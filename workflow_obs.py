"""Workflow to extract obs data."""
import atexit
import logging
import os

import xarray as xr
from dask import config as dskconf
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import xclim
import xscen as xs
from xscen.config import CONFIG
from copy import copy

# Load configuration
xs.load_config(
    "paths_obs.yml", "config_obs.yml", verbose=(__name__ == "__main__"), reset=True
)

# get logger
if "logging" in CONFIG:
    logger = logging.getLogger("xscen")

if __name__ == "__main__":
    # set dask  configuration
    daskkws = CONFIG["dask"].get("client", {})
    dskconf.set(**{k: v for k, v in CONFIG["dask"].items() if k != "client"})

    # set xclim config to compute indicators on 3H data
    xclim.set_options(data_validation="log")

    # copy config to the top of the log file
    if "logging" in CONFIG and "file" in CONFIG["logging"]["handlers"]:
        f1 = open(CONFIG["logging"]["handlers"]["file"]["filename"], "a+")
        f2 = open("config_obs.yml")
        f1.write(f2.read())
        f1.close()
        f2.close()

    # set email config
    if "scripting" in CONFIG:
        atexit.register(xs.send_mail_on_exit, subject=CONFIG["scripting"]["subject"])

    # initialize Project Catalog (only do this once, if the file doesn't already exist)
    if not os.path.exists(CONFIG["paths"]["project_catalog"]):
        pcat = xs.ProjectCatalog.create(
            CONFIG["paths"]["project_catalog"],
            project=CONFIG["project"],
        )

    # load project catalog
    pcat = xs.ProjectCatalog(CONFIG["paths"]["project_catalog"])

    # set some recurrent variables
    if CONFIG.get("to_dataset_dict", False):
        tdd = CONFIG["to_dataset_dict"]

    # --- EXTRACT---
    if "extract" in CONFIG["tasks"]:
        # iterate on types to extract (reconstruction, simulation)
        for source_type, type_dict in CONFIG["extract"].items():
            # filter catalog for data that we want
            cat = xs.search_data_catalogs(**type_dict["search_data_catalogs"])

            # iterate over ids from the search
            for ds_id, dc in cat.items():
                # attrs of current iteration that are relevant now
                cur = {
                    "id": ds_id,
                    "xrfreq": "D",
                    "processing_level": "extracted",
                }
                # check if steps was already done
                if not pcat.exists_in_cat(**cur):
                    with (
                        Client(**type_dict["dask"], **daskkws),
                        xs.measure_time(name=f"extract {ds_id}", logger=logger),
                    ):
                        # create dataset from sub-catalog with right domain and periods
                        ds_dict = xs.extract_dataset(
                            catalog=dc,
                            **type_dict["extract_dataset"],
                        )

                        # iterate over the different datasets/frequencies
                        for key_freq, ds in ds_dict.items():
                            if type_dict.get("floor", False):
                                # won't be needing this when data is completely cleaned
                                ds["time"] = ds.time.dt.floor(type_dict["floor"])

                            # drop nans and stack lat/lon in 1d loc (makes code faster)
                            if type_dict.get("stack_drop_nans", False):
                                ds = xs.utils.stack_drop_nans(
                                    ds,
                                    ds[list(ds.data_vars)[0]]
                                    .isel(time=0, drop=True)
                                    .notnull(),
                                )

                            # save to zarr
                            path = CONFIG["paths"]["task"].format(**cur)
                            # try:
                            #     xs.save_to_netcdf(ds=ds, filename=path.replace('zarr', 'nc'),
                            #                       rechunk=type_dict["save"]["rechunk"],
                            #                       # for some reason h5netcdf doesn't write correct files here
                            #                       netcdf_kwargs={'engine': 'netcdf4'})
                            # except:
                            #     print(f'Couln\'t write file: {path.replace("zarr", "nc")}')
                            xs.save_to_zarr(ds=ds, filename=path, **type_dict["save"])
                            pcat.update_from_ds(ds=ds, path=path)

    # --- REGRID ---
    if "regrid" in CONFIG["tasks"]:
        # get input and iter over datasets
        input_dict = pcat.search(**CONFIG["regrid"]["inputs"]).to_dataset_dict(**tdd)
        for key_input, ds_input in input_dict.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "regridded",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["regrid"]["dask"], **daskkws),
                    xs.measure_time(name=f"regrid {key_input}", logger=logger),
                ):
                    # get output grid
                    ds_grid = pcat.search(**CONFIG["regrid"]["output"]).to_dataset(
                        **tdd
                    )

                    # do regridding
                    ds_regrid = xs.regrid_dataset(
                        ds=ds_input,
                        ds_grid=ds_grid,
                    )

                # save to zarr
                path = f"{CONFIG['paths']['task']}".format(**cur)
                xs.save_to_zarr(ds=ds_regrid, filename=path, **CONFIG["regrid"]["save"])
                pcat.update_from_ds(ds=ds_regrid, path=path, info_dict=cur)

    # --- BIAS ADJUST ---
    if "biasadjust" in CONFIG["tasks"]:
        # iter over each variable that needs to be adjusted
        for var, ba_dict in CONFIG["biasadjust"].items():
            # get all input simulations and iter over them
            dict_sim = pcat.search(**ba_dict["sim_inputs"]).to_dataset_dict(**tdd)
            for id_sim, ds_sim in dict_sim.items():
                cur = {
                    "id": ds_sim.attrs["cat:id"],
                    "xrfreq": ds_sim.attrs["cat:xrfreq"],
                    "processing_level": "biasadjusted",
                    "variable": var,
                }
                if not pcat.exists_in_cat(**cur):
                    with (
                        Client(**ba_dict["dask"], **daskkws),
                        xs.measure_time(
                            name=f" bias adjust {var} {id_sim}", logger=logger
                        ),
                    ):
                        # get reference
                        ds_ref = pcat.search(**ba_dict["ref_input"]).to_dataset(**tdd)
                        # training
                        ds_tr = xs.train(
                            dref=ds_ref,
                            dhist=ds_sim,
                            var=[var],
                            **ba_dict["xscen_train"],
                        )

                        # might need to save ds_tr in between if data is too big.

                        # adjusting
                        ds_scen = xs.adjust(
                            dsim=ds_sim,
                            dtrain=ds_tr,
                            to_level="biasadjusted",
                            **ba_dict["xscen_adjust"],
                        )

                        # save and update
                        path = f"{CONFIG['paths']['ba']}".format(**cur)
                        xs.save_to_zarr(ds=ds_scen, filename=path, **ba_dict["save"])
                        pcat.update_from_ds(ds=ds_scen, path=path)

    # --- CLEAN UP ---
    if "cleanup" in CONFIG["tasks"]:
        # get all datasets to clean up and iter
        cu_cats = xs.search_data_catalogs(**CONFIG["cleanup"]["search_data_catalogs"])
        for cu_id, cu_cat in cu_cats.items():
            cur = {
                "id": cu_id,
                "xrfreq": "D",
                "processing_level": "cleaned",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["cleanup"]["dask"], **daskkws),
                    xs.measure_time(name=f"clean {cu_id}", logger=logger),
                ):
                    # put the individually adjusted variables back together in one ds
                    freq_dict = xs.extract_dataset(catalog=cu_cat)

                    # iter over dataset of different frequencies (usually just 'D')
                    for key_freq, ds in freq_dict.items():
                        # clean up the dataset
                        ds_clean = xs.clean_up(
                            ds=ds,
                            **CONFIG["cleanup"]["xscen_clean_up"],
                        )

                        # save and update
                        path = f"{CONFIG['paths']['task']}".format(**cur)
                        xs.save_to_zarr(ds_clean, path, **CONFIG["cleanup"]["save"])
                        pcat.update_from_ds(ds=ds_clean, path=path)

    # --- RECHUNK and store final daily data ---
    if "rechunk" in CONFIG["tasks"]:
        # get inputs and iter over them
        dict_input = pcat.search(**CONFIG["rechunk"]["inputs"]).to_dataset_dict(**tdd)
        for key_input, ds_input in dict_input.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "domain": ds_input.attrs["cat:domain"],
                "processing_level": "final",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["rechunk"]["dask"], **daskkws),
                    xs.measure_time(name=f"rechunk {key_input}", logger=logger),
                ):
                    # final path for the data
                    path_out = f"{CONFIG['paths']['task']}".format(**cur)

                    # rechunk and put in final path
                    xs.io.rechunk(
                        path_in=ds_input.attrs["cat:path"],
                        path_out=path_out,
                        **CONFIG["rechunk"]["xscen_rechunk"],
                    )

                    # update catalog
                    ds = xr.open_zarr(path_out)
                    pcat.update_from_ds(
                        ds=ds,
                        path=str(path_out),
                        info_dict={"processing_level": "final"},
                    )

    # --- DIAGNOSTICS ---
    if "diagnostics" in CONFIG["tasks"]:
        # compute properties (and measures) on different kinds of data (ref, sim,scen)
        for kind, kind_dict in CONFIG["diagnostics"]["kind"].items():
            # iterate on inputs
            dict_input = pcat.search(**kind_dict["inputs"]).to_dataset_dict(**tdd)
            my_kinds_properties = copy(kind_dict["properties_and_measures"]["properties"])
            for key_input, ds_input in dict_input.items():
                cur = {
                    "id": ds_input.attrs["cat:id"],
                    "processing_level": kind_dict["properties_and_measures"].get(
                        "to_level_prop", "diag-properties"
                    ),
                    "xrfreq": "fx",
                }

                if not pcat.exists_in_cat(**cur):
                    with (
                        Client(**CONFIG["diagnostics"]["dask"], **daskkws),
                        xs.measure_time(name=f"diagnostics {key_input}", logger=logger),
                    ):
                        # get the reference for the measures
                        dref_for_measure = None
                        if "dref_for_measure" in kind_dict:
                            dref_for_measure = pcat.search(
                                **kind_dict["dref_for_measure"],
                            ).to_dataset(**tdd)

                        # filter for properties for which variables are available
                        kind_dict["properties_and_measures"]["properties"] = \
                            xs.indicators.select_inds_for_avail_vars(ds_input, my_kinds_properties)

                        # compute properties per period
                        all_periods = []
                        for period in CONFIG["diagnostics"]["periods"]:
                            # skip all data except bcs for period 1980-1985
                            if int(period[0]) == 1980 and int(period[1]) == 1985 and 'bcs' not in key_input:
                                continue
                            # compute properties for period
                            if ds_input.time.dt.year.min() <= int(period[0]) and \
                               ds_input.time.dt.year.max() >= int(period[1]):
                                logger.info(f"Computing properties for {key_input} for period {period}")
                                periods_props, _ = xs.properties_and_measures(
                                    ds=ds_input,
                                    dref_for_measure=dref_for_measure,
                                    **kind_dict["properties_and_measures"],
                                    period=period,
                                )
                                periods_props = periods_props.assign_coords(period=f'{period[0]}-{period[1]}')
                                periods_props = periods_props.expand_dims(dim='period')
                                if 'month' in periods_props.coords.keys():
                                    periods_props = periods_props.assign_coords(
                                        {'month': list(xr.coding.cftime_offsets._MONTH_ABBREVIATIONS.values())})
                                # TODO: does this work accross all datasets? Ok for now for ERA5-Land, AHCCD, ...
                                if 'rlat' not in periods_props.coords.keys() and \
                                   'rlon' not in periods_props.coords.keys():
                                    periods_props['lat'].attrs = {"standard_name": "latitude", "axis": "Y"}
                                    periods_props['lon'].attrs = {"standard_name": "longitude", "axis": "X"}
                                all_periods.append(periods_props)

                        logger.info(f"Merging properties of periods for {key_input}")
                        prop = xr.merge(all_periods, combine_attrs='override')

                        # save to zarr
                        # don't save if empty
                        if len(prop.data_vars) > 0:
                            path_diag = f"{CONFIG['paths']['task']}".format(**cur)
                            xs.save_to_zarr(prop, path_diag, **kind_dict["save"])
                            pcat.update_from_ds(ds=prop, path=path_diag)

        # # summary of diagnostics
        # get sim measures
        meas_dict = pcat.search(processing_level="diag-measures-sim").to_dataset_dict(
            **tdd
        )
        for id_meas, ds_meas_sim in meas_dict.items():
            cur = {
                "id": ds_meas_sim.attrs["cat:id"],
                "processing_level": "diag-improved",
                "xrfreq": "fx",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["diagnostics"]["dask"], **daskkws),
                    xs.measure_time(name=f"summary diag {cur['id']}", logger=logger),
                ):
                    # get scen meas associated with sim
                    meas_datasets = {}
                    meas_datasets["sim"] = ds_meas_sim
                    meas_datasets["scen"] = pcat.search(
                        processing_level="diag-measures-scen",
                        id=cur["id"],
                    ).to_dataset(**tdd)

                    # compute heatmap
                    hm = xs.diagnostics.measures_heatmap(meas_datasets)

                    # compute improved
                    ip = xs.diagnostics.measures_improvement(meas_datasets)

                    # save and update
                    for ds in [hm, ip]:
                        cur["processing_level"] = ds.attrs["cat:processing_level"]
                        path_diag = f"{CONFIG['paths']['task']}".format(**cur)
                        xs.save_to_zarr(ds=ds, filename=path_diag, mode="o")
                        pcat.update_from_ds(ds=ds, path=path_diag)

    # --- INDICATORS ---
    if "indicators" in CONFIG["tasks"]:
        # get input and iter
        dict_input = pcat.search(**CONFIG["indicators"]["inputs"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in dict_input.items():
            with (
                Client(**CONFIG["indicators"]["dask"], **daskkws),
                xs.measure_time(name=f"indicators {key_input}", logger=logger),
            ):

                # compute indicators
                dict_indicator = xs.compute_indicators(
                    ds=ds_input,
                    indicators=xs.indicators.select_inds_for_avail_vars(ds_input, CONFIG["indicators"]["path_yml"]),
                )

                # iter over output dictionary (keys are freqs)
                for key_freq, ds_ind in dict_indicator.items():
                    cur = {
                        "id": ds_input.attrs["cat:id"],
                        "xrfreq": key_freq,
                        "processing_level": "indicators",
                    }
                    if not pcat.exists_in_cat(**cur):
                        # save to zarr
                        path_ind = f"{CONFIG['paths']['task']}".format(**cur)
                        xs.save_to_zarr(
                            ds_ind, path_ind, **CONFIG["indicators"]["save"]
                        )
                        pcat.update_from_ds(ds=ds_ind, path=path_ind)

    # --- CLIMATOLOGICAL MEAN ---
    if "climatology" in CONFIG["tasks"]:
        # iterate over inputs
        ind_dict = pcat.search(**CONFIG["aggregate"]["input"]["obs"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in ind_dict.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "climatology",
            }

            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["aggregate"]["dask"], **daskkws),
                    xs.measure_time(name=f"climatology {key_input}", logger=logger),
                ):
                    # compute climatological mean
                    all_periods = []
                    for period in CONFIG["aggregate"]["periods"]:
                        # skip all data except bcs for period 1980-1985 ToDo: remove when tests finished
                        if int(period[0]) == 1980 and int(period[1]) == 1985 and 'bcs' not in key_input:
                            continue
                        # compute properties for period when contained in data
                        if ds_input.time.dt.year.min() <= int(period[0]) and \
                           ds_input.time.dt.year.max() >= int(period[1]):

                            logger.info(f"Computing climatology for {key_input} for period {period}")

                            # Calculate climatological mean
                            ds_mean = xs.climatological_mean(
                                ds=ds_input,
                                **CONFIG["aggregate"]["climatological_mean"],
                                periods=period
                            )
                            ds_mean = ds_mean.assign_coords(period=f'{period[0]}-{period[1]}')
                            ds_mean = ds_mean.expand_dims(dim='period')
                            ds_mean = ds_mean.drop_vars('horizon')
                            all_periods.append(ds_mean)

                            # Calculate interannual standard deviation
                            ds_std = xs.aggregate.climatological_op(
                                ds=ds_input,
                                **CONFIG["aggregate"]["climatological_op"],
                                periods=period
                            )
                            ds_std = ds_std.assign_coords(period=f'{period[0]}-{period[1]}')
                            ds_std = ds_std.expand_dims(dim='period')
                            ds_std = ds_std.drop_vars('horizon')
                            ds_std = ds_std.rename(dict(zip(
                                ds_std.data_vars,
                                [f"{var}_std" for var in ds_std.data_vars]))
                            )
                            all_periods.append(ds_std)

                            # Calculate intra monthly/seasonal standard deviation
                            # ToDo

                            # Calculate climatological standard deviation
                            # ToDo

                            # Calculate trends
                            # ToDo

                    # remove all dates so that periods can be merged
                    new_time = {1: {'year': ['ANN']},
                                4: {'season': ['MAM', 'JJA', 'SON', 'DJF']},
                                12: {'month': list(xr.coding.cftime_offsets._MONTH_ABBREVIATIONS.values())},
                    }
                    all_periods = [ds.rename({'time': list(new_time[ds.time.size].keys())[0]})
                                   .assign_coords(new_time[ds.time.size]) for ds in all_periods]
                    logger.info(f"Merging climatology of periods for {key_input}")
                    ds_clim = xr.merge(all_periods, combine_attrs='override')

                    # save to zarr
                    path = f"{CONFIG['paths']['task']}".format(**cur)
                    xs.save_to_zarr(ds_clim, path, **CONFIG["aggregate"]["save"])
                    pcat.update_from_ds(ds=ds_clim, path=path)

    # --- DELTAS ---
    if "delta" in CONFIG["tasks"]:
        # iterate over inputs
        ind_dict = pcat.search(**CONFIG["aggregate"]["input"]["delta"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in ind_dict.items():
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "delta",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["aggregate"]["dask"], **daskkws),
                    xs.measure_time(name=f"delta {key_input}", logger=logger),
                ):
                    # compute deltas
                    ds_delta = xs.aggregate.compute_deltas(ds=ds_input)

                    # save to zarr
                    path = f"{CONFIG['paths']['task']}".format(**cur)
                    xs.save_to_zarr(ds_delta, path, **CONFIG["aggregate"]["save"])
                    pcat.update_from_ds(ds=ds_delta, path=path)

    # --- ENSEMBLES ---
    if "ensembles" in CONFIG["tasks"]:
        # one ensemble (file) per level, per experiment, per xrfreq
        for processing_level in CONFIG["ensembles"]["processing_levels"]:
            ind_df = pcat.search(processing_level=processing_level).df
            # iterate through available xrfreq, exp and variables
            for experiment in ind_df.experiment.unique():
                for xrfreq in ind_df.xrfreq.unique():
                    # get all datasets that go in the ensemble
                    ind_dict = pcat.search(
                        processing_level=processing_level,
                        experiment=experiment,
                        xrfreq=xrfreq,
                    ).to_dataset_dict(**tdd)

                    cur = {
                        "processing_level": f"ensemble-{processing_level}",
                        "experiment": experiment,
                        "xrfreq": xrfreq,
                    }
                    if not pcat.exists_in_cat(**cur):
                        with (
                            Client(**CONFIG["ensembles"]["dask"], **daskkws),
                            xs.measure_time(
                                name=f"ens-{processing_level} {experiment} {xrfreq}",
                                logger=logger,
                            ),
                        ):
                            ens_stats = xs.ensemble_stats(
                                datasets=ind_dict,
                                to_level=f"ensemble-{processing_level}",
                            )

                            # add new id
                            cur["id"] = ens_stats.attrs["cat:id"]

                            # save to zarr
                            path = f"{CONFIG['paths']['task']}".format(**cur)
                            xs.save_to_zarr(
                                # ens_stats, path, **CONFIG["ensembles"]["save"]
                            )
                            pcat.update_from_ds(ds=ens_stats, path=path)

    xs.send_mail(
        subject="ObsFlow - Message",
        msg="Congratulations! All tasks of the workflow are done!",
    )
