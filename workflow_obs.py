"""Workflow to extract obs data."""
import atexit
import logging
import os
import warnings
import xarray as xr
import numpy as np
import xarray.plot
from dask import config as dskconf
from dask.distributed import Client
from dask.diagnostics import ProgressBar
import xclim
import xscen as xs
from xscen.config import CONFIG
import xsdba

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

    # set xclim config to compute indicators on 3H data FixMe: can this be removed?
    xclim.set_options(**CONFIG["set_options"]) #TODO: change check_messing to "wmo" and figure out chunking issues

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
    xs.catalog.ID_COLUMNS.append("type")

    # add a column to allow searching during performance task
    pcat.df['performance_base']=[np.nan]*len(pcat.df)
    pcat.update()

    # set some recurrent variables
    if CONFIG.get("to_dataset_dict", False):
        tdd = CONFIG["to_dataset_dict"]

    # --- EXTRACT---
    if "extract" in CONFIG["tasks"]:
        # iterate on types to extract (reconstruction, station-obs temp and pr)
        for source_type, type_dict in CONFIG["extract"].items():
            # filter catalog for data that we want
            cat = xs.search_data_catalogs(**type_dict["search_data_catalogs"])

            # iterate over ids from the search
            for ds_id, dc in cat.items():
                # check if steps was already done
                if not pcat.exists_in_cat(
                    id=f'{ds_id}_{source_type}',
                    processing_level='extracted'
                ):
                    #for parallelisation
                    with (
                        Client(**type_dict["dask"], **daskkws),
                        xs.measure_time(name=f"extract {ds_id}", logger=logger),
                    ):
                        # create dataset from sub-catalog with right domain and periods
                        ds = xs.extract_dataset(
                            catalog=dc,
                            **type_dict["extract_dataset"],
                        )['D']
                        
                        # update 'cat:type' attribute (little hack to separate station-pr from station-tas)
                        ds.attrs["cat:type"] = source_type
                        ds.attrs["cat:id"] = f'{ds_id}_{source_type}'


                        # save to zarr # format(*cur) is now done in the fonction based on cat attrs
                        xs.save_and_update(ds=ds, pcat=pcat, path=CONFIG['paths']['task'], save_kwargs=type_dict["save"])


    # --- INDICATORS ---
    if "indicators" in CONFIG["tasks"]:
        # get input and iter
        dict_input = pcat.search(**CONFIG["indicators"]["inputs"]).to_dataset_dict(**tdd)
        for key_input, ds_input in sorted(dict_input.items()):
            with (
                Client(**CONFIG["indicators"]["dask"], **daskkws),
                xs.measure_time(name=f"indicators {key_input}", logger=logger)
            ):
                #might be useful later
                # if 'pr' in ds_input.data_vars:
                #     # reference percentiles for precipitation
                #     ref_period = slice(CONFIG["indicators"]["ref_percentiles"]["from"],
                #                        CONFIG["indicators"]["ref_percentiles"]["to"])
                #     ds_input['pr_per95'] = (ds_input['pr']
                #                             .sel(time=ref_period)
                #                             .quantile(0.95, dim='time', keep_attrs=True))
                #     ds_input['pr_per99'] = (ds_input['pr']
                #                             .sel(time=ref_period)
                #                             .quantile(0.99, dim='time', keep_attrs=True))

                
                
                # create module with only indicators that are available for this input
                cur_mod = xs.indicators.select_inds_for_avail_vars(ds_input, CONFIG["indicators"]["path_yml"])
                # compute indicators and write to disk individually, this way it is easier to add more later
                for name, ind in cur_mod.iter_indicators():
                    outfreq = ind.injected_parameters["freq"]
                    outnames = [cfatt["var_name"] for cfatt in ind.cf_attrs]
                    
                    if not pcat.exists_in_cat(
                        id= ds_input.attrs["cat:id"],
                        processing_level='indicators',
                        xrfreq=outfreq, 
                        variable=outnames
                    ):
                        logger.info(f"Computing {outfreq} {outnames}")
                        
                        #TODO: add missing check
                        _, ds_ind = xs.compute_indicators(
                            ds=ds_input,
                            indicators=[(name, ind)],
                        ).popitem()

                        xs.save_and_update(ds=ds_ind, pcat=pcat,
                            path=CONFIG['paths']['task'],save_kwargs=CONFIG["indicators"]["save"])


    # --- CLIMATOLOGIES ---
    if "climatologies" in CONFIG["tasks"]:
        # iterate over inputs
        ind_dict = pcat.search(**CONFIG["aggregate"]["input"]).to_dataset_dict(**tdd)
        for key_input, ds_input in sorted(ind_dict.items()):
            if not pcat.exists_in_cat(
                id=ds_input.attrs['cat:id'],
                xrfreq=ds_input.attrs["cat:xrfreq"],
                processing_level='climatology'
            ):
                with (Client(**CONFIG["aggregate"]["dask"], **daskkws) as client,
                      xs.measure_time(name=f"climatologies {key_input}", logger=logger)
                      ):
                    # compute climatological mean
                    all_horizons = []
                    for period in CONFIG["aggregate"]["periods"]:
                        # compute climatologies for period when contained in data
                        if ds_input.time.dt.year.min() <= int(period[0]) and \
                                ds_input.time.dt.year.max() >= int(period[1]):

                            logger.info(f"Computing climatology for {key_input} for period {period}")

                            # Calculate climatological mean --------------------------------
                            logger.info(f"- Computing climatological mean for {key_input} for period {period}")
                            #TODO: add missing check
                            ds_mean = xs.aggregate.climatological_op(
                                ds=ds_input,
                                **CONFIG["aggregate"]["climatological_mean"],
                                periods=period,
                                rename_variables=True,
                                horizons_as_dim=True,
                            )
                            print(ds_mean)
                            all_horizons.append(ds_mean)

                            # Calculate interannual standard deviation, skipping intra-[freq] std --------------------
                            logger.info(f"- Computing interannual standard deviation for {key_input} for period {period}")
                            # exclude intra-[freq] standard deviation
                            ds_input_std = ds_input[[v for v in CONFIG["aggregate"]["vars_for_interannual_std"]
                                                     if v in ds_input.data_vars]]
                            print(ds_input_std)
                            ds_std = xs.aggregate.climatological_op(
                                ds=ds_input_std,
                                **CONFIG["aggregate"]["climatological_std"],
                                periods=period,
                                rename_variables=True,
                                horizons_as_dim=True,
                            )
                            all_horizons.append(ds_std)
                            print(ds_std)

                    logger.info(f"Merging climatology of periods for {key_input}")
                    ds_clim = xr.merge([ds.drop_vars('time') for ds in all_horizons], combine_attrs='override')
                    print(ds_clim)
                    #TODO: do it by var
                    xs.save_and_update(ds=ds_clim, pcat=pcat,path=CONFIG['paths']['task'])

    # --- PERFORMANCE ---
    performance_path = CONFIG['paths']['performance']

    if "performance" in CONFIG["tasks"]:
        for statistic_name, search_param_dicts in CONFIG["performance"]["statistics"].items():
            statistic_func = getattr(xsdba.measures, statistic_name)
            for search_param_dict in search_param_dicts:
                # search_param_dict provides parameters for pcat.search, enabling selection
                # of equivalent datasets (e.g., same variable; but from different sources)
                
                variable_name = search_param_dict["variable"] # The variable for which we're computing the measure
                performance_variable_name = f"{variable_name}_{statistic_name}"

                obs_dict = pcat.search(
                    **search_param_dict, # Shared search criteria (e.g., variable)
                    **CONFIG["performance"]["input"]["observation"] # Observation-exclusive search criteria
                ).to_dataset_dict()

                rec_dict = pcat.search(
                    **search_param_dict,
                    **CONFIG["performance"]["input"]["reconstruction"] # Reconstruction-exclusive search criteria
                ).to_dataset_dict()

                for obs_dataset_id, obs_dataset in obs_dict.items(): # For each observation dataset
                    obs_source = obs_dataset.attrs['cat:source']

                    for rec_dataset_id, rec_dataset in rec_dict.items(): # For each reconstruction dataset
                        rec_source = rec_dataset.attrs['cat:source']
                        
                        if pcat.exists_in_cat(id=rec_dataset_id, processing_level="performance", performance_base=obs_dataset_id):
                            logger.info(f"Skipping existing performance for: {performance_variable_name} ({rec_source} vs {obs_source})")
                            continue

                        with (
                            Client(**CONFIG["performance"]["dask"], **daskkws),
                            xs.measure_time(name=f"performance {performance_variable_name} ({rec_source} vs {obs_source})", logger=logger)
                        ):
                            logger.info(f"Computing {statistic_name} between {rec_dataset_id} and {obs_dataset_id}")
                            ## Selecting grid points located on stations ##
                            station_lats = obs_dataset.lat.values
                            station_lons = obs_dataset.lon.values

                            # drop the nans, to avoid choosing a grid cell in the sea during the subsetting
                            rec_nanless=xs.utils.stack_drop_nans(rec_dataset,
                                                                 mask = rec_dataset.isel(time=10, drop=True).notnull().compute())
                            rec_subset = xs.spatial.subset(
                                rec_nanless,
                                method='gridpoint',
                                lat=station_lats,
                                lon=station_lons
                            )
                            
                            # Rename the dimension added by xs.spatial.subset
                            if 'site' in rec_subset.dims:
                                rec_subset = rec_subset.rename({'site': 'station'})
                            else:
                                raise ValueError(f"Expected 'site' dimension in rec_subset, but found: {dict(rec_subset.dims)}")
                            
                            # put back the unique coords of the obs on the rec
                            station_coords=set(obs_dataset.coords) - set(rec_subset.coords)
                            for c in station_coords:
                                rec_subset.coords[c]=obs_dataset.coords[c]

                            # unstack date to have one stat per season
                            rec_unstacked=xs.utils.unstack_dates(rec_subset)
                            obs_unstacked=xs.utils.unstack_dates(obs_dataset)

                            ## Selecting a common time slice, as their start/end dates might differ
                            common_time = np.intersect1d(obs_unstacked['time'], rec_unstacked['time'])
                            obs_sliced = obs_unstacked.sel(time=common_time)
                            rec_sliced = rec_unstacked.sel(time=common_time)

                            #check if stations have a least n years of data, if not fill it with nan
                            min_years=CONFIG['performance']['minimum_n_years']
                            obs_filtered = obs_sliced.where((obs_sliced.count(dim='time')>=min_years).compute())


                            ## Computing the performance metric ##
                            da_output = statistic_func( # The output data array
                                sim=rec_sliced[variable_name],
                                ref=obs_filtered[variable_name]
                            )
                            ds_output = da_output.to_dataset(name=performance_variable_name) # The output dataset
                            
                            ds_output.attrs["cat:xrfreq"]= "fx" # Frequency is fixed, as there is no time axis
                            ds_output.attrs["cat:variable"] = performance_variable_name
                            ds_output.attrs["cat:processing_level"] = "performance"
                            ds_output.attrs["cat:source"] = rec_source
                            ds_output.attrs["cat:performance_base"] = obs_dataset.attrs["cat:id"]
                            ds_output.attrs["cat:id"] = rec_dataset.attrs["cat:id"]
                            ds_output.attrs["cat:domain"] = rec_dataset.attrs["cat:domain"]

                            del ds_output.station.encoding['filters'] # Existing value in encoding's "filters" breaks "save_and_update"
                            
                            xs.save_and_update(
                                ds=ds_output,
                                pcat=pcat,
                                path=performance_path,
                                save_kwargs=CONFIG["performance"]["save"]
                            )




    xs.send_mail(
        subject="ObsFlow - Message",
        msg="Congratulations! All tasks of the workflow were completed!",
    )

