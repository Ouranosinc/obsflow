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
    xs.catalog.ID_COLUMNS.append("type")

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
                    "id": f'{ds_id}_{source_type}',
                    "xrfreq": "D",
                    "processing_level": "extracted",
                    "type": source_type,
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
                            # update 'cat:type' attribute (to separate station-pr from station-tas)
                            ds.attrs["cat:type"] = cur["type"]
                            ds.attrs["cat:id"] = cur['id']

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

    # --- CLEAN UP ---
    if "cleanup" in CONFIG["tasks"]:
        # get all datasets to clean up and iter
        # cu_cats = xs.search_data_catalogs(**CONFIG["cleanup"]["search_data_catalogs"])
        # for cu_id, cu_cat in cu_cats.items():
        input_dict = pcat.search(**CONFIG["cleanup"]["inputs"]).to_dataset_dict(**tdd)
        # iter over datasets
        for ds_id, ds_input in input_dict.items():
            cur = {
                "id": '.'.join(ds_id.split('.')[:2]),
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "extracted-cleaned",
            }
            if not pcat.exists_in_cat(**cur):
                with (
                    Client(**CONFIG["cleanup"]["dask"], **daskkws),
                    xs.measure_time(name=f"clean {ds_id}", logger=logger),
                ):

                    # clean up the dataset
                    ds_clean = xs.clean_up(
                        ds=ds_input,
                        **CONFIG["cleanup"]["xscen_clean_up"],
                    )

                    # save and update
                    path = f"{CONFIG['paths']['task']}".format(**cur)
                    save_kwargs = CONFIG["cleanup"]["save"]
                    if "encoding" in save_kwargs:
                        save_kwargs["encoding"] = {
                            k: v
                            for k, v in save_kwargs["encoding"].items()
                            if k in ds_clean.data_vars
                        }
                    xs.save_to_zarr(ds_clean, path, **save_kwargs)
                    pcat.update_from_ds(ds=ds_clean, path=path)

    # --- INDICATORS ---
    if "indicators" in CONFIG["tasks"]:
        # get input and iter
        dict_input = pcat.search(**CONFIG["indicators"]["inputs"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in sorted(dict_input.items()):
            with (
                Client(**CONFIG["indicators"]["dask"], **daskkws),
                xs.measure_time(name=f"indicators {key_input}", logger=logger)
            ):
                if 'pr' in ds_input.data_vars:
                    # reference percentiles for precipitation
                    ref_period = slice(CONFIG["indicators"]["ref_percentiles"]["from"],
                                       CONFIG["indicators"]["ref_percentiles"]["to"])
                    ds_input['pr_per95'] = (ds_input['pr']
                                            .sel(time=ref_period)
                                            .quantile(0.95, dim='time', keep_attrs=True))
                    ds_input['pr_per99'] = (ds_input['pr']
                                            .sel(time=ref_period)
                                            .quantile(0.99, dim='time', keep_attrs=True))

                # compute indicators
                dict_indicator = xs.compute_indicators(
                    ds=ds_input,
                    indicators=xs.indicators.select_inds_for_avail_vars(ds_input,
                                                                        CONFIG["indicators"]["path_yml"]),
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
                        xs.save_to_zarr(ds_ind, path_ind, **CONFIG["indicators"]["save"])
                        pcat.update_from_ds(ds=ds_ind, path=path_ind)

    # --- CLIMATOLOGIES ---
    if "climatologies" in CONFIG["tasks"]:
        # iterate over inputs
        ind_dict = pcat.search(**CONFIG["aggregate"]["input"]).to_dataset_dict(
            **tdd
        )
        for key_input, ds_input in sorted(ind_dict.items()):
            cur = {
                "id": ds_input.attrs["cat:id"],
                "xrfreq": ds_input.attrs["cat:xrfreq"],
                "processing_level": "climatology",
            }


            # if key_input not in 'GovCan_AHCCD_CAN_station-pr.Quebec.indicators.MS':  # FixMe: remove this
            #    continue

            if not pcat.exists_in_cat(**cur):
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
                            ds_mean = xs.aggregate.climatological_op(
                                ds=ds_input,
                                **CONFIG["aggregate"]["climatological_mean"],
                                periods=period,
                                rename_variables=True,
                                horizons_as_dim=True,
                            )
                            all_horizons.append(ds_mean)

                            # Calculate interannual standard deviation, skipping intra-[freq] std --------------------
                            logger.info(f"- Computing interannual standard deviation for {key_input} for period {period}")
                            # exclude intra-[freq] standard deviation
                            ds_input_std = ds_input[[v for v in CONFIG["aggregate"]["vars_for_interannual_std"]
                                                     if v in ds_input.data_vars]]
                            # ds_input_std = ds_input.filter_by_attrs(
                            #     description=lambda s: 'standard deviation' not in str(s)
                            # )
                            ds_std = xs.aggregate.climatological_op(
                                ds=ds_input_std,
                                **CONFIG["aggregate"]["climatological_std"],
                                periods=period,
                                rename_variables=True,
                                horizons_as_dim=True,
                            )
                            all_horizons.append(ds_std)

                            # Calculate climatological standard deviation for pr, tg, tn, tx only --------------------
                            logger.info(
                                f"- Computing climatological standard deviation for {key_input} for period {period}")
                            # ToDo: This could be generic to work for all terms involved, depending on the input freq
                            ds_std_clim = xr.Dataset()
                            with xr.set_options(keep_attrs=True):
                                for v_type in set([name.split('_')[0]
                                                   for name in CONFIG["aggregate"]["vars_for_climatological_std"]
                                                   if name in ds_input.data_vars]):
                                    # pick the interannual standard deviation of the variable
                                    ds_std_varname = [n for n in ds_std.data_vars if f"{v_type}_mean" in n and 'std' in n]
                                    if len(ds_std_varname) == 1:
                                        ds_std_varname = ds_std_varname[0]
                                    else:
                                        raise ValueError(
                                            f"More than one variable found containing "
                                            f"'{v_type}' and 'std' in {ds_std_varname}"
                                        )
                                    # pick the mean of intra-annual/monthly/seasonal standard deviation of the variable
                                    ds_mean_varname = [n for n in ds_mean.data_vars if f"{v_type}_std" in n]
                                    if len(ds_mean_varname) == 1:
                                        ds_mean_varname = ds_mean_varname[0]
                                    else:
                                        raise ValueError(
                                            f"More than one variable found containing "
                                            f"'{v_type}' and 'std' in {ds_mean_varname}"
                                        )
                                    # calculate the total climatological standard deviation
                                    new_varname = f'{v_type}_std_clim_total'
                                    ds_std_clim[new_varname] = np.sqrt(
                                        np.square(ds_std[ds_std_varname]) +
                                        np.square(ds_mean[ds_mean_varname])
                                    )
                                    # add attributes
                                    ds_std_clim[new_varname].attrs['description'] = \
                                        f"{xclim.core.formatting.default_formatter.format_field(ds_mean.attrs['cat:xrfreq'], 'adj').capitalize()} " \
                                        f"total standard deviation of {' '.join(ds_std[ds_std_varname].attrs['description'].split(' ')[-3:])}"
                                    ds_std_clim[new_varname].attrs['long_name'] = ds_std_clim[new_varname].attrs[
                                        'description']
                            if ds_std_clim:
                                all_horizons.append(ds_std_clim)

                            # Calculate trends -----------------------------------------------
                            logger.info(f"- Computing climatological linregress for {key_input} for period {period}")

                            # ds_input_trend1 = ds_input[[v for v in ds_input.data_vars if 'mean' in v]]
                            ds_input_trend = ds_input[[v for v in CONFIG["aggregate"]["vars_for_climatological_trend"]
                                                       if v in ds_input.data_vars]]
                            ds_trend = xs.aggregate.climatological_op(
                                ds=ds_input_trend,
                                **CONFIG["aggregate"]["climatological_trend"],
                                periods=period,
                                min_periods=0.7,
                                rename_variables=True,
                                horizons_as_dim=True,
                            )
                            all_horizons.append(ds_trend)

                    logger.info(f"Merging climatology of periods for {key_input}")
                    # all_horizons = client.scatter(all_horizons)
                    ds_clim = xr.merge([ds.drop_vars('time') for ds in all_horizons], combine_attrs='override')

                    # save to zarr
                    path = f"{CONFIG['paths']['task']}".format(**cur)
                    xs.save_to_zarr(ds_clim, path, **CONFIG["aggregate"]["save"])
                    pcat.update_from_ds(ds=ds_clim, path=path)

    # --- PLOTTING ---
    if "plotting" in CONFIG["tasks"]:
        # get input and iter
        dict_input = pcat.search(**CONFIG["plotting"]["data"]).to_dataset_dict(**tdd)
        for key_input, ds_input in sorted(dict_input.items()):
            if 'AHCCD' in key_input:  # currently AHCCD data are only plotted on top of reanalysis data
                continue
            with (Client(**CONFIG["plotting"]["dask"], **daskkws),
                    xs.measure_time(name=f"plotting {key_input}", logger=logger)):

                import cartopy.crs as ccrs
                import matplotlib.pyplot as plt
                import figanos.matplotlib as fg
                from textwrap import wrap
                from pathlib import Path
                import matplotlib

                # matplotlib.use('wxAgg')
                # matplotlib.use("TkAgg")
                # matplotlib.use("macosx")
                # matplotlib.use('Qt5Agg')
                # matplotlib.use('Agg')

                fg.utils.set_mpl_style('ouranos')
                projection = ccrs.LambertConformal()
                freq = CONFIG['plotting'][ds_input.attrs['cat:xrfreq']]['xrfreq_name']

                for horizon in ds_input.horizon.values:  # ['1991-2018']:  # ds_input.horizon.values:  #['1981-2010']:  # FixMe: remove this
                    for da_grid in ds_input.data_vars.values():

                        try:
                            # Filter specific variables/indicators/periods
                            if da_grid.name not in CONFIG["plotting"]["indicators"]:
                                continue

                                # if 'dtr' not in da_grid.name: continue  # FixMe: remove this
                            # if '1981-2010' not in horizon: continue  # FixMe: remove this
                            # if 'tx_mean_clim_linregress' not in da_grid.name: continue  # FixMe: remove this
                            # if any(n in da_grid.name for n in ['tg', 'tx', 'tn']):
                            # if 'RDRS' not in key_input: continue  # FixMe: remove this
                            #if 'year' in freq: continue  # FixMe: remove this
                            if 'season' in freq: continue  # FixMe: remove this
                            if 'month' in freq: continue  # FixMe: remove this
                            #if 'YS-JUL' not in ds_input.attrs['cat:xrfreq']: continue

                            # Wait! if we don't have that indicator in the config, let's configure it
                            logger.info(f"Coming up: {da_grid.name}.")
                            if da_grid.name not in CONFIG["plotting"]:
                                warnings.warn(f"Variable {da_grid.name} not in CONFIG['plotting']!")
                                continue
                            # else:
                            #     logger.info(f"Nope, I'm not plotting '{da_grid.name}' for '{key_input}' ({horizon}) "
                            #                 f"this time! I can't find it in the configuration!")
                            #     continue

                            # Fix heat/cold spell long names # ToDo: remove this
                            if 'spell' in da_grid.name and 'class' not in da_grid.attrs['long_name']:
                                da_grid.attrs['long_name'] = \
                                    [da_grid.attrs['long_name'][:-1] + ' ' + s \
                                    for s in da_grid.name.split('_') if 'class' in s][0]

                            # get a plot_id for labeling and file naming -------------------------------------------
                            plot_id = (f"{CONFIG['plotting'][da_grid.name]['label']} "
                                       f"{da_grid.attrs['long_name'].lower()} - "
                                       f"{ds_input.attrs['source']} ({horizon})")
                            # print(f"Raw ID: ..................... {da_grid.name} --- {plot_id}")

                            # trim the plot_id ---------------------------------------------------------------------
                            adj = {'year': 'annual', 'month': 'monthly', 'season': 'seasonal'}
                            changes = {
                                'interannual 30-year climatological': 'interannual',
                                'intra 30-year climatological average of ': 'intra-',
                                '30-year climatological average': f'{adj[freq]} climate average',
                                '30-year climatological linregress': 'linear climate trend',
                                'total standard deviation': 'climate standard deviation',
                                '.': '',
                                '˚c': '°C',
                            }
                            for k, v in changes.items():
                                plot_id = plot_id.replace(k, v).strip()
                            #logger.info(f"Variable: {da_grid.name} --- Plotting {plot_id}") # ToDo: reactivate!
                            # print(f'Trimmed ID: {da_grid.name} --- {plot_id}')

                            # get the corresponding AHCCD observation data -----------------------------------------
                            try:
                                da_station = pcat.search(
                                    source='AHCCD',
                                    processing_level='climatology',
                                    xrfreq=ds_input.attrs.get('cat:xrfreq'),
                                    variable=da_grid.name,
                                ).to_dataset(**tdd)[da_grid.name]
                            except ValueError:
                                da_station = None

                            if horizon not in da_station.horizon.values:
                                da_station = None

                            # selection and scaling of data ToDo: the scaling should be done in the clean-up step --
                            sel_kwargs = {
                                "horizon": horizon}  # | {freq: da_grid[freq].values} if freq not in 'year' else {}

                            scale_factor = 1
                            if 'linregress' in da_grid.name:
                                sel_kwargs.setdefault('linreg_param', 'slope')
                                scale_factor = 10

                            # figure arguments -
                            fig_kwargs = CONFIG["plotting"][freq]["fig_kwargs"]

                            # levels and ticks -----------------------------------------------------------------------
                            if 'linspace' in CONFIG["plotting"][da_grid.name]["ticks"]:
                                levels = list(np.linspace(CONFIG["plotting"][da_grid.name]["limits"][freq]["vmin"],
                                                          CONFIG["plotting"][da_grid.name]["limits"][freq]["vmax"],
                                                          CONFIG["plotting"][da_grid.name]["levels"]))
                                ticks = levels[0::2]
                            else:
                                levels = CONFIG["plotting"][da_grid.name]["levels"]
                                ticks = CONFIG["plotting"][da_grid.name]["ticks"]

                            # print(f'Freq: {freq}\nLevels: {levels}\nTicks: {ticks}\n'
                            #       f'{CONFIG["plotting"][da_grid.name]["limits"][freq]}')  # FixMe: remove this

                            # plot kwargs ----------------------------------------------------------------------------
                            plot_kwargs_grid = {
                                **CONFIG["plotting"][freq]["plot_kwargs_grid"],
                                **CONFIG["plotting"][da_grid.name]["limits"][freq],
                            }
                            plot_kwargs_grid = plot_kwargs_grid | {} if 'year' in freq else plot_kwargs_grid | {
                                "col": freq,
                                "col_wrap": CONFIG["plotting"][freq]["col_wrap"],
                            }
                            plot_kwargs_grid['cbar_kwargs'].setdefault('ticks', ticks)
                            # cbar_label = da_grid.attrs['units'] if (
                            #         'linregress' not in da_grid.name) else da_grid.attrs['units'] + ' per decade'
                            # cbar_label = da_grid.attrs['units'] if (
                            #         'linregress' not in da_grid.name and
                            #         'growing' in da_grid.name) else 'GDD per decade'
                            # @ToDo: cbar_label could be in the config file!
                            # We made sure station data has the same units as the grid data
                            cbar_label = da_grid.attrs['units']
                            cbar_label = 'GGD' if 'growing' in da_grid.name else cbar_label
                            cbar_label = cbar_label + ' per decade' if 'linregress' in da_grid.name else cbar_label
                            plot_kwargs_grid['cbar_kwargs'].setdefault('label', cbar_label)
                            # print(f"Raw ID: ..................... {da_grid.name} --- {plot_id} --- cbar_label: {cbar_label}")
                            # continue

                            # gridmap kwargs -------------------------------------------------------------------------
                            gridmap_kwargs = {
                                "projection": projection,
                                "transform": ccrs.PlateCarree(),
                                "divergent": CONFIG["plotting"][da_grid.name]["divergent"],
                                "levels": levels,
                                **CONFIG["plotting"]["gridmap_kwargs"],
                            }

                            # let's get to work plotting -------------------------------------------------------------
                            # do the facetgrid plot
                            with xr.set_options(keep_attrs=True):
                                fax = fg.gridmap(
                                    data=da_grid.sel(sel_kwargs) * scale_factor,
                                    plot_kw=plot_kwargs_grid,
                                    fig_kw=fig_kwargs,
                                    **gridmap_kwargs,
                                )

                                # plt.show(block=False)

                                # prepare kwargs for station data ToDo: Simplify this, too
                                plot_kwargs_station = {
                                                          k: plot_kwargs_grid[k]
                                                          for k in ['x', 'y', 'vmin', 'vmax', 'xlim', 'ylim']
                                                      } | {
                                                          'edgecolors': '#FFF6FF',
                                                          # pale pink, '#F8F8E7' bright yellow,  # '#6F6F6F' grey,
                                                          'marker': 'o',
                                                          'linewidths': 0.5,
                                                          **CONFIG["plotting"][freq]["plot_kwargs_station"],
                                                          'add_colorbar': False,
                                                          'label': 'AHCCD-Stations'
                                                      }
                                scattermap_kwargs = {k: gridmap_kwargs[k] for k in
                                                     ['transform', 'divergent', 'levels', 'frame']}
                                # iterate over facets to add hatching for trends and station data
                                for ax, sel_kwarg in zip(
                                        fax.axs.flat if isinstance(fax, xarray.plot.FacetGrid) else [fax],
                                        [{}] if freq in 'year' else
                                        [{freq: f} for f in da_grid[freq].values]):
                                    title = ax.get_title()
                                    # add significance hatching for reanalysis when plotting linear trends
                                    legend_elements = []
                                    if 'linregress' in da_grid.name:
                                        sel_sig_kwargs = (sel_kwargs | sel_kwarg).copy()
                                        sel_sig_kwargs['linreg_param'] = 'pvalue'
                                        sig_gridmap_kwargs = {k: v for k, v in gridmap_kwargs.items()
                                                              if k in ['projection', 'transform', 'frame']}
                                        plt.rcParams['hatch.linewidth'] = 0.3
                                        plt.rcParams['hatch.color'] = '#6F6F6F'  # 'dimgray'
                                        hatches = ['////']
                                        ax = fg.hatchmap(
                                            data=da_grid.sel(sel_sig_kwargs).where(da_grid.sel(sel_sig_kwargs) > 0.05),
                                            ax=ax,
                                            plot_kw={'hatches': hatches,
                                                     'x': 'lon',
                                                     'y': 'lat',
                                                     'xlim': plot_kwargs_grid['xlim'],
                                                     'ylim': plot_kwargs_grid['ylim'],
                                                     },
                                            **sig_gridmap_kwargs,
                                            legend_kw=False,
                                        )
                                        legend_elements.append(
                                            matplotlib.patches.Patch(hatch=hatches[0],
                                                                     color='#333333',
                                                                     fill=False,
                                                                     label='non-significant'
                                                                     )
                                        )
                                    # add station data
                                    if da_station is not None:
                                        data = da_station.sel(sel_kwargs | sel_kwarg) * scale_factor
                                        if 'linregress' not in da_station.name:
                                            plot_kwargs_station['marker'] = 'o'
                                            plot_kwargs_station['label'] = 'AHCCD-Stations'
                                            ax = fg.scattermap(
                                                data=data,
                                                ax=ax,
                                                plot_kw=plot_kwargs_station,
                                                **scattermap_kwargs,
                                                # cmap=[c.cmap for c in ax.get_children() if hasattr(c, 'cmap')][0],
                                            )
                                            legend_elements.append(
                                                matplotlib.lines.Line2D([0], [0],
                                                                        linestyle='none',
                                                                        marker=plot_kwargs_station['marker'],
                                                                        markerfacecolor='none',
                                                                        markeredgecolor='#333333',
                                                                        label=plot_kwargs_station['label']),
                                            )
                                        else:
                                            data_non_sig = data.where(da_station.sel(sel_sig_kwargs) > 0.05).dropna(
                                                'station')
                                            plot_kwargs_station['marker'] = 'X'
                                            plot_kwargs_station['label'] = 'AHCCD-Stations\n(non-significant)'
                                            if data_non_sig.size > 0:
                                                ax = fg.scattermap(
                                                    data=data_non_sig,
                                                    ax=ax,
                                                    plot_kw=plot_kwargs_station,
                                                    **scattermap_kwargs,
                                                )
                                            legend_elements.append(
                                                matplotlib.lines.Line2D([0], [0],
                                                                        linestyle='none',
                                                                        marker=plot_kwargs_station['marker'],
                                                                        markerfacecolor='none',
                                                                        markeredgecolor='#333333',
                                                                        label=plot_kwargs_station['label'])
                                            )
                                            data_sig = data.where(da_station.sel(sel_sig_kwargs) <= 0.05).dropna(
                                                'station')
                                            plot_kwargs_station['marker'] = 'o'
                                            plot_kwargs_station['label'] = 'AHCCD-Stations\n(significant)'
                                            if data_sig.size > 0:
                                                ax = fg.scattermap(
                                                    data=data_sig,
                                                    ax=ax,
                                                    plot_kw=plot_kwargs_station,
                                                    **scattermap_kwargs,
                                                )
                                            legend_elements.append(
                                                matplotlib.lines.Line2D([0], [0],
                                                                        linestyle='none',
                                                                        marker=plot_kwargs_station['marker'],
                                                                        markerfacecolor='none',
                                                                        markeredgecolor='#333333',
                                                                        label=plot_kwargs_station['label'])
                                            )
                                    ax.set_title('')
                                    facet_label = title if 'year' not in freq else 'ANN'
                                    ax.text(x=0.8, y=0.93, s=facet_label, fontsize=14, fontweight='bold',
                                            transform=ax.transAxes)
                                    ax.set_extent(
                                        plot_kwargs_station['xlim'] +
                                        plot_kwargs_station['ylim'],
                                        crs=ccrs.PlateCarree()
                                    )

                                fig = fax.fig if isinstance(fax, xarray.plot.FacetGrid) else fax.axes.figure
                                if legend_elements:
                                    legend_elements.reverse()
                                    ax.legend(
                                        handles=legend_elements,
                                        **CONFIG["plotting"][freq]["legend_kwargs"]
                                    )
                                fig.subplots_adjust(**CONFIG["plotting"][freq]["subplots_adjust_kwargs"])
                                title = '\n'.join(wrap(plot_id[:1].upper() + plot_id[1:], len(plot_id) / 2 + 5))
                                # CONFIG["plotting"][freq]["suptitle_wrap"]))
                                fig.suptitle(title, **CONFIG["plotting"][freq]["suptitle_kwargs"])
                            # plt.show(block=False)

                            # prepare file_name, directory and save -------------------------------------------------
                            changes = {
                                'standard deviation': 'std',
                                '- ': '',
                                '(': '',
                                ')': '',
                                ' ': '_',
                                '>': 'gt',
                                '°C': 'degC',
                            }
                            file_name = plot_id
                            for k, v in changes.items():
                                file_name = file_name.replace(k, v)
                            cur = {
                                "processing_level": "test_figures",
                                "horizon": horizon,
                                "freq": freq,
                                "file_name": file_name,
                            }
                            out_file = Path(f"{CONFIG['paths']['figures']}".format(**cur))
                            if not out_file.parent.exists():
                                out_file.parent.mkdir(parents=True, exist_ok=True)

                            # save to png
                            logger.info(f"Saving {out_file} ...")
                            fig.savefig(out_file, **CONFIG["plotting"]["savefig_kwargs"])
                            plt.close(fig)
                        except Exception as e:
                            logger.error(f"Error plotting {da_grid.name} for {key_input} ({horizon}): {e}",
                                         exc_info=True)
                            # raise e
                            continue

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
        msg="Congratulations! All tasks of the workflow were completed!",
    )
