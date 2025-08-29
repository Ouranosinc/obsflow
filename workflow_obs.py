"""Workflow to extract obs data."""
import atexit
import logging
import os

import numpy as np
import xarray as xr

from dask import config as dskconf
from dask.distributed import Client

import xclim
from xclim.core import units

import xscen as xs
from xscen.config import CONFIG

import xsdba

import geopandas as gpd
from shapely import Polygon, MultiPolygon

# Load configuration
xs.load_config(
    "paths_obs.yml", "config_obs.yml", verbose=(__name__ == "__main__"), reset=True
)

# get logger
if "logging" in CONFIG:
    logger = logging.getLogger("xscen")

def clean_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Clean dataset for Zarr saving: fix encodings and rechunk any multi-chunk variable."""
    for var in ds.variables:
        ds[var].encoding.pop("chunks", None)

        da = ds[var]
        if hasattr(da, "chunks") and da.chunks is not None:
            # If any dimension has multiple chunks, rechunk fully
            if any(len(dim_chunks) > 1 for dim_chunks in da.chunks):
                ds[var] = da.chunk({dim: -1 for dim in da.dims})

    return ds

# Load region file
gdf = gpd.read_file(CONFIG["regional_mean"]["region"]["shape"])

def _remove_small_parts(geom, minarea):
    def p(p):
        holes = [i for i in p.interiors if i.area >= minarea]
        return Polygon(shell=p.exterior, holes=holes)

    def mp(mp):
        return MultiPolygon([p(i) for i in mp.geoms if i.area >= minarea])

    if isinstance(geom, Polygon):
        return p(geom)
    elif isinstance(geom, MultiPolygon):
        return mp(geom)
    else:
        return geom

# Adjusting shapes for xs.aggregate.spatial_mean
gdf["geometry"] = gdf.geometry.apply(_remove_small_parts, minarea=0.001).simplify(tolerance=0.01).buffer(0).segmentize(1)

if __name__ == "__main__":
    # set dask  configuration
    daskkws = CONFIG["dask"].get("client", {})
    dskconf.set(**{k: v for k, v in CONFIG["dask"].items() if k != "client"})

    # set xclim config to compute indicators on 3H data FixMe: can this be removed?
    xclim.set_options(**CONFIG["set_options"])

    # set email config
    if "scripting" in CONFIG:
        atexit.register(xs.send_mail_on_exit, subject=CONFIG["scripting"]["subject"])

    # initialize Project Catalog (only do this once, if the file doesn't already exist)
    if not os.path.exists(CONFIG["paths"]["project_catalog"]):
        pcat = xs.ProjectCatalog.create(
            CONFIG["paths"]["project_catalog"],
            project=CONFIG["project"],
        )
        # add a column to allow searching during performance task
        pcat.df['performance_base']=[np.nan]*len(pcat.df)
        pcat.update()

    # load project catalog
    pcat = xs.ProjectCatalog(CONFIG["paths"]["project_catalog"])



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
                        
                        # Renaming sources
                        source_remap = type_dict.get("source_renaming", {})
                        source_val = ds.attrs.get("cat:source")
                        if source_val in source_remap:
                            ds.attrs["cat:source"] = source_remap[source_val]

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

                ds_input = ds_input.unify_chunks()

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
                        
                        ds_ind = clean_for_zarr(ds_ind)

                        for var in ds_ind.data_vars:
                            da_ind = ds_ind[var]
                            if "units" in da_ind.attrs and da_ind.attrs["units"] == "K":
                                ds_ind[var] = units.convert_units_to(da_ind, "degC")

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
                            all_horizons.append(ds_mean)

                            # Calculate interannual standard deviation, skipping intra-[freq] std --------------------
                            logger.info(f"- Computing interannual standard deviation for {key_input} for period {period}")
                            # exclude intra-[freq] standard deviation
                            ds_input_std = ds_input[[v for v in CONFIG["aggregate"]["vars_for_interannual_std"]
                                                     if v in ds_input.data_vars]]

                            ds_std = xs.aggregate.climatological_op(
                                ds=ds_input_std,
                                **CONFIG["aggregate"]["climatological_std"],
                                periods=period,
                                rename_variables=True,
                                horizons_as_dim=True,
                            )
                            all_horizons.append(ds_std)

                    logger.info(f"Merging climatology of periods for {key_input}")
                    ds_clim = xr.merge([ds.drop_vars('time') for ds in all_horizons], combine_attrs='override')
                    
                    # Saving dataset by var
                    for var_name in ds_clim.data_vars:
                        logger.info(f"Saving climatology for variable {var_name}")

                        ds_var = ds_clim[[var_name]].copy()
                        ds_var.attrs.update(ds_clim.attrs)
                        ds_var.attrs["cat:variable"] = var_name

                        xs.save_and_update(
                            ds=ds_var,
                            pcat=pcat,
                            path=CONFIG['paths']['task']
                        )

    # --- PERFORMANCE ---

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
                        if pcat.exists_in_cat(id=rec_dataset.attrs["cat:id"], processing_level="performance",
                                              performance_base=obs_dataset.attrs["cat:id"],
                                              variable=performance_variable_name):
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
                            drec=xs.utils.stack_drop_nans(rec_dataset,
                                mask = rec_dataset.isel(time=slice(1,-1)).notnull().any(dim="time").compute())
                            drec = xs.spatial.subset(
                                drec,
                                method='gridpoint',
                                lat=station_lats,
                                lon=station_lons
                            )

                            # put back the unique coords of the obs on the rec
                            drec=drec.rename({'site':'station'})
                            station_coords=set(obs_dataset.coords) - set(drec.coords)
                            for c in station_coords:
                                drec.coords[c]=obs_dataset.coords[c]

                            # unstack date to have one stat per season
                            drec=xs.utils.unstack_dates(drec)
                            dobs=xs.utils.unstack_dates(obs_dataset)

                            # Select time period from config
                            start_year, end_year = CONFIG['performance']['period']
                            time_slice = slice(str(start_year), str(end_year))

                            # Apply the time slice to both datasets
                            dobs = dobs.sel(time=time_slice)
                            drec = drec.sel(time=time_slice)

                            # Make a mask for stations with at least n years of data
                            min_years = CONFIG["performance"]["minimum_years"]
                            valid_mask = (dobs.count(dim='time')>=min_years).compute()

                            # Drop stations that don’t meet the requirement
                            dobs = dobs.where(valid_mask, drop=True)
                            drec = drec.where(valid_mask, drop=True)

                            # Rechunk both timeseries into a single chunk each
                            dobs = dobs.chunk({"time": -1})
                            drec = drec.chunk({"time": -1})

                            ## Computing the performance metric ##
                            da_output = statistic_func( # The output data array
                                sim=drec[variable_name],
                                ref=dobs[variable_name]
                            )
                            ds_output = da_output.to_dataset(name=performance_variable_name) # The output dataset

                            ds_output.attrs=rec_dataset.attrs # inherit most attrs from the rec input
                            ds_output.attrs["cat:xrfreq"] = "fx"
                            ds_output.attrs["cat:variable"] = performance_variable_name
                            ds_output.attrs["cat:processing_level"] = "performance"
                            ds_output.attrs["cat:performance_base"] = obs_dataset.attrs["cat:id"]


                            del ds_output.station.encoding['filters'] # Existing value in encoding's "filters" breaks "save_and_update"
                            
                            xs.save_and_update(
                                ds=ds_output,
                                pcat=pcat,
                                path=CONFIG['paths']['task'],
                                save_kwargs=CONFIG["performance"]["save"],
                            )


    # --- REGIONAL MEAN ---
    if "regional_mean" in CONFIG["tasks"]:
        regions = [(gdf[gdf["id"] == row.id], row.name) for row in gdf.itertuples(index=False)]

        for search_param in CONFIG["regional_mean"]["search_params"]:
            dict_input = pcat.search(**search_param, processing_level='performance').to_dataset_dict(**tdd)
            if pcat.exists_in_cat(id='multiple', processing_level="regional_mean",variable=search_param['variable']):
                logger.info(f"Skipping existing regional mean for: {search_param['variable']})")
                continue

            source_datasets = [] # The datasets whose variables have been averaged

            for dataset_id, ds_input in dict_input.items():
                source_name = ds_input.attrs["cat:source"]

                region_means = [] # The regional averages of variables in the current "source_dataset"
                for region_shape, region_name in regions:
                    try: 
                        ds_sub = xs.spatial.subset(
                            ds_input,
                            method="shape",
                            shape=region_shape,
                            tile_buffer=0,
                        )
                    except ValueError as e:
                        logger.warning(f"Error subsetting region '{region_name}' for source '{source_name}': {e}")
                        if "No grid cell centroids found" in str(e):
                            logger.info(f"Skipping region '{region_name}' for source '{source_name}' - no data found.")
                            continue
                        raise
                    ds_sub = ds_sub.drop_vars("crs", errors="ignore") # Removing Coordinate Reference System info

                    # Compute mean
                    ds_mean = ds_sub.mean(dim="station", skipna=True, keep_attrs=True)
                    ds_mean = ds_mean.expand_dims({"region": [region_name]})

                    # Compute count (shared coordinate)
                    nstation = ds_sub.count(dim="station")
                    nstation = nstation.expand_dims({"region": [region_name]})
                    nstation = nstation.to_array().mean("variable")  # collapse across vars if needed
                    nstation.name = "nstation"

                    # Attach nstation as shared coordinate
                    for var in ds_mean.data_vars:
                        ds_mean[var] = ds_mean[var].assign_coords(nstation=nstation)

                    region_means.append(ds_mean)

                if not region_means:
                    logger.warning(f"No data found in any region for source {source_name}. Skipping.")
                    continue

                ds_source = xr.concat(region_means, dim="region").expand_dims({"source": [source_name]})
                source_datasets.append(ds_source)

            if not source_datasets:
                logger.warning(f"No sources available for search_param {search_param}. Skipping.")
                continue

            # Keep only common coordinates across all source datasets
            common_coords = set.intersection(*(set(ds.coords) for ds in source_datasets))
            source_datasets = [
                ds.drop_vars(set(ds.coords) - common_coords, errors="ignore")
                for ds in source_datasets
            ]

            combined_ds = xr.concat(source_datasets, dim="source")

            # Setting attributes for the new dataset
            combined_ds.attrs["cat:processing_level"] = "regional_mean"
            combined_ds.attrs["cat:source"] = "multiple"
            combined_ds.attrs["cat:id"] = "multiple"
            combined_ds.attrs["cat:domain"] = ds_input.attrs["cat:domain"]
            combined_ds.attrs["cat:xrfreq"] = "fx"
            combined_ds.attrs["cat:variable"]= ds_input.attrs["cat:variable"]

            combined_ds = clean_for_zarr(combined_ds)
            xs.save_and_update(
                ds=combined_ds,
                pcat=pcat,
                path=CONFIG['paths']['task']
            )

    # --- COHERENCE ---
    if "coherence" in CONFIG["tasks"]:
        logger.info("Started coherence task")
        for search_param_dict in CONFIG["coherence"]["search_params"]:
            logger.info(f"Checking: {search_param_dict}")
            ds_dict = pcat.search(**search_param_dict).to_dataset_dict()

            variable = search_param_dict["variable"]
            
            # Check if this coherence dataset already exists
            if pcat.exists_in_cat(
                id="multiple",
                source="multiple",
                processing_level="coherence",
                variable=variable
            ):
                logger.info(f"Skipping: {search_param_dict}")
                continue
            
            spatial_means = []

            for name, ds in ds_dict.items():
                # Clean variable attributes
                ds[variable].attrs.pop("grid_mapping", None)

                # Compute spatial mean
                ds_spatial_mean = xs.aggregate.spatial_mean(
                    ds,
                    method="xesmf",
                    region={"method": "shape", "shape": gdf},
                    kwargs={"skipna": True},
                )

                ds_spatial_mean[variable].attrs = ds[variable].attrs.copy() # Copy over the attributes from the original data array (units, etc)

                ds_spatial_mean = ds_spatial_mean.rename({"geom": "region"}) # TODO: once xscen updates, remove this line and add {"geom_dim_name": "region"} in the kwargs dict

                # Drop bounds if present (additional information on lat,lon and/or rlat,rlon)
                ds_spatial_mean = ds_spatial_mean.drop_dims("bounds", errors="ignore")

                # Source name
                src = ds_spatial_mean.attrs.get("cat:source", name)

                spatial_means.append(ds_spatial_mean.expand_dims(source=[src]))

            # Concatenate across sources
            ds_all = xr.concat(spatial_means, dim="source")

            # Set region dimension to use region names instead of integer indices
            ds_all = ds_all.assign_coords(region=("region", ds_all["name"].values))

            # Set attributes
            ds_all.attrs.update({
                "cat:processing_level": "coherence",
                "cat:source": "multiple",
                "cat:id": "multiple",
                "cat:xrfreq": "fx",
                "cat:variable": variable,
            })

            ds_all = clean_for_zarr(ds_all)
            xs.save_and_update(
                ds=ds_all,
                pcat=pcat,
                path=CONFIG['paths']['task']
            )






    xs.send_mail(
        subject="ObsFlow - Message",
        msg="Congratulations! All tasks of the workflow were completed!",
    )

