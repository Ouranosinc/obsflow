import pytest
import xscen as xs
from xscen.config import CONFIG
from pathlib import Path
from clisops.core.subset import subset_bbox
import xarray as xr
import pandas as pd
import random


@pytest.mark.parametrize('processing_level', ['extracted', 'indicators', 'climatology'])
@pytest.mark.parametrize('xrfreq', ['AS-JAN', "QS-DEC", "MS", "D"])
def test_processing_levels(pcat, processing_level, xrfreq, ref_path, bbox):

    cat = pcat.search(processing_level=processing_level, xrfreq=xrfreq)

    # run test for random dates during the period 1981-2010, or the period itself for climatology
    for p in cat.df.path.values:

        # the obsflow output file
        ds_path = Path(p)
        ds = xr.open_dataset(ds_path, engine='zarr')
        ds_sub = subset_bbox(ds, **bbox).drop_vars('crs', errors='ignore')
        ds.close()

        fname_ref = (ref_path / processing_level /
                     f'{ds_path.stem}_bbox_{bbox["lat_bnds"][0]}_{bbox["lat_bnds"][1]}'
                     f'_{bbox["lon_bnds"][0]}_{bbox["lon_bnds"][1]}.zarr')

        # the reference file
        # maintain an option to create reference files
        create_ref = False
        if create_ref:
            # create reference files in the ref_path (might create new reference files from ds_path)
            ds_ref = xr.open_dataset(ref_path / processing_level / ds_path.name, engine='zarr')
            ds_ref_sub = subset_bbox(ds_ref, **bbox).drop_vars('crs', errors='ignore')
            # following https://github.com/pydata/xarray/issues/3476
            for v in list(ds_ref_sub.coords.keys()):
                if ds_ref_sub.coords[v].dtype == object:
                    ds_ref_sub[v].encoding.clear()
            for v in list(ds_ref_sub.variables.keys()):
                if ds_ref_sub[v].dtype == object:
                    ds_ref_sub[v].encoding.clear()
            ds_ref_sub.to_zarr(fname_ref, mode='w')
            ds_ref.close()
        else:
            ds_ref_sub = xr.open_dataset(fname_ref, engine='zarr').drop_vars('crs', errors='ignore')

        # select random dates for the test
        time_sel = []
        while len(time_sel) < 3:
            if processing_level == 'climatology':
                time_sel.append({'horizon': '1981-2010'})
                break
            else:
                date = pd.to_datetime(random.choice(ds_sub.time.values))
                if 1981 <= date.year <= 2010:
                    time_sel.append({'time': date})

        # test the reference file against the obsflow output file
        for var in ds_sub.data_vars.keys():
            for t_sel in time_sel:
                # print(f"Testing file {ds_path.name} doing {var} for {t_sel}")
                xr.testing.assert_allclose(
                    ds_ref_sub[var].sel(**t_sel),
                    ds_sub[var].sel(**t_sel),
                    rtol=1e-5,
                    atol=1e-5,
                )
