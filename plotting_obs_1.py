# plot_module.py

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import figanos.matplotlib as fg
from textwrap import wrap
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_subplots(da_grid, config, frequency, horizon):
    """Generate subplots based on the frequency and horizon."""
    projection = ccrs.LambertConformal()
    fig_kwargs = config[frequency]["fig_kwargs"]
    plot_kwargs_grid = config[frequency]["plot_kwargs_grid"]
    gridmap_kwargs = config["gridmap_kwargs"]

    if 'linspace' in config[da_grid.name]["ticks"]:
        levels = list(np.linspace(config[da_grid.name]["limits"][frequency]["vmin"],
                                  config[da_grid.name]["limits"][frequency]["vmax"],
                                  config[da_grid.name]["levels"]))
        ticks = levels[0::2]
    else:
        levels = config[da_grid.name]["levels"]
        ticks = config[da_grid.name]["ticks"]

    plot_kwargs_grid.update(config[da_grid.name]["limits"][frequency])
    plot_kwargs_grid['cbar_kwargs'].setdefault('ticks', ticks)
    plot_kwargs_grid['cbar_kwargs'].setdefault('label', da_grid.attrs['units'])

    gridmap_kwargs.update({
        "projection": projection,
        "transform": ccrs.PlateCarree(),
        "divergent": config[da_grid.name]["divergent"],
        "levels": levels,
    })

    if frequency == 'year':
        fig, ax = plt.subplots(subplot_kw={'projection': projection}, **fig_kwargs)
        fax = fg.gridmap(
            data=da_grid.sel(horizon=horizon) * 10 if 'linregress' in da_grid.name else da_grid.sel(horizon=horizon),
            plot_kw=plot_kwargs_grid,
            fig_kw=fig_kwargs,
            ax=ax,
            **gridmap_kwargs,
        )
    else:
        fax = fg.gridmap(
            data=da_grid.sel(horizon=horizon) * 10 if 'linregress' in da_grid.name else da_grid.sel(horizon=horizon),
            plot_kw=plot_kwargs_grid,
            fig_kw=fig_kwargs,
            **gridmap_kwargs,
        )

    return fax

def plot_data(da_grid, config, frequency, horizon):
    """Plot data using the provided configuration."""
    fax = generate_subplots(da_grid, config, frequency, horizon)
    fig = fax.fig if isinstance(fax, xr.plot.FacetGrid) else fax.axes.figure

    plot_id = (f"{config[da_grid.name]['label']} "
               f"{da_grid.attrs['long_name'].lower()} - "
               f"{da_grid.attrs['source']} ({horizon})")

    title = '\n'.join(wrap(plot_id[:1].upper() + plot_id[1:], len(plot_id) / 2 + 5))
    fig.suptitle(title, **config[frequency]["suptitle_kwargs"])

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
        "freq": frequency,
        "file_name": file_name,
    }
    out_file = Path(f"{config['paths']['figures']}".format(**cur))
    if not out_file.parent.exists():
        out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {out_file} ...")
    fig.savefig(out_file, **config["plotting"]["savefig_kwargs"])
    plt.close(fig)

    return fig