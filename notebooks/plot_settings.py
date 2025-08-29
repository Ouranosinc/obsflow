import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature

import os
import xscen
from xscen.config import CONFIG
# Load configuration
xscen.load_config(
    "../"+"paths_obs.yml", "../"+"config_obs.yml", verbose=(__name__ == "__main__"), reset=True
)

PROJECTION = ccrs.LambertConformal()
EXTENT = [-80, -60, 45.5, 60.5]
SCATTER_SIZE = 5
SCATTER_EDGE = 2
PAGE_WIDTH = 8.5*2
PAGE_HEIGHT = 11*2
PATH = CONFIG["paths"]["figures"]
DPI=300

SOURCE_COLORS = {
    "CaSR v2.1": "#5DADE2",  # medium blue
    "CaSR v3.1": "#D35454",   # deeper red
    "EMDNA": "#E6A700",      # darker amber
    "ERA5-Land": "#B266B2",  # medium purple
    "PCICBlend": "#66C266"  # deeper green
}

import geopandas as gpd
REGIONS = gpd.read_file("../"+CONFIG["regional_mean"]["region"]["shape"])

def make_standardizer(region_kw=None, coastline_kw=None):
    default_region_kw = {
        "edgecolor": "dimgray",
        "facecolor": "none",
        "linewidths": 0.4,
        "zorder": -1,
    }
    default_coastline_kw = {
        "color": "gray",
        "linewidths": 0.4,
        "zorder": -2,
    }

    region_kw = {**default_region_kw, **(region_kw or {})}
    coastline_kw = {**default_coastline_kw, **(coastline_kw or {})}

    region_feature = ShapelyFeature(REGIONS.boundary, crs=ccrs.PlateCarree(), **region_kw)

    def standardize(ax):
        ax.coastlines(**coastline_kw)
        ax.add_feature(region_feature)
        ax.set_extent(EXTENT)

    return standardize

def save_plot(figure, processing_level, horizon, freq, file_name):
    path = PATH.format(processing_level=processing_level, horizon=horizon, freq=freq, file_name=file_name)
    
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    figure.savefig(path, dpi=DPI, bbox_inches="tight")

