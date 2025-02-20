from __future__ import annotations

import numpy as np
import xarray as xr
from xclim.core.units import convert_units_to, declare_units
try:
    from xclim.core import Quantified, DayOfYearStr
except ImportError:
    from xclim.core.utils import Quantified, DayOfYearStr
from xclim.indices import last_spring_frost, first_day_temperature_below


@declare_units(tasmin="[temperature]", thresh="[temperature]")
def frost_free_season(
    tasmin: xr.DataArray,
    thresh: Quantified = "0.0 degC",
    window: int = 2,
    mid_date: DayOfYearStr = "08-01",
    freq: str = "YS",
):
    r"""Warm season.

    The warm season starts the day after the last sequence of N days with temperature under the threshold, before a given date (last spring frost).
    It ends the day before the first sequence of N days under the threshold after a given date (first_day_temperature_below).
    Its length is the end minus the start.

    Parameters
    ----------
    tasmin : xarray.DataArray
        Minimum daily temperature.
    thresh : Quantified
        Threshold temperature on which to base evaluation.
    window : int
        Minimum number of consecutive days below the threshold defining a "frost".
    mid_date : DayOfYearStr
        A date that must be included in the season.
    freq : str
        Resampling frequency.

    Returns
    -------
    xarray.DataArray, [dimensionless]
        Day of the year when the frost free season starts.
    xarray.DataArray, [dimensionless]
        Day of the year when the frost free season ends.
    xarray.DataArray, [time]
        Length of the frost free season in days.
    """
    if freq != "YS":
        raise ValueError('This function only works with freq = YS')

    start = last_spring_frost(tasmin=tasmin, thresh=thresh, op='<=', before_date=mid_date, window=window, freq=freq)
    start = xr.where( # No start => first day, actual start is the day after the last frost
        start.isnull(),
        1,
        start + 1
    )

    end = first_day_temperature_below(tas=tasmin, thresh=thresh, op='<=', after_date=mid_date, window=window, freq=freq) - 1
    end = xr.where(  # No end  -> last day
        end.isnull(),
        end.time.dt.days_in_year,
        end - 1
    )

    # If they are equal or >, this means there was no frost free season
    no_frost_free = start > end
    start = start.where(~no_frost_free)
    end = end.where(~no_frost_free)
    length = end - start + 1
    start.attrs.update(units="", is_dayofyear=np.int32(1), calendar=tasmin.time.dt.calendar)
    end.attrs.update(units="", is_dayofyear=np.int32(1), calendar=tasmin.time.dt.calendar)
    length.attrs.update(units="days")
    return start, end, length
