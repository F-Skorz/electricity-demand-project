## #############################
###############################
##                                                          ##
##     src/filter_group_aggr_utils.py       ##
##                                                          ##
###############################
###############################

from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_datetime64tz_dtype,
    is_timedelta64_dtype,
    is_extension_array_dtype,
)
from typing import Sequence, Tuple, List, Iterable
from typing import Callable, Iterable, Optional, Dict, Any
import numpy as np



#############################
##      filter weekdays                        ##
############################# 


def make_weekday_mask(
    df: pd.DataFrame,
    ts_col: str,
    weekdays: Iterable[int] = (0, 1, 2, 3, 4),
    *,
    treat_as_hour_end: bool = True,
    epsilon: pd.Timedelta | str = "1ns",
) -> pd.Series:
    """
    Build a boolean mask selecting rows whose weekday—as defined by `ts_col`—is in `weekdays`.
    Weekday encoding: Monday=0, ..., Sunday=6.

    Hour-ending convention
    ----------------------
    If your timestamps label the *end* of the hour (OPSD/ENTSO-E style), set
    `treat_as_hour_end=True` (default). This computes the weekday at `t - epsilon`
    (inside the hour [t-1h, t)) so an hour ending at local midnight counts toward
    the *preceding* local day. `epsilon` defaults to 1ns to avoid DST pitfalls.

    Assumptions
    -----------
    - `ts_col` is a datetime-like Series (tz-aware or naive). Use the calendar you care about:
      e.g., pass a local-time column like `cet_cest_timestamp` if you want German weekdays.

    Parameters
    ----------
    df : pd.DataFrame
    ts_col : str
        Name of the timestamp column to inspect.
    weekdays : Iterable[int], default (0..4)
        Allowed weekdays (Mon=0 .. Sun=6).
    treat_as_hour_end : bool, default True
        If True, use `timestamp - epsilon` before computing weekday.
    epsilon : pd.Timedelta | str, default "1ns"
        Tiny offset applied when `treat_as_hour_end=True`.

    Returns
    -------
    pd.Series (bool)
        True where the row's (adjusted) weekday is in `weekdays`. NaT -> False.

    Raises
    ------
    KeyError, TypeError
    """
    if ts_col not in df.columns:
        raise KeyError(f"make_weekday_mask: column '{ts_col}' not found.")
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        raise TypeError(f"make_weekday_mask: '{ts_col}' is not datetime-like.")

    ts = df[ts_col]
    if treat_as_hour_end:
        eps = pd.Timedelta(epsilon)
        ts = ts - eps  # step just inside the hour ([t-1h, t))

    wk = ts.dt.weekday  # Monday=0 .. Sunday=6
    wset = set(weekdays)
    return wk.isin(wset).fillna(False)





def _coerce_boundary_to_ts_tz(ts: pd.Series, dt) -> pd.Timestamp:
    """Coerce a user-provided start/end into the timezone of `ts` for safe comparison."""
    out = pd.to_datetime(dt)
    if ts.dt.tz is not None:
        # Make tz-aware in the same tz as ts if naive; otherwise convert
        out = out.tz_localize(ts.dt.tz) if out.tzinfo is None else out.tz_convert(ts.dt.tz)
    return out

#############################
##     get  local DST switch dates       ##
############################# 

def get_local_dst_switch_dates(df: pd.DataFrame, ts_col: str) -> set:
    """
    Return dates (as datetime.date) that are DST transition days for the given local tz series.
    Only meaningful if `ts_col` is local time (e.g., 'cet_cest_timestamp'). It counts the
    hours of a local calender day and returns the days with a count not equal to 24 
    as switch days. 
    """
    s = df[ts_col]
    if s.dt.tz is None:
        return set()
    # Count distinct local hours per calendar day; 23 or 25 implies a DST switch day.
    per_day = s.groupby(s.dt.date).apply(lambda x: x.dt.hour.nunique())
    return set(per_day.index[(per_day != 24)].tolist())


def aggregate_hourly_profile(
    df: pd.DataFrame,
    value_col: str,
    start,
    end,
    *,
    ts_col: str,
    agg: str = "mean",
    mask: Optional[pd.Series] = None,
    mask_fn: Optional[Callable[[pd.DataFrame, str], pd.Series]] = None,
    drop_dst_switch_dates: bool = True,
    return_counts: bool = True,
    return_std: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute an hourly (0–23) aggregated profile of `value_col` over rows that match an optional mask,
    using day/hour boundaries defined by `ts_col` ('utc_timestamp' or 'cet_cest_timestamp').

    - Filters to [start, end] in the timezone of `ts_col`.
    - If `ts_col` is local (e.g., CET/CEST) and `drop_dst_switch_dates=True`, excludes DST switch days.
    - Groups by `ts_col.dt.hour` and aggregates per  `agg`  method(e.g., 'mean','median','sum').

    Returns
    -------
    profile_df : DataFrame with columns:
        hour (int 0..23), value (float), and optionally n, std
    meta : dict with keys:
        timestamp_used, date_range, agg, excluded_dates (list[str]), notes
    """
    if ts_col not in df.columns:
        raise KeyError(f"The function aggregate_hourly_profile  could not find the column '{ts_col}'.")
    if value_col not in df.columns:
        raise KeyError(f"The function aggregate_hourly_profile could not find the column '{value_col}'.")
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        raise TypeError(f"The function aggregate_hourly_profile does not recognize  '{ts_col}' as datetime-like.")

    # Time window filtering in the tz of ts_col
    ts = df[ts_col]
    start_ts = _coerce_boundary_to_ts_tz(ts, start)
    end_ts   = _coerce_boundary_to_ts_tz(ts, end)
    in_window = (ts >= start_ts) & (ts <= end_ts)

    # Mask handling
    if mask is not None:
        if len(mask) != len(df):
            raise ValueError("The function aggregate_hourly_profile found: The provided mask's length does not match DataFrame's length. ")
        mask_eff = mask.fillna(False)
        mask_desc = "provided mask"
    elif mask_fn is not None:
        mask_eff = mask_fn(df, ts_col).fillna(False)
        mask_desc = getattr(mask_fn, "__name__", "mask_fn")
    else:
        mask_eff = pd.Series(True, index=df.index)
        mask_desc = "no mask"
    
    # combine time-window and mask selection
    sel = in_window & mask_eff
    # reduce DataFrame to sel rows, and ts_col, value_col 
    work = df.loc[sel, [ts_col, value_col]].copy()

    # Optionally drop DST switch dates (only meaningful for local tz)
    excluded_dates: list[str] = []
    if drop_dst_switch_dates and work[ts_col].dt.tz is not None:
        # If the series is local (like CET/CEST), drop days with != 24 unique hours.
        # For UTC (fixed offset), get_local_dst_switch_dates will typically return empty.
        switch_dates = get_local_dst_switch_dates(work, ts_col)
        if switch_dates:
            excluded_dates = [d.isoformat() for d in sorted(switch_dates)]
            work = work[~work[ts_col].dt.date.isin(switch_dates)]

    if work.empty:
        # Return an empty 0..23 frame for consistency
        out = pd.DataFrame({"hour": np.arange(24, dtype=int), "value": np.nan})
        meta = {
            "timestamp_used": ts_col,
            "date_range": (start_ts.isoformat(), end_ts.isoformat()),
            "agg": agg,
            "excluded_dates": excluded_dates,
            "mask_desc": mask_desc,
            "notes": "No rows after filtering.",
        }
        return out, meta

    work["hour"] = work[ts_col].dt.hour

    # Aggregate
    grouped = work.groupby("hour")[value_col]
    agg_series = getattr(grouped, agg)() if hasattr(grouped, agg) else grouped.aggregate(agg)
    out = agg_series.rename("value").reindex(range(24)).reset_index()

    if return_counts:
        out = out.merge(grouped.size().rename("n").reindex(range(24)).reset_index(), on="hour", how="left")
    if return_std:
        out = out.merge(grouped.std(ddof=1).rename("std").reindex(range(24)).reset_index(), on="hour", how="left")

    meta = {
        "timestamp_used": ts_col,
        "date_range": (start_ts.isoformat(), end_ts.isoformat()),
        "agg": agg,
        "excluded_dates": excluded_dates,
        "mask_desc": mask_desc,
        "notes": "",
    }
    return out, meta


__all__ = ["make_weekday_mask", 
               "get_local_dst_switch_dates"
                "aggregate_hourly_profile"
]
