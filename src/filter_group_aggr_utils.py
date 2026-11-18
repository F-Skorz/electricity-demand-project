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
    If  timestamps label the *end* of the hour (OPSD/ENTSO-E convention),  
    setting
    `treat_as_hour_end=True` (default) 
    will make `.dt.weekday` return the weekday at `t - epsilon`.
    An hour ending at local midnight counts toward
    the *preceding* local day. `epsilon` defaults to 1ns.

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

def get_local_dst_switch_dates(df, ts_col, *, treat_as_hour_end=True, epsilon="1ns"):
    """
    Lokale Kalendertage (nach Hour-End-Konvention), an denen sich der UTC-Offset ändert.
    Muss auf dem vollständigen DataFrame laufen, nicht auf einem maskierten Teil.
    """
    s = df[ts_col]
    if s.dt.tz is None:
        return set()
    eff = s - pd.Timedelta(epsilon) if treat_as_hour_end else s
    local = eff.dt.tz_localize(None)                              # lokale Wanduhrzeit
    offset = local - eff.dt.tz_convert("UTC").dt.tz_localize(None)
    day = local.dt.normalize()
    n_off = offset.groupby(day).nunique()
    return set(n_off.index[n_off > 1].date)


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
    treat_as_hour_end: bool = True,
    epsilon: pd.Timedelta | str = "1ns",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Stündliches Profil (0-23). Bei treat_as_hour_end=True bezeichnet ein Stempel das ENDE
    der Stunde: Stunde, Tageszuordnung, Datumsfenster und DST-Tage beziehen sich auf
    (ts - epsilon). Fenster [start, end] gilt inklusive, als lokale Kalendertage.
    """
    for col in (ts_col, value_col):
        if col not in df.columns:
            raise KeyError(f"aggregate_hourly_profile: column '{col}' not found.")
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        raise TypeError(f"aggregate_hourly_profile: '{ts_col}' is not datetime-like.")

    # Effektive Zeit: die Stunde, auf die sich der Stempel bezieht
    ts = df[ts_col]
    eff = ts - pd.Timedelta(epsilon) if treat_as_hour_end else ts
    eff_local = eff.dt.tz_localize(None)          # lokale Wanduhrzeit (naiv)
    eff_day = eff_local.dt.normalize()

    # Fenster nach effektivem Kalendertag
    def _to_day(x):
        t = pd.Timestamp(x)
        if t.tzinfo is not None:
            t = t.tz_localize(None)
        return t.normalize()

    start_day, end_day = _to_day(start), _to_day(end)
    in_window = (eff_day >= start_day) & (eff_day <= end_day)

    # Maske
    if mask is not None:
        if len(mask) != len(df):
            raise ValueError("aggregate_hourly_profile: mask length does not match DataFrame length.")
        mask_eff = mask.fillna(False)
        mask_desc = "provided mask"
    elif mask_fn is not None:
        mask_eff = mask_fn(df, ts_col).fillna(False)
        mask_desc = getattr(mask_fn, "__name__", "mask_fn")
    else:
        mask_eff = pd.Series(True, index=df.index)
        mask_desc = "no mask"

    sel = in_window & mask_eff

    # DST-Tage auf dem VOLLEN df bestimmen, dann über effektiven Tag ausschließen
    excluded_dates: list[str] = []
    if drop_dst_switch_dates and ts.dt.tz is not None:
        switch_dates = get_local_dst_switch_dates(
            df, ts_col, treat_as_hour_end=treat_as_hour_end, epsilon=epsilon
        )
        if switch_dates:
            excluded_dates = [d.isoformat() for d in sorted(switch_dates)]
            sel = sel & ~eff_day.dt.date.isin(switch_dates)

    work = pd.DataFrame({"hour": eff_local.dt.hour[sel], value_col: df.loc[sel, value_col]})

    meta = {
        "timestamp_used": ts_col,
        "date_range": (start_day.date().isoformat(), end_day.date().isoformat()),
        "agg": agg,
        "excluded_dates": excluded_dates,
        "mask_desc": mask_desc,
        "hour_end_convention": treat_as_hour_end,
        "notes": "",
    }

    if work.empty:
        meta["notes"] = "No rows after filtering."
        return pd.DataFrame({"hour": np.arange(24, dtype=int), "value": np.nan}), meta

    grouped = work.groupby("hour")[value_col]
    agg_series = getattr(grouped, agg)() if hasattr(grouped, agg) else grouped.aggregate(agg)
    out = agg_series.rename("value").reindex(range(24)).rename_axis("hour").reset_index()

    if return_counts:
        out = out.merge(grouped.size().rename("n").reindex(range(24)).rename_axis("hour").reset_index(), on="hour", how="left")
    if return_std:
        out = out.merge(grouped.std(ddof=1).rename("std").reindex(range(24)).rename_axis("hour").reset_index(), on="hour", how="left")

    return out, meta


__all__ = ["make_weekday_mask", 
               "get_local_dst_switch_dates"
                "aggregate_hourly_profile"
]
