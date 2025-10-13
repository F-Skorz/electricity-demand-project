###################
# src/plotting_utils.py    #
###################

"""
Plotting utilities for aggregating and visualizing time series.

Notes on granularity and pandas defaults
---------------------------------------
- Allowed granularities for downsampling: {"H", "D", "W", "M", "Y"}.
- Pandas resampling defaults:
    - W  : week ends on Sunday (W-SUN)
    - M  : month-end
    - Y  : Dec-end (A-DEC)
  label='left', closed='left'.
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional

__all__ = [
    "aggregate_timeseries",
    "plot_dual_timeseries",
]

# Time resolutions to downsample to
ALLOWED_GRANULARITY = {"H", "D", "W", "M", "Y"}
#   Defaults for pandas downsampling:
# - W  : week ends on Sunday (W-SUN)
# - M  : month-end
# - Y  : Dec-end (A-DEC)
# - label='left', closed='left'


# Replace _ensure_dt_index with this safer version
def _ensure_dt_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Return a copy of df with a UTC DatetimeIndex. Does not mutate input.

    Requirements
    -----------
    - If df already has a DatetimeIndex, it is converted/localized to UTC.
    - Otherwise, df must contain `time_col`; it will be parsed to UTC and set as index.
    """
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
        return out

    if time_col not in out.columns:
        raise KeyError(
            f"DataFrame has no DatetimeIndex and is missing the time column {time_col!r}."
        )

    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    out = out.set_index(time_col)
    return out


def aggregate_timeseries(
    df: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    column: str,
    granularity: str,
    time_col: str = "utc_timestamp",
    *,
    coverage_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Create an aggregated time series for `column` between `start_date` and `end_date`,
    indexed by `time_col` (default: "utc_timestamp").

    Steps
    -----
    1) Ensure a UTC DatetimeIndex using `time_col`.
    2) Slice the series to the closed interval [start_date, end_date].
    3) If `granularity` is not hourly, resample to the chosen frequency using the mean.
    4) Apply a coverage rule per bin: only keep the mean if coverage ≥ `coverage_threshold`
       (default 0.5); otherwise set the bin to NA.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing `time_col` and `column`.
    start_date, end_date : pd.Timestamp | str
        Date bounds (inclusive) for slicing.
    column : str
        Target column to aggregate.
    granularity : {"H","D","W","M","Y"}
        Resampling frequency. Must be one of ALLOWED_GRANULARITY.
    time_col : str, default "utc_timestamp"
        Timestamp column used to build/set the DatetimeIndex (UTC).
    coverage_threshold : float, default 0.5
        Minimum fraction of non-NA values within a resampling bin to keep the mean.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame named `column`, indexed by UTC timestamps at the requested
        granularity, with bins failing the coverage rule set to NA.

    Raises
    ------
    KeyError
        If `column` is missing.
    ValueError
        If `granularity` is not allowed, or the selected window is empty.
    """
    if column not in df.columns:
        raise KeyError(f"Column not found: {column!r}")
    if granularity not in ALLOWED_GRANULARITY:
        raise ValueError(f"granularity must be one of {sorted(ALLOWED_GRANULARITY)}")

    start, end = _coerce_dates(start_date, end_date)
    # Ensure datetime index on the full frame first
    x = _ensure_dt_index(df, time_col)
    # Now select the single column
    x = x[[column]]
    # Slice closed interval
    sliced = x.loc[start:end]
    if sliced.empty:
        raise ValueError(f"Selected window {start}–{end} has no rows for column {column!r}.")
    # Downsample with mean and apply coverage rule
    grouped = sliced.resample(granularity).agg({column: ["mean", "count", "size"]})
    mean = grouped[(column, "mean")].to_frame(name=column)
    count = grouped[(column, "count")]
    size = grouped[(column, "size")].clip(lower=1)
    coverage = (count / size).astype("float64")
    mean.loc[coverage < coverage_threshold, column] = pd.NA
    return mean


def plot_dual_timeseries(
    df: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    column_name_one: str,
    granularity_one: str,
    column_name_two: str,
    granularity_two: str,
    time_col: str = "utc_timestamp",
    *,
    coverage_threshold: float = 0.5,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plot two aggregated time series (possibly at different granularities) on twin y-axes.

    The function mirrors `aggregate_timeseries` for each column and plots the results on
    left and right y-axes, respectively. Returns the figure and both axes for further
    customization.

    Notes
    -----
    - Default anchoring for resampling: W→Sunday-end, M→month-end, Y→Dec-end (pandas defaults).
    - Different granularities imply different x-densities.
    - Raises if either slice is empty or a requested column is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing both columns and `time_col`.
    start_date, end_date : pd.Timestamp | str
        Date bounds (inclusive) for slicing both series.
    column_name_one : str
        First series column name.
    granularity_one : {"H","D","W","M","Y"}
        Resampling frequency for the first series.
    column_name_two : str
        Second series column name.
    granularity_two : {"H","D","W","M","Y"}
        Resampling frequency for the second series.
    time_col : str, default "utc_timestamp"
        Timestamp column used to build/set the DatetimeIndex (UTC).
    coverage_threshold : float, default 0.5
        Minimum fraction of non-NA values within a resampling bin to keep that bin’s mean.
    title : Optional[str], default None
        Optional custom figure title.

    Returns
    -------
    (fig, ax_left, ax_right) : Tuple[plt.Figure, plt.Axes, plt.Axes]
        Matplotlib figure and the two axes (left and right).

    Raises
    ------
    KeyError
        If a requested column is missing.
    ValueError
        If either aggregated time series turns out empty after slicing/aggregation.
    """
    # Validate columns early
    for col in (column_name_one, column_name_two):
        if col not in df.columns:
            raise KeyError(f"Column not found: {col!r}")

    ts1 = aggregate_timeseries(
        df, start_date, end_date, column_name_one, granularity_one, time_col, coverage_threshold=coverage_threshold
    )
    ts2 = aggregate_timeseries(
        df, start_date, end_date, column_name_two, granularity_two, time_col, coverage_threshold=coverage_threshold
    )

    if ts1.empty:
        raise ValueError(f"First time series is empty after slicing/aggregation: {column_name_one!r}.")
    if ts2.empty:
        raise ValueError(f"Second time series is empty after slicing/aggregation: {column_name_two!r}.")

    # Ensure both indices are UTC DatetimeIndex (aggregate_timeseries already enforces this)
    assert isinstance(ts1.index, pd.DatetimeIndex) and ts1.index.tz is not None
    assert isinstance(ts2.index, pd.DatetimeIndex) and ts2.index.tz is not None

    # Plot (no explicit colors to keep styling neutral)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(ts1.index, ts1.iloc[:, 0], label=f"{column_name_one} ({granularity_one})")
    ax1.set_xlabel(f"Time from {pd.to_datetime(start_date)} until {pd.to_datetime(end_date)}")
    ax1.set_ylabel(column_name_one)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(ts2.index, ts2.iloc[:, 0], label=f"{column_name_two} ({granularity_two})")
    ax2.set_ylabel(column_name_two)

    # Titles & legends
    fig.suptitle(title or f"{column_name_one} ({granularity_one})  vs  {column_name_two} ({granularity_two})", y=0.98)

    # Combine legends from both axes (optional but helpful)
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    return fig, ax1, ax2
