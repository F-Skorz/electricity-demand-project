##############################
# src/plotting_utils.py      #
##############################

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
from typing import Optional, Sequence, Tuple

__all__ = [
    "_coerce_dates",
    "_ensure_dt_index",
    "aggregate_timeseries",
    "plot_dual_timeseries",
     "plot_nan_count"
]

# Time resolutions to downsample to
ALLOWED_GRANULARITY = {"H", "D", "W", "M", "Y"}
# Pandas resampling defaults:
# - W  : week ends on Sunday (W-SUN)
# - M  : month-end
# - Y  : Dec-end (A-DEC)
# - label='left', closed='left'


def _coerce_dates(start_date, end_date) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """Accept strings / datetime-like / None and return (start_ts, end_ts) as pandas Timestamps."""
    start_ts = pd.to_datetime(start_date) if start_date is not None else None
    end_ts   = pd.to_datetime(end_date)   if end_date   is not None else None
    return start_ts, end_ts


def _ensure_dt_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Return a copy of df with a UTC DatetimeIndex. Does not mutate input."""
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
    """
    if column not in df.columns:
        raise KeyError(f"Column not found: {column!r}")
    if granularity not in ALLOWED_GRANULARITY:
        raise ValueError(f"granularity must be one of {sorted(ALLOWED_GRANULARITY)}")

    start, end = _coerce_dates(start_date, end_date)
    x = _ensure_dt_index(df, time_col)
    x = x[[column]]
    sliced = x.loc[start:end]
    if sliced.empty:
        raise ValueError(f"Selected window {start}–{end} has no rows for column {column!r}.")

    grouped = sliced.resample(granularity).agg({column: ["mean", "count", "size"]})
    mean = grouped[(column, "mean")].to_frame(name=column)
    count = grouped[(column, "count")]
    size = grouped[(column, "size")].clip(lower=1)
    coverage = (count / size).astype("float64")
    mean.loc[coverage < coverage_threshold, column] = pd.NA
    return mean


def _first_col_as_series(df: pd.DataFrame) -> pd.Series:
    """Return the first (and only) column as a Series, preserving the index/name."""
    if not isinstance(df, pd.DataFrame) or df.shape[1] == 0:
        raise ValueError("Expected a single-column DataFrame.")
    s = df.iloc[:, 0]
    # ensure series has a name for legend
    if s.name is None and df.columns.size:
        s.name = df.columns[0]
    return s


def plot_dual_timeseries(
    df: pd.DataFrame,
    start_date,
    end_date,
    column_name_one: str,
    granularity_one: str,
    column_name_two: str,
    granularity_two: str,
    time_col: str | None = None,
    coverage_threshold: float = 0.5,
    title: str | None = None,
    *,
    color_one: str | None = None,   # default: matplotlib cycle
    color_two: str = "red",         # make second series red
    label_one: str | None = None,
    label_two: str | None = None,
):
    """
    Plot two time series with separate y-axes. Second series (right axis) is red by default.

    Returns
    -------
    (fig, ax1, ax2)
    """
    # Correctly pass keyword-only coverage_threshold
    res1 = aggregate_timeseries(
        df=df,
        start_date=start_date,
        end_date=end_date,
        column=column_name_one,
        granularity=granularity_one,
        time_col=time_col or "utc_timestamp",
        coverage_threshold=coverage_threshold,
    )
    res2 = aggregate_timeseries(
        df=df,
        start_date=start_date,
        end_date=end_date,
        column=column_name_two,
        granularity=granularity_two,
        time_col=time_col or "utc_timestamp",
        coverage_threshold=coverage_threshold,
    )

    s1 = _first_col_as_series(res1)
    s2 = _first_col_as_series(res2)

    if s1.empty or s2.empty:
        raise ValueError("One of the aggregated series is empty. Check date window / column names.")

    fig, ax1 = plt.subplots(figsize=(10, 5))

    line1_kwargs = {}
    if color_one is not None:
        line1_kwargs["color"] = color_one
    l1, = ax1.plot(s1.index, s1.values, label=label_one or s1.name, **line1_kwargs)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(label_one or s1.name)

    ax2 = ax1.twinx()
    l2, = ax2.plot(s2.index, s2.values, label=label_two or s2.name, color=color_two)
    ax2.set_ylabel(label_two or s2.name, color=color_two)
    ax2.tick_params(axis="y", colors=color_two)
    if "right" in ax2.spines:
        ax2.spines["right"].set_color(color_two)

    if title:
        ax1.set_title(title)

    lines = [l1, l2]
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    return fig, ax1, ax2

def plot_dual_timeseries_old(
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


















def plot_nan_count(
    df: pd.DataFrame,
    *,
    time_col: Optional[str] = "utc_timestamp",
    use_existing_col: Optional[str] = None,
    include_cols_for_count: Optional[Sequence[str]] = None,
    resample: Optional[str] = None,          # e.g. "D", "W", "M"
    start_date: Optional[pd.Timestamp | str] = None,
    end_date: Optional[pd.Timestamp | str] = None,
    title: Optional[str] = "NaN count per row",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the number of NaNs per row over time, optionally restricted to a time window.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame. Must have a DatetimeIndex or a `time_col`.
    time_col : str or None, default "utc_timestamp"
        Name of timestamp column if the index is not already datetime.
    use_existing_col : str or None
        If provided, use this column as the NaN count (e.g. a precomputed 'nan_count').
    include_cols_for_count : sequence of str or None
        Subset of columns across which to count NaNs when computing (ignored if
        `use_existing_col` is set). If None, use all columns.
    resample : str or None
        Optional pandas frequency to aggregate counts (sum) per period.
    start_date, end_date : Timestamp | str | None
        Optional inclusive time-span to plot. Naive datetimes are treated as UTC.
    title : str or None
        Plot title.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    x = _ensure_dt_index(df, time_col)
    s_ts, e_ts = _coerce_dates(start_date, end_date)

    # Compute series
    if use_existing_col:
        if use_existing_col not in x.columns:
            raise KeyError(f"Column {use_existing_col!r} not found.")
        s = x[use_existing_col].astype("float64")
    else:
        cols = list(include_cols_for_count) if include_cols_for_count is not None else list(x.columns)
        if time_col in cols:
            cols.remove(time_col)
        s = x[cols].isna().sum(axis=1).astype("float64")

    # Slice to window (inclusive)
    if s_ts is not None or e_ts is not None:
        s = s.loc[s_ts:e_ts]

    # Optional resample (sum across window bins)
    if resample:
        s = s.resample(resample).sum(min_count=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s.index, s.values, linewidth=1.25)
    ax.set_xlabel("Time")
    ax.set_ylabel("NaN count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax