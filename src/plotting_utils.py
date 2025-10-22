###############################
###############################
##                                                          ##
##          src/plotting_utils.py                 ##
##                                                          ##
###############################
###############################

"""
Plotting utilities for aggregating and visualizing time series.

Notes on granularity and pandas defaults
---------------------------------------
- Allowed granularities for downsampling (normalized internally):
    - hour:        'h'                 (lowercase)
    - day:         'd'                 (lowercase)
    - week:        'W' or 'W-<DAY>'    (uppercase; pandas expects caps)
    - month-end:   'ME'                (explicit MonthEnd; replaces old 'M'/'m')
    - year-end:    'YE'                (explicit YearEnd; replaces old 'Y'/'y')
- Pandas resampling defaults:
    - W  : week ends on Sunday (W-SUN) unless an anchor is provided
    - ME : month-end
    - YE : Dec-end (A-DEC)
  label='left', closed='left'.
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple, Literal
from pandas.api.types import is_numeric_dtype

__all__ = [
    "coerce_dates",
    "ensure_dt_index",
    "resample_with_coverage",
    "plot_resampled",
    "plot_dual_resampled",
]

# Time resolutions to downsample to after normalization
# (non-weekly are 'h', 'd', 'ME', 'YE'; weekly remains 'W' / 'W-<DAY>')
ALLOWED_NONWEEKLY = {"h", "d", "ME", "YE"}


def _normalize_granularity(granularity: str) -> str:
    """
    Normalize a user-provided resample rule to pandas' current casing/aliases:

    Weekly rules (stay uppercase):
      - "W" or "W-<DAY>" (e.g., "W-MON", "W-SUN")  -> returned in uppercase.

    Non-weekly rules:
      - Hour:  "H"/"h"  -> "h"
      - Day:   "D"/"d"  -> "d"
      - Month: "M"/"m"/"ME" -> "ME"   (MonthEnd explicit)
      - Year:  "Y"/"y"/"YE" -> "YE"   (YearEnd explicit)

    Unknown strings are returned as-is (so pandas can raise clearly).
    """
    r = str(granularity).strip()
    if not r:
        return r

    # Weekly: 'W' or 'W-<DAY>'
    r_up = r.upper()
    if r_up == "W" or r_up.startswith("W-"):
        return r_up

    # Hour / Day (lowercase)
    if r_up == "H" or r == "h":
        return "h"
    if r_up == "D" or r == "d":
        return "d"

    # Month-End / Year-End (explicit aliases)
    if r_up in {"M", "ME"} or r == "m":
        return "ME"
    if r_up in {"Y", "YE"} or r == "y":
        return "YE"

    # Fallback: let pandas validate/complain
    return r


#############################
##       coerce_dates                           ##
#############################
def coerce_dates(
    start_date: Optional[object],
    end_date: Optional[object],
    *,
    interpret_as: Literal["utc", "local"] = "utc",
    local_tz: str = "Europe/Berlin",
    target_tz: Optional[str] = "UTC",
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
Coerce start/end into tz-aware pandas Timestamps with explicit, UTC-centric rules.

This function accepts strings or datetime-like objects and returns a pair
(start_ts, end_ts) that are tz-aware (by default in UTC). It is designed to be
transparent and predictable about how **naive** vs **tz-aware** inputs are handled.

Behavior
--------
- If a value is None -> returns None.
- If a value is **tz-aware** (has an offset or tzinfo):
    * Its own timezone/offset **wins**. The parameters `interpret_as` and
      `local_tz` are **ignored** for that value.
    * The timestamp is then **converted** to `target_tz` (default "UTC") if provided.
- If a value is **tz-naive** (no offset/tzinfo):
    * `interpret_as="utc"`   -> the naive time is **localized** as UTC
      (i.e., treated as UTC clock time).
    * `interpret_as="local"` -> the naive time is **localized** to `local_tz`
      (default "Europe/Berlin"), with DST checks, and then converted to `target_tz`
      if provided.

DST and invalid local times
---------------------------
- Ambiguous fall-back times (hour repeats) and nonexistent spring-forward times
  (hour skipped) in `local_tz` are **not auto-resolved**; this function **raises**
  with explicit, function-named error messages to avoid silent misinterpretations.
  Example error text:
      "In Europe/Berlin time, your start_date is ambiguous. The function
       coerce_dates cannot handle ambiguous local dates. Please enter your date
       in UTC or disambiguate the local time."

Timezone parameters
-------------------
interpret_as : {"utc", "local"}
    How to interpret **naive** inputs (tz-aware inputs ignore this).
local_tz : str or tzinfo, default "Europe/Berlin"
    IANA timezone used when `interpret_as="local"`. Prefer strings like
    "Europe/Berlin", "America/New_York", "UTC". Avoid ambiguous abbreviations
    like "CET"/"PST".
target_tz : str or tzinfo or None, default "UTC"
    Final timezone for returned timestamps. If None, retains each value's tz.

Returns
-------
(start_ts, end_ts) : tuple[pd.Timestamp | None, pd.Timestamp | None]
    Tz-aware timestamps (in `target_tz` if provided), or None where input was None.

Raises
------
TypeError
    If an input cannot be parsed into a timestamp. The error message includes the
    function name (`coerce_dates`) and which parameter failed.
ValueError
    If a naive local time is ambiguous or nonexistent in `local_tz`, or if
    start_ts > end_ts after coercion. Messages explicitly name `coerce_dates`.

Notes
-----
- **Tz-aware inputs ignore `interpret_as` and `local_tz`.** Their embedded
  offset/tz is honored, then converted to `target_tz`. This is intentional to
  avoid double-interpreting already-qualified timestamps.
- Using **tz-aware UTC** as the `target_tz` is recommended for downstream
  slicing/resampling to prevent tz-naive/aware comparison errors.

Examples
--------
# 1) Naive inputs interpreted as local Berlin time, then converted to UTC
coerce_dates("2017-07-01 10:00", "2017-07-01 12:00",
             interpret_as="local", local_tz="Europe/Berlin", target_tz="UTC")
# -> returns 08:00Z and 10:00Z (CEST is UTC+02:00 in July)

# 2) Tz-aware inputs: their offsets win; interpret_as/local_tz are ignored
coerce_dates("2017-02-03T21:45:22+01:00", "2017-07-01T15:15:00+06:00",
             interpret_as="local", local_tz="Europe/Berlin", target_tz="UTC")
# -> returns 20:45:22Z and 09:15:00Z

# 3) Ambiguous local time raises (fall-back hour in Berlin)
coerce_dates("2019-10-27 02:30", None,
             interpret_as="local", local_tz="Europe/Berlin", target_tz="UTC")
# -> ValueError naming coerce_dates; instructs to use UTC or disambiguate.

# 4) Nonexistent local time raises (spring-forward gap in Berlin)
coerce_dates("2019-03-31 02:30", None,
             interpret_as="local", local_tz="Europe/Berlin", target_tz="UTC")
# -> ValueError naming coerce_dates; instructs to use a valid local time.

"""
    func_name = "coerce_dates"

    def _process_one(label: str, value: Optional[object]) -> Optional[pd.Timestamp]:
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
        except Exception as e:
            raise TypeError(
                f"The function {func_name} could not parse {label!r}: {value!r}. "
                f"Original error: {e}"
            ) from e

        if ts.tzinfo is not None:
            # Already tz-aware: convert if target_tz requested.
            return ts.tz_convert(target_tz) if target_tz else ts

        # tz-naive: apply interpretation policy
        if interpret_as == "utc":
            ts = ts.tz_localize("UTC")
            return ts.tz_convert(target_tz) if target_tz and target_tz != "UTC" else ts

        if interpret_as == "local":
            try:
                # Raise on ambiguous/nonexistent local times (explicit policy)
                ts_local = ts.tz_localize(
                    local_tz,
                    ambiguous="raise",
                    nonexistent="raise",
                )
            except Exception as e:
                msg = str(e).lower()
                if "ambiguous" in msg:
                    raise ValueError(
                        f"In {local_tz} time, your {label} is ambiguous. "
                        f"The function {func_name} cannot handle ambiguous local dates. "
                        f"Please enter your date in UTC or disambiguate the local time."
                    ) from e
                if "nonexistent" in msg or "does not exist" in msg:
                    raise ValueError(
                        f"In {local_tz} time, your {label} does not exist due to a DST jump. "
                        f"The function {func_name} cannot handle nonexistent local dates. "
                        f"Please enter your date in UTC or choose a valid local time."
                    ) from e
                # Fallback for other localization errors
                raise ValueError(
                    f"The function {func_name} could not localize {label!r}={value!r} to {local_tz}: {e}"
                ) from e

            # Convert to target tz if requested (default: UTC)
            return ts_local.tz_convert(target_tz) if target_tz else ts_local

        # Unknown interpret_as
        raise ValueError(
            f"The function {func_name} received invalid interpret_as={interpret_as!r}. "
            f"Expected 'utc' or 'local'."
        )

    s = _process_one("start_date", start_date)
    e = _process_one("end_date",   end_date)

    if s is not None and e is not None and s > e:
        raise ValueError(
            f"The function {func_name} received start_date > end_date after coercion: "
            f"{s} > {e}. Please correct your inputs."
        )

    return s, e


#############################
##       ensure_dt_index                      ##
#############################
def ensure_dt_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Return a copy of `df` indexed by a **UTC DatetimeIndex**.

    Precedence rules
    ----------------
    1) If `df.index` is a DatetimeIndex:
       - tz-naive  -> interpret as **UTC clock time** (`tz_localize("UTC")`)
       - tz-aware  -> **convert** to UTC (`tz_convert("UTC")`)
       In this case, `time_col` is **ignored**.
    2) Otherwise (no DatetimeIndex present):
       - Require `time_col` to exist.
       - Use `pd.to_datetime(df[time_col], utc=True)` and set as the index.
         * tz-aware values are converted to UTC
         * tz-naive values are treated as **UTC**, not local time

    Notes
    -----
    - This function does **not** mutate `df`; it returns a copy.
    - If an existing tz-naive index actually encodes **local** clock times,
      localizing as UTC will misinterpret them—ensure inputs are correct upstream.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    time_col : str
        Name of the timestamp column to use **only** when there is no DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        A new DataFrame indexed by a **UTC** DatetimeIndex.

    Raises
    ------
    KeyError
        If no DatetimeIndex exists and `time_col` is missing.
        The error message explicitly names this function.
    """
    func_name = "ensure_dt_index"
    out: pd.DataFrame = df.copy()

    # Case 1: existing DatetimeIndex -> normalize to UTC, ignore `time_col`
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
        return out

    # Case 2: no DatetimeIndex -> require `time_col` and coerce to UTC
    if time_col not in out.columns:
        raise KeyError(
            f"The function {func_name} expects a DatetimeIndex or a valid time column, "
            f"yet the DataFrame has no DatetimeIndex and is missing the time column {time_col!r}."
        )

    out[time_col] = pd.to_datetime(out[time_col], utc=True)
    out = out.set_index(time_col)
    return out


#############################
##       resample_with_coverage         ##
#############################
def resample_with_coverage(
    df: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    column: str,
    granularity: Literal["H", "D", "W", "M", "Y", "h", "d", "w", "m", "y", "ME", "YE"] | str,
    time_col: str = "utc_timestamp",
    *,
    how: Literal["mean", "sum", "min", "max", "median"] = "mean",
    coverage_threshold: float = 0.5,
) -> pd.Series:
    """
    Downsample one column by `granularity` and aggregate via `how`, then mask bins
    with insufficient **observed** coverage (count/size). Returns a **pd.Series**
    with nullable float dtype (Float64).

    Notes
    -----
    - Weekly anchors like "W-MON"/"W-SUN" remain uppercase.
    - Non-weekly normalized outputs:
        'H'/'h' -> 'h', 'D'/'d' -> 'd', 'M'/'m' -> 'ME', 'Y'/'y' -> 'YE'.
    """
    func_name = "resample_with_coverage"

    # Validate column presence
    if column not in df.columns:
        raise KeyError(
            f"The function {func_name} received a non-existent column {column!r}."
        )

    # Normalize + validate granularity
    g_rule = _normalize_granularity(str(granularity))
    g_up = g_rule.upper()
    is_weekly = (g_up == "W") or g_up.startswith("W-")
    if not is_weekly and g_rule not in ALLOWED_NONWEEKLY:
        raise ValueError(
            f"The function {func_name} received an invalid granularity {granularity!r} "
            f"(normalized to {g_rule!r}). Allowed are {sorted(ALLOWED_NONWEEKLY)} "
            f"or weekly rules like 'W'/'W-MON'."
        )

    resample_rule = g_rule

    # Validate how
    allowed_how = {"mean", "sum", "min", "max", "median"}
    if how not in allowed_how:
        raise ValueError(
            f"The function {func_name} received an invalid how={how!r}. "
            f"Allowed are {sorted(allowed_how)}."
        )

    # Coerce dates (UTC-aware)
    start_ts, end_ts = coerce_dates(start_date, end_date)
    if start_ts is None or end_ts is None:
        raise ValueError(
            f"The function {func_name} requires both start_date and end_date; "
            f"got start={start_ts!r}, end={end_ts!r}."
        )

    # Normalize index to UTC
    x: pd.DataFrame = ensure_dt_index(df, time_col=time_col)

    # Slice window
    s: pd.Series = x[column]
    sliced: pd.Series = s.loc[start_ts:end_ts]
    if sliced.empty:
        raise ValueError(
            f"The function {func_name} selected an empty window for column {column!r} "
            f"using {start_ts}–{end_ts}."
        )

    # Type check for numeric aggregations
    if not is_numeric_dtype(sliced):
        raise ValueError(
            f"The function {func_name} requires a numeric dtype for column {column!r} "
            f"when how={how!r}, but received dtype {sliced.dtype}."
        )

    # Resample & aggregate
    g = sliced.resample(resample_rule)
    try:
        agg: pd.Series = getattr(g, how)()
    except Exception as e:
        raise ValueError(
            f"The function {func_name} failed during resample aggregation with how={how!r}: {e}"
        ) from e

    # Observed coverage (no expected-count penalty)
    count: pd.Series = g.count()
    size: pd.Series = g.size().clip(lower=1)  # avoid div-by-zero for empty bins
    coverage: pd.Series = (count / size).astype("float64")

    # Ensure nullable float before masking with pd.NA
    result: pd.Series = agg.astype("Float64")
    result[coverage < coverage_threshold] = pd.NA

    # Preserve name for legend-friendly plotting
    if result.name is None:
        result.name = column

    return result


def _pretty_col(name: str) -> str:
    """Humanize a column name for display (underscores → spaces)."""
    return str(name).replace("_", " ")


def _pretty_freq(rule: str) -> str:
    """
    Humanize a pandas resample rule for display:
      'H'/'h' -> 'hour'
      'D'/'d' -> 'day'
      'W'     -> 'week'
      'W-<DAY>' -> 'week (<Day> end)'
      'ME'    -> 'month'
      'YE'    -> 'year'
    """
    r = str(rule).upper()
    if r in {"H", "HOUR"}:
        return "hour"
    if r in {"D", "DAY"}:
        return "day"
    if r == "W":
        return "week"
    if r.startswith("W-") and len(r) > 2:
        day = r.split("-", 1)[1].capitalize()
        return f"week ({day} end)"
    if r == "ME":
        return "month"
    if r == "YE":
        return "year"
    return r  # fallback


#############################
##       plot_resampled                      ##
#############################
def plot_resampled(
    df: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    column: str,
    granularity: Literal["H", "D", "W", "M", "Y", "h", "d", "w", "m", "y", "ME", "YE"] | str,
    time_col: str = "utc_timestamp",
    *,
    how: Literal["mean", "sum", "min", "max", "median"] = "mean",
    coverage_threshold: float = 0.5,
    color: str = "tab:blue",
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Slice `df` to [start_date, end_date] (UTC), resample `column` by `granularity`
    using `how`, apply coverage masking, and plot a single series.

    Title/labels include the column, humanized granularity, how, coverage
    threshold, and the UTC time slice. `title` overrides the default if provided.
    """
    func_name = "plot_resampled"

    # Coerce once for display + downstream calls
    try:
        start_ts, end_ts = coerce_dates(start_date, end_date)
    except Exception as e:
        raise ValueError(
            f"The function {func_name} failed to coerce the time window: {e}"
        ) from e

    if start_ts is None or end_ts is None:
        raise ValueError(
            f"The function {func_name} requires both start_date and end_date; "
            f"got start={start_ts!r}, end={end_ts!r}."
        )

    # Build the resampled series
    try:
        series: pd.Series = resample_with_coverage(
            df=df,
            start_date=start_ts,
            end_date=end_ts,
            column=column,
            granularity=granularity,
            time_col=time_col,
            how=how,
            coverage_threshold=coverage_threshold,
        )
    except Exception as e:
        raise ValueError(
            f"The function {func_name} could not produce a resampled series: {e}"
        ) from e

    pretty = _pretty_col(column)
    freq_label = _pretty_freq(granularity)
    threshold_pct = int(round(coverage_threshold * 100))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(series.index, series.values, label=pretty, color=color, linewidth=1.5)

    # Labels: concise and uncluttered
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(pretty)
    ax.grid(axis="y", alpha=0.3)

    # Title (override if provided)
    auto_title = f"{pretty} — {how} per {freq_label}"
    ax.set_title(title if title is not None else auto_title)

    # Small caption beneath the plot area
    caption = f"Coverage ≥ {threshold_pct}%; window: {start_ts} to {end_ts} (UTC)"
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)

    ax.legend()
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # leave room for caption
    return fig, ax


def plot_dual_resampled(
    df: pd.DataFrame,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    column_one: str,
    column_two: str,
    granularity_one: Literal["H", "D", "W", "M", "Y", "h", "d", "w", "m", "y", "ME", "YE"] | str,
    granularity_two: Literal["H", "D", "W", "M", "Y", "h", "d", "w", "m", "y", "ME", "YE"] | str,
    time_col: str = "utc_timestamp",
    *,
    how_one: Literal["mean", "sum", "min", "max", "median"] = "mean",
    how_two: Literal["mean", "sum", "min", "max", "median"] = "mean",
    coverage_threshold_one: float = 0.5,
    coverage_threshold_two: float = 0.5,
    color_one: str = "tab:blue",
    color_two: str = "tab:red",
    title: Optional[str] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Slice `df` to [start_date, end_date] (UTC), resample two columns independently
    (each with its own granularity/how/coverage), and plot them on a shared x-axis
    with two y-axes (left/right). `title` overrides the default if provided.
    """
    func_name = "plot_dual_resampled"

    # Coerce once for display + downstream calls
    try:
        start_ts, end_ts = coerce_dates(start_date, end_date)
    except Exception as e:
        raise ValueError(
            f"The function {func_name} failed to coerce the time window: {e}"
        ) from e

    if start_ts is None or end_ts is None:
        raise ValueError(
            f"The function {func_name} requires both start_date and end_date; "
            f"got start={start_ts!r}, end={end_ts!r}."
        )

    # Build both resampled series
    try:
        s1: pd.Series = resample_with_coverage(
            df=df,
            start_date=start_ts,
            end_date=end_ts,
            column=column_one,
            granularity=granularity_one,
            time_col=time_col,
            how=how_one,
            coverage_threshold=coverage_threshold_one,
        )
    except Exception as e:
        raise ValueError(
            f"The function {func_name} could not resample column_one {column_one!r}: {e}"
        ) from e

    try:
        s2: pd.Series = resample_with_coverage(
            df=df,
            start_date=start_ts,
            end_date=end_ts,
            column=column_two,
            granularity=granularity_two,
            time_col=time_col,
            how=how_two,
            coverage_threshold=coverage_threshold_two,
        )
    except Exception as e:
        raise ValueError(
            f"The function {func_name} could not resample column_two {column_two!r}: {e}"
        ) from e

    pretty1 = _pretty_col(column_one)
    pretty2 = _pretty_col(column_two)
    freq1 = _pretty_freq(granularity_one)
    freq2 = _pretty_freq(granularity_two)
    th1 = int(round(coverage_threshold_one * 100))
    th2 = int(round(coverage_threshold_two * 100))

    # Plot
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()

    line1, = ax_left.plot(s1.index, s1.values, color=color_one, linewidth=1.5, label=pretty1)
    line2, = ax_right.plot(s2.index, s2.values, color=color_two, linewidth=1.5, label=pretty2)

    ax_left.set_xlabel("Time (UTC)")
    ax_left.set_ylabel(f"{pretty1}  ({how_one} @ {freq1}, ≥{th1}% cov)")
    ax_right.set_ylabel(f"{pretty2} ({how_two} @ {freq2}, ≥{th2}% cov)")
    ax_left.grid(axis="y", alpha=0.3)

    auto_title = f"{pretty1} vs {pretty2}"
    ax_left.set_title(title if title is not None else auto_title)

    caption = f"Window: {start_ts} to {end_ts} (UTC)"
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)

    handles = [line1, line2]
    labels = [pretty1, pretty2]
    ax_left.legend(handles, labels, loc="upper left")

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig, (ax_left, ax_right)
