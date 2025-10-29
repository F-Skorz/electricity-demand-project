###############################
###############################
##                                                          ##
##          src/eda_utils.py                       ##
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
from typing import Sequence, Tuple, List
from typing import Callable, Iterable, Optional, Dict, Any
import numpy as np


#############################
##       Missingness per Column        ##
############################# 


#######################################
#   Absolute count and proportion                           #                
#    Output: list of triples:                                          #
#     (<ColumnName>, Count, Fraction)                   #
#######################################

def calculate_abs_rel_missingness_per_column(
    df: pd.DataFrame
) -> List[Tuple[str, int, float]]:
    """
    Return a list of (column_name, abs_nans, rel_nans) for each column in `df`.

    rel_nans ∈ [0, 1]; abs_nans is an integer count. If `df` is empty, rel_nans
    will be NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    list[tuple[str, int, float]]
        A list of triples: (column_name, abs_nans, rel_nans).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The function alculate_abs_rel_missingness_per_column expects a pandas DataFrame as input.")
    abs_nans = df.isna().sum()
    n = len(df)
    rel_nans = (abs_nans / n) if n > 0 else pd.Series(float("nan"), index=df.columns)
    return [(str(col), int(abs_nans[col]), float(rel_nans[col])) for col in df.columns]


####################
##  We sort the triples     # 
#################### 

def sort_triples_by_index(
    triples: List[Tuple[str, int, float]],
    index: int = 2,
    descending: bool = True,
    nan_position: str = "last",
) -> List[Tuple[str, int, float]]:
    """
    Return a new list of tuples sorted by the value at `index`.
    NaNs (and None) are treated as missing and placed per `nan_position`.
    """
    def _is_nan(x):
        return x is None or (isinstance(x, float) and x != x)

    normal = [t for t in triples if not _is_nan(t[index])]
    missing = [t for t in triples if _is_nan(t[index])]

    normal.sort(key=lambda t: t[index], reverse=descending)
    return (missing + normal) if nan_position == "first" else (normal + missing)



########################################
#   list of triples -->  3-column-DataFrame with         #
#   three columns                                                        #
########################################

def triples_to_df(triples: List[Tuple[str, int, float]]) -> pd.DataFrame:
    """
    Convert a list of triples into a DataFrame with columns:
    ['column', 'n_missing', 'frac_missing'].

    Accepts items as either:
      - (col, n, frac), or
      - ((col, n, frac),)

    Parameters
    ----------
    triples : list[tuple]
        Iterable of triples (or one-tuple-wrapped triples).

    Returns
    -------
    pd.DataFrame
        Columns: ["column", "n_missing", "frac_missing"].
    """
    rows: list[tuple[str, int, float]] = []
    for item in triples:
        # (col, n, frac)
        if isinstance(item, (list, tuple)) and len(item) == 3:
            col, n, frac = item
        # ((col, n, frac),)
        elif (
            isinstance(item, (list, tuple)) and len(item) == 1
            and isinstance(item[0], (list, tuple)) and len(item[0]) == 3
        ):
            col, n, frac = item[0]
        else:
            raise ValueError(f"The function triples_to_df expected: a triple. But it got: {item!r}")
        rows.append((str(col), int(n), float(frac)))

    return pd.DataFrame(rows, columns=["column", "n_missing", "frac_missing"])


def build_columnwise_missingness_report(
    df: pd.DataFrame,
    *,
    sort_index: int = 2,
    descending: bool = True,
    nan_position: str = "last",
) -> pd.DataFrame:
    """
    Compute and return a sorted, tidy DataFrame with column-wise missingness
    statistics (absolute & relative).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The function build_columnwise_missingness_report expected a pandas DataFrame.")

    triples = calculate_abs_rel_missingness_per_column(df)
    sorted_triples = sort_triples_by_index(
        triples,
        index=sort_index,
        descending=descending,
        nan_position=nan_position,
    )
    return triples_to_df(sorted_triples)





def missingness_mask(
    df: pd.DataFrame,
    time_cols: Tuple[str, ...] = ("utc_timestamp", "cet_cest_timestamp"),
) -> pd.DataFrame:
    """
    Build a missingness mask:
      - Columns listed in `time_cols` are copied through unchanged (even if not datetime dtype yet).
      - Other columns become float masks: 1.0 if NaN, 0.0 if not NaN.

    The index is preserved as-is.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The functio missingness_mask expects a pandas DataFrame.")

    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in time_cols:
            out[col] = df[col]  # pass through untouched (string timestamps are OK)
        else:
            out[col] = df[col].isna().astype("float64")
    return out


#############################
##       Missingness per  Row            ##
############################# 


def count_missing_per_row(
    df: pd.DataFrame,
    time_cols: Sequence[str] = ("utc_timestamp", "cet_cest_timestamp"),
    out_col: str = "miss_count",
) -> pd.DataFrame:
    """
    Return a DataFrame with the original index and:
      - any timestamp columns from `time_cols` that exist in `df` (possibly none),
      - one column with the number of missing values in each row
        (counted across ALL columns via pandas.isna()).

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    time_cols : Sequence[str], default ("utc_timestamp", "cet_cest_timestamp")
        Timestamp columns to include in the output if present.
    out_col : str, default "nan_count"
        Name of the per-row missing count column.

    Returns
    -------
    pd.DataFrame
        Columns: [<existing time cols...>, out_col]
    """
    existing_time_cols = [c for c in time_cols if c in df.columns]
    missing = df.isna().sum(axis=1)
    out = df.loc[:, existing_time_cols].copy()
    out[out_col] = missing  # or: missing.astype("Int64")
    return out


def row_missing_breakdown(
    df: pd.DataFrame,
    time_cols: Tuple[str, ...] = ("utc_timestamp", "cet_cest_timestamp"),
    out_prefix: str = "missing_",
) -> pd.DataFrame:
    """
    Per-row missingness breakdown by sentinel:
      - total  : any missing as defined by pandas.isna()
      - nan    : np.nan in float/complex dtypes + np.nan present inside object
      - nat    : NaT in datetime/timedelta dtypes
      - none   : None inside object dtype
      - pdNA   : <NA> in pandas' nullable dtypes (Int64, Boolean, String, Float64, etc.) + <NA> in object

    Returns a DataFrame with the original index, the requested time columns,
    and the above breakdown columns.
    """
    out = df.loc[:, [c for c in time_cols if c in df.columns]].copy()

    # total (all standard missings)
    total = df.isna().sum(axis=1)

    # --- NaT in datetime/timedelta dtypes
    dt_cols = [c for c in df.columns
               if is_datetime64_any_dtype(df[c]) or is_timedelta64_dtype(df[c])]
    nat = df[dt_cols].isna().sum(axis=1) if dt_cols else 0

    # --- np.nan in numeric float/complex dtypes
    nan_num = df.select_dtypes(include=["float", "complex"]).isna().sum(axis=1)

    # --- nullable dtypes (<NA>)
    pdna_cols = [c for c in df.columns if is_extension_array_dtype(df[c]) and df[c].dtype.na_value is pd.NA]
    pdna_ext = df[pdna_cols].isna().sum(axis=1) if pdna_cols else 0

    # --- object dtype breakdown: None, np.nan, <NA>
    obj = df.select_dtypes(include=["object"])
    if not obj.empty:
        none_obj = obj.applymap(lambda x: x is None).sum(axis=1)
        nan_obj  = obj.applymap(lambda x: isinstance(x, float) and np.isnan(x)).sum(axis=1)
        pdna_obj = obj.applymap(lambda x: x is pd.NA).sum(axis=1)
    else:
        none_obj = nan_obj = pdna_obj = 0

    # combine disjoint pieces
    nan  = nan_num.add(nan_obj, fill_value=0)
    pdNA = (pd.Series(0, index=df.index) if isinstance(pdna_ext, int)
            else pdna_ext).add(pdna_obj if not isinstance(pdna_obj, int) else 0, fill_value=0)

    # assemble output
    out[out_prefix + "total"] = total
    out[out_prefix + "nan"]   = nan
    out[out_prefix + "nat"]   = nat
    out[out_prefix + "none"]  = none_obj
    out[out_prefix + "pdNA"]  = pdNA

    # (optional) sanity check—should be equal; tiny mismatches mean a category wasn’t captured
    # remainder = out[out_prefix + "total"] - (out[out_prefix + "nan"] + out[out_prefix + "nat"] +
    #                                          out[out_prefix + "none"] + out[out_prefix + "pdNA"])
    # print("Max remainder:", remainder.max())

    return out


