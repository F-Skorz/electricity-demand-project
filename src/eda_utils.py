################
# src/eda_utils.py    #
################

from __future__ import annotations
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

__all__ = [
    "nan_stats",
    "sort_triples_by_index",
    "triples_to_df",
    "build_nan_stats",
    "missingness_mask",
]


def nan_stats(df: pd.DataFrame) -> list[tuple[str, int, float]]:
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
        raise TypeError("nan_stats expects a pandas DataFrame.")
    abs_nans = df.isna().sum()
    n = len(df)
    rel_nans = (abs_nans / n) if n > 0 else pd.Series(float("nan"), index=df.columns)
    return [(str(col), int(abs_nans[col]), float(rel_nans[col])) for col in df.columns]


def sort_triples_by_index(
    triples: list[tuple],
    index: int = 2,
    descending: bool = True,
    nan_position: str = "last",  # "last" or "first"
) -> list[tuple]:
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


def triples_to_df(triples: list[tuple]) -> pd.DataFrame:
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
            isinstance(item, (list, tuple)) and len(item) == 1 and
            isinstance(item[0], (list, tuple)) and len(item[0]) == 3
        ):
            col, n, frac = item[0]
        else:
            raise ValueError(f"Expected a triple, got: {item!r}")
        rows.append((str(col), int(n), float(frac)))

    return pd.DataFrame(rows, columns=["column", "n_missing", "frac_missing"])


def build_nan_stats(
    df: pd.DataFrame,
    *,
    sort_index: int = 2,
    descending: bool = True,
    nan_position: str = "last",  # "last" or "first"
) -> tuple[list[tuple[str, int, float]], pd.DataFrame]:
    """
    Compute NaN stats for each column of a DataFrame, sort them, and return
    both the sorted list and a tidy 3-column DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    sort_index : int, default 2
        Index within each triple to sort by (0=column, 1=n_missing, 2=frac_missing).
    descending : bool, default True
        Sort order.
    nan_position : {"last","first"}, default "last"
        Where to plac
   """