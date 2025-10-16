from __future__ import annotations
import pandas as pd
import numpy as np
import holidays  # add alongside your other imports
import warnings
from typing import Dict, List, Tuple

__all__ = ["make_de_hol_df", "update_public_DE_hol_df_fct", "update_school_DE_hol_df_fct"]

GERMAN_STATES_ABBREVIATIONS: Sequence[str] = ['BW', 'BY', 'BE', 'BB', 'HB', 'HH', 'HE',
                                                                                     'MV', 'NI', 'NW', 'RP', 'SL', 'SN', 'ST', 'SH', 'TH']

GERMAN_STATES_ABBREVIATIONS_WITH_DE_PREFIX: tuple[str, ...] = (
    "DE_BW","DE_BY","DE_BE","DE_BB","DE_HB","DE_HH","DE_HE","DE_MV",
    "DE_NI","DE_NW","DE_RP","DE_SL","DE_SN","DE_ST","DE_SH","DE_TH"
)

GERMAN_STATES_ABBREVIATIONS_PREFIX_CUT: dict[str, str] = {
    "DE_BW": "BW", "DE_BY": "BY", "DE_BE": "BE", "DE_BB": "BB",
    "DE_HB": "HB", "DE_HH": "HH", "DE_HE": "HE", "DE_MV": "MV",
    "DE_NI": "NI", "DE_NW": "NW", "DE_RP": "RP", "DE_SL": "SL",
    "DE_SN": "SN", "DE_ST": "ST", "DE_SH": "SH", "DE_TH": "TH",
}


# Canonical list of German state codes with DE_ prefix

def make_de_hol_df(
    start: str = "2015-01-01",
    end: str = "2035-12-31",
    tz: str = "Europe/Berlin",
    states: tuple[str, ...] = GERMAN_STATES_ABBREVIATIONS_WITH_DE_PREFIX,
) -> pd.DataFrame:
    """
    Build a daily local-calendar DataFrame for German holidays/school-free flags.

    Columns:
      - local_date : daily calendar day (naive, local date label)
      - local_start : tz-aware local midnight (Europe/Berlin) at the start of the day
      - local_end   : tz-aware local midnight of the next day
      - utc_start   : local_start converted to UTC
      - utc_end     : local_end converted to UTC
      - <STATE>_hol, <STATE>_school_free : UInt8 flags (all initialized to 0)

    Notes
    -----
    - local_date is intended for merges by day label .
    - utc_start/utc_end reflect the UTC interval covering the local day; on DST
      switch dates the UTC span is 23h or 25h, which is correct.
    """
    # 1) Daily local calendar (naive date labels)
    local_dates = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame({"local_date": local_dates})
    
    # 2) Local midnight start/end as tz-aware, and UTC projections
    local_start = df["local_date"].dt.tz_localize(tz)
    local_end   = (df["local_date"] + pd.Timedelta(days=1)).dt.tz_localize(tz)

    df["local_start"] = local_start
    df["local_end"]   = local_end
    df["utc_start"]   = local_start.dt.tz_convert("UTC")
    df["utc_end"]     = local_end.dt.tz_convert("UTC")

    # 3) Initialize flags as UInt8 zeros (memory-friendly)
    zeros = np.zeros(len(df), dtype="uint8")
    for st in states:
        df[f"{st}_hol"] = zeros
        df[f"{st}_school_free"] = zeros

    return df


def update_public_DE_hol_df_fct(
    df: pd.DataFrame,
    *,
    states: tuple[str, ...] = GERMAN_STATES_ABBREVIATIONS_WITH_DE_PREFIX,
    holiday_implies_school_free: bool = True,
    years: tuple[int, int] | None = None,
) -> pd.DataFrame:
    if "local_date" not in df.columns:
        raise KeyError("Expected column 'local_date' in df (from make_de_hol_df).")

    if years is None:
        y0 = int(pd.to_datetime(df["local_date"].min()).year)
        y1 = int(pd.to_datetime(df["local_date"].max()).year)
    else:
        y0, y1 = years

    day_array = df["local_date"].dt.date.to_numpy()

    for st in states:
        subdiv = GERMAN_STATES_ABBREVIATIONS_PREFIX_CUT.get(st)
        if subdiv is None:
            raise ValueError(f"Unknown state code: {st!r}")

        hol = holidays.Germany(subdiv=subdiv, years=range(y0, y1 + 1))
        is_hol = np.fromiter((d in hol for d in day_array), count=len(day_array), dtype=bool)

        hol_col = f"{st}_hol"
        school_col = f"{st}_school_free"

        df[hol_col] = is_hol.astype("uint8")
        if holiday_implies_school_free and school_col in df.columns:
            df[school_col] = np.maximum(df[school_col].astype("uint8"), df[hol_col].astype("uint8"))

    return df




def update_school_DE_hol_df_fct(
    start_year: int,
    end_year: int,
    df: pd.DataFrame,
    semester_registry: Dict[Tuple[int, int], Dict[str, List[Tuple[str, str]]]],
    *,
    add_weekends: bool = True,
) -> pd.DataFrame:
    """
    OR-only update of school-free flags from semester dictionaries.
    - No reset.
    - Sets 1 where school-vacation intervals (or weekends) apply.
    """
    start_date = pd.Timestamp(f"{start_year}-01-01")
    end_date = pd.Timestamp(f"{end_year}-12-31")
    in_window = (df["local_date"] >= start_date) & (df["local_date"] <= end_date)

    for st in GERMAN_STATES_ABBREVIATIONS:
        col = f"DE_{st}_school_free"
        if col not in df.columns:
            df[col] = np.uint8(0)
        else:
            df[col] = df[col].astype("uint8", copy=False)

    for year in range(start_year, end_year + 1):
        for half in (1, 2):
            d = semester_registry.get((year, half))
            if not d:
                continue
            for st, ranges in d.items():
                if st not in GERMAN_STATES_ABBREVIATIONS:
                    raise ValueError(f"Unknown state code in registry: {st}")
                col = f"DE_{st}_school_free"
                for s_str, e_str in ranges:
                    s = pd.Timestamp(s_str); e = pd.Timestamp(e_str)
                    mask = in_window & (df["local_date"] >= s) & (df["local_date"] <= e)
                    df.loc[mask, col] = 1

    if add_weekends:
        wk_mask = in_window & df["local_date"].dt.dayofweek.isin([5, 6])
        for st in GERMAN_STATES_ABBREVIATIONS:
            df.loc[wk_mask, f"DE_{st}_school_free"] = 1

    return df