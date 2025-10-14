from __future__ import annotations
import pandas as pd
import numpy as np
import holidays  # add alongside your other imports

__all__ = ["make_de_hol_df", "populate_de_public_holidays"]

STATE_HOLIDAY_CODES: dict[str, str] = {
    "DE_BW": "BW", "DE_BY": "BY", "DE_BE": "BE", "DE_BB": "BB",
    "DE_HB": "HB", "DE_HH": "HH", "DE_HE": "HE", "DE_MV": "MV",
    "DE_NI": "NI", "DE_NW": "NW", "DE_RP": "RP", "DE_SL": "SL",
    "DE_SN": "SN", "DE_ST": "ST", "DE_SH": "SH", "DE_TH": "TH",
}



# Canonical list of German state codes with DE_ prefix
STATE_CODES = (
    "DE_BW","DE_BY","DE_BE","DE_BB","DE_HB","DE_HH","DE_HE","DE_MV",
    "DE_NI","DE_NW","DE_RP","DE_SL","DE_SN","DE_ST","DE_SH","DE_TH"
)

def make_de_hol_df(
    start: str = "2015-01-01",
    end: str = "2035-12-31",
    tz: str = "Europe/Berlin",
    states: tuple[str, ...] = STATE_CODES,
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


def populate_de_public_holidays(
    df: pd.DataFrame,
    *,
    states: tuple[str, ...] = STATE_CODES,
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
        subdiv = STATE_HOLIDAY_CODES.get(st)
        if subdiv is None:
            raise ValueError(f"Unknown state code: {st!r}")

        hol = holidays.Germany(subdiv=subdiv, years=range(y0, y1 + 1))
        is_hol = np.fromiter((d in hol for d in day_array), count=len(day_array), dtype=bool)

        hol_col = f"{st}_hol"
        school_col = f"{st}_school_free"

        df[hol_col] = is_hol.astype("uint8")
        if holiday_implies_school_free and school_col in df.columns:
            df[school_col] = df[hol_col].astype("uint8")

    return df
