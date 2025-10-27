###############################
###############################
##                                                          ##
##          src/dwd_utils.py                      ##
##                                                          ##
###############################
###############################
from __future__ import annotations

import re
import time
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Callable
from functools import reduce

import requests
import warnings
import numpy as np
import pandas as pd


BASE_INDEX_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/hourly/air_temperature/historical/"
)
####################
##  Download Chapter  ## 
####################

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _http_get(
    url: str,
    *,
    timeout: int = 30,
    max_retries: int = 3,
    backoff: float = 1.5,
    stream: bool = False,
) -> requests.Response:
    headers = {"User-Agent": "electricity-demand-project/1.0 (requests)"}
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, stream=stream)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff ** (attempt + 1))
            else:
                raise
    if last_exc:
        raise last_exc
    raise RuntimeError("_http_get: Unknown HTTP error")





def _parse_listing_for_station_zips(listing_html: str, station_id: str) -> List[str]:
    """
    Return all zip filenames for the given station_id found in the index HTML.
    Matches names like: stundenwerte_TU_00403_19990101_20250709_hist.zip
    """
    # Extract all hrefs that end with .zip
    hrefs = re.findall(r'href="([^"]+\.zip)"', listing_html, flags=re.IGNORECASE)

    # Filter by station id token `_STATIONID_` inside the name
    token = f"_{station_id}_"
    candidates = [h for h in hrefs if token in h and h.lower().endswith(".zip")]
    return candidates


def _pick_latest_zip(zip_filenames: List[str]) -> Optional[str]:
    """
    From a list of zip filenames, pick the one with the latest end date
    according to pattern: stundenwerte_TU_<station>_<start>_<end>_hist.zip
    """
    best: Optional[str] = None
    best_end: Optional[str] = None
    pat = re.compile(r"stundenwerte_TU_\d+_(\d{8})_(\d{8})_hist\.zip$", re.IGNORECASE)
    for name in zip_filenames:
        m = pat.search(name)
        if not m:
            # Fallback: keep the last one alphabetically if pattern doesn't match
            if best is None or name > best:
                best = name
            continue
        end = m.group(2)
        if (best_end is None) or (end > best_end):
            best = name
            best_end = end
    return best


def find_latest_station_zip(station_id: str, *, base_index_url: str = BASE_INDEX_URL) -> str:
    """
    Look up the index page and return the BEST matching .zip filename for station_id.
    Returns the filename, not the full URL.
    Raises if none is found.
    """
    resp = _http_get(base_index_url)
    zips = _parse_listing_for_station_zips(resp.text, station_id)
    if not zips:
        raise FileNotFoundError(f"No .zip archives found for station_id={station_id} at {base_index_url}")
    best = _pick_latest_zip(zips)
    if not best:
        raise FileNotFoundError(f"Could not select a latest .zip for station_id={station_id}")
    return best


# --- Download: iterate over the generator, don't treat it as a context manager ---
def download_zip(
    filename: str,
    *,
    dest_dir: Path,
    base_index_url: str = BASE_INDEX_URL,
    skip_existing: bool = True,
) -> Path:
    """
    Download the given filename from the base index URL into dest_dir.
    Returns the local path.
    """
    ensure_dir(dest_dir)
    out_path = dest_dir / filename
    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = f"{base_index_url}{filename}"
    resp = _http_get(url, timeout=60, max_retries=3, stream=True)
    try:
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
    finally:
        resp.close()
    return out_path


def unzip_archive(zip_path: Path, *, dest_dir: Path) -> List[Path]:
    """
    Extract a .zip into dest_dir. Returns list of extracted file paths.
    """
    ensure_dir(dest_dir)
    extracted: List[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = dest_dir / member.filename
            ensure_dir(target.parent)
            zf.extract(member, path=dest_dir)
            extracted.append(target)
    return extracted

def download_and_unzip_dwd_data(
    station_ids: Iterable[str],
    *,
    zip_dir: Path,
    unzip_dir: Path,
    base_index_url: str = BASE_INDEX_URL,
    skip_existing: bool = True,
) -> Dict[str, Dict[str, List[Path] | Path]]:
    """
    Generic downloader/unzipper for DWD hourly products (e.g., TU, RR).
    Looks up the latest archive per station on `base_index_url`, downloads to `zip_dir`,
    unzips into `unzip_dir`, and returns:
      { station_id: {"zip_file": <Path>, "extracted_files": [<Path>, ...]} }
    """
    ensure_dir(zip_dir)
    ensure_dir(unzip_dir)

    results: Dict[str, Dict[str, List[Path] | Path]] = {}
    for sid in station_ids:
        filename = find_latest_station_zip(sid, base_index_url=base_index_url)
        zip_path = download_zip(filename, dest_dir=zip_dir, base_index_url=base_index_url, skip_existing=skip_existing)
        extracted = unzip_archive(zip_path, dest_dir=unzip_dir)
        results[sid] = {"zip_file": zip_path, "extracted_files": extracted}
    return results



####################
## Buillding section       ##
####################

# --- DWD: build per-city TU (temp/RH) + RR (precip) and the national wide table ---


def _pick_station_sequence(city_stations: pd.DataFrame) -> List[pd.Series]:
    """
    Return an ordered list of station rows that 'chain' through time:
    start with earliest from_date, and when overlapping exists, pick the one
    with the latest to_date; otherwise jump to the next future station.
    """
    if city_stations.empty:
        return []

    df = city_stations.copy()
    df["from_date"] = pd.to_datetime(df["from_date"])
    df["to_date"] = pd.to_datetime(df["to_date"])
    df = df.sort_values("from_date")

    picks: List[pd.Series] = []
    current_end: Optional[pd.Timestamp] = None

    while not df.empty:
        if current_end is None:
            selected = df.iloc[0]
        else:
            overlapping = df[df["from_date"] <= current_end]
            if not overlapping.empty:
                selected = overlapping.sort_values("to_date", ascending=False).iloc[0]
            else:
                future = df[df["from_date"] > current_end]
                if future.empty:
                    break
                selected = future.sort_values("from_date").iloc[0]

        picks.append(selected)
        current_end = selected["to_date"]
        df = df[df["station_id"] != selected["station_id"]]

    return picks


def _find_file(unzip_dir: Path, pattern: str) -> Optional[Path]:
    """
    Find a single file under unzip_dir matching pattern (glob, recursive).
    If multiple matches exist, pick the lexicographically last (often newest).
    """
    matches = sorted(unzip_dir.rglob(pattern))
    if not matches:
        return None
    return matches[-1]


def _load_tu(unzip_dir: Path, station_id: str) -> pd.DataFrame:
    """
    Load air temperature & relative humidity (TU product) for one station.
    Returns columns: station_id, utc_timestamp, cet_cest_timestamp, QN_9, temp, RH
    """
    path = _find_file(unzip_dir, f"produkt_tu_stunde_*_*_{station_id}.txt")
    if path is None:
        raise FileNotFoundError(
            f"_load_tu: Could not find TU file for station_id={station_id} "
            f"under {unzip_dir} with pattern 'produkt_tu_stunde_*_*_{station_id}.txt'"
        )

    df = pd.read_csv(
        path, sep=";", encoding="latin1",
        dtype={"STATIONS_ID": str, "MESS_DATUM": str},
    )
    # Column cleanup & rename
    df.columns = df.columns.str.strip()
    required = {"MESS_DATUM", "QN_9", "TT_TU", "RF_TU"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"_load_tu: Missing {sorted(missing)} in TU file for station_id={station_id}. "
            f"Found columns: {df.columns.tolist()}"
        )

    out = pd.DataFrame({
        "station_id": station_id,
        "utc_timestamp": pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", utc=True),
        "QN_9": df["QN_9"],
        "temp": df["TT_TU"],
        "RH": df["RF_TU"],
    })
    out["cet_cest_timestamp"] = out["utc_timestamp"].dt.tz_convert("Europe/Berlin")

    # Optional: handle sentinel values (DWD sometimes uses -999 etc.)
    for col in ("temp", "RH"):
        if np.issubdtype(out[col].dtype, np.number):
            out.loc[out[col] <= -999, col] = np.nan

    # Final column order
    out = out[["station_id", "utc_timestamp", "cet_cest_timestamp", "QN_9", "temp", "RH"]]
    return out


def _load_rr(unzip_dir: Path, station_id: str) -> pd.DataFrame:
    """
    Load precipitation (RR product) for one station.
    Returns columns: station_id, utc_timestamp, cet_cest_timestamp, QN_8, precip_mm
    """
    path = _find_file(unzip_dir, f"produkt_rr_stunde_*_*_{station_id}.txt")
    if path is None:
        raise FileNotFoundError(
            f"_load_rr: Could not find RR file for station_id={station_id} "
            f"under {unzip_dir} with pattern 'produkt_rr_stunde_*_*_{station_id}.txt'"
        )

    df = pd.read_csv(path, sep=";", encoding="latin1", dtype={"STATIONS_ID": str, "MESS_DATUM": str})
    df.columns = df.columns.str.strip()

    required = {"MESS_DATUM", "QN_8", "R1"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"_load_rr: Missing {sorted(missing)} in RR file for station_id={station_id}. "
            f"Found columns: {df.columns.tolist()}"
        )

    out = pd.DataFrame({
        "station_id": station_id,
        "utc_timestamp": pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", utc=True),
        "QN_8": df["QN_8"],
        "precip_mm": df["R1"],
    })
    out["cet_cest_timestamp"] = out["utc_timestamp"].dt.tz_convert("Europe/Berlin")

    # Sentinel handling
    if np.issubdtype(out["precip_mm"].dtype, np.number):
        out.loc[out["precip_mm"] <= -999, "precip_mm"] = np.nan

    out = out[["station_id", "utc_timestamp", "cet_cest_timestamp", "QN_8", "precip_mm"]]
    return out

def _encode_station_id_tp(df: pd.DataFrame) -> pd.Series:
    """
    Encode TU/RR station IDs into a single string column:
      - both present and equal: "00403"
      - both present and different: "00403|01420"
      - only TU present: "00403|-"
      - only RR present: "-|01420"
      - neither: NaN
    """
    t = df.get("station_id_t")
    p = df.get("station_id_p")

    # Ensure they exist; if not, create missing columns filled with NA (won't break the logic)
    if t is None:
        t = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    if p is None:
        p = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    has_t = t.notna()
    has_p = p.notna()

    out = pd.Series(pd.NA, index=df.index, dtype="object")

    # both present
    both = has_t & has_p
    same = both & (t == p)
    diff = both & (t != p)

    out[same] = t[same]  # just "00403"
    out[diff] = (t[diff].astype(str) + "|" + p[diff].astype(str))

    # only one side
    only_t = has_t & ~has_p
    only_p = has_p & ~has_t
    out[only_t] = t[only_t].astype(str) + "|-"
    out[only_p] = "-|" + p[only_p].astype(str)

    return out

def create_DWD_temp_RH_precip_hist_df(
    city_id: str,
    *,
    stations_df: pd.DataFrame,
    unzip_dir: Path,
) -> pd.DataFrame:
    """
    Build a single city-level DataFrame that combines air temperature, relative humidity,
    and precipitation by chaining across the city's stations and merging TU+RR per station.

    Output columns (per city chunk):
      - station_id_t, station_id_p
      - utc_timestamp, cet_cest_timestamp
      - QN_9, temp, RH
      - QN_8, precip_mm

    Raises
    ------
    ValueError, FileNotFoundError with function name in the message.
    """
    fn = "create_DWD_temp_RH_precip_hist_df"

    if stations_df is None or stations_df.empty:
        raise ValueError(f"{fn}: stations_df is empty or None")

    city_stations = stations_df[stations_df["city_id"] == city_id].copy()
    if city_stations.empty:
        raise ValueError(f"{fn}: No station metadata found for city_id={city_id}")

    picks = _pick_station_sequence(city_stations)
    if not picks:
        raise ValueError(f"{fn}: Could not compute a station chain for city_id={city_id}")

    chunks: list[pd.DataFrame] = []
    for sel in picks:
        sid = str(sel["station_id"])

        tu_df = None
        rr_df = None
        try:
            tu_df = _load_tu(unzip_dir, sid).rename(columns={"station_id": "station_id_t"})
        except FileNotFoundError as e:
            import warnings
            warnings.warn(f"{fn}: {e}")
        try:
            rr_df = _load_rr(unzip_dir, sid).rename(columns={"station_id": "station_id_p"})
        except FileNotFoundError as e:
            import warnings
            warnings.warn(f"{fn}: {e}")

        if tu_df is None and rr_df is None:
            import warnings
            warnings.warn(f"{fn}: Skipping station_id={sid} (no TU or RR files found).")
            continue

        merged = rr_df if tu_df is None else tu_df if rr_df is None else pd.merge(
            tu_df, rr_df, on=["utc_timestamp", "cet_cest_timestamp"], how="outer"
        )

        # Soft-clip by the station's declared window (if present)
        if pd.notna(sel.get("from_date", pd.NaT)):
            merged = merged[merged["utc_timestamp"] >= pd.Timestamp(sel["from_date"], tz="UTC")]
        if pd.notna(sel.get("to_date", pd.NaT)):
            merged = merged[merged["utc_timestamp"] <= pd.Timestamp(sel["to_date"], tz="UTC")]

        chunks.append(merged)

    if not chunks:
        raise ValueError(f"{fn}: No valid data chunks constructed for city_id={city_id}")

    city_df = (
        pd.concat(chunks, ignore_index=True)
        .sort_values("utc_timestamp")
        .reset_index(drop=True)
    )

    # Deduplicate any accidental duplicate timestamps (prefer more signal columns)
    key = ["utc_timestamp", "cet_cest_timestamp"]
    sig_cols = [c for c in ("temp", "RH", "precip_mm") if c in city_df.columns]
    if city_df.duplicated(key).any():
        def _best_row(g: pd.DataFrame) -> pd.Series:
            scores = g[sig_cols].notna().sum(axis=1)
            return g.iloc[int(scores.values.argmax())]
        city_df = (
            city_df.groupby(key, as_index=False, group_keys=False).apply(_best_row).reset_index(drop=True)
        )

    # --- NEW: compress station IDs into single column ---
    city_df["station_id_tp"] = _encode_station_id_tp(city_df)

    # Drop separate station id columns
    drop_cols = [c for c in ("station_id_t", "station_id_p") if c in city_df.columns]
    if drop_cols:
        city_df = city_df.drop(columns=drop_cols)

    # --- Reorder columns: timestamps first ---
    desired_order = [
        "utc_timestamp",
        "cet_cest_timestamp",
        "station_id_tp",
        "QN_9", "temp", "RH",
        "QN_8", "precip_mm",
    ]
    # keep only those that exist
    final_cols = [c for c in desired_order if c in city_df.columns]
    # append any other columns (unlikely) at the end to avoid loss
    final_cols += [c for c in city_df.columns if c not in final_cols]
    city_df = city_df[final_cols]

    return city_df

def build_DE_DWD_hist_df(
    stations_df: pd.DataFrame,
    city_builder_fct: Callable[..., pd.DataFrame],
    *,
    unzip_dir: Path,
) -> pd.DataFrame:
    """
    Build the wide Germany-level DWD DataFrame by:
      1) Building one per-city frame with temp, RH, precip (and QN flags)
      2) Prefixing non-timestamp columns with city_id + '_'
      3) Outer-merging all cities on the two timestamps

    Returns
    -------
    pd.DataFrame with keys:
      - utc_timestamp (tz-aware UTC)
      - cet_cest_timestamp (Europe/Berlin)
    and per-city prefixed columns:
      <CITY>_station_id_t, <CITY>_station_id_p, <CITY>_QN_9, <CITY>_temp, <CITY>_RH,
      <CITY>_QN_8, <CITY>_precip_mm
    """
    fn = "build_DE_DWD_hist_df"

    if stations_df is None or stations_df.empty:
        raise ValueError(f"{fn}: stations_df is empty or None")

    city_ids = list(pd.unique(stations_df["city_id"]))
    if not city_ids:
        raise ValueError(f"{fn}: No city_id values found in stations_df")

    city_frames: List[pd.DataFrame] = []

    for city_id in city_ids:
        city_df = city_builder_fct(
            city_id=city_id,
            stations_df=stations_df,
            unzip_dir=unzip_dir,
        )
        # Prefix all non-timestamp columns
        def _ren(c: str) -> str:
            return c if c in ("utc_timestamp", "cet_cest_timestamp") else f"{city_id}_{c}"

        city_df = city_df.rename(columns=_ren)
        city_frames.append(city_df)

    # Merge across all cities
    wide_df = reduce(
        lambda left, right: pd.merge(
            left, right,
            on=["utc_timestamp", "cet_cest_timestamp"],
            how="outer",
        ),
        city_frames,
    )

    # Final sort for cleanliness
    wide_df = wide_df.sort_values("utc_timestamp").reset_index(drop=True)
    return wide_df



__all__ = [
     "find_latest_station_zip", "download_zip",
    "download_and_unzip_dwd_data", "load_tu_file",
    "create_DWD_temp_RH_precip_hist_df",
    "build_DE_DWD_hist_df",
]

