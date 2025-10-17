"""
Lightweight Parquet persistence helpers for your repo.
Writes to `data/processed/` via `src.config.PROC` by default.

Usage:
    from src.persist import save_parquet
    path = save_parquet(df, name="merged_2015_2020")
"""

from pathlib import Path
from typing import Optional
from datetime import datetime
import pandas as pd

# Prefer your repo's processed-data path; fall back to repo_root/data/processed.
try:
    from src.config import PROC as DEFAULT_PROC  # e.g., <repo>/data/processed
except Exception:
    DEFAULT_PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def _ensure_parquet_name(name: str) -> str:
    """Return name with .parquet extension."""
    return name if name.lower().endswith(".parquet") else f"{name}.parquet"


def _build_stem(base: str, suffix: Optional[str], add_date: bool) -> str:
    """Compose filename stem: base[+_suffix][+_YYYYMMDD]."""
    stem = base
    if suffix:
        stem += f"_{suffix}"
    if add_date:
        stem += f"_{datetime.utcnow():%Y%m%d}"
    return stem


def save_parquet(
    df: pd.DataFrame,
    name: str,
    folder: Optional[Path] = None,
    *,
    compression: str = "zstd",
    index: bool = False,
    allow_overwrite: bool = False,
    suffix: Optional[str] = None,
    add_date: bool = False,
) -> Path:
    """
    Persist a DataFrame as Parquet using pyarrow, preserving dtypes (incl. tz-aware datetimes).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    name : str
        Base file name (with or without .parquet).
    folder : Path, optional
        Target directory; defaults to `src.config.PROC` (data/processed).
    compression : str
        Parquet codec: "zstd" (default), "snappy", "gzip", etc.
    index : bool
        Whether to write the DataFrame index. Default False.
    allow_overwrite : bool
        If False and file exists, raise FileExistsError.
    suffix : str, optional
        Optional suffix appended to the base name (before extension).
    add_date : bool
        If True, append UTC YYYYMMDD to the filename.

    Returns
    -------
    Path
        The full path written.
    """
    target_dir = Path(folder) if folder else DEFAULT_PROC
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = _build_stem(base=name.replace(".parquet", ""), suffix=suffix, add_date=add_date)
    filename = _ensure_parquet_name(stem)
    path = target_dir / filename

    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    df.to_parquet(path, engine="pyarrow", compression=compression, index=index)
    return path
