#############
# src/config.py  #
#############

"""
Project configuration paths and file guards.

Provides
--------
- ROOT : Path
    Repository root (this file lives in src/, so parent.parent == repo root).
- DATA, RAW, PROC, EXT : Path
    Canonical data directories used by notebooks/scripts. They are created
    (mkdir with exist_ok=True) on import as a safe no-op.
- OPSD_60min_CSV : Path
    Path to the OPSD hourly base CSV (original upstream filename).
- require_file(p, hint) -> Path
    Helper to assert that a required file exists, raising FileNotFoundError
    with an optional hint if it does not.
"""

from pathlib import Path

# (1) Repo root: this file lives in src/, so parent.parent = repo root
ROOT = Path(__file__).resolve().parent.parent

# (2) Canonical directories used by jupyter notebooks
DATA = ROOT / "data"
RAW  = DATA / "raw"
PROC = DATA / "processed"
EXT  = DATA / "external"

# (3) Important files (original upstream names)
# (3.1) OPSD hourly data base, 
OPSD_60min_CSV = RAW / "OPSD_time_series_60min_singleindex.csv"

# (4) Ensure basic dirs exist (safe no-op)
for d in (DATA, RAW, PROC, EXT):
    d.mkdir(parents=True, exist_ok=True)

# (5) Guard for missing files
def require_file(p: Path, hint: str = "") -> Path:
    """
    Ensure a required file exists.

    Parameters
    ----------
    p : Path
        Path to the required file.
    hint : str, optional
        Optional message appended to the error to help the user locate or
        obtain the file.

    Returns
    -------
    Path
        The same path `p` if the file exists.

    Raises
    ------
    FileNotFoundError
        If `p` does not exist.
    """
    if not p.exists():
        msg = f"This required file has not been found: {p}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)
    return p
