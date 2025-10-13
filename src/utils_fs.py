from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Optional, Iterator, Tuple
import os
import math
import pandas as pd

__all__ = [
    "sizeof_fmt",
    "walk_files",
    "build_file_inventory",
    "print_tree_with_sizes",
]

def sizeof_fmt(num_bytes: int) -> str:
    """Human-readable size for bytes (e.g., 1.2 MB)."""
    if num_bytes is None:
        return "-"
    if num_bytes == 0:
        return "0 B"
    units = ["B","KB","MB","GB","TB","PB","EB"]
    i = int(math.floor(math.log(num_bytes, 1024))) if num_bytes > 0 else 0
    i = min(i, len(units)-1)
    p = math.pow(1024, i)
    s = num_bytes / p
    if s >= 100 or i == 0:
        return f"{int(round(s))} {units[i]}"
    return f"{s:.1f} {units[i]}"

def _is_hidden(path: Path) -> bool:
    name = path.name
    return name.startswith(".") or name.startswith("~")

def _should_skip(path: Path, exclude_dirs: Sequence[str]) -> bool:
    # skip if any path part equals an excluded directory name
    parts = set(path.parts)
    return any(ed in parts for ed in exclude_dirs)

def walk_files(
    root: Path,
    *,
    include_hidden: bool = True,
    exclude_dirs: Sequence[str] = (),
) -> Iterator[Path]:
    """
    Yield all file paths under `root`, honoring hidden/excludes.
    Does not follow symlinks.
    """
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)

        # In-place prune excluded directories for efficiency
        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not _is_hidden(Path(d))]

        for fname in filenames:
            p = current / fname
            if not include_hidden and _is_hidden(p):
                continue
            yield p

def build_file_inventory(
    root: Path,
    *,
    include_hidden: bool = True,
    exclude_dirs: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Return a DataFrame of all files under `root` with sizes.
    Columns: ['rel_path','name','ext','size_bytes','size_hr','parent','depth','abs_path']
    """
    root = Path(root).resolve()
    rows = []
    for p in walk_files(root, include_hidden=include_hidden, exclude_dirs=exclude_dirs):
        try:
            st = p.stat()
            size = int(st.st_size)
        except (FileNotFoundError, PermissionError):
            size = None
        rel = p.relative_to(root)
        rows.append({
            "rel_path": str(rel).replace("\\", "/"),
            "name": p.name,
            "ext": p.suffix.lower(),
            "size_bytes": size,
            "size_hr": sizeof_fmt(size) if size is not None else "-",
            "parent": str(rel.parent).replace("\\", "/"),
            "depth": len(rel.parts),
            "abs_path": str(p),
        })
    df = pd.DataFrame(rows).sort_values(["size_bytes","rel_path"], ascending=[False, True]).reset_index(drop=True)
    return df

def print_tree_with_sizes(
    root: Path,
    *,
    include_hidden: bool = True,
    exclude_dirs: Sequence[str] = (),
    max_depth: Optional[int] = None,
    file_limit_per_dir: Optional[int] = None,
) -> None:
    """
    Pretty-print a tree with file sizes. (Non-destructive; for quick inspection.)
    """
    root = Path(root).resolve()
    base_len = len(root.parts)
    counts_in_dir = {}

    for p in walk_files(root, include_hidden=include_hidden, exclude_dirs=exclude_dirs):
        rel = p.relative_to(root)
        depth = len(rel.parts)
        if max_depth is not None and depth > max_depth:
            continue

        parent = rel.parent
        counts_in_dir[parent] = counts_in_dir.get(parent, 0) + 1
        if file_limit_per_dir is not None and counts_in_dir[parent] > file_limit_per_dir:
            continue  # skip extra files in very large folders

        try:
            size = p.stat().st_size
        except (FileNotFoundError, PermissionError):
            size = None

        indent = "    " * (depth - 1)
        print(f"{indent}├── {rel.name}  [{sizeof_fmt(size)}]")
