"""Path discovery for the in-repo fd-badcat source and sibling FDBench_v3."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable


TACT_DIR = Path(__file__).resolve().parent
PACKAGE_PARENT = TACT_DIR.parent
# New layout: <workspace>/fd-badcat/tact (PACKAGE_PARENT is the repo root).
# Legacy layout: <workspace>/tact (PACKAGE_PARENT is the workspace root).
REPO_ROOT = PACKAGE_PARENT if (PACKAGE_PARENT / "src").is_dir() else None
WORKSPACE_DIR = REPO_ROOT.parent if REPO_ROOT is not None else PACKAGE_PARENT


def _existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path.resolve()
    return None


def fd_badcat_src() -> Path | None:
    src_env = os.getenv("FDBC_SRC_DIR")
    root_env = os.getenv("FDBC_REPO_ROOT")
    return _existing(
        [
            Path(src_env) if src_env else None,
            Path(root_env) / "src" if root_env else None,
            REPO_ROOT / "src" if REPO_ROOT else None,
            WORKSPACE_DIR / "fd-badcat" / "src",
        ]
    )


def fdb_v3_dir() -> Path | None:
    v3_env = os.getenv("FDB_V3_DIR") or os.getenv("FDBENCH_V3_DIR")
    root_env = os.getenv("FDBENCH_REPO_ROOT")
    return _existing(
        [
            Path(v3_env) if v3_env else None,
            Path(root_env) / "v3" if root_env else None,
            WORKSPACE_DIR / "FDBench_v3" / "v3",
        ]
    )


def default_data_dir() -> Path | None:
    data_env = os.getenv("FDB_V3_DATA_DIR")
    v3 = fdb_v3_dir()
    return _existing(
        [
            Path(data_env) if data_env else None,
            v3 / "fdb_v3_data_released" if v3 else None,
        ]
    )


def add_to_syspath(path: Path | None) -> Path | None:
    if path is None:
        return None
    resolved = path.resolve()
    text = str(resolved)
    if text not in sys.path:
        sys.path.insert(0, text)
    return resolved


def configure_external_paths() -> dict[str, str | None]:
    """Add discovered external source roots to sys.path and return what was used."""
    fdbc = add_to_syspath(fd_badcat_src())
    fdb = add_to_syspath(fdb_v3_dir())
    return {
        "fd_badcat_src": str(fdbc) if fdbc else None,
        "fdb_v3_dir": str(fdb) if fdb else None,
    }
