"""Regression checks for the in-repository TACT package migration."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tact  # noqa: E402
from tact import decider, paths, tools, transaction  # noqa: E402


def test_tact_modules_resolve_inside_main_repository():
    package_dir = (ROOT / "tact").resolve()
    for module in (tact, transaction, tools, decider, paths):
        assert Path(module.__file__).resolve().is_relative_to(package_dir)


def test_tact_discovers_new_repository_layout(monkeypatch):
    for name in (
        "FDBC_SRC_DIR",
        "FDBC_REPO_ROOT",
        "FDB_V3_DIR",
        "FDBENCH_V3_DIR",
        "FDBENCH_REPO_ROOT",
        "FDB_V3_DATA_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    assert paths.fd_badcat_src() == (ROOT / "src").resolve()
    assert paths.fdb_v3_dir() == (ROOT.parent / "FDBench_v3" / "v3").resolve()
    assert paths.default_data_dir() == (
        ROOT.parent / "FDBench_v3" / "v3" / "fdb_v3_data_released"
    ).resolve()
