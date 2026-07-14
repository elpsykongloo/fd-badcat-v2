#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/fd-sds/bin/python}"

export FDBC_REPO_ROOT="${FDBC_REPO_ROOT:-/root/autodl-tmp/fd-badcat}"
export FDBC_SRC_DIR="${FDBC_SRC_DIR:-$FDBC_REPO_ROOT/src}"
export FDBENCH_REPO_ROOT="${FDBENCH_REPO_ROOT:-/root/autodl-tmp/FDBench_v3}"
export FDB_V3_DIR="${FDB_V3_DIR:-$FDBENCH_REPO_ROOT/v3}"
export FDB_V3_DATA_DIR="${FDB_V3_DATA_DIR:-$FDB_V3_DIR/fdb_v3_data_released}"
export PYTHONPATH="$ROOT_DIR/..:$FDBC_SRC_DIR:$FDB_V3_DIR:${PYTHONPATH:-}"

MODE="${MODE:-blocking}"
PROVIDER="${PROVIDER:-tact_blocking}"

exec "$PYTHON_BIN" -m tact.offline_runner \
  --data "$FDB_V3_DATA_DIR" \
  --provider "$PROVIDER" \
  --mode "$MODE" \
  "$@"
