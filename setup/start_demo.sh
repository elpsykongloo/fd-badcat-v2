#!/usr/bin/env bash
# Foreground launcher; run in tmux to keep it alive after SSH disconnects.
set -euo pipefail
DEMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEMO_ROOT"
DEMO_PYTHON="${FDBC_DEMO_PYTHON:-/root/miniconda3/envs/fd-sds/bin/python}"
if [[ ! -x "$DEMO_PYTHON" ]]; then
    echo "Backend Python not found: $DEMO_PYTHON. Set FDBC_DEMO_PYTHON." >&2
    exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
[[ "$OMP_NUM_THREADS" != "0" ]] || export OMP_NUM_THREADS=8
[[ "$MKL_NUM_THREADS" != "0" ]] || export MKL_NUM_THREADS=8
exec "$DEMO_PYTHON" scripts/serve_demo.py "$@"
