#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${FDBC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-fd-sds}"
CONFIG="${CONFIG:-$ROOT_DIR/src/config.yaml}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/logs/humdial_smoke}"
TIMEOUT="${TIMEOUT:-300}"
TRAILING_SILENCE="${TRAILING_SILENCE:-2.0}"
LANG_MODE="${LANG_MODE:-test}"
EXP_NAME="${EXP_NAME:-humdial-smoke}"

CN_INPUT="${CN_INPUT:-$ROOT_DIR/data/HumDial-FDBench/extracted/test/cn_test_nondev/pause/0009_0002.wav}"
EN_INPUT="${EN_INPUT:-$ROOT_DIR/data/HumDial-FDBench/extracted/test/en_test_nondev/pause/0006_0052.wav}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost,0.0.0.0}"
export no_proxy="${no_proxy:-127.0.0.1,localhost,0.0.0.0}"

cd "$ROOT_DIR"
bash setup/extract_humdial.sh
mkdir -p "$OUT_DIR"

run_one() {
  local label="$1"
  local input="$2"
  local output="$3"

  if [[ ! -f "$input" ]]; then
    echo "Missing sample input: $input" >&2
    exit 1
  fi

  echo "Running HumDial smoke sample: $label"
  echo "  input:  $input"
  echo "  output: $output"
  conda run -n "$CONDA_ENV" python scripts/smoke_backend_ws.py \
    --config "$CONFIG" \
    --input "$input" \
    --output "$output" \
    --lang "$LANG_MODE" \
    --exp "$EXP_NAME" \
    --timeout "$TIMEOUT" \
    --trailing-silence "$TRAILING_SILENCE"
}

run_one "cn_pause_0009_0002" "$CN_INPUT" "$OUT_DIR/cn_pause_0009_0002_output.wav"
run_one "en_pause_0006_0052" "$EN_INPUT" "$OUT_DIR/en_pause_0006_0052_output.wav"

python - <<'PY'
import wave
from pathlib import Path

for path in sorted(Path("logs/humdial_smoke").glob("*.wav")):
    with wave.open(str(path), "rb") as wav:
        duration = wav.getnframes() / wav.getframerate()
        print(
            f"{path}: {wav.getframerate()} Hz, "
            f"{wav.getnchannels()} ch, {duration:.3f}s, {path.stat().st_size} bytes"
        )
PY
