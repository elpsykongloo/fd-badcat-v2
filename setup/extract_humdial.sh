#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${FDBC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ZIP_PATH="${HUMDIAL_ZIP:-$ROOT_DIR/data/HumDial-FDBench/Humdial-Track2-Test.zip}"
EXTRACT_DIR="${HUMDIAL_EXTRACT_DIR:-$ROOT_DIR/data/HumDial-FDBench/extracted}"

if ! command -v unzip >/dev/null 2>&1; then
  echo "unzip is required. Install it with: apt-get install -y unzip" >&2
  exit 1
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "HumDial zip not found: $ZIP_PATH" >&2
  echo "Run: bash setup/download_assets.sh data" >&2
  exit 1
fi

mkdir -p "$EXTRACT_DIR"

echo "Extracting HumDial zip:"
echo "  zip:  $ZIP_PATH"
echo "  dest: $EXTRACT_DIR"
unzip -q -n "$ZIP_PATH" -d "$EXTRACT_DIR"

wav_count="$(find "$EXTRACT_DIR/test" -type f -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
json_count="$(find "$EXTRACT_DIR/test" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "HumDial extracted: wav=$wav_count json=$json_count"
