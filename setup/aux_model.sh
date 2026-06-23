#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda-envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/conda-pkgs}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$UV_CACHE_DIR"

eval "$(conda shell.bash hook)"

ENV_NAME="${FDSDS_ENV_NAME:-fd-sds}"
PYTHON_VERSION="${FDSDS_PYTHON_VERSION:-3.10}"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
        conda create --override-channels \
        -c https://repo.anaconda.com/pkgs/main \
        -c https://repo.anaconda.com/pkgs/r \
        -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip uv

TORCH_VERSION="${FDSDS_TORCH_VERSION:-2.9.1+cu128}"
TORCHAUDIO_VERSION="${FDSDS_TORCHAUDIO_VERSION:-2.9.1+cu128}"
TORCH_INDEX_URL="${FDSDS_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-http://mirrors.aliyun.com/pypi/simple}"

env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    uv pip install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    uv pip install \
    --index-url "$PYPI_INDEX_URL" \
    -r "$ROOT_DIR/requirements.txt"

bash "$ROOT_DIR/setup/download_assets.sh" asr

ASR_DIR="$ROOT_DIR/model/sherpa-onnx-paraformer-zh-2024-03-09"
if [[ -d "$ASR_DIR" ]]; then
    echo "ASR model ready: $ASR_DIR"
else
    echo "ASR model missing: $ASR_DIR" >&2
    exit 1
fi
