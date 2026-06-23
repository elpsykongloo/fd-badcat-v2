#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda-envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/conda-pkgs}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$UV_CACHE_DIR"

if [[ "${USE_NETWORK_TURBO:-1}" == "1" && -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi

eval "$(conda shell.bash hook)"

ENV_NAME="${INDEX_TTS_ENV_NAME:-index-tts-vllm}"
PYTHON_VERSION="${INDEX_TTS_PYTHON_VERSION:-3.10}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-http://mirrors.aliyun.com/pypi/simple}"

mkdir -p "$ROOT_DIR/model"
if [[ ! -d "$ROOT_DIR/model/index-tts-vllm/.git" ]]; then
    rm -rf "$ROOT_DIR/model/index-tts-vllm"
    git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none \
        https://github.com/Ksuriuri/index-tts-vllm.git "$ROOT_DIR/model/index-tts-vllm"
else
    git -C "$ROOT_DIR/model/index-tts-vllm" pull --ff-only
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
        conda create --override-channels \
        -c https://repo.anaconda.com/pkgs/main \
        -c https://repo.anaconda.com/pkgs/r \
        -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip uv
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    uv pip install \
    --index-url "$PYPI_INDEX_URL" \
    -r "$ROOT_DIR/model/index-tts-vllm/requirements.txt"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    uv pip install \
    --index-url "$PYPI_INDEX_URL" \
    modelscope

python "$ROOT_DIR/scripts/patch_index_tts_vllm.py" "$ROOT_DIR/model/index-tts-vllm"

echo "Index-TTS env ready. Download weights with: bash setup/download_assets.sh index-tts"
