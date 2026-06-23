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

ENV_NAME="${QWEN_ENV_NAME:-fdbc-qwen3o-vllm}"
PYTHON_VERSION="${QWEN_PYTHON_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
VLLM_OMNI_VERSION="${VLLM_OMNI_VERSION:-0.22.0}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-http://mirrors.aliyun.com/pypi/simple}"

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
    uv pip install --upgrade \
    --index-url "$PYPI_INDEX_URL" \
    "vllm==$VLLM_VERSION" \
    "vllm-omni==$VLLM_OMNI_VERSION" \
    qwen-omni-utils \
    accelerate \
    modelscope

python - <<'PY'
import importlib.metadata as metadata

for package in ("vllm", "vllm-omni", "qwen-omni-utils"):
    print(f"{package}=={metadata.version(package)}")
PY

python "$ROOT_DIR/scripts/patch_prometheus_instrumentator.py"
