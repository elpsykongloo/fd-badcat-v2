#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"

export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda-envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/root/autodl-tmp/conda-pkgs}"

if [[ "${USE_NETWORK_TURBO:-1}" == "1" && -f /etc/network_turbo ]]; then
    # AutoDL academic accelerator for GitHub/Hugging Face.
    # Disable with USE_NETWORK_TURBO=0.
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

want_target() {
    local target="$1"
    shift
    if [[ "$#" -eq 0 ]]; then
        return 0
    fi
    for item in "$@"; do
        [[ "$item" == "all" || "$item" == "$target" ]] && return 0
    done
    return 1
}

hf_aria2_download() {
    local repo_id="$1"
    local repo_type="$2"
    local local_dir="$3"
    local input_file="$4"
    shift 4

    python scripts/hf_aria2_download.py \
        --repo-id "$repo_id" \
        --repo-type "$repo_type" \
        --local-dir "$local_dir" \
        --input-file "$input_file" \
        "$@"

    unset all_proxy ALL_PROXY
    aria2c \
        --input-file="$input_file" \
        --continue=true \
        --max-connection-per-server="${ARIA2_X:-8}" \
        --split="${ARIA2_S:-8}" \
        --max-concurrent-downloads="${ARIA2_J:-4}" \
        --min-split-size="${ARIA2_K:-1M}" \
        --file-allocation=none \
        --auto-file-renaming=false \
        --allow-overwrite=true \
        --retry-wait="${ARIA2_RETRY_WAIT:-10}" \
        --max-tries=0 \
        --timeout=60 \
        --connect-timeout=30 \
        --console-log-level=warn \
        --summary-interval=30
}

download_asr() {
    local asr_dir="$ROOT_DIR/model/sherpa-onnx-paraformer-zh-2024-03-09"
    local asr_tar="$ROOT_DIR/model/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"
    local asr_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"

    mkdir -p "$ROOT_DIR/model"
    if [[ -f "$asr_dir/model.onnx" && -f "$asr_dir/tokens.txt" ]]; then
        echo "ASR model already exists: $asr_dir"
        return
    fi

    if [[ "${ASR_SOURCE:-hf}" == "hf" ]]; then
        hf_aria2_download \
            "csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09" \
            "model" \
            "$asr_dir" \
            "$ROOT_DIR/.aria2/sherpa-onnx-paraformer-zh-2024-03-09.txt" \
            --include "README.md" \
            --include "am.mvn" \
            --include "config.yaml" \
            --include "model.onnx" \
            --include "tokens.txt"
        return
    fi

    unset all_proxy ALL_PROXY
    aria2c \
        --continue=true \
        --max-connection-per-server="${ASR_ARIA2_X:-4}" \
        --split="${ASR_ARIA2_S:-4}" \
        --min-split-size="${ARIA2_K:-1M}" \
        --file-allocation=none \
        --console-log-level=warn \
        --summary-interval=60 \
        --dir="$ROOT_DIR/model" \
        --out="$(basename "$asr_tar")" \
        "$asr_url"
    tar xf "$asr_tar" -C "$ROOT_DIR/model"
    rm -f "$asr_tar"
}

download_sensevoice() {
    local sv_dir="$ROOT_DIR/model/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    local sv_tar="$sv_dir.tar.bz2"
    local sv_url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"

    mkdir -p "$ROOT_DIR/model"
    if [[ ( -f "$sv_dir/model.int8.onnx" || -f "$sv_dir/model.onnx" ) && -f "$sv_dir/tokens.txt" ]]; then
        echo "SenseVoice model already exists: $sv_dir"
        return
    fi

    unset all_proxy ALL_PROXY
    aria2c \
        --continue=true \
        --max-connection-per-server="${ASR_ARIA2_X:-4}" \
        --split="${ASR_ARIA2_S:-4}" \
        --min-split-size="${ARIA2_K:-1M}" \
        --file-allocation=none \
        --console-log-level=warn \
        --summary-interval=60 \
        --dir="$ROOT_DIR/model" \
        --out="$(basename "$sv_tar")" \
        "$sv_url"
    tar xf "$sv_tar" -C "$ROOT_DIR/model"
    rm -f "$sv_tar"
}

download_data() {
    local data_dir="$ROOT_DIR/data/HumDial-FDBench"
    local input_file="$ROOT_DIR/.aria2/humdial-fdbench.txt"
    ARIA2_X="${DATA_ARIA2_X:-16}" \
    ARIA2_S="${DATA_ARIA2_S:-16}" \
    ARIA2_J="${DATA_ARIA2_J:-2}" \
    hf_aria2_download \
        "ASLP-lab/HumDial-FDBench" \
        "dataset" \
        "$data_dir" \
        "$input_file"

    local zip_path="$data_dir/Humdial-Track2-Test.zip"
    local extract_dir="$data_dir/extracted"
    if [[ "${EXTRACT_DATA:-1}" == "1" && -f "$zip_path" && ! -d "$extract_dir" ]]; then
        mkdir -p "$extract_dir"
        python - "$zip_path" "$extract_dir" <<'PY'
import sys
import zipfile

zip_path, extract_dir = sys.argv[1:3]
with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(extract_dir)
print(f"Extracted {zip_path} to {extract_dir}")
PY
    fi
}

download_qwen() {
    ARIA2_X="${QWEN_ARIA2_X:-16}" \
    ARIA2_S="${QWEN_ARIA2_S:-16}" \
    ARIA2_J="${QWEN_ARIA2_J:-2}" \
    ARIA2_RETRY_WAIT="${QWEN_ARIA2_RETRY_WAIT:-20}" \
    hf_aria2_download \
        "${QWEN_HF_REPO:-Qwen/Qwen3-Omni-30B-A3B-Instruct}" \
        "model" \
        "$ROOT_DIR/model/Qwen3-Omni-30B-A3B-Instruct" \
        "$ROOT_DIR/.aria2/qwen3-omni-30b-a3b-instruct.txt"
}

prepare_index_tts_repo() {
    mkdir -p "$ROOT_DIR/model"
    if [[ ! -d "$ROOT_DIR/model/index-tts-vllm/.git" ]]; then
        rm -rf "$ROOT_DIR/model/index-tts-vllm"
        git -c http.version=HTTP/1.1 clone --depth 1 --filter=blob:none \
            https://github.com/Ksuriuri/index-tts-vllm.git "$ROOT_DIR/model/index-tts-vllm"
    else
        git -C "$ROOT_DIR/model/index-tts-vllm" pull --ff-only
    fi
}

download_index_tts_weights() {
    prepare_index_tts_repo
    if [[ -d "$ROOT_DIR/model/index-tts-vllm/checkpoints/Index-TTS-1.5-vLLM" ]]; then
        echo "Index-TTS checkpoint already exists."
        return
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required for ModelScope Index-TTS download." >&2
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    if ! conda env list | awk '{print $1}' | grep -qx "index-tts-vllm"; then
        echo "Create index-tts-vllm first: bash setup/indextts_env.sh" >&2
        exit 1
    fi
    conda run -n index-tts-vllm modelscope download \
        --model "${INDEX_TTS_MODELSCOPE_REPO:-kusuriuri/Index-TTS-1.5-vLLM}" \
        --local_dir "$ROOT_DIR/model/index-tts-vllm/checkpoints/Index-TTS-1.5-vLLM"
}

main() {
    local targets=("$@")
    [[ "${#targets[@]}" -gt 0 ]] || targets=("all")

    require_cmd aria2c
    require_cmd python
    mkdir -p "$ROOT_DIR/model" "$ROOT_DIR/data" "$ROOT_DIR/.aria2"

    if want_target qwen "${targets[@]}"; then
        download_qwen
    fi
    if want_target data "${targets[@]}"; then
        download_data
    fi
    if want_target asr "${targets[@]}"; then
        download_asr
    fi
    if want_target sensevoice "${targets[@]}"; then
        download_sensevoice
    fi
    if want_target index-tts-repo "${targets[@]}"; then
        prepare_index_tts_repo
    fi
    if want_target index-tts "${targets[@]}"; then
        download_index_tts_weights
    fi
}

main "$@"
