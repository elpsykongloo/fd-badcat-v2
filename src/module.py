import io
import json
import os
from pathlib import Path

import requests
import sherpa_onnx
import soundfile as sf
import torch
import torchaudio


ROOT_DIR = Path(__file__).resolve().parents[1]

ASR_DIR = Path(
    os.getenv(
        "FDBC_ASR_DIR",
        ROOT_DIR / "model" / "sherpa-onnx-paraformer-zh-2024-03-09",
    )
)
ASR_PROVIDER = os.getenv("FDBC_ASR_PROVIDER", "cpu")
ASR_NUM_THREADS = int(os.getenv("FDBC_ASR_NUM_THREADS", "2"))

INDEX_TTS_URL = os.getenv("FDBC_INDEX_TTS_URL", "http://127.0.0.1:19000/tts")
INDEX_TTS_CHARACTER = os.getenv("FDBC_INDEX_TTS_CHARACTER", "jay_klee")
TTS_PROVIDER = os.getenv("FDBC_TTS_PROVIDER", "index").strip().lower()

QWEN_URL = os.getenv("FDBC_QWEN_URL", "http://127.0.0.1:10004/v1/chat/completions")
QWEN_MODEL = os.getenv("FDBC_QWEN_MODEL", "Qwen3-Omni-30B-A3B-Instruct")
OMNI_TTS_URL = os.getenv("FDBC_OMNI_TTS_URL", QWEN_URL)
OMNI_TTS_SYSTEM_PROMPT = os.getenv(
    "FDBC_OMNI_TTS_SYSTEM_PROMPT",
    "You are a text-to-speech engine. Speak the user's text exactly, without adding or omitting content.",
)

_ASR_MODEL = None
_LOCAL_HTTP = requests.Session()
_LOCAL_HTTP.trust_env = False


def _get_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is not None:
        return _ASR_MODEL

    model_path = ASR_DIR / "model.onnx"
    tokens_path = ASR_DIR / "tokens.txt"
    missing = [str(p) for p in (model_path, tokens_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "ASR model is incomplete. Run setup/download_assets.sh asr first. "
            f"Missing: {', '.join(missing)}"
        )

    _ASR_MODEL = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(model_path),
        tokens=str(tokens_path),
        num_threads=ASR_NUM_THREADS,
        provider=ASR_PROVIDER,
    )
    return _ASR_MODEL


def _call_index_tts(text: str) -> bytes:
    payload = {
        "text": text,
        "character": INDEX_TTS_CHARACTER,
    }
    response = _LOCAL_HTTP.post(INDEX_TTS_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.content


def _extract_omni_audio(response_data: dict) -> bytes:
    choices = response_data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Qwen3-Omni TTS response has no choices: {response_data}")
    message = choices[0].get("message") or {}
    audio = message.get("audio") or {}
    audio_b64 = audio.get("data")
    if not audio_b64:
        raise RuntimeError(f"Qwen3-Omni TTS response has no audio.data: {response_data}")
    import base64

    return base64.b64decode(audio_b64)


def _call_omni_tts(text: str) -> bytes:
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": OMNI_TTS_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "modalities": ["audio"],
        "temperature": float(os.getenv("FDBC_OMNI_TTS_TEMPERATURE", "0")),
        "max_tokens": int(os.getenv("FDBC_OMNI_TTS_MAX_TOKENS", "256")),
        "seed": int(os.getenv("FDBC_OMNI_TTS_SEED", "42")),
    }
    response = _LOCAL_HTTP.post(
        OMNI_TTS_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=int(os.getenv("FDBC_OMNI_TTS_TIMEOUT", "600")),
    )
    response.raise_for_status()
    return _extract_omni_audio(response.json())


def _mono_16k(audio, sr: int):
    tensor = torch.as_tensor(audio, dtype=torch.float32)
    if tensor.ndim == 2:
        # soundfile returns [time, channels]; ASR/TTS pipeline expects mono.
        tensor = tensor.mean(dim=1)
    if sr != 16000:
        tensor = torchaudio.functional.resample(tensor.unsqueeze(0), sr, 16000).squeeze(0)
    return tensor.cpu().numpy()


def tts(text, path):
    if TTS_PROVIDER in {"omni", "qwen3omni", "qwen3-omni"}:
        wav_bytes = _call_omni_tts(text)
    elif TTS_PROVIDER in {"index", "indextts", "index-tts"}:
        wav_bytes = _call_index_tts(text)
    else:
        raise ValueError(f"Unsupported FDBC_TTS_PROVIDER={TTS_PROVIDER!r}")

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    data = _mono_16k(data, sr)
    sf.write(str(path), data, 16000, subtype="PCM_16")

    return str(path)


def asr(path):
    audio, sr = sf.read(path, dtype="float32")
    audio = _mono_16k(audio, sr)
    stream = _get_asr_model().create_stream()
    stream.accept_waveform(16000, audio)
    _get_asr_model().decode_stream(stream)
    return str(stream.result.text).strip()


def llm_qwen3o(messages: list):
    payload = {
        "model": QWEN_MODEL,
        "temperature": float(os.getenv("FDBC_QWEN_TEMPERATURE", "0")),
        "top_p": float(os.getenv("FDBC_QWEN_TOP_P", "0.7")),
        "top_k": int(os.getenv("FDBC_QWEN_TOP_K", "40")),
        "presence_penalty": float(os.getenv("FDBC_QWEN_PRESENCE_PENALTY", "1.2")),
        "frequency_penalty": float(os.getenv("FDBC_QWEN_FREQUENCY_PENALTY", "0.8")),
        "max_tokens": int(os.getenv("FDBC_QWEN_MAX_TOKENS", "256")),
        "seed": int(os.getenv("FDBC_QWEN_SEED", "42")),
        "modalities": ["text"],
        "messages": messages,
    }
    try:
        response = _LOCAL_HTTP.post(
            QWEN_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=int(os.getenv("FDBC_QWEN_TIMEOUT", "300")),
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[QWEN REQUEST ERROR] {exc}")
        return ""
