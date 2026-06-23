#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path

import requests


def extract_audio_b64(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"missing choices in response: {response}")
    message = choices[0].get("message") or {}
    audio = message.get("audio") or {}
    data = audio.get("data")
    if not data:
        raise RuntimeError(f"missing choices[0].message.audio.data: {json.dumps(response)[:2000]}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Qwen3-Omni audio output through vllm-omni.")
    parser.add_argument("--url", default="http://127.0.0.1:10003/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--out", default="logs/qwen3_omni_audio_smoke.wav")
    parser.add_argument("--text", default="用中文自然地说一句：你好，我已经可以直接生成语音了。")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.text}],
        "modalities": ["audio"],
        "temperature": 0,
        "max_tokens": 128,
        "seed": 42,
    }

    session = requests.Session()
    session.trust_env = False
    response = session.post(args.url, json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    wav_b64 = extract_audio_b64(data)
    wav_bytes = base64.b64decode(wav_b64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav_bytes)
    print(json.dumps({"out": str(out), "bytes": len(wav_bytes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
