#!/usr/bin/env python3
"""Minimal websocket smoke test for fd-badcat backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
import yaml


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 256


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--input", default="logs/module_tts_smoke.wav")
    parser.add_argument("--output", default="logs/backend_ws_tts.wav")
    parser.add_argument("--lang", default="test")
    parser.add_argument("--exp", default="exp-1")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--trailing-silence", type=float, default=2.0)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    port = cfg["server"]["port"]
    ws_url = f"ws://127.0.0.1:{port}/realtime"

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")
    silence = np.zeros(int(args.trailing_silence * SAMPLE_RATE), dtype=np.float32)
    audio = np.concatenate([audio, silence])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events: list[str] = []
    got_tts = asyncio.Event()

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"event": "config", "data": {"lang": args.lang, "exp": args.exp}}))

        async def sender() -> None:
            frame_time = CHUNK_SAMPLES / SAMPLE_RATE
            for i in range(0, len(audio), CHUNK_SAMPLES):
                chunk = audio[i : i + CHUNK_SAMPLES]
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                await ws.send(chunk.astype(np.float32).tobytes())
                await asyncio.sleep(frame_time)

        async def receiver() -> None:
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    output_path.write_bytes(msg)
                    print(f"received_audio_bytes={len(msg)} path={output_path}", flush=True)
                    got_tts.set()
                    return
                obj = json.loads(msg)
                event = obj.get("event", "")
                events.append(event)
                print(f"event={event} data={obj.get('data', {})}", flush=True)

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        started = time.perf_counter()
        try:
            await asyncio.wait_for(got_tts.wait(), timeout=args.timeout)
        finally:
            sender_task.cancel()
            receiver_task.cancel()
            await ws.close()

    print(f"events={events}", flush=True)
    print(f"elapsed={time.perf_counter() - started:.3f}s", flush=True)
    return 0 if output_path.exists() and output_path.stat().st_size > 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
