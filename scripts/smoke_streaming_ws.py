#!/usr/bin/env python3
"""Serial whole-WAV vs pcm16.v1 WebSocket probe (synthetic/user-owned input).

No benchmark corpus is selected automatically. Playback acknowledgements use a
paced client simulator, not an actual speaker: report arrival latency, NOT
microphone-to-acoustic latency. --output is a JSON receipt, not a score report.
"""
import argparse
import asyncio
import io
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speech_stream import PCM_HEADER


async def trial(url, audio, streaming):
    events, packets = [], 0
    received = played = 0
    rate = 0
    sid = None
    seq = 0
    t0 = time.perf_counter()
    first_audio = None
    last_play_at = None
    done = asyncio.Event()
    play_queue = asyncio.Queue()
    pcm_pieces = []
    async with websockets.connect(url, max_size=16 * 1024 * 1024, proxy=None) as ws:
        await ws.send(json.dumps({"event": "config", "data": {
            "exp": f"stream-probe-{uuid.uuid4().hex}", "lang": "synthetic",
            **({"audio_protocol": "pcm16.v1"} if streaming else {})}}))

        async def play():
            nonlocal played
            while True:
                deadline, count = await play_queue.get()
                await asyncio.sleep(max(0, deadline - time.perf_counter()))
                played = count
                await ws.send(json.dumps({"event": "playback_progress", "data": {
                    "utterance_id": sid, "played_samples": played}}))

        async def receive():
            nonlocal packets, received, rate, sid, seq, first_audio, last_play_at
            async for msg in ws:
                now = time.perf_counter()
                if isinstance(msg, str):
                    obj = json.loads(msg)
                    kind, data = obj.get("event"), obj.get("data") or {}
                    # Record compact non-sensitive timings; only model output text.
                    events.append({"event": kind, "at_s": now - t0,
                                   **{k: data[k] for k in ("kind", "infer_time", "content", "text", "samples", "error") if k in data}})
                    if kind in ("error", "speech_error", "speech_cancelled"):
                        raise RuntimeError(f"Unexpected {kind}: {data}")
                    if kind == "speech_start":
                        sid = data["utterance_id"]
                    if kind == "speech_audio_end":
                        assert received == data["samples"]
                        done.set()
                else:
                    if first_audio is None:
                        first_audio = now - t0
                    packets += 1
                    if streaming:
                        magic, packet_id, index, packet_rate = PCM_HEADER.unpack(msg[:16])
                        assert magic == b"FDS1" and packet_id == sid and index == seq
                        assert not rate or packet_rate == rate
                        rate = packet_rate
                        seq += 1
                        pcm_pieces.append(msg[16:])
                        count = len(msg[16:]) // 2
                        received += count
                        assert received - played <= int(rate * .6) + 1
                        last_play_at = max(last_play_at or now + .08, now) + count / rate
                        await play_queue.put((last_play_at, received))
                    else:
                        wav, rate = sf.read(io.BytesIO(msg), dtype="int16")
                        received = len(wav)
                        pcm_pieces.append(wav.tobytes())
                        done.set()
                if done.is_set():
                    return

        async def microphone():
            # Include quiet tail so the frozen audio-clock END_HOLD can fire.
            data = np.pad(audio, (0, 16000 * 30))
            anchor = time.perf_counter()
            for pos in range(0, len(data), 256):
                if done.is_set():
                    return
                await asyncio.sleep(max(0, anchor + pos / 16000 - time.perf_counter()))
                await ws.send(np.asarray(data[pos:pos + 256], dtype="<f4").tobytes())

        tasks = [asyncio.create_task(receive()), asyncio.create_task(microphone()), asyncio.create_task(play())]
        try:
            await asyncio.wait_for(tasks[0], 60)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    hold = next(e["at_s"] for e in events if e["event"] == "vad_640_done")
    text_done = next(e for e in events if e["event"] == "llm_done" and e.get("kind") == "response")
    return {"streaming": streaming, "post_hold_first_audio_ms": round((first_audio - hold) * 1000, 1),
            "post_hold_text_done_ms": round((text_done["at_s"] - hold) * 1000, 1),
            "response": text_done.get("content"), "packets": packets,
            "sample_rate": rate, "audio_samples": received, "events": events}, b"".join(pcm_pieces)


async def main(args):
    audio, sr = sf.read(args.input, dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("Input must be mono 16 kHz WAV")
    rows = []
    for repeat in range(args.repeats):
        # Alternate order to expose warmup/order effects rather than hide them.
        for streaming in ([False, True] if repeat % 2 == 0 else [True, False]):
            row, pcm = await trial(args.url, audio, streaming)
            row["repeat"] = repeat
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != "events"}, ensure_ascii=False), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                name = args.output.with_name(f"probe_{repeat}_{'stream' if streaming else 'whole'}.wav")
                sf.write(name, np.frombuffer(pcm, dtype="<i2"), row["sample_rate"], subtype="PCM_16")
                args.output.write_text(json.dumps({"scope": "serial synthetic WebSocket smoke; simulated playback",
                                                   "input": str(args.input), "trials": rows}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--url", default="ws://127.0.0.1:18000/realtime")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    asyncio.run(main(parser.parse_args()))
