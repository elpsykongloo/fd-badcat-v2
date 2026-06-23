#!/usr/bin/env python3
"""Run a seeded HumDial sample through the websocket backend."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
import soundfile as sf
import websockets
import yaml


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 256
RESPONSE_PROMPT_MARKERS = (
    "你是一个自然聊天的语音助手",
    "只回复15个字",
)
RESPONSE_PROMPT_HINTS = (
    "根据用户音频进行回应",
    "重复助手上一轮的回答",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: Path
    relpath: str
    lang: str
    category: str
    duration: float


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def wav_duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"{path} must be {SAMPLE_RATE} Hz, got {info.samplerate}")
    return info.frames / info.samplerate


def build_sample_pool(data_root: Path) -> dict[tuple[str, str], list[Path]]:
    test_root = data_root / "test"
    if not test_root.exists():
        raise FileNotFoundError(f"HumDial test root not found: {test_root}")

    groups: dict[tuple[str, str], list[Path]] = {}
    for path in sorted(test_root.rglob("*.wav")):
        if path.name.startswith("clean_"):
            continue
        rel = path.relative_to(test_root)
        if len(rel.parts) < 3:
            continue
        lang, category = rel.parts[0], rel.parts[1]
        groups.setdefault((lang, category), []).append(path)
    return groups


def choose_samples(
    data_root: Path,
    count: int,
    seed: int,
    balanced: bool,
) -> list[Sample]:
    rng = random.Random(seed)
    groups = build_sample_pool(data_root)
    selected: list[Path] = []

    if balanced:
        keys = sorted(groups)
        per_group = count // len(keys)
        remainder = count % len(keys)
        for i, key in enumerate(keys):
            take = per_group + (1 if i < remainder else 0)
            candidates = groups[key]
            if take > len(candidates):
                raise ValueError(f"Not enough samples in {key}: need {take}, have {len(candidates)}")
            selected.extend(rng.sample(candidates, take))
    else:
        all_paths = [path for paths in groups.values() for path in paths]
        if count > len(all_paths):
            raise ValueError(f"Not enough samples: need {count}, have {len(all_paths)}")
        selected = rng.sample(all_paths, count)

    test_root = data_root / "test"
    samples: list[Sample] = []
    for idx, path in enumerate(selected, start=1):
        rel = path.relative_to(test_root)
        lang, category = rel.parts[0], rel.parts[1]
        stem = safe_id("_".join(rel.with_suffix("").parts))
        samples.append(
            Sample(
                sample_id=f"{idx:04d}_{stem}",
                path=path,
                relpath=str(rel),
                lang=lang,
                category=category,
                duration=wav_duration(path),
            )
        )
    return samples


def write_manifest(samples: list[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.__dict__ | {"path": str(sample.path)}, ensure_ascii=False) + "\n")


def read_mono_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} must be {SAMPLE_RATE} Hz, got {sr}")
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32, copy=False)


def mix_segment(base: np.ndarray, segment: np.ndarray, start_sample: int) -> np.ndarray:
    if segment.ndim == 2:
        segment = segment.mean(axis=1)
    end_sample = max(start_sample, 0) + len(segment)
    if end_sample > len(base):
        base = np.pad(base, (0, end_sample - len(base)))
    if start_sample < 0:
        segment = segment[-start_sample:]
        start_sample = 0
    base[start_sample : start_sample + len(segment)] += segment
    np.clip(base, -1.0, 1.0, out=base)
    return base


def llm_event_expects_tts(data: dict[str, object]) -> bool:
    """Batch-side classifier for backend LLM events that spawn TTS.

    The backend emits the same llm_done event for control prompts and response
    prompts. Do not infer TTS from the generated text; infer it from the system
    prompt that caused the LLM call.
    """
    prompt = data.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return False
    first = prompt[0]
    if not isinstance(first, dict):
        return False
    system_prompt = first.get("content")
    if not isinstance(system_prompt, str):
        return False
    return all(marker in system_prompt for marker in RESPONSE_PROMPT_MARKERS) and any(
        hint in system_prompt for hint in RESPONSE_PROMPT_HINTS
    )


async def run_one(
    sample: Sample,
    ws_url: str,
    out_dir: Path,
    exp: str,
    lang_mode: str,
    trailing_silence: float,
    post_send_wait: float,
    sample_timeout: float,
    tts_wait_timeout: float,
) -> dict[str, object]:
    audio = read_mono_16k(sample.path)
    original_duration = len(audio) / SAMPLE_RATE
    silence = np.zeros(int(trailing_silence * SAMPLE_RATE), dtype=np.float32)
    send_audio = np.concatenate([audio, silence])

    sample_dir = out_dir / "samples" / sample.sample_id
    tts_dir = sample_dir / "tts"
    sample_dir.mkdir(parents=True, exist_ok=True)
    tts_dir.mkdir(parents=True, exist_ok=True)
    events_path = sample_dir / "events.jsonl"
    output_path = sample_dir / "output.wav"

    events: list[dict[str, object]] = []
    tts_timestamps: list[float] = []
    tts_texts: list[str] = []
    expected_tts_count = 0
    tts_wait_started_at: float | None = None
    asr_texts: list[str] = []
    tts_count = 0
    mixed = np.zeros(len(send_audio), dtype=np.float32)
    sender_done = asyncio.Event()
    last_message_at = time.perf_counter()

    async with websockets.connect(ws_url, max_size=None) as ws:
        backend_exp = f"{exp}/{sample.sample_id}"
        await ws.send(json.dumps({"event": "config", "data": {"lang": lang_mode, "exp": backend_exp}}))

        async def sender() -> None:
            frame_time = CHUNK_SAMPLES / SAMPLE_RATE
            for i in range(0, len(send_audio), CHUNK_SAMPLES):
                chunk = send_audio[i : i + CHUNK_SAMPLES]
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                await ws.send(chunk.astype(np.float32).tobytes())
                await asyncio.sleep(frame_time)
            sender_done.set()

        async def receiver() -> None:
            nonlocal expected_tts_count, last_message_at, mixed, tts_count, tts_wait_started_at
            while True:
                now = time.perf_counter()
                pending_tts = expected_tts_count > tts_count or bool(tts_timestamps)
                if pending_tts and tts_wait_started_at is None:
                    tts_wait_started_at = now
                elif not pending_tts:
                    tts_wait_started_at = None

                if (
                    pending_tts
                    and sender_done.is_set()
                    and tts_wait_timeout > 0
                    and tts_wait_started_at is not None
                    and now - tts_wait_started_at >= tts_wait_timeout
                ):
                    events.append(
                        {
                            "event": "batch_tts_wait_timeout",
                            "waited": round(now - tts_wait_started_at, 3),
                            "expected_tts_count": expected_tts_count,
                            "tts_count": tts_count,
                            "pending_tts_done": len(tts_timestamps),
                        }
                    )
                    expected_tts_count = tts_count
                    tts_timestamps.clear()
                    tts_wait_started_at = None
                    last_message_at = now
                    continue

                if (
                    sender_done.is_set()
                    and not pending_tts
                    and time.perf_counter() - last_message_at >= post_send_wait
                ):
                    return
                timeout = 1.0
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                last_message_at = time.perf_counter()

                if isinstance(msg, bytes):
                    timestamp = tts_timestamps.pop(0) if tts_timestamps else 0.0
                    data, sr = sf.read(io.BytesIO(msg), dtype="float32")
                    if sr != SAMPLE_RATE:
                        raise ValueError(f"backend returned {sr} Hz for {sample.sample_id}")
                    if data.ndim == 2:
                        data = data.mean(axis=1)
                    tts_count += 1
                    expected_tts_count = max(expected_tts_count, tts_count)
                    tts_path = tts_dir / f"tts_{tts_count:02d}.wav"
                    sf.write(str(tts_path), data, SAMPLE_RATE, subtype="PCM_16")
                    mixed = mix_segment(mixed, data, int(timestamp * SAMPLE_RATE))
                    events.append(
                        {
                            "event": "tts_audio",
                            "timestamp": timestamp,
                            "bytes": len(msg),
                            "path": str(tts_path),
                        }
                    )
                    continue

                obj = json.loads(msg)
                events.append(obj)
                event = obj.get("event")
                data = obj.get("data", {})
                if event == "tts_done":
                    tts_timestamps.append(float(data.get("timestamp", 0.0)))
                    expected_tts_count = max(expected_tts_count, tts_count + len(tts_timestamps))
                elif event == "llm_done":
                    content = str(data.get("content", ""))
                    if content and llm_event_expects_tts(data):
                        expected_tts_count += 1
                        tts_texts.append(content)
                elif event == "asr_done":
                    asr_texts.append(str(data.get("content", "")))

        started = time.perf_counter()
        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())
        try:
            await asyncio.wait_for(asyncio.gather(sender_task, receiver_task), timeout=sample_timeout)
        finally:
            sender_task.cancel()
            receiver_task.cancel()
            await ws.close()

    sf.write(str(output_path), mixed, SAMPLE_RATE, subtype="PCM_16")
    with events_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    return {
        "sample_id": sample.sample_id,
        "relpath": sample.relpath,
        "lang": sample.lang,
        "category": sample.category,
        "input_duration": round(original_duration, 3),
        "output_path": str(output_path),
        "events_path": str(events_path),
        "tts_count": tts_count,
        "asr_text": " | ".join(text for text in asr_texts if text),
        "reply_text": " | ".join(text for text in tts_texts if text),
        "elapsed": round(time.perf_counter() - started, 3),
        "status": "ok",
        "error": "",
    }


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "relpath",
        "lang",
        "category",
        "input_duration",
        "output_path",
        "events_path",
        "tts_count",
        "asr_text",
        "reply_text",
        "elapsed",
        "status",
        "error",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_summary_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def completed_sample_ids(rows: list[dict[str, object]]) -> set[str]:
    return {
        str(row["sample_id"])
        for row in rows
        if row.get("status") == "ok"
        and row.get("output_path")
        and Path(str(row["output_path"])).exists()
    }


def error_row(sample: Sample, exc: BaseException) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "relpath": sample.relpath,
        "lang": sample.lang,
        "category": sample.category,
        "input_duration": round(sample.duration, 3),
        "output_path": "",
        "events_path": "",
        "tts_count": 0,
        "asr_text": "",
        "reply_text": "",
        "elapsed": 0,
        "status": "error",
        "error": repr(exc),
    }


async def run_samples(
    *,
    samples: list[Sample],
    ws_url: str,
    out_dir: Path,
    summary_path: Path,
    exp: str,
    lang_mode: str,
    trailing_silence: float,
    post_send_wait: float,
    sample_timeout: float,
    tts_wait_timeout: float = float(os.getenv("FDBC_BATCH_TTS_WAIT_TIMEOUT", "120")),
    resume: bool,
    workers: int,
    on_row: Callable[[dict[str, object]], Awaitable[None] | None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = load_summary_rows(summary_path) if resume else []
    completed = completed_sample_ids(rows)
    rows_by_id = {str(row["sample_id"]): row for row in rows}

    pending: list[tuple[int, Sample]] = []
    for idx, sample in enumerate(samples, start=1):
        if sample.sample_id in completed:
            print(f"[{idx}/{len(samples)}] skip {sample.sample_id}", flush=True)
            continue
        pending.append((idx, sample))

    if not pending:
        write_summary(list(rows_by_id.values()), summary_path)
        return list(rows_by_id.values())

    semaphore = asyncio.Semaphore(max(1, workers))

    async def run_indexed(idx: int, sample: Sample) -> tuple[int, Sample, dict[str, object]]:
        async with semaphore:
            print(f"[{idx}/{len(samples)}] run {sample.sample_id} {sample.relpath}", flush=True)
            try:
                row = await run_one(
                    sample=sample,
                    ws_url=ws_url,
                    out_dir=out_dir,
                    exp=exp,
                    lang_mode=lang_mode,
                    trailing_silence=trailing_silence,
                    post_send_wait=post_send_wait,
                    sample_timeout=sample_timeout,
                    tts_wait_timeout=tts_wait_timeout,
                )
            except Exception as exc:
                row = error_row(sample, exc)
                print(f"[{idx}/{len(samples)}] error {sample.sample_id}: {exc!r}", flush=True)
            else:
                print(
                    f"[{idx}/{len(samples)}] ok {sample.sample_id} "
                    f"tts={row['tts_count']} elapsed={row['elapsed']}s",
                    flush=True,
                )
            return idx, sample, row

    tasks = [asyncio.create_task(run_indexed(idx, sample)) for idx, sample in pending]
    for task in asyncio.as_completed(tasks):
        _, _, row = await task
        rows_by_id[str(row["sample_id"])] = row
        write_summary(list(rows_by_id.values()), summary_path)
        if on_row is not None:
            maybe_awaitable = on_row(row)
            if maybe_awaitable is not None:
                await maybe_awaitable

    return list(rows_by_id.values())


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--data-root", default="data/HumDial-FDBench/extracted")
    parser.add_argument("--out-dir", default="logs/humdial_100")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp", default="humdial-100")
    parser.add_argument("--lang", default="batch")
    parser.add_argument("--trailing-silence", type=float, default=2.0)
    parser.add_argument("--post-send-wait", type=float, default=8.0)
    parser.add_argument("--sample-timeout", type=float, default=240.0)
    parser.add_argument("--tts-wait-timeout", type=float, default=float(os.getenv("FDBC_BATCH_TTS_WAIT_TIMEOUT", "120")))
    parser.add_argument("--no-balanced", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.getenv("FDBC_BATCH_WORKERS", "1")))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ws_url = f"ws://127.0.0.1:{cfg['server']['port']}/realtime"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = choose_samples(
        data_root=Path(args.data_root),
        count=args.count,
        seed=args.seed,
        balanced=not args.no_balanced,
    )
    write_manifest(samples, out_dir / "manifest.jsonl")

    summary_path = out_dir / "summary.csv"
    rows = await run_samples(
        samples=samples,
        ws_url=ws_url,
        out_dir=out_dir,
        summary_path=summary_path,
        exp=args.exp,
        lang_mode=args.lang,
        trailing_silence=args.trailing_silence,
        post_send_wait=args.post_send_wait,
        sample_timeout=args.sample_timeout,
        tts_wait_timeout=args.tts_wait_timeout,
        resume=args.resume,
        workers=args.workers,
    )

    ok = sum(1 for row in rows if row.get("status") == "ok")
    with_tts = sum(1 for row in rows if row.get("status") == "ok" and int(row.get("tts_count", 0)) > 0)
    errors = sum(1 for row in rows if row.get("status") != "ok")
    print(f"done total={len(rows)} ok={ok} with_tts={with_tts} errors={errors}", flush=True)
    print(f"manifest={out_dir / 'manifest.jsonl'}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
