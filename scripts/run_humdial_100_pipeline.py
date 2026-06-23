#!/usr/bin/env python3
"""Concurrent HumDial 100-sample generation and evaluation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import yaml

import evaluate_humdial_100 as eval100
from run_humdial_batch import Sample, choose_samples, run_samples, safe_id, wav_duration, write_manifest


DEFAULT_SKIP_CATEGORIES = {"pause", "others_talk_to_user_before"}


def clean_counterparts(samples: list[Sample], data_root: Path, skip_categories: set[str]) -> list[Sample]:
    test_root = data_root / "test"
    clean_samples: list[Sample] = []
    for sample in samples:
        if sample.category in skip_categories:
            continue
        clean_path = sample.path.with_name(f"clean_{sample.path.name}")
        if not clean_path.exists():
            raise FileNotFoundError(f"Missing clean counterpart: {clean_path}")
        rel = clean_path.relative_to(test_root)
        clean_samples.append(
            Sample(
                sample_id=f"clean_{sample.sample_id}",
                path=clean_path,
                relpath=str(rel),
                lang=sample.lang,
                category=sample.category,
                duration=wav_duration(clean_path),
            )
        )
    return clean_samples


def count_rows(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ok = sum(1 for row in rows if row.get("status") == "ok")
    errors = sum(1 for row in rows if row.get("status") != "ok")
    with_tts = sum(1 for row in rows if row.get("status") == "ok" and int(row.get("tts_count", 0) or 0) > 0)
    return ok, with_tts, errors


def deepseek_enabled(args: argparse.Namespace) -> bool:
    return not args.no_judge and bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def run_eval_pass(args: argparse.Namespace, final: bool = False) -> None:
    eval_root = Path(args.eval_root)
    stats = eval100.sync_eval_layout(
        eval_root,
        manifest_path=Path(args.noisy_out_dir) / "manifest.jsonl",
        noisy_summary_path=Path(args.noisy_out_dir) / "summary.csv",
        clean_summary_path=Path(args.clean_out_dir) / "summary.csv",
        require_clean=False,
    )
    print(f"[pipeline:evaluate] sync {stats}", flush=True)

    eval100.run_transcribe(
        eval_root,
        workers=args.asr_workers,
        asr_threads=args.asr_threads,
        asr_engine=args.asr_engine,
        asr_device=args.asr_device,
        asr_batch_size=args.asr_batch_size,
    )
    eval100.prepare_speech_directed_second(eval_root, incremental=True)
    eval100.run_second_transcribe(
        eval_root,
        workers=args.asr_workers,
        asr_threads=args.asr_threads,
        asr_engine=args.asr_engine,
        asr_device=args.asr_device,
        asr_batch_size=args.asr_batch_size,
    )
    if deepseek_enabled(args):
        eval100.judge(eval_root, workers=args.judge_workers, include_timing=final, force_timing=final)
    elif final and not args.no_judge:
        print("[pipeline:evaluate] DEEPSEEK_API_KEY is not set; skipped judge", flush=True)


async def evaluator_loop(args: argparse.Namespace, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.to_thread(run_eval_pass, args, False)
        except Exception as exc:
            print(f"[pipeline:evaluate] pass failed: {exc!r}", flush=True)
        await asyncio.sleep(args.poll_interval)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--data-root", default="data/HumDial-FDBench/extracted")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-balanced", action="store_true")
    parser.add_argument("--noisy-out-dir", default="logs/humdial_100")
    parser.add_argument("--clean-out-dir", default="logs/humdial_100_clean_refs")
    parser.add_argument("--eval-root", default="logs/humdial_100_eval")
    parser.add_argument("--exp", default="humdial-100-pipeline")
    parser.add_argument("--lang", default="batch")
    parser.add_argument("--trailing-silence", type=float, default=2.0)
    parser.add_argument("--post-send-wait", type=float, default=8.0)
    parser.add_argument("--sample-timeout", type=float, default=240.0)
    parser.add_argument("--clean-sample-timeout", type=float, default=300.0)
    parser.add_argument("--gen-workers", type=int, default=int(os.getenv("FDBC_GEN_WORKERS", "2")))
    parser.add_argument("--clean-workers", type=int, default=int(os.getenv("FDBC_CLEAN_WORKERS", "2")))
    parser.add_argument("--asr-workers", type=int, default=int(os.getenv("ASR_WORKERS", "8")))
    parser.add_argument("--asr-threads", type=int, default=int(os.getenv("ASR_THREADS", "4")))
    parser.add_argument("--asr-engine", choices=["fast", "folder"], default=os.getenv("ASR_ENGINE", "fast"))
    parser.add_argument("--asr-device", choices=["cpu", "cuda"], default=os.getenv("ASR_DEVICE", "cpu"))
    parser.add_argument("--asr-batch-size", type=int, default=int(os.getenv("ASR_BATCH_SIZE", "8")))
    parser.add_argument("--judge-workers", type=int, default=int(os.getenv("DEEPSEEK_WORKERS", "16")))
    parser.add_argument("--poll-interval", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--serial-generation",
        action="store_true",
        help="Run noisy samples first, then clean samples, with one websocket session at a time.",
    )
    parser.add_argument(
        "--skip-clean-category",
        action="append",
        default=sorted(DEFAULT_SKIP_CATEGORIES),
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ws_url = f"ws://127.0.0.1:{cfg['server']['port']}/realtime"

    data_root = Path(args.data_root)
    noisy_out = Path(args.noisy_out_dir)
    clean_out = Path(args.clean_out_dir)
    noisy_out.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)
    Path(args.eval_root).mkdir(parents=True, exist_ok=True)

    samples = choose_samples(
        data_root=data_root,
        count=args.count,
        seed=args.seed,
        balanced=not args.no_balanced,
    )
    clean_samples = clean_counterparts(samples, data_root, set(args.skip_clean_category))
    write_manifest(samples, noisy_out / "manifest.jsonl")
    write_manifest(clean_samples, clean_out / "manifest.jsonl")

    if args.serial_generation:
        stop_event = asyncio.Event()
        evaluator = asyncio.create_task(evaluator_loop(args, stop_event))
        try:
            await run_samples(
                samples=samples,
                ws_url=ws_url,
                out_dir=noisy_out,
                summary_path=noisy_out / "summary.csv",
                exp=args.exp,
                lang_mode=args.lang,
                trailing_silence=args.trailing_silence,
                post_send_wait=args.post_send_wait,
                sample_timeout=args.sample_timeout,
                resume=args.resume,
                workers=1,
            )
            await run_samples(
                samples=clean_samples,
                ws_url=ws_url,
                out_dir=clean_out,
                summary_path=clean_out / "summary.csv",
                exp=f"{args.exp}-clean",
                lang_mode=args.lang,
                trailing_silence=args.trailing_silence,
                post_send_wait=args.post_send_wait,
                sample_timeout=args.clean_sample_timeout,
                resume=args.resume,
                workers=1,
            )
        finally:
            stop_event.set()
            await evaluator
    else:
        stop_event = asyncio.Event()
        evaluator = asyncio.create_task(evaluator_loop(args, stop_event))
        noisy_task = asyncio.create_task(
            run_samples(
                samples=samples,
                ws_url=ws_url,
                out_dir=noisy_out,
                summary_path=noisy_out / "summary.csv",
                exp=args.exp,
                lang_mode=args.lang,
                trailing_silence=args.trailing_silence,
                post_send_wait=args.post_send_wait,
                sample_timeout=args.sample_timeout,
                resume=args.resume,
                workers=args.gen_workers,
            )
        )
        clean_task = asyncio.create_task(
            run_samples(
                samples=clean_samples,
                ws_url=ws_url,
                out_dir=clean_out,
                summary_path=clean_out / "summary.csv",
                exp=f"{args.exp}-clean",
                lang_mode=args.lang,
                trailing_silence=args.trailing_silence,
                post_send_wait=args.post_send_wait,
                sample_timeout=args.clean_sample_timeout,
                resume=args.resume,
                workers=args.clean_workers,
            )
        )

        await asyncio.gather(noisy_task, clean_task)
        stop_event.set()
        await evaluator

    print("[pipeline] final evaluation drain", flush=True)
    run_eval_pass(args, final=True)

    noisy_ok, noisy_tts, noisy_errors = count_rows(noisy_out / "summary.csv")
    clean_ok, clean_tts, clean_errors = count_rows(clean_out / "summary.csv")
    result = {
        "noisy": {"ok": noisy_ok, "with_tts": noisy_tts, "errors": noisy_errors},
        "clean": {"ok": clean_ok, "with_tts": clean_tts, "errors": clean_errors},
        "eval_root": str(Path(args.eval_root)),
        "summary": str(Path(args.eval_root) / "summary.json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if noisy_errors == 0 and clean_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
