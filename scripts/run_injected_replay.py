#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI runner for injected replay (W1 fast evaluation).

Usage:
    python scripts/run_injected_replay.py \\
        --golden traces/golden_rerun/ask_0001_0004.jsonl \\
        --wav exp/golden/actor_ask_0001_0004/stream_turn0_input.wav \\
        --out exp/replay_output/ask_0001_0004

Supports:
    - Single session replay with determinism verification
    - Batch replay for concurrent evaluation
    - Speed benchmarking (target: 60× real-time)
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import yaml
from injected_replay import (
    InjectedReplaySession,
    compare_traces,
    extract_decisions_summary,
    load_golden_trace,
)


async def replay_single(golden_path, wav_path, config, output_dir, mode='injected',
                        verify_determinism=False):
    """Run a single injected replay session."""
    print(f"Replaying: {Path(golden_path).name}")
    print(f"  Golden: {golden_path}")
    print(f"  WAV: {wav_path}")
    print(f"  Mode: {mode}")

    session = InjectedReplaySession(
        golden_trace=golden_path,
        wav_path=wav_path,
        config=config,
        output_dir=output_dir
    )

    t0 = time.perf_counter()
    trace, engine = await session.replay(mode=mode)
    elapsed = time.perf_counter() - t0

    # Calculate audio duration
    audio_dur = max(
        (ev.get('data', {}).get('t_audio', 0) for ev in trace),
        default=0.0
    )
    speedup = audio_dur / elapsed if elapsed > 0 else 0

    print(f"  Events: {len(trace)}")
    print(f"  Time: {elapsed:.3f}s (audio={audio_dur:.1f}s, {speedup:.1f}× real-time)")

    # Save trace
    trace_path = Path(output_dir) / "trace.jsonl"
    session.save_trace(trace_path)
    print(f"  Trace: {trace_path}")

    # Extract summary
    summary = extract_decisions_summary(trace)
    print(f"  Decisions: {len(summary['judge'])} judge, "
          f"{len(summary['response'])} response, "
          f"{len(summary['asr'])} asr, "
          f"{summary['tts_count']} tts")

    # Verify determinism if requested
    if verify_determinism:
        print("  Verifying determinism (running again)...")
        session2 = InjectedReplaySession(
            golden_trace=golden_path,
            wav_path=wav_path,
            config=config,
            output_dir=Path(output_dir).parent / f"{Path(output_dir).name}_verify"
        )
        trace2, _ = await session2.replay(mode=mode)
        identical, diffs = compare_traces(trace, trace2)

        if identical:
            print("  ✓ Deterministic: traces identical")
        else:
            print(f"  ✗ NOT deterministic: {len(diffs)} differences")
            for diff in diffs[:5]:
                print(f"    - {diff}")

    return trace, engine


async def replay_batch(golden_dir, wav_dir, config, output_dir, mode='injected',
                       concurrency=8, limit=None):
    """Run batch replay with concurrency."""
    golden_dir = Path(golden_dir)
    wav_dir = Path(wav_dir)
    output_dir = Path(output_dir)

    # Find all golden trace files
    golden_traces = sorted(golden_dir.glob("*.jsonl"))
    if limit:
        golden_traces = golden_traces[:limit]

    print(f"Batch replay: {len(golden_traces)} sessions")
    print(f"  Concurrency: {concurrency}")
    print(f"  Mode: {mode}")

    async def run_one(golden_path):
        # Find corresponding wav
        session_name = golden_path.stem
        # Try different naming patterns
        wav_candidates = [
            wav_dir / f"actor_{session_name}" / "stream_turn0_input.wav",
            wav_dir / session_name / "stream_turn0_input.wav",
        ]
        wav_path = None
        for candidate in wav_candidates:
            if candidate.exists():
                wav_path = candidate
                break

        if not wav_path:
            return None, f"WAV not found for {session_name}"

        out = output_dir / session_name
        session = InjectedReplaySession(golden_path, wav_path, config, out)

        try:
            t0 = time.perf_counter()
            trace, _ = await session.replay(mode=mode)
            elapsed = time.perf_counter() - t0
            session.save_trace(out / "trace.jsonl")
            return (session_name, len(trace), elapsed, None), None
        except Exception as e:
            return None, f"{session_name}: {e}"

    # Run with concurrency limit
    sem = asyncio.Semaphore(concurrency)

    async def run_with_sem(golden_path):
        async with sem:
            return await run_one(golden_path)

    t_start = time.perf_counter()
    results = await asyncio.gather(*[run_with_sem(g) for g in golden_traces])
    t_total = time.perf_counter() - t_start

    # Report
    successes = [r for r, e in results if r]
    errors = [e for r, e in results if e]

    print(f"\n=== BATCH RESULTS ===")
    print(f"  Total: {len(results)}")
    print(f"  Success: {len(successes)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {t_total:.1f}s")

    if errors:
        print("\nErrors:")
        for err in errors[:10]:
            print(f"  - {err}")

    if successes:
        total_events = sum(r[1] for r in successes)
        total_elapsed = sum(r[2] for r in successes)
        avg_speedup = len(successes) * 10.0 / total_elapsed  # assume ~10s avg audio
        print(f"\nStats:")
        print(f"  Events: {total_events}")
        print(f"  Avg speedup: ~{avg_speedup:.1f}× real-time")
        print(f"  Throughput: {len(successes) / t_total:.1f} sessions/sec")


def main():
    ap = argparse.ArgumentParser(description="Injected replay runner")
    ap.add_argument("--golden", help="Golden trace .jsonl file or directory")
    ap.add_argument("--wav", help="Input WAV file or directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--mode", choices=["injected", "oracle"], default="injected")
    ap.add_argument("--config", default="src/config.yaml", help="Engine config")
    ap.add_argument("--batch", action="store_true", help="Batch mode (golden/wav are dirs)")
    ap.add_argument("--concurrency", type=int, default=8, help="Batch concurrency")
    ap.add_argument("--limit", type=int, help="Batch limit (for testing)")
    ap.add_argument("--verify", action="store_true", help="Verify determinism (single mode)")
    args = ap.parse_args()

    # Load config
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.batch:
        asyncio.run(replay_batch(
            args.golden, args.wav, config, args.out,
            mode=args.mode, concurrency=args.concurrency, limit=args.limit
        ))
    else:
        if not args.golden or not args.wav:
            print("Error: --golden and --wav required in single mode")
            sys.exit(1)
        asyncio.run(replay_single(
            args.golden, args.wav, config, args.out,
            mode=args.mode, verify_determinism=args.verify
        ))


if __name__ == "__main__":
    main()
