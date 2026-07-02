#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1 D3.4 perception-freeze before/after measurement (paper Fig.1 candidate).

Metric: gap between consecutive per-frame VAD probe hits (nominal 16ms).
  - legacy engine: decisions are awaited inline in the receive loop, so during a
    judge/shift/response the probe goes silent — gaps spike to decision latency.
  - actor engine: decisions are forked; the probe should tick at ~16ms always.

Same probe point, same wav, same scripted VAD events, same mock latencies
(sleep-based, so the 1-core container's CPU doesn't confound the contrast).

Usage:
  python scripts/measure_freeze.py --wav exp/exp-1/test/test-000001.wav \
      --llm-sleep 0.8 --runs 2 --out docs/w1_freeze_data.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_session import (  # noqa: E402
    MockModels, install_light_stubs, load_vad_script, run_actor, run_legacy,
)

install_light_stubs()

import yaml  # noqa: E402


def gap_stats(probe):
    gaps = [(b - a) * 1e3 for a, b in zip(probe, probe[1:])]
    if not gaps:
        return {}
    g = sorted(gaps)
    pick = lambda p: g[min(len(g) - 1, int(p * len(g)))]
    return {
        "n": len(g),
        "p50_ms": round(pick(0.50), 3), "p95_ms": round(pick(0.95), 3),
        "p99_ms": round(pick(0.99), 3), "max_ms": round(g[-1], 3),
        "gaps_over_100ms": sum(1 for x in g if x > 100),
        "gaps_over_500ms": sum(1 for x in g if x > 500),
        "total_stall_ms": round(sum(x - 16.0 for x in g if x > 100), 1),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="exp/exp-1/test/test-000001.wav")
    ap.add_argument("--llm-sleep", type=float, default=0.8)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--judge", default="switch")
    ap.add_argument("--out", default="docs/w1_freeze_data.json")
    args = ap.parse_args()

    with open(ROOT / "src/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vad_script = load_vad_script(args.wav)
    assert vad_script, f"run scripts/extract_vad_events.py {args.wav} first"

    report = {"wav": args.wav, "llm_sleep_s": args.llm_sleep, "runs": args.runs,
              "legacy": [], "actor": [], "actor_frame_lag": []}

    for r in range(args.runs):
        probe_l = []
        mocks = MockModels(cfg["prompts"], judge=args.judge, llm_sleep=args.llm_sleep)
        await run_legacy(args.wav, mocks, ROOT / "exp/replay_tmp/freeze_l",
                         vad_script=vad_script, vad_probe=probe_l)
        s = gap_stats(probe_l)
        report["legacy"].append(s)
        print(f"[legacy run{r}] {s}")

        probe_a = []
        mocks = MockModels(cfg["prompts"], judge=args.judge, llm_sleep=args.llm_sleep)
        _, eng = await run_actor(args.wav, mocks, ROOT / "exp/replay_tmp/freeze_a",
                                 mode="realtime", vad_script=vad_script, vad_probe=probe_a)
        s = gap_stats(probe_a)
        report["actor"].append(s)
        report["actor_frame_lag"].append(eng.freeze_stats())
        print(f"[actor  run{r}] {s}")
        print(f"[actor  lag  ] {eng.freeze_stats()}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
