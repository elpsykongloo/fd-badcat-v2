#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1 D2.2 mock-equivalence matrix in ONE lightweight process (2GB container).

Uses precomputed VAD event scripts (scripts/extract_vad_events.py) + stubbed
torch/silero, so both engines see IDENTICAL perception without loading torch.
Emits docs/w1_equivalence_data.json + per-run trace files.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_session import (  # noqa: E402
    MockModels, apply_vad_script, install_light_stubs, load_vad_script,
    run_actor, run_legacy, save_trace,
)

install_light_stubs()

import yaml  # noqa: E402
from trace_diff import diff_traces  # noqa: E402

WAVS = [
    ROOT / "exp/exp-1/test/test-000001.wav",
    ROOT / "data/HumDial-FDBench/extracted/test/cn_test_nondev/ask/0001_0004.wav",
]
CONFIGS = [  # (judge, interrupt, shift)
    ("switch", "continue", "no"),
    ("continue", "continue", "no"),
    ("switch", "switch", "no"),
    ("switch", "continue", "yes"),
]


async def one_pair(wav, judge, interrupt, shift, out_root):
    with open(ROOT / "src/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tag = f"{wav.stem}__j{judge}_i{interrupt}_s{shift}"
    vad_script = load_vad_script(wav)
    assert vad_script is not None, f"run scripts/extract_vad_events.py {wav} first"

    mocks_l = MockModels(cfg["prompts"], judge=judge, interrupt=interrupt, shift=shift)
    trace_l, _ = await run_legacy(wav, mocks_l, out_root / f"l_{tag}", vad_script=vad_script)
    save_trace(trace_l, ROOT / f"traces/mock_eq/legacy_{tag}.jsonl")

    mocks_a = MockModels(cfg["prompts"], judge=judge, interrupt=interrupt, shift=shift)
    trace_a, eng = await run_actor(wav, mocks_a, out_root / f"a_{tag}", mode="realtime",
                                   vad_script=vad_script)
    save_trace(trace_a, ROOT / f"traces/mock_eq/actor_{tag}.jsonl")

    res = diff_traces(trace_l, trace_a)
    return {
        "tag": tag,
        "events_legacy": res["a_len"], "events_actor": res["b_len"],
        "l1": res["l1"], "sequence_equal": res["sequence_equal"],
        "first_divergence": res["first_divergence"],
        "soft_time_mismatches": [(i, round(dt, 3)) for i, dt in res["soft_time_mismatches"]],
        "info_counts": res["info_counts"],
        "actor_freeze": eng.freeze_stats(),
    }


async def main():
    out_root = ROOT / "exp/replay_tmp"
    results = []
    t0 = time.time()
    for wav in WAVS:
        if not wav.exists():
            print(f"skip missing {wav}")
            continue
        for judge, interrupt, shift in CONFIGS:
            r = await one_pair(wav, judge, interrupt, shift, out_root)
            verdict = "L1" if r["l1"] else ("SEQ_EQ+time" if r["sequence_equal"] else f"L2@{r['first_divergence']}")
            print(f"[{time.time()-t0:6.1f}s] {r['tag']}: {verdict} "
                  f"({r['events_legacy']}/{r['events_actor']} events, "
                  f"{len(r['soft_time_mismatches'])} soft)")
            results.append(r)
    out = ROOT / "docs/w1_equivalence_data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    l1 = sum(1 for r in results if r["l1"])
    seq = sum(1 for r in results if r["sequence_equal"])
    print(f"\nL1: {l1}/{len(results)}  sequence-equal: {seq}/{len(results)}  -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
