#!/usr/bin/env python3
"""
w3_server_drift_probe.py — bound server-side decision drift on IDENTICAL prompts.

Question (W3 D1 memo §3): the full-probe decision flips (finance_23, housing_17b)
ran on a temporary serving stack (exp/w3/qwen3_omni_audio_84g.yaml, 85GB card) —
are they explained by the epsilon-shifted prefixes alone, or does the stack itself
drift T=0/seed=42 outputs on byte-identical prompts?

Method: replay the CORE-mode message sequence for the probe clips (state advanced
by the CACHED W2 decisions, so every prompt is a W2-provenance byte-exact replica),
and for every cache HIT also fire the same messages directly at the live server,
then diff raw outputs. Cache file is never saved (no mutation). Temporary
result_<provider> files are removed afterwards.

Zero drift => the stack is innocent on these prompts; the full-probe flips are
prefix effects. Any drift => server numerics contribute; record which keys.

Usage: python scripts/w3_server_drift_probe.py [--out exp/w3/server_drift_probe.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/src")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/scripts")

from w2r_stream_replay import DATA, DecisionCache, run_example, _llm_call  # noqa: E402
from tact.decider import parse_decision                                    # noqa: E402

CLIPS = [
    "finance_23_66f59c766e7e22e1f90d08f6",   # full-probe decision flip (narrate-no-op)
    "housing_17_69a9cf80f4d7668d5c815038",   # full-probe decision component (wait@EoU1)
    "ecommerce_19_66f59c766e7e22e1f90d08f6",  # structural clip, multi-decision
    "housing_25_66f59c766e7e22e1f90d08f6",   # structural clip, multi-launch
    "finance_12_69a9cf80f4d7668d5c815038",   # barrier flip clip
]
PROVIDER = "w3drift_tmp"


class DriftCache(DecisionCache):
    """Cache wrapper: on every HIT, also call the live server with the identical
    messages and record the diff. Misses pass through (and are NOT persisted)."""

    def __init__(self, path, records):
        super().__init__(path)
        self.records = records

    def call(self, msgs):
        k = self.key(msgs)
        cached = self.data.get(k)
        raw, infer = super().call(msgs)
        if cached is not None:
            fresh = _llm_call(msgs)
            identical = (fresh == cached["raw"])
            rec = {"key": k[:16], "identical": identical,
                   "ops_equal": None, "parsed_equal": None}
            if not identical:
                pc, pf = parse_decision(cached["raw"]), parse_decision(fresh)
                pc.pop("_parse_error", None), pf.pop("_parse_error", None)
                rec["parsed_equal"] = (pc == pf)
                # ops-level equivalence: the only field the official track scores
                rec["ops_equal"] = (pc.get("ops") == pf.get("ops"))
                rec["cached_raw"] = cached["raw"][:500]
                rec["fresh_raw"] = fresh[:500]
            self.records.append(rec)
        return raw, infer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default="/root/autodl-tmp/fd-badcat/exp/w2_rerun/decision_cache.json")
    ap.add_argument("--out",
                    default="/root/autodl-tmp/fd-badcat/exp/w3/server_drift_probe.json")
    args = ap.parse_args()

    records = []
    cache = DriftCache(args.cache, records)
    per_clip = {}
    for name in CLIPS:
        folder = DATA / name
        if not folder.is_dir():
            print(f"MISSING {name}")
            continue
        n0 = len(records)
        try:
            run_example(folder, PROVIDER, 1.5, cache, mode="tact", force=True,
                        infer_nominal=1.0, barrier=True, engine="core")
        except Exception as e:
            print(f"{name}: ERROR {type(e).__name__} {e}")
        finally:
            tmp = folder / f"result_{PROVIDER}.json"
            if tmp.exists():
                tmp.unlink()
        per_clip[name] = records[n0:]
        n_id = sum(1 for r in records[n0:] if r["identical"])
        print(f"{name}: {n_id}/{len(records[n0:])} identical "
              f"(cache misses this clip: see below)")

    n = len(records)
    n_id = sum(1 for r in records if r["identical"])
    n_ops = sum(1 for r in records
                if not r["identical"] and r.get("ops_equal"))
    n_bad = n - n_id - n_ops
    print(f"\nTOTAL compared: {n} | byte-identical: {n_id} | "
          f"surface-only drift (ops equal): {n_ops} | OPS DRIFT: {n_bad}")
    print(f"cache misses during probe (prompts w/o W2 reference): {cache.misses}")
    verdict = ("SERVER INNOCENT (byte-exact on all compared prompts)" if n_id == n
               else "OPS-STABLE (drift confined to say/dialogue surface fields)"
               if n_bad == 0 else "OPS DRIFT PRESENT — see records")
    print("VERDICT:", verdict)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": n, "byte_identical": n_id,
                               "surface_only_drift": n_ops,
                               "ops_drift": n_bad,
                               "cache_misses": cache.misses,
                               "verdict": verdict,
                               "records": records}, indent=1, ensure_ascii=False))
    print("report ->", out)


if __name__ == "__main__":
    main()
