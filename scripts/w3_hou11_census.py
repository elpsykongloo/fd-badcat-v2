#!/usr/bin/env python3
"""
w3_hou11_census.py — two censuses over benchmark_data_v2.json (06 零 GPU 批).

1. 裁断③ census: scenarios whose GOLD tool args reference values that appear ONLY
   in PRIOR dialogue turns (not in the final user turn) — the housing_11 class.
   The released audio covers the final turn; whether the official contract feeds
   prior turns decides if our replay is under-informed (see contract audit).

2. 裁断 A-iv census: scenarios with >=2 expected calls with NO $RESULT dependency
   between them (chain-parallelizable) — the DAG-parallel claim base.

Heuristic for (1): for each expected arg VALUE (stringified), check case-insensitive
containment in the final user turn text vs any prior user turn text. Values not
found anywhere are counted separately (world knowledge / normalization).

Usage: python scripts/w3_hou11_census.py [--out exp/w3/hou11_census.json]
"""
import argparse
import json
import re
from pathlib import Path

BENCH = Path("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def val_in(val, text):
    v, t = norm(val), norm(text)
    if not v:
        return True
    if v in t:
        return True
    # numeric convenience: 1600 vs "1,600"/"1600 a month" handled by norm; also
    # bare-number containment on token boundary
    return re.search(rf"(?:^| ){re.escape(v)}(?: |$)", t) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/fd-badcat/exp/w3/hou11_census.json")
    args = ap.parse_args()

    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench

    prior_only, nowhere, multiturn = [], [], 0
    chain, chain_dep = [], []
    for x in items:
        dlg = x.get("dialogue") or []
        users = [t.get("user", "") for t in dlg if t.get("user")]
        if len(users) > 1:
            multiturn += 1
        final, prior = (users[-1] if users else ""), " ".join(users[:-1])
        po_hits, nw_hits = [], []
        for call in x.get("expected_tool_calls", []):
            for k, v in (call.get("args") or {}).items():
                if isinstance(v, str) and v.startswith("$RESULT"):
                    continue
                if v is None or isinstance(v, bool):
                    continue
                if val_in(v, final):
                    continue
                if prior and val_in(v, prior):
                    po_hits.append({"fn": call.get("function"), "arg": k, "value": v})
                else:
                    nw_hits.append({"fn": call.get("function"), "arg": k, "value": v})
        if po_hits:
            prior_only.append({"id": x["id"], "args": po_hits})
        if nw_hits:
            nowhere.append({"id": x["id"], "args": nw_hits})

        calls = x.get("expected_tool_calls", [])
        if len(calls) >= 2:
            deps = any(isinstance(v, str) and v.startswith("$RESULT")
                       for c in calls for v in (c.get("args") or {}).values())
            (chain_dep if deps else chain).append(x["id"])

    summary = {
        "n_scenarios": len(items),
        "multiturn_dialogues": multiturn,
        "gold_arg_only_in_prior_turns": {"n": len(prior_only),
                                         "ids": [r["id"] for r in prior_only]},
        "gold_arg_found_nowhere": {"n": len(nowhere),
                                   "ids": [r["id"] for r in nowhere]},
        "chain_ge2_calls_independent": {"n": len(chain), "ids": chain},
        "chain_ge2_calls_with_deps": {"n": len(chain_dep), "ids": chain_dep},
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "prior_only": prior_only,
                               "nowhere": nowhere}, indent=1, ensure_ascii=False))
    print("report ->", out)


if __name__ == "__main__":
    main()
