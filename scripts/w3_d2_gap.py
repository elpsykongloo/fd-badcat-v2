#!/usr/bin/env python3
"""
w3_d2_gap.py — D2 live-vs-offline gap accounting (H6 residual, full-100).

Compares two providers on the official exact scorer, clip by clip:
  --live : the unified-engine live-perception run (--engine full), e.g. w3_tactfull_live
  --ref  : the offline reference (--engine core / W2 archive), e.g. w2r_tact_full

For every flip it reports the first-order perception attribution: n_eou change
(streaming VADIterator vs offline oracle EoU — the epsilon-band existence effect)
vs same-structure decision drift. Residuals beyond that go to the H1->H2->H5 chase.

Usage:
  python scripts/w3_d2_gap.py --live w3_tactfull_live --ref w2r_tact_full \
      [--only-rollback] [--out exp/w3/d2_gap.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/FDBench_v3/v3")
from evaluate_pass_rate import evaluate_scenario_pass  # noqa: E402

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
BENCH = Path("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json")
_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def load(folder, provider):
    p = folder / f"result_{provider}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--only-rollback", action="store_true")
    ap.add_argument("--out", default="/root/autodl-tmp/fd-badcat/exp/w3/d2_gap.json")
    args = ap.parse_args()

    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench
    by_id = {x["id"]: x for x in items}
    rb = {x["id"] for x in items if x.get("state_rollback_test")}

    rows, missing = [], []
    for folder in sorted(DATA.iterdir()):
        m = _FOLDER_RE.match(folder.name) if folder.is_dir() else None
        if not m or m.group(1) not in by_id:
            continue
        sid = m.group(1)
        if args.only_rollback and sid not in rb:
            continue
        live, ref = load(folder, args.live), load(folder, args.ref)
        if live is None or ref is None:
            missing.append((folder.name, live is None, ref is None))
            continue

        def score(res):
            return evaluate_scenario_pass(by_id[sid], res.get("actual_tool_calls", []),
                                          res.get("transcript", ""), res)["passed"]
        pl, pr = score(live), score(ref)
        rows.append({
            "folder": folder.name, "id": sid, "rollback": sid in rb,
            "live": pl, "ref": pr,
            "n_eou_live": (live.get("latency") or {}).get("n_eou"),
            "n_eou_ref": (ref.get("latency") or {}).get("n_eou"),
            "calls_live": len(live.get("actual_tool_calls", [])),
            "calls_ref": len(ref.get("actual_tool_calls", [])),
        })

    n = len(rows)
    live_pass = sum(r["live"] for r in rows)
    ref_pass = sum(r["ref"] for r in rows)
    flips = [r for r in rows if r["live"] != r["ref"]]
    eou_flips = [r for r in flips if r["n_eou_live"] != r["n_eou_ref"]]
    same_eou_flips = [r for r in flips if r["n_eou_live"] == r["n_eou_ref"]]

    print(f"n={n}  live({args.live})={live_pass}  ref({args.ref})={ref_pass}  "
          f"gap={live_pass - ref_pass:+d}")
    print(f"flips: {len(flips)} total | EoU-structure changed: {len(eou_flips)} "
          f"| same-structure (decision drift / other): {len(same_eou_flips)}")
    for r in flips:
        kind = "EOU" if r["n_eou_live"] != r["n_eou_ref"] else "DEC"
        print(f"  [{kind}] {r['folder']}: live={'P' if r['live'] else 'F'} "
              f"ref={'P' if r['ref'] else 'F'} "
              f"n_eou {r['n_eou_ref']}->{r['n_eou_live']} "
              f"calls {r['calls_ref']}->{r['calls_live']}"
              f"{'  [rollback]' if r['rollback'] else ''}")
    if missing:
        print(f"missing results: {len(missing)} (run both providers first)")
        for name, ml, mr in missing[:10]:
            print(f"  {name}: live_missing={ml} ref_missing={mr}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "live": args.live, "ref": args.ref, "n": n,
        "live_pass": live_pass, "ref_pass": ref_pass,
        "flips": flips, "missing": [m[0] for m in missing], "rows": rows,
    }, indent=1, ensure_ascii=False))
    print("report ->", out)


if __name__ == "__main__":
    main()
