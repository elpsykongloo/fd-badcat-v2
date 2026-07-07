#!/usr/bin/env python3
"""
w3_oracle_frontier.py — the premium-economics three numbers (06 教义二 / E4).

Premium (保费) per scenario i:  P_i = max(0, completion_i^TACT@δ* − completion_i^blocking)
computed on BOTH clocks:
  nominal : max(t_commit) − t_user_end   (deterministic, wall-free — primary)
  wall    : archived task_completion_s   (tool wall included — reference)

Arms (archived W2 serial-live full-100, δ*=1.5):
  TACT     = result_w2r_tact_full.json
  blocking = result_w2r_sblock_full.json

Three numbers, for full-100 and the rollback subset:
  fixed premium   = Σ P_i over all scenarios              (every clip pays δ*)
  oracle-A        = Σ P_i over rollback-flagged scenarios (window only where a
                    revision exists — scenario-level oracle, 教义二's "21×δ*")
  recovery        = 1 − oracle/fixed                      (W4–W5 learning target)

Sharper mechanism-level oracle from the W3 ledger (true exposure set):
  oracle-B        = Σ P_i over {eco19, hou25, fin12b}     (clips where the window
                    is causally load-bearing at δ*; travel_10 needs δ>3.91 and is
                    reported separately with the cancel-rule cost comparison)

travel_10 case arithmetic (裁断 A 顺带 / 06 零 GPU 批): the "δ>3.91 rescue" route
vs the "announce-cancel prompt rule" route (zero-premium rescue).

Usage: python scripts/w3_oracle_frontier.py [--out exp/w3/oracle_frontier.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
BENCH = Path("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json")
_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")

TACT, BLOCK = "w2r_tact_full", "w2r_sblock_full"
DELTA_STAR = 1.5
# mechanism-level true-exposure clips (W3 ledger §3; folder names)
ORACLE_B = {"ecommerce_19_66f59c766e7e22e1f90d08f6",
            "housing_25_66f59c766e7e22e1f90d08f6",
            "finance_12_69a9cf80f4d7668d5c815038"}
TRAVEL10 = "travel_10_5f4a4da1575d605c43bef871"
TRAVEL10_DELTA = 3.91   # silence-budget threshold from the ledger


def completion(res):
    """(nominal, wall) completion anchors; None if no commits."""
    tr = res.get("trace") or {}
    commits = tr.get("commits") or []
    segs = tr.get("segs") or []
    if not commits or not segs:
        return None, None
    t_user_end = segs[-1][1]
    nominal = max(c["t_commit"] for c in commits) - t_user_end
    wall = (res.get("latency") or {}).get("task_completion_s")
    return max(0.0, nominal), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/autodl-tmp/fd-badcat/exp/w3/oracle_frontier.json")
    args = ap.parse_args()

    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench
    rb_ids = {x["id"] for x in items if x.get("state_rollback_test")}

    rows, skipped = [], []
    for folder in sorted(DATA.iterdir()):
        m = _FOLDER_RE.match(folder.name) if folder.is_dir() else None
        if not m:
            continue
        tp, bp = folder / f"result_{TACT}.json", folder / f"result_{BLOCK}.json"
        if not (tp.exists() and bp.exists()):
            continue
        rt, rb = json.loads(tp.read_text()), json.loads(bp.read_text())
        tn, tw = completion(rt)
        bn, bw = completion(rb)
        if tn is None or bn is None:
            skipped.append({"folder": folder.name,
                            "tact_commits": tn is not None,
                            "block_commits": bn is not None})
            continue
        rows.append({
            "folder": folder.name, "id": m.group(1),
            "rollback": m.group(1) in rb_ids,
            "premium_nominal": round(max(0.0, tn - bn), 3),
            "premium_wall": (round(max(0.0, tw - bw), 3)
                             if tw is not None and bw is not None else None),
        })

    def agg(sel, key="premium_nominal"):
        vals = [r[key] for r in sel if r[key] is not None]
        return round(sum(vals), 2), len(vals)

    def frontier(rows, label):
        fixed, n = agg(rows)
        oa, na = agg([r for r in rows if r["rollback"]])
        ob, nb = agg([r for r in rows if r["folder"] in ORACLE_B])
        out = {
            "set": label, "n": n,
            "fixed_premium_s": fixed,
            "oracle_A_premium_s": oa, "oracle_A_n": na,
            "oracle_B_premium_s": ob, "oracle_B_n": nb,
            "recovery_A": round(1 - oa / fixed, 3) if fixed else None,
            "recovery_B": round(1 - ob / fixed, 3) if fixed else None,
            "mean_premium_s": round(fixed / n, 3) if n else None,
        }
        fw, _ = agg(rows, "premium_wall")
        oaw, _ = agg([r for r in rows if r["rollback"]], "premium_wall")
        out["fixed_premium_wall_s"] = fw
        out["recovery_A_wall"] = round(1 - oaw / fw, 3) if fw else None
        return out

    full = frontier(rows, "full-100")
    rb17 = frontier([r for r in rows if r["rollback"]], "rollback-subset")

    # travel_10 case arithmetic from its archived TACT trace
    t10 = None
    tp = DATA / TRAVEL10 / f"result_{TACT}.json"
    if tp.exists():
        r = json.loads(tp.read_text())
        tr = r["trace"]
        t_user_end = tr["segs"][-1][1]
        t_dec_final = tr["eous"][-1][1] + tr["decisions"][-1]["infer_s"]
        base = round(t_dec_final - t_user_end, 3)   # decision-ready anchor
        t10 = {
            "rescue_route": {"delta_needed": TRAVEL10_DELTA,
                             "completion_s": round(base + TRAVEL10_DELTA, 3),
                             "per_clip_premium_s": TRAVEL10_DELTA,
                             "fleet_ripple": f"fixed-arm delta 1.5 -> {TRAVEL10_DELTA} "
                                             f"costs +{round((TRAVEL10_DELTA-DELTA_STAR)*100,1)}s "
                                             f"per 100 scenarios"},
            "cancel_rule_route": {"mechanism": "announce-cancel at EoU1 (revision "
                                               "announced, content pending) -> relaunch "
                                               "at final EoU under delta*",
                                  "completion_s": round(base + DELTA_STAR, 3),
                                  "per_clip_premium_s": DELTA_STAR},
            "ratio_rescue_over_cancel": round(TRAVEL10_DELTA / DELTA_STAR, 2),
        }

    report = {"delta_star": DELTA_STAR, "arms": {"tact": TACT, "blocking": BLOCK},
              "clock": "nominal = max(t_commit) - t_user_end (wall-free, primary)",
              "full": full, "rollback": rb17, "travel_10": t10,
              "skipped": skipped,
              "top10_premiums": sorted(rows, key=lambda r: -r["premium_nominal"])[:10]}
    print(json.dumps({k: report[k] for k in
                      ("full", "rollback", "travel_10")}, indent=1, ensure_ascii=False))
    if skipped:
        print(f"skipped (missing commits in one arm): "
              f"{[s['folder'] for s in skipped]}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("report ->", out)


if __name__ == "__main__":
    main()
