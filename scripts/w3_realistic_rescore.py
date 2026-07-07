#!/usr/bin/env python3
"""
w3_realistic_rescore.py — realistic-profile re-scoring of EXISTING result
archives (W3 D4/D6; zero GPU — 教义: the realistic profile is pure accounting,
so it is a deterministic replay-arithmetic function of a finished trace).

Never mutates the archived result files; writes a per-provider summary to
exp/w3/realistic_{provider}.json.

Usage:
  # single arm
  python scripts/w3_realistic_rescore.py --provider w2r_tact_full
  # paired 对表 against the preregistered prediction table (docs/latency_calibration.md §4)
  python scripts/w3_realistic_rescore.py --compare w2r_blocking w2r_tact_full
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, "/root/autodl-tmp")

import latency_realistic as lr  # noqa: E402
from tact_dag import OpDag  # noqa: E402

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")

WRITE_BOOKING = {fn for fn, c in lr.TOOL_CLASS.items() if c == "write_booking"}


class _OpShim:
    def __init__(self, rec):
        self.op_id = rec["op_id"]
        self.fn = rec["fn"]
        self.args = rec.get("args", {}) or {}
        self.patch_history = []


def rebuild_edges(tx_log):
    """Offline DAG reconstruction from the archived tx_log launch order."""
    dag = OpDag(ledger=None)
    for rec in tx_log:
        if rec.get("op") == "launch":
            dag.register_launch(_OpShim(rec))
    return dag.edges


def rescore(provider):
    rows = []
    for folder in sorted(DATA.iterdir()):
        m = _FOLDER_RE.match(folder.name) if folder.is_dir() else None
        if not m:
            continue
        rp = folder / f"result_{provider}.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        segs = res.get("trace", {}).get("segs") or []
        if not segs:
            # legacy archive without a trace (e.g. w2r_blocking, the W1-pipeline
            # accuracy baseline): not rescorable — the latency arms are the
            # driver-generated ones (sblock = streaming blocking).
            continue
        t_user_end = segs[-1][1]
        edges = rebuild_edges(res.get("tx_log", []))
        got = lr.attach(res, t_user_end, edges=edges)   # in-memory only
        fns = [p["fn"] for p in got["per_op"]]
        rows.append({
            "id": m.group(1), "folder": folder.name, "mode": res.get("mode"),
            "n_calls": len(fns), "has_booking_write": bool(set(fns) & WRITE_BOOKING),
            "chained": len(fns) >= 2,
            "edges": {str(k): v for k, v in edges.items()},
            "official": {"first": res["latency"].get("first_response_s"),
                         "completion": res["latency"].get("task_completion_s"),
                         "completion_nominal": res["latency"].get("completion_nominal_s")},
            "realistic": {"first": got["first_response_s"],
                          "completion": got["task_completion_s"]},
            "sum_lat": round(sum(p["lat_s"] for p in got["per_op"]), 3),
            "max_lat": round(max((p["lat_s"] for p in got["per_op"]), default=0.0), 3),
        })
    return rows


def p50(xs):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[len(xs) // 2], 3) if xs else None


def summarize(provider, rows):
    firsts = [r["realistic"]["first"] for r in rows]
    comps = [r["realistic"]["completion"] for r in rows]
    out = {"provider": provider, "n": len(rows), "profile": lr.PROFILE_VERSION,
           "first_p50": p50(firsts), "completion_p50": p50(comps),
           "first_p50_booking": p50([r["realistic"]["first"] for r in rows
                                     if r["has_booking_write"]]),
           "official_first_p50": p50([r["official"]["first"] for r in rows]),
           "rows": rows}
    print(f"{provider}: n={out['n']} realistic first p50={out['first_p50']} "
          f"(booking-write subset {out['first_p50_booking']}) "
          f"completion p50={out['completion_p50']} "
          f"[official first p50={out['official_first_p50']}]")
    dst = Path(f"/root/autodl-tmp/fd-badcat/exp/w3/realistic_{provider}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"  -> {dst}")
    return out


def compare(prov_b, prov_t):
    rb = {r["folder"]: r for r in rescore(prov_b)}
    rt = {r["folder"]: r for r in rescore(prov_t)}
    ids = sorted(set(rb) & set(rt))
    print(f"\n=== 对表 (preregistered docs/latency_calibration.md §4) | "
          f"{prov_b} vs {prov_t} | {len(ids)} paired ===")

    # P-1 first-response ratio
    ratios = [(rt[i]["realistic"]["first"], rb[i]["realistic"]["first"]) for i in ids
              if rt[i]["realistic"]["first"] is not None
              and rb[i]["realistic"]["first"]]
    r_all = p50([a / b for a, b in ratios])
    book = [(rt[i]["realistic"]["first"], rb[i]["realistic"]["first"]) for i in ids
            if rb[i]["has_booking_write"]
            and rt[i]["realistic"]["first"] is not None and rb[i]["realistic"]["first"]]
    r_book = p50([a / b for a, b in book])
    print(f"P-1 first-response ratio p50: ALL={r_all} (prereg ≈0.45-0.50, gate ≤0.50) | "
          f"booking-write={r_book} (prereg ≈0.26, gate ≤0.30)")

    # P-2 completion premium
    prem = [rt[i]["realistic"]["completion"] - rb[i]["realistic"]["completion"]
            for i in ids if rt[i]["realistic"]["completion"] is not None
            and rb[i]["realistic"]["completion"] is not None]
    print(f"P-2 completion premium p50: {p50(prem)} (prereg ≈1.49, gate [1.3, 1.7])")

    # P-3 chained conditional: TACT <= blocking iff sum-max >= delta
    delta = 1.5
    cond_rows, wrong = [], 0
    for i in ids:
        r = rt[i]
        if not r["chained"]:
            continue
        saving = rb[i]["sum_lat"] - rb[i]["max_lat"]
        pred_tact_wins = saving >= delta
        actual_tact_wins = (r["realistic"]["completion"] is not None
                            and rb[i]["realistic"]["completion"] is not None
                            and r["realistic"]["completion"]
                            <= rb[i]["realistic"]["completion"])
        okp = pred_tact_wins == actual_tact_wins
        wrong += (not okp)
        cond_rows.append((i, round(saving, 2), pred_tact_wins, actual_tact_wins, okp))
    print(f"P-3 chained conditional (TACT≤blocking iff Σlat−max ≥ δ={delta}): "
          f"{len(cond_rows) - wrong}/{len(cond_rows)} direction-correct "
          f"(gate: wrong ≤2)")
    for i, sv, pw, aw, okp in cond_rows:
        if not okp:
            print(f"    MISS {i}: saving={sv} pred_win={pw} actual_win={aw}")

    # P-4 first-response invariance across profiles for the say-anchored arm
    same = sum(1 for i in ids
               if rt[i]["realistic"]["first"] == rt[i]["official"]["first"])
    print(f"P-4 tact first-response profile-invariance: {same}/{len(ids)} identical "
          f"(say anchor is tool-independent; non-identical = no-ack fallback cases)")

    out = Path(f"/root/autodl-tmp/fd-badcat/exp/w3/realistic_compare_{prov_b}__{prov_t}.json")
    out.write_text(json.dumps({
        "pair": [prov_b, prov_t], "n": len(ids),
        "P1_ratio_all_p50": r_all, "P1_ratio_booking_p50": r_book,
        "P2_premium_p50": p50(prem),
        "P3": [{"id": i, "saving": sv, "pred_win": pw, "actual_win": aw, "ok": okp}
               for i, sv, pw, aw, okp in cond_rows],
        "P4_invariant": same}, indent=1, ensure_ascii=False))
    print(f"  -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider")
    ap.add_argument("--compare", nargs=2, metavar=("BLOCKING", "TACT"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.provider:
        summarize(args.provider, rescore(args.provider))
    else:
        ap.error("need --provider or --compare")


if __name__ == "__main__":
    main()
