#!/usr/bin/env python3
"""
w4_ladder_report.py — one-shot report for the W4 adaptive ladder (rungs 1-3).

Reads result_{provider}.json files (already produced by w2r_stream_replay.py
--delta-policy ...) and emits, per arm:
  exact / state / done_p50, completion premium vs the blocking arm,
  premium recovery vs the fixed delta* arm, raw/two-point policy-window counts,
  per-kappa policy windows and
  realized commit delays (P2 prediction iii), finality-label stats (prompted
  arm), per-clip flips vs fixed, and the G2'-relevant verdict lines.

All comparisons are nominal-clock and against the SAME-regime comparators
(text-stack, workers-12, nominal infer) — latency-profile-immune premium only;
first-response is untouched by the ladder (ack path unchanged).

Usage:
  w4_ladder_report.py --arms w4k0_tact w4ks_tact w4kr_tact w4pf_tact \
      [--fixed w3p31_tact_d150] [--blocking w3p31_sblock] \
      [--out exp/w4/ladder_v0.json]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from w2r_score_grid import score, pct, DATA, _FOLDER_RE     # noqa: E402
from tact.tools import REVERSIBILITY                        # noqa: E402

KAPPAS = ("READ", "REV", "COMP", "IRR")


def kname(fn):
    r = REVERSIBILITY.get(fn)
    return r.name if r is not None else "IRR"


def rmap(rows):
    """rows -> {(id, occurrence_idx): row} (duplicate ids = multiple clips)."""
    c, m = Counter(), {}
    for r in rows:
        m[(r["id"], c[r["id"]])] = r
        c[r["id"]] += 1
    return m


def per_kappa_stats(provider):
    """Walk result files: policy windows + realized commit delays by kappa,
    plus finality-label stats when present."""
    win_by_k = defaultdict(list)      # policy window budgets (op_windows)
    win_counts = Counter()            # exact raw policy-window support
    win_counts_by_k = defaultdict(Counter)
    delay_by_k = defaultdict(list)    # realized t_commit - t_dec(launch)
    fin_labels_all, fin_labels_rb = Counter(), Counter()
    fin_infers, fin_unparsed = [], 0
    for folder in sorted(DATA.iterdir()):
        if not (folder.is_dir() and _FOLDER_RE.match(folder.name)):
            continue
        rp = folder / f"result_{provider}.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        tr = res.get("trace", {})
        launch_at, op_fn = {}, {}
        rb = None
        for d in tr.get("decisions", []):
            t_dec = d.get("t_dec")
            if t_dec is None:                       # non-spec core: t_eou + infer
                t_dec = round(d["t_eou"] + (d.get("infer_s") or 0.0), 3)
            for a in d.get("ops", []):
                if a.get("type") == "launch":
                    launch_at[a["op_id"]] = t_dec
                    op_fn[a["op_id"]] = a["fn"]
            for oid, w in (d.get("op_windows") or {}).items():
                fn = op_fn.get(int(oid))
                if fn:
                    k = kname(fn)
                    w = float(w)
                    win_by_k[k].append(w)
                    win_counts[w] += 1
                    win_counts_by_k[k][w] += 1
            if "finality" in d:
                fin_labels_all[d["finality"]] += 1
                if d.get("finality_infer_s") is not None:
                    fin_infers.append(d["finality_infer_s"])
                fin_unparsed += 1 if d.get("finality_unparsed") else 0
                if rb is None:
                    rb = _is_rollback(res.get("example_id"))
                if rb:
                    fin_labels_rb[d["finality"]] += 1
        for c in tr.get("commits", []):
            oid = c["op_id"]
            if oid in launch_at and oid in op_fn:
                delay_by_k[kname(op_fn[oid])].append(
                    round(c["t_commit"] - launch_at[oid], 3))
    def counts(c):
        return {f"{w:.1f}": c[w] for w in sorted(c)}

    n_windows = sum(win_counts.values())
    return {
        "n_policy_windows": n_windows,
        "policy_window_counts": counts(win_counts),
        "policy_window_counts_by_kappa": {
            k: counts(v) for k, v in sorted(win_counts_by_k.items())},
        "protect_fraction": (round(sum(n for w, n in win_counts.items() if w > 0)
                                   / n_windows, 3) if n_windows else None),
        "policy_window_mean": {k: round(sum(v) / len(v), 3)
                               for k, v in win_by_k.items() if v},
        "commit_delay_mean": {k: round(sum(v) / len(v), 3)
                              for k, v in delay_by_k.items() if v},
        "commit_delay_p50": {k: pct(v, .5) for k, v in delay_by_k.items() if v},
        "n_commits_by_kappa": {k: len(v) for k, v in delay_by_k.items()},
        "finality": ({"labels_all": dict(fin_labels_all),
                      "labels_rollback_clips": dict(fin_labels_rb),
                      "infer_p50": pct(fin_infers, .5),
                      "infer_p90": pct(fin_infers, .9),
                      "unparsed": fin_unparsed} if fin_labels_all else None),
    }


_RB_IDS = None
def _is_rollback(sid):
    global _RB_IDS
    if _RB_IDS is None:
        bench = json.load(open("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json"))
        items = bench["scenarios"] if isinstance(bench, dict) else bench
        _RB_IDS = {x["id"] for x in items if x.get("state_rollback_test")}
    return sid in _RB_IDS


def monotone(d, order=KAPPAS, ascending=True):
    xs = [d[k] for k in order if k in d]
    if len(xs) < 2:
        return None
    pairs = zip(xs, xs[1:])
    return all(a <= b for a, b in pairs) if ascending else all(a >= b for a, b in pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--fixed", default="w3p31_tact_d150")
    ap.add_argument("--blocking", default="w3p31_sblock")
    ap.add_argument("--out", default="/root/autodl-tmp/fd-badcat/exp/w4/ladder_v0.json")
    args = ap.parse_args()

    fixed = score(args.fixed);    fmap = rmap(fixed["rows"])
    block = score(args.blocking); bmap = rmap(block["rows"])

    def premium(arm_map):
        pres = [arm_map[k]["done"] - bmap[k]["done"] for k in arm_map
                if k in bmap and arm_map[k]["done"] is not None
                and bmap[k]["done"] is not None]
        return {"n_paired": len(pres), "sum_s": round(sum(pres), 1),
                "p50_s": pct(pres, .5)}

    fixed_prem = premium(fmap)
    report = {"fixed": {"provider": args.fixed, "n": fixed["n"],
                        "exact": fixed["exact"],
                        "state": fixed["state"], "done_p50": fixed["done_p50"],
                        "premium": fixed_prem},
              "blocking": {"provider": args.blocking, "n": block["n"],
                           "exact": block["exact"],
                           "done_p50": block["done_p50"]},
              "arms": []}

    hdr = (f"{'arm':14s} {'exact':>6s} {'state':>6s} {'done50':>7s} "
           f"{'prem_sum':>9s} {'recov_s':>8s} {'recov%':>7s} {'dExact':>7s}")
    print(f"[fixed  ] {args.fixed}: exact={fixed['exact']} done_p50={fixed['done_p50']} "
          f"premium_sum={fixed_prem['sum_s']}s (n={fixed_prem['n_paired']})")
    print(f"[block  ] {args.blocking}: exact={block['exact']} done_p50={block['done_p50']}")
    print(hdr)

    for arm in args.arms:
        s = score(arm)
        amap = rmap(s["rows"])
        prem = premium(amap)
        recov = [fmap[k]["done"] - amap[k]["done"] for k in amap
                 if k in fmap and amap[k]["done"] is not None
                 and fmap[k]["done"] is not None]
        recov_sum = round(sum(recov), 1)
        recov_frac = (round(recov_sum / fixed_prem["sum_s"], 3)
                      if fixed_prem["sum_s"] else None)
        # strict common-support variant (arm ∩ fixed ∩ blocking, all done
        # non-null) — the housing_25 audit caliber; both reported.
        common = [k for k in amap if k in fmap and k in bmap
                  and amap[k]["done"] is not None and fmap[k]["done"] is not None
                  and bmap[k]["done"] is not None]
        rs = round(sum(fmap[k]["done"] - amap[k]["done"] for k in common), 1)
        fs = sum(fmap[k]["done"] - bmap[k]["done"] for k in common)
        recov_strict = round(rs / fs, 3) if fs else None
        gains = sorted(k for k in amap if amap[k]["exact"]
                       and k in fmap and not fmap[k]["exact"])
        losses = sorted(k for k in amap if not amap[k]["exact"]
                        and fmap.get(k, {}).get("exact"))
        ks = per_kappa_stats(arm)
        entry = {"provider": arm, "n": s["n"],
                 "exact": s["exact"], "state": s["state"],
                 "done_p50": s["done_p50"], "premium": prem,
                 "recovery_vs_fixed_s": recov_sum,
                 "recovery_frac_of_fixed_premium": recov_frac,
                 "recovery_frac_strict_common_support": recov_strict,
                 "n_strict_common": len(common),
                 "d_exact_vs_fixed": round(s["exact"] - fixed["exact"], 3),
                 "flips_vs_fixed": {"gain": gains, "loss": losses},
                 "kappa_stats": ks,
                 "verdicts": {
                     "G2iii_exact_ge_minus1pt": s["exact"] - fixed["exact"] >= -0.011,
                     "premium_below_fixed": (prem["sum_s"] < fixed_prem["sum_s"]),
                     "delay_monotone_in_kappa":
                         monotone(ks["commit_delay_mean"], ascending=True),
                 }}
        report["arms"].append(entry)
        print(f"{arm:14s} {s['exact']:6.3f} {s['state']:6.3f} "
              f"{str(s['done_p50']):>7s} {prem['sum_s']:9.1f} {recov_sum:8.1f} "
              f"{str(recov_frac):>7s} {entry['d_exact_vs_fixed']:+7.3f}"
              f"   [strict n={len(common)}: {recov_strict}]")
        print(f"   windows(count) {ks['policy_window_counts']} "
              f"protect={ks['protect_fraction']} | "
              f"windows(mean) {ks['policy_window_mean']} | "
              f"delay(mean) {ks['commit_delay_mean']} | "
              f"monotone={entry['verdicts']['delay_monotone_in_kappa']}")
        if ks["finality"]:
            f = ks["finality"]
            print(f"   finality all={f['labels_all']} rb={f['labels_rollback_clips']} "
                  f"infer p50/p90={f['infer_p50']}/{f['infer_p90']} unparsed={f['unparsed']}")
        if gains or losses:
            print(f"   flips vs fixed: +{[g[0] for g in gains]} -{[l[0] for l in losses]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
