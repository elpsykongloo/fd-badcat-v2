#!/usr/bin/env python3
"""
w4_ceiling_diag.py — causal-separability CEILING diagnostic on FDB (zero GPU).

Question: on FDB's own feature distribution (the same 18 features the stopping
head uses), are will-be-revised ops separable from safe ops AT ALL?  This is a
hindsight DIAGNOSTIC in the same class as w3_oracle_frontier — it never
produces a deployable policy and nothing here enters training (firewall doc:
w4_ladder_design §10.5-next).

Method:
  1. Build per-op records from the FIXED arm's traces (launch features incl.
     args from tx_log; finality per EoU cross-referenced from the w4pf arm —
     identical audio => identical finality calls, cache-verified 217/0).
  2. Hindsight label: op revised iff (a) a later decision patches it, or
     (b) ledger patch_after_commit references it, or (c) the same fn is
     re-launched at a later decision (relaunch-style revision). gap = silence
     clock between launch decision and revising decision (from segs).
  3. Leave-one-CLIP-out logistic regression -> out-of-fold risk scores -> AUC.
  4. Ceiling frontier: threshold sweep; protected ops wait W_CAP (post-rescue
     window 0 = hindsight-optimal, learnable per v1's rescue negatives),
     unprotected commit at once. clips_lost = clips with any unprotected
     revised op; premium = per-clip max leftover past the final decision.
     Sanity: the same simulator with W=1.5 everywhere must reproduce the
     measured fixed premium (~109.6s).

Usage: w4_ceiling_diag.py [--fixed w3p31_tact_d150] [--finality-arm w4pf_tact]
Out:   exp/w4/ceiling_diag.json + printed report.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from stophead import (featurize, FEATS, kappa_name,          # noqa: E402
                      slots_missing_from_args, chain_dep_from_args)

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
W_PROTECT = 4.0


def silence_between(t1, t2, segs):
    sil = t2 - t1
    for s, e in segs:
        sil -= max(0.0, min(t2, e) - max(t1, s))
    return max(0.0, sil)


def build_records(fixed, fin_arm):
    clips = []
    for folder in sorted(DATA.iterdir()):
        rp = folder / f"result_{fixed}.json"
        if not (folder.is_dir() and rp.exists()):
            continue
        r = json.loads(rp.read_text())
        pf = folder / f"result_{fin_arm}.json"
        fin = ([d.get("finality") for d in
                json.loads(pf.read_text())["trace"]["decisions"]]
               if pf.exists() else [])
        segs, decs = r["trace"]["segs"], r["trace"]["decisions"]
        args_by_op = {e["op_id"]: e.get("args", {}) for e in r["tx_log"]
                      if e.get("op") == "launch"}
        t_dec = [d["t_eou"] + (d.get("infer_s") or 0.0) for d in decs]
        ops, n_prior = [], 0
        for i, d in enumerate(decs):
            seg = segs[d["seg_idx"]]
            for a in d.get("ops", []):
                if a["type"] != "launch":
                    continue
                args = args_by_op.get(a["op_id"], {})
                ops.append({
                    "op_id": a["op_id"], "fn": a["fn"], "eou_idx": i,
                    "utt_dur": round(seg[1] - seg[0], 3),
                    "gap_prev": round(seg[0] - (segs[d["seg_idx"] - 1][1]
                                                if d["seg_idx"] else 0.0), 3),
                    "n_prior_ops": n_prior,
                    "slots_missing": slots_missing_from_args(a["fn"], args),
                    "chain_dep": chain_dep_from_args(args),
                    "kappa": kappa_name(a["fn"]),
                    "finality": fin[i] if i < len(fin) else None,
                    "domain": r.get("category"),
                    "rev_gap": None})
            n_prior = len(ops)
        # hindsight revision labels (+ kind: patch / relaunch / pac)
        by_id = {o["op_id"]: o for o in ops}
        for j, d in enumerate(decs):
            for a in d.get("ops", []):
                if a["type"] == "patch" and a.get("op_id") in by_id:
                    o = by_id[a["op_id"]]
                    if j > o["eou_idx"] and o["rev_gap"] is None:
                        o["rev_gap"] = silence_between(
                            t_dec[o["eou_idx"]], t_dec[j], segs)
                        o["rev_kind"] = "patch"
                elif a["type"] == "launch":          # relaunch heuristic
                    prev = [o for o in ops if o["fn"] == a["fn"]
                            and o["eou_idx"] < j and o["op_id"] != a["op_id"]]
                    if prev and prev[0]["rev_gap"] is None:
                        prev[0]["rev_gap"] = silence_between(
                            t_dec[prev[0]["eou_idx"]], t_dec[j], segs)
                        prev[0]["rev_kind"] = "relaunch"
        for pac in r.get("ledger", {}).get("patch_after_commit", []):
            oid = (pac.get("ref") or {}).get("op_id")
            if oid in by_id and by_id[oid]["rev_gap"] is None:
                by_id[oid]["rev_gap"] = 999.0        # revised, gap unknown/late
                by_id[oid]["rev_kind"] = "pac"
        # silence from each launch to the final decision (for premium sim)
        for o in ops:
            o["sil_to_final"] = silence_between(
                t_dec[o["eou_idx"]], t_dec[-1], segs)
        clips.append({"clip": folder.name, "ops": ops,
                      "rollback": None})
    return clips


def fit_lr(X, y, epochs=400, lr=0.5, l2=1e-3):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xn = (X - mu) / sd
    w_pos = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))
    sw = np.where(y == 1, w_pos, 1.0); sw /= sw.mean()
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        g = (p - y) * sw
        w -= lr * (Xn.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b, mu, sd


def predict(w, b, mu, sd, X):
    return 1 / (1 + np.exp(-np.clip(((X - mu) / sd) @ w + b, -30, 30)))


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def frontier(clips, scores, w_protect):
    """Threshold sweep -> (clips_lost, premium_sum). Lost is counted on
    LOAD-BEARING ops only (patch-rescued under fixed AND clip passes there);
    protected ops that are not load-bearing-rescued pay leftover premium."""
    rows = []
    flat = [(o, s) for c, ss in zip(clips, scores) for o, s in zip(c["ops"], ss)]
    for th in sorted({round(s, 4) for _, s in flat} | {0.0, 1.1}):
        lost = prem = 0.0
        for c, ss in zip(clips, scores):
            leftovers, dead = [0.0], False
            for o, s in zip(c["ops"], ss):
                prot = s >= th
                if o.get("lb"):
                    # barrier rescue: an expiry inside the patching decision's
                    # guard (t_disp..t_dec, ~1.0s nominal infer) is deferred and
                    # rescued by the patch => effective gap = gap - 1.0.
                    if not prot or w_protect <= max(0.0, min(o["rev_gap"], 4.0) - 1.0):
                        dead = True
                    # rescued lb op: post-rescue window 0 -> no premium
                elif prot:
                    leftovers.append(max(0.0, w_protect - o["sil_to_final"]))
            lost += dead
            prem += max(leftovers)
        rows.append({"th": th, "clips_lost": int(lost), "premium_s": round(prem, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default="w3p31_tact_d150")
    ap.add_argument("--finality-arm", default="w4pf_tact")
    args = ap.parse_args()
    clips = build_records(args.fixed, args.finality_arm)
    n_ops = sum(len(c["ops"]) for c in clips)
    n_rev = sum(1 for c in clips for o in c["ops"] if o["rev_gap"] is not None)
    print(f"clips={len(clips)} ops={n_ops} revised={n_rev}")

    # load-bearing = patch-rescued under fixed AND the clip passes there
    from collections import Counter as _C
    grid = {e["provider"]: e for e in json.loads(Path(
        "/root/autodl-tmp/fd-badcat/exp/w2_rerun/grid_full_v31.json").read_text())}
    cnt, exact_map = _C(), {}
    for row in grid[args.fixed]["rows"]:
        exact_map[(row["id"], cnt[row["id"]])] = row["exact"]
        cnt[row["id"]] += 1
    occ = _C()
    kinds, gaps_by_kind = _C(), {}
    n_lb = 0
    for c in clips:
        sid = "_".join(c["clip"].split("_")[:-1])
        passed = exact_map.get((sid, occ[sid]), False)
        occ[sid] += 1
        for o in c["ops"]:
            if o["rev_gap"] is not None:
                k = o.get("rev_kind", "?")
                kinds[k] += 1
                gaps_by_kind.setdefault(k, []).append(round(o["rev_gap"], 2))
                o["lb"] = (k == "patch" and passed)
                n_lb += o["lb"]
    print(f"revision kinds: {dict(kinds)} | load-bearing (patch & clip-pass): {n_lb}")
    for k, g in gaps_by_kind.items():
        print(f"  gaps[{k}]: {sorted(g)}")

    # sanity: fixed-sim premium (all windows 1.5, restart 1.5)
    prem = 0.0
    for c in clips:
        lo = [0.0]
        for o in c["ops"]:
            g = o["rev_gap"]
            if g is not None and 1.5 > g:      # rescued; restart 1.5 at rev
                lo.append(max(0.0, 1.5 - max(0.0, o["sil_to_final"] - g)))
            elif g is None:
                lo.append(max(0.0, 1.5 - o["sil_to_final"]))
        prem += max(lo)
    print(f"sanity fixed-sim premium = {prem:.1f}s (measured ladder: 109.6s)")

    X = np.array([featurize(o, 0.0) for c in clips for o in c["ops"]])
    y_any = np.array([float(o["rev_gap"] is not None)
                      for c in clips for o in c["ops"]])
    y = np.array([float(bool(o.get("lb"))) for c in clips for o in c["ops"]])
    # leave-one-clip-out
    sizes = [len(c["ops"]) for c in clips]
    idx0, oof = 0, np.zeros(len(y))
    bounds = []
    for s in sizes:
        bounds.append((idx0, idx0 + s)); idx0 += s
    for lo_, hi in bounds:
        mask = np.ones(len(y), bool); mask[lo_:hi] = False
        w, b, mu, sd = fit_lr(X[mask], y[mask])
        oof[lo_:hi] = predict(w, b, mu, sd, X[lo_:hi])
    print(f"LOO(clip) AUC on LOAD-BEARING task = {auc(y, oof):.3f} "
          f"(any-revision task, in-sample ref: {auc(y_any, oof):.3f})")

    # single-feature AUCs + full-fit weights (signal attribution)
    print("single-feature AUC (top):")
    fa = sorted(((auc(y, X[:, i]) or 0.5, FEATS[i]) for i in range(X.shape[1])),
                key=lambda t: -abs(t[0] - 0.5))
    for a_, f_ in fa[:8]:
        print(f"  {f_:16s} {a_:.3f}")

    scores = []
    i0 = 0
    for s in sizes:
        scores.append(oof[i0:i0 + s]); i0 += s
    fixed_prem = 109.6
    all_rows = {}
    for wp in (1.0, 1.5, 2.0, 2.5, 4.0):
        rows = frontier(clips, scores, wp)
        all_rows[wp] = rows
        # best premium at each lost-count (the ceiling is the lower envelope)
        best_by_lost = {}
        for r in rows:
            if (r["clips_lost"] not in best_by_lost
                    or r["premium_s"] < best_by_lost[r["clips_lost"]]):
                best_by_lost[r["clips_lost"]] = r["premium_s"]
        line = "  ".join(f"lost={k}:{(fixed_prem - v) / fixed_prem:.0%}"
                         for k, v in sorted(best_by_lost.items())[:6])
        print(f"W_protect={wp}: {line}")
    rows = all_rows[W_PROTECT]
    print("ceiling frontier (LOO scores; lost<=? vs recovery):")
    seen = set()
    for r in rows:
        key = (r["clips_lost"],)
        if key in seen:
            continue
        seen.add(key)
        rec = (fixed_prem - r["premium_s"]) / fixed_prem
        print(f"  th={r['th']:<7} lost={r['clips_lost']:2d} "
              f"premium={r['premium_s']:6.1f}s recovery={rec:6.1%}")
    # oracle-features line: protect exactly the load-bearing ops
    oracle = [np.array([1.0 if o.get("lb") else 0.0
                        for o in c["ops"]]) for c in clips]
    orow = frontier(clips, oracle, 2.0)
    best = min(orow, key=lambda r: (r["clips_lost"], r["premium_s"]))
    print(f"oracle-features: lost={best['clips_lost']} premium={best['premium_s']}s "
          f"recovery={(fixed_prem - best['premium_s']) / fixed_prem:.1%}")

    # transfer check: how well do the firewall-trained models rank FDB lb ops?
    transfer = {}
    from stophead import StopHead
    for tag in ("v0", "v1"):
        mp = Path(f"/root/autodl-tmp/fd-badcat/exp/w4/stophead_{tag}.json")
        if not mp.exists():
            continue
        m = StopHead.load(mp)
        s = np.array([m.hazard(featurize(o, 0.0)) for c in clips for o in c["ops"]])
        sc, i0 = [], 0
        for c in clips:
            sc.append(s[i0:i0 + len(c["ops"])]); i0 += len(c["ops"])
        env = {}
        for r in frontier(clips, sc, 1.5):
            if r["clips_lost"] not in env or r["premium_s"] < env[r["clips_lost"]]:
                env[r["clips_lost"]] = r["premium_s"]
        transfer[tag] = {"auc": auc(y, s),
                         "w15_frontier": {k: round((fixed_prem - v) / fixed_prem, 3)
                                          for k, v in sorted(env.items())[:5]}}
        print(f"transfer stophead_{tag}: AUC={transfer[tag]['auc']:.3f} "
              f"W=1.5 frontier={transfer[tag]['w15_frontier']}")

    out = Path("/root/autodl-tmp/fd-badcat/exp/w4/ceiling_diag.json")
    out.write_text(json.dumps({"n_ops": n_ops, "n_rev": n_rev, "n_lb": n_lb,
                               "loo_auc": auc(y, oof),
                               "w_sweep_env": {str(wp): {
                                   str(r["clips_lost"]): r["premium_s"]
                                   for r in all_rows[wp]} for wp in all_rows},
                               "frontier_w4": rows,
                               "transfer": transfer,
                               "feature_auc": [(f, a) for a, f in fa]}, indent=1))
    print("->", out)


if __name__ == "__main__":
    main()
