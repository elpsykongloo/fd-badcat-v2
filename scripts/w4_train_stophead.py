#!/usr/bin/env python3
"""
w4_train_stophead.py — train stopping-head v0 (logistic hazard, pure numpy).

Split: by dialogue id (did % 5 == 0 -> val) — no op/step leakage across splits.
Model: standardized logistic regression, class-balanced, full-batch GD.
c_w selection (preregistered objective, synthetic-val only — FDB never touches
training or selection): for each candidate c_w, apply the P2 threshold policy
to every val op and score
    cost = sum(window_assigned) + MISS_PEN * sum(C_kappa over missed revisions)
with MISS_PEN = 3.0 s-equivalents per unit C_kappa. Pick argmin.

Usage: w4_train_stophead.py [--tag v0] [--epochs 300] [--lr 0.5] [--l2 1e-4]
Out:   exp/w4/stophead_{tag}.json  (+ printed val AUC / calibration / sweep)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/src")
from stophead import StopHead, C_KAPPA, KAPPAS, T_GRID, FEATS   # noqa: E402

MISS_PEN = 3.0
CW_GRID = [0.02, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 0.8, 1.2]


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args()
    base = Path("/root/autodl-tmp/fd-badcat/exp/w4/synth")
    z = np.load(base / f"hazard_{args.tag}.npz", allow_pickle=True)
    X, y, did = z["X"].astype(np.float64), z["y"].astype(np.float64), z["did"]
    val = (did % 5 == 0)
    Xtr, ytr, Xva, yva = X[~val], y[~val], X[val], y[val]

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xn, Xv = (Xtr - mu) / sd, (Xva - mu) / sd
    n, d = Xn.shape
    w_pos = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    sw = np.where(ytr == 1, w_pos, 1.0); sw /= sw.mean()

    w, b = np.zeros(d), 0.0
    for ep in range(args.epochs):
        p = 1 / (1 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        g = (p - ytr) * sw
        w -= args.lr * (Xn.T @ g / n + args.l2 * w)
        b -= args.lr * g.mean()
    b -= np.log(w_pos)   # prior correction: undo class-weight inflation so
    #                      lambda_hat is a (approximately) calibrated probability
    pv = 1 / (1 + np.exp(-np.clip(Xv @ w + b, -30, 30)))
    print(f"train n={n} pos={ytr.mean():.2%} | val n={len(yva)} "
          f"AUC={auc(yva, pv):.3f}")
    bins = np.quantile(pv, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (pv >= bins[i]) & (pv <= bins[i + 1])
        if m.sum():
            print(f"  calib bin {i}: pred={pv[m].mean():.3f} actual={yva[m].mean():.3f} n={m.sum()}")

    model = StopHead({"version": f"stophead_{args.tag}", "feats": FEATS,
                      "mean": mu.tolist(), "std": sd.tolist(),
                      "w": w.tolist(), "b": float(b),
                      "t_grid": T_GRID, "c_w": None})

    # -- c_w sweep on val ops (policy-level, preregistered cost) -------------
    ops = [json.loads(l) for l in (base / f"ops_{args.tag}.jsonl").open()
           if json.loads(l)["did"] % 5 == 0]
    print(f"c_w sweep on {len(ops)} val ops (cost = sum(w) + {MISS_PEN}*C_k*miss):")
    best = None
    for cw in CW_GRID:
        tot_w = tot_pen = miss = 0
        for o in ops:
            wnd = model.window(o, c_w=cw)
            tot_w += wnd
            if o["gap_silence"] is not None and wnd < o["gap_silence"]:
                miss += 1
                tot_pen += MISS_PEN * C_KAPPA[o["kappa"]]
        cost = tot_w + tot_pen
        rev = sum(1 for o in ops if o["gap_silence"] is not None)
        print(f"  c_w={cw:<5} mean_w={tot_w / len(ops):.3f} miss={miss}/{rev} "
              f"cost={cost:.0f}")
        if best is None or cost < best[1]:
            best = (cw, cost)
    model.d["c_w"] = best[0]
    print(f"chosen c_w={best[0]}")

    out = Path(f"/root/autodl-tmp/fd-badcat/exp/w4/stophead_{args.tag}.json")
    out.write_text(json.dumps(model.d))
    print("->", out)


if __name__ == "__main__":
    main()
