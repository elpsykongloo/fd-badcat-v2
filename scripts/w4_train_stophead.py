#!/usr/bin/env python3
"""
w4_train_stophead.py — train stopping-head v1 (logistic hazard, pure numpy).

Split: by dialogue id (did % 5 == 0 -> val) — no op/step leakage across splits.
Model: standardized logistic regression, class-balanced + prior correction
(lambda_hat is an approximately calibrated probability).

c_w selection (v1, PREMIUM-FAITHFUL — preregistered in w4_ladder_design §10;
synthetic-val only, FDB never touches training or selection):
replay the policy on each val dialogue's silence timeline:
  - op revised at gap g: window w0 <= g -> MISS (scenario kill); else rescued,
    restarted at the revising EoU with the policy's new window there;
  - premium(dialogue) = max over ops of the budget LEFT OVER past the final
    decision (mid-dialogue waiting is free; only tail spill delays completion)
  cost = sum(premium) + KILL_PEN * misses,  KILL_PEN = 50.0 s/clip
(derived from the G2' gate exchange rate: -1pt <-> >=51.5s per 100 scenarios).
v0's uniform per-second cost + 3s miss price was a mis-specification of the
published scoring rules — both corrections are metric-structural, no FDB
content or statistics enter (leakage firewall intact).

Usage: w4_train_stophead.py [--tag v1] [--epochs 300] [--lr 0.5] [--l2 1e-4]
Out:   exp/w4/stophead_{tag}.json  (+ printed AUC / calibration / sweep / structure)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/src")
from stophead import StopHead, T_GRID, W_CAP, FEATS, KAPPAS   # noqa: E402

KILL_PEN = 50.0
CW_GRID = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def dialogue_cost(model, dlg, cw):
    """Premium-faithful replay of the policy on one dialogue's timeline."""
    sig, M = dlg["sigmas"], len(dlg["eous"])
    tail = lambda i: sum(sig[i:M - 1])       # silence from decision i to final
    leftovers, miss, wins = [0.0], 0, []
    for o in dlg["ops"]:
        w0 = model.window(o, c_w=cw)
        wins.append((o, w0))
        j = o.get("rev_eou")
        if j is not None:
            if w0 <= o["gap_silence"]:
                miss += 1                    # committed stale mid-dialogue
                continue
            e = dlg["eous"][j]
            s = {**o, "eou_idx": j, "utt_dur": e["utt_dur"],
                 "gap_prev": e["gap_prev"], "finality": e["finality"],
                 "slots_missing": 0}
            leftovers.append(max(0.0, model.window(s, c_w=cw) - tail(j)))
        else:
            leftovers.append(max(0.0, w0 - tail(o["eou_idx"])))
    return max(leftovers), miss, wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
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
    b -= np.log(w_pos)   # prior correction: calibrated lambda_hat
    pv = 1 / (1 + np.exp(-np.clip(Xv @ w + b, -30, 30)))
    print(f"train n={n} pos={ytr.mean():.2%} | val n={len(yva)} "
          f"AUC={auc(yva, pv):.3f}")
    bins = np.quantile(pv, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        m = (pv >= bins[i]) & (pv <= bins[i + 1])
        if m.sum():
            print(f"  calib bin {i}: pred={pv[m].mean():.3f} "
                  f"actual={yva[m].mean():.3f} n={m.sum()}")

    model = StopHead({"version": f"stophead_{args.tag}", "feats": FEATS,
                      "mean": mu.tolist(), "std": sd.tolist(),
                      "w": w.tolist(), "b": float(b),
                      "t_grid": T_GRID, "w_cap": W_CAP, "c_w": None})

    # -- premium-faithful c_w sweep on val dialogues --------------------------
    dlgs = [json.loads(l) for l in (base / f"dialogues_{args.tag}.jsonl").open()]
    dlgs = [d for d in dlgs if d["did"] % 5 == 0]
    n_rev = sum(1 for d in dlgs for o in d["ops"] if o["gap_silence"] is not None)
    print(f"c_w sweep on {len(dlgs)} val dialogues "
          f"(cost = sum(tail premium) + {KILL_PEN}*miss; {n_rev} revised ops):")
    best = None
    for cw in CW_GRID:
        prem = miss = 0.0
        for dlg in dlgs:
            p, m, _ = dialogue_cost(model, dlg, cw)
            prem += p; miss += m
        cost = prem + KILL_PEN * miss
        print(f"  c_w={cw:<6} premium={prem:8.0f}s ({prem / len(dlgs):.3f}/dlg) "
              f"miss={int(miss)}/{n_rev} cost={cost:.0f}")
        if best is None or cost < best[1]:
            best = (cw, cost)
    model.d["c_w"] = best[0]
    print(f"chosen c_w={best[0]}")

    # -- structure diagnostics at the chosen point ----------------------------
    by_k, by_pos = {k: [] for k in KAPPAS}, {"final_eou": [], "earlier": []}
    for dlg in dlgs:
        M = len(dlg["eous"])
        _, _, wins = dialogue_cost(model, dlg, best[0])
        for o, w0 in wins:
            by_k[o["kappa"]].append(w0)
            by_pos["final_eou" if o["eou_idx"] == M - 1 else "earlier"].append(w0)
    print("mean window by kappa:",
          {k: round(sum(v) / len(v), 3) for k, v in by_k.items() if v})
    print("mean window by position:",
          {k: round(sum(v) / len(v), 3) for k, v in by_pos.items() if v})

    out = Path(f"/root/autodl-tmp/fd-badcat/exp/w4/stophead_{args.tag}.json")
    out.write_text(json.dumps(model.d))
    print("->", out)


if __name__ == "__main__":
    main()
