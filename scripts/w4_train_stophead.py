#!/usr/bin/env python3
"""
w4_train_stophead.py — train stopping-head v2 (two-stage policy, pure numpy).

Split: by dialogue id (did % 5 == 0 -> val) — no op/step leakage across splits.
Heads: (a) standardized logistic regression (primary, ~10^1 params) and
(b) one-hidden-layer tanh MLP h=16 (preregistered ablation, ~10^2 params) —
both class-balanced with prior correction. Feature list comes from the npz
(FEATS_V2 signal core for tag v2).

Policy (v2, preregistered in w4_ladder_design §12): TWO-STAGE —
    risk(op) = 1 - prod_{t < RISK_HORIZON} (1 - lambda_hat(t))
    window   = W_PROTECT (1.5) if risk >= theta else 0.0
Ranking is the only free variable; protect-all == the fixed delta* arm, so the
premium downside is structurally bounded (v1's 4s-window collapse is
unexpressible).

theta selection (synthetic-val only; FDB never touches training or selection):
replay each val dialogue's silence timeline with BARRIER GRACE (lever e,
metric-structural): a window expiring inside the next decision's guard
(GRACE = 1.0s nominal infer) defers under the commit barrier and is
patch-rescued, so an op is rescued iff  w > 0 and w > gap - GRACE;
premium(dialogue) = max op budget LEFT OVER past the final decision;
    cost = sum(premium) + KILL_PEN * misses,   KILL_PEN = 50.0 s/clip
(G2' exchange rate, unchanged from v1). Model (lr vs mlp) and theta are
selected JOINTLY by this cost; ties prefer lr (fewer params).

Usage: w4_train_stophead.py [--tag v2] [--epochs 300] [--lr 0.5] [--l2 1e-4]
Out:   exp/w4/stophead_{tag}.json           (selected head, policy=twostage)
       exp/w4/stophead_{tag}_{lr,mlp}.json  (audit copies, each with own theta)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from stophead import (StopHead, T_GRID, W_CAP, W_PROTECT, GRACE,   # noqa: E402
                      RISK_HORIZON)

KILL_PEN = 50.0
THETA_GRID = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08,
              0.12, 0.20, 0.35, 0.50]
MLP_HIDDEN = 16
MLP_EPOCHS = 400
MLP_LR = 0.01
MLP_SEED = 0


def auc(y, p):
    order = np.argsort(p)
    r = np.empty(len(p)); r[order] = np.arange(1, len(p) + 1)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_lr(Xn, ytr, sw, w_pos, epochs, lr, l2):
    n, d = Xn.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = sigmoid(Xn @ w + b)
        g = (p - ytr) * sw
        w -= lr * (Xn.T @ g / n + l2 * w)
        b -= lr * g.mean()
    b -= np.log(w_pos)   # prior correction: calibrated lambda_hat
    return w, b


def train_mlp(Xn, ytr, Xv, yva, sw, w_pos, l2):
    """Full-batch Adam, one tanh hidden layer; best-val-AUC snapshot kept."""
    rng = np.random.default_rng(MLP_SEED)
    n, d = Xn.shape
    W1 = rng.normal(0, 1 / np.sqrt(d), (d, MLP_HIDDEN))
    b1 = np.zeros(MLP_HIDDEN)
    W2 = rng.normal(0, 1 / np.sqrt(MLP_HIDDEN), MLP_HIDDEN)
    b2 = 0.0
    params = [W1, b1, W2, np.array([b2])]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best = (-1.0, None)
    for ep in range(1, MLP_EPOCHS + 1):
        Hh = np.tanh(Xn @ params[0] + params[1])
        p = sigmoid(Hh @ params[2] + params[3][0])
        gz = (p - ytr) * sw / n
        gW2 = Hh.T @ gz + l2 * params[2]
        gb2 = np.array([gz.sum()])
        gH = np.outer(gz, params[2]) * (1 - Hh ** 2)
        gW1 = Xn.T @ gH + l2 * params[0]
        gb1 = gH.sum(0)
        for i, g in enumerate([gW1, gb1, gW2, gb2]):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            mh = m[i] / (1 - beta1 ** ep)
            vh = v[i] / (1 - beta2 ** ep)
            params[i] = params[i] - MLP_LR * mh / (np.sqrt(vh) + eps)
        if ep % 20 == 0 or ep == MLP_EPOCHS:
            Hv = np.tanh(Xv @ params[0] + params[1])
            a = auc(yva, sigmoid(Hv @ params[2] + params[3][0]))
            if a > best[0]:
                best = (a, [p.copy() for p in params])
    W1, b1, W2, b2a = best[1]
    return W1, b1, W2, float(b2a[0] - np.log(w_pos)), best[0]


# -- two-stage replay (premium-faithful + barrier grace) ----------------------
def precompute(model, dlgs):
    """Per-op (risk_at_launch, risk_at_rescue, tail silences) — risk is
    theta-independent, so the sweep re-runs only the cheap accounting."""
    pre = []
    for dlg in dlgs:
        sig, M = dlg["sigmas"], len(dlg["eous"])
        tail = lambda i: sum(sig[i:M - 1])
        rows = []
        for o in dlg["ops"]:
            j = o.get("rev_eou")
            rr = tailr = None
            if j is not None:
                e = dlg["eous"][j]
                s = {**o, "eou_idx": j, "utt_dur": e["utt_dur"],
                     "gap_prev": e["gap_prev"], "finality": e["finality"],
                     "slots_missing": 0}
                rr, tailr = model.risk(s), tail(j)
            rows.append({"o": o, "r0": model.risk(o), "rr": rr,
                         "tail0": tail(o["eou_idx"]), "tailr": tailr})
        pre.append(rows)
    return pre


def replay_cost(pre, theta, wp=W_PROTECT):
    prem = 0.0
    miss = n_prot = n_ops = cov_n = cov_d = 0
    for rows in pre:
        leftovers = [0.0]
        for r in rows:
            o = r["o"]; n_ops += 1
            w0 = wp if r["r0"] >= theta else 0.0
            n_prot += w0 > 0
            g = o["gap_silence"]
            if g is not None:
                rescuable = (g - GRACE) < wp
                cov_d += rescuable
                if w0 <= 0.0 or w0 <= g - GRACE:   # barrier grace (lever e)
                    miss += 1
                    continue
                cov_n += rescuable
                wr = wp if r["rr"] >= theta else 0.0
                leftovers.append(max(0.0, wr - r["tailr"]))
            else:
                leftovers.append(max(0.0, w0 - r["tail0"]))
        prem += max(leftovers)
    return {"premium": prem, "miss": miss,
            "protect_rate": n_prot / max(1, n_ops),
            "coverage": cov_n / max(1, cov_d), "n_rescuable": cov_d,
            "cost": prem + KILL_PEN * miss}


def sweep(name, pre, n_dlg, verbose=True):
    n_rev = sum(1 for rows in pre for r in rows if r["o"]["gap_silence"] is not None)
    if verbose:
        print(f"theta sweep [{name}] on {n_dlg} dialogues "
              f"(cost = sum(tail premium) + {KILL_PEN}*miss; {n_rev} revised ops):")
    best = None
    for th in THETA_GRID:
        r = replay_cost(pre, th)
        if verbose:
            print(f"  theta={th:<6} premium={r['premium']:8.0f}s "
                  f"({r['premium'] / n_dlg:.3f}/dlg) miss={r['miss']}/{n_rev} "
                  f"protect={r['protect_rate']:.1%} cover(resc)={r['coverage']:.1%} "
                  f"cost={r['cost']:.0f}")
        if best is None or r["cost"] < best[1]["cost"]:
            best = (th, r)
    if verbose:
        print(f"  [{name}] best theta={best[0]} cost={best[1]['cost']:.0f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-4)
    args = ap.parse_args()
    base = Path("/root/autodl-tmp/fd-badcat/exp/w4/synth")
    z = np.load(base / f"hazard_{args.tag}.npz", allow_pickle=True)
    X, y, did = z["X"].astype(np.float64), z["y"].astype(np.float64), z["did"]
    feats = [str(f) for f in z["feats"]]
    val = (did % 5 == 0)
    Xtr, ytr, Xva, yva = X[~val], y[~val], X[val], y[val]

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    Xn, Xv = (Xtr - mu) / sd, (Xva - mu) / sd
    w_pos = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    sw = np.where(ytr == 1, w_pos, 1.0); sw /= sw.mean()
    print(f"feats({len(feats)}) = {feats}")

    w, b = train_lr(Xn, ytr, sw, w_pos, args.epochs, args.lr, args.l2)
    pv = sigmoid(Xv @ w + b)
    print(f"[lr]  train n={len(ytr)} pos={ytr.mean():.2%} | val n={len(yva)} "
          f"AUC={auc(yva, pv):.3f}")
    bins = np.quantile(pv, [0, .2, .4, .6, .8, 1.0])
    for i in range(5):
        msk = (pv >= bins[i]) & (pv <= bins[i + 1])
        if msk.sum():
            print(f"  calib bin {i}: pred={pv[msk].mean():.3f} "
                  f"actual={yva[msk].mean():.3f} n={msk.sum()}")

    W1, b1, W2, b2, mlp_auc = train_mlp(Xn, ytr, Xv, yva, sw, w_pos, args.l2)
    print(f"[mlp] hidden={MLP_HIDDEN} seed={MLP_SEED} best-val AUC={mlp_auc:.3f}")

    common = {"feats": feats, "mean": mu.tolist(), "std": sd.tolist(),
              "t_grid": T_GRID, "w_cap": W_CAP, "policy": "twostage",
              "theta": None, "w_protect": W_PROTECT,
              "risk_horizon": RISK_HORIZON, "grace": GRACE, "c_w": None}
    heads = {
        "lr": StopHead({**common, "version": f"stophead_{args.tag}_lr",
                        "w": w.tolist(), "b": float(b)}),
        "mlp": StopHead({**common, "version": f"stophead_{args.tag}_mlp",
                         "arch": "mlp", "W1": W1.T.tolist(),
                         "b1": b1.tolist(), "W2": W2.tolist(), "b2": b2}),
    }

    # -- joint (model, theta) selection on val dialogues ----------------------
    # Selection slice (prereg §12): the BOTTOM TERCILE of val dialogues by cfg
    # rev_intensity. The randomized mixture is a TEACHING distribution whose
    # high-revision regimes are deliberately exaggerated (rank robustness);
    # its pooled economics over-protect (v1's confirmed death mode). The
    # low-intensity band carries the deployment-representative economics.
    # Full-mixture and per-tercile theta* are reported for the record.
    dlgs = [json.loads(l) for l in (base / f"dialogues_{args.tag}.jsonl").open()]
    dlgs = [d for d in dlgs if d["did"] % 5 == 0]
    intens = np.array([d["cfg"]["rev_intensity"] for d in dlgs])
    t1, t2 = np.quantile(intens, [1 / 3, 2 / 3])
    idx_low = [i for i, x in enumerate(intens) if x <= t1]
    print(f"selection slice = bottom tercile by rev_intensity "
          f"(<= {t1:.3f}; n={len(idx_low)}/{len(dlgs)} val dialogues)")
    results, pres = {}, {}
    for name in ("lr", "mlp"):                      # lr first: ties prefer lr
        pre = precompute(heads[name], dlgs)
        pres[name] = pre
        th, r = sweep(name, [pre[i] for i in idx_low], len(idx_low))
        heads[name].d["theta"] = th
        results[name] = r
        for lab, keep in (("mid", lambda x: t1 < x <= t2),
                          ("high", lambda x: x > t2),
                          ("full", lambda x: True)):
            sub = [pre[i] for i, x in enumerate(intens) if keep(x)]
            bt, br = sweep(f"{name}:{lab}", sub, max(1, len(sub)), verbose=False)
            print(f"  [{name}] {lab}-mixture theta*={bt} "
                  f"(cost={br['cost']:.0f}, protect={br['protect_rate']:.1%}) "
                  f"— record only")
    winner = "lr" if results["lr"]["cost"] <= results["mlp"]["cost"] else "mlp"
    sel = heads[winner]
    sel.d["selected_from"] = {k: round(v["cost"], 1) for k, v in results.items()}
    sel.d["selection_slice"] = {"by": "cfg.rev_intensity", "tercile": "bottom",
                                "cut": round(float(t1), 4), "n": len(idx_low)}
    print(f"SELECTED head = {winner} (theta={sel.d['theta']}, "
          f"low-slice cost lr={results['lr']['cost']:.0f} vs "
          f"mlp={results['mlp']['cost']:.0f})")

    # -- structure diagnostics at the chosen point ----------------------------
    th = sel.d["theta"]
    by = {"finality": {}, "style": {}, "position": {}, "kind_cover": {}}
    for rows, dlg in zip(pres[winner], dlgs):
        M = len(dlg["eous"])
        for r in rows:
            o, prot = r["o"], r["r0"] >= th
            for key, val_ in (("finality", o["finality"]), ("style", o["style"]),
                              ("position", "final_eou" if o["eou_idx"] == M - 1
                               else "earlier")):
                c = by[key].setdefault(val_, [0, 0]); c[0] += prot; c[1] += 1
            if o["gap_silence"] is not None and (o["gap_silence"] - GRACE) < W_PROTECT:
                c = by["kind_cover"].setdefault(o["revision_kind"], [0, 0])
                c[0] += prot; c[1] += 1
    for key, tab in by.items():
        print(f"protect by {key}:",
              {k: f"{a}/{n} ({a / n:.0%})" for k, (a, n) in sorted(tab.items())})

    outdir = Path("/root/autodl-tmp/fd-badcat/exp/w4")
    for name, head in heads.items():
        (outdir / f"stophead_{args.tag}_{name}.json").write_text(json.dumps(head.d))
    out = outdir / f"stophead_{args.tag}.json"
    out.write_text(json.dumps(sel.d))
    print("->", out, f"(+ audit copies stophead_{args.tag}_{{lr,mlp}}.json)")


if __name__ == "__main__":
    main()
