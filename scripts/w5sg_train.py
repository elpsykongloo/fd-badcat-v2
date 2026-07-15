#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w5sg_train.py — W5-SG Phase-1 probe (K2 gate) and Phase-2 trainer
(docs/w5_specgate_design.md §4/§6/§7).

Probe (Phase-1, zero deployment training):
  $PY scripts/w5sg_train.py --probe
      grouped 5-fold OOF logistic probe on exp/w5sg/events.jsonl ->
      OOF AUC + precision@recall>=0.85; K2 verdict (kill if < 0.50).

Train (Phase-2, only after the §9 numeric freeze):
  $PY scripts/w5sg_train.py
      group-hash split (folds 0-3 train / fold 4 val), LR main arm + MLP(h=16)
      ablation; theta per the FROZEN §6 rule — argmax_theta precision(theta)
      s.t. recall(theta) >= 0.85 ON THE HUMDIAL VAL SPLIT (never FDB) — then
      writes exp/w5sg/specgate_v0.json (stophead-convention JSON + provenance).

  $PY scripts/w5sg_train.py --selftest   # stdlib+numpy synthetic check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from specgate import FEATS_SG  # noqa: E402
from w4v3_common import fit_lr, predict_lr, auc, group_fold  # noqa: E402

RECALL_FLOOR = 0.85            # frozen operating rule (design §6)
K2_PREC_BAR = 0.50             # frozen kill: precision@recall0.85 below => cancel
VAL_FOLD = 4


def load_events(path):
    import numpy as np
    X, y, g = [], [], []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        X.append(r["f"])
        y.append(r["y"])
        g.append(r["g"])
    meta_p = Path(str(path).replace(".jsonl", "_meta.json"))
    feats = (json.loads(meta_p.read_text())["feats"] if meta_p.exists()
             else list(FEATS_SG))
    return np.asarray(X, float), np.asarray(y, int), g, feats


def precision_at_recall(y, p, recall_floor=RECALL_FLOOR):
    """Best (precision, theta, recall) with recall >= floor; falls back to the
    max-recall point if the floor is unreachable. Thresholds swept on the
    observed score set (monotone; exact)."""
    import numpy as np
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    npos = int(y.sum())
    if npos == 0:
        return None
    k = np.arange(1, len(ys) + 1)
    rec = tp / npos
    prec = tp / k
    ok = rec >= recall_floor
    if ok.any():
        i = int(np.argmax(np.where(ok, prec, -1.0)))
    else:
        i = len(ys) - 1
    theta = float(p[order][i])
    return {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
            "theta": theta, "n_dispatch": int(k[i]), "n_pos": npos}


def fit_mlp(X, y, hidden=16, epochs=300, lr=0.1, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xn = (X - mu) / sd
    d = X.shape[1]
    W1 = rng.normal(0, 0.5, (hidden, d))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.5, hidden)
    b2 = 0.0
    w_pos = max(1.0, (y == 0).sum() / max(1, (y == 1).sum()))
    sw = np.where(y == 1, w_pos, 1.0)
    sw = sw / sw.mean()
    for _ in range(epochs):
        H = np.tanh(Xn @ W1.T + b1)
        z = np.clip(H @ W2 + b2, -30, 30)
        p = 1 / (1 + np.exp(-z))
        gz = (p - y) * sw / len(y)
        gW2 = H.T @ gz
        gb2 = gz.sum()
        gH = np.outer(gz, W2) * (1 - H ** 2)
        gW1 = gH.T @ Xn
        gb1 = gH.sum(0)
        W2 -= lr * gW2; b2 -= lr * gb2; W1 -= lr * gW1; b1 -= lr * gb1
    return {"arch": "mlp", "W1": W1.tolist(), "b1": b1.tolist(),
            "W2": W2.tolist(), "b2": float(b2),
            "mean": mu.tolist(), "std": sd.tolist()}


def predict_mlp(m, X):
    import numpy as np
    Xn = (X - np.asarray(m["mean"])) / np.asarray(m["std"])
    H = np.tanh(Xn @ np.asarray(m["W1"]).T + np.asarray(m["b1"]))
    z = np.clip(H @ np.asarray(m["W2"]) + m["b2"], -30, 30)
    return 1 / (1 + np.exp(-z))


def probe(events_path, out_path):
    import numpy as np
    X, y, g, feats = load_events(events_path)
    folds = np.asarray([group_fold(x) for x in g])
    oof = np.zeros(len(y))
    for k in range(5):
        tr = folds != k
        m = fit_lr(X[tr], y[tr])
        oof[folds == k] = predict_lr(m, X[folds == k])
    a = auc(y, oof)
    par = precision_at_recall(y, oof)
    report = {"n": len(y), "base_rate": round(float(y.mean()), 4),
              "oof_auc": round(a, 4) if a else None,
              "precision_at_recall85": par,
              "k2_bar": K2_PREC_BAR,
              "k2_pass": bool(par and par["precision"] >= K2_PREC_BAR),
              "feats": feats}
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def train(events_path, out_path):
    import numpy as np
    X, y, g, feats = load_events(events_path)
    folds = np.asarray([group_fold(x) for x in g])
    tr, va = folds != VAL_FOLD, folds == VAL_FOLD
    w, b, mu, sd = fit_lr(X[tr], y[tr])
    p_va = predict_lr((w, b, mu, sd), X[va])
    par = precision_at_recall(y[va], p_va)
    mlp = fit_mlp(X[tr], y[tr])
    p_mlp = predict_mlp(mlp, X[va])
    par_mlp = precision_at_recall(y[va], p_mlp)
    a_lr, a_mlp = auc(y[va], p_va), auc(y[va], p_mlp)
    print(f"val AUC lr={a_lr:.4f} mlp={a_mlp:.4f}; "
          f"prec@rec85 lr={par['precision']} mlp={par_mlp['precision']}")
    model = {"feats": feats, "w": w.tolist(), "b": float(b),
             "mean": mu.tolist(), "std": sd.tolist(), "theta": par["theta"],
             "specgate": {"design": "docs/w5_specgate_design.md",
                          "rule": f"argmax precision s.t. recall>={RECALL_FLOOR} on HumDial val",
                          "val_auc": round(a_lr, 4), "val_point": par,
                          "mlp_ablation": {"val_auc": round(a_mlp, 4), "val_point": par_mlp},
                          "n_train": int(tr.sum()), "n_val": int(va.sum())}}
    Path(out_path).write_text(json.dumps(model))
    print(f"wrote {out_path} (theta={par['theta']:.6f})")
    return 0


def selftest():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 4000
    # synthetic: gap-driven world — one informative feature (index 1 = gap1)
    x_inf = rng.normal(0, 1, n)
    X = rng.normal(0, 1, (n, len(FEATS_SG)))
    X[:, 1] = x_inf
    y = (x_inf + rng.normal(0, 0.5, n) > 0.4).astype(int)
    g = [f"g{i % 50}" for i in range(n)]
    folds = np.asarray([group_fold(x) for x in g])
    m = fit_lr(X[folds != 4], y[folds != 4])
    p = predict_lr(m, X[folds == 4])
    a = auc(y[folds == 4], p)
    par = precision_at_recall(y[folds == 4], p)
    ck = {"auc_learns": a is not None and a > 0.85,
          "recall_floor_respected": par["recall"] >= RECALL_FLOOR,
          "precision_sane": 0 < par["precision"] <= 1,
          "mlp_runs": auc(y[folds == 4],
                          predict_mlp(fit_mlp(X[folds != 4], y[folds != 4]),
                                      X[folds == 4])) > 0.8,
          "degenerate_no_pos": precision_at_recall(np.zeros(5, int),
                                                   np.ones(5) * 0.5) is None}
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", default="exp/w5sg/events.jsonl")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.probe:
        return probe(args.events, args.out or "exp/w5sg/probe_report.json")
    return train(args.events, args.out or "exp/w5sg/specgate_v0.json")


if __name__ == "__main__":
    sys.exit(main())
