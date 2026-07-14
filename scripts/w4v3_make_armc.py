#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w4v3_make_armc.py — build the Arm-C0 prior-shift stophead files
(docs/w4v3_design.md §10; zero GPU, zero training).

C0 isolates the CALIBRATION axis: the frozen v2 head (weights, features,
two-stage policy) is reused bit-identically; only the operating threshold is
remapped by a closed-form prior shift

    risk'      = sigmoid(logit(risk) + logit(pi * Q_H) - logit(M_TRAIN))
    theta_eff  = sigmoid(logit(THETA) - logit(pi * Q_H) + logit(M_TRAIN))

so `risk >= theta_eff(pi)` is exactly `risk' >= THETA` — no harness changes,
each grid point is just a stophead JSON with a remapped theta. Constants are
FROZEN from committed artifacts (verified on load when present):

    THETA   = 0.03            economic anchor W/KILL_PEN (v2, unchanged)
    Q_H     = 731/748         P(pause <= RISK_HORIZON | self-continuation),
                              measured HumDial pauses (humdial_cuts.jsonl)
    M_TRAIN = 3307/19288      P(revision arrives in-horizon), v2 synthetic
                              training marginal (ops_v2.jsonl; = 4744 revised
                              x 69.7% rescuable, matching the gen receipt)
    pi      = deployment op-level ANY-revision rate (user-facing parameter);
              the horizon event rate is pi * Q_H

Run (server):
  $PY scripts/w4v3_make_armc.py            # writes exp/w4v3/stophead_v3c_pi*.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

THETA = 0.03
Q_H_NUM, Q_H_DEN = 731, 748
M_NUM, M_DEN = 3307, 19288
Q_H = Q_H_NUM / Q_H_DEN
M_TRAIN = M_NUM / M_DEN
PI_GRID = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30)
PI_STAR = 0.10
RISK_HORIZON = 2.5


def logit(x):
    return math.log(x / (1.0 - x))


def sig(z):
    return 1.0 / (1.0 + math.exp(-z))


def theta_eff(pi, theta=THETA, q_h=Q_H, m_train=M_TRAIN):
    return sig(logit(theta) - logit(pi * q_h) + logit(m_train))


def pi_tag(pi):
    return f"{int(round(pi * 1000)):03d}"


def verify_constants(repo_root):
    """Recompute the frozen constants from the committed artifacts when they
    are present; abort on mismatch (wrong artifact revision)."""
    cuts = repo_root / "exp/w4v3/humdial_cuts.jsonl"
    if cuts.exists():
        pauses = [json.loads(l)["pause_dur"] for l in cuts.read_text().splitlines()
                  if json.loads(l)["kind"] == "break_mid"]
        got = (sum(1 for p in pauses if p <= RISK_HORIZON), len(pauses))
        if got != (Q_H_NUM, Q_H_DEN):
            raise SystemExit(f"Q_H mismatch: artifacts give {got[0]}/{got[1]}, "
                             f"frozen {Q_H_NUM}/{Q_H_DEN}")
        print(f"Q_H verified against {cuts.name}: {got[0]}/{got[1]}")
    ops = repo_root / "exp/w4/synth/ops_v2.jsonl"
    if ops.exists():
        n = in_h = 0
        for line in ops.read_text().splitlines():
            o = json.loads(line)
            n += 1
            if o.get("revision_kind") and o.get("gap_silence") is not None \
                    and 0 < o["gap_silence"] <= RISK_HORIZON:
                in_h += 1
        if (in_h, n) != (M_NUM, M_DEN):
            raise SystemExit(f"M_TRAIN mismatch: artifacts give {in_h}/{n}, "
                             f"frozen {M_NUM}/{M_DEN}")
        print(f"M_TRAIN verified against {ops.name}: {in_h}/{n}")


def m_train_from_ops(ops_path):
    """Recompute the horizon-event training marginal from an ops file
    (the C1 branch: a re-anchored world has its own m_train)."""
    n = in_h = 0
    for line in Path(ops_path).read_text().splitlines():
        o = json.loads(line)
        n += 1
        if o.get("revision_kind") and o.get("gap_silence") is not None \
                and 0 < o["gap_silence"] <= RISK_HORIZON:
            in_h += 1
    return in_h, n


def build(stophead_path, outdir, pi_grid=PI_GRID, variant="v3c", m_frac=None):
    base = json.loads(Path(stophead_path).read_text())
    if base.get("policy") != "twostage":
        raise SystemExit("Arm C requires a twostage head")
    m_num, m_den = m_frac if m_frac else (M_NUM, M_DEN)
    m_train = m_num / m_den
    if m_frac is None and abs(base.get("theta", -1) - THETA) > 1e-9:
        raise SystemExit(f"base theta {base.get('theta')} != anchor {THETA}")
    if m_frac is not None:
        print(f"C1 variant: trainer-selected theta {base.get('theta')} ignored "
              f"(remapped from the {THETA} anchor); m_train = {m_num}/{m_den} "
              f"= {m_train:.5f}")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pi in pi_grid:
        te = theta_eff(pi, m_train=m_train)
        d = dict(base)
        d["theta"] = round(te, 6)
        d["armc"] = {"pi": pi, "q_h": [Q_H_NUM, Q_H_DEN],
                     "m_train": [m_num, m_den], "theta_base": THETA,
                     "variant": variant,
                     "formula": "sigmoid(logit(theta)-logit(pi*q_H)+logit(m_train))",
                     "source": "docs/w4v3_design.md §10.2/§11.3"}
        out = outdir / f"stophead_{variant}_pi{pi_tag(pi)}.json"
        out.write_text(json.dumps(d))
        rows.append((pi, te, out))
    print(f"\n{'pi':>6} {'pi*q_H':>8} {'theta_eff':>10}  provider / file")
    for pi, te, out in rows:
        star = "  <- pi*" if abs(pi - PI_STAR) < 1e-9 else ""
        print(f"{pi:>6.2f} {pi * Q_H:>8.4f} {te:>10.4f}  "
              f"w4{variant}_pi{pi_tag(pi)}_tact  {out}{star}")
    print(f"\nimplied deployment prior of the base head (theta_eff==theta): "
          f"pi = m_train/q_H = {m_train / Q_H:.4f}")
    return rows


def selftest():
    ck = {}
    # identity: at pi = m_train/q_H the remap is exactly the v2 anchor
    ck["identity_at_train_prior"] = abs(theta_eff(M_TRAIN / Q_H) - THETA) < 1e-12
    # monotone decreasing in pi
    tes = [theta_eff(pi) for pi in PI_GRID]
    ck["monotone_decreasing"] = all(a > b for a, b in zip(tes, tes[1:]))
    # frozen spot values (4dp, computed at freeze time)
    ck["spot_pi010"] = abs(theta_eff(0.10) - 0.0558) < 5e-4
    ck["spot_pi002"] = abs(theta_eff(0.02) - 0.2430) < 5e-4
    ck["spot_pi030"] = abs(theta_eff(0.30) - 0.0152) < 5e-4
    # roundtrip on a stub head: theta remapped, everything else untouched
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="w4v3_armc_st_"))
    stub = {"policy": "twostage", "theta": THETA, "w_protect": 1.5,
            "grace": 1.0, "risk_horizon": 2.5, "feats": ["t", "utt_dur"],
            "w": [0.1, -0.2], "b": 0.3, "mean": [0, 0], "std": [1, 1]}
    (tmp / "stophead_v2.json").write_text(json.dumps(stub))
    rows = build(tmp / "stophead_v2.json", tmp, pi_grid=(0.10,))
    got = json.loads(rows[0][2].read_text())
    ck["roundtrip_theta"] = abs(got["theta"] - theta_eff(0.10)) < 1e-6
    ck["roundtrip_frozen_rest"] = all(got[k] == stub[k] for k in stub
                                      if k != "theta")
    ck["roundtrip_provenance"] = got["armc"]["pi"] == 0.10
    # C1 path: m_train recomputed from a stub ops file; trainer theta waived;
    # variant naming
    ops = tmp / "ops_stub.jsonl"
    ops.write_text("\n".join(json.dumps(o) for o in [
        {"revision_kind": "cutoff_continuation", "gap_silence": 2.0},
        {"revision_kind": "afterthought", "gap_silence": 3.0},
        {"revision_kind": None, "gap_silence": None},
        {"revision_kind": "slot_completion", "gap_silence": 1.8}]))
    mf = m_train_from_ops(ops)
    ck["c1_m_train"] = mf == (2, 4)
    (tmp / "stophead_c1.json").write_text(json.dumps(dict(stub, theta=0.07)))
    rows = build(tmp / "stophead_c1.json", tmp, pi_grid=(0.10,),
                 variant="v3c1", m_frac=mf)
    got = json.loads(rows[0][2].read_text())
    ck["c1_variant_name"] = rows[0][2].name == "stophead_v3c1_pi100.json"
    ck["c1_theta"] = abs(got["theta"] - theta_eff(0.10, m_train=0.5)) < 1e-6
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stophead", default="exp/w4/stophead_v2.json")
    ap.add_argument("--outdir", default="exp/w4v3")
    ap.add_argument("--variant", default="v3c",
                    help="output naming: v3c (C0, default) | v3c1 (C1 branch)")
    ap.add_argument("--ops",
                    help="C1: ops jsonl of the re-anchored world; m_train is "
                         "recomputed from it (and the base-theta check is "
                         "waived — theta is always remapped from the anchor)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    repo_root = Path(__file__).resolve().parent.parent
    verify_constants(repo_root)
    m_frac = m_train_from_ops(args.ops) if args.ops else None
    build(args.stophead, args.outdir, variant=args.variant, m_frac=m_frac)
    return 0


if __name__ == "__main__":
    sys.exit(main())
