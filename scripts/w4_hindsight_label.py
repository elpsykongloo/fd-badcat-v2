#!/usr/bin/env python3
"""
w4_hindsight_label.py — hindsight labeler + hazard sample export (rung 4).

Per op (future known): minimal safe window w* = gap_silence + eps (or 0 if no
revision ever hits it) — the closed-form layer; chains are handled by the
generator marking `upstream` revisions on the child directly (conservative
rule: the child's own gap IS its label). Full DP over compensation-coupled
schedules is a later refinement (official track has zero compensation).

Hazard samples (discrete-time survival): for each op and each t in T_GRID
while the op is alive (t < gap, or all steps if never revised):
    y(op, t) = 1  iff  gap_silence in (t, t+H]
This trains lambda_hat(t) directly — the head learns "risk now", not the
hindsight ACTION, so the hindsight-optimism bias stays out of the target.

Usage: w4_hindsight_label.py [--tag v0]
In:  exp/w4/synth/dialogues_{tag}.jsonl
Out: exp/w4/synth/hazard_{tag}.npz   (X, y, t, did, kappa_idx, gap)
     exp/w4/synth/ops_{tag}.jsonl    (op records + w_star, for c_w sweep/eval)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/src")
from stophead import featurize, FEATS, T_GRID, H, KAPPAS   # noqa: E402

EPS = 0.05   # w* = gap + EPS (just past the last revision)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    base = Path("/root/autodl-tmp/fd-badcat/exp/w4/synth")
    src = base / f"dialogues_{args.tag}.jsonl"

    X, y, ts, dids, kidx, gaps = [], [], [], [], [], []
    n_ops = n_pos = n_rescue = 0

    def emit(feat_src, gap, did, kappa):
        nonlocal n_pos
        for t in T_GRID:
            if gap is not None and t >= gap:
                break                         # op no longer alive at t
            pos = int(gap is not None and t < gap <= t + H)
            X.append(featurize(feat_src, t))
            y.append(pos)
            ts.append(t)
            dids.append(did)
            kidx.append(KAPPAS.index(kappa))
            gaps.append(-1.0 if gap is None else gap)
            n_pos += pos

    with (base / f"ops_{args.tag}.jsonl").open("w") as ops_out:
        for line in src.open():
            d = json.loads(line)
            for o in d["ops"]:
                n_ops += 1
                gap = o["gap_silence"]
                o["w_star"] = 0.0 if gap is None else round(gap + EPS, 3)
                o["did"] = d["did"]
                ops_out.write(json.dumps(o) + "\n")
                emit(o, gap, d["did"], o["kappa"])
                # post-rescue state (v1): after the patch lands the op is safe
                # (generator: <=1 revision/op) — all-negative samples with the
                # revising EoU's features teach "rescued -> commit fast".
                j = o.get("rev_eou")
                if j is not None and j < len(d.get("eous", [])):
                    e = d["eous"][j]
                    s = {**o, "eou_idx": j, "utt_dur": e["utt_dur"],
                         "gap_prev": e["gap_prev"], "finality": e["finality"],
                         "slots_missing": 0}
                    emit(s, None, d["did"], o["kappa"])
                    n_rescue += 1

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    np.savez_compressed(base / f"hazard_{args.tag}.npz",
                        X=X, y=y, t=np.asarray(ts, np.float32),
                        did=np.asarray(dids, np.int32),
                        kappa_idx=np.asarray(kidx, np.int8),
                        gap=np.asarray(gaps, np.float32),
                        feats=np.asarray(FEATS))
    print(f"ops={n_ops} rescue_states={n_rescue} hazard_samples={len(y)} "
          f"positives={n_pos} ({n_pos / max(1, len(y)):.2%}) dims={X.shape[1]}")
    print("->", base / f"hazard_{args.tag}.npz", "and", base / f"ops_{args.tag}.jsonl")


if __name__ == "__main__":
    main()
