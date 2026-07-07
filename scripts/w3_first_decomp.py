#!/usr/bin/env python3
"""
w3_first_decomp.py — first-response decomposition (06 裁断 B 回填, zero GPU).

Input: the archived W2 serial-live full-100 results (result_w2r_tact_full.json,
infer_mode=live, delta=1.5) + the blocking arm (result_w2r_sblock_full.json).

Architecture note: in Phase-B there is no separate judge — judge+Phase-B are ONE
fused transactional decision per EoU. On the audio clock the first-response anchor
decomposes EXACTLY as
    first = HOLD(0.64) + infer(final decision)          [ack path]
because the final EoU fires at last_seg_end + HOLD and t_user_end = last_seg_end.
Ack synthesis is excluded by the text-ready anchor on both arms (W2 design); the
measured ack-v0 audio cost is a separate known number (0.429s vs 0.933s full).

Outputs:
  - component stats: hold (const), final-decision infer distribution, residual
    (should be ~0 on the ack path; fallback-path examples listed)
  - speculative-dispatch projection (judge fires at vad_end instead of hold expiry):
    first' = max(HOLD, infer_final) per example  ->  p50/p90, and the P3 ratio
    first'_p50 / blocking_first_p50 (criterion: <= 50%)

Usage: python scripts/w3_first_decomp.py [--provider w2r_tact_full]
       [--blocking w2r_sblock_full] [--out exp/w3/first_decomp.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")
HOLD = 0.64


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(p * len(v)))], 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="w2r_tact_full")
    ap.add_argument("--blocking", default="w2r_sblock_full")
    ap.add_argument("--out", default="/root/autodl-tmp/fd-badcat/exp/w3/first_decomp.json")
    args = ap.parse_args()

    rows, fallback, no_first = [], [], []
    blocking_firsts = []
    for folder in sorted(DATA.iterdir()):
        if not (folder.is_dir() and _FOLDER_RE.match(folder.name)):
            continue
        rp = folder / f"result_{args.provider}.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            lat = r.get("latency", {})
            first = lat.get("first_response_s")
            decs = (r.get("trace") or {}).get("decisions") or []
            infer_final = decs[-1]["infer_s"] if decs else None
            if first is None or infer_final is None:
                no_first.append(folder.name)
                continue
            row = {"id": folder.name, "first": first, "ack": lat.get("ack_emitted"),
                   "infer_final": infer_final, "n_eou": lat.get("n_eou"),
                   "residual": round(first - HOLD - infer_final, 3),
                   "spec_first": round(max(HOLD, infer_final), 3)}
            if not lat.get("ack_emitted"):
                fallback.append(row)   # first anchored on result-ready, not say
            rows.append(row)
        bp = folder / f"result_{args.blocking}.json"
        if bp.exists():
            b = json.loads(bp.read_text())
            bf = (b.get("latency") or {}).get("first_response_s")
            if bf is not None:
                blocking_firsts.append(bf)

    ack_rows = [r for r in rows if r["ack"]]
    firsts = [r["first"] for r in ack_rows]
    infers = [r["infer_final"] for r in ack_rows]
    residuals = [abs(r["residual"]) for r in ack_rows]
    spec = [r["spec_first"] for r in ack_rows]
    b50, b90 = pct(blocking_firsts, .5), pct(blocking_firsts, .9)

    summary = {
        "n": len(rows), "n_ack_path": len(ack_rows),
        "n_fallback_path": len(fallback), "n_no_first": len(no_first),
        "hold_s": HOLD,
        "first_p50": pct(firsts, .5), "first_p90": pct(firsts, .9),
        "infer_final_p50": pct(infers, .5), "infer_final_p90": pct(infers, .9),
        "identity_check_max_abs_residual": max(residuals) if residuals else None,
        "decomposition": "first = 0.64 hold + infer_final (fused judge+PhaseB); "
                         "ack synthesis excluded by text anchor (ack-v0: 0.429s)",
        "speculative_projection_p50": pct(spec, .5),
        "speculative_projection_p90": pct(spec, .9),
        "n_infer_gt_hold": sum(1 for x in infers if x > HOLD),
        "blocking_first_p50": b50, "blocking_first_p90": b90,
        "P3_ratio_now": (round(pct(firsts, .5) / b50, 3)
                         if firsts and b50 else None),
        "P3_ratio_speculative": (round(pct(spec, .5) / b50, 3)
                                 if spec and b50 else None),
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if fallback:
        print("\nfallback-path examples (no ack say; first = result-ready):")
        for r in fallback:
            print(f"  {r['id']}: first={r['first']} infer_final={r['infer_final']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows,
                               "fallback": fallback}, indent=1, ensure_ascii=False))
    print("report ->", out)


if __name__ == "__main__":
    main()
