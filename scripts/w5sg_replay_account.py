#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w5sg_replay_account.py — W5-SG Phase-2 FDB single-shot accounting
(docs/w5_specgate_design.md §5; zero GPU — pure replay arithmetic).

Inputs: per-clip VAD event files (scripts/extract_vad_events.py format:
{seq: {"start": s}|{"end": s}}, seconds) + the trained gate JSON + an
infer-time source. For every vad-end the gate decides dispatch; the recorded
timeline decides confirmation (gap >= 0.64). Frozen accounting (§2):

  waste  = voided dispatches / dispatches          (gate G1: <= 0.35)
  missed = gated-out confirmed EoUs / confirmed    (gate G2a: <= 0.15)
  first  = max(0.64, infer) if dispatched else 0.64 + infer   per confirmed EoU
           (G2b: p50 == 0.640; G2c: p90 <= always-dispatch p90 + 0.15)

FIREWALL: theta comes frozen from HumDial val. This script is SINGLE-SHOT on
FDB — rerunning with a retuned theta violates the preregistration (K3).

  $PY scripts/w5sg_replay_account.py --events-dir traces/vad_scripts \\
      --gate exp/w5sg/specgate_v0.json --infer-p50 0.561
  $PY scripts/w5sg_replay_account.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from specgate import HOLD_S, SpecGate, events_to_rows  # noqa: E402

G1_WASTE = 0.35
G2A_MISSED = 0.15
G2B_FIRST_P50 = 0.640
G2C_P90_SLACK = 0.15


def segments_from_event_file(path):
    ev = json.loads(Path(path).read_text())
    items = sorted(((int(k), v) for k, v in ev.items()), key=lambda x: x[0])
    segs, cur = [], None
    for _seq, e in items:
        if "start" in e and cur is None:
            cur = float(e["start"])
        elif "end" in e and cur is not None:
            segs.append((cur, float(e["end"])))
            cur = None
    return segs


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[k]


def account(streams, gate, infer_of):
    """streams: {clip_id: [(t_start, t_end)]}; infer_of(clip_id, eou_idx)->s.
    Returns the accounting dict for gate arm + always-dispatch baseline."""
    n_disp = n_void = n_conf = n_missed = 0
    first_gate, first_base = [], []
    n_disp_base = n_void_base = 0
    for clip, segs in sorted(streams.items()):
        # tail vad-end of a clip: no further speech recorded => confirmed
        for i, (feats, y, _gap) in enumerate(events_to_rows(segs)):
            infer = infer_of(clip, i)
            allow = gate.allow(feats)
            n_disp_base += 1
            if not y:
                n_void_base += 1
            if allow:
                n_disp += 1
                if y:
                    n_conf += 1
                    first_gate.append(max(HOLD_S, infer))
                else:
                    n_void += 1
            elif y:
                n_conf += 1
                n_missed += 1
                first_gate.append(HOLD_S + infer)
            if y:
                first_base.append(max(HOLD_S, infer))
    waste = n_void / n_disp if n_disp else 0.0
    waste_base = n_void_base / n_disp_base if n_disp_base else 0.0
    missed = n_missed / n_conf if n_conf else 0.0
    p50, p90 = pctl(first_gate, 0.5), pctl(first_gate, 0.9)
    p90_base = pctl(first_base, 0.9)
    out = {
        "dispatched": n_disp, "voided": n_void, "confirmed": n_conf,
        "missed": n_missed,
        "waste": round(waste, 4), "missed_rate": round(missed, 4),
        "first_p50": round(p50, 3) if p50 is not None else None,
        "first_p90": round(p90, 3) if p90 is not None else None,
        "baseline_always": {"dispatched": n_disp_base, "voided": n_void_base,
                            "waste": round(waste_base, 4),
                            "first_p50": round(pctl(first_base, 0.5) or 0, 3),
                            "first_p90": round(p90_base or 0, 3)},
        "gates": {
            "G1_waste<=0.35": waste <= G1_WASTE,
            "G2a_missed<=0.15": missed <= G2A_MISSED,
            "G2b_first_p50==0.640": p50 is not None and abs(p50 - G2B_FIRST_P50) < 1e-9,
            "G2c_p90_bound": (p90 is not None and p90_base is not None
                              and p90 <= p90_base + G2C_P90_SLACK),
        },
    }
    out["gates"]["AND"] = all(out["gates"].values())
    return out


def selftest():
    gate = SpecGate({"feats": ["utt_dur", "gap1", "gap2", "gap3", "n_segs_10s",
                               "speech_ratio_5s"],
                     "mean": [0] * 6, "std": [1] * 6,
                     "w": [0, 8.0, 0, 0, 0, 0], "b": -4.0, "theta": 0.5})
    # stream: gap1 features carry history — second vad-end sees gap1=0.3 (void
    # history) etc. Construct three clips with known confirm pattern.
    streams = {
        "a": [(0.0, 2.0), (2.3, 3.0), (4.5, 6.0)],   # gaps 0.3(void), 1.5(conf), tail(conf)
        "b": [(0.0, 1.0), (1.2, 2.0)],               # gap 0.2(void), tail(conf)
    }
    out = account(streams, gate, lambda c, i: 0.5)
    ck = {}
    ck["confirm_count"] = out["confirmed"] == 3
    ck["baseline_voids"] = out["baseline_always"]["voided"] == 2
    # gate uses gap1 (PREVIOUS gap) — first vad-end of each clip has gap1=0 ->
    # prob<theta -> not dispatched; those are the void ones in clip a/b => waste 0
    ck["waste_zero_here"] = out["waste"] == 0.0
    ck["first_floor_when_dispatched"] = out["baseline_always"]["first_p50"] == 0.64
    ck["missed_counts"] = 0 <= out["missed_rate"] <= 1
    ck["gates_keys"] = set(out["gates"]) == {"G1_waste<=0.35", "G2a_missed<=0.15",
                                             "G2b_first_p50==0.640", "G2c_p90_bound",
                                             "AND"}
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events-dir", help="dir of per-clip VAD event JSONs")
    ap.add_argument("--gate", default="exp/w5sg/specgate_v0.json")
    ap.add_argument("--infer-p50", type=float, default=0.561,
                    help="constant infer fallback (v3.1 live p50)")
    ap.add_argument("--infer-json",
                    help="optional {clip: [infer_s per EoU]} for exact replay")
    ap.add_argument("--out", default="exp/w5sg/fdb_account.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    gate = SpecGate.load(args.gate)
    streams = {}
    for p in sorted(Path(args.events_dir).glob("*.json")):
        segs = segments_from_event_file(p)
        if segs:
            streams[p.stem] = segs
    infer_map = json.loads(Path(args.infer_json).read_text()) if args.infer_json else {}

    def infer_of(clip, i):
        xs = infer_map.get(clip)
        if xs and i < len(xs):
            return float(xs[i])
        return args.infer_p50

    out = account(streams, gate, infer_of)
    out["theta"] = gate.theta
    out["n_clips"] = len(streams)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
