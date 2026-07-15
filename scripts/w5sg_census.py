#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w5sg_census.py — W5-SG Phase-1 item 1+2: engine-caliber vad-end census on
HumDial train (docs/w5_specgate_design.md §3/§8).

Extracts, with the ENGINE-IDENTICAL VAD path (silero VADIterator, default
params, SR 16000, CHUNK 256, 2-chunk feed — byte-for-byte the loop of
scripts/extract_vad_events.py / src/engine.py), every vad-end event of every
HumDial train sample, and emits:

  exp/w5sg/events.jsonl    one row per vad-end: features (FEATS_SG), label
                           y = 1[gap >= 0.64], gap_next, lang, group (sample
                           hash), utt_dur — NUMERIC ONLY (assert_no_text).
  exp/w5sg/census_report.json
                           base rate, gap quantiles (per lang), ASR-lag audit
                           (projected lag = RTF * utt_dur vs the frozen 50ms
                           bar -> F_text in/out verdict), K1 label-count check.
  exp/w5sg/pause_prior.json
                           the SHARED RB timing prior (rb_design v2 §4 red
                           line: timing statistics only): inter-segment gap
                           quantiles + histogram; consumed by rb/generator.

Run (server):
  $PY scripts/w5sg_census.py --root /root/autodl-tmp/HumDial_train
  $PY scripts/w5sg_census.py --selftest      # no torch/VAD needed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from specgate import FEATS_SG, HOLD_S, events_to_rows  # noqa: E402

SR = 16000
CHUNK = 256                      # engine audio-clock unit (t = seq*256/16000)
ASR_RTF = 0.021                  # SenseVoice offline RTF (W1 acceptance)
ASR_LAG_BAR_S = 0.050            # frozen F_text availability bar (design §4)
K1_MIN_EVENTS = 20000            # frozen kill: fewer labels => cancel (design §7)


def vad_segments(wav_path, model, VADIterator, torch, sf, np):
    """Engine-caliber scan -> [(t_start, t_end)] in seconds (mono; asserts SR)."""
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    assert sr == SR, f"{wav_path}: sr={sr}"
    it = VADIterator(model, sampling_rate=SR)
    buf = np.zeros(0, dtype=np.float32)
    segs, cur = [], None
    for i in range(0, len(data), CHUNK):
        pcm = data[i:i + CHUNK]
        if len(pcm) < CHUNK:
            pcm = np.pad(pcm, (0, CHUNK - len(pcm)))
        buf = np.concatenate([buf, pcm])
        if len(buf) >= 2 * CHUNK:
            ev = it(torch.from_numpy(buf[:2 * CHUNK]), return_seconds=True)
            buf = np.zeros(0, dtype=np.float32)
            if ev:
                if "start" in ev and cur is None:
                    cur = float(ev["start"])
                elif "end" in ev and cur is not None:
                    segs.append((cur, float(ev["end"])))
                    cur = None
    if cur is not None:                      # speech ran to EOF: close at file end
        segs.append((cur, len(data) / SR))
    return segs


def quantiles(xs, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not xs:
        return {}
    xs = sorted(xs)
    out = {}
    for p in ps:
        k = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
        out[f"p{int(p * 100)}"] = round(xs[k], 4)
    return out


def census(root, out_dir, limit=None):
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import load_silero_vad, VADIterator
    from w4v3_common import resolve_root, iter_train_samples, assert_no_text

    torch.set_num_threads(2)
    model = load_silero_vad()
    root = resolve_root(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_f = (out_dir / "events.jsonl").open("w")
    n_events = n_samples = 0
    gaps = {"zh": [], "en": []}
    utt_durs, labels, lag_proj = [], [], []
    for smp in iter_train_samples(root):
        if limit and n_samples >= limit:
            break
        wav = smp.get("wav")
        if not wav or not Path(wav).exists():
            continue
        n_samples += 1
        lang = smp.get("lang", "zh")
        group = hashlib.sha256(str(smp.get("key", wav)).encode()).hexdigest()[:12]
        try:
            segs = vad_segments(wav, model, VADIterator, torch, sf, np)
        except Exception as e:                     # unreadable wav: count and skip
            print(f"skip {wav}: {e}")
            continue
        for feats, y, gap in events_to_rows(segs):
            rec = {"f": feats, "y": y, "gap": gap, "lang": lang, "g": group}
            assert_no_text(rec)
            rows_f.write(json.dumps(rec) + "\n")
            n_events += 1
            labels.append(y)
            utt_durs.append(feats[0])
            lag_proj.append(ASR_RTF * feats[0])
            if gap is not None:
                gaps[lang].append(gap)
        if n_samples % 500 == 0:
            print(f"  {n_samples} samples / {n_events} events")
    rows_f.close()

    lag_p50 = quantiles(lag_proj).get("p50", 0.0)
    all_gaps = gaps["zh"] + gaps["en"]
    report = {
        "n_samples": n_samples, "n_events": n_events,
        "k1_min_events": K1_MIN_EVENTS, "k1_pass": n_events >= K1_MIN_EVENTS,
        "base_rate_confirm": round(sum(labels) / max(1, len(labels)), 4),
        "gap_quantiles": {k: quantiles(v) for k, v in gaps.items()},
        "gap_below_hold_frac": round(
            sum(1 for x in all_gaps if x < HOLD_S) / max(1, len(all_gaps)), 4),
        "utt_dur_quantiles": quantiles(utt_durs),
        "asr_lag_audit": {"rtf": ASR_RTF, "bar_s": ASR_LAG_BAR_S,
                          "proj_lag_p50": lag_p50,
                          "f_text_admitted": lag_p50 <= ASR_LAG_BAR_S},
        "feats": list(FEATS_SG), "hold_s": HOLD_S,
        "vad": "VADIterator(defaults) SR16000 CHUNK256 feed=2xCHUNK (engine-identical)",
    }
    (out_dir / "census_report.json").write_text(json.dumps(report, indent=2))
    hist_edges = [i / 10 for i in range(0, 51)]           # 0..5.0s, 0.1 bins
    hist = [0] * (len(hist_edges) - 1)
    for x in all_gaps:
        k = min(len(hist) - 1, int(x * 10))
        hist[k] += 1
    prior = {"source": "HumDial train vad-end census (timing statistics only)",
             "gap_quantiles": quantiles(all_gaps), "hist_edges_s": hist_edges,
             "hist_counts": hist, "n": len(all_gaps)}
    (out_dir / "pause_prior.json").write_text(json.dumps(prior, indent=2))
    print(json.dumps(report, indent=2))
    return 0


def selftest():
    ck = {}
    rows = events_to_rows([(0.5, 2.0), (2.3, 3.0), (4.5, 6.0)])
    ck["rows"] = len(rows) == 3 and [r[1] for r in rows] == [0, 1, 1]
    ck["quantiles"] = quantiles([1, 2, 3, 4, 5])["p50"] == 3
    empty = quantiles([])
    ck["quantiles_empty"] = empty == {}
    # ASR-lag audit arithmetic: 4s utterance -> 84ms > 50ms bar
    ck["lag_projection"] = ASR_RTF * 4.0 > ASR_LAG_BAR_S
    ck["feats_len"] = len(rows[0][0]) == len(FEATS_SG)
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/HumDial_train")
    ap.add_argument("--out", default="exp/w5sg")
    ap.add_argument("--limit", type=int, help="debug: cap sample count")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return census(args.root, Path(args.out), args.limit)


if __name__ == "__main__":
    sys.exit(main())
