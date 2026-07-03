#!/usr/bin/env python3
"""
w2r_delta_hist.py — W2 D2: empirical Δt histogram for the rollback subset (REAL alignment).

Δt = (correction-cue speech onset) − (end of the utterance segment stating the initial intent)

Method (no heuristics about "50-60% of audio"):
  1. silero VAD → speech segments of input.wav (the ruler; this is what the engine sees).
  2. SenseVoice (sherpa-onnx) token timestamps → locate the FIRST correction cue
     occurring after the initial (wrong) intent value.
  3. If the cue lies in a LATER VAD segment than the intent value:
         Δt = cue_segment.start − intent_segment.end          (exposed: EoU could fire)
     If the cue lies in the SAME VAD segment:
         Δt = 0.0                                             (never exposed at EoU granularity)
  4. Emit per-scenario details for human verification + quantile table.

Output: exp/w2_rerun/delta_hist.json
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("FDBC_ASR_BACKEND", "sensevoice")

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
BENCH = Path("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json")
OUT = Path("/root/autodl-tmp/fd-badcat/exp/w2_rerun/delta_hist.json")

CUES = ["scratch that", "no wait", "actually", "wait", "i meant", "changed my mind",
        "change that", "make that", "make it", "hold on", "sorry", "instead"]


def load_16k(path):
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        import torch
        import torchaudio
        a = torchaudio.functional.resample(
            torch.from_numpy(a).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    return a


def vad_segments(audio):
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    ts = get_speech_timestamps(audio, model, sampling_rate=16000,
                               min_silence_duration_ms=400, speech_pad_ms=30)
    return [(t["start"] / 16000.0, t["end"] / 16000.0) for t in ts]


def asr_tokens(audio, asr):
    s = asr.create_stream()
    s.accept_waveform(16000, audio)
    asr.decode_stream(s)
    r = s.result
    return list(zip([t.lower() for t in r.tokens], list(r.timestamps))), r.text


def find_cue(tokens):
    """Return (cue_text, t_start) of the first correction cue in the token stream."""
    joined = ""
    starts = []          # char offset -> token index
    for tok, ts in tokens:
        starts.append((len(joined), ts))
        joined += tok
    for m in re.finditer("|".join(re.escape(c.replace(" ", "")) for c in CUES),
                         joined.replace(" ", "")):
        # map char position in space-stripped text back to token timestamp
        pos, stripped = m.start(), 0
        for (off, ts), (tok, _) in zip(starts, tokens):
            stripped_tok = tok.replace(" ", "")
            if stripped + len(stripped_tok) > pos:
                return m.group(0), ts
            stripped += len(stripped_tok)
    return None, None


def main():
    from module import _get_asr_model
    asr = _get_asr_model()
    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench
    rollback = [x for x in items if x.get("state_rollback_test")]

    rows = []
    for sc in rollback:
        folders = sorted(glob.glob(str(DATA / f"{sc['id']}_*")))
        if not folders:
            rows.append({"id": sc["id"], "status": "no_audio"})
            continue
        for folder in folders:
            audio = load_16k(Path(folder) / "input.wav")
            segs = vad_segments(audio)
            toks, text = asr_tokens(audio, asr)
            cue, cue_t = find_cue(toks)
            if cue_t is None:
                rows.append({"id": sc["id"], "folder": Path(folder).name,
                             "status": "cue_not_found", "asr_text": text[:200]})
                continue
            # segment containing the cue vs segment before it
            cue_seg = next((i for i, (s, e) in enumerate(segs) if s <= cue_t <= e), None)
            prev_end = None
            if cue_seg is not None and cue_seg > 0:
                prev_end = segs[cue_seg - 1][1]
            exposed = (cue_seg is not None and cue_seg > 0
                       and abs(segs[cue_seg][0] - cue_t) < 1.0)
            # Δt: gap between previous segment end and cue segment start when the cue
            # opens a new segment; else 0 (same-segment correction: never exposed at EoU)
            if exposed and prev_end is not None:
                dt = round(segs[cue_seg][0] - prev_end, 3)
            else:
                dt = 0.0
            rows.append({"id": sc["id"], "folder": Path(folder).name, "status": "ok",
                         "cue": cue, "cue_t": round(cue_t, 2),
                         "n_segs": len(segs),
                         "segs": [[round(s, 2), round(e, 2)] for s, e in segs],
                         "cue_seg": cue_seg, "exposed_at_eou": bool(exposed),
                         "delta_t": dt, "asr_text": text[:250]})

    ok = [r for r in rows if r.get("status") == "ok"]
    dts = sorted(r["delta_t"] for r in ok)
    exposed_dts = sorted(r["delta_t"] for r in ok if r["exposed_at_eou"])
    q = (lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None)
    summary = {
        "n_rollback_bench": len(rollback),
        "n_with_audio": len({r['id'] for r in rows if r.get('status') != 'no_audio'}),
        "n_ok": len(ok),
        "n_exposed_at_eou": len(exposed_dts),
        "delta_t_all": dts,
        "delta_t_exposed": exposed_dts,
        "quantiles_exposed": {p: q(exposed_dts, x) for p, x in
                              [("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90)]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if r.get("status") == "ok":
            print(f"{r['id']:14s} cue={r['cue']:12s} cue_t={r['cue_t']:6.2f} "
                  f"exposed={str(r['exposed_at_eou']):5s} dt={r['delta_t']:5.2f} segs={r['n_segs']}")
        else:
            print(f"{r['id']:14s} {r['status']}")


if __name__ == "__main__":
    main()
