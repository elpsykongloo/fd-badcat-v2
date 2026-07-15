#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w5sg_asr_features.py — SG v1 Phase-1 (design §12): rebuild the vad-end event
table with F_time + F_text_asr (ASR-prefix structural features).

Same engine-caliber VAD scan as w5sg_census; additionally runs SenseVoice
(sherpa-onnx offline) on every speech segment and, at each vad-end k, extracts
NUMERIC structural features of the utterance-so-far transcript
(segments 0..k): n_chars, n_tokens + w4v3 t1_features (trailing-token lexicon
classes — same implementation as the w4v3 probe's near-saturating family).

COMPLIANCE (§12.3): raw transcripts are cached OUT OF REPO
(--cache-dir, default /root/autodl-tmp/w5sg_asr_cache); in-repo outputs
(events_v1.jsonl + events_v1_meta.json) are numeric-only (assert_no_text).

  $PY scripts/w5sg_asr_features.py --root /root/autodl-tmp/HumDial_train \\
      --model-dir /root/autodl-tmp/models/sensevoice --workers 32
      # dir with model.int8.onnx (preferred) or model.onnx + tokens.txt
  $PY scripts/w5sg_asr_features.py --selftest
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from specgate import FEATS_SG, events_to_rows  # noqa: E402
from w4v3_common import t1_features  # noqa: E402

SR = 16000


T1_KEYS = ("ends_filler", "ends_conn", "ends_prep", "ends_det_or_part",
           "ends_aux", "ends_num", "single_token")
F_TEXT_NAMES = ["n_chars", "n_tokens"] + [f"t1_{k}" for k in T1_KEYS]

_WORKER_DEPS = None


def text_feats(prefix_text, lang):
    toks = prefix_text.split() if lang == "en" else list(prefix_text)
    t1 = t1_features(prefix_text, lang)
    return [float(len(prefix_text)), float(len(toks))] + [
        float(t1[k]) for k in T1_KEYS]


def resolve_model_files(model_dir):
    """Resolve the same SenseVoice asset variants accepted by src/module.py."""
    model_dir = Path(model_dir)
    model_path = model_dir / "model.int8.onnx"
    if not model_path.exists():
        model_path = model_dir / "model.onnx"
    tokens_path = model_dir / "tokens.txt"
    missing = [str(p) for p in (model_path, tokens_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "SenseVoice model missing (need model.int8.onnx or model.onnx "
            f"plus tokens.txt). Missing: {', '.join(missing)}")
    return model_path, tokens_path


def make_sherpa_asr(model_dir, threads=2):
    import sherpa_onnx
    import soundfile as sf
    model_path, tokens_path = resolve_model_files(model_dir)
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(model_path),
        tokens=str(tokens_path),
        num_threads=threads, use_itn=True)

    # A sample commonly has many VAD segments. Retaining the most recently
    # opened waveform avoids decoding that same file once per segment.
    wav_cache = {"path": None, "audio": None, "sr": None}

    def asr(wav_path, s, e):
        wav_path = str(wav_path)
        if wav_cache["path"] != wav_path:
            a, sr = sf.read(wav_path, dtype="float32")
            if a.ndim == 2:
                a = a.mean(axis=1)
            wav_cache.update(path=wav_path, audio=a, sr=sr)
        a, sr = wav_cache["audio"], wav_cache["sr"]
        seg = a[int(s * sr): int(e * sr)]
        st = rec.create_stream()
        st.accept_waveform(sr, seg)
        rec.decode_stream(st)
        return st.result.text.strip()
    return asr


class SegASR:
    """Per-segment ASR with an out-of-repo JSON cache (key = wav:seg times)."""

    def __init__(self, asr_fn, cache_dir):
        self.asr = asr_fn
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, wav, s, e):
        k = hashlib.sha256(f"{wav}:{s:.3f}:{e:.3f}".encode()).hexdigest()[:24]
        p = self.dir / f"{k}.json"
        if p.exists():
            try:
                return json.loads(p.read_text())["text"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                # An interrupted pre-atomic run may have left a partial entry.
                p.unlink(missing_ok=True)
        t = self.asr(wav, s, e)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"text": t}, ensure_ascii=False))
        os.replace(tmp, p)
        return t


def _process_sample(task, deps):
    """VAD + ASR one sample; only numeric rows leave the worker."""
    from w5sg_census import vad_segments

    seq, wav, lang, group = task
    model, VADIterator, torch, sf, np, seg_asr = deps
    try:
        segs = vad_segments(Path(wav), model, VADIterator, torch, sf, np)
    except Exception as exc:
        return seq, wav, [], f"{type(exc).__name__}: {exc}", []

    prefix = ""
    records, asr_errors = [], []
    for k, ((s, e), (feats, y, gap)) in enumerate(
            zip(segs, events_to_rows(segs))):
        try:
            seg_text = seg_asr.get(wav, s, e)
        except Exception as exc:
            asr_errors.append((k, f"{type(exc).__name__}: {exc}"))
            seg_text = ""
        prefix = (prefix + " " + seg_text).strip() if lang == "en" \
            else prefix + seg_text
        records.append({"f": feats + text_feats(prefix, lang), "y": y,
                        "gap": gap, "lang": lang, "g": group})
    return seq, wav, records, None, asr_errors


def _init_worker(model_dir, cache_dir):
    """Load independent VAD/ASR instances per process (thread-safe scaling)."""
    global _WORKER_DEPS
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import load_silero_vad, VADIterator

    torch.set_num_threads(1)
    _WORKER_DEPS = (load_silero_vad(), VADIterator, torch, sf, np,
                    SegASR(make_sherpa_asr(model_dir, threads=1), cache_dir))


def _process_sample_worker(task):
    if _WORKER_DEPS is None:
        raise RuntimeError("ASR feature worker was not initialized")
    return _process_sample(task, _WORKER_DEPS)


def build(root, out_dir, model_dir, cache_dir, limit=None, workers=1):
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import load_silero_vad, VADIterator
    from w4v3_common import resolve_root, iter_train_samples, assert_no_text
    from w5sg_census import sample_language

    root = resolve_root(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for smp in iter_train_samples(root):
        if limit and len(tasks) >= limit:
            break
        wav = smp.get("wav")
        if not wav or not Path(wav).exists():
            continue
        lang = sample_language(smp)
        group = hashlib.sha256(str(smp.get("key", wav)).encode()).hexdigest()[:12]
        tasks.append((len(tasks), str(wav), lang, group))

    workers = max(1, int(workers))
    if workers == 1:
        torch.set_num_threads(2)
        deps = (load_silero_vad(), VADIterator, torch, sf, np,
                SegASR(make_sherpa_asr(model_dir), cache_dir))
        results = (_process_sample(task, deps) for task in tasks)
        pool = None
    else:
        # spawn avoids inheriting unsafe torch/onnx runtime state. pool.map
        # preserves loader order, so output is byte-stable across worker counts.
        ctx = multiprocessing.get_context("spawn")
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_init_worker,
            initargs=(str(model_dir), str(cache_dir)))
        results = pool.map(_process_sample_worker, tasks, chunksize=1)

    n_ev = 0
    rows_f = (out_dir / "events_v1.jsonl").open("w")
    try:
        for done, (_seq, wav, records, error, asr_errors) in enumerate(results, 1):
            if error is not None:
                print(f"skip {wav}: {error}")
                continue
            for k, error in asr_errors:
                print(f"asr fail {wav}[{k}]: {error}")
            for rec in records:
                assert_no_text(rec)
                rows_f.write(json.dumps(rec) + "\n")
                n_ev += 1
            if done % 200 == 0:
                print(f"  {done} samples / {n_ev} events")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        rows_f.close()

    feat_names = list(FEATS_SG) + F_TEXT_NAMES
    meta = {"feats": feat_names, "n_events": n_ev, "n_samples": len(tasks),
            "design": "w5_specgate_design.md §12", "cache_dir": str(cache_dir)}
    (out_dir / "events_v1_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "feats"}, indent=2))
    return 0


def selftest():
    import tempfile
    ck = {}
    tf = text_feats("帮我订一张去杭州的", "zh")
    tf2 = text_feats("帮我订一张去杭州的票。", "zh")
    ck["numeric_only"] = all(isinstance(x, float) for x in tf)
    ck["length_grows"] = tf2[0] > tf[0]
    ck["t1_present"] = len(tf) > 2
    en = text_feats("book me a train to", "en")
    ck["en_tokens"] = en[1] == 5.0
    calls = []
    fake = lambda w, s, e: (calls.append((w, s, e)) or f"seg{len(calls)}")  # noqa: E731
    sa = SegASR(fake, Path(tempfile.mkdtemp()) / "cache")
    a1 = sa.get("w.wav", 0.0, 1.0)
    a2 = sa.get("w.wav", 0.0, 1.0)
    ck["cache_hit"] = a1 == a2 and len(calls) == 1
    model_dir = Path(tempfile.mkdtemp())
    (model_dir / "model.onnx").touch()
    (model_dir / "model.int8.onnx").touch()
    (model_dir / "tokens.txt").touch()
    model_path, tokens_path = resolve_model_files(model_dir)
    ck["int8_preferred"] = (model_path.name == "model.int8.onnx" and
                             tokens_path.name == "tokens.txt")
    (model_dir / "model.int8.onnx").unlink()
    ck["full_fallback"] = resolve_model_files(model_dir)[0].name == "model.onnx"
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/HumDial_train")
    ap.add_argument("--out", default="exp/w5sg")
    ap.add_argument("--model-dir", help="SenseVoice dir (model.int8.onnx or "
                                        "model.onnx + tokens.txt)")
    ap.add_argument("--cache-dir", default="/root/autodl-tmp/w5sg_asr_cache")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1,
                    help="sample processes (default 1; use 32 on a large CPU host)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.model_dir:
        ap.error("--model-dir required (SenseVoice model.int8.onnx or "
                 "model.onnx + tokens.txt)")
    return build(args.root, Path(args.out), args.model_dir, args.cache_dir,
                 args.limit, args.workers)


if __name__ == "__main__":
    sys.exit(main())
