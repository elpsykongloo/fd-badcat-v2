#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w4v3_omni_readout.py — Omni zero-shot readouts over HumDial cut points
(the T3/X feature family of the probe; docs/w4v3_design.md §5).

Rationale: the biggest model in the system (Qwen3-Omni 30B) is already deployed
as the finality judge (rung 3; 84% recovery came from its labels). This script
reuses that exact wire format to label HumDial census cut points, producing
external features the text probe consumes via --extra-jsonl. Needs the vLLM
stack (:10004 proxy) — GPU day only.

Modes (comma list, --modes):
  finality  audio tail (<= FINALITY_TAIL_S) up to the cut -> frozen
            FINALITY_PROMPT -> final/hesitant/unfinished (identical call shape
            to the rung-3 arm: build_finality_msgs + _audio_block)
  amend     transcript prefix up to the cut -> frozen AMEND_PROMPT_V0 ->
            continue/done (text-only; probes SEMANTIC continuation signal)

Output jsonl lines: {"key": "<sample>#prefix|#full", "feats": {...}} — keys
already match the probe's instance keys. Labels only; NO transcript text and
NO audio reach the output (compliance red line, w4v3_design.md §3).

Run (server, vLLM text/audio stack up):
  $PY scripts/w4v3_omni_readout.py --cuts exp/w4v3/humdial_cuts.jsonl \
      --root /root/autodl-tmp/HumDial_train --modes finality,amend \
      --out exp/w4v3/omni_readout.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
from w4v3_common import (BREAK, SR, assert_no_text, resolve_root,  # noqa: E402
                         read_textgrid_segments, strip_break)

# Frozen one-word amendment prompt (v0, preregistered w4v3_design.md §5; the
# text-side sibling of delta_policy.FINALITY_PROMPT).
AMEND_PROMPT_V0 = (
    "You will read the transcript of one user turn from a task-oriented phone "
    "call, cut at a moment of silence. Judge whether the user is likely to say "
    "more within the next few seconds - continuing the sentence, adding a "
    "constraint, or correcting themselves - or whether the request is complete "
    "as stated. Answer with EXACTLY one word:\n"
    "continue - more speech from this user is likely imminent.\n"
    "done - the request sounds complete."
)
_AMEND_RE = re.compile(r"\b(continue|done)\b", re.I)


def _llm_call(messages):
    """Same payload envelope as the replay harness (w2r_stream_replay._llm_call:
    T=0/seed=42, modalities text) so readouts share the decider's regime."""
    import requests
    payload = {
        "model": os.getenv("FDBC_QWEN_MODEL", "Qwen3-Omni-30B-A3B-Instruct"),
        "temperature": float(os.getenv("FDBC_QWEN_TEMPERATURE", "0")),
        "top_p": float(os.getenv("FDBC_QWEN_TOP_P", "0.7")),
        "top_k": int(os.getenv("FDBC_QWEN_TOP_K", "40")),
        "presence_penalty": float(os.getenv("FDBC_QWEN_PRESENCE_PENALTY", "1.2")),
        "frequency_penalty": float(os.getenv("FDBC_QWEN_FREQUENCY_PENALTY", "0.8")),
        "max_tokens": int(os.getenv("FDBC_QWEN_MAX_TOKENS", "16")),
        "seed": int(os.getenv("FDBC_QWEN_SEED", "42")),
        "modalities": ["text"],
        "messages": messages,
    }
    s = _llm_call._session
    if s is None:
        import requests as _r
        s = _r.Session()
        s.trust_env = False               # local service; bypass proxy env
        _llm_call._session = s
    r = s.post(os.getenv("FDBC_QWEN_URL",
                         "http://127.0.0.1:10004/v1/chat/completions"),
               headers={"Content-Type": "application/json"},
               data=json.dumps(payload),
               timeout=int(os.getenv("FDBC_QWEN_TIMEOUT", "300")))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


_llm_call._session = None


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.hits = self.misses = 0

    def call(self, msgs):
        k = hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
        if k in self.data:
            self.hits += 1
            return self.data[k]["raw"], self.data[k]["infer"]
        t0 = time.time()
        raw = _llm_call(msgs)
        self.misses += 1
        self.data[k] = {"raw": raw, "infer": round(time.time() - t0, 3)}
        return raw, self.data[k]["infer"]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))


def finality_feats(rec, root, cache, dry):
    import numpy as np                                     # GPU-day deps
    import soundfile as sf
    from decider_b import _audio_block                     # exact decider wire
    import delta_policy as dp
    data, sr = sf.read(str(root / rec["wav"]), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    assert sr == SR
    cut = rec["cut_t"]
    tail = data[int(max(0.0, cut - dp.FINALITY_TAIL_S) * SR):int(cut * SR)]
    if len(tail) < SR // 10:
        return None
    msgs = dp.build_finality_msgs(_audio_block(tail))
    if dry:
        print(f"[dry] finality {rec['key']}: tail {len(tail) / SR:.2f}s, "
              f"b64 {len(msgs[1]['content'][0]['audio_url']['url'])}B")
        return {}
    raw, _ = cache.call(msgs)
    label, ok = dp.parse_finality(raw)
    return {"fin_final": float(label == "final"),
            "fin_hesitant": float(label == "hesitant"),
            "fin_unfinished": float(label == "unfinished"),
            "fin_parsed": float(ok)}


def amend_feats(rec, root, cache, dry):
    segs = read_textgrid_segments(root / rec["textgrid"])
    if not segs:
        return None
    text = segs[rec.get("seg_idx", 0)]["text"]
    cut_text = (text.split(BREAK)[0] if rec["kind"] == "break_mid"
                else strip_break(text)).strip()
    if not cut_text:
        return None
    msgs = [{"role": "system", "content": AMEND_PROMPT_V0},
            {"role": "user", "content": cut_text}]
    if dry:
        print(f"[dry] amend {rec['key']}: {len(cut_text)} chars")
        return {}
    raw, _ = cache.call(msgs)
    m = _AMEND_RE.search(raw or "")
    return {"amend_continue": float(bool(m and m.group(1).lower() == "continue")),
            "amend_parsed": float(bool(m))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cuts", required=True,
                    help="census --emit-cuts jsonl (break_mid needs --vad there)")
    ap.add_argument("--root", default="/root/autodl-tmp/HumDial_train")
    ap.add_argument("--modes", default="finality,amend")
    ap.add_argument("--cache", default="exp/w4v3/omni_readout_cache.json")
    ap.add_argument("--out", default="exp/w4v3/omni_readout.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = resolve_root(args.root)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cache = Cache(args.cache)
    recs = [json.loads(x) for x in Path(args.cuts).read_text().splitlines()]
    if args.limit:
        recs = recs[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    with open(out, "w", encoding="utf-8") as fh:
        for i, rec in enumerate(recs):
            base = rec["key"].rsplit("#", 1)[0]
            probe_key = f"{base}#{'prefix' if rec['kind'] == 'break_mid' else 'full'}"
            feats = {}
            for mode in modes:
                fn = finality_feats if mode == "finality" else amend_feats
                f = fn(rec, root, cache, args.dry_run)
                if f is None:
                    feats = None
                    break
                feats.update(f)
            if feats is None:
                n_skip += 1
                continue
            if not args.dry_run:
                line = {"key": probe_key, "kind": rec["kind"], "feats": feats}
                assert_no_text(line)
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                n_ok += 1
            if args.dry_run and i >= 2:
                break
            if (i + 1) % 100 == 0:
                cache.save()
                print(f"  {i + 1}/{len(recs)} (cache {cache.hits}h/{cache.misses}m)",
                      flush=True)
    if not args.dry_run:
        cache.save()
    print(f"readout: {n_ok} labeled / {n_skip} skipped -> {out}")
    print(f"cache: {cache.hits} hits / {cache.misses} misses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
