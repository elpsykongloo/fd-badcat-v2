#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""One-shot VAD event extraction (the only torch-loading step of the mock pipeline).

Replicates BOTH engines' perception exactly: 256-sample frames, 2-frame buffer,
silero VADIterator(return_seconds=True). Emits {frame_seq: event} JSON per wav so
equivalence/freeze runs can use a scripted VAD in a ~150MB process
(the no-GPU dev container has a 2GB cgroup limit — see AGENTS.md).
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

CHUNK = 256
SR = 16000


def _pre_pressure(mb=400, step=25):
    """Gently force the kernel to reclaim page cache inside the 2GB cgroup
    BEFORE torch's allocation burst (which otherwise races reclaim and gets
    OOM-killed). Gradual touch -> direct reclaim -> free -> headroom."""
    blocks = []
    try:
        for _ in range(mb // step):
            b = bytearray(step * 1024 * 1024)
            for i in range(0, len(b), 4096):
                b[i] = 1
            blocks.append(b)
    except MemoryError:
        pass
    finally:
        blocks.clear()
        import gc
        gc.collect()


def extract(wav_path, model, VADIterator, torch):
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    assert sr == SR, f"{wav_path}: {sr}"
    it = VADIterator(model, sampling_rate=SR)
    buf = np.zeros(0, dtype=np.float32)
    events = {}
    seq = 0
    for i in range(0, len(data), CHUNK):
        pcm = data[i:i + CHUNK]
        if len(pcm) < CHUNK:
            pcm = np.pad(pcm, (0, CHUNK - len(pcm)))
        seq += 1
        buf = np.concatenate([buf, pcm])
        if len(buf) >= 2 * CHUNK:
            ev = it(torch.from_numpy(buf[:2 * CHUNK]), return_seconds=True)
            buf = np.zeros(0, dtype=np.float32)
            if ev:
                events[seq] = ev
    return events, seq


def main():
    out_dir = Path("traces/vad_scripts")
    out_dir.mkdir(parents=True, exist_ok=True)
    _pre_pressure()
    import torch
    from silero_vad import load_silero_vad, VADIterator
    model = load_silero_vad()
    for arg in sys.argv[1:]:
        wav = Path(arg)
        events, n = extract(wav, model, VADIterator, torch)
        out = out_dir / f"{wav.stem}.json"
        out.write_text(json.dumps({"wav": str(wav), "n_frames": n,
                                   "events": {str(k): v for k, v in events.items()}},
                                  ensure_ascii=False, indent=1))
        kinds = [("start" if "start" in e else "end") for e in events.values()]
        print(f"{wav.name}: {n} frames, {kinds.count('start')} starts / {kinds.count('end')} ends -> {out}")


if __name__ == "__main__":
    main()
