# -*- coding: utf-8 -*-
"""rb/audio.py — RB v2 TTS backends + arm-A timeline assembly
(docs/rb_design.md v2 §4).

Backends return (samples, sr) with samples = array('h') mono PCM16 @16k.
  SilenceStub  — deterministic placeholder tone (dry builds: validates the
                 assembly/manifest pipeline; NOT VAD-triggerable, engine runs
                 need a real backend).
  QwenTTSBackend — Qwen3-TTS open-source (CustomVoice, 9 preset voices).
                 >>> USER WIRES THIS: fill synthesize() with your local
                 model/endpoint call; map VOICES cv01..cv09 to the CustomVoice
                 preset names in VOICE_MAP. Everything else is ready. <<<

Assembly: pieces are placed sequentially by gap_before (exact silence-clock
control — the revision cue time IS end-of-previous-piece + gap); pieces with
`at_after_eou` are placed at (end of last sequential piece + HOLD + value) and
MIXED if they overlap (bystander injections). Emits wav + a cue table with
per-piece [t_start, t_end] — the build-time ground truth for binning."""
from __future__ import annotations

import array
import hashlib
import math
import wave
from pathlib import Path

SR = 16000
HOLD_S = 0.64


class TTSBackend:
    def synthesize(self, text, voice, lang, rate=1.0):
        raise NotImplementedError


class SilenceStub(TTSBackend):
    """Deterministic tone whose duration mimics speech length (zh ~0.18s/char,
    en ~0.075s/char) — placeholder ONLY."""

    def synthesize(self, text, voice, lang, rate=1.0):
        per = 0.18 if lang == "zh" else 0.075
        dur = max(0.4, min(12.0, per * len(text) / max(rate, 0.25)))
        n = int(dur * SR)
        f0 = 180 + (int(hashlib.sha256(voice.encode()).hexdigest()[:4], 16) % 120)
        amp = 3000
        return array.array("h", (int(amp * math.sin(2 * math.pi * f0 * i / SR))
                                 for i in range(n))), SR


class QwenTTSBackend(TTSBackend):
    """Qwen3-TTS open-source, CustomVoice model, 9 preset voices.

    TODO(user): implement synthesize() against your local deployment and fill
    VOICE_MAP with the real preset names. Contract: return mono PCM16 @16k as
    (array('h'), 16000) — resample if the model emits another rate. Keep
    synthesis DETERMINISTIC (fixed seed / no sampling temperature) so builds
    are reproducible; cache by sha256(text|voice|lang|rate) under cache_dir."""

    VOICE_MAP = {f"cv{i:02d}": f"CUSTOMVOICE_PRESET_{i}" for i in range(1, 10)}

    def __init__(self, endpoint=None, model_dir=None, cache_dir="exp/rb/tts_cache"):
        self.endpoint = endpoint
        self.model_dir = model_dir
        self.cache_dir = Path(cache_dir)

    def synthesize(self, text, voice, lang, rate=1.0):
        raise NotImplementedError(
            "wire Qwen3-TTS here: local CustomVoice inference or HTTP endpoint; "
            "return (array('h') PCM16 mono, 16000)")


def _mix_into(buf, samples, at_s):
    i0 = int(at_s * SR)
    need = i0 + len(samples)
    if need > len(buf):
        buf.extend([0] * (need - len(buf)))
    for j, v in enumerate(samples):
        s = buf[i0 + j] + v
        buf[i0 + j] = max(-32768, min(32767, s))


def assemble_episode(episode, backend, out_wav):
    """Render one episode's user-channel wav. Returns the cue table."""
    buf = array.array("h")
    cues = []
    t = 0.0
    seq_end = 0.0
    for p in episode["pieces"]:
        samples, sr = backend.synthesize(p["text"], p["voice"], p["lang"])
        assert sr == SR
        if "at_after_eou" in p:
            at = seq_end + HOLD_S + float(p["at_after_eou"])
            _mix_into(buf, samples, at)
            cues.append({"role": p["role"], "t_start": round(at, 3),
                         "t_end": round(at + len(samples) / SR, 3),
                         "scheduled": "lifecycle_nominal"})
            continue
        t = max(t, len(buf) / SR) + float(p.get("gap_before", 0.0))
        _mix_into(buf, samples, t)
        end = t + len(samples) / SR
        cues.append({"role": p["role"], "t_start": round(t, 3),
                     "t_end": round(end, 3), "scheduled": "sequential"})
        t = end
        seq_end = max(seq_end, end)
    # tail room so the engine can reach its final EoU + windows
    buf.extend([0] * int(6.0 * SR))
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(buf.tobytes())
    return cues


def measured_gaps(cues):
    """Post-hoc gap table (sequential user pieces): the v1 §3 'oversample and
    re-bin' principle, mechanized — report actual gaps for layer binning."""
    seq = [c for c in cues if c["scheduled"] == "sequential" and c["role"] == "user"]
    return [round(b["t_start"] - a["t_end"], 3) for a, b in zip(seq, seq[1:])]
