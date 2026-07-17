# -*- coding: utf-8 -*-
"""rb/audio.py — RB v2 TTS backends + arm-A timeline assembly
(docs/rb_design.md v2 §4).

Backends return (samples, sr) with samples = array('h') mono PCM16 @16k.
  SilenceStub  — deterministic placeholder tone (dry builds: validates the
                 assembly/manifest pipeline; NOT VAD-triggerable, engine runs
                 need a real backend).
  QwenTTSBackend — Qwen3-TTS open-source (CustomVoice): WIRED to the local
                 deployment (subprocess to QWEN3TTS_DIR/scripts/synthesize.py,
                 out-of-repo cache, 16k resample). Voice map:
                 exp/rb/tts_voices.json (cv01..cv09 -> preset names).

Assembly: pieces are placed sequentially by gap_before (exact silence-clock
control — the revision cue time IS end-of-previous-piece + gap); pieces with
`at_after_eou` are placed at (end of last sequential piece + HOLD + value) and
MIXED if they overlap (bystander injections). Emits wav + a cue table with
per-piece [t_start, t_end] — the build-time ground truth for binning."""
from __future__ import annotations

import array
from contextlib import contextmanager
import hashlib
import json
import math
import os
import threading
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SR = 16000
HOLD_S = 0.64


def episode_tts_requests(episodes, include_events=False):
    """Return TTS request dictionaries in deterministic episode order.

    Build-time audio is assembled serially so cue placement and WAV bytes stay
    independent of request completion order.  The expensive synthesis itself
    can be prewarmed concurrently through :meth:`QwenTTSBackend.prewarm`.
    Arm-B callers set ``include_events=True`` because its reactive turns are
    rendered on demand rather than baked into its episode WAVs.
    """
    requests = []
    for episode in episodes:
        lang_default = episode.get("lang") or "zh"
        for piece in episode.get("pieces", []):
            text = piece.get("text")
            if text:
                requests.append({"text": text,
                                 "voice": piece.get("voice") or "cv01",
                                 "lang": piece.get("lang") or lang_default,
                                 "rate": piece.get("rate", 1.0)})
        if include_events:
            for event in episode.get("events", []):
                text = event.get("text")
                if text:
                    requests.append({"text": text,
                                     "voice": event.get("voice") or "cv01",
                                     "lang": event.get("lang") or lang_default,
                                     "rate": event.get("rate", 1.0)})
    return requests


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
    """Local Qwen3-TTS (CustomVoice) via the user's deployment at
    /root/autodl-tmp/Qwen3TTS (override: env QWEN3TTS_DIR): calls
    `<dir>/.venv/bin/python <dir>/scripts/synthesize.py --text .. --voice ..
    --language Chinese|English --output <wav>`, caches OUT OF REPO
    (env RB_TTS_CACHE, default /root/autodl-tmp/rb_tts_cache), resamples to
    16k mono PCM16. Deterministic per (text, preset, lang, rate) — the cache
    key IS the reproducibility unit; a rebuilt cache re-synthesizes.  The
    ``prewarm`` API deduplicates keys and sends isolated local requests in
    parallel; per-key thread and file locks make cache publication safe across
    workers and processes.

    VOICE_MAP: cv01..cv09 -> CustomVoice preset names, loaded from
    exp/rb/tts_voices.json when present ({"cv01": "Vivian", ...}). Fallback
    (until the user lists all 9 presets): zh->Vivian, en->Ryan — flagged in
    the manifest as voice_map_complete=false."""

    LANG = {"zh": "Chinese", "en": "English"}
    FALLBACK = {"zh": "Vivian", "en": "Ryan"}
    _CACHE_LOCKS = {}
    _CACHE_LOCKS_GUARD = threading.Lock()

    def __init__(self, tts_dir=None, cache_dir=None,
                 voice_map_path="exp/rb/tts_voices.json"):
        import os
        self.dir = Path(tts_dir or os.getenv("QWEN3TTS_DIR",
                                             "/root/autodl-tmp/Qwen3TTS"))
        self.cache = Path(cache_dir or os.getenv("RB_TTS_CACHE",
                                                 "/root/autodl-tmp/rb_tts_cache"))
        self.cache.mkdir(parents=True, exist_ok=True)
        p = Path(voice_map_path)
        self.voice_map = json.loads(p.read_text()) if p.exists() else {}
        self.voice_map_complete = len(self.voice_map) >= 9
        # Set by the build after a successful parallel prewarm.  A cache miss
        # during serial assembly is then a coverage bug, not a quiet fallback
        # to one-at-a-time synthesis.
        self.cache_only = False

    def _preset(self, voice, lang):
        return self.voice_map.get(voice) or self.FALLBACK[lang]

    def cache_entry(self, text, voice, lang, rate=1.0):
        """Return ``(key, preset, wav_path)`` for a synthesis request."""
        if lang not in self.LANG:
            raise ValueError(f"unsupported Qwen TTS language: {lang!r}")
        preset = self._preset(voice, lang)
        key = hashlib.sha256(
            f"{text}|{preset}|{lang}|{rate}".encode()).hexdigest()[:24]
        return key, preset, self.cache / f"{key}.wav"

    @classmethod
    def _thread_lock(cls, wav):
        """One in-process mutex per cache key (the file lock covers peers)."""
        with cls._CACHE_LOCKS_GUARD:
            return cls._CACHE_LOCKS.setdefault(str(wav), threading.Lock())

    @contextmanager
    def _cache_guard(self, wav):
        """Serialize writers for one cache key across threads and processes."""
        with self._thread_lock(wav):
            lock_path = wav.with_suffix(wav.suffix + ".lock")
            with lock_path.open("a+") as lock_file:
                try:
                    import fcntl
                except ImportError:  # pragma: no cover - RB runs on Linux
                    fcntl = None
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _synthesize_to_path(self, text, preset, lang, tmp):
        """Issue one isolated HTTP synthesis through the existing CLI shim."""
        import subprocess
        cmd = [str(self.dir / ".venv/bin/python"),
               str(self.dir / "scripts/synthesize.py"),
               "--text", text, "--voice", preset,
               "--language", self.LANG[lang], "--output", str(tmp)]
        r = subprocess.run(cmd, cwd=str(self.dir), capture_output=True,
                           text=True, timeout=300)
        if r.returncode != 0 or not tmp.exists():
            raise RuntimeError(f"qwen3-tts failed for [{preset}/{lang}]: "
                               f"{r.stderr[-400:]}")

    def _ensure_wav(self, text, voice, lang, rate=1.0):
        """Ensure one atomically published cache WAV; return ``(path, hit)``."""
        _key, preset, wav = self.cache_entry(text, voice, lang, rate)
        if wav.exists():
            return wav, True
        if self.cache_only:
            raise RuntimeError(f"Qwen TTS cache miss after prewarm: {wav.name}")
        with self._cache_guard(wav):
            if wav.exists():
                return wav, True
            # A unique temporary path prevents a second process from ever
            # clobbering this writer's output before the atomic replace.
            tmp = wav.with_name(
                f".{wav.stem}.{os.getpid()}.{threading.get_ident()}.tmp.wav")
            try:
                self._synthesize_to_path(text, preset, lang, tmp)
                if not tmp.exists():
                    raise RuntimeError(f"qwen3-tts produced no WAV: {wav.name}")
                tmp.replace(wav)
            finally:
                tmp.unlink(missing_ok=True)
        return wav, False

    def prewarm(self, requests, workers=16):
        """Concurrently synthesize each distinct cache key exactly once.

        ``requests`` is an iterable of dictionaries with ``text``, ``voice``,
        ``lang`` and optional ``rate``.  Submission order is deterministic;
        completion order is intentionally irrelevant because each task writes
        a distinct atomic cache object.  The returned counts are suitable for
        the P2 receipt and distinguish pre-existing hits from actual synthesis.
        """
        if workers < 1:
            raise ValueError("TTS workers must be >= 1")
        unique = {}
        requested = 0
        for request in requests:
            requested += 1
            try:
                text = request["text"]
                voice = request.get("voice") or "cv01"
                lang = request["lang"]
                rate = request.get("rate", 1.0)
            except (AttributeError, KeyError) as exc:
                raise ValueError(f"invalid TTS prewarm request: {request!r}") from exc
            _key, _preset, wav = self.cache_entry(text, voice, lang, rate)
            unique.setdefault(str(wav), (text, voice, lang, rate, wav))

        entries = list(unique.values())
        pending = [entry for entry in entries if not entry[4].exists()]
        initial_hits = len(entries) - len(pending)
        synthesized = 0
        raced_hits = 0
        errors = []
        if pending:
            used_workers = min(workers, len(pending))
            with ThreadPoolExecutor(max_workers=used_workers,
                                    thread_name_prefix="rb-tts") as pool:
                futures = {
                    pool.submit(self._ensure_wav, text, voice, lang, rate): wav
                    for text, voice, lang, rate, wav in pending
                }
                for future in as_completed(futures):
                    wav = futures[future]
                    try:
                        _path, hit = future.result()
                    except Exception as exc:  # preserve every independent failure
                        errors.append(f"{wav.name}: {exc}")
                    else:
                        if hit:
                            raced_hits += 1
                        else:
                            synthesized += 1
            if errors:
                excerpt = "\n".join(errors[:4])
                raise RuntimeError(f"Qwen TTS prewarm failed ({len(errors)} keys):\n{excerpt}")
        else:
            used_workers = 0
        return {"requested": requested, "unique": len(entries),
                "initial_cache_hits": initial_hits, "cache_misses": len(pending),
                "synthesized": synthesized, "raced_cache_hits": raced_hits,
                "workers_requested": workers, "workers_used": used_workers}

    def synthesize(self, text, voice, lang, rate=1.0):
        import soundfile as sf
        wav, _hit = self._ensure_wav(text, voice, lang, rate)
        a, sr = sf.read(str(wav), dtype="float32")
        if a.ndim == 2:
            a = a.mean(axis=1)
        if sr != SR:
            import torch
            import torchaudio
            a = torchaudio.functional.resample(
                torch.from_numpy(a).unsqueeze(0), sr, SR).squeeze(0).numpy()
        pcm = array.array("h", (int(max(-1.0, min(1.0, float(x))) * 32767)
                                for x in a))
        return pcm, SR


def _mix_into(buf, samples, at_s):
    i0 = int(at_s * SR)
    need = i0 + len(samples)
    if need > len(buf):
        buf.extend([0] * (need - len(buf)))
    for j, v in enumerate(samples):
        s = buf[i0 + j] + v
        buf[i0 + j] = max(-32768, min(32767, s))


# ---------------------------------------------------------------------------
# v2.3 seeded perturbation family (design §4 promise): rate / gain / scene SNR.
# Applied POST-synthesis so the TTS cache stays keyed on clean text+voice.
# ---------------------------------------------------------------------------
def perturb_samples(samples, rate=1.0, gain_db=0.0):
    """Speed (linear resample; pitch shifts with it — declared) + gain."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    if abs(rate - 1.0) > 1e-6 and len(x) > 1:
        n_out = max(1, int(round(len(x) / rate)))
        x = np.interp(np.linspace(0, len(x) - 1, n_out),
                      np.arange(len(x)), x)
    if abs(gain_db) > 1e-6:
        x = x * (10.0 ** (gain_db / 20.0))
    x = np.clip(x, -32768, 32767)
    return array.array("h", x.astype("int16").tolist())


def noise_sigma(first_piece_samples, snr_db):
    """Scene-noise sigma from the FIRST piece's RMS (known first in both
    arms — keeps arm-B lazy prefixes consistent) at the target SNR."""
    import numpy as np
    if snr_db is None:
        return 0.0
    x = np.asarray(first_piece_samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    return rms / (10.0 ** (snr_db / 20.0))


def noise_block(episode_id, n, sigma):
    """Deterministic scene noise, reproducible for any prefix length."""
    import numpy as np
    if sigma <= 0.0 or n <= 0:
        return np.zeros(n, dtype=np.float32)
    seed = int(hashlib.sha256(f"{episode_id}:noise".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * sigma


def assemble_episode(episode, backend, out_wav):
    """Render one episode's user-channel wav. Returns the cue table."""
    import numpy as np
    buf = array.array("h")
    cues = []
    t = 0.0
    seq_end = 0.0
    pb = episode.get("perturb") or {}
    rate, gain = pb.get("rate", 1.0), pb.get("gain_db", 0.0)
    sigma = None
    for p in episode["pieces"]:
        samples, sr = backend.synthesize(p["text"], p["voice"], p["lang"])
        assert sr == SR
        samples = perturb_samples(samples, rate, gain)
        if sigma is None:
            sigma = noise_sigma(samples, pb.get("snr_db"))
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
    if sigma and sigma > 0.0:
        x = np.asarray(buf, dtype=np.float32) + \
            noise_block(episode["id"], len(buf), sigma)
        buf = array.array("h", np.clip(x, -32768, 32767).astype("int16").tolist())
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
