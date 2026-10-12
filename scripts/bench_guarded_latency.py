#!/usr/bin/env python3
"""Guarded-path end-to-end latency bench through the real backend WebSocket.

Browser-protocol client (``client: humdial-web``, ``pcm16.ref.v1`` FDM1 mic/ref
packets, same-origin header), a Python mirror of ``src/static/speech-player.js``
(playAt = first packet + startup_ms, ``speech_hold`` pause/resume, playback
ACKs and demo telemetry) and a line-for-line mirror of the client pre-playback
local hold (client-hold-v1, ``OnsetDetector``/``LocalHold`` in speech-player.js;
mode, params and resume lead come from ``/api/demo/info`` ``client_hold`` exactly
as demo.js reads them, ``--local-pause`` forces mode "on").  The backend under test
is launched in isolation (backend-only, reusing the Omni on :10003) from any
code tree (``--backend-root``) with per-profile config overrides.

Subcommands
  episodes  cut listening-period user episodes from recorded real sessions (CPU only)
  fixtures  synthesize self-authored pause/continuation inputs via :10003 TTS (GPU)
  run       launch an isolated backend and replay synthetic and/or real suites (GPU)
  analyze   parse a run directory into per-END rows and group P50/P90 tables
  report    merge several analyzed runs into one markdown table

All model-calling commands must be wrapped by the caller in
``flock /tmp/lat5_gpu.lock`` for the whole measurement.
"""
import argparse
import asyncio
import copy
import glob
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

SR = 16000
FRAME = 256
FRAME_S = FRAME / SR
INPUT_HEADER = struct.Struct("<4sIII")
OUTPUT_HEADER = struct.Struct("<4sIII")
PY = "/root/miniconda3/envs/fd-sds/bin/python"
DEFAULT_SRC_ROOT = Path("/root/autodl-tmp/fd-badcat")
REAL_SESSIONS = ["d90e22", "fad453", "b55937", "771115", "aee007", "8f06b7",
                 "a6651f", "b1ef5e", "b4cb83", "13045d", "06a491"]
PROFILES = {
    "base_640_350": {},
    "base_400_350": {"time.end_hold_frame": 0.4},
    "base_640_300": {"engine.stream_startup_ms": 300},
}
PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")

# Self-authored inputs (never HumDial).  (name, pieces, gaps_s)
FIXTURE_TEXTS = [
    ("pause_zh_a", ["我想了解一下，", "去云南旅游，哪个季节最合适？"], [0.5]),
    ("pause_zh_b", ["帮我想想，", "周末在家可以做些什么有意思的事情？"], [0.7]),
    ("pause_zh_c", ["关于学习编程，", "初学者应该先学哪门语言比较好？"], [0.4]),
    ("pause_en_a", ["I was wondering,", "what is the best way to keep houseplants alive?"], [0.6]),
    ("cont_g08", ["我想问一个问题，", "从上海到杭州坐高铁大概要多久？"], [0.8]),
    ("cont_g10", ["我想问一个问题，", "从上海到杭州坐高铁大概要多久？"], [1.0]),
    ("cont_g12", ["我想问一个问题，", "从上海到杭州坐高铁大概要多久？"], [1.2]),
]


def now():
    return time.monotonic()


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def jrows(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def read16k(path):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim == 2:
        x = x.mean(1)
    if sr != SR:  # e.g. 24 kHz p0 fixtures; the browser worklet also delivers 16 kHz
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


# --------------------------------------------------------------------------
# Ground-truth speech segments (audio clock): Silero proposes, energy refines.
# --------------------------------------------------------------------------
_VAD = None


def _silero():
    global _VAD
    if _VAD is None:
        import torch
        torch.set_num_threads(2)
        from silero_vad import load_silero_vad
        _VAD = load_silero_vad()
    return _VAD


def speech_segments(x, merge_gap=0.15):
    """Return [{start,end,silero_start,silero_end,end_rms01}] in seconds.

    Silero (threshold 0.5, no padding) finds speech; each boundary is refined to
    the first/last 10 ms frame above max(0.003, 4 x noise floor, 3 % of the
    segment peak) within +-150 ms.  ``end_rms01`` is the 853 ms baseline rule
    (last 10 ms frame with RMS > 0.01).
    """
    import torch
    from silero_vad import get_speech_timestamps
    x = np.asarray(x, dtype=np.float32)
    if len(x) < SR // 10:
        return []
    ts = get_speech_timestamps(torch.from_numpy(x.copy()), _silero(), sampling_rate=SR,
                               threshold=0.5, min_speech_duration_ms=80,
                               min_silence_duration_ms=int(merge_gap * 1000), speech_pad_ms=0)
    hop = SR // 100
    n = len(x) // hop
    rms = np.sqrt((x[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1))
    floor = float(np.percentile(rms, 10)) if n else 0.0
    segs = []
    for t in ts:
        s0, e0 = t["start"] / SR, t["end"] / SR
        lo, hi = max(0, int((s0 - .15) * 100)), min(n, int((e0 + .15) * 100) + 1)
        s, e, e01 = s0, e0, None
        win = rms[lo:hi]
        if len(win):
            thr = max(.003, 4 * floor, .03 * float(win.max()))
            act = np.nonzero(win >= thr)[0]
            if len(act):
                s, e = (lo + act[0]) / 100, (lo + act[-1] + 1) / 100
            a01 = np.nonzero(win > .01)[0]
            if len(a01):
                e01 = (lo + a01[-1] + 1) / 100
        segs.append({"start": round(s, 3), "end": round(e, 3), "silero_start": round(s0, 3),
                     "silero_end": round(e0, 3), "end_rms01": None if e01 is None else round(e01, 3)})
    merged = []
    for seg in sorted(segs, key=lambda d: d["start"]):
        if merged and seg["start"] - merged[-1]["end"] < merge_gap:
            m = merged[-1]
            m["end"] = max(m["end"], seg["end"])
            m["silero_end"] = max(m["silero_end"], seg["silero_end"])
            if seg["end_rms01"] is not None:
                m["end_rms01"] = max(m["end_rms01"] or 0, seg["end_rms01"])
        else:
            merged.append(dict(seg))
    return merged


# --------------------------------------------------------------------------
# episodes: listening-period user episodes from recorded real sessions
# --------------------------------------------------------------------------
def _playback_intervals(ev):
    starts, ends, last = {}, {}, {}
    for r in ev:
        d, t = r.get("data") or {}, r.get("t_audio")
        uid = d.get("utterance_id")
        if t is None or uid is None:
            continue
        name = r["event"]
        if name == "speech_playback_started":
            starts.setdefault(uid, t)
        elif name == "speech_cancelled":
            ends.setdefault(uid, t)
        elif name == "playback_progress":
            last[uid] = t
            if d.get("ended") is True:
                ends.setdefault(uid, t)
        elif name in ("speech_played",):
            ends.setdefault(uid, t)
    out = []
    for uid, s in starts.items():
        e = ends.get(uid, last.get(uid, s))
        out.append((s, max(e, s), uid))
    return sorted(out)


def cmd_episodes(a):
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    manifest = {"version": "lat5-real-episodes-v1", "src_root": str(a.src_root),
                "rule": "listening-period user episodes: from 0.5 s before the first vad_start after the "
                        "previous assistant playback ended, to the original assistant playback start; "
                        "reference channel zeroed; capped at max_episode_s", "max_episode_s": a.max_episode_s,
                "sessions": []}
    for sid in a.sessions:
        dirs = sorted(glob.glob(str(Path(a.src_root) / "exp" / f"web-demo-{sid}*" / "realtimeout_live")))
        if len(dirs) != 1:
            raise SystemExit(f"session {sid}: {dirs}")
        d = Path(dirs[0])
        ev = jrows(d / "events.jsonl")
        mic = read16k(d / "diagnostics/capture/browser_mic.wav")
        dur = len(mic) / SR
        plays = _playback_intervals(ev)
        vstarts = [r["t_audio"] for r in ev if r["event"] == "vad_start" and r.get("t_audio") is not None]
        vdones = [r["t_audio"] for r in ev if r["event"] == "vad_done" and r.get("t_audio") is not None]

        def playing(t):
            return any(s <= t < e for s, e, _ in plays)

        eps, cursor = [], 0.0
        for s, e, uid in plays + [(None, None, None)]:
            limit = dur if s is None else s
            cand = [t for t in vstarts if cursor < t < limit and not playing(t)]
            if cand:
                st = max(cursor, cand[0] - .5)
                if s is None:
                    after = [t for t in vdones if t > cand[0]]
                    en = min(dur, (max(after) if after else cand[-1]) + 2.0)
                else:
                    en = s
                capped = en - st > a.max_episode_s
                en = min(en, st + a.max_episode_s)
                if en - st >= .4:
                    eps.append({"orig_start": round(st, 3), "orig_end": round(en, 3),
                                "orig_reply_utterance_id": uid, "ends_at_orig_playback": s is not None,
                                "capped": capped})
            if s is not None:
                cursor = max(cursor, e)
        sess = {"sid": sid, "session_dir": str(d), "duration_s": round(dur, 3), "episodes": []}
        for k, ep in enumerate(eps):
            x = mic[int(ep["orig_start"] * SR):int(ep["orig_end"] * SR)]
            name = f"{sid}_{k:03d}"
            path = out / "audio" / f"{name}.wav"
            sf.write(str(path), x, SR, subtype="PCM_16")
            segs = speech_segments(x)
            ep.update(name=name, path=str(path), duration_s=round(len(x) / SR, 3), segments=segs,
                      orig_vad_starts=[round(t - ep["orig_start"], 3) for t in vstarts
                                       if ep["orig_start"] <= t < ep["orig_end"]],
                      orig_vad_dones=[round(t - ep["orig_start"], 3) for t in vdones
                                      if ep["orig_start"] <= t < ep["orig_end"]])
            sess["episodes"].append(ep)
        manifest["sessions"].append(sess)
        print(sid, "episodes", len(eps), "audio_s", round(sum(e["duration_s"] for e in sess["episodes"]), 1),
              "segments", sum(len(e["segments"]) for e in sess["episodes"]), flush=True)
    jdump(out / "episodes.json", manifest)
    n = sum(len(s["episodes"]) for s in manifest["sessions"])
    print("TOTAL episodes", n, "->", out / "episodes.json")


# --------------------------------------------------------------------------
# fixtures: self-authored multi-piece inputs through the :10003 verbatim TTS
# --------------------------------------------------------------------------
def _trim(x, thr=.005):
    hop = SR // 100
    n = len(x) // hop
    rms = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
    act = np.nonzero(rms > thr)[0]
    if not len(act):
        return x
    return x[max(0, act[0] * hop - 320):min(len(x), (act[-1] + 1) * hop + 320)]


def cmd_fixtures(a):
    import base64
    import io
    import requests
    root = Path(a.backend_root)
    sys.path.insert(0, str(root / "src"))
    from module import verbatim_tts_payload, _mono_16k
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    sess.trust_env = False
    cache, meta = {}, {}
    for name, pieces, gaps in FIXTURE_TEXTS:
        parts = []
        for text in pieces:
            if text not in cache:
                p = verbatim_tts_payload(text, voice_control={"speaker": "chelsie", "seed": 42})
                d = sess.post("http://127.0.0.1:10003/v1/chat/completions", json=p, timeout=180).json()
                au = [c["message"] for c in d["choices"] if c["message"].get("audio")][0]
                y, sr = sf.read(io.BytesIO(base64.b64decode(au["audio"]["data"])), dtype="float32")
                cache[text] = _trim(_mono_16k(y, sr).astype(np.float32))
            parts.append(cache[text])
        x = [np.zeros(4000, np.float32)]
        for i, part in enumerate(parts):
            if i:
                x.append(np.zeros(int(gaps[i - 1] * SR), np.float32))
            x.append(part)
        x.append(np.zeros(1600, np.float32))
        x = np.concatenate(x)
        sf.write(str(out / f"{name}.wav"), x, SR, subtype="PCM_16")
        meta[name] = {"pieces": pieces, "gaps_s": gaps, "duration_s": round(len(x) / SR, 3),
                      "source": "self_authored_tts_chelsie_seed42", "segments": speech_segments(x)}
        print(name, meta[name]["duration_s"], [(s["start"], s["end"]) for s in meta[name]["segments"]], flush=True)
    jdump(out / "fixtures.json", meta)


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------
def synthetic_suite(a):
    web = Path(a.src_root) / "exp/web_demo"
    q = web / "thinker_fp8_20261012/e2e/inputs"
    p0 = web / "p0_pro6000_20261007/fixtures"
    p1a = web / "p1_input_diagnosis_20261008/audio"
    p1f = web / "p1_latency_diagnosis_20261008/fixtures"
    fx = Path(a.fixtures)
    S = []
    for i in range(8):
        S.append(("short_q%d" % i, "short", [q / f"q{i}.wav"]))
    S.append(("long_story", "long", [p0 / "long.wav"]))
    S.append(("normal_zh", "long", [p0 / "normal_zh.wav"]))
    S.append(("normal_en", "long", [p0 / "normal_en.wav"]))
    for name in ("pause_zh_a", "pause_zh_b", "pause_zh_c", "pause_en_a"):
        S.append((name, "long", [fx / f"{name}.wav"]))
    S.append(("cont_wait_045", "cont", [p1f / "unfinished_then_complete.wav"]))
    for name in ("cont_g08", "cont_g10", "cont_g12"):
        S.append((name, "cont", [fx / f"{name}.wav"]))
    S.append(("follow_zh", "history", [p0 / "normal_zh.wav", p1a / "tts_question_4_s42.wav"]))
    S.append(("follow_en", "history", [p0 / "normal_en.wav", p1a / "tts_question_8_s42.wav"]))
    return S


# --------------------------------------------------------------------------
# Simulated browser: speech-player.js mirror + telemetry + optional local pause
# --------------------------------------------------------------------------
class Link:
    """Ordered client<->server transport with an optional one-way delay each way."""

    def __init__(self, ws, delay_s):
        self.ws, self.delay = ws, delay_s
        self.out = asyncio.Queue()
        self.closed = False

    def send_json(self, event, data):
        self.out.put_nowait((now() + self.delay, json.dumps({"event": event, "data": data})))

    def send_bytes(self, raw):
        self.out.put_nowait((now() + self.delay, raw))

    async def sender(self):
        while True:
            due, payload = await self.out.get()
            if payload is None:
                return
            wait = due - now()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                await self.ws.send(payload)
            except Exception:
                self.closed = True
                return


# ---- client-hold-v1 mirror: keep line-for-line with src/static/speech-player.js
LOCAL_HOLD_DEFAULTS = dict(
    floorInitDb=-60, floorMinDb=-80, floorMaxDb=-35,
    fallRate=0.2, riseRate=0.02, riseVoicedRate=0.0005,
    marginDb=12, absMinDb=-50, onsetFrames=2, sustainFrames=125,
    refGateDb=-60, refHangFrames=10, echoTailMs=150,
    timerMinMs=300, timerRttFactor=2, timerPadMs=200, fallbackRttMs=150,
)


def _to_db(value):
    return 20 * math.log10(max(value, 1e-9))


def _from_db(value):
    return math.pow(10, value / 20)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class OnsetDetector:
    """Python twin of speech-player.js OnsetDetector (same arithmetic, same order)."""

    def __init__(self, params=None):
        self.p = dict(LOCAL_HOLD_DEFAULTS)
        for key in LOCAL_HOLD_DEFAULTS:
            if _finite((params or {}).get(key)):
                self.p[key] = (params or {})[key]
        self.floor_db = self.p["floorInitDb"]
        self.run = 0
        self.since_ref = math.inf

    def frame(self, mic_rms, ref_rms=0.0, output_recent=False):
        p, db, ref_db = self.p, _to_db(mic_rms), _to_db(ref_rms)
        self.since_ref = 0 if ref_db > p["refGateDb"] else self.since_ref + 1
        echo = bool(output_recent) or self.since_ref <= p["refHangFrames"]
        threshold_db, floor_db = max(self.floor_db + p["marginDb"], p["absMinDb"]), self.floor_db
        above = db > threshold_db
        if not echo:
            rate = (p["fallRate"] if db < self.floor_db
                    else p["riseVoicedRate"] if above and self.run < p["sustainFrames"] else p["riseRate"])
            self.floor_db = min(p["floorMaxDb"], max(p["floorMinDb"], self.floor_db + (db - self.floor_db) * rate))
        self.run = self.run + 1 if above and not echo else 0
        return {"onset": self.run >= p["onsetFrames"], "echo": echo, "db": db, "ref_db": ref_db,
                "floor_db": floor_db, "threshold_db": threshold_db}


def client_hold_mode(value):
    """LocalHold constructor: true/"on" -> on, "shadow" -> shadow, else off."""
    return "on" if value is True or value == "on" else "shadow" if value == "shadow" else "off"


def resume_lead(value):
    """SpeechPlayer.setResumeLead: finite in [0, 1000] ms, else None (= startup)."""
    return value if _finite(value) and 0 <= value <= 1000 else None


class Speech:
    def __init__(self, uid, startup_ms, buffer_ms, at):
        self.id, self.startup_ms, self.limit = uid, startup_ms, buffer_ms
        self.seq = self.received = self.played = 0
        self.next = 0.0
        self.rate = None
        self.eof = False
        self.started = False
        self.held = False
        self.local_hold = None  # client-hold-v1: {at, timer_at, timer_ms, done, shadow}
        self.pending = []
        self.nodes = []
        self.play_at = None
        self.first_audio = None
        self.start_at = at
        self.underruns = 0
        self.underrun_ms = 0.0
        self.max_underrun_ms = 0.0
        self.tel_first = self.tel_started = self.tel_end = False
        self.text_at = None


class Browser:
    def __init__(self, link, args, log, t0_ref):
        self.link, self.a, self.log = link, args, log
        self.t0 = t0_ref  # capture time of audio sample 0 (set by the mic loop)
        self.speech = None
        self.telemetry = False
        self.ping_seq = 0
        self.ping_pending = None
        self.last_ping = -1e9
        self.rtts = []
        self.last_output_end = -1e9
        self.last_activity = now()
        self.replies_done = 0
        self.session_id = None
        self.ready = asyncio.Event()
        self.errors = []
        self.utt = {}  # uid -> client-side record
        # client-hold-v1 exactly as demo.js wires it from /api/demo/info.client_hold.
        hold_info = getattr(args, "client_hold", None) or {}
        self.local_mode = "on" if args.local_pause else client_hold_mode(hold_info.get("local_hold"))
        self.detector = OnsetDetector(hold_info.get("params") or {})
        self.resume_lead_ms = (resume_lead(args.release_lead_ms) if args.release_lead_ms is not None
                               else resume_lead(hold_info.get("resume_lead_ms")))
        self.last_rtt_ms = None  # demo.js: latest telemetry RTT (s.rttMs)

    # ---- clocks / logging
    def T(self, t=None):
        return round(((now() if t is None else t) - self.t0[0]), 4)

    def rec(self, uid):
        return self.utt.setdefault(uid, {"utterance_id": uid, "holds": []})

    def emit(self, kind, **data):
        if self.telemetry:
            self.link.send_json("demo_telemetry", {"kind": kind, "client_ms": now() * 1000,
                                                   "upload_buffer_bytes": 0, **data})

    def tick_ping(self):
        t = now()
        if self.telemetry and t - self.last_ping >= 5:
            self.last_ping = t
            self.ping_seq += 1
            self.ping_pending = (self.ping_seq, t)
            self.emit("ping", seq=self.ping_seq)

    def rtt_s(self):
        return float(np.median(self.rtts[-5:])) / 1000 if self.rtts else 2 * self.a.net_delay_ms / 1000

    # ---- speech-player.js mirror
    def _partial_played(self, s, t):
        played = s.played
        for n in s.nodes:
            if t >= n["start"]:
                played = max(played, n["offset"] + max(0, min(n["count"], int((t - n["start"]) * s.rate))))
        return min(played, s.received)

    def cancel(self, reason="replaced"):
        s = self.speech
        if s is None:
            return
        t = now()
        played = self._partial_played(s, t)
        if any(t >= n["start"] for n in s.nodes):
            self.last_output_end = t
        self.speech = None
        r = self.rec(s.id)
        r.setdefault("cancelled_at", self.T(t))
        r.setdefault("cancel_reason", reason)
        r["played_samples_at_cancel"] = played
        self.link.send_json("playback_stopped", {"utterance_id": s.id, "played_samples": played})

    def start(self, d):
        self.cancel("replaced")
        uid = d.get("utterance_id")
        startup = d.get("startup_ms", 80)
        self.speech = Speech(uid, startup, d.get("buffer_ms", 600), now())
        r = self.rec(uid)
        r.update(speech_start_at=self.T(), startup_ms=startup, buffer_ms=d.get("buffer_ms"),
                 candidate_id=d.get("candidate_id"), parent_id=d.get("parent_id"))

    def progress(self, s):
        if self.speech is not s:
            return
        t = now()
        was = s.started
        s.started = s.started or s.played > 0 or (s.received > 0 and s.play_at is not None and t >= s.play_at)
        self.link.send_json("playback_progress", {"utterance_id": s.id, "played_samples": s.played,
                                                  "started": bool(s.started),
                                                  "ended": bool(s.eof and s.played == s.received),
                                                  "underruns": s.underruns})
        r = self.rec(s.id)
        if s.started and not was:
            r["play_start_at"] = round(s.play_at - self.t0[0], 4) if s.play_at not in (None, math.inf) else self.T(t)
            r["started_detected_at"] = self.T(t)
        # demo-telemetry.js playback()
        if s.first_audio is not None and not s.tel_first:
            s.tel_first = True
            self.emit("first_audio", utterance_id=s.id, first_audio_ms=s.first_audio,
                      scheduled_lead_ms=s.startup_ms, base_latency_ms=5.333, output_latency_ms=self.a.output_latency_ms)
        if s.started and not s.tel_started:
            s.tel_started = True
            self.emit("playback_start", utterance_id=s.id, audio_context_ms=t * 1000,
                      scheduled_start_ms=(s.play_at if s.play_at not in (None, math.inf) else t) * 1000,
                      played_samples=s.played)
        if s.eof and s.received == s.played and not s.tel_end:
            s.tel_end = True
            r["ended_at"] = self.T(t)
            self.replies_done += 1
            self.last_activity = t
            self.emit("playback_end", utterance_id=s.id, played_samples=s.played, received_samples=s.received,
                      underruns=s.underruns, underrun_ms=s.underrun_ms, max_underrun_ms=s.max_underrun_ms)

    def poll_start(self):
        s = self.speech
        if s and not s.started and s.received and s.play_at is not None and now() >= s.play_at:
            self.progress(s)

    def hold(self, held, source):
        s = self.speech
        if s is None:
            return
        self.poll_start()
        r = self.rec(s.id)
        r["holds"].append({"at": self.T(), "held": held, "source": source, "started": s.started,
                           "effective": (not s.started) and (held != s.held)})
        if s.started:
            return
        if held and not s.held:
            s.held = True
            for n in s.nodes:
                s.pending.append(n)
            s.nodes = []
            s.pending.sort(key=lambda n: n["offset"])
            s.play_at = math.inf
        elif not held and s.held:
            s.held = False
            lead = min(s.startup_ms, s.startup_ms if self.resume_lead_ms is None else self.resume_lead_ms)
            s.next = now() + lead / 1000
            s.play_at = s.next
            r.setdefault("release_play_at", []).append(round(s.play_at - self.t0[0], 4))
            pend, s.pending = s.pending, []
            for n in pend:
                self.schedule(s, n["offset"], n["count"])

    def schedule(self, s, offset, count):
        n = {"start": s.next, "offset": offset, "count": count, "end": s.next + count / s.rate}
        s.nodes.append(n)
        s.next += count / s.rate

    def packet(self, raw):
        if len(raw) < 18 or (len(raw) - 16) % 2:
            raise ValueError("invalid PCM packet")
        magic, uid, seq, rate = OUTPUT_HEADER.unpack_from(raw)
        if magic != b"FDS1":
            raise ValueError("unknown audio protocol")
        s = self.speech
        if s is None or uid != s.id:
            return
        if seq != s.seq or (s.rate and rate != s.rate):
            raise ValueError("audio order/rate error")
        s.seq += 1
        if s.eof:
            raise ValueError("audio after end")
        s.rate = rate
        count = (len(raw) - 16) // 2
        s.received += count
        if (s.received - s.played) / rate * 1000 > s.limit + 1:
            raise ValueError("playback buffer overrun")
        t = now()
        self.last_activity = t
        r = self.rec(s.id)
        if s.first_audio is None:
            s.first_audio = round((t - s.start_at) * 1000)
            s.next = t + s.startup_ms / 1000
            s.play_at = s.next
            r.update(first_packet_at=self.T(t), initial_play_at=round(s.play_at - self.t0[0], 4))
        elif not s.held and s.next < t:
            s.underruns += 1
            gap = (t - s.next + .08) * 1000
            s.underrun_ms += gap
            s.max_underrun_ms = max(s.max_underrun_ms, gap)
            s.next = t + .08
        if s.held:
            s.pending.append({"offset": s.received - count, "count": count})
            s.play_at = math.inf
        else:
            self.schedule(s, s.received - count, count)
        self.progress(s)

    def tick(self):
        """Node onended emulation + local-pause timer."""
        s = self.speech
        if s is None:
            return
        t = now()
        done = [n for n in s.nodes if t >= n["end"]]
        if done:
            s.nodes = [n for n in s.nodes if t < n["end"]]
            for n in sorted(done, key=lambda n: n["offset"]):
                s.played = max(s.played, n["offset"] + n["count"])
                self.last_output_end = max(self.last_output_end, n["end"])
                if self.speech is s:
                    self.progress(s)
        if s.nodes and any(t >= n["start"] for n in s.nodes):
            self.last_output_end = t
        hold = s.local_hold
        if hold is not None and not hold["done"] and hold.get("timer_at") is not None and t >= hold["timer_at"]:
            self.local_expire(s, hold)

    # ---- client-hold-v1 mirror (speech-player.js LocalHold), one call per uploaded mic frame
    def recent_output(self, window_s):
        """SpeechPlayer.recentOutput: started output still playing, or ended < window ago."""
        t = now()
        until = self.last_output_end
        s = self.speech
        if s is not None and any(n["start"] <= t for n in s.nodes):
            until = max(until, t)
        return t < until + window_s

    def local_report(self, s, action, **data):
        self.emit("client_hold", utterance_id=s.id, action=action, **data)

    def local_pause(self, mic_rms, ref_rms=0.0):
        if self.local_mode == "off":  # demo.js does not even run the detector
            return None
        p = self.detector.p
        latency = max(0.0, self.a.output_latency_ms / 1000)
        r = self.detector.frame(mic_rms, ref_rms, self.recent_output(latency + p["echoTailMs"] / 1000))
        if not r["onset"]:
            return r
        s = self.speech
        if s is None or not s.received or s.held or s.local_hold is not None:
            return r
        self.poll_start()
        if s.started:
            return r
        rtt = self.last_rtt_ms
        rtt_ms = rtt if _finite(rtt) and rtt >= 0 else p["fallbackRttMs"]
        timer_ms = max(p["timerMinMs"], p["timerRttFactor"] * rtt_ms + p["timerPadMs"])
        energy = dict(frame_rms=mic_rms, ref_rms=ref_rms, noise_floor_rms=_from_db(r["floor_db"]),
                      threshold_rms=_from_db(r["threshold_db"]), timer_ms=timer_ms, rtt_ms=rtt_ms,
                      output_latency_ms=latency * 1000)
        t = now()
        if self.local_mode == "shadow":
            s.local_hold = {"at": t, "done": True, "shadow": True}
            self.rec(s.id).setdefault("local_shadow_at", []).append(self.T(t))
            self.local_report(s, "shadow", **energy)
            return r
        self.hold(True, "local_onset")
        if not s.held:
            return r
        s.local_hold = {"at": t, "timer_ms": timer_ms, "timer_at": t + timer_ms / 1000, "done": False}
        self.rec(s.id).setdefault("local_hold_at", []).append(self.T(t))
        self.local_report(s, "hold", **energy)
        return r

    def local_expire(self, s, hold):
        if hold["done"]:
            return
        hold["done"] = True
        if self.speech is not s or not s.held:
            return
        t = now()
        self.rec(s.id).setdefault("local_release_at", []).append(self.T(t))
        self.hold(False, "local_timer")
        self.local_report(s, "timeout", hold_ms=(t - hold["at"]) * 1000, timer_ms=hold["timer_ms"])

    def local_server(self, held):
        """LocalHold.server: a server speech_hold clears the timer; the server's value wins."""
        s = self.speech
        hold = s.local_hold if s is not None else None
        if hold is not None and not hold["done"]:
            hold["done"] = True
            self.local_report(s, "release", hold_ms=(now() - hold["at"]) * 1000, timer_ms=hold["timer_ms"],
                              server_held=1 if held else 0)
        self.hold(held, "server")

    def local_cancel(self):
        """LocalHold.cancel: before the player's utterance is replaced or cancelled."""
        s = self.speech
        hold = s.local_hold if s is not None else None
        if hold is not None and not hold["done"]:
            hold["done"] = True

    # ---- control messages
    def control(self, msg):
        ev, d = msg.get("event"), msg.get("data") or {}
        t = now()
        keep = {k: d[k] for k in ("utterance_id", "input_id", "candidate_id", "parent_id", "held", "reason",
                                  "route", "samples", "startup_ms", "seq") if k in d}
        self.log.append({"at": self.T(t), "event": ev, **keep})
        if ev == "error":
            self.errors.append(d.get("message"))
            return
        if ev == "demo_ready":
            self.session_id = d.get("session_id")
            self.telemetry = d.get("observability") in ("demo-trace-v1", "demo-trace-v2")
            self.ready.set()
            return
        if ev == "demo_pong":
            if self.ping_pending and d.get("seq") == self.ping_pending[0]:
                ms = (t - self.ping_pending[1]) * 1000
                self.ping_pending = None
                self.rtts.append(ms)
                self.last_rtt_ms = ms
                self.emit("rtt", seq=d.get("seq"), rtt_ms=ms)
            return
        if ev in ("vad_start", "vad_done", "speech_start", "input_waiting", "speech_cancelled"):
            self.last_activity = t
        if ev == "speech_start":
            self.local_cancel()
            self.start(d)
            return
        s = self.speech
        if ev == "speech_cancelled" and s is not None and d.get("utterance_id") == s.id:
            self.emit("cancel", utterance_id=s.id, played_samples=s.played, received_samples=s.received,
                      underruns=s.underruns, underrun_ms=s.underrun_ms, max_underrun_ms=s.max_underrun_ms)
            self.rec(s.id)["server_cancel_reason"] = d.get("reason")
            self.local_cancel()
            self.cancel(d.get("reason") or "cancelled")
            self.last_activity = t
            return
        if ev == "speech_cancelled":
            self.rec(d.get("utterance_id"))["server_cancel_reason"] = d.get("reason")
            return
        if s is not None and d.get("utterance_id") == s.id:
            if ev == "speech_hold":
                self.local_server(d.get("held") is True)
            elif ev == "speech_text_delta":
                s.text_at = s.text_at or t
                self.rec(s.id).setdefault("text_at", self.T(t))
            elif ev == "speech_audio_end":
                if d.get("samples") != s.received:
                    self.errors.append("audio end count mismatch")
                s.eof = True
                self.rec(s.id)["audio_end_at"] = self.T(t)
                self.progress(s)
        elif ev == "speech_text_delta":
            self.rec(d.get("utterance_id")).setdefault("text_at", self.T(t))

    def busy(self):
        s = self.speech
        return s is not None and not (s.eof and s.played == s.received)


async def run_trial(a, url, origin, plan, out_dir, meta):
    """plan: list of dicts {label, audio(np.float32), segments(rel), kind}."""
    import websockets
    out_dir.mkdir(parents=True, exist_ok=True)
    log, t0 = [], [now()]
    receipt = dict(meta, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), items=[],
                   net_delay_ms=a.net_delay_ms, local_pause=a.local_pause, release_lead_ms=a.release_lead_ms,
                   client_hold=getattr(a, "client_hold", None))
    connect_kw = dict(origin=origin, max_size=2 ** 22, ping_interval=None, compression=None)
    try:
        ws = await websockets.connect(url, proxy=None, **connect_kw)
    except TypeError:
        ws = await websockets.connect(url, **connect_kw)
    link = Link(ws, a.net_delay_ms / 1000)
    br = Browser(link, a, log, t0)
    tasks = []
    stop_mic = asyncio.Event()
    mic = {"queue": [], "sent": 0, "late": []}

    async def receiver():
        inbox = asyncio.Queue()

        async def process():
            while True:
                due, m = await inbox.get()
                if m is None:
                    return
                w = due - now()
                if w > 0:
                    await asyncio.sleep(w)
                try:
                    if isinstance(m, (bytes, bytearray)):
                        br.packet(bytes(m))
                    else:
                        br.control(json.loads(m))
                except Exception as exc:
                    br.errors.append(f"{type(exc).__name__}: {exc}")
        proc = asyncio.create_task(process())
        try:
            async for m in ws:
                inbox.put_nowait((now() + link.delay, m))
        except Exception:
            pass
        inbox.put_nowait((now(), None))
        await proc

    async def mic_loop():
        start = now() + .05
        t0[0] = start
        k = 0
        while not stop_mic.is_set():
            target = start + (k + 1) * FRAME_S
            w = target - now()
            if w > 0:
                await asyncio.sleep(w)
            else:
                mic["late"].append(-w)
            need, parts = FRAME, []
            while need and mic["queue"]:
                head = mic["queue"][0]
                take = head[:need]
                parts.append(take)
                need -= len(take)
                if len(take) == len(head):
                    mic["queue"].pop(0)
                else:
                    mic["queue"][0] = head[len(take):]
            if need:
                parts.append(np.zeros(need, np.float32))
            x = np.concatenate(parts) if len(parts) > 1 else parts[0]
            br.poll_start()
            pcm = np.clip(np.round(x * 32768), -32768, 32767).astype("<i2")
            if br.local_mode != "off":
                # demo.js: RMS of the int16 mic channel /32768 (the reference channel is silent here).
                mic_rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768) ** 2)))
                br.local_pause(mic_rms, 0.0)
            inter = np.empty(FRAME * 2, "<i2")
            inter[0::2], inter[1::2] = pcm, 0
            link.send_bytes(INPUT_HEADER.pack(b"FDM1", k, FRAME, SR) + inter.tobytes())
            mic["sent"] = (k + 1) * FRAME
            k += 1
            br.tick_ping()

    async def ticker():
        while not stop_mic.is_set():
            br.tick()
            await asyncio.sleep(.005)

    def queued():
        return sum(len(q) for q in mic["queue"])

    async def wait_reply(t_input_end):
        """Wait until the reply finished and 1 s idle, or no reply/activity for no_reply_s."""
        done_before = br.replies_done
        cap = now() + a.reply_cap_s
        while now() < cap:
            await asyncio.sleep(.05)
            if br.busy():
                continue
            idle = now() - max(br.last_activity, t_input_end)
            if br.replies_done > done_before and idle >= a.idle_s:
                return "reply_done"
            if idle >= a.no_reply_s:
                return "no_reply"
        return "cap"

    tasks.append(asyncio.create_task(link.sender()))
    tasks.append(asyncio.create_task(receiver()))
    try:
        link.send_json("config", {"client": "humdial-web", "audio_protocol": "pcm16.v1",
                                  "input_protocol": "pcm16.ref.v1", "diagnostic_capture": False})
        await asyncio.wait_for(br.ready.wait(), 60)
        receipt["session_id"] = br.session_id
        link.send_json("input_settings", {"echoCancellation": False, "noiseSuppression": True,
                                          "autoGainControl": True, "sampleRate": 16000, "channelCount": 1})
        tasks.append(asyncio.create_task(mic_loop()))
        tasks.append(asyncio.create_task(ticker()))
        await asyncio.sleep(a.lead_s)
        for item in plan:
            # Align injection to the next packet boundary (audio clock = samples / 16 kHz).
            start_sample = mic["sent"] + queued()
            if start_sample % FRAME:
                pad = FRAME - start_sample % FRAME
                mic["queue"].append(np.zeros(pad, np.float32))
                start_sample += pad
            mic["queue"].append(item["audio"].astype(np.float32))
            info = {k: v for k, v in item.items() if k != "audio"}
            info.update(start_T=start_sample / SR, end_T=(start_sample + len(item["audio"])) / SR)
            receipt["items"].append(info)
            while queued():
                await asyncio.sleep(.02)
            await asyncio.sleep(FRAME_S * 2)
            info["wait_result"] = await wait_reply(now())
            info["wait_done_T"] = br.T()
            print(f"  {meta['name']} {item['label']} -> {info['wait_result']}", flush=True)
            if link.closed:
                br.errors.append("socket closed")
                break
        await asyncio.sleep(.5)
    except Exception as exc:
        br.errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if br.telemetry:
            s = br.speech
            br.emit("stop", **({"utterance_id": s.id, "played_samples": s.played} if s else {}))
        stop_mic.set()
        await asyncio.sleep(.2 + link.delay)
        link.out.put_nowait((now(), None))
        try:
            await ws.close()
        except Exception:
            pass
        for t in tasks:
            try:
                await asyncio.wait_for(t, 5)
            except Exception:
                t.cancel()
    late = np.array(mic["late"] or [0.0]) * 1000
    receipt.update(client_log=log, utterances=list(br.utt.values()), errors=br.errors, rtt_ms=br.rtts,
                   mic_late_ms={"n": len(mic["late"]), "p99": round(float(np.percentile(late, 99)), 2),
                                "max": round(float(late.max()), 2)},
                   samples_sent=mic["sent"], finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # Collect the backend trace once its writer has closed.
    if br.session_id:
        src = Path(a.backend_root) / "exp" / br.session_id / "realtimeout_live" / "events.jsonl"
        deadline = now() + 20
        while now() < deadline:
            rows = jrows(src)
            if rows and rows[-1].get("event") == "trace_closed":
                break
            await asyncio.sleep(.25)
        if src.exists():
            shutil.copyfile(src, out_dir / "events.jsonl")
            receipt["trace_closed"] = any(r.get("event") == "trace_closed" for r in jrows(out_dir / "events.jsonl"))
    jdump(out_dir / "receipt.json", receipt)
    return receipt


# --------------------------------------------------------------------------
# Isolated backend (backend-only; reuses the Omni on :10003)
# --------------------------------------------------------------------------
def set_dotted(cfg, key, value):
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def build_config(a, out):
    root = Path(a.backend_root)
    cfg = yaml.safe_load((root / "src/config.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load((root / "configs/demo_chat.yaml").read_text(encoding="utf-8"))
    for section, values in overlay.items():  # same merge as backend.load_runtime_config(demo_chat=True)
        cfg.setdefault(section, {}).update(values)
    overrides = dict(PROFILES.get(a.profile, {}))
    if a.hold_ms is not None:
        overrides["time.end_hold_frame"] = a.hold_ms / 1000
    if a.startup_ms is not None:
        overrides["engine.stream_startup_ms"] = a.startup_ms
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        overrides[k] = yaml.safe_load(v)
    overrides.setdefault("engine.diagnostics_retention_days", None)
    for k, v in overrides.items():
        set_dotted(cfg, k, v)
    path = out / "backend_config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path, overrides, cfg


def backend_env():
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    for k in ("QWEN_EXTRA_ARGS", "QWEN_DEPLOY_CONFIG", "FDBC_DEMO_RESIDUAL_OPT"):
        env.pop(k, None)
    ep = "http://127.0.0.1:10003/v1/chat/completions"
    env.update(QWEN_HOST="127.0.0.1", QWEN_PORT="10003", FDBC_PROXY_HOST="127.0.0.1", FDBC_PROXY_PORT="10004",
               FDBC_VLLM_URL=ep, FDBC_QWEN_URL=ep, FDBC_OMNI_TTS_URL=ep, FDBC_DEMO_VOICE_ADAPTER="1",
               FDBC_DEMO_INFERENCE_MODE="direct", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=env.get("OMP_NUM_THREADS") or "8", MKL_NUM_THREADS=env.get("MKL_NUM_THREADS") or "8")
    env.setdefault("FDBC_DEMO_TALKER_NUMERICS", "native")
    env.setdefault("FDBC_DEMO_CODEC_CHUNKS", "4:12")
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1,localhost"
    return env


def port_free(port):
    import socket
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def launch_backend(a, cfg_path, out):
    import urllib.request
    free_by = now() + 60  # a backend stopped just before may still hold the port briefly
    while not port_free(a.port):
        if now() > free_by:
            raise SystemExit(f"port {a.port} busy")
        time.sleep(1)
    log = (out / "backend.log").open("a")
    proc = subprocess.Popen([PY, "src/backend.py", "--config", str(cfg_path), "--streaming",
                             "--host", "127.0.0.1", "--port", str(a.port)], cwd=a.backend_root,
                            env=backend_env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = now() + 240
    while now() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"backend exited {proc.returncode}; see {out / 'backend.log'}")
        try:
            info = json.loads(opener.open(f"http://127.0.0.1:{a.port}/api/demo/info", timeout=2).read())
            if info.get("streaming") is True:
                return proc, log, info
        except Exception:
            pass
        time.sleep(1)
    stop_backend(proc, log)
    raise SystemExit("backend not ready")


def stop_backend(proc, log):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    log.close()


def cmd_run(a):
    out = Path(a.out) / a.profile
    out.mkdir(parents=True, exist_ok=True)
    cfg_path, overrides, cfg = build_config(a, out)
    rev = subprocess.run(["git", "-c", "safe.directory=*", "-C", a.backend_root, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-c", "safe.directory=*", "-C", a.backend_root, "status", "--short",
                            "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    meta = {"profile": a.profile, "backend_root": a.backend_root, "backend_rev": rev, "backend_dirty": dirty,
            "overrides": overrides, "end_hold_frame": cfg["time"]["end_hold_frame"],
            "stream_startup_ms": cfg["engine"].get("stream_startup_ms"), "net_delay_ms": a.net_delay_ms,
            "local_pause": a.local_pause, "release_lead_ms": a.release_lead_ms, "port": a.port,
            "suites": a.suites, "repeats": a.repeats, "argv": sys.argv, "timings": {}}
    meta_path = out / "run_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        meta["timings"] = old.get("timings", {})
    t_all = now()
    proc = log = None
    if not a.no_launch:
        t = now()
        proc, log, info = launch_backend(a, cfg_path, out)
        meta["timings"]["backend_start_s"] = round(now() - t, 1)
        meta["backend_info"] = info
    else:
        import urllib.request
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        info = json.loads(opener.open(f"http://127.0.0.1:{a.port}/api/demo/info", timeout=5).read())
    # The simulated player takes local-hold mode/params and resume lead from the backend
    # exactly like demo.js; absent (older backend) keeps the previous behavior.
    a.client_hold = info.get("client_hold") or {}
    meta["client_hold"] = a.client_hold
    url, origin = f"ws://127.0.0.1:{a.port}/realtime", f"http://127.0.0.1:{a.port}"
    try:
        if "synthetic" in a.suites:
            t = now()
            suite = ([(f"adhoc_{Path(w).stem}", "adhoc", [Path(w)]) for w in a.wav] if a.wav
                     else synthetic_suite(a))
            for name, group, paths in suite:
                if a.only and not any(name.startswith(o) for o in a.only):
                    continue
                for rep in range(a.repeats):
                    tdir = out / "trials" / f"syn_{name}_r{rep}"
                    if (tdir / "receipt.json").exists():
                        continue
                    plan = []
                    for i, p in enumerate(paths):
                        x = read16k(p)
                        plan.append({"label": f"{name}#{i}", "kind": "synthetic", "path": str(p),
                                     "audio": x, "segments": speech_segments(x)})
                    print(f"[{a.profile}] syn {name} r{rep}", flush=True)
                    asyncio.run(run_trial(a, url, origin, plan, tdir,
                                          {"name": f"syn_{name}_r{rep}", "suite": "synthetic", "scenario": name,
                                           "group": group, "repeat": rep}))
                    time.sleep(a.between_s)
            meta["timings"]["synthetic_s"] = round(meta["timings"].get("synthetic_s", 0) + now() - t, 1)
        if "real" in a.suites:
            t = now()
            man = json.loads(Path(a.episodes).read_text())
            for sess in man["sessions"]:
                if a.only and not any(sess["sid"].startswith(o) for o in a.only):
                    continue
                tdir = out / "trials" / f"real_{sess['sid']}"
                if (tdir / "receipt.json").exists() or not sess["episodes"]:
                    continue
                plan = [{"label": ep["name"], "kind": "real", "path": ep["path"], "audio": read16k(ep["path"]),
                         "segments": ep["segments"], "orig_start": ep["orig_start"]} for ep in sess["episodes"]]
                print(f"[{a.profile}] real {sess['sid']} episodes={len(plan)}", flush=True)
                asyncio.run(run_trial(a, url, origin, plan, tdir,
                                      {"name": f"real_{sess['sid']}", "suite": "real", "scenario": sess["sid"],
                                       "group": "real", "repeat": 0}))
                time.sleep(a.between_s)
            meta["timings"]["real_s"] = round(meta["timings"].get("real_s", 0) + now() - t, 1)
    finally:
        if proc is not None:
            stop_backend(proc, log)
        meta["timings"]["total_s"] = round(meta["timings"].get("total_s", 0) + now() - t_all, 1)
        jdump(meta_path, meta)
    analyze_dir(out)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _first(rows, pred):
    for r in rows:
        if pred(r):
            return r
    return None


def analyze_trial(tdir):
    rec = json.loads((tdir / "receipt.json").read_text())
    ev = jrows(tdir / "events.jsonl")
    utt = {u["utterance_id"]: u for u in rec.get("utterances", []) if u.get("utterance_id") is not None}
    items = rec.get("items", [])
    segs = []  # absolute audio-clock GT segments, tagged with item index
    for i, it in enumerate(items):
        for s in it.get("segments", []):
            segs.append({"item": i, "start": it["start_T"] + s["start"], "end": it["start_T"] + s["end"],
                         "end01": None if s.get("end_rms01") is None else it["start_T"] + s["end_rms01"]})
    D = lambda r: r.get("data") or {}
    by = {}
    for r in ev:
        by.setdefault(r["event"], []).append(r)
    rows = []
    dispatches = [r for r in by.get("input_dispatch", []) if D(r).get("closed")]
    for vd in by.get("vad_done", []):
        tv = vd["t_audio"]
        iid = D(vd).get("input_id")
        disp = _first(dispatches, lambda r: D(r).get("input_id") == iid and r["seq"] > vd["seq"]
                      and r["seq"] - vd["seq"] < 50)
        prev = [s for s in segs if s["start"] < tv and s["end"] <= tv + .02]
        gt = max(prev, key=lambda s: s["end"]) if prev else None
        before = [i for i, it in enumerate(items) if it["start_T"] <= tv]
        item = gt["item"] if gt else (before[-1] if before else None)
        row = {"trial": rec["name"], "suite": rec["suite"], "group": rec["group"], "scenario": rec["scenario"],
               "repeat": rec.get("repeat"), "input_id": iid, "vad_done_T": tv, "item": item,
               "item_label": items[item]["label"] if item is not None else None}
        if item is not None and rec["group"] == "history":
            row["group"] = "history" if item >= 1 else "history_t1"
        if gt:
            row.update(stop_T=round(gt["end"], 4), V_ms=round((tv - gt["end"]) * 1000, 1),
                       V_unmatched=(tv - gt["end"]) > 1.0)
            nxt = [s for s in segs if s["item"] == gt["item"] and s["start"] > gt["end"] + 1e-3]
            row["cont_T"] = round(min(s["start"] for s in nxt), 4) if nxt else None
            row["stop01_T"] = gt["end01"]
            stop_server = vd["server_ms"] - (tv - gt["end"]) * 1000
        else:
            row.update(stop_T=None, V_ms=None, V_unmatched=True, cont_T=None)
            stop_server = None
        if disp is None:
            row["dispatch"] = None
            rows.append(row)
            continue
        P = D(disp)["parent_id"]
        row.update(parent_id=P, playing=D(disp).get("playing"), context_samples=D(disp).get("context_audio_samples"))
        dec = _first(by.get("input_decision", []), lambda r: D(r).get("parent_id") == P)
        if dec:
            au = (D(dec).get("audit") or {}).get("route") or {}
            row.update(route=D(dec).get("route"), R_ms=round(dec["server_ms"] - disp["server_ms"], 1),
                       transcript=au.get("transcript"), route_fallback=au.get("fallback"),
                       route_timed_out=au.get("timed_out"))
        else:
            rev = D(disp).get("revision")
            stale = _first(by.get("input_decision_stale", []), lambda r: D(r).get("input_id") == iid
                           and D(r).get("revision") == rev)
            row["route"] = "stale" if stale else "superseded"
        rc = _first(by.get("model_call_dispatch", []), lambda r: D(r).get("kind") == "input_route"
                    and D(r).get("parent_id") == P)
        if rc:
            cid = D(rc)["call_id"]
            fo = _first(by.get("model_call_first_output", []), lambda r: D(r).get("call_id") == cid)
            dn = _first(by.get("model_call_done", []), lambda r: D(r).get("call_id") == cid)
            cap = _first(by.get("capacity_acquired", []), lambda r: D(r).get("call_id") == cid)
            row.update(route_ttft_ms=D(fo).get("elapsed_ms") if fo else None,
                       route_out_chunks=D(dn).get("output_chunks") if dn else None,
                       route_call_ms=D(dn).get("elapsed_ms") if dn else None,
                       route_capacity_wait_ms=D(cap).get("wait_ms") if cap else None)
        cc = _first(by.get("candidate_created", []), lambda r: D(r).get("parent_id") == P)
        if not cc:
            rows.append(row)
            continue
        C = D(cc)["candidate_id"]
        row["candidate_id"] = C
        resp = _first(by.get("model_call_dispatch", []), lambda r: D(r).get("kind") == "response"
                      and D(r).get("candidate_id") == C and D(r).get("parent_id") == P)
        U = D(resp).get("utterance_id") if resp else None
        ss = _first(by.get("speech_start", []), lambda r: D(r).get("candidate_id") == C and D(r).get("parent_id") == P)
        if U is None and ss:
            U = D(ss).get("utterance_id")
        row["utterance_id"] = U
        if resp:
            fo = _first(by.get("model_call_first_output", []), lambda r: D(r).get("call_id") == D(resp)["call_id"])
            row["response_ttft_ms"] = D(fo).get("elapsed_ms") if fo else None
        tc = _first(by.get("speech_timing", []), lambda r: D(r).get("utterance_id") == U
                    and D(r).get("phase") == "tts_chunk") if U is not None else None
        if tc:
            row["G_ms"] = round(tc["server_ms"] - cc["server_ms"], 1)
            row["RG_ms"] = round(tc["server_ms"] - disp["server_ms"], 1)
            if stop_server is not None:
                row["stop_to_audio_ready_ms"] = round(tc["server_ms"] - stop_server, 1)
        v640 = _first(by.get("vad_640_done", []), lambda r: D(r).get("candidate_id") == C and D(r).get("parent_id") == P)
        if v640:
            row["hold_release_after_vad_ms"] = round(v640["server_ms"] - vd["server_ms"], 1)
            if tc:
                row["hold_bound"] = tc["server_ms"] <= v640["server_ms"]
        cx = _first(by.get("candidate_cancelled", []), lambda r: D(r).get("candidate_id") == C and D(r).get("parent_id") == P)
        if cx:
            row["candidate_cancel_reason"] = D(cx).get("reason")
            row["candidate_published"] = D(cx).get("published")
        if ss:
            row["published"] = True
            sfa = _first(by.get("socket_first_audio_sent", []), lambda r: D(r).get("utterance_id") == U)
            if sfa and stop_server is not None:
                row["stop_to_socket_ms"] = round(sfa["server_ms"] - stop_server, 1)
            row["server_hold"] = any(D(r).get("utterance_id") == U and D(r).get("held") for r in by.get("speech_hold", []))
            sc = _first(by.get("speech_cancelled", []), lambda r: D(r).get("utterance_id") == U)
            row["speech_cancel_reason"] = D(sc).get("reason") if sc else None
        u = utt.get(U) if U is not None else None
        if u:
            row["client_held"] = any(h.get("held") and h.get("effective") for h in u.get("holds", []))
            row["local_held"] = bool(u.get("local_hold_at"))
            ps = u.get("play_start_at")
            row["played"] = ps is not None
            row["text_shown"] = u.get("text_at") is not None
            row["flash"] = bool(row.get("published") and u.get("text_at") is not None and ps is None)
            if row.get("stop_T") is not None:
                if u.get("first_packet_at") is not None:
                    row["stop_to_client_first_packet_ms"] = round((u["first_packet_at"] - row["stop_T"]) * 1000, 1)
                if u.get("initial_play_at") is not None:
                    row["stop_to_initial_playat_ms"] = round((u["initial_play_at"] - row["stop_T"]) * 1000, 1)
                if ps is not None:
                    row["stop_to_playat_ms"] = round((ps - row["stop_T"]) * 1000, 1)
                    if row.get("stop01_T") is not None:
                        row["stop01_to_playat_ms"] = round((ps - row["stop01_T"]) * 1000, 1)
                    ct = row.get("cont_T")
                    row["barge_audible"] = bool(ct is not None and ps <= ct < row["stop_T"] + 1.4)
                    row["played_despite_resume"] = bool(ct is not None and row["stop_T"] < ct < ps)
                    row["played_despite_resume_1400"] = bool(row["played_despite_resume"] and ct < row["stop_T"] + 1.4)
                    row["resume_gap_ms"] = None if ct is None else round((ct - row["stop_T"]) * 1000, 1)
        elif row.get("published"):
            row["played"] = False
            row["flash"] = False
        rows.append(row)
    return rec, rows


def pct(v, q):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(float(np.percentile(v, q)), 1) if v else None


GROUP_ORDER = ["short", "long", "cont", "history", "real", "real_first", "all_synth", "adhoc"]


def summarize(rows):
    out = {}
    groups = {g: [] for g in GROUP_ORDER}
    for r in rows:
        g = r["group"]
        if g in groups:
            groups[g].append(r)
        if r["suite"] == "synthetic" and g != "history_t1":
            groups["all_synth"].append(r)
        if g == "real" and r.get("first_in_item"):
            groups["real_first"].append(r)
    for g, rs in groups.items():
        lis = [r for r in rs if not r.get("playing") and r.get("parent_id")]
        cand = [r for r in lis if r.get("candidate_id") is not None]
        played = [r for r in cand if r.get("played")]
        pub = [r for r in cand if r.get("published")]
        hb = [r for r in cand if r.get("hold_bound") is not None]
        good_v = [r["V_ms"] for r in lis if r.get("V_ms") is not None and not r.get("V_unmatched")]
        s = {"ends_listening": len(lis), "candidates": len(cand), "published": len(pub), "played": len(played)}
        for key, src in (("V", good_v), ("R", [r.get("R_ms") for r in lis]),
                         ("route_ttft", [r.get("route_ttft_ms") for r in lis]),
                         ("route_tokens", [r.get("route_out_chunks") for r in lis]),
                         ("G", [r.get("G_ms") for r in cand]), ("RG", [r.get("RG_ms") for r in cand]),
                         ("stop_to_audio_ready", [r.get("stop_to_audio_ready_ms") for r in cand]),
                         ("stop_to_socket", [r.get("stop_to_socket_ms") for r in played]),
                         ("stop_to_playat", [r.get("stop_to_playat_ms") for r in played]),
                         ("stop01_to_playat", [r.get("stop01_to_playat_ms") for r in played])):
            s[key + "_p50"], s[key + "_p90"] = pct(src, 50), pct(src, 90)
        s["hold_bound"] = f"{sum(1 for r in hb if r['hold_bound'])}/{len(hb)}"
        s["hold_bound_frac"] = round(sum(1 for r in hb if r["hold_bound"]) / len(hb), 3) if hb else None
        s["server_hold"] = sum(1 for r in pub if r.get("server_hold"))
        s["local_hold"] = sum(1 for r in pub if r.get("local_held"))
        s["barge_audible"] = sum(1 for r in played if r.get("barge_audible"))
        s["played_despite_resume"] = sum(1 for r in played if r.get("played_despite_resume"))
        s["played_despite_resume_1400"] = sum(1 for r in played if r.get("played_despite_resume_1400"))
        s["flash"] = f"{sum(1 for r in pub if r.get('flash'))}/{len(pub)}"
        s["routes"] = {}
        for r in lis:
            s["routes"][str(r.get("route"))] = s["routes"].get(str(r.get("route")), 0) + 1
        out[g] = s
    return out


def analyze_dir(out):
    out = Path(out)
    all_rows, problems = [], []
    for tdir in sorted((out / "trials").glob("*")):
        if not (tdir / "receipt.json").exists():
            continue
        try:
            rec, rows = analyze_trial(tdir)
        except Exception as exc:
            problems.append({"trial": tdir.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        seen = set()
        for r in rows:
            key = r.get("item")
            r["first_in_item"] = key not in seen and not r.get("playing")
            if not r.get("playing"):
                seen.add(key)
        if rec.get("errors") or not rec.get("trace_closed"):
            problems.append({"trial": rec["name"], "errors": rec.get("errors"), "trace_closed": rec.get("trace_closed")})
        all_rows.extend(rows)
    with (out / "ends.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = summarize(all_rows)
    meta = json.loads((out / "run_meta.json").read_text()) if (out / "run_meta.json").exists() else {}
    jdump(out / "summary.json", {"profile": out.name, "meta": meta, "groups": summ, "problems": problems})
    (out / "summary.md").write_text(markdown([(out.name, summ, meta)]), encoding="utf-8")
    print((out / "summary.md").read_text())
    if problems:
        print("PROBLEMS", json.dumps(problems, ensure_ascii=False)[:2000])


COLS = [("n END/候选/开播", lambda s: f"{s['ends_listening']}/{s['candidates']}/{s['played']}"),
        ("V P50/P90", lambda s: f"{s['V_p50']}/{s['V_p90']}"),
        ("R P50/P90", lambda s: f"{s['R_p50']}/{s['R_p90']}"),
        ("R tok P50/P90", lambda s: f"{s['route_tokens_p50']}/{s['route_tokens_p90']}"),
        ("G P50/P90", lambda s: f"{s['G_p50']}/{s['G_p90']}"),
        ("R+G P50/P90", lambda s: f"{s['RG_p50']}/{s['RG_p90']}"),
        ("hold 卡住", lambda s: f"{s['hold_bound']}"),
        ("停嘴→socket P50/P90", lambda s: f"{s['stop_to_socket_p50']}/{s['stop_to_socket_p90']}"),
        ("停嘴→playAt P50/P90", lambda s: f"{s['stop_to_playat_p50']}/{s['stop_to_playat_p90']}"),
        ("speech_hold", lambda s: str(s["server_hold"])),
        ("实测抢话", lambda s: str(s["barge_audible"])),
        ("早于 playAt 续说仍开播（1.4 s 内）", lambda s: f"{s['played_despite_resume']}（{s['played_despite_resume_1400']}）"),
        ("闪现", lambda s: s["flash"])]


def markdown(runs):
    lines = []
    for name, summ, meta in runs:
        lines.append(f"### `{name}`（hold {meta.get('end_hold_frame')} s，起播 {meta.get('stream_startup_ms')} ms，"
                     f"耗时 {meta.get('timings', {}).get('total_s')} s）\n")
        lines.append("| 组 | " + " | ".join(c for c, _ in COLS) + " |")
        lines.append("|" + "---|" * (len(COLS) + 1))
        for g in GROUP_ORDER:
            s = summ.get(g)
            if not s or not s["ends_listening"]:
                continue
            lines.append(f"| {g} | " + " | ".join(str(f(s)) for _, f in COLS) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def cmd_analyze(a):
    analyze_dir(a.run_dir)


def cmd_report(a):
    runs = []
    for d in a.run_dirs:
        j = json.loads((Path(d) / "summary.json").read_text())
        runs.append((j["profile"], j["groups"], j.get("meta", {})))
    text = markdown(runs)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("episodes")
    e.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT))
    e.add_argument("--sessions", nargs="+", default=REAL_SESSIONS)
    e.add_argument("--max-episode-s", type=float, default=30.0)
    e.add_argument("--out", required=True)
    f = sub.add_parser("fixtures")
    f.add_argument("--backend-root", required=True)
    f.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--backend-root", required=True)
    r.add_argument("--profile", required=True, help="name; built-ins: " + ", ".join(PROFILES))
    r.add_argument("--hold-ms", type=float, help="override time.end_hold_frame (ms)")
    r.add_argument("--startup-ms", type=int, help="override engine.stream_startup_ms")
    r.add_argument("--set", action="append", help="extra dotted override, e.g. engine.stream_prefetch_ms=2000")
    r.add_argument("--port", type=int, default=18010)
    r.add_argument("--no-launch", action="store_true", help="use an already running backend on --port")
    r.add_argument("--suites", nargs="+", default=["synthetic", "real"], choices=["synthetic", "real"])
    r.add_argument("--only", nargs="*", help="scenario/session name prefixes")
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--wav", nargs="*", help="ad-hoc 16 kHz wavs replacing the synthetic suite (group adhoc)")
    r.add_argument("--src-root", default=str(DEFAULT_SRC_ROOT), help="where the synthetic inputs live")
    r.add_argument("--fixtures", required=False, default="", help="dir produced by `fixtures`")
    r.add_argument("--episodes", default="", help="episodes.json produced by `episodes`")
    r.add_argument("--out", required=True)
    r.add_argument("--net-delay-ms", type=float, default=0.0, help="artificial one-way delay, both directions")
    r.add_argument("--local-pause", action="store_true",
                   help="force the client-hold-v1 local hold on (default: info.client_hold.local_hold, as demo.js)")
    r.add_argument("--release-lead-ms", type=float, default=None,
                   help="restart lead after a hold release (default: info.client_hold.resume_lead_ms, as demo.js)")
    r.add_argument("--output-latency-ms", type=float, default=27.0)
    r.add_argument("--lead-s", type=float, default=1.0)
    r.add_argument("--idle-s", type=float, default=1.0)
    r.add_argument("--no-reply-s", type=float, default=8.0)
    r.add_argument("--reply-cap-s", type=float, default=120.0)
    r.add_argument("--between-s", type=float, default=0.5)
    an = sub.add_parser("analyze")
    an.add_argument("run_dir")
    rp = sub.add_parser("report")
    rp.add_argument("run_dirs", nargs="+")
    rp.add_argument("--out")
    a = p.parse_args()
    for k in PROXY_VARS:
        os.environ.pop(k, None)
    {"episodes": cmd_episodes, "fixtures": cmd_fixtures, "run": cmd_run, "analyze": cmd_analyze,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
