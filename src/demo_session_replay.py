"""Replay an opted-in browser demo session through the current ActorEngine.

The replay uses captured microphone/reference frames, recorded browser controls,
recorded VAD boundaries and saved model outputs.  It never calls a model.  At
speed=1 the saved first-output delays and input timing are wall paced; faster
speeds are useful for debugging but are not latency evidence.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
import json
from pathlib import Path
import time
import wave

import numpy as np

from demo_diagnostics import load_jsonl
from input_audio import INPUT_HEADER
from stream_transport import PCMChunk


SIGNATURE_EVENTS = frozenset({
    "input_decision", "input_decision_stale", "candidate_control_done",
    "candidate_confirmed", "candidate_cancelled", "speech_start",
    "speech_cancelled", "speech_playback_started", "turn_finished",
    "history_write", "history_remove", "llm_stale_dropped", "speech_error",
})
SIGNATURE_FIELDS = (
    "route", "label", "kind", "reason", "source", "role", "turn",
    "input_id", "revision", "candidate_id", "utterance_id",
)


class _NullVAD:
    def __call__(self, *_args, **_kwargs):
        return None

    def reset_states(self):
        return None


def _read_pcm16(path):
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"Replay track must be mono PCM16: {path}")
        rate = wav.getframerate()
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
    return data, rate


class MemoryTrace:
    def __init__(self):
        self.rows = []
        self.origin = time.perf_counter()

    def observe(self, event, data=None, **context):
        self.rows.append({"event": event, "server_ms": round((time.perf_counter()-self.origin)*1000, 3),
                          "data": dict(data or {}), **context})

    def sent(self, *args):
        return None

    def input_health(self, *args, **kwargs):
        return None

    def client(self, data):
        return data


class RecordedStreams:
    def __init__(self, session_id, cases_root, trace_rows, prompts, speed=1.0):
        self.speed = float(speed)
        self.prompts = prompts
        self.by_prompt = {value: key for key, value in prompts.items()}
        self.queues = defaultdict(deque)
        self.asr_outputs = deque(str((row.get("data") or {}).get("content", ""))
            for row in trace_rows if row.get("event") == "asr_done")
        starts = [r for r in trace_rows if r.get("event") == "model_case_started"]
        first_ms = {}
        dispatch_ms = {}
        for row in trace_rows:
            data = row.get("data") or {}
            call_id = data.get("call_id")
            if row.get("event") == "model_call_dispatch" and call_id:
                dispatch_ms[call_id] = row.get("server_ms", 0)
            elif row.get("event") == "model_call_first_output" and call_id:
                first_ms[call_id] = max(0, row.get("server_ms", 0) - dispatch_ms.get(call_id, row.get("server_ms", 0)))
        for row in starts:
            data = row.get("data") or {}
            case_id, kind = data.get("case_id"), data.get("kind")
            path = Path(cases_root) / "captures" / str(case_id)
            try:
                case = json.loads((path / "case.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (case.get("context") or {}).get("session_id") != session_id:
                continue
            case["_path"] = path
            call_id = (case.get("context") or {}).get("call_id")
            case["_first_ms"] = first_ms.get(call_id, min((case.get("outcome") or {}).get("elapsed_ms", 0), 2000))
            self.queues[kind].append(case)

    async def _take(self, kind):
        if not self.queues[kind]:
            raise RuntimeError(f"Recorded model queue exhausted: {kind}")
        case = self.queues[kind].popleft()
        delay = case.get("_first_ms", 0) / 1000 / self.speed
        if delay:
            await asyncio.sleep(delay)
        return case

    async def text(self, messages, **_):
        prompt = messages[0].get("content") if messages else None
        kind = self.by_prompt.get(prompt, "response")
        case = await self._take(kind)
        text = str((case.get("outcome") or {}).get("text", ""))
        if text:
            yield text

    async def audio(self, text, **_):
        case = await self._take("tts")
        path = case["_path"] / "output.wav"
        pcm, rate = _read_pcm16(path)
        chunk = max(1, int(rate * .12))
        for start in range(0, len(pcm), chunk):
            yield PCMChunk(pcm[start:start + chunk].tobytes(), rate)

    def asr(self, _path):
        return self.asr_outputs.popleft() if self.asr_outputs else ""

    def remaining(self):
        return {kind: len(queue) for kind, queue in self.queues.items() if queue}


class ReplayWebSocket:
    def __init__(self, capture_dir, trace_rows, *, speed=1.0):
        self.speed = float(speed)
        meta = load_jsonl(Path(capture_dir) / "frames.jsonl")
        mic, mic_rate = _read_pcm16(Path(capture_dir) / "browser_mic.wav")
        ref, ref_rate = _read_pcm16(Path(capture_dir) / "render_reference.wav")
        if mic_rate != 16000 or ref_rate != 16000 or len(mic) != len(ref):
            raise ValueError("Replay tracks must be aligned at 16 kHz")
        timeline = []
        for row in meta:
            start, count = row["offset"], row["samples"]
            frame_mic, frame_ref = mic[start:start+count], ref[start:start+count]
            interleaved = np.column_stack((frame_mic, frame_ref)).astype("<i2").tobytes()
            packet = INPUT_HEADER.pack(b"FDM1", row["seq"]-1, count, 16000) + interleaved
            timeline.append((float(row["t_audio"]), {"type": "websocket.receive", "bytes": packet}))
        for row in trace_rows:
            if row.get("event") not in {"playback_progress", "input_settings", "playback_stopped"}:
                continue
            timeline.append((float(row.get("t_audio", 0)), {"type": "websocket.receive", "text":
                json.dumps({"event": row["event"], "data": row.get("data") or {}})}))
        self.timeline = deque(sorted(timeline, key=lambda item: item[0]))
        self.base_audio = self.timeline[0][0] if self.timeline else 0
        self.origin = None
        self.sent_text = []
        self.sent_bytes = 0
        self.closed = False

    async def receive(self):
        if self.origin is None:
            self.origin = time.perf_counter()
        if not self.timeline:
            return {"type": "websocket.disconnect"}
        at, message = self.timeline.popleft()
        delay = self.origin + (at - self.base_audio) / self.speed - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        return message

    async def send_text(self, value):
        self.sent_text.append(json.loads(value))

    async def send_bytes(self, value):
        self.sent_bytes += len(value)

    async def close(self):
        self.closed = True


def _vad_script(rows):
    events = defaultdict(deque)
    for row in rows:
        if row.get("event") in {"vad_start", "vad_done"}:
            events[round(float(row.get("t_audio", 0)), 3)].append(
                {"start": row.get("t_audio")} if row["event"] == "vad_start" else {"end": row.get("t_audio")})
    return events


def signature(rows):
    result = []
    for row in rows:
        if row.get("event") not in SIGNATURE_EVENTS:
            continue
        data = row.get("data") or {}
        result.append({"event": row["event"], **{key: data.get(key) for key in SIGNATURE_FIELDS if key in data}})
    return result


async def replay_session(session_id, diagnostics_dir, cases_root, config, output_dir, *, speed=1.0):
    from engine import ActorEngine

    diagnostics_dir, output_dir = Path(diagnostics_dir), Path(output_dir)
    manifest = json.loads((diagnostics_dir / "manifest.json").read_text(encoding="utf-8"))
    capture = json.loads((diagnostics_dir / "capture" / "capture.json").read_text(encoding="utf-8"))
    if not manifest.get("capture", {}).get("enabled"):
        raise ValueError("Session has no opted-in replay capture")
    if capture.get("truncated"):
        raise ValueError("Replay capture is truncated; full-session replay is unavailable")
    rows = load_jsonl(diagnostics_dir.parent / "events.jsonl")
    streams = RecordedStreams(session_id, cases_root, rows, config["prompts"], speed)
    engine_cfg = dict(config.get("engine") or {})
    engine_cfg.update((manifest.get("effective") or {}).get("engine") or {})
    engine_cfg.update(stream_response=True, input_protocol="pcm16.ref.v1")
    memory = MemoryTrace()
    websocket = ReplayWebSocket(diagnostics_dir / "capture", rows, speed=speed)
    engine = ActorEngine(websocket=websocket, prompts=config["prompts"], delay=config["time"],
        llm_cfg=config.get("llm") or {}, engine_cfg=engine_cfg,
        llm_fn=lambda _messages: "", asr_fn=streams.asr, tts_fn=lambda text, path: path,
        text_stream_fn=streams.text, tts_stream_fn=streams.audio,
        vad_iterator=_NullVAD())
    engine.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.chmod(0o700)
    engine.demo_trace = memory
    scripted_vad = _vad_script(rows)

    def detect(_):
        queue = scripted_vad.get(round(engine.t_audio, 3))
        return queue.popleft() if queue else None
    engine.detect_vad_frame = detect
    await engine.run_realtime(websocket)
    original, replayed = signature(rows), signature(memory.rows)
    mismatches = []
    for index in range(max(len(original), len(replayed))):
        left = original[index] if index < len(original) else None
        right = replayed[index] if index < len(replayed) else None
        if left != right:
            mismatches.append({"index": index, "recorded": left, "replayed": right})
    report = {"version": "demo-session-replay-v1", "session_id": session_id,
        "speed": speed, "timing_comparable": speed == 1,
        "recorded_signature_events": len(original), "replayed_signature_events": len(replayed),
        "mismatch_count": len(mismatches), "mismatches": mismatches[:200],
        "remaining_model_cases": streams.remaining()}
    output = output_dir / "replay_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.chmod(0o600)
    return report
