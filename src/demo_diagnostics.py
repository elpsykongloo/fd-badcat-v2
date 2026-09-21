"""Private demo diagnostic sessions, replay audio, audits and human reviews.

The diagnostic store is observational.  It never authorizes an engine action and
never mutates ActorEngine state.  A session is the durable join boundary:

    manifest -> events -> turns -> model cases -> review revisions

Raw/browser audio capture is separately negotiated and disabled by default.
All files produced here are local, private and bounded; none are dataset gold
until a human review says so.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Iterable
from uuid import uuid4
import wave

import numpy as np


DIAGNOSTICS_VERSION = "demo-diagnostics-v1"
MANIFEST_VERSION = "demo-session-v1"
TURN_VERSION = "demo-turn-v1"
SPAN_VERSION = "demo-span-v1"
SUMMARY_VERSION = "demo-session-summary-v1"
REVIEW_VERSION = "demo-turn-review-v1"
CAPTURE_VERSION = "demo-audio-ring-v1"

REVIEW_LABELS = frozenset({
    "vad_or_truncation", "transcription", "route", "third_party_or_echo",
    "response_content", "tts_fidelity", "voice", "stutter_or_underrun",
    "cancellation", "history_pollution", "latency", "cannot_determine",
    "acceptable",
})


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _private_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def private_json(path: Path, value):
    """Atomically replace one private JSON file without integrity hashes."""
    _private_dir(path.parent)
    tmp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(tmp, flags, 0o600), "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    path.chmod(0o600)


def private_jsonl(path: Path, rows: Iterable[dict]):
    _private_dir(path.parent)
    tmp = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    path.chmod(0o600)


def repository_revision(root: Path):
    """Read the checked-out revision identifier; do not compute file hashes."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="ascii").strip()
        if head.startswith("ref: "):
            ref = head[5:]
            revision = (git / ref).read_text(encoding="ascii").strip()
            return {"ref": ref, "revision": revision[:12]}
        return {"ref": "detached", "revision": head[:12]}
    except (OSError, UnicodeError):
        return {"ref": "unknown", "revision": os.environ.get("FDBC_BUILD_REVISION", "unknown")[:64]}


def build_manifest(*, session_id, repository_root, profile, engine_cfg, delay,
                   llm_cfg, asr_cfg, capture_enabled=False, capture_seconds=180):
    """Return an allowlisted effective runtime manifest with no prompts/secrets."""
    engine_keys = (
        "chat_demo", "stream_response", "stream_packet_ms", "stream_buffer_ms",
        "stream_startup_ms", "stream_prefetch_ms", "stream_diagnostics",
        "playback_autoend", "control_validation", "speculative_response",
        "cancellable_response", "guarded_turns", "input_decision_timeout_s",
        "input_preroll_ms", "max_input_seconds", "request_total_limit",
        "normal_request_limit", "demo_voice_control", "demo_voice_speaker",
        "demo_voice_seed", "input_protocol", "case_capture",
        "case_capture_max_bytes", "case_capture_max_cases",
        "case_capture_kind_weights", "case_capture_success_sample_rate",
        "diagnostics_review", "diagnostics_retention_days",
        "diagnostic_audio_capture_allowed", "diagnostic_audio_capture_seconds",
    )
    llm_keys = ("model", "decision_timeout_s", "audio_block")
    asr_keys = ("backend", "provider", "num_threads")
    return {
        "version": MANIFEST_VERSION,
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "session_id": session_id,
        "created_utc": utc_now(),
        "profile": profile,
        "code": repository_revision(Path(repository_root)),
        "protocols": {
            "trace": "demo-trace-v2", "case": "demo-case-v2",
            "audio": "pcm16.v1", "input": engine_cfg.get("input_protocol"),
            "route": "transcript-first-v1", "reply": "played-reply-v1",
            "speech_reference": "speech-reference-v1",
            "tts": "verbatim-grammar-v2", "continuity": "demo-continuity-v1",
        },
        "effective": {
            "engine": {k: engine_cfg[k] for k in engine_keys if k in engine_cfg},
            "time": {**{k: delay[k] for k in ("end_hold_frame", "after_continue_time") if k in delay},
                     "long_interrupt_s": 1.5},
            "input_timing": {"vad_threshold": 0.5, "vad_silence_ms": 100,
                             "preroll_ms": int(engine_cfg.get("input_preroll_ms", 160))},
            "llm": {k: llm_cfg[k] for k in llm_keys if k in llm_cfg},
            "asr": {k: asr_cfg[k] for k in asr_keys if k in asr_cfg},
            "runtime": {
                "talker_numerics": os.environ.get("FDBC_DEMO_TALKER_NUMERICS", "off")[:32],
                "codec_chunks": os.environ.get("FDBC_DEMO_CODEC_CHUNKS", "default")[:32],
            },
        },
        "browser_audio": {},
        "capture": {
            "enabled": bool(capture_enabled), "version": CAPTURE_VERSION,
            "max_seconds": int(capture_seconds), "truncated": False,
        },
    }


@dataclass
class _CaptureFrame:
    seq: int
    t_audio: float
    mic: np.ndarray
    reference: np.ndarray | None
    clean: np.ndarray | None = None


class AudioRingCapture:
    """Bounded in-memory browser mic/reference/clean tracks for opted-in sessions."""

    def __init__(self, path, *, sample_rate=16000, max_seconds=180):
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.max_samples = self.sample_rate * int(max_seconds)
        self.frames = deque()
        self.samples = 0
        self.truncated = False
        self.settings = {}

    def raw(self, seq, t_audio, mic, reference=None):
        mic = np.asarray(mic, dtype=np.float32).copy()
        reference = None if reference is None else np.asarray(reference, dtype=np.float32).copy()
        frame = _CaptureFrame(int(seq), float(t_audio), mic, reference)
        self.frames.append(frame)
        self.samples += len(mic)
        while self.samples > self.max_samples and self.frames:
            self.samples -= len(self.frames.popleft().mic)
            self.truncated = True

    def clean(self, seq, pcm):
        if self.frames and self.frames[-1].seq == seq:
            self.frames[-1].clean = np.asarray(pcm, dtype=np.float32).copy()

    def input_settings(self, settings):
        self.settings = dict(settings)

    @staticmethod
    def _pcm16(parts):
        if not parts:
            return b""
        data = np.concatenate(parts)
        return (np.clip(data, -1, 32767 / 32768) * 32768).round().astype("<i2").tobytes()

    def _wav(self, path, parts):
        raw = self._pcm16(parts)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
            with wave.open(handle, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(self.sample_rate)
                wav.writeframes(raw)

    def close(self):
        _private_dir(self.path)
        rows = []
        offset = 0
        frames = list(self.frames)
        for frame in frames:
            rows.append({"seq": frame.seq, "t_audio": round(frame.t_audio, 6),
                         "offset": offset, "samples": len(frame.mic),
                         "has_reference": frame.reference is not None,
                         "has_clean": frame.clean is not None})
            offset += len(frame.mic)
        self._wav(self.path / "browser_mic.wav", [f.mic for f in frames])
        self._wav(self.path / "render_reference.wav", [f.reference if f.reference is not None
                  else np.zeros_like(f.mic) for f in frames])
        self._wav(self.path / "engine_clean.wav", [f.clean if f.clean is not None
                  else f.mic for f in frames])
        private_jsonl(self.path / "frames.jsonl", rows)
        meta = {"version": CAPTURE_VERSION, "sample_rate": self.sample_rate,
                "frames": len(rows), "samples": offset, "truncated": self.truncated,
                "first_seq": rows[0]["seq"] if rows else None,
                "last_seq": rows[-1]["seq"] if rows else None,
                "input_settings": self.settings}
        private_json(self.path / "capture.json", meta)
        return meta


def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _event_turn(row):
    data = row.get("data") or {}
    generation = row.get("generation", data.get("generation", 0))
    turn = data.get("turn", row.get("turn"))
    if type(turn) is not int:
        return None
    return int(generation or 0), turn


def _case_index(cases_root, session_id):
    result = {}
    captures = Path(cases_root) / "captures" if cases_root else None
    if not captures or not captures.exists():
        return result
    for case_file in captures.glob("*/case.json"):
        try:
            case = json.loads(case_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (case.get("context") or {}).get("session_id") == session_id:
            result[case["case_id"]] = {"path": str(case_file.parent), "case": case}
    return result


def build_turns(rows, cases):
    grouped = defaultdict(list)
    for row in rows:
        key = _event_turn(row)
        if key is not None:
            grouped[key].append(row)
    turns = []
    for (generation, turn), events in sorted(grouped.items()):
        data_rows = [(e.get("event"), e.get("data") or {}) for e in events]
        # Every observed event carries the engine's current turn as context.
        # Lifecycle/health-only sessions therefore must not become fake turns.
        turn_prefixes = ("vad_", "input_", "candidate_", "model_", "speech_",
                         "asr_", "history_", "turn_", "llm_", "shift_")
        session_input_events = {"input_settings", "input_health", "input_evidence"}
        if not any(isinstance(event, str) and event not in session_input_events
                   and event.startswith(turn_prefixes)
                   for event, _ in data_rows):
            continue
        case_ids = sorted({d.get("case_id") for _, d in data_rows if d.get("case_id")})
        call_ids = sorted({d.get("call_id") for _, d in data_rows if d.get("call_id")})
        input_ids = sorted({d.get("input_id") for _, d in data_rows if type(d.get("input_id")) is int})
        routes = [{"input_id": d.get("input_id"), "revision": d.get("revision"),
                   "route": d.get("route"), "audit": d.get("audit")}
                  for event, d in data_rows if event == "input_decision"]
        utterances = {}
        for event, data in data_rows:
            sid = data.get("utterance_id")
            if type(sid) is not int:
                continue
            item = utterances.setdefault(str(sid), {"utterance_id": sid})
            if event == "speech_start":
                item.update(started=True, parent_id=data.get("parent_id"))
            elif event == "speech_cancelled":
                item.update(cancelled=True, cancel_reason=data.get("reason"))
            elif event in {"speech_audio_end", "speech_played", "playback_progress",
                           "client_playback_end", "playback_stopped"}:
                for field in ("samples", "sent_samples", "played_samples", "received_samples", "underruns"):
                    if field in data:
                        item[field] = data[field]
                if event == "speech_played":
                    item["played_complete"] = True
        outcomes = []
        for event, data in data_rows:
            if event in {"input_admitted", "input_rejected", "input_waiting", "input_ignored",
                         "candidate_confirmed", "candidate_cancelled", "turn_finished",
                         "llm_stale_dropped", "speech_error", "engine_error"}:
                outcomes.append({"event": event, **{k: v for k, v in data.items()
                    if k not in {"content", "prompt", "text"}}})
        history = [{k: data.get(k) for k in (
                    "role", "source", "text", "utterance_id", "answer_id", "parent_id")
                    if data.get(k) is not None}
                   for event, data in data_rows if event in {"history_write", "history_remove"}]
        turns.append({
            "version": TURN_VERSION, "turn_id": f"g{generation}-t{turn}",
            "generation": generation, "turn": turn,
            "first_event_seq": events[0].get("seq"), "last_event_seq": events[-1].get("seq"),
            "first_server_ms": events[0].get("server_ms"), "last_server_ms": events[-1].get("server_ms"),
            "input_ids": input_ids, "call_ids": call_ids, "case_ids": case_ids,
            "routes": routes, "utterances": list(utterances.values()), "outcomes": outcomes,
            "history": history,
            "case_status": {case_id: (cases.get(case_id, {}).get("case", {}).get("outcome") or {}).get("status", "missing")
                            for case_id in case_ids},
        })
    return turns


def build_spans(rows):
    """Join request, utterance and clock-sync milestones on one server clock."""
    calls, utterances = {}, {}
    pings = defaultdict(dict)
    for row in rows:
        event, data = row.get("event"), row.get("data") or {}
        at = row.get("server_ms")
        call_id = data.get("call_id")
        if call_id:
            span = calls.setdefault(call_id, {"version": SPAN_VERSION, "span_type": "model",
                "call_id": call_id, "parent_id": data.get("parent_id"), "kind": data.get("kind"),
                "clock": "server_monotonic_ms", "milestones": {}})
            span["parent_id"] = span.get("parent_id") or data.get("parent_id")
            span["kind"] = span.get("kind") or data.get("kind")
            if event == "model_call_dispatch":
                span["milestones"]["dispatch"] = at
                span["transport"] = data.get("transport")
            elif event == "capacity_acquired":
                span["milestones"]["capacity_acquired"] = at
                span["capacity_wait_ms"] = data.get("wait_ms")
                span["capacity"] = data.get("capacity")
            elif event == "model_call_first_output":
                span["milestones"]["first_output"] = at
                span["first_output_ms"] = data.get("elapsed_ms")
                span["upstream_request_id"] = data.get("request_id")
                span["upstream_sse_received_ms"] = data.get("sse_received_ms")
            elif event == "model_call_done":
                span["milestones"]["done"] = at
                span["elapsed_ms"] = data.get("elapsed_ms")
                span["status"] = data.get("status")
                span["error_type"] = data.get("error_type")
            elif event == "model_case_started":
                span.setdefault("case_ids", []).append(data.get("case_id"))
        sid = data.get("utterance_id")
        if type(sid) is int:
            span = utterances.setdefault(sid, {"version": SPAN_VERSION, "span_type": "utterance",
                "utterance_id": sid, "parent_id": data.get("parent_id"),
                "clock": "server_monotonic_ms", "milestones": {}, "tts": []})
            span["parent_id"] = span.get("parent_id") or data.get("parent_id")
            milestone = {
                "speech_start": "start", "speech_text_done": "text_done",
                "speech_first_audio": "first_pcm", "socket_first_audio_sent": "first_socket_send",
                "speech_audio_end": "pcm_done", "speech_playback_started": "playback_started_ack",
                "speech_played": "playback_done_ack", "speech_cancelled": "cancel_requested",
                "playback_stopped": "playback_stopped_ack", "client_playback_start": "client_playback_start",
                "client_playback_end": "client_playback_end", "client_first_audio": "client_first_audio",
            }.get(event)
            if milestone:
                span["milestones"][milestone] = at
            if event == "speech_timing":
                span["tts"].append({k: v for k, v in data.items() if k != "utterance_id"})
            if event == "speech_cancelled":
                span["cancel_reason"] = data.get("reason")
            for field in ("samples", "played_samples", "received_samples", "underruns"):
                if field in data:
                    span[field] = data[field]
        if event in {"client_ping", "demo_pong", "client_rtt"} and type(data.get("seq")) is int:
            pings[data["seq"]][event] = (at, data)
    spans = list(calls.values()) + list(utterances.values())
    for seq, parts in sorted(pings.items()):
        if not {"client_ping", "demo_pong", "client_rtt"} <= parts.keys():
            continue
        t1, ping = parts["client_ping"]
        t2, _ = parts["demo_pong"]
        _, rtt = parts["client_rtt"]
        t0, t3 = ping.get("client_ms"), rtt.get("client_ms")
        if not all(type(v) in (int, float) for v in (t0, t1, t2, t3)):
            continue
        spans.append({"version": SPAN_VERSION, "span_type": "clock_sync", "seq": seq,
            "server_minus_client_ms": round(((t1-t0)+(t2-t3))/2, 3),
            "uncertainty_ms": round(max(0, ((t3-t0)-(t2-t1))/2), 3),
            "rtt_ms": rtt.get("rtt_ms"),
            "claim": "offset interval only; not physical playout or one-way network latency"})
    return spans


def audit_session(rows, turns, cases, manifest, spans=None):
    findings = []

    def add(code, severity, message, **context):
        findings.append({"code": code, "severity": severity, "message": message, **context})

    if not rows or rows[-1].get("event") != "trace_closed":
        add("trace_not_closed", "error", "Trace has no closing record")
    elif rows[-1].get("dropped", 0):
        add("trace_dropped", "error", "Trace writer dropped events", dropped=rows[-1]["dropped"])
    seqs = [r.get("seq") for r in rows if r.get("event") != "trace_closed"]
    if seqs and seqs != list(range(1, len(seqs) + 1)):
        add("event_sequence_gap", "error", "Event sequence is not contiguous")

    started = {}
    queued = {}
    call_done = set()
    confirmed_candidates = set()
    cancelled_at = {}
    last_played = defaultdict(int)
    sent = defaultdict(int)
    response_block = None
    stale_inputs = set()
    tts_proof_at = {}
    final = None
    for index, row in enumerate(rows):
        event, data = row.get("event"), row.get("data") or {}
        if event == "model_case_started":
            started[data.get("case_id")] = row
        elif event == "model_case_queued":
            queued[data.get("case_id")] = data
        elif event == "model_call_done":
            call_done.add(data.get("call_id"))
        elif event == "candidate_confirmed":
            confirmed_candidates.add(data.get("candidate_id"))
        elif event == "speech_start" and data.get("candidate_id") is not None:
            if data.get("candidate_id") not in confirmed_candidates:
                add("private_candidate_published", "error", "Candidate was published before confirmation",
                    candidate_id=data.get("candidate_id"))
            if response_block is not None:
                add("response_without_ready_input", "error",
                    "A new response started after stop/wait without a ready input",
                    route=response_block, utterance_id=data.get("utterance_id"))
        elif event == "speech_start" and response_block is not None:
            add("response_without_ready_input", "error",
                "A new response started after stop/wait without a ready input",
                route=response_block, utterance_id=data.get("utterance_id"))
        elif event == "speech_cancelled":
            cancelled_at[data.get("utterance_id")] = index
        elif event in {"speech_first_audio", "socket_first_audio_sent"}:
            sid = data.get("utterance_id")
            if sid in cancelled_at and index > cancelled_at[sid]:
                add("audio_after_cancel", "error", "Audio appeared after utterance cancellation", utterance_id=sid)
        elif event in {"playback_progress", "playback_stopped", "speech_played", "client_playback_end"}:
            sid = data.get("utterance_id")
            played = data.get("played_samples")
            if type(sid) is int and type(played) is int:
                if played < last_played[sid]:
                    add("playback_regressed", "error", "Playback samples moved backwards", utterance_id=sid)
                last_played[sid] = max(last_played[sid], played)
                upper = data.get("received_samples", sent.get(sid, 0))
                if upper and played > upper:
                    add("playback_exceeds_sent", "error", "Playback exceeds sent/received samples", utterance_id=sid)
        elif event == "speech_audio_end":
            sent[data.get("utterance_id")] = data.get("samples", 0)
        elif event == "session_final":
            final = data
        elif event == "session_reset":
            response_block = None

        if event == "capacity_acquired":
            cap = data.get("capacity") or {}
            if cap.get("active_total", 0) > cap.get("total_limit", 10**9):
                add("capacity_total_exceeded", "error", "Total request capacity exceeded")
            if cap.get("active_normal", 0) > cap.get("normal_limit", 10**9):
                add("capacity_normal_exceeded", "error", "Normal request capacity exceeded")
        elif event == "input_decision":
            route = data.get("route")
            response_block = route if route in {"stop_only", "yield_wait"} else (
                None if route == "yield_ready" else response_block)
            if (data.get("input_id"), data.get("revision")) in stale_inputs:
                add("stale_input_mutated_state", "error", "A stale input revision later produced a decision",
                    input_id=data.get("input_id"), revision=data.get("revision"))
        elif event == "input_decision_stale":
            stale_inputs.add((data.get("input_id"), data.get("revision")))
        elif event == "speech_timing" and data.get("phase") == "tts_chunk" \
                and data.get("text_verified_ms") is not None:
            tts_proof_at.setdefault(data.get("utterance_id"), index)
        elif event == "speech_first_audio":
            sid = data.get("utterance_id")
            if (manifest.get("effective") or {}).get("engine", {}).get("stream_diagnostics"):
                if tts_proof_at.get(sid, index + 1) >= index:
                    add("pcm_before_tts_proof", "error", "First PCM had no earlier literal-text proof",
                        utterance_id=sid)
        elif event == "history_write" and data.get("role") == "assistant":
            text = str(data.get("text", ""))
            if "此回答已被打断，其余内容未确认播完" in text:
                add("internal_metadata_in_history", "error", "Internal interruption metadata entered assistant history",
                    turn=data.get("turn"))
        elif event in {"engine_error", "speech_error", "model_case_error"}:
            add(event, "error", "Runtime diagnostic error", error_type=data.get("error_type") or data.get("type"))
        elif event in {"client_underrun", "socket_slow_send", "llm_timeout"}:
            add(event, "warning", "Runtime health anomaly", utterance_id=data.get("utterance_id"))
        elif event == "control_validation":
            if data.get("fallback") or data.get("timed_out") or data.get("repaired"):
                add("control_repair_or_fallback", "warning", "Control output required repair/fallback",
                    kind=data.get("kind"), fallback=data.get("fallback"), repaired=data.get("repaired"),
                    timed_out=data.get("timed_out"))
        elif event == "model_call_done" and data.get("status") == "error":
            add("model_call_error", "error", "Model call failed", call_id=data.get("call_id"),
                kind=data.get("kind"), error_type=data.get("error_type"))

    for case_id in set(started) - set(queued):
        add("case_not_queued", "warning", "Started model case was not queued", case_id=case_id)
    for case_id, data in queued.items():
        if data.get("queued") and case_id not in cases:
            add("case_not_saved", "warning", "Queued model case is absent after archive flush", case_id=case_id)
        elif data.get("queued") is False:
            sample_rate = ((manifest.get("effective") or {}).get("engine") or {}).get(
                "case_capture_success_sample_rate", 1)
            sampled = data.get("status") == "completed" and sample_rate < 1
            add("case_not_retained", "info" if sampled else "warning",
                "Model case was explicitly declined by archive policy",
                case_id=case_id, status=data.get("status"), sampled=sampled)
    for row in rows:
        if row.get("event") == "model_call_dispatch":
            call_id = (row.get("data") or {}).get("call_id")
            if call_id and call_id not in call_done:
                add("call_without_terminal", "warning", "Model call has no terminal event", call_id=call_id)

    for turn in turns:
        seen = defaultdict(set)
        for route in turn["routes"]:
            key = (route.get("input_id"), route.get("revision"))
            seen[key].add(route.get("route"))
        for key, values in seen.items():
            if len(values) > 1:
                add("multiple_final_routes", "error", "One input revision has conflicting routes",
                    turn_id=turn["turn_id"], input_id=key[0], revision=key[1])

    if final is None:
        add("session_final_missing", "warning", "Session cleanup snapshot is absent")
    else:
        if final.get("state") != "LISTEN":
            add("illegal_final_state", "error", "Actor did not finish in LISTEN", state=final.get("state"))
        if final.get("pending_tasks", 0) or final.get("active_capacity", 0):
            add("resources_not_drained", "warning", "Tasks or capacity remained at session close",
                pending_tasks=final.get("pending_tasks"), active_capacity=final.get("active_capacity"))

    anomalies = [f for f in findings if f["severity"] in {"warning", "error"}]
    return {
        "version": SUMMARY_VERSION, "session_id": manifest.get("session_id"),
        "created_utc": utc_now(), "trace_events": len(rows), "turns": len(turns),
        "cases": len(cases), "spans": len(spans or []),
        "findings": findings, "anomalies": anomalies,
        "status": "error" if any(f["severity"] == "error" for f in findings)
                  else "warning" if anomalies else "ok",
    }


def finalize_session(diagnostics_dir, events_path, cases_root):
    diagnostics_dir, events_path = Path(diagnostics_dir), Path(events_path)
    manifest_path = diagnostics_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_jsonl(events_path)
    cases = _case_index(cases_root, manifest["session_id"])
    turns = build_turns(rows, cases)
    spans = build_spans(rows)
    summary = audit_session(rows, turns, cases, manifest, spans)
    private_jsonl(diagnostics_dir / "turns.jsonl", turns)
    private_jsonl(diagnostics_dir / "spans.jsonl", spans)
    private_json(diagnostics_dir / "summary.json", summary)
    return summary


class DiagnosticStore:
    """Read/review/prune access shared by the local API and CLI."""

    def __init__(self, exp_root="exp", cases_root="exp/demo_cases"):
        self.exp_root = Path(exp_root)
        self.cases_root = Path(cases_root)

    def _session_dirs(self):
        if not self.exp_root.exists():
            return []
        return sorted(self.exp_root.glob("web-demo-*/realtimeout_live/diagnostics"), reverse=True)

    def session_dir(self, session_id):
        if not isinstance(session_id, str) or not session_id.startswith("web-demo-") \
                or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-" for ch in session_id):
            raise ValueError("Invalid session ID")
        target = (self.exp_root / session_id / "realtimeout_live" / "diagnostics").resolve()
        if target.parent.parent.parent != self.exp_root.resolve():
            raise ValueError("Session path escapes diagnostics root")
        return target

    def list_sessions(self, *, anomalies_only=False):
        sessions = []
        for path in self._session_dirs():
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                summary = json.loads((path / "summary.json").read_text(encoding="utf-8")) \
                    if (path / "summary.json").exists() else {"status": "active", "anomalies": []}
                reviews = sum(1 for _ in (path / "reviews").glob("*/*.json")) if (path / "reviews").exists() else 0
            except (OSError, ValueError):
                continue
            item = {"session_id": manifest.get("session_id"), "created_utc": manifest.get("created_utc"),
                    "status": summary.get("status"), "turns": summary.get("turns"),
                    "cases": summary.get("cases"), "anomaly_count": len(summary.get("anomalies", [])),
                    "reviews": reviews, "capture": manifest.get("capture", {})}
            if not anomalies_only or item["anomaly_count"]:
                sessions.append(item)
        return sessions

    def session_detail(self, session_id):
        path = self.session_dir(session_id)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        turns = load_jsonl(path / "turns.jsonl")
        spans = load_jsonl(path / "spans.jsonl")
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8")) \
            if (path / "summary.json").exists() else {"status": "active", "anomalies": []}
        cases = _case_index(self.cases_root, session_id)
        case_view = {}
        for case_id, item in cases.items():
            case = item["case"]
            outcome = case.get("outcome") or {}
            case_view[case_id] = {"case_id": case_id, "kind": case.get("kind"),
                "context": case.get("context"), "status": outcome.get("status"),
                "elapsed_ms": outcome.get("elapsed_ms"), "text": outcome.get("text"),
                "tts_expected_text": case.get("tts_expected_text"),
                "audio": [p.name for p in Path(item["path"]).glob("*.wav")]}
        reviews = []
        review_root = path / "reviews"
        if review_root.exists():
            for review in sorted(review_root.glob("*/*.json")):
                try:
                    reviews.append(json.loads(review.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    pass
        return {"manifest": manifest, "summary": summary, "turns": turns, "spans": spans,
                "cases": case_view, "reviews": reviews}

    def add_review(self, session_id, turn_id, *, labels, note, reviewer):
        path = self.session_dir(session_id)
        known = {t.get("turn_id") for t in load_jsonl(path / "turns.jsonl")}
        if turn_id not in known:
            raise ValueError("Unknown turn ID")
        labels = sorted(set(labels or []))
        if not labels or any(label not in REVIEW_LABELS for label in labels):
            raise ValueError("Unknown or empty review labels")
        if not isinstance(note, str) or not note.strip() or len(note) > 4000:
            raise ValueError("A bounded review note is required")
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 80:
            raise ValueError("A bounded reviewer name is required")
        revision = len(list((path / "reviews" / turn_id).glob("*.json"))) + 1
        review = {"version": REVIEW_VERSION, "review_id": uuid4().hex,
                  "session_id": session_id, "turn_id": turn_id, "revision": revision,
                  "labels": labels, "note": note.strip(), "reviewer": reviewer.strip(),
                  "utc": utc_now()}
        target = path / "reviews" / turn_id / f"{revision:04d}-{review['review_id']}.json"
        private_json(target, review)
        return review

    def case_audio(self, case_id, name):
        if not isinstance(case_id, str) or not case_id or Path(case_id).name != case_id:
            raise ValueError("Invalid case ID")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".wav"):
            raise ValueError("Invalid audio name")
        path = (self.cases_root / "captures" / case_id / name).resolve()
        if path.parent.parent != (self.cases_root / "captures").resolve() or not path.is_file():
            raise FileNotFoundError(path)
        return path

    def delete_session(self, session_id):
        diag = self.session_dir(session_id)
        if not diag.exists():
            raise FileNotFoundError(session_id)
        for item in _case_index(self.cases_root, session_id).values():
            shutil.rmtree(item["path"])
        container = diag.parent.parent
        shutil.rmtree(container)

    def prune(self, retention_days):
        days = int(retention_days)
        if days < 1:
            raise ValueError("retention_days must be >= 1")
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        removed = []
        for item in self.list_sessions():
            try:
                created = datetime.fromisoformat(item["created_utc"])
            except (TypeError, ValueError):
                continue
            if created < cutoff:
                self.delete_session(item["session_id"])
                removed.append(item["session_id"])
        return removed
