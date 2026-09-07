"""Demo-only, bounded, fail-open diagnostics; never used for turn decisions.

All server durations use one monotonic clock. Client numbers are untrusted
measurements from another clock, never subtracted from server timestamps.
"""
import asyncio
import json
import math
import queue
import struct
import threading
import time
from datetime import datetime, timezone


class DemoTrace:
    def __init__(self, path, *, clock=time.perf_counter, capacity=4096):
        self.path, self.clock = path, clock
        self.origin = clock()
        self.queue = queue.Queue(maxsize=capacity)
        self.closed = threading.Event()
        self.dropped = 0
        self.error = None
        self.anchor = None
        self.replies = {}
        self.client_tokens = 20.0
        self.last_client = clock()
        self.last_health = -math.inf
        self.thread = threading.Thread(target=self._write, daemon=True, name="demo-trace")
        self.thread.start()
        self.record("session_start", {"version": "demo-trace-v1", "utc":
                    datetime.now(timezone.utc).isoformat()})

    def record(self, event, data=None, **context):
        if self.closed.is_set():
            return
        rec = {"event": event, "server_ms": round((self.clock() - self.origin) * 1000, 3),
               "data": dict(data or {}), **context}
        try:
            self.queue.put_nowait(rec)
        except queue.Full:
            self.dropped += 1

    def observe(self, event, data, **context):
        self.record(event, data, **context)
        now = (self.clock() - self.origin) * 1000
        key = (context.get("generation"), context.get("epoch"))
        if event == "vad_done":
            self.anchor = {"key": key, "vad_ms": now, "hold_ms": None}
        elif event == "vad_640_done" and self.anchor and self.anchor["key"] == key:
            self.anchor["hold_ms"] = now
        elif event == "speech_start":
            anchor = self.anchor if self.anchor and self.anchor["key"] == key else {}
            # Bounded even if every reply is cancelled before receiving audio.
            if len(self.replies) >= 64:
                self.replies.pop(next(iter(self.replies)))
            self.replies[data["utterance_id"]] = {**anchor, "start_ms": now}
        elif event == "speech_cancelled":
            self.replies.pop(data.get("utterance_id"), None)
        elif event == "speech_first_audio":
            timing = self.replies.pop(data.get("utterance_id"), None)
            if timing:
                vad, hold, start = timing.get("vad_ms"), timing.get("hold_ms"), timing["start_ms"]
                def diff(end, begin):
                    return round(end - begin, 1) if end is not None and begin is not None else None
                return {"utterance_id": data["utterance_id"],
                        "hold_ms": diff(hold, vad), "decision_ms": diff(start, hold),
                        "generation_ms": diff(now, start), "vad_to_audio_ms": diff(now, vad)}
        return None

    def client(self, data):
        """Allow numeric telemetry only, with a rate/size bound. Never ingest text/audio."""
        if not isinstance(data, dict):
            return None
        now = self.clock()
        self.client_tokens = min(20, self.client_tokens + (now - self.last_client) * 4)
        self.last_client = now
        if self.client_tokens < 1:
            return None
        self.client_tokens -= 1
        event = data.get("kind")
        if not isinstance(event, str) or event not in {"ping", "rtt", "first_audio", "playback_end", "cancel", "stop"}:
            return None
        fields = ("seq", "utterance_id", "client_ms", "rtt_ms", "first_audio_ms",
                  "scheduled_lead_ms", "base_latency_ms", "output_latency_ms",
                  "played_samples", "received_samples", "underruns", "upload_buffer_bytes")
        clean = {"kind": event}
        for key in fields:
            value = data.get(key)
            if type(value) in (int, float) and 0 <= value <= 1e12 and math.isfinite(value):
                clean[key] = value
        if event == "ping" and (type(clean.get("seq")) is not int):
            return None
        self.record("client_" + event, clean)
        return clean

    def sent(self, payload, queued, started, finished):
        data = {"queue_ms": round((started - queued) * 1000, 3),
                "send_ms": round((finished - started) * 1000, 3)}
        first = False
        if isinstance(payload, bytes) and len(payload) >= 16:
            magic, sid, seq, _ = struct.unpack_from("<4sIII", payload)
            if magic == b"FDS1":
                data.update(utterance_id=sid, packet_seq=seq)
                first = seq == 0
        if first or finished - queued >= .05:
            self.record("socket_first_audio_sent" if first else "socket_slow_send", data)

    def input_health(self, received_at, **context):
        if received_at is None:  # offline/injected frames have no receipt clock
            return
        now = self.clock()
        if now - self.last_health >= 1:
            self.last_health = now
            self.record("input_health", {"reader_to_actor_ms":
                        round((now - received_at) * 1000, 3)}, **context)

    def _write(self):
        try:
            # Dedicated thread; no disk I/O in the perception or socket tasks.
            with self.path.open("x", encoding="utf-8", buffering=1) as handle:
                while not self.closed.is_set() or not self.queue.empty():
                    try:
                        rec = self.queue.get(timeout=.1)
                    except queue.Empty:
                        continue
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                handle.write(json.dumps({"event": "trace_closed", "dropped": self.dropped}) + "\n")
        except Exception as exc:
            self.error = type(exc).__name__
            print(f"Demo trace unavailable ({self.error}): {self.path}", flush=True)

    async def close(self):
        self.closed.set()
        await asyncio.to_thread(self.thread.join, 2)
        if self.thread.is_alive():
            print(f"Demo trace writer still draining: {self.path}", flush=True)
