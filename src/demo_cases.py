"""Private, bounded recordings of actual demo model calls (not gold labels).

No filesystem work in the streaming task. Completed, failed and cancelled calls
are queued to one process-owned writer; quota/full/disk failures never affect
conversation decisions. Hard process termination can lose in-flight/queued cases.
"""
import asyncio
import base64
import copy
from collections import defaultdict
import io
import json
import os
from pathlib import Path
import queue
import random
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import wave

VERSION = "demo-case-v2"
SUPPORTED_VERSIONS = frozenset({"demo-case-v1", VERSION})
MAX_REQUEST = 4 * 1024 * 1024
MAX_OUTPUT = 2 * 1024 * 1024


def private_write(path, data):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        f.write(data)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def request_size(obj):
    # Bound before copying/queueing; UTF-8 upper bound without serializing audio.
    if isinstance(obj, str):
        return len(obj) * 4
    if isinstance(obj, dict):
        return sum(request_size(k) + request_size(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return sum(map(request_size, obj))
    return 32


def pack_audio(payload):
    """Externalize exact WAV bytes; restore_request reverses this losslessly."""
    payload = copy.deepcopy(payload)
    files = {}
    for message in payload.get("messages", []):
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "audio_url":
                holder, field = block["audio_url"], "url"
            elif block.get("type") == "input_audio":
                holder, field = block["input_audio"], "data"
            else:
                continue
            encoded = holder[field]
            prefix = ""
            if encoded.startswith("data:audio/") and ";base64," in encoded:
                prefix, encoded = encoded.split(",", 1)
                prefix += ","
            elif field == "url":
                raise ValueError("Only inline audio can be archived/replayed")
            raw = base64.b64decode(encoded, validate=True)
            if not (raw.startswith(b"RIFF") and raw[8:12] == b"WAVE"):
                raise ValueError("Case audio must be WAV")
            name = f"input-{len(files):02d}.wav"
            files[name] = raw
            holder[field] = {"case_file": name, "data_prefix": prefix}
    return payload, files


def load_case(path):
    path = Path(path).resolve()
    case = json.loads((path / "case.json").read_text())
    if case.get("version") not in SUPPORTED_VERSIONS or case.get("case_id") != path.name:
        raise ValueError("Unsupported case or mismatched case ID")
    return case


def restore_request(path, case=None):
    path = Path(path).resolve()
    case = case or load_case(path)
    payload = copy.deepcopy(case["request"])
    for message in payload.get("messages", []):
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "audio_url":
                holder, field = block["audio_url"], "url"
            elif block.get("type") == "input_audio":
                holder, field = block["input_audio"], "data"
            else:
                continue
            ref = holder[field]
            name = ref["case_file"]
            target = (path / name).resolve()
            if target.parent != path or not name.startswith("input-") or not name.endswith(".wav"):
                raise ValueError("Audio reference escapes the case")
            prefix = ref["data_prefix"]
            if prefix and prefix != "data:audio/wav;base64,":
                raise ValueError("Unsupported audio data prefix")
            holder[field] = prefix + base64.b64encode(target.read_bytes()).decode("ascii")
    return payload


class CaseCall:
    def __init__(self, archive, kind, payload, context, expected_text):
        self.archive = archive
        self.start = time.perf_counter()
        self.case = {"version": VERSION, "case_id": datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:12],
            "utc": datetime.now(timezone.utc).isoformat(), "kind": kind,
            "request": copy.deepcopy(payload), "context": copy.deepcopy(context),
            "expected": None, "tts_expected_text": expected_text,
            "boundary": "model-input-after-evidence-filter; output-before-playback; single-call"}
        self.text, self.pcm, self.rate = [], bytearray(), None
        self.output_bytes = 0
        self.truncated = False

    def feed(self, item):
        if isinstance(item, str):
            size = len(item.encode("utf-8"))
            if self.output_bytes + size <= MAX_OUTPUT and not self.truncated:
                self.text.append(item)
            else:
                self.truncated = True
        else:
            size = len(item.pcm)
            if self.rate is not None and self.rate != item.sample_rate:
                self.truncated = True
            self.rate = self.rate or item.sample_rate
            if self.output_bytes + size <= MAX_OUTPUT and not self.truncated:
                self.pcm.extend(item.pcm)
            else:
                self.truncated = True
        self.output_bytes += size

    def finish(self, status, error=None):
        self.case["outcome"] = {"status": status, "error_type": error,
            "elapsed_ms": round((time.perf_counter() - self.start) * 1000, 3),
            "text": "".join(self.text), "output_truncated": self.truncated,
            "observed_output_bytes": self.output_bytes,
            "sample_rate": self.rate, "saved_samples": len(self.pcm) // 2,
            "audio": "output.wav" if self.pcm else None}
        return self.archive.submit(self.case, self.pcm)


class CaseArchive:
    def __init__(self, root, *, max_bytes=536870912, max_cases=2000, capacity=8,
                 kind_weights=None, success_sample_rate=1.0, retention_days=None,
                 random_fn=random.random):
        self.root = Path(root)
        self.max_bytes, self.max_cases = max_bytes, max_cases
        if not 0 < float(success_sample_rate) <= 1:
            raise ValueError("success_sample_rate must be in (0, 1]")
        self.success_sample_rate = float(success_sample_rate)
        self.retention_days = None if retention_days is None else int(retention_days)
        if self.retention_days is not None and self.retention_days < 1:
            raise ValueError("retention_days must be >= 1")
        weights = dict(kind_weights or {})
        if weights and (any(float(v) <= 0 for v in weights.values()) or sum(map(float, weights.values())) > 1.000001):
            raise ValueError("kind_weights must be positive and sum to <= 1")
        self.kind_limits = {str(k): int(self.max_bytes * float(v)) for k, v in weights.items()}
        self.random_fn = random_fn
        self.queue = queue.Queue(maxsize=capacity)
        self.closed = threading.Event()
        self.saved = self.dropped = self.used_bytes = self.existing_cases = 0
        self.used_by_kind = defaultdict(int)
        self.dropped_by_reason = defaultdict(int)
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True, name="demo-cases")
        self.thread.start()

    def stats(self):
        return {"version": VERSION, "saved": self.saved, "dropped": self.dropped,
                "queued": self.queue.qsize(), "error": self.error,
                "used_bytes": self.used_bytes, "max_bytes": self.max_bytes,
                "max_cases": self.max_cases, "used_by_kind": dict(self.used_by_kind),
                "kind_limits": self.kind_limits,
                "dropped_by_reason": dict(self.dropped_by_reason),
                "success_sample_rate": self.success_sample_rate,
                "retention_days": self.retention_days}

    def begin(self, kind, payload, context, expected_text=None):
        reason = None
        if self.closed.is_set():
            reason = "archive_closed"
        elif self.error:
            reason = "writer_error"
        elif request_size(payload) > MAX_REQUEST:
            reason = "request_too_large"
        if reason is not None:
            self.dropped += 1
            self.dropped_by_reason[reason] += 1
            return None
        return CaseCall(self, kind, payload, context, expected_text)

    def submit(self, case, pcm):
        if (case.get("outcome") or {}).get("status") == "completed" \
                and getattr(self, "success_sample_rate", 1) < 1 \
                and self.random_fn() > self.success_sample_rate:
            self.dropped += 1
            if hasattr(self, "dropped_by_reason"):
                self.dropped_by_reason["success_sampling"] += 1
            return False
        if not self.closed.is_set() and not self.error:
            try:
                self.queue.put_nowait((case, pcm))
                return True
            except queue.Full:
                pass
        self.dropped += 1
        if hasattr(self, "dropped_by_reason"):
            self.dropped_by_reason["queue_or_closed"] += 1
        return False

    def _prune_expired(self, captures):
        if self.retention_days is None:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        for case_file in captures.glob("*/case.json"):
            try:
                case = json.loads(case_file.read_text(encoding="utf-8"))
                created = datetime.fromisoformat(case["utc"])
            except (OSError, KeyError, ValueError):
                continue
            if created < cutoff:
                shutil.rmtree(case_file.parent)

    def _scan_existing(self, captures):
        self.used_bytes = self.existing_cases = 0
        self.used_by_kind.clear()
        for case_file in captures.glob("*/case.json"):
            try:
                case = json.loads(case_file.read_text(encoding="utf-8"))
                size = sum(p.stat().st_size for p in case_file.parent.iterdir() if p.is_file())
            except (OSError, ValueError):
                continue
            self.used_bytes += size
            self.existing_cases += 1
            self.used_by_kind[str(case.get("kind", "other"))] += size

    def _write(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.root.chmod(0o700)
            captures = self.root / "captures"
            captures.mkdir(exist_ok=True, mode=0o700)
            self._prune_expired(captures)
            self._scan_existing(captures)
            while not self.closed.is_set() or not self.queue.empty():
                try:
                    case, pcm = self.queue.get(timeout=.1)
                except queue.Empty:
                    continue
                try:
                    if self.error:
                        self.dropped += 1
                        continue
                    case["request"], files = pack_audio(case["request"])
                    if pcm:
                        buf = io.BytesIO()
                        with wave.open(buf, "wb") as wav:
                            wav.setnchannels(1)
                            wav.setsampwidth(2)
                            wav.setframerate(case["outcome"]["sample_rate"])
                            wav.writeframes(pcm)
                        files["output.wav"] = buf.getvalue()
                    files["case.json"] = json_bytes(case)
                    size = sum(map(len, files.values()))
                    kind = str(case.get("kind", "other"))
                    kind_limit = self.kind_limits.get(kind)
                    if (self.used_bytes + size > self.max_bytes
                            or self.existing_cases >= self.max_cases):
                        self.dropped += 1
                        self.dropped_by_reason["total_quota"] += 1
                        continue
                    if kind_limit is not None and self.used_by_kind[kind] + size > kind_limit:
                        self.dropped += 1
                        self.dropped_by_reason["kind_quota:" + kind] += 1
                        continue
                    staging = captures / ("." + case["case_id"] + ".partial")
                    staging.mkdir(mode=0o700)
                    for name, content in files.items():
                        private_write(staging / name, content)
                    staging.rename(captures / case["case_id"])
                    self.used_bytes += size
                    self.used_by_kind[kind] += size
                    self.existing_cases += 1
                    self.saved += 1
                except Exception as exc:
                    self.dropped += 1
                    self.error = type(exc).__name__
                    print(f"[DEMO CASES] capture stopped: {self.error}", flush=True)
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.error = type(exc).__name__
            print(f"[DEMO CASES] capture unavailable: {self.error}", flush=True)
        finally:
            # Release queued payload memory even after initialization failure.
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                    self.dropped += 1
                except queue.Empty:
                    break

    async def close(self):
        self.closed.set()
        await asyncio.to_thread(self.thread.join, 2)
        print("[DEMO CASES] " + json.dumps(self.stats()), flush=True)

    async def flush(self):
        """Wait until cases queued before this call are durable."""
        await asyncio.to_thread(self.queue.join)
