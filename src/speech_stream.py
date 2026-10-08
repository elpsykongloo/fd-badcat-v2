"""Bounded text -> sentence -> streaming TTS pipeline, independent of turn policy.

Only SpeechEvents cross into the actor. One audio packet can await delivery;
at most two complete sentences queue for synthesis. Played-sample credit bounds
client lookahead. Optional bounded prefetch decouples synthesis from delivery.
Native sample rate avoids stateless per-chunk resampling.
"""
import asyncio
import struct
import time
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass, field

from tts_sentence import StreamingSentenceBuffer
from async_utils import cancellable_wait

PCM_HEADER = struct.Struct("<4sIII")  # magic, utterance id, packet sequence, rate
PROTOCOL = "pcm16.v1"
MAX_PREFETCH_CHUNK_BYTES = 512 * 1024
MAX_RESPONSE_CHARS = 16384
MAX_READ_AHEAD_PCM_BYTES = 8 * 1024 * 1024
MAX_READ_AHEAD_PCM_CHUNKS = 4096


async def _read_ahead(source, timeout, measure, max_size, max_items, error_message, name,
                     on_complete=None):
    """A whole-source budget prevents downstream credit from holding its lease.

    No queue put can block the model reader. The source closes before terminal
    delivery; errors follow received items. Consumer cancellation joins it.
    """
    pending = asyncio.Queue(maxsize=max_items + 1)
    closing = False

    async def read():
        size = count = 0
        error = None
        try:
            async with aclosing(source):
                while True:
                    try:
                        item = await cancellable_wait(source.__anext__(), timeout)
                    except StopAsyncIteration:
                        break
                    cost = measure(item)
                    size += cost
                    count += 1
                    if size > max_size or count > max_items:
                        raise RuntimeError(error_message)
                    if not cost:
                        continue
                    pending.put_nowait((item, None))
        except asyncio.CancelledError as exc:
            if closing:
                raise
            # Only consumer cancellation is a normal pipeline cancellation.
            # A self-cancelled source must surface as an explicit failure after
            # buffered items, rather than silently finishing the public stream.
            error = RuntimeError("Speech source cancelled unexpectedly")
            error.__cause__ = exc
        except Exception as exc:
            error = exc
        if on_complete is not None:
            on_complete(time.perf_counter())
        # The source (and its capacity permit) is closed before terminal delivery.
        pending.put_nowait((None, error))

    reader = asyncio.create_task(read(), name=name)
    try:
        while True:
            delta, error = await pending.get()
            if delta is None:
                if error is not None:
                    raise error
                return
            yield delta
    finally:
        closing = True
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


def drain_text(source, timeout, *, on_complete=None):
    def measure(delta):
        if not isinstance(delta, str):
            raise TypeError("Response stream must contain text")
        return len(delta)
    return _read_ahead(source, timeout, measure, MAX_RESPONSE_CHARS, MAX_RESPONSE_CHARS,
                       "Response exceeds streaming text limit", "speech-text-reader", on_complete)


def drain_pcm(source, timeout, *, max_bytes=MAX_READ_AHEAD_PCM_BYTES,
              max_chunks=MAX_READ_AHEAD_PCM_CHUNKS):
    if type(max_bytes) is not int or max_bytes < 1 or type(max_chunks) is not int or max_chunks < 1:
        raise ValueError("Invalid PCM read-ahead budget")
    rate = None
    def measure(chunk):
        nonlocal rate
        if not isinstance(chunk.pcm, bytes) or not chunk.pcm or len(chunk.pcm) % 2:
            raise RuntimeError("Empty or unaligned PCM chunk")
        if len(chunk.pcm) > MAX_PREFETCH_CHUNK_BYTES:
            raise RuntimeError("TTS chunk exceeds bounded read-ahead input")
        if type(chunk.sample_rate) is not int or not 8000 <= chunk.sample_rate <= 96000:
            raise RuntimeError("Invalid sample rate")
        if rate is not None and rate != chunk.sample_rate:
            raise RuntimeError("Sample rate changed inside utterance")
        rate = chunk.sample_rate
        return len(chunk.pcm)
    return _read_ahead(source, timeout, measure, max_bytes, max_chunks,
                       "TTS source exceeds bounded read-ahead budget", "speech-pcm-reader")


class AudioHealth:
    """Content-free 100 ms RMS diagnostics; NEVER trim or gate speech."""
    def __init__(self):
        self.tail = b""
        self.windows = self.quiet = self.run = self.longest = 0

    def feed(self, pcm, rate):
        import numpy as np
        data = self.tail + pcm
        size = max(1, rate // 10)
        count = len(data) // (size * 2)
        self.tail = data[count * size * 2:]
        if not count:
            return
        frames = np.frombuffer(data[:count * size * 2], dtype="<i2").astype(np.float32).reshape(count, size)
        low = np.mean(frames * frames, axis=1) < (.003 * 32768) ** 2
        for value in low:
            self.windows += 1
            self.quiet += int(value)
            self.run = self.run + 1 if value else 0
            self.longest = max(self.longest, self.run)

    def summary(self):
        return {"rms_window_ms": 100, "rms_threshold": .003,
                "low_energy_ms": self.quiet * 100,
                "longest_low_energy_ms": self.longest * 100,
                "analyzed_ms": self.windows * 100}


class AudioAhead:
    """FIFO with a hard PCM byte budget; at most two sentence markers in flight.

    One decoded source chunk and the sender's current packet are outside this
    queue and separately bounded. Cancellation belongs to the pipeline tasks.
    """
    def __init__(self):
        self.items = deque()
        self.bytes = self.peak_bytes = 0
        self.changed = asyncio.Condition()

    async def put(self, item, size=0, limit=0):
        async with self.changed:
            await self.changed.wait_for(lambda: not size or self.bytes + size <= limit)
            self.items.append((item, size))
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.changed.notify_all()

    async def get(self):
        async with self.changed:
            await self.changed.wait_for(lambda: bool(self.items))
            item, size = self.items.popleft()
            self.bytes -= size
            self.changed.notify_all()
            return item


@dataclass
class SpeechEvent:
    sid: int
    kind: str
    data: dict = field(default_factory=dict)
    delivered: object = None


class SocketOutbox:
    """One socket writer, never a network await in the perception actor.

    A stalled client is disconnected, not given an unbounded queue. Tagged old
    audio is dropped before sending; a packet already on wire is fenced by id.
    """
    def __init__(self, websocket, valid, failed, timeout=5, observed=None):
        self.websocket, self.valid, self.failed = websocket, valid, failed
        self.timeout = timeout
        self.observed = observed
        self.queue = asyncio.Queue(maxsize=128)
        self.task = asyncio.create_task(self.run())

    def put(self, payload, sid=None, delivered=None):
        try:
            self.queue.put_nowait((payload, sid, delivered, time.perf_counter()))
        except asyncio.QueueFull:
            self.failed()
            if delivered is not None and not delivered.done():
                delivered.cancel()

    async def run(self):
        try:
            while True:
                payload, sid, delivered, queued = await self.queue.get()
                try:
                    if sid is None or self.valid(sid):
                        started = time.perf_counter()
                        send = (self.websocket.send_bytes(payload) if isinstance(payload, bytes)
                                else self.websocket.send_text(payload))
                        await cancellable_wait(send, self.timeout)
                        if self.observed is not None:
                            self.observed(payload, queued, started, time.perf_counter())
                finally:
                    if delivered is not None and not delivered.done():
                        delivered.set_result(None)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failed()

    async def close(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        while not self.queue.empty():
            _, _, delivered, _ = self.queue.get_nowait()
            if delivered is not None and not delivered.done():
                delivered.cancel()


class SpeechPipeline:
    def __init__(self, sid, queue, messages, text_fn, tts_fn, *, timeout=15,
                 retry=True, apology="", packet_ms=40, buffer_ms=600, precompute_gate=None,
                 track_sentences=False, startup_ms=80, prefetch_ms=0, diagnostics=False,
                 spoken_text=False, text_read_ahead=False, first_clause_chars=0,
                 pcm_read_ahead=False):
        if not 10 <= packet_ms <= 100 or not 2 * packet_ms <= buffer_ms <= 2000:
            raise ValueError("stream packet/buffer sizes outside safe limits")
        if not 0 <= startup_ms <= 1000:
            raise ValueError("stream startup outside safe limits")
        if prefetch_ms and not 2 * packet_ms <= prefetch_ms <= 5000:
            raise ValueError("stream prefetch outside safe limits")
        if type(first_clause_chars) is not int or not 0 <= first_clause_chars <= 160:
            raise ValueError("First clause threshold outside safe limits")
        self.sid, self.queue = sid, queue
        self.messages, self.text_fn, self.tts_fn = messages, text_fn, tts_fn
        self.timeout, self.retry, self.apology = timeout, retry, apology
        self.packet_ms, self.buffer_ms = packet_ms, buffer_ms
        self.sent = self.played = 0
        self.rate = None
        self.credit = asyncio.Event()
        self.sentences = asyncio.Queue(maxsize=2)
        self.precompute_gate = precompute_gate
        self.track_sentences = track_sentences
        self.startup_ms, self.prefetch_ms = startup_ms, prefetch_ms
        self.diagnostics = diagnostics
        self.spoken_text = spoken_text
        self.text_read_ahead = text_read_ahead
        self.first_clause_chars = first_clause_chars
        self.pcm_read_ahead = pcm_read_ahead
        self.text_source_finished_at = None
        self.ahead = AudioAhead() if prefetch_ms else None
        self.sentence_slots = asyncio.Semaphore(2)
        self.send_seq = 0
        self.origin = time.perf_counter()
        self.task = asyncio.create_task(self.run())

    async def metric(self, phase, **data):
        if self.diagnostics:
            await self.emit("timing", phase=phase,
                            pipeline_ms=round((time.perf_counter() - self.origin) * 1000, 3), **data)

    def progress(self, samples):
        # Called only by actor; stale IDs are filtered before this point.
        if type(samples) is not int or not self.played <= samples <= self.sent:
            return False
        self.played = samples
        self.credit.set()
        return True

    def cancel(self):
        self.task.cancel()

    async def emit(self, kind, **data):
        ack = asyncio.get_running_loop().create_future()
        self.queue.put_nowait(SpeechEvent(self.sid, kind, data, ack))
        await ack

    async def produce(self):
        splitter = StreamingSentenceBuffer(first_clause_chars=self.first_clause_chars)
        normalizer = None
        if self.spoken_text:
            from spoken_text import SpokenTextBuffer
            normalizer = SpokenTextBuffer()
        chunks = []
        raw_chars = 0
        t0 = time.perf_counter()
        timed_out = False
        def source_finished(at):
            self.text_source_finished_at = at
        async def publish(delta):
            if not delta:
                return
            chunks.append(delta)
            await self.emit("text_delta", text=delta)
            for sentence in splitter.feed(delta):
                await self.sentences.put(sentence)
        for attempt in range(2 if self.retry else 1):
            try:
                source = self.text_fn(self.messages)
                if self.text_read_ahead:
                    source = drain_text(source, self.timeout, on_complete=source_finished)
                async with aclosing(source) as source:
                    while True:
                        try:
                            delta = await cancellable_wait(source.__anext__(), self.timeout)
                        except StopAsyncIteration:
                            break
                        raw_chars += len(delta)
                        if raw_chars > MAX_RESPONSE_CHARS:
                            raise RuntimeError("Response exceeds streaming text limit")
                        await publish(normalizer.feed(str(delta)) if normalizer else delta)
                break
            except asyncio.TimeoutError:
                # Never replay a partially exposed response: that would duplicate speech.
                if raw_chars:
                    raise
                if self.retry and attempt == 0:
                    continue
                timed_out = True
                await publish(self.apology)
        if normalizer:
            await publish(normalizer.flush())
            await self.metric("spoken_text", protocol="spoken-text-v1", raw_chars=raw_chars,
                              spoken_chars=normalizer.spoken_chars)
        for sentence in splitter.flush():
            await self.sentences.put(sentence)
        text = "".join(chunks)
        if not text.strip():
            raise RuntimeError("Empty response stream")
        delivered_at = time.perf_counter()
        infer_end = self.text_source_finished_at or delivered_at
        await self.emit("text_done", text=text, infer=round(infer_end - t0, 3),
                        timed_out=timed_out,
                        **({"delivery_ms": round((delivered_at - t0) * 1000, 3)}
                           if self.text_read_ahead else {}))
        await self.sentences.put(None)

    def validate_chunk(self, chunk):
        if not chunk.pcm or len(chunk.pcm) % 2:
            raise RuntimeError("Empty or unaligned PCM chunk")
        if not 8000 <= chunk.sample_rate <= 96000:
            raise RuntimeError("Invalid sample rate")
        if self.ahead is not None and len(chunk.pcm) > MAX_PREFETCH_CHUNK_BYTES:
            raise RuntimeError("TTS chunk exceeds bounded prefetch input")
        if self.rate is None:
            self.rate = chunk.sample_rate
        if self.rate != chunk.sample_rate:
            raise RuntimeError("Sample rate changed inside utterance")

    async def send_pcm(self, pcm, t0):
        count = len(pcm) // 2
        waited = 0.0
        while self.sent + count - self.played > self.rate * self.buffer_ms / 1000:
            self.credit.clear()
            began = time.perf_counter()
            await cancellable_wait(self.credit.wait(), 5)
            waited += time.perf_counter() - began
        self.sent += count
        wire = PCM_HEADER.pack(b"FDS1", self.sid, self.send_seq, self.rate) + pcm
        await self.emit("audio", wire=wire, seq=self.send_seq, samples=count,
                        rate=self.rate, elapsed=time.perf_counter() - t0)
        self.send_seq += 1
        return waited

    async def synthesize(self):
        sentence_index = 0
        t0 = time.perf_counter()
        while True:
            sentence = await self.sentences.get()
            if sentence is None:
                break
            if not sentence.strip():
                continue
            if sentence_index and self.precompute_gate is not None:
                await self.precompute_gate.wait()
            if self.ahead is not None:
                await self.sentence_slots.acquire()
            sentence_index += 1
            if self.ahead is not None:
                await self.ahead.put(("sentence", sentence_index, sentence))
            else:
                await self.emit("sentence", text=sentence, start_sample=self.sent)
            await self.metric("tts_request", sentence_index=sentence_index, sentence_chars=len(sentence))
            request_started = time.perf_counter()
            credit_wait = enqueue_wait = 0.0
            sentence_samples = chunk_index = 0
            health = AudioHealth() if self.diagnostics else None
            produced = False
            source = self.tts_fn(sentence)
            if self.pcm_read_ahead:
                source = drain_pcm(source, 60)
            async with aclosing(source) as source:
                while True:
                    try:
                        chunk = await cancellable_wait(source.__anext__(), 60)
                    except StopAsyncIteration:
                        break
                    self.validate_chunk(chunk)
                    produced = True
                    chunk_index += 1
                    sentence_samples += len(chunk.pcm) // 2
                    if health is not None:
                        health.feed(chunk.pcm, self.rate)
                    await self.metric("tts_chunk", sentence_index=sentence_index,
                        chunk_index=chunk_index, samples=len(chunk.pcm) // 2, rate=self.rate,
                        request_ms=round((time.perf_counter() - request_started) * 1000, 3),
                        **(getattr(chunk, "timing", None) or {}))
                    packet_bytes = int(self.rate * self.packet_ms / 1000) * 2
                    for offset in range(0, len(chunk.pcm), packet_bytes):
                        pcm = chunk.pcm[offset:offset + packet_bytes]
                        if self.ahead is None:
                            credit_wait += await self.send_pcm(pcm, t0)
                        else:
                            began = time.perf_counter()
                            await self.ahead.put(("audio", sentence_index, pcm), len(pcm),
                                                 int(self.rate * self.prefetch_ms / 1000) * 2)
                            enqueue_wait += time.perf_counter() - began
            if not produced:
                raise RuntimeError("Empty TTS sentence stream")
            await self.metric("tts_complete", sentence_index=sentence_index,
                consume_ms=round((time.perf_counter() - request_started) * 1000, 3),
                audio_ms=round(sentence_samples / self.rate * 1000, 3), chunks=chunk_index,
                credit_wait_ms=round(credit_wait * 1000, 3),
                prefetch_wait_ms=round(enqueue_wait * 1000, 3),
                sentence_chars=len(sentence), **(health.summary() if health else {}))
            if health is not None and health.longest >= 50:
                await self.metric("tts_audio_warning", sentence_index=sentence_index,
                    reason="long_low_energy_run", **health.summary())
            if self.ahead is not None:
                await self.ahead.put(("sentence_end", sentence_index, sentence))
            elif self.track_sentences:
                await self.emit("sentence_end", text=sentence, end_sample=self.sent,
                                **self.reply_boundary(sentence_index, sentence))
        if self.ahead is not None:
            await self.ahead.put(("end", 0, None))
        else:
            await self.emit("audio_end", samples=self.sent, rate=self.rate,
                            packets=self.send_seq, elapsed=time.perf_counter() - t0)

    async def send_ahead(self):
        credit_wait = 0.0
        while True:
            kind, index, data = await self.ahead.get()
            if kind == "end":
                await self.emit("audio_end", samples=self.sent, rate=self.rate,
                    packets=self.send_seq, elapsed=time.perf_counter() - self.origin)
                return
            if kind == "sentence":
                credit_wait = 0.0
                await self.emit("sentence", text=data, start_sample=self.sent)
            elif kind == "audio":
                credit_wait += await self.send_pcm(data, self.origin)
            elif kind == "sentence_end":
                if self.track_sentences:
                    await self.emit("sentence_end", text=data, end_sample=self.sent,
                                    **self.reply_boundary(index, data))
                await self.metric("sentence_sent", sentence_index=index,
                    end_sample=self.sent, credit_wait_ms=round(credit_wait * 1000, 3),
                    prefetch_peak_bytes=self.ahead.peak_bytes)
                self.sentence_slots.release()

    def reply_boundary(self, index, text):
        # Earlier first-clause playback must not make an unfinished sentence
        # eligible as a completed request in played-reply review.
        if self.first_clause_chars:
            return {"reply_complete": not (index == 1 and text.rstrip().endswith((',', '，')))}
        return {}

    async def run(self):
        children = [asyncio.create_task(self.produce()), asyncio.create_task(self.synthesize())]
        if self.ahead is not None:
            children.append(asyncio.create_task(self.send_ahead()))
        try:
            await asyncio.gather(*children)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.emit("error", error=f"{type(exc).__name__}: {exc}",
                            terminal=bool(getattr(exc, "terminal", False)),
                            pipeline_state={"sent_samples": self.sent, "played_samples": self.played,
                                "rate": self.rate, "packets": self.send_seq,
                                "queued_sentences": self.sentences.qsize(),
                                "prefetch_bytes": self.ahead.bytes if self.ahead else 0},
                            **({"tts_operation_id": exc.operation_id, "attempts": exc.attempts}
                               if hasattr(exc, "operation_id") else {}),
                            **({"code": exc.code} if getattr(exc, "code", None) else {}))
        finally:
            for child in children:
                child.cancel()
            await asyncio.gather(*children, return_exceptions=True)
            if self.ahead is not None:
                self.ahead.items.clear()
                self.ahead.bytes = 0
            self.queue.put_nowait(SpeechEvent(self.sid, "finished"))
