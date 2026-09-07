"""Bounded text -> sentence -> streaming TTS pipeline, independent of turn policy.

Only SpeechEvents cross into the actor. One audio packet can await delivery;
at most two complete sentences queue for synthesis. Played-sample credit bounds
client lookahead. Native sample rate avoids stateless per-chunk resampling.
"""
import asyncio
import struct
import time
from contextlib import aclosing
from dataclasses import dataclass, field

from tts_sentence import StreamingSentenceBuffer

PCM_HEADER = struct.Struct("<4sIII")  # magic, utterance id, packet sequence, rate
PROTOCOL = "pcm16.v1"


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
                        await asyncio.wait_for(send, self.timeout)
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
                 retry=True, apology="", packet_ms=40, buffer_ms=600):
        if not 10 <= packet_ms <= 100 or not 2 * packet_ms <= buffer_ms <= 2000:
            raise ValueError("stream packet/buffer sizes outside safe limits")
        self.sid, self.queue = sid, queue
        self.messages, self.text_fn, self.tts_fn = messages, text_fn, tts_fn
        self.timeout, self.retry, self.apology = timeout, retry, apology
        self.packet_ms, self.buffer_ms = packet_ms, buffer_ms
        self.sent = self.played = 0
        self.rate = None
        self.credit = asyncio.Event()
        self.sentences = asyncio.Queue(maxsize=2)
        self.task = asyncio.create_task(self.run())

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
        splitter = StreamingSentenceBuffer()
        chunks = []
        t0 = time.perf_counter()
        timed_out = False
        for attempt in range(2 if self.retry else 1):
            try:
                async with aclosing(self.text_fn(self.messages)) as source:
                    while True:
                        try:
                            delta = await asyncio.wait_for(source.__anext__(), self.timeout)
                        except StopAsyncIteration:
                            break
                        chunks.append(delta)
                        if sum(map(len, chunks)) > 16384:
                            raise RuntimeError("Response exceeds streaming text limit")
                        await self.emit("text_delta", text=delta)
                        for sentence in splitter.feed(delta):
                            await self.sentences.put(sentence)
                break
            except asyncio.TimeoutError:
                # Never replay a partially exposed response: that would duplicate speech.
                if chunks:
                    raise
                if self.retry and attempt == 0:
                    continue
                timed_out = True
                chunks = [self.apology]
                await self.emit("text_delta", text=self.apology)
                for sentence in splitter.feed(self.apology):
                    await self.sentences.put(sentence)
        for sentence in splitter.flush():
            await self.sentences.put(sentence)
        text = "".join(chunks)
        if not text.strip():
            raise RuntimeError("Empty response stream")
        await self.emit("text_done", text=text, infer=round(time.perf_counter() - t0, 3),
                        timed_out=timed_out)
        await self.sentences.put(None)

    async def synthesize(self):
        seq = 0
        t0 = time.perf_counter()
        while True:
            sentence = await self.sentences.get()
            if sentence is None:
                break
            if not sentence.strip():
                continue
            await self.emit("sentence", text=sentence)
            produced = False
            async with aclosing(self.tts_fn(sentence)) as source:
                while True:
                    try:
                        chunk = await asyncio.wait_for(source.__anext__(), 60)
                    except StopAsyncIteration:
                        break
                    if not chunk.pcm or len(chunk.pcm) % 2:
                        raise RuntimeError("Empty or unaligned PCM chunk")
                    if not 8000 <= chunk.sample_rate <= 96000:
                        raise RuntimeError("Invalid sample rate")
                    if self.rate is None:
                        self.rate = chunk.sample_rate
                    if self.rate != chunk.sample_rate:
                        raise RuntimeError("Sample rate changed inside utterance")
                    produced = True
                    packet_bytes = int(self.rate * self.packet_ms / 1000) * 2
                    for offset in range(0, len(chunk.pcm), packet_bytes):
                        pcm = chunk.pcm[offset:offset + packet_bytes]
                        count = len(pcm) // 2
                        while self.sent + count - self.played > self.rate * self.buffer_ms / 1000:
                            self.credit.clear()
                            await asyncio.wait_for(self.credit.wait(), 5)
                        self.sent += count
                        wire = PCM_HEADER.pack(b"FDS1", self.sid, seq, self.rate) + pcm
                        await self.emit("audio", wire=wire, seq=seq, samples=count,
                                        rate=self.rate, elapsed=time.perf_counter() - t0)
                        seq += 1
            if not produced:
                raise RuntimeError("Empty TTS sentence stream")
        await self.emit("audio_end", samples=self.sent, rate=self.rate,
                        packets=seq, elapsed=time.perf_counter() - t0)

    async def run(self):
        children = [asyncio.create_task(self.produce()), asyncio.create_task(self.synthesize())]
        try:
            await asyncio.gather(*children)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.emit("error", error=f"{type(exc).__name__}: {exc}")
        finally:
            for child in children:
                child.cancel()
            await asyncio.gather(*children, return_exceptions=True)
            self.queue.put_nowait(SpeechEvent(self.sid, "finished"))
