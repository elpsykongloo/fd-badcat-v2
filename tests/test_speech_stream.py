"""Streaming contract tests: no model, corpus, credentials, or GPU required."""
import asyncio
import base64
import io
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speech_stream import PCM_HEADER, SpeechEvent, SpeechPipeline, SocketOutbox
from stream_transport import PCMChunk, audio_stream, sse_json, text_stream
from tts_sentence import StreamingSentenceBuffer, split_sentences


@pytest.mark.parametrize("text,expected", [
    ("你好。今天晴天！明天呢？", ["你好。", "今天晴天！", "明天呢？"]),
    ("Dr. Li paid 3.14 dollars. Next!", ["Dr. Li paid 3.14 dollars.", " Next!"]),
    ("Visit the U.S. office, e.g. Boston. Fine.", ["Visit the U.S. office, e.g. Boston.", " Fine."]),
    ('他说：“你好！”然后离开。', ['他说：“你好！”', '然后离开。']),
    ("Wait... Really?\n好。", ["Wait...", " Really?", "\n", "好。"]),
    ("See https://a.example/path. OK.", ["See https://a.example/path.", " OK."]),
    ("", []), ("3.14159", ["3.14159"]),
])
def test_splitter_is_lossless_and_independent_of_token_boundaries(text, expected):
    for width in range(1, max(2, len(text) + 1)):
        splitter, actual = StreamingSentenceBuffer(), []
        for start in range(0, len(text), width):
            actual += splitter.feed(text[start:start + width])
        actual += splitter.flush()
        assert "".join(actual) == text
        assert actual == expected, (width, actual)


def test_splitter_legacy_unchanged_and_bounded_clauses():
    assert split_sentences("Done. Ok.") == ["Done. Ok."]
    splitter = StreamingSentenceBuffer(max_chars=16)
    text = "this is a long unpunctuated sentence " * 20
    chunks = splitter.feed(text) + splitter.flush()
    assert len(chunks) > 10 and max(map(len, chunks)) < 40
    assert "".join(chunks) == text


@asynccontextmanager
async def server(handler):
    app = web.Application()
    app.router.add_post("/chat", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/chat"
    finally:
        await runner.cleanup()


def sse(obj):
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\r\n\r\n").encode()


async def test_real_http_sse_fragmented_utf8_and_audio_wav_dialect():
    wav = io.BytesIO()
    sf.write(wav, np.array([0, 0.25, -0.5]), 24000, format="WAV", subtype="PCM_16")
    async def handler(request):
        payload = await request.json()
        assert payload["stream"] is True
        audio = payload.get("audio")
        obj = {"modality": "audio" if audio else "text", "choices": [{"delta": {
            "content": base64.b64encode(wav.getvalue()).decode() if audio else "你好。"}}]}
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for byte in sse(obj) + b"data: [DONE]\n\n":
            await response.write(bytes([byte]))
        return response
    async with server(handler) as url:
        assert [x async for x in text_stream(url, {})] == ["你好。"]
        chunks = [x async for x in audio_stream(url, {"audio": True})]
        assert len(chunks) == 1 and chunks[0].sample_rate == 24000
        assert np.frombuffer(chunks[0].pcm, dtype="<i2").tolist() == [0, 8192, -16384]


@pytest.mark.parametrize("body", [b"data: {}\n\n", b"data: invalid\n\n",
                                  b'data: {"error":"broken"}\n\n'])
async def test_truncated_or_bad_sse_is_not_silent_success(body):
    async def handler(request):
        return web.Response(body=body, content_type="text/event-stream")
    async with server(handler) as url:
        with pytest.raises((ValueError, RuntimeError)):
            [x async for x in sse_json(url, {})]


async def collect(pipeline, queue, events, *, credit=True, audio_hook=None):
    while True:
        ev = await queue.get()
        events.append(ev)
        if ev.kind == "audio":
            magic, sid, seq, rate = PCM_HEADER.unpack(ev.data["wire"][:16])
            assert (magic, sid, rate) == (b"FDS1", pipeline.sid, 24000)
            if credit:
                pipeline.progress(pipeline.sent)
            if audio_hook:
                audio_hook()
        if ev.delivered is not None and not ev.delivered.done():
            ev.delivered.set_result(None)
        if ev.kind == "finished":
            return


async def test_first_audio_precedes_text_and_tts_eof_order_lossless_history_input():
    first_audio = asyncio.Event()
    tts_closed = []
    sentences = []
    async def text_fn(messages):
        yield "第一句。第"
        await first_audio.wait()  # deadlock if whole text is awaited
        yield "二句。"
    async def tts_fn(sentence):
        sentences.append(sentence)
        try:
            yield PCMChunk(b"\0\0" * 960, 24000)
            await first_audio.wait()  # deadlock if whole TTS is awaited
            yield PCMChunk(b"\1\0" * 480, 24000)
        finally:
            tts_closed.append(sentence)
    queue, events = asyncio.Queue(), []
    p = SpeechPipeline(7, queue, [], text_fn, tts_fn)
    await asyncio.wait_for(collect(p, queue, events, audio_hook=first_audio.set), 2)
    await p.task
    assert "".join(sentences) == "第一句。第二句。"
    assert sentences == tts_closed
    assert [e.data["seq"] for e in events if e.kind == "audio"] == [0, 1, 2, 3]
    assert sum(e.kind == "text_done" for e in events) == 1
    assert next(e.data["text"] for e in events if e.kind == "text_done") == "第一句。第二句。"
    assert next(i for i,e in enumerate(events) if e.kind == "audio") < next(
        i for i,e in enumerate(events) if e.kind == "text_done")


async def test_backpressure_bounded_and_cancel_closes_both_generators():
    text_closed, tts_closed = asyncio.Event(), asyncio.Event()
    async def text_fn(messages):
        try:
            while True:
                yield "甲。乙。"
                await asyncio.sleep(0)
        finally:
            text_closed.set()
    async def tts_fn(sentence):
        try:
            while True:
                yield PCMChunk(b"\0\0" * 960, 24000)
        finally:
            tts_closed.set()
    queue, events = asyncio.Queue(), []
    p = SpeechPipeline(1, queue, [], text_fn, tts_fn, buffer_ms=120)
    reader = asyncio.create_task(collect(p, queue, events, credit=False))
    for _ in range(100):
        if p.sent == 2880 and p.sentences.full():
            break
        await asyncio.sleep(.001)
    assert p.sent == 2880 and p.sentences.qsize() == 2
    assert not p.progress(p.sent + 1) and not p.progress(-1)
    p.cancel()
    await asyncio.gather(p.task, return_exceptions=True)
    await asyncio.wait_for(reader, 1)
    assert text_closed.is_set() and tts_closed.is_set()
    assert not any(e.kind == "audio_end" for e in events)


async def test_partial_response_failure_never_retries_or_duplicates_speech():
    calls = []
    async def text_fn(messages):
        calls.append(1)
        yield "已说的内容。"
        raise asyncio.TimeoutError()
    async def tts_fn(sentence):
        yield PCMChunk(b"\0\0" * 960, 24000)
    queue, events = asyncio.Queue(), []
    p = SpeechPipeline(1, queue, [], text_fn, tts_fn)
    await collect(p, queue, events)
    await p.task
    assert len(calls) == 1
    assert sum(e.kind == "error" for e in events) == 1
    assert not any(e.kind == "text_done" for e in events)


async def test_outbox_drops_stale_audio_and_does_not_block_actor():
    gate, started = asyncio.Event(), asyncio.Event()
    writes = []
    class Socket:
        async def send_text(self, text):
            started.set()
            await gate.wait()
            writes.append(text)
        async def send_bytes(self, data):
            writes.append(data)
    active = 1
    out = SocketOutbox(Socket(), lambda sid: sid == active, lambda: None)
    out.put("busy")
    await started.wait()
    ack = asyncio.get_running_loop().create_future()
    out.put(b"old audio", sid=1, delivered=ack)
    active = 2
    gate.set()
    await asyncio.wait_for(ack, 1)
    await out.close()
    assert writes == ["busy"]


async def test_actor_stream_default_off_and_cancel_fence_single_history():
    from engine import ActorEngine, ControlMsg, ModelDone
    from test_engine import ScriptedVAD
    closed = asyncio.Event()
    async def text_fn(messages):
        yield "完整回答。"
    async def tts_fn(sentence):
        try:
            yield PCMChunk(b"\0\0" * 960, 24000)
            await asyncio.Event().wait()
        finally:
            closed.set()
    e = ActorEngine(engine_cfg={"stream_response": True}, vad_iterator=ScriptedVAD({}),
                    llm_fn=lambda m: "old", asr_fn=lambda p: "", tts_fn=lambda t,p: p,
                    text_stream_fn=text_fn, tts_stream_fn=tts_fn)
    e.dispatch_llm("response", "unchanged prompt", None, 0, add_to_history=True)
    while not any(r["event"] == "speech_first_audio" for r in e.trace):
        await e._process_event(await asyncio.wait_for(e.q.get(), 1))
    old_id = e._speech.sid
    assert e.STATE == "SPEAK" and e.assistant_history == ["完整回答。"]
    await e._process_event(ControlMsg("session_end"))
    await asyncio.gather(*e._speech_tasks, return_exceptions=True)
    assert closed.is_set()
    await e._process_event(SpeechEvent(old_id, "text_done", {"text": "stale"}))
    await e._process_event(SpeechEvent(old_id, "audio", {"seq": 0}))
    assert e.STATE == "LISTEN" and e.assistant_history == [] and e._inflight == 0


async def test_proxy_streams_before_eof_and_closes_upstream_on_cancel(monkeypatch):
    import qwen3_api as proxy
    released, disconnected = asyncio.Event(), asyncio.Event()
    async def handler(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(sse({"choices": [{"delta": {"content": "first"}}]}))
        while not released.is_set():
            if request.transport is None or request.transport.is_closing():
                disconnected.set()
                return response
            await asyncio.sleep(.005)
        return response
    class Request:
        app = proxy.app
        async def json(self):
            return {"stream": True}
    async with server(handler) as url:
        monkeypatch.setattr(proxy, "VLLM_URL", url)
        async with proxy.lifespan(proxy.app):
            response = await proxy.chat_proxy(Request())
            data = await asyncio.wait_for(response.body_iterator.__anext__(), 1)
            assert b"first" in data and not released.is_set()
            pending = asyncio.create_task(response.body_iterator.__anext__())
            await asyncio.sleep(.01)
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await asyncio.wait_for(disconnected.wait(), 1)
        released.set()


async def test_proxy_preserves_nonstream_payload_and_http_failure(monkeypatch):
    import qwen3_api as proxy
    async def handler(request):
        payload = await request.json()
        return web.json_response({"echo": payload}, status=429)
    class Request:
        app = proxy.app
        async def json(self):
            return {"messages": [{"role": "user", "content": "unchanged"}]}
    async with server(handler) as url:
        monkeypatch.setattr(proxy, "VLLM_URL", url)
        async with proxy.lifespan(proxy.app):
            response = await proxy.chat_proxy(Request())
            assert response.status_code == 429
            payload = json.loads(response.body)["echo"]
            assert payload["messages"][0]["content"] == "unchanged"
            assert payload["model"] == proxy.QWEN_MODEL


async def test_text_timeout_before_output_retries_then_apology_once():
    calls = []
    async def text_fn(messages):
        calls.append(1)
        raise asyncio.TimeoutError()
        yield  # async generator contract
    async def tts_fn(sentence):
        yield PCMChunk(b"\0\0" * 960, 24000)
    queue, events = asyncio.Queue(), []
    p = SpeechPipeline(1, queue, [], text_fn, tts_fn, apology="抱歉。")
    await collect(p, queue, events)
    await p.task
    assert len(calls) == 2
    assert [e.data["text"] for e in events if e.kind == "sentence"] == ["抱歉。"]
    result = next(e for e in events if e.kind == "text_done")
    assert result.data["text"] == "抱歉。" and result.data["timed_out"]


async def test_confirmed_short_and_long_interrupt_cancel_but_rejected_does_not():
    from engine import ActorEngine, FrameEvent, ModelDone
    from test_engine import ScriptedVAD
    async def text_fn(messages):
        yield "旧回答。"
        await asyncio.Event().wait()
    async def tts_fn(sentence):
        yield PCMChunk(b"\0\0" * 960, 24000)
    for kind in ("continue", "switch", "long"):
        e = ActorEngine(engine_cfg={"stream_response": True}, vad_iterator=ScriptedVAD({}),
                        llm_fn=lambda m: "", asr_fn=lambda p: "", tts_fn=lambda t,p: p,
                        text_stream_fn=text_fn, tts_stream_fn=tts_fn)
        e.dispatch_llm("response", "R", None, 0)
        old = e._speech
        # Keep this a policy-unit test: no ASR/model calls on accepted interruption.
        e.dispatch_llm = lambda *a, **kw: None
        e.dispatch_asr = lambda *a, **kw: None
        e.STATE = "SPEAK"
        if kind == "long":
            e.IN_SPEECH = True
            e.t_audio = 2
            e.t_interrupt_start = .4
            e.interrupt_buf = [np.zeros(256, dtype=np.float32)]
            await e._speak_frame(FrameEvent(1, 2, None, np.zeros(256)), None)
            assert e.STATE == "LISTEN" and e.TURN_IDX == 1
        else:
            await e._on_interrupt(ModelDone(kind="interrupt", gen=0, epoch=0, turn=0, text=kind))
        assert (e._speech is old) == (kind == "continue")
        e._cancel_speech("cleanup")
        await asyncio.gather(*e._speech_tasks, return_exceptions=True)


async def test_slow_websocket_does_not_freeze_actor_perception():
    from engine import ActorEngine, ControlMsg, FrameEvent
    from test_engine import ScriptedVAD
    gate, entered, tts_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class Socket:
        async def send_text(self, text):
            entered.set()
            await gate.wait()
        async def send_bytes(self, data):
            await gate.wait()
    async def text_fn(messages):
        yield "你好。"
    async def tts_fn(sentence):
        tts_entered.set()
        yield PCMChunk(b"\0\0" * 960, 24000)
    socket = Socket()
    e = ActorEngine(websocket=socket, engine_cfg={"stream_response": True},
                    vad_iterator=ScriptedVAD({}), llm_fn=lambda m: "",
                    asr_fn=lambda p: "", tts_fn=lambda t,p: p,
                    text_stream_fn=text_fn, tts_stream_fn=tts_fn)
    e._outbox = SocketOutbox(socket, lambda sid: True, lambda: None)
    e.dispatch_llm("response", "R", None, 0)
    loop = asyncio.create_task(e.engine_loop())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(tts_entered.wait(), 1)
        for seq in range(1, 21):
            e.q.put_nowait(FrameEvent(seq, seq * .016, None, np.zeros(256, dtype=np.float32)))
        # Queue barrier: engine processes all earlier frames before this ack.
        ack = asyncio.get_running_loop().create_future()
        e.q.put_nowait(SpeechEvent(-1, "barrier", delivered=ack))
        await asyncio.wait_for(ack, 1)
        assert e.t_audio == .32 and not gate.is_set()
    finally:
        e._cancel_speech("cleanup")
        e.q.put_nowait(ControlMsg("disconnect"))
        await asyncio.gather(loop, *e._speech_tasks, return_exceptions=True)
        await e._outbox.close()


async def test_empty_tts_is_explicit_failure_not_audio_done():
    async def text_fn(messages):
        yield "有效文本。"
    async def tts_fn(sentence):
        if False:
            yield
    queue, events = asyncio.Queue(), []
    p = SpeechPipeline(1, queue, [], text_fn, tts_fn)
    await collect(p, queue, events)
    await p.task
    assert any(e.kind == "error" and "Empty TTS" in e.data["error"] for e in events)
    assert not any(e.kind in ("audio", "audio_end") for e in events)
