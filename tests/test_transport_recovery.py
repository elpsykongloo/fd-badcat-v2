"""HTTP fault injection, saved state, no-repeat and cancellation contracts."""
import asyncio
from contextlib import aclosing
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import aiohttp
from aiohttp import web
import pytest

import module
from demo_cases import CaseArchive, load_case
from demo_session_replay import RecordedStreams
from speech_stream import SocketOutbox
from stream_transport import PCMChunk, sse_json
from transport_diagnostics import SpeechSynthesisError, safe_endpoint, TruncatedStreamError
from test_actor_candidate import actor, ended, pump, frame, cleanup
from test_speech_stream import server, sse


def configured(fn):
    e, _, socket = actor()
    e.TTS_TRANSPORT_RETRIES = 1
    e.tts_stream_fn = fn
    return e, socket


def http_error(status):
    return aiohttp.ClientResponseError(SimpleNamespace(real_url="http://localhost/test"), (), status=status)


async def test_real_http_failure_has_content_free_correlated_evidence():
    seen = []
    async def handler(request):
        seen.append(request.headers["x-request-id"])
        return web.json_response({"secret": "NEVER-SAVE-BODY"}, status=502,
                                 headers={"X-Request-Id": "upstream-1"})
    state = {"request_id": "call-synthetic"}
    async with server(handler) as url:
        with pytest.raises(aiohttp.ClientResponseError) as failure:
            [p async for p in sse_json(url + "?key=SECRET", {}, transport_context=state)]
    details = failure.value.transport_failure
    assert details["phase"] == "headers" and details["http_status"] == 502
    assert details["received_bytes"] == 0 and details["sse_events"] == 0
    assert details["request_id"] == "call-synthetic" and details["upstream_request_id"] == "upstream-1"
    assert seen == ["call-synthetic"]
    assert details["elapsed_ms"] >= details["headers_ms"] >= 0
    assert "SECRET" not in json.dumps(details) and "NEVER-SAVE-BODY" not in json.dumps(details)
    assert safe_endpoint("http://user:secret@localhost:123/path?key=secret") == "http://localhost:123/path"


async def test_retry_before_pcm_has_distinct_saved_calls_and_replays(tmp_path, monkeypatch):
    attempts = []
    async def fake(text, **options):
        attempts.append(dict(options["transport_context"]))
        if len(attempts) == 1:
            exc = http_error(502)
            exc.transport_failure = {"phase": "headers", "http_status": 502,
                                     "request_id": options["transport_context"]["request_id"]}
            raise exc
        yield PCMChunk(b"\1\0" * 10, 24000)
    monkeypatch.setattr(module, "tts_omni_stream", fake)
    e, _ = configured(fake)
    rows = []
    e._observe = lambda event, data=None: rows.append({"event": event, "data": data or {}})
    e.demo_cases = CaseArchive(tmp_path)
    e.demo_session_id = "synthetic"
    output = [p async for p in e._response_tts_stream("自造短句。", parent_id="input-synthetic")]
    assert len(output) == 1 and len(attempts) == 2
    assert attempts[0]["request_id"] != attempts[1]["request_id"]
    assert e.request_capacity.snapshot()["active_total"] == 0
    await e.demo_cases.close()
    cases = sorted((load_case(p) for p in (tmp_path / "captures").iterdir()), key=lambda c:c["context"]["tts_attempt"])
    assert [c["outcome"]["status"] for c in cases] == ["error", "completed"]
    assert cases[0]["outcome"]["transport"]["http_status"] == 502
    assert cases[0]["context"]["tts_operation_id"] == cases[1]["context"]["tts_operation_id"]
    assert cases[0]["request"] == cases[1]["request"]
    from demo_diagnostics import build_spans
    calls = [r for r in build_spans(rows) if r["span_type"] == "model"]
    assert [r["tts_attempt"] for r in calls] == [1, 2]
    assert len({r["tts_operation_id"] for r in calls}) == 1
    replay = RecordedStreams("synthetic", tmp_path, rows, {}, speed=1000)
    e.tts_stream_fn = replay.audio
    assert [p async for p in e._response_tts_stream("自造短句。")] == output
    assert not replay.remaining()


@pytest.mark.parametrize("status,exposed,expected_calls", [(502, True, 1), (400, False, 1), (429, False, 1), (502, False, 2)])
async def test_retry_is_bounded_and_never_replays_any_exposed_pcm(status, exposed, expected_calls):
    calls = []
    async def fake(text):
        calls.append(text)
        if exposed:
            yield PCMChunk(b"\1\0" * 10, 24000)
        raise http_error(status)
    e, _ = configured(fake)
    pcm = []
    with pytest.raises(SpeechSynthesisError) as failure:
        async for p in e._response_tts_stream("同一句。"):
            pcm.append(p)
    assert len(calls) == expected_calls
    assert len(pcm) == int(exposed)
    assert failure.value.attempts == expected_calls
    assert e.request_capacity.snapshot()["active_total"] == 0


async def test_cancel_during_retry_backoff_never_sends_second_request():
    failed = asyncio.Event()
    calls = []
    async def fake(text):
        calls.append(text)
        failed.set()
        raise http_error(502)
        yield
    e, _ = configured(fake)
    task = asyncio.create_task(anext(e._response_tts_stream("短句。")))
    await failed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) == 1 and e.request_capacity.snapshot()["active_total"] == 0


async def test_generator_close_never_triggers_recovery():
    closed = asyncio.Event()
    calls = []
    async def fake(text):
        calls.append(text)
        try:
            yield PCMChunk(b"\1\0" * 10, 24000)
            await asyncio.Event().wait()
        finally:
            closed.set()
    e, _ = configured(fake)
    async with aclosing(e._response_tts_stream("短句。")) as source:
        await anext(source)
    assert closed.is_set() and len(calls) == 1
    assert e.request_capacity.snapshot()["active_total"] == 0


@pytest.mark.parametrize("error,retryable", [
    (aiohttp.ServerDisconnectedError, True),
    (aiohttp.ClientPayloadError, True),
    (asyncio.TimeoutError, True),
    (TruncatedStreamError, True),
    (ValueError, False),
])
async def test_only_transient_transport_classes_retry(error, retryable):
    calls = []
    async def fail(text):
        calls.append(text)
        raise error()
        yield
    e, _ = configured(fail)
    with pytest.raises(SpeechSynthesisError):
        [p async for p in e._response_tts_stream("测试。")]
    assert len(calls) == (2 if retryable else 1)
    assert e.request_capacity.snapshot()["active_total"] == 0


async def test_failed_private_tts_never_resets_budget_by_restarting_response():
    e, models, sock = actor()
    e.TTS_TRANSPORT_RETRIES = 1
    calls = []
    async def fail(text):
        calls.append(text)
        raise http_error(502)
        yield
    e.tts_stream_fn = fail
    try:
        await ended(e)
        c = e._candidate
        await pump(e, lambda:c.error is not None)
        assert len(calls) == 2 and models.responses == 1
        assert not sock.audio
        await frame(e, .704)
        await pump(e, lambda:e._speech is None and e._candidate is None)
        assert len(calls) == 2 and models.responses == 1
        assert any(r["event"] == "speech_error" and r["data"]["code"] == "tts_unavailable" for r in sock.events)
        assert not any(r["event"] == "speech_text_delta" for r in sock.events)
        assert not sock.audio
    finally:
        await cleanup(e)


@pytest.mark.parametrize("code", [502, "tts_unavailable", "speech_text_invalid", "response_incomplete"])
async def test_all_terminal_error_notices_survive_cancel_fence(code):
    e, _, socket = actor()
    e._outbox = SocketOutbox(socket, lambda sid: False, lambda: None)
    try:
        await e.send_control("speech_error", {"utterance_id": 1, "code": code})
        await e.send_control("speech_cancelled", {"utterance_id": 1, "reason": "stream_error"})
        async def sent():
            while len(socket.events) < 2:
                await asyncio.sleep(0)
        await asyncio.wait_for(sent(), 1)
        assert [r["event"] for r in socket.events] == ["speech_error", "speech_cancelled"]
    finally:
        await e._outbox.close()


async def test_compatibility_proxy_fresh_connections_and_error_log(monkeypatch, caplog):
    import qwen3_api as proxy
    connections = []
    async def handler(request):
        connections.append(request.transport)
        return web.json_response({"ok": True})
    class Request:
        app = proxy.app
        headers = {"x-request-id": "call-proxy-test"}
        async def json(self):
            return {"model": "test", "stream": False}
    async with server(handler) as url:
        monkeypatch.setattr(proxy, "VLLM_URL", url)
        async with proxy.lifespan(proxy.app):
            for _ in range(2):
                result = await proxy.chat_proxy(Request())
                assert result.status_code == 200
                assert result.headers["x-request-id"] == "call-proxy-test"
            async def disconnected(*args, **kwargs):
                raise aiohttp.ServerDisconnectedError("DO-NOT-LOG-RAW-SECRET")
            monkeypatch.setattr(proxy.app.state.http, "post", disconnected)
            with caplog.at_level("INFO", logger="uvicorn.error"):
                result = await proxy.chat_proxy(Request())
            assert result.status_code == 502
    assert connections[0] is not connections[1]
    assert "ServerDisconnectedError" in caplog.text and "call-proxy-test" in caplog.text
    assert "DO-NOT-LOG-RAW-SECRET" not in caplog.text
