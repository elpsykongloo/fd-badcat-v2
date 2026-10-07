"""Demo length completion: real SSE boundaries and injected Actor/PCM lifecycle."""
import asyncio
from contextlib import aclosing
from pathlib import Path
import sys

import pytest
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import module
from engine import ActorEngine
from response_completion import ResponseIncompleteError, continuation_messages, response_length_config
from stream_transport import TextDelta, PCMChunk, text_stream
from speech_stream import SpeechPipeline
from test_engine import ScriptedVAD
from test_speech_stream import server, sse, collect


def end(reason):
    return TextDelta("", finish_reason=reason)


MESSAGES = [{"role": "system", "content": "response"},
            {"role": "user", "content": [{"type": "audio_url", "audio_url": {
                "url": "data:audio/wav;base64,original-audio"}}]}]


def actor(stream, **config):
    async def tts(_):
        yield PCMChunk(b"\0\0" * 960, 24000)
    return ActorEngine(prompts={"response": "response", "response_completion": "complete the promise"},
        engine_cfg={"chat_demo": True, "stream_response": True, "response_length_repair": True,
                    "response_completion_repair": True, **config},
        vad_iterator=ScriptedVAD({}), llm_fn=lambda _: "", asr_fn=lambda _: "",
        tts_fn=lambda *_: "", text_stream_fn=stream, tts_stream_fn=tts)


@pytest.mark.parametrize("reason", ["stop", "length"])
async def test_terminal_reason_survives_content_and_empty_records(reason):
    async def handler(_):
        return web.Response(body=sse({"choices": [{"index": 0, "delta": {"content": "末尾"},
            "finish_reason": reason}]}) + sse({"choices": [], "usage": {}}) + b"data: [DONE]\n\n",
            content_type="text/event-stream")
    async with server(handler) as url:
        parts = [p async for p in text_stream(url, {}, report_finish=True)]
        assert parts == ["末尾", ""] and parts[-1].finish_reason == reason
        assert [p async for p in text_stream(url, {})] == ["末尾"]


@pytest.mark.parametrize("records,done", [
    ([{"choices": [{"delta": {"content": "missing reason"}}]}], True),
    ([{"choices": [{"delta": {}, "finish_reason": "stop"}]}], False),
    ([{"choices": [{"delta": {}, "finish_reason": "content_filter"}]}], True),
    ([{"choices": [{"delta": {}, "finish_reason": "stop"}]},
      {"choices": [{"delta": {"content": "late"}}]}], True),
    ([{"choices": [{"delta": {}, "finish_reason": "stop"}]},
      {"choices": [{"delta": {}, "finish_reason": "length"}]}], True),
    ([{"choices": [{"index": 1, "delta": {}, "finish_reason": "stop"}]}], True),
])
async def test_malformed_or_unproven_completion_is_not_success(records, done):
    async def handler(_):
        return web.Response(body=b"".join(sse(r) for r in records) + (b"data: [DONE]\n\n" if done else b""),
                            content_type="text/event-stream")
    async with server(handler) as url:
        with pytest.raises(RuntimeError):
            [p async for p in text_stream(url, {}, report_finish=True)]


async def test_native_prefix_continues_multiple_times_without_extra_space_or_duplicate_audio(monkeypatch):
    calls = []
    answers = iter([("从前小猫发现了一颗", "length"), ("星星。它把星星送回", "length"), ("天空。", "stop")])
    async def native(messages, **options):
        calls.append((messages, options))
        text, reason = next(answers)
        yield text
        yield end(reason)
    monkeypatch.setattr(module, "llm_qwen3o_stream", native)
    e = actor(native)
    spoken = []
    async def tts(text):
        spoken.append(text)
        yield PCMChunk(b"\0\0" * 960, 24000)
    events, queue = [], asyncio.Queue()
    pipeline = SpeechPipeline(1, queue, MESSAGES, e._response_text_stream, tts, track_sentences=True)
    await collect(pipeline, queue, events)
    await pipeline.task
    final = "从前小猫发现了一颗星星。它把星星送回天空。"
    assert "".join(spoken) == final
    assert [ev.data["text"] for ev in events if ev.kind == "text_done"] == [final]
    assert not any(ev.kind == "error" for ev in events)
    assert len(calls) == 3 and calls[0][0] is MESSAGES
    assert calls[1][0] == [*MESSAGES, {"role": "assistant", "content": "从前小猫发现了一颗"}]
    assert calls[2][0][-1]["content"] == "从前小猫发现了一颗星星。它把星星送回"
    assert calls[1][1] == {"report_finish": True, "max_tokens": 512, "continue_final_message": True}
    assert e.request_capacity.snapshot()["active_total"] == 0


@pytest.mark.parametrize("outputs,limit,expected", [
    ([("前文", "length"), ("后文", "length")], 1, "continuation_limit"),
    ([("前文", "length"), ("", "length")], 3, "continuation_no_progress"),
    ([("", "stop")], 3, "empty_response"),
    ([("前文", None)], 3, "missing_finish_reason"),
    ([("好，我给你讲一个。", "stop"), ("别急，我这就开始。", "stop")], 3, "promise_not_completed"),
])
async def test_incomplete_answers_never_report_success(outputs, limit, expected):
    calls = []
    async def source(messages):
        text, reason = outputs[len(calls)]
        calls.append(messages)
        if text:
            yield text
        if reason:
            yield end(reason)
    e = actor(source, response_max_continuations=limit)
    with pytest.raises(ResponseIncompleteError, match=expected):
        [p async for p in e._response_text_stream(MESSAGES)]
    assert len(calls) == len(outputs)
    assert e.request_capacity.snapshot()["active_total"] == 0


async def test_promise_then_truncated_body_shares_length_budget():
    outputs = iter([("好，我给你讲一个。", "stop"), ("小猫发现了一颗", "length"), ("星星。", "stop")])
    calls = []
    async def source(messages):
        calls.append(messages)
        text, reason = next(outputs)
        yield text
        yield end(reason)
    e = actor(source, response_max_continuations=1)
    assert "".join([p async for p in e._response_text_stream(MESSAGES)]) == "好，我给你讲一个。 小猫发现了一颗星星。"
    assert calls[1][-1] == {"role": "user", "content": "complete the promise"}
    assert calls[2] == [*MESSAGES, {"role": "assistant", "content": "好，我给你讲一个。 小猫发现了一颗"}]


async def test_partial_continuation_failure_has_explicit_ui_error_and_no_text_done():
    calls = 0
    async def source(_):
        nonlocal calls
        calls += 1
        yield "已说完。后面的" if calls == 1 else "半句"
        if calls == 2:
            raise ConnectionError("synthetic outage")
        yield end("length")
    e = actor(source)
    queue, events = asyncio.Queue(), []
    pipeline = SpeechPipeline(1, queue, MESSAGES, e._response_text_stream, e.tts_stream_fn)
    await collect(pipeline, queue, events)
    await pipeline.task
    assert [ev.data["code"] for ev in events if ev.kind == "error"] == ["response_incomplete"]
    assert not any(ev.kind in {"text_done", "audio_end"} for ev in events)
    assert calls == 2


async def test_cancel_inflight_continuation_closes_source_releases_capacity_and_never_retries():
    entered, closed = asyncio.Event(), asyncio.Event()
    calls = 0
    async def source(_):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "前文"
            yield end("length")
        else:
            try:
                entered.set()
                await asyncio.Event().wait()
            finally:
                closed.set()
    e = actor(source)
    async def consume():
        async with aclosing(e._response_text_stream(MESSAGES)) as parts:
            return [p async for p in parts]
    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set() and calls == 2 and e.request_capacity.snapshot()["active_total"] == 0


def test_continuation_evicts_only_whole_old_pairs_keeps_audio_and_exact_prefix():
    old = [{"role": "user", "content": [{"type": "text", "text": "old" * 40}]},
           {"role": "assistant", "content": "old answer"}]
    recent = [{"role": "user", "content": "new"}, {"role": "assistant", "content": "answer"}]
    messages = [MESSAGES[0], *old, *recent, MESSAGES[-1]]
    request, removed = continuation_messages(messages, "prefix", 100)
    assert removed == 1 and request == [MESSAGES[0], *recent, MESSAGES[-1],
                                        {"role": "assistant", "content": "prefix"}]
    assert len(messages) == 6 and request[-2] is MESSAGES[-1]
    with pytest.raises(ResponseIncompleteError, match="context_budget"):
        continuation_messages(MESSAGES, "中" * 40, 100)


def test_response_payload_options_and_frozen_defaults(monkeypatch):
    monkeypatch.delenv("FDBC_QWEN_MAX_TOKENS", raising=False)
    assert module.qwen_text_payload(MESSAGES)["max_tokens"] == 256
    request = [*MESSAGES, {"role": "assistant", "content": "unfinished"}]
    payload = module.qwen_text_payload(request, max_tokens=512, continue_final_message=True)
    assert payload["max_tokens"] == 512 and payload["continue_final_message"] is True
    assert payload["add_generation_prompt"] is False
    assert "continue_final_message" not in module.qwen_text_payload(MESSAGES, route=True)
    assert response_length_config({"response_length_repair": True}) is None
    with pytest.raises(ValueError):
        response_length_config({"chat_demo": True, "response_length_repair": True, "response_max_tokens": True})


async def test_disabled_demo_or_other_prompt_retains_old_stream_contract():
    calls = []
    async def source(messages):
        calls.append(messages)
        yield "unchanged"
    for config, messages in [({"response_length_repair": False}, MESSAGES),
                             ({"chat_demo": False}, MESSAGES),
                             ({}, [{"role": "system", "content": "shift"}])]:
        e = actor(source, **config)
        assert [p async for p in e._response_text_stream(messages)] == ["unchanged"]
    assert len(calls) == 3


async def test_actual_call_archive_records_overrides_prefix_and_finish_reason(tmp_path, monkeypatch):
    from demo_cases import CaseArchive, load_case, restore_request
    from demo_session_replay import RecordedStreams
    from test_demo_cases import request as audio_request
    messages = audio_request()["messages"]
    messages[0]["content"] = "response"
    calls = []
    async def native(messages, **options):
        calls.append((messages, options))
        yield "前半" if len(calls) == 1 else "后半。"
        yield end("length" if len(calls) == 1 else "stop")
    monkeypatch.setattr(module, "llm_qwen3o_stream", native)
    e = actor(native)
    archive = CaseArchive(tmp_path)
    e.demo_cases, e.demo_session_id = archive, "synthetic"
    assert "".join([p async for p in e._response_text_stream(messages)]) == "前半后半。"
    await archive.close()
    paths = sorted((tmp_path / "captures").iterdir())
    cases = [load_case(p) for p in paths]
    assert [c["outcome"]["finish_reason"] for c in cases] == ["length", "stop"]
    assert [c["request"]["max_tokens"] for c in cases] == [512, 512]
    assert restore_request(paths[1])["messages"] == calls[1][0]
    assert cases[1]["request"]["continue_final_message"] is True
    assert cases[1]["request"]["add_generation_prompt"] is False
    rows = [{"event": "model_case_started", "data": {"case_id": c["case_id"], "kind": "response"}}
            for c in cases]
    replay = RecordedStreams("synthetic", tmp_path, rows, {"response": "response"}, speed=1000)
    engine = actor(replay.text)
    assert "".join([p async for p in engine._response_text_stream(messages)]) == "前半后半。"
    assert not replay.remaining()


async def test_incomplete_notice_survives_output_cancellation_fence():
    from test_actor_candidate import actor as candidate_actor
    from speech_stream import SocketOutbox
    e, _, socket = candidate_actor()
    e._outbox = SocketOutbox(socket, lambda sid: False, lambda: None)
    try:
        await e.send_control("speech_text_delta", {"utterance_id": 1, "text": "stale"})
        await e.send_control("speech_error", {"utterance_id": 1, "code": "response_incomplete"})
        await e.send_control("speech_cancelled", {"utterance_id": 1, "reason": "stream_error"})
        async def wait_sent():
            while len(socket.events) < 2:
                await asyncio.sleep(0)
        await asyncio.wait_for(wait_sent(), 1)
        assert [r["event"] for r in socket.events] == ["speech_error", "speech_cancelled"]
    finally:
        await e._outbox.close()


async def test_private_completion_failure_never_restarts_budget_or_leaks_draft():
    import json
    from test_actor_candidate import actor as candidate_actor, Models, ended, frame, pump, cleanup
    class Truncated(Models):
        async def text(self, messages):
            if messages[0]["content"] == "response":
                self.responses += 1
                yield "不能公开的未完成草稿"
                yield end("length")
            else:
                async for part in super().text(messages):
                    yield part
    e, models, socket = candidate_actor(Truncated())
    e.response_length = response_length_config({"chat_demo": True, "response_length_repair": True,
                                               "response_max_continuations": 0})
    try:
        await ended(e)
        await pump(e, lambda: e._candidate.error is not None)
        candidate = e._candidate
        await frame(e, .672)
        assert models.responses == 1 and candidate.restarts == 0
        assert "不能公开" not in json.dumps(socket.events, ensure_ascii=False)
        assert any(r["event"] == "speech_error" and r["data"]["code"] == "response_incomplete"
                   for r in socket.events)
        assert not e.assistant_history and not socket.audio
    finally:
        await cleanup(e)
