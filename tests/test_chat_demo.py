"""Focused chat-profile contracts; no GPU/model weights or wall-clock sleeps."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from control_labels import parse_label, decide_control
from engine import ActorEngine, ControlMsg, ModelDone, FrameEvent, response_needs_completion
from test_engine import ScriptedVAD


def actor(**config):
    return ActorEngine(engine_cfg={"chat_demo": True, "playback_autoend": True,
                       "control_validation": True, **config}, vad_iterator=ScriptedVAD({}),
                       llm_fn=lambda m: "", asr_fn=lambda p: "", tts_fn=lambda t,p: p)


@pytest.mark.parametrize("text", [
    "好，我给你讲一个。", "别急，我这就开始。", "Sure, I'll tell you one.",
])
def test_short_promise_only_response_needs_completion(text):
    assert response_needs_completion(text)


@pytest.mark.parametrize("text", [
    "我来回答：北京。", "好，我给你讲一个。深夜，门突然响了。",
    "我来解释一下，电源线可能松了。", "答案是四。",
])
def test_response_with_payload_does_not_need_completion(text):
    assert not response_needs_completion(text)


async def test_demo_response_completion_appends_once_with_original_audio_history():
    calls = []

    async def text_stream(messages):
        calls.append(messages)
        yield "别急，我这就开始。" if len(calls) == 1 else "深夜，门外响起了脚步声。"

    async def tts_stream(_):
        if False:
            yield

    prompts = {"response": "R", "response_completion": "续写实际内容。"}
    e = ActorEngine(prompts=prompts,
        engine_cfg={"chat_demo": True, "stream_response": True,
                    "response_completion_repair": True},
        vad_iterator=ScriptedVAD({}), llm_fn=lambda _: "", asr_fn=lambda _: "",
        tts_fn=lambda *_: None, text_stream_fn=text_stream, tts_stream_fn=tts_stream)
    audio = {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,fixture"}}
    messages = [{"role": "system", "content": "R"},
                {"role": "user", "content": [audio]}]
    output = "".join([part async for part in e._response_text_stream(messages)])
    assert output == "别急，我这就开始。 深夜，门外响起了脚步声。"
    assert len(calls) == 2 and calls[0] is messages
    assert calls[1][:-2] == messages
    assert calls[1][-2:] == [
        {"role": "assistant", "content": "别急，我这就开始。"},
        {"role": "user", "content": "续写实际内容。"},
    ]


async def test_completion_repair_failure_keeps_the_streamed_draft():
    calls = 0

    async def text_stream(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("repair unavailable")
        yield "好，我给你讲一个。"

    async def tts_stream(_):
        if False:
            yield

    e = ActorEngine(prompts={"response": "R", "response_completion": "continue"},
        engine_cfg={"chat_demo": True, "stream_response": True,
                    "response_completion_repair": True},
        vad_iterator=ScriptedVAD({}), llm_fn=lambda _: "", asr_fn=lambda _: "",
        tts_fn=lambda *_: None, text_stream_fn=text_stream, tts_stream_fn=tts_stream)
    messages = [{"role": "system", "content": "R"}]
    assert "".join([part async for part in e._response_text_stream(messages)]) == "好，我给你讲一个。"
    assert calls == 2


@pytest.mark.parametrize("kind,good,bad", [("judge", "'SWITCH'.", "do not switch"),
    ("interrupt", "continue", "continue or switch"), ("shift", "yes", "yesterday")])
def test_exact_control_labels(kind, good, bad):
    assert parse_label(kind, good) is not None
    assert parse_label(kind, bad) is None
    assert parse_label(kind, "") is None


@pytest.mark.parametrize("kind,fallback", [("judge", "continue"), ("interrupt", "continue"), ("shift", "no")])
async def test_one_repair_and_safe_fallback(kind, fallback):
    seen = []
    messages = [{"role": "user", "content": "original audio placeholder"}]
    async def call(request):
        seen.append(request)
        return "invalid explanation"
    label, audit = await decide_control(call, messages, kind, 1)
    assert label == fallback and audit["fallback"] and len(seen) == 2
    assert seen[1][:1] == messages and len(messages) == 1


async def test_repair_success_and_timeout_do_not_loop():
    answers = iter(["", "switch"])
    async def call(_): return next(answers)
    label, audit = await decide_control(call, [], "judge", 1)
    assert label == "switch" and audit["repaired"] and not audit["fallback"]
    async def timeout(_): raise asyncio.TimeoutError
    label, audit = await decide_control(timeout, [], "shift", 1)
    assert label == "no" and audit["timed_out"] and audit["attempts"] == 1


async def test_finish_turn_once_and_keep_ongoing_user_audio():
    e = actor()
    e.STATE = "SPEAK"; e.IN_SPEECH = True
    e.BUFFER = [np.array([1.], dtype=np.float32)]
    e.interrupt_buf = [np.array([2.], dtype=np.float32)]
    e._speech_audio_done = True
    e._speech = SimpleNamespace(sid=1, played=10, sent=10, progress=lambda n: True)
    e._speech_meta = SimpleNamespace(turn=0)
    ack = ControlMsg("playback_progress", {"utterance_id": 1, "played_samples": 10, "ended": True})
    await e._process_event(ack)
    await e._process_event(ack)
    assert e.TURN_IDX == 1 and e.STATE == "LISTEN" and not e.interrupt_buf
    assert e._speech is None
    e._cancel_speech("superseded")
    assert e.q.empty(), "a completed reply must not be cancelled by the next reply"
    await e._listen_frame(FrameEvent(1, .016, None, np.array([3.], dtype=np.float32)), None)
    assert np.concatenate(e.BUFFER).tolist() == [2., 3.]
    assert len([r for r in e.trace if r["event"] == "turn_finished"]) == 1


async def test_finished_closed_segment_rejudges_and_old_interrupt_cannot_apply():
    e = actor(); e.STATE = "SPEAK"; e.IN_SPEECH = e._seg_closed = True
    e.interrupt_buf = [np.zeros(256)]
    calls = []
    e.dispatch_llm = lambda kind, *a, **k: calls.append(kind)
    await e._finish_turn(0, "played")
    assert calls == ["judge"] and e.TURN_IDX == 1
    await e._on_model_done(ModelDone("interrupt", 0, 0, 0, text="switch"))
    assert e.TURN_IDX == 1 and calls == ["judge"]


async def test_late_asr_pairs_by_turn_not_arrival_order():
    e = actor()
    e._assistants_by_turn = {0: "answer zero", 1: "answer one"}
    for turn in [1, 0]:
        await e._on_asr(ModelDone("asr", 0, 0, turn, text="user " + str(turn)))
    msgs = e.build_messages("system", None, True, False)
    assert [m["content"][0]["text"] for m in msgs if m["role"] == "user"] == ["user 0", "user 1"]
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == ["answer zero", "answer one"]


async def test_empty_control_never_auto_switches_or_dead_ends():
    e = actor(); e._judged_seg_end = 0
    calls = []
    e.dispatch_llm = lambda kind, *a, **k: calls.append(kind)
    e.dispatch_asr = lambda *a: None
    await e._on_model_done(ModelDone("judge", 0, 0, 0, text=""))
    assert e.CONTINUE_ARMED and not calls
    await e._on_model_done(ModelDone("shift", 0, 0, 0, text=""))
    assert calls == ["response"]


def test_demo_profile_is_separate_and_asr_configuration_reaches_factory(monkeypatch):
    from backend import load_runtime_config
    import module
    base = Path(__file__).resolve().parents[1] / "src/config.yaml"
    frozen, demo = load_runtime_config(base), load_runtime_config(base, True)
    assert not frozen["engine"]["playback_autoend"] and "15" in frozen["prompts"]["response"]
    assert demo["engine"]["control_validation"] and demo["engine"]["warmup"]
    assert "15" not in demo["prompts"]["response"]
    assert demo["prompts"]["interrupt"] == frozen["prompts"]["interrupt"]
    assert frozen["time"] == demo["time"]
    for key in ("FDBC_ASR_BACKEND", "FDBC_ASR_PROVIDER", "FDBC_ASR_NUM_THREADS"):
        monkeypatch.delenv(key, raising=False)
    for name in ("ASR_BACKEND", "ASR_PROVIDER", "ASR_NUM_THREADS", "_ASR_MODEL"):
        monkeypatch.setattr(module, name, getattr(module, name))
    module.configure_asr(demo["asr"])
    assert module.ASR_BACKEND == "sensevoice" and module.ASR_PROVIDER == "cpu"
    monkeypatch.setenv("FDBC_ASR_BACKEND", "paraformer_zh")
    module.configure_asr(demo["asr"])
    assert module.ASR_BACKEND == "paraformer_zh"


def test_warmup_blocks_readiness_and_failure_is_not_ready(monkeypatch):
    from backend import create_app
    from fastapi.testclient import TestClient
    import demo_startup
    calls = []
    async def warm(prompts, engine_cfg):
        assert engine_cfg["warmup"]
        calls.append(prompts)
    monkeypatch.setattr(demo_startup, "warmup", warm)
    with TestClient(create_app({"response": "R"}, {}, engine_cfg={"warmup": True})) as client:
        assert calls == [{"response": "R"}]
        assert client.get("/api/demo/info").status_code == 200
    async def fail(_, engine_cfg): raise RuntimeError("warmup failed")
    monkeypatch.setattr(demo_startup, "warmup", fail)
    with pytest.raises(RuntimeError, match="warmup failed"):
        with TestClient(create_app({}, {}, engine_cfg={"warmup": True})):
            pass
