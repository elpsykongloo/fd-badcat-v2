"""Focused chat-profile contracts; no GPU/model weights or wall-clock sleeps."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from control_labels import parse_label, decide_control
from engine import ActorEngine, ControlMsg, ModelDone, FrameEvent
from test_engine import ScriptedVAD


def actor(**config):
    return ActorEngine(engine_cfg={"chat_demo": True, "playback_autoend": True,
                       "control_validation": True, **config}, vad_iterator=ScriptedVAD({}),
                       llm_fn=lambda m: "", asr_fn=lambda p: "", tts_fn=lambda t,p: p)


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
    async def warm(prompts): calls.append(prompts)
    monkeypatch.setattr(demo_startup, "warmup", warm)
    with TestClient(create_app({"response": "R"}, {}, engine_cfg={"warmup": True})) as client:
        assert calls == [{"response": "R"}]
        assert client.get("/api/demo/info").status_code == 200
    async def fail(_): raise RuntimeError("warmup failed")
    monkeypatch.setattr(demo_startup, "warmup", fail)
    with pytest.raises(RuntimeError, match="warmup failed"):
        with TestClient(create_app({}, {}, engine_cfg={"warmup": True})):
            pass
