"""Deterministic candidate races. Fake models, real actor/queues/SSE cancellation."""
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from engine import ActorEngine, ControlMsg, FrameEvent, ModelDone
from actor_candidate import CandidateResult
from request_capacity import RequestCapacity, NORMAL, CONTROL
from stream_transport import PCMChunk
from speech_stream import SocketOutbox
from test_engine import ScriptedVAD


class Socket:
    def __init__(self):
        self.events, self.audio = [], []
    async def send_text(self, value): self.events.append(json.loads(value))
    async def send_bytes(self, value): self.audio.append(value)


class Models:
    def __init__(self, judge="switch", shift="no", blocked=None, fail_first=False):
        self.labels = {"judge": judge, "shift": shift}
        self.blocked = blocked
        self.gate = asyncio.Event()
        self.entered = {k: asyncio.Event() for k in ["judge", "shift", "response", "shift_re", "tts"]}
        self.closed, self.calls, self.tts_calls = [], [], []
        self.fail_first = fail_first
        self.responses = 0
    async def text(self, messages):
        kind = messages[0]["content"]
        self.calls.append((kind, deepcopy(messages)))
        self.entered[kind].set()
        try:
            if self.blocked == kind:
                await self.gate.wait()
            if kind in self.labels:
                yield self.labels[kind]
            else:
                self.responses += 1
                if self.fail_first and self.responses == 1:
                    yield "不能泄露的失败草稿"
                    raise RuntimeError("private failure")
                yield "第一句。第二句。"
        finally:
            self.closed.append(kind)
    async def tts(self, sentence):
        self.tts_calls.append(sentence)
        self.entered["tts"].set()
        try:
            if self.blocked == "tts":
                await self.gate.wait()
            for _ in range(4):
                yield PCMChunk(b"\0\0" * 960, 24000)
        finally:
            self.closed.append("tts")


def actor(models=None, speculative=True):
    m = models or Models()
    sock = Socket()
    e = ActorEngine(websocket=sock, prompts={k:k for k in ["judge", "shift", "response", "interrupt", "shift_s"]},
        engine_cfg={"stream_response": True, "chat_demo": True, "playback_autoend": True,
                    "control_validation": True, "speculative_response": speculative, "cancellable_response": True},
        vad_iterator=ScriptedVAD({}), llm_fn=lambda _: "continue", asr_fn=lambda _: "转写", tts_fn=lambda *a: None,
        text_stream_fn=m.text, tts_stream_fn=m.tts, request_capacity=RequestCapacity(4, 3))
    e.SHIFT_RE_PROMPT = "shift_re"
    e.asr_calls = []
    e.dispatch_asr = lambda audio, turn, answer_id=0: e.asr_calls.append((audio, turn, answer_id))
    return e, m, sock


async def frame(e, t, event=None, value=.1):
    e.detect_vad_frame = lambda _: event
    await e._process_event(FrameEvent(1, t, None, np.full(256, value, dtype=np.float32)))


async def pump(e, predicate):
    async def run():
        while not predicate():
            await e._process_event(await e.q.get())
    await asyncio.wait_for(run(), 2)


async def cleanup(e):
    e._reset_session()
    await asyncio.gather(*e._speech_tasks, *e._candidate_tasks, return_exceptions=True)
    await asyncio.sleep(0)  # deliver task completion callbacks
    while not e.q.empty():
        await e._process_event(e.q.get_nowait())
    assert e.request_capacity.snapshot()["active_total"] == 0
    assert e._inflight == 0


async def ended(e):
    await frame(e, .016, {"start": 0}, .1)
    await frame(e, .032, {"end": .032}, .2)


async def test_private_pipeline_runs_through_first_tts_without_any_public_output_or_history():
    e, m, sock = actor()
    try:
        await ended(e)
        c = e._candidate
        frozen = deepcopy(c.messages)
        await pump(e, lambda: c.pipeline is not None and c.pipeline.sent == 3840
                   and any(s.kind == "text_done" for s in c.stash))
        assert m.tts_calls == ["第一句。"], "only the first TTS sentence is speculative"
        assert not sock.audio and not e.asr_calls and not e.assistant_history
        assert not any(x["event"].startswith("speech_") for x in sock.events)
        e._users_by_turn[99], e._assistants_by_turn[99] = "late ASR", "late answer"
        await frame(e, .672)
        assert c.confirmed and c.published and c.messages == frozen
        assert len(c.audio) == 512 and not c.audio.flags.writeable, "hold silence is excluded"
        assert sock.audio and not e.assistant_history and not e.asr_calls
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": c.pipeline.sid, "played_samples": 0, "started": True}))
        assert len(e.asr_calls) == 1 and e.asr_calls[0][2] == c.cid
        assert e.assistant_history == ["第一句。第二句。"]
        assert e._candidate is None
        await pump(e, lambda: e._speech_audio_done)
        assert len(sock.audio) == 8 and len(m.tts_calls) == 2
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": c.pipeline.sid, "played_samples": 7680, "ended": True}))
        assert e.TURN_IDX == 1 and e.STATE == "LISTEN"
    finally:
        await cleanup(e)


@pytest.mark.parametrize("published", [False, True])
async def test_resume_cancels_private_or_published_unplayed_pipeline_and_keeps_original_input(published):
    e, m, sock = actor()
    try:
        await ended(e)
        c = e._candidate
        await pump(e, lambda: c.pipeline is not None and c.pipeline.sent == 3840)
        if published:
            await frame(e, .672, value=.3)
        before = len(sock.audio)
        await frame(e, .688 if published else .2, {"start": .18}, .9)
        assert e._candidate is None and e._speech is None
        assert e.STATE == "LISTEN" and e.IN_SPEECH and e.TURN_IDX == 0
        assert np.concatenate(e.BUFFER)[0] == np.float32(.1)
        assert np.concatenate(e.BUFFER)[-1] == np.float32(.9)
        await asyncio.gather(c.pipeline.task, return_exceptions=True)
        while not e.q.empty(): await e._process_event(e.q.get_nowait())
        assert len(sock.audio) == before and not e.asr_calls and not e.assistant_history
        assert c.pipeline.task.done() and "tts" in m.closed
        assert len([x for x in sock.events if x["event"] == "speech_cancelled"]) == int(published)
    finally:
        await cleanup(e)


@pytest.mark.parametrize("stage", ["judge", "shift", "response", "tts"])
async def test_resume_closes_actual_inflight_generator_at_each_stage(stage):
    e, m, sock = actor(Models(blocked=stage))
    if stage == "shift": e.TURN_IDX = 1
    try:
        await ended(e)
        c = e._candidate
        if stage != "judge":
            await pump(e, lambda: c.stage == stage or (stage == "tts" and c.pipeline is not None))
        if stage == "tts":
            # TTS enters only after the actor acknowledges the sentence event.
            await pump(e, lambda: any(ev.kind == "sentence" for ev in c.stash))
        await asyncio.wait_for(m.entered[stage].wait(), 1)
        await frame(e, .2, {"start": .18})
        await asyncio.gather(*c.tasks, *([c.pipeline.task] if c.pipeline else []), return_exceptions=True)
        assert stage in m.closed and not sock.audio and not e.assistant_history
        assert e.request_capacity.snapshot()["active_total"] == 0
    finally:
        await cleanup(e)


async def test_deferred_mode_uses_identical_end_snapshot_and_does_not_dispatch_early():
    early, _, _ = actor(Models(blocked="judge"))
    late, lm, lsock = actor(speculative=False)
    try:
        await ended(early); await ended(late)
        assert early._candidate.messages == late._candidate.messages
        assert not lm.calls and not lsock.audio
        await frame(late, .672)
        await pump(late, lambda: late._speech is not None)
        assert lm.calls[0][1] == early._candidate.messages["judge"]
    finally:
        await cleanup(early); await cleanup(late)


async def test_cancel_fences_queued_text_start_and_audio_on_slow_socket():
    e, _, sock = actor()
    blocked, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_send = sock.send_text

    async def send(value):
        event = json.loads(value)["event"]
        if event == "wire_block":
            blocked.set()
            await release.wait()
        await original_send(value)
        if event == "speech_cancelled":
            cancelled.set()

    sock.send_text = send
    e._outbox = SocketOutbox(sock, lambda sid: e._speech is not None and e._speech.sid == sid,
                             lambda: pytest.fail("unexpected socket failure"))
    try:
        await ended(e)
        c = e._candidate
        await pump(e, lambda: any(ev.kind == "audio" for ev in c.stash))
        await e.send_control("wire_block")
        await asyncio.wait_for(blocked.wait(), 1)
        await frame(e, .672)
        await frame(e, .7, {"start": .69})
        while not e.q.empty():
            await e._process_event(e.q.get_nowait())
        release.set()
        await asyncio.wait_for(cancelled.wait(), 1)
        assert not sock.audio
        assert not any(ev["event"] in ("speech_start", "speech_text_delta", "speech_text_done")
                       for ev in sock.events)
    finally:
        release.set()
        await cleanup(e)
        await e._outbox.close()


async def test_continue_never_generates_early_and_retains_original_continue_timeout():
    e, m, sock = actor(Models(judge="continue"))
    try:
        await ended(e)
        await pump(e, lambda: e._candidate.stage == "continue")
        assert m.responses == 0 and not e.CONTINUE_ARMED
        await frame(e, .672)
        assert e.CONTINUE_ARMED and e.t_continue_anchor == .672
        await frame(e, 3.1)
        assert m.responses == 0
        await frame(e, 3.18)
        await pump(e, lambda: e._speech is not None)
        assert e._candidate.stage == "response"
    finally:
        await cleanup(e)


async def test_shift_yes_never_generates_normal_response_or_commits_user_history():
    e, m, sock = actor(Models(shift="yes")); e.TURN_IDX = 1
    try:
        await ended(e)
        await pump(e, lambda: e._candidate.pipeline is not None)
        assert "response" not in [k for k, _ in m.calls]
        await frame(e, .672)
        await pump(e, lambda: e._speech_audio_done)
        assert [k for k, _ in m.calls] == ["judge", "shift", "shift_re"]
        p = e._speech
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": p.sid, "played_samples": p.sent, "ended": True}))
        assert not e.asr_calls and not e.assistant_history and e.TURN_IDX == 2
    finally:
        await cleanup(e)


async def test_failed_private_draft_is_discarded_before_one_confirmed_restart():
    e, m, sock = actor(Models(fail_first=True))
    try:
        await ended(e)
        await pump(e, lambda: e._candidate.error is not None)
        old_sid = e._candidate.pipeline.sid
        await frame(e, .672)
        await pump(e, lambda: e._speech_audio_done)
        assert e._speech.sid != old_sid and m.responses == 2
        assert "不能泄露" not in json.dumps(sock.events, ensure_ascii=False)
    finally:
        await cleanup(e)


async def test_normal_slots_bound_speculation_and_cancel_removes_waiter_without_using_control_reserve():
    e, m, _ = actor()
    try:
        async with AsyncExitStack() as stack:
            for _ in range(3): await stack.enter_async_context(e.request_capacity.slot(NORMAL))
            await ended(e)
            await asyncio.sleep(0); await asyncio.sleep(0)
            async with e.request_capacity.slot(CONTROL):
                assert not m.calls and e.request_capacity.snapshot()["active_total"] == 4
                await frame(e, .2, {"start": .18})
            await asyncio.gather(*e._candidate_tasks, return_exceptions=True)
            assert e.request_capacity.snapshot()["waiting_normal"] == 0
        assert not m.calls
    finally:
        await cleanup(e)


async def test_reset_and_late_asr_cannot_revive_candidate_or_overwrite_new_answer():
    e, m, sock = actor(Models(blocked="response"))
    try:
        await ended(e)
        await pump(e, lambda: e._candidate.pipeline is not None)
        old = e._candidate
        e._reset_session()
        await e._on_candidate_result(CandidateResult(old.cid, "judge", "switch", accounted=False))
        e._answer_versions[0] = 9
        await e._on_model_done(ModelDone("asr", e.session_gen, e.seg_epoch, 0,
                                        text="obsolete", answer_id=old.cid))
        assert not e.user_history and not sock.audio and e._candidate is None
    finally:
        await cleanup(e)


async def test_speech_after_playback_started_keeps_existing_interrupt_policy():
    e, m, sock = actor()
    try:
        await ended(e); await frame(e, .672)
        await pump(e, lambda: len(sock.audio) > 0)
        p = e._speech
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": p.sid, "played_samples": 0, "started": True}))
        await frame(e, .8, {"start": .78})
        assert e._speech is p and e.STATE == "SPEAK" and e.IN_SPEECH
        assert not any(x["event"] == "speech_cancelled" for x in sock.events)
    finally:
        await cleanup(e)
