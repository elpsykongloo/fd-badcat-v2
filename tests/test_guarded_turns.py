"""Real queues/cancellation with deterministic model and VAD evidence; no sleeps."""
import asyncio
import json
import struct

import numpy as np
import pytest

from test_actor_candidate import actor, Models, frame, pump, ended
from engine import ControlMsg, ModelDone, FrameEvent
from guarded_turns import InputDecision, route_messages
from input_audio import EchoEvidence, decode_input_packet, INPUT_HEADER
from control_labels import parse_label


def guarded(route="yield_ready", interrupt="switch", blocked=None):
    m = Models(blocked=blocked)
    m.labels.update(input_route=route, interrupt=interrupt)
    m.entered.update(input_route=asyncio.Event(), interrupt=asyncio.Event())
    e, _, sock = actor(m)
    e.engine_cfg["guarded_turns"] = True
    e.prompts["input_route"] = "input_route"
    e._init_guarded()
    return e, m, sock


async def cleanup(e):
    e._reset_session()
    await asyncio.gather(*e._guard_tasks, *e._speech_tasks, *e._candidate_tasks, return_exceptions=True)
    await asyncio.sleep(0)
    while not e.q.empty():
        await e._process_event(e.q.get_nowait())
    assert e._inflight == 0
    assert e.request_capacity.snapshot()["active_total"] == 0


async def private(e):
    await ended(e)
    await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None
               and e._candidate.pipeline.sent > 0)


async def playing(e):
    await private(e)
    await frame(e, .704)
    await pump(e, lambda: e._speech is not None and e.STATE == "SPEAK")
    await e._process_event(ControlMsg("playback_progress", {
        "utterance_id": e._speech.sid, "played_samples": 0, "started": True}))


async def new_input(e, t=1.):
    await frame(e, t, {"start": t}, .2)
    await frame(e, t + .032, {"end": t + .032}, .3)


@pytest.mark.parametrize("route", ["keep", "nonsense", "keep or yield_ready"])
async def test_uncertain_or_invalid_audio_does_not_cancel_or_add_history(route):
    e, m, sock = guarded()
    try:
        await playing(e)
        sid, epoch = e._speech.sid, e.seg_epoch
        m.labels["input_route"] = route
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        assert e._speech.sid == sid and e.seg_epoch == epoch
        assert m.responses == 1 and len(e.asr_calls) == 1
        await frame(e, 10)
        assert m.responses == 1
    finally:
        await cleanup(e)


async def test_typed_route_does_not_inherit_binary_false_negative_or_issue_two_requests():
    e, m, _ = guarded()
    try:
        await playing(e)
        m.labels["interrupt"] = "continue"
        m.labels["input_route"] = "stop_only"
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        assert e._speech is None and m.responses == 1
        assert not any(kind == "interrupt" for kind, _ in m.calls)
    finally:
        await cleanup(e)


@pytest.mark.parametrize("route,reason", [("stop_only", "stop_only"), ("yield_wait", "awaiting_user")])
async def test_stop_and_explicit_wait_remain_silent_after_old_timeout(route, reason):
    e, m, _ = guarded()
    try:
        await playing(e)
        m.labels["input_route"] = route
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        assert e._speech is None and e.STATE == "LISTEN"
        assert e._guard_wait_reason == reason
        for t in [2, 4, 8, 15]:
            await frame(e, t)
        assert e._candidate is None and m.responses == 1 and len(e.asr_calls) == 1
    finally:
        await cleanup(e)


async def test_wait_then_continuation_produces_one_combined_answer():
    e, m, _ = guarded()
    try:
        await playing(e)
        m.labels["input_route"] = "yield_wait"
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        waiting = sum(map(len, e._guard_wait_audio))
        m.labels["input_route"] = "yield_ready"
        await new_input(e, 5)
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
        assert len(e._candidate.audio) > waiting and e._candidate.turn == 1
        await frame(e, 5.704)
        await pump(e, lambda: e._speech is not None and e._speech_meta.turn == 1)
        assert m.responses == 2
    finally:
        await cleanup(e)


async def test_long_vad_triggers_judgement_not_unconditional_stop():
    e, m, _ = guarded()
    try:
        await playing(e)
        sid = e._speech.sid
        m.labels["input_route"] = "keep"
        await frame(e, 1, {"start": 1})
        await frame(e, 2.6)
        await pump(e, lambda: e._guard_input.provisional_keep)
        assert e._speech.sid == sid and m.responses == 1
        m.labels["input_route"] = "yield_ready"
        await frame(e, 4.2)
        await pump(e, lambda: e._guard_input.admitted)
        assert e._speech is None and e._candidate is None, "ongoing speech cannot publish READY"
        await frame(e, 4.3, {"end": 4.3})
        await pump(e, lambda: e._candidate is not None)
    finally:
        await cleanup(e)


async def test_noise_before_publication_preserves_original_candidate_and_epoch():
    e, m, sock = guarded()
    try:
        await private(e)
        original, epoch = e._candidate, e.seg_epoch
        m.labels["input_route"] = "keep"
        await new_input(e, .2)
        await pump(e, lambda: e._guard_input.decided)
        assert e._candidate is original and e.seg_epoch == epoch
        await frame(e, .8)
        assert e._candidate is original and e._candidate.published
        assert m.responses == 1 and not e.asr_calls
    finally:
        await cleanup(e)


async def test_accepted_preplay_resume_merges_original_question_without_advancing_turn():
    e, m, _ = guarded()
    try:
        await private(e)
        old = e._candidate
        await new_input(e, .2)
        await pump(e, lambda: e._candidate is not None and e._candidate is not old)
        assert e.TURN_IDX == 0 and len(e._candidate.audio) > len(old.audio)
        assert not e.asr_calls and not e.assistant_history
    finally:
        await cleanup(e)


async def test_preplay_noise_holds_and_releases_scheduled_audio_without_cancelling():
    e, m, sock = guarded()
    try:
        await private(e)
        await frame(e, .704)
        await pump(e, lambda: e._speech is not None and e.STATE == "SPEAK")
        sid = e._speech.sid
        m.labels["input_route"] = "keep"
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        holds = [x["data"]["held"] for x in sock.events if x["event"] == "speech_hold"]
        assert holds == [True, False]
        assert e._speech.sid == sid and not e._speech_started
    finally:
        await cleanup(e)


async def test_eof_does_not_reclassify_pending_input_or_resurrect_rejected_audio():
    e, m, _ = guarded()
    try:
        await playing(e)
        m.blocked = "input_route"
        m.labels["input_route"] = "keep"
        await new_input(e)
        span = e._guard_input
        await e._finish_turn(0, "played")
        assert e._guard_input is span and e._candidate is None
        m.gate.set()
        await pump(e, lambda: span.decided)
        await frame(e, 8)
        assert m.responses == 1 and e._candidate is None
    finally:
        await cleanup(e)


async def test_reset_closes_pending_input_sse_and_stale_result_never_answers():
    e, m, _ = guarded(blocked="input_route")
    await ended(e)
    await asyncio.wait_for(m.entered["input_route"].wait(), 1)
    stale = InputDecision(e.session_gen, e._guard_input.sid, e._guard_input.revision,
                          True, "yield_ready")
    await cleanup(e)
    await e._process_event(stale)
    assert e._candidate is None and m.responses == 0 and "input_route" in m.closed


async def test_legacy_interrupt_callback_cannot_bypass_guard():
    e, m, _ = guarded()
    try:
        await playing(e)
        sid = e._speech.sid
        await e._on_interrupt(ModelDone("interrupt", e.session_gen, e.seg_epoch, 0, text="switch"))
        assert e._speech.sid == sid and m.responses == 1
    finally:
        await cleanup(e)


async def test_cancel_ack_updates_only_confirmed_sentence_prefix_and_ignores_invalid_ack():
    e, m, _ = guarded()
    try:
        await playing(e)
        sid = e._speech.sid
        e._guard_outputs[sid]["sentences"] = [(100, "第一句。"), (100000, "没播完的句子。")]
        e._guard_mark_cancelled()
        await e._guard_control("playback_stopped", {"utterance_id": sid, "played_samples": 100})
        assert e._assistants_by_turn[0].startswith("第一句。")
        assert "没播完的句子" not in e._assistants_by_turn[0]
        old = e._assistants_by_turn[0]
        await e._guard_control("playback_stopped", {"utterance_id": sid, "played_samples": 100000})
        assert e._assistants_by_turn[0] == old
    finally:
        await cleanup(e)


async def test_reset_discards_cancel_history_and_late_ack():
    e, m, _ = guarded()
    await playing(e)
    sid = e._speech.sid
    await cleanup(e)
    await e._guard_control("playback_stopped", {"utterance_id": sid, "played_samples": 0})
    assert not e._guard_outputs and not e._assistants_by_turn


async def test_ready_audio_is_frozen_at_end_not_model_return():
    e, m, _ = guarded(blocked="input_route")
    try:
        await ended(e)
        await m.entered["input_route"].wait()
        for t in [.05, .08, .12]:
            await frame(e, t, value=.8)
        m.gate.set()
        await pump(e, lambda: e._candidate is not None)
        assert len(e._candidate.audio) == 512
        assert np.max(e._candidate.audio) == np.float32(.2)
    finally:
        await cleanup(e)


async def test_playback_start_race_requires_updated_context_before_stopping():
    e, m, _ = guarded()
    try:
        await private(e)
        await frame(e, .704)
        await pump(e, lambda: e._speech is not None)
        m.blocked = "input_route"
        m.labels["input_route"] = "stop_only"
        await new_input(e)
        span = e._guard_input
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": e._speech.sid, "played_samples": 0, "started": True}))
        await e._on_input_decision(InputDecision(e.session_gen, span.sid, span.revision,
            True, "stop_only", {"playing": False}))
        assert not span.admitted and e._speech is not None and span.mode == "playing"
        m.labels["input_route"] = "keep"
        m.gate.set()
        await pump(e, lambda: span.decided)
        assert e._speech is not None and m.responses == 1
    finally:
        await cleanup(e)


async def test_reference_echo_is_rejected_before_model_dispatch_in_actor():
    e, m, _ = guarded()
    try:
        rng = np.random.default_rng(42)
        reference = rng.normal(0, .1, 256 * 32).astype(np.float32)
        echo = np.concatenate([np.zeros(1024, dtype=np.float32), reference[:-1024] * .4])
        for i in range(32):
            event = {"start": 0} if i == 4 else ({"end": .5} if i == 31 else None)
            e.detect_vad_frame = lambda pcm, event=event: event
            await e._process_event(FrameEvent(i, (i + 1) * .016, None,
                echo[i*256:(i+1)*256], reference[i*256:(i+1)*256]))
        await pump(e, lambda: e._guard_input.decided)
        assert e._guard_input.route == "keep" and not m.calls and e.seg_epoch == 0
    finally:
        await cleanup(e)


def test_input_packet_is_bounded_and_sequenced():
    pcm = np.tile(np.array([8192, -16384], dtype="<i2"), 256)
    raw = INPUT_HEADER.pack(b"FDM1", 0, 256, 16000) + pcm.tobytes()
    mic, ref = decode_input_packet(raw, 0)
    assert np.all(mic == .25) and np.all(ref == -.5)
    for invalid in [raw[:-1], raw + b"x", b"xxxx" + raw[4:]]:
        with pytest.raises(ValueError): decode_input_packet(invalid, 0)
    with pytest.raises(ValueError): decode_input_packet(raw, 1)


def test_echo_evidence_rejects_delayed_echo_but_preserves_double_talk():
    rng = np.random.default_rng(9)
    reference = rng.normal(0, .1, 256 * 50).astype(np.float32)
    echo = np.concatenate([np.zeros(1024, dtype=np.float32), reference[:-1024] * .3])
    near = rng.normal(0, .08, len(reference)).astype(np.float32)
    # Slow near-end energy must not vanish behind the whitened-correlation test.
    low_near = (.08 * np.sin(np.arange(len(reference)) * 2 * np.pi * 50 / 16000)).astype(np.float32)
    for mic, expected in [(echo, True), (echo + near, False), (echo + low_near, False)]:
        guard = EchoEvidence()
        for start in range(0, len(mic), 256):
            evidence = guard.process(mic[start:start+256], reference[start:start+256])
        assert evidence["echo_only"] is expected
    assert len(guard.ref) <= 38 and len(guard.mic) <= 8


def test_route_protocol_is_exact_and_has_no_implicit_ready():
    assert parse_label("input_route", '{"transcript":"停","label":"stop_only"}') == "stop_only"
    for value in ["", "keep", "'STOP_ONLY'.", "probably yield_ready", "keep or yield_ready", None]:
        assert parse_label("input_route", value) is None


def test_route_metadata_cannot_absorb_following_microphone_block():
    content = {"type": "audio_url", "audio_url": {"url": "test"}}
    for reference in ["", "参考文字：\n\"不是用户输入\"" * 100]:
        messages = route_messages("route", content, playing=True, reference=reference)
        metadata, delim = messages[1]["content"][0]["text"].split("\n")
        fields = json.loads(metadata.removeprefix("情境资料："))
        assert fields == {"assistant_playing": True}
        assert "麦克风采样" in delim
        assert messages[1]["content"][1] is content
