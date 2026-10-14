"""Real queues/cancellation with deterministic model and VAD evidence; no sleeps."""
import asyncio
import json
import re
import struct
import sys

import numpy as np
import pytest

from test_actor_candidate import actor, Models, frame, pump, ended
from engine import ControlMsg, ModelDone, FrameEvent
from guarded_turns import InputDecision, route_messages
from input_audio import EchoEvidence, decode_input_packet, INPUT_HEADER
from control_labels import parse_label


@pytest.fixture(autouse=True, params=[0, 2000], ids=['ordered-tts', 'prefetch-tts'])
def streaming_schedule(request, monkeypatch):
    original = actor
    def configured(*args, **kwargs):
        e, m, sock = original(*args, **kwargs)
        e.engine_cfg.update(stream_prefetch_ms=request.param, stream_startup_ms=350,
                            stream_diagnostics=True)
        return e, m, sock
    monkeypatch.setattr(sys.modules[__name__], 'actor', configured)


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
        assert e._assistants_by_turn[0] == "第一句。"
        assert "没播完的句子" not in e._assistants_by_turn[0]
        e._users_by_turn[0] = "继续。"
        messages = e.build_messages("system", None, True, False)
        assert [x["content"] for x in messages if x["role"] == "assistant"] == ["第一句。"]
        old = e._assistants_by_turn[0]
        await e._guard_control("playback_stopped", {"utterance_id": sid, "played_samples": 100000})
        assert e._assistants_by_turn[0] == old
    finally:
        await cleanup(e)


async def test_cancel_before_a_complete_sentence_removes_unheard_answer_from_history():
    e, m, _ = guarded()
    try:
        await playing(e)
        sid = e._speech.sid
        e._guard_outputs[sid]["sentences"] = [(1000, "尚未播完。")]
        e._guard_mark_cancelled()
        await e._guard_control("playback_stopped", {"utterance_id": sid, "played_samples": 100})
        assert 0 not in e._assistants_by_turn
        assert not any(x["role"] == "assistant"
                       for x in e.build_messages("system", None, True, False))
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


async def test_accepted_followup_during_playback_answers_without_audio_shift_veto():
    e, m, _ = guarded()
    try:
        await playing(e)
        m.labels["shift"] = "yes"
        await new_input(e)
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
        assert e.TURN_IDX == 1 and e._candidate.stage == "response"
        await frame(e, 1.704)
        await pump(e, lambda: e._speech is not None and e._speech_meta.turn == 1)
        assert m.responses == 2
        assert not any(kind in ("shift", "shift_s") for kind, _ in m.calls)
    finally:
        await cleanup(e)


def released(seconds=2.0):
    e, m, sock = guarded()
    e.engine_cfg["input_wait_release_s"] = seconds
    e._init_guarded()
    return e, m, sock


async def test_stalled_wait_is_answered_after_user_silence():
    e, m, _ = released()
    try:
        await playing(e)
        m.labels["input_route"] = "yield_wait"
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        held = np.concatenate(e._guard_wait_audio[:-1]).copy()
        for t in (1.5, 2.0, 2.9):
            await frame(e, t)
        assert e._candidate is None and e._guard_wait_reason == "awaiting_user"
        await frame(e, 3.1)
        assert e._candidate is not None and e._candidate.stage == "response"
        np.testing.assert_array_equal(e._candidate.audio, held)
        assert not e._guard_wait_audio and e._guard_wait_end is None and e._guard_wait_reason is None
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
        await frame(e, 3.2)
        await pump(e, lambda: e._speech is not None and e._speech_meta.turn == 1)
        assert m.responses == 2
    finally:
        await cleanup(e)


async def test_continuation_before_release_still_merges_into_one_answer():
    e, m, _ = released()
    try:
        await playing(e)
        m.labels["input_route"] = "yield_wait"
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        waiting = sum(map(len, e._guard_wait_audio))
        m.labels["input_route"] = "yield_ready"
        await new_input(e, 2.5)
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
        assert len(e._candidate.audio) > waiting
        for t in (3.2, 5, 8):
            await frame(e, t)
        await pump(e, lambda: e._speech is not None and e._speech_meta.turn == 1)
        assert m.responses == 2
    finally:
        await cleanup(e)


@pytest.mark.parametrize("route", ["stop_only", "keep"])
async def test_release_never_answers_stop_or_rejected_audio(route):
    e, m, _ = released()
    try:
        await playing(e)
        m.labels["input_route"] = route
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        for t in (3, 6, 12):
            await frame(e, t)
        assert m.responses == 1
    finally:
        await cleanup(e)


def test_release_setting_is_bounded():
    e, _, _ = guarded()
    e.engine_cfg["input_wait_release_s"] = 30
    with pytest.raises(ValueError):
        e._init_guarded()


def route_context(messages):
    head = messages[1]["content"][0]["text"]
    return json.loads(head.split("情境资料：", 1)[1].split("\n", 1)[0])


def test_route_conversation_is_opt_in_and_never_carries_unheard_text():
    e, _, _ = guarded()
    span = type("Span", (), {"reply_context": "小狐狸发现了小鸟。"})()
    e._users_by_turn, e._assistants_by_turn = {0: "讲个故事"}, {}
    assert e._guard_conversation(span, True) is None
    e.engine_cfg["input_route_conversation"] = True
    playing = e._guard_conversation(span, True)
    assert playing == {"previous_user_request": "讲个故事", "assistant_speech_heard": "小狐狸发现了小鸟。"}
    # Idle: only a turn with both sides in ACK-based history; never a reply still being generated.
    e._users_by_turn[1] = "然后呢"
    assert e._guard_conversation(span, False) == {"previous_user_request": "然后呢", "assistant_speech_heard": ""}
    e._assistants_by_turn[0] = "小狐狸发现了小鸟。"
    assert e._guard_conversation(span, False)["assistant_speech_heard"] == "小狐狸发现了小鸟。"
    e._users_by_turn, e._assistants_by_turn = {}, {}
    span.reply_context = ""
    assert e._guard_conversation(span, True) is None


def test_route_request_carries_conversation_only_when_enabled():
    msgs = route_messages("p", {"type": "audio_url"}, playing=True)
    assert "conversation" not in route_context(msgs)
    conv = {"previous_user_request": "讲个故事", "assistant_speech_heard": "从前有座山。"}
    msgs = route_messages("p", {"type": "audio_url"}, playing=True, conversation=conv)
    assert route_context(msgs)["conversation"] == conv and route_context(msgs)["assistant_playing"] is True


async def test_live_route_dispatch_sends_heard_conversation():
    e, m, _ = guarded()
    e.engine_cfg["input_route_conversation"] = True
    e._init_guarded()
    try:
        await playing(e)
        e._users_by_turn[0] = "讲个故事"
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        sent = [msgs for kind, msgs in m.calls if kind.startswith("input_route")]
        assert sent, m.calls
        meta = route_context(sent[-1])
        assert meta["conversation"]["previous_user_request"] == "讲个故事"
        assert meta["conversation"]["assistant_speech_heard"] == e._guard_input.reply_context
    finally:
        await cleanup(e)


EMPTY_KEEP = '{"transcript":"","label":"keep"}'
# Audio-clock comparisons must not depend on float rounding of t - since.
EPS = 1e-6


def recovering(seconds=2.0, **config):
    e, m, sock = guarded()
    e.engine_cfg.update(input_wait_release_s=seconds, input_wait_uncertain_recover=True, **config)
    e._init_guarded()
    seen = []
    def observe(event, data=None):
        seen.append((event, data))
    e._observe = observe
    return e, m, sock, seen


def observed(seen, event):
    return [data for name, data in seen if name == event]


async def cut_then_keep(e, m, t=1., keep=EMPTY_KEEP, admit="yield_ready"):
    """Admit an unclosed input (playback is cut), then close it with KEEP."""
    m.labels["input_route"] = admit
    await frame(e, t, {"start": t})
    await frame(e, t + 1.6)
    await pump(e, lambda: e._guard_input.admitted)
    assert e._speech is None and e._guard_wait_reason == "user_speaking"
    m.labels["input_route"] = keep
    await frame(e, t + 1.7, {"end": t + 1.7})
    await pump(e, lambda: e._guard_input.decided)
    assert e._guard_wait_reason == "uncertain_after_yield"
    return t + 1.7


async def started(e, kind):
    await pump(e, lambda: e._speech is not None and e._speech_meta.kind == kind and e._speech.sent)
    await e._process_event(ControlMsg("playback_progress", {
        "utterance_id": e._speech.sid, "played_samples": 0, "started": True}))


async def recovery_played(e, m, tts_before):
    await started(e, "recover")
    await pump(e, lambda: e._speech_audio_done)
    assert "".join(m.tts_calls[tts_before:]) == e._speech_candidate.fixed_text
    await e._process_event(ControlMsg("playback_progress", {
        "utterance_id": e._speech.sid, "played_samples": e._speech.sent, "ended": True}))
    assert e._speech is None


def test_every_wait_reason_has_an_explicit_policy():
    import re
    from pathlib import Path
    import guarded_turns
    from guarded_turns import WAIT_POLICIES
    source = Path(guarded_turns.__file__).read_text(encoding="utf-8")
    # The setter is the only writer; every reason it is given needs a policy.
    assert re.findall(r"_guard_wait_reason = ", source) == ["_guard_wait_reason = "] * 3
    assert set(re.findall(r'_guard_set_wait\("(\w+)"\)', source)) == set(WAIT_POLICIES)
    assert set(WAIT_POLICIES.values()) <= {"none", "release", "recover"}
    # Deliberate silence is an explicit, reviewed whitelist (see WAIT_POLICIES comments).
    assert {r for r, p in WAIT_POLICIES.items() if p == "none"} == {
        "user_speaking", "stop_only", "input_limit"}
    e, _, _ = guarded()
    with pytest.raises(ValueError):
        e._guard_set_wait("unlisted")


async def wait_awaiting_user(e, m):
    m.labels["input_route"] = "yield_wait"
    await new_input(e)
    await pump(e, lambda: e._guard_input.decided)
    return e._guard_wait_end


async def wait_uncertain_after_yield(e, m):
    return await cut_then_keep(e, m)


@pytest.mark.parametrize("reason", ["awaiting_user", "uncertain_after_yield"])
async def test_each_non_silent_wait_policy_fires_exactly_once(reason):
    from guarded_turns import WAIT_POLICIES
    assert {r for r, p in WAIT_POLICIES.items() if p != "none"} == {"awaiting_user", "uncertain_after_yield"}
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await globals()["wait_" + reason](e, m)
        assert e._guard_wait_reason == reason
        await frame(e, since + 1.984)
        assert e._candidate is None
        tts_before = len(m.tts_calls)
        await frame(e, since + 2 + EPS)
        assert e._candidate is not None and e._guard_wait_reason is None
        if WAIT_POLICIES[reason] == "release":
            await pump(e, lambda: e._speech is not None and e._speech_meta.turn == 1 and m.responses == 2)
        else:
            await recovery_played(e, m, tts_before)
            assert m.responses == 1
        for t in (since + 4, since + 8, since + 15):
            await frame(e, t)
        fired = observed(seen, "input_wait_released") + observed(seen, "input_cut_recovery")
        assert len(fired) == 1 and fired[0]["silence_s"] == 2.0
        assert len(observed(seen, "candidate_created")) == 2
    finally:
        await cleanup(e)


@pytest.mark.parametrize("keep,cause", [(EMPTY_KEEP, "empty_transcript"), ("nonsense", "fallback")])
async def test_cut_with_nothing_heard_asks_once_without_model_or_user_history(keep, cause):
    e, m, sock, seen = recovering()
    try:
        await playing(e)
        users, asr = dict(e._users_by_turn), list(e.asr_calls)
        since = await cut_then_keep(e, m, keep=keep)
        calls = len(m.calls)
        tts_before = len(m.tts_calls)
        await frame(e, since + 2 + EPS)
        c = e._candidate
        assert c.stage == "recover" and c.fixed_text == "抱歉，我刚才没听清，请再说一遍。"
        await recovery_played(e, m, tts_before)
        assert observed(seen, "input_cut_recovery") == [{"reason": cause, "streak": 1, "silence_s": 2.0}]
        assert len(m.calls) == calls and m.responses == 1, "no model call for the fixed sentence"
        assert e.asr_calls == asr and e._users_by_turn == users
        assert not any("没听清" in text for text in e.assistant_history)
        assert not any("没听清" in text for text in e._assistants_by_turn.values())
        assert not e._guard_wait_audio and e._guard_wait_end is None
    finally:
        await cleanup(e)


async def test_recovery_speaks_english_after_english_route_transcript():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m, admit='{"transcript":"what about tomorrow","label":"yield_ready"}')
        await frame(e, since + 2 + EPS)
        assert e._candidate.fixed_text == "Sorry, I didn't catch that. Could you say it again?"
    finally:
        await cleanup(e)


@pytest.mark.parametrize("enabled", [False, True])
async def test_heard_keep_stop_and_flag_off_stay_silent(enabled):
    e, m, sock, seen = recovering()
    e.engine_cfg["input_wait_uncertain_recover"] = enabled
    e._init_guarded()
    e._observe = lambda event, data=None: seen.append((event, data))
    try:
        await playing(e)
        # Heard content (backchannel / third party) is never asked to be repeated.
        since = await cut_then_keep(e, m, keep="keep")
        for t in (since + 2, since + 6):
            await frame(e, t)
        # A STOP that was followed by an unusable close is still a STOP.
        since = await cut_then_keep(e, m, t=since + 7, admit="stop_only")
        for t in (since + 2, since + 6):
            await frame(e, t)
        assert e._candidate is None and m.responses == 1
        assert not observed(seen, "input_cut_recovery")
        skipped = [x["reason"] for x in observed(seen, "input_cut_recovery_skipped")]
        assert skipped == (["heard_content", "stop_requested"] if enabled else [])
    finally:
        await cleanup(e)


async def test_flag_off_cut_without_content_matches_flag_absent():
    runs = []
    for config in ({}, {"input_wait_uncertain_recover": False, "input_recover_max": 1}):
        e, m, sock = guarded()
        e.engine_cfg.update(input_wait_release_s=2.0, **config)
        e._init_guarded()
        seen = []
        e._observe = lambda event, data=None: seen.append(event)
        try:
            await playing(e)
            since = await cut_then_keep(e, m)
            for t in (since + 2 + EPS, since + 4, since + 12):
                await frame(e, t)
            assert e._candidate is None and e._guard_wait_reason == "uncertain_after_yield"
            assert not any(name.startswith("input_cut_recovery") for name in seen)
            runs.append(([x["event"] for x in sock.events], seen,
                         [kind for kind, _ in m.calls], list(m.tts_calls)))
        finally:
            await cleanup(e)
    assert runs[0] == runs[1]


async def test_onset_one_frame_before_expiry_pauses_and_rejected_input_restarts_count():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        # Unusable again (noise): the rejection only pauses the recovery.
        m.labels["input_route"] = EMPTY_KEEP
        await frame(e, since + 1.984, {"start": since + 1.984}, .2)
        await frame(e, since + 2.0, value=.2)
        assert e._candidate is None
        await frame(e, since + 2.1, {"end": since + 2.1}, .3)
        await pump(e, lambda: e._guard_input.decided and not e._guard_input.admitted)
        await frame(e, since + 4.0)
        assert e._candidate is None and not observed(seen, "input_cut_recovery")
        await frame(e, since + 4.1 + EPS)
        assert e._candidate.stage == "recover"
        assert observed(seen, "input_cut_recovery")[0]["silence_s"] == 2.0
    finally:
        await cleanup(e)


async def test_onset_on_the_expiry_frame_yields_one_outcome():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        m.labels["input_route"] = "yield_ready"
        await new_input(e, since + 2 + EPS)
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
        assert e._candidate.stage == "response"
        await frame(e, since + 2.7)
        await pump(e, lambda: e._speech is not None and e._speech_meta.kind == "response" and m.responses == 2)
        for t in (since + 5, since + 9):
            await frame(e, t)
        assert not observed(seen, "input_cut_recovery") and m.responses == 2
    finally:
        await cleanup(e)


async def test_late_expiry_after_session_reset_does_not_recover():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        e._reset_session()
        for t in (since + 2, since + 5):
            await frame(e, t)
        assert e._candidate is None and not observed(seen, "input_cut_recovery")
        assert e._guard_recover_since is None and e._guard_recover_streak == 0
    finally:
        await cleanup(e)


async def test_interrupted_recovery_does_not_loop_until_a_real_answer_plays():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        await frame(e, since + 2 + EPS)
        await started(e, "recover")
        # Cutting the recovery prompt with another unusable input does not re-ask.
        since = await cut_then_keep(e, m, t=since + 2.1)
        for t in (since + 2, since + 6):
            await frame(e, t)
        assert e._candidate is None and len(observed(seen, "input_cut_recovery")) == 1
        assert [x["reason"] for x in observed(seen, "input_cut_recovery_skipped")] == ["streak_limit"]
        # A played real answer resets the streak; the next cut may ask again.
        m.labels["input_route"] = "yield_ready"
        await new_input(e, since + 7)
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
        await frame(e, since + 7.7)
        await started(e, "response")
        assert e._guard_recover_streak == 0 and m.responses == 2
        since = await cut_then_keep(e, m, t=since + 8)
        await frame(e, since + 2 + EPS)
        assert e._candidate.stage == "recover" and len(observed(seen, "input_cut_recovery")) == 2
    finally:
        await cleanup(e)


def test_recovery_settings_are_bounded():
    for config in ({"input_wait_uncertain_recover": True},
                   {"input_wait_release_s": 2.0, "input_recover_max": 4},
                   {"input_wait_release_s": 2.0, "input_recover_max": "1"}):
        e, _, _ = guarded()
        e.engine_cfg.update(config)
        with pytest.raises(ValueError):
            e._init_guarded()


def recoveries(seen):
    return observed(seen, "input_cut_recovery")


def skipped_reasons(seen):
    return [x["reason"] for x in observed(seen, "input_cut_recovery_skipped")]


def recovering_with_context():
    """Continuation context on; the route prompt then carries the context protocol."""
    e, m, sock, seen = recovering(input_continuation_context=True)
    async def text(messages):
        if messages[0]["content"].startswith("input_route"):
            messages = [{**messages[0], "content": "input_route"}, *messages[1:]]
        async for part in m.text(messages):
            yield part
    e.text_stream_fn = text
    return e, m, sock, seen


async def test_heard_content_after_arming_cancels_the_pending_recovery():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        # The user then backchannels or talks to someone else: kept, rejected, heard.
        m.labels["input_route"] = "keep"
        await new_input(e, since + 1)
        await pump(e, lambda: e._guard_input.decided and not e._guard_input.admitted)
        for t in (since + 2 + EPS, since + 3.1, since + 8):
            await frame(e, t)
        assert e._candidate is None and not recoveries(seen)
        assert skipped_reasons(seen) == ["heard_content"]
    finally:
        await cleanup(e)


async def test_admitted_then_heard_input_replaces_the_pending_recovery():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        # Admitted while armed (user_speaking), then closed with heard content: the
        # earlier arming must not survive the wait-reason transitions.
        since = await cut_then_keep(e, m, t=since + .5, keep="keep")
        for t in (since + 2 + EPS, since + 8):
            await frame(e, t)
        assert e._candidate is None and not recoveries(seen)
        assert skipped_reasons(seen) == ["heard_content"]
    finally:
        await cleanup(e)


async def test_any_other_answer_disarms_recovery():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        # Whatever path starts an answer owns the floor; no repeat request after it.
        e._begin_candidate(np.zeros(1600, dtype=np.float32), stage="response")
        await frame(e, since + .1)
        assert e._guard_recover_since is None
        e._invalidate_candidate("test")
        for t in (since + 2 + EPS, since + 6):
            await frame(e, t)
        assert not recoveries(seen)
    finally:
        await cleanup(e)


async def test_rejection_recounts_from_the_input_end_not_the_decision():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        m.labels["input_route"] = EMPTY_KEEP
        await frame(e, since + 1, {"start": since + 1}, .2)
        await frame(e, since + 1.2, {"end": since + 1.2}, .3)
        # The decision is only processed 0.4 s after the input ended.
        await frame(e, since + 1.6)
        await pump(e, lambda: e._guard_input.decided)
        await frame(e, since + 3.2 - .016)
        assert not recoveries(seen)
        await frame(e, since + 3.2 + EPS)
        assert e._candidate.stage == "recover" and recoveries(seen)[0]["silence_s"] == 2.0
    finally:
        await cleanup(e)


@pytest.mark.parametrize("audit,cause", [
    ({"acoustic_reject": True, "error": "X"}, "acoustic_reject"),
    ({"error": "TimeoutError"}, "error"),
    ({"route": {"fallback": True, "timed_out": True, "transcript": "测试"}}, "timeout"),
    ({"route": {"fallback": True, "timed_out": False}}, "fallback"),
    ({"route": {"transcript": "  "}}, "empty_transcript"),
    ({"route": {"transcript": "嗯"}}, None),
])
async def test_recovery_cause_classification(audit, cause):
    from guarded_turns import _unheard_cause
    assert _unheard_cause(audit) == cause
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        e._guard_set_wait("uncertain_after_yield")
        e._guard_arm_recovery(audit, since)
        await frame(e, since + 2 + EPS)
        if cause is None:
            assert not recoveries(seen) and skipped_reasons(seen) == ["heard_content"]
        else:
            assert [x["reason"] for x in recoveries(seen)] == [cause]
    finally:
        await cleanup(e)


async def test_stop_stays_sticky_when_the_user_keeps_talking():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        m.labels["input_route"] = "stop_only"
        await frame(e, 1., {"start": 1.})
        await frame(e, 2.6)
        await pump(e, lambda: e._guard_input.admitted)
        # A later unclosed check of the same input says yield_wait.
        m.labels["input_route"] = "yield_wait"
        await frame(e, 4.2)
        await pump(e, lambda: any(d["route"] == "yield_wait" and not d["closed"]
                                  for d in observed(seen, "input_decision")))
        m.labels["input_route"] = EMPTY_KEEP
        await frame(e, 4.3, {"end": 4.3})
        await pump(e, lambda: e._guard_input.decided)
        for t in (6.3 + EPS, 9):
            await frame(e, t)
        assert e._guard_input.stop_requested and not recoveries(seen)
        assert skipped_reasons(seen) == ["stop_requested"]
    finally:
        await cleanup(e)


async def test_recovery_prompt_never_enters_history_even_when_interrupted():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        await frame(e, since + 2 + EPS)
        await started(e, "recover")
        await pump(e, lambda: e._speech_audio_done)
        # Fully heard, then the user barges in before the END ACK.
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": e._speech.sid, "played_samples": e._speech.sent}))
        m.labels["input_route"] = "yield_ready"
        await new_input(e, since + 4)
        await pump(e, lambda: e._guard_input.decided)
        assert not any("没听清" in v for v in e._assistants_by_turn.values())
        assert not any("没听清" in (d or {}).get("text", "") for name, d in seen if name == "history_write")
    finally:
        await cleanup(e)


async def test_unheard_recovery_does_not_count_toward_the_limit():
    e, m, sock, seen = recovering_with_context()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        await frame(e, since + 2 + EPS)
        assert e._candidate.stage == "recover"
        # The user starts repeating before the prompt is audible (preplay).
        m.labels["input_route"] = "yield_ready"
        t = since + 2.05
        await frame(e, t, {"start": t})
        span = e._guard_input
        assert span.mode == "preplay"
        await frame(e, t + .25)
        await pump(e, lambda: span.admitted)
        assert e._candidate is None and e._guard_recover_streak == 0
        m.labels["input_route"] = EMPTY_KEEP
        await frame(e, t + .4, {"end": t + .4})
        await pump(e, lambda: span.decided)
        # The never-played prompt left no audio context in either route request.
        dispatches = [d for d in observed(seen, "input_dispatch") if d["input_id"] == span.sid]
        assert len(dispatches) == 2
        assert all(d["context_audio_samples"] == 0 and d["context_source"] is None for d in dispatches)
        await frame(e, t + .4 + 2 + EPS)
        assert e._candidate.stage == "recover" and len(recoveries(seen)) == 2
        assert [x["streak"] for x in recoveries(seen)] == [1, 1]
    finally:
        await cleanup(e)


async def test_input_during_recovery_playback_gets_no_replied_audio_context():
    e, m, _, seen = recovering_with_context()
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        await frame(e, since + 2 + EPS)
        await started(e, "recover")
        assert e._guard_recover_streak == 1
        m.labels["input_route"] = "keep"
        await new_input(e, since + 2.5)
        await pump(e, lambda: e._guard_input.decided)
        dispatch = [d for d in observed(seen, "input_dispatch") if d["input_id"] == e._guard_input.sid]
        assert dispatch and dispatch[0]["playing"]
        assert dispatch[0]["context_audio_samples"] == 0 and dispatch[0]["context_source"] is None
    finally:
        await cleanup(e)


async def test_session_reset_clears_streak_and_language():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        since = await cut_then_keep(e, m, admit='{"transcript":"what about tomorrow","label":"yield_ready"}')
        await frame(e, since + 2 + EPS)
        await started(e, "recover")
        assert e._guard_recover_streak == 1 and e._speech_candidate.fixed_text.startswith("Sorry")
        e._reset_session()
        assert e._guard_recover_streak == 0 and e._guard_last_transcript == ""
        assert e._guard_recover_language() == "zh"
    finally:
        await cleanup(e)


async def test_unadmitted_speech_does_not_pick_the_language():
    e, m, _, seen = recovering()
    try:
        await playing(e)
        # Speech to someone else (or an echo) is kept and never admitted.
        m.labels["input_route"] = '{"transcript":"hold on a second","label":"keep"}'
        before = e._guard_last_transcript
        await new_input(e, .9)
        await pump(e, lambda: e._guard_input.decided and not e._guard_input.admitted)
        assert e._guard_last_transcript == before != "hold on a second"
        assert e._guard_recover_language() == "zh"
    finally:
        await cleanup(e)


async def test_input_limit_rejection_disarms_recovery_without_continuation():
    e, m, sock, seen = recovering(max_input_seconds=3)
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        m.labels["input_route"] = "keep"
        t = since + .5
        await frame(e, t, {"start": t})
        k = 1
        while not e._guard_input.decided and k < 400:
            await frame(e, t + .016 * k)
            k += 1
        assert [x["event"] for x in sock.events].count("input_notice") == 1
        for dt in (2 + EPS, 5):
            await frame(e, t + .016 * k + dt)
        assert not recoveries(seen) and skipped_reasons(seen) == ["input_limit"]
    finally:
        await cleanup(e)



def playing_end_window(e, ms):
    e.engine_cfg["input_vad_end_ms_playing"] = ms
    e._init_guarded()


def observe_rows(e):
    rows, original = [], e._observe
    def observe(event, data=None):
        rows.append((event, round(e.t_audio, 3), dict(data or {})))
        return original(event, data)
    e._observe = observe
    return rows


def closed_dispatches(rows):
    return [(t, d["input_id"], d.get("vad_end_silence_ms"), d.get("vad_end_merges"))
            for event, t, d in rows if event == "input_dispatch" and d["closed"]]


def applied(rows):
    return [(t, d["input_id"], d["route"]) for event, t, d in rows if event == "input_decision"]


async def holed_input(e, hole, settle=None):
    """Speech 1.0-1.2 s, a browser-AEC hole of `hole` s, then speech until 1.5+hole s.
    Silero ends 100 ms into each silence and restarts when speech resumes; `settle`
    lets the fragment decision return inside the hole (the harmful race)."""
    await frame(e, 1., {"start": 1.}, .2)
    await frame(e, 1.2, value=.2)
    await frame(e, 1.3, {"end": 1.3}, .001)
    if settle:
        await settle()
    await frame(e, 1.2 + hole - .016, value=.001)
    await frame(e, 1.2 + hole, {"start": 1.2 + hole}, .4)
    await frame(e, 1.5 + hole, value=.4)
    await frame(e, 1.6 + hole, {"end": 1.6 + hole}, .001)


# 0.5 s with a 500 ms window puts the resumed start on the frame that ends the
# window: a start always wins over the expiry on the same frame.
@pytest.mark.parametrize("ms,hole", [(400, .3), (500, .3), (500, .5)])
async def test_playback_aec_hole_inside_end_window_stays_one_input(ms, hole):
    e, m, sock = guarded()
    playing_end_window(e, ms)
    try:
        await playing(e)
        rows = observe_rows(e)
        sid = e._speech.sid
        m.labels["input_route"] = "yield_wait"
        await holed_input(e, hole, lambda: pump(e, lambda: e._guard_input.deferred))
        span, end, hold = e._guard_input, round(1.6 + hole, 3), (ms - 100) / 1000
        merged = [d for event, _, d in rows if event == "input_vad_end_merged"]
        assert [(d["gap_s"], d["discarded_routes"]) for d in merged] == [(round(hole - .1, 3), ["yield_wait"])]
        assert not applied(rows) and e._speech.sid == sid, "the fragment decision never acts"
        m.labels["input_route"] = "keep"
        await pump(e, lambda: span.deferred)
        await frame(e, end + hold - .016, value=.001)
        assert not span.decided and not applied(rows)
        await frame(e, end + hold, value=.001)
        assert applied(rows) == [(round(end + hold, 3), span.sid, "keep")] and span.decided
        assert closed_dispatches(rows) == [(1.3, span.sid, ms, 0), (end, span.sid, ms, 1)]
        assert span.end == 1.6 + hole and span.revision == 3
        assert [x["event"] for x in sock.events if x["event"] in ("vad_start", "vad_done")
                and x["data"].get("input_id") == span.sid] == [
            "vad_start", "vad_done", "vad_start", "vad_done"], "raw VAD events stay replayable"
        assert e._speech.sid == sid and m.responses == 1
    finally:
        await cleanup(e)


@pytest.mark.parametrize("ms", [None, 400])
async def test_fragment_decision_inside_hole_acts_only_without_end_window(ms):
    e, m, sock = guarded()
    if ms:
        playing_end_window(e, ms)
    try:
        await playing(e)
        rows = observe_rows(e)
        m.labels["input_route"] = "yield_wait"
        first = e._guard_input
        await holed_input(e, .3, lambda: pump(e, lambda: e._guard_input is not first and (
            e._guard_input.decided or e._guard_input.deferred)))
        if ms:
            assert e._speech is not None and not applied(rows)
            assert len({d[1] for d in closed_dispatches(rows)}) == 1
        else:
            # Baseline: the hole's fragment already stopped playback and the
            # resumed speech became a separate input.
            assert e._speech is None and applied(rows)[0][2] == "yield_wait"
            assert len({d[1] for d in closed_dispatches(rows)}) == 2
    finally:
        await cleanup(e)


def stable(value):
    """Drop wall-clock measurements and random diagnostic ids; keep everything else."""
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if not (k == "timestamp" or (
            type(v) is float and re.search(r"_ms$|time|elapsed|latency", k)))}
    if isinstance(value, (list, tuple)):
        return type(value)(stable(v) for v in value)
    if isinstance(value, str) and re.fullmatch(r"[a-z_]+-[0-9a-f]{16}", value):
        return value.split("-")[0] + "-*"
    return value


async def run_holed(ms, *, play, hole=.3):
    e, m, sock = guarded()
    if ms is not None:
        playing_end_window(e, ms)
    try:
        if play:
            await playing(e)
        rows = observe_rows(e)
        m.labels["input_route"] = "keep"
        await holed_input(e, hole)
        for t in [2., 2.5, 3.]:
            await frame(e, t, value=.001)
            await pump(e, lambda: e._guard_input.decided and e._inflight == 0)
        return stable(sock.events), [x for x in m.calls if x[0] == "input_route"], stable(rows)
    finally:
        await cleanup(e)


def without_window_fields(rows):
    return [(event, t, {k: v for k, v in d.items() if k not in ("vad_end_silence_ms", "vad_end_merges")})
            for event, t, d in rows]


@pytest.mark.parametrize("play", [False, True])
async def test_end_window_absent_or_zero_is_identical_and_keeps_split_baseline(play):
    absent, zero = await run_holed(None, play=play), await run_holed(0, play=play)
    assert absent == zero
    events, calls, rows = absent
    assert not any(event.startswith(("input_vad_end", "input_decision_deferred")) for event, _, _ in rows)
    assert not any("vad_end_silence_ms" in d or "deferred_from" in str(d) for _, _, d in rows)
    assert [t for t, *_ in closed_dispatches(rows)] == [1.3, 1.9]
    assert calls and len(calls) <= 2, "the reopened first check may be cancelled before its call"


async def test_end_window_never_changes_listening_input():
    off, on = await run_holed(None, play=False), await run_holed(400, play=False)
    assert off[0] == on[0] and off[1] == on[1]
    assert without_window_fields(off[2]) == without_window_fields(on[2])
    assert [(t, ms, merges) for t, _, ms, merges in closed_dispatches(on[2])] == [(1.3, 100, 0), (1.9, 100, 0)]


async def test_pause_longer_than_playing_window_still_splits():
    e, m, _ = guarded()
    playing_end_window(e, 400)
    try:
        await playing(e)
        rows = observe_rows(e)
        m.labels["input_route"] = "keep"
        await frame(e, 1., {"start": 1.}, .2)
        await frame(e, 1.2, {"end": 1.2}, .001)
        first = e._guard_input
        await pump(e, lambda: first.deferred)
        await frame(e, 1.48, value=.001)
        assert not first.decided
        # The held decision was accounted when it returned: applying it must not
        # release another in-flight task's slot (an extra one keeps it above zero).
        e._inflight += 1
        inflight = e._inflight
        await frame(e, 1.504, value=.001)
        assert first.decided and applied(rows) == [(1.504, first.sid, "keep")]
        assert e._inflight == inflight
        e._inflight -= 1
        await frame(e, 1.8, {"start": 1.8}, .4)
        assert e._guard_input is not first and e._guard_input.sid == first.sid + 1
        assert closed_dispatches(rows) == [(1.2, first.sid, 400, 0)]
        assert not any(event == "input_vad_end_merged" for event, _, _ in rows)
    finally:
        await cleanup(e)


@pytest.mark.parametrize("ms", [None, 400])
async def test_playback_stop_takes_effect_after_the_playing_window_without_loss(ms):
    e, m, sock = guarded()
    if ms:
        playing_end_window(e, ms)
    try:
        await playing(e)
        rows = observe_rows(e)
        m.labels["input_route"] = "stop_only"
        await frame(e, 1., {"start": 1.}, .2)
        await frame(e, 1.2, {"end": 1.2}, .001)
        span = e._guard_input
        await pump(e, lambda: span.decided or span.deferred)
        if ms:
            await frame(e, 1.4, value=.001)
            assert e._speech is not None and not span.decided
            await frame(e, 1.5, value=.001)
        assert closed_dispatches(rows)[0][0] == 1.2, "the route call is never postponed"
        assert applied(rows) == [(1.5 if ms else 1.2, span.sid, "stop_only")]
        assert e._speech is None and e.STATE == "LISTEN" and e._guard_wait_reason == "stop_only"
        await pump(e, lambda: any(x["event"] == "speech_cancelled" for x in sock.events))
        assert m.responses == 1
    finally:
        await cleanup(e)


async def test_decision_slower_than_the_window_is_not_delayed():
    e, m, _ = guarded()
    playing_end_window(e, 400)
    try:
        await playing(e)
        rows = observe_rows(e)
        m.blocked = "input_route"
        m.labels["input_route"] = "stop_only"
        await frame(e, 1., {"start": 1.}, .2)
        await frame(e, 1.2, {"end": 1.2}, .001)
        span = e._guard_input
        await frame(e, 1.6, value=.001)
        assert span.hold_until is None, "an empty window closes on time"
        m.gate.set()
        await pump(e, lambda: span.decided)
        assert applied(rows) == [(1.6, span.sid, "stop_only")]
        assert not any(event == "input_decision_deferred" for event, _, _ in rows)
    finally:
        await cleanup(e)


async def test_playing_window_is_fixed_at_the_silero_end():
    e, m, _ = guarded()
    playing_end_window(e, 400)
    try:
        await playing(e)
        rows = observe_rows(e)
        m.labels["input_route"] = "keep"
        await frame(e, 1., {"start": 1.}, .2)
        await frame(e, 1.2, {"end": 1.2}, .001)
        span = e._guard_input
        await e._finish_turn(0, "played")
        assert e._speech is None
        await pump(e, lambda: span.deferred)
        await frame(e, 1.3, value=.001)
        assert not span.decided, "playback EOF does not shorten the window"
        await frame(e, 1.5, value=.001)
        assert applied(rows) == [(1.5, span.sid, "keep")]
    finally:
        await cleanup(e)


async def test_preplay_input_keeps_listening_end_window():
    e, m, _ = guarded()
    playing_end_window(e, 400)
    rows = observe_rows(e)
    try:
        await private(e)
        m.labels["input_route"] = "keep"
        await new_input(e, .2)
        span = e._guard_input
        assert span.mode == "preplay" and span.hold_until is None
        await pump(e, lambda: span.decided)
        assert [x for x in closed_dispatches(rows) if x[1] == span.sid] == [(.232, span.sid, 100, 0)]
        assert [x for x in applied(rows) if x[1] == span.sid] == [(.232, span.sid, "keep")]
    finally:
        await cleanup(e)


@pytest.mark.parametrize("ms", [None, 400])
async def test_offline_tail_closes_the_playing_window(ms):
    e, m, _ = guarded()
    if ms:
        playing_end_window(e, ms)
    try:
        await playing(e)
        rows = observe_rows(e)
        m.blocked = "input_route"
        m.labels["input_route"] = "stop_only"
        await frame(e, 1., {"start": 1.}, .2)
        await frame(e, 1.2, {"end": 1.2}, .001)
        span = e._guard_input
        m.gate.set()
        # No further frames: run_offline must still let the held stop act.
        await e.run_offline([], quiesce_timeout=3)
        assert span.decided and not span.deferred and e._speech is None
        assert applied(rows) == [(1.5 if ms else 1.2, span.sid, "stop_only")]
    finally:
        await cleanup(e)


@pytest.mark.parametrize("ms", [None, 400])
async def test_input_limit_ignores_silence_held_after_a_playing_end(ms):
    e, m, sock = guarded()
    if ms:
        playing_end_window(e, ms)
    try:
        await playing(e)
        rows = observe_rows(e)
        m.labels["input_route"] = "stop_only"
        await frame(e, 1., {"start": 1.}, .2)
        await frame(e, 1.2, {"end": 1.2}, .001)
        span = e._guard_input
        await pump(e, lambda: span.decided or span.deferred)
        e._guard_max_samples = sum(map(len, span.frames)) + 4 * 256
        for k in range(1, 8):
            await frame(e, round(1.2 + .016 * k, 3), value=.001)
        await frame(e, 1.5, value=.001)
        assert not any(event == "input_rejected" for event, _, _ in rows)
        assert not any(x["event"] == "input_notice" for x in sock.events)
        assert applied(rows) == [(1.5 if ms else 1.2, span.sid, "stop_only")] and e._speech is None
    finally:
        await cleanup(e)


@pytest.mark.parametrize("started_in_window", [False, True])
async def test_published_but_unstarted_speech_keeps_listening_end_window(started_in_window):
    e, m, _ = guarded()
    playing_end_window(e, 400)
    try:
        await private(e)
        await frame(e, .704)
        await pump(e, lambda: e._speech is not None and e.STATE == "SPEAK")
        rows = observe_rows(e)
        m.blocked = "input_route"
        m.labels["input_route"] = "keep"
        await new_input(e)
        span = e._guard_input
        assert not e._speech_started and span.hold_until is None
        if started_in_window:
            # Audible playback beginning after the end never lengthens it.
            await e._process_event(ControlMsg("playback_progress", {
                "utterance_id": e._speech.sid, "played_samples": 0, "started": True}))
            assert e._speech_started
        m.gate.set()
        await pump(e, lambda: span.decided)
        assert closed_dispatches(rows) == [(1.032, span.sid, 100, 0)]
        assert applied(rows) == [(1.032, span.sid, "keep")]
        assert not any(event == "input_decision_deferred" for event, _, _ in rows)
    finally:
        await cleanup(e)


async def test_merged_input_ending_after_playback_eof_acts_at_once():
    e, m, _ = guarded()
    playing_end_window(e, 400)
    try:
        await playing(e)
        rows = observe_rows(e)
        m.labels["input_route"] = "yield_wait"
        await frame(e, 1., {"start": 1.}, .2)
        await frame(e, 1.2, {"end": 1.2}, .001)
        span = e._guard_input
        await pump(e, lambda: span.deferred)
        await e._finish_turn(0, "played")
        assert e._speech is None
        await frame(e, 1.4, {"start": 1.4}, .4)
        assert e._guard_input is span and span.hold_until is None and not span.deferred
        m.labels["input_route"] = "keep"
        await frame(e, 1.6, value=.4)
        await frame(e, 1.7, {"end": 1.7}, .001)
        assert span.hold_until is None and span.end_silence_ms == 100
        await pump(e, lambda: span.decided)
        assert applied(rows) == [(1.7, span.sid, "keep")]
        assert closed_dispatches(rows) == [(1.2, span.sid, 400, 0), (1.7, span.sid, 100, 1)]
    finally:
        await cleanup(e)


# Interactions of the three opt-in switches: cut recovery (input_wait_uncertain_recover),
# the playing end window (input_vad_end_ms_playing) and the heard partial in reply review
# (input_reply_partial_heard_s). Invariants locked here:
# - recovery is armed only after playback was cut, so no playing end window can be open
#   on the cut input or on any input before the recovery fires;
# - the recovery prompt is itself playback: input over it gets the window, and a decision
#   held there takes effect only when the window closes;
# - the fixed recovery sentences never yield a heard partial, so reply review sees the
#   same recovery text with the partial switch on or off.

INTERACTION_EVENTS = ("input_decision", "input_decision_deferred", "input_vad_end_merged",
                      "input_admitted", "input_rejected", "input_cut_recovery",
                      "input_cut_recovery_skipped", "candidate_created", "history_write",
                      "history_remove")


def interaction_order(rows):
    details = {
        "input_decision": lambda d: (d["closed"], d["route"]),
        "input_decision_deferred": lambda d: (d["closed"], d["route"], d["until"]),
        "input_vad_end_merged": lambda d: tuple(d["discarded_routes"]),
        "input_admitted": lambda d: d["route"],
        "input_rejected": lambda d: d["reason"],
        "input_cut_recovery": lambda d: (d["reason"], d["streak"], d["silence_s"]),
        "input_cut_recovery_skipped": lambda d: d["reason"],
        "candidate_created": lambda d: "recover" if d["audio_samples"] == 0 else "answer",
    }
    return [(event, t, details.get(event, lambda d: (d.get("role"), d.get("turn")))(d))
            for event, t, d in rows if event in INTERACTION_EVENTS]


@pytest.mark.parametrize("ms", [None, 400])
async def test_recovery_clock_is_unchanged_by_the_playing_end_window(ms):
    e, m, _, seen = recovering()
    if ms:
        playing_end_window(e, ms)
    try:
        await playing(e)
        rows = observe_rows(e)
        since = await cut_then_keep(e, m)
        cut = e._guard_input
        # The admission already cut playback: the cut input's own end never waits.
        assert cut.hold_until is None and not cut.deferred and e._guard_recover_since == since
        # An unusable input still open when the recovery would expire: it is a listening
        # input (nothing plays while armed), so no window holds it; the expiry waits for
        # its decision, and the rejection recounts from its Silero end, not the decision.
        m.labels["input_route"] = EMPTY_KEEP
        t = since + 1.9
        await frame(e, t, {"start": t}, .2)
        await frame(e, since + 2 + EPS, value=.2)
        span = e._guard_input
        assert span.mode == "listening" and not span.decided and not recoveries(seen)
        await frame(e, t + .2, {"end": t + .2}, .3)
        assert span.hold_until is None
        await frame(e, t + .4, value=.001)
        await pump(e, lambda: span.decided)
        assert e._guard_recover_since == t + .2 and not recoveries(seen)
        await frame(e, t + .2 + 2 - .016)
        assert not recoveries(seen)
        await frame(e, t + .2 + 2 + EPS)
        assert e._candidate.stage == "recover" and recoveries(seen)[0]["silence_s"] == 2.0
        assert not any(event in ("input_decision_deferred", "input_vad_end_merged") for event, _, _ in rows)
        assert [x[2] for x in closed_dispatches(rows)] == ([100, 100] if ms else [None, None])
    finally:
        await cleanup(e)


async def test_unusable_input_over_the_recovery_prompt_waits_for_the_window_then_stays_silent():
    e, m, _, seen = recovering(input_vad_end_ms_playing=400)
    try:
        await playing(e)
        since = await cut_then_keep(e, m)
        await frame(e, since + 2 + EPS)
        await started(e, "recover")
        sid = e._speech.sid
        assert e._guard_recover_streak == 1 and e._guard_recover_since is None
        rows = observe_rows(e)
        m.labels["input_route"] = EMPTY_KEEP
        t = since + 2.5
        await frame(e, t, {"start": t}, .2)
        await frame(e, t + .2, {"end": t + .2}, .001)
        span = e._guard_input
        assert span.mode == "playing" and span.hold_until == pytest.approx(t + .5)
        await pump(e, lambda: span.deferred)
        # The prompt ends inside the window; playback EOF does not shorten it.
        await pump(e, lambda: e._speech_audio_done)
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": sid, "played_samples": e._speech.sent, "ended": True}))
        assert e._speech is None
        await frame(e, t + .5 - .016, value=.001)
        assert not span.decided and not applied(rows)
        await frame(e, t + .5, value=.001)
        assert span.decided and not span.admitted
        assert applied(rows) == [(round(t + .5, 3), span.sid, "keep")]
        for dt in (2 + EPS, 6):
            await frame(e, t + .2 + dt)
        # Not admitted: nothing re-arms, and the streak already counted the played prompt.
        assert e._candidate is None and len(recoveries(seen)) == 1 and not skipped_reasons(seen)
        assert e._guard_recover_since is None and e._guard_recover_streak == 1
    finally:
        await cleanup(e)


async def test_all_switches_reply_over_the_recovery_prompt_in_order():
    e, m, sock, seen = recovering(input_vad_end_ms_playing=400, input_reply_partial_heard_s=.5)
    e.prompts["input_reply"] = "input_reply"
    m.labels["input_reply"] = "yield_ready"
    m.entered["input_reply"] = asyncio.Event()
    try:
        await playing(e)
        rows = observe_rows(e)
        since = await cut_then_keep(e, m)
        await frame(e, since + 2 + EPS)
        await started(e, "recover")
        sid = e._speech.sid
        await pump(e, lambda: e._guard_outputs.get(sid, {}).get("sentences"))
        record = e._guard_outputs[sid]
        (end, text), = record["sentences"]
        assert text == "抱歉，我刚才没听清，请再说一遍。"
        await e._process_event(ControlMsg("playback_progress", {"utterance_id": sid, "played_samples": end}))
        # The user answers over the fully heard prompt; a browser-AEC hole splits the speech.
        m.labels["input_route"] = "yield_wait"
        t = since + 3.
        await frame(e, t, {"start": t}, .2)
        span = e._guard_input
        assert span.mode == "playing" and span.reply_context == text
        assert span.reply_context_kind == "completed_sentences"
        assert span.reply_review_context == "" and span.reply_partial_chars == 0
        await frame(e, t + .2, {"end": t + .2}, .001)
        await pump(e, lambda: span.deferred)
        m.labels["input_route"] = "keep"
        await frame(e, t + .4, {"start": t + .4}, .4)
        await frame(e, t + .6, value=.4)
        await frame(e, t + .7, {"end": t + .7}, .001)
        await pump(e, lambda: span.deferred)
        await frame(e, t + 1. - .016, value=.001)
        assert not span.decided and e._speech is not None and e._speech.sid == sid
        await frame(e, t + 1., value=.001)
        assert span.decided and span.admitted and span.route == "yield_ready" and e._speech is None
        review = [json.loads(msgs[1]["content"]) for kind, msgs in m.calls if kind == "input_reply"]
        assert review == [{"played_assistant": text, "user_transcript": "测试语音"}]
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
        await frame(e, t + 1.7)
        await started(e, "response")
        assert e._guard_recover_streak == 0 and m.responses == 2
        cut_end, recovered, hold_end = round(since, 3), round(since + 2 + EPS, 3), round(t + 1., 3)
        assert interaction_order(rows) == [
            ("input_decision", 2.6, (False, "yield_ready")),
            # The first answer is cut before any sentence was heard (baseline: admission and
            # _cancel_speech each record the cut).
            ("history_remove", 2.6, ("assistant", 0)),
            ("history_remove", 2.6, ("assistant", 0)),
            ("input_admitted", 2.6, "yield_ready"),
            ("input_decision", cut_end, (True, "keep")),
            ("input_rejected", cut_end, "insufficient_evidence"),
            ("input_cut_recovery", recovered, ("empty_transcript", 1, 2.0)),
            ("candidate_created", recovered, "recover"),
            ("input_decision_deferred", round(t + .2, 3), (True, "yield_wait", round(t + .5, 3))),
            ("input_vad_end_merged", round(t + .4, 3), ("yield_wait",)),
            ("input_decision_deferred", round(t + .7, 3), (True, "yield_ready", hold_end)),
            ("input_decision", hold_end, (True, "yield_ready")),
            ("input_admitted", hold_end, "yield_ready"),
            ("candidate_created", hold_end, "answer"),
            # Only the real answer writes history (its own turn); the cut prompt wrote none.
            ("history_write", round(t + 1.7, 3), ("assistant", 2)),
        ]
        # The prompt never reaches history, before or after the reply.
        assert not any("没听清" in v for v in e._assistants_by_turn.values())
        assert not any("没听清" in d.get("text", "") for event, _, d in rows if event == "history_write")
    finally:
        await cleanup(e)


async def recovery_prompt_onsets(lang, first_clause, partial):
    """Reply-review context of an input starting at each 50 ms of the recovery prompt."""
    from stream_transport import PCMChunk
    from guarded_turns import RECOVER_TEXT
    e, m, _, seen = recovering(input_reply_partial_heard_s=partial, stream_first_clause_chars=first_clause)
    async def long_tts(sentence):
        # 1.2 s units, so playing positions past the partial threshold are reachable.
        m.tts_calls.append(sentence)
        for _ in range(30):
            yield PCMChunk(b"\0\0" * 960, 24000)
    e.tts_stream_fn = long_tts
    try:
        await playing(e)
        admit = '{"transcript":"what about tomorrow","label":"yield_ready"}' if lang == "en" else "yield_ready"
        since = await cut_then_keep(e, m, admit=admit)
        await frame(e, since + 2 + EPS)
        await started(e, "recover")
        sid, rate = e._speech.sid, e._speech.rate
        while not e._speech_audio_done:
            # Acknowledge playback so the 600 ms send credit lets the whole prompt out.
            sent = e._speech.sent
            await e._process_event(ControlMsg("playback_progress", {"utterance_id": sid, "played_samples": sent}))
            await pump(e, lambda: e._speech_audio_done or e._speech.sent > sent)
        record = e._guard_outputs[sid]
        units = [text for _, text in record["reference"].sentences]
        assert "".join(units) == RECOVER_TEXT[lang] and len(record["sentences"]) == len(units)
        assert len(units) == (1 if lang == "zh" and not first_clause else 2)
        onsets = []
        for played in range(0, e._speech.sent + 1, rate // 20):
            e._speech.played = played
            await frame(e, since + 3, {"start": since + 3}, .2)
            span = e._guard_input
            assert span.mode == "playing" and span.task is None
            onsets.append((played, span.reply_context_kind, span.reply_context,
                           span.reply_review_context, span.reply_partial_chars))
            # Drop this probe onset; nothing was dispatched for it.
            e._guard_input = None
        return onsets
    finally:
        await cleanup(e)


@pytest.mark.parametrize("first_clause", [0, 8])
@pytest.mark.parametrize("lang", ["zh", "en"])
async def test_recovery_prompt_never_adds_a_heard_partial_to_reply_review(lang, first_clause):
    off = await recovery_prompt_onsets(lang, first_clause, 0)
    on = await recovery_prompt_onsets(lang, first_clause, .25)
    assert on == off
    assert all(review == "" and chars == 0 for *_, review, chars in on)
    assert {kind for _, kind, *_ in on} >= {None, "completed_sentences" if lang == "en" or first_clause == 0
                                            else "sentence_window"}, "the prompt does become review context"
