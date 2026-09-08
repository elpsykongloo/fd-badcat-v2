"""Incremental reference is not generated history or proof of heard words."""
import json

from test_actor_candidate import frame, pump, ended
from speech_reference import SpeechReference
from speech_stream import SpeechEvent
from guarded_turns import InputSpan
from engine import ControlMsg
from test_guarded_turns import guarded, cleanup, playing, new_input


def test_bounded_generation_and_playback_window_exclude_future_and_old_opening():
    ref = SpeechReference()
    ref.observe("text_delta", {"text": "开场" * 600})
    ref.observe("text_delta", {"text": "尚未播放的未来内容"})
    tail, source = ref.snapshot(started=False, played=0, sent=0)
    assert len(tail) == 512 and tail.endswith("尚未播放的未来内容")
    assert source == "generated_unplayed"
    for i in range(8):
        ref.observe("sentence", {"start_sample": i * 1000, "text": f"第{i}句。"})
    text, source = ref.snapshot(started=True, played=6500, sent=7000)
    assert text == "第4句。第5句。第6句。" and source == "playback_sentence_window"
    assert "未来" not in text and "第0句" not in text and "第7句" not in text
    assert ref.snapshot(started=True, played=7000, sent=7000)[0] == text
    assert ref.snapshot(started=True, played=7000, sent=7100)[0] == "第5句。第6句。第7句。"
    for i in range(100):
        ref.observe("sentence", {"start_sample": (i + 8) * 1000, "text": "长句" * 600})
    assert len(ref.sentences) == 64 and all(len(t) <= 512 for _, t in ref.sentences)
    assert len(ref.snapshot(started=True, played=107001, sent=107002)[0]) <= 512


def test_no_audio_boundary_does_not_pass_future_generation_as_playback():
    ref = SpeechReference()
    ref.observe("text_done", {"text": "全部生成完，但没有音频。"})
    assert ref.snapshot(started=True, played=0, sent=0) == ("", "unavailable")


async def test_real_pipeline_reference_available_while_long_text_done_is_backpressured():
    e, m, _ = guarded()
    original = m.text
    long_text = "正在播放的开场句。" + "后面尚未播放的内容。" * 70
    async def text(messages):
        if messages[0]["content"] == "response":
            yield long_text
        else:
            async for part in original(messages):
                yield part
    e.text_stream_fn = text
    try:
        await ended(e)
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None
                   and e._candidate.pipeline.sent > 0)
        private_sid = e._candidate.pipeline.sid
        assert e._speech is None and private_sid not in e._guard_outputs
        assert e._candidate.meta.text == "" and not e._assistants_by_turn
        await frame(e, .704)
        await pump(e, lambda: e._speech is not None and e.STATE == "SPEAK")
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": e._speech.sid, "played_samples": 0, "started": True}))
        assert e._speech_meta.text == ""  # reproduce the incident's original gap
        m.labels["input_route"] = "keep"
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        msg = [request for kind, request in m.calls if kind == "input_route"][-1]
        context = msg[1]["content"][0]["text"].split("\n")[0].removeprefix("情境资料：")
        assert json.loads(context) == {"assistant_playing": True}
        assert e._guard_input.reference_text == "正在播放的开场句。"
        assert e._guard_input.reference_kind == "playback_sentence_window"
        assert e._speech_meta.text == "" and not e._assistants_by_turn
    finally:
        await cleanup(e)


async def test_input_snapshot_is_frozen_and_cancel_reset_cannot_leak_old_reference():
    e, m, _ = guarded()
    try:
        await playing(e)
        sid = e._speech.sid
        span = InputSpan(99, 0, "playing", [])
        e._guard_capture_reference(span)
        original = span.reference_text
        assert original
        await e._on_speech_event(SpeechEvent(sid, "text_delta", {"text": "不属于输入开始时的未来文本。"}))
        assert span.reference_text == original
        e._guard_mark_cancelled()
        e._cancel_speech("accepted_interrupt")
        empty = InputSpan(100, 0, "listening", [])
        e._guard_capture_reference(empty)
        assert empty.reference_text == ""
        await e._on_speech_event(SpeechEvent(sid, "text_delta", {"text": "迟到的旧文本。"}))
        assert "迟到" not in e._guard_outputs[sid]["reference"].generated_tail
        e._reset_session()
        await e._on_speech_event(SpeechEvent(sid, "sentence", {"text": "重置后迟到", "start_sample": 0}))
        assert not e._guard_outputs
    finally:
        await cleanup(e)


async def test_preplay_snapshot_refreshes_same_stream_without_partial_history_commit():
    e, m, _ = guarded()
    try:
        await ended(e)
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None
                   and e._candidate.pipeline.sent > 0)
        await frame(e, .704)
        await pump(e, lambda: e._speech is not None and e.STATE == "SPEAK")
        span = InputSpan(99, 0, "preplay", [])
        e._guard_capture_reference(span)
        assert span.reference_kind == "generated_unplayed"
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": e._speech.sid, "played_samples": 0, "started": True}))
        e._guard_capture_reference(span)
        assert span.reference_kind == "playback_sentence_window"
        frozen = span.reference_text
        span.reference_sid += 1  # another answer cannot overwrite this input's reference
        e._guard_capture_reference(span)
        assert span.reference_text == frozen
    finally:
        await cleanup(e)


async def test_eof_retains_existing_input_snapshot_but_new_input_gets_no_stale_meta():
    e, m, _ = guarded()
    try:
        await playing(e)
        span = InputSpan(99, 0, "playing", [])
        e._guard_capture_reference(span)
        frozen = span.reference_text
        assert frozen
        await e._guard_finish_turn(e.TURN_IDX, "played")
        assert e._speech is None
        new = InputSpan(100, 0, "listening", [])
        e._guard_capture_reference(new)
        assert new.reference_text == "" and span.reference_text == frozen
    finally:
        await cleanup(e)
