"""Pending accepted audio reaches readiness; noise/stop/reset stay fenced."""
import asyncio
import base64
import io
import json

import numpy as np
import pytest
import soundfile as sf

from test_guarded_turns import guarded, new_input, cleanup
from test_actor_candidate import frame, pump
from engine import ControlMsg


def with_context(*, enabled=True, gap=20):
    e, m, sock = guarded()
    e.engine_cfg.update(input_continuation_context=enabled, input_context_max_gap_s=gap)
    e._init_guarded()
    labels = ["yield_wait"]
    seen = []
    async def text(messages):
        if messages[0]["content"].startswith("input_route"):
            content = messages[-1]["content"]
            parts = []
            for block in content:
                if block["type"] == "audio_url":
                    wave = base64.b64decode(block["audio_url"]["url"].split(",", 1)[1])
                    audio, rate = sf.read(io.BytesIO(wave), dtype="float32")
                    parts.append(audio)
            audio = np.concatenate(parts)
            seen.append((messages, audio))
            yield json.dumps({"transcript": "自造测试", "label": labels[-1]}, ensure_ascii=False)
        else:
            async for part in m.text(messages):
                yield part
    e.text_stream_fn = text
    return e, m, sock, labels, seen


async def wait_prefix(e):
    await new_input(e)
    await pump(e, lambda:e._guard_input.decided)
    assert e._guard_wait_reason == "awaiting_user"
    return np.concatenate(e._guard_wait_audio).copy()


@pytest.mark.parametrize("enabled", [False, True])
async def test_pending_audio_routes_and_answers_without_duplication(enabled):
    e, m, _, labels, seen = with_context(enabled=enabled)
    try:
        old = await wait_prefix(e)
        labels.append("yield_ready")
        await new_input(e, 3)
        await pump(e, lambda:e._candidate is not None)
        messages, routed = seen[-1]
        context = json.loads(messages[-1]["content"][0]["text"].split("情境资料：", 1)[1].split("\n", 1)[0])
        if enabled:
            assert context["pending_user_audio"]["current_start_ms"] == len(old) / 16
            np.testing.assert_allclose(routed[:len(old)], old, atol=1/32768)
            # The response sees exactly the context used for this decision, not
            # pending audio concatenated for a second time by the new routing.
            np.testing.assert_allclose(routed, e._candidate.audio, atol=1/32768)
        else:
            assert "pending_user_audio" not in context
            assert len(routed) < len(e._candidate.audio)
        assert not e._guard_wait_audio and e._guard_wait_end is None
        assert not any("assistant" == msg["role"] for msg in messages)
    finally:
        await cleanup(e)


async def test_multiple_waits_accumulate_once_then_clear_on_ready():
    e, _, _, labels, seen = with_context()
    try:
        first = await wait_prefix(e)
        await new_input(e, 3)
        await pump(e, lambda:e._guard_input.decided)
        second = np.concatenate(e._guard_wait_audio).copy()
        assert len(second) == len(seen[-1][1]) + 1600
        np.testing.assert_array_equal(second[:len(first)], first)
        labels.append("yield_ready")
        await new_input(e, 5)
        await pump(e, lambda:e._candidate is not None)
        assert len(e._candidate.audio) == len(seen[-1][1])
        np.testing.assert_allclose(seen[-1][1][:len(second)], second, atol=1/32768)
    finally:
        await cleanup(e)


@pytest.mark.parametrize("label", ["yield_ready", "yield_wait", "keep", "stop_only"])
@pytest.mark.parametrize("previous_played", [False, True])
async def test_unplayed_candidate_input_reaches_route_before_admission(label, previous_played):
    e, _, _, labels, seen = with_context()
    try:
        labels.append("yield_ready")
        await new_input(e)
        await pump(e, lambda:e._candidate is not None)
        original = e._candidate
        old = original.audio.copy()
        # A completed prior turn may leave its ACK flag set while no active
        # speech exists; this must not hide the new unplayed candidate input.
        e._speech_started = previous_played
        labels.append(label)
        await new_input(e, 1.3)
        await pump(e, lambda:e._guard_input.decided)
        routed = seen[-1][1]
        np.testing.assert_allclose(routed[:len(old)], old, atol=1/32768)
        if label == "yield_ready":
            assert e._candidate is not original
            np.testing.assert_allclose(routed, e._candidate.audio, atol=1/32768)
        elif label == "yield_wait":
            np.testing.assert_allclose(routed, np.concatenate(e._guard_wait_audio)[:-1600], atol=1/32768)
            assert e._candidate is None
        elif label == "keep":
            assert e._candidate is original and not e._guard_wait_audio
        else:
            assert e._candidate is None and not e._guard_wait_audio
    finally:
        await cleanup(e)


@pytest.mark.parametrize("label", ["stop_only", "yield_wait", "keep"])
async def test_latest_route_can_stop_wait_or_reject_despite_context(label):
    e, _, _, labels, seen = with_context()
    try:
        await wait_prefix(e)
        labels.append(label)
        await new_input(e, 3)
        await pump(e, lambda:e._guard_input.decided)
        assert e._candidate is None
        assert (not e._guard_wait_audio) if label == "stop_only" else bool(e._guard_wait_audio)
        if label == "stop_only":
            assert e._guard_wait_end is None
        assert "不能只凭先前已保存的语音再次授权回答" in seen[-1][0][0]["content"]
    finally:
        await cleanup(e)


async def test_old_speech_cannot_authorize_new_silence():
    e, _, _, labels, seen = with_context()
    try:
        await wait_prefix(e)
        old_count = len(seen)
        labels.append("yield_ready")
        e._guard_preroll.clear()
        await frame(e, 3, {"start": 3}, value=0)
        await frame(e, 3.032, {"end": 3.032}, value=0)
        await pump(e, lambda:e._guard_input.decided)
        assert len(seen) == old_count and e._candidate is None
    finally:
        await cleanup(e)


async def test_context_expiry_uses_audio_clock_not_model_wall_clock():
    e, _, _, labels, seen = with_context(gap=2)
    try:
        await wait_prefix(e)
        labels.append("keep")
        await new_input(e, 4)
        await pump(e, lambda:e._guard_input.decided)
        assert not e._guard_wait_audio and e._guard_wait_end is None
        assert "pending_user_audio" not in seen[-1][0][-1]["content"][0]["text"]
    finally:
        await cleanup(e)


async def test_context_and_current_input_share_one_audio_budget():
    e, _, _, _, seen = with_context()
    try:
        await wait_prefix(e)
        e._guard_max_samples = sum(map(len, e._guard_wait_audio)) + 255
        old_count = len(seen)
        await frame(e, 3, {"start": 3})
        assert e._guard_input.decided and e._candidate is None
        assert not e._guard_wait_audio and len(seen) == old_count
        assert any(r["event"] == "input_notice" for r in e.trace)
    finally:
        await cleanup(e)


async def test_reset_clears_context_and_late_decision_cannot_restore_it():
    e, _, _, _, _ = with_context()
    try:
        await wait_prefix(e)
        e._reset_session()
        assert not e._guard_wait_audio and e._guard_wait_end is None
        assert e._guard_input is None
    finally:
        await cleanup(e)


@pytest.mark.parametrize("label", ["yield_ready", "yield_wait", "stop_only", "keep"])
async def test_playing_continuation_has_onset_frozen_user_audio_not_assistant_text(label):
    e, _, _, labels, seen = with_context()
    try:
        labels.append("yield_ready")
        await new_input(e)
        await pump(e, lambda:e._candidate is not None and e._candidate.pipeline is not None
                   and e._candidate.pipeline.sent > 0)
        old = e._candidate
        await frame(e, 1.8)
        await pump(e, lambda:e._speech is not None)
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": e._speech.sid, "played_samples": 0, "started": True}))
        assert e._speech_started and e._candidate is None
        labels.append(label)
        await new_input(e, 3)
        await pump(e, lambda:e._guard_input.decided)
        messages, routed = seen[-1]
        metadata = json.loads(messages[-1]["content"][0]["text"].split("情境资料：")[1].split("\n")[0])
        assert metadata["pending_user_audio"]["source"] == "recent_replied_user_audio"
        assert metadata["pending_user_audio"]["route_only"] is True
        assert metadata["pending_user_audio"]["transcript_scope"] == "new_audio_only"
        assert metadata["pending_user_audio"]["audio_layout"] == "two_blocks"
        assert sum(x["type"] == "audio_url" for x in messages[-1]["content"]) == 2
        assert "transcript只逐字转写起点之后的新增语音" in messages[0]["content"]
        np.testing.assert_allclose(routed[:len(old.audio)], old.audio, atol=1/32768)
        assert "后半段只是在附和" in messages[0]["content"]
        assert len(messages) == 2  # No ASR/assistant history in audio router.
        if label == "keep":
            assert e._speech is old.pipeline and e._candidate is None
        elif label == "yield_ready":
            assert e._candidate is not None
            assert len(e._candidate.audio) == len(routed) - len(old.audio) - 1600
        else:
            assert e._speech is None and e._candidate is None
    finally:
        await cleanup(e)


@pytest.mark.parametrize("boundary", ["disabled", "expired", "combined_limit"])
async def test_optional_playing_context_does_not_steal_new_audio_budget(boundary):
    e, _, sock, labels, seen = with_context(enabled=boundary != "disabled")
    try:
        labels.append("yield_ready")
        await new_input(e)
        await pump(e, lambda:e._candidate is not None and e._candidate.pipeline is not None
                   and e._candidate.pipeline.sent > 0)
        await frame(e, 1.8)
        await pump(e, lambda:e._speech is not None)
        await e._process_event(ControlMsg("playback_progress", {
            "utterance_id": e._speech.sid, "played_samples": 0, "started": True}))
        assert e._speech_started
        if boundary == "combined_limit":
            e._guard_max_samples = 1500  # New frames fit, optional old + gap do not.
        labels.append("keep")
        await new_input(e, 25 if boundary == "expired" else 3)
        await pump(e, lambda:e._guard_input.decided)
        assert "pending_user_audio" not in seen[-1][0][-1]["content"][0]["text"]
        assert e._speech is not None
        assert not any(row["event"] == "input_notice" for row in sock.events)
    finally:
        await cleanup(e)
