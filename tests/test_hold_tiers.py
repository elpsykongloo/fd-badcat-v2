"""hold-tier-v1: pure tier rules (table-driven) and per-candidate hold in the guarded actor."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hold_tiers
from hold_tiers import classify, resolve, hold_config

TIERS = {"low": .4, "default": .576, "keep": .64}
CFG = {**TIERS, "rules": True, "shadow": False, "min_rtt_ms": 0, "evaluate": True, "apply": True}


def row(text, *, s=1.0, label="yield_ready", playing=False, first=False, rtt=None, min_rtt=0):
    return classify(text, duration_s=s, label=label, playing=playing, first_turn=first,
                    rtt_ms=rtt, min_rtt_ms=min_rtt)


@pytest.mark.parametrize("text,kwargs,expected", [
    # low: ready, < 1.5 s, (question or first turn), Chinese, no opener
    ("你是谁？", {}, ("low", "short_question")),
    ("现在几点了", {}, ("low", "short_question")),
    ("今天天气怎么样", {}, ("low", "short_question")),
    ("你好", {"first": True}, ("low", "short_first_turn")),
    ("你好", {}, ("default", "not_question")),
    # bare 么/几/哪 are not question cues (那么/茶几/哪怕)
    ("把茶几搬走", {}, ("default", "not_question")),
    ("哪怕下雨也去", {}, ("default", "not_question")),
    ("讲个笑话吧", {}, ("low", "short_question")),
    # openers never take low
    ("请问你是谁？", {}, ("default", "opener:请问")),
    ("我想问一下几点了", {}, ("default", "opener:我想问一下")),
    ("就是你是谁", {"first": True}, ("default", "opener:就是")),
    ("那么现在呢？", {}, ("default", "opener:那么")),
    # keep: dangling function word / connective (whole words only)
    ("我想去北京然后", {}, ("keep", "tail_word:然后")),
    ("因为。", {"first": True}, ("keep", "tail_word:因为")),
    ("这个问题是关于", {}, ("keep", "tail_word:关于")),
    ("我们去吃的", {}, ("keep", "tail_word:的")),
    ("你先帮我把", {}, ("keep", "tail_word:把")),
    ("我想问一下，", {}, ("keep", "tail_comma")),
    # complete words ending in a tail character are not dangling
    ("好的", {"first": True}, ("low", "short_first_turn")),
    ("是不是", {}, ("low", "short_question")),
    ("你在不在", {}, ("low", "short_question")),
    ("现在", {"first": True}, ("low", "short_first_turn")),
    ("是。", {"first": True}, ("low", "short_first_turn")),
    ("这是谁的？", {}, ("low", "short_question")),
    # duration
    ("你是谁？", {"s": 1.5}, ("default", "not_short")),
    ("你是谁？", {"s": 3.99}, ("default", "not_short")),
    ("你是谁？", {"s": 4.0}, ("keep", "long_input")),
    # playback-time candidates and English never lower
    ("你是谁？", {"playing": True}, ("keep", "playing")),
    ("Who are you?", {}, ("keep", "not_chinese")),
    ("what is AI", {"first": True}, ("keep", "not_chinese")),
    ("AI是什么？", {}, ("low", "short_question")),
    # only yield_ready may lower
    ("你是谁？", {"label": "yield_wait"}, ("default", "not_ready")),
    ("你是谁？", {"label": "wait_release"}, ("default", "not_ready")),
    ("", {"first": True}, ("default", "no_transcript")),
    # optional low-RTT keep (disabled by default)
    ("你是谁？", {"rtt": 10}, ("low", "short_question")),
    ("你是谁？", {"rtt": 10, "min_rtt": 30}, ("keep", "low_rtt")),
    ("你是谁？", {"rtt": 45, "min_rtt": 30}, ("low", "short_question")),
    ("你是谁？", {"rtt": None, "min_rtt": 30}, ("low", "short_question")),
])
def test_rule_table(text, kwargs, expected):
    assert row(text, **kwargs) == expected


@pytest.mark.parametrize("tier,reason,elapsed,expected", [
    ("low", "short_question", 0.0, ("low", "short_question", .4)),
    ("low", "short_question", .399, ("low", "short_question", .4)),
    # transcript arrived after the low hold would have expired: default, never immediate
    ("low", "short_question", .4, ("default", "low_late", .576)),
    ("low", "short_question", 1.2, ("default", "low_late", .576)),
    ("keep", "tail_word:然后", 2.0, ("keep", "tail_word:然后", .64)),
    ("default", "not_question", .5, ("default", "not_question", .576)),
])
def test_resolve_late_transcript_never_lowers(tier, reason, elapsed, expected):
    assert resolve(CFG, tier, reason, elapsed_s=elapsed) == expected


def test_config_defaults_reproduce_single_hold():
    cfg = hold_config({}, .64)
    assert (cfg["low"], cfg["default"], cfg["keep"]) == (.64, .64, .64)
    assert not cfg["evaluate"] and not cfg["apply"] and cfg["min_rtt_ms"] == 0
    shadow = hold_config({"end_hold_tiers": TIERS, "hold_tier_rules": True, "hold_tier_shadow": True}, .64)
    assert shadow["evaluate"] and not shadow["apply"]


@pytest.mark.parametrize("cfg", [
    {"end_hold_tiers": {"low": .4, "default": .576}},
    {"end_hold_tiers": {"low": .6, "default": .576, "keep": .64}},
    {"end_hold_tiers": {"low": .4, "default": .576, "keep": 3}},
    {"end_hold_tiers": {"low": True, "default": .576, "keep": .64}},
    {"end_hold_tiers": {"low": float("nan"), "default": .576, "keep": .64}},
    {"end_hold_tiers": [.4, .576, .64]},
    {"hold_tier_rules": "yes"}, {"hold_tier_shadow": 1},
    {"hold_min_rtt_ms": -1}, {"hold_min_rtt_ms": True},
])
def test_config_rejects_invalid(cfg):
    with pytest.raises(ValueError):
        hold_config(cfg, .64)


# ---- guarded actor: per-candidate hold ----

from test_actor_candidate import frame, pump, ended
from test_guarded_turns import guarded, cleanup, playing, new_input


def route(transcript, label="yield_ready"):
    return json.dumps({"transcript": transcript, "label": label}, ensure_ascii=False)


def configure(e, **cfg):
    e.engine_cfg.update(cfg)
    e._init_guarded()


def hold_events(sock):
    return [x["data"] for x in sock.events if x["event"] == "vad_640_done"]


async def test_default_config_keeps_legacy_hold_and_records_diagnostics():
    e, m, sock = guarded()
    try:
        await ended(e)
        await pump(e, lambda: e._candidate is not None)
        c = e._candidate
        assert (c.hold_s, c.hold_tier, c.hold_reason, c.hold_rule) == (.64, "default", "initial", None)
        assert c.hold_text == "测试语音" and c.hold_transcript_ms is not None
        await frame(e, .66)
        assert not c.confirmed
        await frame(e, .704)
        assert c.confirmed
        [done] = hold_events(sock)
        assert done["hold_s"] == .64 and done["hold_tier"] == "default" and "hold_rule_tier" not in done
        vad = next(x["data"] for x in sock.events if x["event"] == "vad_done")
        assert vad["vad_silence_ms"] == 100 and vad["vad_effective_end_ms"] == 160
    finally:
        await cleanup(e)


async def test_low_tier_releases_early_and_records_tier():
    e, m, sock = guarded()
    configure(e, end_hold_tiers=TIERS, hold_tier_rules=True)
    m.labels["input_route"] = route("你是谁？")
    try:
        await ended(e)
        await pump(e, lambda: e._candidate is not None)
        c = e._candidate
        assert (c.hold_s, c.hold_tier, c.hold_reason) == (.4, "low", "short_question")
        await frame(e, .4)
        assert not c.confirmed
        await frame(e, .44)
        assert c.confirmed
        [done] = hold_events(sock)
        assert (done["hold_s"], done["hold_tier"], done["hold_reason"]) == (.4, "low", "short_question")
        assert done["hold_rule_tier"] == "low" and done["transcript_after_anchor_ms"] < 400
    finally:
        await cleanup(e)


async def test_late_transcript_does_not_lower_hold():
    e, m, sock = guarded(blocked="input_route")
    configure(e, end_hold_tiers=TIERS, hold_tier_rules=True)
    m.labels["input_route"] = route("你是谁？")
    try:
        await ended(e)
        await m.entered["input_route"].wait()
        for t in (.1, .2, .3, .4, .5):
            await frame(e, t)
        m.gate.set()
        await pump(e, lambda: e._candidate is not None)
        c = e._candidate
        assert (c.hold_s, c.hold_tier, c.hold_reason) == (.576, "default", "low_late")
        await frame(e, .6)
        assert not c.confirmed
        await frame(e, .62)
        assert c.confirmed
        assert hold_events(sock)[0]["transcript_after_anchor_ms"] >= 400
    finally:
        await cleanup(e)


async def test_tier_is_judged_on_merged_full_text_not_newest_segment():
    e, m, _ = guarded()
    configure(e, end_hold_tiers=TIERS, hold_tier_rules=True)
    try:
        m.labels["input_route"] = route("我想问一下", "yield_wait")
        await new_input(e, 1.)
        await pump(e, lambda: e._guard_input.decided)
        assert e._guard_wait_text == "我想问一下"
        m.labels["input_route"] = route("你是谁？")
        await new_input(e, 2.)
        await pump(e, lambda: e._candidate is not None)
        c = e._candidate
        assert c.hold_text == "我想问一下你是谁？"
        assert (c.hold_tier, c.hold_reason) == ("default", "opener:我想问一下")
        assert e._guard_wait_text == ""
    finally:
        await cleanup(e)


async def test_accepted_unanswered_context_transcript_is_already_full_text():
    e, m, _ = guarded()
    configure(e, end_hold_tiers=TIERS, hold_tier_rules=True, input_continuation_context=True)
    async def text(messages):
        # The continuation prompt extends the system text; route it like input_route.
        if messages[0]["content"].startswith("input_route"):
            yield m.labels["input_route"]
        else:
            async for part in m.text(messages):
                yield part
    e.text_stream_fn = text
    try:
        m.labels["input_route"] = route("我想问一下", "yield_wait")
        await new_input(e, 1.)
        await pump(e, lambda: e._guard_input.decided)
        # The route heard both blocks and transcribed the whole merged audio.
        m.labels["input_route"] = route("我想问一下你是谁？")
        await new_input(e, 2.)
        await pump(e, lambda: e._candidate is not None)
        assert e._guard_input.context_source == "accepted_unanswered_user_audio"
        assert e._candidate.hold_text == "我想问一下你是谁？"
    finally:
        await cleanup(e)


async def test_playback_time_candidate_never_lowers():
    e, m, _ = guarded()
    configure(e, end_hold_tiers=TIERS, hold_tier_rules=True)
    try:
        await playing(e)
        m.labels["input_route"] = route("你是谁？")
        await new_input(e, 1.)
        await pump(e, lambda: e._candidate is not None and e._candidate.turn == 1)
        c = e._candidate
        assert (c.hold_s, c.hold_tier, c.hold_reason) == (.64, "keep", "playing")
    finally:
        await cleanup(e)


async def test_english_candidate_never_lowers():
    e, m, _ = guarded()
    configure(e, end_hold_tiers=TIERS, hold_tier_rules=True)
    m.labels["input_route"] = route("Who are you?")
    try:
        await ended(e)
        await pump(e, lambda: e._candidate is not None)
        assert (e._candidate.hold_s, e._candidate.hold_tier) == (.64, "keep")
    finally:
        await cleanup(e)


async def test_shadow_records_rule_tier_without_changing_hold():
    e, m, sock = guarded()
    configure(e, end_hold_tiers=TIERS, hold_tier_rules=True, hold_tier_shadow=True)
    m.labels["input_route"] = route("你是谁？")
    try:
        await ended(e)
        await pump(e, lambda: e._candidate is not None)
        c = e._candidate
        assert (c.hold_s, c.hold_tier, c.hold_reason) == (.576, "default", "shadow")
        assert c.hold_rule == {"tier": "low", "reason": "short_question", "hold_s": .4}
        await frame(e, .44)
        assert not c.confirmed
        await frame(e, .62)
        [done] = hold_events(sock)
        assert done["hold_s"] == .576 and done["hold_rule_tier"] == "low"
    finally:
        await cleanup(e)


async def test_tiers_without_rules_apply_default_to_every_guarded_candidate():
    e, m, _ = guarded()
    configure(e, end_hold_tiers=TIERS)
    m.labels["input_route"] = route("你是谁？")
    try:
        await ended(e)
        await pump(e, lambda: e._candidate is not None)
        c = e._candidate
        assert (c.hold_s, c.hold_tier, c.hold_rule) == (.576, "default", None)
        assert e.END_HOLD == .64, "the legacy path keeps end_hold_frame"
    finally:
        await cleanup(e)


async def test_continuation_gap_and_first_audio_are_recorded():
    e, m, sock = guarded()
    try:
        await ended(e)
        await pump(e, lambda: e._candidate is not None and e._candidate.first_audio_ms is not None)
        await frame(e, .3, {"start": .3}, .2)
        start = [x["data"] for x in sock.events if x["event"] == "vad_start"][-1]
        assert start["prev_end_gap_ms"] == pytest.approx(268, abs=.2)
        assert start["held_candidate_id"] == e._candidate.cid and start["held_hold_s"] == .64
        assert start["vad_state"] == "preplay"
    finally:
        await cleanup(e)


async def test_client_hold_and_rtt_telemetry_reach_the_trace_with_actor_context():
    from engine import ControlMsg
    from demo_session_replay import MemoryTrace
    e, m, _ = guarded()
    configure(e, end_hold_tiers=TIERS, hold_tier_rules=True, hold_min_rtt_ms=30)
    e.demo_trace = MemoryTrace()
    m.labels["input_route"] = route("你是谁？")
    try:
        for rtt in (8, 12, 10):
            await e._process_event(ControlMsg("demo_telemetry", {"kind": "rtt", "rtt_ms": rtt}))
        await ended(e)
        await pump(e, lambda: e._candidate is not None)
        assert (e._candidate.hold_tier, e._candidate.hold_reason) == ("keep", "low_rtt")
        await e._process_event(ControlMsg("demo_telemetry", {"kind": "client_hold", "action": "hold",
                                                             "utterance_id": 1}))
        [row] = [r for r in e.demo_trace.rows if r["event"] == "client_hold_actor"]
        assert row["data"]["action"] == "hold" and row["data"]["candidate_id"] == e._candidate.cid
        assert row["data"]["hold_tier"] == "keep" and row["t_audio"] == pytest.approx(.032)
        decision = next(r for r in e.demo_trace.rows if r["event"] == "input_decision")
        assert decision["data"]["route_ms"] is not None and decision["data"]["decision_after_end_ms"] == 0
    finally:
        await cleanup(e)
