"""Conditional reply review: evidence, shared deadline and real Actor races."""
import asyncio
import json

import pytest

from test_guarded_turns import guarded, playing, cleanup, new_input, route_context
from test_actor_candidate import frame, pump
from control_labels import decide_input_route, parse_label
from speech_reference import completed_context
from engine import ControlMsg


async def decision(base="keep", transcript="都可以", reply="yield_ready", **kwargs):
    calls = []
    async def call(messages, stage):
        calls.append((stage, messages))
        return json.dumps({"transcript": transcript, "label": base}) if stage == "input_route" else reply
    result = await decide_input_route(call, [], 2, reply_prompt="review", reply_context="你想听哪种故事。",
                                      playing=True, **kwargs)
    return result, calls


async def test_review_preserves_transcript_and_uses_only_text():
    (label, audit), calls = await decision()
    assert label == "yield_ready" and audit["base_label"] == "keep"
    assert audit["transcript"] == "都可以" and len(calls) == 2
    assert json.loads(calls[1][1][1]["content"]) == {
        "played_assistant": "你想听哪种故事。", "user_transcript": "都可以"}


@pytest.mark.parametrize("base,transcript,closed", [
    ("keep", "", True), ("keep", "  ", True), ("keep", "都可以", False),
    ("stop_only", "停", False), ("yield_wait", "等等", True), ("yield_ready", "你是谁", True)])
async def test_ineligible_decisions_never_call_review(base, transcript, closed):
    (label, _), calls = await decision(base, transcript, closed=closed)
    assert label == base and len(calls) == 1


@pytest.mark.parametrize("base,reply,expected", [
    ("stop_only", "yield_ready", "yield_ready"), ("stop_only", "keep", "stop_only"),
    ("stop_only", "not a label", "stop_only"), ("keep", "keep", "keep")])
async def test_review_promotes_keep_or_stop_only_to_ready(base, reply, expected):
    (label, audit), calls = await decision(base, "不可以", reply=reply)
    assert label == expected and audit["base_label"] == base and len(calls) == 2
    review = audit["reply_review"]
    assert review["label"] == (reply if reply in ("keep", "yield_ready") else "keep")
    assert review["fallback"] == (reply not in ("keep", "yield_ready"))


@pytest.mark.parametrize("base", ["keep", "stop_only"])
async def test_review_error_keeps_audio_label(base):
    calls = []
    async def call(messages, stage):
        calls.append(stage)
        if stage == "input_reply":
            raise RuntimeError("review unavailable")
        return json.dumps({"transcript": "不可以", "label": base})
    label, audit = await decide_input_route(call, [], 2, playing=True, reply_prompt="review",
                                            reply_context="我可以继续吗？")
    assert label == base and audit["reply_review"]["fallback"]
    assert calls == ["input_route", "input_reply"]


@pytest.mark.parametrize("playing,context,prompt", [(False, "问题", "review"),
    (True, "", "review"), (True, "问题", None)])
async def test_missing_onset_context_or_disabled_review_costs_no_call(playing, context, prompt):
    calls = []
    async def call(messages, stage):
        calls.append(stage)
        return '{"transcript":"都可以","label":"keep"}'
    label, _ = await decide_input_route(call, [], 2, playing=playing,
                                        reply_context=context, reply_prompt=prompt)
    assert label == "keep" and calls == ["input_route"]


@pytest.mark.parametrize("base", ["keep", "stop_only"])
@pytest.mark.parametrize("raw", ["stop_only", "yield_wait", "ready", "keep or yield_ready",
    '"yield_ready"', '{"label":"yield_ready"}', "yield_ready because yes", ""])
async def test_review_cannot_expand_authority_or_repair(raw, base):
    (label, audit), calls = await decision(base, reply=raw)
    assert label == base and len(calls) == 2 and audit["reply_review"]["fallback"]
    assert parse_label("input_reply", raw) is None


async def test_invalid_audio_fallback_cannot_be_promoted():
    calls = []
    async def call(messages, stage):
        calls.append(stage)
        return 'not valid'
    label, audit = await decide_input_route(call, [], 2, playing=True, reply_prompt="review", reply_context="问句")
    assert label == "keep" and audit["fallback"] and calls == ["input_route"] * 2
    assert "reply_review" not in audit


async def test_shared_deadline_includes_audio_time_and_cancels_review(monkeypatch):
    import control_labels as c
    now = [100.]
    monkeypatch.setattr(c.time, "perf_counter", lambda: now[0])
    budgets = []
    async def wait(awaitable, timeout):
        budgets.append(timeout)
        return await awaitable
    monkeypatch.setattr(c, "cancellable_wait", wait)
    async def call(messages, stage):
        if stage == "input_route":
            now[0] += 1.6
            return '{"transcript":"都可以","label":"keep"}'
        raise asyncio.TimeoutError
    label, audit = await decide_input_route(call, [], 2, playing=True, reply_prompt="review", reply_context="问句")
    assert label == "keep" and budgets == pytest.approx([2, .4])
    assert audit["reply_review"]["timed_out"]


def test_context_requires_end_ack_and_preserves_whole_sentences():
    sentences = [(100, "想听童话吗？"), (200, "还是科幻故事？")]
    assert completed_context(sentences, 99) == ""
    assert completed_context(sentences, 100) == "想听童话吗？"
    assert completed_context(sentences, 199) == "想听童话吗？"
    assert completed_context(sentences, 200) == "想听童话吗？还是科幻故事？"
    assert completed_context([(1, "旧问题？"), (2, "长" * 513)], 2) == ""
    assert completed_context([(i, str(i)) for i in range(5)], 4) == "234"


async def prepare(blocked=None, **cfg):
    e, m, sock = guarded()
    if cfg:
        e.engine_cfg.update(cfg)
        e._init_guarded()
    e.prompts["input_reply"] = "input_reply"
    m.labels["input_reply"] = "yield_ready"
    m.entered["input_reply"] = asyncio.Event()
    await playing(e)
    await pump(e, lambda: len(e._guard_outputs[e._speech.sid]["sentences"]) >= 2)
    record = e._guard_outputs[e._speech.sid]
    record["sentences"][0] = (record["sentences"][0][0], "请告诉我想听哪种故事。")
    m.labels["input_route"] = "keep"
    m.blocked = blocked
    return e, m, sock


async def ack_first(e):
    end = e._guard_outputs[e._speech.sid]["sentences"][0][0]
    await e._process_event(ControlMsg("playback_progress", {
        "utterance_id": e._speech.sid, "played_samples": end, "started": True}))


async def test_on_review_hook_runs_once_with_base_label_only_when_review_runs():
    seen = []
    (label, _), calls = await decision("stop_only", "不可以", on_review=seen.append)
    assert seen == ["stop_only"] and len(calls) == 2 and label == "yield_ready"
    seen.clear()
    (label, _), calls = await decision("yield_wait", "等等", on_review=seen.append)
    assert seen == [] and len(calls) == 1 and label == "yield_wait"


def test_heard_window_requires_started_and_heard_units():
    from speech_reference import SpeechReference
    r = SpeechReference()
    for start, text in ((0, "第一句，"), (100, "第二句。"), (200, "第三句？")):
        r.observe("sentence", {"start_sample": start, "text": text})
    assert r.heard_window(played=0, sent=300, min_heard=0) == ""
    assert r.heard_window(played=100, sent=300, min_heard=0) == "第一句，"
    assert r.heard_window(played=101, sent=300, min_heard=0) == "第一句，第二句。"
    assert r.heard_window(played=150, sent=300, min_heard=60) == "第一句，"
    assert r.heard_window(played=150, sent=120, min_heard=0) == "第一句，第二句。"
    assert r.heard_window(played=150, sent=100, min_heard=0) == "第一句，"
    assert r.heard_window(played=50, sent=300, min_heard=60) == ""
    assert r.heard_window(played=59, sent=300, min_heard=60) == ""
    assert r.heard_window(played=60, sent=300, min_heard=60) == "第一句，"


async def onset_after(e, played):
    e._speech.played = played
    await frame(e, 1, {"start": 1}, .2)
    return e._guard_input


async def test_unheard_onset_snapshot_stays_empty_and_skips_review(monkeypatch):
    import guarded_turns
    monkeypatch.setattr(guarded_turns, "REPLY_MIN_HEARD_S", 1000)
    e, m, _ = await prepare()
    try:
        record = e._guard_outputs[e._speech.sid]
        span = await onset_after(e, record["reference"].sentences[0][0] + 1)
        assert span.reply_context == "" and span.reply_context_kind is None
        await ack_first(e)
        await frame(e, 1.1, {"end": 1.1}, .2)
        await pump(e, lambda: span.decided)
        # A later ACK never fills an empty onset snapshot.
        assert span.reply_context == "" and span.reply_context_kind is None
        assert not m.entered["input_reply"].is_set() and e._speech is not None
    finally:
        await cleanup(e)


async def test_heard_onset_window_is_frozen_and_sent_to_review(monkeypatch):
    import guarded_turns
    monkeypatch.setattr(guarded_turns, "REPLY_MIN_HEARD_S", 0)
    e, m, _ = await prepare()
    try:
        record = e._guard_outputs[e._speech.sid]
        start, text = record["reference"].sentences[0]
        span = await onset_after(e, start + 1)
        assert span.reply_context_kind == "sentence_window" and span.reply_context == text
        await ack_first(e)
        await frame(e, 1.1, {"end": 1.1}, .2)
        await pump(e, lambda: span.decided)
        assert span.reply_context == text and span.reply_context_kind == "sentence_window"
        review = next(msg for kind, msg in m.calls if kind == "input_reply")
        assert json.loads(review[1]["content"])["played_assistant"] == text
    finally:
        await cleanup(e)


@pytest.mark.parametrize("reply,final", [("yield_ready", "yield_ready"), ("keep", "stop_only"),
                                         ("not a label", "stop_only")])
async def test_stop_cancels_playback_before_review_returns(reply, final):
    e, m, sock = await prepare(blocked="input_reply")
    m.labels["input_route"] = "stop_only"
    m.labels["input_reply"] = reply
    try:
        await ack_first(e)
        await new_input(e)
        await asyncio.wait_for(m.entered["input_reply"].wait(), 1)
        await pump(e, lambda: any(x["event"] == "speech_cancelled" for x in sock.events))
        # Playback stopped while the review is still blocked.
        assert not e._guard_input.decided and m.responses == 1
        m.gate.set()
        await pump(e, lambda: e._guard_input.decided)
        assert e._guard_input.route == final
        assert sum(x["event"] == "speech_cancelled" for x in sock.events) == 1
        if final == "stop_only":
            await asyncio.sleep(0.05)
            assert m.responses == 1 and e._guard_wait_reason == "stop_only"
            assert any(x["event"] == "input_waiting" and x["data"].get("reason") == "stop_only"
                       for x in sock.events)
        else:
            await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
            await frame(e, 2)
            await pump(e, lambda: e._speech is not None and e._speech_meta.turn == 1)
            await pump(e, lambda: m.responses == 2)
    finally:
        await cleanup(e)


@pytest.mark.parametrize("reset", [False, True])
async def test_interim_stop_then_continuation_or_reset_never_cancels_twice(reset):
    e, m, sock = await prepare(blocked="input_reply")
    m.labels["input_route"] = "stop_only"
    try:
        await ack_first(e)
        await new_input(e)
        await asyncio.wait_for(m.entered["input_reply"].wait(), 1)
        await pump(e, lambda: any(x["event"] == "speech_cancelled" for x in sock.events))
        span = e._guard_input
        if reset:
            e._reset_session()
        else:
            await frame(e, 1.2, {"start": 1.2}, .2)
        m.gate.set()
        await asyncio.gather(*e._guard_tasks, return_exceptions=True)
        await asyncio.sleep(0)
        while not e.q.empty():
            await e._process_event(e.q.get_nowait())
        assert sum(x["event"] == "speech_cancelled" for x in sock.events) == 1
        assert m.responses == 1
        if reset:
            assert e._guard_input is None
        else:
            # The continuation made the in-flight final decision stale.
            assert not span.decided and span.admitted
    finally:
        await cleanup(e)


async def test_interim_stop_survives_old_turn_eof_and_answers():
    e, m, sock = await prepare(blocked="input_reply")
    m.labels["input_route"] = "stop_only"
    try:
        await ack_first(e)
        old_turn = e.TURN_IDX
        await new_input(e)
        await asyncio.wait_for(m.entered["input_reply"].wait(), 1)
        await pump(e, lambda: any(x["event"] == "speech_cancelled" for x in sock.events))
        await e._guard_finish_turn(old_turn, "played")
        m.gate.set()
        await pump(e, lambda: e._guard_input.decided)
        assert e._guard_input.route == "yield_ready" and e._candidate is not None
        assert sum(x["event"] == "speech_cancelled" for x in sock.events) == 1
    finally:
        await cleanup(e)


async def test_onset_window_threshold_is_half_second_of_pipeline_audio(monkeypatch):
    import speech_reference
    seen = {}
    original = speech_reference.SpeechReference.heard_window

    def spy(self, **kwargs):
        seen.update(kwargs)
        return original(self, **kwargs)
    monkeypatch.setattr(speech_reference.SpeechReference, "heard_window", spy)
    e, m, _ = await prepare()
    try:
        record = e._guard_outputs[e._speech.sid]
        span = await onset_after(e, record["reference"].sentences[0][0] + 1)
        rate = getattr(e._speech, "rate", None) or 24000
        assert seen["min_heard"] == int(rate * 0.5) and seen["played"] == span.reply_played_samples
    finally:
        await cleanup(e)


@pytest.mark.parametrize("base", ["keep", "stop_only"])
async def test_reply_admission_cancels_once_and_starts_one_response(base):
    e, m, sock = await prepare()
    m.labels["input_route"] = base
    try:
        await ack_first(e)
        await new_input(e)
        await pump(e, lambda: e._candidate is not None and e._candidate.pipeline is not None)
        await frame(e, 2)
        await pump(e, lambda: e._speech is not None and e._speech_meta.turn == 1)
        await pump(e, lambda: m.responses == 2)
        assert e._guard_input.route == "yield_ready"
        assert sum(x["event"] == "speech_cancelled" for x in sock.events) == 1
        review = next(msg for kind, msg in m.calls if kind == "input_reply")
        assert json.loads(review[1]["content"])["played_assistant"] == "请告诉我想听哪种故事。"
    finally:
        await cleanup(e)


@pytest.mark.parametrize("reset", [False, True])
async def test_continuation_or_reset_cancels_inflight_review(reset):
    e, m, _ = await prepare(blocked="input_reply")
    try:
        await ack_first(e)
        await new_input(e)
        await asyncio.wait_for(m.entered["input_reply"].wait(), 1)
        epoch, sid = e.seg_epoch, e._speech.sid
        if reset:
            e._reset_session()
        else:
            await frame(e, 1.2, {"start": 1.2}, .2)
        m.gate.set()
        await asyncio.gather(*e._guard_tasks, return_exceptions=True)
        await asyncio.sleep(0)
        while not e.q.empty():
            await e._process_event(e.q.get_nowait())
        if reset:
            assert e._speech is None and e._guard_input is None
        else:
            assert e._speech.sid == sid and e.seg_epoch == epoch and not e._guard_input.admitted
    finally:
        await cleanup(e)


async def test_eof_keeps_onset_context_for_pending_review():
    e, m, _ = await prepare(blocked="input_reply")
    try:
        await ack_first(e)
        await new_input(e)
        await asyncio.wait_for(m.entered["input_reply"].wait(), 1)
        snapshot = e._guard_input.reply_context
        await e._guard_finish_turn(e.TURN_IDX, "played")
        m.gate.set()
        await pump(e, lambda: e._guard_input.decided)
        assert e._guard_input.route == "yield_ready" and e._guard_input.reply_context == snapshot
        assert e._candidate is not None
    finally:
        await cleanup(e)


async def test_review_failure_releases_capacity_and_keeps_playback():
    e, m, _ = await prepare()
    original = e.text_stream_fn
    async def broken(messages):
        if messages[0]["content"] == "input_reply":
            raise RuntimeError("review unavailable")
        async for part in original(messages):
            yield part
    e.text_stream_fn = broken
    try:
        await ack_first(e)
        sid = e._speech.sid
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        assert e._speech.sid == sid and not e._guard_input.admitted
    finally:
        await cleanup(e)


STORY = "小狐狸在林子里，发现了一只小鸟，它摔伤了翅膀。"
STORY_BASE = "请告诉我想听哪种故事。第二句。"
# Worst captured demo TTS units against the duration bound (text, measured seconds).
CAPTURED_UNITS = [
    ("第二天：上午西溪湿地或中国丝绸博物馆，下午南宋御街逛吃。", 7.63), ("啊，9能被3整除！", 2.42),
    ("第二天：河坊街+南宋御街，傍晚钱塘江边散步。", 5.84), ("适合孩子也深受大人喜爱，能唤起纯真回忆。", 5.27),
    ("I'm Qwen-Omni, a multimodal large-scale language model developed by Alibaba's Tongyi Lab.", 7.72),
    ("第一天：上午到杭州，先去西湖景区逛逛，推荐断桥残雪、苏堤春晓；", 7.92), ("晚上河坊街吃小吃。", 2.23),
    ("嗨，好啊！", 1.24), ("包括八大行星、它们的卫星、矮行星以及无数的小行星和彗星等。", 7.16),
    ("而科幻故事则基于科学想象，探讨未来科技、宇宙探索或人工智能等主题，更偏理性与悬念。", 10.11),
    ("11不能被3整除哦。", 2.42), ("是的，11不能被3整除。", 2.66), ("我是Qwen-Omni，", 1.62),
    ("因为3乘以3等于9，3乘以4等于12，中间没有11这个数。", 6.07),
    ("Cats can make over 100 different sounds, which is more than any other domestic animal except humans.", 7.08),
    (" which is more than any other animal in the feline family.", 3.57)]


def test_unit_duration_bound_covers_captured_tts():
    from speech_reference import unit_seconds_bound
    for text, seconds in CAPTURED_UNITS:
        assert unit_seconds_bound(text) >= 1.05 * seconds, text


def test_clause_prefix_never_ends_inside_a_phrase():
    from speech_reference import clause_prefix
    assert clause_prefix("接下来你想听童话还是科幻？", 8) == ""
    assert clause_prefix("你想听童话还是科幻故事，我都会讲。", 14) == "你想听童话还是科幻故事，"
    assert clause_prefix("好的、我们开始吧。", 8) == ""
    assert clause_prefix("It costs 3.5 dollars, ok?", 12) == ""
    assert clause_prefix("It costs 3.5 dollars, ok?", 23) == "It costs 3.5 dollars,"
    assert clause_prefix("Why? Because", 4) == "Why?" and clause_prefix("Why?", 9) == "Why?"


def test_heard_prefix_counts_only_the_played_clauses_of_the_playing_unit():
    from speech_reference import SpeechReference
    r = SpeechReference()
    units = ((0, 100, "第一句。"), (100, 300, STORY), (300, 400, "第三句？"))
    for start, _, text in units:
        r.observe("sentence", {"start_sample": start, "text": text})
    known = dict(sent=400, ends=[100, 300], min_heard=0, rate=100)
    assert r.heard_prefix(played=100, **known) == ""
    # A phrase cut mid-way is dropped, not offered as a (different) question.
    assert r.heard_prefix(played=150, **known) == ""
    assert r.heard_prefix(played=200, **known) == STORY[:8]
    assert r.heard_prefix(played=299, **known) == STORY[:16]
    # An ended unit is completed context, never a partial; the next one has not started.
    assert r.heard_prefix(played=300, **known) == ""
    assert r.heard_prefix(played=200, **{**known, "min_heard": 101}) == ""
    assert r.heard_prefix(played=200, **{**known, "min_heard": 100}) == STORY[:8]
    assert r.heard_prefix(played=200, **{**known, "sent": 100}) == ""
    assert r.heard_prefix(played=299, **known, limit=10) == STORY[:8]
    assert r.heard_prefix(played=299, **known, limit=7) == ""
    assert r.heard_prefix(played=350, **known) == ""
    for played in range(401):
        heard = r.heard_prefix(played=played, **known)
        start, end, text = max((u for u in units if u[0] < played), default=(0, 1, ""))
        assert text.startswith(heard) and (heard != text or not text)
        assert len(heard) * (end - start) <= len(text) * (played - start)
        assert heard == "" or heard[-1] in "，。？"
    # Unsent END: bound the unit by its text, never by `sent`, which runs ahead of playback.
    r = SpeechReference()
    for start, _, text in units[:2]:
        r.observe("sentence", {"start_sample": start, "text": text})
    for played in range(101, 900):
        heard = r.heard_prefix(played=played, sent=played + 60, ends=[100], min_heard=0, rate=100)
        assert STORY.startswith(heard) and heard != STORY
        assert len(heard) * 690 <= len(STORY) * (played - 100)
    assert r.heard_prefix(played=339, sent=399, ends=[100], min_heard=0, rate=100) == ""
    assert r.heard_prefix(played=340, sent=400, ends=[100], min_heard=0, rate=100) == STORY[:8]
    long = SpeechReference()
    long.observe("sentence", {"start_sample": 0, "text": "长，" * 300})
    assert long.heard_prefix(played=50, sent=100, ends=[], min_heard=0, rate=100) == ""


def test_heard_prefix_without_end_never_outruns_sent_or_the_whole_unit():
    from speech_reference import SpeechReference, unit_seconds_bound
    r = SpeechReference()
    text = "Hi, yo, ok, go, no."
    r.observe("sentence", {"start_sample": 0, "text": text})
    assert round(100 * unit_seconds_bound(text)) == 250
    # The text bound is short here: the unit has still not ended by `sent`.
    heard = r.heard_prefix(played=500, sent=1000, ends=[], min_heard=0, rate=100)
    assert heard == "Hi, yo," and len(heard) * 1000 <= len(text) * 500
    # Starved playback (played == sent) without END: never the whole unit.
    r = SpeechReference()
    r.observe("sentence", {"start_sample": 0, "text": "Hi, yo."})
    assert r.heard_prefix(played=300, sent=300, ends=[], min_heard=0, rate=100) == "Hi,"


@pytest.mark.parametrize("value,valid", [(0, True), (0.5, True), (2, True), (0.1, False), (3, False)])
def test_partial_heard_setting_is_bounded(value, valid):
    e, _, _ = guarded()
    e.engine_cfg["input_reply_partial_heard_s"] = value
    if valid:
        e._init_guarded()
    else:
        with pytest.raises(ValueError):
            e._init_guarded()


async def story_review(played, end=None, base=None, **cfg):
    """Input onset `played` samples into a long third unit (negative: before it starts).

    `end` is the unit's length once fully sent; None means its END has not been sent.
    Returns the span, model calls, input_dispatch rows, case contexts, decision audits
    and the assistant history of the interrupted turn.
    """
    e, m, _ = await prepare(input_route_conversation=True, **cfg)
    rows, cases = [], []
    e._observe = lambda event, data=None: rows.append((event, dict(data or {})))
    capacity = e._capacity_stream

    def record_case(kind, fn, arg, **kw):
        if "decision_stage" in (kw.get("case_context") or {}):
            cases.append(dict(kw["case_context"]))
        return capacity(kind, fn, arg, **kw)
    e._capacity_stream = record_case
    e._users_by_turn[0] = "讲个故事"
    try:
        record = e._guard_outputs[e._speech.sid]
        if base is not None:
            record["sentences"][0] = (record["sentences"][0][0], base)
        start = record["sentences"][-1][0]
        record["reference"].observe("sentence", {"start_sample": start, "text": STORY})
        if end is not None:
            record["sentences"].append((start + end, STORY))
        e._speech.sent = max(e._speech.sent, start + (end or 0), start + played + 14400)
        span = await onset_after(e, start + played)
        await frame(e, 1.1, {"end": 1.1}, .2)
        await pump(e, lambda: span.decided)
        if span.route == "yield_ready":
            await pump(e, lambda: record.get("cancelled"))
        calls = [(kind, msgs) for kind, msgs in m.calls if kind in ("input_route", "input_reply")]
        dispatch = [{k: v for k, v in data.items() if k != "parent_id"}
                    for event, data in rows if event == "input_dispatch"]
        audits = [data["audit"] for event, data in rows if event == "input_decision" and data["closed"]]
        return span, calls, dispatch, cases, audits, e._assistants_by_turn.get(record["turn"])
    finally:
        await cleanup(e)


@pytest.mark.parametrize("played,end,partial", [(9600, 72000, ""), (12000, 72000, ""),
    (48000, 72000, STORY[:8]), (70000, 72000, STORY[:16]), (48000, None, ""), (72000, None, STORY[:8])])
async def test_partial_heard_adds_only_the_played_clauses_to_review(played, end, partial):
    span, calls, dispatch, cases, audits, _ = await story_review(played, end, input_reply_partial_heard_s=0.5)
    reviews = [json.loads(msgs[1]["content"]) for kind, msgs in calls if kind == "input_reply"]
    assert reviews == [{"played_assistant": STORY_BASE + partial, "user_transcript": "测试语音"}]
    # Addressee context and the frozen onset snapshot keep completed sentences only.
    assert span.reply_context == STORY_BASE and span.reply_partial_chars == len(partial)
    assert span.reply_review_context == (STORY_BASE + partial if partial else "")
    assert span.reply_context_kind == ("completed_and_playing_prefix" if partial else "completed_sentences")
    route = [msgs for kind, msgs in calls if kind == "input_route"]
    assert route_context(route[-1])["conversation"]["assistant_speech_heard"] == STORY_BASE
    assert dispatch[-1]["reply_context_chars"] == len(STORY_BASE + partial)
    assert dispatch[-1]["reply_partial_chars"] == len(partial)
    assert cases[-1]["reply_partial_chars"] == len(partial)
    assert audits[-1]["route"]["reply_review"]["partial_chars"] == len(partial)


async def test_partial_heard_off_keeps_route_and_review_payloads_byte_identical():
    absent = await story_review(48000, 72000)
    zero = await story_review(48000, 72000, input_reply_partial_heard_s=0)
    on = await story_review(48000, 72000, input_reply_partial_heard_s=0.5)
    assert absent[1] == zero[1] and absent[2] == zero[2] and absent[3] == zero[3]
    assert [kind for kind, _ in absent[1]][-2:] == ["input_route", "input_reply"]
    assert absent[1][-1][1][1]["content"] == json.dumps(
        {"played_assistant": STORY_BASE, "user_transcript": "测试语音"}, ensure_ascii=False)
    assert all("reply_partial_chars" not in row for row in absent[2] + absent[3])
    assert "partial_chars" not in absent[4][-1]["route"]["reply_review"]
    assert absent[0].reply_review_context == "" and absent[0].reply_partial_chars == 0
    # Enabled: the audio route request is unchanged; only the review text grows.
    assert [msgs for kind, msgs in on[1] if kind == "input_route"] == \
           [msgs for kind, msgs in absent[1] if kind == "input_route"]
    assert json.loads(on[1][-1][1][1]["content"])["played_assistant"] == STORY_BASE + STORY[:8]
    # The answer's history is the same either way: completed sentences only.
    assert absent[5] == on[5] == STORY_BASE


async def test_partial_never_displaces_completed_sentences_from_review():
    long = "你想听哪个？" + "长" * 490 + "。"
    off = await story_review(70000, 72000, base=long)
    on = await story_review(70000, 72000, base=long, input_reply_partial_heard_s=0.5)
    heard = json.loads(on[1][-1][1][1]["content"])["played_assistant"]
    base = json.loads(off[1][-1][1][1]["content"])["played_assistant"]
    assert base.startswith("你想听哪个？") and len(base) == 501
    assert heard == base + STORY[:8] and on[0].reply_partial_chars == 8


@pytest.mark.parametrize("end", [72000, None])
@pytest.mark.parametrize("played", [-7580, -4000, 0, 9600, 12000, 48000, 70000, 72000])
async def test_partial_heard_never_changes_which_inputs_are_reviewed(played, end):
    off_span, off_calls, *_ = await story_review(played, end)
    on_span, on_calls, *_ = await story_review(played, end, input_reply_partial_heard_s=0.5)
    assert [kind for kind, _ in on_calls] == [kind for kind, _ in off_calls]
    assert on_span.route == off_span.route and on_span.reply_context == off_span.reply_context
    for (_, off), (kind, on) in zip(off_calls, on_calls):
        if kind == "input_reply":
            heard = json.loads(on[1]["content"])["played_assistant"]
            base = json.loads(off[1]["content"])["played_assistant"]
            assert heard.startswith(base) and STORY.startswith(heard[len(base):])
