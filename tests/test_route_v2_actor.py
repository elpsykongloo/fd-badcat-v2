"""transcript-first-v2 and label-close early decision inside the real actor/queues."""
import asyncio
import json

import pytest

from test_actor_candidate import ended, pump
from test_guarded_turns import guarded, cleanup
from control_labels import route_prefill_message, ROUTE_PREFIX
from demo_cases import CaseArchive, load_case


def instrument(e, pieces_by_call):
    """Route calls stream scripted deltas; other calls use the fake models."""
    events, route_requests, consumed = [], [], []
    original = e.text_stream_fn
    e._observe = lambda event, data=None: events.append((event, data or {}))

    async def text(messages):
        if messages[0]["content"] != "input_route":
            async for part in original(messages):
                yield part
            return
        route_requests.append(messages)
        pieces = pieces_by_call[min(len(route_requests), len(pieces_by_call)) - 1]
        count = 0
        try:
            for piece in pieces:
                count += 1
                yield piece
                await asyncio.sleep(0)
        finally:
            consumed.append(count)
    e.text_stream_fn = text
    return events, route_requests, consumed


def configure(e, **cfg):
    e.engine_cfg.update(input_control_reserve=True, **cfg)
    e._init_guarded()


def decision(events):
    rows = [d for ev, d in events if ev == "input_decision"]
    assert rows
    return rows[-1]


async def test_v2_prefill_early_close_cancels_stream_after_label():
    e, m, _ = guarded()
    configure(e, input_route_prefill=True)
    pieces = ['你', '是谁","la', 'bel":"yield_', 'ready"', '}', '\nIGNORED']
    events, requests, consumed = instrument(e, [pieces])
    try:
        await ended(e)
        await pump(e, lambda: e._guard_input is not None and e._guard_input.decided)
        assert requests[0][-1] == route_prefill_message()
        assert requests[0][-2]["role"] == "user"
        row = decision(events)
        assert row["route"] == "yield_ready" and row["early_closed"] is True
        assert row["route_output_chars"] == len("".join(pieces[:4]))
        assert row["route_label_closed_ms"] is not None and row["route_label_closed_ms"] >= 0
        audit = row["audit"]["route"]
        assert audit["protocol"] == "transcript-first-v2" and audit["transcript"] == "你是谁"
        assert audit["trailing"] == "" and audit["raw"] == ['你是谁","label":"yield_ready"}']
        # The stream was closed right after the label's closing quote arrived.
        assert consumed == [4]
        done = [d for ev, d in events if ev == "model_call_done" and d["kind"] == "input_route"]
        assert [d["status"] for d in done] == ["early_closed"]
        dispatch = [d for ev, d in events if ev == "model_call_dispatch"]
        assert dispatch
    finally:
        await cleanup(e)


async def test_v1_early_close_matches_full_parse_and_records_trailing():
    e, m, _ = guarded()
    configure(e)  # defaults: v1, early close on
    raw = '```json\n{\n  "transcript": "停",\n  "label": "stop_only"\n}\n```'
    cut = raw.index('"stop_only"') + len('"stop_only"')
    pieces = [raw[:cut - 3], raw[cut - 3:cut + 2], raw[cut + 2:]]
    events, requests, consumed = instrument(e, [pieces])
    try:
        await ended(e)
        await pump(e, lambda: e._guard_input is not None and e._guard_input.decided)
        assert requests[0][-1]["role"] == "user"  # no prefill in v1
        row = decision(events)
        audit = row["audit"]["route"]
        assert row["route"] == "stop_only" and row["early_closed"] is True
        assert audit["protocol"] == "transcript-first-v1" and audit["transcript"] == "停"
        assert audit["trailing"] == "\n}"
        assert consumed == [2]
    finally:
        await cleanup(e)


async def test_early_close_off_reads_to_eos_but_still_times_label():
    e, m, _ = guarded()
    configure(e, input_route_prefill=True, input_route_early_close=False)
    pieces = ['你是谁","label":"yield_ready"', '}']
    events, _, consumed = instrument(e, [pieces])
    try:
        await ended(e)
        await pump(e, lambda: e._guard_input is not None and e._guard_input.decided)
        row = decision(events)
        assert row["route"] == "yield_ready" and row["early_closed"] is False
        assert row["route_label_closed_ms"] is not None and consumed == [2]
        assert row["route_output_chars"] == len("".join(pieces))
        done = [d for ev, d in events if ev == "model_call_done" and d["kind"] == "input_route"]
        assert [d["status"] for d in done] == ["completed"]
    finally:
        await cleanup(e)


async def test_trailing_junk_is_audit_only_with_early_close_but_invalid_without():
    junk = ['停","label":"stop_only"}\n```extra']
    for early, label in [(True, "stop_only"), (False, "keep")]:
        e, m, _ = guarded()
        configure(e, input_route_prefill=True, input_route_early_close=early)
        events, requests, _ = instrument(e, [junk])
        try:
            await ended(e)
            await pump(e, lambda: e._guard_input is not None and e._guard_input.decided)
            row = decision(events)
            assert row["route"] == label
            if early:
                assert row["audit"]["route"]["trailing"] == "}\n```extra" and len(requests) == 1
            else:
                # Without early close the full parse rejects; one repair, then fallback keep.
                assert row["audit"]["route"]["fallback"] and len(requests) == 2
                assert requests[1][-1] == route_prefill_message() and requests[1][-2]["role"] == "user"
        finally:
            await cleanup(e)


async def test_invalid_head_does_not_close_and_repairs():
    e, m, _ = guarded()
    configure(e, input_route_prefill=True)
    pieces_by_call = [['","label":"stop_only"}'], ['","label":"keep"}']]  # empty transcript + action is invalid
    events, requests, consumed = instrument(e, pieces_by_call)
    try:
        await ended(e)
        await pump(e, lambda: e._guard_input is not None and e._guard_input.decided)
        row = decision(events)
        audit = row["audit"]["route"]
        assert row["route"] == "keep" and audit["repaired"] and len(requests) == 2
        assert [s["early_closed"] for s in audit["streams"]] == [False, True]
        assert requests[1][-1] == route_prefill_message()
    finally:
        await cleanup(e)


async def test_case_archive_keeps_prefill_request_and_early_closed_status(tmp_path, monkeypatch):
    import module, stream_transport
    payloads = []
    async def stream(url, payload, timeout, **options):
        payloads.append(payload)
        for piece in ['停","label":"stop_only"', '}', 'never read']:
            yield piece
    monkeypatch.setattr(stream_transport, "text_stream", stream)
    e, _, _ = guarded()
    configure(e, input_route_prefill=True)
    e.demo_cases = CaseArchive(tmp_path)
    e.demo_session_id = "synthetic-route-v2"
    e.text_stream_fn = module.llm_qwen3o_stream
    try:
        state = {}
        messages = [{"role": "system", "content": "input_route"}, route_prefill_message()]
        async with __import__("contextlib").aclosing(e._capacity_stream("input_route", module.llm_qwen3o_stream,
                messages, route=True, response_options={"continue_final_message": True, "max_tokens": 64},
                stream_state=state)) as source:
            async for part in source:
                state["early_closed"] = True
                break
        await e.demo_cases.close()
        cases = [load_case(p) for p in (tmp_path / "captures").iterdir()]
        assert len(cases) == 1 and cases[0]["outcome"]["status"] == "early_closed"
        assert cases[0]["request"] == {**payloads[0], "stream": True}
        assert payloads[0]["continue_final_message"] is True and payloads[0]["messages"][-1]["content"] == ROUTE_PREFIX
    finally:
        await cleanup(e)


def test_compact_prompt_reaches_the_route_request():
    e, m, _ = guarded()
    e.prompts["input_route"] = "规则。\n只输出一个JSON对象，必须先输出transcript字段，再输出label字段，不要其他字段或解释。\n尾。"
    e.prompts["input_route_compact_format"] = "只输出单行紧凑JSON。"
    configure(e, input_route_compact_prompt=True)
    from guarded_turns import route_prompt
    assert route_prompt(e.prompts, e.engine_cfg) == "规则。\n只输出单行紧凑JSON。\n尾。"
    e.prompts.pop("input_route_compact_format")
    with pytest.raises(ValueError):
        e._init_guarded()
