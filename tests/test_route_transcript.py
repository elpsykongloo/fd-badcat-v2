"""One-call transcript-first routing contract, including fail-closed repairs."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from control_labels import parse_route, parse_label, decide_control, ROUTE_PROTOCOL


@pytest.mark.parametrize("raw", [
    'keep', '{"label":"stop_only","transcript":"停"}',
    '{"transcript":"停","label":"stop_only","label":"keep"}',
    '{"transcript":"停","label":"stop_only","extra":1}',
    '{"transcript":"","label":"stop_only"}',
    '{"transcript":"  ","label":"yield_ready"}',
    '{"transcript":null,"label":"keep"}',
    '{"transcript":{},"label":"keep"}',
    '{"transcript":"停","label":true}',
    '{"transcript":"停","label":"STOP_ONLY"}',
    '{"transcript":"停","label":"stop_only"',
    '[["transcript","停"],["label","stop_only"]]',
    'prefix {"transcript":"停","label":"stop_only"}',
    '{"transcript":"停","label":"stop_only"} trailing',
    json.dumps({"transcript":"a"*513,"label":"yield_ready"}),
])
def test_route_rejects_invalid_or_unsupported_evidence(raw):
    assert parse_route(raw) is None
    assert parse_label("input_route",raw) is None


@pytest.mark.parametrize("label", ["keep","stop_only","yield_wait","yield_ready"])
def test_transcript_cannot_override_label(label):
    raw=json.dumps({"transcript":"keep stop_only yield_wait yield_ready", "label":label})
    assert parse_label("input_route",raw)==label
    assert parse_label("input_route",'```json\n'+raw+'\n```')==label
    assert parse_label("input_route",label,legacy_route=True)==label


async def test_valid_result_uses_one_call_and_does_not_mutate_messages():
    requests=[]
    async def call(messages):
        requests.append(messages)
        return '{"transcript":"你是谁","label":"yield_ready"}'
    original=[{"role":"user","content":"audio placeholder"}]
    label,audit=await decide_control(call,original,"input_route",2)
    assert label=="yield_ready" and len(requests)==1
    assert audit["transcript"]=="你是谁" and audit["protocol"]==ROUTE_PROTOCOL
    assert len(original)==1 and not audit["repaired"] and not audit["fallback"]


async def test_repair_does_not_echo_invalid_transcription_as_evidence():
    requests=[]
    async def call(messages):
        requests.append(messages)
        return 'invented stop' if len(requests)==1 else '{"transcript":"","label":"keep"}'
    original=[{"role":"user","content":"audio placeholder"}]
    label,audit=await decide_control(call,original,"input_route",2)
    assert label=="keep" and audit["repaired"] and len(requests)==2
    assert requests[1][:-1]==original
    assert 'invented stop' not in str(requests[1]) and 'JSON' in requests[1][-1]['content']


async def test_invalid_or_timeout_never_implicitly_stops():
    async def invalid(_):return '{"transcript":"","label":"stop_only"}'
    label,audit=await decide_control(invalid,[],"input_route",2)
    assert label=="keep" and audit["fallback"] and audit["attempts"]==2
    async def timeout(_):raise asyncio.TimeoutError
    label,audit=await decide_control(timeout,[],"input_route",2)
    assert label=="keep" and audit["timed_out"] and audit["attempts"]==1
