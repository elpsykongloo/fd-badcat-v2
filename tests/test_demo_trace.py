"""Diagnostic clocks, persistence, failure isolation, and policy equivalence."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demo_trace import DemoTrace


async def test_three_stage_clock_and_no_cross_epoch_attribution(tmp_path):
    clock = [10.0]
    trace = DemoTrace(tmp_path / "events.jsonl", clock=lambda: clock[0])
    context = {"generation": 0, "epoch": 1}
    def event(at, name, **data):
        clock[0] = 10 + at
        return trace.observe(name, data, **context)
    event(0, "vad_done")
    event(.641, "vad_640_done")
    event(1.004, "speech_start", utterance_id=1)
    summary = event(1.522, "speech_first_audio", utterance_id=1)
    assert summary == {"utterance_id": 1, "hold_ms": 641, "decision_ms": 363,
                       "generation_ms": 518, "vad_to_audio_ms": 1522}
    context["epoch"] = 2
    event(2, "speech_start", utterance_id=2)
    summary = event(2.5, "speech_first_audio", utterance_id=2)
    assert summary["generation_ms"] == 500
    assert summary["hold_ms"] is summary["decision_ms"] is summary["vad_to_audio_ms"] is None
    event(3, "speech_start", utterance_id=3)
    event(3.1, "speech_cancelled", utterance_id=3)
    assert event(3.2, "speech_first_audio", utterance_id=3) is None
    context.update(generation=1, epoch=1)
    event(4, "speech_start", utterance_id=4)
    assert event(4.5, "speech_first_audio", utterance_id=4)["vad_to_audio_ms"] is None
    await trace.close()
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert rows[-1] == {"event": "trace_closed", "dropped": 0}
    assert rows[1]["server_ms"] == 0


async def test_client_fields_rate_bound_and_independent_clock(tmp_path):
    trace = DemoTrace(tmp_path / "events.jsonl", clock=lambda: 100)
    clean = trace.client({"kind": "first_audio", "client_ms": 999999,
                          "first_audio_ms": 449, "output_latency_ms": float("nan"),
                          "upload_buffer_bytes": -1, "prompt": "never save me"})
    assert clean == {"kind": "first_audio", "client_ms": 999999, "first_audio_ms": 449}
    assert trace.client({"kind": "ping", "seq": 1})["seq"] == 1
    assert trace.client({"kind": "rtt", "rtt_ms": 2})["rtt_ms"] == 2  # burst allowed
    assert trace.client({"kind": "ping", "seq": "bad"}) is None
    assert trace.client(["bad"]) is None
    assert trace.client({"kind": []}) is None
    assert trace.client({"kind": "rtt", "rtt_ms": 10 ** 500}) == {"kind": "rtt"}
    assert sum(trace.client({"kind": "stop"}) is not None for _ in range(100)) <= 20
    await trace.close()
    text = trace.path.read_text()
    assert "never save me" not in text and "NaN" not in text


async def test_disk_failure_is_visible_and_bounded_not_a_conversation_failure(tmp_path):
    path = tmp_path / "missing" / "events.jsonl"
    trace = DemoTrace(path, capacity=2)
    await asyncio.to_thread(trace.thread.join, 1)
    assert trace.error == "FileNotFoundError"
    for _ in range(10):
        trace.record("test")
    assert trace.queue.qsize() <= 2 and trace.dropped > 0
    await trace.close()


@pytest.mark.parametrize("decision", ["continue", "yes"])
async def test_observer_does_not_change_actor_dispatch_or_history(tmp_path, decision):
    from test_engine import make_engine, Script, silence
    async def run(enabled):
        script = Script({"judge": [{"text": decision, "infer": .3}],
                         "response": [{"text": "你好", "infer": .2}]})
        engine = make_engine({1: {"start": .06}, 31: {"end": 1.02}}, script)
        if enabled:
            engine.demo_trace = DemoTrace(tmp_path / "events.jsonl")
        try:
            await engine.run_offline(silence(5))
        finally:
            if enabled:
                await engine.demo_trace.close()
        return script.calls, engine.STATE, engine.user_history, engine.assistant_history
    assert await run(False) == await run(True)


async def test_socket_records_first_send_and_slow_queue_without_pcm_payload(tmp_path):
    from speech_stream import PCM_HEADER
    trace = DemoTrace(tmp_path / "events.jsonl")
    payload = PCM_HEADER.pack(b"FDS1", 7, 0, 24000) + bytes(1920)
    trace.sent(payload, 10, 10.05, 10.07)
    trace.sent("control", 11, 11, 11.001)
    await trace.close()
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    records = [r for r in rows if r["event"].startswith("socket_")]
    assert len(records) == 1
    assert records[0]["data"] == {"queue_ms": 50, "send_ms": 20, "utterance_id": 7, "packet_seq": 0}
