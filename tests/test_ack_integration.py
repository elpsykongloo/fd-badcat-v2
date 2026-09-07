#!/usr/bin/env python3
"""
Integration test for ack-v0 on the Phase-B v1 engine (W3 D1 API).

Drives TactEngineWithAck through the REAL engine loop: dispatch a transactional
decision, let the worker + queue + handlers run, then inspect the TTS artifacts.

Tests:
1. ack-v0 is triggered for appropriate responses (two-phase files produced)
2. ack-v0 disabled -> baseline single-phase TTS
3. ack-v0 skipped for short responses (<= 8 words)
4. First-response latency is reduced vs baseline (pure tts_ack, no engine)
"""

import asyncio
import json
import sys
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf

from engine import ControlMsg
from engine_b_ack import TactEngineWithAck


# ---------------------------------------------------------------------------
# mocks
# ---------------------------------------------------------------------------
def mock_llm(messages):
    """Mock LLM returns a launch(+commit) decision with a long say."""
    return json.dumps({
        "dialogue": "speak",
        "ops": [
            {"type": "launch", "fn": "search_flights",
             "args": {"destination": "NYC", "date": "July 15"}},
            {"type": "commit", "op_id": 1}
        ],
        "say": "I've searched for flights to New York on July 15th and found several options."
    })


def mock_asr(path):
    return "find flights to New York on July 15th"


def mock_tts(text: str, path):
    """Write a tiny valid WAV; timing is tested with a virtual clock below."""
    samples = max(160, len(text.split()) * 160)
    audio = np.zeros(samples, dtype=np.float32)
    sf.write(str(path), audio, 16000, subtype='PCM_16')
    return str(path)


def mock_tool_executor(fn, args):
    return {"status": "success", "fn": fn, "args": args}


class _NoVAD:
    """Perception stub: the tests dispatch decisions directly, no frames flow."""
    def __call__(self, *a, **k):
        return None

    def reset_states(self):
        pass


def _mk_engine(tmpdir, engine_cfg, llm=mock_llm):
    cfg = {"phase": "b", "mode": "blocking", "tool_sync": True, **engine_cfg}
    eng = TactEngineWithAck(
        websocket=None,
        prompts={},
        delay={"end_hold_frame": 0.64, "after_continue_time": 2.5},
        llm_cfg={"audio_block": "audio_url", "decision_timeout_s": 30},
        engine_cfg=cfg,
        llm_fn=llm, asr_fn=mock_asr, tts_fn=mock_tts,
        vad_iterator=_NoVAD(),
        tool_executor=mock_tool_executor,
    )
    eng.output_dir = Path(tmpdir)
    return eng


async def _drive(engine):
    """Run one decision and stop as soon as its worker/queue graph is idle."""
    loop_task = asyncio.create_task(engine.engine_loop())
    engine.t_audio = 1.0
    engine.t_end_anchor = 1.0
    engine._ledger_t = 1.0
    engine._session_frames = [np.zeros(16000, dtype=np.float32)]
    engine.dispatch_tact_decision(t_eou=1.0, turn=0)

    async def wait_until_idle():
        while engine._inflight or not engine.q.empty():
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(wait_until_idle(), timeout=5.0)
    finally:
        engine.q.put_nowait(ControlMsg("disconnect"))
        await asyncio.wait_for(loop_task, timeout=5.0)


# ---------------------------------------------------------------------------
async def test_ack_enabled(tmp_path):
    engine = _mk_engine(tmp_path, {"ack_enabled": True,
                                   "ack_strategy": "context", "ack_seed": 42})
    await _drive(engine)
    assert engine.tx.committed, "decision did not commit the launched op"
    assert len(list(tmp_path.glob("*_ack.wav"))) == 1
    assert len(list(tmp_path.glob("*_main.wav"))) == 1


async def test_ack_disabled(tmp_path):
    engine = _mk_engine(tmp_path, {"ack_enabled": False})
    await _drive(engine)
    assert not list(tmp_path.glob("*_ack.wav"))
    assert len(list(tmp_path.glob("turn0_tts.wav"))) == 1


async def test_short_response_skip(tmp_path):
    def mock_llm_short(messages):
        return json.dumps({
            "dialogue": "speak",
            "ops": [{"type": "launch", "fn": "search_flights",
                     "args": {"destination": "NYC", "date": "July 15"}}],
            "say": "Okay, searching now."  # 3 words
        })

    engine = _mk_engine(tmp_path, {"ack_enabled": True}, llm=mock_llm_short)
    await _drive(engine)
    assert not list(tmp_path.glob("*_ack.wav"))
    assert len(list(tmp_path.glob("turn0_tts.wav"))) == 1


async def test_latency_improvement(tmp_path, monkeypatch):
    """Compare the two synthesis paths without waiting on wall-clock sleeps."""
    say_text = ("I've searched for flights to New York on July 15th "
                "and found several options.")

    class VirtualClock:
        now = 0.0

        def __call__(self):
            return self.now

        def advance_for(self, text):
            self.now += len(text.split()) * 0.02

    clock = VirtualClock()

    def timed_tts(text, path):
        clock.advance_for(text)
        return mock_tts(text, path)

    import tts_ack
    monkeypatch.setattr(tts_ack, "time", SimpleNamespace(perf_counter=clock))
    baseline_path, baseline_lat = await tts_ack.synthesize_baseline(
        say_text, timed_tts, tmp_path, turn=0)
    ack_path, main_path, ack_lat, _main_lat, _total_lat = \
        await tts_ack.synthesize_with_ack(
            say_text, timed_tts, tmp_path, turn=1,
            ops=[{"type": "launch", "fn": "search_flights"}],
            strategy="context", seed=42)

    improvement_pct = ((baseline_lat - ack_lat) / baseline_lat) * 100
    assert improvement_pct > 20
    assert all(Path(p).exists() for p in (baseline_path, ack_path, main_path))
