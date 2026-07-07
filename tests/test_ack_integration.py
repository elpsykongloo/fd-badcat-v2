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
import tempfile
import time
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
    """Mock TTS generates silent audio proportional to text length."""
    time.sleep(len(text.split()) * 0.02)   # 20ms per word
    duration = len(text.split()) * 0.15    # 150ms per word
    samples = int(16000 * duration)
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


async def _drive(engine, settle=2.0):
    """Run the engine loop, inject one EoU decision, let it settle, shut down."""
    loop_task = asyncio.create_task(engine.engine_loop())
    engine.t_audio = 1.0
    engine.t_end_anchor = 1.0
    engine._ledger_t = 1.0
    engine._session_frames = [np.zeros(16000, dtype=np.float32)]
    engine.dispatch_tact_decision(t_eou=1.0, turn=0)
    await asyncio.sleep(settle)
    engine.q.put_nowait(ControlMsg("disconnect"))
    await asyncio.wait_for(loop_task, timeout=10.0)


# ---------------------------------------------------------------------------
async def test_ack_enabled():
    print("\n[Test 1] ack-v0 enabled for appropriate responses")
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = _mk_engine(tmpdir, {"ack_enabled": True,
                                     "ack_strategy": "context", "ack_seed": 42})
        await _drive(engine)
        ack_files = list(Path(tmpdir).glob("*_ack.wav"))
        main_files = list(Path(tmpdir).glob("*_main.wav"))
        assert engine.tx.committed, "decision did not commit the launched op"
        if ack_files and main_files:
            print(f"  ✓ ack-v0 triggered: {len(ack_files)} ack + {len(main_files)} main files")
            return True
        print(f"  ✗ ack-v0 NOT triggered (ack:{len(ack_files)}, main:{len(main_files)})")
        return False


async def test_ack_disabled():
    print("\n[Test 2] ack-v0 disabled when flag is false")
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = _mk_engine(tmpdir, {"ack_enabled": False})
        await _drive(engine)
        ack_files = list(Path(tmpdir).glob("*_ack.wav"))
        baseline_files = list(Path(tmpdir).glob("turn0_tts.wav"))
        if not ack_files and baseline_files:
            print("  ✓ ack-v0 correctly disabled: baseline TTS used")
            return True
        print(f"  ✗ Unexpected behavior (ack:{len(ack_files)}, baseline:{len(baseline_files)})")
        return False


async def test_short_response_skip():
    print("\n[Test 3] ack-v0 skipped for short responses (<= 8 words)")

    def mock_llm_short(messages):
        return json.dumps({
            "dialogue": "speak",
            "ops": [{"type": "launch", "fn": "search_flights",
                     "args": {"destination": "NYC", "date": "July 15"}}],
            "say": "Okay, searching now."  # 3 words
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = _mk_engine(tmpdir, {"ack_enabled": True}, llm=mock_llm_short)
        await _drive(engine)
        ack_files = list(Path(tmpdir).glob("*_ack.wav"))
        baseline_files = list(Path(tmpdir).glob("turn0_tts.wav"))
        if not ack_files and baseline_files:
            print("  ✓ ack-v0 correctly skipped for short response")
            return True
        print(f"  ✗ ack:{len(ack_files)} baseline:{len(baseline_files)} (want 0 / >0)")
        return False


async def test_latency_improvement():
    print("\n[Test 4] First-response latency improvement")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        say_text = ("I've searched for flights to New York on July 15th "
                    "and found several options.")

        from tts_ack import synthesize_baseline, synthesize_with_ack
        baseline_path, baseline_lat = await synthesize_baseline(
            say_text, mock_tts, tmpdir, turn=0)
        ack_path, main_path, ack_lat, main_lat, total_lat = await synthesize_with_ack(
            say_text, mock_tts, tmpdir, turn=1,
            ops=[{"type": "launch", "fn": "search_flights"}],
            strategy="context", seed=42)

        improvement = baseline_lat - ack_lat
        improvement_pct = (improvement / baseline_lat) * 100
        print(f"  Baseline first-response: {baseline_lat:.3f}s")
        print(f"  ack-v0 first-response: {ack_lat:.3f}s")
        print(f"  Improvement: {improvement:.3f}s ({improvement_pct:.1f}%)")
        if improvement > 0 and improvement_pct > 20:
            print("  ✓ Significant improvement achieved")
            return True
        print("  ✗ Improvement insufficient")
        return False


async def main():
    print("=" * 70)
    print("ack-v0 Integration Test Suite (Phase-B v1)")
    print("=" * 70)

    results = []
    for test in (test_ack_enabled, test_ack_disabled,
                 test_short_response_skip, test_latency_improvement):
        try:
            results.append(await test())
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ✗ Test failed with error: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 70)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
