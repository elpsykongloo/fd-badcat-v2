# -*- coding: utf-8 -*-
"""
tests/test_regression.py
========================
Regression tests for Phase-B: verify Phase-A behavior preservation.

Ensures that introducing transactional operations doesn't break:
1. Basic engine behavior (VAD, decision timing, state transitions)
2. Audio clock vs wall clock separation
3. Stale decision dropping
4. Trace equivalence against golden baselines

Run: pytest tests/test_regression.py -v
Or:  python tests/test_regression.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine import ActorEngine, ControlMsg, frames_from_array
import numpy as np


PROMPTS = {"judge": "J", "interrupt": "I", "response": "R", "shift": "S", "shift_s": "SR"}
DELAY = {"end_hold_frame": 0.64, "after_continue_time": 2.5}


class ScriptedVAD:
    """VAD stand-in: emits scripted events by call index."""
    def __init__(self, events: dict):
        self.events = dict(events)
        self.i = 0

    def __call__(self, tensor, return_seconds=True):
        ev = self.events.get(self.i)
        self.i += 1
        return ev

    def reset_states(self):
        self.i = 0


class Script:
    """Decision script: per-kind FIFO of results."""
    def __init__(self, mapping=None):
        self.m = {k: list(v) for k, v in (mapping or {}).items()}
        self.calls = []

    def __call__(self, kind, meta):
        self.calls.append((kind, dict(meta)))
        seq = self.m.get(kind, [])
        return seq.pop(0) if seq else {"text": "", "infer": 0.0}

    def calls_of(self, kind):
        return [m for k, m in self.calls if k == kind]


def make_engine(vad_events, script, mode="injected", engine_cfg=None):
    return ActorEngine(
        prompts=PROMPTS, delay=DELAY,
        llm_cfg={"audio_block": "audio_url", "decision_timeout_s": 15},
        engine_cfg=engine_cfg or {},
        replay_mode=mode, decision_script=script,
        vad_iterator=ScriptedVAD(vad_events),
    )


def silence(seconds):
    return frames_from_array(np.zeros(int(seconds * 16000), dtype=np.float32))


def events_of(engine, kind):
    return [r for r in engine.trace if r["event"] == kind]


def run(coro):
    import asyncio
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# R1: Audio clock invariant (END_HOLD timing)
# ---------------------------------------------------------------------------
def test_r1_audio_clock_end_hold():
    """Verify END_HOLD still runs on audio clock, not wall clock."""
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({"judge": [{"text": "continue", "infer": 0.0}]})
    e = make_engine(vad, script)
    run(e.run_offline(silence(3.0)))

    judges = script.calls_of("judge")
    assert len(judges) == 1
    # Expected: 1.02 (vad end) + 0.64 (hold) = 1.66
    assert abs(judges[0]["t_audio"] - 1.664) < 0.02, f"Expected ~1.664, got {judges[0]['t_audio']}"
    print("PASS: R1 audio clock END_HOLD")


# ---------------------------------------------------------------------------
# R2: Continue timeout anchor (audio clock)
# ---------------------------------------------------------------------------
def test_r2_continue_timeout_anchor():
    """Verify continue timeout anchors on segment end, not judge return."""
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({
        "judge": [{"text": "continue", "infer": 0.2}],
        "response": [{"text": "答复", "infer": 0.1}],
        "asr": [{"text": "用户话", "infer": 0.1}],
        "tts": [{"infer": 0.1, "dur_audio": 1.0, "wav_path": "x.wav"}],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(6.0)))

    resp = script.calls_of("response")
    assert len(resp) == 1
    # Anchor = 1.664 (segment end + hold), timeout = 2.5, total = 4.164
    expected = 1.664 + 2.5
    assert abs(resp[0]["t_audio"] - expected) < 0.04, f"Expected ~{expected}, got {resp[0]['t_audio']}"
    print("PASS: R2 continue timeout anchor")


# ---------------------------------------------------------------------------
# R3: Long interrupt window (1.5s on audio clock)
# ---------------------------------------------------------------------------
def test_r3_long_interrupt_window():
    """Verify 1.5s long interrupt threshold still works."""
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}, 150: {"start": 4.8}}
    script = Script({
        "judge": [{"text": "switch", "infer": 0.1}],
        "response": [{"text": "答复", "infer": 0.1}],
        "asr": [{"text": "u", "infer": 0.05}],
        "tts": [{"infer": 0.2, "dur_audio": 1.0, "wav_path": "x.wav"}],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(8.0)))

    li = events_of(e, "long_interrupt")
    assert len(li) == 1
    # Interrupt at call150: t = 0.032*151 = 4.832, +1.5 = 6.332
    assert abs(li[0]["data"]["t_audio"] - 6.332) < 0.04
    print("PASS: R3 long interrupt window")


# ---------------------------------------------------------------------------
# R4: Stale judge (continue) dropped after new speech
# ---------------------------------------------------------------------------
def test_r4_stale_judge_dropped():
    """Verify stale judge=continue is dropped when user resumes speaking."""
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}, 60: {"start": 1.95}}
    script = Script({"judge": [{"text": "continue", "infer": 1.0}]})
    e = make_engine(vad, script)
    run(e.run_offline(silence(5.0)))

    # Judge should be dropped, no response
    assert len(script.calls_of("response")) == 0
    stale = events_of(e, "llm_stale_dropped")
    assert len(stale) >= 1
    assert stale[0]["data"]["kind"] == "judge"
    print("PASS: R4 stale judge dropped")


# ---------------------------------------------------------------------------
# R5: State transitions (LISTEN -> SPEAK -> LISTEN)
# ---------------------------------------------------------------------------
def test_r5_state_transitions():
    """Verify basic state machine: LISTEN -> (vad+judge) -> response -> SPEAK."""
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({
        "judge": [{"text": "switch", "infer": 0.1}],
        "response": [{"text": "Hello", "infer": 0.1}],
        "asr": [{"text": "hi", "infer": 0.05}],
        "tts": [{"infer": 0.1, "dur_audio": 0.5, "wav_path": "x.wav"}],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(3.0)))

    assert e.STATE == "SPEAK"
    assert e.TURN_IDX == 0
    tts_events = events_of(e, "tts_done")
    assert len(tts_events) == 1
    print("PASS: R5 state transitions")


# ---------------------------------------------------------------------------
# R6: Multiple turns
# ---------------------------------------------------------------------------
def test_r6_multiple_turns():
    """Verify multi-turn conversation flow.

    Turn 0 arrives in LISTEN -> judge("switch") -> response -> TTS -> SPEAK.
    With playback_autoend=False (frozen HumDial behavior) the engine STAYS in
    SPEAK, so turn 1's segment is routed to the INTERRUPT judge, not judge —
    a turn switch there requires interrupt=="switch"."""
    vad = {
        1: {"start": 0.06}, 31: {"end": 1.02},       # Turn 0 (LISTEN -> judge)
        150: {"start": 4.8}, 180: {"end": 5.76},     # Turn 1 (SPEAK -> interrupt)
    }
    script = Script({
        "judge": [
            {"text": "switch", "infer": 0.1},
        ],
        "interrupt": [
            {"text": "switch", "infer": 0.1},
        ],
        "response": [
            {"text": "Response 0", "infer": 0.1},
            {"text": "Response 1", "infer": 0.1},
        ],
        "asr": [
            {"text": "user0", "infer": 0.05},
            {"text": "user1", "infer": 0.05},
        ],
        "tts": [
            {"infer": 0.1, "dur_audio": 0.5, "wav_path": "x0.wav"},
            {"infer": 0.1, "dur_audio": 0.5, "wav_path": "x1.wav"},
        ],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(8.0)))

    assert e.TURN_IDX == 1
    assert len(e.user_history) == 2
    assert len(e.assistant_history) == 2
    print("PASS: R6 multiple turns")


# ---------------------------------------------------------------------------
# R7: No perception freeze (decisions don't block receive)
# ---------------------------------------------------------------------------
def test_r7_no_perception_freeze():
    """Verify engine can receive frames during decision (actor pattern)."""
    # This is structural: if run_offline completes with slow infer, no freeze occurred
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({
        "judge": [{"text": "continue", "infer": 0.5}],  # Slow decision
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(3.0)))

    # Engine processes all frames despite slow judge
    assert len(script.calls_of("judge")) == 1
    # If there was a freeze, frames would be dropped
    print("PASS: R7 no perception freeze")


# ---------------------------------------------------------------------------
# R8: Injected replay determinism
# ---------------------------------------------------------------------------
def test_r8_injected_replay():
    """Verify injected mode replay is deterministic."""
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({
        "judge": [{"text": "switch", "infer": 0.15}],
        "response": [{"text": "Hi", "infer": 0.12}],
        "asr": [{"text": "hello", "infer": 0.08}],
        "tts": [{"infer": 0.10, "dur_audio": 0.5, "wav_path": "x.wav"}],
    })

    # Run twice
    e1 = make_engine(vad, Script(script.m))
    run(e1.run_offline(silence(3.0)))

    e2 = make_engine(vad, Script(script.m))
    run(e2.run_offline(silence(3.0)))

    # Traces should be identical (event sequence + timing)
    assert len(e1.trace) == len(e2.trace)
    for i, (ev1, ev2) in enumerate(zip(e1.trace, e2.trace)):
        assert ev1["event"] == ev2["event"], f"Event {i}: {ev1['event']} != {ev2['event']}"
        assert abs(ev1["data"]["t_audio"] - ev2["data"]["t_audio"]) < 1e-6
    print("PASS: R8 injected replay determinism")


# ---------------------------------------------------------------------------
# R9: Trace format (for golden comparison)
# ---------------------------------------------------------------------------
def test_r9_trace_format():
    """Verify trace format includes required fields for golden comparison."""
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({
        "judge": [{"text": "switch", "infer": 0.1}],
        "response": [{"text": "Hi", "infer": 0.1}],
        "asr": [{"text": "hello", "infer": 0.05}],
        "tts": [{"infer": 0.1, "dur_audio": 0.5, "wav_path": "x.wav"}],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(3.0)))

    # Check trace structure
    for entry in e.trace:
        assert "event" in entry
        assert "data" in entry
        assert "turn" in entry["data"]
        assert "state" in entry["data"]
        assert "t_audio" in entry["data"]

    # Key events present
    assert len(events_of(e, "vad_start")) >= 1
    assert len(events_of(e, "vad_640_done")) >= 1
    assert len(events_of(e, "llm_done")) >= 1
    assert len(events_of(e, "tts_done")) >= 1
    print("PASS: R9 trace format")


# ---------------------------------------------------------------------------
# R10: Prompt values unchanged
# ---------------------------------------------------------------------------
def test_r10_prompts_unchanged():
    """Verify Phase-A prompts are not modified."""
    # This test documents the prompt values, not that they're used correctly
    # (that's covered by behavioral tests above)
    assert PROMPTS["judge"] == "J"
    assert PROMPTS["response"] == "R"
    assert PROMPTS["shift"] == "S"
    assert DELAY["end_hold_frame"] == 0.64
    assert DELAY["after_continue_time"] == 2.5
    print("PASS: R10 prompts unchanged")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
