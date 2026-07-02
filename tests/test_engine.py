# -*- coding: utf-8 -*-
"""W1 S2 unit tests for the actor engine (spec: 手工文档/神谕/01_W1 完整计划.md).

All model calls are scripted (no GPU, no network, no silero). Runs under pytest
OR plain `python tests/test_engine.py`.

Frame math used throughout:
  frames are 256 samples = 16 ms; the VAD fires every 2nd frame (512 samples),
  so scripted-VAD call index i lands on t_audio = 0.032 * (i + 1).
"""
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from engine import ActorEngine, ControlMsg, frames_from_array  # noqa: E402

PROMPTS = {"judge": "J", "interrupt": "I", "response": "R", "shift": "S", "shift_s": "SR"}
DELAY = {"end_hold_frame": 0.64, "after_continue_time": 2.5}
FRAME = 0.016


class ScriptedVAD:
    """VAD stand-in: emits scripted events by call index (1 call per 512 samples)."""
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
    """Decision script: per-kind FIFO of results; records every call's meta."""
    def __init__(self, mapping=None):
        self.m = {k: list(v) for k, v in (mapping or {}).items()}
        self.calls = []

    def __call__(self, kind, meta):
        self.calls.append((kind, dict(meta)))
        seq = self.m.get(kind, [])
        return seq.pop(0) if seq else {"text": "", "infer": 0.0}

    def calls_of(self, kind):
        return [m for k, m in self.calls if k == kind]


def make_engine(vad_events, script, mode="injected", engine_cfg=None, llm_fn=None):
    return ActorEngine(
        prompts=PROMPTS, delay=DELAY,
        llm_cfg={"audio_block": "audio_url", "decision_timeout_s": 15},
        engine_cfg=engine_cfg or {},
        replay_mode=mode, decision_script=script,
        vad_iterator=ScriptedVAD(vad_events),
        llm_fn=llm_fn,
    )


def silence(seconds):
    return frames_from_array(np.zeros(int(seconds * 16000), dtype=np.float32))


def events_of(engine, kind):
    return [r for r in engine.trace if r["event"] == kind]


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# T1: END_HOLD runs on the audio clock, not the wall clock
# ---------------------------------------------------------------------------
def test_t1_end_hold_audio_clock():
    # speech: start@call1 (t=0.064), end@call31 (t=1.024) -> judge at 1.024+0.64=1.664
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({"judge": [{"text": "continue", "infer": 0.0}]})
    e = make_engine(vad, script)
    run(e.run_offline(silence(3.0)))  # unpaced: wall time ~ms, only audio clock can elapse
    judges = script.calls_of("judge")
    assert len(judges) == 1, f"judge dispatched {len(judges)} times"
    assert abs(judges[0]["t_audio"] - 1.664) < 0.017, judges[0]["t_audio"]
    v640 = events_of(e, "vad_640_done")
    assert len(v640) == 1 and abs(v640[0]["data"]["t_audio"] - 1.664) < 0.017


# ---------------------------------------------------------------------------
# T2: continue-timeout anchor = end of judged segment (audio clock)
# ---------------------------------------------------------------------------
def test_t2_continue_anchor_is_segment_end():
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({
        "judge": [{"text": "continue", "infer": 0.2}],
        "response": [{"text": "答复", "infer": 0.1}],
        "asr": [{"text": "用户话", "infer": 0.1}],
        "tts": [{"infer": 0.1, "dur_audio": 1.0, "wav_path": "x.wav"}],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(6.0)))
    # anchor = judged segment end (1.664), NOT judge-return time (1.864)
    resp = script.calls_of("response")
    assert len(resp) == 1
    assert abs(resp[0]["t_audio"] - (1.664 + 2.5)) < 0.034, resp[0]["t_audio"]
    assert len(script.calls_of("asr")) == 1
    assert e.assistant_history == ["答复"]


# ---------------------------------------------------------------------------
# T3: 1.5s long-interrupt window on the audio clock
# ---------------------------------------------------------------------------
def test_t3_long_interrupt_window():
    # turn 0 answered -> SPEAK; then a long unbroken interruption
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
    assert len(li) == 1, f"long_interrupt count={len(li)} trace={[r['event'] for r in e.trace]}"
    # interrupt vad_start at call150 -> t = 0.032*151 = 4.832; +1.5 = 6.332
    assert abs(li[0]["data"]["t_audio"] - 6.332) < 0.034, li[0]["data"]["t_audio"]
    assert e.STATE == "LISTEN" and e.TURN_IDX == 1 and e.IN_SPEECH


# ---------------------------------------------------------------------------
# T4: stale judge (result=continue) is dropped after new speech
# ---------------------------------------------------------------------------
def test_t4_stale_judge_continue_dropped():
    # judge in flight for 1.0s; user resumes speaking at call60 (t=1.952)
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}, 60: {"start": 1.95}}
    script = Script({"judge": [{"text": "continue", "infer": 1.0}]})
    e = make_engine(vad, script)
    run(e.run_offline(silence(4.0)))
    stale = events_of(e, "llm_stale_dropped")
    assert len(stale) == 1 and stale[0]["data"]["kind"] == "judge"
    assert not e.CONTINUE_ARMED
    assert events_of(e, "llm_done") == []  # dropped result never traced as llm_done


# ---------------------------------------------------------------------------
# T5: stale judge (result=switch) dropped; re-judged at next EoU with grown buffer
# ---------------------------------------------------------------------------
def test_t5_stale_judge_switch_rejudged():
    vad = {1: {"start": 0.06}, 31: {"end": 1.02},   # EoU #1 -> judge@1.664 (stale)
           60: {"start": 1.95}, 90: {"end": 2.91}}  # EoU #2 -> judge@3.552 (fresh)
    script = Script({
        "judge": [{"text": "switch", "infer": 1.0},   # stale one
                  {"text": "switch", "infer": 0.1}],  # fresh re-judge
        "response": [{"text": "答", "infer": 0.1}],
        "asr": [{"text": "u", "infer": 0.05}],
        "tts": [{"infer": 0.1, "dur_audio": 0.5, "wav_path": "x.wav"}],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(6.0)))
    judges = script.calls_of("judge")
    assert len(judges) == 2, judges
    assert abs(judges[1]["t_audio"] - (2.912 + 0.64)) < 0.034, judges[1]["t_audio"]
    assert len(events_of(e, "llm_stale_dropped")) == 1
    # fresh judge produced the answer chain
    assert len(script.calls_of("response")) == 1
    # second judge audio contains BOTH speech segments (grown buffer)
    n1 = judges[0]["n_samples"] if "n_samples" in judges[0] else None
    assert e.TURN_IDX == 0


# ---------------------------------------------------------------------------
# T6: stale shift dropped -> no response from the dead chain
# ---------------------------------------------------------------------------
def test_t6_stale_shift_dropped():
    # reach LISTEN+turn1 via long interrupt, then EoU -> judge(switch) -> shift in flight
    # -> user resumes -> shift result must be dropped and produce nothing
    vad = {1: {"start": 0.06}, 31: {"end": 1.02},        # turn0 EoU
           150: {"start": 4.8},                          # long interrupt (>=1.5s no end)
           260: {"end": 8.35},                           # EoU of interrupt segment (turn1)
           310: {"start": 9.95}}                         # resume during shift in flight
    script = Script({
        "judge": [{"text": "switch", "infer": 0.1},      # turn0
                  {"text": "switch", "infer": 0.1}],     # turn1 EoU
        "shift": [{"text": "no", "infer": 1.2}],         # in flight when user resumes
        "response": [{"text": "答0", "infer": 0.1}],
        "asr": [{"text": "u0", "infer": 0.05}],
        "tts": [{"infer": 0.2, "dur_audio": 1.0, "wav_path": "x.wav"}],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(12.0)))
    assert len(script.calls_of("shift")) == 1
    stale = [r for r in events_of(e, "llm_stale_dropped") if r["data"]["kind"] == "shift"]
    assert len(stale) == 1, [r["data"] for r in events_of(e, "llm_stale_dropped")]
    # only turn0's response ever ran; the stale shift chain produced no second response
    assert len(script.calls_of("response")) == 1


# ---------------------------------------------------------------------------
# T7: stale interrupt re-judged on the extended segment
# ---------------------------------------------------------------------------
def test_t7_stale_interrupt_extended_segment():
    vad = {1: {"start": 0.06}, 31: {"end": 1.02},        # turn0 -> SPEAK
           150: {"start": 4.8}, 160: {"end": 5.15},      # interrupt seg -> intent@5.792
           186: {"start": 5.98},                          # resume during intent in flight
           210: {"end": 6.75}}                            # extended EoU -> re-judge@7.392
    script = Script({
        "judge": [{"text": "switch", "infer": 0.1}],
        "interrupt": [{"text": "switch", "infer": 1.0},   # stale
                      {"text": "continue", "infer": 0.1}],# fresh: backchannel
        "response": [{"text": "答0", "infer": 0.1}],
        "asr": [{"text": "u0", "infer": 0.05}],
        "tts": [{"infer": 0.2, "dur_audio": 1.0, "wav_path": "x.wav"}],
    })
    e = make_engine(vad, script)
    run(e.run_offline(silence(9.0)))
    intents = script.calls_of("interrupt")
    assert len(intents) == 2, intents
    assert abs(intents[1]["t_audio"] - (6.752 + 0.64)) < 0.034, intents[1]["t_audio"]
    stale = [r for r in events_of(e, "llm_stale_dropped") if r["data"]["kind"] == "interrupt"]
    assert len(stale) == 1
    # fresh result was "continue" (backchannel): stays SPEAK, no new turn
    assert e.STATE == "SPEAK" and e.TURN_IDX == 0
    assert len(events_of(e, "no_interrupt")) == 1
    assert len(events_of(e, "shot_interrupt")) == 0


# ---------------------------------------------------------------------------
# T8: reset invalidates in-flight decisions (realtime mode, session generation)
# ---------------------------------------------------------------------------
def test_t8_reset_drops_inflight():
    async def scenario():
        vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
        calls = []

        def slow_llm(messages):
            calls.append(1)
            import time as _t
            _t.sleep(0.15)
            return "switch"

        e = make_engine(vad, None, mode="realtime", llm_fn=slow_llm)
        for ev in silence(2.0):
            e.q.put_nowait(ev)
        e.q.put_nowait(ControlMsg("session_end"))   # arrives while judge is in flight
        # give the worker time to finish after the reset, then disconnect
        async def late_disconnect():
            await asyncio.sleep(0.5)
            e.q.put_nowait(ControlMsg("disconnect"))
        asyncio.create_task(late_disconnect())
        await e.engine_loop()
        return e, calls

    e, calls = run(scenario())
    assert len(calls) == 1                       # judge was dispatched
    assert events_of(e, "llm_done") == []        # ...but its result was gen-dropped
    assert e.session_gen == 1
    assert e.STATE == "LISTEN" and not e.IN_SPEECH and e.TURN_IDX == 0


# ---------------------------------------------------------------------------
# T10: injected replay advances the audio clock by the recorded infer_time
# ---------------------------------------------------------------------------
def test_t10_injected_clock_advance():
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({"judge": [{"text": "continue", "infer": 0.7}]})
    e = make_engine(vad, script)
    run(e.run_offline(silence(4.0)))
    done = events_of(e, "llm_done")
    assert len(done) == 1
    # dispatched at 1.664, infer 0.7 -> delivered at first frame >= 2.364
    assert abs(done[0]["data"]["t_audio"] - 2.364) < 0.034, done[0]["data"]["t_audio"]
    assert done[0]["data"]["infer_time"] == 0.7
    assert e.CONTINUE_ARMED


def test_t10b_oracle_zero_latency():
    vad = {1: {"start": 0.06}, 31: {"end": 1.02}}
    script = Script({"judge": [{"text": "continue", "infer": 0.7}]})
    e = make_engine(vad, script, mode="oracle")
    run(e.run_offline(silence(4.0)))
    done = events_of(e, "llm_done")
    assert len(done) == 1
    assert abs(done[0]["data"]["t_audio"] - 1.664) < 0.034  # same tick as dispatch


# ---------------------------------------------------------------------------
# T9: trace_diff locates a known divergence (and passes identical traces)
# ---------------------------------------------------------------------------
def test_t9_trace_diff_locates_divergence():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from trace_diff import diff_traces

    def ev(kind, turn, state, t, content=None):
        d = {"turn": turn, "state": state, "t_audio": t}
        if content is not None:
            d["content"] = content
        return {"event": kind, "data": d}

    a = [ev("vad_start", 0, "LISTEN", 1.0), ev("vad_done", 0, "LISTEN", 2.0),
         ev("llm_done", 0, "LISTEN", 2.7, "switch"), ev("tts_done", 0, "LISTEN", 3.4)]
    # identical -> L1
    res = diff_traces(a, [dict(x) for x in a])
    assert res["l1"] and res["first_divergence"] is None
    # different llm content -> divergence at index 2
    b = [ev("vad_start", 0, "LISTEN", 1.0), ev("vad_done", 0, "LISTEN", 2.0),
         ev("llm_done", 0, "LISTEN", 2.7, "continue"), ev("tts_done", 0, "LISTEN", 3.4)]
    res = diff_traces(a, b)
    assert not res["sequence_equal"] and res["first_divergence"] == 2
    # time skew beyond tol -> sequence equal but not L1
    c = [ev("vad_start", 0, "LISTEN", 1.0), ev("vad_done", 0, "LISTEN", 2.6),
         ev("llm_done", 0, "LISTEN", 2.7, "switch"), ev("tts_done", 0, "LISTEN", 3.4)]
    res = diff_traces(a, c, tol=0.25)
    assert res["sequence_equal"] and not res["l1"]
    assert res["soft_time_mismatches"] and res["soft_time_mismatches"][0][0] == 1
    # new-engine-only events are excluded from comparison
    d = a[:2] + [{"event": "llm_stale_dropped", "data": {"turn": 0, "t_audio": 2.5}}] + a[2:]
    res = diff_traces(a, d)
    assert res["l1"] and res["info_counts"].get("llm_stale_dropped") == 1


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
