"""
tests/test_commit_barrier.py
============================
W3 D1 unit tests for the unified TACT semantics (src/tact_core.py) and the
Phase-B v1 engine (src/engine_b.py TactEngine).

Pure CPU: no model, no network, no wav files, no real VAD (a scripted iterator
is injected). Run:  /root/miniconda3/envs/fd-sds/bin/python tests/test_commit_barrier.py

Covers (engine_b spec / 06 §一 clauses):
  1. silence-budget law: speech freezes the countdown, gaps burn it, nominal
     deadline stamps are exact
  2. commit barrier: expiry inside a decision's flight window defers; patch
     rescues (window restarts with a full budget); unrescued defers commit at
     their NOMINAL stamp at sweep (dual stamp: deferred_s > 0)
  3. continuous-clock ablation (barrier off): expiries commit mid-flight; the
     late patch is dropped and logged (patch_after_commit)
  4. release paths: stale / timeout sweeps leave nothing deferred
  5. delta<=0 / blocking-mode immediate commits
  6. launch dedup, nested-args unwrap, schema coercion
  7. TactEngine offline smoke (injected replay, scripted VAD + decisions):
     EoU -> launch -> tail expiry commit; barrier rescue through the engine loop
"""
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/root/autodl-tmp")

import numpy as np  # noqa: E402

import tact_core  # noqa: E402
from tact_core import (WindowLedger, apply_decision_ops, advance_over,  # noqa: E402
                       coerce_args, silent_spans)
from tact.transaction import Transaction  # noqa: E402

PASS = []


def ok(label, cond):
    assert cond, f"FAIL: {label}"
    PASS.append(label)
    print(f"  ok - {label}")


class Recorder:
    """Commit sink: records (op_id, nominal, actual) and commits into tx."""
    def __init__(self, tx):
        self.tx = tx
        self.commits = []

    def __call__(self, op_id, t_nominal, t_actual):
        if op_id not in self.tx.pending:
            return
        self.tx.commit(op_id, lambda fn, args: {"status": "success"}, t=t_nominal)
        self.commits.append({"op_id": op_id, "nominal": round(t_nominal, 3),
                             "actual": round(t_actual, 3)})


def launch(tx, ledger, fn="search_flights", args=None, t=0.0):
    dec = {"ops": [{"type": "launch", "fn": fn,
                    "args": args or {"destination": "NYC", "date": "July 15"}}],
           "say": "x"}
    rec = Recorder(tx)
    applied = apply_decision_ops(tx, ledger, dec, t, immediate=False, commit_cb=rec)
    assert applied and applied[0]["type"] == "launch"
    return applied[0]["op_id"]


# ---------------------------------------------------------------------------
def test_silence_budget_law():
    print("== 1. silence-budget law ==")
    tx, ledger = Transaction(), WindowLedger(1.0, barrier=True)
    rec = Recorder(tx)
    # fin12b shape: launch at 7.398 inside seg1(6.91,11.84); gap 0.48; seg2 ends 14.398
    segs = [(1.03, 5.76), (6.91, 11.84), (12.32, 14.398)]
    oid = launch(tx, ledger, t=7.398)
    advance_over(ledger, 7.398, 15.038, segs, rec)   # up to the final EoU
    ok("expiry fires at burn-exact nominal 14.918",
       rec.commits and abs(rec.commits[0]["nominal"] - 14.918) < 1e-6)
    ok("op committed (single)", len(tx.committed) == 1 and not tx.pending)

    # same shape, delta 1.5: must SURVIVE to the EoU (deadline 15.418 > 15.038)
    tx2, ledger2 = Transaction(), WindowLedger(1.5, barrier=True)
    rec2 = Recorder(tx2)
    oid2 = launch(tx2, ledger2, t=7.398)
    advance_over(ledger2, 7.398, 15.038, segs, rec2)
    ok("delta 1.5 survives the EoU (silence budget 1.12 < 1.5)",
       not rec2.commits and oid2 in tx2.pending)
    ok("remaining budget ≈ 0.38", abs(ledger2.remaining(oid2) - 0.38) < 1e-6)

    # pure-speech interval burns nothing
    spans = silent_spans(8.0, 11.0, segs)
    ok("silent_spans inside speech is empty", spans == [])


def test_barrier_defer_rescue_and_commit():
    print("== 2. barrier: defer + rescue / defer + nominal commit ==")
    segs = [(0.0, 10.0)]   # speech ends at 10.0; everything after is silence
    for rescued in (True, False):
        tx, ledger = Transaction(), WindowLedger(1.0, barrier=True)
        rec = Recorder(tx)
        oid = launch(tx, ledger, args={"destination": "NYC"}, t=9.0)
        # silence burns 10.0->11.0 => nominal deadline 11.0; EoU at 10.64 dispatches
        advance_over(ledger, 9.0, 10.64, segs, rec)
        tx.snapshot_for_prompt()                      # builds the local-id map
        ledger.begin_decision("d2", set(tx.pending))
        advance_over(ledger, 10.64, 11.64, segs, rec)  # expiry 11.0 in flight window
        ok(f"[resc={rescued}] expiry deferred, not committed",
           not rec.commits and oid in ledger.expired)
        dec = {"ops": ([{"type": "patch", "op_id": 1,
                         "diff": {"destination": "Boston"}}] if rescued else []),
               "say": ""}
        apply_decision_ops(tx, ledger, dec, 11.64, immediate=False, commit_cb=rec)
        ledger.end_decision("d2")
        ledger.sweep(11.64, rec, cause="decision_done")
        if rescued:
            ok("patch rescued: nothing committed, window restarted with full budget",
               not rec.commits and abs(ledger.remaining(oid) - 1.0) < 1e-9)
            ok("deferral audited as rescued_patch",
               ledger.deferrals and ledger.deferrals[0]["outcome"] == "rescued_patch")
            advance_over(ledger, 11.64, math.inf, segs, rec)   # tail
            ok("restarted window expires at 12.64 with PATCHED args",
               rec.commits[0]["nominal"] == 12.64
               and tx.committed[0].args["destination"] == "Boston")
        else:
            ok("unrescued deferral commits at sweep, stamped at NOMINAL 11.0",
               rec.commits and rec.commits[0]["nominal"] == 11.0
               and rec.commits[0]["actual"] == 11.64)
            d = ledger.deferrals[0]
            ok("dual stamp audited: deferred_s = 0.64",
               d["outcome"] == "committed" and abs(d["deferred_s"] - 0.64) < 1e-9)


def test_continuous_clock_ablation():
    print("== 3. barrier OFF (continuous clock) ==")
    segs = [(0.0, 10.0)]
    tx, ledger = Transaction(), WindowLedger(1.0, barrier=False)
    rec = Recorder(tx)
    oid = launch(tx, ledger, args={"destination": "NYC"}, t=9.0)
    advance_over(ledger, 9.0, 10.64, segs, rec)
    tx.snapshot_for_prompt()
    ledger.begin_decision("d2", set(tx.pending))
    advance_over(ledger, 10.64, 11.64, segs, rec)      # expiry 11.0 commits NOW
    ok("expiry commits mid-flight at nominal=actual=11.0",
       rec.commits and rec.commits[0]["nominal"] == 11.0
       and rec.commits[0]["actual"] == 11.0)
    dec = {"ops": [{"type": "patch", "op_id": 1, "diff": {"destination": "Boston"}}],
           "say": ""}
    apply_decision_ops(tx, ledger, dec, 11.64, immediate=False, commit_cb=rec)
    ledger.end_decision("d2")
    ledger.sweep(11.64, rec)
    ok("late patch dropped: committed args UNCHANGED",
       tx.committed[0].args["destination"] == "NYC")
    ok("patch_after_commit logged", len(ledger.patch_after_commit) == 1)


def test_release_paths():
    print("== 4. release paths: stale / timeout leave nothing deferred ==")
    segs = [(0.0, 5.0)]
    tx, ledger = Transaction(), WindowLedger(0.5, barrier=True)
    rec = Recorder(tx)
    launch(tx, ledger, t=4.0)
    ledger.begin_decision("dX", set(tx.pending))
    advance_over(ledger, 4.0, 6.0, segs, rec)          # expiry 5.5 deferred
    ok("deferred under guard", not rec.commits and len(ledger.expired) == 1)
    # stale/timeout path: guard released WITHOUT ops, then sweep
    ledger.end_decision("dX")
    ledger.sweep(6.0, rec, cause="timeout")
    ok("fail-open sweep commits at nominal 5.5",
       rec.commits and rec.commits[0]["nominal"] == 5.5
       and not ledger.expired and not ledger.win)
    ok("deferral cause audited as timeout",
       ledger.deferrals[-1]["cause"] == "timeout")

    # overlapping guards: op stays deferred until the LAST guard releases
    tx2, ledger2 = Transaction(), WindowLedger(0.5, barrier=True)
    rec2 = Recorder(tx2)
    oid = launch(tx2, ledger2, t=4.0)
    ledger2.begin_decision("a", {oid})
    ledger2.begin_decision("b", {oid})
    advance_over(ledger2, 4.0, 6.0, segs, rec2)
    ledger2.end_decision("a")
    ledger2.sweep(6.0, rec2)
    ok("still guarded by b: not committed", not rec2.commits)
    ledger2.end_decision("b")
    ledger2.sweep(6.5, rec2)
    ok("released by b: committed at nominal", rec2.commits[0]["nominal"] == 5.5)


def test_immediate_and_hygiene():
    print("== 5/6. immediate commits, dedup, coercion ==")
    tx, ledger = Transaction(), WindowLedger(1.5, barrier=True)
    rec = Recorder(tx)
    dec = {"ops": [{"type": "launch", "fn": "search_apartments",
                    "args": {"args": {"city": "Austin"}, "max_price": "1800",
                             "bedrooms": "2"}},
                   {"type": "launch", "fn": "search_apartments",     # exact dup
                    "args": {"city": "Austin", "max_price": 1800, "bedrooms": 2}},
                   {"type": "launch", "fn": "update_search_filter",
                    "args": {"filter_name": "pets_allowed", "value": "true"}}],
           "say": "x"}
    applied = apply_decision_ops(tx, ledger, dec, 3.0, immediate=True, commit_cb=rec)
    ok("immediate mode commits at t_dec (nominal == actual == 3.0)",
       all(c["nominal"] == 3.0 and c["actual"] == 3.0 for c in rec.commits))
    ok("nested-args unwrapped + numerics coerced",
       tx.committed[0].args == {"city": "Austin", "max_price": 1800, "bedrooms": 2})
    ok("exact duplicate deduped", [a["type"] for a in applied].count("launch_dedup") == 1)
    ok("poly field coerced to bool", tx.committed[1].args["value"] is True)
    ok("coerce leaves non-schema strings alone",
       coerce_args({"query": "12 keyboards"})["query"] == "12 keyboards")


# ---------------------------------------------------------------------------
# eco19-shaped end-to-end (mini replay driver over tact_core, scripted decisions)
# ---------------------------------------------------------------------------
def drive(segs, eous, decisions, delta, barrier, infer=1.0):
    """Minimal replica of the replay driver loop (scripted decisions)."""
    tx, ledger = Transaction(), WindowLedger(delta, barrier=barrier)
    rec = Recorder(tx)
    cursor = 0.0
    for k, (seg_idx, t_eou) in enumerate(eous):
        advance_over(ledger, cursor, t_eou, segs, rec)
        cursor = max(cursor, t_eou)
        tx.snapshot_for_prompt()
        ledger.begin_decision(k, set(tx.pending))
        t_dec = t_eou + infer
        advance_over(ledger, cursor, t_dec, segs, rec)
        cursor = max(cursor, t_dec)
        apply_decision_ops(tx, ledger, decisions[k], t_dec,
                           immediate=(delta <= 0), commit_cb=rec)
        ledger.end_decision(k)
        ledger.sweep(t_dec, rec)
    advance_over(ledger, cursor, math.inf, segs, rec)
    ledger.sweep(cursor, rec, cause="finalize")
    return tx, ledger, rec


def test_eco19_shape():
    print("== 7. eco19-SHAPED flip (real value diff): barrier on passes, off fails ==")
    # NOTE: segments/EoUs mirror ecommerce_19, but the patch here carries a REAL
    # value change (laptop->tablet). Real eco19's patch was value-neutral (its flip
    # is a snapshot effect, barrier-independent — see w3_ledger §4A erratum); this
    # test exercises the hou25/fin12b-class mechanics where the barrier IS causal.
    segs = [(0.962, 8.254), (8.642, 10.718), (11.426, 18.91)]
    eous = [(1, 11.358), (2, 19.55)]
    decisions = [
        {"ops": [{"type": "launch", "fn": "search_products",
                  "args": {"query": "laptop"}}], "say": "Searching."},
        {"ops": [{"type": "patch", "op_id": 1, "diff": {"query": "tablet"}},
                 {"type": "launch", "fn": "add_to_cart",
                  "args": {"product_id": "P1", "quantity": 1}}], "say": "Updating."},
    ]
    # barrier ON, delta=1.0: search expiry 19.91 lands in (19.55, 20.55] -> deferred,
    # patch rescues -> both calls commit with the CORRECTED query
    tx, ledger, rec = drive(segs, eous, decisions, 1.0, barrier=True)
    calls = tx.to_actual_tool_calls()
    ok("on: 2 calls, query patched to tablet",
       len(calls) == 2 and calls[0]["args"]["query"] == "tablet")
    ok("on: rescue audited",
       any(d["outcome"] == "rescued_patch" for d in ledger.deferrals))
    # barrier OFF, same delta: search commits at 19.91 with the DIRTY query;
    # the patch drops; add_to_cart still launches
    tx2, ledger2, rec2 = drive(segs, eous, decisions, 1.0, barrier=False)
    calls2 = tx2.to_actual_tool_calls()
    ok("off: dirty query committed at nominal 19.91",
       calls2[0]["args"]["query"] == "laptop"
       and any(c["nominal"] == 19.91 for c in rec2.commits))
    ok("off: patch_after_commit logged", len(ledger2.patch_after_commit) == 1)


# ---------------------------------------------------------------------------
# TactEngine offline smoke (injected replay, scripted VAD + scripted decisions)
# ---------------------------------------------------------------------------
class FakeVAD:
    """Scripted VADIterator: emits {'start'|'end': t} when the audio clock crosses
    the scripted times. Called once per 512 samples (0.032 s)."""
    def __init__(self, events):                 # events: [(t_sec, 'start'|'end')]
        self.events = sorted(events)
        self.t = 0.0
        self.i = 0

    def __call__(self, tensor, return_seconds=True):
        self.t += 0.032
        if self.i < len(self.events) and self.t >= self.events[self.i][0]:
            t, kind = self.events[self.i]
            self.i += 1
            return {kind: t}
        return None

    def reset_states(self):
        self.t, self.i = 0.0, 0


def _mk_engine(events, decisions, delta, barrier):
    from engine_b import TactEngine

    def script(kind, meta):
        if kind == "tact":
            dec = decisions[min(script.n, len(decisions) - 1)]
            script.n += 1
            return {"decision": dec["decision"], "infer": dec["infer"]}
        return {"text": "", "infer": 0.0, "wav_path": "", "dur_audio": 0.0}
    script.n = 0

    return TactEngine(
        prompts={}, delay={"end_hold_frame": 0.64, "after_continue_time": 2.5},
        llm_cfg={"decision_timeout_s": 30},
        engine_cfg={"phase": "b", "mode": "tact", "delta": delta,
                    "commit_barrier": barrier, "tool_sync": True,
                    "tts_enabled": False, "asr_enabled": False},
        llm_fn=lambda m: "", asr_fn=lambda p: "",
        tts_fn=lambda t, **k: ("", 0.0),
        replay_mode="injected", decision_script=script,
        vad_iterator=FakeVAD(events),
        tool_executor=lambda fn, args: {"status": "success"})


def test_engine_offline_smoke():
    print("== 8. TactEngine offline smoke (injected, scripted) ==")
    from engine import frames_from_array

    # one utterance 0.5-2.0s in 8s of audio; EoU ~2.64; launch; delta=1.0
    audio = np.zeros(int(8.0 * 16000), dtype=np.float32)
    eng = _mk_engine(
        [(0.5, "start"), (2.0, "end")],
        [{"decision": {"dialogue": "speak",
                       "ops": [{"type": "launch", "fn": "search_flights",
                                "args": {"destination": "NYC", "date": "July 15"}}],
                       "say": "Searching."},
          "infer": 0.5}],
        delta=1.0, barrier=True)
    asyncio.run(eng.run_offline(frames_from_array(audio)))
    eng.finalize_windows()
    ok("engine: one EoU, one decision", len(eng.tact_eous) == 1
       and len(eng.tact_decisions) == 1)
    ok("engine: op committed on the tail", len(eng.tx.committed) == 1)
    r = eng.commit_records[0]
    # anchor ≈ 2.0 (+VAD chunk granularity) -> eou ≈ 2.64 -> dec ≈ 3.14 -> +1.0
    ok("engine: nominal commit stamp ≈ 4.14 (audio clock)",
       abs(r["t_commit"] - 4.14) < 0.15)
    ok("engine: say/ack recorded", len(eng.say_events) == 1)
    exp = eng.export_fdb_result("smoke_001", "unit")
    ok("engine: export has the call",
       exp["actual_tool_calls"][0]["function"] == "search_flights")

    # barrier rescue through the engine loop: two utterances; the 2nd decision
    # (infer 0.5) overlaps the 1st op's expiry; patch must rescue it.
    audio2 = np.zeros(int(9.0 * 16000), dtype=np.float32)
    eng2 = _mk_engine(
        [(0.5, "start"), (2.0, "end"), (3.5, "start"), (5.0, "end")],
        [{"decision": {"dialogue": "speak",
                       "ops": [{"type": "launch", "fn": "search_flights",
                                "args": {"destination": "NYC", "date": "July 15"}}],
                       "say": "Searching."}, "infer": 0.2},
         {"decision": {"dialogue": "speak",
                       "ops": [{"type": "patch", "op_id": 1,
                                "diff": {"destination": "Boston"}}],
                       "say": "Updating."}, "infer": 0.5}],
        delta=1.5, barrier=True)
    asyncio.run(eng2.run_offline(frames_from_array(audio2)))
    eng2.finalize_windows()
    calls = eng2.tx.to_actual_tool_calls()
    ok("engine barrier: single call, destination patched to Boston",
       len(calls) == 1 and calls[0]["args"]["destination"] == "Boston")
    ok("engine barrier: rescue audited in the ledger",
       any(d["outcome"] == "rescued_patch" for d in eng2.ledger.deferrals))

    # v0 compatibility surface (test_phase_b.py contract)
    from engine_b import TactEngine
    eng3 = TactEngine(prompts={}, delay={}, llm_cfg={},
                      engine_cfg={"phase": "b", "blocking": True, "delta": 2.0},
                      llm_fn=lambda m: "", asr_fn=lambda p: "",
                      tts_fn=lambda t, **k: ("", 0.0),
                      replay_mode="oracle", vad_iterator=FakeVAD([]),
                      tool_executor=lambda fn, a: {})
    ok("v0 surface: phase/blocking_mode/delta/export intact",
       eng3.phase == "b" and eng3.blocking_mode is True and eng3.delta == 2.0
       and eng3.export_fdb_result("t", "p")["status"] == "completed")


if __name__ == "__main__":
    test_silence_budget_law()
    test_barrier_defer_rescue_and_commit()
    test_continuous_clock_ablation()
    test_release_paths()
    test_immediate_and_hygiene()
    test_eco19_shape()
    test_engine_offline_smoke()
    print(f"\nALL {len(PASS)} CHECKS PASSED")
