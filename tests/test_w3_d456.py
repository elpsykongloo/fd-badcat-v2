# -*- coding: utf-8 -*-
"""
tests/test_w3_d456.py — pure-CPU unit tests for the W3 D4–D6 mechanisms.

No model, no network, no wav files, no real VAD. Run:
  /root/miniconda3/envs/fd-sds/bin/python tests/test_w3_d456.py

Covers:
  A. latency_realistic — per-instance determinism, class percentiles vs the
     preregistered table, serial (blocking) vs parallel/DAG (tact) scheduling,
     attach() first-response conventions
  B. tact_dag — edge detection, patch propagation (reparam / stale / window
     restart), compensation plans, κ-faithful registry + idempotency, and the
     apply_decision_ops(dag=…) wiring incl. launch idem keys
  C. normalize_entity — the closed N1–N8 rule set, positives AND negatives
  D. tts_sentence / floor_policy — splitting, tier rule, unconditional
     narration interruptibility
  E. TactEngine speculative_dispatch — adopt-at-EoU, invalidate-on-resume,
     barrier interplay (deferral through a spec flight), finalize release,
     and parity of committed calls vs the non-spec arm
  F. TactEngine tts_split + floor-holding — sentence events, barge-in drop
"""
import asyncio
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/root/autodl-tmp")

import numpy as np  # noqa: E402

import tact_core  # noqa: E402
from tact_core import WindowLedger, apply_decision_ops  # noqa: E402
from tact.transaction import Transaction  # noqa: E402

import latency_realistic as lr  # noqa: E402
import tact_dag  # noqa: E402
from tact_dag import OpDag, CompensationRegistry, make_idem_key  # noqa: E402
import normalize_entity as ne  # noqa: E402
from tts_sentence import split_sentences  # noqa: E402
import floor_policy  # noqa: E402

PASS = []


def ok(label, cond):
    assert cond, f"FAIL: {label}"
    PASS.append(label)
    print(f"  ok - {label}")


# ---------------------------------------------------------------------------
# A. latency_realistic
# ---------------------------------------------------------------------------
def test_realistic_sampler():
    print("== A1. realistic sampler: determinism + calibration ==")
    a = lr.sample_latency("ex1", "book_flight", {"passenger_name": "kim"}, 0)
    b = lr.sample_latency("ex1", "book_flight", {"passenger_name": "kim"}, 0)
    c = lr.sample_latency("ex1", "book_flight", {"passenger_name": "kim"}, 1)
    d = lr.sample_latency("ex2", "book_flight", {"passenger_name": "kim"}, 0)
    ok("same instance -> same draw", a == b)
    ok("occurrence & example vary the draw", a != c and a != d)
    for fn, (p50t, p95t) in [("get_exchange_rate", (0.30, 0.74)),
                             ("search_flights", (0.75, 1.33)),
                             ("add_to_cart", (0.40, 0.91)),
                             ("book_flight", (3.00, 5.00))]:
        draws = sorted(lr.sample_latency("cal", fn, {"k": i}, 0) for i in range(4000))
        p50, p95 = draws[2000], draws[3800]
        ok(f"{fn}: p50 {p50:.2f} ≈ {p50t} (±15%)", abs(p50 - p50t) / p50t < 0.15)
        ok(f"{fn}: p95 {p95:.2f} ≈ {p95t} (±20%)", abs(p95 - p95t) / p95t < 0.20)
    cap = math.exp(lr.CLASS_PARAMS["write_booking"][0] + 3.09 * lr.CLASS_PARAMS["write_booking"][1])
    draws = [lr.sample_latency("cap", "book_flight", {"k": i}, 0) for i in range(4000)]
    ok("p999 cap enforced", max(draws) <= cap + 5e-4)   # 3-decimal rounding slack


def _mk_result(mode, commits, example_id="sched"):
    """commits: [(op_id, fn, args, t_commit)]"""
    return {"example_id": example_id, "mode": mode,
            "latency": {"first_response_s": 1.0, "ack_emitted": True},
            "trace": {"commits": [{"op_id": o, "t_commit": t} for o, _, _, t in commits]},
            "tx_log": [{"op": "commit", "op_id": o, "fn": fn, "args": a, "t": t}
                       for o, fn, a, t in commits]}


def test_realistic_schedule():
    print("== A2. realistic schedule: serial vs parallel vs DAG ==")
    commits = [(1, "search_flights", {"destination": "NYC", "date": "July 15"}, 5.0),
               (2, "get_exchange_rate", {"amount": 100, "from_currency": "EUR",
                                         "to_currency": "USD"}, 5.0)]
    lat1 = lr.sample_latency("sched", "search_flights", commits[0][2], 0)
    lat2 = lr.sample_latency("sched", "get_exchange_rate", commits[1][2], 0)

    ser = lr.schedule(_mk_result("blocking", commits))
    ok("blocking is serial: ready = t + lat1 + lat2",
       abs(ser["result_ready"] - (5.0 + lat1 + lat2)) < 1e-3)
    par = lr.schedule(_mk_result("tact", commits))
    ok("tact independent ops run in parallel: ready = t + max(lat)",
       abs(par["result_ready"] - (5.0 + max(lat1, lat2))) < 1e-3)
    dag = lr.schedule(_mk_result("tact", commits), edges={2: [1]})
    ok("DAG child waits for parent: ready = t + lat1 + lat2",
       abs(dag["result_ready"] - (5.0 + lat1 + lat2)) < 1e-3)

    # attach conventions
    r = _mk_result("blocking", commits)
    got = lr.attach(r, t_user_end=4.0)
    ok("blocking first response = realistic completion",
       got["first_response_s"] == got["task_completion_s"])
    r2 = _mk_result("tact", commits)
    got2 = lr.attach(r2, t_user_end=4.0)
    ok("tact first response keeps the say anchor (tool-independent)",
       got2["first_response_s"] == 1.0)
    ok("premium = completion difference only",
       got["task_completion_s"] > got2["task_completion_s"])


# ---------------------------------------------------------------------------
# B. tact_dag
# ---------------------------------------------------------------------------
def _launch(tx, dag, ledger, fn, args, t):
    dec = {"ops": [{"type": "launch", "fn": fn, "args": args}], "say": "x"}
    applied = apply_decision_ops(tx, ledger, dec, t, immediate=False,
                                 commit_cb=lambda *a: None, dag=dag)
    return applied[0]["op_id"]


def test_dag_propagation():
    print("== B1. DAG: edges + patch propagation ==")
    tx, ledger = Transaction(), WindowLedger(1.5, barrier=True)
    dag = OpDag(ledger)
    p = _launch(tx, dag, ledger, "search_apartments",
                {"city": "Austin", "bedrooms": 2, "max_price": 1600}, 1.0)
    c = _launch(tx, dag, ledger, "calculate_commute",
                {"origin_address": "downtown Austin", "destination_address": "the office",
                 "mode": "driving"}, 2.0)
    ok("edge detected via value flow (city ⊂ origin_address)",
       dag.edges.get(c) == [p])
    ok("launch idem keys assigned + deterministic",
       tx.pending[p].idem_key == make_idem_key("", "search_apartments",
                                               {"city": "Austin", "bedrooms": 2,
                                                "max_price": 1600}, 0))
    ledger.win[c] = 0.2          # nearly expired: reparam must restart it
    dec = {"ops": [{"type": "patch", "op_id": 1, "diff": {"city": "Dallas"}}], "say": ""}
    tx._localmap = {1: p}
    apply_decision_ops(tx, ledger, dec, 3.0, immediate=False,
                       commit_cb=lambda *a: None, dag=dag)
    ok("parent patched", tx.pending[p].args["city"] == "Dallas")
    ok("child REPARAMETERIZED (Austin -> Dallas in origin_address)",
       tx.pending[c].args["origin_address"] == "downtown Dallas")
    ok("child window restarted to full delta", ledger.remaining(c) == 1.5)
    ok("dag event logged", dag.events and dag.events[-1]["kind"] == "dag_reparam")

    # derived-field staleness: search_products -> add_to_cart
    tx2, led2 = Transaction(), WindowLedger(1.0, barrier=True)
    dag2 = OpDag(led2)
    p2 = _launch(tx2, dag2, led2, "search_products", {"query": "watch", "max_price": 200}, 1.0)
    c2 = _launch(tx2, dag2, led2, "add_to_cart", {"product_id": "PROD1", "quantity": 1}, 2.0)
    ok("declared derived edge (no value flow needed)", dag2.edges.get(c2) == [p2])
    tx2._localmap = {1: p2}
    apply_decision_ops(tx2, led2, {"ops": [{"type": "patch", "op_id": 1,
                                            "diff": {"query": "nice watch"}}], "say": ""},
                       3.0, immediate=False, commit_cb=lambda *a: None, dag=dag2)
    ok("result-derived child marked stale",
       dag2.events[-1]["kind"] == "dag_stale"
       and dag2.events[-1]["stale_fields"] == ["product_id"])

    # committed child -> compensation PLAN (priced, not executed)
    tx3, led3 = Transaction(), WindowLedger(1.0, barrier=True)
    dag3, reg3 = OpDag(led3), CompensationRegistry()
    p3 = _launch(tx3, dag3, led3, "search_products", {"query": "watch"}, 1.0)
    c3 = _launch(tx3, dag3, led3, "add_to_cart", {"product_id": "PROD1", "quantity": 1}, 2.0)
    tx3.commit(c3, lambda fn, a: {"status": "success"}, t=3.0)
    led3.close(c3)
    tx3._localmap = {1: p3}
    apply_decision_ops(tx3, led3, {"ops": [{"type": "patch", "op_id": 1,
                                            "diff": {"query": "gold watch"}}], "say": ""},
                       4.0, immediate=False, commit_cb=lambda *a: None,
                       dag=dag3, comp_registry=reg3)
    ev = dag3.events[-1]
    ok("committed child -> comp plan", ev["kind"] == "dag_comp_plan"
       and ev["plan"]["fn"] == "remove_from_cart"
       and ev["plan"]["action"] == "execute")


def test_comp_registry():
    print("== B2. compensation registry: κ-faithful + idempotent ==")
    from tact.transaction import PendingOp, Reversibility
    reg = CompensationRegistry()
    book = PendingOp(fn="book_flight", args={"passenger_name": "kim"},
                     reversibility=Reversibility.COMP, idem_key="abc123")
    book.result = {"status": "success", "booking_ref": "B789"}
    plan = reg.plan(book)
    ok("book_flight -> cancel_booking(booking_ref from result)",
       plan["fn"] == "cancel_booking" and plan["args"]["booking_ref"] == "B789"
       and plan["idem_key"] == "comp:abc123")
    calls = []
    ex = lambda fn, args: calls.append((fn, args)) or {"status": "success"}
    r1 = reg.execute(plan, ex)
    r2 = reg.execute(plan, ex)
    ok("at-most-once per idem key", len(calls) == 1
       and r2["status"] == "skipped_idempotent")

    irr = PendingOp(fn="update_identity_doc", args={"doc_type": "passport"},
                    reversibility=Reversibility.IRR)
    p_irr = reg.plan(irr)
    ok("IRR refuses", p_irr["action"] == "refuse")
    try:
        reg.execute(p_irr, ex)
        ok("IRR execute raises", False)
    except ValueError:
        ok("IRR execute raises", True)
    rd = PendingOp(fn="search_flights", args={}, reversibility=Reversibility.READ)
    ok("READ is a no-op comp", reg.plan(rd)["action"] == "noop")

    usf = PendingOp(fn="update_search_filter",
                    args={"filter_name": "max price", "value": 2500},
                    reversibility=Reversibility.REV)
    usf.patch_history.append({"t": 1.0, "diff": {"value": 2500}, "before": {"value": 2000}})
    pu = reg.plan(usf)
    ok("REV exact inverse recovers the prior value",
       pu["fn"] == "update_search_filter" and pu["args"]["value"] == 2000)


# ---------------------------------------------------------------------------
# C. normalizer  /  D. sentence split + floor rule
# ---------------------------------------------------------------------------
def test_normalizer():
    print("== C. normalizer (closed rule set) ==")
    pos = [("June 3rd", "june 3"), ("august 20th", "august 20"), ("K-2", "k2"),
           ("P-5-2", "p52"), ("driver's license", "driver license"),
           ("the gym", "gym"), ("The University", "university"),
           ("mechanical keyboards", "mechanical keyboard"), ("7.0", 7),
           ("D-L-5-5-5", "dl555"), ("V-4-4", "v44"), ("  Boston ", "boston")]
    for a, b in pos:
        ok(f"norm-equal: {a!r} == {b!r}", ne.values_equal(a, b))
    neg = [("vegas", "las vegas"), ("bob", "pop"), ("de", "deliv"),
           ("nice watch", "watch"), ("800", "1800"), ("p88990011", "p99990011"),
           ("address", "addres")]
    for a, b in neg:
        ok(f"norm-distinct: {a!r} != {b!r}", not ne.values_equal(a, b))
    ok("idempotent", ne.normalize_value(ne.normalize_value("June 3rd")) ==
       ne.normalize_value("June 3rd"))


def test_sentence_and_floor():
    print("== D. sentence split + floor rule v0 ==")
    s = split_sentences("Updating that to Chicago. Checking the commute now. "
                        "Anything else today?")
    ok("3 sentences", len(s) == 3 and s[0].endswith("Chicago."))
    ok("trailing short fragment merges backward",
       split_sentences("Updating that to Chicago. Done!")
       == ["Updating that to Chicago. Done!"])
    ok("short fragments merge", len(split_sentences("Done. Ok.")) == 1)
    ok("empty -> []", split_sentences("") == [])
    ok("single stays whole", split_sentences("Booking that flight now.") ==
       ["Booking that flight now."])

    ok("narration always yields", floor_policy.decide("narration", 0.1, 3.0) == "yield")
    ok("ack always yields", floor_policy.decide("ack", 0.05, 3.0) == "yield")
    ok("confirmation + imminent commit finishes the clause",
       floor_policy.decide("confirmation", 0.8, 3.0) == "finish_clause")
    ok("confirmation + distant commit yields",
       floor_policy.decide("confirmation", 2.5, 3.0) == "yield")
    ok("confirmation + no pending yields",
       floor_policy.decide("confirmation", None, None) == "yield")
    ok("unknown kind fails open", floor_policy.decide("???", 0.1, 1.0) == "yield")
    ok("v0 never holds a full utterance",
       all(floor_policy.decide(k, w, 3.0) != "finish_utterance"
           for k in ("narration", "ack", "confirmation", "x")
           for w in (None, 0.1, 0.9, 5.0)))
    ok("eta prior = class p50", floor_policy.eta_prior(["book_flight"]) == 3.0)


# ---------------------------------------------------------------------------
# E. TactEngine speculative dispatch (scripted engine, injected replay)
# ---------------------------------------------------------------------------
class FakeVAD:
    def __init__(self, events):
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


def _mk_engine(events, decisions, delta, barrier, cfg_extra=None, tts_script=None):
    from engine_b import TactEngine

    def script(kind, meta):
        if kind == "tact":
            dec = decisions[min(script.n, len(decisions) - 1)]
            script.n += 1
            return {"decision": dec["decision"], "infer": dec["infer"]}
        if tts_script is not None:
            return tts_script(kind, meta)
        return {"text": "", "infer": 0.0, "wav_path": "", "dur_audio": 0.0}
    script.n = 0

    cfg = {"phase": "b", "mode": "tact", "delta": delta,
           "commit_barrier": barrier, "tool_sync": True,
           "tts_enabled": False, "asr_enabled": False}
    cfg.update(cfg_extra or {})
    eng = TactEngine(
        prompts={}, delay={"end_hold_frame": 0.64, "after_continue_time": 2.5},
        llm_cfg={"decision_timeout_s": 30},
        engine_cfg=cfg,
        llm_fn=lambda m: "", asr_fn=lambda p: "",
        tts_fn=lambda t, **k: ("", 0.0),
        replay_mode="injected", decision_script=script,
        vad_iterator=FakeVAD(events),
        tool_executor=lambda fn, args: {"status": "success"})
    return eng


LAUNCH_DEC = {"decision": {"dialogue": "speak",
                           "ops": [{"type": "launch", "fn": "search_flights",
                                    "args": {"destination": "NYC", "date": "July 15"}}],
                           "say": "Searching."},
              "infer": 0.3}


def _run(eng, seconds=8.0):
    from engine import frames_from_array
    audio = np.zeros(int(seconds * 16000), dtype=np.float32)
    asyncio.run(eng.run_offline(frames_from_array(audio)))
    eng.finalize_windows()
    return eng


def test_speculative_adopt():
    print("== E1. speculative: dispatch at vad end, adopt at EoU ==")
    ev = [(0.5, "start"), (2.0, "end")]
    spec = _run(_mk_engine(ev, [LAUNCH_DEC], 1.0, True,
                           {"speculative_dispatch": True}))
    base = _run(_mk_engine(ev, [LAUNCH_DEC], 1.0, True))
    ok("spec: one EoU, one decision", len(spec.tact_eous) == 1
       and len(spec.tact_decisions) == 1)
    strip = lambda calls: [(c["function"], tuple(sorted(c["args"].items())))
                           for c in calls]
    ok("spec: committed call set identical to non-spec arm",
       strip(spec.tx.to_actual_tool_calls()) == strip(base.tx.to_actual_tool_calls()))
    ok("spec: commit stamp EARLIER by the overlapped infer (0.3s)",
       abs(base.commit_records[0]["t_commit"]
           - spec.commit_records[0]["t_commit"] - 0.3) < 1e-6)
    t_spec = spec.say_events[0][0]
    t_base = base.say_events[0][0]
    # infer 0.3 < hold 0.64: spec say lands AT the EoU; base at EoU + 0.3
    ok(f"spec say at EoU ({t_spec:.3f} ≈ 2.64 < base {t_base:.3f})",
       abs(t_spec - 2.656) < 0.05 and t_base - t_spec > 0.25)
    ok("spec trace has dispatch event",
       any(e["event"] == "tact_spec_dispatch" for e in spec.trace))


def test_speculative_invalidate():
    print("== E2. speculative: speech resumes inside hold -> invalidated ==")
    ev = [(0.5, "start"), (2.0, "end"), (2.3, "start"), (3.5, "end")]
    eng = _run(_mk_engine(ev, [LAUNCH_DEC, LAUNCH_DEC], 1.0, True,
                          {"speculative_dispatch": True}))
    ok("only the second anchor became an EoU", len(eng.tact_eous) == 1
       and abs(eng.tact_eous[0][1] - 4.14) < 0.05)
    ok("exactly one decision applied", len(eng.tact_decisions) == 1)
    ok("one committed call (no double launch)", len(eng.tx.committed) == 1)
    ok("invalidation traced",
       any(e["event"] in ("tact_spec_discarded",)
           for e in eng.trace) or
       any(e.get("data", {}).get("event") == "tact_decision_spec_invalid"
           for e in eng.trace) or
       any(e["event"] == "tact_decision_spec_invalid" for e in eng.trace))
    ok("no stuck guards / spec entries",
       not eng.ledger.guards and not eng._spec_inflight)


def test_speculative_barrier_interplay():
    print("== E3. speculative: expiry inside spec flight defers (barrier) ==")
    # EoU1 launches (delta 2.0). Burn 2.656->4.0 (1.344), frozen 4.0->5.0,
    # residual 0.656 expires ~5.68 — inside the spec-2 flight (dispatch ~5.0,
    # infer 2.0 -> lands ~7.0) => defer, then commit at nominal on the sweep.
    decs = [LAUNCH_DEC,
            {"decision": {"dialogue": "speak", "ops": [], "say": "Noted."},
             "infer": 2.0}]
    eng = _run(_mk_engine([(0.5, "start"), (2.0, "end"),
                           (4.0, "start"), (5.0, "end")],
                          decs, 2.0, True, {"speculative_dispatch": True}),
               seconds=10.0)
    ok("two EoUs, two decisions", len(eng.tact_eous) == 2
       and len(eng.tact_decisions) == 2)
    ok("op committed exactly once", len(eng.tx.committed) == 1)
    defs = [d for d in eng.ledger.deferrals if d["outcome"] == "committed"]
    ok("expiry was deferred by the spec-flight barrier and dual-stamped",
       len(defs) == 1 and defs[0]["deferred_s"] > 0)
    rec = eng.commit_records[0]
    ok("nominal stamp preserved (audio-clock truth)",
       rec["t_commit"] < rec["actual_commit"])


def test_speculative_tail_release():
    print("== E4. speculative: audio ends inside hold -> guard released ==")
    # vad end at 7.9; audio ends at 8.0 < EoU 8.54: the spec decision must be
    # discarded at finalize and op1's window still commits on the tail.
    decs = [LAUNCH_DEC, LAUNCH_DEC]
    eng = _run(_mk_engine([(0.5, "start"), (2.0, "end"),
                           (7.0, "start"), (7.9, "end")],
                          decs, 3.0, True, {"speculative_dispatch": True}))
    ok("second anchor never confirmed (1 EoU)", len(eng.tact_eous) == 1)
    ok("spec guard released at finalize",
       not eng.ledger.guards and not eng._spec_inflight)
    ok("op1 committed on the tail despite the dangling spec",
       len(eng.tx.committed) == 1)
    ok("finalize discard traced",
       any(e["event"] == "tact_spec_discarded" and
           e["data"].get("at") == "finalize" for e in eng.trace))


# ---------------------------------------------------------------------------
# F. sentence-split TTS + floor-holding through the engine
# ---------------------------------------------------------------------------
def test_tts_split_and_floor():
    print("== F. tts_split + floor decision ==")

    def tts_script(kind, meta):
        return {"text": meta.get("text", ""), "infer": 0.4, "wav_path": "",
                "dur_audio": 1.0}

    say3 = {"decision": {"dialogue": "speak", "ops": [
                {"type": "launch", "fn": "book_flight",
                 "args": {"passenger_name": "kim"}}],
            "say": "Booking that flight now. I will confirm the seat next. "
                   "Anything else?"},
            "infer": 0.3}
    eng = _mk_engine([(0.5, "start"), (2.0, "end")], [say3], 1.0, True,
                     {"tts_enabled": True, "tts_split": True,
                      "floor_holding": True}, tts_script=tts_script)
    _run(eng, seconds=12.0)
    sent = [e for e in eng.trace if e["event"] == "tts_sent_done"]
    ok("3 sentences dispatched and delivered", len(sent) == 3
       and sent[0]["data"]["first_sentence"] is True)
    ok("utterance classified as confirmation (COMP launch)",
       eng._say_kind == "confirmation")

    # floor decision unit path: open utterance mid-flight, barge-in
    eng2 = _mk_engine([], [], 1.0, True, {"tts_split": True, "floor_holding": True})
    eng2._utt_state[1] = {"kind": "narration", "n": 3, "allow_upto": None,
                          "last_sent": 0}
    eng2._apply_floor_decision()
    ok("narration barge-in: yield => nothing beyond last_sent",
       eng2._utt_state[1]["allow_upto"] == 0)
    fd = [e for e in eng2.trace if e["event"] == "floor_decision"]
    ok("floor decision traced with tier", fd and fd[0]["data"]["tier"] == "yield")

    eng3 = _mk_engine([], [], 1.0, True, {"tts_split": True, "floor_holding": True})
    dec = {"ops": [{"type": "launch", "fn": "book_flight",
                    "args": {"passenger_name": "kim"}}], "say": ""}
    apply_decision_ops(eng3.tx, eng3.ledger, dec, 0.0, immediate=False,
                       commit_cb=lambda *a: None)
    eng3.ledger.win[list(eng3.tx.pending)[0]] = 0.5      # commit imminent
    eng3._utt_state[7] = {"kind": "confirmation", "n": 3, "allow_upto": None,
                          "last_sent": 0}
    eng3._apply_floor_decision()
    ok("confirmation + imminent commit: finish_clause => one more sentence",
       eng3._utt_state[7]["allow_upto"] == 1)

    # handler drop: sentence beyond allow_upto is dropped
    from engine_b import TtsSentDone
    ev = TtsSentDone(uid=7, idx=2, gen=eng3.session_gen, turn=0, text="tail",
                     dur_audio=1.0)
    asyncio.run(eng3._on_tts_sent(ev))
    ok("sentence beyond allow_upto dropped",
       any(e["event"] == "tts_sent_dropped" for e in eng3.trace))


if __name__ == "__main__":
    test_realistic_sampler()
    test_realistic_schedule()
    test_dag_propagation()
    test_comp_registry()
    test_normalizer()
    test_sentence_and_floor()
    test_speculative_adopt()
    test_speculative_invalidate()
    test_speculative_barrier_interplay()
    test_speculative_tail_release()
    test_tts_split_and_floor()
    print(f"\nALL PASS ({len(PASS)} checks)")
