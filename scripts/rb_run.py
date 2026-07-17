#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rb_run.py — RB v2 runner: arm A (fixed timeline) + arm B (reactive co-sim),
core engine track (docs/rb_design.md v2 §2; bit-parity semantics with the
w2r_stream_replay core driver: same WindowLedger / barrier / apply path /
prompt v3.1 / decision cache / EoU rule).

  $PY scripts/rb_run.py --build exp/rb/build_v2 --split dev --arm A \\
      --system tact --delta 1.5 --provider rbdev_tact_d150
  $PY scripts/rb_run.py ... --system blocking --provider rbdev_sblock
  $PY scripts/rb_run.py ... --floor-commit-tiers v1        # W5-FC arm (L9/L8)
  $PY scripts/rb_run.py ... --arm B --tts qwen             # reactive, live TTS
  $PY scripts/rb_run.py --selftest                         # oracle decider, no LLM

Input calibers: --input audio (default; needs `rb_build --audio` wavs — the
decider HEARS the TTS user, FDB-caliber) | --input text (transcript-grounded
smoke; different cache keys, never mixed into audio-caliber tables).
Arm B pieces are synthesized on demand (--tts qwen|stub) and injected at the
times the reactive user fires (lifecycle anchors via rb/simulator).
Deciders: llm (decision cache, T=0) | oracle (gold-policy sanity, no network).
FIREWALL: RB-test is single-shot per system version; scorer frozen before it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import tact_core                                                  # noqa: E402
from tact_core import (WindowLedger, apply_decision_ops, advance_over,  # noqa: E402
                       decide_from_msgs, ack_fallback, HOLD, SR)
from tact.transaction import Transaction, Reversibility           # noqa: E402
from tact.tools import REVERSIBILITY as _REV                      # noqa: E402
from tact.decider import build_decider_messages                   # noqa: E402
import math                                                        # noqa: E402

from rb.registry import TOOLS, DOMAINS, SCENARIOS, ARG_FORMAT, canon_value  # noqa: E402
from rb.sandbox import Sandbox, LATENCY, mint_id                  # noqa: E402
from rb import scorer as rb_scorer                                # noqa: E402
from rb.simulator import ReactiveUser                             # noqa: E402

STUB_ZH_S, STUB_EN_S, STUB_MIN_S = 0.18, 0.075, 0.4


# ---------------------------------------------------------------------------
# prompt + reversibility injection (RB toolset replaces the FDB catalog; the
# v3.1 rule stack is installed unchanged on top)
# ---------------------------------------------------------------------------
def rb_catalog(v24=False):
    lines = []
    for d in DOMAINS:
        lines.append(d.capitalize() + ":")
        for fn, spec in TOOLS.items():
            if spec["domain"] == d:
                args = ", ".join(
                    f"{a} ({ARG_FORMAT[a]})" if a in ARG_FORMAT else a
                    for a in spec["required"])
                lines.append(f"  {fn}({args})")
    lines.append(
        "Every tool returns {\"id\": \"...\"}. To use an earlier op's result "
        "as an argument write exactly \"$RESULT_<n>.id\", where <n> is the "
        "op_id shown in PENDING OPS, or the 0-based position of an earlier "
        "launch in YOUR CURRENT ops list. Write integer arguments as plain "
        "digits and currencies as ISO codes. Write string values VERBATIM in "
        "the user's language (成都 stays 成都, never Chengdu; keep multiword "
        "values whole: \"first class\", not \"first\").")
    if v24:
        # v2.4 builds only (episode caps gate the semantics; v2.3 replays keep
        # the exact v2.3 catalog bytes above).
        lines.append(
            "If the user changes their mind AFTER an op was executed: while "
            "that tool is still running you may ABORT it by sending "
            "{\"type\": \"cancel\", \"op_id\": \"X<id>\"} with the id shown "
            "in ALREADY EXECUTED; once it has finished, undo it by calling "
            "its reverse tool (e.g. cancel_hotel / remove_item / "
            "unschedule_payment / reverse_transfer) with the executed op's "
            "returned id, then launch the tool again with the corrected "
            "arguments. When asked to confirm a value you set, state the "
            "current value plainly; if it later changes, say the correction "
            "out loud as well as fixing the tools.")
    return "\n".join(lines)


# W5 attribution gate (the WHICH-axis of window admission control; user
# ruling 2026-07-16). v1 wording — dev-iterable at most twice (five-target
# discipline), then frozen. Installed only under --attr on; the constant is
# additive so non-attr providers' prompts stay byte-identical.
PROMPT_RB_ATTR = (
    "16. REVISION TARGET BINDING: when the user revises a value, bind it to "
    "the op or action whose declared argument that value belongs to (check "
    "the catalog argument formats). If the value names an argument of an "
    "action you have NOT launched yet, LAUNCH that action carrying the new "
    "value - do NOT patch an open op whose arguments the value does not fit. "
    "Only patch an op when the revised value is a valid replacement for one "
    "of that op's own arguments.")


def install_rb_attr():
    td = tact_core.tact_decider
    if getattr(td, "_RB_ATTR_INSTALLED", False):
        return
    td.SYSTEM_PROMPT = td.SYSTEM_PROMPT + "\n" + PROMPT_RB_ATTR
    td._RB_ATTR_INSTALLED = True


def _snapshot_v24(tx):
    """v2.4 snapshot: identical to tact_core._snapshot_v2 EXCEPT the
    ALREADY-EXECUTED section shows each op's id in the collision-free
    "X<global id>" namespace (never confusable with the local 1..n pending
    ids that killed admission v1), so a decider can address an executing op
    for abort. Bound PER EPISODE (instance attribute) when the episode's
    caps declare snapshot v24 — the class default stays _snapshot_v2, so
    v2.3 builds and every FDB path keep byte-identical prompts."""
    tx._localmap = {}
    parts = []
    if tx.committed:
        parts.append("ALREADY EXECUTED (do NOT launch these again; while one "
                     "is still running you may abort it with "
                     "{\"type\":\"cancel\",\"op_id\":\"X<id>\"}; once "
                     "finished, use its reverse tool):")
        for op in tx.committed:
            parts.append(f"  - id=X{op.op_id} fn={op.fn} "
                         f"args={json.dumps(op.args, ensure_ascii=False)}")
    parts.append("PENDING (not yet executed, patch/cancel by id):")
    if not tx.pending:
        parts.append("  (none)")
    else:
        for local_id, op in enumerate(tx.pending.values(), 1):
            tx._localmap[local_id] = op.op_id
            parts.append(f"  - id={local_id} fn={op.fn} "
                         f"args={json.dumps(op.args, ensure_ascii=False)} "
                         f"status={op.status.value}")
    return "\n".join(parts)


def install_rb(prompt="v3.1", v24=False):
    td = tact_core.tact_decider
    cat = rb_catalog(v24=v24)
    if getattr(td, "_RB_INSTALLED", False):
        return
    td.SYSTEM_PROMPT = td.SYSTEM_PROMPT.replace(td.TOOL_CATALOG, cat)
    td.SYSTEM_PROMPT = td.SYSTEM_PROMPT.replace(
        "You have 12 tools", f"You have {len(TOOLS)} tools")
    td.TOOL_CATALOG = cat
    td._RB_INSTALLED = True
    for fn, spec in TOOLS.items():
        _REV.setdefault(fn, Reversibility[spec["kappa"]])
    import tact_dag
    tact_dag.DAG_TEMPLATES.update({          # RB chain dependency declarations
        "hold_seat": {"search_trains": {"passthrough": {}, "derived": ["train_id"],
                      "key_fields": ["origin", "destination", "date"]}},
        "purchase_ticket": {"hold_seat": {"passthrough": {}, "derived": ["hold_id"],
                            "key_fields": ["train_id", "seat_class"]}},
        "add_item": {"search_catalog": {"passthrough": {}, "derived": ["item_id"],
                     "key_fields": ["query"]}},
        "place_order": {"add_item": {"passthrough": {}, "derived": ["cart_id"],
                        "key_fields": ["item_id", "qty"]}},
        "book_viewing": {"search_rentals": {"passthrough": {}, "derived": ["listing_id"],
                         "key_fields": ["city", "beds", "max_rent"]}},
    })
    if prompt == "v3.1":
        tact_core.install_prompt_v31()


# ---------------------------------------------------------------------------
# timeline: (s, e, text, role) segments from episode cues or the stub model
# ---------------------------------------------------------------------------
def stub_dur(text, lang):
    per = STUB_ZH_S if lang == "zh" else STUB_EN_S
    return max(STUB_MIN_S, min(12.0, per * len(text)))


def plan_segments(ep, use_cues=True):
    cues = ep.get("cues") if use_cues else None
    pieces = ep["pieces"]
    if ep.get("arm") == "B":                 # reactive arm: nominal-timed
        pieces = [q for q in pieces if "at_after_eou" not in q]   # pieces out
    segs = []
    if cues:
        for p, c in zip(pieces, cues):
            segs.append({"s": c["t_start"], "e": c["t_end"],
                         "text": p["text"], "role": p["role"],
                         "voice": p.get("voice"), "lang": p.get("lang")})
    else:
        t, seq_end = 0.0, 0.0
        for p in pieces:
            d = stub_dur(p["text"], p["lang"])
            if "at_after_eou" in p:
                s = seq_end + HOLD + float(p["at_after_eou"])
            else:
                s = max(t, seq_end) + float(p.get("gap_before", 0.0))
                seq_end = s + d
            segs.append({"s": round(s, 3), "e": round(s + d, 3),
                         "text": p["text"], "role": p["role"],
                         "voice": p.get("voice"), "lang": p.get("lang")})
            t = s + d
    return sorted(segs, key=lambda x: x["s"])


class LiveAudio:
    """Arm-B audio assembler: synthesizes pieces on demand and mixes them at
    their segment times; prefix(t) yields the decider's audio input. Segment
    DURATIONS come from the synthesized audio (v2.3: START times too — the
    caller schedules synthesize-first, same structure as the arm-A
    assembler; v2.2 scheduled by stub estimates and only backfilled the
    endpoint, which produced physically impossible self-overlapping speech).
    The episode's seeded perturbation family (rate/gain/scene SNR) is applied
    here, post-synthesis, so TTS cache keys stay clean."""

    def __init__(self, backend, episode=None):
        import array
        self.backend = backend
        self.buf = array.array("h")
        pb = (episode or {}).get("perturb") or {}
        self.rate = pb.get("rate", 1.0)
        self.gain = pb.get("gain_db", 0.0)
        self.snr = pb.get("snr_db")
        self.eid = (episode or {}).get("id", "")
        self.sigma = None

    def place(self, seg):
        from rb.audio import perturb_samples, noise_sigma
        samples, _sr = self.backend.synthesize(
            seg["text"], seg.get("voice") or "cv01", seg.get("lang") or "zh")
        samples = perturb_samples(samples, self.rate, self.gain)
        if self.sigma is None:
            self.sigma = noise_sigma(samples, self.snr)
        i0 = int(seg["s"] * SR)
        need = i0 + len(samples)
        if need > len(self.buf):
            self.buf.extend([0] * (need - len(self.buf)))
        for j, v in enumerate(samples):
            x = self.buf[i0 + j] + v
            self.buf[i0 + j] = max(-32768, min(32767, x))
        seg["e"] = round(seg["s"] + len(samples) / SR, 3)   # true duration

    def prefix(self, t):
        import numpy as np
        from rb.audio import noise_block
        n = int(t * SR)
        if n > len(self.buf):
            self.buf.extend([0] * (n - len(self.buf)))
        x = np.asarray(self.buf[:n], dtype=np.float32)
        if self.sigma and self.sigma > 0.0:
            x = x + noise_block(self.eid, n, self.sigma)
        return np.clip(x, -32768.0, 32767.0) / 32768.0


# ---------------------------------------------------------------------------
# oracle decider — gold-policy sanity arm (no LLM): launch the scenario steps
# with the slot values HEARD so far; patch the pending op when a revision is
# heard; cancel on cancel utterances; ignore bystander speech.
# ---------------------------------------------------------------------------
class OracleDecider:
    def __init__(self, ep):
        self.ep = ep
        self.scn = SCENARIOS[ep["scenario"]]
        self.launched = False
        self.applied_revs = set()
        # slot -> [(step_idx, arg), ...] — MULTI-map: the same slot may feed
        # several steps (fin_transfer: amount -> get_fx_quote AND
        # transfer_funds); a revision must patch EVERY pending op using it
        # (single-value map was the L3/L4 oracle gate failure, 2026-07-16).
        self.slot_args = {}
        for i, st in enumerate(self.scn["steps"]):
            for a, v in st["args"].items():
                if isinstance(v, str) and v.startswith("{"):
                    self.slot_args.setdefault(v.strip("{}"), []).append((i, a))

    def heard(self, segs_done):
        """Slot values after the user pieces heard so far."""
        slots = dict(self.ep["slots"])
        texts = [s["text"] for s in segs_done if s["role"] == "user"]
        blob = "".join(texts)
        revs = []
        for r in self.ep.get("revisions", []):
            if r["by"] == "user" and r["new"] in blob and \
                    (len(texts) > 1 or r["kind"] == "inline"):
                slots[r["slot"]] = r["new"]
                revs.append(r["slot"])
        # The oracle follows the episode's declared action, not a tiny lexical
        # whitelist.  Content-bank paraphrases intentionally include forms
        # such as "Scratch that" / "算了，不弄了"; once their second user
        # segment is heard they are the same L8 cancel action.  Keep the
        # lexical fallback for hand-authored episodes without l8_action.
        cancel = (
            self.ep.get("l8_action") == "cancel" and len(texts) > 1
        ) or any(("别办" in t or "先别" in t or "hold off" in t.lower())
                 for t in texts[1:])
        return slots, revs, cancel

    def __call__(self, tx, segs_done, op_ids):
        slots, revs, cancel = self.heard(segs_done)
        ops = []
        if cancel:
            for oid in list(tx.pending):
                ops.append({"type": "cancel", "op_id": oid})
            return {"dialogue": "stay", "ops": ops, "say": "好的，先不办了。"}
        if not self.launched:
            self.launched = True
            for i, st in enumerate(self.scn["steps"]):
                args = {}
                for a, v in st["args"].items():
                    if isinstance(v, str) and v.startswith("$R"):
                        ref_fn = self.scn["steps"][int(v[2:])]["fn"]
                        args[a] = mint_id(self.ep["id"], ref_fn, 0)
                    elif isinstance(v, str) and v.startswith("{"):
                        sl = v.strip("{}")
                        args[a] = canon_value(sl, slots[sl])
                    else:
                        args[a] = v
                ops.append({"type": "launch", "fn": st["fn"], "args": args})
            self.applied_revs.update(revs)
            return {"dialogue": "stay", "ops": ops, "say": "好的，马上办。"}
        for slot in revs:
            if slot in self.applied_revs or slot not in self.slot_args:
                continue
            self.applied_revs.add(slot)
            for step_i, arg in self.slot_args[slot]:
                fn = self.scn["steps"][step_i]["fn"]
                patched = False
                for oid in tx.pending:
                    if tx.pending[oid].fn == fn:
                        ops.append({"type": "patch", "op_id": oid,
                                    "diff": {arg: canon_value(slot, slots[slot])}})
                        patched = True
                        break
                if patched or not any(op.fn == fn for op in tx.committed):
                    continue
                # v2.3 compensation route (L7 arena): the target op is already
                # COMMITTED — reverse it via its catalog reverse tool, then
                # relaunch with the revised value (nets to forward(new)).
                # v2.4 (caps-gated): FIRST try to abort it in place — if its
                # execution window is still open the abort voids it for free
                # and the reverse call then errors harmlessly ("no live
                # forward"); if it already completed the abort errors and the
                # reverse nets it out. One static op sequence, both branches
                # reach forward(new) — the route taken is the measurement.
                rev_fn = TOOLS[fn].get("reverse")
                if rev_fn is None:
                    continue
                if (self.ep.get("caps") or {}).get("abort_on_cancel"):
                    gid = next((op.op_id for op in tx.committed
                                if op.fn == fn), None)
                    if gid is not None:
                        ops.append({"type": "cancel", "op_id": f"X{gid}"})
                from rb.registry import REVERSE_TARGET_ARG
                ops.append({"type": "launch", "fn": rev_fn,
                            "args": {REVERSE_TARGET_ARG[rev_fn]:
                                     mint_id(self.ep["id"], fn, 0)}})
                new_args = {}
                for a, v in self.scn["steps"][step_i]["args"].items():
                    if isinstance(v, str) and v.startswith("$R"):
                        ref_fn = self.scn["steps"][int(v[2:])]["fn"]
                        new_args[a] = mint_id(self.ep["id"], ref_fn, 0)
                    elif isinstance(v, str) and v.startswith("{"):
                        sl = v.strip("{}")
                        new_args[a] = canon_value(sl, slots[sl])
                    else:
                        new_args[a] = v
                ops.append({"type": "launch", "fn": fn, "args": new_args})
        return {"dialogue": "stay", "ops": ops,
                "say": "已更新。" if ops else ""}


# ---------------------------------------------------------------------------
# episode run (core semantics; mirrors w2r run_example)
# ---------------------------------------------------------------------------
def eta_of(fn, profile):
    cls = "heavy" if profile == "heavy" and TOOLS[fn]["kappa"] != "READ" \
        else TOOLS[fn]["latency"]
    return round(math.exp(LATENCY[cls][0]), 3)


def _resolve_ref(v, by_op, by_step, batch=None, base=None):
    """Resolve chained-arg references at commit time. "$RESULT_<n>.<path>":
    <n> is tried as (i) the 0-based position of a launch in the SAME decision
    batch (what the dev smoke showed the LLM emits for ops it just launched),
    then (ii) a global op_id. The field path is IGNORED — RB results carry a
    single value ({"id": ...}), so any guessed schema ("trains[0].train_id")
    resolves to that id (catalog documents the {"id"} form). Unresolvable ->
    literal (a real, scored system failure)."""
    if not isinstance(v, str) or not v.startswith("$R"):
        return v
    try:
        head, _field = v.split(".", 1) if "." in v else (v, "id")
        if head.startswith("$RESULT_"):
            n = int(head[len("$RESULT_"):])
            res = None
            if batch is not None and n < len(batch):
                res = by_op.get(batch[n])
            elif base is not None and base + n < len(by_step):
                # same-batch 0-based ref committing DURING apply (blocking
                # immediate mode / model-emitted commit ops): batch_of is not
                # populated yet, but this decision's commits land in launch
                # order at by_step[base:], so base+n is that same op.
                res = by_step[base + n]
            if res is None:
                res = by_op.get(n)
            if res is None:
                return v
        elif head.startswith("$RSTEP_"):
            res = by_step[int(head[len("$RSTEP_"):])]
        else:
            return v
        return res["result"].get("id", v)
    except Exception:
        return v


def run_episode(ep, decider, cache=None, mode="tact", delta=1.5, barrier=True,
                fc_mode=None, input_kind="text", audio=None, tts_backend=None,
                infer_nominal=1.0, dag_on=True,
                delta_policy="fixed", stophead=None, fin_cache=None,
                admission=None):
    random.seed(42)
    learned = delta_policy.startswith("learned")
    if learned:
        assert mode == "tact", "--delta-policy requires --system tact"
        assert stophead is not None, "learned:* needs a loaded stophead model"
        import stophead as stophead_mod
        import delta_policy as delta_policy_mod
    import itertools
    import tact.transaction as _txm
    _txm._uid = itertools.count(1)          # per-episode op ids (reproducible runs)
    sandbox = Sandbox(ep["id"], profile=ep.get("profile", "default"),
                      lat_ns=ep.get("lat_ns"))
    tx = Transaction()
    caps = ep.get("caps") or {}             # v2.4 capability gates (v2.3: {})
    if caps.get("snapshot") == "v24":
        # instance-level binding: the class default stays _snapshot_v2, so
        # v2.3 episodes in the same process keep byte-identical prompts.
        tx.snapshot_for_prompt = (lambda _tx=tx: _snapshot_v24(_tx))
    ledger = WindowLedger(delta, barrier=barrier)
    dag = comp_reg = None
    if dag_on and mode == "tact":
        from tact_dag import OpDag, CompensationRegistry
        dag = OpDag(ledger)
        comp_reg = CompensationRegistry()
    sim = ReactiveUser(ep) if ep["arm"] == "B" else None
    live = None
    if input_kind == "audio" and ep["arm"] == "B":
        assert tts_backend is not None, "arm B audio runs need --tts"
        # v2.3 synthesize-first scheduling: true duration decides the next
        # start (arm-A assembler structure) — no stub estimates on the clock.
        live = LiveAudio(tts_backend, ep)
        segs = []
        t = 0.0
        for p in ep["pieces"]:
            if "at_after_eou" in p:
                continue                      # arm B: lifecycle pieces -> events
            s = t + float(p.get("gap_before", 0.0))
            seg = {"s": round(s, 3), "e": None, "text": p["text"],
                   "role": p["role"], "voice": p.get("voice"),
                   "lang": p.get("lang")}
            live.place(seg)                   # sets true e, mixes at s
            segs.append(seg)
            t = seg["e"]
    else:
        segs = plan_segments(ep, use_cues=input_kind == "audio"
                             and ep["arm"] == "A")
    say_events, commits, decisions = [], [], []
    trace_events = []
    results_by_op, results_by_step = {}, []
    batch_of = {}                  # op_id -> ordered launch op_ids of its decision
    cur_base = {"v": 0}            # len(results_by_step) at the current apply

    def do_commit(op_id, t_nominal, t_actual):
        if op_id not in tx.pending:
            return
        fn = tx.pending[op_id].fn
        lat = sandbox.latency_of(fn) if fn in TOOLS else 0.0
        ledger.close(op_id)

        def _exec(f, a):
            batch = batch_of.get(op_id)
            ra = {k: _resolve_ref(v, results_by_op, results_by_step, batch,
                                  base=cur_base["v"])
                  for k, v in (a or {}).items()}
            res = sandbox.execute(f, ra, idem_key=f"{ep['id']}:{op_id}",
                                  t=t_nominal)
            if res.get("status") == "success":
                results_by_op[op_id] = res
                results_by_step.append(res)
            return res
        tx.commit(op_id, _exec, t=t_nominal)
        commits.append({"op_id": op_id, "fn": fn, "t_commit": round(t_nominal, 3),
                        "tool_nominal_s": lat,
                        "deferred_s": round(max(0.0, t_actual - t_nominal), 3)})
        trace_events.append({"event": "tact_op_applied", "t": t_nominal,
                             "data": {"t_audio": t_nominal,
                                      "op": {"type": "commit", "fn": fn}}})

    trace_sent = {"n": 0}                  # feed_sim watermark over trace_events

    def feed_sim(evts, t_now):
        """arm B: engine events -> reactive-user actions -> new segments.
        Injections never overlap existing speech (physical single mouth)."""
        if sim is None:
            return
        for ev in evts:
            for act in sim.on_event(ev):
                p = act["piece"]
                floor = max((sg["e"] for sg in segs), default=0.0) + 0.05
                s = act["at"] if act["at"] >= floor else floor
                s = max(s, t_now)
                seg = {"s": round(s, 3), "e": None,
                       "text": p["text"], "role": p["role"],
                       "voice": p.get("voice"), "lang": p.get("lang")}
                if live is not None:
                    live.place(seg)          # true duration
                else:
                    seg["e"] = round(seg["s"] + stub_dur(p["text"], p["lang"]), 3)
                lo = 0
                while lo < len(segs) and segs[lo]["s"] <= seg["s"]:
                    lo += 1
                segs.insert(lo, seg)

    def feed_sim_traces(t_now):
        """Deliver trace events (commits) accrued since the last watermark —
        the v2.2 `trace_events[len(trace_events):]` was a constant empty
        slice, so the `committed` lifecycle anchor never fired."""
        mark = trace_sent["n"]
        trace_sent["n"] = len(trace_events)
        if mark < trace_sent["n"]:
            feed_sim(trace_events[mark:], t_now)

    cursor, i, n_eou = 0.0, 0, 0
    tuple_segs = lambda: [(x["s"], x["e"]) for x in segs]        # noqa: E731
    while i < len(segs):
        e_i = segs[i]["e"]
        nxt = segs[i + 1]["s"] if i + 1 < len(segs) else None
        last = nxt is None
        if (not last and nxt - e_i < HOLD) or (mode == "blocking" and not last):
            i += 1
            continue
        t_eou = e_i + HOLD
        n_eou += 1
        advance_over(ledger, cursor, t_eou, tuple_segs(), do_commit)
        cursor = max(cursor, t_eou)
        seen = segs[:i + 1]
        ledger.begin_decision(i, set(tx.pending))
        if isinstance(decider, OracleDecider):
            dec, infer = decider(tx, seen, len(tx.pending) + len(tx.committed)), 0.05
        else:
            if input_kind == "audio":
                prefix = live.prefix(e_i) if live is not None \
                    else audio[: int(e_i * SR)]
                msgs = build_decider_messages(tx, "LISTEN", audio=prefix)
            else:
                blob = " / ".join(s["text"] for s in seen)
                msgs = build_decider_messages(tx, "LISTEN", user_text=blob)
            dec, infer = decide_from_msgs(cache.call, msgs)
        t_dec = t_eou + (infer_nominal if infer_nominal is not None else infer)
        advance_over(ledger, cursor, t_dec, tuple_segs(), do_commit)
        cursor = max(cursor, t_dec)
        say = dec.get("say", "") if mode == "blocking" else ack_fallback(dec)
        # W4 learned stopping head (batch 2): per-op objection windows from the
        # FROZEN model. Finality is judged AFTER the decision call on the same
        # audio tail convention as the FDB harness (separate cache; its wall
        # time never advances the audio clock — deployed it overlaps the hold).
        # The decider messages are untouched: decision cache keys stay caliber.
        # Admission control (patch schema gate), rb_design 16.7/16.8:
        #   schema   = v1 (RETIRED after test-911 R-ADM1 — kept ONLY for
        #              archival replay of the rbt23_*_adm arms; gated at the
        #              raw layer, which misread snapshot-local ids)
        #   schema11 = v1.1: same rule at the POST-RESOLUTION layer, using
        #              the engine's own resolve_ref + nested-args unwrap.
        adm_audit = None
        if admission == "schema" and mode == "tact":
            from admission import admit_decision_ops
            new_ops, adm_audit = admit_decision_ops(
                dec.get("ops") or [],
                {oid: tx.pending[oid].fn for oid in tx.pending},
                {f: spec["required"] for f, spec in TOOLS.items()})
            if adm_audit:
                dec = dict(dec)
                dec["ops"] = new_ops
        elif admission == "schema11" and mode == "tact":
            from admission import admit_decision_ops_v11
            from tact_core import resolve_ref as _rref

            def _resolve_pending(op):
                r = _rref(tx, op)
                return r if r in tx.pending else None
            new_ops, adm_audit = admit_decision_ops_v11(
                dec.get("ops") or [], _resolve_pending,
                {oid: tx.pending[oid].fn for oid in tx.pending},
                {f: spec["required"] for f, spec in TOOLS.items()})
            if adm_audit:
                dec = dict(dec)
                dec["ops"] = new_ops
        # v2.4 abort semantics (caps-gated; rb_design §17.4): a cancel whose
        # op_id is "X<id>" addresses an ALREADY-EXECUTED op (the _snapshot_v24
        # namespace — collision-free with local pending ids). If that op's
        # sandbox execution window is still open, the effect is aborted in
        # place (voided, fee-free); otherwise the attempt fails and the
        # catalog reverse tool remains the only route. Intercepted BEFORE
        # apply_decision_ops, which would silently drop non-pending cancels.
        abort_recs = []
        if caps.get("abort_on_cancel"):
            kept_ops = []
            for op_ in dec.get("ops") or []:
                oid_s = op_.get("op_id") if isinstance(op_, dict) else None
                if isinstance(op_, dict) and op_.get("type") == "cancel" \
                        and isinstance(oid_s, str) \
                        and re.fullmatch(r"[Xx]\d+", oid_s.strip()):
                    gid = int(oid_s.strip()[1:])
                    res_e = results_by_op.get(gid)
                    rid = ((res_e or {}).get("result") or {}).get("id")
                    r_ab = sandbox.abort(rid, t=t_dec) if rid else \
                        {"status": "error", "error": "unknown executed op"}
                    rec = {"type": "abort", "op_id": oid_s, "target": rid,
                           "ok": r_ab.get("status") == "success"}
                    if not rec["ok"]:
                        rec["error"] = r_ab.get("error")
                    abort_recs.append(rec)
                else:
                    kept_ops.append(op_)
            if abort_recs:
                dec = dict(dec)
                dec["ops"] = kept_ops
        delta_fn = None
        finality = None
        fin_parsed = True
        if learned:
            if input_kind == "audio" and fin_cache is not None:
                if live is not None:
                    pref = live.prefix(e_i)
                    tail = pref[int(max(0.0, e_i - delta_policy_mod.
                                        FINALITY_TAIL_S) * SR):]
                else:
                    tail = audio[int(max(0.0, e_i - delta_policy_mod.
                                         FINALITY_TAIL_S) * SR): int(e_i * SR)]
                from decider_b import _audio_block
                fraw, _fin_infer = fin_cache.call(
                    delta_policy_mod.build_finality_msgs(_audio_block(tail)))
                finality, fin_parsed = delta_policy_mod.parse_finality(fraw)
            else:       # text / cache-less oracle smoke: declared, deterministic
                finality = delta_policy_mod.FINALITY_FALLBACK
            s0 = segs[i]["s"]
            ctx = {"eou_idx": n_eou - 1, "utt_dur": round(e_i - s0, 3),
                   "gap_prev": round(s0 - (segs[i - 1]["e"] if i else 0.0), 3),
                   "n_prior_ops": len(tx.pending) + len(tx.committed),
                   "finality": finality,
                   "domain": TOOLS[SCENARIOS[ep["scenario"]]["steps"][0]["fn"]
                                   ]["domain"]}
            delta_fn = stophead_mod.make_learned_delta_fn(stophead, ctx)
        cur_base["v"] = len(results_by_step)
        applied = apply_decision_ops(tx, ledger, dec, t_dec,
                                     immediate=(mode == "blocking" or delta <= 0),
                                     commit_cb=do_commit,
                                     dag=dag, comp_registry=comp_reg,
                                     delta_fn=delta_fn)
        ledger.end_decision(i)
        ledger.sweep(t_dec, do_commit, cause="decision_done")
        launches = [a for a in applied if a.get("type") == "launch"]
        batch_ids = [a.get("op_id") for a in launches if a.get("op_id") is not None]
        for oid_ in batch_ids:
            batch_of[oid_] = batch_ids
        tier = None
        if fc_mode and mode != "blocking" and launches:
            from floor_policy import commit_tier, tier_utterance, worst_kappa
            fns = [a.get("fn", "") for a in launches]
            eta = max((eta_of(f, ep.get("profile", "default"))
                       for f in fns if f in TOOLS), default=None)
            tier = ("silence" if fc_mode == "always_silent" else
                    "filler" if fc_mode == "always_filler" else
                    commit_tier(eta_s=eta, kappa=worst_kappa(fns)))
            say = tier_utterance(tier, lang=ep["lang"], fns=fns, eta_s=eta)
        if say:
            say_events.append((round(t_dec, 3), say))
        dec_entry = {"i": i, "t_eou": round(t_eou, 3),
                     "ops": applied + abort_recs,
                     "say": say, **({"fc_tier": tier} if tier else {})}
        if learned:                     # audit fields (absent on the frozen path)
            ow = {}
            for a in applied:
                oid = a.get("op_id")
                if a["type"] in ("launch", "patch") and oid is not None:
                    rem = ledger.remaining(oid)
                    if rem is not None:
                        ow[str(oid)] = round(rem, 3)
            dec_entry["op_windows"] = ow
            dec_entry["finality"] = finality
            if not fin_parsed:
                dec_entry["finality_unparsed"] = True
        if adm_audit:                   # admission audit (flag-gated path only)
            dec_entry["admission"] = adm_audit
        decisions.append(dec_entry)
        evts = [{"event": "tact_eou", "t": t_eou}]
        for a in launches:
            evts.append({"event": "tact_op_applied", "t": t_dec,
                         "data": {"t_audio": t_dec,
                                  "op": {"type": "launch", "fn": a.get("fn")}}})
        if say:
            evts.append({"event": "tts_start", "t": t_dec})
        feed_sim(evts, t_dec)
        feed_sim_traces(t_dec)               # commits since last decision
        i += 1
        if i >= len(segs):
            # end of current speech: close remaining windows on tail silence,
            # then deliver their commit events — a `committed`-anchored user
            # event may still inject speech and resume the loop (v2.3).
            advance_over(ledger, cursor, math.inf, tuple_segs(), do_commit)
            ledger.sweep(cursor, do_commit, cause="finalize")
            feed_sim_traces(cursor)

    t_user_end = max((s["e"] for s in segs if s["role"] == "user"), default=0.0)
    done = max((c["t_commit"] + c["tool_nominal_s"] for c in commits),
               default=t_user_end)
    done_s = round(max(0.0, done - t_user_end), 3)
    final_says = [t for t, _ in say_events if t >= t_user_end]
    first = round(min(final_says) - t_user_end, 3) if final_says else done_s
    row = rb_scorer.score_episode(
        ep, sandbox.calls, sandbox.state, say_events, done_s)
    row.update({"first_response_s": first, "n_eou": n_eou,
                "n_commits": len(commits), "fees": sandbox.fees,
                "decisions": decisions, "say_events": say_events,
                "segs": [(s["s"], s["e"], s["role"]) for s in segs]})
    if "abort_feasible" in ep:                       # v2.4 L15 route analysis
        row["abort_feasible"] = ep["abort_feasible"]
    if ep["arm"] == "B":
        # v2.3 arm-B timing acceptance receipt (arm-A-grade): overlaps must
        # be zero by construction; measured gaps are the re-binning truth.
        user = sorted((s["s"], s["e"]) for s in segs if s["role"] == "user")
        row["armb_timing"] = {
            "user_overlaps": sum(1 for a, b in zip(user, user[1:])
                                 if b[0] < a[1] - 1e-9),
            "measured_gaps": [round(b[0] - a[1], 3)
                              for a, b in zip(user, user[1:])]}
    return row


def aggregate(rows):
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["layer"]].append(r)
    def rate(xs, k):                                              # noqa: E306
        return round(sum(1 for x in xs if x[k]) / max(1, len(xs)), 4)
    def p50(xs, k):                                               # noqa: E306
        v = sorted(x[k] for x in xs)
        return v[len(v) // 2] if v else None
    rep = {"n": len(rows),
           "exact": rate(rows, "exact"),
           "state_verbatim": rate(rows, "state_verbatim"),
           "state_normalized": rate(rows, "state_normalized"),
           "U_mean": round(sum(r["U"] for r in rows) / max(1, len(rows)), 4),
           "wrong_commits": sum(r["commit_repair"]["wrong_commits"] for r in rows),
           "unrepaired": sum(r["commit_repair"]["unrepaired"] for r in rows),
           "first_p50": p50(rows, "first_response_s"),
           "done_p50": p50(rows, "done_s"),
           "by_layer": {L: {"n": len(xs), "exact": rate(xs, "exact"),
                            "U": round(sum(x["U"] for x in xs) / len(xs), 4)}
                        for L, xs in sorted(by.items())}}
    bt = [r["armb_timing"] for r in rows if "armb_timing" in r]
    if bt:
        rep["armb_timing"] = {
            "episodes_with_overlap": sum(1 for x in bt if x["user_overlaps"]),
            "total_overlaps": sum(x["user_overlaps"] for x in bt)}
    # v2.4 preregistered instruments (in-report, not ad-hoc):
    # WHO axis over L10 (benign adoption / adversarial intrusion double rate)
    l10 = by.get("L10", [])
    ben = [r for r in l10 if r.get("rev_adopted") is not None]
    adv = [r for r in l10 if r.get("intruder_present") is not None]
    if ben or adv:
        rep["who_axis"] = {
            "benign": {"n": len(ben),
                       "adopted": round(sum(1 for r in ben if r["rev_adopted"])
                                        / max(1, len(ben)), 4)},
            "adversarial": {"n": len(adv),
                            "intruded": round(sum(1 for r in adv
                                                  if r["intruder_present"])
                                              / max(1, len(adv)), 4)}}
    # lifecycle pair grid over L13 (per (who, state) cell)
    l13 = [r for r in rows if r.get("pair")]
    if l13:
        cells = defaultdict(list)
        for r in l13:
            cells[(r["pair"]["who"], r["pair"]["state"])].append(r)
        rep["pair_axis"] = {
            f"{w}:{s}": {"n": len(xs), "exact": rate(xs, "exact"),
                         "intruded": round(sum(1 for x in xs
                                               if x.get("intruder_present"))
                                           / max(1, len(xs)), 4)
                         if w == "bystander" else None}
            for (w, s), xs in sorted(cells.items())}
    # abort/compensation route counters (L15 + everywhere they occur)
    ab = sum(r.get("aborted_ops", 0) for r in rows)
    cc = sum(r.get("comp_calls", 0) for r in rows)
    if ab or cc or "L15" in by:
        l15 = by.get("L15", [])
        rep["route_axis"] = {
            "aborted_ops": ab, "comp_calls": cc,
            "l15": {"n": len(l15),
                    "exact": rate(l15, "exact") if l15 else None,
                    "aborted": sum(r.get("aborted_ops", 0) for r in l15),
                    "feasible": sum(1 for r in l15 if r.get("abort_feasible"))}}
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", default="exp/rb/build_v2")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--arm", default="A", choices=["A", "B"])
    ap.add_argument("--system", default="tact", choices=["tact", "blocking"])
    ap.add_argument("--delta", type=float, default=1.5)
    ap.add_argument("--commit-barrier", default="on", choices=["on", "off"])
    ap.add_argument("--floor-commit-tiers", default=None,
                    choices=[None, "v1", "always_filler", "always_silent"])
    ap.add_argument("--decider", default="llm", choices=["llm", "oracle"])
    ap.add_argument("--input", default="audio", choices=["audio", "text"])
    ap.add_argument("--prompt", default="v3.1")
    ap.add_argument("--infer-nominal", type=float, default=1.0)
    ap.add_argument("--dag", default="on", choices=["on", "off"])
    ap.add_argument("--tts", default=None, choices=[None, "qwen", "stub"],
                    help="arm-B audio synthesis backend")
    ap.add_argument("--attr", default="off", choices=["on", "off"],
                    help="W5 attribution PROMPT rule (SEALED at dev round 1, "
                         "rb_design 16.6 — kept for archival replay only). "
                         "Default off = prompt byte-identical.")
    ap.add_argument("--admission", default=None,
                    choices=[None, "schema", "schema11"],
                    help="Patch schema gate. schema = v1, RETIRED (archival "
                         "replay of rbt23_*_adm only — raw-layer gating "
                         "misread snapshot-local ids, test-911 R-ADM1). "
                         "schema11 = v1.1: same rejection rule at the post-"
                         "resolution layer (engine resolve_ref + nested-args "
                         "unwrap). Default off = frozen path byte-identical.")
    ap.add_argument("--delta-policy", default="fixed",
                    choices=["fixed", "learned:v2"],
                    help="fixed (default; frozen batch-1 path, bit-identical) "
                         "| learned:v2 = frozen two-stage stopping head "
                         "(test-897 batch 2; needs --stophead-model; judges "
                         "finality on the audio tail via --finality-cache). "
                         "EVALUATION ONLY: never tune theta/weights on RB.")
    ap.add_argument("--stophead-model", default="exp/w4/stophead_v2.json",
                    help="frozen model JSON (v2 / C0 / C1 pi-point remaps)")
    ap.add_argument("--finality-cache", default=None,
                    help="finality-call cache path (default: "
                         "<build>/finality_cache_rb_<arm>.json — shared across "
                         "learned providers; arm-A audio is provider-invariant)")
    ap.add_argument("--provider", required=False, default="rbdev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.split == "test":                     # design §8-2: scorer frozen
        fz = json.loads((ROOT / "exp/rb/scorer_freeze.json").read_text())
        import hashlib as _h
        for f, want in fz["hashes"].items():
            got = _h.sha256((ROOT / f).read_bytes()).hexdigest()
            assert got == want, f"scorer freeze violated: {f} changed"
    # v2.4 builds get the extended catalog (abort/reverse/confirm guidance);
    # v2.3 builds keep the exact v2.3 catalog bytes — one process runs one
    # build, so the process-global install is safe to gate on its manifest.
    mpath = Path(args.build) / "manifest.json"
    build_ver = ""
    if mpath.exists():
        build_ver = json.loads(mpath.read_text()).get("version", "")
    install_rb(args.prompt, v24=build_ver >= "rb_v2.4")
    if args.admission and args.system != "tact":
        ap.error("--admission requires --system tact (patches are a tact op)")
    if args.admission and "adm" not in args.provider:
        print("WARNING: --admission without an adm-tagged provider; "
              "recommend *_adm so grids stay separable.")
    if args.attr == "on":
        install_rb_attr()
        if "attr" not in args.provider:
            print("WARNING: --attr on without an attr-tagged provider; "
                  "recommend *_attr so grids stay separable.")
    stophead = fin_cache = None
    if args.delta_policy != "fixed":
        if args.system != "tact":
            ap.error("--delta-policy requires --system tact")
        if args.delta <= 0:
            ap.error("--delta-policy needs --delta > 0 (delta<=0 selects the "
                     "immediate-commit path and would bypass per-op windows)")
        if args.decider == "llm" and args.input != "audio":
            ap.error("learned + llm requires --input audio (finality is an "
                     "audio-tail judgment; text is smoke-only via oracle)")
        from rb.learned import install_stophead_rb, load_head, head_summary
        n_inst = install_stophead_rb()
        stophead = load_head(ROOT / args.stophead_model, expect=args.delta_policy)
        print(f"stophead: {json.dumps(head_summary(stophead))} "
              f"| RB required-args installed: {n_inst}")
        if not args.provider.startswith(("rbtest_lh", "rbdev_lh", "lh")):
            print("WARNING: learned provider without an lh tag; recommend "
                  "rbtest_lh_* / rbdev_lh_* so grids stay separable.")
    cache = None
    if args.decider == "llm":
        sys.path.insert(0, str(ROOT / "scripts"))
        from w2r_stream_replay import DecisionCache
        cache = DecisionCache(Path(args.build) / f"decision_cache_{args.provider}.json")
    if args.delta_policy != "fixed" and args.decider == "llm":
        from w2r_stream_replay import DecisionCache as _FC
        fin_path = args.finality_cache or str(
            Path(args.build) / f"finality_cache_rb_{args.arm.lower()}.json")
        fin_cache = _FC(fin_path)
    tts_backend = None
    if args.tts == "qwen":
        from rb.audio import QwenTTSBackend
        tts_backend = QwenTTSBackend()
    elif args.tts == "stub":
        from rb.audio import SilenceStub
        tts_backend = SilenceStub()
    epdir = Path(args.build) / "episodes"
    outdir = Path(args.build) / "results" / args.provider
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(epdir.glob(f"{args.arm}_*.json")):
        ep = json.loads(p.read_text())
        if args.split != "all" and ep["split"] != args.split:
            continue
        if args.limit and len(rows) >= args.limit:
            break
        audio = None
        if args.input == "audio" and ep["arm"] == "A":
            import soundfile as sf
            wav = Path(args.build) / "audio" / f"{ep['id']}.wav"
            if not wav.exists():
                print(f"skip {ep['id']}: no audio (build with --audio qwen)")
                continue
            audio, _sr = sf.read(str(wav), dtype="float32")
        decider = OracleDecider(ep) if args.decider == "oracle" else "llm"
        kw = dict(cache=cache, mode=args.system, delta=args.delta,
                  barrier=args.commit_barrier == "on",
                  fc_mode=args.floor_commit_tiers,
                  input_kind=args.input, audio=audio,
                  infer_nominal=args.infer_nominal,
                  dag_on=args.dag == "on", tts_backend=tts_backend,
                  delta_policy=args.delta_policy, stophead=stophead,
                  fin_cache=fin_cache, admission=args.admission)
        row = run_episode(ep, decider, **kw) if args.decider == "oracle" \
            else run_episode(ep, "llm", **kw)
        (outdir / f"{ep['id']}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=1))
        rows.append(row)
    if cache is not None:
        cache.save()
        print(f"decision cache: {cache.hits} hits / {cache.misses} misses")
    if fin_cache is not None:
        fin_cache.save()
        print(f"finality cache: {fin_cache.hits} hits / {fin_cache.misses} misses")
    rep = aggregate(rows)
    rep["provider"] = args.provider
    rep["config"] = {k: getattr(args, k) for k in
                     ("arm", "system", "delta", "commit_barrier", "input",
                      "decider", "floor_commit_tiers", "split", "infer_nominal",
                      "dag", "attr", "admission")}
    if args.delta_policy != "fixed":    # absent on frozen batch-1 reports
        from rb.learned import head_summary
        rep["config"]["delta_policy"] = args.delta_policy
        rep["config"]["stophead_model"] = args.stophead_model
        rep["config"]["stophead"] = head_summary(stophead)
        prot = sum(1 for r in rows for d in r["decisions"]
                   for w in d.get("op_windows", {}).values() if w > 0)
        tot = sum(len(d.get("op_windows", {})) for r in rows
                  for d in r["decisions"])
        rep["learned_windows"] = {"ops": tot, "protected": prot,
                                  "protect_rate": round(prot / tot, 4) if tot
                                  else None}
    (Path(args.build) / f"rb_report_{args.provider}.json").write_text(
        json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


def selftest():
    from rb.generator import (make_episode, config_hash, build_all,
                              ARM_A_QUOTA, ARM_B_QUOTA)
    from rb.grammar import L13_STATES as L13_STATES_
    install_rb("v3.1")
    ch = config_hash()
    ck = {}

    def run_oracle(arm, layer, idx, **kw):
        ep = make_episode(arm, layer, idx, ch)
        return ep, run_episode(ep, OracleDecider(ep), mode="tact",
                               input_kind="text", **kw)

    ep2, r2 = run_oracle("A", "L2", 0)
    ck["l2_exact"] = r2["exact"]
    ep4, r4 = run_oracle("A", "L4", 0)
    ck["l4_exact_patch_rescued"] = r4["exact"]
    _, r4f = run_oracle("A", "L4", 1)           # finance: shared-slot chain
    ck["l4_shared_slot_multimap"] = r4f["exact"]
    _, rb5 = run_oracle("B", "L4", 5)           # the reported B_L4_0005 case
    ck["b_l4_0005_regression"] = rb5["exact"]
    _, r10 = run_oracle("A", "L10", 0)          # bystander command ignored
    ck["l10_bystander_ignored"] = r10["exact"]
    _, r10b = run_oracle("A", "L10", 2)         # benign user revision applies
    ck["l10_benign_applied"] = r10b["exact"]
    ep5, r5 = run_oracle("A", "L5", 3)
    ck["l5_scored"] = isinstance(r5["exact"], bool) and r5["n_eou"] >= 2
    epb, rb1 = run_oracle("B", "L8", 2)         # progress query event fires
    _, rb2 = run_oracle("B", "L8", 2)
    ck["armb_deterministic"] = json.dumps(rb1, sort_keys=True) == \
        json.dumps(rb2, sort_keys=True)
    ck["armb_event_injected"] = rb1["n_eou"] >= 2
    # bank-paraphrased cancel — pick a cancel cell whose event lands while
    # the window is still open BY CONSTRUCTION (frac x anchor wall < 1.2s;
    # post-commit cancel of a chain is a designed ceiling: committed READs
    # have no reverse, so gold=[] is unreachable there — draw-dependent idx
    # made the v2.3 form of this test luck-sensitive).
    def _precommit_cancel_idx():
        for k in range(1, 60, 3):                    # idx%3==1 -> cancel
            e = make_episode("B", "L8", k, ch)
            i_nr = next((j for j, fn in enumerate(e["scenario_steps"])
                         if TOOLS[fn]["kappa"] != "READ"), 0)
            if e["events"] and e["events"][0]["offset"] * \
                    e["step_latencies"][i_nr] < 1.2:
                return k
        return None
    kbc = _precommit_cancel_idx()
    epbc, rbc = run_oracle("B", "L8", kbc)
    ck["v23_bank_cancel_oracle"] = (
        kbc is not None and epbc["l8_action"] == "cancel" and rbc["exact"])
    _, rfc = run_oracle("A", "L9", 0, fc_mode="v1")
    tiers = [d.get("fc_tier") for d in rfc["decisions"] if d.get("fc_tier")]
    ck["fc_tier_recorded"] = bool(tiers)
    rep = aggregate([r2, r4, r10, r5])
    ck["aggregate_shape"] = "by_layer" in rep and rep["n"] == 4
    ck["blocking_runs"] = run_episode(make_episode("A", "L5", 1, ch),
                                      OracleDecider(make_episode("A", "L5", 1, ch)),
                                      mode="blocking", input_kind="text",
                                      )["n_eou"] == 1
    # blocking same-batch $RESULT refs (immediate commits) must resolve: an
    # L3 chain episode decided ONCE post-hoc must be exact under blocking.
    epb3 = make_episode("A", "L3", 2, ch)
    rb3 = run_episode(epb3, OracleDecider(epb3), mode="blocking",
                      input_kind="text")
    ck["blocking_chain_refs_resolve"] = rb3["exact"]
    # unit: same-batch 0-based $RESULT under immediate commits (base fallback)
    steps = [{"result": {"id": "AAA"}}, {"result": {"id": "BBB"}}]
    ck["resolver_base_fallback"] = (
        _resolve_ref("$RESULT_0.trains[0].id", {}, steps, None, base=0) == "AAA"
        and _resolve_ref("$RESULT_1.id", {}, steps, None, base=0) == "BBB"
        and _resolve_ref("$RESULT_7.id", {}, steps, None, base=0) == "$RESULT_7.id")

    # -- learned-head RB adaptation (test-897 batch 2; frozen v2 weights) ----
    from rb.learned import install_stophead_rb, load_head, RB_REQUIRED_ARGS
    import stophead as sh_mod
    install_stophead_rb()
    ck["lh_required_args_installed"] = all(
        fn in sh_mod.REQUIRED_ARGS for fn in RB_REQUIRED_ARGS) and \
        "search_flights" in sh_mod.REQUIRED_ARGS          # FDB keys intact
    ck["lh_kappa_resolves"] = all(
        sh_mod.kappa_name(fn) in sh_mod.KAPPAS for fn in RB_REQUIRED_ARGS)
    v2 = load_head(ROOT / "exp/w4/stophead_v2.json")
    base_ctx = {"eou_idx": 0, "utt_dur": 2.0, "gap_prev": 0.0,
                "n_prior_ops": 0, "domain": "travel", "kappa": "REV",
                "slots_missing": 0, "chain_dep": 0}
    r_unf = v2.risk({**base_ctx, "finality": "unfinished"})
    r_fin = v2.risk({**base_ctx, "finality": "final"})
    ck["lh_finality_feature_flows"] = r_unf > r_fin       # frozen v2 loading
    # protect-all theta clone == fixed delta* arm on scored fields (structural
    # bound of the twostage policy; window audit fields differ by design)
    protect_all = sh_mod.StopHead({**v2.d, "theta": -1.0})
    commit_now = sh_mod.StopHead({**v2.d, "theta": 1.1})
    ep4l = make_episode("A", "L4", 0, ch)
    fixed = run_episode(ep4l, OracleDecider(ep4l), mode="tact",
                        input_kind="text")
    pall = run_episode(ep4l, OracleDecider(ep4l), mode="tact",
                       input_kind="text", delta_policy="learned:v2",
                       stophead=protect_all)
    scored = ("exact", "state_verbatim", "state_normalized", "U", "done_s",
              "first_response_s", "n_commits", "fees", "comp_cost")
    ck["lh_protect_all_equals_fixed"] = all(
        fixed[k] == pall[k] for k in scored) and all(
        w == 1.5 for d in pall["decisions"]
        for w in d.get("op_windows", {}).values())
    cnow = run_episode(ep4l, OracleDecider(ep4l), mode="tact",
                       input_kind="text", delta_policy="learned:v2",
                       stophead=commit_now)
    # v2.4 economics update: abort is first-class (caps-gated), so an ORACLE
    # under commit-now can escape through the still-open EXECUTION window
    # (abort + relaunch, fee-free) where v2.3 had to pay compensation or
    # lose. The fixed window still patches with ZERO aborts — the window is
    # the content-blind mechanism; the abort route requires knowing which
    # executed op the revision binds to (the attribution capability LLMs
    # lack, H-COMP2/H-ABORT2). Assert the mechanism split explicitly.
    ck["lh_commit_now_abort_escape"] = (
        fixed["exact"] and fixed["fees"] == 0 and fixed["comp_cost"] == 0.0
        and fixed["aborted_ops"] == 0
        and all(w == 0.0 for d in cnow["decisions"]
                for w in d.get("op_windows", {}).values())
        and ((cnow["exact"] and cnow["aborted_ops"] >= 1)
             or (not cnow["exact"])
             or cnow["fees"] > 0 or cnow["comp_cost"] > 0.0))
    cnow2 = run_episode(ep4l, OracleDecider(ep4l), mode="tact",
                        input_kind="text", delta_policy="learned:v2",
                        stophead=commit_now)
    ck["lh_deterministic"] = json.dumps(cnow, sort_keys=True) == \
        json.dumps(cnow2, sort_keys=True)
    ck["lh_frozen_path_no_audit_keys"] = not any(
        "op_windows" in d or "finality" in d for d in fixed["decisions"])

    # -- v2.3: transactional arena + arm-B mechanics + attribution gate ------
    sb = Sandbox("t_v23")
    r1 = sb.execute("reserve_hotel", {"city": "杭州", "checkin": "5月3日",
                                      "nights": 2}, t=0.0)
    rid1 = r1["result"]["id"]
    rr = sb.execute("cancel_hotel", {"booking_id": rid1})
    from rb.scorer import net_calls, comp_cost as _cc, score_state as _ss
    ck["v23_reverse_netting"] = (rr["status"] == "success"
                                 and sb.live_state() == {}
                                 and sb.fees == 1
                                 and net_calls(sb.calls, sb.state) == []
                                 and _ss(sb.state, {})
                                 and _cc(sb.state) == 4.0)
    sb2 = Sandbox("t_v23b")
    r2b = sb2.execute("transfer_funds", {"from_acct": "checking",
                                         "to_acct": "savings", "amount": 800},
                      t=10.0)
    ca = sb2.state[f"transfer_funds#{r2b['result']['id']}"]["completes_at"]
    a1 = sb2.abort(r2b["result"]["id"], t=10.1)
    a2 = sb2.abort(r2b["result"]["id"], t=10.2)
    r2c = sb2.execute("purchase_ticket", {"hold_id": "HO123456",
                                          "passenger": "Alex Chen"}, t=0.0)
    a3 = sb2.abort(r2c["result"]["id"], t=0.01)
    ck["v23_abort_primitive"] = (ca > 10.0 and a1["status"] == "success"
                                 and a2["status"] == "error"
                                 and a3["status"] == "error"
                                 and _cc(sb2.state) == 0.0)
    ep7, r7 = run_oracle("A", "L7", 3)          # idx 3 -> travel (COMP terminal)
    ck["v23_l7_comp_oracle"] = (r7["exact"] and r7["fees"] == 1
                                and r7["comp_cost"] == 4.0
                                and ep7["revisions"][0]["gap"] > 3.4)
    r7b = run_episode(ep7, OracleDecider(ep7), mode="blocking",
                      input_kind="text")
    ck["v23_l7_patchless_blocking"] = r7b["exact"] and r7b["fees"] == 0
    ep12, r12 = run_oracle("A", "L12", 0)
    step0 = {v.strip("{}") for v in
             __import__("rb.registry", fromlist=["SCENARIOS"]).SCENARIOS[
                 ep12["scenario"]]["steps"][0]["args"].values()
             if isinstance(v, str) and v.startswith("{")}
    ck["v23_l12_step2_slot"] = r12["exact"] and \
        ep12["revisions"][0]["slot"] not in step0
    epb6, rb6 = run_oracle("B", "L6", 1)
    segs_user = [s for s in rb6["segs"] if s[2] == "user"]
    ck["v23_b_l6_committed_anchor"] = (
        len(epb6["events"]) == 2 and len(segs_user) == 3
        and rb6["exact"]
        and epb6["events"][1]["text"].count(epb6["revisions"][1]["new"]) > 0)
    epb11, rb11 = run_oracle("B", "L11", 2)
    ck["v23_b_l11_tts_barge"] = (rb11["exact"]
                                 and len([s for s in rb11["segs"]
                                          if s[2] == "user"]) == 2
                                 and rb11["armb_timing"]["user_overlaps"] == 0)
    ck["v23_armb_receipts"] = all(
        r.get("armb_timing", {}).get("user_overlaps") == 0
        for r in (rb1, rb6, rb11))
    from rb.simulator import ReactiveUser as _RU
    ru = _RU({"id": "x", "lang": "zh", "step_latencies": [0.5, 2.0],
              "scenario_steps": ["search_trains", "hold_seat"],
              "events": [{"state": "inflight", "offset": 0.5, "action":
                          "revise", "role": "user", "voice": "cv01",
                          "text": "t"}]})
    no_fire = ru.on_event({"event": "tact_op_applied", "t": 1.0,
                           "data": {"t_audio": 1.0, "op": {"type": "launch",
                                                           "fn": "search_trains"}}})
    fire = ru.on_event({"event": "tact_op_applied", "t": 2.0,
                        "data": {"t_audio": 2.0, "op": {"type": "launch",
                                                        "fn": "hold_seat"}}})
    ck["v23_inflight_anchor_nonread"] = (no_fire == [] and len(fire) == 1
                                         and fire[0]["at"] == 3.0)
    from rb.generator import make_episode as _mk, config_hash as _chf
    _ch = _chf()
    langs_per_domain = {}
    for i in range(40):
        e = _mk("A", "L1", i, _ch)
        langs_per_domain.setdefault(e["domain"], set()).add(e["lang"])
    ck["v23_lang_domain_decoupled"] = all(
        len(v) == 2 for v in langs_per_domain.values())
    td = tact_core.tact_decider
    before = "REVISION TARGET BINDING" in td.SYSTEM_PROMPT
    install_rb_attr()
    ck["v23_attr_additive"] = (not before) and \
        "REVISION TARGET BINDING" in td.SYSTEM_PROMPT
    from rb.audio import perturb_samples, noise_block
    import array as _arr
    base = _arr.array("h", [1000] * 1600)
    p1 = perturb_samples(base, rate=1.06, gain_db=-3.0)
    p2 = perturb_samples(base, rate=1.06, gain_db=-3.0)
    nb1 = noise_block("e1", 100, 50.0)
    nb2 = noise_block("e1", 100, 50.0)
    ck["v23_perturb_deterministic"] = (p1 == p2 and len(p1) < len(base)
                                       and list(nb1) == list(nb2))

    # -- admission v1 (patch schema gate; rb_design 16.7) --------------------
    from admission import admit_decision_ops
    req = {f: spec["required"] for f, spec in TOOLS.items()}
    ops_in = [{"type": "launch", "fn": "check_stock", "args": {"item_id": "A100"}},
              {"type": "patch", "op_id": 2, "diff": {"seat_class": "coach"}},
              {"type": "patch", "op_id": 2, "diff": {"query": "鞋", "qty": 2}},
              {"type": "patch", "op_id": 9, "diff": {"seat_class": "coach"}}]
    out, aud = admit_decision_ops(ops_in, {2: "search_catalog"}, req)
    # redirect candidates come only from PENDING ops passed in: none here
    ck["adm_unit_gate"] = (
        len(out) == 3                                   # junk-only patch dropped
        and out[0] is ops_in[0]                         # non-patch passthrough
        and out[1] == {"type": "patch", "op_id": 2, "diff": {"query": "鞋"}}
        and out[2] is ops_in[3]                         # unknown target untouched
        and len(aud) == 2
        and aud[0]["rejected_keys"] == ["seat_class"] and aud[0]["dropped"]
        and aud[0]["redirect_candidates"] == []
        and aud[1]["rejected_keys"] == ["qty"] and not aud[1]["dropped"])

    class JunkPatchDecider(OracleDecider):
        """Launches the full scenario with FINAL canon values, then emits a
        spurious illegal-field patch on the first pending op — the exact
        junk-arg corruption shape the census found (25 eps, 0 passes).
        Subclasses OracleDecider only for the runner's oracle-path dispatch."""
        def __init__(self, ep):
            self.ep = ep
            self.scn = SCENARIOS[ep["scenario"]]
            self.launched = False
        def __call__(self, tx, segs_done, op_ids):
            if not self.launched:
                self.launched = True
                ops = []
                for st in self.scn["steps"]:
                    args = {a: (self.ep["slots_canon"][v.strip("{}")]
                                if isinstance(v, str) and v.startswith("{") else v)
                            for a, v in st["args"].items()}
                    ops.append({"type": "launch", "fn": st["fn"], "args": args})
                return {"dialogue": "stay", "ops": ops, "say": "ok"}
            if tx.pending:
                oid = sorted(tx.pending)[0]
                return {"dialogue": "stay", "say": "",
                        "ops": [{"type": "patch", "op_id": oid,
                                 "diff": {"seat_class": "junk"}}]}
            return {"dialogue": "stay", "ops": [], "say": ""}

    ep12a = make_episode("A", "L12", 0, ch)
    r_off = run_episode(ep12a, JunkPatchDecider(ep12a), mode="tact",
                        input_kind="text")
    r_on = run_episode(ep12a, JunkPatchDecider(ep12a), mode="tact",
                       input_kind="text", admission="schema")
    ck["adm_flip_junk_arg_failure"] = (not r_off["exact"]) and r_on["exact"]
    ck["adm_audit_recorded"] = (
        any("admission" in d for d in r_on["decisions"])
        and not any("admission" in d for d in r_off["decisions"]))
    r_off2 = run_episode(ep12a, JunkPatchDecider(ep12a), mode="tact",
                         input_kind="text")
    ck["adm_default_off_identity"] = json.dumps(r_off, sort_keys=True) == \
        json.dumps(r_off2, sort_keys=True)
    # legal patches under the gate are byte-untouched: oracle run identical
    ro_off = run_episode(ep12a, OracleDecider(ep12a), mode="tact",
                         input_kind="text")
    ro_on = run_episode(ep12a, OracleDecider(ep12a), mode="tact",
                        input_kind="text", admission="schema")
    ck["adm_legal_patches_untouched"] = json.dumps(ro_off, sort_keys=True) == \
        json.dumps(ro_on, sort_keys=True)

    # -- admission v1.1: post-resolution gate (rb_design 16.8) ---------------
    from admission import admit_decision_ops_v11
    from tact_core import resolve_ref as _rref
    from tact.transaction import Transaction as _Tx, Reversibility as _Rev
    import itertools as _it
    import tact.transaction as _txm
    _txm._uid = _it.count(1)      # per-episode id reset (mirrors run_episode) —
    # real ids become 2/3 while local snapshot ids are 1/2: the exact
    # collision geometry that made v1 misread local ids as global ones.
    tx11 = _Tx()
    p_bal = tx11.launch("get_balance", {"account": "checking"}, _Rev.READ, t=0.0)
    p_al = tx11.launch("set_alert", {"threshold": 500, "account": "checking"},
                       _Rev.REV, t=0.0)
    tx11._localmap = {1: p_bal.op_id, 2: p_al.op_id}   # snapshot-local numbering
    fnmap = {o.op_id: tx11.pending[o.op_id].fn for o in (p_bal, p_al)}

    def res11(op):
        r = _rref(tx11, op)
        return r if r in tx11.pending else None
    # the exact op that killed v1: model's LOCAL id 2 = set_alert; threshold
    # is legal there. v1 misread it as global id -> get_balance -> rejected.
    killer = {"type": "patch", "op_id": 2, "diff": {"threshold": 1000}}
    out11, aud11 = admit_decision_ops_v11([killer], res11, fnmap, req)
    out_v1, aud_v1 = admit_decision_ops([killer], fnmap, req)
    ck["adm11_localmap_correct_patch_untouched"] = (
        out11 == [killer] and aud11 == []
        and len(aud_v1) == 1)            # contrast: v1 would have rejected it
    # post-resolution illegal: local 1 -> get_balance, threshold illegal there
    bad11 = {"type": "patch", "op_id": 1, "diff": {"threshold": 1000}}
    o2, a2 = admit_decision_ops_v11([bad11], res11, fnmap, req)
    ck["adm11_postres_illegal_dropped"] = (
        o2 == [] and len(a2) == 1 and a2[0]["resolved_op_id"] == p_bal.op_id
        and a2[0]["gate"] == "v1.1" and a2[0]["dropped"])
    # nested-args wire shape: engine unwraps -> gate checks unwrapped form
    nest_ok = {"type": "patch", "op_id": 2,
               "diff": {"args": {"threshold": 800}}}
    nest_bad = {"type": "patch", "op_id": 2,
                "diff": {"args": {"threshold": 800, "seat_class": "coach"}}}
    o3, a3 = admit_decision_ops_v11([nest_ok, nest_bad], res11, fnmap, req)
    ck["adm11_nested_args_engine_mirror"] = (
        o3[0] is nest_ok                                  # legal: untouched
        and o3[1]["diff"] == {"threshold": 800}           # illegal key stripped
        and len(a3) == 1 and a3[0]["wire_unwrapped"]
        and a3[0]["rejected_keys"] == ["seat_class"])
    # unresolvable target: pass through byte-identical (engine drops it)
    ghost = {"type": "patch", "op_id": 99, "diff": {"seat_class": "x"}}
    o4, a4 = admit_decision_ops_v11([ghost], res11, fnmap, req)
    ck["adm11_unresolvable_passthrough"] = o4 == [ghost] and a4 == []
    # e2e: the junk-arg flip still holds under v1.1 (raw-id fallback path)
    r11 = run_episode(ep12a, JunkPatchDecider(ep12a), mode="tact",
                      input_kind="text", admission="schema11")
    ck["adm11_flip_junk_arg_failure"] = r11["exact"] and (not r_off["exact"]) \
        and any("admission" in d for d in r11["decisions"])
    ro11 = run_episode(ep12a, OracleDecider(ep12a), mode="tact",
                       input_kind="text", admission="schema11")
    ck["adm11_legal_path_untouched"] = json.dumps(ro_off, sort_keys=True) == \
        json.dumps(ro11, sort_keys=True)

    # ---- v2.4 (rb_design §17) ----------------------------------------------
    from rb.grammar import REV_UTT as _REV_UTT, revision_text as _rt
    # L4 text fix: single-{new} contrastive templates; rendered text carries
    # the new value exactly once and the old value once (the erratum's net)
    ck["v24_value_first_single_new"] = all(
        _REV_UTT[lg]["value_first"].count("{new}") == 1
        and _REV_UTT[lg]["value_first"].count("{old}") == 1
        for lg in ("zh", "en"))
    t_vf = _rt("zh", "value_first", "美元", old="日元")
    ck["v24_value_first_render"] = t_vf.count("美元") == 1 and "日元" in t_vf
    ep4v = make_episode("A", "L4", 0, ch)
    rv = ep4v["revisions"][0]
    vtxt = ep4v["pieces"][1]["text"]
    ck["v24_l4_piece_contrastive"] = (
        vtxt.count(str(rv["new"])) == 1 and vtxt.startswith(str(rv["new"])))
    # guard: a bank variant with two {new} is discarded for the frozen template
    ck["v24_double_new_guard"] = _rt(
        "en", "default", "Denver",
        content_hook=None, rng=None, old="Austin").count("Denver") == 1 and \
        "{new}" not in _rt("en", "value_first", "Denver", old="Austin")
    # catalog gating: v2.3 catalog bytes are frozen (sha pinned), v2.4 extends
    import hashlib as _hh
    V23_CATALOG_SHA = "32e09323ed7e8e98806c7aee5d342afa82719b461a19904b61a338e9587ac3f5"
    ck["v24_catalog_v23_frozen"] = (
        _hh.sha256(rb_catalog(v24=False).encode()).hexdigest() == V23_CATALOG_SHA
        and "ABORT" in rb_catalog(v24=True)
        and rb_catalog(v24=True).startswith(rb_catalog(v24=False)))
    # snapshot gating: caps episodes get X-ids for committed ops; a cap-less
    # tx keeps the byte-exact _snapshot_v2 text
    ep15 = make_episode("B", "L15", 0, ch)
    ck["v24_l15_shape"] = (ep15["profile"] == "heavy"
                          and ep15["events"][0]["state"] == "executing"
                          and "abort_feasible" in ep15
                          and ep15["caps"]["abort_on_cancel"])
    # find a feasible L15 episode and run the oracle end-to-end: exact pass
    # with a successful abort (fee-free) recorded
    ep15f = next((make_episode("B", "L15", k, ch) for k in range(50)
                  if make_episode("B", "L15", k, ch)["abort_feasible"]), None)
    if ep15f is not None:
        r15 = run_episode(ep15f, OracleDecider(ep15f), mode="tact",
                          input_kind="text")
        ck["v24_l15_abort_route"] = (r15["exact"] and r15["aborted_ops"] == 1
                                     and r15["fees"] == 0
                                     and any(d.get("ops") and any(
                                         o.get("type") == "abort" and o.get("ok")
                                         for o in d["ops"])
                                         for d in r15["decisions"]))
    else:
        ck["v24_l15_abort_route"] = False
    # cap-less episode: an X-cancel is NOT intercepted (v2.3 semantics keep)
    ep7o = make_episode("A", "L7", 3, ch)
    ep7o = dict(ep7o)
    ep7o.pop("caps")

    class XCancelDecider(OracleDecider):
        def __call__(self, tx, segs_done, op_ids):
            dec = super().__call__(tx, segs_done, op_ids)
            if tx.committed:
                dec = dict(dec)
                dec["ops"] = ([{"type": "cancel",
                                "op_id": f"X{tx.committed[0].op_id}"}]
                              + list(dec["ops"]))
            return dec
    r7x = run_episode(ep7o, XCancelDecider(ep7o), mode="tact",
                      input_kind="text")
    ck["v24_abort_gated_off_without_caps"] = (
        r7x["aborted_ops"] == 0 and
        not any(o.get("type") == "abort" for d in r7x["decisions"]
                for o in d["ops"]))
    # L13 pairing: the 8 cells of a family share content; states/who differ;
    # user cells' gold carries the revision, bystander cells' gold does not
    fam0 = [make_episode("B", "L13", k, ch) for k in range(8)]
    ck["v24_l13_family_shared_content"] = (
        len({e["scenario"] for e in fam0}) == 1
        and len({e["lang"] for e in fam0}) == 1
        and len({json.dumps(e["slots"], sort_keys=True) for e in fam0}) == 1
        and len({e["lat_ns"] for e in fam0}) == 1
        and len({json.dumps(e["step_latencies"]) for e in fam0}) == 1
        and [e["pair"]["who"] for e in fam0] == ["user"] * 4 + ["bystander"] * 4
        and [e["pair"]["state"] for e in fam0] ==
        list(L13_STATES_) * 2)
    u_new = fam0[0]["revisions"][0]["new"]
    ck["v24_l13_gold_split"] = (
        all(e["revisions"] and e["revisions"][0]["new"] == u_new
            for e in fam0[:4])
        and all(not e["revisions"] and e["bystander"]["other"] == u_new
                for e in fam0[4:])
        and json.dumps(fam0[4]["gold_calls"], sort_keys=True) !=
        json.dumps(fam0[0]["gold_calls"], sort_keys=True))
    # all four user states run to gold under the oracle (window patch or
    # abort/reverse route, state-dependent)
    l13_ok = []
    for e in fam0[:4]:
        rr = run_episode(e, OracleDecider(e), mode="tact", input_kind="text")
        l13_ok.append(rr["exact"])
    ck["v24_l13_user_states_oracle"] = all(l13_ok)
    rb0 = run_episode(fam0[4], OracleDecider(fam0[4]), mode="tact",
                      input_kind="text")
    ck["v24_l13_bystander_ignored"] = rb0["exact"] and \
        rb0["intruder_present"] is False and "pair" in rb0
    # L14: confirm + late revision both fire; oracle reaches gold on tools
    ep14 = make_episode("B", "L14", 0, ch)
    r14 = run_episode(ep14, OracleDecider(ep14), mode="tact",
                      input_kind="text")
    ck["v24_l14_two_events_and_gold"] = (
        len(ep14["events"]) == 2 and r14["exact"]
        and sum(1 for s in r14["segs"] if s[2] == "user") == 3)
    # WHO flags: benign L10 adopted / adversarial not intruded under oracle
    e10b = make_episode("A", "L10", 2, ch)          # idx%3==2 -> benign
    r10b = run_episode(e10b, OracleDecider(e10b), mode="tact",
                       input_kind="text")
    e10a = make_episode("A", "L10", 0, ch)          # idx%3==0 -> command
    r10a = run_episode(e10a, OracleDecider(e10a), mode="tact",
                       input_kind="text")
    ck["v24_who_flags"] = (r10b["rev_adopted"] is True
                           and r10b["intruder_present"] is None
                           and r10a["rev_adopted"] is None
                           and r10a["intruder_present"] is False
                           and e10a["bystander"].get("slot"))
    # stratified split: floors + family atomicity + L10 cell balance
    chq, epsq = build_all(quota_a={"L10": 12, "L4": 12},
                          quota_b={"L13": 32, "L14": 12})
    from collections import Counter as _C
    spl = _C((e["layer"], e["split"]) for e in epsq)
    fam_splits = {}
    for e in epsq:
        if e["layer"] == "L13":
            fam_splits.setdefault(e["pair"]["family"], set()).add(e["split"])
    ck["v24_split_stratified"] = (
        spl[("L4", "dev")] == 6 and spl[("L14", "dev")] == 6
        and spl[("L10", "dev")] == 6           # 2 per idx%3 cell
        and all(len(v) == 1 for v in fam_splits.values()))
    # report instruments present
    agg_rows = [r10b, r10a, rb0, r15] if ep15f is not None else \
        [r10b, r10a, rb0]
    rep24 = aggregate(agg_rows)
    ck["v24_report_axes"] = ("who_axis" in rep24 and "pair_axis" in rep24
                            and "route_axis" in rep24)
    # quota arithmetic (frozen): totals + every L10 cell clears n>=30 on test
    ck["v24_quota_arith"] = (
        sum(ARM_A_QUOTA.values()) == 666 and sum(ARM_B_QUOTA.values()) == 698
        and ARM_A_QUOTA["L10"] % 3 == 0 and ARM_B_QUOTA["L10"] % 3 == 0
        and ARM_A_QUOTA["L10"] // 3 - 2 >= 30
        and ARM_B_QUOTA["L10"] // 3 - 2 >= 30
        and ARM_B_QUOTA["L13"] % 8 == 0)
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
