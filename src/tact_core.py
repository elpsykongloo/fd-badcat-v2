# -*- coding: utf-8 -*-
"""
src/tact_core.py — the SINGLE SOURCE of TACT transactional semantics (W3 D1).

Everything semantic that was previously implemented inside scripts/w2r_stream_replay.py
(and therefore lived only in the offline harness) is factored here so that the live
engine (src/engine_b.py TactEngine) and the replay driver (scripts/w2r_stream_replay.py)
execute the SAME code. The two entry points differ only in perception and clock
advancement:

    live   : silero VADIterator frames -> per-frame silence burning (causal)
    replay : offline get_speech_timestamps segments -> interval silence burning

Semantics (engine_b spec, 手工文档/神谕/06 §一 — the five clauses):

  1. TIMER-DRIVEN COMMIT SCHEDULER on the SILENCE CLOCK: every pending op carries a
     silence budget (= delta at launch; reset to delta on patch). User speech freezes
     the countdown; silent audio-clock time burns it. Budget exhausted => the op's
     commit timer fires at `nominal = burn_start + remaining` (audio clock).
     delta<=0 or blocking mode => commit immediately at decision-apply time.

  2. COMMIT BARRIER ("decision-atomic commit", the W3 ruling): while a decision whose
     dispatch-time snapshot contains op X is in flight, X's expiry commit is DEFERRED
     (never dropped). DecisionDone processing order: apply the decision's ops FIRST
     (patch/cancel may rescue X; patch restarts the window), THEN sweep at the current
     clock. A deferred-then-unrescued commit keeps its NOMINAL stamp (audio-clock truth
     of window expiry); the ACTUAL execution time is recorded alongside (dual stamp).
     `commit_barrier=False` is the semantics ABLATION arm (continuous clock): expiries
     commit the moment they fire, even mid-decision; a late patch then finds the op
     committed and is dropped + logged (`patch_after_commit`).

  3. THREE BARRIER RELEASE PATHS (all funnel through end_decision + sweep):
     normal DecisionDone | staleness invalidation (gen/epoch mismatch) | decision
     timeout fail-open. The engine traces each; timeout should be zero-triggered.

  4. DUAL-STAMP ACCOUNTING: each commit records {t_commit (=nominal), actual_commit};
     actual - nominal = barrier deferral, a first-class metric bounded by decision
     latency p99. Deferral events are kept in WindowLedger.deferrals.

  5. The live EoU detector carries VAD frame latency epsilon (budget <=100ms); the
     replay EoU is oracle (retroactive VAD). This is a documented perception delta
     (D2 residual item), NOT a semantics delta — semantics live here, once.

Bit-parity contract: with commit_barrier=True and the same decision cache, the replay
driver reproduces the W2 grid v1 result files field-for-field on actual_tool_calls,
latency and tx_log (new fields are strictly additive). scripts/w3_barrier_probe.py
enforces this plus the four-clip on/off probe from the 06 ruling.
"""

from __future__ import annotations

import json
import math
import re
import sys

for _p in ("/root/autodl-tmp", "/root/autodl-tmp/fd-badcat/src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tact.transaction import Transaction, Reversibility  # noqa: E402
from tact.tools import ToolRegistry, REVERSIBILITY       # noqa: E402
import tact.decider as tact_decider                      # noqa: E402
from tact.decider import build_decider_messages, parse_decision  # noqa: E402

HOLD = 0.64          # frozen engine END_HOLD (iron rule 1: value reused, not changed)
SR = 16000
_EPS = 1e-9

# ---------------------------------------------------------------------------
# Prompt v2 (W2 rerun addenda) — applied IDENTICALLY to both arms & both paths.
# Moved verbatim from scripts/w2r_stream_replay.py; the text is FROZEN (any edit
# invalidates the decision cache and the W2 grids).
# ---------------------------------------------------------------------------
PROMPT_V2_ADDENDUM = """
ADDITIONAL RULES:
8. If the user BOTH revises an existing pending op AND adds a new request in the
   same utterance, emit the `patch` AND the new `launch` together in the same ops list.
9. `say` must NEVER be empty when ops is non-empty: briefly announce what you are
   doing (e.g. "Updating that to Chicago and checking the commute now.").
"""

_installed = {"prompt": False, "snapshot": False, "prompt_v3": False}


def install_prompt_v2():
    if not _installed["prompt"]:
        tact_decider.SYSTEM_PROMPT = tact_decider.SYSTEM_PROMPT + PROMPT_V2_ADDENDUM
        _installed["prompt"] = True


# ---------------------------------------------------------------------------
# Prompt v3 (W3 D5, the five-target batch — 06 §二④). DEFAULT OFF: installing
# it changes every decision-cache key, so it is armed only by an explicit
# `--prompt v3` / engine_cfg prompt:"v3". Targets (evidence: docs/w3_ledger §4):
#   10 post-commit paralysis (§4B)   11 declare-then-cancel (§4C, travel_10)
#   12 patch the corrected field (housing_17b)
#   13 self-interrupted requests (§4G, finance_23)
#   14 canonical entity forms (travel_19/16 + 裁断 C four cases)
# Draft + 30-subset protocol: docs/prompt_v3_five_targets.md.
# ---------------------------------------------------------------------------
PROMPT_V3_ADDENDUM = """
10. If ALREADY EXECUTED shows a call whose arguments the user has since \
corrected, do not just acknowledge it in `say`: emit the corrective op in this \
reply — re-launch the corrected call for search/lookup tools. And never drop \
the user's other requests: every distinct request still gets its own op.
11. If the user announces a change but has NOT yet said the new value (e.g. \
"wait, my schedule just changed"), immediately `cancel` the affected pending \
op — cancelling while pending costs nothing; re-launch once the corrected \
details arrive.
12. When emitting a `patch`, patch exactly the field the user corrected: check \
the user's words against each current arg value and change the one they \
contradicted (origin vs destination is the common mix-up). Do not repeat \
unchanged fields in the diff.
13. When the user interrupts their own request and replaces it ("check my \
balance— actually, never mind, just set up autopay"), the abandoned request is \
dropped but the REPLACEMENT is a real request: launch its op now.
14. Write argument values in canonical form: full official place names ("Las \
Vegas", not "Vegas"); plain cardinal dates ("June 3", not "June 3rd"); compact \
IDs without hyphens or spaces ("DL555", not "D-L-5-5-5"); no leading articles \
("gym", not "the gym"); no possessives ("driver license"); the noun exactly as \
the user said it, without added adjectives.
"""


def install_prompt_v3():
    install_prompt_v2()
    if not _installed["prompt_v3"]:
        tact_decider.SYSTEM_PROMPT = tact_decider.SYSTEM_PROMPT + PROMPT_V3_ADDENDUM
        _installed["prompt_v3"] = True


# ---------------------------------------------------------------------------
# Snapshot v2 (W2 rerun, both arms): include committed ops as ALREADY-EXECUTED
# context (prevents re-launch after window-expiry commit) and renumber pending
# ops with LOCAL ids so the prompt text is independent of the global op_id
# counter (cache-key stability under concurrency). Verbatim from the W2 harness.
# ---------------------------------------------------------------------------
def _snapshot_v2(self):
    self._localmap = {}
    parts = []
    if self.committed:
        parts.append("ALREADY EXECUTED (do NOT launch these again):")
        for op in self.committed:
            parts.append(f"  - fn={op.fn} args={json.dumps(op.args, ensure_ascii=False)}")
    parts.append("PENDING (not yet executed, patch/cancel by id):")
    if not self.pending:
        parts.append("  (none)")
    else:
        for local_id, op in enumerate(self.pending.values(), 1):
            self._localmap[local_id] = op.op_id
            parts.append(f"  - id={local_id} fn={op.fn} "
                         f"args={json.dumps(op.args, ensure_ascii=False)} "
                         f"status={op.status.value}")
    return "\n".join(parts)


def install_snapshot_v2():
    if not _installed["snapshot"]:
        Transaction.snapshot_for_prompt = _snapshot_v2
        _installed["snapshot"] = True


# Both installs are unconditional at import: every consumer of tact_core speaks
# the same prompt dialect => identical cache keys across live and replay.
install_prompt_v2()
install_snapshot_v2()


# ---------------------------------------------------------------------------
# Schema-typed argument coercion at the executor boundary (both arms), verbatim.
# ---------------------------------------------------------------------------
NUMERIC_FIELDS = {"amount", "max_price", "bedrooms", "quantity"}
POLY_FIELDS = {"value"}      # update_search_filter.value: bool | number | string


def coerce_args(args):
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and (k in NUMERIC_FIELDS or k in POLY_FIELDS):
            s = v.strip()
            if s.lower() in ("true", "false"):
                v = s.lower() == "true"
            else:
                try:
                    v = int(s) if re.fullmatch(r"-?\d+", s) else float(s)
                except ValueError:
                    pass
        out[k] = v
    return out


# ack-v0 (TACT arm only): template + slot filling when the model launched ops but
# said nothing — the announce IS the objection-window opener. Verbatim.
ACK_TEMPLATES = {
    "search_flights": "Let me search those flights for you.",
    "book_flight": "Booking that flight now.",
    "update_identity_doc": "Updating that document now.",
    "get_card_benefits": "Let me pull up those card benefits.",
    "get_exchange_rate": "Let me get that exchange rate.",
    "modify_autopay": "Setting up that autopay change.",
    "search_apartments": "Searching apartments for you now.",
    "calculate_commute": "Let me check that commute.",
    "update_search_filter": "Updating your search filter.",
    "track_order": "Let me track that order.",
    "search_products": "Searching for that now.",
    "add_to_cart": "Adding that to your cart.",
}


def ack_fallback(dec):
    """Fill dec['say'] from the ack template when ops launch but say is empty.
    Returns the (possibly template) say string. Marks dec['_ack_template']."""
    say = dec.get("say", "")
    launched_fns = [op.get("fn", "") for op in dec.get("ops", [])
                    if op.get("type") == "launch"]
    if not say and launched_fns:
        say = ACK_TEMPLATES.get(launched_fns[0], "I'm on it.")
        dec["_ack_template"] = True
    return say


# ---------------------------------------------------------------------------
# Robust decision parsing: strict -> one repair retry -> salvage. Verbatim W2.
# ---------------------------------------------------------------------------
REPAIR_TEXT = ("Your previous output was NOT valid JSON (check bracket balance — "
               "the \"ops\" array must be closed with ] before \"say\"). "
               "Reply with ONLY the corrected JSON object, nothing else.")


def salvage(raw):
    """Last-resort tolerant extraction when strict JSON parsing fails: pull op
    objects and the say string out of malformed output (e.g. a missing `]` before
    `"say"`). Only accepts ops whose own JSON parses; never invents fields."""
    ops = []
    for m in re.finditer(r'\{\s*"type"\s*:\s*"(launch|patch|cancel|commit|noop)".*?\}(?=\s*[,\]\}])',
                         raw, re.S):
        frag = m.group(0)
        extra, depth = 0, frag.count("{") - frag.count("}")
        end = m.end()
        while depth > 0 and end + extra < len(raw):
            ch = raw[end + extra]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            extra += 1
        frag = raw[m.start(): end + extra]
        try:
            ops.append(json.loads(frag))
        except Exception:
            continue
    say = ""
    ms = re.search(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if ms:
        try:
            say = json.loads('"' + ms.group(1) + '"')
        except Exception:
            say = ms.group(1)
    if ops or say:
        return {"dialogue": "stay", "ops": ops, "say": say, "_salvaged": True}
    return None


def build_msgs(tx, audio_prefix, state="LISTEN"):
    """The one true prompt builder (snapshot v2 + prompt v2 already installed)."""
    return build_decider_messages(tx, state, audio=audio_prefix)


def decide_from_msgs(call_fn, msgs):
    """Parse-with-repair over any `call_fn(msgs) -> (raw_text, infer_seconds)`.
    Returns (decision_dict, total_infer_seconds). Identical control flow to the
    W2 harness `decide()` — cache keys and repair text are frozen."""
    raw, infer = call_fn(msgs)
    dec = parse_decision(raw)
    if "_parse_error" in dec:                      # one repair retry (W2 plan D4)
        msgs2 = msgs + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": [{"type": "text", "text": REPAIR_TEXT}]}]
        raw2, infer2 = call_fn(msgs2)
        dec = parse_decision(raw2)
        infer += infer2
        dec["_repaired"] = True
        if "_parse_error" in dec:                  # salvage before conservative fallback
            sal = salvage(raw2) or salvage(raw)
            if sal is not None:
                sal["_repaired"] = True
                return sal, infer
    return dec, infer


def resolve_ref(tx, op):
    """Resolve a model-emitted patch/cancel reference to a real pending op_id.
    The model echoes the LOCAL id it saw in the snapshot; translate back."""
    if op.get("op_id") is not None:
        try:
            lid = int(op["op_id"])
        except Exception:
            return None
        localmap = getattr(tx, "_localmap", {})
        if lid in localmap:
            return localmap[lid]
        return lid if lid in tx.pending else None
    if "fn" in op:
        p = tx.find_pending_by_fn(op["fn"])
        return p.op_id if p else None
    p = tx.latest_pending()
    return p.op_id if p else None


# ---------------------------------------------------------------------------
# WindowLedger — objection windows on the silence clock + the commit barrier.
# Single-writer object: only the engine loop / the replay driver thread touches it.
# ---------------------------------------------------------------------------
class WindowLedger:
    def __init__(self, delta, barrier=True):
        self.delta = float(delta)
        self.barrier = bool(barrier)
        self.win = {}          # op_id -> remaining silence budget (seconds)
        self.expired = {}      # op_id -> nominal deadline (expiry fired, commit deferred)
        self.guards = {}       # decision key -> frozenset(op_ids in its dispatch snapshot)
        self.deferrals = []    # audit: {op_id, nominal, released, deferred_s, outcome, cause}
        self.patch_after_commit = []  # barrier-off observability (and any late patch)

    # -- window lifecycle -------------------------------------------------
    def open(self, op_id):
        self.win[op_id] = self.delta

    def restart(self, op_id):
        """Patch restarts the window with a FULL budget. If the op's expiry was
        deferred by the barrier, the patch RESCUES it (deferred commit voided)."""
        nominal = self.expired.pop(op_id, None)
        if nominal is not None:
            self.deferrals.append({"op_id": op_id, "nominal": round(nominal, 3),
                                   "released": None, "deferred_s": None,
                                   "outcome": "rescued_patch", "cause": "decision_ops"})
        self.win[op_id] = self.delta

    def close(self, op_id):
        """Cancel or commit: the op leaves the ledger entirely."""
        self.win.pop(op_id, None)
        nominal = self.expired.pop(op_id, None)
        if nominal is not None:
            self.deferrals.append({"op_id": op_id, "nominal": round(nominal, 3),
                                   "released": None, "deferred_s": None,
                                   "outcome": "cancelled", "cause": "decision_ops"})

    def remaining(self, op_id):
        return self.win.get(op_id)

    # -- barrier ------------------------------------------------------------
    def begin_decision(self, key, snapshot_op_ids):
        self.guards[key] = frozenset(snapshot_op_ids)

    def end_decision(self, key):
        self.guards.pop(key, None)

    def guarded(self, op_id):
        return any(op_id in s for s in self.guards.values())

    # -- clock --------------------------------------------------------------
    def advance_silence(self, t0, t1, commit_cb):
        """Burn silence over [t0, t1] (t1 may be math.inf at finalize). The DRIVER
        guarantees the interval is user-silent. Expiries fire in deadline order
        (matches the v1 sweep's deadline sort => identical commit order)."""
        span = t1 - t0
        if span <= _EPS:
            return
        expiring = sorted((rem, op_id) for op_id, rem in self.win.items()
                          if rem <= span + _EPS)
        for rem, op_id in expiring:
            del self.win[op_id]
            nominal = t0 + rem
            if self.barrier and self.guarded(op_id):
                self.expired[op_id] = nominal          # defer: barrier holds it
            else:
                commit_cb(op_id, nominal, nominal)     # timer fires on time
        if not math.isinf(span):
            for op_id in list(self.win):
                self.win[op_id] -= span

    def sweep(self, t_now, commit_cb, cause="decision_done"):
        """Release path: commit every deferred expiry that is no longer guarded
        and was not rescued. Called AFTER decision ops apply (barrier ordering)."""
        for op_id, nominal in sorted(self.expired.items(), key=lambda kv: kv[1]):
            if self.guarded(op_id):
                continue                               # another decision still holds it
            del self.expired[op_id]
            self.deferrals.append({"op_id": op_id, "nominal": round(nominal, 3),
                                   "released": round(t_now, 3),
                                   "deferred_s": round(t_now - nominal, 3),
                                   "outcome": "committed", "cause": cause})
            commit_cb(op_id, nominal, t_now)

    def note_patch_after_commit(self, t, ref, diff):
        self.patch_after_commit.append({"t": round(t, 3), "ref": ref, "diff": diff})

    def export(self):
        return {"barrier": self.barrier, "delta": self.delta,
                "deferrals": list(self.deferrals),
                "patch_after_commit": list(self.patch_after_commit)}


# ---------------------------------------------------------------------------
# Decision application — the exact W2 op semantics, single-sourced.
# ---------------------------------------------------------------------------
def apply_decision_ops(tx, ledger, dec, t_dec, immediate, commit_cb,
                       dag=None, comp_registry=None):
    """Apply a parsed decision's ops to (tx, ledger) at audio time t_dec.

    immediate : blocking arm or delta<=0 -> launch commits on the spot.
    commit_cb : commit_cb(op_id, t_nominal, t_actual) — the ONLY commit path;
                the caller owns tool execution and double-commit protection.
    dag       : optional tact_dag.OpDag (W3 D5, DEFAULT None = frozen v1
                behavior). When armed: launches register into the DAG and
                patches propagate to dependent ops (reparam / stale / comp plan).
    Returns the `applied` list in the frozen v1 trace shape (dag events are
    exported separately — v1 trace parity).
    """
    applied = []
    for op in dec.get("ops", []):
        typ = op.get("type", "noop")
        if typ == "launch":
            fn = op.get("fn", "")
            args = op.get("args", {}) or {}
            if "args" in args and isinstance(args["args"], dict):
                args = {**args["args"], **{k: v for k, v in args.items() if k != "args"}}
            args = coerce_args(args)
            # PendingSet idempotence: an identical intent (fn+args) already
            # pending or committed is NOT a new intent — drop the duplicate.
            dup = (any(p.fn == fn and p.args == args for p in tx.pending.values())
                   or any(c.fn == fn and c.args == args for c in tx.committed))
            if dup:
                applied.append({"type": "launch_dedup", "fn": fn})
                continue
            rev = REVERSIBILITY.get(fn, Reversibility.IRR)
            occ = sum(1 for o in list(tx.pending.values()) + tx.committed
                      if o.fn == fn)
            from tact_dag import make_idem_key
            p = tx.launch(fn, args, rev, t=t_dec,
                          idem_key=make_idem_key("", fn, args, occ))
            if dag is not None:
                dag.register_launch(p)
            if immediate:
                commit_cb(p.op_id, t_dec, t_dec)
            else:
                ledger.open(p.op_id)
            applied.append({"type": "launch", "fn": fn, "op_id": p.op_id})
        elif typ == "patch":
            oid = resolve_ref(tx, op)
            if oid is not None and oid in tx.pending:
                diff = op.get("diff", {}) or {}
                if set(diff.keys()) == {"args"} and isinstance(diff["args"], dict):
                    diff = diff["args"]      # unwrap model's nested-args habit
                diff = coerce_args(diff)
                tx.patch(oid, diff, t=t_dec)
                ledger.restart(oid)          # window restarts (rescues a deferred expiry)
                if dag is not None:
                    dag.on_patch(tx, oid, diff, t=t_dec,
                                 comp_registry=comp_registry)
                applied.append({"type": "patch", "op_id": oid, "diff": diff})
            else:
                # Target already committed (continuous-clock arm) or unresolvable:
                # the patch is DROPPED (official-track behavior) but logged for the
                # semantics probe. Not appended to `applied` (v1 trace parity).
                ledger.note_patch_after_commit(
                    t_dec, {k: op.get(k) for k in ("op_id", "fn") if k in op},
                    op.get("diff", {}) or {})
        elif typ == "cancel":
            oid = resolve_ref(tx, op)
            if oid is not None and oid in tx.pending:
                tx.cancel(oid, t=t_dec, executor=None)
                ledger.close(oid)
                applied.append({"type": "cancel", "op_id": oid})
        # model-emitted commit / noop: ignored (the harness owns commit timing)
    return applied


# ---------------------------------------------------------------------------
# Silence-span helper for interval drivers (replay): complement of VAD segments.
# ---------------------------------------------------------------------------
def silent_spans(t0, t1, segs):
    """Silent sub-intervals of [t0, t1) given speech segments (sorted). t1 may be inf."""
    spans = []
    t = t0
    for s, e in segs:
        if e <= t:
            continue
        if s >= t1:
            break
        if s > t:
            spans.append((t, min(s, t1)))
        t = max(t, e)
        if t >= t1:
            break
    if t < t1:
        spans.append((t, t1))
    return spans


def advance_over(ledger, t0, t1, segs, commit_cb):
    """Burn the ledger over every silent span of [t0, t1) (replay driver path)."""
    if t1 < t0:
        return          # replay fiction: decision return can overshoot the next EoU
    for a, b in silent_spans(t0, t1, segs):
        ledger.advance_silence(a, b, commit_cb)
