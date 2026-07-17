# -*- coding: utf-8 -*-
"""admission.py — decision-op admission control, v1: the patch SCHEMA GATE
(W5 WHICH-axis mechanism; rb_design §16.7).

Motivation (forensic census over the RB archives, 2026-07-17): 9.7% of all
patches emitted by the decider in the v2.2.1 test archives (496/5,097; dev23
8.8%) carry diff FIELDS that do not exist on the target op's argument schema
— e.g. {seat_class: ...} patched into search_trains. Such a patch can NEVER
be correct behavior: gold args are exactly the tool's declared arguments, so
a junk key in the committed call is a guaranteed canonical mismatch. In the
main v2.2.1 arm, all 25 episodes containing one were failures.

v1 policy = REJECT-ONLY, which is provably non-harmful:
  * stripping keys that are not in the target's schema can only remove
    guaranteed-mismatch material from the committed call — the canonical
    distance to ANY gold cannot increase, on either the exact or the state
    track;
  * redirect (re-targeting the stripped value onto a schema-matching op) is
    NOT applied — the model's asserted value may itself be stale, and
    redirecting a stale value could corrupt an op that was already correct.
    Feasible redirects are COUNTED in the audit (evidence base for a v2).

The legal-field misbinding form ({destination: Thailand} — field legal,
semantics wrong; 4 of the 6 historical anti-window cells) is out of scope
for any mechanical gate and remains the measured open problem (L12).

Pure function; harness-agnostic (the required-args map is an input, so the
same gate runs on FDB tools or RB tools). Default-off wiring keeps frozen
paths byte-identical.

v1.1 (2026-07-17, after the test-911 R-ADM1 autopsy retired v1): the SAME
rejection rule moved to the layer it was proved on. v1 read the model's
snapshot-LOCAL ids as global op ids and killed correct patches (the engine's
resolve_ref had been translating local ids correctly all along), and it did
not model the engine's nested-{"args": {...}} unwrap. v1.1 therefore:

  * resolves each patch with THE ENGINE'S OWN resolver (a `resolve` callable
    bound to the live tx — tact_core.resolve_ref), on the same tx state the
    apply step will see;
  * mirrors the engine's wire unwrap rule byte-for-byte (a single-key
    {"args": {...}} diff is checked in its unwrapped form, exactly what
    tact_core.apply_decision_ops will apply);
  * gates ONLY patches that resolve cleanly to a currently-pending op; every
    unresolvable / stale / oddly-shaped op passes through untouched (the
    engine drops or handles those identically with or without the gate);
  * makes NO blanket harmlessness theorem claim this time. The claim is
    scoped: a stripped key is one the engine would have applied as a junk
    argument onto the RESOLVED op (a guaranteed canonical mismatch in the
    committed call). Second-order trajectory effects (snapshot text, dedup
    interactions) are checked EMPIRICALLY: dev zero-loss gate first, then
    the single-shot R-ADM1' zero-loss hard gate on test.

v1 is kept below, unchanged, for archival replay of the retired
rbt23_*_adm arms only.
"""
from __future__ import annotations


def _engine_unwrap(diff):
    """Mirror tact_core.apply_decision_ops: a single-key {"args": {...}} diff
    is applied in its unwrapped form."""
    if isinstance(diff, dict) and set(diff.keys()) == {"args"} \
            and isinstance(diff["args"], dict):
        return diff["args"], True
    return diff, False


def admit_decision_ops_v11(ops, resolve, pending_fn_of, required_map):
    """v1.1 post-resolution schema gate.

    ops           : parsed decision op dicts (model layer, pre-apply)
    resolve       : callable(op) -> real PENDING op_id or None; must be the
                    engine's resolver bound to the live tx (same tx state the
                    apply step will read), with non-pending results mapped
                    to None by the caller
    pending_fn_of : {real op_id -> fn} for currently pending ops
    required_map  : {fn -> [declared arg names]}

    Returns (admitted_ops, audit). audit entries carry gate="v1.1" and the
    RESOLVED target; `wire_unwrapped` marks nested-args diffs."""
    out, audit = [], []
    for op in ops:
        if not isinstance(op, dict) or op.get("type") != "patch" \
                or not isinstance(op.get("diff"), dict):
            out.append(op)
            continue
        rid = resolve(op)
        fn = pending_fn_of.get(rid)
        schema = required_map.get(fn)
        if rid is None or schema is None:
            out.append(op)               # engine drops/handles it identically
            continue
        diff, unwrapped = _engine_unwrap(op["diff"])
        bad = sorted(k for k in diff if k not in schema)
        if not bad:
            out.append(op)               # legal (in engine-applied form): untouched
            continue
        legal = {k: v for k, v in diff.items() if k in schema}
        audit.append({"op_id": op.get("op_id"), "resolved_op_id": rid,
                      "target_fn": fn, "rejected_keys": bad,
                      "kept_keys": sorted(legal), "dropped": not legal,
                      "wire_unwrapped": unwrapped, "gate": "v1.1"})
        if legal:
            kept = dict(op)
            kept["diff"] = legal          # already engine-unwrapped form
            out.append(kept)
    return out, audit


def admit_decision_ops(ops, pending_fn_of, required_map):
    """Filter a decision's op list through the schema gate.

    ops           : list of parsed op dicts ({"type","op_id","diff",...})
    pending_fn_of : {op_id -> fn} for CURRENTLY PENDING ops (snapshot ids —
                    same-decision launches have no ids yet and cannot be
                    patch targets by id)
    required_map  : {fn -> [required arg names]} (the tool schema)

    Returns (admitted_ops, audit). Non-patch ops and patches on unknown /
    non-pending targets pass through untouched (stale handling stays with
    the engine). audit = list of {op_id, target_fn, rejected_keys,
    kept_keys, dropped, redirect_candidates}.
    """
    out = []
    audit = []
    for op in ops:
        if not isinstance(op, dict) or op.get("type") != "patch" \
                or not isinstance(op.get("diff"), dict):
            out.append(op)
            continue
        fn = pending_fn_of.get(op.get("op_id"))
        schema = required_map.get(fn)
        if fn is None or schema is None:
            out.append(op)
            continue
        legal = {k: v for k, v in op["diff"].items() if k in schema}
        bad = [k for k in op["diff"] if k not in schema]
        if not bad:
            out.append(op)
            continue
        # counterfactual only (v1 never applies a redirect): pending ops
        # whose schema would accept ALL rejected keys
        cands = sorted({f for oid, f in pending_fn_of.items()
                        if f != fn and f in required_map
                        and all(k in required_map[f] for k in bad)})
        entry = {"op_id": op.get("op_id"), "target_fn": fn,
                 "rejected_keys": sorted(bad),
                 "kept_keys": sorted(legal),
                 "dropped": not legal,
                 "redirect_candidates": cands}
        audit.append(entry)
        if legal:
            kept = dict(op)
            kept["diff"] = legal
            out.append(kept)
        # else: the whole patch was junk — drop it
    return out, audit
