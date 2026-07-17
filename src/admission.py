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
"""
from __future__ import annotations


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
