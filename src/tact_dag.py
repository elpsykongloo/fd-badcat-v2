# -*- coding: utf-8 -*-
"""
src/tact_dag.py — dependency propagation + compensation registry (W3 D5).

Two mechanisms, both DEFAULT-OFF (iron rule 1 — they change window timing and
must not perturb the frozen W2/W3 grids unless explicitly armed):

1. OpDag — result-dependency DAG over a Transaction's ops.
   Edges are DECLARED (parent_fn -> child_fn, which child args derive from the
   parent) and instantiated per-op by launch order + value-flow evidence.
   On patch(parent, diff) hitting a DEPENDED-UPON field:
     * pending child, passthrough field  -> REPARAMETERIZE: rewrite the flowed
       value inside the child's arg (a derived patch; its window restarts —
       any arg change reopens the objection window, same rule as a user patch);
     * pending child, result-derived field -> mark STALE: the child's cached
       derivation is invalid; window restarts; re-derivation happens at commit
       (in the mock world result-derived ids are constant, so v0 logs + restarts);
     * committed child -> emit a COMPENSATION PLAN (priced, not auto-executed:
       on the official track compensation is a death sentence — 教义一; the
       state track / realistic profile are where the plan is COSTED).
   The same edges feed latency_realistic.schedule() as the parallel-execution
   dependency set (裁断 A: chained scenarios, ActExecutor parallel vs serial).

2. CompensationRegistry — reverse templates + idempotency keys (蓝图 §2.3).
   κ-faithful: READ -> no-op (nothing to undo), REV/COMP -> declared inverse,
   IRR -> REFUSE (update_identity_doc must never be auto-undone).
   Idempotency: every plan carries `comp:<parent idem_key>`; execute() is
   at-most-once per key (safe against retry storms / double-application).
"""

from __future__ import annotations

import hashlib
import json

from tact.transaction import Reversibility
from tact.tools import REVERSIBILITY


def make_idem_key(room, fn, args, occurrence=0):
    payload = f"{room}|{fn}|{json.dumps(args or {}, sort_keys=True)}|{occurrence}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Declared dependency templates: child_fn -> {parent_fn: spec}
#   passthrough : child args whose VALUE flows from a parent arg (checked by
#                 containment at edge-instantiation time — evidence, not faith)
#   derived     : child args that come from the parent's RESULT (not visible
#                 pre-commit; staleness is all we can assert)
#   key_fields  : parent args that parameterize the derivation — a patch is
#                 propagation-relevant iff it hits one of these
# ---------------------------------------------------------------------------
DAG_TEMPLATES = {
    "book_flight": {
        "search_flights": {"passthrough": {}, "derived": ["flight_id"],
                           "key_fields": ["destination", "date"]},
    },
    "add_to_cart": {
        "search_products": {"passthrough": {}, "derived": ["product_id"],
                            "key_fields": ["query", "max_price", "category"]},
    },
    "calculate_commute": {
        "search_apartments": {"passthrough": {"origin_address": "city",
                                              "destination_address": "city"},
                              "derived": ["origin_address"],
                              "key_fields": ["city", "bedrooms", "max_price"]},
    },
}


class OpDag:
    """Single-writer (engine loop / replay thread) dependency tracker."""

    def __init__(self, ledger=None):
        self.ledger = ledger          # WindowLedger; None => no window coupling
        self.ops = {}                 # op_id -> op (live PendingOp references)
        self.order = []               # op_ids in launch order
        self.edges = {}               # child_op_id -> [parent_op_id, ...]
        self.edge_spec = {}           # (parent_id, child_id) -> template spec
        self.events = []              # audit: reparam / stale / comp_plan

    # -- registration ------------------------------------------------------
    def register_launch(self, op):
        self.ops[op.op_id] = op
        for prev_id in reversed(self.order):        # nearest matching parent wins
            parent = self.ops.get(prev_id)
            if parent is None:
                continue
            spec = DAG_TEMPLATES.get(op.fn, {}).get(parent.fn)
            if spec is None:
                continue
            flowed = self._flow_evidence(parent, op, spec)
            if spec["derived"] or flowed:
                self.edges.setdefault(op.op_id, []).append(prev_id)
                self.edge_spec[(prev_id, op.op_id)] = {**spec, "flowed": flowed}
                self.events.append({
                    "kind": "dag_edge",
                    "parent_op": prev_id,
                    "parent_fn": parent.fn,
                    "child_op": op.op_id,
                    "child_fn": op.fn,
                    "flowed": flowed,
                    "derived_fields": list(spec.get("derived", [])),
                })
                break
        self.order.append(op.op_id)

    @staticmethod
    def _flow_evidence(parent, child, spec):
        """Passthrough fields where the parent's value verifiably appears
        inside the child's value (case-insensitive containment)."""
        flowed = {}
        for c_field, p_field in spec.get("passthrough", {}).items():
            pv, cv = parent.args.get(p_field), child.args.get(c_field)
            if isinstance(pv, str) and isinstance(cv, str) and pv and \
                    pv.lower() in cv.lower():
                flowed[c_field] = p_field
        return flowed

    # -- propagation -------------------------------------------------------
    def on_patch(self, tx, parent_id, diff, t=None, comp_registry=None):
        """Propagate a patch on `parent_id` to its dependent children.
        Returns the list of propagation events appended (also kept in .events)."""
        out = []
        for (p_id, c_id), spec in list(self.edge_spec.items()):
            if p_id != parent_id:
                continue
            hit = [k for k in diff if k in spec["key_fields"]]
            if not hit:
                continue
            child = self.ops.get(c_id)
            if child is None:
                continue
            if c_id in tx.pending:
                ev = self._propagate_pending(tx, child, spec, diff, hit, t)
            else:
                ev = self._plan_compensation(tx, child, hit, t, comp_registry)
            if ev:
                self.events.append(ev)
                out.append(ev)
        return out

    def _propagate_pending(self, tx, child, spec, diff, hit, t):
        reparam = {}
        for c_field, p_field in spec.get("flowed", {}).items():
            if p_field in diff:
                old, new = None, diff[p_field]
                cv = child.args.get(c_field)
                # the pre-patch parent value we matched at registration time
                for h in reversed(self.ops[self.edges[child.op_id][0]].patch_history):
                    if p_field in h["before"]:
                        old = h["before"][p_field]
                        break
                if old and isinstance(cv, str) and isinstance(new, str) and \
                        old.lower() in cv.lower():
                    idx = cv.lower().index(old.lower())
                    reparam[c_field] = cv[:idx] + new + cv[idx + len(old):]
        stale_fields = [f for f in spec.get("derived", []) if f in child.args]
        if reparam:
            tx.patch(child.op_id, reparam, t=t)
        if self.ledger is not None and (reparam or stale_fields):
            self.ledger.restart(child.op_id)    # derived patch reopens the window
        if not (reparam or stale_fields):
            return None
        return {"kind": "dag_reparam" if reparam else "dag_stale",
                "t": round(t, 3) if t is not None else None,
                "parent_op": self.edges[child.op_id][0], "child_op": child.op_id,
                "child_fn": child.fn, "hit_fields": hit,
                "reparam": reparam, "stale_fields": stale_fields}

    def _plan_compensation(self, tx, child, hit, t, comp_registry):
        plan = (comp_registry or CompensationRegistry()).plan(child)
        return {"kind": "dag_comp_plan",
                "t": round(t, 3) if t is not None else None,
                "parent_op": self.edges[child.op_id][0], "child_op": child.op_id,
                "child_fn": child.fn, "hit_fields": hit,
                "plan": plan}

    def export(self):
        return {"edges": {str(c): ps for c, ps in self.edges.items()},
                "events": list(self.events)}


# ---------------------------------------------------------------------------
# Compensation registry: reverse templates + idempotency (priced, gated exec)
# ---------------------------------------------------------------------------
def _comp_book_flight(op):
    ref = (op.result or {}).get("booking_ref", "$RESULT.booking_ref")
    return {"fn": "cancel_booking", "args": {"booking_ref": ref}}


def _comp_modify_autopay(op):
    return {"fn": "revert_autopay",
            "args": {"bill_type": op.args.get("bill_type"),
                     "source_account": op.args.get("source_account")}}


def _comp_add_to_cart(op):
    return {"fn": "remove_from_cart",
            "args": {"product_id": op.args.get("product_id"),
                     "quantity": op.args.get("quantity", 1)}}


def _comp_update_search_filter(op):
    prior = "$PRIOR"
    for h in op.patch_history:                    # exact inverse if we saw the prior value
        if "value" in h["before"]:
            prior = h["before"]["value"]
            break
    return {"fn": "update_search_filter",
            "args": {"filter_name": op.args.get("filter_name"), "value": prior}}


COMP_TEMPLATES = {
    "book_flight": _comp_book_flight,
    "modify_autopay": _comp_modify_autopay,
    "add_to_cart": _comp_add_to_cart,
    "update_search_filter": _comp_update_search_filter,
}


class CompensationRegistry:
    def __init__(self):
        self.executed_keys = set()
        self.plans = []

    def plan(self, op):
        """Reverse plan for a committed op. κ-faithful; never executes."""
        rev = REVERSIBILITY.get(op.fn, Reversibility.IRR)
        base = {"parent_op": op.op_id, "parent_fn": op.fn,
                "kappa": rev.name,
                "idem_key": f"comp:{op.idem_key or make_idem_key('tx', op.fn, op.args)}"}
        if rev == Reversibility.READ:
            p = {**base, "action": "noop", "reason": "READ has no side effect"}
        elif rev == Reversibility.IRR:
            p = {**base, "action": "refuse", "reason": "IRR: no inverse exists"}
        else:
            tmpl = COMP_TEMPLATES.get(op.fn)
            if tmpl is None:
                p = {**base, "action": "refuse", "reason": "no template declared"}
            else:
                p = {**base, "action": "execute", **tmpl(op)}
        self.plans.append(p)
        return p

    def execute(self, plan, executor):
        """At-most-once execution of an `execute` plan. Returns the tool result,
        {'status':'skipped_idempotent'} on a replayed key, or raises on refuse."""
        if plan["action"] == "refuse":
            raise ValueError(f"refusing to compensate {plan['parent_fn']}: "
                             f"{plan['reason']}")
        if plan["action"] == "noop":
            return {"status": "success", "noop": True}
        if plan["idem_key"] in self.executed_keys:
            return {"status": "skipped_idempotent", "idem_key": plan["idem_key"]}
        self.executed_keys.add(plan["idem_key"])
        return executor(plan["fn"], plan["args"])
