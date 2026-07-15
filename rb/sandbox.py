# -*- coding: utf-8 -*-
"""rb/sandbox.py — RB v2 deterministic tool sandbox (docs/rb_design.md v2 §3).

Deterministic everywhere: result ids are minted from sha256(episode_id, fn,
call_index) so gold calls and the gold end-state are precomputable at build
time; latency is seeded lognormal per class with a heavy long-tail profile for
L9. Idempotency keys dedupe retries (at-most-once side effects); reverse tools
compensate (COMP compensation records an audit fee residual — P1's ≈id)."""
from __future__ import annotations

import hashlib
import json
import math
import random

from .registry import TOOLS

# lognormal (mu, sigma) seconds per latency class; heavy = L9 long-tail profile
LATENCY = {"short": (math.log(0.5), 0.4), "mid": (math.log(2.0), 0.5),
           "heavy": (math.log(20.0), 0.6)}
LATENCY_CAP_S = 60.0
COMP_FEE = 1                     # audit residual units per COMP compensation


def _rid(episode_id, fn, idx):
    return fn[:2].upper() + hashlib.sha256(
        f"{episode_id}:{fn}:{idx}".encode()).hexdigest()[:6]


def mint_id(episode_id, fn, k=0):
    """Public deterministic id minting: k-th call OF THIS FN in the episode.
    Independent of global call order, so window-policy reordering (patch
    restarts) cannot shift ids away from the build-time gold."""
    return _rid(episode_id, fn, k)


class Sandbox:
    """One episode's tool world. execute() mutates state and returns the FDB-
    style envelope {"status","result"}. State keys are '<fn>#<rid>' -> args
    (compensated entries flip 'void': True and add 'fee')."""

    def __init__(self, episode_id, profile="default", seed=0):
        self.episode_id = episode_id
        self.profile = profile
        self.rng = random.Random(f"{episode_id}:{seed}:lat")
        self.state = {}
        self.calls = []            # committed calls in order: {fn, args, rid}
        self.fees = 0
        self._idem = {}
        self._n_fn = {}

    def latency_of(self, fn):
        cls = "heavy" if self.profile == "heavy" and TOOLS[fn]["kappa"] != "READ" \
            else TOOLS[fn]["latency"]
        mu, sig = LATENCY[cls]
        return round(min(LATENCY_CAP_S, math.exp(self.rng.gauss(mu, sig))), 3)

    def execute(self, fn, args, idem_key=None):
        if fn not in TOOLS:
            return {"status": "error", "error": f"unknown tool {fn}"}
        missing = [a for a in TOOLS[fn]["required"]
                   if not str((args or {}).get(a, "")).strip()]
        if missing:
            return {"status": "error", "error": f"missing {missing}"}
        if idem_key and idem_key in self._idem:
            return self._idem[idem_key]
        k = self._n_fn.get(fn, 0)
        self._n_fn[fn] = k + 1
        rid = _rid(self.episode_id, fn, k)
        self.state[f"{fn}#{rid}"] = dict(args)
        self.calls.append({"fn": fn, "args": dict(args), "rid": rid})
        res = {"status": "success", "result": {"id": rid, "fn": fn}}
        if idem_key:
            self._idem[idem_key] = res
        return res

    def compensate(self, fn, rid):
        """Reverse the effect of an earlier call (fn, rid). COMP leaves a fee
        residual; REV is free; READ/IRR are not compensable here."""
        key = f"{fn}#{rid}"
        if key not in self.state or self.state[key].get("void"):
            return {"status": "error", "error": "nothing to compensate"}
        kappa = TOOLS[fn]["kappa"]
        if TOOLS[fn]["reverse"] is None:
            return {"status": "error", "error": f"{fn} has no reverse"}
        self.state[key]["void"] = True
        if kappa == "COMP":
            self.state[key]["fee"] = COMP_FEE
            self.fees += COMP_FEE
        return {"status": "success", "result": {"reversed": key}}

    def live_state(self):
        return {k: v for k, v in self.state.items() if not v.get("void")}


def oracle_run(episode_id, steps, slots, profile="default"):
    """Execute a resolved scenario (steps with {slot}/$R refs) against a fresh
    sandbox — used at BUILD time to precompute gold calls / end-state /
    per-step latencies. Returns (gold_calls, gold_state, latencies)."""
    sb = Sandbox(episode_id, profile=profile)
    results, gold_calls, lats = [], [], []
    for st in steps:
        args = {}
        for k, v in st["args"].items():
            if isinstance(v, str) and v.startswith("$R"):
                args[k] = results[int(v[2:])]["result"]["id"]
            elif isinstance(v, str):
                args[k] = v.format(**slots)
            else:
                args[k] = v
        lats.append(sb.latency_of(st["fn"]))
        res = sb.execute(st["fn"], args)
        assert res["status"] == "success", (st, res)
        results.append(res)
        gold_calls.append({"fn": st["fn"], "args": args})
    return gold_calls, sb.live_state(), lats


def canonical_calls(calls):
    """Canonical-sort multiset (rb_design v2 §5: replaces FDB's pop(0)
    positional alignment): group by fn, sort each group by the canonical JSON
    of args. Scoring compares these lists."""
    out = {}
    for c in calls:
        out.setdefault(c["fn"], []).append(
            json.dumps(c["args"], sort_keys=True, ensure_ascii=False))
    return {fn: sorted(v) for fn, v in out.items()}
