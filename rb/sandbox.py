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

from .registry import TOOLS, REVERSE_OF, REVERSE_TARGET_ARG

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
        """Wall latency of the NEXT call of fn. v2.3: keyed by (episode, fn,
        occurrence) — interleaving-immune, same lesson as mint_id (the v2.2
        streamed draw shifted under window-policy reordering)."""
        return self.latency_of_at(fn, self._n_fn.get(fn, 0))

    def _match_forward(self, rev_fn, target):
        """Live forward entry a reverse call nets out: by minted id first,
        then by arg-value equality (remove_item / unsave_listing class);
        latest match wins. Returns the state key or None."""
        fwd = REVERSE_OF[rev_fn]
        tv = str(target)
        best = None
        for key, v in self.state.items():
            if not key.startswith(fwd + "#") or v.get("void"):
                continue
            if key.split("#", 1)[1] == tv or \
                    any(str(x) == tv for x in v.values()):
                best = key
        return best

    def execute(self, fn, args, idem_key=None, t=None):
        """v2.3: `t` (audio-clock commit time) opens an execution window
        [t, t+latency] during which abort() may cancel a REV/COMP effect;
        reverse-tool calls that match a live forward entry NET IT OUT (the
        compensation path is an ordinary catalog call, scoreable for any
        system). Without t, effects are instantaneous (v2.2 behavior)."""
        if fn not in TOOLS:
            return {"status": "error", "error": f"unknown tool {fn}"}
        missing = [a for a in TOOLS[fn]["required"]
                   if not str((args or {}).get(a, "")).strip()]
        if missing:
            return {"status": "error", "error": f"missing {missing}"}
        if idem_key and idem_key in self._idem:
            return self._idem[idem_key]
        if fn in REVERSE_OF:
            key = self._match_forward(fn, (args or {}).get(REVERSE_TARGET_ARG[fn]))
            if key is None:
                return {"status": "error",
                        "error": f"{fn}: no live {REVERSE_OF[fn]} to reverse"}
            fwd_kappa = TOOLS[REVERSE_OF[fn]]["kappa"]
            self.state[key]["void"] = True
            if fwd_kappa == "COMP":
                self.state[key]["fee"] = COMP_FEE
                self.fees += COMP_FEE
            self.calls.append({"fn": fn, "args": dict(args),
                               "rid": key.split("#", 1)[1], "comp": True})
            res = {"status": "success",
                   "result": {"id": key.split("#", 1)[1], "fn": fn,
                              "reversed": key}}
            if idem_key:
                self._idem[idem_key] = res
            return res
        k = self._n_fn.get(fn, 0)
        self._n_fn[fn] = k + 1
        rid = _rid(self.episode_id, fn, k)
        entry = dict(args)
        if t is not None:
            entry["completes_at"] = round(float(t) + self.latency_of_at(fn, k), 3)
        self.state[f"{fn}#{rid}"] = entry
        self.calls.append({"fn": fn, "args": dict(args), "rid": rid})
        res = {"status": "success", "result": {"id": rid, "fn": fn}}
        if idem_key:
            self._idem[idem_key] = res
        return res

    def latency_of_at(self, fn, k):
        """Deterministic per-(fn, occurrence) latency — replayable regardless
        of interleaving (the streamed rng draw depends on call order)."""
        r = random.Random(f"{self.episode_id}:{fn}:{k}:lat")
        cls = "heavy" if self.profile == "heavy" and TOOLS[fn]["kappa"] != "READ" \
            else TOOLS[fn]["latency"]
        mu, sig = LATENCY[cls]
        return round(min(LATENCY_CAP_S, math.exp(r.gauss(mu, sig))), 3)

    def abort(self, rid_or_key, t):
        """Abort an op WHILE IT IS STILL EXECUTING (t < completes_at): the
        effect never lands (entry voided, no fee — the def2 alternative to
        wait-then-compensate). IRR ops cannot be aborted; completed ops
        cannot be aborted (use the reverse tool)."""
        key = rid_or_key if "#" in str(rid_or_key) else next(
            (k for k in self.state if k.endswith(f"#{rid_or_key}")), rid_or_key)
        v = self.state.get(key)
        if v is None or v.get("void"):
            return {"status": "error", "error": "nothing to abort"}
        fn = key.split("#")[0]
        if TOOLS[fn]["kappa"] == "IRR":
            return {"status": "error", "error": f"{fn} is irreversible"}
        ca = v.get("completes_at")
        if ca is None or float(t) >= ca:
            return {"status": "error", "error": "already completed - compensate"}
        v["void"] = True
        v["aborted_at"] = round(float(t), 3)
        return {"status": "success", "result": {"aborted": key}}

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
            elif isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                args[k] = slots[v.strip("{}")]          # keep canonical type
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
        norm = {k: (str(v) if isinstance(v, (int, float)) else v)
                for k, v in c["args"].items()}
        out.setdefault(c["fn"], []).append(
            json.dumps(norm, sort_keys=True, ensure_ascii=False))
    return {fn: sorted(v) for fn, v in out.items()}
