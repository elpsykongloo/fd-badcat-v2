# -*- coding: utf-8 -*-
"""
src/latency_realistic.py — the REALISTIC latency profile (W3 D4, 裁断 A).

The official FDB-v3 sandbox executes tools at p50 ≈ 0.315 s wall ("official
profile") — R9: no overlap headroom, so the dual-profile claim (05 §二 裁断 A)
needs a second, DEFENSIBLY CALIBRATED profile. This module is that profile:

  * κ-class → lognormal(μ, σ) fits, anchored on public measurements
    (calibration table + sources: docs/latency_calibration.md; parameters are
    PREREGISTERED there — any post-run edit is forbidden by the prereg rule).
  * PER-INSTANCE deterministic sampling: seed = sha256(example_id|fn|args|k).
    Independent of call order, thread interleaving and process restarts — this
    is the per-instance-RNG fix the W2 harness lacked (global random.seed(42)),
    implemented locally (upstream PR cancelled 7/07: repo is local-only).
  * PURE ACCOUNTING: tool latency in this profile never sleeps and never feeds
    back into decisions (tool results do not enter the decider prompt), so a
    realistic-profile score is a deterministic function of an existing trace.
    => the realistic δ grid = re-scoring existing runs; zero new GPU decisions.

Executor semantics (裁断 A / 教义二: the completion anchor is where the two
arms differ under real latencies):

  blocking : SERIAL — the official SUT issues call i+1 only after call i
             returns:  done_i = max(t_commit_i, done_{i-1}) + lat_i
  tact     : PARALLEL per DAG — ActExecutor starts an op at its commit stamp,
             but an op that depends on a parent's RESULT starts only after the
             parent finishes: done_i = max(t_commit_i, max_parent done) + lat_i

Both arms sample THE SAME latency for the same (example, fn, args, occurrence)
=> paired comparison is exact; the premium decomposition stays clean.
"""

from __future__ import annotations

import hashlib
import json
import math
import random

PROFILE_VERSION = "realistic-v1"

# ---------------------------------------------------------------------------
# κ-class → lognormal parameters (seconds). PREREGISTERED — see
# docs/latency_calibration.md for the anchor measurements and the fit.
# p50 = exp(mu); p95 = exp(mu + 1.645*sigma). Samples are capped at p999
# (exp(mu + 3.09*sigma)): public measurements show minute-long outliers
# (arXiv:1903.07712) that would let a single draw dominate a 100-scenario mean.
# ---------------------------------------------------------------------------
CLASS_PARAMS = {
    #                 mu            sigma   (p50, p95 for the docstring/table)
    "read_lookup":  (math.log(0.30), 0.55),   # p50 0.30  p95 0.74
    "read_search":  (math.log(0.75), 0.35),   # p50 0.75  p95 1.33
    "write_light":  (math.log(0.40), 0.50),   # p50 0.40  p95 0.91
    "write_booking": (math.log(3.00), 0.31),  # p50 3.00  p95 5.00
}

TOOL_CLASS = {
    # READ, light lookups (rate/benefit/track/route grade)
    "get_card_benefits":    "read_lookup",
    "get_exchange_rate":    "read_lookup",
    "track_order":          "read_lookup",
    "calculate_commute":    "read_lookup",
    # READ, inventory searches (flight/hotel/product-search grade)
    "search_flights":       "read_search",
    "search_apartments":    "read_search",
    "search_products":      "read_search",
    # REV writes (cart/preference grade)
    "update_search_filter": "write_light",
    "add_to_cart":          "write_light",
    # COMP/IRR writes (booking/identity grade)
    "book_flight":          "write_booking",
    "modify_autopay":       "write_booking",
    "update_identity_doc":  "write_booking",
    # compensators inherit the booking-write grade (used when a comp plan is priced)
    "cancel_booking":       "write_booking",
    "revert_autopay":       "write_booking",
    "remove_from_cart":     "write_light",
}


def tool_class(fn):
    return TOOL_CLASS.get(fn, "read_lookup")


def sample_latency(example_id, fn, args, occurrence=0):
    """Deterministic per-instance draw. Same (example, fn, args, occurrence)
    => same latency in every run, arm, thread and process."""
    mu, sigma = CLASS_PARAMS[tool_class(fn)]
    key = f"{example_id}|{fn}|{json.dumps(args or {}, sort_keys=True)}|{occurrence}"
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    v = rng.lognormvariate(mu, sigma)
    return round(min(v, math.exp(mu + 3.09 * sigma)), 3)


# ---------------------------------------------------------------------------
# Completion accounting over a finished trace.
# ---------------------------------------------------------------------------
def _commit_ops(result):
    """[(op_id, fn, args, t_commit)] in commit order, from tx_log (authoritative
    op identity) joined with the trace/commits nominal stamps."""
    stamps = {c["op_id"]: c["t_commit"]
              for c in (result.get("trace", {}).get("commits")
                        or result.get("commits") or [])}
    out = []
    for rec in result.get("tx_log", []):
        if rec.get("op") == "commit":
            oid = rec["op_id"]
            out.append((oid, rec["fn"], rec.get("args", {}) or {},
                        stamps.get(oid, rec.get("t") or 0.0)))
    out.sort(key=lambda x: (x[3], x[0]))
    return out


def schedule(result, edges=None):
    """Realistic-profile execution schedule for one result file.

    edges : {child_op_id: [parent_op_id, ...]} result-dependency DAG (tact arm);
            None/{} => fully independent. Blocking arm ignores edges (serial).
    Returns {per_op: [...], result_ready, mode, profile}.
    """
    mode = result.get("mode", "tact")
    ops = _commit_ops(result)
    example_id = result.get("example_id", "")
    seen, per_op, done_at = {}, [], {}
    serial_free = 0.0
    for oid, fn, args, t_commit in ops:
        k = (fn, json.dumps(args, sort_keys=True))
        occ = seen.get(k, 0)
        seen[k] = occ + 1
        lat = sample_latency(example_id, fn, args, occ)
        if mode == "blocking":
            start = max(t_commit, serial_free)
        else:
            parents = (edges or {}).get(oid, [])
            start = max([t_commit] + [done_at[p] for p in parents if p in done_at])
        end = round(start + lat, 3)
        serial_free = end
        done_at[oid] = end
        per_op.append({"op_id": oid, "fn": fn, "class": tool_class(fn),
                       "lat_s": lat, "t_commit": round(t_commit, 3),
                       "t_start": round(start, 3), "t_done": end})
    return {"profile": PROFILE_VERSION, "mode": mode, "per_op": per_op,
            "result_ready": max(done_at.values()) if done_at else None}


def attach(result, t_user_end, edges=None):
    """Compute and attach result['latency_realistic'] (additive; frozen official
    fields untouched). First-response convention mirrors the frozen W2 formulas:
    blocking speaks only after results; tact's say anchor is tool-independent."""
    sched = schedule(result, edges=edges)
    ready = sched["result_ready"]
    completion = round(max(0.0, ready - t_user_end), 3) if ready is not None else None
    if result.get("mode") == "blocking":
        first = completion
    else:
        first = result.get("latency", {}).get("first_response_s")
        if not result.get("latency", {}).get("ack_emitted", False):
            first = completion            # no say fell back to result_ready
    result["latency_realistic"] = {
        "profile": PROFILE_VERSION,
        "first_response_s": first,
        "task_completion_s": completion,
        "per_op": sched["per_op"],
    }
    return result["latency_realistic"]


if __name__ == "__main__":
    for fn in ("get_exchange_rate", "search_flights", "add_to_cart", "book_flight"):
        draws = [sample_latency("selftest", fn, {"x": 1}, k) for k in range(1000)]
        draws.sort()
        print(f"{fn:22s} {tool_class(fn):14s} p50={draws[500]:.3f} p95={draws[950]:.3f}")
