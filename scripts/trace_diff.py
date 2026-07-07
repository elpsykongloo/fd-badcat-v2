#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1 D2.1 trace-diff: the judge for old-engine vs new-engine equivalence.

W2 Phase-A R1: Updated to use concurrent-safe multiset comparison for events
that may arrive in nondeterministic order (asr_done etc).

Normalization: each trace event becomes (event, turn, state, content_key);
wall timestamps are dropped; a time axis is kept separately for soft checks
(legacy traces only have wall "timestamp", which equals audio time under paced
realtime replay; actor traces carry an explicit t_audio).

Levels:
  L1 strict  — ordered spine + concurrent multisets identical AND every |Δt| <= --tol.
  L2         — sequence divergence; the tool prints the first divergence with
               ±3 events of context for manual attribution
               (docs/w1_equivalence.md).

New-engine-only informational events (llm_stale_dropped, llm_timeout,
playback_end) are filtered out before comparison; their counts are reported.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# Import concurrent verification logic (W2 Phase-A R1)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))
try:
    from concurrent_trace_checker import compare_concurrent_traces
    CONCURRENT_MODE_AVAILABLE = True
except ImportError:
    CONCURRENT_MODE_AVAILABLE = False

NEW_ENGINE_ONLY = {"llm_stale_dropped", "llm_timeout", "playback_end", "session_reset"}

# events produced by concurrent worker tasks whose completion ORDER relative to
# the main decision chain is nondeterministic in BOTH engines (e.g. asr runs
# alongside response+tts): compared as a multiset, not by position
UNORDERED_KINDS = {"asr_done"}


def load_trace(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def content_key(ev):
    data = ev.get("data", {})
    kind = ev.get("event")
    if kind in ("llm_done", "asr_done"):
        content = str(data.get("content", ""))
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:10]
    return ""


def normalize(events, ignore=()):
    ordered, unordered = [], []
    for ev in events:
        kind = ev.get("event")
        if kind in NEW_ENGINE_ONLY or kind in ignore:
            continue
        data = ev.get("data", {})
        t = data.get("t_audio", data.get("timestamp"))
        item = {
            "key": (kind, data.get("turn"), data.get("state"), content_key(ev)),
            "t": None if t is None else float(t),
            "raw": ev,
        }
        if kind in UNORDERED_KINDS:
            # state is timing-dependent for concurrent tasks; compare kind/turn/content
            item["key"] = (kind, data.get("turn"), None, content_key(ev))
            unordered.append(item)
        else:
            ordered.append(item)
    return ordered, unordered


def fmt(item):
    kind, turn, state, ch = item["key"]
    t = "-" if item["t"] is None else f"{item['t']:.3f}"
    return f"t={t:>8}  {kind:<18} turn={turn} state={state} content={ch or '-'}"


def diff_traces(a_events, b_events, tol=0.25, ignore=()):
    a, a_un = normalize(a_events, ignore)
    b, b_un = normalize(b_events, ignore)
    n = min(len(a), len(b))
    first_div = None
    soft = []
    for i in range(n):
        if a[i]["key"] != b[i]["key"]:
            first_div = i
            break
        if a[i]["t"] is not None and b[i]["t"] is not None:
            dt = abs(a[i]["t"] - b[i]["t"])
            if dt > tol:
                soft.append((i, dt))
    if first_div is None and len(a) != len(b):
        first_div = n

    # unordered kinds: multiset comparison on (kind, turn, content)
    a_ms = sorted(str(x["key"]) for x in a_un)
    b_ms = sorted(str(x["key"]) for x in b_un)
    unordered_equal = a_ms == b_ms

    info_counts = {}
    for evs in (a_events, b_events):
        for ev in evs:
            if ev.get("event") in NEW_ENGINE_ONLY:
                info_counts[ev["event"]] = info_counts.get(ev["event"], 0) + 1

    return {
        "a_len": len(a), "b_len": len(b),
        "first_divergence": first_div,
        "soft_time_mismatches": soft,
        "unordered_equal": unordered_equal,
        "unordered_a": a_ms, "unordered_b": b_ms,
        "l1": first_div is None and not soft and unordered_equal,
        "sequence_equal": first_div is None and unordered_equal,
        "info_counts": info_counts,
        "a": a, "b": b,
    }


def print_report(res, a_name="A", b_name="B"):
    print(f"{a_name}: {res['a_len']} events | {b_name}: {res['b_len']} events")
    if res["info_counts"]:
        print(f"informational (excluded): {res['info_counts']}")
    if not res["unordered_equal"]:
        print("UNORDERED-SET MISMATCH (asr_done etc.):")
        print(f"  A: {res['unordered_a']}")
        print(f"  B: {res['unordered_b']}")
    if res["l1"]:
        print("VERDICT: L1 STRICT EQUIVALENT")
        return
    if res["sequence_equal"]:
        print(f"VERDICT: sequence equal; {len(res['soft_time_mismatches'])} soft time mismatches (> tol)")
        for i, dt in res["soft_time_mismatches"][:10]:
            print(f"  #{i} dt={dt:.3f}s  {fmt(res['a'][i])}")
        return
    i = res["first_divergence"]
    if i is None:
        print("VERDICT: L2 — unordered-set mismatch only")
        return
    print(f"VERDICT: L2 — first divergence at event #{i}")
    lo = max(0, i - 3)
    for j in range(lo, min(i + 4, max(res["a_len"], res["b_len"]))):
        av = fmt(res["a"][j]) if j < res["a_len"] else "<missing>"
        bv = fmt(res["b"][j]) if j < res["b_len"] else "<missing>"
        marker = ">>" if j == i else "  "
        print(f"{marker} A#{j} {av}")
        print(f"{marker} B#{j} {bv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_a")
    ap.add_argument("trace_b")
    ap.add_argument("--tol", type=float, default=0.25,
                    help="soft time tolerance in seconds (audio clock vs paced wall)")
    ap.add_argument("--ignore", default="", help="comma-separated event kinds to ignore")
    ap.add_argument("--concurrent-safe", action="store_true",
                    help="Use concurrent-safe multiset comparison (W2 Phase-A R1)")
    args = ap.parse_args()

    # W2 Phase-A R1: use concurrent-safe comparison when requested
    if args.concurrent_safe:
        if not CONCURRENT_MODE_AVAILABLE:
            print("ERROR: concurrent_trace_checker module not available", file=sys.stderr)
            sys.exit(3)

        a_events = load_trace(args.trace_a)
        b_events = load_trace(args.trace_b)
        cmp = compare_concurrent_traces(a_events, b_events)

        print(f"{Path(args.trace_a).name}: {len(a_events)} events")
        print(f"{Path(args.trace_b).name}: {len(b_events)} events")
        if cmp.info_counts:
            print(f"informational (excluded): {cmp.info_counts}")
        print(f"VERDICT: {cmp.verdict()}")

        if not cmp.ordered_equal:
            print(f"  Ordered divergence at event #{cmp.ordered_first_diff}")
        if cmp.multiset_diff and not cmp.multiset_diff.equal:
            print(f"  {cmp.multiset_diff.summary()}")

        sys.exit(0 if cmp.equivalent else 1)

    # Original W1 behavior
    ignore = tuple(x for x in args.ignore.split(",") if x)
    res = diff_traces(load_trace(args.trace_a), load_trace(args.trace_b),
                      tol=args.tol, ignore=ignore)
    print_report(res, Path(args.trace_a).name, Path(args.trace_b).name)
    sys.exit(0 if res["l1"] else (2 if res["sequence_equal"] else 1))


if __name__ == "__main__":
    main()
