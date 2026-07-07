#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Concurrent trace verification toolkit (W2 Phase-A R1).

W1 discovered: in concurrent execution modes (吞吐轨), completion events from
parallel worker tasks arrive in nondeterministic order. The multiset of events
must be identical, but sequence comparison is too strict.

This module provides:
  1. Multiset comparison for concurrent events
  2. Partial-order verification (preserves causal dependencies)
  3. Aggregate statistics over batch runs
  4. Thread-safe trace collector for concurrent workers
"""
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set


# Events produced by concurrent workers whose relative ORDER is nondeterministic
# but whose MULTISET must match (trace_diff.py line 30)
CONCURRENT_EVENT_KINDS = {"asr_done"}

# Events that are causally ordered (must preserve sequence)
ORDERED_EVENT_KINDS = {
    "vad_start", "vad_end", "vad_640_done",
    "llm_dispatch", "llm_done",
    "decision", "state_change",
    "playback_start", "playback_end",
    "session_reset",
}

# Informational events (actor-only, excluded from equivalence)
INFORMATIONAL_KINDS = {"llm_stale_dropped", "llm_timeout", "session_reset"}


@dataclass
class EventSignature:
    """Normalized event identity for multiset comparison."""
    kind: str
    turn: int
    content_hash: str  # md5[:10] of text/audio content

    def __hash__(self):
        return hash((self.kind, self.turn, self.content_hash))

    def __eq__(self, other):
        return (self.kind == other.kind and
                self.turn == other.turn and
                self.content_hash == other.content_hash)

    def __str__(self):
        return f"{self.kind}@turn{self.turn}:{self.content_hash}"


@dataclass
class TracePartition:
    """Partition trace into ordered spine + concurrent events."""
    ordered: List[dict] = field(default_factory=list)
    concurrent: List[dict] = field(default_factory=list)
    informational: List[dict] = field(default_factory=list)

    def ordered_signatures(self) -> List[str]:
        """Return sequence of ordered event keys."""
        return [_event_key(e) for e in self.ordered]

    def concurrent_multiset(self) -> Counter:
        """Return multiset (counter) of concurrent event signatures."""
        return Counter(_make_signature(e) for e in self.concurrent)


def _content_hash(event: dict) -> str:
    """Extract content hash for deduplication."""
    data = event.get("data", {})
    kind = event.get("event")
    if kind in ("llm_done", "asr_done"):
        content = str(data.get("content", ""))
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:10]
    if kind == "playback_start":
        # TTS output may differ if regenerated; hash the audio bytes
        audio = data.get("audio_bytes", b"")
        if audio:
            return hashlib.md5(audio).hexdigest()[:10]
    return ""


def _event_key(event: dict) -> str:
    """Ordered event key (kind, turn, state, content_hash)."""
    data = event.get("data", {})
    kind = event.get("event")
    turn = data.get("turn", -1)
    state = data.get("state", "")
    ch = _content_hash(event)
    return f"{kind}@{turn}:{state}:{ch}"


def _make_signature(event: dict) -> EventSignature:
    """Build EventSignature for multiset comparison."""
    data = event.get("data", {})
    return EventSignature(
        kind=event.get("event"),
        turn=data.get("turn", -1),
        content_hash=_content_hash(event),
    )


def partition_trace(events: List[dict]) -> TracePartition:
    """Split trace into ordered spine, concurrent events, and informational."""
    part = TracePartition()
    for ev in events:
        kind = ev.get("event")
        if kind in INFORMATIONAL_KINDS:
            part.informational.append(ev)
        elif kind in CONCURRENT_EVENT_KINDS:
            part.concurrent.append(ev)
        else:
            part.ordered.append(ev)
    return part


@dataclass
class MultisetDiff:
    """Diff result for concurrent event multisets."""
    equal: bool
    only_in_a: Counter = field(default_factory=Counter)
    only_in_b: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        if self.equal:
            return "MULTISET EQUAL"
        lines = ["MULTISET MISMATCH:"]
        if self.only_in_a:
            lines.append(f"  Only in A: {dict(self.only_in_a)}")
        if self.only_in_b:
            lines.append(f"  Only in B: {dict(self.only_in_b)}")
        return "\n".join(lines)


def compare_multisets(a: Counter, b: Counter) -> MultisetDiff:
    """Compare two event multisets."""
    if a == b:
        return MultisetDiff(equal=True)
    only_a = a - b
    only_b = b - a
    return MultisetDiff(equal=False, only_in_a=only_a, only_in_b=only_b)


@dataclass
class ConcurrentTraceComparison:
    """Complete comparison result for concurrent traces."""
    ordered_equal: bool
    ordered_first_diff: int = -1  # index of first ordered divergence
    multiset_diff: MultisetDiff = None
    info_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def equivalent(self) -> bool:
        """True if traces are equivalent under concurrent semantics."""
        return self.ordered_equal and (self.multiset_diff is None or self.multiset_diff.equal)

    def verdict(self) -> str:
        if self.equivalent:
            return "CONCURRENT-EQUIVALENT"
        if not self.ordered_equal:
            return f"ORDERED-DIVERGENCE at event #{self.ordered_first_diff}"
        return "MULTISET-MISMATCH (concurrent events)"


def compare_concurrent_traces(a_events: List[dict], b_events: List[dict]) -> ConcurrentTraceComparison:
    """Compare two traces with concurrent-safe semantics.

    Returns:
        ConcurrentTraceComparison with detailed diff information.
    """
    a_part = partition_trace(a_events)
    b_part = partition_trace(b_events)

    # Compare ordered spine (must match exactly)
    a_seq = a_part.ordered_signatures()
    b_seq = b_part.ordered_signatures()
    ordered_equal = a_seq == b_seq
    first_diff = -1
    if not ordered_equal:
        for i, (av, bv) in enumerate(zip(a_seq, b_seq)):
            if av != bv:
                first_diff = i
                break
        if first_diff == -1:  # length mismatch
            first_diff = min(len(a_seq), len(b_seq))

    # Compare concurrent multisets
    a_ms = a_part.concurrent_multiset()
    b_ms = b_part.concurrent_multiset()
    ms_diff = compare_multisets(a_ms, b_ms)

    # Collect informational counts
    info_counts = Counter()
    for ev in a_part.informational + b_part.informational:
        info_counts[ev.get("event")] += 1

    return ConcurrentTraceComparison(
        ordered_equal=ordered_equal,
        ordered_first_diff=first_diff,
        multiset_diff=ms_diff,
        info_counts=dict(info_counts),
    )


def load_trace(path: Path) -> List[dict]:
    """Load trace from jsonl file."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


@dataclass
class BatchVerificationResult:
    """Aggregate verification result over multiple trace pairs."""
    total: int = 0
    equivalent: int = 0
    ordered_diverged: int = 0
    multiset_diverged: int = 0
    failures: List[Tuple[str, str]] = field(default_factory=list)  # (name, reason)

    @property
    def pass_rate(self) -> float:
        return self.equivalent / self.total if self.total > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Batch verification: {self.equivalent}/{self.total} equivalent ({self.pass_rate:.1%})",
            f"  Ordered diverged: {self.ordered_diverged}",
            f"  Multiset diverged: {self.multiset_diverged}",
        ]
        if self.failures:
            lines.append(f"\nFailures ({len(self.failures)}):")
            for name, reason in self.failures[:10]:
                lines.append(f"  {name}: {reason}")
            if len(self.failures) > 10:
                lines.append(f"  ... and {len(self.failures) - 10} more")
        return "\n".join(lines)


def verify_batch(trace_pairs: List[Tuple[Path, Path]], verbose: bool = False) -> BatchVerificationResult:
    """Verify equivalence across a batch of trace pairs.

    Args:
        trace_pairs: List of (legacy_trace, actor_trace) path pairs
        verbose: Print per-pair results

    Returns:
        BatchVerificationResult with aggregate statistics
    """
    result = BatchVerificationResult(total=len(trace_pairs))

    for a_path, b_path in trace_pairs:
        try:
            a_events = load_trace(a_path)
            b_events = load_trace(b_path)
            cmp = compare_concurrent_traces(a_events, b_events)

            if cmp.equivalent:
                result.equivalent += 1
            else:
                if not cmp.ordered_equal:
                    result.ordered_diverged += 1
                    result.failures.append((a_path.stem, cmp.verdict()))
                if cmp.multiset_diff and not cmp.multiset_diff.equal:
                    result.multiset_diverged += 1
                    result.failures.append((a_path.stem, cmp.multiset_diff.summary()))

            if verbose:
                print(f"{a_path.stem}: {cmp.verdict()}")
                if not cmp.equivalent:
                    if not cmp.ordered_equal:
                        print(f"  Ordered diff at #{cmp.ordered_first_diff}")
                    if cmp.multiset_diff and not cmp.multiset_diff.equal:
                        print(f"  {cmp.multiset_diff.summary()}")

        except Exception as e:
            result.failures.append((a_path.stem, f"ERROR: {e}"))

    return result


# Thread-safe trace collector for concurrent workers
class ConcurrentTraceCollector:
    """Thread-safe trace event collector for concurrent replay.

    Each worker writes to its own buffer; final traces are deterministically
    sorted by (turn, t_audio, seq) to enable multiset comparison.
    """
    def __init__(self):
        self.buffers: Dict[int, List[dict]] = defaultdict(list)
        self._lock = None  # lazy init in async context

    async def append(self, worker_id: int, event: dict):
        """Append event to worker's buffer (async-safe)."""
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()
        async with self._lock:
            self.buffers[worker_id].append(event)

    def merge(self) -> List[dict]:
        """Merge all buffers into a single trace, sorted by logical time."""
        all_events = []
        for events in self.buffers.values():
            all_events.extend(events)

        # Sort by (turn, t_audio, seq) for deterministic output
        def sort_key(ev):
            data = ev.get("data", {})
            return (
                data.get("turn", -1),
                data.get("t_audio", data.get("timestamp", 0)),
                data.get("seq", 0),
            )

        return sorted(all_events, key=sort_key)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Verify concurrent trace equivalence")
    ap.add_argument("trace_a", help="First trace (legacy or actor)")
    ap.add_argument("trace_b", help="Second trace (actor or regenerated)")
    ap.add_argument("--batch", help="Batch mode: compare all trace pairs in two directories")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print per-pair results")
    args = ap.parse_args()

    if args.batch:
        # Batch mode: compare matching stems in two directories
        dir_a = Path(args.trace_a)
        dir_b = Path(args.trace_b)
        pairs = []
        for a_path in sorted(dir_a.glob("*.jsonl")):
            b_path = dir_b / a_path.name
            if b_path.exists():
                pairs.append((a_path, b_path))

        result = verify_batch(pairs, verbose=args.verbose)
        print(result.summary())
        exit(0 if result.pass_rate == 1.0 else 1)

    else:
        # Single-pair mode
        a_events = load_trace(Path(args.trace_a))
        b_events = load_trace(Path(args.trace_b))
        cmp = compare_concurrent_traces(a_events, b_events)

        print(f"Trace A: {len(a_events)} events")
        print(f"Trace B: {len(b_events)} events")
        if cmp.info_counts:
            print(f"Informational (excluded): {cmp.info_counts}")
        print(f"\nVERDICT: {cmp.verdict()}")

        if not cmp.ordered_equal:
            print(f"  First ordered divergence at event #{cmp.ordered_first_diff}")

        if cmp.multiset_diff and not cmp.multiset_diff.equal:
            print(f"\n{cmp.multiset_diff.summary()}")

        exit(0 if cmp.equivalent else 1)
