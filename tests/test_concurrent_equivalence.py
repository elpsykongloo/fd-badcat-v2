#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W2 Phase-A R1: Test concurrent trace multiset comparison.

Validates that:
  1. Identical ordered spines + identical concurrent multisets => EQUIVALENT
  2. Same multiset but different order => EQUIVALENT (this is the W1 fix)
  3. Different multisets => NOT EQUIVALENT
  4. Ordered spine divergence => NOT EQUIVALENT (even if multisets match)
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from concurrent_trace_checker import (  # noqa: E402
    EventSignature,
    TracePartition,
    compare_concurrent_traces,
    compare_multisets,
    partition_trace,
)


def make_event(kind, turn=0, content="", state="LISTEN", t_audio=0.0):
    """Helper: create a trace event."""
    ev = {"event": kind, "data": {"turn": turn, "state": state, "t_audio": t_audio}}
    if content:
        ev["data"]["content"] = content
    return ev


# ---------------------------------------------------------------------------
# T1: EventSignature identity and hashing
# ---------------------------------------------------------------------------
def test_signature_identity():
    """EventSignatures with same (kind, turn, content_hash) are equal."""
    s1 = EventSignature("asr_done", 1, "abc123")
    s2 = EventSignature("asr_done", 1, "abc123")
    s3 = EventSignature("asr_done", 2, "abc123")
    s4 = EventSignature("asr_done", 1, "def456")

    assert s1 == s2, "identical signatures must be equal"
    assert s1 != s3, "different turns must not be equal"
    assert s1 != s4, "different content_hash must not be equal"
    assert hash(s1) == hash(s2), "equal signatures must hash identically"

    # Counter deduplication works
    c = Counter([s1, s2, s3])
    assert c[s1] == 2, "s1 and s2 should collapse to same key"
    assert c[s3] == 1


# ---------------------------------------------------------------------------
# T2: Trace partitioning
# ---------------------------------------------------------------------------
def test_partition_ordered_vs_concurrent():
    """Partition separates ordered spine, concurrent events, and informational."""
    events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("llm_dispatch", turn=1, t_audio=0.2),
        make_event("asr_done", turn=1, content="hello", t_audio=0.3),  # concurrent
        make_event("llm_done", turn=1, content="response", t_audio=0.5),
        make_event("asr_done", turn=1, content="world", t_audio=0.6),  # concurrent
        make_event("llm_stale_dropped", turn=1, t_audio=0.7),  # informational
        make_event("state_change", turn=2, state="SPEAK", t_audio=0.8),
    ]

    part = partition_trace(events)
    assert len(part.ordered) == 4, "should have 4 ordered events"
    assert len(part.concurrent) == 2, "should have 2 concurrent (asr_done) events"
    assert len(part.informational) == 1, "should have 1 informational event"

    ordered_kinds = [e["event"] for e in part.ordered]
    assert ordered_kinds == ["vad_start", "llm_dispatch", "llm_done", "state_change"]

    concurrent_kinds = [e["event"] for e in part.concurrent]
    assert concurrent_kinds == ["asr_done", "asr_done"]


# ---------------------------------------------------------------------------
# T3: Multiset comparison (the core fix)
# ---------------------------------------------------------------------------
def test_multiset_equal():
    """Identical multisets are equal regardless of order."""
    # Create two traces with asr_done events in DIFFERENT order
    a_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("asr_done", turn=1, content="A", t_audio=0.2),
        make_event("asr_done", turn=1, content="B", t_audio=0.3),
        make_event("llm_done", turn=1, content="response", t_audio=0.5),
    ]

    b_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("asr_done", turn=1, content="B", t_audio=0.25),  # B arrives first
        make_event("asr_done", turn=1, content="A", t_audio=0.35),  # A arrives second
        make_event("llm_done", turn=1, content="response", t_audio=0.5),
    ]

    cmp = compare_concurrent_traces(a_events, b_events)
    assert cmp.equivalent, "traces should be equivalent despite reordering"
    assert cmp.ordered_equal, "ordered spine should match"
    assert cmp.multiset_diff.equal, "concurrent multiset should match"


def test_multiset_mismatch_missing():
    """Multisets with different counts are not equal."""
    a_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("asr_done", turn=1, content="hello", t_audio=0.2),
        make_event("asr_done", turn=1, content="world", t_audio=0.3),
    ]

    b_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("asr_done", turn=1, content="hello", t_audio=0.2),
        # missing second asr_done
    ]

    cmp = compare_concurrent_traces(a_events, b_events)
    assert not cmp.equivalent, "traces should NOT be equivalent"
    assert not cmp.multiset_diff.equal, "multiset should mismatch"
    assert len(cmp.multiset_diff.only_in_a) > 0, "should report extra in A"


def test_multiset_mismatch_different_content():
    """Multisets with different content are not equal."""
    a_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("asr_done", turn=1, content="hello", t_audio=0.2),
    ]

    b_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("asr_done", turn=1, content="goodbye", t_audio=0.2),  # different
    ]

    cmp = compare_concurrent_traces(a_events, b_events)
    assert not cmp.equivalent
    assert not cmp.multiset_diff.equal


# ---------------------------------------------------------------------------
# T4: Ordered spine divergence detection
# ---------------------------------------------------------------------------
def test_ordered_divergence():
    """Ordered spine divergence is detected even if multisets match."""
    a_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("llm_done", turn=1, content="A", t_audio=0.3),
        make_event("state_change", turn=1, state="SPEAK", t_audio=0.4),
    ]

    b_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("state_change", turn=1, state="SPEAK", t_audio=0.25),  # reordered
        make_event("llm_done", turn=1, content="A", t_audio=0.3),
    ]

    cmp = compare_concurrent_traces(a_events, b_events)
    assert not cmp.equivalent, "ordered spine divergence => NOT equivalent"
    assert not cmp.ordered_equal
    assert cmp.ordered_first_diff == 1, "divergence at index 1"


def test_ordered_length_mismatch():
    """Length mismatch in ordered spine is divergence."""
    a_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("vad_end", turn=1, t_audio=0.5),
        make_event("llm_done", turn=1, content="A", t_audio=0.8),
    ]

    b_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("vad_end", turn=1, t_audio=0.5),
        # missing llm_done
    ]

    cmp = compare_concurrent_traces(a_events, b_events)
    assert not cmp.equivalent
    assert not cmp.ordered_equal
    assert cmp.ordered_first_diff == 2, "divergence at length boundary"


# ---------------------------------------------------------------------------
# T5: Informational events are excluded
# ---------------------------------------------------------------------------
def test_informational_excluded():
    """Informational events (actor-only) don't affect equivalence."""
    a_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("llm_done", turn=1, content="A", t_audio=0.3),
    ]

    b_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("llm_stale_dropped", turn=1, t_audio=0.2),  # actor-only
        make_event("llm_timeout", turn=1, t_audio=0.25),        # actor-only
        make_event("llm_done", turn=1, content="A", t_audio=0.3),
    ]

    cmp = compare_concurrent_traces(a_events, b_events)
    assert cmp.equivalent, "informational events should not affect equivalence"
    assert cmp.info_counts.get("llm_stale_dropped", 0) == 1
    assert cmp.info_counts.get("llm_timeout", 0) == 1


# ---------------------------------------------------------------------------
# T6: Real-world scenario (W1 HumDial regression pattern)
# ---------------------------------------------------------------------------
def test_w1_regression_pattern():
    """Simulate W1 HumDial pattern: asr races with response+tts.

    Legacy: asr may complete before or after response due to async races.
    Actor: same nondeterminism under concurrent mode.
    Both are correct as long as multisets match.
    """
    # Legacy trace: asr finishes after response
    legacy = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("vad_end", turn=1, t_audio=1.0),
        make_event("llm_dispatch", turn=1, t_audio=1.7),  # judge+response
        make_event("llm_done", turn=1, content="回复内容", t_audio=2.0),
        make_event("state_change", turn=1, state="SPEAK", t_audio=2.1),
        make_event("playback_start", turn=1, t_audio=2.2),
        make_event("asr_done", turn=1, content="用户说的话", t_audio=2.5),  # late
    ]

    # Actor trace: asr finishes before response (concurrent mode, different timing)
    actor = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("vad_end", turn=1, t_audio=1.0),
        make_event("asr_done", turn=1, content="用户说的话", t_audio=1.2),  # early
        make_event("llm_dispatch", turn=1, t_audio=1.7),
        make_event("llm_done", turn=1, content="回复内容", t_audio=2.0),
        make_event("state_change", turn=1, state="SPEAK", t_audio=2.1),
        make_event("playback_start", turn=1, t_audio=2.2),
    ]

    cmp = compare_concurrent_traces(legacy, actor)
    assert cmp.equivalent, "W1 pattern: asr timing variance should not break equivalence"
    assert cmp.ordered_equal, "ordered spine (vad, llm, state) should match"
    assert cmp.multiset_diff.equal, "asr_done content should match as multiset"


# ---------------------------------------------------------------------------
# T7: Duplicate concurrent events (same content, different timing)
# ---------------------------------------------------------------------------
def test_duplicate_concurrent_events():
    """Multiple asr_done with identical content (e.g., retries) must match as multiset."""
    a_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("asr_done", turn=1, content="重复", t_audio=0.2),
        make_event("asr_done", turn=1, content="重复", t_audio=0.3),  # duplicate
        make_event("vad_end", turn=1, t_audio=0.5),
    ]

    b_events = [
        make_event("vad_start", turn=1, t_audio=0.1),
        make_event("asr_done", turn=1, content="重复", t_audio=0.25),
        make_event("asr_done", turn=1, content="重复", t_audio=0.35),  # duplicate, diff time
        make_event("vad_end", turn=1, t_audio=0.5),
    ]

    cmp = compare_concurrent_traces(a_events, b_events)
    assert cmp.equivalent, "duplicate concurrent events should match as multiset"

    # Now remove one duplicate in B
    b_events_short = b_events[:-2] + b_events[-1:]  # remove one asr_done
    cmp2 = compare_concurrent_traces(a_events, b_events_short)
    assert not cmp2.equivalent, "missing duplicate should cause mismatch"


# ---------------------------------------------------------------------------
# T8: Empty traces
# ---------------------------------------------------------------------------
def test_empty_traces():
    """Empty traces are equivalent."""
    cmp = compare_concurrent_traces([], [])
    assert cmp.equivalent


def test_one_empty_trace():
    """One empty trace vs non-empty is not equivalent."""
    a = [make_event("vad_start", turn=1, t_audio=0.1)]
    b = []
    cmp = compare_concurrent_traces(a, b)
    assert not cmp.equivalent


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
def run_all():
    tests = [
        test_signature_identity,
        test_partition_ordered_vs_concurrent,
        test_multiset_equal,
        test_multiset_mismatch_missing,
        test_multiset_mismatch_different_content,
        test_ordered_divergence,
        test_ordered_length_mismatch,
        test_informational_excluded,
        test_w1_regression_pattern,
        test_duplicate_concurrent_events,
        test_empty_traces,
        test_one_empty_trace,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
        except Exception as e:
            print(f"✗ {t.__name__}: EXCEPTION {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
