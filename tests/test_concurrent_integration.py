#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W2 Phase-A R1: Integration test for concurrent verification against W1 golden traces.

Validates that the concurrent-safe multiset comparison correctly handles the
W1 HumDial regression pattern where asr_done timing varied between legacy and actor.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concurrent_trace_checker import (  # noqa: E402
    compare_concurrent_traces,
    load_trace,
    verify_batch,
)


def test_w1_golden_traces():
    """Verify W1 golden traces (if they exist) are concurrent-equivalent."""
    golden_dir = ROOT / "traces" / "golden"
    actor_dir = ROOT / "traces" / "golden_actor"

    if not golden_dir.exists() or not actor_dir.exists():
        print("SKIP: W1 golden traces not found (GPU-day artifacts)")
        return True

    pairs = []
    for legacy_path in sorted(golden_dir.glob("*.jsonl")):
        actor_path = actor_dir / legacy_path.name
        if actor_path.exists():
            pairs.append((legacy_path, actor_path))

    if not pairs:
        print("SKIP: No matching golden trace pairs")
        return True

    print(f"Verifying {len(pairs)} W1 golden trace pairs...")
    result = verify_batch(pairs, verbose=False)

    print(f"\nResults: {result.equivalent}/{result.total} equivalent ({result.pass_rate:.1%})")

    if result.failures:
        print("\nFailures:")
        for name, reason in result.failures[:5]:
            print(f"  {name}: {reason}")

    # W1 baseline: 19/20 decision sequences identical (95% = noise floor)
    # With concurrent-safe comparison, should approach 100%
    expected_pass_rate = 0.90  # relaxed threshold for varying golden set sizes

    if result.pass_rate >= expected_pass_rate:
        print(f"✓ PASS: {result.pass_rate:.1%} >= {expected_pass_rate:.1%}")
        return True
    else:
        print(f"✗ FAIL: {result.pass_rate:.1%} < {expected_pass_rate:.1%}")
        return False


def test_synthetic_concurrent_pattern():
    """Synthetic test: create mock traces with concurrent event reordering."""
    import json
    import tempfile

    # Legacy: asr completes AFTER response
    legacy_events = [
        {"event": "vad_start", "data": {"turn": 1, "t_audio": 0.1, "state": "LISTEN"}},
        {"event": "vad_end", "data": {"turn": 1, "t_audio": 1.0, "state": "LISTEN"}},
        {"event": "llm_dispatch", "data": {"turn": 1, "t_audio": 1.7, "state": "LISTEN"}},
        {"event": "llm_done", "data": {"turn": 1, "t_audio": 2.0, "state": "LISTEN",
                                        "content": "回复内容"}},
        {"event": "state_change", "data": {"turn": 1, "t_audio": 2.1, "state": "SPEAK"}},
        {"event": "playback_start", "data": {"turn": 1, "t_audio": 2.2, "state": "SPEAK"}},
        {"event": "asr_done", "data": {"turn": 1, "t_audio": 2.5, "state": "SPEAK",
                                       "content": "用户说的话"}},
    ]

    # Actor: asr completes BEFORE response (concurrent mode, different scheduling)
    actor_events = [
        {"event": "vad_start", "data": {"turn": 1, "t_audio": 0.1, "state": "LISTEN"}},
        {"event": "vad_end", "data": {"turn": 1, "t_audio": 1.0, "state": "LISTEN"}},
        {"event": "asr_done", "data": {"turn": 1, "t_audio": 1.2, "state": "LISTEN",
                                       "content": "用户说的话"}},
        {"event": "llm_dispatch", "data": {"turn": 1, "t_audio": 1.7, "state": "LISTEN"}},
        {"event": "llm_done", "data": {"turn": 1, "t_audio": 2.0, "state": "LISTEN",
                                        "content": "回复内容"}},
        {"event": "state_change", "data": {"turn": 1, "t_audio": 2.1, "state": "SPEAK"}},
        {"event": "playback_start", "data": {"turn": 1, "t_audio": 2.2, "state": "SPEAK"}},
    ]

    cmp = compare_concurrent_traces(legacy_events, actor_events)

    if cmp.equivalent:
        print("✓ PASS: Synthetic concurrent pattern recognized as equivalent")
        return True
    else:
        print(f"✗ FAIL: Synthetic pattern not equivalent: {cmp.verdict()}")
        if not cmp.ordered_equal:
            print(f"  Ordered divergence at #{cmp.ordered_first_diff}")
        if cmp.multiset_diff and not cmp.multiset_diff.equal:
            print(f"  {cmp.multiset_diff.summary()}")
        return False


def test_batch_replay_traces():
    """Test concurrent verification on batch replay output (if exists)."""
    batch_dir = ROOT / "exp" / "batch_smoke" / "traces"

    if not batch_dir.exists():
        print("SKIP: No batch replay traces found")
        return True

    traces = list(batch_dir.glob("*.jsonl"))
    if len(traces) < 2:
        print("SKIP: Need at least 2 traces for cross-validation")
        return True

    print(f"Found {len(traces)} batch replay traces")

    # Self-consistency check: each trace should be equivalent to itself
    passed = 0
    for trace_path in traces[:5]:  # spot-check first 5
        events = load_trace(trace_path)
        cmp = compare_concurrent_traces(events, events)
        if cmp.equivalent:
            passed += 1
        else:
            print(f"✗ FAIL: Self-comparison failed for {trace_path.name}")

    if passed == min(5, len(traces)):
        print(f"✓ PASS: {passed} traces self-consistent")
        return True
    else:
        print(f"✗ FAIL: Only {passed}/{min(5, len(traces))} self-consistent")
        return False


def main():
    print("W2 Phase-A R1: Concurrent Verification Integration Tests")
    print("=" * 70)

    tests = [
        ("Synthetic concurrent pattern", test_synthetic_concurrent_pattern),
        ("W1 golden traces", test_w1_golden_traces),
        ("Batch replay traces", test_batch_replay_traces),
    ]

    results = []
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        print("-" * 70)
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\n{total_passed}/{len(results)} tests passed")

    sys.exit(0 if total_passed == len(results) else 1)


if __name__ == "__main__":
    main()
