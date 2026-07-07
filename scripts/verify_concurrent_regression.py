#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W2 Phase-A R1: Concurrent regression verification harness.

Validates that Phase-B concurrent execution produces equivalent results to
Phase-A baseline using multiset comparison for concurrent events.

Usage:
  # Compare actor traces against legacy golden set
  python scripts/verify_concurrent_regression.py \
      --baseline traces/golden/ \
      --candidate traces/golden_actor/ \
      --verbose

  # Batch verify HumDial regression under concurrent mode
  python scripts/verify_concurrent_regression.py \
      --baseline exp/humdial_100_legacy/traces/ \
      --candidate exp/humdial_100_actor_concurrent/traces/ \
      --concurrency-mode injected
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from concurrent_trace_checker import (  # noqa: E402
    compare_concurrent_traces,
    load_trace,
    verify_batch,
    BatchVerificationResult,
)


def find_trace_pairs(baseline_dir: Path, candidate_dir: Path):
    """Find all matching trace pairs between baseline and candidate."""
    pairs = []
    for baseline_path in sorted(baseline_dir.glob("*.jsonl")):
        candidate_path = candidate_dir / baseline_path.name
        if candidate_path.exists():
            pairs.append((baseline_path, candidate_path))
        else:
            print(f"WARNING: No candidate trace for {baseline_path.name}", file=sys.stderr)
    return pairs


def verify_single_pair(baseline_path: Path, candidate_path: Path, verbose: bool = True):
    """Verify a single trace pair and print detailed report."""
    baseline_events = load_trace(baseline_path)
    candidate_events = load_trace(candidate_path)

    cmp = compare_concurrent_traces(baseline_events, candidate_events)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Comparing: {baseline_path.name}")
        print(f"  Baseline:  {baseline_path}")
        print(f"  Candidate: {candidate_path}")
        print(f"{'='*70}")
        print(f"Baseline events:  {len(baseline_events)}")
        print(f"Candidate events: {len(candidate_events)}")

        if cmp.info_counts:
            print(f"\nInformational events (excluded from comparison):")
            for kind, count in sorted(cmp.info_counts.items()):
                print(f"  {kind}: {count}")

        print(f"\nVERDICT: {cmp.verdict()}")

        if not cmp.ordered_equal:
            print(f"\n⚠ Ordered spine divergence at event #{cmp.ordered_first_diff}")
            # Show context around divergence
            baseline_part = baseline_events[max(0, cmp.ordered_first_diff-2):cmp.ordered_first_diff+3]
            candidate_part = candidate_events[max(0, cmp.ordered_first_diff-2):cmp.ordered_first_diff+3]
            print("\nBaseline context:")
            for i, ev in enumerate(baseline_part, start=max(0, cmp.ordered_first_diff-2)):
                marker = ">>>" if i == cmp.ordered_first_diff else "   "
                print(f"  {marker} #{i}: {ev.get('event')} {ev.get('data', {}).get('state', '')}")
            print("\nCandidate context:")
            for i, ev in enumerate(candidate_part, start=max(0, cmp.ordered_first_diff-2)):
                marker = ">>>" if i == cmp.ordered_first_diff else "   "
                print(f"  {marker} #{i}: {ev.get('event')} {ev.get('data', {}).get('state', '')}")

        if cmp.multiset_diff and not cmp.multiset_diff.equal:
            print(f"\n⚠ Concurrent event multiset mismatch:")
            print(cmp.multiset_diff.summary())

        if cmp.equivalent:
            print("\n✓ PASS: Traces are concurrent-equivalent")
        else:
            print("\n✗ FAIL: Traces are NOT equivalent")

    return cmp.equivalent


def main():
    ap = argparse.ArgumentParser(
        description="Verify concurrent execution equivalence using multiset comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify Phase-A golden set equivalence
  python scripts/verify_concurrent_regression.py \\
      --baseline traces/golden/ --candidate traces/golden_actor/

  # Batch verify with summary report
  python scripts/verify_concurrent_regression.py \\
      --baseline traces/legacy/ --candidate traces/actor_concurrent/ \\
      --batch --summary-only

  # Single pair with full detail
  python scripts/verify_concurrent_regression.py \\
      --baseline traces/golden/sample001.jsonl \\
      --candidate traces/golden_actor/sample001.jsonl
        """
    )
    ap.add_argument("--baseline", required=True,
                    help="Baseline trace file or directory")
    ap.add_argument("--candidate", required=True,
                    help="Candidate trace file or directory")
    ap.add_argument("--batch", action="store_true",
                    help="Batch mode: compare all pairs in directories")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Verbose per-pair output")
    ap.add_argument("--summary-only", action="store_true",
                    help="Only print summary statistics (implies --batch)")
    ap.add_argument("--fail-fast", action="store_true",
                    help="Exit on first failure")
    ap.add_argument("--concurrency-mode", choices=["realtime", "injected", "oracle"],
                    help="Document the concurrency mode for audit trail")

    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)

    if not baseline_path.exists():
        print(f"ERROR: Baseline path does not exist: {baseline_path}", file=sys.stderr)
        sys.exit(1)

    if not candidate_path.exists():
        print(f"ERROR: Candidate path does not exist: {candidate_path}", file=sys.stderr)
        sys.exit(1)

    # Print audit header
    print("Concurrent Regression Verification")
    print("=" * 70)
    print(f"Baseline:  {baseline_path}")
    print(f"Candidate: {candidate_path}")
    if args.concurrency_mode:
        print(f"Concurrency mode: {args.concurrency_mode}")
    print("=" * 70)

    # Determine mode
    if args.summary_only or (args.batch and baseline_path.is_dir()):
        # Batch mode
        if not baseline_path.is_dir() or not candidate_path.is_dir():
            print("ERROR: --batch requires both paths to be directories", file=sys.stderr)
            sys.exit(1)

        pairs = find_trace_pairs(baseline_path, candidate_path)
        if not pairs:
            print("ERROR: No matching trace pairs found", file=sys.stderr)
            sys.exit(1)

        print(f"\nFound {len(pairs)} trace pairs to verify\n")

        result = verify_batch(pairs, verbose=(args.verbose and not args.summary_only))

        print("\n" + "=" * 70)
        print("BATCH VERIFICATION SUMMARY")
        print("=" * 70)
        print(result.summary())

        exit_code = 0 if result.pass_rate == 1.0 else 1

        if result.failures and not args.summary_only:
            print("\nFailed traces:")
            for name, reason in result.failures:
                print(f"  {name}")
                print(f"    {reason}")

        sys.exit(exit_code)

    else:
        # Single-pair mode
        if baseline_path.is_dir() or candidate_path.is_dir():
            print("ERROR: Single-pair mode requires file paths, not directories", file=sys.stderr)
            print("       Use --batch for directory comparison", file=sys.stderr)
            sys.exit(1)

        success = verify_single_pair(baseline_path, candidate_path, verbose=True)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
