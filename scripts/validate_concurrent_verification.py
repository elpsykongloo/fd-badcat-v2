#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W2 Phase-A R1: Quick validation — run all concurrent verification tests.

This script is the single entry point for validating the concurrent trace
comparison implementation. Run before merging Phase-A R1.

Exit codes:
  0 — All tests pass, ready for Phase-B
  1 — Test failures, review output
  2 — Missing dependencies
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_test(name, cmd, required=True):
    """Run a test command and report result."""
    print(f"\n{'='*70}")
    print(f"[TEST] {name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=ROOT)

    if result.returncode == 0:
        print(f"\n✓ PASS: {name}")
        return True
    else:
        print(f"\n✗ FAIL: {name} (exit code {result.returncode})")
        if required:
            print("   This is a BLOCKING failure")
        return False


def main():
    print("W2 Phase-A R1: Concurrent Verification — Full Validation Suite")
    print("="*70)

    python = "/root/miniconda3/envs/fd-sds/bin/python"

    tests = [
        # Core unit tests (BLOCKING)
        ("Unit tests (multiset comparison)",
         [python, "tests/test_concurrent_equivalence.py"],
         True),

        # Integration tests (INFORMATIONAL — golden traces may have real divergences)
        ("Integration tests (golden traces)",
         [python, "tests/test_concurrent_integration.py"],
         False),

        # Existing W1 engine tests (BLOCKING — must not regress)
        ("W1 engine tests (S2 suite)",
         [python, "tests/test_engine.py"],
         True),
    ]

    results = []
    blocking_failed = False

    for name, cmd, required in tests:
        passed = run_test(name, cmd, required)
        results.append((name, passed, required))
        if not passed and required:
            blocking_failed = True

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    blocking_passed = sum(1 for _, p, req in results if p and req)
    blocking_total = sum(1 for _, _, req in results if req)
    info_passed = sum(1 for _, p, req in results if p and not req)
    info_total = sum(1 for _, _, req in results if not req)

    for name, passed, required in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        req_label = "[BLOCKING]" if required else "[INFO]"
        print(f"{status} {req_label} {name}")

    print()
    print(f"Blocking: {blocking_passed}/{blocking_total} passed")
    print(f"Informational: {info_passed}/{info_total} passed")

    if blocking_failed:
        print("\n✗ VALIDATION FAILED: Fix blocking failures before merging")
        sys.exit(1)
    else:
        print("\n✓ VALIDATION PASSED: Ready for Phase-B")
        sys.exit(0)


if __name__ == "__main__":
    main()
