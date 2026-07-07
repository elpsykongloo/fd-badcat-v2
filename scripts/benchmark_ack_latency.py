#!/usr/bin/env python3
"""
Benchmark ack-v0 first-response latency improvement.

Measures:
- Baseline: single-phase TTS latency (full response)
- ack-v0: two-phase TTS latency (ack first, then main)
- First-response improvement: baseline_lat - ack_lat

Uses real Qwen3-Omni TTS to get accurate measurements.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tts_ack import synthesize_with_ack, synthesize_baseline
from module import tts


# Test cases: realistic Phase-B responses
TEST_CASES = [
    {
        "id": "search_flights",
        "say": "I've searched for flights to New York on July 15th and found several options available.",
        "ops": [{"type": "launch", "fn": "search_flights", "args": {"destination": "New York", "date": "July 15"}}],
    },
    {
        "id": "book_flight",
        "say": "I've booked the flight for you under the name John Smith and sent the confirmation to your email.",
        "ops": [{"type": "commit", "fn": "book_flight", "args": {"passenger_name": "John Smith"}}],
    },
    {
        "id": "search_apartments",
        "say": "I found twelve apartments in Boston with two bedrooms under three thousand dollars per month.",
        "ops": [{"type": "launch", "fn": "search_apartments", "args": {"city": "Boston", "bedrooms": 2, "max_price": 3000}}],
    },
    {
        "id": "update_filter",
        "say": "I've updated the search filter to show only pet-friendly apartments now.",
        "ops": [{"type": "patch", "op_id": 5, "diff": {"pets_allowed": True}}],
    },
    {
        "id": "track_order",
        "say": "Your order is currently in transit and should arrive by Friday afternoon.",
        "ops": [{"type": "launch", "fn": "track_order", "args": {"order_id": "ORD-12345"}}],
    },
]


async def benchmark_case(case: dict, output_dir: Path, seed: int = 42) -> dict:
    """Benchmark one test case."""
    case_id = case["id"]
    say_text = case["say"]
    ops = case["ops"]

    print(f"\n[{case_id}]")
    print(f"  Say: {say_text}")

    # Baseline
    print("  Running baseline...", end="", flush=True)
    baseline_path, baseline_lat = await synthesize_baseline(
        say_text, tts, output_dir, turn=0
    )
    print(f" {baseline_lat:.3f}s")

    # ack-v0 (context-aware)
    print("  Running ack-v0 (context)...", end="", flush=True)
    ack_path, main_path, ack_lat, main_lat, total_lat = await synthesize_with_ack(
        say_text, tts, output_dir, turn=1, ops=ops, strategy="context", seed=seed
    )
    improvement_abs = baseline_lat - ack_lat
    improvement_pct = (improvement_abs / baseline_lat) * 100
    print(f" ack={ack_lat:.3f}s, main={main_lat:.3f}s, total={total_lat:.3f}s")
    print(f"  First-response improvement: {improvement_abs:.3f}s ({improvement_pct:.1f}%)")

    return {
        "case_id": case_id,
        "say_text": say_text,
        "baseline_latency": round(baseline_lat, 4),
        "ack_latency": round(ack_lat, 4),
        "main_latency": round(main_lat, 4),
        "total_latency": round(total_lat, 4),
        "first_response_improvement_s": round(improvement_abs, 4),
        "first_response_improvement_pct": round(improvement_pct, 2),
        "baseline_path": str(baseline_path),
        "ack_path": str(ack_path),
        "main_path": str(main_path),
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark ack-v0 latency improvement")
    parser.add_argument("--output-dir", type=str, default="analysis/ack_benchmark",
                       help="Output directory for results and audio files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for ack selection")
    parser.add_argument("--cases", type=str, default="all",
                       help="Test cases to run (comma-separated IDs or 'all')")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select cases
    if args.cases == "all":
        cases = TEST_CASES
    else:
        case_ids = set(args.cases.split(","))
        cases = [c for c in TEST_CASES if c["id"] in case_ids]

    if not cases:
        print("Error: No test cases selected")
        return 1

    print(f"Benchmarking {len(cases)} cases with real Qwen3-Omni TTS...")
    print(f"Output directory: {output_dir}")

    results = []
    for case in cases:
        case_dir = output_dir / case["id"]
        case_dir.mkdir(exist_ok=True)
        try:
            result = await benchmark_case(case, case_dir, seed=args.seed)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"case_id": case["id"], "error": str(e)})

    # Aggregate statistics
    successful = [r for r in results if "error" not in r]
    if successful:
        avg_baseline = sum(r["baseline_latency"] for r in successful) / len(successful)
        avg_ack = sum(r["ack_latency"] for r in successful) / len(successful)
        avg_improvement = sum(r["first_response_improvement_s"] for r in successful) / len(successful)
        avg_improvement_pct = sum(r["first_response_improvement_pct"] for r in successful) / len(successful)

        summary = {
            "total_cases": len(cases),
            "successful_cases": len(successful),
            "failed_cases": len(cases) - len(successful),
            "avg_baseline_latency_s": round(avg_baseline, 4),
            "avg_ack_latency_s": round(avg_ack, 4),
            "avg_first_response_improvement_s": round(avg_improvement, 4),
            "avg_first_response_improvement_pct": round(avg_improvement_pct, 2),
            "results": results,
        }

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Cases: {len(successful)}/{len(cases)} successful")
        print(f"Average baseline latency: {avg_baseline:.3f}s")
        print(f"Average ack latency: {avg_ack:.3f}s")
        print(f"Average first-response improvement: {avg_improvement:.3f}s ({avg_improvement_pct:.1f}%)")
        print(f"\nTarget: 0.66s baseline → ~0.35s ack (47% improvement)")
        if avg_ack <= 0.40:
            print("✓ Target met!")
        else:
            print(f"⚠ Target not met (ack latency {avg_ack:.3f}s > 0.40s)")

        # Save results
        result_file = output_dir / "ack_latency_improvement.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {result_file}")

        return 0
    else:
        print("\nAll cases failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
