#!/usr/bin/env python3
"""Run FDB binary pass-rate judging without silent exact-match fallback."""

import argparse
import json
import os
import pathlib
import sys
import time


FDB_DIR = pathlib.Path("/root/autodl-tmp/FDBench_v3/v3")
sys.path.insert(0, str(FDB_DIR))

import evaluate_pass_rate as evaluator  # noqa: E402
from llm_judge import add_llm_judge_args, configure_judge_from_args  # noqa: E402


class JudgeFallbackError(RuntimeError):
    """Raised when the upstream evaluator would silently use exact matching."""


def _disable_exact_fallback(*_args, **_kwargs):
    raise JudgeFallbackError("LLM judge failed; exact fallback is disabled")


def install_retrying_judge(retries: int, token_budget: int) -> None:
    upstream_judge = evaluator.llm_judge_argument
    upstream_chat = evaluator.judge_chat_completion

    def chat_with_reasoning_budget(messages, max_tokens, temperature=0):
        return upstream_chat(
            messages=messages,
            max_tokens=token_budget,
            temperature=temperature,
        )

    evaluator.judge_chat_completion = chat_with_reasoning_budget
    evaluator.exact_match_args = _disable_exact_fallback

    def strict_judge(expected_args, actual_args, function_name):
        last_error = None
        for attempt in range(retries + 1):
            try:
                return upstream_judge(expected_args, actual_args, function_name)
            except JudgeFallbackError as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(min(4.0, 0.5 * (2 ** attempt)))
        raise JudgeFallbackError(
            f"judge failed after {retries + 1} attempts for {function_name}"
        ) from last_error

    evaluator.llm_judge_argument = strict_judge


def load_entries(benchmark, results_dir: pathlib.Path, provider: str):
    scenario_map = {s["id"]: s for s in benchmark.get("scenarios", [])}
    entries = []
    for result_file in results_dir.rglob(f"result_{provider}.json"):
        result = json.loads(result_file.read_text())
        example_id = result.get("example_id")
        if not example_id or example_id not in scenario_map:
            continue
        entries.append({
            "scenario": scenario_map[example_id],
            "calls": result.get("actual_tool_calls", []),
            "transcript": result.get("transcript", ""),
            "result_data": result,
        })
    return entries


def main():
    # The upstream FDB helper already creates a thread-local client per worker.
    # Keep its CLI override, but make concurrent DeepSeek judging the default.
    os.environ.setdefault("FDB_LLM_WORKERS", os.getenv("DEEPSEEK_WORKERS", "100"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(FDB_DIR / "benchmark_data_v2.json"))
    parser.add_argument("--results-dir", default=str(FDB_DIR / "fdb_v3_data_released"))
    parser.add_argument("--provider", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    add_llm_judge_args(parser)
    args = parser.parse_args()

    if (str(args.llm_base_url or "").startswith("https://api.deepseek.com")
            or str(args.llm_model or "").startswith("deepseek")):
        for proxy_var in ("all_proxy", "ALL_PROXY", "http_proxy", "https_proxy",
                          "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)

    configure_judge_from_args(args)
    install_retrying_judge(
        max(0, args.retries),
        max(256, args.judge_max_tokens),
    )

    benchmark = json.loads(pathlib.Path(args.benchmark).read_text())
    entries = load_entries(benchmark, pathlib.Path(args.results_dir), args.provider)
    if len(entries) != 100:
        raise RuntimeError(f"expected 100 results for {args.provider}, found {len(entries)}")

    report = evaluator.evaluate_all_pass_rate(
        benchmark,
        entries,
        use_llm=True,
        workers=max(1, args.llm_workers),
        stream_results=False,
    )
    errors = [
        row for row in report["scenario_results"]
        if "evaluation_error" in row.get("checks", {})
    ]
    if errors:
        details = "; ".join(
            f"{row['scenario_id']}: "
            f"{row['checks']['evaluation_error'].get('message', 'unknown error')}"
            for row in errors
        )
        raise RuntimeError(f"judge errors in {len(errors)} scenarios: {details}")

    pathlib.Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(
        f"{args.provider}: {report['passed']}/{report['total_scenarios']} "
        f"({report['overall_pass_rate']:.3f})"
    )


if __name__ == "__main__":
    main()
