#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline counterfactual census for RB admission v1 versus v1.1.

The archived RB rows intentionally contain only admission actions that were
actually taken.  R-ADM3' additionally asks how many raw patches admission v1
would have rejected even though v1.1, after the engine's local-id resolution
and wire unwrapping, passes them through.  That raw geometry is available only
at the live gate call, before ``apply_decision_ops`` normalizes it.

This wrapper monkey-patches the pure v1.1 gate during a normal cache replay. It
returns the original v1.1 output byte-for-byte, while evaluating v1 and v1.1
counterfactuals on each raw patch.  It must be used only with a complete
decision cache and services stopped; the wrapped runner's ordinary ``0
misses`` line and archive hashes remain the replay validity checks.

Example:

  python scripts/rb_adm11_counterfactual.py \
    --audit-output /tmp/audit-a.json -- \
    --build exp/rb/build_v23 --split test --arm A --system tact \
    --delta 1.5 --admission schema11 --decider llm --input audio \
    --provider rbt23_tact_d150_adm11
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import admission  # noqa: E402
import rb_run  # noqa: E402


def _caller_context():
    """Return episode/decision context from rb_run.run_episode's gate call."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame else None
        while frame is not None:
            ep = frame.f_locals.get("ep")
            if isinstance(ep, dict) and "id" in ep:
                return {
                    "episode_id": ep["id"],
                    "layer": ep.get("layer"),
                    "decision_index": frame.f_locals.get("i"),
                }
            frame = frame.f_back
    finally:
        del frame
    return {"episode_id": None, "layer": None, "decision_index": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    if "--admission" not in runner_args or "schema11" not in runner_args:
        parser.error("wrapped runner must use --admission schema11")

    orig_v1 = admission.admit_decision_ops
    orig_v11 = admission.admit_decision_ops_v11
    counts = Counter()
    details = []

    def wrapped_v11(ops, resolve, pending_fn_of, required_map):
        out, audit = orig_v11(ops, resolve, pending_fn_of, required_map)
        ctx = _caller_context()
        for op_index, op in enumerate(ops):
            if not isinstance(op, dict) or op.get("type") != "patch" \
                    or not isinstance(op.get("diff"), dict):
                continue
            counts["raw_patch_events"] += 1
            v1_out, v1_audit = orig_v1([op], pending_fn_of, required_map)
            v11_out, v11_audit = orig_v11(
                [op], resolve, pending_fn_of, required_map)
            rid = resolve(op)
            resolved_pending = rid in pending_fn_of
            counts[
                "resolved_pending_patch_events"
                if resolved_pending else "unresolved_or_stale_patch_events"
            ] += 1
            v11_passthrough = (
                not v11_audit and len(v11_out) == 1 and v11_out[0] == op
            )
            if v1_audit:
                counts["v1_would_reject"] += 1
            if v11_audit:
                counts["v11_reject"] += 1
            if v1_audit and v11_passthrough:
                if resolved_pending:
                    category = "v1_false_positive_resolved"
                else:
                    category = "v1_reject_v11_abstained_unresolved"
                counts[category] += 1
                a1 = v1_audit[0]
                details.append({
                    **ctx,
                    "op_index": op_index,
                    "raw_op_id": op.get("op_id"),
                    "resolved_op_id": rid,
                    "resolved_target_fn": pending_fn_of.get(rid),
                    "v1_target_fn": a1.get("target_fn"),
                    "v1_rejected_keys": a1.get("rejected_keys", []),
                    "wire_nested_args": (
                        set(op["diff"]) == {"args"}
                        and isinstance(op["diff"].get("args"), dict)
                    ),
                    "category": category,
                })
            if v1_audit and v11_audit:
                counts["both_reject"] += 1
            if not v1_audit and v11_audit:
                counts["v11_only_reject"] += 1
        return out, audit

    admission.admit_decision_ops_v11 = wrapped_v11
    old_argv = sys.argv
    try:
        sys.argv = [str(ROOT / "scripts/rb_run.py"), *runner_args]
        rc = rb_run.main()
    finally:
        sys.argv = old_argv
        admission.admit_decision_ops_v11 = orig_v11

    payload = {
        "schema": "rb-admission-v1.1-counterfactual-v1",
        "runner_args": runner_args,
        "counts": dict(sorted(counts.items())),
        "details": details,
    }
    out_path = Path(args.audit_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("ADMISSION_COUNTERFACTUAL " + json.dumps(
        payload["counts"], sort_keys=True))
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
