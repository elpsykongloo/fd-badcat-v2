#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
state_track_scorer.py
=====================
State track evaluator for Phase-B transactional tool calling.

Evaluates transaction semantic correctness beyond tool call accuracy:
  - Completeness: no dangling pending ops at unit end
  - Legality: no illegal state transitions, reversibility constraints respected
  - Efficiency: no redundant operations

References:
  - state-track terminal-state semantics (see scripts/w2r_state_track.py)
  - tact/transaction.py (algebra implementation)
  - 手工文档/神谕/00_系统蓝图.md §3.2 P1 (Intent Serializability)
"""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import math


# ==============================================================================
# State Space (mirrors tact/transaction.py OpStatus)
# ==============================================================================

class OpStatus(Enum):
    PENDING = "pending"
    STAGED = "staged"
    IN_FLIGHT = "in_flight"
    COMMITTED = "committed"
    CANCELLED = "cancelled"
    COMPENSATED = "compensated"


class Reversibility(Enum):
    READ = 0   # pure, no side effect
    REV = 1    # exact inverse exists
    COMP = 2   # compensating action exists
    IRR = 3    # irreversible


# Legal state transitions (blueprint §2.3)
# Note: PENDING -> COMMITTED is legal (synchronous execution path, no async IN_FLIGHT)
LEGAL_TRANSITIONS = {
    (OpStatus.PENDING, OpStatus.STAGED),
    (OpStatus.PENDING, OpStatus.IN_FLIGHT),
    (OpStatus.PENDING, OpStatus.COMMITTED),    # synchronous commit (no async)
    (OpStatus.PENDING, OpStatus.CANCELLED),
    (OpStatus.STAGED, OpStatus.COMMITTED),
    (OpStatus.STAGED, OpStatus.CANCELLED),
    (OpStatus.IN_FLIGHT, OpStatus.COMMITTED),
    (OpStatus.COMMITTED, OpStatus.COMPENSATED),
}

TERMINAL_STATES = {OpStatus.COMMITTED, OpStatus.CANCELLED, OpStatus.COMPENSATED}


# ==============================================================================
# State Track Data Model
# ==============================================================================

@dataclass
class StateEvent:
    """Single event in the state track."""
    t: float                    # audio-relative time
    op: str                     # operation: launch/patch/cancel/commit/compensate/speculate
    op_id: int
    fn: str                     # function name
    args: dict
    status: str                 # OpStatus value
    reversibility: Optional[str] = None  # Reversibility value
    patch_diff: Optional[dict] = None    # for patch ops


@dataclass
class StateTrackReport:
    """Complete state track for one conversational unit."""
    unit_id: int
    events: List[StateEvent]
    final_status: Dict[int, str]  # op_id -> final OpStatus
    violations: List[str]          # list of violation descriptions
    metrics: Dict[str, float]      # completeness, legality, efficiency, score


# ==============================================================================
# Violation Detectors
# ==============================================================================

def check_completeness(events: List[StateEvent], final_status: Dict[int, str]) -> Tuple[float, List[str]]:
    """
    P1: No dangling ops — all ops must reach terminal state.

    Returns:
        (completeness_score, violations)
        completeness_score = (terminal_count / total_ops)
    """
    if not final_status:
        return 1.0, []

    violations = []
    terminal_count = 0

    for op_id, status in final_status.items():
        if OpStatus(status) in TERMINAL_STATES:
            terminal_count += 1
        else:
            violations.append(f"op_{op_id} dangling in state {status}")

    score = terminal_count / len(final_status) if final_status else 1.0
    return score, violations


def check_legality(events: List) -> Tuple[float, List[str]]:
    """
    P2: State monotonicity — only legal transitions allowed.
    P3: Reversibility constraint — IRR ops cannot be STAGED.
    P4: Compensation constraint — only COMP ops can be COMPENSATED.

    Returns:
        (legality_score, violations)
    """
    violations = []
    op_history: Dict[int, List[str]] = {}  # op_id -> [status_sequence]
    op_reversibility: Dict[int, str] = {}  # op_id -> reversibility

    # Build state history for each op
    for event in events:
        # Support both StateEvent objects and dicts
        if isinstance(event, dict):
            op_id = event["op_id"]
            status = event["status"]
            reversibility = event.get("reversibility")
        else:
            op_id = event.op_id
            status = event.status
            reversibility = event.reversibility

        if op_id not in op_history:
            op_history[op_id] = []
            if reversibility:
                op_reversibility[op_id] = reversibility
        op_history[op_id].append(status)

    # Check transitions
    illegal_transition_count = 0
    total_transition_count = 0

    for op_id, history in op_history.items():
        for i in range(len(history) - 1):
            from_status = OpStatus(history[i])
            to_status = OpStatus(history[i + 1])

            # Skip non-transitions (same state repeated, e.g., patch events keep status=pending)
            if from_status == to_status:
                continue

            total_transition_count += 1

            # P2: Legal transition check
            if (from_status, to_status) not in LEGAL_TRANSITIONS:
                violations.append(f"op_{op_id}: illegal transition {from_status.value} → {to_status.value}")
                illegal_transition_count += 1

            # P3: IRR cannot be STAGED
            if to_status == OpStatus.STAGED:
                rev = op_reversibility.get(op_id)
                if rev and Reversibility[rev] == Reversibility.IRR:
                    violations.append(f"op_{op_id}: IRR operation cannot be STAGED (must commit with confirmation)")
                    illegal_transition_count += 1

            # P4: Only COMP can be COMPENSATED
            if to_status == OpStatus.COMPENSATED:
                rev = op_reversibility.get(op_id)
                if rev and Reversibility[rev] != Reversibility.COMP:
                    violations.append(f"op_{op_id}: only COMP operations can be COMPENSATED (got {rev})")
                    illegal_transition_count += 1

    if total_transition_count == 0:
        legality_score = 1.0
    else:
        legality_score = 1.0 - (illegal_transition_count / total_transition_count)

    return legality_score, violations


def check_efficiency(events: List) -> Tuple[float, List[str]]:
    """
    Redundancy detection:
      - Same idem_key committed multiple times
      - Launch immediately followed by cancel with no patches (false starts)

    Returns:
        (efficiency_score, violations)
    """
    violations = []
    redundancy_penalty = 0.0

    # Group by op_id
    op_events: Dict[int, List] = {}
    for event in events:
        # Support both StateEvent objects and dicts
        if isinstance(event, dict):
            op_id = event["op_id"]
        else:
            op_id = event.op_id

        if op_id not in op_events:
            op_events[op_id] = []
        op_events[op_id].append(event)

    # Check for false starts: launch → cancel with no patches
    for op_id, evts in op_events.items():
        if len(evts) == 2:
            # Get op field from first and second event
            first_op = evts[0]["op"] if isinstance(evts[0], dict) else evts[0].op
            second_op = evts[1]["op"] if isinstance(evts[1], dict) else evts[1].op

            if first_op == "launch" and second_op == "cancel":
                has_patch = any(
                    (e["op"] if isinstance(e, dict) else e.op) == "patch"
                    for e in evts
                )
                if not has_patch:
                    violations.append(f"op_{op_id}: false start (launch → cancel with no patches)")
                    redundancy_penalty += 0.1

    # Check for idempotency key reuse (multiple commits of same idem_key)
    # This would require idem_key in events, which current schema doesn't have
    # Left as future extension

    efficiency_score = max(0.0, 1.0 - redundancy_penalty)
    return efficiency_score, violations


# ==============================================================================
# Main Scorer
# ==============================================================================

class StateTrackScorer:
    """
    State track scorer for Phase-B evaluation.

    Scoring formula:
        StateTrackScore = α·Completeness + β·Legality + γ·Efficiency

    Default weights: α=0.5, β=0.4, γ=0.1
    (Completeness and Legality are load-bearing; Efficiency is a tie-breaker)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.4, gamma: float = 0.1):
        """
        Args:
            alpha: weight for completeness (no dangling ops)
            beta: weight for legality (valid transitions + reversibility)
            gamma: weight for efficiency (no redundancy)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"

    def score(self, state_track: Dict[str, Any]) -> StateTrackReport:
        """
        Score a state track from a single conversational unit.

        Args:
            state_track: dict with keys:
                - unit_id: int
                - events: List[dict] (each dict has t, op, op_id, fn, args, status, reversibility)
                - final_status: Dict[int, str] (op_id -> final status string)

        Returns:
            StateTrackReport with scores and violation details
        """
        unit_id = state_track.get("unit_id", 0)
        events_raw = state_track.get("events", [])
        final_status = state_track.get("final_status", {})

        # Parse events
        events = []
        for e in events_raw:
            events.append(StateEvent(
                t=e["t"],
                op=e["op"],
                op_id=e["op_id"],
                fn=e["fn"],
                args=e["args"],
                status=e["status"],
                reversibility=e.get("reversibility"),
                patch_diff=e.get("patch_diff"),
            ))

        # Run checks
        completeness, c_violations = check_completeness(events, final_status)
        legality, l_violations = check_legality(events)
        efficiency, e_violations = check_efficiency(events)

        # Aggregate
        all_violations = c_violations + l_violations + e_violations
        state_track_score = (
            self.alpha * completeness +
            self.beta * legality +
            self.gamma * efficiency
        )

        metrics = {
            "completeness": completeness,
            "legality": legality,
            "efficiency": efficiency,
            "state_track_score": state_track_score,
        }

        return StateTrackReport(
            unit_id=unit_id,
            events=events,
            final_status=final_status,
            violations=all_violations,
            metrics=metrics,
        )

    def score_batch(self, state_tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Score multiple state tracks and aggregate.

        Returns:
            {
                "per_unit": List[StateTrackReport],
                "aggregate": {
                    "avg_completeness": float,
                    "avg_legality": float,
                    "avg_efficiency": float,
                    "avg_state_track_score": float,
                    "total_violations": int,
                    "violation_rate": float,
                }
            }
        """
        reports = [self.score(st) for st in state_tracks]

        if not reports:
            return {
                "per_unit": [],
                "aggregate": {
                    "avg_completeness": 0.0,
                    "avg_legality": 0.0,
                    "avg_efficiency": 0.0,
                    "avg_state_track_score": 0.0,
                    "total_violations": 0,
                    "violation_rate": 0.0,
                }
            }

        n = len(reports)
        avg_completeness = sum(r.metrics["completeness"] for r in reports) / n
        avg_legality = sum(r.metrics["legality"] for r in reports) / n
        avg_efficiency = sum(r.metrics["efficiency"] for r in reports) / n
        avg_score = sum(r.metrics["state_track_score"] for r in reports) / n
        total_violations = sum(len(r.violations) for r in reports)
        violation_rate = total_violations / n

        return {
            "per_unit": reports,
            "aggregate": {
                "avg_completeness": avg_completeness,
                "avg_legality": avg_legality,
                "avg_efficiency": avg_efficiency,
                "avg_state_track_score": avg_score,
                "total_violations": total_violations,
                "violation_rate": violation_rate,
            }
        }


# ==============================================================================
# Integration with FDB Evaluator
# ==============================================================================

def integrate_state_track_score(
    fdb_report: Dict[str, Any],
    state_track_report: Dict[str, Any],
    weight_tool: float = 0.25,
    weight_arg: float = 0.25,
    weight_response: float = 0.25,
    weight_state: float = 0.25,
) -> Dict[str, Any]:
    """
    Combine FDB tool calling scores with state track scores.

    Args:
        fdb_report: output from evaluate_tool_calls.py
        state_track_report: output from StateTrackScorer.score_batch()
        weight_*: weights for each component

    Returns:
        Extended report with "phase_b_score" field
    """
    assert abs(weight_tool + weight_arg + weight_response + weight_state - 1.0) < 1e-6

    # Extract FDB metrics
    by_metric = fdb_report.get("by_metric", {})
    tool_sel = by_metric.get("tool_selection_acc", 0.0)
    arg_acc = by_metric.get("argument_acc", 0.0)
    resp_qual = by_metric.get("response_qual", 0.0)

    # Extract state track score
    state_score = state_track_report["aggregate"]["avg_state_track_score"]

    # Combine
    phase_b_score = (
        weight_tool * tool_sel +
        weight_arg * arg_acc +
        weight_response * resp_qual +
        weight_state * state_score
    )

    # Extend report
    extended = dict(fdb_report)
    extended["state_track"] = state_track_report["aggregate"]
    extended["phase_b_score"] = phase_b_score
    extended["phase_b_weights"] = {
        "tool_selection": weight_tool,
        "argument": weight_arg,
        "response_quality": weight_response,
        "state_track": weight_state,
    }

    return extended


# ==============================================================================
# CLI Test
# ==============================================================================

if __name__ == "__main__":
    # Test case 1: Normal self-correction
    test_case_1 = {
        "unit_id": 1,
        "events": [
            {"t": 2.10, "op": "launch", "op_id": 1, "fn": "search_flights",
             "args": {"destination": "New York"}, "status": "pending", "reversibility": "READ"},
            {"t": 2.85, "op": "patch", "op_id": 1, "fn": "search_flights",
             "args": {"destination": "Boston"}, "status": "pending", "reversibility": "READ",
             "patch_diff": {"destination": "Boston"}},
            {"t": 3.05, "op": "commit", "op_id": 1, "fn": "search_flights",
             "args": {"destination": "Boston"}, "status": "committed", "reversibility": "READ"},
        ],
        "final_status": {1: "committed"}
    }

    # Test case 2: Dangling op
    test_case_2 = {
        "unit_id": 2,
        "events": [
            {"t": 2.10, "op": "launch", "op_id": 2, "fn": "book_flight",
             "args": {"destination": "Boston"}, "status": "pending", "reversibility": "IRR"},
        ],
        "final_status": {2: "pending"}
    }

    # Test case 3: Illegal speculation
    test_case_3 = {
        "unit_id": 3,
        "events": [
            {"t": 2.10, "op": "launch", "op_id": 3, "fn": "transfer_money",
             "args": {"amount": 500}, "status": "pending", "reversibility": "IRR"},
            {"t": 2.30, "op": "speculate", "op_id": 3, "fn": "transfer_money",
             "args": {"amount": 500}, "status": "staged", "reversibility": "IRR"},
        ],
        "final_status": {3: "staged"}
    }

    # Test case 4: Legal compensation
    test_case_4 = {
        "unit_id": 4,
        "events": [
            {"t": 2.10, "op": "launch", "op_id": 4, "fn": "reserve_table",
             "args": {"time": "7pm"}, "status": "pending", "reversibility": "COMP"},
            {"t": 3.00, "op": "commit", "op_id": 4, "fn": "reserve_table",
             "args": {"time": "7pm"}, "status": "committed", "reversibility": "COMP"},
            {"t": 5.50, "op": "compensate", "op_id": 4, "fn": "reserve_table",
             "args": {"time": "7pm"}, "status": "compensated", "reversibility": "COMP"},
        ],
        "final_status": {4: "compensated"}
    }

    scorer = StateTrackScorer()

    print("=" * 70)
    print("State Track Scorer — Test Cases")
    print("=" * 70)

    for i, test_case in enumerate([test_case_1, test_case_2, test_case_3, test_case_4], 1):
        print(f"\n--- Case {i}: {test_case.get('unit_id')} ---")
        report = scorer.score(test_case)
        print(f"Completeness: {report.metrics['completeness']:.2f}")
        print(f"Legality:     {report.metrics['legality']:.2f}")
        print(f"Efficiency:   {report.metrics['efficiency']:.2f}")
        print(f"State Track Score: {report.metrics['state_track_score']:.2f}")
        if report.violations:
            print(f"Violations ({len(report.violations)}):")
            for v in report.violations:
                print(f"  - {v}")
        else:
            print("Violations: none")

    # Batch test
    print("\n" + "=" * 70)
    print("Batch Scoring")
    print("=" * 70)
    batch_report = scorer.score_batch([test_case_1, test_case_2, test_case_3, test_case_4])
    agg = batch_report["aggregate"]
    print(f"Avg Completeness:    {agg['avg_completeness']:.2f}")
    print(f"Avg Legality:        {agg['avg_legality']:.2f}")
    print(f"Avg Efficiency:      {agg['avg_efficiency']:.2f}")
    print(f"Avg State Track Score: {agg['avg_state_track_score']:.2f}")
    print(f"Total Violations:    {agg['total_violations']}")
    print(f"Violation Rate:      {agg['violation_rate']:.2f} per unit")

    print("\n✓ All test cases executed. See scripts/w2r_state_track.py for the calibrated scorer.")
