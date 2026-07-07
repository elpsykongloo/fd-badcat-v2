#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_state_track.py
===================
Unit tests for state track scorer.

Coverage:
  - Normal self-correction flow
  - Dangling operations
  - Illegal state transitions
  - Reversibility constraint violations
  - Compensation semantics
  - Batch scoring
  - Integration with FDB reports
"""

import sys
import os
import unittest
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from state_track_scorer import (
    StateTrackScorer,
    OpStatus,
    Reversibility,
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    check_completeness,
    check_legality,
    check_efficiency,
    integrate_state_track_score,
)


class TestStateSpace(unittest.TestCase):
    """Test state space constants and definitions."""

    def test_legal_transitions(self):
        """Verify legal transition set matches formalism."""
        expected = {
            (OpStatus.PENDING, OpStatus.STAGED),
            (OpStatus.PENDING, OpStatus.IN_FLIGHT),
            (OpStatus.PENDING, OpStatus.COMMITTED),    # synchronous commit path
            (OpStatus.PENDING, OpStatus.CANCELLED),
            (OpStatus.STAGED, OpStatus.COMMITTED),
            (OpStatus.STAGED, OpStatus.CANCELLED),
            (OpStatus.IN_FLIGHT, OpStatus.COMMITTED),
            (OpStatus.COMMITTED, OpStatus.COMPENSATED),
        }
        self.assertEqual(LEGAL_TRANSITIONS, expected)

    def test_terminal_states(self):
        """Verify terminal states."""
        expected = {OpStatus.COMMITTED, OpStatus.CANCELLED, OpStatus.COMPENSATED}
        self.assertEqual(TERMINAL_STATES, expected)

    def test_reversibility_lattice(self):
        """Verify reversibility ordering: READ ⪯ REV ⪯ COMP ⪯ IRR."""
        self.assertLess(Reversibility.READ.value, Reversibility.REV.value)
        self.assertLess(Reversibility.REV.value, Reversibility.COMP.value)
        self.assertLess(Reversibility.COMP.value, Reversibility.IRR.value)


class TestCompleteness(unittest.TestCase):
    """Test completeness checking (P1: no dangling ops)."""

    def test_all_committed(self):
        """All ops reach terminal state → score 1.0."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {}, "status": "pending"},
            {"t": 2.0, "op": "commit", "op_id": 1, "fn": "f1", "args": {}, "status": "committed"},
        ]
        final_status = {1: "committed"}
        score, violations = check_completeness(events, final_status)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(violations), 0)

    def test_dangling_pending(self):
        """Op stuck in PENDING → score 0.0."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {}, "status": "pending"},
        ]
        final_status = {1: "pending"}
        score, violations = check_completeness(events, final_status)
        self.assertEqual(score, 0.0)
        self.assertIn("op_1 dangling in state pending", violations[0])

    def test_mixed_completion(self):
        """2 committed, 1 dangling → score 0.67."""
        events = []
        final_status = {1: "committed", 2: "committed", 3: "in_flight"}
        score, violations = check_completeness(events, final_status)
        self.assertAlmostEqual(score, 2.0 / 3.0, places=2)
        self.assertEqual(len(violations), 1)

    def test_cancelled_is_terminal(self):
        """CANCELLED counts as terminal."""
        events = []
        final_status = {1: "cancelled"}
        score, violations = check_completeness(events, final_status)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(violations), 0)


class TestLegality(unittest.TestCase):
    """Test legality checking (P2: transitions, P3: reversibility, P4: compensation)."""

    def test_legal_transition(self):
        """PENDING → COMMITTED is legal."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {}, "status": "pending"},
            {"t": 2.0, "op": "commit", "op_id": 1, "fn": "f1", "args": {}, "status": "in_flight"},
            {"t": 3.0, "op": "commit", "op_id": 1, "fn": "f1", "args": {}, "status": "committed"},
        ]
        score, violations = check_legality(events)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(violations), 0)

    def test_illegal_transition(self):
        """CANCELLED → COMMITTED is illegal."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {}, "status": "pending"},
            {"t": 2.0, "op": "cancel", "op_id": 1, "fn": "f1", "args": {}, "status": "cancelled"},
            {"t": 3.0, "op": "commit", "op_id": 1, "fn": "f1", "args": {}, "status": "committed"},
        ]
        score, violations = check_legality(events)
        self.assertLess(score, 1.0)
        self.assertGreater(len(violations), 0)
        self.assertIn("illegal transition", violations[0])

    def test_irr_cannot_be_staged(self):
        """P3: IRR operation cannot enter STAGED state."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "transfer", "args": {},
             "status": "pending", "reversibility": "IRR"},
            {"t": 2.0, "op": "speculate", "op_id": 1, "fn": "transfer", "args": {},
             "status": "staged", "reversibility": "IRR"},
        ]
        score, violations = check_legality(events)
        self.assertLess(score, 1.0)
        self.assertTrue(any("IRR operation cannot be STAGED" in v for v in violations))

    def test_read_can_be_staged(self):
        """READ operations can safely enter STAGED."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "search", "args": {},
             "status": "pending", "reversibility": "READ"},
            {"t": 2.0, "op": "speculate", "op_id": 1, "fn": "search", "args": {},
             "status": "staged", "reversibility": "READ"},
        ]
        score, violations = check_legality(events)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(violations), 0)

    def test_only_comp_can_be_compensated(self):
        """P4: Only COMP operations can be COMPENSATED."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "reserve", "args": {},
             "status": "pending", "reversibility": "IRR"},
            {"t": 2.0, "op": "commit", "op_id": 1, "fn": "reserve", "args": {},
             "status": "committed", "reversibility": "IRR"},
            {"t": 3.0, "op": "compensate", "op_id": 1, "fn": "reserve", "args": {},
             "status": "compensated", "reversibility": "IRR"},
        ]
        score, violations = check_legality(events)
        self.assertLess(score, 1.0)
        self.assertTrue(any("only COMP operations can be COMPENSATED" in v for v in violations))

    def test_comp_can_be_compensated(self):
        """COMP operations can legally be COMPENSATED."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "reserve", "args": {},
             "status": "pending", "reversibility": "COMP"},
            {"t": 2.0, "op": "commit", "op_id": 1, "fn": "reserve", "args": {},
             "status": "committed", "reversibility": "COMP"},
            {"t": 3.0, "op": "compensate", "op_id": 1, "fn": "reserve", "args": {},
             "status": "compensated", "reversibility": "COMP"},
        ]
        score, violations = check_legality(events)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(violations), 0)


class TestEfficiency(unittest.TestCase):
    """Test efficiency checking (redundancy detection)."""

    def test_false_start(self):
        """Launch → Cancel with no patches is a false start."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {}, "status": "pending"},
            {"t": 1.5, "op": "cancel", "op_id": 1, "fn": "f1", "args": {}, "status": "cancelled"},
        ]
        score, violations = check_efficiency(events)
        self.assertLess(score, 1.0)
        self.assertTrue(any("false start" in v for v in violations))

    def test_patch_then_cancel_not_false_start(self):
        """Launch → Patch → Cancel is legitimate (user changed mind)."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {}, "status": "pending"},
            {"t": 1.5, "op": "patch", "op_id": 1, "fn": "f1", "args": {}, "status": "pending"},
            {"t": 2.0, "op": "cancel", "op_id": 1, "fn": "f1", "args": {}, "status": "cancelled"},
        ]
        score, violations = check_efficiency(events)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(violations), 0)

    def test_normal_flow_efficient(self):
        """Launch → Commit is perfectly efficient."""
        events = [
            {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {}, "status": "pending"},
            {"t": 2.0, "op": "commit", "op_id": 1, "fn": "f1", "args": {}, "status": "committed"},
        ]
        score, violations = check_efficiency(events)
        self.assertEqual(score, 1.0)
        self.assertEqual(len(violations), 0)


class TestStateTrackScorer(unittest.TestCase):
    """Integration tests for StateTrackScorer."""

    def setUp(self):
        self.scorer = StateTrackScorer(alpha=0.5, beta=0.4, gamma=0.1)

    def test_perfect_track(self):
        """Normal self-correction: launch → patch → commit."""
        track = {
            "unit_id": 1,
            "events": [
                {"t": 2.10, "op": "launch", "op_id": 1, "fn": "search_flights",
                 "args": {"destination": "NYC"}, "status": "pending", "reversibility": "READ"},
                {"t": 2.85, "op": "patch", "op_id": 1, "fn": "search_flights",
                 "args": {"destination": "Boston"}, "status": "pending", "reversibility": "READ"},
                {"t": 3.05, "op": "commit", "op_id": 1, "fn": "search_flights",
                 "args": {"destination": "Boston"}, "status": "committed", "reversibility": "READ"},
            ],
            "final_status": {1: "committed"}
        }
        report = self.scorer.score(track)
        self.assertEqual(report.metrics["completeness"], 1.0)
        self.assertEqual(report.metrics["legality"], 1.0)
        self.assertEqual(report.metrics["efficiency"], 1.0)
        self.assertEqual(report.metrics["state_track_score"], 1.0)
        self.assertEqual(len(report.violations), 0)

    def test_dangling_op(self):
        """Incomplete transaction."""
        track = {
            "unit_id": 2,
            "events": [
                {"t": 2.10, "op": "launch", "op_id": 2, "fn": "book_flight",
                 "args": {}, "status": "pending", "reversibility": "IRR"},
            ],
            "final_status": {2: "pending"}
        }
        report = self.scorer.score(track)
        self.assertEqual(report.metrics["completeness"], 0.0)
        self.assertEqual(report.metrics["legality"], 1.0)
        self.assertLess(report.metrics["state_track_score"], 1.0)
        self.assertGreater(len(report.violations), 0)

    def test_illegal_speculation(self):
        """IRR operation illegally staged."""
        track = {
            "unit_id": 3,
            "events": [
                {"t": 2.10, "op": "launch", "op_id": 3, "fn": "transfer_money",
                 "args": {"amount": 500}, "status": "pending", "reversibility": "IRR"},
                {"t": 2.30, "op": "speculate", "op_id": 3, "fn": "transfer_money",
                 "args": {"amount": 500}, "status": "staged", "reversibility": "IRR"},
            ],
            "final_status": {3: "staged"}
        }
        report = self.scorer.score(track)
        self.assertEqual(report.metrics["completeness"], 0.0)  # not in terminal state
        self.assertLess(report.metrics["legality"], 1.0)
        self.assertGreater(len(report.violations), 1)  # dangling + illegal staging

    def test_batch_scoring(self):
        """Score multiple tracks and aggregate."""
        track1 = {
            "unit_id": 1,
            "events": [
                {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {},
                 "status": "pending", "reversibility": "READ"},
                {"t": 2.0, "op": "commit", "op_id": 1, "fn": "f1", "args": {},
                 "status": "committed", "reversibility": "READ"},
            ],
            "final_status": {1: "committed"}
        }
        track2 = {
            "unit_id": 2,
            "events": [
                {"t": 1.0, "op": "launch", "op_id": 2, "fn": "f2", "args": {},
                 "status": "pending", "reversibility": "IRR"},
            ],
            "final_status": {2: "pending"}
        }
        batch_report = self.scorer.score_batch([track1, track2])

        agg = batch_report["aggregate"]
        self.assertEqual(len(batch_report["per_unit"]), 2)
        self.assertAlmostEqual(agg["avg_completeness"], 0.5, places=2)  # (1.0 + 0.0) / 2
        self.assertGreater(agg["total_violations"], 0)


class TestFDBIntegration(unittest.TestCase):
    """Test integration with FDB evaluation reports."""

    def test_integrate_scores(self):
        """Combine FDB tool metrics with state track score."""
        fdb_report = {
            "by_metric": {
                "tool_selection_acc": 0.90,
                "argument_acc": 0.85,
                "response_qual": 0.80,
            },
            "total_scenarios": 10,
        }

        state_track_report = {
            "aggregate": {
                "avg_completeness": 0.95,
                "avg_legality": 1.0,
                "avg_efficiency": 0.98,
                "avg_state_track_score": 0.97,
                "total_violations": 1,
            }
        }

        extended = integrate_state_track_score(
            fdb_report, state_track_report,
            weight_tool=0.25, weight_arg=0.25, weight_response=0.25, weight_state=0.25
        )

        expected_score = 0.25 * 0.90 + 0.25 * 0.85 + 0.25 * 0.80 + 0.25 * 0.97
        self.assertAlmostEqual(extended["phase_b_score"], expected_score, places=4)
        self.assertIn("state_track", extended)
        self.assertIn("phase_b_weights", extended)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_empty_track(self):
        """No operations → perfect score."""
        track = {"unit_id": 0, "events": [], "final_status": {}}
        scorer = StateTrackScorer()
        report = scorer.score(track)
        self.assertEqual(report.metrics["state_track_score"], 1.0)

    def test_multiple_patches(self):
        """Multiple patches before commit is legitimate."""
        track = {
            "unit_id": 1,
            "events": [
                {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {"x": 1},
                 "status": "pending", "reversibility": "READ"},
                {"t": 2.0, "op": "patch", "op_id": 1, "fn": "f1", "args": {"x": 2},
                 "status": "pending"},
                {"t": 3.0, "op": "patch", "op_id": 1, "fn": "f1", "args": {"x": 3},
                 "status": "pending"},
                {"t": 4.0, "op": "commit", "op_id": 1, "fn": "f1", "args": {"x": 3},
                 "status": "committed"},
            ],
            "final_status": {1: "committed"}
        }
        scorer = StateTrackScorer()
        report = scorer.score(track)
        self.assertEqual(report.metrics["completeness"], 1.0)
        self.assertEqual(report.metrics["legality"], 1.0)
        # Multiple patches don't hurt efficiency (self-correction is encouraged)
        self.assertEqual(report.metrics["efficiency"], 1.0)

    def test_cancelled_after_patches(self):
        """Cancel after patches is not a false start."""
        track = {
            "unit_id": 1,
            "events": [
                {"t": 1.0, "op": "launch", "op_id": 1, "fn": "f1", "args": {},
                 "status": "pending"},
                {"t": 2.0, "op": "patch", "op_id": 1, "fn": "f1", "args": {},
                 "status": "pending"},
                {"t": 3.0, "op": "cancel", "op_id": 1, "fn": "f1", "args": {},
                 "status": "cancelled"},
            ],
            "final_status": {1: "cancelled"}
        }
        scorer = StateTrackScorer()
        report = scorer.score(track)
        self.assertEqual(report.metrics["completeness"], 1.0)  # cancelled is terminal
        self.assertEqual(report.metrics["efficiency"], 1.0)    # not a false start


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
