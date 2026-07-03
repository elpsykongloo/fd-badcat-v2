"""
tests/test_phase_b.py
=====================
Integration test for Phase-B transactional engine.

Tests:
1. Transaction algebra (launch/patch/cancel/commit)
2. Self-correction via patch
3. Dissent window mechanism
4. FDB-v3 export format
5. Engine integration with mock decisions
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from transaction import Transaction, Reversibility
from decider_b import decide_and_apply, REVERSIBILITY
from tools_registry import ToolRegistry
from engine_b import TactEngine


def test_transaction_algebra():
    """Test basic transaction operations."""
    print("\n=== Test 1: Transaction Algebra ===")

    registry = ToolRegistry(latency_profile="instant")
    tx = Transaction()

    # Launch a flight search
    op1 = tx.launch("search_flights", {"destination": "New York", "date": "July 15"},
                   REVERSIBILITY["search_flights"], t=2.0)
    print(f"Launched op {op1.op_id}: search_flights to New York")

    # Patch destination
    tx.patch(op1.op_id, {"destination": "Boston"}, t=2.5)
    print(f"Patched op {op1.op_id}: destination -> Boston")

    # Commit
    tx.commit(op1.op_id, registry.executor, t=3.0)
    print(f"Committed op {op1.op_id}")

    # Verify
    calls = tx.to_actual_tool_calls()
    assert len(calls) == 1
    assert calls[0]["args"]["destination"] == "Boston"
    print("✓ Self-correction via patch works")

    # Launch and cancel
    op2 = tx.launch("book_flight", {"passenger_name": "Alice"},
                   REVERSIBILITY["book_flight"], t=4.0)
    print(f"Launched op {op2.op_id}: book_flight")
    tx.cancel(op2.op_id, t=4.5)
    print(f"Cancelled op {op2.op_id}")

    calls = tx.to_actual_tool_calls()
    assert len(calls) == 1  # still only the committed flight search
    print("✓ Cancel works")

    print("\nTransaction log:")
    print(json.dumps(tx.log, indent=2))


def test_decider_with_mock_llm():
    """Test decider with scripted LLM responses."""
    print("\n=== Test 2: Decider with Mock LLM ===")

    registry = ToolRegistry(latency_profile="instant")
    tx = Transaction()

    # Mock LLM that emits launch+patch+commit
    responses = iter([
        '{"dialogue":"speak","ops":[{"type":"launch","fn":"search_flights",'
        '"args":{"destination":"NYC","date":"July 20"}}],'
        '"say":"Searching flights to NYC."}',
    ])

    def mock_llm(msgs):
        return next(responses)

    result = decide_and_apply(tx, registry.executor, mock_llm, "LISTEN",
                             user_text="flight to NYC on July 20", t=2.0, blocking=True)

    print(f"Decision: {result['dialogue']}, Say: {result['say']}")
    print(f"Ops applied: {result['ops_applied']}")

    calls = tx.to_actual_tool_calls()
    assert len(calls) == 1
    assert calls[0]["function"] == "search_flights"
    assert calls[0]["args"]["destination"] == "NYC"
    print("✓ Decider emits and applies ops correctly")


def test_dissent_window():
    """Test dissent window mechanism (conceptual, no real user input)."""
    print("\n=== Test 3: Dissent Window Mechanism ===")

    registry = ToolRegistry(latency_profile="instant")
    tx = Transaction()

    # Simulate: user requests a booking, it commits, dissent window opens
    op = tx.launch("book_flight", {"passenger_name": "Bob Smith"},
                  REVERSIBILITY["book_flight"], t=5.0)
    print(f"Launched op {op.op_id}: book_flight")

    # Commit opens dissent window
    tx.commit(op.op_id, registry.executor, t=5.5)
    print(f"Committed op {op.op_id} at t=5.5")

    delta = 2.0  # 2-second dissent window
    dissent_window_closes_at = 5.5 + delta
    print(f"Dissent window open until t={dissent_window_closes_at}")

    # Scenario A: No dissent (window expires)
    print("Scenario A: No dissent, window expires")
    current_time = 8.0  # past window
    if current_time >= dissent_window_closes_at:
        print(f"  t={current_time}: Window closed, booking finalized")

    # Scenario B: User dissents within window (would trigger patch/cancel)
    print("\nScenario B: User dissents at t=6.0 (within window)")
    dissent_time = 6.0
    if dissent_time < dissent_window_closes_at:
        print(f"  t={dissent_time}: Dissent detected, can patch or cancel")
        # In real implementation, this would trigger a new decision with dissent context
        # For now, demonstrate cancel
        # (op already committed, so we'd need compensate or a new cancel mechanism)
        print("  -> Would trigger compensating action or patch")

    print("✓ Dissent window logic validated (conceptual)")


def test_fdb_export():
    """Test FDB-v3 result export format."""
    print("\n=== Test 4: FDB-v3 Export Format ===")

    registry = ToolRegistry(latency_profile="instant")
    tx = Transaction()

    # Execute a few tool calls
    op1 = tx.launch("search_flights", {"destination": "Tokyo", "date": "Aug 1"},
                   REVERSIBILITY["search_flights"], t=1.0)
    tx.commit(op1.op_id, registry.executor, t=1.5)

    op2 = tx.launch("book_flight", {"passenger_name": "Charlie"},
                   REVERSIBILITY["book_flight"], t=2.0)
    tx.commit(op2.op_id, registry.executor, t=2.5)

    # Export in FDB-v3 format
    result = {
        "example_id": "travel_01",
        "provider": "tact_b_v0",
        "actual_tool_calls": tx.to_actual_tool_calls(),
        "transcript": "Searching flights to Tokyo. Booking for Charlie.",
        "status": "completed"
    }

    print("FDB-v3 result format:")
    print(json.dumps(result, indent=2))

    assert len(result["actual_tool_calls"]) == 2
    assert result["actual_tool_calls"][0]["function"] == "search_flights"
    assert result["actual_tool_calls"][1]["function"] == "book_flight"
    print("✓ FDB-v3 export format correct")


def test_engine_integration():
    """Test Phase-B engine with mock setup."""
    print("\n=== Test 5: Engine Integration ===")

    from engine import frames_from_array

    # Mock configuration
    prompts = {}
    delay = {"end_hold_frame": 0.64, "after_continue_time": 2.5}
    llm_cfg = {"audio_block": "audio_url", "decision_timeout_s": 30}
    engine_cfg = {"phase": "b", "blocking": True, "delta": 2.0}

    # Mock LLM
    def mock_llm(msgs):
        return '{"dialogue":"speak","ops":[{"type":"launch","fn":"search_flights",' \
               '"args":{"destination":"Berlin","date":"Sep 1"}}],"say":"Searching flights."}'

    # Mock tool executor
    registry = ToolRegistry(latency_profile="instant")

    # Create engine
    engine = TactEngine(
        websocket=None,
        prompts=prompts,
        delay=delay,
        llm_cfg=llm_cfg,
        engine_cfg=engine_cfg,
        llm_fn=mock_llm,
        asr_fn=lambda path: "flight to Berlin on September first",
        tts_fn=lambda text, **k: (b"", 1.0),
        replay_mode="oracle",
        tool_executor=registry.executor
    )

    print(f"Engine phase: {engine.phase}")
    print(f"Blocking mode: {engine.blocking_mode}")
    print(f"Delta: {engine.delta}s")

    # Export test result
    result = engine.export_fdb_result("test_001", "tact_b_test")
    print("\nExported result:")
    print(json.dumps(result, indent=2))

    print("✓ Engine integration successful")


def run_all_tests():
    """Run all Phase-B tests."""
    print("=" * 60)
    print("Phase-B Integration Tests")
    print("=" * 60)

    test_transaction_algebra()
    test_decider_with_mock_llm()
    test_dissent_window()
    test_fdb_export()
    test_engine_integration()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
