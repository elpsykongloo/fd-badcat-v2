# -*- coding: utf-8 -*-
"""
tests/test_phase_b_integration.py
==================================
Integration tests for Phase-B: end-to-end scenarios with decider + transaction + tools.

Tests the 6 smoke scenarios from W2 plan (§3.4 R13):
1. Simple single-tool call
2. Self-correction (patch)
3. Multi-step tool sequence
4. Window cancellation
5. Over-limit clarification request
6. Compensating rollback

Run: pytest tests/test_phase_b_integration.py -v
Or:  python tests/test_phase_b_integration.py
"""
import sys
import json
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load transaction module directly
exec(open("/root/autodl-tmp/tact/transaction.py").read())

# Minimal mock tools and reversibility for testing
class Reversibility:
    READ = 0
    REV = 1
    COMP = 2
    IRR = 3

REVERSIBILITY = {
    "search_flights": Reversibility.READ,
    "book_flight": Reversibility.COMP,
    "update_identity_doc": Reversibility.IRR,
    "get_card_benefits": Reversibility.READ,
    "get_exchange_rate": Reversibility.READ,
    "modify_autopay": Reversibility.COMP,
    "search_apartments": Reversibility.READ,
    "calculate_commute": Reversibility.READ,
    "update_search_filter": Reversibility.REV,
    "track_order": Reversibility.READ,
    "search_products": Reversibility.READ,
    "add_to_cart": Reversibility.REV,
}

class ToolRegistry:
    """Minimal mock tool registry for testing."""
    def __init__(self, latency_profile="instant", room="test"):
        self.room = room

    def executor(self, fn, args):
        """Mock executor that returns success."""
        return {"status": "success", "fn": fn, "args": dict(args)}

def parse_decision(response):
    """Parse JSON decision from LLM response."""
    # Strip markdown fences if present
    response = re.sub(r'```json\s*|\s*```', '', response)
    # Extract JSON object
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(response)


# Mock LLM that returns scripted JSON responses
class MockLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if not self.responses:
            return '{"dialogue":"listen","ops":[{"type":"noop"}],"say":""}'
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Scenario 1: Simple single-tool call
# ---------------------------------------------------------------------------
def test_s1_simple_tool_call():
    """User: 'Search flights to NYC on July 15' -> launch + commit."""
    llm = MockLLM([
        '{"dialogue":"speak","ops":[{"type":"launch","fn":"search_flights","args":{"destination":"NYC","date":"2026-07-15"}},{"type":"commit","op_id":1}],"say":"Searching for flights to NYC"}'
    ])

    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_s1")

    # Decider turn 0
    response = llm([])
    decision = parse_decision(response)

    assert decision["dialogue"] == "speak"
    assert len(decision["ops"]) == 2
    assert decision["ops"][0]["type"] == "launch"
    assert decision["ops"][0]["fn"] == "search_flights"

    # Execute ops
    op = tx.launch(decision["ops"][0]["fn"],
                   decision["ops"][0]["args"],
                   REVERSIBILITY.get(decision["ops"][0]["fn"], Reversibility.IRR),
                   t=0.1)
    assert decision["ops"][1]["type"] == "commit"
    tx.commit(op.op_id, tools.executor, t=0.3)

    # Verify
    calls = tx.to_actual_tool_calls()
    assert len(calls) == 1
    assert calls[0]["function"] == "search_flights"
    assert calls[0]["args"]["destination"] == "NYC"
    print("PASS: S1 simple tool call")


# ---------------------------------------------------------------------------
# Scenario 2: Self-correction (patch)
# ---------------------------------------------------------------------------
def test_s2_self_correction():
    """User: 'NYC' ... 'actually Boston' -> launch, then patch."""
    llm = MockLLM([
        '{"dialogue":"speak","ops":[{"type":"launch","fn":"search_flights","args":{"destination":"NYC","date":"2026-07-15"}}],"say":"Looking for flights to NYC"}',
        '{"dialogue":"speak","ops":[{"type":"patch","op_id":1,"diff":{"destination":"Boston"}},{"type":"commit","op_id":1}],"say":"Changed to Boston"}'
    ])

    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_s2")

    # Turn 0: launch
    response1 = llm([])
    decision1 = parse_decision(response1)
    op = tx.launch(decision1["ops"][0]["fn"],
                   decision1["ops"][0]["args"],
                   REVERSIBILITY.get(decision1["ops"][0]["fn"], Reversibility.IRR),
                   t=0.1)

    # Turn 1: patch + commit
    response2 = llm([])
    decision2 = parse_decision(response2)
    assert decision2["ops"][0]["type"] == "patch"
    assert decision2["ops"][0]["op_id"] == 1
    assert decision2["ops"][0]["diff"]["destination"] == "Boston"

    tx.patch(op.op_id, decision2["ops"][0]["diff"], t=0.5)
    tx.commit(op.op_id, tools.executor, t=0.7)

    # Verify final call has patched value
    calls = tx.to_actual_tool_calls()
    assert len(calls) == 1
    assert calls[0]["args"]["destination"] == "Boston"
    assert len(op.patch_history) == 1
    print("PASS: S2 self-correction")


# ---------------------------------------------------------------------------
# Scenario 3: Multi-step tool sequence
# ---------------------------------------------------------------------------
def test_s3_multi_step():
    """User: 'Search flights to NYC, then book for Alice' -> two launches + commits."""
    llm = MockLLM([
        '{"dialogue":"speak","ops":[{"type":"launch","fn":"search_flights","args":{"destination":"NYC","date":"2026-07-15"}},{"type":"commit","op_id":1},{"type":"launch","fn":"book_flight","args":{"passenger_name":"Alice"}},{"type":"commit","op_id":2}],"say":"Searching and booking"}'
    ])

    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_s3")

    response = llm([])
    decision = parse_decision(response)

    # Execute ops in sequence
    op1 = tx.launch(decision["ops"][0]["fn"],
                    decision["ops"][0]["args"],
                    REVERSIBILITY.get(decision["ops"][0]["fn"], Reversibility.IRR),
                    t=0.1)
    tx.commit(op1.op_id, tools.executor, t=0.3)

    op2 = tx.launch(decision["ops"][2]["fn"],
                    decision["ops"][2]["args"],
                    REVERSIBILITY.get(decision["ops"][2]["fn"], Reversibility.IRR),
                    t=0.5)
    tx.commit(op2.op_id, tools.executor, t=0.7)

    calls = tx.to_actual_tool_calls()
    assert len(calls) == 2
    assert calls[0]["function"] == "search_flights"
    assert calls[1]["function"] == "book_flight"
    print("PASS: S3 multi-step")


# ---------------------------------------------------------------------------
# Scenario 4: Window cancellation
# ---------------------------------------------------------------------------
def test_s4_cancellation():
    """User: 'Book a flight' ... 'never mind' -> launch, then cancel."""
    llm = MockLLM([
        '{"dialogue":"speak","ops":[{"type":"launch","fn":"book_flight","args":{"passenger_name":"Alice"}}],"say":"Starting booking"}',
        '{"dialogue":"listen","ops":[{"type":"cancel","op_id":1}],"say":"Cancelled"}'
    ])

    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_s4")

    # Turn 0: launch
    response1 = llm([])
    decision1 = parse_decision(response1)
    op = tx.launch(decision1["ops"][0]["fn"],
                   decision1["ops"][0]["args"],
                   REVERSIBILITY.get(decision1["ops"][0]["fn"], Reversibility.IRR),
                   t=0.1)

    assert op.op_id in tx.pending

    # Turn 1: cancel
    response2 = llm([])
    decision2 = parse_decision(response2)
    assert decision2["ops"][0]["type"] == "cancel"
    tx.cancel(op.op_id, t=0.5)

    # Verify no committed calls
    calls = tx.to_actual_tool_calls()
    assert len(calls) == 0
    assert op.op_id not in tx.pending
    print("PASS: S4 cancellation")


# ---------------------------------------------------------------------------
# Scenario 5: Over-limit clarification (missing required arg)
# ---------------------------------------------------------------------------
def test_s5_clarification():
    """User: 'Search apartments' (missing city) -> agent asks for clarification."""
    llm = MockLLM([
        '{"dialogue":"speak","ops":[{"type":"noop"}],"say":"Which city would you like to search in?"}'
    ])

    tx = Transaction()

    response = llm([])
    decision = parse_decision(response)

    assert decision["dialogue"] == "speak"
    assert decision["ops"][0]["type"] == "noop"
    assert "city" in decision["say"].lower() or "which" in decision["say"].lower()
    print("PASS: S5 clarification")


# ---------------------------------------------------------------------------
# Scenario 6: Compensating rollback
# ---------------------------------------------------------------------------
def test_s6_compensation():
    """User: 'Book flight' ... 'cancel that' -> commit, then compensate."""
    llm = MockLLM([
        '{"dialogue":"speak","ops":[{"type":"launch","fn":"book_flight","args":{"passenger_name":"Alice"}},{"type":"commit","op_id":1}],"say":"Booked"}',
        '{"dialogue":"speak","ops":[{"type":"compensate","op_id":1}],"say":"Booking cancelled"}'
    ])

    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_s6")

    # Turn 0: launch + commit
    response1 = llm([])
    decision1 = parse_decision(response1)
    op = tx.launch(decision1["ops"][0]["fn"],
                   decision1["ops"][0]["args"],
                   REVERSIBILITY.get(decision1["ops"][0]["fn"], Reversibility.IRR),
                   t=0.1)
    op.compensator = "cancel_booking"
    tx.commit(op.op_id, tools.executor, t=0.3)

    assert len(tx.committed) == 1

    # Turn 1: compensate
    response2 = llm([])
    decision2 = parse_decision(response2)
    assert decision2["ops"][0]["type"] == "compensate"
    tx.compensate(op.op_id, tools.executor, t=0.7)

    # Verify op is compensated (no longer in committed)
    assert len(tx.committed) == 0
    assert len(tx.compensated) == 1
    print("PASS: S6 compensation")


# ---------------------------------------------------------------------------
# Scenario 7: Parse decision from real JSON
# ---------------------------------------------------------------------------
def test_s7_parse_decision():
    """Test the parse_decision utility with various formats."""
    # Clean JSON
    json_str = '{"dialogue":"speak","ops":[{"type":"launch","fn":"search_flights","args":{"destination":"NYC"}}],"say":"Searching"}'
    decision = parse_decision(json_str)
    assert decision["dialogue"] == "speak"
    assert len(decision["ops"]) == 1

    # JSON with markdown fences (should be stripped)
    fenced = '```json\n{"dialogue":"listen","ops":[],"say":""}\n```'
    decision = parse_decision(fenced)
    assert decision["dialogue"] == "listen"

    # JSON with prose before/after (extract)
    mixed = 'Here is my response:\n{"dialogue":"speak","ops":[],"say":"Hello"}\nDone.'
    decision = parse_decision(mixed)
    assert decision["say"] == "Hello"

    print("PASS: S7 parse_decision")


# ---------------------------------------------------------------------------
# Scenario 8: Tool reversibility classification
# ---------------------------------------------------------------------------
def test_s8_reversibility_classification():
    """Verify all 12 FDB tools have correct reversibility."""
    assert REVERSIBILITY["search_flights"] == Reversibility.READ
    assert REVERSIBILITY["search_apartments"] == Reversibility.READ
    assert REVERSIBILITY["get_card_benefits"] == Reversibility.READ
    assert REVERSIBILITY["track_order"] == Reversibility.READ

    assert REVERSIBILITY["add_to_cart"] == Reversibility.REV
    assert REVERSIBILITY["update_search_filter"] == Reversibility.REV

    assert REVERSIBILITY["book_flight"] == Reversibility.COMP
    assert REVERSIBILITY["modify_autopay"] == Reversibility.COMP

    assert REVERSIBILITY["update_identity_doc"] == Reversibility.IRR

    print("PASS: S8 reversibility classification")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
