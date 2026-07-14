# -*- coding: utf-8 -*-
"""
tests/test_transaction.py
==========================
Unit tests for PendingSet transaction operations (Phase-B, W2 §3.4 R13).

Tests the core algebraic operations: launch, patch, cancel, commit, compensate.
Covers window cancellation, window restart, patch merge, and over-limit scenarios.

Run: pytest tests/test_transaction.py -v
Or:  python tests/test_transaction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tact.transaction import Transaction, Reversibility, OpStatus, PendingOp


def mock_executor(fn, args):
    """Trivial executor: echoes the call as success."""
    return {"status": "success", "fn": fn, "args": dict(args)}


# ---------------------------------------------------------------------------
# T1: Basic launch + commit
# ---------------------------------------------------------------------------
def test_t1_launch_commit():
    tx = Transaction()
    op = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                   Reversibility.READ, t=1.0)
    assert op.op_id in tx.pending
    assert op.status == OpStatus.PENDING

    tx.commit(op.op_id, mock_executor, t=1.5)
    assert op.op_id not in tx.pending
    assert op in tx.committed
    assert op.status == OpStatus.COMMITTED
    assert op.result["fn"] == "search_flights"
    assert op.committed_at == 1.5


# ---------------------------------------------------------------------------
# T2: Patch (self-correction)
# ---------------------------------------------------------------------------
def test_t2_patch():
    tx = Transaction()
    op = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                   Reversibility.READ, t=1.0)

    # User corrects: "actually Boston"
    tx.patch(op.op_id, {"destination": "Boston"}, t=1.3)
    assert op.args["destination"] == "Boston"
    assert op.args["date"] == "2026-07-15"  # unchanged
    assert len(op.patch_history) == 1
    assert op.patch_history[0]["diff"]["destination"] == "Boston"
    assert op.patch_history[0]["before"]["destination"] == "NYC"

    tx.commit(op.op_id, mock_executor, t=1.8)
    # Final committed call has the patched value
    calls = tx.to_actual_tool_calls()
    assert len(calls) == 1
    assert calls[0]["args"]["destination"] == "Boston"


# ---------------------------------------------------------------------------
# T3: Multiple patches (patch merge)
# ---------------------------------------------------------------------------
def test_t3_patch_merge():
    tx = Transaction()
    op = tx.launch("search_apartments",
                   {"city": "NYC", "bedrooms": 2, "max_price": 3000},
                   Reversibility.READ, t=1.0)

    tx.patch(op.op_id, {"bedrooms": 3}, t=1.2)
    tx.patch(op.op_id, {"max_price": 3500}, t=1.5)
    tx.patch(op.op_id, {"city": "Boston"}, t=1.7)

    assert op.args == {"city": "Boston", "bedrooms": 3, "max_price": 3500}
    assert len(op.patch_history) == 3

    tx.commit(op.op_id, mock_executor, t=2.0)
    calls = tx.to_actual_tool_calls()
    assert calls[0]["args"]["city"] == "Boston"
    assert calls[0]["args"]["bedrooms"] == 3


# ---------------------------------------------------------------------------
# T4: Cancel (window cancellation)
# ---------------------------------------------------------------------------
def test_t4_cancel():
    tx = Transaction()
    op1 = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                    Reversibility.READ, t=1.0)
    op2 = tx.launch("book_flight", {"passenger_name": "Alice"},
                    Reversibility.COMP, t=1.2)

    # User: "never mind the booking"
    tx.cancel(op2.op_id, t=1.5)

    assert op2.status == OpStatus.CANCELLED
    assert op2.op_id not in tx.pending
    assert op1.op_id in tx.pending  # op1 unaffected

    tx.commit(op1.op_id, mock_executor, t=1.8)
    calls = tx.to_actual_tool_calls()
    assert len(calls) == 1  # only op1 committed
    assert calls[0]["function"] == "search_flights"


# ---------------------------------------------------------------------------
# T5: Window restart (cancel then launch again)
# ---------------------------------------------------------------------------
def test_t5_window_restart():
    tx = Transaction()
    op1 = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                    Reversibility.READ, t=1.0)
    tx.cancel(op1.op_id, t=1.3)

    # User restarts with different params
    op2 = tx.launch("search_flights", {"destination": "LAX", "date": "2026-08-01"},
                    Reversibility.READ, t=1.8)

    assert op1.status == OpStatus.CANCELLED
    assert op2.op_id in tx.pending
    assert op1.op_id != op2.op_id

    tx.commit(op2.op_id, mock_executor, t=2.2)
    calls = tx.to_actual_tool_calls()
    assert len(calls) == 1
    assert calls[0]["args"]["destination"] == "LAX"


# ---------------------------------------------------------------------------
# T6: Compensate (undo committed COMP operation)
# ---------------------------------------------------------------------------
def test_t6_compensate():
    tx = Transaction()
    op = tx.launch("book_flight", {"passenger_name": "Alice", "flight_id": "FL123"},
                   Reversibility.COMP, t=1.0)
    op.compensator = "cancel_booking"

    tx.commit(op.op_id, mock_executor, t=1.5)
    assert op in tx.committed

    # User: "cancel that booking"
    tx.compensate(op.op_id, mock_executor, t=2.0)
    assert op.status == OpStatus.COMPENSATED
    assert op not in tx.committed
    assert op in tx.compensated


# ---------------------------------------------------------------------------
# T7: Multi-op transaction
# ---------------------------------------------------------------------------
def test_t7_multi_op():
    tx = Transaction()
    op1 = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                    Reversibility.READ, t=1.0)
    op2 = tx.launch("get_exchange_rate", {"amount": 500, "from_currency": "USD", "to_currency": "EUR"},
                    Reversibility.READ, t=1.2)
    op3 = tx.launch("book_flight", {"passenger_name": "Alice"},
                    Reversibility.COMP, t=1.5)

    tx.commit(op1.op_id, mock_executor, t=1.8)
    tx.commit(op2.op_id, mock_executor, t=2.0)
    tx.commit(op3.op_id, mock_executor, t=2.3)

    calls = tx.to_actual_tool_calls()
    assert len(calls) == 3
    assert calls[0]["function"] == "search_flights"
    assert calls[1]["function"] == "get_exchange_rate"
    assert calls[2]["function"] == "book_flight"


# ---------------------------------------------------------------------------
# T8: Reversibility levels
# ---------------------------------------------------------------------------
def test_t8_reversibility():
    tx = Transaction()

    read_op = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                        Reversibility.READ, t=1.0)
    rev_op = tx.launch("add_to_cart", {"product_id": "PROD1", "quantity": 2},
                       Reversibility.REV, t=1.2)
    comp_op = tx.launch("book_flight", {"passenger_name": "Alice"},
                        Reversibility.COMP, t=1.5)
    irr_op = tx.launch("update_identity_doc", {"doc_type": "passport", "doc_number": "P123456"},
                       Reversibility.IRR, t=1.8)

    assert read_op.reversibility == Reversibility.READ
    assert rev_op.reversibility == Reversibility.REV
    assert comp_op.reversibility == Reversibility.COMP
    assert irr_op.reversibility == Reversibility.IRR

    # Verify lattice order: READ < REV < COMP < IRR
    assert Reversibility.READ.value < Reversibility.REV.value
    assert Reversibility.REV.value < Reversibility.COMP.value
    assert Reversibility.COMP.value < Reversibility.IRR.value


# ---------------------------------------------------------------------------
# T9: Pending snapshot (for prompt feedback)
# ---------------------------------------------------------------------------
def test_t9_pending_snapshot():
    tx = Transaction()

    # The standalone algebra returns ``(none)``.  Importing tact_core installs
    # the W2 snapshot overlay on this same class, so an aggregate pytest run
    # sees the richer active-engine form.  Both are intentional layers.
    snap = tx.snapshot_for_prompt()
    core_snapshot = snap != "(none)"
    if core_snapshot:
        assert snap == "PENDING (not yet executed, patch/cancel by id):\n  (none)"

    op1 = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                    Reversibility.READ, t=1.0)
    op2 = tx.launch("book_flight", {"passenger_name": "Alice"},
                    Reversibility.COMP, t=1.2)

    snap = tx.snapshot_for_prompt()
    if core_snapshot:
        assert "id=1" in snap and "id=2" in snap
    else:
        assert f"id={op1.op_id}" in snap
        assert f"id={op2.op_id}" in snap
    assert "search_flights" in snap
    assert "book_flight" in snap
    assert "NYC" in snap
    assert "Alice" in snap


# ---------------------------------------------------------------------------
# T10: Audit log completeness
# ---------------------------------------------------------------------------
def test_t10_audit_log():
    tx = Transaction()
    op = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                   Reversibility.READ, t=1.0)
    tx.patch(op.op_id, {"destination": "Boston"}, t=1.3)
    tx.commit(op.op_id, mock_executor, t=1.8)

    # Log should contain launch, patch, commit
    assert len(tx.log) == 3
    assert tx.log[0]["op"] == "launch"
    assert tx.log[0]["fn"] == "search_flights"
    assert tx.log[0]["t"] == 1.0

    assert tx.log[1]["op"] == "patch"
    assert tx.log[1]["op_id"] == op.op_id

    assert tx.log[2]["op"] == "commit"
    assert tx.log[2]["status"] == "committed"


# ---------------------------------------------------------------------------
# T11: Timestamp tracking
# ---------------------------------------------------------------------------
def test_t11_timestamps():
    tx = Transaction()
    op = tx.launch("search_flights", {"destination": "NYC", "date": "2026-07-15"},
                   Reversibility.READ, t=1.234)

    assert op.launched_at == 1.234
    assert op.committed_at is None

    tx.commit(op.op_id, mock_executor, t=1.987)
    assert op.committed_at == 1.987

    call = op.to_call()
    assert call["timestamp_start"] == 1.234
    assert call["timestamp_end"] == 1.987


# ---------------------------------------------------------------------------
# T12: Idempotency key support
# ---------------------------------------------------------------------------
def test_t12_idempotency():
    tx = Transaction()
    op1 = tx.launch("book_flight", {"passenger_name": "Alice"},
                    Reversibility.COMP, idem_key="booking-123", t=1.0)
    op2 = tx.launch("book_flight", {"passenger_name": "Bob"},
                    Reversibility.COMP, idem_key="booking-456", t=1.2)

    assert op1.idem_key == "booking-123"
    assert op2.idem_key == "booking-456"

    # Find ops by idempotency key
    found = [op for op in tx.pending.values() if op.idem_key == "booking-123"]
    assert len(found) == 1
    assert found[0].args["passenger_name"] == "Alice"


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
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
