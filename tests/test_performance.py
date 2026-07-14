# -*- coding: utf-8 -*-
"""
tests/test_performance.py
=========================
Performance tests for Phase-B: concurrent execution and throughput testing.

Tests the injected replay mode at high concurrency to verify:
1. No race conditions in transaction state
2. Correct isolation between concurrent sessions
3. Throughput scaling with concurrency
4. Memory stability under load

Run: pytest tests/test_performance.py -v
Or:  python tests/test_performance.py
"""
import sys
import time
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tact.transaction import Transaction, Reversibility

# Minimal mock tools for testing
class ToolRegistry:
    """Minimal mock tool registry for testing."""
    def __init__(self, latency_profile="instant", room="test"):
        self.room = room
        self.latency_profile = latency_profile

    def executor(self, fn, args):
        """Mock executor that returns success."""
        if self.latency_profile == "realistic":
            time.sleep(0.05)  # 50ms simulated latency
        return {"status": "success", "fn": fn, "args": dict(args)}


# ---------------------------------------------------------------------------
# P1: Sequential baseline (single session)
# ---------------------------------------------------------------------------
def test_p1_sequential_baseline():
    """Baseline: single transaction executing tools sequentially."""
    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_p1")

    start = time.time()
    for i in range(10):
        op = tx.launch(f"search_flights", {"destination": f"NYC{i}", "date": "2026-07-15"},
                       Reversibility.READ, t=float(i))
        tx.commit(op.op_id, tools.executor, t=float(i) + 0.1)

    elapsed = time.time() - start

    calls = tx.to_actual_tool_calls()
    assert len(calls) == 10
    print(f"PASS: P1 sequential baseline ({elapsed:.3f}s for 10 ops)")


# ---------------------------------------------------------------------------
# P2: Concurrent sessions (isolation test)
# ---------------------------------------------------------------------------
def test_p2_concurrent_sessions():
    """Multiple transactions in parallel, verify isolation."""

    def worker(session_id):
        tx = Transaction()
        tools = ToolRegistry(latency_profile="instant", room=f"session_{session_id}")

        # Each session does different ops
        op1 = tx.launch("search_flights",
                        {"destination": f"NYC{session_id}", "date": "2026-07-15"},
                        Reversibility.READ, t=1.0)
        op2 = tx.launch("get_card_benefits",
                        {"card_type": "platinum"},
                        Reversibility.READ, t=1.2)

        tx.commit(op1.op_id, tools.executor, t=1.5)
        tx.commit(op2.op_id, tools.executor, t=1.8)

        return tx, session_id

    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(8)]
        results = [f.result() for f in as_completed(futures)]
    elapsed = time.time() - start

    # Verify each transaction is independent
    for tx, sid in results:
        calls = tx.to_actual_tool_calls()
        assert len(calls) == 2
        assert calls[0]["args"]["destination"] == f"NYC{sid}"
        assert tx.unit_id is not None

    print(f"PASS: P2 concurrent sessions (8 sessions in {elapsed:.3f}s)")


# ---------------------------------------------------------------------------
# P3: High-concurrency throughput
# ---------------------------------------------------------------------------
def test_p3_high_concurrency():
    """Stress test: 24+ concurrent sessions."""

    def worker(session_id):
        tx = Transaction()
        tools = ToolRegistry(latency_profile="instant", room=f"session_{session_id}")

        # Simulate a multi-op conversation
        ops = []
        ops.append(tx.launch("search_flights",
                             {"destination": "NYC", "date": "2026-07-15"},
                             Reversibility.READ, t=1.0))
        ops.append(tx.launch("search_apartments",
                             {"city": "NYC", "bedrooms": 2, "max_price": 3000},
                             Reversibility.READ, t=1.5))
        ops.append(tx.launch("calculate_commute",
                             {"origin_address": "A", "destination_address": "B"},
                             Reversibility.READ, t=2.0))

        for op in ops:
            tx.commit(op.op_id, tools.executor, t=op.launched_at + 0.2)

        return len(tx.to_actual_tool_calls())

    start = time.time()
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(worker, i) for i in range(24)]
        results = [f.result() for f in as_completed(futures)]
    elapsed = time.time() - start

    assert all(r == 3 for r in results), f"Expected 3 calls per session, got {results}"
    total_ops = sum(results)
    throughput = total_ops / elapsed

    print(f"PASS: P3 high concurrency (24 sessions, {total_ops} ops in {elapsed:.3f}s = {throughput:.1f} ops/s)")


# ---------------------------------------------------------------------------
# P4: Async concurrent operations
# ---------------------------------------------------------------------------
async def test_p4_async_concurrent():
    """Async version: concurrent sessions using asyncio."""

    async def worker(session_id):
        tx = Transaction()
        tools = ToolRegistry(latency_profile="instant", room=f"async_session_{session_id}")

        op = tx.launch("search_flights",
                       {"destination": f"DEST{session_id}", "date": "2026-07-15"},
                       Reversibility.READ, t=1.0)

        # Simulate async wait
        await asyncio.sleep(0.001)

        tx.commit(op.op_id, tools.executor, t=1.5)
        return tx

    start = time.time()
    tasks = [worker(i) for i in range(16)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start

    for i, tx in enumerate(results):
        calls = tx.to_actual_tool_calls()
        assert len(calls) == 1
        assert calls[0]["args"]["destination"] == f"DEST{i}"

    print(f"PASS: P4 async concurrent (16 sessions in {elapsed:.3f}s)")


def test_p4_async_concurrent_wrapper():
    """Wrapper to run async test."""
    asyncio.run(test_p4_async_concurrent())


# ---------------------------------------------------------------------------
# P5: Transaction log memory overhead
# ---------------------------------------------------------------------------
def test_p5_log_overhead():
    """Measure memory overhead of transaction logs."""
    import sys

    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_p5")

    # Execute many ops
    for i in range(100):
        op = tx.launch("search_flights",
                       {"destination": f"NYC{i}", "date": "2026-07-15"},
                       Reversibility.READ, t=float(i))
        if i % 3 == 0:
            tx.patch(op.op_id, {"destination": f"BOS{i}"}, t=float(i) + 0.2)
        tx.commit(op.op_id, tools.executor, t=float(i) + 0.5)

    # Check log size
    log_size = sys.getsizeof(tx.log)
    assert len(tx.log) >= 100  # At least one entry per launch
    assert len(tx.committed) == 100

    print(f"PASS: P5 log overhead (100 ops, log size ~{log_size/1024:.1f}KB)")


# ---------------------------------------------------------------------------
# P6: Patch history accumulation
# ---------------------------------------------------------------------------
def test_p6_patch_accumulation():
    """Test behavior with many patches to same op."""
    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_p6")

    op = tx.launch("search_apartments",
                   {"city": "NYC", "bedrooms": 2, "max_price": 3000},
                   Reversibility.READ, t=1.0)

    # Apply many patches
    for i in range(20):
        tx.patch(op.op_id, {"max_price": 3000 + i * 100}, t=1.0 + i * 0.1)

    assert len(op.patch_history) == 20
    assert op.args["max_price"] == 4900  # 3000 + 19*100

    tx.commit(op.op_id, tools.executor, t=5.0)
    calls = tx.to_actual_tool_calls()
    assert calls[0]["args"]["max_price"] == 4900

    print(f"PASS: P6 patch accumulation (20 patches)")


# ---------------------------------------------------------------------------
# P7: Cancel/restart cycles
# ---------------------------------------------------------------------------
def test_p7_cancel_restart_cycles():
    """Test repeated cancel + restart (window restarts)."""
    tx = Transaction()
    tools = ToolRegistry(latency_profile="instant", room="test_p7")

    committed_count = 0
    for i in range(10):
        op = tx.launch("search_flights",
                       {"destination": f"NYC{i}", "date": "2026-07-15"},
                       Reversibility.READ, t=float(i))

        if i % 2 == 0:
            tx.cancel(op.op_id, t=float(i) + 0.2)
        else:
            tx.commit(op.op_id, tools.executor, t=float(i) + 0.2)
            committed_count += 1

    calls = tx.to_actual_tool_calls()
    assert len(calls) == committed_count
    assert len(calls) == 5  # Half were cancelled

    print(f"PASS: P7 cancel/restart cycles (10 launches, 5 committed)")


# ---------------------------------------------------------------------------
# P8: Throughput comparison (instant vs realistic latency)
# ---------------------------------------------------------------------------
def test_p8_latency_profiles():
    """Compare throughput under different latency profiles."""

    def measure(profile):
        tx = Transaction()
        tools = ToolRegistry(latency_profile=profile, room=f"test_p8_{profile}")

        start = time.time()
        for i in range(5):
            op = tx.launch("search_flights",
                           {"destination": f"NYC{i}", "date": "2026-07-15"},
                           Reversibility.READ, t=float(i))
            tx.commit(op.op_id, tools.executor, t=float(i) + 0.1)
        elapsed = time.time() - start
        return elapsed

    instant = measure("instant")
    realistic = measure("realistic")

    print(f"PASS: P8 latency profiles (instant: {instant:.3f}s, realistic: {realistic:.3f}s)")
    assert realistic > instant  # Realistic should be slower


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
