"""
tact/act_executor.py
====================
The asynchronous ACTION TRACK (blueprint §3.1).

Core TACT structural move: tool calls run on a parallel track, NOT as a third
serial state. The dialogue loop stays in listen/speak; this executor runs the
side-effecting calls in the background (via asyncio.to_thread so the event loop
is never blocked), tracks them, and reports when results are ready.

Two implementations with the SAME interface, so the engine code path is identical
and the BLOCKING-vs-ASYNC comparison is a one-line switch (the key MVP ablation):

  BlockingActTrack : launch() executes synchronously and returns the result.
                     -> reproduces the cascade's 'occupied silence'. The BASELINE.
  AsyncActTrack    : launch() spawns a background task and returns immediately.
                     ready_ops()/in_flight_ops() let the dialogue loop keep going
                     and let the floor-holding controller decide when to narrate.

Usage inside backend.py's async context:

    track = AsyncActTrack()                      # or BlockingActTrack()
    ...
    track.launch(pending_op, registry.executor)  # fire (non-blocking)
    ...
    for op in track.ready_ops():                  # poll each tick
        tx.commit(op.op_id, registry.executor, t=now)   # surface result (cheap; already computed)
    lam = track.max_remaining_estimate()          # feed floor-holding (Week 3)
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from .transaction import PendingOp, OpStatus
from .tools import REVERSIBILITY
from .latency_estimates import estimate_seconds   # see file below; safe heuristic


# ---------------------------------------------------------------------------
# Blocking baseline
# ---------------------------------------------------------------------------
class BlockingActTrack:
    """Executes tool calls synchronously. The dialogue loop is blocked while a tool
    runs — this is exactly the failure mode FDB-v3 punishes (10.12 s cascade)."""

    def __init__(self):
        self._done = []   # ops whose result is in hand and not yet committed

    def launch(self, op: PendingOp, executor: Callable[[str, dict], dict]) -> dict:
        op.status = OpStatus.IN_FLIGHT
        op.result = executor(op.fn, op.args)      # BLOCKS here
        op.status = OpStatus.STAGED               # result ready; awaiting commit
        self._done.append(op)
        return op.result

    def ready_ops(self):
        out, self._done = self._done, []
        return out

    def in_flight_ops(self):
        return []

    def max_remaining_estimate(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# Async action track (TACT)
# ---------------------------------------------------------------------------
class AsyncActTrack:
    """Runs each tool call as a background asyncio task. Non-blocking: the dialogue
    loop continues to listen/speak while calls are in flight."""

    def __init__(self):
        self._tasks: "dict[int, asyncio.Task]" = {}     # op_id -> task
        self._ops: "dict[int, PendingOp]" = {}          # op_id -> op
        self._started_at: "dict[int, float]" = {}       # op_id -> wall start (for remaining estimate)
        self._ready: "list[PendingOp]" = []

    def launch(self, op: PendingOp, executor: Callable[[str, dict], dict]) -> None:
        op.status = OpStatus.IN_FLIGHT
        self._ops[op.op_id] = op
        self._started_at[op.op_id] = time.time()

        async def _run():
            # run the (latency-injected, blocking) tool off the event loop
            result = await asyncio.to_thread(executor, op.fn, op.args)
            op.result = result
            op.status = OpStatus.STAGED            # ready; engine will COMMIT to surface it
            self._ready.append(op)

        self._tasks[op.op_id] = asyncio.create_task(_run())

    def ready_ops(self):
        """Ops whose background call has finished since the last poll. The engine
        should COMMIT these into the transaction (cheap; result already computed)."""
        done = []
        for op_id, task in list(self._tasks.items()):
            if task.done():
                self._tasks.pop(op_id, None)
                self._started_at.pop(op_id, None)
        out, self._ready = self._ready, []
        return out

    def in_flight_ops(self):
        return [self._ops[i] for i in self._tasks.keys()]

    def remaining_estimate(self, op: PendingOp) -> float:
        """Estimated seconds until this op finishes (for floor-holding). Uses the
        per-tool latency prior minus elapsed; clamped at 0."""
        prior = estimate_seconds(op.fn)
        elapsed = time.time() - self._started_at.get(op.op_id, time.time())
        return max(0.0, prior - elapsed)

    def max_remaining_estimate(self) -> float:
        ifo = self.in_flight_ops()
        return max((self.remaining_estimate(o) for o in ifo), default=0.0)

    async def drain(self):
        """Await all in-flight tasks (call at end of an example to flush)."""
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            self._tasks.clear()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_act_track(mode: str):
    mode = (mode or "blocking").lower()
    if mode in ("async", "tact"):
        return AsyncActTrack()
    return BlockingActTrack()
