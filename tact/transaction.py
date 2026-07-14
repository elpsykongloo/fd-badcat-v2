"""
tact/transaction.py
====================
The deterministic pending-set / conversational-transaction data structure.

This is the load-bearing structure behind TACT: the MLLM decision center only
*emits operations* against a Transaction; this structure does all the bookkeeping
deterministically (so it is reproducible, serializable, and directly maps to the
intent-serializability proof obligations in the blueprint).

Nothing in here calls a model or an API. It is pure state + algebra. Tool
execution is delegated to a callable `executor(fn_name, args) -> dict` that the
caller supplies (see tact/act_executor.py and tact/tools.py).

Algebra (matches the blueprint §2.4):
    launch(fn, args, reversibility)   create a pending op
    patch(op_id, diff)                structured diff over a pending op's args   <-- self-correction
    cancel(op_id)                     drop a pending op (false-start)
    commit(op_id, executor)           make the side effect real (gated for IRR)
    compensate(op_id)                 undo an already-committed COMP op
    speculate(op_id, executor)        execute into an isolated staging slot (READ/REV/COMP only)
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Reversibility lattice  (blueprint §2.3):  READ ⪯ REV ⪯ COMP ⪯ IRR
# ---------------------------------------------------------------------------
class Reversibility(Enum):
    READ = 0   # pure / read-only, no side effect            -> always safe to speculate
    REV = 1    # side effect with a cheap exact inverse        -> safe to speculate (w/ inverse)
    COMP = 2   # side effect undoable via a compensating call  -> safe to speculate (w/ comp. plan)
    IRR = 3    # no inverse                                     -> NEVER speculate; commit only after Conf


class OpStatus(Enum):
    PENDING = "pending"          # proposed, not executed
    STAGED = "staged"            # speculatively executed, effect isolated (READ/REV/COMP)
    IN_FLIGHT = "in_flight"      # async execution launched, awaiting result
    COMMITTED = "committed"      # executed, effect surfaced / durable
    CANCELLED = "cancelled"      # dropped before commit
    COMPENSATED = "compensated"  # committed then undone


_uid = itertools.count(1)


# ---------------------------------------------------------------------------
# A single tentative operation (a would-be tool call)
# ---------------------------------------------------------------------------
@dataclass
class PendingOp:
    fn: str
    args: dict
    reversibility: Reversibility = Reversibility.IRR  # safe default: treat unknown as irreversible
    op_id: int = field(default_factory=lambda: next(_uid))
    status: OpStatus = OpStatus.PENDING
    idem_key: str = ""                       # idempotency key (dedupe / safe retry)
    result: Optional[dict] = None            # staged or committed result
    launched_at: Optional[float] = None      # AUDIO-RELATIVE seconds (not wall clock)
    committed_at: Optional[float] = None
    compensator: Optional[str] = None        # name of compensating fn for COMP ops
    patch_history: list = field(default_factory=list)  # audit trail of self-corrections

    def apply_patch(self, diff: dict, t: Optional[float] = None) -> None:
        before = {k: self.args.get(k) for k in diff}
        self.args.update(diff)
        self.patch_history.append({"t": t, "diff": dict(diff), "before": before})

    def to_call(self) -> dict:
        """Export in the FDB-v3 `actual_tool_calls` entry format."""
        d = {"function": self.fn, "args": dict(self.args)}
        if self.launched_at is not None:
            d["timestamp_start"] = round(self.launched_at, 3)
        if self.committed_at is not None:
            d["timestamp_end"] = round(self.committed_at, 3)
        return d


# ---------------------------------------------------------------------------
# A scoped transaction, bound to one conversational unit
# ---------------------------------------------------------------------------
@dataclass
class Transaction:
    unit_id: int = field(default_factory=lambda: next(_uid))
    pending: "dict[int, PendingOp]" = field(default_factory=dict)
    committed: "list[PendingOp]" = field(default_factory=list)
    compensated: "list[PendingOp]" = field(default_factory=list)
    log: list = field(default_factory=list)  # full audit trail for the paper's case studies

    # ---- audit ----
    def _log(self, op_name: str, op: PendingOp, t: Optional[float], extra: Optional[dict] = None):
        rec = {
            "t": round(t, 3) if t is not None else None,
            "op": op_name,
            "op_id": op.op_id,
            "fn": op.fn,
            "args": dict(op.args),
            "status": op.status.value,
        }
        if extra:
            rec.update(extra)
        self.log.append(rec)

    # ---- algebra ----
    def launch(self, fn: str, args: dict, reversibility: Reversibility,
               idem_key: str = "", t: Optional[float] = None) -> PendingOp:
        op = PendingOp(fn=fn, args=dict(args), reversibility=reversibility,
                       idem_key=idem_key, launched_at=t)
        self.pending[op.op_id] = op
        self._log("launch", op, t)
        return op

    def patch(self, op_id: int, diff: dict, t: Optional[float] = None) -> PendingOp:
        """Self-correction: structured diff over a pending op's args. The signature
        TACT mechanism — `{destination: NYC -> Boston}` is just `patch(id, {'destination':'Boston'})`."""
        op = self.pending[op_id]
        op.apply_patch(diff, t)
        self._log("patch", op, t, extra={"diff": dict(diff)})
        return op

    def cancel(self, op_id: int, t: Optional[float] = None,
               executor: Optional[Callable[[str, dict], dict]] = None) -> PendingOp:
        """False-start: drop a pending op. If it was STAGED with a side effect, run its inverse."""
        op = self.pending.pop(op_id)
        if op.status == OpStatus.STAGED and op.reversibility in (Reversibility.REV, Reversibility.COMP) \
                and op.compensator and executor is not None:
            executor(op.compensator, op.args)  # roll back the staged effect
        op.status = OpStatus.CANCELLED
        self._log("cancel", op, t)
        return op

    def commit(self, op_id: int, executor: Callable[[str, dict], dict],
               t: Optional[float] = None) -> PendingOp:
        """Make the side effect real. If the op was already STAGED (speculated), the
        result is already in hand -> negative-latency commit; otherwise execute now."""
        op = self.pending.pop(op_id)
        if op.result is None:                       # not staged -> execute synchronously here
            op.result = executor(op.fn, op.args)
        op.status = OpStatus.COMMITTED
        op.committed_at = t
        self.committed.append(op)
        self._log("commit", op, t)
        return op

    def speculate(self, op_id: int, executor: Callable[[str, dict], dict],
                  t: Optional[float] = None) -> PendingOp:
        """Optimistic execution: run a pending op ahead of commit into an isolated slot.
        SAFE only for READ/REV/COMP (never IRR). Result is held, not surfaced, until commit."""
        op = self.pending[op_id]
        if op.reversibility == Reversibility.IRR:
            raise ValueError(f"refusing to speculate an IRREVERSIBLE op: {op.fn}")
        op.result = executor(op.fn, op.args)
        op.status = OpStatus.STAGED
        self._log("speculate", op, t)
        return op

    def compensate(self, op_id_or_op, executor: Callable[[str, dict], dict],
                   t: Optional[float] = None) -> PendingOp:
        """Post-commit correction: undo an already-committed COMP op via its compensator."""
        op = op_id_or_op if isinstance(op_id_or_op, PendingOp) else \
            next(o for o in self.committed if o.op_id == op_id_or_op)
        if op.reversibility == Reversibility.IRR or not op.compensator:
            raise ValueError(f"cannot compensate {op.fn}: no compensator / irreversible")
        executor(op.compensator, op.args)
        op.status = OpStatus.COMPENSATED
        self.committed.remove(op)
        self.compensated.append(op)
        self._log("compensate", op, t)
        return op

    # ---- helpers the decider needs ----
    def find_pending_by_fn(self, fn: str) -> Optional[PendingOp]:
        for op in self.pending.values():
            if op.fn == fn:
                return op
        return None

    def latest_pending(self) -> Optional[PendingOp]:
        return list(self.pending.values())[-1] if self.pending else None

    def snapshot_for_prompt(self) -> str:
        """Compact human-readable view of pending ops, to feed back into the MLLM
        so it knows what is already in flight and can patch/commit/cancel correctly."""
        if not self.pending:
            return "(none)"
        lines = []
        for op in self.pending.values():
            lines.append(f"  - id={op.op_id} fn={op.fn} args={json.dumps(op.args, ensure_ascii=False)} "
                         f"status={op.status.value}")
        return "\n".join(lines)

    # ---- export for FDB-v3 scoring ----
    def to_actual_tool_calls(self) -> list:
        """The list that goes into result_{provider}.json['actual_tool_calls'].
        Only COMMITTED ops count as real tool calls (compensated ones are netted out)."""
        return [op.to_call() for op in self.committed]


# ---------------------------------------------------------------------------
# Quick self-test:  python -m tact.transaction
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # toy executor: echoes the call
    def _exec(fn, args):
        return {"status": "success", "fn": fn, "args": args}

    tx = Transaction()
    # user: "book a flight to New York ... actually, Boston"
    op = tx.launch("search_flights", {"destination": "New York", "date": "July 15"},
                   Reversibility.READ, t=2.10)
    tx.patch(op.op_id, {"destination": "Boston"}, t=2.85)      # <-- self-correction
    tx.commit(op.op_id, _exec, t=3.05)

    print("committed tool calls:")
    print(json.dumps(tx.to_actual_tool_calls(), indent=2, ensure_ascii=False))
    print("\naudit log:")
    print(json.dumps(tx.log, indent=2, ensure_ascii=False))
    assert tx.to_actual_tool_calls()[0]["args"]["destination"] == "Boston"
    print("\nOK: final committed destination == Boston (rollback via patch worked).")
