"""
src/engine_b.py
===============
Phase-B engine: ActorEngine + transactional decision space.

Extends src/engine.py (Phase-A) with:
- Transaction / PendingSet management
- Extended decision algebra (launch/patch/cancel/commit)
- Dissent window mechanism (delta parameter)

W1 iron laws still apply:
  1. Behavior preservation (behind flags)
  2. Single writer principle
  3. Audio clock for all intervals

Phase-B v0: blocking mode (launch+commit immediate).
Phase-B v1+: async speculation (W3+).
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from engine import ActorEngine, FrameEvent, ModelDone, ControlMsg, SAMPLE_RATE
from transaction import Transaction, Reversibility
from decider_b import decide_and_apply, REVERSIBILITY, COMPENSATORS


# ---------------------------------------------------------------------------
# Dissent window events
# ---------------------------------------------------------------------------
@dataclass
class DissentWindowOpen:
    """Dissent window opened for a committed op (user has delta seconds to object)."""
    op_id: int
    t_open: float       # audio clock when window opened
    delta: float        # window duration in seconds


@dataclass
class DissentWindowClose:
    """Dissent window closed (no objection, op is finalized)."""
    op_id: int
    t_close: float


# ---------------------------------------------------------------------------
# Phase-B Engine
# ---------------------------------------------------------------------------
class TactEngine(ActorEngine):
    """
    Phase-B transactional engine.

    Extends ActorEngine with:
    - Transaction management (PendingSet)
    - Extended decision space (launch/patch/cancel/commit)
    - Dissent window mechanism

    Config (engine_cfg):
        phase: "b" (enables transactional mode)
        blocking: true (v0: launch+commit immediate) | false (async speculation, W3+)
        delta: 2.0 (dissent window duration in seconds, audio clock)
        tool_executor: callable (fn_name, args) -> result (or None for mock)
    """

    def __init__(self, websocket=None, prompts: dict = None, delay: dict = None,
                 llm_cfg: dict = None, engine_cfg: dict = None,
                 llm_fn=None, asr_fn=None, tts_fn=None,
                 replay_mode: str = "realtime", decision_script=None,
                 trace_path=None, vad_model=None, vad_iterator=None,
                 tool_executor: Optional[Callable] = None):

        super().__init__(websocket, prompts, delay, llm_cfg, engine_cfg,
                        llm_fn, asr_fn, tts_fn, replay_mode, decision_script,
                        trace_path, vad_model, vad_iterator)

        # Phase-B config
        self.phase = self.engine_cfg.get("phase", "a")
        self.blocking_mode = self.engine_cfg.get("blocking", True)
        self.delta = float(self.engine_cfg.get("delta", 2.0))

        # Transaction state (single-writer: only engine loop touches these)
        self.tx = Transaction()
        self.tool_executor = tool_executor or self._mock_executor

        # Dissent window tracking (op_id -> window open time)
        self.dissent_windows: dict[int, float] = {}

        # Tool call telemetry for FDB-v3 scoring
        self.tool_call_log = []

    def _mock_executor(self, fn: str, args: dict) -> dict:
        """Mock tool executor for testing without FDB integration."""
        return {"status": "success", "function": fn, "args": args}

    def _reset_session(self):
        """Override to reset transaction state."""
        super()._reset_session()
        self.tx = Transaction()
        self.dissent_windows.clear()
        self.tool_call_log.clear()

    # ------------------------------------------------------------------
    # Phase-B decision dispatch (replaces Phase-A's judge/shift/response)
    # ------------------------------------------------------------------
    def dispatch_transactional_decision(self, kind: str, user_audio: np.ndarray, turn: int):
        """
        Dispatch a transactional decision (Phase-B).

        This replaces the Phase-A judge->shift->response chain with a single
        transactional decision that can emit launch/patch/cancel/commit ops.

        kind: "eou_decision" (end-of-utterance) | "dissent" (user objection in window)
        """
        gen, epoch = self.session_gen, self.seg_epoch

        if self.replay_mode != "realtime":
            res = self.decision_script(kind, {"turn": turn, "t_audio": self.t_audio,
                                              "epoch": epoch, "tx_snapshot": self.tx.snapshot_for_prompt()})
            infer = 0.0 if self.replay_mode == "oracle" else float(res.get("infer", 0.0))
            # In replay mode, decision_script returns full decision JSON
            done = ModelDone(kind=kind, gen=gen, epoch=epoch, turn=turn,
                           text=json.dumps(res.get("decision", {})), infer=infer,
                           prompt_snapshot=[])
            self._schedule(self.t_audio + infer, done)
            return

        async def _run():
            t0 = time.perf_counter()
            timed_out = False
            text = ""
            error = ""

            try:
                # Build decider messages (uses tx snapshot)
                from decider_b import build_decider_messages
                msgs = build_decider_messages(self.tx, self.STATE, audio=user_audio)

                text = await asyncio.wait_for(
                    asyncio.to_thread(self.llm_fn, msgs), self.DECISION_TIMEOUT)
            except asyncio.TimeoutError:
                timed_out = True
                text = '{"dialogue":"stay","ops":[],"say":""}'  # conservative fallback
            except Exception as exc:
                error = str(exc)
                text = '{"dialogue":"stay","ops":[],"say":""}'

            infer = round(time.perf_counter() - t0, 3)
            self.q.put_nowait(ModelDone(kind=kind, gen=gen, epoch=epoch, turn=turn,
                                      text=str(text), infer=infer,
                                      prompt_snapshot=[], timed_out=timed_out,
                                      error=error))

        self._inflight += 1
        asyncio.create_task(_run())

    # ------------------------------------------------------------------
    # Decision result handlers (Phase-B)
    # ------------------------------------------------------------------
    async def _on_transactional_decision(self, ev: ModelDone):
        """Handle transactional decision result."""
        await self.send_control("tact_decision_done", {
            "timestamp": self._wall_ts(),
            "infer_time": ev.infer,
            "turn": ev.turn,
            "state": self.STATE,
            "kind": ev.kind,
        })

        # Parse and apply decision
        try:
            decision_json = json.loads(ev.text)
        except json.JSONDecodeError:
            # Fallback: do nothing
            return

        result = decide_and_apply(
            self.tx, self.tool_executor, lambda _: ev.text,
            self.STATE, audio=None, t=self.t_audio,
            blocking=self.blocking_mode
        )

        # Log applied ops
        for op_applied in result["ops_applied"]:
            await self.send_control("tact_op_applied", {
                "timestamp": self._wall_ts(),
                "t_audio": round(self.t_audio, 3),
                "op": op_applied
            })

            # Open dissent window for committed ops
            if op_applied["type"] == "commit" and self.delta > 0:
                op_id = op_applied["op_id"]
                self.dissent_windows[op_id] = self.t_audio
                self._schedule(self.t_audio + self.delta,
                             DissentWindowClose(op_id=op_id, t_close=self.t_audio + self.delta))
                await self.send_control("dissent_window_open", {
                    "timestamp": self._wall_ts(),
                    "t_audio": round(self.t_audio, 3),
                    "op_id": op_id,
                    "delta": self.delta
                })

        # TTS for the "say" field
        say = result.get("say", "")
        if say:
            self.assistant_history.append(say)
            self.dispatch_tts(say, ev.turn)

    # ------------------------------------------------------------------
    # Dissent window management
    # ------------------------------------------------------------------
    async def _on_dissent_window_close(self, ev: DissentWindowClose):
        """Handle dissent window closing (no objection received)."""
        if ev.op_id in self.dissent_windows:
            del self.dissent_windows[ev.op_id]
            await self.send_control("dissent_window_closed", {
                "timestamp": self._wall_ts(),
                "t_audio": round(ev.t_close, 3),
                "op_id": ev.op_id
            })

    def check_dissent_windows(self, current_audio_time: float):
        """Check for expired dissent windows (called on each frame)."""
        expired = [op_id for op_id, t_open in self.dissent_windows.items()
                  if current_audio_time >= t_open + self.delta]
        return expired

    # ------------------------------------------------------------------
    # Override event processing to route Phase-B decisions
    # ------------------------------------------------------------------
    async def _process_event(self, ev):
        """Override to handle Phase-B events."""
        if isinstance(ev, DissentWindowClose):
            await self._on_dissent_window_close(ev)
            return

        if isinstance(ev, ModelDone):
            # Route Phase-B decision kinds
            if ev.kind in ("eou_decision", "dissent"):
                await self._on_transactional_decision(ev)
                return

        # Fall back to Phase-A handlers
        await super()._process_event(ev)

    # ------------------------------------------------------------------
    # Override LISTEN state to use transactional decisions
    # ------------------------------------------------------------------
    async def _listen_frame(self, ev: FrameEvent, event):
        """Override to dispatch transactional decisions instead of judge->shift->response."""
        if self.phase != "b":
            # Phase-A mode: use original logic
            await super()._listen_frame(ev, event)
            return

        # Phase-B: transactional decision on EoU
        frame, t = ev.pcm, ev.t_audio

        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "turn": self.TURN_IDX, "state": self.STATE, "timestamp": self._wall_ts()})
            self.seg_epoch += 1
            self.IN_SPEECH = True
            self.BUFFER = [frame]
            return

        if not self.IN_SPEECH:
            return

        self.BUFFER.append(frame)

        if event and "end" in event:
            self.SILENCE_COUNTER = 1
            self.t_end_anchor = t
            await self.send_control("vad_done", {
                "timestamp": self._wall_ts(), "turn": self.TURN_IDX, "state": self.STATE})
            return

        if self.SILENCE_COUNTER > 0:
            if event and "start" in event:
                self.SILENCE_COUNTER = 0
                return
            elif (t - self.t_end_anchor) >= self.END_HOLD:
                self.SILENCE_COUNTER = 0
                await self.send_control("vad_640_done", {
                    "timestamp": self._wall_ts(), "turn": self.TURN_IDX, "state": self.STATE})
                user_audio = np.concatenate(self.BUFFER)
                self._judged_seg_end = t

                # Dispatch transactional decision (Phase-B)
                self.dispatch_transactional_decision("eou_decision", user_audio, self.TURN_IDX)

                # Also dispatch ASR for history
                self.dispatch_asr(user_audio, self.TURN_IDX)
                return

        if self.CONTINUE_ARMED:
            if (t - self.t_continue_anchor) >= self.AFTER_CONTINUE_TIMEOUT:
                user_audio = np.concatenate(self.BUFFER)
                self.CONTINUE_ARMED = False
                self.t_continue_anchor = None
                self.IN_SPEECH = False
                self.BUFFER = []

                # Dispatch transactional decision (Phase-B)
                self.dispatch_transactional_decision("eou_decision", user_audio, self.TURN_IDX)
                self.dispatch_asr(user_audio, self.TURN_IDX)
                return
            if event and "start" in event:
                self.CONTINUE_ARMED = False
                self.t_continue_anchor = None

    # ------------------------------------------------------------------
    # Export for FDB-v3 evaluation
    # ------------------------------------------------------------------
    def export_fdb_result(self, example_id: str, provider: str = "tact_b") -> dict:
        """Export transaction state in FDB-v3 result format."""
        return {
            "example_id": example_id,
            "provider": provider,
            "actual_tool_calls": self.tx.to_actual_tool_calls(),
            "transcript": " ".join(self.assistant_history),
            "status": "completed",
            "transaction_log": self.tx.log,
            "dissent_windows_used": len(self.dissent_windows) > 0,
        }


# ---------------------------------------------------------------------------
# Quick test: python -m src.engine_b
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Phase-B engine loaded. Use integration test script to run.")
