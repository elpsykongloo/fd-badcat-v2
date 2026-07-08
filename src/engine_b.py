# -*- coding: utf-8 -*-
"""
src/engine_b.py — Phase-B v1: TactEngine, the LIVE side of the unified TACT engine.

W3 D1 rewrite ("engine_b 接线"). Phase-B v0 (commit-then-dissent-window) is retired;
the semantics that won the W2 delta scan — launch -> pending -> silence-clock
objection window -> timer-driven commit, under the COMMIT BARRIER — are now
implemented ONCE in src/tact_core.py and shared with the replay driver
(scripts/w2r_stream_replay.py). This module supplies the live/causal driver:

  perception   : inherited ActorEngine VADIterator state machine (LISTEN EoU =
                 vad end + 0.64s hold, unchanged legacy values — iron rule 1)
  clock        : audio clock; the WindowLedger burns SILENT audio time per frame
                 (user speech freezes every countdown), commit timers are re-armable
                 scheduled events on the same clock (fire via frame drain in
                 realtime, via the offline tail in run_offline)
  decisions    : one transactional decision per EoU over the CUMULATIVE session
                 audio prefix + PendingSet snapshot (same prompt builder, snapshot
                 v2 and prompt v2 as the replay => identical decision-cache keys)
  barrier      : dispatch registers the snapshot's pending set; expiries of guarded
                 ops are deferred; DecisionDone applies ops FIRST (patch rescues,
                 window restarts), THEN sweeps at the current clock (dual-stamp)
  release paths: normal DecisionDone | staleness (gen/epoch, policy-gated) |
                 decision timeout fail-open — each releases the barrier and sweeps

W1 iron laws still apply:
  1. Behavior preservation: Phase-A humdial behavior untouched (phase != "b" falls
     through to ActorEngine); all Phase-B behavior sits behind `phase: b`.
  2. Single writer: tx / ledger / records are mutated ONLY in the engine loop.
  3. Audio clock for every interval; wall clock only measures model latency.

Config (engine_cfg):
    phase: "b"                    enable the transactional engine (default "a")
    mode: "tact" | "blocking"     blocking = single decision, immediate commits
    blocking: bool                legacy v0 alias for mode=blocking (kept for tests)
    delta: 1.5                    objection window (seconds of SILENT audio time)
    commit_barrier: true          false = continuous-clock ablation (06 §一 probe)
    stale_eou_policy: "apply"     "drop" = epoch-gate EoU decisions (W2-parity: apply;
                                  an EoU decision's evidence is the pause itself, its
                                  ops stay meaningful under dedup + later patching)
    tool_sync: false              true = execute tools inline in the engine loop
                                  (offline/eval); false + realtime = worker thread
    tts_enabled / asr_enabled     live side-channels; disable for eval parity
    speculative_dispatch: false   W3 D6: dispatch the EoU decision at VAD END
                                  (not hold expiry). The result is GATED: it can
                                  only apply once the hold actually expires (the
                                  anchor is a confirmed EoU); speech resuming
                                  inside the hold invalidates it (release path 2
                                  variant, cause="spec_invalidated"). First-
                                  response floor: 0.64 + max(0, infer - 0.64).
    dag: false                    W3 D5: arm tact_dag.OpDag (patch propagation to
                                  dependent pending ops + compensation planning)
    tts_split: false              W3 D4: per-sentence TTS (first-audio anchor =
                                  sentence 1; barge-in drops unplayed sentences)
    floor_holding: false          W3 D4: floor rule v0 on barge-in during SPEAK
                                  (narration unconditionally yields; see
                                  src/floor_policy.py). Needs tts_split for
                                  sentence-granular actuation.
    sv_alpha: null                W4 placeholder (speaker-verification gate); MUST
                                  stay null in W3 — any non-null value would have to
                                  enter the prompt and would invalidate cache parity
"""

import asyncio
import heapq
import json
import time
from dataclasses import dataclass, field

import numpy as np

from engine import ActorEngine, FrameEvent, ModelDone, ControlMsg, SAMPLE_RATE
import tact_core
from tact_core import WindowLedger, apply_decision_ops, build_msgs, decide_from_msgs


# ---------------------------------------------------------------------------
# Phase-B events (queue/scheduled items; single-writer handles them in the loop)
# ---------------------------------------------------------------------------
@dataclass
class TactDecisionDone:
    key: int                  # barrier guard key (dispatch sequence number)
    gen: int
    epoch: int
    turn: int
    t_eou: float              # audio clock of the EoU that triggered the decision
    t_due: float              # audio clock when the result lands (eou + infer)
    decision: dict = None     # parsed decision (parse/repair done in the worker)
    infer: float = 0.0
    timed_out: bool = False
    error: str = ""


@dataclass
class WindowTimer:
    op_id: int
    gen: int
    due: float


@dataclass
class ToolDone:
    op_id: int
    gen: int
    result: dict = None
    wall: float = 0.0
    error: str = ""


@dataclass
class TtsSentDone:
    """One synthesized sentence of a split utterance (tts_split path)."""
    uid: int                  # utterance id
    idx: int                  # sentence index within the utterance
    gen: int = 0
    turn: int = 0
    text: str = ""
    infer: float = 0.0
    dur_audio: float = 0.0
    audio_bytes: bytes = b""
    cancelled: bool = False   # worker-side early exit (floor already yielded)
    error: str = ""


# Deprecated v0 events, kept importable for engine_b_ack.py / old tests.
@dataclass
class DissentWindowOpen:
    op_id: int
    t_open: float
    delta: float


@dataclass
class DissentWindowClose:
    op_id: int
    t_close: float


class TactEngine(ActorEngine):
    """Phase-B transactional engine (live driver of tact_core semantics)."""

    def __init__(self, websocket=None, prompts: dict = None, delay: dict = None,
                 llm_cfg: dict = None, engine_cfg: dict = None,
                 llm_fn=None, asr_fn=None, tts_fn=None,
                 replay_mode: str = "realtime", decision_script=None,
                 trace_path=None, vad_model=None, vad_iterator=None,
                 tool_executor=None):
        super().__init__(websocket, prompts, delay, llm_cfg, engine_cfg,
                         llm_fn, asr_fn, tts_fn, replay_mode, decision_script,
                         trace_path, vad_model, vad_iterator)

        cfg = self.engine_cfg
        self.phase = cfg.get("phase", "a")
        # v0 compat: `blocking: true` == `mode: blocking`
        self.mode = cfg.get("mode", "blocking" if cfg.get("blocking", False) else "tact")
        self.blocking_mode = (self.mode == "blocking")
        self.delta = float(cfg.get("delta", 1.5))
        self.commit_barrier = bool(cfg.get("commit_barrier", True))
        self.stale_eou_policy = cfg.get("stale_eou_policy", "apply")
        self.tool_sync = bool(cfg.get("tool_sync", False))
        self.tts_enabled = bool(cfg.get("tts_enabled", True))
        self.asr_enabled = bool(cfg.get("asr_enabled", True))
        self.speculative = bool(cfg.get("speculative_dispatch", False))
        self.dag_on = bool(cfg.get("dag", False))
        self.tts_split = bool(cfg.get("tts_split", False))
        self.floor_holding = bool(cfg.get("floor_holding", False))
        self.sv_alpha = cfg.get("sv_alpha", None)   # W4 placeholder — keep None in W3
        if cfg.get("prompt", "v2") == "v3":
            tact_core.install_prompt_v3()   # live-side arming (driver uses --prompt v3)
        elif cfg.get("prompt", "v2") == "v3.1":
            tact_core.install_prompt_v31()  # W3 close-out repair batch

        # transactional state (single-writer: engine loop only)
        self.tx = tact_core.Transaction()
        self.ledger = WindowLedger(self.delta, barrier=self.commit_barrier)
        self.dag = None
        self.comp_registry = None
        if self.dag_on:
            from tact_dag import OpDag, CompensationRegistry
            self.dag = OpDag(self.ledger)
            self.comp_registry = CompensationRegistry()
        self.tool_executor = tool_executor or self._mock_executor
        self.commit_records = []        # {op_id, t_commit(=nominal), actual_commit, tool_wall_s}
        self.say_events = []            # (t_audio, text)
        self.tact_decisions = []        # v1-trace-shaped decision entries
        self.tact_eous = []             # (seg_idx, t_eou) — seg_idx = running EoU index
        self.tool_call_log = []         # v0-compat attribute (unused by v1 paths)

        # cumulative session audio (decision input = prefix up to the EoU anchor)
        self._session_frames = []
        self._session_samples = 0

        # ledger clock + timers
        self._ledger_t = 0.0
        self._armed = set()             # op_ids with a live scheduled WindowTimer
        self._tact_seq = 0
        self._tact_inflight = {}        # key -> {"t_eou":…, "epoch":…}
        self._eou_count = 0

        # speculative dispatch state (at most one open spec per hold window)
        self._spec_inflight = {}        # key -> {"state": pending|confirmed|invalid, "stash": ev}
        self._spec_open = None          # key of the spec tied to the armed hold

        # sentence-split TTS state
        self._utt_seq = 0
        self._utt_state = {}            # uid -> {kind, n, allow_upto, last_sent}
        self._say_kind = "narration"

    # ------------------------------------------------------------------
    def _mock_executor(self, fn: str, args: dict) -> dict:
        return {"status": "success", "function": fn, "args": args}

    def _immediate(self):
        return self.blocking_mode or self.delta <= 0

    def _reset_session(self):
        super()._reset_session()
        self.tx = tact_core.Transaction()
        self.ledger = WindowLedger(self.delta, barrier=self.commit_barrier)
        if self.dag_on:
            from tact_dag import OpDag, CompensationRegistry
            self.dag = OpDag(self.ledger)
            self.comp_registry = CompensationRegistry()
        self.commit_records = []
        self.say_events = []
        self.tact_decisions = []
        self.tact_eous = []
        self.tool_call_log = []
        self._session_frames = []
        self._session_samples = 0
        self._ledger_t = 0.0
        self._armed = set()
        self._tact_inflight = {}
        self._eou_count = 0
        self._spec_inflight = {}
        self._spec_open = None
        self._utt_seq = 0
        self._utt_state = {}
        self._say_kind = "narration"

    # ------------------------------------------------------------------
    # silence clock: burn the ledger over (self._ledger_t, t] when user-silent
    # ------------------------------------------------------------------
    def _burn_to(self, t):
        if t <= self._ledger_t:
            return
        if not self.IN_SPEECH:
            self.ledger.advance_silence(self._ledger_t, t, self._commit_op)
        self._ledger_t = t

    def _arm_timers(self):
        """(Re-)schedule a commit timer for every open window while user-silent.
        Timers are re-armable hints: on fire they burn to their due time and, if
        the op survived (speech froze the budget), re-project. Audio-clock driven:
        they fire via frame drain (realtime) or the offline tail (run_offline)."""
        if self.IN_SPEECH:
            return
        # `self.t_audio` can lag while _drain_due() is handling scheduled events
        # before the current frame is applied. The ledger clock is the authoritative
        # base for window deadlines after any burn/decision/timer event.
        base = self._ledger_t
        for op_id, rem in self.ledger.win.items():
            if op_id not in self._armed:
                self._armed.add(op_id)
                self._schedule(base + rem,
                               WindowTimer(op_id=op_id, gen=self.session_gen,
                                           due=base + rem))

    # ------------------------------------------------------------------
    # commit path (single funnel; dual stamp)
    # ------------------------------------------------------------------
    def _commit_op(self, op_id, t_nominal, t_actual):
        if op_id not in self.tx.pending:
            return
        self.ledger.close(op_id)
        self._armed.discard(op_id)
        rec = {"op_id": op_id, "t_commit": round(t_nominal, 3),
               "actual_commit": round(t_actual, 3),
               "deferred_s": round(max(0.0, t_actual - t_nominal), 3),
               "tool_wall_s": None}
        if self.replay_mode == "realtime" and not self.tool_sync:
            # async tool track: mark committed now, execute in a worker
            op = self.tx.commit(op_id, lambda fn, args: {"status": "in_flight"},
                                t=t_nominal)
            gen = self.session_gen
            self._inflight += 1

            async def _run():
                t0 = time.perf_counter()
                res, err = None, ""
                try:
                    res = await asyncio.to_thread(self.tool_executor, op.fn, op.args)
                except Exception as exc:
                    err = str(exc)
                self.q.put_nowait(ToolDone(op_id=op_id, gen=gen, result=res,
                                           wall=round(time.perf_counter() - t0, 3),
                                           error=err))
            asyncio.create_task(_run())
        else:
            t0 = time.time()
            self.tx.commit(op_id, self.tool_executor, t=t_nominal)
            rec["tool_wall_s"] = round(time.time() - t0, 3)
        self.commit_records.append(rec)
        self.trace.append({"event": "act_commit", "data": dict(rec, t_audio=round(self.t_audio, 3))})
        if rec["deferred_s"] > 0:
            self.trace.append({"event": "act_commit_deferred",
                               "data": {"op_id": op_id, "nominal": rec["t_commit"],
                                        "actual": rec["actual_commit"],
                                        "deferred_s": rec["deferred_s"],
                                        "t_audio": round(self.t_audio, 3)}})

    # ------------------------------------------------------------------
    # decision dispatch (one transactional decision per EoU)
    # ------------------------------------------------------------------
    def _cumulative_prefix(self, t_end):
        """Session audio from t=0 up to the EoU segment-end anchor (cache parity
        with the replay driver, which slices the recorded file at seg end)."""
        if not self._session_frames:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._session_frames)
        n = int(t_end * SAMPLE_RATE)
        return audio[:n] if 0 < n <= len(audio) else audio

    def dispatch_tact_decision(self, t_eou, turn, spec=False):
        self._tact_seq += 1
        key = self._tact_seq
        gen, epoch = self.session_gen, self.seg_epoch
        if spec:
            # EoU not confirmed yet: bookkeeping (eou list, seg idx) happens at
            # hold expiry; the guard is registered NOW (dispatch-time snapshot).
            self._spec_inflight[key] = {"state": "pending", "stash": None,
                                        "t_eou": t_eou}
            self._spec_open = key
        else:
            self._eou_count += 1
            self.tact_eous.append([self._eou_count - 1, round(t_eou, 3)])
        self.ledger.begin_decision(key, set(self.tx.pending))
        self._tact_inflight[key] = {"t_eou": t_eou, "epoch": epoch}
        anchor = self.t_end_anchor if self.t_end_anchor is not None else self.t_audio
        prefix = self._cumulative_prefix(anchor)
        msgs = build_msgs(self.tx, prefix)

        if self.replay_mode != "realtime":
            res = self.decision_script("tact", {
                "turn": turn, "t_audio": self.t_audio, "t_eou": t_eou,
                "epoch": epoch, "messages": msgs,
                "tx_snapshot": self.tx.snapshot_for_prompt()})
            infer = 0.0 if self.replay_mode == "oracle" else float(res.get("infer", 0.0))
            dec = res.get("decision")
            if dec is None:
                try:
                    dec = json.loads(res.get("text", "") or "{}")
                except Exception:
                    dec = {}
            dec = {"dialogue": "stay", "ops": [], "say": "", **(dec or {})}
            self._schedule(self.t_audio + infer,
                           TactDecisionDone(key=key, gen=gen, epoch=epoch, turn=turn,
                                            t_eou=t_eou, t_due=self.t_audio + infer,
                                            decision=dec, infer=infer))
            return

        async def _run():
            t0 = time.perf_counter()
            timed_out, error = False, ""
            dec, infer = {"dialogue": "stay", "ops": [], "say": ""}, 0.0

            def _call(m):
                w0 = time.perf_counter()
                raw = self.llm_fn(m)
                return str(raw), round(time.perf_counter() - w0, 3)

            try:
                dec, infer = await asyncio.wait_for(
                    asyncio.to_thread(decide_from_msgs, _call, msgs),
                    self.DECISION_TIMEOUT)
            except asyncio.TimeoutError:
                timed_out = True
            except Exception as exc:
                error = str(exc)
            wall = round(time.perf_counter() - t0, 3)
            self.q.put_nowait(TactDecisionDone(
                key=key, gen=gen, epoch=epoch, turn=turn, t_eou=t_eou,
                t_due=self.t_audio + wall, decision=dec,
                infer=infer if not timed_out else wall,
                timed_out=timed_out, error=error))
        self._inflight += 1
        asyncio.create_task(_run())

    # ------------------------------------------------------------------
    # decision result: three release paths, barrier ordering
    # ------------------------------------------------------------------
    async def _on_tact_decision(self, ev: TactDecisionDone):
        # speculative gating: a spec result is inert until its anchor is a
        # CONFIRMED EoU (hold expired). Pending -> stash; invalid -> release.
        sp = self._spec_inflight.get(ev.key)
        if sp is not None:
            if sp["state"] == "pending":
                sp["stash"] = ev
                return
            self._spec_inflight.pop(ev.key, None)
            if sp["state"] == "invalid":
                self._tact_inflight.pop(ev.key, None)
                t_rel = max(ev.t_due, self._ledger_t)
                self._burn_to(t_rel)
                self.ledger.end_decision(ev.key)
                self.ledger.sweep(t_rel, self._commit_op, cause="spec_invalidated")
                self._arm_timers()
                await self.send_control("tact_decision_spec_invalid", {
                    "timestamp": self._wall_ts(), "turn": ev.turn,
                    "epoch": ev.epoch, "current_epoch": self.seg_epoch})
                return
            # confirmed: fall through to the normal release paths below

        info = self._tact_inflight.pop(ev.key, {})
        t_dec = max(ev.t_due, self._ledger_t)
        self._burn_to(t_dec)   # guarded expiries in (cursor, t_dec] defer (barrier on)

        # release path 2: staleness (session reset always; epoch only under "drop")
        stale = (ev.gen != self.session_gen or
                 (self.stale_eou_policy == "drop" and ev.epoch != self.seg_epoch))
        if stale:
            self.ledger.end_decision(ev.key)
            self.ledger.sweep(t_dec, self._commit_op, cause="stale")
            self._arm_timers()
            await self.send_control("tact_decision_stale", {
                "timestamp": self._wall_ts(), "turn": ev.turn,
                "epoch": ev.epoch, "current_epoch": self.seg_epoch})
            return

        # release path 3: decision timeout -> fail-open sweep (expected zero-trigger)
        if ev.timed_out:
            self.ledger.end_decision(ev.key)
            self.ledger.sweep(t_dec, self._commit_op, cause="timeout")
            self._arm_timers()
            await self.send_control("tact_decision_timeout", {
                "timestamp": self._wall_ts(), "turn": ev.turn, "infer": ev.infer})
            return

        # release path 1: normal return — ops apply FIRST, then sweep (the barrier)
        dec = ev.decision or {"dialogue": "stay", "ops": [], "say": ""}
        await self.send_control("tact_decision_done", {
            "timestamp": self._wall_ts(), "infer_time": ev.infer, "turn": ev.turn,
            "state": self.STATE, "n_ops": len(dec.get("ops", []))})
        if ev.error:
            await self.send_control("tact_decision_error", {
                "timestamp": self._wall_ts(), "error": ev.error})

        say = dec.get("say", "") if self.blocking_mode else tact_core.ack_fallback(dec)
        applied = apply_decision_ops(self.tx, self.ledger, dec, t_dec,
                                     immediate=self._immediate(),
                                     commit_cb=self._commit_op,
                                     dag=self.dag, comp_registry=self.comp_registry)
        self.ledger.end_decision(ev.key)
        self.ledger.sweep(t_dec, self._commit_op, cause="decision_done")
        self._arm_timers()

        self.tact_decisions.append({
            "seg_idx": len(self.tact_eous) - 1, "t_eou": round(info.get("t_eou", ev.t_eou), 3),
            "infer_s": ev.infer, "say": dec.get("say", ""), "ops": applied,
            "repaired": bool(dec.get("_repaired"))})
        for op_applied in applied:
            await self.send_control("tact_op_applied", {
                "timestamp": self._wall_ts(), "t_audio": round(self.t_audio, 3),
                "op": op_applied})

        if say:
            self.say_events.append((t_dec, say))
            self.assistant_history.append(say)
            if self.tts_enabled:
                self._say_kind = self._classify_say(dec, applied)
                self._emit_say(say, applied, ev.turn)

    @staticmethod
    def _classify_say(dec, applied):
        """Floor-policy utterance kind: announcing an imminent COMP/IRR commit
        is a 'confirmation'; template acks are 'ack'; everything else is
        narration (unconditionally interruptible)."""
        from tact.tools import REVERSIBILITY
        from tact.transaction import Reversibility
        for a in applied:
            if a.get("type") == "launch" and REVERSIBILITY.get(
                    a.get("fn", ""), Reversibility.IRR) in (
                    Reversibility.COMP, Reversibility.IRR):
                return "confirmation"
        return "ack" if dec.get("_ack_template") else "narration"

    def _emit_say(self, say, applied, turn):
        """TTS emission hook (single-phase baseline). engine_b_ack.TactEngineWithAck
        overrides this with the two-phase ack-v0 path — audio-side only; transaction
        semantics, windows and the barrier are untouched by any override."""
        if self.tts_split:
            self.dispatch_tts_sentences(say, turn, kind=self._say_kind)
        else:
            self.dispatch_tts(say, turn)

    # ------------------------------------------------------------------
    # sentence-split TTS (tts_split flag; W3 D4) + floor-holding v0
    # ------------------------------------------------------------------
    def dispatch_tts_sentences(self, say, turn, kind="narration"):
        from tts_sentence import split_sentences
        sents = split_sentences(say)
        if len(sents) <= 1:
            self.dispatch_tts(say, turn)
            return
        self._utt_seq += 1
        uid = self._utt_seq
        # allow_upto None = no restriction; set by the floor decision on barge-in
        self._utt_state[uid] = {"kind": kind, "n": len(sents),
                                "allow_upto": None, "last_sent": -1}
        gen = self.session_gen

        if self.replay_mode != "realtime":
            t = self.t_audio
            for idx, s in enumerate(sents):
                res = self.decision_script("tts", {"turn": turn, "t_audio": t,
                                                   "text": s})
                infer = 0.0 if self.replay_mode == "oracle" else float(res.get("infer", 0.0))
                t += infer
                self._schedule(t, TtsSentDone(uid=uid, idx=idx, gen=gen, turn=turn,
                                              text=s, infer=infer,
                                              dur_audio=float(res.get("dur_audio", 0.0))))
            return

        async def _run():
            for idx, s in enumerate(sents):
                st = self._utt_state.get(uid)
                if st and st["allow_upto"] is not None and idx > st["allow_upto"]:
                    # best-effort early exit (authoritative drop is in the handler)
                    self.q.put_nowait(TtsSentDone(uid=uid, idx=idx, gen=gen,
                                                  turn=turn, text=s, cancelled=True))
                    continue
                t0 = time.perf_counter()
                err, path, dur, raw = "", "", 0.0, b""

                def _work(sent=s, i=idx):
                    import soundfile as sf
                    p = None
                    if self.output_dir is not None:
                        from pathlib import Path
                        p = Path(self.output_dir) / f"turn{turn}_tts_s{i}.wav"
                    fp = self.tts_fn(sent, p)
                    data, sr = sf.read(fp, dtype="float32")
                    with open(fp, "rb") as f:
                        return fp, len(data) / sr, f.read()
                try:
                    path, dur, raw = await asyncio.to_thread(_work)
                except Exception as exc:
                    err = str(exc)
                    print(f"[TTS SENT ERROR] {exc}")
                self.q.put_nowait(TtsSentDone(
                    uid=uid, idx=idx, gen=gen, turn=turn, text=s,
                    infer=round(time.perf_counter() - t0, 3),
                    dur_audio=dur, audio_bytes=raw, error=err))
        self._inflight += len(sents)
        asyncio.create_task(_run())

    async def _on_tts_sent(self, ev: TtsSentDone):
        if self.replay_mode == "realtime":
            self._inflight = max(0, self._inflight - 1)
        if ev.gen != self.session_gen:
            return
        st = self._utt_state.get(ev.uid)
        dropped = (ev.cancelled or
                   (st is not None and st["allow_upto"] is not None
                    and ev.idx > st["allow_upto"]))
        if dropped:
            self.trace.append({"event": "tts_sent_dropped",
                               "data": {"uid": ev.uid, "idx": ev.idx,
                                        "t_audio": round(self.t_audio, 3)}})
            return
        if self.websocket is not None and ev.audio_bytes:
            await self.websocket.send_bytes(ev.audio_bytes)
        if st is not None:
            st["last_sent"] = max(st["last_sent"], ev.idx)
        await self.send_control("tts_sent_done", {
            "timestamp": self._wall_ts(), "turn": ev.turn, "uid": ev.uid,
            "idx": ev.idx, "infer_time": ev.infer,
            "dur_audio": round(ev.dur_audio, 3),
            "first_sentence": ev.idx == 0})   # the completion-anchor instrument
        if not ev.error and ev.dur_audio:
            self.STATE = "SPEAK"
            self.playback_end_audio = self.t_audio + ev.dur_audio

    def _apply_floor_decision(self):
        """Barge-in during SPEAK (floor_holding flag): decide the fate of the
        not-yet-played sentences of the open utterance. Sentence-granular:
        yield = nothing further; finish_clause = at most one more sentence.
        The user's speech is ALWAYS processed as a revision regardless."""
        import floor_policy
        uid, st = None, None
        for u in sorted(self._utt_state, reverse=True):
            s = self._utt_state[u]
            if s["last_sent"] < s["n"] - 1 and s["allow_upto"] is None:
                uid, st = u, s
                break
        if st is None:
            return
        win = [r for r in self.ledger.win.values()]
        fns = [op.fn for op in self.tx.pending.values()]
        tier = floor_policy.decide(st["kind"],
                                   window_remaining_s=min(win) if win else None,
                                   eta_prior_s=floor_policy.eta_prior(fns))
        st["allow_upto"] = st["last_sent"] + (1 if tier == "finish_clause" else 0)
        self.trace.append({"event": "floor_decision",
                           "data": {"uid": uid, "tier": tier, "kind": st["kind"],
                                    "last_sent": st["last_sent"],
                                    "allow_upto": st["allow_upto"],
                                    "t_audio": round(self.t_audio, 3)}})


    # ------------------------------------------------------------------
    # timers / tool results
    # ------------------------------------------------------------------
    async def _on_window_timer(self, ev: WindowTimer):
        self._armed.discard(ev.op_id)
        if ev.gen != self.session_gen:
            return
        self._burn_to(ev.due)
        # op survived (speech froze the budget) -> re-project from the current clock
        self._arm_timers()

    async def _on_tool_done(self, ev: ToolDone):
        if self.replay_mode == "realtime":
            self._inflight = max(0, self._inflight - 1)
        if ev.gen != self.session_gen:
            return
        for op in self.tx.committed:
            if op.op_id == ev.op_id:
                op.result = ev.result
                break
        for rec in self.commit_records:
            if rec["op_id"] == ev.op_id and rec["tool_wall_s"] is None:
                rec["tool_wall_s"] = ev.wall
                break
        await self.send_control("act_result", {
            "timestamp": self._wall_ts(), "op_id": ev.op_id,
            "tool_wall_s": ev.wall, "error": ev.error})

    # ------------------------------------------------------------------
    # event routing
    # ------------------------------------------------------------------
    async def _drain_due(self, t_audio):
        while self._scheduled and self._scheduled[0][0] <= t_audio:
            _, _, done = heapq.heappop(self._scheduled)
            if isinstance(done, ModelDone):
                await self._on_model_done(done)
            else:
                await self._process_event(done)

    async def _process_event(self, ev):
        if isinstance(ev, TactDecisionDone):
            if self.replay_mode == "realtime":
                self._inflight = max(0, self._inflight - 1)
            await self._on_tact_decision(ev)
            return True
        if isinstance(ev, WindowTimer):
            await self._on_window_timer(ev)
            return True
        if isinstance(ev, ToolDone):
            await self._on_tool_done(ev)
            return True
        if isinstance(ev, TtsSentDone):
            await self._on_tts_sent(ev)
            return True
        if isinstance(ev, DissentWindowClose):    # retired v0 event: ignore
            return True
        return await super()._process_event(ev)

    # ------------------------------------------------------------------
    # frames: accumulate session audio + burn the silence clock + hold check
    # ------------------------------------------------------------------
    async def _on_frame(self, ev: FrameEvent):
        if self.phase == "b":
            self._session_frames.append(ev.pcm)
            self._session_samples += len(ev.pcm)
            self._burn_to(ev.t_audio)   # pre-frame IN_SPEECH decides freeze vs burn
        await super()._on_frame(ev)
        if self.phase == "b":
            await self._on_frame_hold_check(ev.t_audio)
            self._arm_timers()

    # ------------------------------------------------------------------
    # LISTEN: Phase-B replaces judge->shift->response with ONE tact decision.
    # SPEAK: v1 has no floor-holding (D4 item) — user speech is always processed
    # as a potential revision through the same LISTEN logic.
    # ------------------------------------------------------------------
    async def _listen_frame(self, ev: FrameEvent, event):
        if self.phase != "b":
            await super()._listen_frame(ev, event)
            return

        frame, t = ev.pcm, ev.t_audio
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "turn": self.TURN_IDX, "state": self.STATE, "timestamp": self._wall_ts()})
            self.IN_SPEECH = True
            self._hold_armed = False    # speech resumed within hold: no EoU
            self._invalidate_open_spec()
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
            # user went silent: the ledger resumes burning from here on
            self.IN_SPEECH = False
            self._hold_armed = True
            if self.speculative and self.mode == "tact":
                # W3 D6: dispatch NOW; the result stays inert until the hold
                # confirms this anchor as a real EoU (see _on_tact_decision).
                await self.send_control("tact_spec_dispatch", {
                    "timestamp": self._wall_ts(), "turn": self.TURN_IDX,
                    "anchor": round(t, 3)})
                self.dispatch_tact_decision(t + self.END_HOLD, self.TURN_IDX,
                                            spec=True)
            return

    def _invalidate_open_spec(self):
        """Speech resumed inside the hold: the projected EoU never happened.
        A stashed result releases its barrier guard immediately; an in-flight
        one is marked invalid and released on arrival (release path 2 variant)."""
        key = self._spec_open
        self._spec_open = None
        if key is None:
            return
        sp = self._spec_inflight.get(key)
        if sp is None:
            return
        if sp["stash"] is not None:
            self._spec_inflight.pop(key, None)
            self._tact_inflight.pop(key, None)
            self.ledger.end_decision(key)
            self.ledger.sweep(self._ledger_t, self._commit_op,
                              cause="spec_invalidated")
            self._arm_timers()
            self.trace.append({"event": "tact_spec_discarded",
                               "data": {"key": key,
                                        "t_audio": round(self.t_audio, 3)}})
        else:
            sp["state"] = "invalid"

    async def _speak_frame(self, ev: FrameEvent, event):
        if self.phase != "b":
            await super()._speak_frame(ev, event)
            return
        if (self.floor_holding and event and "start" in event
                and not self.IN_SPEECH):
            self._apply_floor_decision()
        await self._listen_frame(ev, event)

    async def _on_frame_hold_check(self, t):
        """END_HOLD expiry => EoU => the transactional decision applies.
        Speculative path: the decision was already dispatched at vad end —
        confirm its anchor and (if the result raced ahead) process the stash."""
        if getattr(self, "_hold_armed", False) and self.t_end_anchor is not None \
                and (t - self.t_end_anchor) >= self.END_HOLD:
            self._hold_armed = False
            self.SILENCE_COUNTER = 0
            t_eou = self.t_end_anchor + self.END_HOLD
            await self.send_control("vad_640_done", {
                "timestamp": self._wall_ts(), "turn": self.TURN_IDX, "state": self.STATE})
            user_audio = np.concatenate(self.BUFFER) if self.BUFFER else None
            self._judged_seg_end = t
            self.BUFFER = []
            spec_key = self._spec_open
            self._spec_open = None
            sp = self._spec_inflight.get(spec_key) if spec_key is not None else None
            if sp is not None:
                # EoU bookkeeping deferred from dispatch time (spec path)
                self._eou_count += 1
                self.tact_eous.append([self._eou_count - 1, round(t_eou, 3)])
                sp["state"] = "confirmed"
                if sp["stash"] is not None:
                    await self._on_tact_decision(sp["stash"])
            else:
                self.dispatch_tact_decision(t_eou, self.TURN_IDX)
            if self.asr_enabled and user_audio is not None and len(user_audio):
                self.dispatch_asr(user_audio, self.TURN_IDX)

    # ------------------------------------------------------------------
    # session finalize (offline runs): expire remaining windows on the tail
    # ------------------------------------------------------------------
    def finalize_windows(self):
        """Force-expire every remaining window into the infinite tail silence.
        Offline driver calls this after run_offline quiesces; commits are stamped
        at their nominal deadlines (audio-clock truth)."""
        import math as _math
        self.IN_SPEECH = False
        # Unconfirmed speculative decisions die with the tape: their hold never
        # expired, so the projected EoU never existed. Release the barrier
        # guards or the tail sweep would defer their snapshot ops forever.
        for key, sp in list(self._spec_inflight.items()):
            if sp["state"] != "confirmed":
                self._spec_inflight.pop(key, None)
                self._tact_inflight.pop(key, None)
                self.ledger.end_decision(key)
                self.trace.append({"event": "tact_spec_discarded",
                                   "data": {"key": key, "at": "finalize"}})
        self._spec_open = None
        self.ledger.advance_silence(self._ledger_t, _math.inf, self._commit_op)
        self.ledger.sweep(self._ledger_t, self._commit_op, cause="finalize")

    # ------------------------------------------------------------------
    # export (v0-compatible signature; w2r-compatible content)
    # ------------------------------------------------------------------
    def export_fdb_result(self, example_id: str, provider: str = "tact_b") -> dict:
        out = {
            "example_id": example_id,
            "provider": provider,
            "actual_tool_calls": self.tx.to_actual_tool_calls(),
            "transcript": self.say_events[-1][1] if self.say_events else "",
            "status": "completed",
            "transaction_log": self.tx.log,
            "commits": list(self.commit_records),
            "ledger": self.ledger.export(),
            "decisions": list(self.tact_decisions),
            "eous": list(self.tact_eous),
            "say_events": [[round(t, 3), s] for t, s in self.say_events],
        }
        if self.dag is not None:
            out["dag"] = self.dag.export()
        if self.speculative:
            out["speculative_dispatch"] = True
        return out


if __name__ == "__main__":
    print("Phase-B v1 engine (tact_core wired). Drive via scripts/w2r_stream_replay.py "
          "--engine full, or live via src/backend.py with engine.phase: b.")
