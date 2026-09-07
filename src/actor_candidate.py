"""Actor-only speculative turns; policy state is owned by the actor coroutine.

Unlike TACT's result-only invalidation, every candidate owns cancellable SSE
tasks. Both speculative and deferred modes use the SAME VAD-END snapshot.
Nothing from a private speech pipeline is published before hold confirmation.
"""
import asyncio
import time
from contextlib import aclosing
from dataclasses import dataclass, field

import numpy as np

from control_labels import decide_control
from messages import build_audio_content, scrub_audio_blocks
from speech_stream import SpeechEvent, SpeechPipeline, PROTOCOL


@dataclass
class CandidateResult:
    cid: int
    kind: str
    text: str = ""
    audit: dict = field(default_factory=dict)
    infer: float = 0.0
    cancelled: bool = False
    accounted: bool = True


@dataclass
class Candidate:
    cid: int
    gen: int
    epoch: int
    turn: int
    audio: object
    messages: dict
    resume_frames: list
    anchor: float
    confirmed: bool = False
    stage: str = "judge"
    launched: bool = False
    published: bool = False
    controls: list = field(default_factory=list)
    reported: int = 0
    tasks: list = field(default_factory=list)
    pipeline: object = None
    meta: object = None
    stash: list = field(default_factory=list)
    error: object = None
    restarts: int = 0
    created: float = field(default_factory=time.perf_counter)


class CandidateTurns:
    """Optional Actor mixin. No TACT ledger, tool parser or learned gate involved."""

    def _init_candidates(self):
        self.SPECULATIVE_RESPONSE = bool(self.engine_cfg.get("speculative_response", False))
        self.CANCELLABLE_RESPONSE = bool(self.engine_cfg.get("cancellable_response", False))
        self.CANDIDATE_TURNS = self.STREAMING and (self.SPECULATIVE_RESPONSE or self.CANCELLABLE_RESPONSE)
        self._candidate = self._speech_candidate = None
        self._candidate_serial = 0
        self._candidate_tasks = []
        self._answer_versions = {}
        self._speech_started = False

    def _candidate_log(self, event, **data):
        self.trace.append({"event": event, "data": {**data, "t_audio": self.t_audio}})
        self._observe(event, data)

    def _candidate_current(self, c):
        return (self._candidate is c and c.gen == self.session_gen
                and c.epoch == self.seg_epoch and c.turn == self.TURN_IDX)

    def _begin_candidate(self, audio, *, confirmed=False, stage="judge"):
        self._invalidate_candidate("replaced")
        self._candidate_serial += 1
        audio = np.array(audio, dtype=np.float32, copy=True)
        audio.setflags(write=False)
        # Encode once; all messages and history are captured synchronously, so
        # late ASR cannot change the candidate between judge, shift and response.
        content = build_audio_content(audio, 16000, self.AUDIO_BLOCK)
        prompts = {"judge": (self.JUDGE_PROMPT, False, False),
                   "shift": (self.SHIFT_PROMPT, False, True),
                   "response": (self.RESPONSE_PROMPT, True, False),
                   "shift_re": (self.SHIFT_RE_PROMPT, False, True)}
        messages = {kind: self.build_messages(prompt, None if kind == "shift_re" else audio,
                    history, shift_history, audio_content=content)
                    for kind, (prompt, history, shift_history) in prompts.items()}
        c = Candidate(self._candidate_serial, self.session_gen, self.seg_epoch,
                      self.TURN_IDX, audio, messages, self.BUFFER,
                      self.t_end_anchor if self.t_end_anchor is not None else self.t_audio,
                      confirmed=confirmed, stage=stage)
        self._candidate = c
        self._candidate_log("candidate_created", candidate_id=c.cid, turn=c.turn,
                            speculative=self.SPECULATIVE_RESPONSE and not confirmed,
                            audio_samples=len(audio), snapshot="vad-end" if not confirmed else "confirmed-input")
        if confirmed or self.SPECULATIVE_RESPONSE:
            self._launch_candidate(c)
        return c

    def _launch_candidate(self, c):
        c.launched = True
        if c.stage in ("response", "shift_re"):
            self.q.put_nowait(CandidateResult(c.cid, c.stage, accounted=False))
            return
        kind, messages = c.stage, c.messages[c.stage]
        # Even speculative judges use NORMAL. Only a non-speculative, confirmed
        # judge may take the control reserve. No synchronous detached HTTP here.
        request_kind = kind if c.confirmed else "spec_" + kind
        self._candidate_log("candidate_dispatch", candidate_id=c.cid, kind=kind,
                            confirmed=c.confirmed, request_class=self._request_class(request_kind))

        async def call(request):
            chunks, size = [], 0
            async with aclosing(self._capacity_stream(request_kind, self.text_stream_fn, request)) as source:
                async for part in source:
                    chunks.append(part)
                    size += len(part)
                    if size > 1024:
                        raise ValueError("Oversized binary control response")
            return "".join(chunks)

        async def run():
            started = time.perf_counter()
            label, audit = await decide_control(call, messages, kind, self.DECISION_TIMEOUT)
            return CandidateResult(c.cid, kind, label, audit, time.perf_counter() - started)

        task = asyncio.create_task(run())
        c.tasks.append(task)
        self._candidate_tasks = [t for t in self._candidate_tasks if not t.done()]
        self._candidate_tasks.append(task)
        self._inflight += 1

        def done(task):
            if task.cancelled():
                result = CandidateResult(c.cid, kind, cancelled=True)
            else:
                try:
                    result = task.result()
                except Exception as exc:
                    result = CandidateResult(c.cid, kind,
                        "no" if kind == "shift" else "continue",
                        {"kind": kind, "fallback": True, "errors": [type(exc).__name__]})
            # This callback also runs if cancellation happened before run().
            self.q.put_nowait(result)
        task.add_done_callback(done)

    async def _report_candidate_controls(self, c):
        from engine import ModelDone
        while c.reported < len(c.controls):
            ev = c.controls[c.reported]
            c.reported += 1
            await self.send_control("control_validation", {**ev.audit, "turn": c.turn,
                                    "candidate_id": c.cid, "timestamp": self._wall_ts()})
            await self._trace_llm_done(ModelDone(ev.kind, c.gen, c.epoch, c.turn,
                text=ev.text, infer=round(ev.infer, 3), timed_out=ev.audit.get("timed_out", False),
                prompt_snapshot=scrub_audio_blocks(c.messages[ev.kind])))

    async def _on_candidate_result(self, ev):
        if ev.accounted:
            self._inflight = max(0, self._inflight - 1)
        c = self._candidate
        if c is None or c.cid != ev.cid or not self._candidate_current(c) or ev.cancelled:
            self._candidate_log("candidate_result_discarded", candidate_id=ev.cid,
                                kind=ev.kind, cancelled=ev.cancelled)
            return
        if ev.kind in ("judge", "shift"):
            c.controls.append(ev)
            self._candidate_log("candidate_control_done", candidate_id=c.cid,
                                kind=ev.kind, label=ev.text, infer_ms=round(ev.infer * 1000, 2),
                                audit=ev.audit)
            if c.confirmed:
                await self._report_candidate_controls(c)
        if ev.kind == "judge":
            if ev.text == "continue":
                c.stage = "continue"
                if c.confirmed:
                    self._candidate_continue(c)
                return
            c.stage = "shift" if c.turn else "response"
            if c.stage == "shift":
                self._launch_candidate(c)
                return
        elif ev.kind == "shift":
            c.stage = "shift_re" if ev.text == "yes" else "response"
        await self._start_candidate_speech(c)

    def _candidate_continue(self, c):
        self._candidate = None
        self.CONTINUE_ARMED = self.IN_SPEECH = True
        self.t_continue_anchor = self._judged_seg_end
        self._candidate_log("candidate_continue", candidate_id=c.cid)

    async def _confirm_candidate(self):
        c = self._candidate
        if c is None or not self._candidate_current(c):
            return
        c.confirmed = True
        await self.send_control("candidate_confirmed", {
            "candidate_id": c.cid, "turn": c.turn, "stage": c.stage,
            "speculative": self.SPECULATIVE_RESPONSE,
            "prepared_audio_samples": c.pipeline.sent if c.pipeline else 0,
            "prepared_audio_ms": round(c.pipeline.sent / c.pipeline.rate * 1000, 1)
                if c.pipeline and c.pipeline.rate else 0,
            "precompute_ms": round((time.perf_counter() - c.created) * 1000, 1)})
        await self._report_candidate_controls(c)
        if not c.launched:
            self._launch_candidate(c)
        elif c.stage == "continue":
            self._candidate_continue(c)
        elif c.pipeline is not None:
            if c.error is not None:
                # Failed, still-private text/audio must not leak on confirmation.
                # One fresh pipeline may retry the same frozen selected response.
                c.restarts += 1
                self._discard_candidate_pipeline(c)
                c.error = None
                await self._start_candidate_speech(c)
            else:
                await self._publish_candidate(c)

    async def _start_candidate_speech(self, c):
        from engine import ModelDone, RESPONSE_TIMEOUT_APOLOGY
        if not self._candidate_current(c):
            return
        self._speech_serial += 1
        c.meta = ModelDone(c.stage, c.gen, c.epoch, c.turn,
            add_to_history=c.stage == "response", answer_id=c.cid,
            prompt_snapshot=scrub_audio_blocks(c.messages[c.stage]))
        gate = asyncio.Event()
        c.pipeline = SpeechPipeline(self._speech_serial, self.q, c.messages[c.stage],
            self._response_text_stream, self._response_tts_stream,
            timeout=self.DECISION_TIMEOUT, retry=c.stage == "response" and c.restarts == 0,
            apology=RESPONSE_TIMEOUT_APOLOGY,
            packet_ms=int(self.engine_cfg.get("stream_packet_ms", 40)),
            buffer_ms=int(self.engine_cfg.get("stream_buffer_ms", 600)), precompute_gate=gate,
            track_sentences=self.GUARDED_TURNS)
        self._speech_jobs[c.pipeline.sid] = c.pipeline
        self._speech_tasks = [t for t in self._speech_tasks if not t.done()]
        self._speech_tasks.append(c.pipeline.task)
        self._inflight += 1
        self._candidate_log("candidate_speech_started", candidate_id=c.cid,
                            utterance_id=c.pipeline.sid, confirmed=c.confirmed)
        if c.confirmed:
            await self._publish_candidate(c)

    def _stage_candidate_speech(self, c, ev):
        if ev.kind == "error":
            c.error = ev.data
        # Coalesce token events: memory is bounded by the pipeline's text limit,
        # two sentences and 600 ms audio credit, not one object per token.
        if ev.kind == "text_delta" and c.stash and c.stash[-1].kind == "text_delta":
            c.stash[-1].data["text"] += ev.data["text"]
        else:
            c.stash.append(SpeechEvent(ev.sid, ev.kind, dict(ev.data)))

    async def _publish_candidate(self, c):
        if not self._candidate_current(c) or not c.confirmed or c.published:
            return
        if self.GUARDED_TURNS and not self._guard_can_publish(c):
            return
        c.published = True
        self._speech = c.pipeline
        self._speech_meta = c.meta
        self._speech_candidate = c
        self._speech_started = False
        self._speech_audio_done = self._speech_played_reported = False
        self._answer_versions[c.turn] = c.cid
        if self.GUARDED_TURNS:
            if len(self._guard_outputs) >= 64:
                self._guard_outputs.pop(next(iter(self._guard_outputs)))
            self._guard_outputs[c.pipeline.sid] = {"turn": c.turn, "sentences": []}
        await self.send_control("speech_start", {"utterance_id": c.pipeline.sid,
            "turn": c.turn, "protocol": PROTOCOL, "buffer_ms": c.pipeline.buffer_ms,
            "candidate_id": c.cid, "timestamp": self._wall_ts()})
        c.pipeline.precompute_gate.set()
        staged, c.stash = c.stash, []
        for ev in staged:
            await self._on_speech_event(ev)

    def _discard_candidate_pipeline(self, c):
        if c.pipeline is not None:
            c.pipeline.cancel()
            if self._speech_jobs.pop(c.pipeline.sid, None) is not None:
                self._inflight = max(0, self._inflight - 1)
        c.stash.clear()

    def _invalidate_candidate(self, reason):
        c = self._candidate
        if c is None:
            return None
        self._candidate = None
        for task in c.tasks:
            task.cancel()
        self._candidate_log("candidate_cancelled", candidate_id=c.cid, reason=reason,
            stage=c.stage, published=c.published,
            prepared_audio_samples=c.pipeline.sent if c.pipeline else 0,
            wasted_ms=round((time.perf_counter() - c.created) * 1000, 1))
        if c.published and self._speech is c.pipeline:
            self._cancel_speech(reason)
        else:
            self._discard_candidate_pipeline(c)
        if self._answer_versions.get(c.turn) == c.cid:
            self._answer_versions.pop(c.turn, None)
        return c

    def _resume_candidate_input(self):
        c = self._invalidate_candidate("user_resumed_before_playback")
        if c is None:
            return
        self.BUFFER = list(c.resume_frames)
        self.STATE = "LISTEN"
        self.IN_SPEECH = True
        self.interrupt_buf = []
        self.SILENCE_COUNTER = 0
        self.CONTINUE_ARMED = self._seg_closed = False
        self.t_continue_anchor = self.t_interrupt_start = None
        self.playback_end_audio = self._playback_turn = None

    async def _candidate_playback_started(self):
        c = self._speech_candidate
        if c is None or self._speech_started or not c.pipeline.sent:
            return
        self._speech_started = True
        if self._candidate is c:
            self._candidate = None
        c.resume_frames.clear()
        if c.meta.add_to_history:
            self.dispatch_asr(c.audio, c.turn, answer_id=c.cid)
        self._commit_speech_history(c.meta)
        await self.send_control("speech_playback_started", {
            "utterance_id": c.pipeline.sid, "turn": c.turn, "candidate_id": c.cid})

    def _commit_speech_history(self, meta):
        if meta.add_to_history and meta.text and not meta.history_written:
            if meta.answer_id and self._answer_versions.get(meta.turn) != meta.answer_id:
                return
            self.assistant_history.append(meta.text)
            if self.CHAT_DEMO:
                self._assistants_by_turn[meta.turn] = meta.text
                if self.GUARDED_TURNS:
                    for sid, record in self._guard_outputs.items():
                        if record["turn"] == meta.turn and record.get("cancelled"):
                            self._guard_history_progress(sid, record.get("played", 0))
            meta.history_written = True

    async def _listen_candidate_frame(self, ev, event):
        frame, t = ev.pcm, ev.t_audio
        if event and "start" in event:
            if not self.IN_SPEECH:
                self.BUFFER = []
            self.IN_SPEECH = True
            self.BUFFER.append(frame)
            self.SILENCE_COUNTER = 0
            self.CONTINUE_ARMED = False
            self.t_continue_anchor = None
            await self.send_control("vad_start", {"turn": self.TURN_IDX,
                "state": self.STATE, "timestamp": self._wall_ts()})
            return
        if not self.IN_SPEECH:
            if self._candidate is not None:
                self._candidate.resume_frames.append(frame)
            return
        self.BUFFER.append(frame)
        if event and "end" in event:
            self.SILENCE_COUNTER = 1
            self.t_end_anchor = t
            await self.send_control("vad_done", {"turn": self.TURN_IDX,
                "state": self.STATE, "timestamp": self._wall_ts()})
            self._begin_candidate(np.concatenate(self.BUFFER))
            return
        if self.SILENCE_COUNTER and t - self.t_end_anchor >= self.END_HOLD:
            self.SILENCE_COUNTER = 0
            self._judged_seg_end = t
            await self.send_control("vad_640_done", {"turn": self.TURN_IDX,
                "state": self.STATE, "timestamp": self._wall_ts()})
            await self._confirm_candidate()
            return
        if self.CONTINUE_ARMED and t - self.t_continue_anchor >= self.AFTER_CONTINUE_TIMEOUT:
            self.CONTINUE_ARMED = False
            self.t_continue_anchor = None
            self._begin_candidate(np.concatenate(self.BUFFER), confirmed=True,
                                  stage="shift" if self.TURN_IDX else "response")
