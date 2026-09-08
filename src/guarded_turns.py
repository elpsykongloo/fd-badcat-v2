"""Demo-only input admission and independent floor/response ownership.

Raw VAD does NOT advance the accepted-input epoch. Every callback is fenced by
session, input id and revision; the actor remains the only state writer.
"""
import asyncio
import json
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass, field

import numpy as np

from control_labels import decide_control
from input_audio import EchoEvidence, INPUT_PROTOCOL
from messages import build_audio_content


def route_messages(prompt, content, *, playing=False, reference=""):
    # Quote/bound metadata and explicitly delimit the following microphone block.
    # A bare trailing "assistant reference:" made Omni misattribute even a clear
    # user question to the assistant, especially when that reference was empty.
    context = "情境资料：" + json.dumps({"assistant_playing": bool(playing),
        "assistant_reference_text": reference[:512]}, ensure_ascii=False)
    context += "\n接下来的音频块是本次待判断的麦克风采样，请分类。"
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": [{"type": "text", "text": context}, content]}]


@dataclass
class InputSpan:
    sid: int
    start: float
    mode: str
    frames: list
    revision: int = 0
    end: float = None
    voice: bool = True
    admitted: bool = False
    decided: bool = False
    route: str = None
    checked_until: float = 0.
    task: object = None
    held_candidate: object = None
    echo_frames: int = 0
    frame_count: int = 0
    provisional_keep: bool = False
    reference_text: str = ""
    end_index: int = None


@dataclass
class InputDecision:
    gen: int
    sid: int
    revision: int
    closed: bool
    route: str = "keep"
    audit: dict = field(default_factory=dict)


class GuardedTurns:
    def _init_guarded(self):
        self.GUARDED_TURNS = bool(self.engine_cfg.get("guarded_turns", False) and self.CANDIDATE_TURNS)
        self.INPUT_REFERENCE = self.engine_cfg.get("input_protocol") == INPUT_PROTOCOL
        self._input_serial = 0
        self._guard_tasks = []
        self._guard_input = None
        self._guard_preroll = deque(maxlen=10)
        self._guard_wait_audio = []
        self._guard_wait_reason = None
        self._guard_outputs = {}
        self._guard_echo = EchoEvidence()
        self._guard_evidence = {}
        self._guard_last_evidence_t = -1.
        self._guard_max_samples = int(float(self.engine_cfg.get("max_input_seconds", 20)) * 16000)
        if not 16000 <= self._guard_max_samples <= 16000 * 30:
            raise ValueError("max_input_seconds must be in [1, 30]")
        if self.GUARDED_TURNS and not self.prompts.get("input_route"):
            raise ValueError("guarded_turns requires an independent input_route prompt")

    def _guard_detect(self, ev):
        pcm = ev.pcm
        if ev.reference is not None:
            self._guard_evidence = self._guard_echo.process(pcm, ev.reference)
            if self._guard_evidence["echo_only"]:
                pcm = np.zeros_like(pcm)
            if self.t_audio - self._guard_last_evidence_t >= .25:
                self._guard_last_evidence_t = self.t_audio
                self._observe("input_evidence", self._guard_evidence)
        return pcm

    def _guard_can_publish(self, c):
        span = self._guard_input
        return span is None or span.decided or span.admitted or span.provisional_keep

    async def _guard_hold(self, c, held):
        if c is not None and c.published and self._speech is c.pipeline and not self._speech_started:
            await self.send_control("speech_hold", {"utterance_id": c.pipeline.sid, "held": held})

    async def _guard_frame(self, ev, event):
        frame, t = ev.pcm, ev.t_audio
        span = self._guard_input
        if event and "start" in event:
            if span is not None and not span.decided:
                span.revision += 1
                if span.task is not None:
                    span.task.cancel()
                span.voice, span.end = True, None
                span.end_index = None
                span.provisional_keep = False
                span.frames.append(frame.copy())
            else:
                self._input_serial += 1
                held = self._candidate
                mode = "playing" if self._speech is not None and self._speech_started else (
                    "preplay" if held is not None else "listening")
                span = InputSpan(self._input_serial, t, mode,
                                 list(self._guard_preroll) + [frame.copy()], held_candidate=held)
                span.reference_text = self._speech_meta.text if self._speech_meta else ""
                self._guard_input = span
                await self._guard_hold(held, True)
            self.IN_SPEECH = True
            await self.send_control("vad_start", {"turn": self.TURN_IDX, "state": self.STATE,
                "input_id": span.sid, "timestamp": self._wall_ts()})
        elif span is not None and not span.decided:
            span.frames.append(frame.copy())
        self._guard_preroll.append(frame.copy())
        if span is not None and not span.decided:
            span.frame_count += 1
            span.echo_frames += bool(self._guard_evidence.get("echo_only"))
            if sum(map(len, span.frames)) > self._guard_max_samples:
                await self._guard_reject(span, "input_limit")
                await self.send_control("input_notice", {"message": "这段语音过长，请分段说。"})
            elif event and "end" in event:
                span.voice, span.end = False, t
                span.end_index = len(span.frames)
                span.revision += 1
                if span.task is not None:
                    span.task.cancel()
                await self.send_control("vad_done", {"turn": self.TURN_IDX, "state": self.STATE,
                    "input_id": span.sid, "timestamp": self._wall_ts()})
                self._guard_dispatch(span, closed=True)
            elif span.voice and (span.task is None or span.task.done()):
                # Early preplay verification prevents a noise blip from destroying
                # a valid candidate. Long input triggers a decision, never a stop.
                interval = .192 if span.mode == "preplay" and not span.checked_until else 1.5
                if t - max(span.start, span.checked_until) >= interval:
                    self._guard_dispatch(span, closed=False)
        c = self._candidate
        if c is not None and not c.confirmed and self._guard_can_publish(c) and t - c.anchor >= self.END_HOLD:
            self._judged_seg_end = t
            await self.send_control("vad_640_done", {"turn": self.TURN_IDX, "state": self.STATE,
                "timestamp": self._wall_ts()})
            await self._confirm_candidate()

    def _guard_dispatch(self, span, *, closed):
        span.checked_until = self.t_audio
        gen, revision = self.session_gen, span.revision
        audio = np.concatenate(span.frames[:span.end_index] if closed else span.frames)
        if not audio.size or float(np.max(np.abs(audio))) < 1e-5 or (
                span.echo_frames >= 3 and span.echo_frames / max(span.frame_count, 1) >= .5):
            self.q.put_nowait(InputDecision(gen, span.sid, revision, closed,
                audit={"acoustic_reject": True}))
            return
        content = build_audio_content(audio, 16000, self.AUDIO_BLOCK)
        playing = not span.admitted and (span.mode == "playing" or (
            self._speech is not None and self._speech_started))
        if playing:
            span.mode = "playing"
        messages = route_messages(self.prompts["input_route"], content,
                                  playing=playing, reference=span.reference_text)
        self._observe("input_dispatch", {"input_id": span.sid, "revision": revision,
            "closed": closed, "playing": playing, "audio_samples": len(audio)})

        async def classify(kind, request):
            async def call(msgs):
                parts = []
                request_kind = "interrupt" if playing else "spec_judge"
                async with aclosing(self._capacity_stream(request_kind, self.text_stream_fn, msgs,
                        case_context={"input_id": span.sid, "revision": revision, "closed": closed,
                                      "input_generation": gen, "playing": playing})) as source:
                    async for part in source:
                        parts.append(part)
                        if sum(map(len, parts)) > 1024:
                            raise ValueError("Oversized input decision")
                return "".join(parts)
            return await decide_control(call, request, kind,
                min(self.DECISION_TIMEOUT, float(self.engine_cfg.get("input_decision_timeout_s", 2))))

        async def run():
            # One typed decision owns admission, stopping and readiness. The
            # original HumDial binary classifier remains the flag-off baseline,
            # not a second veto: an AND gate compounds its false negatives.
            route, audit = await classify("input_route", messages)
            return InputDecision(gen, span.sid, revision, closed, route,
                                 {"route": audit, "playing": playing})

        task = asyncio.create_task(run())
        span.task = task
        self._guard_tasks = [t for t in self._guard_tasks if not t.done()] + [task]
        self._inflight += 1
        def done(task):
            result = InputDecision(gen, span.sid, revision, closed, audit={"cancelled": True})
            if not task.cancelled():
                try:
                    result = task.result()
                except Exception as exc:
                    result.audit = {"error": type(exc).__name__}
            result.audit["accounted"] = True
            self.q.put_nowait(result)
        task.add_done_callback(done)

    async def _guard_reject(self, span, reason):
        span.decided, span.route = True, "keep"
        self.IN_SPEECH = False
        if span.task is not None and not span.task.done():
            span.task.cancel()
        self._observe("input_rejected", {"input_id": span.sid, "reason": reason})
        await self.send_control("input_ignored", {"input_id": span.sid, "reason": reason})
        span.frames.clear()
        await self._guard_hold(span.held_candidate, False)
        c = self._candidate
        if c is not None and c.confirmed and c.pipeline is not None:
            await self._publish_candidate(c)

    async def _on_input_decision(self, ev):
        if ev.audit.get("accounted"):
            self._inflight = max(0, self._inflight - 1)
        span = self._guard_input
        if (ev.gen != self.session_gen or span is None or span.sid != ev.sid
                or span.revision != ev.revision or span.decided or ev.audit.get("cancelled")):
            self._observe("input_decision_stale", {"input_id": ev.sid, "revision": ev.revision})
            return
        self._observe("input_decision", {"input_id": ev.sid, "revision": ev.revision,
            "route": ev.route, "audit": ev.audit})
        if ev.route == "keep":
            if ev.closed:
                if span.admitted:
                    kept = span.frames[:span.end_index]
                    if sum(map(len, self._guard_wait_audio + kept)) + 1600 <= self._guard_max_samples:
                        self._guard_wait_audio += [np.concatenate(kept), np.zeros(1600, dtype=np.float32)]
                    else:
                        self._guard_wait_audio = []
                    self._guard_wait_reason = "uncertain_after_yield"
                await self._guard_reject(span, "insufficient_evidence")
            else:
                span.provisional_keep = True
                await self._guard_hold(span.held_candidate, False)
            return
        if (self._speech is not None and self._speech_started and not span.admitted
                and not ev.audit.get("playing")):
            # Playback may have started while a preplay decision was in flight.
            # Re-evaluate with playback context before cancelling audible speech
            # (e.g. "嗯嗯" can be a greeting when idle, a backchannel when playing).
            span.mode = "playing"
            self._guard_dispatch(span, closed=ev.closed)
            return
        # No speech synthesis is initiated by yielding the floor.
        if not span.admitted:
            span.admitted = True
            old_candidate = self._candidate
            prefix = []
            if old_candidate is not None and (self._speech is None or not self._speech_started) and ev.route != "stop_only":
                prefix = [old_candidate.audio, np.zeros(1600, dtype=np.float32)]
            self._invalidate_candidate("accepted_user_input")
            if self._speech is not None:
                self._guard_mark_cancelled()
                self._cancel_speech("accepted_interrupt")
                self.TURN_IDX += 1
            self.seg_epoch += 1
            self.STATE = "LISTEN"
            self.CONTINUE_ARMED = self.SILENCE_COUNTER = 0
            self.playback_end_audio = self._playback_turn = None
            self._guard_wait_audio = prefix + self._guard_wait_audio
            self._observe("input_admitted", {"input_id": span.sid, "route": ev.route})
        if not ev.closed:
            self._guard_wait_reason = "user_speaking"
            return
        span.decided, span.route = True, ev.route
        self.IN_SPEECH = False
        if ev.route == "stop_only":
            self._guard_wait_audio = []
            self._guard_wait_reason = "stop_only"
            span.frames.clear()
            await self.send_control("input_waiting", {"reason": "stop_only", "input_id": span.sid})
            return
        audio_parts = self._guard_wait_audio + span.frames[:span.end_index]
        self._guard_wait_audio = []
        if sum(map(len, audio_parts)) > self._guard_max_samples:
            self._guard_wait_reason = "input_limit"
            span.frames.clear()
            await self.send_control("input_notice", {"message": "这段语音过长，请分段说。"})
            return
        audio = np.concatenate(audio_parts)
        span.frames.clear()
        if ev.route == "yield_wait":
            self._guard_wait_audio = [audio, np.zeros(1600, dtype=np.float32)]
            self._guard_wait_reason = "awaiting_user"
            await self.send_control("input_waiting", {"reason": "awaiting_user", "input_id": span.sid})
            return
        self._guard_wait_reason = None
        self.BUFFER = [audio]
        self.t_end_anchor = span.end
        # Route READY supplies input/readiness admission; retain the existing
        # third-party shift gate, frozen snapshots and private first-sentence TTS.
        self._begin_candidate(audio, stage="shift" if self.TURN_IDX else "response")

    async def _guard_finish_turn(self, turn, reason):
        if turn is None or turn != self.TURN_IDX:
            return
        self._guard_history_progress(self._speech.sid if self._speech else None,
                                     self._speech.played if self._speech else 0, completed=reason == "played")
        if self._speech is not None:
            old = self._speech
            old.cancel()
            if self._speech_jobs.pop(old.sid, None) is not None:
                self._inflight = max(0, self._inflight - 1)
        self._speech = self._speech_candidate = None
        # A still-running input decision keeps its identity and original mode;
        # EOF never routes the same raw fragment through a different admission.
        self.TURN_IDX += 1
        self.STATE = "LISTEN"
        self.playback_end_audio = self._playback_turn = None
        await self.send_control("turn_finished", {"turn": turn, "next_turn": self.TURN_IDX,
            "timestamp": self._wall_ts(), "reason": reason})

    def _guard_mark_cancelled(self):
        if self._speech is None:
            return
        sid = self._speech.sid
        record = self._guard_outputs.setdefault(sid, {"turn": self._speech_meta.turn, "sentences": []})
        record.update(cancelled=True, sent=self._speech.sent, played=self._speech.played)
        self._guard_history_progress(sid, self._speech.played)

    def _guard_history_progress(self, sid, samples, *, completed=False):
        record = self._guard_outputs.get(sid)
        if record is None:
            return
        record["played"] = max(record.get("played", 0), samples)
        if record.get("cancelled"):
            prefix = "".join(text for end, text in record["sentences"] if end <= record["played"])
            self._assistants_by_turn[record["turn"]] = prefix + "（此回答已被打断，其余内容未确认播完。）"
        elif completed:
            record["completed"] = True

    async def _guard_control(self, kind, data):
        if kind == "playback_stopped":
            sid, samples = data.get("utterance_id"), data.get("played_samples")
            record = self._guard_outputs.get(sid)
            if record and record.get("cancelled") and type(samples) is int and 0 <= samples <= record["sent"]:
                self._guard_history_progress(sid, samples)
                self._observe("playback_stopped", {"utterance_id": sid, "played_samples": samples})
        elif kind == "input_settings":
            safe = {k: v for k, v in data.items() if k in {"echoCancellation", "noiseSuppression", "autoGainControl", "sampleRate", "channelCount"}
                    and type(v) in (int, bool) and 0 <= v <= 192000}
            self._observe("input_settings", safe)

    def _guard_reset(self):
        for task in self._guard_tasks:
            task.cancel()
        self._guard_input = None
        self._guard_preroll.clear()
        self._guard_wait_audio = []
        self._guard_wait_reason = None
        self._guard_outputs.clear()
        self._guard_echo = EchoEvidence()
        self._guard_evidence = {}
        self._guard_last_evidence_t = -1.
