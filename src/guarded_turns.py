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

from control_labels import decide_input_route, ROUTE_PROTOCOL, ROUTE_MAX_OUTPUT, REPLY_PROTOCOL
from input_audio import EchoEvidence, INPUT_PROTOCOL
from messages import build_audio_content
from speech_reference import SpeechReference, completed_context


def input_timing(config=None):
    """Only pre-roll is tuned; keep the evaluated Silero threshold/end window."""
    config = config or {}
    preroll = config.get("input_preroll_ms", 160)
    if type(preroll) is not int or not 0 <= preroll <= 960 or preroll % 16:
        raise ValueError("input_preroll_ms must be a multiple of 16 in [0, 960]")
    return {"vad_threshold": 0.5, "vad_silence_ms": 100, "preroll_ms": preroll}


CONTINUATION_PROTOCOL = "pending-user-audio-v1"


def route_messages(prompt, content, *, playing=False, reference="", continuation=None,
                   context_content=None):
    # Keep the keyword for existing callers, but never send assistant text: the
    # transcriber can copy it into fictitious microphone speech. Acoustic
    # reference filtering and playback-aligned diagnostic snapshots are intact.
    metadata = {"assistant_playing": bool(playing)}
    if continuation:
        metadata["pending_user_audio"] = continuation
        replied_context = continuation.get("source") == "recent_replied_user_audio"
        transcript_rule = ("transcript只逐字转写起点之后的新增语音，不得复制前半段旧语音；"
                           if replied_context else "transcript逐字转写整个音频。")
        prompt += ("\n本次音频包含两部分：先是此前已接纳、尚未回答完的用户表达，再是本次新增语音。"
                   "情境资料给出新增语音的起点。请听完整段音频，把真正的续说与前文一起理解，"
                   "不能仅因新增部分单独看不像完整句就丢弃。" + transcript_rule +
                   "label判断用户最新意图：新增部分明确停止则stop_only，要求继续等则yield_wait；"
                   "新增部分完成此前表达或提出独立完整新请求才可yield_ready。"
                   "如果新增部分只是噪声、回声、无关碎音或明显向第三方讲话，仍为keep；"
                   "不能只凭先前已保存的语音再次授权回答。不要使用助手的未播内容。")
        if replied_context:
            prompt += ("\n前半段用户语音的回答已经开始；它仅用来理解后半段的续说、补充或纠正，"
                       "不是一个需要再次回应的旧请求。只按后半段最新意图决定动作；"
                       "若后半段只是在附和、赞同或鼓励，必须keep。不要让前半段的完整请求覆盖后半段。")
    context = "情境资料：" + json.dumps(metadata, ensure_ascii=False)
    if context_content is not None:
        # Distinct audio blocks are a real boundary. A timestamp inside one
        # concatenated waveform did NOT stop Omni from copying old speech into
        # the new transcript and treating a backchannel as a repeated request.
        prompt += ("\n有两个独立音频块：第一块是已接纳的此前用户语音，仅作上下文；"
                   "第二块是当前新增麦克风语音。只转写第二块到transcript；"
                   "听两块来理解续说，但label只按第二块最新意图判断。"
                   "第一块已有的完整请求不能让第二块的附和变成新请求。")
        return [{"role": "system", "content": prompt}, {"role": "user", "content": [
            {"type": "text", "text": context + "\n第一块：此前用户语音，仅作上下文。"}, context_content,
            {"type": "text", "text": "第二块：新增用户语音，请只转写这一块并判断最新意图。"}, content]}]
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
    reference_sid: int = None
    reference_kind: str = "unavailable"
    reference_played: int = 0
    end_index: int = None
    preroll_samples: int = 0
    reply_context: str = ""
    reply_played_samples: int = 0
    # Immutable user audio frozen at input onset; route context only, not a
    # second copy in ASR/response history or an assistant-text reference.
    replied_user_audio: object = None


@dataclass
class InputDecision:
    gen: int
    sid: int
    revision: int
    closed: bool
    route: str = "keep"
    audit: dict = field(default_factory=dict)
    call_ids: list = field(default_factory=list)
    parent_id: str = None


class GuardedTurns:
    def _init_guarded(self):
        self.GUARDED_TURNS = bool(self.engine_cfg.get("guarded_turns", False) and self.CANDIDATE_TURNS)
        self.INPUT_REFERENCE = self.engine_cfg.get("input_protocol") == INPUT_PROTOCOL
        self._input_serial = 0
        self._guard_tasks = []
        self._guard_input = None
        self.input_timing = input_timing(self.engine_cfg if self.GUARDED_TURNS else {})
        self._guard_preroll = deque(maxlen=self.input_timing["preroll_ms"] // 16)
        self._guard_wait_audio = []
        self._guard_wait_end = None
        self.CONTINUATION_CONTEXT = bool(self.GUARDED_TURNS and self.engine_cfg.get("input_continuation_context", False))
        self._guard_context_gap = float(self.engine_cfg.get("input_context_max_gap_s", 20))
        if not 1 <= self._guard_context_gap <= 30:
            raise ValueError("input_context_max_gap_s must be in [1, 30]")
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

    def _guard_reference_event(self, ev):
        # Called AFTER the engine's private-candidate and stale-utterance gates.
        # Do not update ModelDone.text: that would commit partial history on ACK.
        record = self._guard_outputs.setdefault(ev.sid, {
            "turn": self._speech_meta.turn, "sentences": []})
        if "reference" not in record:
            record["reference"] = SpeechReference()
        record["reference"].observe(ev.kind, ev.data)

    def _guard_capture_reference(self, span):
        speech = self._speech
        if speech is None:
            return
        # Never replace the onset snapshot with a different answer's text.
        if span.reference_sid is not None and span.reference_sid != speech.sid:
            return
        record = self._guard_outputs.get(speech.sid, {})
        reference = record.get("reference")
        if reference is None or record.get("cancelled") or record.get("completed"):
            return
        span.reference_text, span.reference_kind = reference.snapshot(
            started=self._speech_started, played=speech.played, sent=speech.sent)
        span.reference_sid, span.reference_played = speech.sid, speech.played

    async def _guard_hold(self, c, held):
        if c is not None and c.published and self._speech is c.pipeline and not self._speech_started:
            await self.send_control("speech_hold", {"utterance_id": c.pipeline.sid, "held": held})

    def _guard_pending_audio(self, span):
        """Read-only snapshot of audio that admission would merge into a reply.

        Before admission a held, unplayed candidate still owns its input. After
        admission that exact prefix is in wait_audio instead: never attach both.
        Playback-start races discard this candidate context on re-evaluation.
        """
        if not self.CONTINUATION_CONTEXT:
            return []
        prefix = []
        held = span.held_candidate
        if (not span.admitted and held is not None and held is self._candidate
                and (self._speech is None or not self._speech_started)):
            prefix = [held.audio, np.zeros(1600, dtype=np.float32)]
        return prefix + self._guard_wait_audio

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
                if (self.CONTINUATION_CONTEXT and self._guard_wait_audio
                        and self._guard_wait_end is not None
                        and t - self._guard_wait_end > self._guard_context_gap):
                    self._observe("input_context_expired", {"protocol": CONTINUATION_PROTOCOL,
                        "gap_s": round(t - self._guard_wait_end, 3),
                        "context_samples": sum(map(len, self._guard_wait_audio))})
                    self._guard_wait_audio = []
                    self._guard_wait_end = None
                    self._guard_wait_reason = None
                self._input_serial += 1
                held = self._candidate
                mode = "playing" if self._speech is not None and self._speech_started else (
                    "preplay" if held is not None else "listening")
                span = InputSpan(self._input_serial, t, mode,
                                 list(self._guard_preroll) + [frame.copy()], held_candidate=held,
                                 preroll_samples=sum(map(len, self._guard_preroll)))
                self._guard_capture_reference(span)
                # Freeze once at VAD onset, including an empty snapshot. Later
                # ACKs/END/first-playback refreshes must never fill in the future.
                if mode == "playing":
                    record = self._guard_outputs.get(self._speech.sid, {})
                    if not record.get("cancelled") and not record.get("completed"):
                        span.reply_played_samples = self._speech.played
                        span.reply_context = completed_context(record.get("sentences", []),
                                                               span.reply_played_samples)
                    c = self._speech_candidate
                    if (self.CONTINUATION_CONTEXT and c is not None
                            and 0 <= t - c.anchor <= self._guard_context_gap):
                        span.replied_user_audio = c.audio
                elif mode == "preplay" and self.CONTINUATION_CONTEXT:
                    # If playback wins an in-flight admission race, the same
                    # onset-frozen USER input is still available to re-route.
                    span.replied_user_audio = held.audio
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
            context_size = sum(map(len, self._guard_pending_audio(span)))
            if context_size + sum(map(len, span.frames)) > self._guard_max_samples:
                if self.CONTINUATION_CONTEXT:
                    self._guard_wait_audio = []
                    self._guard_wait_end = None
                    self._guard_wait_reason = "input_limit"
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
        operation_id = self._diagnostic_id("input")
        audio = np.concatenate(span.frames[:span.end_index] if closed else span.frames)
        current_samples = len(audio)
        if not audio.size or float(np.max(np.abs(audio))) < 1e-5 or (
                span.echo_frames >= 3 and span.echo_frames / max(span.frame_count, 1) >= .5):
            self.q.put_nowait(InputDecision(gen, span.sid, revision, closed,
                audit={"acoustic_reject": True}))
            return
        # Acoustic admission is checked on the NEW frames above. Old accepted
        # speech must never make current silence/echo look like a new request.
        context_audio = self._guard_pending_audio(span)
        context_source = "accepted_unanswered_user_audio" if context_audio else None
        # During playback, a fragment such as "of the refugee crisis" must not
        # lose the recent question it completes. This is route-only context:
        # normal response history and new-user ASR retain their existing inputs.
        # Optional played context cannot consume the new input's audio budget.
        previous = span.replied_user_audio
        if (not context_audio and previous is not None
                and len(previous) + 1600 + current_samples <= self._guard_max_samples):
            context_audio = [previous, np.zeros(1600, dtype=np.float32)]
            context_source = "recent_replied_user_audio"
        context_samples = sum(map(len, context_audio))
        continuation = None
        context_content = None
        if context_samples:
            continuation = {"protocol": CONTINUATION_PROTOCOL,
                "current_start_ms": round(context_samples / 16, 3),
                "source": context_source}
            if context_source == "recent_replied_user_audio":
                continuation.update(route_only=True, transcript_scope="new_audio_only", audio_layout="two_blocks")
                context_content = build_audio_content(np.concatenate(context_audio), 16000, self.AUDIO_BLOCK)
            else:
                audio = np.concatenate([*context_audio, audio])
        content = build_audio_content(audio, 16000, self.AUDIO_BLOCK)
        playing = not span.admitted and (span.mode == "playing" or (
            self._speech is not None and self._speech_started))
        if playing:
            span.mode = "playing"
            # An input may start before first playback/first text and be checked
            # again after the start ACK. Refresh that same utterance once; normal
            # playing input retains its onset snapshot across END/repair/EOF.
            if span.reference_kind != "playback_sentence_window":
                self._guard_capture_reference(span)
        messages = route_messages(self.prompts["input_route"], content,
                                  playing=playing, reference=span.reference_text,
                                  continuation=continuation, context_content=context_content)
        self._observe("input_dispatch", {"input_id": span.sid, "revision": revision,
            "closed": closed, "playing": playing, "audio_samples": current_samples + context_samples,
            "current_audio_samples": current_samples, "context_audio_samples": context_samples,
            "context_source": context_source,
            "continuation_protocol": CONTINUATION_PROTOCOL if continuation else None,
            "parent_id": operation_id,
            "reference_kind": span.reference_kind, "reference_utterance_id": span.reference_sid,
            "reference_chars": len(span.reference_text), "reference_played_samples": span.reference_played,
            "reply_context_chars": len(span.reply_context), "reply_played_samples": span.reply_played_samples})
        reference_context = {"reference_kind": span.reference_kind,
            "reference_utterance_id": span.reference_sid,
            "reference_played_samples": span.reference_played,
            "route_protocol": ROUTE_PROTOCOL, "reference_sent_to_model": False,
            "input_timing": self.input_timing, "preroll_samples": span.preroll_samples,
            "route_penalties": {"presence": 0.0, "frequency": 0.0}}
        reply_context = span.reply_context
        reference_context.update(reply_protocol=REPLY_PROTOCOL,
            reply_context_chars=len(reply_context), reply_played_samples=span.reply_played_samples)
        reference_context.update(continuation_protocol=CONTINUATION_PROTOCOL if continuation else None,
            context_audio_samples=context_samples, current_audio_samples=current_samples,
            context_source=context_source)

        async def call(msgs, stage):
            parts = []
            call_id = self._diagnostic_id("call")
            call_ids.append(call_id)
            request_kind = "interrupt" if playing else "spec_judge"
            async with aclosing(self._capacity_stream(request_kind, self.text_stream_fn, msgs, route=True,
                    case_context={"input_id": span.sid, "revision": revision, "closed": closed,
                                  "input_generation": gen, "playing": playing,
                                  **reference_context, "decision_stage": stage,
                                  "reference_sent_to_model": stage == "input_reply"},
                                  call_id=call_id, parent_id=operation_id)) as source:
                async for part in source:
                    parts.append(part)
                    if sum(map(len, parts)) > (128 if stage == "input_reply" else ROUTE_MAX_OUTPUT):
                        raise ValueError("Oversized input decision")
            return "".join(parts)

        call_ids = []

        async def run():
            # One typed decision owns admission, stopping and readiness. The
            # original HumDial binary classifier remains the flag-off baseline,
            # not a second veto: an AND gate compounds its false negatives.
            route, audit = await decide_input_route(call, messages,
                min(self.DECISION_TIMEOUT, float(self.engine_cfg.get("input_decision_timeout_s", 2))),
                playing=playing, closed=closed, reply_prompt=self.prompts.get("input_reply"),
                reply_context=reply_context)
            return InputDecision(gen, span.sid, revision, closed, route,
                                 {"route": audit, "playing": playing},
                                 list(call_ids), operation_id)

        task = asyncio.create_task(run())
        span.task = task
        self._guard_tasks = [t for t in self._guard_tasks if not t.done()] + [task]
        self._inflight += 1
        def done(task):
            result = InputDecision(gen, span.sid, revision, closed,
                                   audit={"cancelled": True}, parent_id=operation_id)
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
            "route": ev.route, "audit": ev.audit, "call_ids": ev.call_ids,
            "parent_id": ev.parent_id})
        if ev.route == "keep":
            if ev.closed:
                if span.admitted:
                    kept = span.frames[:span.end_index]
                    if sum(map(len, self._guard_wait_audio + kept)) + 1600 <= self._guard_max_samples:
                        self._guard_wait_audio += [np.concatenate(kept), np.zeros(1600, dtype=np.float32)]
                        self._guard_wait_end = span.end
                    else:
                        self._guard_wait_audio = []
                        self._guard_wait_end = None
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
            self._guard_wait_end = None
            self._guard_wait_reason = "stop_only"
            span.frames.clear()
            await self.send_control("input_waiting", {"reason": "stop_only", "input_id": span.sid})
            return
        audio_parts = self._guard_wait_audio + span.frames[:span.end_index]
        self._guard_wait_audio = []
        self._guard_wait_end = None
        if sum(map(len, audio_parts)) > self._guard_max_samples:
            self._guard_wait_reason = "input_limit"
            span.frames.clear()
            await self.send_control("input_notice", {"message": "这段语音过长，请分段说。"})
            return
        audio = np.concatenate(audio_parts)
        span.frames.clear()
        if ev.route == "yield_wait":
            self._guard_wait_audio = [audio, np.zeros(1600, dtype=np.float32)]
            self._guard_wait_end = span.end
            self._guard_wait_reason = "awaiting_user"
            await self.send_control("input_waiting", {"reason": "awaiting_user", "input_id": span.sid})
            return
        self._guard_wait_reason = None
        self.BUFFER = [audio]
        self.t_end_anchor = span.end
        # Route READY supplies input/readiness admission; retain the existing
        # third-party shift gate, frozen snapshots and private first-sentence TTS.
        self._begin_candidate(audio, stage="shift" if self.TURN_IDX else "response",
                              parent_id=ev.parent_id)

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
            # Model-facing history is a transcript, so it must contain only
            # words that the user actually heard.  Keep interruption metadata
            # in this record/trace; natural-language annotations in an
            # assistant message are liable to be copied into later answers.
            if prefix:
                self._assistants_by_turn[record["turn"]] = prefix
                self._observe("history_write", {"role": "assistant", "turn": record["turn"],
                    "text": prefix, "source": "played_prefix", "utterance_id": sid})
            else:
                self._assistants_by_turn.pop(record["turn"], None)
                self._observe("history_remove", {"role": "assistant", "turn": record["turn"],
                    "source": "no_confirmed_playback", "utterance_id": sid})
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
            if self.demo_trace is not None:
                self.demo_trace.update_manifest("browser_audio", safe)
            if self.demo_capture is not None:
                self.demo_capture.input_settings(safe)

    def _guard_reset(self):
        for task in self._guard_tasks:
            task.cancel()
        self._guard_input = None
        self._guard_preroll.clear()
        self._guard_wait_audio = []
        self._guard_wait_end = None
        self._guard_wait_reason = None
        self._guard_outputs.clear()
        self._guard_echo = EchoEvidence()
        self._guard_evidence = {}
        self._guard_last_evidence_t = -1.
