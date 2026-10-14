"""Demo-only input admission and independent floor/response ownership.

Raw VAD does NOT advance the accepted-input epoch. Every callback is fenced by
session, input id and revision; the actor remains the only state writer.
"""
import asyncio
import json
import math
import statistics
import time
from collections import deque
from contextlib import aclosing
from dataclasses import dataclass, field

import numpy as np

import hold_tiers
from control_labels import (decide_input_route, route_protocol, parse_route, RouteScanner,
                            ROUTE_MAX_OUTPUT, REPLY_PROTOCOL)
from input_audio import EchoEvidence, INPUT_PROTOCOL
from messages import build_audio_content
from speech_reference import REFERENCE_CHARS, SpeechReference, completed_context, looks_like_question


VAD_WINDOW_MS = 32  # Silero 16 kHz window: 512 samples per VADIterator call


def _vad_silence(config, key):
    """Silero END needs (1 + ceil(ms/32)) low windows; only window multiples are distinct.
    Absent keys keep the evaluated legacy 100 ms (4 windows, 160 ms effective)."""
    if key not in config:
        return 100
    value = config[key]
    if type(value) is not int or not (value == 100 or (
            32 <= value <= 480 and value % VAD_WINDOW_MS == 0)):
        raise ValueError(f"{key} must be a multiple of 32 in [32, 480] (or the legacy 100)")
    return value


def vad_end_windows(silence_ms):
    """Low windows counted after the first low window before Silero reports END."""
    return math.ceil(silence_ms / VAD_WINDOW_MS)


def vad_effective_end_ms(silence_ms):
    """Trailing silence from the first low window to END: (1 + windows) x 32 ms."""
    return VAD_WINDOW_MS * (1 + vad_end_windows(silence_ms))


def input_timing(config=None):
    """Pre-roll and (guarded only) per-state VAD end silence; threshold stays 0.5.

    `vad_silence_ms` applies while listening and before playback starts,
    `vad_silence_ms_playing` while the reply is audibly playing (see
    GuardedTurns._guard_vad_state). Callers pass {} when not guarded, so the
    HumDial/legacy baseline always gets 100 ms.
    """
    config = config or {}
    preroll = config.get("input_preroll_ms", 160)
    if type(preroll) is not int or not 0 <= preroll <= 960 or preroll % 16:
        raise ValueError("input_preroll_ms must be a multiple of 16 in [0, 960]")
    legacy = config.get("input_vad_end_ms_playing", 0)
    if type(legacy) is not int or not (legacy == 0 or 200 <= legacy <= 1000):
        raise ValueError("input_vad_end_ms_playing must be 0 or an integer in [200, 1000]")
    if legacy and any(config.get(k, 100) != 100 for k in
                      ("input_vad_silence_ms", "input_vad_silence_ms_playing")):
        raise ValueError("legacy f2 and Silero silence tuning cannot both be enabled")
    silence = _vad_silence(config, "input_vad_silence_ms")
    playing = _vad_silence(config, "input_vad_silence_ms_playing")
    return {"vad_threshold": 0.5, "vad_silence_ms": silence, "preroll_ms": preroll,
            "vad_end_windows": vad_end_windows(silence),
            "vad_effective_end_ms": vad_effective_end_ms(silence),
            "vad_silence_ms_playing": playing,
            "vad_end_windows_playing": vad_end_windows(playing),
            "vad_effective_end_ms_playing": vad_effective_end_ms(playing),
            **({"vad_decision_end_ms_playing": legacy} if legacy else {})}


CONTINUATION_PROTOCOL = "pending-user-audio-v1"
# The only prompt line input_route_compact_prompt replaces (output format, not semantics).
ROUTE_FORMAT_LINE = "只输出一个JSON对象，必须先输出transcript字段，再输出label字段，不要其他字段或解释。"


def route_options(engine_cfg=None):
    """transcript-first-v2 switches (prefill, compact prompt, early close); all default off."""
    cfg = engine_cfg or {}
    # All three default OFF (baseline-identical). Early close is NOT decision-identical
    # to the full parse: it cannot see output after the label, so e.g.
    # `..."label":"keep","x":1}` or `...}` + trailing text (rejected -> repair by the
    # full parse) is accepted by the head parse. Enable it explicitly.
    options = {"prefill": cfg.get("input_route_prefill", False),
               "compact_prompt": cfg.get("input_route_compact_prompt", False),
               "early_close": cfg.get("input_route_early_close", False)}
    for key, value in options.items():
        if type(value) is not bool:
            raise ValueError(f"input_route_{key} must be a boolean")
    return options


def route_prompt(prompts, engine_cfg=None):
    """The input_route system prompt, optionally with the compact one-line format rule."""
    prompt = prompts["input_route"]
    if not route_options(engine_cfg)["compact_prompt"]:
        return prompt
    compact = (prompts.get("input_route_compact_format") or "").strip()
    if not compact or "\n" in compact or prompt.count(ROUTE_FORMAT_LINE) != 1:
        raise ValueError("input_route_compact_prompt needs prompts.input_route_compact_format "
                         "(one line) and exactly one original format line in input_route")
    return prompt.replace(ROUTE_FORMAT_LINE, compact)
# Before any unit has fully played, reply review may use the units already heard;
# the newest one must have played at least this long to count as heard.
REPLY_MIN_HEARD_S = 0.5

# Every reason the guarded path can wait on, and what ends that wait besides the
# next user input. Only _guard_set_wait/_guard_clear_wait write the reason.
#   none    - deliberately silent until the user speaks again (rationale below)
#   release - after input_wait_release_s of user silence, answer the held input
#   recover - after the same silence, ask once for a repeat (opt-in, see below)
WAIT_POLICIES = {
    # The admitted input is still open; its own closing decision always follows.
    "user_speaking": "none",
    # The user asked the assistant to stop; speaking again would defy that.
    "stop_only": "none",
    # The user was already told (input_notice) to split an over-long input.
    "input_limit": "none",
    # yield_wait: the user paused mid-utterance; answer what was heard.
    "awaiting_user": "release",
    # Playback was cut, then the input was kept. Recovery only applies when nothing
    # usable was heard; heard content (backchannel, third party) stays silent.
    "uncertain_after_yield": "recover",
}
# Fixed repeat request; spoken verbatim, never sent to a model or user history.
RECOVER_TEXT = {"zh": "抱歉，我刚才没听清，请再说一遍。",
                "en": "Sorry, I didn't catch that. Could you say it again?"}


def _unheard_cause(audit):
    """Why a kept decision carried nothing usable, or None when content was heard."""
    route = audit.get("route") or {}
    if audit.get("acoustic_reject"):
        return "acoustic_reject"
    if audit.get("error"):
        return "error"
    if route.get("fallback"):
        return "timeout" if route.get("timed_out") else "fallback"
    if not str(route.get("transcript") or "").strip():
        return "empty_transcript"
    return None


def route_messages(prompt, content, *, playing=False, reference="", continuation=None,
                   context_content=None, conversation=None):
    # Keep the keyword for existing callers, but never send the speech reference: the
    # transcriber can copy it into fictitious microphone speech. Acoustic
    # reference filtering and playback-aligned diagnostic snapshots are intact.
    # `conversation` is labelled addressee context only (latest answered request and
    # assistant speech the listener already heard); the prompt forbids transcribing it.
    metadata = {"assistant_playing": bool(playing)}
    if conversation:
        metadata["conversation"] = conversation
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
    # Any unclosed STOP after admission; sticky, so recovery never follows a STOP.
    stop_requested: bool = False
    reference_text: str = ""
    reference_sid: int = None
    reference_kind: str = "unavailable"
    reference_played: int = 0
    end_index: int = None
    preroll_samples: int = 0
    reply_context: str = ""
    reply_played_samples: int = 0
    reply_context_kind: str = None
    # Opt-in reply-review view: reply_context plus the heard clauses of the unit still
    # playing at onset. Empty when off or no partial; addressee context never uses it.
    reply_review_context: str = ""
    reply_partial_chars: int = 0
    reply_history_prefix: str = ""
    # Immutable user audio frozen at input onset; route context only, not a
    # second copy in ASR/response history or an assistant-text reference.
    replied_user_audio: object = None
    # Route context of the latest dispatch: "accepted_unanswered_user_audio" means
    # the route transcript already covers every merged prefix (hold-tier features).
    context_source: str = None
    # A Silero end heard during playback cannot take effect before hold_until:
    # its decisions wait in `deferred`, and speech resuming inside the window
    # (e.g. a browser-AEC hole) reopens this same input instead.
    hold_until: float = None
    deferred: list = field(default_factory=list)
    end_silence_ms: int = None
    vad_merges: int = 0


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
    elapsed_ms: float = None  # wall time from dispatch to decision (R), diagnostics only


class GuardedTurns:
    def _init_guarded(self):
        self.GUARDED_TURNS = bool(self.engine_cfg.get("guarded_turns", False) and self.CANDIDATE_TURNS)
        self.INPUT_REFERENCE = self.engine_cfg.get("input_protocol") == INPUT_PROTOCOL
        self._input_serial = 0
        self._guard_tasks = []
        self._guard_input = None
        self.input_timing = input_timing(self.engine_cfg if self.GUARDED_TURNS else {})
        self._guard_preroll = deque(maxlen=self.input_timing["preroll_ms"] // 16)
        self._guard_vad_ms = self.input_timing["vad_silence_ms"]
        self._guard_vad_mode = "listening"
        self._guard_last_end = None
        # Per-candidate END hold (hold-tier-v1). Without end_hold_tiers every tier is
        # END_HOLD; without hold_tier_rules every candidate keeps the default tier.
        self.HOLD_TIERS = hold_tiers.hold_config(self.engine_cfg if self.GUARDED_TURNS else {},
                                                 self.END_HOLD)
        # Transcripts of the audio held in _guard_wait_audio, in the same order.
        self._guard_wait_text = ""
        self._client_rtts = deque(maxlen=32)
        # Legacy f2 delays decisions; do not also lengthen Silero itself.
        legacy_ms = self.engine_cfg.get("input_vad_end_ms_playing", 0) if self.GUARDED_TURNS else 0
        self._guard_end_hold = (legacy_ms - 100) / 1000 if legacy_ms else 0.
        self._guard_wait_audio = []
        self._guard_wait_end = None
        self.CONTINUATION_CONTEXT = bool(self.GUARDED_TURNS and self.engine_cfg.get("input_continuation_context", False))
        self._guard_context_gap = float(self.engine_cfg.get("input_context_max_gap_s", 20))
        if not 1 <= self._guard_context_gap <= 30:
            raise ValueError("input_context_max_gap_s must be in [1, 30]")
        self._guard_wait_reason = None
        # After yield_wait, answer the held input once the user has been silent this long
        # (seconds of audio clock); 0 keeps the old wait-indefinitely behaviour.
        self._guard_wait_release = float(self.engine_cfg.get("input_wait_release_s", 0) or 0)
        if self._guard_wait_release and not 0.5 <= self._guard_wait_release <= 10:
            raise ValueError("input_wait_release_s must be 0 or in [0.5, 10]")
        # Opt-in: after a cut playback whose kept input carried nothing usable, ask for
        # a repeat once the user has been silent input_wait_release_s.
        self._guard_recover = bool(self.engine_cfg.get("input_wait_uncertain_recover", False))
        self._guard_recover_max = self.engine_cfg.get("input_recover_max", 1)
        if type(self._guard_recover_max) is not int or not 0 <= self._guard_recover_max <= 3:
            raise ValueError("input_recover_max must be an integer in [0, 3]")
        if self._guard_recover and not self._guard_wait_release:
            raise ValueError("input_wait_uncertain_recover requires input_wait_release_s")
        self._guard_recover_cause = self._guard_recover_since = None
        self._guard_recover_streak = 0
        self._guard_last_transcript = ""
        # Reply review may also see the heard part of the unit still playing at onset
        # once it has played this long (seconds); 0 keeps completed sentences only.
        self._guard_partial_heard = float(self.engine_cfg.get("input_reply_partial_heard_s", 0) or 0)
        if self._guard_partial_heard and not 0.25 <= self._guard_partial_heard <= 2:
            raise ValueError("input_reply_partial_heard_s must be 0 or in [0.25, 2]")
        self._guard_partial_history = self.engine_cfg.get("input_reply_partial_heard_history", False)
        if type(self._guard_partial_history) is not bool:
            raise ValueError("input_reply_partial_heard_history must be a boolean")
        # Only admit the heard prefix when it reads as a question to the user, so an
        # announcement or explanation in progress does not invite a stop.
        self._guard_partial_questions = self.engine_cfg.get("input_reply_partial_heard_questions_only", False)
        if type(self._guard_partial_questions) is not bool:
            raise ValueError("input_reply_partial_heard_questions_only must be a boolean")
        self._guard_wait_parent = None
        self._guard_outputs = {}
        self._guard_echo = EchoEvidence()
        self._guard_evidence = {}
        self._guard_last_evidence_t = -1.
        self._guard_max_samples = int(float(self.engine_cfg.get("max_input_seconds", 20)) * 16000)
        if not 16000 <= self._guard_max_samples <= 16000 * 30:
            raise ValueError("max_input_seconds must be in [1, 30]")
        if self.GUARDED_TURNS and not self.prompts.get("input_route"):
            raise ValueError("guarded_turns requires an independent input_route prompt")
        self.ROUTE_OPTIONS = route_options(self.engine_cfg if self.GUARDED_TURNS else {})
        if self.GUARDED_TURNS:
            route_prompt(self.prompts, self.engine_cfg)  # fail at startup, not on first input

    def _guard_vad_state(self):
        """Which VAD end silence applies to the next frame (actor state, not span mode).

        - "playing": a published reply whose playback start the client acknowledged
          (`_speech is not None and _speech_started`; `_candidate_playback_started`
          sets it on playback_progress started/played>0) until the reply finishes
          (`_guard_finish_turn`) or is cancelled (`_cancel_speech` clears `_speech`).
        - "preplay" (before playback): nothing audible yet, but a reply exists: a
          private or held candidate (`_candidate is not None`, from route decision to
          playback start) or a published reply still inside the client startup window
          (`_speech is not None and not _speech_started`).
        - "listening": neither.
        Listening and preplay use input_vad_silence_ms; playing uses
        input_vad_silence_ms_playing (browser AEC can carve holes in user speech).
        """
        if self._speech is not None and self._speech_started:
            return "playing"
        if self._candidate is not None or self._speech is not None:
            return "preplay"
        return "listening"

    def _guard_vad_tune(self):
        """Set the Silero end silence for this frame. utils_vad reads
        min_silence_samples on every call, so a change applies to this frame; it is
        only written when it differs (the 100/100 default never touches it)."""
        mode = self._guard_vad_state()
        timing = self.input_timing
        ms = timing["vad_silence_ms_playing" if mode == "playing" else "vad_silence_ms"]
        self._guard_vad_ms, self._guard_vad_mode = ms, mode
        iterator = self.vad_iterator
        samples = 16000 * ms / 1000
        if hasattr(iterator, "min_silence_samples") and iterator.min_silence_samples != samples:
            iterator.min_silence_samples = samples

    def _guard_rtt_median(self):
        return statistics.median(self._client_rtts) if self._client_rtts else None

    def _guard_client_telemetry(self, clean):
        """Actor-side context for client telemetry (diagnostics only, never a decision input)."""
        if not clean:
            return
        kind = clean.get("kind")
        if kind == "rtt" and type(clean.get("rtt_ms")) in (int, float):
            self._client_rtts.append(float(clean["rtt_ms"]))
        elif kind == "client_hold":
            c, speech = self._candidate, self._speech
            self._observe("client_hold_actor", {"action": clean.get("action"),
                "utterance_id": clean.get("utterance_id"),
                "speech_utterance_id": speech.sid if speech is not None else None,
                "speech_started": bool(self._speech_started) if speech is not None else None,
                "candidate_id": c.cid if c is not None else None,
                "candidate_published": c.published if c is not None else None,
                "after_anchor_ms": round((self.t_audio - c.anchor) * 1000, 1) if c is not None else None,
                "hold_s": c.hold_s if c is not None else None,
                "hold_tier": c.hold_tier if c is not None else None,
                "vad_state": self._guard_vad_state()})

    def _guard_assign_hold(self, c, *, text, label, playing, transcript_at=None):
        """Classify one new candidate's END hold (it starts at default; see hold_tiers)."""
        cfg = self.HOLD_TIERS
        c.hold_text = text or ""
        c.hold_transcript_ms = (round((transcript_at - c.anchor) * 1000, 1)
                                if transcript_at is not None else None)
        if not cfg["evaluate"]:
            return
        users = getattr(self, "_users_by_turn", {})
        assistants = getattr(self, "_assistants_by_turn", {})
        tier, reason = hold_tiers.classify(c.hold_text, duration_s=len(c.audio) / 16000,
            label=label, playing=playing, first_turn=not (users or assistants),
            rtt_ms=self._guard_rtt_median(), min_rtt_ms=cfg["min_rtt_ms"],
            low_max_s=cfg["low_max_s"], not_chinese_tier=cfg["not_chinese_tier"])
        tier, reason, hold_s = hold_tiers.resolve(cfg, tier, reason,
                                                  elapsed_s=self.t_audio - c.anchor)
        c.hold_rule = {"tier": tier, "reason": reason, "hold_s": hold_s}
        if cfg["apply"]:
            c.hold_s, c.hold_tier, c.hold_reason = hold_s, tier, reason
        else:
            c.hold_reason = "shadow"

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

    def _guard_record(self, sid):
        meta = self._speech_meta
        record = self._guard_outputs.setdefault(sid, {"turn": meta.turn, "sentences": []})
        if meta.kind == "recover":
            record["no_history"] = True
        return record

    def _guard_reference_event(self, ev):
        # Called AFTER the engine's private-candidate and stale-utterance gates.
        # Do not update ModelDone.text: that would commit partial history on ACK.
        record = self._guard_record(ev.sid)
        if "reference" not in record:
            record["reference"] = SpeechReference()
        record["reference"].observe(ev.kind, ev.data)

    def _guard_sentence_end(self, sid, data):
        record = self._guard_record(sid)
        end, text = data["end_sample"], data["text"]
        if not data.get("reply_complete", True):
            if "reply_sentences" not in record:
                record["reply_sentences"] = list(record["sentences"])
            record["reply_pending"] = record.get("reply_pending", "") + text
        elif "reply_sentences" in record:
            record["reply_sentences"].append((end, record.pop("reply_pending", "") + text))
        # Actual completed TTS units still delimit played history; generated
        # future text never enters either history or the completed reply view.
        record["sentences"].append((end, text))

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
                and held.stage != "recover"
                and (self._speech is None or not self._speech_started)):
            prefix = [held.audio, np.zeros(1600, dtype=np.float32)]
        return prefix + self._guard_wait_audio

    async def _guard_frame(self, ev, event):
        frame, t = ev.pcm, ev.t_audio
        span = self._guard_input
        policy = WAIT_POLICIES.get(self._guard_wait_reason)
        if (policy == "release" and self._guard_wait_release
                and self._guard_wait_audio and self._guard_wait_end is not None
                and not (event and "start" in event) and (span is None or span.decided)
                and self._candidate is None and t - self._guard_wait_end >= self._guard_wait_release):
            await self._guard_release_wait(t)
        elif policy == "recover" and self._guard_recover_since is not None:
            if self._candidate is not None or self._speech is not None:
                # Another answer owns the floor now; never ask after it.
                self._guard_recover_cause = self._guard_recover_since = None
            elif (not (event and "start" in event) and (span is None or span.decided)
                    and t - self._guard_recover_since >= self._guard_wait_release):
                await self._guard_recover_wait(t)
        if event and "start" in event:
            if span is not None and not span.decided:
                if span.hold_until is not None:
                    # A start on the frame that ends the window still wins.
                    if span.deferred or t <= span.hold_until + 1e-6:
                        span.vad_merges += 1
                        self._observe("input_vad_end_merged", {"input_id": span.sid,
                            "revision": span.revision, "gap_s": round(t - span.end, 3),
                            "discarded_routes": [d.route for d in span.deferred]})
                    span.hold_until, span.deferred = None, []
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
                    self._guard_wait_text = ""
                    self._guard_wait_end = None
                    self._guard_clear_wait()
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
                        span.reply_context = completed_context(record.get("reply_sentences", record.get("sentences", [])),
                                                               span.reply_played_samples)
                        rate = getattr(self._speech, "rate", None) or 24000
                        if span.reply_context:
                            span.reply_context_kind = "completed_sentences"
                            if self._guard_partial_heard and record.get("reference") is not None:
                                # A follow-up such as "why?" may target the sentence still
                                # playing: append only its clauses already played at this
                                # onset, after (never displacing) the completed sentences.
                                partial = record["reference"].heard_prefix(
                                    played=span.reply_played_samples, sent=self._speech.sent,
                                    ends=[end for end, _ in record.get("sentences", [])],
                                    min_heard=int(rate * self._guard_partial_heard), rate=rate,
                                    limit=REFERENCE_CHARS - len(span.reply_context))
                                if partial and self._guard_partial_questions and not looks_like_question(partial):
                                    partial = ""
                                if partial:
                                    span.reply_history_prefix = "".join(
                                        text for end, text in record.get("sentences", [])
                                        if end <= span.reply_played_samples) + partial
                                    span.reply_partial_chars = len(partial)
                                    span.reply_review_context = span.reply_context + partial
                                    span.reply_context_kind = "completed_and_playing_prefix"
                        elif record.get("reference") is not None:
                            # Nothing has fully played yet: use only units the user has
                            # already heard, frozen at this onset (never a unit that
                            # has not started, nor one heard for under REPLY_MIN_HEARD_S).
                            span.reply_context = record["reference"].heard_window(
                                played=span.reply_played_samples, sent=self._speech.sent,
                                min_heard=int(rate * REPLY_MIN_HEARD_S))
                            if span.reply_context:
                                span.reply_context_kind = "sentence_window"
                    c = self._speech_candidate
                    if (self.CONTINUATION_CONTEXT and c is not None and c.stage != "recover"
                            and 0 <= t - c.anchor <= self._guard_context_gap):
                        span.replied_user_audio = c.audio
                elif mode == "preplay" and self.CONTINUATION_CONTEXT and held.stage != "recover":
                    # If playback wins an in-flight admission race, the same
                    # onset-frozen USER input is still available to re-route.
                    span.replied_user_audio = held.audio
                self._guard_input = span
                await self._guard_hold(held, True)
            self.IN_SPEECH = True
            held = self._candidate
            await self.send_control("vad_start", {"turn": self.TURN_IDX, "state": self.STATE,
                "input_id": span.sid, "timestamp": self._wall_ts(),
                # g: onset minus the previous VAD END (audio clock), for counterfactual holds.
                "prev_end_gap_ms": (round((t - self._guard_last_end) * 1000, 1)
                                    if self._guard_last_end is not None else None),
                "vad_state": self._guard_vad_mode,
                "held_candidate_id": held.cid if held is not None else None,
                "held_candidate_published": held.published if held is not None else None,
                "held_after_anchor_ms": (round((t - held.anchor) * 1000, 1)
                                         if held is not None else None),
                "held_hold_s": held.hold_s if held is not None else None})
        elif span is not None and not span.decided:
            span.frames.append(frame.copy())
        self._guard_preroll.append(frame.copy())
        if span is not None and not span.decided:
            span.frame_count += 1
            span.echo_frames += bool(self._guard_evidence.get("echo_only"))
            context_size = sum(map(len, self._guard_pending_audio(span)))
            # Silence after a held playback end joins the input only if speech resumes.
            frames = span.frames[:span.end_index] if span.hold_until is not None else span.frames
            if context_size + sum(map(len, frames)) > self._guard_max_samples:
                if self.CONTINUATION_CONTEXT:
                    self._guard_wait_audio = []
                    self._guard_wait_text = ""
                    self._guard_wait_end = None
                    self._guard_set_wait("input_limit")
                await self._guard_reject(span, "input_limit")
                await self.send_control("input_notice", {"message": "这段语音过长，请分段说。"})
            elif event and "end" in event:
                span.voice, span.end = False, t
                span.end_index = len(span.frames)
                span.revision += 1
                if span.task is not None:
                    span.task.cancel()
                self._guard_last_end = t
                if self._guard_end_hold:
                    # Chosen once, here: only audible assistant audio (browser AEC
                    # can carve holes into speech) lengthens the effective end.
                    playing = self._speech is not None and self._speech_started
                    span.hold_until = t + self._guard_end_hold if playing else None
                    span.end_silence_ms = (self.engine_cfg["input_vad_end_ms_playing"]
                                           if playing else self.input_timing["vad_silence_ms"])
                await self.send_control("vad_done", {"turn": self.TURN_IDX, "state": self.STATE,
                    "input_id": span.sid, "timestamp": self._wall_ts(),
                    "vad_state": self._guard_vad_mode, "vad_silence_ms": self._guard_vad_ms,
                    "vad_effective_end_ms": vad_effective_end_ms(self._guard_vad_ms)})
                self._guard_dispatch(span, closed=True)
            elif span.hold_until is not None and t >= span.hold_until - 1e-6:
                # The window passed without resumed speech: apply what was decided.
                await self._guard_close_end_window(span)
            elif span.voice and (span.task is None or span.task.done()):
                # Early preplay verification prevents a noise blip from destroying
                # a valid candidate. Long input triggers a decision, never a stop.
                interval = .192 if span.mode == "preplay" and not span.checked_until else 1.5
                if t - max(span.start, span.checked_until) >= interval:
                    self._guard_dispatch(span, closed=False)
        c = self._candidate
        if c is not None and not c.confirmed and self._guard_can_publish(c) and t - c.anchor >= c.hold_s:
            self._judged_seg_end = t
            data = {"turn": self.TURN_IDX, "state": self.STATE,
                "candidate_id": c.cid, "parent_id": c.parent_id,
                "timestamp": self._wall_ts(),
                "hold_s": c.hold_s, "hold_tier": c.hold_tier, "hold_reason": c.hold_reason,
                "hold_elapsed_ms": round((t - c.anchor) * 1000, 1),
                "transcript_after_anchor_ms": c.hold_transcript_ms,
                "first_audio_ms": c.first_audio_ms}
            if c.hold_rule is not None:
                data.update(hold_rule_tier=c.hold_rule["tier"], hold_rule_reason=c.hold_rule["reason"],
                            hold_rule_s=c.hold_rule["hold_s"])
            await self.send_control("vad_640_done", data)
            await self._confirm_candidate()

    async def _guard_close_end_window(self, span):
        deferred, span.hold_until, span.deferred = span.deferred, None, []
        for decision in deferred:
            await self._on_input_decision(decision)

    def _guard_held_until(self):
        """Audio time at which held decisions take effect; None when none wait."""
        span = self._guard_input
        if span is None or span.decided or not span.deferred:
            return None
        return span.hold_until

    def _guard_conversation(self, span, playing):
        """Addressee context for input_route: the latest answered user request and only
        assistant speech the listener has heard. While playing that is the onset-frozen
        played window; otherwise the ACK-based history, which never holds unplayed text."""
        if not self.engine_cfg.get("input_route_conversation", False):
            return None
        users = getattr(self, "_users_by_turn", {})
        assistants = getattr(self, "_assistants_by_turn", {})
        previous = users[max(users)] if users else ""
        if playing:
            heard = span.reply_context
        else:
            answered = sorted(users.keys() & assistants.keys())
            heard = assistants[answered[-1]] if answered else ""
        if not previous and not heard:
            return None
        return {"previous_user_request": previous[-300:], "assistant_speech_heard": heard[-400:]}

    def _guard_dispatch(self, span, *, closed):
        dispatched = time.perf_counter()
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
        span.context_source = context_source
        # During playback, a fragment such as "of the refugee crisis" must not
        # lose the recent question it completes. This is route-only context:
        # normal response history and new-user ASR retain their existing inputs.
        # Optional played context cannot consume the new input's audio budget.
        previous = span.replied_user_audio
        if (not context_audio and previous is not None
                and len(previous) + 1600 + current_samples <= self._guard_max_samples):
            context_audio = [previous, np.zeros(1600, dtype=np.float32)]
            context_source = "recent_replied_user_audio"
            span.context_source = context_source
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
        conversation = self._guard_conversation(span, playing)
        options = self.ROUTE_OPTIONS
        prefill = options["prefill"]
        reply_context = span.reply_review_context or span.reply_context
        messages = route_messages(route_prompt(self.prompts, self.engine_cfg), content,
                                  playing=playing, reference=span.reference_text,
                                  continuation=continuation, context_content=context_content,
                                  conversation=conversation)
        self._observe("input_dispatch", {"input_id": span.sid, "revision": revision,
            "closed": closed, "playing": playing, "audio_samples": current_samples + context_samples,
            "current_audio_samples": current_samples, "context_audio_samples": context_samples,
            "context_source": context_source,
            "continuation_protocol": CONTINUATION_PROTOCOL if continuation else None,
            "parent_id": operation_id,
            "reference_kind": span.reference_kind, "reference_utterance_id": span.reference_sid,
            "reference_chars": len(span.reference_text), "reference_played_samples": span.reference_played,
            "reply_context_chars": len(reply_context), "reply_played_samples": span.reply_played_samples,
            "reply_context_kind": span.reply_context_kind,
            **({"vad_end_silence_ms": span.end_silence_ms if closed else None,
                "vad_end_merges": span.vad_merges} if self._guard_end_hold else {}),
            **({"reply_partial_chars": span.reply_partial_chars} if self._guard_partial_heard else {})})
        reference_context = {"reference_kind": span.reference_kind,
            "reference_utterance_id": span.reference_sid,
            "reference_played_samples": span.reference_played,
            "route_protocol": route_protocol(prefill), "reference_sent_to_model": False,
            "input_timing": self.input_timing, "preroll_samples": span.preroll_samples,
            "route_penalties": {"presence": 0.0, "frequency": 0.0},
            "conversation_chars": sum(map(len, (conversation or {}).values()))}
        reference_context.update(reply_protocol=REPLY_PROTOCOL,
            reply_context_chars=len(reply_context), reply_played_samples=span.reply_played_samples,
            reply_context_kind=span.reply_context_kind)
        if self._guard_partial_heard:
            reference_context["reply_partial_chars"] = span.reply_partial_chars
        reference_context.update(continuation_protocol=CONTINUATION_PROTOCOL if continuation else None,
            context_audio_samples=context_samples, current_audio_samples=current_samples,
            context_source=context_source)

        async def call(msgs, stage):
            parts = []
            call_id = self._diagnostic_id("call")
            call_ids.append(call_id)
            request_kind = "interrupt" if playing else "spec_judge"
            if self.CHAT_DEMO and self.engine_cfg.get("input_control_reserve", False):
                request_kind = "input_reply" if stage == "input_reply" else "input_route"
            is_route = stage != "input_reply"
            response_options = {}
            if self.CHAT_DEMO and self.engine_cfg.get("input_route_token_budget", False) and is_route:
                from control_labels import input_route_token_budget
                response_options["max_tokens"] = input_route_token_budget(current_samples + context_samples)
            if prefill and is_route:
                # decide_control appended the v2 prefix as the last (assistant) message.
                response_options["continue_final_message"] = True
            # Label-close scan on every route call (its timing is recorded even with
            # early close off). Only a full parse_route of the completed head decides.
            scanner = RouteScanner(prefilled=prefill) if is_route else None
            stats = {"stage": stage, "call_id": call_id, "output_chars": 0,
                     "label_closed_ms": None, "early_closed": False} if is_route else None
            stream_state = {}
            result = None
            if stats is not None:
                route_calls.append(stats)
            async with aclosing(self._capacity_stream(request_kind, self.text_stream_fn, msgs, route=True,
                    response_options=response_options or None, stream_state=stream_state,
                    case_context={"input_id": span.sid, "revision": revision, "closed": closed,
                                  "input_generation": gen, "playing": playing,
                                  **reference_context, "decision_stage": stage,
                                  "reference_sent_to_model": stage == "input_reply"},
                                  call_id=call_id, parent_id=operation_id)) as source:
                async for part in source:
                    parts.append(part)
                    if stats is not None:
                        stats["output_chars"] += len(part)
                    if sum(map(len, parts)) > (ROUTE_MAX_OUTPUT if is_route else 128):
                        raise ValueError("Oversized input decision")
                    if scanner is not None and scanner.feed(part):
                        stats["label_closed_ms"] = round((time.perf_counter() - dispatched) * 1000, 3)
                        if options["early_close"]:
                            candidate = scanner.candidate()
                            if parse_route(candidate, prefilled=prefill) is not None:
                                # Final decision; closing the stream cancels the server request.
                                # Output after the label is audit-only.
                                stats.update(early_closed=True, trailing=scanner.trailing()[:128])
                                stream_state["early_closed"] = True
                                result = candidate
                                break
            return "".join(parts) if result is None else result

        call_ids = []
        route_calls = []

        async def run():
            # One typed decision owns admission, stopping and readiness. The
            # original HumDial binary classifier remains the flag-off baseline,
            # not a second veto: an AND gate compounds its false negatives.
            def stop_first(label):
                # A STOP must not wait for the reply review. Cancel playback now via
                # an interim (unclosed) decision; the final decision then either keeps
                # the stop or, if the review asks for a reply, answers from this input.
                if label == "stop_only":
                    self.q.put_nowait(InputDecision(gen, span.sid, revision, False, "stop_only",
                        {"route": {"interim_stop": True, "base_label": label}, "playing": playing},
                        list(call_ids), operation_id))

            route, audit = await decide_input_route(call, messages,
                min(self.DECISION_TIMEOUT, float(self.engine_cfg.get("input_decision_timeout_s", 2))),
                playing=playing, closed=closed, reply_prompt=self.prompts.get("input_reply"),
                reply_context=reply_context, on_review=stop_first, route_prefill=prefill)
            if route_calls:
                last = route_calls[-1]
                audit.update(output_chars=last["output_chars"], label_closed_ms=last["label_closed_ms"],
                             early_closed=last["early_closed"], streams=[dict(c) for c in route_calls])
                if "trailing" in last:
                    audit["trailing"] = last["trailing"]
            if self._guard_partial_heard and "reply_review" in audit:
                # Count reviews (and replies) that saw a partial unit the answer's
                # history never holds: history keeps only completed sentences.
                audit["reply_review"]["partial_chars"] = span.reply_partial_chars
            return InputDecision(gen, span.sid, revision, closed, route,
                                 {"route": audit, "playing": playing},
                                 list(call_ids), operation_id,
                                 elapsed_ms=round((time.perf_counter() - dispatched) * 1000, 3))

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

    async def _guard_reject(self, span, reason, audit=None):
        span.decided, span.route = True, "keep"
        self.IN_SPEECH = False
        if self._guard_recover_since is not None:
            skipped = ("input_limit" if reason == "input_limit" else
                       "heard_content" if audit is not None and _unheard_cause(audit) is None else None)
            if skipped:
                # The user was heard (or already told to split the input): never follow
                # that with a repeat request.
                self._observe("input_cut_recovery_skipped", {"reason": skipped,
                    "cause": self._guard_recover_cause, "streak": self._guard_recover_streak})
                self._guard_recover_cause = self._guard_recover_since = None
            else:
                # Nothing usable again: the rejection only paused it; recount from its end.
                self._guard_recover_since = span.end if span.end is not None else self.t_audio
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
            self._observe("input_decision_stale", {"input_id": ev.sid, "revision": ev.revision,
                "parent_id": ev.parent_id, "closed": ev.closed, "input_generation": ev.gen})
            return
        if span.hold_until is not None and self.t_audio < span.hold_until - 1e-6:
            # Accounted above; it is applied (or dropped by resumed speech) later.
            ev.audit.pop("accounted", None)
            ev.audit["deferred_from"] = round(self.t_audio, 3)
            span.deferred.append(ev)
            self._observe("input_decision_deferred", {"input_id": ev.sid, "revision": ev.revision,
                "closed": ev.closed, "route": ev.route, "until": round(span.hold_until, 3)})
            return
        route_audit = ev.audit.get("route") or {}
        self._observe("input_decision", {"input_id": ev.sid, "revision": ev.revision,
            "closed": ev.closed, "input_generation": ev.gen,
            "route": ev.route, "audit": ev.audit, "call_ids": ev.call_ids,
            "parent_id": ev.parent_id,
            "route_output_chars": route_audit.get("output_chars"),
            "route_label_closed_ms": route_audit.get("label_closed_ms"),
            "early_closed": bool(route_audit.get("early_closed")),
            # R (dispatch -> decision, wall) and decision arrival after VAD END (audio clock).
            "route_ms": ev.elapsed_ms,
            "decision_after_end_ms": (round((self.t_audio - span.end) * 1000, 1)
                                      if ev.closed and span.end is not None else None)})
        transcript = (ev.audit.get("route") or {}).get("transcript")
        if (isinstance(transcript, str) and transcript.strip()
                and (span.admitted or ev.route != "keep")):
            # Only speech addressed to the assistant picks the recovery language;
            # echoes and third-party speech are kept and never admitted.
            self._guard_last_transcript = transcript
        if ev.route == "keep":
            if ev.closed:
                if span.admitted:
                    kept = span.frames[:span.end_index]
                    if sum(map(len, self._guard_wait_audio + kept)) + 1600 <= self._guard_max_samples:
                        self._guard_wait_audio += [np.concatenate(kept), np.zeros(1600, dtype=np.float32)]
                        self._guard_wait_text = self._guard_merged_text(span, route_audit)
                        self._guard_wait_end = span.end
                    else:
                        self._guard_wait_audio = []
                        self._guard_wait_text = ""
                        self._guard_wait_end = None
                    self._guard_set_wait("uncertain_after_yield")
                    if self._guard_recover:
                        self._guard_arm_recovery(ev.audit, span.end)
                await self._guard_reject(span, "insufficient_evidence", ev.audit)
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
            if (old_candidate is not None and old_candidate.stage != "recover"
                    and (self._speech is None or not self._speech_started) and ev.route != "stop_only"):
                prefix = [old_candidate.audio, np.zeros(1600, dtype=np.float32)]
                self._guard_wait_text = old_candidate.hold_text + self._guard_wait_text
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
            span.stop_requested |= ev.route == "stop_only"
            self._guard_set_wait("user_speaking")
            return
        span.decided, span.route = True, ev.route
        self.IN_SPEECH = False
        if ev.route == "stop_only":
            self._guard_wait_audio = []
            self._guard_wait_text = ""
            self._guard_wait_end = None
            self._guard_set_wait("stop_only")
            span.frames.clear()
            await self.send_control("input_waiting", {"reason": "stop_only", "input_id": span.sid})
            return
        audio_parts = self._guard_wait_audio + span.frames[:span.end_index]
        merged_text = self._guard_merged_text(span, route_audit)
        self._guard_wait_audio = []
        self._guard_wait_text = ""
        self._guard_wait_end = None
        if sum(map(len, audio_parts)) > self._guard_max_samples:
            self._guard_set_wait("input_limit")
            span.frames.clear()
            await self.send_control("input_notice", {"message": "这段语音过长，请分段说。"})
            return
        audio = np.concatenate(audio_parts)
        span.frames.clear()
        if ev.route == "yield_wait":
            self._guard_wait_audio = [audio, np.zeros(1600, dtype=np.float32)]
            self._guard_wait_text = merged_text
            self._guard_wait_end = span.end
            self._guard_set_wait("awaiting_user")
            self._guard_wait_parent = ev.parent_id
            await self.send_control("input_waiting", {"reason": "awaiting_user", "input_id": span.sid})
            return
        self._guard_clear_wait()
        self._guard_commit_partial_history(span)
        self.BUFFER = [audio]
        self.t_end_anchor = span.end
        # input_route already judged the addressee (speech to a third party is keep). The
        # audio-only shift gate never caught a real third party in captured sessions, vetoed
        # accepted follow-ups, and its reply omits the user audio, so it is not run here.
        c = self._begin_candidate(audio, stage="response", parent_id=ev.parent_id)
        # The route transcript arrived with this decision (now). Hold features use the
        # merged full text and the whole merged audio, never the newest segment alone.
        self._guard_assign_hold(c, text=merged_text, label=ev.route,
                                playing=span.mode == "playing" or bool(ev.audit.get("playing")),
                                transcript_at=self.t_audio)

    def _guard_commit_partial_history(self, span):
        """Answer from the same onset-frozen heard clauses that review used.

        Store the full completed prefix, rather than the bounded last-three-sentence
        review window. Pin it in the cancelled record so late STOP ACKs cannot
        replace it with an unplayed full sentence or remove the partial again.
        """
        if not (self._guard_partial_history and self._guard_partial_heard
                and span.reply_partial_chars and span.reply_history_prefix):
            return
        record = self._guard_outputs.get(span.reference_sid)
        if not record or record.get("no_history"):
            return
        record["partial_history_prefix"] = span.reply_history_prefix
        self._assistants_by_turn[record["turn"]] = span.reply_history_prefix
        self._observe("history_write", {"role": "assistant", "turn": record["turn"],
            "text": span.reply_history_prefix, "source": "reply_partial_heard",
            "utterance_id": span.reference_sid, "played_samples": span.reply_played_samples})

    def _guard_merged_text(self, span, route_audit):
        """Full transcript of the audio merged with this span.

        With accepted-unanswered context the route heard (and transcribed) every
        merged prefix; otherwise prepend the stored transcripts of the held audio."""
        transcript = str(route_audit.get("transcript") or "")
        if span.context_source == "accepted_unanswered_user_audio":
            return transcript
        return self._guard_wait_text + transcript

    async def _guard_release_wait(self, t):
        """The user went quiet after an input that looked unfinished. Answer what was
        heard (the reply prompt allows one short clarification) rather than stay silent."""
        audio = np.concatenate(self._guard_wait_audio[:-1] or self._guard_wait_audio)
        silence = t - self._guard_wait_end
        self.t_end_anchor = self._guard_wait_end
        parent = self._guard_wait_parent
        text = self._guard_wait_text
        self._guard_wait_audio, self._guard_wait_end = [], None
        self._guard_wait_text = ""
        self._guard_clear_wait()
        self._guard_wait_parent = None
        self.BUFFER = [audio]
        self._observe("input_wait_released", {"silence_s": round(silence, 3),
            "audio_samples": len(audio), "parent_id": parent})
        c = self._begin_candidate(audio, stage="response", parent_id=parent)
        self._guard_assign_hold(c, text=text, label="wait_release", playing=False)

    def _guard_set_wait(self, reason):
        """The only writer of the wait reason. Any transition cancels a pending recovery;
        the caller arms a new one explicitly."""
        if reason not in WAIT_POLICIES:
            raise ValueError(f"wait reason {reason!r} has no policy")
        self._guard_wait_reason = reason
        self._guard_recover_cause = self._guard_recover_since = None

    def _guard_clear_wait(self):
        self._guard_wait_reason = None
        self._guard_recover_cause = self._guard_recover_since = None

    def _guard_arm_recovery(self, audit, end):
        """Arm recovery only when the kept input carried nothing usable. Heard content
        (a backchannel or speech to someone else) and an explicit STOP stay silent."""
        cause = _unheard_cause(audit)
        span = self._guard_input
        skipped = ("heard_content" if cause is None else
                   "stop_requested" if span is not None and span.stop_requested else
                   "streak_limit" if self._guard_recover_streak >= self._guard_recover_max else None)
        if skipped:
            self._observe("input_cut_recovery_skipped", {"reason": skipped, "cause": cause,
                "streak": self._guard_recover_streak})
            return
        self._guard_recover_cause, self._guard_recover_since = cause, end

    async def _guard_recover_wait(self, t):
        """Playback was cut but nothing usable was heard, and the user stayed quiet.
        Ask once for a repeat with a fixed sentence: no model call, no user history.
        The unusable audio is dropped so the repeat is answered on its own. The streak
        counts only once the request is audible (see _candidate_playback_started)."""
        cause, since = self._guard_recover_cause, self._guard_recover_since
        self._guard_wait_audio, self._guard_wait_end = [], None
        self._guard_wait_text = ""
        self._guard_clear_wait()
        self.t_end_anchor = since
        self._observe("input_cut_recovery", {"reason": cause, "streak": self._guard_recover_streak + 1,
            "silence_s": round(t - since, 3)})
        self._begin_candidate(np.zeros(0, dtype=np.float32), stage="recover",
                              text=RECOVER_TEXT[self._guard_recover_language()])

    def _guard_recover_language(self):
        users = getattr(self, "_users_by_turn", {})
        for text in (self._guard_last_transcript, users[max(users)] if users else ""):
            if any(0x3040 <= ord(ch) <= 0x9fff for ch in text):
                return "zh"
            if any(ch.isascii() and ch.isalpha() for ch in text):
                return "en"
        return "zh"

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
        record = self._guard_record(sid)
        record.update(cancelled=True, sent=self._speech.sent, played=self._speech.played)
        self._guard_history_progress(sid, self._speech.played)

    def _guard_history_progress(self, sid, samples, *, completed=False):
        record = self._guard_outputs.get(sid)
        if record is None:
            return
        record["played"] = max(record.get("played", 0), samples)
        if record.get("cancelled"):
            if record.get("no_history"):
                # Fixed-text speech (a repeat request) is not part of the conversation.
                return
            prefix = record.get("partial_history_prefix", "".join(
                text for end, text in record["sentences"] if end <= record["played"]))
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
        self._guard_wait_text = ""
        self._guard_wait_end = None
        self._guard_clear_wait()
        self._guard_recover_streak = 0
        self._guard_last_transcript = ""
        self._guard_last_end = None
        self._guard_outputs.clear()
        self._guard_echo = EchoEvidence()
        self._guard_evidence = {}
        self._guard_last_evidence_t = -1.
