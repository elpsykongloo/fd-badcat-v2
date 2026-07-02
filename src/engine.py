# -*- coding: utf-8 -*-
"""Actor-model realtime engine (W1 refactor of src/backend.py).

Old architecture (frozen at src/backend_legacy.py):
    single coroutine: receive() -> VAD -> inline `await llm` (perception freezes
    for up to judge+shift+response serial latency; wall-clock interval logic
    drifts against the audio stream while frames pile up in the ws queue).

New architecture (this file):
    websocket -> reader task (receive ONLY) -> asyncio.Queue -> engine loop
                                                     | dispatch (create_task)
                                     LLM / ASR / TTS worker tasks
                                                     +-- results come back as
                                                         events on the same queue

W1 iron laws (AGENTS.md):
  1. Behavior preservation: judge/interrupt/shift prompts, 0.64s end-hold,
     2.5s continue timeout, 1.5s long-interrupt threshold are UNCHANGED.
     Behavioral deltas hide behind default-off flags or are documented as
     known deviations in docs/w1_equivalence.md.
  2. Single writer: every mutable engine field is touched only by the engine
     loop. Workers communicate exclusively via queue events.
  3. Audio clock: all interval logic runs on cumulative-sample time
     (t_audio = received samples / 16000). Wall clock is only used to measure
     model latency and to stamp traces for latency accounting.

Staleness protocol (W1 core):
  - `session_gen` bumps on reset/disconnect: ALL in-flight results are dropped.
  - `seg_epoch` bumps on every vad_start, on turn switches, and on reset:
    judge/shift/interrupt results from an older epoch are dropped (their
    evidence -- "the user finished speaking" -- has been falsified by new
    speech). `continue` results are equally droppable because their semantics
    is "do nothing". response/shift_re are exempt in W1 (legacy semantics:
    an answer in flight completes; flag engine.cancellable_response reserved
    for W3).

Replay modes:
  - realtime: models actually run in worker threads (production path).
  - injected: models never run; a decision script supplies (text, infer) and
    the result is delivered when the audio clock has advanced by `infer`
    -- deterministic, faster than real time.
  - oracle:   like injected but zero latency (pure state-machine testing).
"""
import asyncio
import heapq
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

import module as _module
from messages import build_audio_content, scrub_audio_blocks

SAMPLE_RATE = 16000
WINDOW_SIZE = 256
LONG_INTERRUPT_SEC = 1.5  # legacy hardcoded threshold (backend_legacy.py:381)

# Conservative fallback per decision kind when llm.decision_timeout_s expires.
TIMEOUT_FALLBACK = {
    "judge": "continue",     # wait longer rather than misjudge EoU
    "interrupt": "continue", # do not yield the floor on a hunch
    "shift": "no",           # treat as same-topic (normal answer path)
}
RESPONSE_TIMEOUT_APOLOGY = "抱歉，我刚才没有听清，请再说一遍。"


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
@dataclass
class FrameEvent:
    seq: int
    t_audio: float        # end-of-frame time on the audio clock (cum samples / SR)
    t_wall: float         # perf_counter at reader receive (freeze measurement)
    pcm: np.ndarray


@dataclass
class ModelDone:
    kind: str             # judge|shift|shift_re|interrupt|response|asr|tts
    gen: int              # session_gen at dispatch
    epoch: int            # seg_epoch at dispatch
    turn: int
    text: str = ""
    infer: float = 0.0
    add_to_history: bool = False
    wav_path: str = ""
    dur_audio: float = 0.0
    audio_bytes: bytes = b""
    prompt_snapshot: list = None
    timed_out: bool = False
    error: str = ""


@dataclass
class ControlMsg:
    kind: str             # "session_end" | "disconnect"
    data: dict = field(default_factory=dict)


class ActorEngine:
    """Single-writer conversation engine. All state mutations happen in engine_loop."""

    def __init__(self, websocket=None, prompts: dict = None, delay: dict = None,
                 llm_cfg: dict = None, engine_cfg: dict = None,
                 llm_fn=None, asr_fn=None, tts_fn=None,
                 replay_mode: str = "realtime", decision_script=None,
                 trace_path=None, vad_model=None, vad_iterator=None):
        self.websocket = websocket
        self.q: asyncio.Queue = asyncio.Queue()

        # ---- injectable model functions (tests / mocks) ----
        self.llm_fn = llm_fn or _module.llm_qwen3o
        self.asr_fn = asr_fn or _module.asr
        self.tts_fn = tts_fn or _module.tts

        # ---- config ----
        self.prompts = prompts or {}
        self.delay = delay or {}
        self.llm_cfg = llm_cfg or {}
        self.engine_cfg = engine_cfg or {}
        self.END_HOLD = float(self.delay.get("end_hold_frame", 0.64))
        self.AFTER_CONTINUE_TIMEOUT = float(self.delay.get("after_continue_time", 2.5))
        self.JUDGE_PROMPT = self.prompts.get("judge", "")
        self.INTERRUPT_PROMPT = self.prompts.get("interrupt", "")
        self.RESPONSE_PROMPT = self.prompts.get("response", "")
        self.SHIFT_PROMPT = self.prompts.get("shift", "")
        self.SHIFT_RE_PROMPT = self.prompts.get("shift_s", "")
        self.AUDIO_BLOCK = self.llm_cfg.get("audio_block", "audio_url")
        self.DECISION_TIMEOUT = float(self.llm_cfg.get("decision_timeout_s", 15))
        self.PLAYBACK_AUTOEND = bool(self.engine_cfg.get("playback_autoend", False))

        # ---- replay ----
        assert replay_mode in ("realtime", "injected", "oracle"), replay_mode
        self.replay_mode = replay_mode
        self.decision_script = decision_script
        self._scheduled = []          # heap of (due_t_audio, seq, ModelDone)
        self._sched_seq = 0

        # ---- audio clock / perception ----
        self.t_audio = 0.0
        if vad_iterator is not None:
            # injected perception (unit tests / scripted VAD)
            self.vad_model = None
            self.vad_iterator = vad_iterator
        else:
            self.vad_model = vad_model if vad_model is not None else load_silero_vad()
            self.vad_iterator = VADIterator(self.vad_model, sampling_rate=SAMPLE_RATE)
        self._vad_buf = np.zeros(0, dtype=np.float32)

        # ---- conversation state (legacy names kept for trace/diff parity) ----
        self.STATE = "LISTEN"
        self.IN_SPEECH = False
        self.BUFFER = []
        self.TURN_IDX = 0
        self.SILENCE_COUNTER = 0
        self.CONTINUE_ARMED = False
        self.interrupt_buf = []
        self.assistant_history = []
        self.user_history = []

        # ---- audio-clock anchors (replace legacy wall-clock anchors) ----
        self.t_end_anchor = None          # vad end   -> END_HOLD window
        self.t_continue_anchor = None     # judged segment end -> continue timeout
        self.t_interrupt_start = None     # interrupt vad_start -> 1.5s window
        self._judged_seg_end = None
        self._seg_closed = False          # SPEAK segment dispatched to intent judge;
                                          # suppresses the long-interrupt timer until
                                          # the segment is consumed or speech resumes

        # ---- staleness protocol ----
        self.session_gen = 0
        self.seg_epoch = 0
        self._pending_audio = {}          # (kind, epoch) -> np.ndarray snapshot at dispatch
        self._pending_frames = {}         # (kind, epoch) -> list[np.ndarray] snapshot

        # ---- playback bookkeeping (record-only unless playback_autoend) ----
        self.playback_end_audio = None

        # ---- observability ----
        self.output_dir = None
        self.start_wall = None
        self.trace = []
        self.trace_path = Path(trace_path) if trace_path else None
        self._trace_fh = None
        self.frame_lags = []              # perf-lag reader->engine per frame (freeze metric)
        self.max_queue_depth = 0
        self._inflight = 0                # realtime worker tasks not yet reported back
        self._running = True

    # ------------------------------------------------------------------
    # perception helpers
    # ------------------------------------------------------------------
    def detect_vad_frame(self, chunk):
        self._vad_buf = np.concatenate([self._vad_buf, chunk])
        if len(self._vad_buf) >= 2 * WINDOW_SIZE:
            tensor = torch.from_numpy(self._vad_buf[: 2 * WINDOW_SIZE])
            event = self.vad_iterator(tensor, return_seconds=True)
            self._vad_buf = np.zeros(0, dtype=np.float32)
            return event
        return None

    # ------------------------------------------------------------------
    # trace / control
    # ------------------------------------------------------------------
    async def send_control(self, event_type: str, data=None):
        payload = {"event": event_type, "data": data or {}}
        rec = dict(payload)
        rec["data"] = dict(payload["data"])
        rec["data"].setdefault("t_audio", round(self.t_audio, 3))
        self.trace.append(rec)
        if self._trace_fh:
            self._trace_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.websocket is not None:
            await self.websocket.send_text(json.dumps(payload))

    def _wall_ts(self):
        if self.start_wall is None:
            return 0.0
        return round(time.time() - self.start_wall, 3)

    # ------------------------------------------------------------------
    # message building (identical semantics to legacy build_messages)
    # ------------------------------------------------------------------
    def build_messages(self, system_prompt, user_audio, use_history, shift_history):
        messages = [{"role": "system", "content": system_prompt}]
        user_history, assistant_history = self.user_history, self.assistant_history
        if not shift_history and ((len(user_history) == 0 and len(assistant_history) == 0) or not use_history):
            if user_audio is not None:
                messages.append({
                    "role": "user",
                    "content": [build_audio_content(user_audio, SAMPLE_RATE, self.AUDIO_BLOCK)]
                })
            return messages
        rounds = min(len(user_history), len(assistant_history))
        for i in range(rounds):
            messages.append({"role": "user", "content": [{"type": "text", "text": user_history[i]}]})
            messages.append({"role": "assistant", "content": assistant_history[i]})
        if user_audio is not None:
            messages.append({
                "role": "user",
                "content": [build_audio_content(user_audio, SAMPLE_RATE, self.AUDIO_BLOCK)]
            })
        return messages

    # ------------------------------------------------------------------
    # dispatch: fork a decision and return to the event loop immediately
    # ------------------------------------------------------------------
    def dispatch_llm(self, kind, system_prompt, user_audio, turn,
                     add_to_history=False, shift_history=False):
        messages = self.build_messages(system_prompt, user_audio, add_to_history, shift_history)
        snapshot = scrub_audio_blocks(messages)
        gen, epoch = self.session_gen, self.seg_epoch
        if user_audio is not None and kind in ("judge", "shift", "interrupt"):
            # snapshot consumed by the handler when the (fresh) result arrives
            self._pending_audio[(kind, epoch)] = user_audio

        if self.replay_mode != "realtime":
            res = self.decision_script(kind, {"turn": turn, "t_audio": self.t_audio,
                                              "epoch": epoch, "prompt": system_prompt})
            infer = 0.0 if self.replay_mode == "oracle" else float(res.get("infer", 0.0))
            done = ModelDone(kind=kind, gen=gen, epoch=epoch, turn=turn,
                             text=str(res.get("text", "")), infer=infer,
                             add_to_history=add_to_history, prompt_snapshot=snapshot)
            self._schedule(self.t_audio + infer, done)
            return

        async def _run():
            t0 = time.perf_counter()
            timed_out = False
            text = ""
            error = ""
            retries = 1 if kind == "response" else 0
            while True:
                try:
                    text = await asyncio.wait_for(
                        asyncio.to_thread(self.llm_fn, messages), self.DECISION_TIMEOUT)
                    break
                except asyncio.TimeoutError:
                    if retries > 0:
                        retries -= 1
                        continue
                    timed_out = True
                    text = RESPONSE_TIMEOUT_APOLOGY if kind in ("response", "shift_re") \
                        else TIMEOUT_FALLBACK.get(kind, "")
                    break
                except Exception as exc:  # never leak a worker: accounting depends on it
                    error = str(exc)
                    text = ""
                    break
            infer = round(time.perf_counter() - t0, 3)
            self.q.put_nowait(ModelDone(kind=kind, gen=gen, epoch=epoch, turn=turn,
                                        text=str(text), infer=infer,
                                        add_to_history=add_to_history,
                                        prompt_snapshot=snapshot, timed_out=timed_out,
                                        error=error))
        self._inflight += 1
        asyncio.create_task(_run())

    def dispatch_asr(self, user_audio, turn):
        gen, epoch = self.session_gen, self.seg_epoch
        out_path = None
        if self.output_dir is not None:
            out_path = Path(self.output_dir) / f"stream_turn{turn}_input.wav"

        if self.replay_mode != "realtime":
            res = self.decision_script("asr", {"turn": turn, "t_audio": self.t_audio})
            infer = 0.0 if self.replay_mode == "oracle" else float(res.get("infer", 0.0))
            self._schedule(self.t_audio + infer,
                           ModelDone(kind="asr", gen=gen, epoch=epoch, turn=turn,
                                     text=str(res.get("text", "")), infer=infer))
            return

        async def _run():
            t0 = time.perf_counter()
            text = ""
            error = ""
            def _work():
                import soundfile as sf
                if out_path is not None:
                    sf.write(out_path, user_audio, SAMPLE_RATE)
                    return self.asr_fn(str(out_path))
                # no output dir (pure in-memory replay): temp-free path via asr on array
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                    sf.write(tmp.name, user_audio, SAMPLE_RATE)
                    return self.asr_fn(tmp.name)
            try:
                text = await asyncio.to_thread(_work)
            except Exception as exc:  # ASR must never kill the session
                error = str(exc)
                print(f"[ASR ERROR] {exc}")
            infer = round(time.perf_counter() - t0, 3)
            self.q.put_nowait(ModelDone(kind="asr", gen=gen, epoch=epoch, turn=turn,
                                        text=str(text), infer=infer, error=error))
        self._inflight += 1
        asyncio.create_task(_run())

    def dispatch_tts(self, text, turn):
        gen, epoch = self.session_gen, self.seg_epoch
        tts_path = None
        if self.output_dir is not None:
            tts_path = Path(self.output_dir) / f"turn{turn}_tts.wav"

        if self.replay_mode != "realtime":
            res = self.decision_script("tts", {"turn": turn, "t_audio": self.t_audio, "text": text})
            infer = 0.0 if self.replay_mode == "oracle" else float(res.get("infer", 0.0))
            self._schedule(self.t_audio + infer,
                           ModelDone(kind="tts", gen=gen, epoch=epoch, turn=turn, text=text,
                                     infer=infer, wav_path=str(res.get("wav_path", "")),
                                     dur_audio=float(res.get("dur_audio", 0.0))))
            return

        async def _run():
            t0 = time.perf_counter()
            error = ""
            path, dur, raw = "", 0.0, b""
            def _work():
                import soundfile as sf
                p = self.tts_fn(text, tts_path)
                data, sr = sf.read(p, dtype="float32")
                with open(p, "rb") as f:
                    return p, len(data) / sr, f.read()
            try:
                path, dur, raw = await asyncio.to_thread(_work)
            except Exception as exc:
                error = str(exc)
                print(f"[TTS ERROR] {exc}")
            infer = round(time.perf_counter() - t0, 3)
            self.q.put_nowait(ModelDone(kind="tts", gen=gen, epoch=epoch, turn=turn, text=text,
                                        infer=infer, wav_path=str(path), dur_audio=dur,
                                        audio_bytes=raw, error=error))
        self._inflight += 1
        asyncio.create_task(_run())

    def _schedule(self, due_t_audio, done: ModelDone):
        self._sched_seq += 1
        heapq.heappush(self._scheduled, (due_t_audio, self._sched_seq, done))

    async def _drain_due(self, t_audio):
        """Deliver injected/oracle results whose audio-clock due time has passed.
        Loops because a delivered decision may dispatch follow-ups due at the same t."""
        while self._scheduled and self._scheduled[0][0] <= t_audio:
            _, _, done = heapq.heappop(self._scheduled)
            await self._on_model_done(done)

    # ------------------------------------------------------------------
    # engine loop (the single writer)
    # ------------------------------------------------------------------
    async def _process_event(self, ev):
        """Single-writer event dispatch. Returns False when the session must end."""
        if isinstance(ev, FrameEvent):
            depth = self.q.qsize()
            if depth > self.max_queue_depth:
                self.max_queue_depth = depth
            if ev.t_wall is not None:
                self.frame_lags.append(time.perf_counter() - ev.t_wall)
            await self._drain_due(ev.t_audio)
            await self._on_frame(ev)
        elif isinstance(ev, ModelDone):
            self._inflight = max(0, self._inflight - 1) if self.replay_mode == "realtime" else self._inflight
            await self._on_model_done(ev)
        elif isinstance(ev, ControlMsg):
            if ev.kind == "session_end":
                self._reset_session()
            elif ev.kind == "disconnect":
                return False
        return True

    async def engine_loop(self):
        while self._running:
            ev = await self.q.get()
            if not await self._process_event(ev):
                break

    async def _on_frame(self, ev: FrameEvent):
        self.t_audio = ev.t_audio
        event = self.detect_vad_frame(ev.pcm)
        if event and "start" in event:
            # new speech falsifies any in-flight judge/shift/interrupt evidence
            self.seg_epoch += 1
        if self.STATE == "LISTEN":
            await self._listen_frame(ev, event)
        else:
            await self._speak_frame(ev, event)
        # playback bookkeeping (flag default False = legacy: stay in SPEAK forever)
        if (self.PLAYBACK_AUTOEND and self.STATE == "SPEAK"
                and self.playback_end_audio is not None
                and ev.t_audio >= self.playback_end_audio):
            self.STATE = "LISTEN"
            self.playback_end_audio = None
            await self.send_control("playback_end", {
                "timestamp": self._wall_ts(), "turn": self.TURN_IDX, "state": self.STATE})

    # ------------------------------------------------------------------
    # LISTEN state (port of legacy handle_listen, audio clock + dispatch)
    # ------------------------------------------------------------------
    async def _listen_frame(self, ev: FrameEvent, event):
        frame, t = ev.pcm, ev.t_audio
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "timestamp": self._wall_ts(), "turn": self.TURN_IDX, "state": self.STATE})
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
                self.dispatch_llm("judge", self.JUDGE_PROMPT, user_audio, self.TURN_IDX)
                return

        if self.CONTINUE_ARMED:
            if (t - self.t_continue_anchor) >= self.AFTER_CONTINUE_TIMEOUT:
                user_audio = np.concatenate(self.BUFFER)
                self.CONTINUE_ARMED = False
                self.t_continue_anchor = None
                self.IN_SPEECH = False
                self.BUFFER = []
                self._start_answer_chain(user_audio, use_shift=(self.TURN_IDX != 0))
                return
            if event and "start" in event:
                self.CONTINUE_ARMED = False
                self.t_continue_anchor = None

    # ------------------------------------------------------------------
    # SPEAK state (port of legacy handle_speak)
    # ------------------------------------------------------------------
    async def _speak_frame(self, ev: FrameEvent, event):
        frame, t = ev.pcm, ev.t_audio
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "turn": self.TURN_IDX, "state": self.STATE, "timestamp": self._wall_ts()})
            self.IN_SPEECH = True
            self.interrupt_buf = [frame]
            self.SILENCE_COUNTER = 0
            self.t_interrupt_start = t
            return

        if self.IN_SPEECH:
            self.interrupt_buf.append(frame)

            if event and "start" in event and self._seg_closed:
                # speech resumed AFTER the segment's intent judge was dispatched:
                # that decision is already stale (epoch bumped in _on_frame); the
                # sustained-speech window restarts at the resume point (legacy's
                # post-freeze drain would likewise have re-anchored a new segment)
                self._seg_closed = False
                self.t_interrupt_start = t

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
                    self._seg_closed = True
                    seg_audio = np.concatenate(self.interrupt_buf)
                    self._pending_frames[("interrupt", self.seg_epoch)] = list(self.interrupt_buf)
                    self.dispatch_llm("interrupt", self.INTERRUPT_PROMPT, seg_audio,
                                      self.TURN_IDX, add_to_history=False)
                    return

            # long interrupt: >=1.5s of speech with no endpoint forces LISTEN
            # (suppressed while the segment is closed awaiting its intent judge;
            #  t_interrupt_start None = carried-over segment from a LISTEN->SPEAK
            #  transition; legacy's wall-clock zero anchor made this fire at once)
            if (self.interrupt_buf and self.SILENCE_COUNTER == 0
                    and not self._seg_closed
                    and (t - (self.t_interrupt_start or 0.0)) >= LONG_INTERRUPT_SEC):
                self.TURN_IDX += 1
                self.seg_epoch += 1
                self.STATE = "LISTEN"
                await self.send_control("long_interrupt", {
                    "timestamp": self._wall_ts(), "turn": self.TURN_IDX, "state": self.STATE})
                self.BUFFER = list(self.interrupt_buf)
                self.IN_SPEECH = True
                self.interrupt_buf = []
                self.SILENCE_COUNTER = 0
                return

    # ------------------------------------------------------------------
    # decision results
    # ------------------------------------------------------------------
    def _start_answer_chain(self, user_audio, use_shift):
        """LISTEN EoU / continue-timeout: shift gate (non-first turn) then answer.
        Interrupt switch path calls with use_shift=False (legacy parity)."""
        if use_shift:
            self.dispatch_llm("shift", self.SHIFT_PROMPT, user_audio, self.TURN_IDX,
                              add_to_history=False, shift_history=True)
        else:
            self.dispatch_asr(user_audio, self.TURN_IDX)
            self.dispatch_llm("response", self.RESPONSE_PROMPT, user_audio, self.TURN_IDX,
                              add_to_history=True)

    async def _trace_llm_done(self, ev: ModelDone):
        await self.send_control("llm_done", {
            "timestamp": self._wall_ts(),
            "infer_time": ev.infer,
            "content": ev.text,
            "prompt": ev.prompt_snapshot,
            "turn": ev.turn,
            "state": self.STATE,
        })
        if ev.timed_out:
            await self.send_control("llm_timeout", {
                "timestamp": self._wall_ts(), "kind": ev.kind, "turn": ev.turn})

    async def _on_model_done(self, ev: ModelDone):
        # --- staleness gates ---
        if ev.gen != self.session_gen:
            return  # pre-reset result: drop silently (legacy could not even reach here)
        if ev.kind in ("judge", "shift", "interrupt") and ev.epoch != self.seg_epoch:
            await self.send_control("llm_stale_dropped", {
                "timestamp": self._wall_ts(), "kind": ev.kind, "turn": ev.turn,
                "epoch": ev.epoch, "current_epoch": self.seg_epoch,
                "content": ev.text, "infer_time": ev.infer})
            self._pending_audio.pop((ev.kind, ev.epoch), None)
            self._pending_frames.pop((ev.kind, ev.epoch), None)
            return

        handler = {
            "judge": self._on_judge, "shift": self._on_shift,
            "interrupt": self._on_interrupt, "response": self._on_response,
            "shift_re": self._on_response, "asr": self._on_asr, "tts": self._on_tts,
        }[ev.kind]
        await handler(ev)

    async def _on_judge(self, ev: ModelDone):
        await self._trace_llm_done(ev)
        audio = self._pending_audio.pop(("judge", ev.epoch), None)
        if "continue" in ev.text.lower():
            self.CONTINUE_ARMED = True
            # audio-clock anchor = end of the judged segment (documented deviation:
            # legacy anchored at judge-return wall time, stacking judge latency
            # onto the user-perceived wait)
            self.t_continue_anchor = self._judged_seg_end
            self.IN_SPEECH = True
            return
        self.IN_SPEECH = False
        self._start_answer_chain(audio, use_shift=(self.TURN_IDX != 0))

    async def _on_shift(self, ev: ModelDone):
        await self._trace_llm_done(ev)
        audio = self._pending_audio.pop(("shift", ev.epoch), None)
        low = ev.text.lower()
        if "no" in low:
            self.dispatch_asr(audio, self.TURN_IDX)
            self.dispatch_llm("response", self.RESPONSE_PROMPT, audio, self.TURN_IDX,
                              add_to_history=True)
        elif "yes" in low:
            self.dispatch_llm("shift_re", self.SHIFT_RE_PROMPT, None, self.TURN_IDX,
                              add_to_history=False, shift_history=True)
        # neither -> dead end (legacy parity: no response is produced)

    async def _on_interrupt(self, ev: ModelDone):
        await self._trace_llm_done(ev)
        audio = self._pending_audio.pop(("interrupt", ev.epoch), None)
        frames = self._pending_frames.pop(("interrupt", ev.epoch), [])
        if "switch" in ev.text.lower():
            await self.send_control("shot_interrupt", {
                "timestamp": self._wall_ts(), "turn": self.TURN_IDX, "state": self.STATE})
            self.BUFFER = list(frames)
            self.TURN_IDX += 1
            self.seg_epoch += 1
            self.dispatch_asr(audio, self.TURN_IDX)
            self.dispatch_llm("response", self.RESPONSE_PROMPT, audio, self.TURN_IDX,
                              add_to_history=True)
        else:
            await self.send_control("no_interrupt", {
                "timestamp": self._wall_ts(), "turn": self.TURN_IDX, "state": self.STATE})
            self.BUFFER = list(frames)
        self.IN_SPEECH = False
        self.interrupt_buf = []
        self.SILENCE_COUNTER = 0
        self._seg_closed = False

    async def _on_response(self, ev: ModelDone):
        if ev.add_to_history:
            self.assistant_history.append(str(ev.text))
        await self._trace_llm_done(ev)
        self.dispatch_tts(ev.text, ev.turn)

    async def _on_asr(self, ev: ModelDone):
        await self.send_control("asr_done", {
            "timestamp": self._wall_ts(), "turn": ev.turn,
            "state": self.STATE, "content": ev.text})
        self.user_history.append(str(ev.text))
        # deviation (documented): legacy cleared BUFFER here from a worker task,
        # which could clobber a fresh segment; the buffer is snapshot-consumed at
        # dispatch time instead, so no clear is needed.

    async def _on_tts(self, ev: ModelDone):
        await self.send_control("tts_done", {
            "timestamp": self._wall_ts(), "infer_time": ev.infer,
            "turn": ev.turn, "state": self.STATE, "dur_audio": round(ev.dur_audio, 3)})
        if self.websocket is not None and ev.audio_bytes:
            await self.websocket.send_bytes(ev.audio_bytes)
        self.STATE = "SPEAK"
        self.playback_end_audio = self.t_audio + ev.dur_audio

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _reset_session(self):
        self.session_gen += 1
        self.seg_epoch += 1
        self.vad_iterator.reset_states()
        self.STATE = "LISTEN"
        self.TURN_IDX = 0
        self.BUFFER = []
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self.IN_SPEECH = False
        self.SILENCE_COUNTER = 0
        self.CONTINUE_ARMED = False
        self.t_end_anchor = None
        self.t_continue_anchor = None
        self.t_interrupt_start = None
        self._seg_closed = False
        self.interrupt_buf = []
        self.assistant_history = []
        self.user_history = []
        self._pending_audio = {}
        self._pending_frames = {}
        self._scheduled = []
        self.playback_end_audio = None

    async def _reader(self, websocket):
        """The ONLY consumer of the websocket. Never awaits models, never touches state."""
        seq = 0
        rx_samples = 0
        while True:
            message = await websocket.receive()
            if "type" in message and message["type"] == "websocket.disconnect":
                self.q.put_nowait(ControlMsg("disconnect"))
                break
            if "text" in message and message["text"] is not None:
                try:
                    obj = json.loads(message["text"])
                except Exception:
                    continue
                if obj.get("event") == "end":
                    self.q.put_nowait(ControlMsg("session_end"))
                continue
            if "bytes" in message and message["bytes"]:
                pcm = np.frombuffer(message["bytes"], dtype=np.float32)
                if pcm.size == 0:
                    continue
                seq += 1
                rx_samples += pcm.size
                self.q.put_nowait(FrameEvent(
                    seq=seq, t_audio=rx_samples / SAMPLE_RATE,
                    t_wall=time.perf_counter(), pcm=pcm))

    async def run_realtime(self, websocket):
        """Signature-compatible with the legacy engine (create_app calls this)."""
        print("client ok (actor engine)")
        self.websocket = websocket
        self.start_wall = time.time()
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_fh = self.trace_path.open("w", encoding="utf-8")
        reader = asyncio.create_task(self._reader(websocket))
        try:
            await self.engine_loop()
        except Exception as e:
            print("Realtime wrong:", e)
        finally:
            reader.cancel()
            self.vad_iterator.reset_states()
            self._reset_session()
            if self._trace_fh:
                self._trace_fh.close()
                self._trace_fh = None
            print("end (actor engine)")

    # ------------------------------------------------------------------
    # offline driving (replay / tests): feed events without a websocket
    # ------------------------------------------------------------------
    async def run_offline(self, frame_iter, *, paced=False, quiesce_timeout=120.0):
        """Drive the engine from an iterable of FrameEvent, then drain to quiescence.

        Tail semantics mirror the legacy engine: the last dispatched decision
        chain COMPLETES before the session is considered over (legacy awaited
        it inline before ever seeing the "end" control message).

        paced=True sleeps real time between frames (needed when model fns
        measure wall latency, e.g. the freeze A/B with a sleeping mock);
        paced=False is the fast path for injected/oracle replay.
        """
        if self.start_wall is None:
            self.start_wall = time.time()
        if self.trace_path and self._trace_fh is None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_fh = self.trace_path.open("w", encoding="utf-8")

        frames = list(frame_iter)

        async def producer():
            for ev in frames:
                if paced:
                    await asyncio.sleep(len(ev.pcm) / SAMPLE_RATE)
                    ev.t_wall = time.perf_counter()  # arrival stamp = put time
                self.q.put_nowait(ev)

        prod = asyncio.create_task(producer())
        deadline = time.perf_counter() + quiesce_timeout
        try:
            while time.perf_counter() < deadline:
                if prod.done() and self.q.empty() and self._inflight == 0:
                    if self._scheduled:
                        # injected/oracle tail: advance the audio clock to the next due
                        due = self._scheduled[0][0]
                        self.t_audio = max(self.t_audio, due)
                        await self._drain_due(self.t_audio)
                        continue
                    break
                try:
                    ev = await asyncio.wait_for(self.q.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                await self._process_event(ev)
        finally:
            if not prod.done():
                prod.cancel()
            if self._trace_fh:
                self._trace_fh.close()
                self._trace_fh = None

    def freeze_stats(self):
        if not self.frame_lags:
            return {}
        lags = sorted(self.frame_lags)
        pick = lambda p: lags[min(len(lags) - 1, int(p * len(lags)))]
        return {"n": len(lags), "p50_ms": pick(0.50) * 1e3, "p95_ms": pick(0.95) * 1e3,
                "p99_ms": pick(0.99) * 1e3, "max_ms": lags[-1] * 1e3,
                "max_queue_depth": self.max_queue_depth}


# ---------------------------------------------------------------------------
# frame builders (replay / tests)
# ---------------------------------------------------------------------------
def frames_from_array(data, chunk=WINDOW_SIZE):
    """Chunk a float32 waveform into FrameEvents exactly like src/frontend.py does
    (256-sample frames, zero-padded tail). t_wall is left None; the paced offline
    producer stamps it at put time."""
    data = np.asarray(data, dtype=np.float32)
    frames = []
    rx = 0
    seq = 0
    for i in range(0, len(data), chunk):
        pcm = data[i:i + chunk]
        if len(pcm) < chunk:
            pcm = np.pad(pcm, (0, chunk - len(pcm)))
        seq += 1
        rx += len(pcm)
        frames.append(FrameEvent(seq=seq, t_audio=rx / SAMPLE_RATE, t_wall=None, pcm=pcm))
    return frames


def frames_from_wav(wav_path, chunk=WINDOW_SIZE):
    import soundfile as sf
    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{wav_path}: sample rate {sr} != {SAMPLE_RATE}")
    return frames_from_array(data, chunk)
