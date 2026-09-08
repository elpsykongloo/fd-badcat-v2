import json, asyncio, time, torch, soundfile as sf, numpy as np, base64, tempfile, io, os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, tts, VERBATIM_TTS_CONTRACT
from messages import build_audio_content, scrub_audio_blocks
import argparse
import uvicorn
import yaml
import copy
# ============================================================
# ConversationEngine
# ============================================================

class ConversationEngine:
    def __init__(self, websocket: WebSocket = None, prompts: dict = None, delay: dict = None, llm_cfg: dict = None):
        self.SAMPLE_RATE = 16000
        self.WINDOW_SIZE = 256
        self.FRAME_SEC = 256 / 16000

        self.STATE = "LISTEN"
        self.IN_SPEECH = False
        self.BUFFER = []
        self.TURN_IDX = 0
        self.INTERRUPT_COUNT = 0
        self.SILENCE_COUNTER = 0
        self.CONTINUE_START_TIME = None
        self.CONTINUE_ARMED = False
        self.interrupt_buf = []
        self.INTERRUPT_START_TIME = 0

        self.output_dir = None
        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(self.vad_model, sampling_rate=self.SAMPLE_RATE)
        self.websocket = websocket

        # yaml
        self.prompts = prompts
        self.delay = delay
        self.llm_cfg = llm_cfg or {}
        self.AUDIO_BLOCK = self.llm_cfg.get("audio_block", "audio_url")
        self.END_HOLD_FRAMES = float(delay["end_hold_frame"])
        self.AFTER_CONTINUE_TIMEOUT_FRAMES = float(delay["after_continue_time"])
        self.JUDGE_PROMPT = prompts.get("judge", "")
        self.INTERRUPT_PROMPT = prompts.get("interrupt", "")
        self.RESPONSE_PROMPT = prompts.get("response", "")
        self.SHIFT_PROMPT = prompts.get("shift", "")
        self.SHIFT_RE_PROMPT = prompts.get("shift_s", "")
        self.semantic_shift = None

        self.assistant_history = []
        self.user_history = []

    # build LLM messages
    def build_messages(self, system_prompt, user_history, assistant_history, user_audio, use_history, shift_history):
        messages = [{"role": "system", "content": system_prompt}]
        # ---------------------------------------------
        # First branch (no history / history disabled)
        # Additionally: if shift_history == True, skip this branch
        # ---------------------------------------------
        if not shift_history and ((len(user_history) == 0 and len(assistant_history) == 0) or not use_history):
            if user_audio is not None:
                messages.append({
                    "role": "user",
                    "content": [build_audio_content(user_audio, self.SAMPLE_RATE, self.AUDIO_BLOCK)]
                })
            return messages
        # with user history
        rounds = min(len(user_history), len(assistant_history))
        for i in range(rounds):
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_history[i]}]
            })
            messages.append({
                "role": "assistant",
                "content": assistant_history[i]
            })
        if user_audio is not None:
            messages.append({
                "role": "user",
                "content": [build_audio_content(user_audio, self.SAMPLE_RATE, self.AUDIO_BLOCK)]
            })
        return messages

    def reset(self):
        self.STATE = "LISTEN"
        self.TURN_IDX = 0
        self.BUFFER.clear()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self.IN_SPEECH = False
        self.SILENCE_COUNTER = 0
        self.CONTINUE_ARMED = False
        self.CONTINUE_START_TIME = None
        self.INTERRUPT_COUNT = 0
        self.interrupt_buf.clear()
        self.assistant_history.clear()
        self.user_history.clear()
        self.semantic_shift = None

    async def send_control(self, event_type: str, data=None):
        if not self.websocket:
            return
        payload = {"event": event_type, "data": data or {}}
        await self.websocket.send_text(json.dumps(payload))


    def detect_vad_frame(self, chunk):
        if not hasattr(self, "_vad_buf"):
            self._vad_buf = np.zeros(0, dtype=np.float32)
        self._vad_buf = np.concatenate([self._vad_buf, chunk])
        if len(self._vad_buf) >= 2 * self.WINDOW_SIZE:
            tensor = torch.from_numpy(self._vad_buf[: 2 * self.WINDOW_SIZE])
            event = self.vad_iterator(tensor, return_seconds=True)
            self._vad_buf = np.zeros(0, dtype=np.float32)
            return event
        return None


    async def async_asr(self, user_audio, turn_id):
        tmp = self.output_dir / f"stream_turn{turn_id}_input.wav"
        sf.write(tmp, user_audio, self.SAMPLE_RATE)
        user_text = await asyncio.to_thread(asr, str(tmp))

        await self.send_control("asr_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "state": self.STATE,
            "content": user_text
        })
        self.user_history.append(str(user_text))
        self.BUFFER.clear()
        return user_text


    async def async_llm(self, system_prompt, user_audio, turn_id, add_to_history=False, shift_history=False):
        messages = self.build_messages(
            system_prompt=system_prompt,
            user_history=self.user_history,
            assistant_history=self.assistant_history,
            user_audio=user_audio,
            use_history=add_to_history,
            shift_history=shift_history
        )
        start_t = time.perf_counter()
        decision = await asyncio.to_thread(llm_qwen3o, messages)
        infer_time = round(time.perf_counter() - start_t, 3)
        # ============  send_control ============
        messages_clean = scrub_audio_blocks(messages)

        await self.send_control("llm_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "infer_time": infer_time,
            "content": decision,
            "prompt": messages_clean,
            "turn": turn_id,
            "state": self.STATE,
        })
        if add_to_history:
            self.assistant_history.append(str(decision))
        self.IN_SPEECH = False
        return decision

    # ==================================================
    async def async_tts(self, text, turn_id):
        tts_path = self.output_dir / f"turn{turn_id}_tts.wav"

        start_t = time.perf_counter()
        tts_file = await asyncio.to_thread(tts, text, tts_path)
        infer_time = round(time.perf_counter() - start_t, 3)
                
        await self.send_control("tts_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "infer_time": infer_time,
            "turn": turn_id,
            "state": self.STATE
        })

        with open(tts_file, "rb") as f:
            await self.websocket.send_bytes(f.read())
        self.STATE = "SPEAK"

    # ==================================================
    # LISTEN / SPEAK state
    # ==================================================
    async def handle_listen(self, frame, event):
        # ----speak start ----
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE
            })
            self.IN_SPEECH = True
            self.BUFFER = [frame]
            return

        if not self.IN_SPEECH:
            return

        self.BUFFER.append(frame)

        # ---- end appear ----
        if event and "end" in event:
            self.SILENCE_COUNTER = 1
            self.INTERRUPT_END_TIME = time.time()
            await self.send_control("vad_done", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE
            })
            return

        if self.SILENCE_COUNTER > 0:
            if event and "start" in event:
                self.SILENCE_COUNTER = 0
                return
            else:
                elapsed_silence = time.time() - self.INTERRUPT_END_TIME
                if elapsed_silence >= self.END_HOLD_FRAMES:
                    self.SILENCE_COUNTER = 0
                    await self.send_control("vad_640_done", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": self.TURN_IDX,
                        "state": self.STATE
                    })

                    user_audio = np.concatenate(self.BUFFER)
                    decision = await self.async_llm(self.JUDGE_PROMPT, user_audio, self.TURN_IDX)
                    if "continue" in decision.lower():
                        self.CONTINUE_ARMED = True
                        self.CONTINUE_START_TIME = time.time()
                        self.IN_SPEECH = True
                        return

                    # --semantic shift--
                    if self.TURN_IDX != 0:
                        shift_judge = await self.async_llm(self.SHIFT_PROMPT, user_audio, self.TURN_IDX, add_to_history=False, shift_history=True)
                        if "no" in shift_judge.lower(): #normal answer
                            asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                            decision = await self.async_llm(self.RESPONSE_PROMPT, user_audio, self.TURN_IDX, add_to_history=True)
                            asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                            return
                        elif "yes" in shift_judge.lower(): #repeat
                            decision = await self.async_llm(self.SHIFT_RE_PROMPT, None, self.TURN_IDX, add_to_history=False, shift_history=True)
                            asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                            return
                    else:
                        asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                        decision = await self.async_llm(self.RESPONSE_PROMPT, user_audio, self.TURN_IDX, add_to_history=True)
                        asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                        return

        # ---- continue overtime ----
        if self.CONTINUE_ARMED:
            elapsed = time.time() - self.CONTINUE_START_TIME
            if elapsed >= self.AFTER_CONTINUE_TIMEOUT_FRAMES:
                user_audio = np.concatenate(self.BUFFER)
                if self.TURN_IDX != 0:
                    shift_judge = await self.async_llm(self.SHIFT_PROMPT, user_audio, self.TURN_IDX, add_to_history=False, shift_history=True)
                    if "no" in shift_judge.lower(): #normal answer
                        asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                        decision = await self.async_llm(self.RESPONSE_PROMPT, user_audio, self.TURN_IDX, add_to_history=True)
                        asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                    elif "yes" in shift_judge.lower(): #repeat
                        decision = await self.async_llm(self.SHIFT_RE_PROMPT, None, self.TURN_IDX, add_to_history=False, shift_history=True)
                        asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                else:
                    asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                    decision = await self.async_llm(self.RESPONSE_PROMPT, user_audio, self.TURN_IDX, add_to_history=True)
                    asyncio.create_task(self.async_tts(decision, self.TURN_IDX))

                self.CONTINUE_ARMED = False
                self.CONTINUE_START_TIME = None
                self.IN_SPEECH = False
                self.BUFFER.clear()
                return

            if event and "start" in event:
                self.CONTINUE_ARMED = False
                self.CONTINUE_START_TIME = None


    async def handle_speak(self, frame, event):
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "turn": self.TURN_IDX,
                "state": self.STATE,
                "timestamp": round(time.time() - self.start_wall, 3)
            })
            self.IN_SPEECH = True
            self.interrupt_buf = [frame]
            self.INTERRUPT_COUNT = 1
            self.SILENCE_COUNTER = 0
            self.INTERRUPT_START_TIME = time.time()
            return

        # interrupt happen
        if self.IN_SPEECH:
            self.interrupt_buf.append(frame)
            self.INTERRUPT_COUNT += 1

            if event and "end" in event:
                self.SILENCE_COUNTER = 1
                await self.send_control("vad_done", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE
                })
                self.INTERRUPT_END_TIME = time.time()
                return

            # 640ms interrupt done
            if self.SILENCE_COUNTER > 0:
                if event and "start" in event:
                    self.SILENCE_COUNTER = 0
                    return
                else:
                    elapsed_silence = time.time() - self.INTERRUPT_END_TIME

                    if elapsed_silence >= self.END_HOLD_FRAMES:
                        seg_audio = np.concatenate(self.interrupt_buf)
                        intent = await self.async_llm(self.INTERRUPT_PROMPT, seg_audio, self.TURN_IDX, add_to_history=False)

                        if "switch" in intent.lower():

                            await self.send_control("shot_interrupt", {
                                "timestamp": round(time.time() - self.start_wall, 3),
                                "turn": self.TURN_IDX,
                                "state": self.STATE
                            })

                            self.BUFFER = self.interrupt_buf.copy()
                            self.TURN_IDX += 1

                            user_audio = np.concatenate(self.interrupt_buf)
                            asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                            decision = await self.async_llm(self.RESPONSE_PROMPT, user_audio, self.TURN_IDX, add_to_history=True)
                            asyncio.create_task(self.async_tts(decision, self.TURN_IDX))

                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            return

                        else:
                            await self.send_control("no_interrupt", {
                                "timestamp": round(time.time() - self.start_wall, 3),
                                "turn": self.TURN_IDX,
                                "state": self.STATE
                            })

                            self.BUFFER = self.interrupt_buf.copy()
                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            return

            # long interrupt: without end
            if (self.interrupt_buf and
                self.SILENCE_COUNTER == 0 and
                time.time() - self.INTERRUPT_START_TIME >= 1.5):

                self.TURN_IDX += 1
                self.STATE = "LISTEN"

                await self.send_control("long_interrupt", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE
                })

                self.BUFFER = self.interrupt_buf.copy()
                self.IN_SPEECH = True
                self.interrupt_buf.clear()
                self.INTERRUPT_COUNT = 0
                self.SILENCE_COUNTER = 0
                return

        return

    async def run_realtime(self, websocket: WebSocket):
        print("client ok")
        self.start_wall = time.time()
        try:
            while True:
                message = await websocket.receive()
                if "type" in message and message["type"] == "websocket.disconnect":
                    break

                # text message 
                if "text" in message and message["text"] is not None:
                    obj = json.loads(message["text"])
                    if obj.get("event") == "end":
                        self.vad_iterator.reset_states()
                        self.reset()
                        continue
                # audio frames
                if "bytes" in message and message["bytes"]:
                    raw = message["bytes"]
                    frame = np.frombuffer(raw, dtype=np.float32)
                    if frame.size == 0:
                        continue
                    event = self.detect_vad_frame(frame)

                    if self.STATE == "LISTEN":
                        await self.handle_listen(frame, event)
                    else:
                        await self.handle_speak(frame, event)

        except WebSocketDisconnect:
            print("WebSocket disconnect")
        except Exception as e:
            print("Realtime wrong:", e)
        finally:
            self.vad_iterator.reset_states()
            self.reset()
            print("end")

# FastAPI
def create_app(prompts, delay, llm_cfg=None, engine_cfg=None) -> FastAPI:
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def lifespan(app):
        if (engine_cfg or {}).get("warmup"):
            from demo_startup import warmup
            await warmup(prompts, engine_cfg)
        app.state.demo_cases = None
        if (engine_cfg or {}).get("case_capture"):
            from demo_cases import CaseArchive
            app.state.demo_cases = CaseArchive(
                (engine_cfg or {}).get("case_capture_dir", "exp/demo_cases"),
                max_bytes=int((engine_cfg or {}).get("case_capture_max_bytes", 536870912)),
                max_cases=int((engine_cfg or {}).get("case_capture_max_cases", 2000)))
        try:
            yield
        finally:
            if app.state.demo_cases is not None:
                await app.state.demo_cases.close()
    app = FastAPI(lifespan=lifespan)
    arch = (engine_cfg or {}).get("arch", "actor")
    from fastapi.staticfiles import StaticFiles
    app.mount("/demo", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="demo")

    @app.get("/api/demo/info")
    async def demo_info():
        # Configuration only, not a claim that the upstream GPU is healthy.
        # Never expose prompts, environment variables, paths, or credentials.
        info = {"protocol": "pcm16.v1", "streaming": bool(
            arch == "actor" and (engine_cfg or {}).get("phase", "a") == "a"
            and (engine_cfg or {}).get("stream_response"))}
        if info["streaming"] and (engine_cfg or {}).get("guarded_turns"):
            from control_labels import ROUTE_PROTOCOL
            from guarded_turns import input_timing
            info.update(input_protocol="pcm16.ref.v1", guarded_turns=True,
                        route_protocol=ROUTE_PROTOCOL, route_reference_text=False,
                        input_timing=input_timing(engine_cfg),
                        route_penalties={"presence": 0.0, "frequency": 0.0})
        archive = getattr(app.state, "demo_cases", None)
        if archive is not None:
            info["case_capture"] = archive.stats()
        return info

    @app.websocket("/realtime")
    async def realtime_ws(websocket: WebSocket):
        await websocket.accept()
        msg = await websocket.receive_json()

        data = msg.get("data", {})
        exp = data.get("exp", {})
        lang = data.get("lang", {})
        web_demo = data.get("client") == "humdial-web"
        if (data.get("input_protocol") not in (None, "pcm16.ref.v1") or (
                data.get("input_protocol") == "pcm16.ref.v1" and (
                    arch != "actor" or (engine_cfg or {}).get("phase", "a") != "a"
                    or not (engine_cfg or {}).get("stream_response")
                    or data.get("audio_protocol") != "pcm16.v1"))):
            await websocket.send_json({"event": "error", "data": {"message": "Unsupported input protocol"}})
            await websocket.close(code=1008)
            return
        if web_demo:
            # The new browser page does not control any filesystem path. Keep
            # historical non-demo clients' experiment naming unchanged.
            from uuid import uuid4
            from urllib.parse import urlsplit
            origin = urlsplit(websocket.headers.get("origin", ""))
            if (origin.scheme not in ("http", "https")
                    or origin.netloc != websocket.headers.get("host")
                    or data.get("audio_protocol") != "pcm16.v1"):
                await websocket.send_json({"event": "error", "data": {
                    "message": "Web demo requires same-origin access and pcm16.v1"}})
                await websocket.close(code=1008)
                return
            exp, lang = "web-demo-" + uuid4().hex, "live"
        if data.get("audio_protocol") == "pcm16.v1" and (
                arch != "actor" or (engine_cfg or {}).get("phase", "a") != "a"
                or not (engine_cfg or {}).get("stream_response")):
            await websocket.send_json({"event": "error", "data": {
                "message": "Use HumDial ActorEngine with --streaming for this demo"}})
            await websocket.close(code=1008)
            return
        if arch == "legacy":
            engine = ConversationEngine(websocket=websocket, prompts=prompts, delay=delay, llm_cfg=llm_cfg)
        elif (engine_cfg or {}).get("phase") == "b":
            # W3 D1: live TACT engine (same tact_core semantics as the replay driver)
            from engine_b import TactEngine
            from tact.tools import ToolRegistry   # path fixed by engine_b->tact_core
            reg = ToolRegistry(
                latency_profile=(engine_cfg or {}).get("latency_profile", "normal"),
                room=f"live-{exp}")
            engine = TactEngine(websocket=websocket, prompts=prompts, delay=delay,
                                llm_cfg=llm_cfg, engine_cfg=engine_cfg,
                                tool_executor=reg.executor)
        else:
            from engine import ActorEngine
            session_cfg = dict(engine_cfg or {})
            # Existing HumDial clients keep their whole-WAV protocol, even when
            # this server enables the new demo. Both sides must opt in.
            session_cfg["stream_response"] = bool(session_cfg.get("stream_response")
                                                   and data.get("audio_protocol") == "pcm16.v1")
            session_cfg["input_protocol"] = data.get("input_protocol")
            session_cfg["guarded_turns"] = bool(session_cfg.get("guarded_turns")
                and data.get("input_protocol") == "pcm16.ref.v1" and session_cfg["stream_response"])
            # Read-only comparison knob: a client can disable speculation, never
            # enable a server-disabled feature. Snapshot/cancellation stay equal.
            if data.get("speculative_response") is False:
                session_cfg["speculative_response"] = False
            engine = ActorEngine(websocket=websocket, prompts=prompts, delay=delay,
                                 llm_cfg=llm_cfg, engine_cfg=session_cfg)
        engine.output_dir = Path("exp") / exp / f"realtimeout_{lang}"
        engine.output_dir.mkdir(parents=True, exist_ok=True)
        if web_demo:
            from demo_trace import DemoTrace
            engine.demo_trace = DemoTrace(engine.output_dir / "events.jsonl")
            engine.demo_cases = getattr(app.state, "demo_cases", None)
            engine.demo_session_id = exp
            # The browser waits before opening the input pipe. VAD construction
            # may take time on the first connection; do not accumulate mic frames.
        try:
            if web_demo:
                await websocket.send_json({"event": "demo_ready", "data": {
                    "session_id": exp, "protocol": "pcm16.v1",
                    "observability": "demo-trace-v1",
                    "tts_contract": VERBATIM_TTS_CONTRACT,
                    "case_capture": engine.demo_cases is not None,
                    "input_protocol": session_cfg.get("input_protocol"),
                    "guarded_turns": bool(session_cfg.get("guarded_turns")),
                    "speculative_response": bool(session_cfg.get("speculative_response")),
                    "cancellable_response": bool(session_cfg.get("cancellable_response")),
                    "profile": "chat-demo-v1" if (engine_cfg or {}).get("chat_demo") else "humdial"}})
            await engine.run_realtime(websocket)
        finally:
            if web_demo:
                await engine.demo_trace.close()
        if (engine_cfg or {}).get("phase") == "b" and hasattr(engine, "trace"):
            trace_file = engine.output_dir / "trace_full.jsonl"
            with trace_file.open("w", encoding="utf-8") as f:
                for rec in engine.trace:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return app


def load_runtime_config(path, demo_chat=False):
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if demo_chat:
        profile = Path(__file__).resolve().parents[1] / "configs/demo_chat.yaml"
        with profile.open(encoding="utf-8") as handle:
            overlay = yaml.safe_load(handle)
        for section, values in overlay.items():
            cfg.setdefault(section, {}).update(values)
        if cfg["engine"].get("arch", "actor") != "actor" or cfg["engine"].get("phase", "a") != "a":
            raise ValueError("Chat demo profile requires the Phase-A ActorEngine")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--streaming", action="store_true", help="Enable negotiated ActorEngine streaming demo")
    parser.add_argument("--demo-chat", action="store_true", help="Chat profile, bilingual ASR, complete turns and startup warmup")
    parser.add_argument("--host", default=None, help="Override server bind address (demo: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    args = parser.parse_args()

    cfg = load_runtime_config(args.config, args.demo_chat)

    prompts_cfg = cfg.get("prompts", {})
    delay_cfg = cfg.get("time", {})
    server_cfg = cfg.get("server", {})
    llm_cfg = cfg.get("llm", {})
    engine_cfg = cfg.get("engine", {})
    if args.streaming:
        if engine_cfg.get("arch", "actor") != "actor" or engine_cfg.get("phase", "a") != "a":
            parser.error("--streaming is only supported by the HumDial ActorEngine (phase a)")
        engine_cfg["stream_response"] = True

    # module is imported above for legacy compatibility; configure the actual
    # factory constants, not environment variables too late to be consumed.
    import module
    module.configure_asr(cfg.get("asr", {}))

    host = args.host if args.host is not None else server_cfg.get("host", {})
    port = args.port if args.port is not None else server_cfg.get("port", {})

    app = create_app(prompts_cfg, delay_cfg, llm_cfg, engine_cfg)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
