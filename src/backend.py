import json, asyncio, time, torch, soundfile as sf, numpy as np, base64, tempfile, io, os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from silero_vad import load_silero_vad, VADIterator
from module import llm_qwen3o, tts
import argparse
import uvicorn
import yaml
import copy
# ============================================================
# ConversationEngine
# ============================================================

class ConversationEngine:
    def __init__(self, websocket: WebSocket = None, prompts: dict = None, delay: dict = None):
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
        self.END_HOLD_FRAMES = float(delay["end_hold_frame"])
        self.AFTER_CONTINUE_TIMEOUT_FRAMES = float(delay["after_continue_time"])
        self.JUDGE_PROMPT = prompts.get("judge", "")
        self.INTERRUPT_PROMPT = prompts.get("interrupt", "")
        self.RESPONSE_PROMPT = prompts.get("response", "")
        self.RESPONSE_WITH_TRANSCRIPT_PROMPT = (
            prompts.get("response_with_transcript", "").strip()
            or self.default_response_with_transcript_prompt(self.RESPONSE_PROMPT)
        )
        self.SHIFT_PROMPT = prompts.get("shift", "")
        self.SHIFT_RE_PROMPT = prompts.get("shift_s", "")
        self.semantic_shift = None

        self.assistant_history = []
        self.user_history = []

    RESPONSE_TRANSCRIPT_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "response_with_transcript",
            "schema": {
                "type": "object",
                "properties": {
                    "transcript": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["transcript", "answer"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    @staticmethod
    def default_response_with_transcript_prompt(response_prompt: str) -> str:
        return (
            "你必须只输出一个 JSON 对象。不要输出普通文本、Markdown、解释、代码块或 JSON 以外的任何字符。\n"
            "JSON 必须严格包含两个字段：{\"transcript\":\"当前用户音频逐字转写\",\"answer\":\"给用户朗读的最终回复\"}\n"
            "transcript 只写当前这一次用户音频的逐字转写，不要写历史对话或助手回复。\n"
            "answer 是给用户朗读的最终回复，所有回答都必须放进 answer 字段，绝不能直接写在 JSON 外面。\n\n"
            "answer 字段的规则：\n"
            + response_prompt.rstrip()
            + "\nanswer 必须严格遵守上面的回复规则，保持简短自然；后续 TTS 只会朗读 answer。\n"
            "如果音频听不清，transcript 填空字符串，但 answer 仍需根据可理解内容回复。"
        )

    # build LLM messages
    def build_messages(self, system_prompt, user_history, assistant_history, user_audio, use_history, shift_history):
        messages = [{"role": "system", "content": system_prompt}]
        # ---------------------------------------------
        # First branch (no history / history disabled)
        # Additionally: if shift_history == True, skip this branch
        # ---------------------------------------------
        if not shift_history and ((len(user_history) == 0 and len(assistant_history) == 0) or not use_history):
            if user_audio is not None:
                # 将音频转为 base64 data URI 格式
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, user_audio, self.SAMPLE_RATE, format='WAV', subtype='PCM_16')
                wav_buffer.seek(0)
                audio_base64 = base64.b64encode(wav_buffer.read()).decode("utf-8")
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
                    ]
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
            # 将音频转为 base64 data URI 格式
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, user_audio, self.SAMPLE_RATE, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            audio_base64 = base64.b64encode(wav_buffer.read()).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
                ]
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


    def clean_messages_for_log(self, messages):
        messages_clean = copy.deepcopy(messages)
        for msg in messages_clean:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "audio_url":
                        block["audio_url"]["url"] = "<AUDIO_BASE64_OMITTED>"
        return messages_clean

    def build_response_messages(self, user_audio):
        content = []
        history_lines = [
            "历史对话仅供 answer 判断是否需要重复或承接；transcript 不得包含历史。"
        ]
        rounds = min(len(self.user_history), len(self.assistant_history))
        for i in range(rounds):
            user_text = self.user_history[i] or "[上一轮用户音频转写为空]"
            assistant_text = self.assistant_history[i]
            history_lines.append(f"用户：{user_text}")
            history_lines.append(f"助手：{assistant_text}")
        history_lines.append("当前用户音频如下。请只输出 JSON。")
        content.append({"type": "text", "text": "\n".join(history_lines)})

        if user_audio is not None:
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, user_audio, self.SAMPLE_RATE, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            audio_base64 = base64.b64encode(wav_buffer.read()).decode("utf-8")
            content.append({
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}
            })

        return [
            {"role": "system", "content": self.RESPONSE_WITH_TRANSCRIPT_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def parse_response_with_transcript(raw: str) -> tuple[str, str]:
        text = str(raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        decoder = json.JSONDecoder()
        start = text.find("{")
        while start != -1:
            try:
                payload, _ = decoder.raw_decode(text[start:])
                if isinstance(payload, dict):
                    transcript = str(payload.get("transcript", "") or "").strip()
                    answer = str(payload.get("answer", "") or "").strip()
                    if answer:
                        return transcript, answer
            except json.JSONDecodeError:
                pass
            start = text.find("{", start + 1)
        return "", text


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
        await self.send_control("llm_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "infer_time": infer_time,
            "content": decision,
            "prompt": self.clean_messages_for_log(messages),
            "turn": turn_id,
            "state": self.STATE,
        })
        if add_to_history:
            self.assistant_history.append(str(decision))
        self.IN_SPEECH = False
        return decision

    async def async_response(self, user_audio, turn_id):
        input_path = None
        if self.output_dir is not None:
            input_path = self.output_dir / f"stream_turn{turn_id}_input.wav"
            sf.write(input_path, user_audio, self.SAMPLE_RATE)

        messages = self.build_response_messages(user_audio)
        start_t = time.perf_counter()
        raw = await asyncio.to_thread(
            llm_qwen3o,
            messages,
            response_format=self.RESPONSE_TRANSCRIPT_SCHEMA,
        )
        infer_time = round(time.perf_counter() - start_t, 3)
        transcript, answer = self.parse_response_with_transcript(raw)

        await self.send_control("llm_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "infer_time": infer_time,
            "content": answer,
            "raw_content": raw,
            "transcript": transcript,
            "prompt": self.clean_messages_for_log(messages),
            "turn": turn_id,
            "state": self.STATE,
            "response_mode": "transcript_answer",
            "structured_output": "json_schema",
            "input_path": str(input_path) if input_path else None,
        })
        await self.send_control("asr_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "state": self.STATE,
            "content": transcript,
            "source": "qwen3omni_response",
        })

        self.user_history.append(transcript)
        self.assistant_history.append(answer)
        self.BUFFER.clear()
        self.IN_SPEECH = False
        return answer

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
                            decision = await self.async_response(user_audio, self.TURN_IDX)
                            asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                            return
                        elif "yes" in shift_judge.lower(): #repeat
                            decision = await self.async_llm(self.SHIFT_RE_PROMPT, None, self.TURN_IDX, add_to_history=False, shift_history=True)
                            asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                            return
                    else:
                        decision = await self.async_response(user_audio, self.TURN_IDX)
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
                        decision = await self.async_response(user_audio, self.TURN_IDX)
                        asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                    elif "yes" in shift_judge.lower(): #repeat
                        decision = await self.async_llm(self.SHIFT_RE_PROMPT, None, self.TURN_IDX, add_to_history=False, shift_history=True)
                        asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
                else:
                    decision = await self.async_response(user_audio, self.TURN_IDX)
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
                            decision = await self.async_response(user_audio, self.TURN_IDX)
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
def create_app(prompts, delay) -> FastAPI:
    app = FastAPI()
    @app.websocket("/realtime")
    async def realtime_ws(websocket: WebSocket):
        await websocket.accept()
        msg = await websocket.receive_json()

        data = msg.get("data", {})
        exp = data.get("exp", {})
        lang = data.get("lang", {})
        engine = ConversationEngine(websocket=websocket, prompts=prompts, delay=delay)
        engine.output_dir = Path("exp") / exp / f"realtimeout_{lang}"
        engine.output_dir.mkdir(parents=True, exist_ok=True)
        await engine.run_realtime(websocket)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts_cfg = cfg.get("prompts", {})
    delay_cfg = cfg.get("time", {})
    server_cfg = cfg.get("server", {})

    host = server_cfg.get("host", {})
    port = server_cfg.get("port", {})

    app = create_app(prompts_cfg, delay_cfg)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
