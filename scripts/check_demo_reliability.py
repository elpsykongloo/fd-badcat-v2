#!/usr/bin/env python3
"""Serial self-authored speech + real browser/Actor/Omni fault diagnostics.

No real-human/physical-audio or formal latency/accuracy claim. HTTP faults are
injected only into a private loopback TTS forwarder owned by this process; the
production service/config is never modified. Raw artifacts stay private.
"""
import argparse
import asyncio
import base64
from contextlib import aclosing
import copy
import io
import json
import os
from pathlib import Path
import socket
import sys
import time

import aiohttp
from aiohttp import web
import numpy as np
import soundfile as sf
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import module
from backend import create_app, load_runtime_config
from speech_stream import AudioHealth
from spoken_text import SpokenTextBuffer
from tts_sentence import StreamingSentenceBuffer
from check_demo_continuity_live import MIC
from control_labels import decide_input_route
from guarded_turns import route_messages, CONTINUATION_PROTOCOL
from messages import build_audio_content

VOICE = {"speaker": "chelsie", "seed": 42}
QUESTION = "请告诉我两个学习语言的小建议。"
REPLY = "这里有两个简单的建议。\n1. **表达清楚**：先想清楚要说什么，再用简短的句子表达。\n2. **每天练习**：每天大声朗读几句，再试着用自己的话复述。"
RAW_TEXTS = [
    "**时态**：时态表示动作发生的时间。",
    "时态：时态表示动作发生的时间。",
    "## Tips\n1. **Clarity**: Use short sentences.\n2. **Practice**: Read them aloud.",
    "要点如下。\n1. **语序**：语序影响句子的意思。\n2. **练习**：先想清楚要表达什么，再慢慢说出来。",
    "The story ends here. 🌫️💡\nGood night.",
]
CONTEXT_PROBES = [
    ("unfinished_en", "I would like to know", "which planet is closest to the Sun?", "yield_ready", False),
    ("unplayed_zh", "请介绍一下火星。", "重点说它的地貌。", "yield_ready", False),
    # The first probe was heard as "停一下，先不要回答" and labeled wait.
    # Both defer the reply; retain the old strict mismatch separately. An
    # unambiguous termination below still must clear via stop_only.
    ("defer_ambiguous", "我想知道", "停下，先不要回答。", ("stop_only", "yield_wait"), False),
    ("latest_stop", "我想知道", "停止说话，不用回答了。", "stop_only", False),
    ("latest_wait", "我想知道", "等一下，我还没说完。", "yield_wait", False),
    ("playing_fragment", "What are the main features", "of a comet?", "yield_ready", True),
    ("playing_backchannel", "请介绍一下火星。", "嗯嗯，我在听。", "keep", True),
    ("playing_new_question", "请介绍一下火星。", "现在请告诉我二加二等于几。", "yield_ready", True),
    ("playing_stop", "请介绍一下火星。", "请停止说话，现在先不要继续讲了。", "stop_only", True),
]


def save(path, value):
    with path.open("w", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


async def synthesize(text):
    chunks = []
    health = AudioHealth()
    began = time.perf_counter()
    async with aclosing(module.tts_omni_stream(text, voice_control=VOICE, timing=True)) as source:
        async for chunk in source:
            chunks.append(chunk)
            health.feed(chunk.pcm, chunk.sample_rate)
    assert chunks
    pcm = b"".join(c.pcm for c in chunks)
    rate = chunks[0].sample_rate
    return pcm, rate, {"text": text, "audio_ms": len(pcm) / 2 / rate * 1000,
        "wall_ms": (time.perf_counter() - began) * 1000, **health.summary()}


class FaultForwarder:
    def __init__(self, upstream):
        self.upstream = upstream
        self.mode = "normal"
        self.calls = []
        self.injected = 0

    async def handle(self, request):
        payload = await request.json()
        index = len(self.calls) + 1
        row = {"index": index, "request_id": request.headers.get("x-request-id"),
               "payload": copy.deepcopy(payload), "injected": None}
        self.calls.append(row)
        inject = (self.mode == "recover" and index == 2) or (self.mode == "fail" and index >= 2) or self.mode == "fail_private"
        if inject:
            self.injected += 1
            row["injected"] = "http_502_before_pcm"
            return web.json_response({"error": "self-authored injected 502"}, status=502,
                                     headers={"X-Request-Id": request.headers.get("x-request-id", "fault")})
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=90)) as client:
            async with client.post(self.upstream, json=payload,
                    headers={"X-Request-Id": request.headers.get("x-request-id", "fault")}) as upstream:
                result = web.StreamResponse(status=upstream.status,
                    headers={"Content-Type": upstream.headers.get("Content-Type", "application/json")})
                await result.prepare(request)
                proof_seen = audio_seen = False
                async for line in upstream.content:
                    await result.write(line)
                    if self.mode == "partial" and index == 2 and line.startswith(b"data:"):
                        try:
                            obj = json.loads(line[5:])
                        except (ValueError, UnicodeDecodeError):
                            continue
                        proof_seen |= obj.get("fd_text_finish_reason") == "stop"
                        audio_seen |= obj.get("modality") == "audio" and any(
                            (choice.get("delta") or {}).get("content") for choice in obj.get("choices", []))
                        if proof_seen and audio_seen:
                            await result.write(b"\n")
                            row["injected"] = "eof_after_text_verified_audio_without_done"
                            self.injected += 1
                            break
                await result.write_eof()
                return result


async def browser_case(browser, cfg, encoded, output, scenario, forwarder):
    settings = {**cfg["engine"], "stream_response": True, "warmup": False,
        "diagnostics_retention_days": None, "case_capture_dir": str(output / scenario / "cases")}
    prompts = dict(cfg["prompts"])
    # Real model, fixed self-authored requested delivery. This is a transport /
    # canonicalization probe, not an open-domain content-accuracy measurement.
    prompts["response"] = "你在进行自造测试。无论音频问什么，只逐字输出下面的两点建议，不增删内容：\n" + REPLY
    delay = dict(cfg["time"])
    if scenario == "fail_private":
        # Deterministic private-state coverage only. Not the production 640ms
        # timing: leave enough hold for both injected failures before publish.
        delay["end_hold_frame"] = 3.0
    app = create_app(prompts, delay, cfg["llm"], settings, cfg["asr"])
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    context = await browser.new_context(permissions=["microphone"])
    await context.add_init_script(MIC)
    page = await context.new_page()
    row = {"scenario": scenario, "events": [], "client": [], "page_errors": [], "passed": False}
    row["end_hold_s"] = delay["end_hold_frame"]
    page.on("pageerror", lambda error: row["page_errors"].append(str(error)))
    def connected(ws):
        ws.on("framereceived", lambda payload: row["events"].append(json.loads(payload)) if isinstance(payload, str) else None)
        ws.on("framesent", lambda payload: row["client"].append(json.loads(payload)) if isinstance(payload, str) else None)
    page.on("websocket", connected)
    def events(name):
        return [e["data"] for e in row["events"] if e["event"] == name]
    async def until(predicate, timeout=120):
        async def poll():
            while not predicate():
                if row["page_errors"]:
                    raise RuntimeError(row["page_errors"])
                await asyncio.sleep(.05)
        await asyncio.wait_for(poll(), timeout)
    forwarder.mode, forwarder.calls, forwarder.injected = scenario, [], 0
    try:
        await until(lambda:server.started, 30)
        await page.goto(f"http://127.0.0.1:{port}/demo/")
        await page.click("#start")
        await until(lambda:bool(events("demo_ready")), 30)
        row["session_id"] = events("demo_ready")[0]["session_id"]
        await page.evaluate("s=>window.__inject(s)", encoded)
        if scenario in {"normal", "recover"}:
            await until(lambda:bool(events("speech_played")))
            assert not events("speech_error"), events("speech_error")
            public = "".join(e["text"] for e in events("speech_text_delta"))
            assert "**" not in public and "1." not in public and "2." not in public
            assert "语音合成失败" not in await page.locator("#notice").inner_text()
            assert "播放完成" in await page.locator("#messages").inner_text()
        else:
            await until(lambda:bool(events("speech_error")))
            if scenario != "fail_private":
                await until(lambda:bool(events("speech_cancelled")))
            assert events("speech_error")[-1]["code"] == "tts_unavailable"
            if scenario != "fail_private":
                assert "回答未完成" in await page.locator("#messages").inner_text()
            assert "语音合成失败" in await page.locator("#notice").inner_text()
            assert not events("speech_played")
        assert forwarder.injected == {"normal": 0, "recover": 1, "fail": 2, "partial": 1, "fail_private": 2}[scenario]
        if scenario in {"recover", "fail"}:
            assert forwarder.calls[1]["payload"] == forwarder.calls[2]["payload"]
            assert forwarder.calls[1]["request_id"] != forwarder.calls[2]["request_id"]
        if scenario == "partial":
            assert len(forwarder.calls) == 2, "Partially emitted sentence was retried"
        if scenario == "fail_private":
            assert len(forwarder.calls) == 2, "Private failure restarted the entire reply"
            assert not events("speech_text_delta") and not events("speech_first_audio")
        await page.click("#stop")
        await page.wait_for_function("window.__tracks.every(t=>t.readyState==='ended')")
        await page.evaluate("window.__micContext.close()")
        await page.screenshot(path=str(output / f"{scenario}.png"), full_page=True)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        await context.close()
        server.should_exit = True
        await task
        listener.close()
        row["tts_requests"] = copy.deepcopy(forwarder.calls)
        if row.get("session_id"):
            trace = ROOT / "exp" / row["session_id"] / "realtimeout_live/events.jsonl"
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            row["trace_path"] = str(trace)
            row["recovery"] = [e["data"] for e in records if e["event"] == "tts_recovery"]
            row["audio_health"] = [e["data"] for e in records if e["event"] == "speech_timing"
                                   and e["data"].get("phase") == "tts_complete"]
            row["closed"] = records[-1]["event"] == "trace_closed" and records[-1].get("dropped", 0) == 0
            row["failed_calls"] = [e["data"] for e in records if e["event"] == "model_call_done" and e["data"].get("status") == "error"]
            if scenario == "partial" and not any(e.get("output_samples", 0) > 0 for e in row["failed_calls"]):
                row["error"] = "Fault did not reach decoded/verified PCM; not a post-output no-replay test"
        row["passed"] = not row.get("error") and bool(row.get("closed")) and not row["page_errors"]
        save(output / f"{scenario}.json", row)
    return row


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chromium", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:10003/v1/chat/completions")
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    cfg = load_runtime_config(ROOT / "src/config.yaml", True)
    module.configure_asr(cfg["asr"])
    module.QWEN_URL = module.OMNI_TTS_URL = args.url
    receipt = {"version": "demo-reliability-v1", "scope": __doc__, "serial": True,
               "physical_audio": False, "independent_holdout": False, "raw": [],
               "context_probes": [], "browser": [], "passed": False}
    save(args.output / "protocol.json", {"texts": RAW_TEXTS, "reply": REPLY,
        "scenarios": ["normal", "recover", "fail", "partial", "fail_private"], "context_probes": CONTEXT_PROBES, "url": args.url,
        "scope": __doc__, "no_production_writes": True})
    try:
        for index, raw in enumerate(RAW_TEXTS):
            n = SpokenTextBuffer()
            canonical = "".join(n.feed(c) for c in raw) + n.flush()
            s = StreamingSentenceBuffer()
            sentences = [v for v in s.feed(canonical) + s.flush() if v.strip()]
            row = {"index": index, "raw": raw, "canonical": canonical, "sentences": []}
            receipt["raw"].append(row)
            for sentence in sentences:
                _, _, result = await synthesize(sentence)
                row["sentences"].append(result)
            print(json.dumps({"raw_index": index, "audio_ms": sum(s["audio_ms"] for s in row["sentences"]),
                "max_low_energy_ms": max(s["longest_low_energy_ms"] for s in row["sentences"])}, ensure_ascii=False), flush=True)
            save(args.output / "receipt.json", receipt)
        # Separate model contract probes, not browser or speech-source accuracy.
        # Reuse only identical self-authored TTS inputs; every route call is live.
        from scipy.signal import resample_poly
        import math
        probe_audio = {}
        for name, prefix, suffix, expected, playing in CONTEXT_PROBES:
            for text in (prefix, suffix):
                if text not in probe_audio:
                    pcm, rate, _ = await synthesize(text)
                    gcd = math.gcd(rate, 16000)
                    probe_audio[text] = resample_poly(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768,
                                                     16000 // gcd, rate // gcd)
            first = np.concatenate([probe_audio[prefix], np.zeros(1600, dtype=np.float32)])
            audio = np.concatenate([first, probe_audio[suffix]])
            context_content = build_audio_content(first, 16000) if playing else None
            metadata = {"protocol": CONTINUATION_PROTOCOL, "current_start_ms": len(first) / 16,
                        "source": "recent_replied_user_audio" if playing else "accepted_unanswered_user_audio"}
            if playing:
                metadata.update(route_only=True, transcript_scope="new_audio_only", audio_layout="two_blocks")
            messages = route_messages(cfg["prompts"]["input_route"],
                build_audio_content(probe_audio[suffix] if playing else audio, 16000),
                playing=playing,
                continuation=metadata, context_content=context_content)
            async def call(msgs, stage):
                return "".join([p async for p in module.llm_qwen3o_stream(msgs, route=True)])
            label, audit = await decide_input_route(call, messages, 2, playing=playing, closed=True)
            row = {"name": name, "expected": expected, "actual": label, "audit": audit,
                   "passed": label in (expected if isinstance(expected, tuple) else (expected,)) and not audit["fallback"]}
            receipt["context_probes"].append(row)
            save(args.output / "receipt.json", receipt)
            print(json.dumps({k: row[k] for k in ("name", "expected", "actual", "passed")}), flush=True)
        pcm, rate, _ = await synthesize(QUESTION)
        mic = io.BytesIO()
        sf.write(mic, np.frombuffer(pcm, dtype="<i2"), rate, format="WAV", subtype="PCM_16")
        encoded = base64.b64encode(mic.getvalue()).decode()
        # Warm CPU ASR once, never create a source-corpus file.
        mic.seek(0)
        await asyncio.to_thread(module.asr, mic)
        fault = FaultForwarder(args.url)
        app = web.Application()
        app.router.add_post("/chat", fault.handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        module.OMNI_TTS_URL = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/chat"
        from playwright.async_api import async_playwright
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, executable_path=str(args.chromium),
                    args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required", "--no-proxy-server"])
                try:
                    for scenario in ("normal", "recover", "fail", "partial", "fail_private"):
                        row = await browser_case(browser, cfg, encoded, args.output, scenario, fault)
                        receipt["browser"].append({k:row[k] for k in ("scenario", "passed", "session_id") if k in row})
                        print(json.dumps({"scenario": scenario, "passed": row["passed"], "error": row.get("error")}, ensure_ascii=False), flush=True)
                        save(args.output / "receipt.json", receipt)
                finally:
                    await browser.close()
        finally:
            await runner.cleanup()
            module.OMNI_TTS_URL = args.url
        receipt["passed"] = all(r["passed"] for r in receipt["browser"] + receipt["context_probes"]) and all(
            s["longest_low_energy_ms"] < 5000 for r in receipt["raw"] for s in r["sentences"])
    finally:
        save(args.output / "receipt.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("Reliability diagnostic failed; all attempts retained")


if __name__ == "__main__":
    asyncio.run(main())
