#!/usr/bin/env python3
"""Real Chromium UI/WebAudio checks. Synthetic protocol fixture by default.

Install playwright + aiohttp in an isolated environment, then:
  python -m playwright install chromium --only-shell
  python scripts/check_web_demo.py --output exp/web_demo/browser_check
Optional live model smoke (only your own audio, no evaluation corpus):
  ... --live-url http://127.0.0.1:18000 --audio OWN_INPUT.wav --output NEW_DIR
No subjective listening/physical speaker latency claim is made by this script.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import wave

from aiohttp import web, WSMsgType
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/static"
TRACKS = """(() => {
  window.__tracks = [];
  const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (...args) => {
    const stream = await original(...args);
    window.__tracks.push(...stream.getTracks());
    return stream;
  };
})();"""


class Fixture:
    """Deliberate sine-tone fixture, never a fallback in the production app."""
    def __init__(self):
        self.ws = None
        self.frames = []
        self.progress = []
        self.closed = 0
        self.sessions = 0
        self.streaming = True

    async def event(self, name, **data):
        await self.ws.send_json({"event": name, "data": data})

    async def respond(self):
        await self.event("speech_start", utterance_id=1, turn=0, buffer_ms=600, protocol="pcm16.v1")
        await self.event("speech_text_delta", utterance_id=1, text="这是一条协议测试消息。")
        # ASR arrives after response text; DOM insertion must still precede it.
        await self.event("asr_done", turn=0, content="<img src=x onerror=alert(1)> 测试转写")
        for seq in range(5):
            samples = [int(4000 * math.sin((seq * 960 + i) * math.tau * 440 / 24000)) for i in range(960)]
            await self.ws.send_bytes(struct.pack("<4sIII", b"FDS1", 1, seq, 24000) + struct.pack("<960h", *samples))
        await self.event("speech_audio_end", utterance_id=1, samples=4800, rate=24000, packets=5)

    async def websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws = ws
        config = await ws.receive_json()
        assert config["data"] == {"client": "humdial-web", "audio_protocol": "pcm16.v1"}
        self.sessions += 1
        await self.event("demo_ready", protocol="pcm16.v1", session_id=f"synthetic-ui-{self.sessions}")
        count = 0
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                assert len(msg.data) == 1024, "256 Float32 mono samples required"
                self.frames.append(msg.data)
                count += 1
                if count == 4:
                    await self.respond()
            elif msg.type == WSMsgType.TEXT:
                self.progress.append(json.loads(msg.data))
        self.closed += 1
        return ws

    @asynccontextmanager
    async def server(self):
        app = web.Application()
        async def info(request):
            return web.json_response({"protocol": "pcm16.v1", "streaming": self.streaming})
        async def index(request):
            return web.FileResponse(STATIC / "index.html")
        app.router.add_get("/api/demo/info", info)
        app.router.add_get("/realtime", self.websocket)
        app.router.add_get("/demo/", index)
        app.router.add_static("/demo/", STATIC)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            if self.ws is not None and not self.ws.closed:
                await self.ws.close()
            await runner.cleanup()


async def connected(page):
    await page.wait_for_function("document.getElementById('connection').dataset.connected === 'true' || (!document.getElementById('notice').hidden && !document.getElementById('start').disabled)", timeout=40000)
    assert await page.locator("#connection").get_attribute("data-connected") == "true", await page.locator("#notice").inner_text()


async def stopped(page):
    await page.wait_for_function("!document.getElementById('start').disabled")
    await page.wait_for_function("window.__tracks.every(t => t.readyState === 'ended')")


async def mock_checks(browser, output):
    fixture = Fixture()
    async with fixture.server() as url:
        context = await browser.new_context(permissions=["microphone"], viewport={"width": 1440, "height": 1050}, reduced_motion="reduce")
        await context.add_init_script(TRACKS)
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.goto(url + "/demo/")
        await page.screenshot(path=str(output / "desktop.png"), full_page=True)
        assert await page.locator("h1").inner_text() == "自然接话，随时打断。"
        assert not await page.evaluate("window.__tracks.length"), "page load must not acquire microphone"
        await page.set_viewport_size({"width": 390, "height": 844})
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "mobile horizontal overflow"
        await page.screenshot(path=str(output / "mobile.png"), full_page=True)
        await page.set_viewport_size({"width": 1366, "height": 768})
        stop_button = await page.locator("#stop").bounding_box()
        assert stop_button["y"] + stop_button["height"] < 768, "laptop controls should be above the fold"
        await page.screenshot(path=str(output / "laptop.png"), full_page=True)
        await page.set_viewport_size({"width": 1440, "height": 1050})
        await page.click("#start")
        await connected(page)
        await page.wait_for_function("document.querySelector('.message-tag') !== null && document.getElementById('messages').innerText.includes('播放完成')")
        assert await page.locator("#messages img").count() == 0, "ASR text must not be interpreted as HTML"
        assert await page.locator(".message").first.get_attribute("class") == "message user"
        assert any(p["data"].get("ended") and p["data"].get("played_samples") == 4800 for p in fixture.progress)
        assert await page.evaluate("window.__tracks[0].readyState") == "live"
        await page.click("#mute")
        await page.wait_for_timeout(180)
        assert await page.evaluate("window.__tracks[0].enabled") is False
        assert fixture.frames[-1] == bytes(1024), "mute must upload silence while preserving clock"
        await page.click("#mute")
        assert await page.evaluate("window.__tracks[0].enabled") is True
        await fixture.event("speech_start", utterance_id=2, turn=1, buffer_ms=600, protocol="pcm16.v1")
        await fixture.event("speech_text_delta", utterance_id=2, text="这条回复将被取消。")
        await fixture.event("speech_cancelled", utterance_id=2, reason="shot_interrupt")
        await fixture.ws.send_bytes(struct.pack("<4sIII", b"FDS1", 2, 0, 24000) + bytes(1920))
        await page.wait_for_function("document.getElementById('messages').innerText.includes('已打断')")
        interrupts = await page.locator("#interrupts").text_content()
        assert interrupts.endswith("/ 1"), interrupts
        await page.screenshot(path=str(output / "conversation-fixture.png"), full_page=True)
        await page.click("#stop")
        await stopped(page)
        await page.click("#clear")
        assert await page.locator(".message").count() == 0
        # Reconnect creates a fresh session. Server-side close releases devices.
        await page.click("#start")
        await connected(page)
        assert fixture.sessions == 2
        await fixture.ws.close()
        await stopped(page)
        assert "断开" in await page.locator("#notice").inner_text()
        # The stream-disabled gate runs before permission / media acquisition.
        fixture.streaming = False
        before = await page.evaluate("window.__tracks.length")
        await page.click("#start")
        await stopped(page)
        assert "未启用" in await page.locator("#notice").inner_text()
        assert await page.evaluate("window.__tracks.length") == before
        fixture.streaming = True
        # Pending permission cancellation, then a new session, then old result.
        await page.evaluate("""() => {
          const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
          let first = true;
          navigator.mediaDevices.getUserMedia = async (...args) => {
            if (first) { first = false; await new Promise(resolve => { window.__permission = resolve; }); }
            return original(...args);
          };
        }""")
        await page.click("#start")
        await page.wait_for_function("typeof window.__permission === 'function'")
        await page.click("#stop")
        await page.click("#start")
        await connected(page)
        await page.evaluate("window.__permission()")
        await page.wait_for_function("window.__tracks.length >= 4 && window.__tracks.at(-1).readyState === 'ended'")
        assert await page.evaluate("window.__tracks.at(-2).readyState") == "live", "late old permission must not stop new session"
        await page.click("#stop")
        await stopped(page)
        # Explicit denied permission gives a recoverable, human-readable error.
        await page.evaluate("() => { navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('denied', 'NotAllowedError')); }")
        await page.click("#start")
        await stopped(page)
        assert "权限" in await page.locator("#notice").inner_text()
        assert not errors, errors
        await context.close()
        return {"kind": "synthetic protocol fixture + real Chromium WebAudio/AudioWorklet",
                "checks": ["desktop/mobile layout", "1366x768 laptop controls above fold", "no mic before start", "256xfloat32 upload", "PCM playback completion ACK",
                           "late ASR ordering", "text XSS escaping", "mute sends silence", "cancel + stale audio fence",
                           "reconnect", "server disconnect cleanup", "stream-disabled preflight", "cancelled permission race",
                           "permission denied", "zero uncaught JS errors"],
                "microphone_frames": len(fixture.frames), "sessions": fixture.sessions, "page_errors": errors}


async def live_check(browser, url, audio, output):
    context = await browser.new_context(permissions=["microphone"], viewport={"width": 1440, "height": 1050}, reduced_motion="reduce")
    await context.add_init_script(TRACKS)
    page = await context.new_page()
    errors, controls, playback_ack = [], [], []
    binary_packets = 0
    uploads = {"frames": 0, "max_rms": 0.0}
    page.on("pageerror", lambda error: errors.append(str(error)))
    def receive(payload):
        nonlocal binary_packets
        if isinstance(payload, bytes):
            binary_packets += 1
        else:
            value = json.loads(payload)
            # Do not archive giant prompt/audio blobs from llm_done.
            if value.get("event", "").startswith(("speech_", "demo_", "vad_")) or value.get("event") == "asr_done":
                controls.append(value)
    def sent(payload):
        if isinstance(payload, bytes):
            uploads["frames"] += 1
            samples = struct.unpack("<" + "f" * (len(payload) // 4), payload)
            uploads["max_rms"] = max(uploads["max_rms"], math.sqrt(sum(x * x for x in samples) / len(samples)))
        else:
            value = json.loads(payload)
            if value.get("event") == "playback_progress":
                playback_ack.append(value["data"])
    def websocket(ws):
        ws.on("framereceived", receive)
        ws.on("framesent", sent)
    page.on("websocket", websocket)
    try:
        await page.goto(url.rstrip("/") + "/demo/")
        await page.click("#start")
        await connected(page)
        await page.wait_for_function("document.getElementById('messages').innerText.includes('播放完成')", timeout=60000)
        await page.click("#mute")
        await page.wait_for_function("document.querySelector('.message.user .message-text:not(.pending)') !== null", timeout=20000)
        await page.screenshot(path=str(output / "live-conversation.png"), full_page=True)
        snapshot = {"first_text": await page.locator("#first-text").text_content(),
                    "first_audio": await page.locator("#first-audio").text_content(),
                    "buffer": await page.locator("#buffer").text_content(),
                    "session_id": await page.locator("#session-id").text_content(),
                    "audio_format": await page.locator("#audio-format").text_content()}
        await page.click("#stop")
        await stopped(page)
        assert binary_packets > 0 and any(p.get("ended") for p in playback_ack)
        assert not errors, errors
        return {"kind": "live ActorEngine + Omni, Chromium virtual microphone/speaker; not physical listening",
                "input": str(audio),
                "binary_packets": binary_packets, "last_playback_ack": playback_ack[-1],
                "ui_snapshot": snapshot, "uploads": uploads, "events": controls, "page_errors": errors}
    except Exception:
        debug = {"uploads": uploads, "events": controls, "page_errors": errors,
                 "ui": await page.locator("body").inner_text()}
        (output / "live_failure.json").write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")
        await page.screenshot(path=str(output / "live_failure.png"), full_page=True)
        raise
    finally:
        await context.close()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite prior evidence")
    parser.add_argument("--live-url", default=None)
    parser.add_argument("--audio", type=Path, help="Your own WAV input, required for --live-url")
    args = parser.parse_args()
    if args.live_url and (not args.audio or not args.audio.is_file()):
        parser.error("--live-url requires an existing --audio WAV owned by you")
    args.output.mkdir(parents=True, exist_ok=False)
    launch_args = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--no-proxy-server"]
    if args.live_url:
        # Chrome loops file capture by default. A repeated short question may
        # never meet EoU hold; feed it ONCE, with time for the initial handshake.
        prepared = args.output / "microphone-input.wav"
        with wave.open(str(args.audio), "rb") as source:
            params = source.getparams()
            pcm = source.readframes(source.getnframes())
        if params.nchannels != 1 or params.sampwidth != 2:
            parser.error("--audio must be mono PCM16 WAV")
        with wave.open(str(prepared), "wb") as destination:
            destination.setparams(params)
            silence = bytes(params.framerate * params.sampwidth)
            destination.writeframes(silence + pcm + silence * 2)
        launch_args.append("--use-file-for-fake-audio-capture=" + str(prepared.resolve()) + "%noloop")
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "physical_listening": False,
               "formal_benchmark": False, "demo_version": "web-demo-v1",
               "sources": [str(p.relative_to(ROOT)) for p in sorted(STATIC.iterdir()) if p.is_file()]}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, args=launch_args)
        receipt["browser"] = browser.version
        try:
            if args.live_url:
                receipt["result"] = await live_check(browser, args.live_url, args.audio.resolve(), args.output)
            else:
                receipt["result"] = await mock_checks(browser, args.output)
            receipt["pass"] = True
        except Exception as exc:
            receipt["pass"] = False
            receipt["error"] = str(exc)
            for context in browser.contexts:
                for page in context.pages:
                    if not page.is_closed():
                        receipt["failure_ui"] = await page.locator("body").inner_text()
                        await page.screenshot(path=str(args.output / "failure.png"), full_page=True)
            raise
        finally:
            await browser.close()
            (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
