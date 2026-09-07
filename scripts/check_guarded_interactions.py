#!/usr/bin/env python3
"""Live guarded Actor + Chromium interactions using only self-authored WAVs.

The synthetic mic is a MediaStream feeding the real capture worklet. Echo is a
96 ms delayed copy of the real render-reference upload, not a physical room test.
No mocks replace VAD, input routing, response, TTS, websocket or playback.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
INIT = r"""(() => {
  const p = window.__probe = {events: [], acks: [], referenceFrames: 0, echo: false};
  let mic, destination;
  navigator.mediaDevices.getUserMedia = async () => {
    mic = new AudioContext(); await mic.resume();
    destination = mic.createMediaStreamDestination();
    return destination.stream;
  };
  window.__feed = async encoded => {
    const raw = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
    const buffer = await mic.decodeAudioData(raw.buffer);
    const source = mic.createBufferSource(); source.buffer = buffer;
    source.connect(destination);
    await new Promise(resolve => { source.onended = resolve; source.start(); });
    source.disconnect();
  };
  const Native = WebSocket;
  window.WebSocket = class extends Native {
    constructor(...args) {
      super(...args); this.references = [];
      this.addEventListener('message', e => {
        if (typeof e.data === 'string') {
          const v = JSON.parse(e.data);
          if (/^(speech_|input_|demo_ready|turn_finished|asr_done)/.test(v.event)) p.events.push(v);
        }
      });
    }
    send(data) {
      if (typeof data === 'string') {
        const v = JSON.parse(data);
        if (v.event === 'playback_progress' || v.event === 'playback_stopped') p.acks.push(v);
      } else if (data instanceof ArrayBuffer) {
        const view = new DataView(data), reference = [];
        if (view.getUint32(0, true) !== 0x314d4446) throw Error('Reference protocol not negotiated');
        for (let i=0; i<256; i++) reference.push(view.getInt16(18 + i*4, true));
        if (reference.some(x => Math.abs(x) > 20)) p.referenceFrames++;
        this.references.push(reference);
        if (this.references.length > 6) {
          const delayed = this.references.shift();
          if (p.echo) for (let i=0; i<256; i++) view.setInt16(16+i*4,
            Math.max(-32768, Math.min(32767, view.getInt16(16+i*4,true) + delayed[i]*.35)), true);
        }
      }
      return super.send(data);
    }
  };
})();"""


async def feed(page, path):
    await page.evaluate("data => window.__feed(data)", base64.b64encode(path.read_bytes()).decode())


async def started(page, n):
    await page.wait_for_function("n => new Set(__probe.acks.filter(x => x.event === 'playback_progress' && x.data.started).map(x => x.data.utterance_id)).size >= n", arg=n, timeout=20000)


async def cancelled(page, n):
    await page.wait_for_function("n => __probe.events.filter(x => x.event === 'speech_cancelled').length >= n", arg=n, timeout=12000)


async def run_case(browser, url, name, output):
    context = await browser.new_context(permissions=["microphone"])
    await context.add_init_script(INIT)
    page = await context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    result = {"case": name, "pass": False}
    try:
        await page.goto(url.rstrip("/") + "/demo/")
        await page.click("#start")
        await page.wait_for_function("document.getElementById('connection').dataset.connected === 'true'", timeout=20000)
        await feed(page, ROOT / "exp/streaming_demo/synthetic_question.wav")
        await started(page, 1)
        if name == "echo_only":
            await page.evaluate("__probe.echo = true")
            await page.wait_for_function("__probe.events.some(x => x.event === 'turn_finished')", timeout=30000)
            snapshot = await page.evaluate("__probe")
            assert snapshot["referenceFrames"] >= 8, "Actual render reference did not reach the microphone worklet"
            assert not any(e["event"] == "speech_cancelled" for e in snapshot["events"])
        else:
            canary = ROOT / "exp/web_demo/guarded_turns_canary"
            await feed(page, canary / "input-2.wav")  # 等一下，我还没说完。
            await cancelled(page, 1)
            await page.wait_for_timeout(3200)  # Must outlast the old 2.5 s forced-answer path.
            assert await page.evaluate("__probe.events.filter(x => x.event === 'speech_start').length") == 1
            await feed(page, canary / "input-3.wav")  # 请用普通话说。
            await started(page, 2)
            await feed(page, canary / "input-1.wav")  # 别说了。
            await cancelled(page, 2)
            await page.wait_for_timeout(3200)
            snapshot = await page.evaluate("__probe")
            assert len([e for e in snapshot["events"] if e["event"] == "speech_start"]) == 2
            cancellations = [e["data"]["reason"] for e in snapshot["events"] if e["event"] == "speech_cancelled"]
            assert cancellations == ["accepted_interrupt", "accepted_interrupt"], cancellations
            assert {"awaiting_user", "stop_only"} <= {e["data"]["reason"] for e in snapshot["events"] if e["event"] == "input_waiting"}
        assert not errors, errors
        assert not any(e["event"] == "speech_error" for e in snapshot["events"])
        await page.screenshot(path=str(output / (name + ".png")))
        session = next(e["data"]["session_id"] for e in snapshot["events"] if e["event"] == "demo_ready")
        await page.click("#stop")
        await page.wait_for_function("!document.getElementById('start').disabled")
        trace_path = ROOT / "exp" / session / "realtimeout_live/events.jsonl"
        rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
        evidence = [r["data"] for r in rows if r["event"] == "input_evidence" and r["data"]["echo_only"]]
        if name == "echo_only":
            assert evidence, "No actual Actor-side echo rejection observed"
        result.update(pass_=True, session_id=session, reference_frames=snapshot["referenceFrames"],
            echo_rejected_windows=len(evidence), events=snapshot["events"],
            stopped_acks=[a for a in snapshot["acks"] if a["event"] == "playback_stopped"],
            input_decisions=[r for r in rows if r["event"] in {"input_decision", "input_rejected"}],
            page_errors=errors)
        result["pass"] = result.pop("pass_")
    except Exception as exc:
        result.update(error=str(exc), probe=await page.evaluate("window.__probe"), page_errors=errors)
        raise
    finally:
        (output / (name + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        await context.close()
    return {k: v for k, v in result.items() if k not in {"events", "input_decisions"}}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"version": "guarded-turns-v1", "physical_acoustic_test": False, "cases": [], "pass": False}
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True,
                args=["--no-proxy-server", "--autoplay-policy=no-user-gesture-required"])
            try:
                for name in ["echo_only", "wait_continue_stop"]:
                    result = await run_case(browser, args.url, name, args.output)
                    report["cases"].append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                report["pass"] = True
            finally:
                await browser.close()
    finally:
        (args.output / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
