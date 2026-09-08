#!/usr/bin/env python3
"""Self-authored long-answer reference smoke; never reuses a user's recording.

fd-sds Python: --make-input --output NEW_DIR (real verbatim TTS).
Playwright Python: --url http://127.0.0.1:18000 --output SAME_DIR.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "请讲一个五百字以上的长故事，分成至少十句话，不要只讲开头。"


async def make_input(output):
    sys.path.insert(0, str(ROOT / "src"))
    import module
    output.mkdir(parents=True, exist_ok=False)
    chunks = [chunk async for chunk in module.tts_omni_stream(QUESTION)]
    assert chunks and len({c.sample_rate for c in chunks}) == 1
    with wave.open(str(output / "question.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(chunks[0].sample_rate)
        wav.writeframes(b"".join(c.pcm for c in chunks))


async def check(url, output):
    from playwright.async_api import async_playwright
    from check_guarded_interactions import INIT, feed, started, cancelled
    if (output / "receipt.json").exists():
        raise ValueError("Never overwrite an existing receipt")
    report = {"version": "speech-reference-v1", "question": QUESTION,
              "physical_listening": False, "formal_benchmark": False, "pass": False}
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True,
                args=["--no-proxy-server", "--autoplay-policy=no-user-gesture-required"])
            try:
                context = await browser.new_context(permissions=["microphone"])
                await context.add_init_script(INIT)
                page = await context.new_page()
                errors = []
                page.on("pageerror", lambda exc: errors.append(str(exc)))
                await page.goto(url.rstrip("/") + "/demo/")
                await page.click("#start")
                await page.wait_for_function("document.getElementById('connection').dataset.connected === 'true'")
                await feed(page, output / "question.wav")
                await started(page, 1)
                ignored = await page.evaluate("__probe.events.filter(e => e.event === 'input_ignored').length")
                await feed(page, ROOT / "exp/web_demo/guarded_turns_canary/input-4.wav")
                await page.wait_for_function("n => __probe.events.filter(e => e.event === 'input_ignored').length > n",
                                             arg=ignored, timeout=10000)
                assert await page.evaluate("__probe.events.filter(e => e.event === 'speech_cancelled').length") == 0
                await feed(page, ROOT / "exp/web_demo/guarded_turns_canary/input-1.wav")
                await cancelled(page, 1)
                await page.wait_for_function("__probe.acks.some(e => e.event === 'playback_stopped')", timeout=5000)
                session = await page.locator("#session-id").text_content()
                snapshot = await page.evaluate("__probe")
                await page.click("#stop")
                await context.close()
                trace_path = ROOT / "exp" / session / "realtimeout_live/events.jsonl"
                rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
                cases = []
                for path in (ROOT / "exp/demo_cases/captures").glob("*/case.json"):
                    case = json.loads(path.read_text())
                    if (case["context"].get("session_id") == session and case["kind"] == "input_route"
                            and case["context"].get("playing")):
                        metadata = json.loads(case["request"]["messages"][1]["content"][0]["text"].split("\n")[0].removeprefix("情境资料："))
                        cases.append({"case_id": case["case_id"], "input_id": case["context"]["input_id"],
                            "reference": metadata["assistant_reference_text"],
                            "source": case["context"].get("reference_kind"),
                            "outcome": case["outcome"]["text"]})
                dispatch = [r for r in rows if r["event"] == "input_dispatch" and r["data"]["playing"]]
                assert dispatch and all(r["data"]["reference_chars"] > 0 for r in dispatch)
                assert not any(r["event"] == "speech_text_done" and r["server_ms"] < dispatch[0]["server_ms"] for r in rows)
                assert len(cases) >= 2 and all(c["reference"] and c["source"] == "playback_sentence_window" for c in cases)
                assert "keep" in {c["outcome"] for c in cases} and "stop_only" in {c["outcome"] for c in cases}
                assert not errors and not any(r["event"] == "engine_error" for r in rows)
                report.update(session_id=session, input_route_cases=cases, text_done_pending_at_barge_in=True,
                    backchannel_kept_playing=True, explicit_stop_cancelled=True,
                    stopped_ack=[e["data"] for e in snapshot["acks"] if e["event"] == "playback_stopped"],
                    page_errors=errors, **{"pass": True})
            finally:
                await browser.close()
    finally:
        (output / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-input", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    args = parser.parse_args()
    asyncio.run(make_input(args.output) if args.make_input else check(args.url, args.output))


if __name__ == "__main__":
    main()
