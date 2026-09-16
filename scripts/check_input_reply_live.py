#!/usr/bin/env python3
"""Real Chromium + VAD + Omni + PCM/ACK; self-authored audio only.

Generate with fd-sds --make-input; run --url with a Playwright-enabled Python.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "question": "请先问我想听童话还是科幻故事，再分别介绍童话和科幻故事的特点。回复必须包含这三句话。",
    "answer": "都可以。",
}


async def make_input(output):
    sys.path.insert(0, str(ROOT / "src"))
    import module
    from demo_startup import verify_spoken_text
    output.mkdir(parents=True, exist_ok=False)
    receipt = {}
    for name, text in INPUTS.items():
        chunks = [c async for c in module.tts_omni_stream(text)]
        path = output / (name + ".wav")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(chunks[0].sample_rate)
            wav.writeframes(b"".join(c.pcm for c in chunks))
        actual = await asyncio.to_thread(module.asr, str(path))
        receipt[name] = {"text": text, "asr": actual}
        (output / "fixtures.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        verify_spoken_text(text, actual)


def trace(session):
    path = ROOT / "exp" / session / "realtimeout_live/events.jsonl"
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines()]


async def check(url, output, fixtures, early=False):
    sys.path.insert(0, str(ROOT / "src"))
    from demo_startup import verify_spoken_text
    fixture_receipt = json.loads((fixtures / "fixtures.json").read_text())
    for name in INPUTS:
        verify_spoken_text(fixture_receipt[name]["text"], fixture_receipt[name]["asr"])
        assert (fixtures / (name + ".wav")).exists()
    from playwright.async_api import async_playwright
    from check_guarded_interactions import INIT, feed, started, cancelled
    output.mkdir(parents=True, exist_ok=False)
    report = {"protocol": "played-reply-v1", "pass": False, "physical_audio": False}
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True,
            args=["--no-proxy-server", "--autoplay-policy=no-user-gesture-required"])
        context = await browser.new_context(permissions=["microphone"])
        await context.add_init_script(INIT)
        page = await context.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            await page.goto(url.rstrip("/") + "/demo/")
            await page.click("#start")
            await page.wait_for_function("document.getElementById('connection').dataset.connected === 'true'", timeout=20000)
            session = await page.locator("#session-id").text_content()
            report["session_id"] = session
            await feed(page, fixtures / "question.wav")
            await started(page, 1)
            if not early:
                # Setup speech can itself contain a real VAD pause/continuation.
                # Let its END decision settle, then anchor this test's short
                # answer to the current stream, not an already cancelled one.
                await page.wait_for_timeout(250)  # final upload + 100ms VAD END
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    rows = trace(session)
                    dispatches = [r["data"] for r in rows if r["event"] == "input_dispatch"]
                    last = dispatches[-1] if dispatches else {}
                    decided = last.get("closed") and any(r["event"] == "input_decision" and
                        r["data"]["input_id"] == last["input_id"] and
                        r["data"]["revision"] == last["revision"] for r in rows)
                    active = await page.evaluate("""() => {
                        const s=__probe.events.filter(x=>x.event==='speech_start').at(-1);
                        return s && !__probe.events.some(x=>x.event==='speech_cancelled' && x.data.utterance_id===s.data.utterance_id)
                          && __probe.acks.some(x=>x.event==='playback_progress' && x.data.started && x.data.utterance_id===s.data.utterance_id);
                    }""")
                    if decided and active:
                        break
                    await page.wait_for_timeout(100)
                assert decided and active, "Setup did not reach a settled END and current playback"
            setup = await page.evaluate("__probe")
            starts = [x for x in setup["events"] if x["event"] == "speech_start"]
            sid = starts[-1]["data"]["utterance_id"]
            initial_starts = len(starts)
            initial_cancels = len([x for x in setup["events"] if x["event"] == "speech_cancelled"])
            report.update(setup_speech_starts=initial_starts, setup_cancellations=initial_cancels)
            if early:
                await feed(page, fixtures / "answer.wav")
                await page.wait_for_function("__probe.events.some(x=>x.event==='input_ignored')", timeout=10000)
                await page.wait_for_timeout(3200)
                snapshot = await page.evaluate("__probe")
                await page.click("#stop")
                await page.wait_for_function("!document.getElementById('start').disabled")
                rows = trace(session)
                decisions = [r for r in rows if r["event"] == "input_decision" and r["data"].get("audit",{}).get("playing")]
                assert decisions and all(r["data"]["route"] == "keep" and
                    "reply_review" not in r["data"]["audit"]["route"] for r in decisions)
                assert not any(x["event"] in {"speech_cancelled", "speech_error"} for x in snapshot["events"])
                assert any(x["event"] == "playback_progress" and x["data"]["utterance_id"] == sid
                           and x["data"]["played_samples"] > 24000 for x in snapshot["acks"])
                assert not errors and not any(r["event"] == "engine_error" for r in rows)
                report.update({"pass": True, "early": True, "decisions": decisions,
                               "old_playback_continued": True, "page_errors": errors})
                return
            # Validate a real generated choice request, then wait for its full
            # PCM endpoint ACK. This text check schedules a test; production has
            # no word/question-mark heuristic.
            deadline = time.monotonic() + 25
            end = None
            while time.monotonic() < deadline:
                for r in trace(session):
                    d = r["data"]
                    if (r["event"] == "speech_sentence_end" and d["utterance_id"] == sid
                            and "童话" in d["text"] and "科幻" in d["text"]
                            and ("想" in d["text"] or "喜欢" in d["text"])):
                        end = d["end_sample"]
                        report["question_sentence"] = d["text"]
                        break
                if end is not None:
                    break
                await page.wait_for_timeout(100)
            assert end is not None, "Model did not produce the requested choice question"
            await page.wait_for_function("v=>__probe.acks.some(x=>x.event==='playback_progress' && x.data.utterance_id===v.sid && x.data.played_samples>=v.end)",
                                         arg={"sid": sid, "end": end}, timeout=15000)
            assert not await page.evaluate("__probe.events.some(x=>x.event==='turn_finished')"), "Old answer already ended"
            report["question_end_sample"] = end
            await feed(page, fixtures / "answer.wav")
            await cancelled(page, initial_cancels + 1)
            await started(page, initial_starts + 1)
            # Let the short-answer response complete, then ensure exactly one
            # successor answer. Stop if the response is unexpectedly too long.
            await page.wait_for_function("__probe.events.some(x=>x.event==='turn_finished')", timeout=45000)
            snapshot = await page.evaluate("__probe")
            await page.click("#stop")
            await page.wait_for_function("!document.getElementById('start').disabled")
            rows = trace(session)
            decisions = [r for r in rows if r["event"] == "input_decision"]
            promoted = [r for r in decisions if r["data"].get("audit",{}).get("route",{}).get("reply_review",{}).get("label") == "yield_ready"]
            assert len(promoted) == 1 and promoted[0]["data"]["route"] == "yield_ready"
            assert len([x for x in snapshot["events"] if x["event"] == "speech_start"]) == initial_starts + 1
            assert len([x for x in snapshot["events"] if x["event"] == "speech_cancelled"]) == initial_cancels + 1
            assert not errors and not any(r["event"] == "engine_error" for r in rows)
            assert not any(x["event"] == "speech_error" for x in snapshot["events"])
            report.update({"pass": True, "decisions": decisions,
                "stopped_acks": [x for x in snapshot["acks"] if x["event"] == "playback_stopped"],
                "page_errors": errors, "speech_starts": initial_starts + 1,
                "cancellations": initial_cancels + 1, "reply_responses": 1, "reply_cancellations": 1})
        except Exception as exc:
            report.update(error=str(exc), probe=await page.evaluate("window.__probe"), page_errors=errors)
            raise
        finally:
            (output / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
            await context.close()
            await browser.close()


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--make-input", action="store_true")
    p.add_argument("--url")
    p.add_argument("--fixtures", type=Path)
    p.add_argument("--early", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.make_input:
        await make_input(args.output)
    else:
        await check(args.url, args.output, args.fixtures, args.early)


if __name__ == "__main__":
    asyncio.run(main())
