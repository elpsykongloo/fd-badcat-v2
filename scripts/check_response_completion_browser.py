#!/usr/bin/env python3
"""Real browser/Actor/Omni completion and exhaustion with synthetic microphone.

Uses isolated loopback backends with small response budgets, current demo
prompts, real model and TTS. No physical audio or latency benchmark claims.
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import socket
import sys

import uvicorn
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import module
from backend import create_app, load_runtime_config
from check_demo_continuity_live import MIC


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--chromium", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = load_runtime_config(ROOT / "src/config.yaml", True)
    module.configure_asr(cfg["asr"])
    encoded = base64.b64encode(args.audio.read_bytes()).decode()
    receipt = {"scope": __doc__, "physical_audio": False, "serial": True, "cases": [], "passed": False}
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, executable_path=str(args.chromium),
                args=["--no-sandbox", "--autoplay-policy=no-user-gesture-required", "--no-proxy-server"])
            for scenario, limit in [("completed", 6), ("exhausted", 0)]:
                settings = {**cfg["engine"], "stream_response": True, "warmup": False,
                    "response_max_tokens": 64, "response_max_continuations": limit,
                    "diagnostics_retention_days": None,
                    "case_capture_dir": str(args.output / scenario / "cases")}
                app = create_app(cfg["prompts"], cfg["time"], cfg["llm"], settings, cfg["asr"])
                listener = socket.socket()
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
                task = asyncio.create_task(server.serve(sockets=[listener]))
                context = await browser.new_context(permissions=["microphone"])
                await context.add_init_script(MIC)
                page = await context.new_page()
                row = {"scenario": scenario, "events": [], "client": [], "page_errors": [], "passed": False}
                receipt["cases"].append(row)
                page.on("pageerror", lambda error: row["page_errors"].append(str(error)))

                def connected(ws):
                    ws.on("framereceived", lambda payload: row["events"].append(json.loads(payload))
                          if isinstance(payload, str) else None)
                    ws.on("framesent", lambda payload: row["client"].append(json.loads(payload))
                          if isinstance(payload, str) else None)

                page.on("websocket", connected)

                def events(name):
                    return [r["data"] for r in row["events"] if r["event"] == name]

                async def until(predicate, timeout=180):
                    async def poll():
                        while not predicate():
                            if row["page_errors"]:
                                raise RuntimeError(row["page_errors"])
                            await asyncio.sleep(.05)
                    await asyncio.wait_for(poll(), timeout)

                try:
                    await until(lambda: server.started, 30)
                    await page.goto(f"http://127.0.0.1:{port}/demo/")
                    await page.click("#start")
                    await until(lambda: bool(events("demo_ready")), 30)
                    row["session_id"] = events("demo_ready")[0]["session_id"]
                    await page.evaluate("s=>window.__inject(s)", encoded)
                    if scenario == "completed":
                        await until(lambda: bool(events("speech_played")))
                        assert not events("speech_error"), events("speech_error")
                        assert "播放完成" in await page.locator("#messages").inner_text()
                    else:
                        await until(lambda: bool(events("speech_error")))
                        await until(lambda: bool(events("speech_cancelled")))
                        assert events("speech_error")[-1]["code"] == "response_incomplete"
                        assert "回答未完成" in await page.locator("#messages").inner_text()
                        assert "没能完整生成" in await page.locator("#notice").inner_text()
                        assert not events("speech_text_done") and not events("speech_played")
                    await page.click("#stop")
                    await page.wait_for_function("window.__tracks.every(t=>t.readyState==='ended')")
                    await page.evaluate("window.__micContext.close()")
                    await page.screenshot(path=str(args.output / f"{scenario}.png"), full_page=True)
                finally:
                    await context.close()
                    server.should_exit = True
                    await task
                    listener.close()
                trace = ROOT / "exp" / row["session_id"] / "realtimeout_live/events.jsonl"
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                endings = [r["data"] for r in rows if r.get("event") == "response_completion_repair"]
                row["completion_events"] = endings
                reasons = [r["finish_reason"] for r in endings if r.get("stage") == "call_finished"]
                row["finish_reasons"] = reasons
                if scenario == "completed":
                    assert "length" in reasons and reasons[-1] == "stop", reasons
                    assert any(r.get("stage") == "completed" for r in endings)
                else:
                    assert any(r.get("reason") == "continuation_limit" for r in endings)
                assert rows[-1]["event"] == "trace_closed" and rows[-1].get("dropped", 0) == 0
                assert not row["page_errors"]
                row["passed"] = True
                print(json.dumps({"scenario": scenario, "passed": True, "finish_reasons": reasons}), flush=True)
            await browser.close()
        receipt["passed"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
