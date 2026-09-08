#!/usr/bin/env python3
"""Small self-authored TTS fidelity smoke; no benchmark/user recordings, no hashes."""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("FDBC_ASR_BACKEND", "sensevoice")
import module
from demo_startup import verify_spoken_text

CASES = ["你那边怎么样？", "你能告诉我你在哪个城市吗？", "Where are you located?",
         "请停止朗读，回答我是谁。", "温度是3.14度，不是4度。"]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--browser-receipt", type=Path,
                        help="Also read back completed PCM received by our browser smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"version": module.VERBATIM_TTS_CONTRACT, "physical_listening": False,
               "formal_benchmark": False, "cases": [], "browser_readback": [], "pass": False}
    try:
        for index, expected in enumerate(CASES):
            start, first = time.perf_counter(), None
            chunks = []
            async for chunk in module.tts_omni_stream(expected):
                if first is None: first = time.perf_counter() - start
                chunks.append(chunk)
            if not chunks or len({c.sample_rate for c in chunks}) != 1:
                raise RuntimeError("Empty/mixed-rate TTS")
            audio = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2")
            path = args.output / f"sentence-{index}.wav"
            sf.write(path, audio, chunks[0].sample_rate, subtype="PCM_16")
            actual = await asyncio.to_thread(module.asr, str(path))
            row = {"expected": expected, "recognized": actual, "wav": str(path),
                   "duration_s": len(audio) / chunks[0].sample_rate, "first_pcm_s": first}
            receipt["cases"].append(row)
            verify_spoken_text(expected, actual)
        if args.browser_receipt:
            browser = json.loads(args.browser_receipt.read_text())["result"]
            texts = {x["data"]["utterance_id"]: x["data"]["text"]
                     for x in browser["events"] if x["event"] == "speech_text_done"}
            completed = {x["utterance_id"] for x in browser["completed_playbacks"]}
            for entry in browser["received_audio_files"]:
                sid = entry["utterance_id"]
                if sid not in completed: continue
                actual = await asyncio.to_thread(module.asr, entry["path"])
                row = {"utterance_id": sid, "expected": texts[sid], "recognized": actual}
                receipt["browser_readback"].append(row)
                verify_spoken_text(texts[sid], actual)
            if len(receipt["browser_readback"]) != browser["turns"]:
                raise RuntimeError("Missing completed browser audio")
        receipt["pass"] = True
    except Exception as exc:
        receipt["error"] = str(exc)
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
