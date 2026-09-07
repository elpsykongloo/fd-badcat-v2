#!/usr/bin/env python3
"""Self-authored real-model input routing canaries; no benchmark or user audio."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import module
from messages import build_audio_content
from control_labels import parse_label
from guarded_turns import route_messages

CASES = [("停。", "stop_only", "switch"),
         ("别说了。", "stop_only", "switch"),
         ("等一下，我还没说完。", "yield_wait", "switch"),
         ("请用普通话说。", "yield_ready", "switch"),
         ("嗯嗯，知道了。", "keep", "continue"),
         ("Wait, let me finish.", "yield_wait", "switch")]


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-audio", type=Path, help="Existing self-authored canaries; no new synthesis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((ROOT / "src/config.yaml").read_text())
    demo = yaml.safe_load((ROOT / "configs/demo_chat.yaml").read_text())
    prompts = {**base["prompts"], **demo["prompts"]}
    report = {"version": "guarded-turns-v1", "physical_acoustic_test": False,
              "cases": [], "pass": False}
    try:
        for index, (text, expected_route, expected_interrupt) in enumerate(CASES):
            if args.reuse_audio:
                path = args.reuse_audio / f"input-{index}.wav"
                pcm, rate = sf.read(path, dtype="int16")
            else:
                chunks = [chunk async for chunk in module.tts_omni_stream(text)]
                pcm = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2")
                rate = chunks[0].sample_rate
                path = args.output / f"input-{index}.wav"
                sf.write(path, pcm, rate, subtype="PCM_16")
            content = build_audio_content(pcm.astype(np.float32) / 32768, rate)
            request = route_messages(prompts["input_route"], content, playing=True,
                                     reference="你好，我可以为你介绍语音助手。")
            interrupt_messages = [{"role": "system", "content": prompts["interrupt"]},
                                  {"role": "user", "content": [content]}]
            route = parse_label("input_route", "".join([p async for p in module.llm_qwen3o_stream(request)]))
            interrupt = parse_label("interrupt", await asyncio.to_thread(module.llm_qwen3o_strict, interrupt_messages))
            row = {"text": text, "expected_route": expected_route, "route": route,
                   "expected_interrupt": expected_interrupt, "interrupt": interrupt,
                   "audio": str(path), "baseline_pass": interrupt == expected_interrupt,
                   "audio_s": len(pcm) / rate,
                   "pass": route == expected_route}
            report["cases"].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        report["pass"] = all(c["pass"] for c in report["cases"])
        if not report["pass"]:
            raise RuntimeError("Input routing canary failed")
    finally:
        (args.output / "receipt.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
