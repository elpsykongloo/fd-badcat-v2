#!/usr/bin/env python3
"""Separate live mixed-request/cancellation smoke, not a latency benchmark."""
import argparse
import asyncio
from contextlib import aclosing
import json
import sys
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import module
from check_demo_voice_rng import TARGETS, INTERFERENCE, comparison


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--serial-receipt", required=True, type=Path)
    parser.add_argument("--asr", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audio").mkdir()
    receipt = {"formal_benchmark": False, "human_audio": False, "physical_device": False,
               "serial_latency": False, "cases": [], "completed": False}
    serial = json.loads(args.serial_receipt.read_text())
    refs = {}
    for index in range(2):
        case = next(c for c in serial["cases"] if c["id"] == f"controlled-t{index}-0")
        refs[index], _ = sf.read(args.serial_receipt.parent / case["audio"], dtype="int16")

    async def synth(name, text, seed=42, target=None, controlled=True):
        voice = {"speaker": "chelsie", "seed": seed} if controlled else None
        chunks = [c async for c in module.tts_omni_stream(text, voice_control=voice)]
        pcm = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2")
        sf.write(args.output / "audio" / f"{name}.wav", pcm, chunks[0].sample_rate, subtype="PCM_16")
        row = {"id": name, "text": text, "seed": seed, "controlled": controlled, "samples": len(pcm),
               "audio": f"audio/{name}.wav"}
        if target is not None: row["vs_serial"] = comparison(refs[target], pcm)
        receipt["cases"].append(row)
        return pcm

    try:
        # One independent request per coroutine/HTTP session; intentionally
        # overlap long-enough utterances to exercise dynamic scheduler batches.
        await asyncio.gather(synth("mixed-a", TARGETS[0], target=0),
                             synth("mixed-b", TARGETS[1], target=1),
                             synth("mixed-legacy", INTERFERENCE[0], controlled=False))
        await asyncio.gather(synth("reordered-b", TARGETS[1], target=1),
                             synth("reordered-a", TARGETS[0], target=0))
        changed = await synth("seed43", TARGETS[0], seed=43, target=0)
        repeated = await synth("seed43-repeat", TARGETS[0], seed=43)
        receipt["seed43_repeat"] = comparison(changed, repeated)

        # Close a live stream after receiving real PCM, then submit a new
        # request with the same seed. No attempt to reuse an upstream request ID.
        async with aclosing(module.tts_omni_stream(
                "我会慢慢讲一个故事。" * 15, voice_control={"speaker": "chelsie", "seed": 42})) as source:
            first = await anext(source)
            receipt["cancelled_after_pcm_bytes"] = len(first.pcm)
        await synth("after-cancel", TARGETS[0], target=0)

        # API boundary failures must be explicit and leave the server healthy.
        async with aiohttp.ClientSession(trust_env=False) as client:
            receipt["invalid_requests"] = []
            for field, value in (("voice", "not-a-speaker"), ("seed", -1),
                                 ("vllm_xargs", {"fd_demo_tts_rng": "unknown"})):
                payload = module.verbatim_tts_payload(TARGETS[0], voice_control={"speaker": "chelsie", "seed": 42})
                payload[field] = value
                async with client.post(module.OMNI_TTS_URL, json={**payload, "stream": True}) as response:
                    receipt["invalid_requests"].append({"field": field, "status": response.status})
                    await response.read()
                    if response.status != 400:
                        raise RuntimeError(f"Expected 400 for invalid {field}, got {response.status}")
        await synth("after-invalid", TARGETS[0], target=0)
        if args.asr:
            module.configure_asr({"backend": "sensevoice", "provider": "cpu", "num_threads": 2})
            from demo_startup import verify_spoken_text
            for row in receipt["cases"]:
                row["recognized"] = await asyncio.to_thread(module.asr, str(args.output / row["audio"]))
                try:
                    verify_spoken_text(row["text"], row["recognized"])
                    row["asr_exact_normalized"] = True
                except RuntimeError:
                    row["asr_exact_normalized"] = False
        receipt["completed"] = True
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
