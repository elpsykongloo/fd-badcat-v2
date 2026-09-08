#!/usr/bin/env python3
"""Literal grammar compilation and live bad-request isolation; no user audio.

Omni Python: --compile-only --output NEW_RECEIPT.json (CPU only).
Backend Python: --output NEW_RECEIPT.json (real TTS, ASR and rejection probes).
Never overwrites a receipt; does not start or stop services.
"""
import argparse
import asyncio
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import module

CASES = [
    "你好。Hello, how are you?",
    "那我给你讲一个奇幻小故事吧：\n",
    "你能告诉我你在哪个城市吗？\r\n\t",
    'He said "Hi". Path: C:\\notes. [a-z]* $3.14.',
    "中😀\u2028\u2029",
    "\n\r\tHello.\n\r\t",
]


def compile_checks(report):
    import xgrammar as xgr
    import yaml
    from vllm_omni.config.stage_config import _parse_stage_deploy
    config = yaml.safe_load((ROOT / "configs/qwen3_omni_audio_single_gpu.yaml").read_text())
    stage = _parse_stage_deploy(config["stages"][0])
    assert stage.engine_extras["structured_outputs_config"]["backend"] == "xgrammar"
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(
        [bytes([i]) for i in range(256)], vocab_type=xgr.VocabType.RAW))
    for index, text in enumerate(CASES):
        grammar = module.verbatim_tts_payload(text)["structured_outputs"]["grammar"]
        compiled = compiler.compile_grammar(xgr.Grammar.from_ebnf(grammar))
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_string(text) and matcher.is_completed()
        matcher = xgr.GrammarMatcher(compiled)
        assert matcher.accept_string(text[:-1]) and not matcher.is_completed()
        assert not xgr.GrammarMatcher(compiled).accept_string(text + "unexpected")
        report["cases"].append({"case": index, "exact_only": True})
    for char in [chr(i) for i in range(32) if i not in (9, 10, 13)] + [chr(127)]:
        try:
            module.verbatim_tts_payload("x" + char + "y")
        except ValueError:
            continue
        raise AssertionError("Unsupported control character was admitted")
    report["fixed_backend"] = True
    report["unsupported_controls_rejected"] = 30


async def live_checks(args, report):
    import httpx
    import numpy as np
    import soundfile as sf
    from demo_startup import verify_spoken_text
    from stream_transport import audio_stream

    async def synthesize(text, readback=False):
        start = time.perf_counter()
        chunks = [c async for c in audio_stream(args.url, module.verbatim_tts_payload(text),
                                               30, expected_text=text)]
        assert chunks and len({c.sample_rate for c in chunks}) == 1
        pcm = b"".join(c.pcm for c in chunks)
        row = {"text": text, "text_proof": True, "samples": len(pcm) // 2,
               "elapsed_ms": round((time.perf_counter() - start) * 1000, 1)}
        if readback:
            wav = io.BytesIO()
            sf.write(wav, np.frombuffer(pcm, dtype="<i2"), chunks[0].sample_rate, format="WAV", subtype="PCM_16")
            wav.seek(0)
            actual = await asyncio.to_thread(module.asr, wav)
            row["readback"] = actual
            try:
                verify_spoken_text(text, actual)
                row["readback_pass"] = True
            except RuntimeError as exc:
                row["readback_pass"] = False
                report.setdefault("readback_failures", []).append(str(exc))
        report["cases"].append(row)

    await synthesize(CASES[0], True)  # Initialize backend before the old trigger.
    for text in (CASES[1], CASES[2], CASES[3], CASES[5]):
        await synthesize(text, text != CASES[3])
    async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
        for bad in [{"choice": [CASES[1]]}, {"grammar": 'root ::= "unterminated'}, {"regex": "("}]:
            payload = module.verbatim_tts_payload("你好。")
            payload.update(structured_outputs=bad, stream=False)
            response = await client.post(args.url, json=payload)
            report.setdefault("rejections", []).append({"kind": next(iter(bad)), "status": response.status_code})
            assert 400 <= response.status_code < 500, (response.status_code, response.text[:300])
            # A rejected request must not poison the existing core.
            await synthesize("你好。", True)
            health = await client.get("http://127.0.0.1:10003/v1/models")
            assert health.status_code == 200 and health.json().get("data")
    report["good_bad_good"] = True
    report["transport_contract_pass"] = True
    # Complete service-isolation probes even if a Talker/ASR readback differs;
    # preserve that independent acoustic failure and fail the overall receipt.
    if report.get("readback_failures"):
        raise AssertionError("Transport passed but acoustic readback mismatched; see readback_failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:10004/v1/chat/completions")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve output before any calls. Always keep failures for inspection.
    with args.output.open("x") as handle:
        report = {"version": module.VERBATIM_TTS_CONTRACT, "mode": "compile" if args.compile_only else "live",
                  "physical_listening": False, "cases": [], "pass": False}
        try:
            compile_checks(report) if args.compile_only else asyncio.run(live_checks(args, report))
            report["pass"] = True
        except Exception as exc:
            report["error"] = str(exc)
            raise
        finally:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
