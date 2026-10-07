#!/usr/bin/env python3
"""Serial live Omni checks for demo EOS/length and native assistant continuation.

Self-authored synthetic audio, no physical device or human quality score.
Writes a new receipt directory; never overwrites an earlier run.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import module
from engine import ActorEngine
from messages import build_audio_content
from response_completion import ResponseIncompleteError


class Trace:
    def __init__(self):
        self.rows = []

    def observe(self, event, data=None, **_):
        self.rows.append({"event": event, "data": data or {}})


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((ROOT / "configs/demo_chat.yaml").read_text())
    prompts = cfg["prompts"]
    receipt = {"scope": __doc__, "serial": True, "physical_audio": False,
               "rows": [], "passed": False}
    inputs = [
        ("zh", "请讲一个两百字左右的森林童话，要有完整结尾，直接讲正文。"),
        ("en", "Tell a complete story in about one hundred words about a cat finding its way home."),
    ]
    try:
        for name, text in inputs:
            chunks = [c async for c in module.tts_omni_stream(text,
                voice_control={"speaker": "chelsie", "seed": 42})]
            rate = chunks[0].sample_rate
            pcm = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2").astype(np.float32) / 32768
            sf.write(args.output / f"{name}-input.wav", pcm, rate, subtype="PCM_16")
            messages = [{"role": "system", "content": prompts["response"]},
                        {"role": "user", "content": [build_audio_content(pcm, rate)]}]
            for mode, tokens, limit in [("forced_length", 64, 6), ("production", 512, 3),
                                        ("exhausted", 64, 0)]:
                trace = Trace()
                engine = ActorEngine(prompts=prompts,
                    engine_cfg={**cfg["engine"], "stream_response": True,
                                "response_max_tokens": tokens, "response_max_continuations": limit},
                    vad_iterator=lambda *_a, **_k: None, llm_fn=lambda _: "", asr_fn=lambda _: "",
                    tts_fn=lambda *_: "", text_stream_fn=module.llm_qwen3o_stream,
                    tts_stream_fn=module.tts_omni_stream)
                engine.demo_trace = trace
                started = time.monotonic()
                output, failure = [], None
                try:
                    async for part in engine._response_text_stream(messages):
                        output.append(str(part))
                except ResponseIncompleteError as exc:
                    failure = exc.reason
                calls = [r["data"] for r in trace.rows if r["event"] == "model_call_done"]
                reasons = [c.get("finish_reason") for c in calls if c.get("kind") == "response"]
                row = {"language": name, "mode": mode, "max_tokens": tokens,
                       "max_continuations": limit, "finish_reasons": reasons,
                       "text": "".join(output), "failure": failure,
                       "elapsed_s": round(time.monotonic() - started, 3),
                       "capacity": engine.request_capacity.snapshot(), "events": trace.rows}
                receipt["rows"].append(row)
                if mode == "exhausted":
                    assert failure == "continuation_limit" and reasons == ["length"], row
                else:
                    assert failure is None and reasons[-1:] == ["stop"], row
                    if mode == "forced_length":
                        assert "length" in reasons and len(reasons) >= 2, row
                assert row["capacity"]["active_total"] == 0
                print(json.dumps({k: row[k] for k in ("language", "mode", "finish_reasons", "failure", "elapsed_s")}), flush=True)
        receipt["passed"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
