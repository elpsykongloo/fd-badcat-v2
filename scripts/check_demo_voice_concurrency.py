#!/usr/bin/env python3
"""Self-authored paired serial/mixed-batch probe; not a latency benchmark."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import module
from stream_transport import audio_stream
from check_demo_voice_rng import comparison, acoustic_summary

TEXTS = [
    "你好，我们接着聊这件事情。",
    "我可以慢慢解释，你也可以随时打断我。",
    "你想先了解原因，还是先看一个具体的例子？",
    "现在是下午三点，距离出发还有二十五分钟。",
    "别着急，我们把问题分成几步，一步一步解决。",
    "The blue cup is next to the window.",
]
INTERFERENCE = "窗外的树叶轻轻摇动，小猫正在阳台上晒太阳。"


def analyze_traces(receipt, directory):
    traces = {}
    for case in receipt["cases"]:
        if not case["controlled"]:
            continue
        files = list(directory.glob(f"chatcmpl-{case['request_id']}-*.jsonl"))
        if len(files) != 1:
            raise ValueError(f"Expected one trace for {case['id']}, found {len(files)}")
        traces[case["id"]] = [json.loads(line) for line in files[0].read_text().splitlines()]
        if not traces[case["id"]]:
            raise ValueError(f"Empty trace for {case['id']}")
    rows = []
    for pair in receipt["comparisons"]:
        if pair["group"].startswith("cross_"):
            continue
        a, b = traces[pair["reference"]], traces[pair["candidate"]]
        row = {key: pair[key] for key in ("group", "reference", "candidate", "pcm_equal")}
        row["trace_steps"] = [len(a), len(b)]
        row["max_active_requests"] = max(len([x for x in r["scheduled_ids"] if x]) for r in b)
        for field in ("primary", "codes", "hidden", "text", "primary_rng_offset", "residual_rng_offset"):
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x[field] != y[field]), None)
            difference = None
            if first is not None:
                x, y = np.asarray(a[first][field]), np.asarray(b[first][field])
                difference = {"step_zero_based": first, "max_abs": float(np.max(abs(x - y)))}
                if field in ("primary", "codes"):
                    difference.update(reference=x.tolist(), candidate=y.tolist())
            row[field + "_first_difference"] = difference
        rows.append(row)
    return {"comparison": "same-text traces aligned by residual decode step; no waveform alignment",
            "limitation": "Tracing copies tensors to CPU and perturbs timing; earliest observed boundary, not full causal attribution",
            "comparisons": rows}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:10003/v1/chat/completions")
    parser.add_argument("--tag", required=True, help="Unique external request prefix for tracing")
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audio").mkdir()
    receipt = {"completed": False, "human_audio": False, "physical_device": False,
               "formal_benchmark": False, "independent_target_texts": len(TEXTS),
               "design": "two serial per target, two mixed rounds (3 controlled + 1 legacy), one serial recovery",
               "cases": [], "comparisons": [], "exclusions": []}
    waveforms, manifest_audio, pairs = {}, [], []

    async def synth(name, text, controlled=True):
        voice = {"speaker": "chelsie", "seed": 42} if controlled else None
        payload = module.verbatim_tts_payload(text, voice_control=voice)
        payload["request_id"] = f"{args.tag}-{name}"
        proof = {"contract": module.DEMO_VOICE_CONTRACT, **voice} if voice else None
        started = time.perf_counter()
        chunks, first_pcm_s = [], None
        async for chunk in audio_stream(args.url, payload, 120, expected_text=text, expected_voice=proof):
            if first_pcm_s is None:
                first_pcm_s = time.perf_counter() - started
            chunks.append(chunk)
        total_s = time.perf_counter() - started
        pcm = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2")
        waveforms[name] = pcm
        sf.write(args.output / "audio" / f"{name}.wav", pcm, chunks[0].sample_rate, subtype="PCM_16")
        row = {"id": name, "request_id": payload["request_id"], "text": text,
               "controlled": controlled, "audio": f"audio/{name}.wav",
               "first_pcm_s_diagnostic": first_pcm_s, "total_s_diagnostic": total_s,
               **acoustic_summary(pcm, chunks[0].sample_rate)}
        receipt["cases"].append(row)
        manifest_audio.append({"id": name, "path": row["audio"], "text": text})
        print(name, len(pcm), flush=True)

    def pair(group, a, b):
        item = {"group": group, "reference": a, "candidate": b}
        pairs.append(item)
        receipt["comparisons"].append({**item, **comparison(waveforms[a], waveforms[b])})

    try:
        for i, text in enumerate(TEXTS):
            await synth(f"solo-{i}-a", text)
            await synth(f"solo-{i}-b", text)
            pair("serial_repeat", f"solo-{i}-a", f"solo-{i}-b")
        for repeat in range(2):
            for batch in range(2):
                indices = list(range(batch * 3, batch * 3 + 3))
                if repeat:
                    indices.reverse()
                await asyncio.gather(
                    *(synth(f"mixed-{i}-{repeat}", TEXTS[i]) for i in indices),
                    synth(f"legacy-{batch}-{repeat}", INTERFERENCE, controlled=False))
                for i in indices:
                    pair("concurrent", f"solo-{i}-a", f"mixed-{i}-{repeat}")
        for i, text in enumerate(TEXTS):
            await synth(f"recovery-{i}", text)
            pair("recovery", f"solo-{i}-a", f"recovery-{i}")
            if i:
                pair("cross_text_zh" if i < 5 else "cross_language",
                     "solo-0-a", f"solo-{i}-a")
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
        if args.trace_dir:
            analysis = analyze_traces(receipt, args.trace_dir)
            (args.output / "trace_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
        receipt["completed"] = True
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        manifest = {"audio": manifest_audio, "pairs": pairs, "design": {
            "source": "check_demo_voice_concurrency.py self-authored fixtures",
            "independent_target_texts": len(TEXTS), "human_audio": False,
            "physical_device": False, "pair_dependence": "Repeated texts/shared serial references; not independent identity trials",
            "negative_speakers": 0, "threshold_calibrated": False}}
        (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
