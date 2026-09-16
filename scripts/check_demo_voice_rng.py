#!/usr/bin/env python3
"""Self-authored serial A/A/B/A/C/A voice probe; PCM comparisons, never hashes.

Three arms on the SAME patched seq4 server: old payload, explicit speaker only,
and demo request RNG. No human recordings, no formal latency/voice-quality score.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import module
from stream_transport import audio_stream

TARGETS = ["你好，我们接着聊这件事情。", "我可以慢慢解释，你也可以随时打断我。"]
INTERFERENCE = ["窗外的树叶轻轻摇动，小猫正在阳台上晒太阳。", "The blue cup is next to the window."]


def acoustic_summary(pcm, rate):
    x = pcm.astype(np.float64) / 32768
    # Windowed normalized autocorrelation, diagnostic only (no SV claim).
    # Remove unvoiced/ambiguous frames and retain the voiced-frame denominator.
    pitches = []
    step, width = round(rate * .01), round(rate * .04)
    lo, hi = int(rate / 450), int(rate / 80)
    frames = 0
    for start in range(0, len(x) - width + 1, step):
        frames += 1
        f = x[start:start + width]
        if np.sqrt(np.mean(f * f)) < .015:
            continue
        f = (f - np.mean(f)) * np.hanning(width)
        spectrum = np.fft.rfft(f, n=2 * width)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[:width]
        if corr[0] <= 0:
            continue
        lag = lo + int(np.argmax(corr[lo:hi + 1]))
        if corr[lag] / corr[0] > .6:
            pitches.append(rate / lag)
    return {"samples": len(pcm), "rate": rate, "duration_s": len(pcm) / rate,
            "rms_dbfs": float(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12))),
            "clipped_samples": int(np.count_nonzero((pcm == 32767) | (pcm == -32768))),
            "f0_method": "windowed-autocorrelation-80-450Hz; diagnostic, octave errors possible",
            "f0_frames": len(pitches), "analysis_frames": frames,
            "f0_median_hz": float(np.median(pitches)) if pitches else None}


def comparison(a, b):
    x, y = a.astype(np.float64), b.astype(np.float64)
    n = min(len(x), len(y))
    same = np.equal(a[:n], b[:n])
    unequal = np.flatnonzero(~same)
    return {"pcm_equal": bool(len(a) == len(b) and same.all()),
            "sample_count_delta": len(b) - len(a), "compared_samples": n,
            "first_unequal_sample": int(unequal[0]) if len(unequal) else (n if len(a) != len(b) else None),
            "equal_sample_fraction_common_prefix": float(same.mean()),
            "unaligned_rmse_common_prefix": float(np.sqrt(np.mean(((x[:n] - y[:n]) / 32768) ** 2)))}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=module.OMNI_TTS_URL)
    parser.add_argument("--arms", nargs="+", choices=["legacy", "speaker", "controlled"],
                        default=["legacy", "speaker", "controlled"])
    parser.add_argument("--asr", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audio").mkdir()
    receipt = {"contract": module.DEMO_VOICE_CONTRACT, "serial_requests": True,
        "deployment": "qwen3_omni_audio_single_gpu.yaml seq4; not serial-eval latency",
        "independent_texts": len(TARGETS), "human_audio": False, "physical_device": False,
        "formal_benchmark": False, "listening_evaluation": False, "exclusions": [],
        "design": "Each arm/text: A,A,B,A,C,A. Repeats are not independent text samples.",
        "cases": [], "comparisons": [], "completed": False}

    def save():
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    try:
        for arm in args.arms:
            for target_index, target in enumerate(TARGETS):
                first = None
                for index, text in enumerate([target, target, INTERFERENCE[0], target, INTERFERENCE[1], target]):
                    name = f"{arm}-t{target_index}-{index}"
                    control = {"speaker": "chelsie", "seed": 42} if arm == "controlled" else None
                    payload = module.verbatim_tts_payload(text, voice_control=control)
                    if arm == "speaker":
                        payload["voice"] = "chelsie"
                    proof = {"contract": module.DEMO_VOICE_CONTRACT, **control} if control else None
                    started = time.perf_counter()
                    chunks = [c async for c in audio_stream(args.url, payload, 120,
                        expected_text=text, expected_voice=proof)]
                    if len({c.sample_rate for c in chunks}) != 1:
                        raise RuntimeError("Mixed sample rates")
                    pcm = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2")
                    rate = chunks[0].sample_rate
                    path = args.output / "audio" / (name + ".wav")
                    sf.write(path, pcm, rate, subtype="PCM_16")
                    row = {"id": name, "arm": arm, "target": target_index, "text": text,
                        "is_target": text == target, "audio": str(path.relative_to(args.output)),
                        "request_payload": payload, "voice_config_acknowledged": proof,
                        "wall_s_diagnostic_only": time.perf_counter() - started,
                        **acoustic_summary(pcm, rate)}
                    receipt["cases"].append(row)
                    if text == target:
                        if first is None:
                            first = (name, pcm.copy())
                        else:
                            receipt["comparisons"].append({"arm": arm, "target": target_index,
                                "reference": first[0], "candidate": name, **comparison(first[1], pcm)})
                    save()
                    print(json.dumps({"done": name, "duration": row["duration_s"]}), flush=True)
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
                save()
        receipt["completed"] = True
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()


if __name__ == "__main__":
    asyncio.run(main())
