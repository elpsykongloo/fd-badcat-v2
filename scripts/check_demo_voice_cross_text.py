#!/usr/bin/env python3
"""Serial content/length/seed diagnostic; self-authored text, no identity threshold."""
import argparse
import asyncio
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import module
from stream_transport import audio_stream
from check_demo_voice_rng import acoustic_summary, comparison

# Nested content separates short acknowledgements from full explanations without
# pretending that text, phonetic content and duration can be fully disentangled.
FAMILIES = [
    ["好的，我明白了。", "好的，我明白了。我们先把问题说清楚，再一起看看有哪些可以尝试的办法。",
     "好的，我明白了。我们先把问题说清楚，再一起看看有哪些可以尝试的办法。你可以从最困扰你的地方开始讲，不需要一下子提供所有细节。我会跟着你的节奏，遇到不确定的地方再向你确认。"],
    ["别急，慢慢来。", "别急，慢慢来。先把桌上的东西收好，然后检查一下钥匙和手机有没有带上。",
     "别急，慢慢来。先把桌上的东西收好，然后检查一下钥匙和手机有没有带上。外面可能有些凉，可以顺手拿一件外套。如果时间还充裕，我们就沿着河边走过去，路上也能看看周围的风景。"],
    ["你想先听哪个？", "你想先听哪个？我们可以先聊这个现象为什么会发生，也可以先看一个简单的例子。",
     "你想先听哪个？我们可以先聊这个现象为什么会发生，也可以先看一个简单的例子。两种方式都能帮助理解，只是出发点不同。你选一个更感兴趣的方向，我再接着往下解释，中途也可以随时换个话题。"],
    ["窗外下雨了。", "窗外下雨了。雨点落在树叶上，发出细细的声响，街上的行人也慢慢撑起了伞。",
     "窗外下雨了。雨点落在树叶上，发出细细的声响，街上的行人也慢慢撑起了伞。屋子里倒是很安静，桌上的茶还冒着热气。小猫趴在窗边看了一会儿，又转过身来，找了个舒服的地方继续睡觉。"],
]
LENGTHS = ["short", "medium", "long"]
SEEDS = [42, 43]


def planned_pairs():
    pairs = []

    def add(group, a, b):
        pairs.append(dict(group=group, reference=a, candidate=b))

    for length in LENGTHS:
        for family in range(4):
            add(f"seed_{length}", f"f{family}-{length}-s42", f"f{family}-{length}-s43")
        for seed in SEEDS:
            for a, b in combinations(range(4), 2):
                add(f"cross_text_{length}_s{seed}", f"f{a}-{length}-s{seed}", f"f{b}-{length}-s{seed}")
    for family in range(4):
        add("serial_recovery", f"f{family}-medium-s42", f"f{family}-recovery")
        for seed in SEEDS:
            for a, b in combinations(LENGTHS, 2):
                add(f"nested_{a}_{b}", f"f{family}-{a}-s{seed}", f"f{family}-{b}-s{seed}")
        full = f"f{family}-long-s42"
        for seconds in (1, 2, 4):
            for side in ("head", "tail"):
                add(f"crop_full_{seconds}s", full, f"{full}-{side}-{seconds}s")
            add(f"crop_disjoint_{seconds}s", f"{full}-head-{seconds}s", f"{full}-tail-{seconds}s")
    return pairs


def aggregate_short_turns(output):
    """Fixed concatenation supplements per-turn scores; never replaces them."""
    original = json.loads((output / "manifest.json").read_text())
    rows = {r["id"]: r for r in original["audio"]}
    audio, pairs = [], []
    for seed in SEEDS:
        ids = [f"f{f}-short-s{seed}" for f in range(4)]
        audio.extend(rows[key] for key in ids)
        waves = [sf.read(output / rows[key]["path"], dtype="int16") for key in ids]
        rates = {rate for _, rate in waves}
        if len(rates) != 1:
            raise ValueError("Cannot concatenate different sample rates")
        name = f"short-concat-s{seed}"
        path = f"audio/{name}.wav"
        sf.write(output / path, np.concatenate([wave for wave, _ in waves]),
                 waves[0][1], subtype="PCM_16")
        audio.append(dict(id=name, path=path, text="".join(rows[key]["text"] for key in ids)))
        for family in range(4):
            ref = f"f{family}-long-s42"
            pairs.append(dict(group=f"concat_to_long_s{seed}", reference=ref, candidate=name))
            for key in ids:
                pairs.append(dict(group=f"single_short_to_long_s{seed}", reference=ref, candidate=key))
    audio.extend(rows[f"f{f}-long-s42"] for f in range(4))
    pairs.append(dict(group="concat_seed", reference="short-concat-s42", candidate="short-concat-s43"))
    design = {
        "rule": "All four short utterances in family order 0,1,2,3; no gaps/trim/repeat/selection",
        "new_syntheses": 0, "human_audio": False, "threshold_calibrated": False,
        "limitation": "Changes duration/content/boundaries together and may average real turn drift; retain individual scores",
    }
    with (output / "aggregation_manifest.json").open("x") as stream:
        json.dump(dict(audio=audio, pairs=pairs, design=design), stream, ensure_ascii=False, indent=2)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:10003/v1/chat/completions")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "contract": "demo-voice-cross-text-v1", "families": FAMILIES,
        "lengths": LENGTHS, "seeds": SEEDS, "planned_syntheses": 28,
        "planned_crops": 24, "pairs": planned_pairs(),
        "deployment": "native Talker, chelsie, unchanged sampling, seq4; serial calls",
        "order": "seed42 family/length, seed43 reverse family/length, four seed42 medium recoveries",
        "crop_rule": "first/last 1,2,4 seconds of each seed42 long waveform; no VAD or selection",
        "human_audio": False, "physical_device": False, "formal_benchmark": False,
        "identity_threshold": None, "negative_speakers": 0,
        "pair_dependence": "Nested content, shared references, two seeds; not independent identity trials",
        "interpretation": "Same-record crops measure joint temporal/content/duration sensitivity; not proof of stable identity within a recording",
    }
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing protocol differs")
    else:
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    if args.prepare_only:
        return
    (args.output / "audio").mkdir(exist_ok=False)
    receipt = {"completed": False, "cases": [], "crops": [], "recovery": [], "exclusions": []}
    audio, pcm_by_id = [], {}

    def save():
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    async def synth(family, length, seed, recovery=False):
        name = f"f{family}-recovery" if recovery else f"f{family}-{LENGTHS[length]}-s{seed}"
        text = FAMILIES[family][length]
        voice = {"speaker": "chelsie", "seed": seed}
        payload = module.verbatim_tts_payload(text, voice_control=voice)
        chunks = [c async for c in audio_stream(args.url, payload, 120, expected_text=text,
                    expected_voice={"contract": module.DEMO_VOICE_CONTRACT, **voice})]
        if not chunks or len({c.sample_rate for c in chunks}) != 1:
            raise RuntimeError("Empty audio or inconsistent rate")
        pcm = np.frombuffer(b"".join(c.pcm for c in chunks), dtype="<i2")
        rate = chunks[0].sample_rate
        pcm_by_id[name] = (pcm, rate)
        path = f"audio/{name}.wav"
        sf.write(args.output / path, pcm, rate, subtype="PCM_16")
        audio.append(dict(id=name, path=path, text=text))
        receipt["cases"].append(dict(id=name, family=family, length=LENGTHS[length], seed=seed,
            text=text, audio=path, **acoustic_summary(pcm, rate)))
        if recovery:
            original = f"f{family}-medium-s42"
            receipt["recovery"].append(dict(reference=original, candidate=name,
                **comparison(pcm_by_id[original][0], pcm)))
        save()
        print(name, len(pcm) / rate, flush=True)

    try:
        for seed in SEEDS:
            indices = [(f, length) for f in range(4) for length in range(3)]
            for family, length in indices if seed == 42 else reversed(indices):
                await synth(family, length, seed)
        for family in range(4):
            await synth(family, 1, 42, recovery=True)
        for family in range(4):
            source = f"f{family}-long-s42"
            pcm, rate = pcm_by_id[source]
            if len(pcm) < 8 * rate:
                raise ValueError("Long waveform too short for disjoint 4s crops")
            for seconds in (1, 2, 4):
                for side in ("head", "tail"):
                    start = 0 if side == "head" else len(pcm) - seconds * rate
                    crop = pcm[start:start + seconds * rate]
                    name = f"{source}-{side}-{seconds}s"
                    path = f"audio/{name}.wav"
                    sf.write(args.output / path, crop, rate, subtype="PCM_16")
                    audio.append(dict(id=name, path=path, source=source))
                    receipt["crops"].append(dict(id=name, source=source, start_sample=start,
                        **acoustic_summary(crop, rate)))
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
        # A failed run retains outputs but cannot be scored as a complete run.
        if receipt["completed"]:
            manifest = {"audio": audio, "pairs": protocol["pairs"], "design": protocol}
            (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            aggregate_short_turns(args.output)


if __name__ == "__main__":
    asyncio.run(main())
