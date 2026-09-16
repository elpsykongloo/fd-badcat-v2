#!/usr/bin/env python3
"""Offline speaker consistency: author's final LMFT weights, no service changes.

The adapter/MFA/ASP implementation follows WeSpeaker (Apache-2.0):
Copyright (c) 2025 Qituan Shangguan; pooling Copyright (c) 2021 Shuai Wang.
https://github.com/wenet-e2e/wespeaker
Licensed under the Apache License, Version 2.0;
https://www.apache.org/licenses/LICENSE-2.0
Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

Uses the standard Transformers backbone and strict author checkpoint loading.
No training, cohort scoring, waveform alignment, integrity hashes or network IO.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
import torchaudio
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertConfig, Wav2Vec2BertModel


class SpeakerEncoder(nn.Module):
    """Full, unpruned 24-layer backbone + 25 adapters + ASP + 256-D projection."""

    def __init__(self, model_dir):
        super().__init__()
        config = Wav2Vec2BertConfig.from_pretrained(model_dir, local_files_only=True)
        if (config.hidden_size, config.num_hidden_layers) != (1024, 24):
            raise ValueError("Expected full w2v-BERT 2.0 (1024, 24)")
        config._attn_implementation = "eager"
        # Preserve author's state-dict names; every inference weight must match.
        self.front = nn.Module()
        self.front.encoder = Wav2Vec2BertModel(config)
        if hasattr(self.front.encoder, "masked_spec_embed"):
            del self.front.encoder.masked_spec_embed
        self.adapter_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(1024, 128), nn.LayerNorm(128),
                          nn.ReLU(), nn.Linear(128, 128)) for _ in range(25)
        ])
        self.pooling = nn.Module()
        self.pooling.attention = nn.Sequential(
            nn.Conv1d(3200, 128, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 3200, 1), nn.Softmax(dim=2))
        self.bottleneck = nn.Linear(6400, 256)
        checkpoint = torch.load(Path(model_dir) / "model_lmft_0.14.pth",
                                map_location="cpu", weights_only=True, mmap=True)
        self.load_state_dict(checkpoint["modules"]["spk_model"], strict=True)
        self.requires_grad_(False).eval()

    def forward(self, features):
        # Match author's whole-utterance, unpadded feature_projection -> layers.
        x = self.front.encoder.feature_projection(features)[0]
        states = [x]
        for layer in self.front.encoder.encoder.layers:
            x = layer(x)[0]
            states.append(x)
        x = torch.cat([adapter(state) for adapter, state in
                       zip(self.adapter_layers, states, strict=True)], dim=-1).transpose(1, 2)
        weights = self.pooling.attention(x)
        mean = (x * weights).sum(dim=2)
        std = ((x.square() * weights).sum(dim=2) - mean.square()).clamp(min=1e-5).sqrt()
        return self.bottleneck(torch.cat([mean, std], dim=1))


def normalize_embedding(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(value)
    if value.shape != (256,) or not np.isfinite(value).all() or not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid 256-D speaker embedding")
    return value / norm


def read_audio(path):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono audio: {path}")
    audio = audio[:, 0]
    if not np.isfinite(audio).all() or len(audio) < rate * .5 or not np.any(audio):
        raise ValueError(f"Empty, silent, nonfinite or <0.5s audio: {path}")
    if len(audio) > rate * 60:
        raise ValueError(f"Audio exceeds author's 60s evaluation limit: {path}")
    if rate != 16000:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), rate, 16000).numpy()
    return audio


def load_manifest(path):
    """{audio: [{id,path,text?}], pairs: [{group,reference,candidate}]}.

    Paths are relative to the manifest; no implicit pair mining or best-of choice.
    """
    manifest = json.loads(path.read_text())
    audio = {}
    for row in manifest["audio"]:
        key = row["id"]
        if key in audio:
            raise ValueError(f"Duplicate audio ID: {key}")
        audio[key] = {**row, "path": path.parent / row["path"]}
    pairs = manifest["pairs"]
    if not audio or not pairs:
        raise ValueError("Need audio and predeclared comparisons")
    for pair in pairs:
        if pair["reference"] not in audio or pair["candidate"] not in audio:
            raise ValueError(f"Unknown audio in comparison: {pair}")
        if pair["reference"] == pair["candidate"]:
            raise ValueError("Self-comparison is not an independent comparison")
    return audio, pairs, manifest.get("design", {})


def score_pairs(audio, pairs, embeddings):
    results, groups = [], defaultdict(list)
    for pair in pairs:
        left, right = pair["reference"], pair["candidate"]
        score = float(np.clip(embeddings[left] @ embeddings[right], -1, 1))
        result = {**pair, "cosine": score, "distance": 1 - score}
        results.append(result)
        groups[pair["group"]].append(result)
    summary = {}
    for name, rows in groups.items():
        values = [r["cosine"] for r in rows]
        used = {r[k] for r in rows for k in ("reference", "candidate")}
        texts = {audio[k].get("text") for k in used} - {None}
        summary[name] = {"pairs": len(rows), "unique_audio_ids": len(used),
                         "unique_texts": len(texts), "mean_cosine": float(np.mean(values)),
                         "min_cosine": min(values), "max_cosine": max(values)}
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("model/speaker-w2vbert-lmft"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audio, pairs, design = load_manifest(args.manifest)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SpeakerEncoder(args.model_dir).to(args.device)
    processor = SeamlessM4TFeatureExtractor.from_pretrained(args.model_dir, local_files_only=True)
    embeddings, durations = {}, {}

    @torch.inference_mode()
    def extract(path):
        signal = read_audio(path)
        features = processor(signal, sampling_rate=16000, return_tensors="pt",
                             padding=False, truncation=False, return_attention_mask=False)
        value = model(features.input_features.to(args.device)).float().cpu().numpy()
        return normalize_embedding(value), len(signal) / 16000

    for index, (key, row) in enumerate(audio.items()):
        embeddings[key], durations[key] = extract(row["path"])
        print(f"{index + 1}/{len(audio)} {key}", flush=True)
    # Re-extract the first audio after all intervening inputs: evaluator's own canary.
    first = next(iter(audio))
    repeated, _ = extract(audio[first]["path"])
    repeat_error = float(np.max(np.abs(repeated - embeddings[first])))
    if repeat_error > 1e-6:
        raise RuntimeError(f"Embedding evaluator is not repeatable: {repeat_error}")
    comparisons, summary = score_pairs(audio, pairs, embeddings)
    import transformers
    receipt = {
        "contract": "demo-speaker-embedding-v1", "completed": True,
        "model": "zl389/w2v-bert-2.0_SV/model_lmft_0.14.pth",
        "architecture": "w2v-BERT2.0 + merged LoRA + layer adapters + MFA/ASP + LMFT",
        "strict_checkpoint_load": True, "parameters": sum(p.numel() for p in model.parameters()),
        "embedding_dimension": 256, "dtype": "float32", "batch_size": 1,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device,
        "preprocessing": "mono, torchaudio resample 16kHz, whole utterance; no VAD/trim/padding",
        "scoring": "L2 normalized raw cosine; no mean centering, AS-Norm or QMF",
        "decision_threshold": None, "human_judgment_required": False,
        "formal_speaker_verification_benchmark": False,
        "design": design, "exclusions": [], "audio_count": len(audio),
        "repeat_extraction_max_abs_error": repeat_error,
        "audio": [{"id": k, "duration_s_16k": durations[k],
                   "embedding_l2_norm": float(np.linalg.norm(embeddings[k]))}
                  for k in audio],
        "comparisons": comparisons, "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents concurrent runs from replacing a receipt.
    with args.output.open("x") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
