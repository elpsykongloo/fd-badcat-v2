#!/usr/bin/env python3
"""Fast manifest-based ASR for HumDial evaluation.

Each process loads one ASR model and writes JSON timestamps for every task in
the manifest. This avoids reloading NeMo/FunASR once per category directory.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_tasks(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_transcript(path: Path, text: str, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"text": text, "chunks": chunks}, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )


def configure_device(device: str, gpu: str, threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def transcribe_cn(tasks: list[dict[str, str]], threads: int) -> None:
    from funasr import AutoModel

    model = AutoModel(
        model=os.getenv("FDBC_FUNASR_MODEL", "paraformer-zh"),
        model_revision=os.getenv("FDBC_FUNASR_MODEL_REVISION", "v2.0.4"),
        vad_model=os.getenv("FDBC_FUNASR_VAD_MODEL", "fsmn-vad"),
        vad_model_revision=os.getenv("FDBC_FUNASR_VAD_MODEL_REVISION", "v2.0.4"),
        disable_update=True,
    )
    for task in tasks:
        wav_path = Path(task["wav_path"])
        json_path = Path(task["json_path"])
        if json_path.exists():
            continue
        res = model.generate(input=str(wav_path), batch_size_s=300)
        text_raw = res[0]["text"]
        timestamps = res[0].get("timestamp", [])
        tokens = text_raw.split()
        chunks = [
            {"text": tok, "timestamp": [start / 1000, end / 1000]}
            for tok, (start, end) in zip(tokens, timestamps)
        ]
        write_transcript(json_path, text_raw, chunks)
        print(f"[CN] {wav_path} -> {json_path}", flush=True)


def _nemo_output_to_json(result: Any) -> tuple[str, list[dict[str, Any]]]:
    word_timestamps = result.timestamp["word"]
    chunks = []
    words = []
    for item in word_timestamps:
        word = item["word"]
        words.append(word)
        chunks.append({"text": word, "timestamp": [item["start"], item["end"]]})
    return " ".join(words).strip(), chunks


def transcribe_en(tasks: list[dict[str, str]], device_name: str, gpu: str, batch_size: int) -> None:
    import soundfile as sf
    import torch
    import nemo.collections.asr as nemo_asr

    device = torch.device("cpu")
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0" if gpu else "cuda")

    nemo_path = os.getenv(
        "FDBC_PARAKEET_NEMO",
        "model/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
    )
    if os.path.exists(nemo_path):
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=str(nemo_path)).to(device)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=os.getenv("FDBC_PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v2")
        ).to(device)

    pending = [task for task in tasks if not Path(task["json_path"]).exists()]
    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start : start + max(1, batch_size)]
        tmp_paths: list[str] = []
        try:
            for task in batch:
                wav_path = Path(task["wav_path"])
                waveform, sr = sf.read(str(wav_path))
                if getattr(waveform, "ndim", 1) > 1:
                    waveform = waveform.mean(axis=1)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.close()
                sf.write(tmp.name, waveform, sr)
                tmp_paths.append(tmp.name)

            try:
                outputs = asr_model.transcribe(tmp_paths, timestamps=True, batch_size=max(1, batch_size))
            except TypeError:
                outputs = asr_model.transcribe(tmp_paths, timestamps=True)

            for task, result in zip(batch, outputs):
                text, chunks = _nemo_output_to_json(result)
                json_path = Path(task["json_path"])
                write_transcript(json_path, text, chunks)
                print(f"[EN] {task['wav_path']} -> {json_path}", flush=True)
        finally:
            for tmp_path in tmp_paths:
                try:
                    os.unlink(tmp_path)
                except FileNotFoundError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lang", choices=["cn", "en"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    configure_device(args.device, args.gpu, args.threads)
    tasks = load_tasks(Path(args.manifest))
    if args.lang == "cn":
        transcribe_cn(tasks, args.threads)
    else:
        transcribe_en(tasks, args.device, args.gpu, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
