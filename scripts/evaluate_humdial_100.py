#!/usr/bin/env python3
"""Prepare transcripts and run DeepSeek judge for the 100-sample HumDial run."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "data/HumDial-FDBench/extracted/test"
DEFAULT_EVAL_ROOT = ROOT / "logs/humdial_100_eval"
NOISY_SUMMARY = ROOT / "logs/humdial_100/summary.csv"
CLEAN_SUMMARY = ROOT / "logs/humdial_100_clean_refs/summary.csv"
MANIFEST = ROOT / "logs/humdial_100/manifest.jsonl"

INTERRUPTION_CATEGORIES = ["ask", "deny", "repeat", "shift", "wait"]
FIRST_RESPONSE_CATEGORIES = [
    "ask",
    "deny",
    "repeat",
    "shift",
    "wait",
    "pause",
    "talk_to_others",
    "others_talk_to_user_after",
    "backchannel",
    "others_talk_to_user_before",
]
REJECTION_DEEPSEEK_CATEGORIES = {
    "backchannel",
    "talk_to_others",
    "others_talk_to_user_after",
}
NO_CLEAN_CATEGORIES = {"pause", "others_talk_to_user_before"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def load_ok_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = load_summary(path)
    return {
        sample_id: row
        for sample_id, row in rows.items()
        if row.get("status") == "ok"
        and row.get("output_path")
        and Path(row["output_path"]).exists()
    }


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_replace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def copy_json(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prefix_from_relpath(relpath: str) -> str:
    name = Path(relpath).stem
    return name.removeprefix("clean_")


def lang_short(lang: str) -> str:
    return "cn" if lang.startswith("cn") else "en"


def data_dir(eval_root: Path, category: str) -> Path:
    return eval_root / "data" / category


def transcribe_dir(eval_root: Path, lang: str, category: str) -> Path:
    return eval_root / "transcribe" / lang / category


def second_transcribe_dir(eval_root: Path, lang: str) -> Path:
    return eval_root / "transcribe_second" / lang / "talk_to_others"


def sync_eval_layout(
    eval_root: Path,
    *,
    manifest_path: Path = MANIFEST,
    noisy_summary_path: Path = NOISY_SUMMARY,
    clean_summary_path: Path = CLEAN_SUMMARY,
    require_clean: bool = False,
) -> dict[str, int]:
    manifest = load_jsonl(manifest_path)
    noisy_summary = load_ok_summary(noisy_summary_path)
    clean_summary = load_ok_summary(clean_summary_path)
    meta_rows = []
    synced_noisy = 0
    synced_clean = 0
    skipped_noisy = 0
    skipped_clean = 0

    for item in manifest:
        sample_id = item["sample_id"]
        category = item["category"]
        lang = item["lang"]
        lang_code = lang_short(lang)
        prefix = prefix_from_relpath(item["relpath"])
        source_wav = Path(item["path"])
        source_stem = source_wav.with_suffix("")
        noisy_row = noisy_summary.get(sample_id)

        task_dir = data_dir(eval_root, category)
        if noisy_row is None:
            skipped_noisy += 1
        else:
            noisy_output = Path(noisy_row["output_path"])
            copy_json(source_stem.with_name(f"{source_stem.name}_timestamp.json"), task_dir / f"{prefix}.json")
            copy_json(source_stem.with_suffix(".json"), task_dir / f"{prefix}_sentence.json")
            symlink_or_replace(source_wav, task_dir / f"{prefix}.wav")
            symlink_or_replace(noisy_output, task_dir / f"{prefix}_output.wav")
            symlink_or_replace(noisy_output, transcribe_dir(eval_root, lang_code, category) / f"{prefix}_output.wav")
            synced_noisy += 1

        if category not in NO_CLEAN_CATEGORIES:
            clean_sample_id = f"clean_{sample_id}"
            clean_row = clean_summary.get(clean_sample_id)
            if clean_row is None:
                skipped_clean += 1
                if require_clean:
                    raise KeyError(f"Missing clean summary row for {clean_sample_id}")
                meta_rows.append(
                    {
                        "sample_id": sample_id,
                        "category": category,
                        "lang": lang,
                        "lang_short": lang_code,
                        "prefix": prefix,
                    }
                )
                continue
            clean_wav = source_wav.with_name(f"clean_{source_wav.name}")
            clean_stem = clean_wav.with_suffix("")
            clean_output = Path(clean_row["output_path"])
            copy_json(
                clean_stem.with_name(f"{clean_stem.name}_timestamp.json"),
                task_dir / f"clean_{prefix}.json",
            )
            symlink_or_replace(clean_wav, task_dir / f"clean_{prefix}.wav")
            symlink_or_replace(clean_output, task_dir / f"clean_{prefix}_output.wav")
            symlink_or_replace(
                clean_output,
                transcribe_dir(eval_root, lang_code, category) / f"clean_{prefix}_output.wav",
            )
            synced_clean += 1

        meta_rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "lang": lang,
                "lang_short": lang_code,
                "prefix": prefix,
            }
        )

    with (eval_root / "manifest_eval.jsonl").open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "manifest": len(manifest),
        "synced_noisy": synced_noisy,
        "synced_clean": synced_clean,
        "skipped_noisy": skipped_noisy,
        "skipped_clean": skipped_clean,
    }


def prepare(eval_root: Path) -> None:
    clean_dir(eval_root / "data")
    clean_dir(eval_root / "transcribe")
    clean_dir(eval_root / "transcribe_second")
    clean_dir(eval_root / "json_group")
    stats = sync_eval_layout(eval_root, require_clean=True)
    print(f"Prepared evaluation layout at {eval_root}: {stats}")


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_cmd_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")


def asr_env(asr_threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(asr_threads)
    env["MKL_NUM_THREADS"] = str(asr_threads)
    env["HF_HOME"] = str(ROOT / ".cache/huggingface")
    env["MODELSCOPE_CACHE"] = str(ROOT / ".cache/modelscope")
    env["FDBC_PARAKEET_NEMO"] = str(ROOT / "model/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo")
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def fast_asr_env(asr_threads: int, asr_device: str) -> dict[str, str]:
    env = asr_env(asr_threads)
    if asr_device == "cuda":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def pending_wavs(folder: Path) -> list[Path]:
    return sorted(wav for wav in folder.glob("*.wav") if not wav.with_suffix(".json").exists())


def run_asr_folder(
    *,
    lang: str,
    category: str,
    folder: Path,
    script: Path,
    eval_root: Path,
    env: dict[str, str],
    stage: str,
) -> tuple[str, str, int, Path]:
    wavs = pending_wavs(folder)
    if not wavs:
        return lang, category, 0, Path()

    job_dir = eval_root / "_asr_jobs" / stage / lang / category
    clean_dir(job_dir)
    for wav in wavs:
        symlink_or_replace(wav, job_dir / wav.name)

    log_path = eval_root / "_asr_logs" / stage / lang / f"{category}.log"
    run_cmd_logged([sys.executable, str(script), "--root_dir", str(job_dir), "--gpu", "0"], env=env, log_path=log_path)
    for json_path in job_dir.glob("*.json"):
        copy_json(json_path, folder / json_path.name)
    return lang, category, len(wavs), log_path


def run_asr_jobs(
    *,
    eval_root: Path,
    jobs: list[tuple[str, str, Path, Path]],
    stage: str,
    workers: int,
    asr_threads: int,
) -> None:
    env = asr_env(asr_threads)
    runnable = [(lang, category, folder, script) for lang, category, folder, script in jobs if pending_wavs(folder)]
    total_wavs = sum(len(pending_wavs(folder)) for _, _, folder, _ in runnable)
    if not runnable:
        print(f"[ASR:{stage}] nothing pending")
        return

    print(f"[ASR:{stage}] queued {len(runnable)} folders / {total_wavs} wavs with workers={workers}, threads={asr_threads}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(
                run_asr_folder,
                lang=lang,
                category=category,
                folder=folder,
                script=script,
                eval_root=eval_root,
                env=env,
                stage=stage,
            )
            for lang, category, folder, script in runnable
        ]
        for future in as_completed(futures):
            lang, category, count, log_path = future.result()
            if count:
                print(f"[ASR:{stage}] done {lang}/{category}: {count} wavs ({log_path})", flush=True)


def chunked(items: list[dict[str, str]], chunks: int) -> list[list[dict[str, str]]]:
    if not items:
        return []
    chunks = max(1, min(chunks, len(items)))
    return [items[i::chunks] for i in range(chunks) if items[i::chunks]]


def run_fast_asr_tasks(
    *,
    eval_root: Path,
    tasks_by_lang: dict[str, list[dict[str, str]]],
    stage: str,
    workers: int,
    asr_threads: int,
    asr_device: str,
    asr_batch_size: int,
) -> None:
    all_tasks = sum(len(items) for items in tasks_by_lang.values())
    if all_tasks == 0:
        print(f"[ASR:{stage}] nothing pending")
        return

    manifest_root = eval_root / "_asr_manifests" / stage
    log_root = eval_root / "_asr_logs" / stage
    clean_dir(manifest_root)
    log_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path, Path]] = []
    for lang, tasks in sorted(tasks_by_lang.items()):
        if not tasks:
            continue
        lang_workers = max(1, workers)
        if lang == "en" and asr_device == "cpu":
            # NeMo restores .nemo archives into temporary directories. Running
            # many CPU Parakeet workers concurrently can exhaust the small root
            # filesystem even when enough RAM/CPU is available.
            lang_workers = max(1, int(os.getenv("EN_ASR_WORKERS", "1")))
        lang_shards = chunked(tasks, lang_workers)
        for idx, shard in enumerate(lang_shards):
            manifest_path = manifest_root / f"{lang}_{idx:02d}.jsonl"
            with manifest_path.open("w", encoding="utf-8") as f:
                for task in shard:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")
            log_path = log_root / f"{lang}_{idx:02d}.log"
            jobs.append((lang, manifest_path, log_path))

    print(
        f"[ASR:{stage}] queued {all_tasks} wavs in {len(jobs)} model shards "
        f"with workers={workers}, threads={asr_threads}, device={asr_device}",
        flush=True,
    )
    env = fast_asr_env(asr_threads, asr_device)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = []
        for lang, manifest_path, log_path in jobs:
            cmd = [
                sys.executable,
                str(ROOT / "scripts/fast_transcribe.py"),
                "--manifest",
                str(manifest_path),
                "--lang",
                lang,
                "--device",
                asr_device,
                "--threads",
                str(asr_threads),
                "--batch-size",
                str(asr_batch_size),
            ]
            futures.append(pool.submit(run_cmd_logged, cmd, env, log_path))
        for future in as_completed(futures):
            future.result()


def collect_transcribe_tasks(eval_root: Path, second: bool = False) -> dict[str, list[dict[str, str]]]:
    tasks_by_lang: dict[str, list[dict[str, str]]] = {"cn": [], "en": []}
    root = eval_root / ("transcribe_second" if second else "transcribe")
    for wav_path in sorted(root.glob("*/*/*.wav")):
        json_path = wav_path.with_suffix(".json")
        if json_path.exists():
            continue
        lang = wav_path.relative_to(root).parts[0]
        if lang not in tasks_by_lang:
            continue
        tasks_by_lang[lang].append(
            {
                "wav_path": str(wav_path.resolve()),
                "json_path": str(json_path.resolve()),
            }
        )
    return tasks_by_lang


def run_transcribe(
    eval_root: Path,
    workers: int = 1,
    asr_threads: int = 8,
    asr_engine: str = "fast",
    asr_device: str = "cpu",
    asr_batch_size: int = 8,
) -> None:
    scripts = {
        "cn": ROOT / "evaluation/get_transcript/infer_cn.py",
        "en": ROOT / "evaluation/get_transcript/infer_en.py",
    }
    if asr_engine == "fast":
        run_fast_asr_tasks(
            eval_root=eval_root,
            tasks_by_lang=collect_transcribe_tasks(eval_root, second=False),
            stage="main",
            workers=workers,
            asr_threads=asr_threads,
            asr_device=asr_device,
            asr_batch_size=asr_batch_size,
        )
    else:
        jobs: list[tuple[str, str, Path, Path]] = []
        for lang in ["cn", "en"]:
            base = eval_root / "transcribe" / lang
            if not base.exists():
                continue
            for folder in sorted(p for p in base.iterdir() if p.is_dir()):
                if any(folder.glob("*.wav")):
                    jobs.append((lang, folder.name, folder, scripts[lang]))
        run_asr_jobs(eval_root=eval_root, jobs=jobs, stage="main", workers=workers, asr_threads=asr_threads)

    for json_path in (eval_root / "transcribe").glob("*/*/*.json"):
        category = json_path.parent.name
        copy_json(json_path, data_dir(eval_root, category) / json_path.name)


def prepare_speech_directed_second(eval_root: Path, incremental: bool = False) -> None:
    first_dir = data_dir(eval_root, "talk_to_others")
    second_dir = eval_root / "data" / "talk_to_others_second"
    if incremental:
        second_dir.mkdir(parents=True, exist_ok=True)
    else:
        clean_dir(second_dir)
    if not first_dir.exists():
        return
    run_cmd(
        [
            sys.executable,
            str(ROOT / "evaluation/rejection/Speech_Directe_at_Others/second_step/prepare_for_eval_first.py"),
            "--input_dir",
            str(first_dir),
            "--output_dir",
            str(second_dir),
        ]
    )

    meta_rows = load_jsonl(eval_root / "manifest_eval.jsonl")
    prefix_to_lang = {
        row["prefix"]: row["lang_short"]
        for row in meta_rows
        if row["category"] == "talk_to_others"
    }
    for wav in second_dir.glob("*.wav"):
        prefix = wav.stem.removesuffix("_output")
        lang = prefix_to_lang[prefix]
        symlink_or_replace(wav, second_transcribe_dir(eval_root, lang) / wav.name)


def run_second_transcribe(
    eval_root: Path,
    workers: int = 1,
    asr_threads: int = 8,
    asr_engine: str = "fast",
    asr_device: str = "cpu",
    asr_batch_size: int = 8,
) -> None:
    scripts = {
        "cn": ROOT / "evaluation/rejection/Speech_Directe_at_Others/second_step/get_transcript_second/infer_cn.py",
        "en": ROOT / "evaluation/rejection/Speech_Directe_at_Others/second_step/get_transcript_second/infer_en.py",
    }
    if asr_engine == "fast":
        run_fast_asr_tasks(
            eval_root=eval_root,
            tasks_by_lang=collect_transcribe_tasks(eval_root, second=True),
            stage="second",
            workers=workers,
            asr_threads=asr_threads,
            asr_device=asr_device,
            asr_batch_size=asr_batch_size,
        )
    else:
        jobs: list[tuple[str, str, Path, Path]] = []
        for lang in ["cn", "en"]:
            folder = second_transcribe_dir(eval_root, lang)
            if folder.exists() and any(folder.glob("*.wav")):
                jobs.append((lang, "talk_to_others", folder, scripts[lang]))
        run_asr_jobs(eval_root=eval_root, jobs=jobs, stage="second", workers=workers, asr_threads=asr_threads)

    second_dir = eval_root / "data" / "talk_to_others_second"
    for json_path in (eval_root / "transcribe_second").glob("*/*/*.json"):
        copy_json(json_path, second_dir / json_path.name)


def read_instruction() -> str:
    return (ROOT / "evaluation/instruction/behavior.txt").read_text(encoding="utf-8")


def load_eval_helpers():
    path = ROOT / "evaluation/interruption/eval.py"
    spec = importlib.util.spec_from_file_location("fdbc_eval_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load eval helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_timing_helpers():
    if os.getenv("FDBC_TIMING_USE_ORIGINAL", "0") != "1":
        return LocalTimingHelpers()

    path = ROOT / "evaluation/interruption/get_timing.py"
    spec = importlib.util.spec_from_file_location("fdbc_timing_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load timing helpers from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    patch_timing_audio_loader(mod)
    return mod


def patch_timing_audio_loader(mod: Any) -> None:
    original_load_wav = mod.load_wav

    def load_wav(path: Path):
        try:
            return original_load_wav(path)
        except ImportError as exc:
            if "TorchCodec" not in str(exc):
                raise

        import numpy as np
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly

        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != mod.SR:
            gcd = np.gcd(sr, mod.SR)
            mono = resample_poly(mono, mod.SR // gcd, sr // gcd).astype("float32")
        return torch.from_numpy(np.ascontiguousarray(mono))

    mod.load_wav = load_wav


def merge_timing_segments(segments: list[tuple[float, float]], gap_threshold: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    merged = [sorted(segments)[0]]
    for start, end in sorted(segments)[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_threshold:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def timing_overlaps(user_segments: list[tuple[float, float]], model_segments: list[tuple[float, float]]) -> list[list[float]]:
    raw = []
    i = j = 0
    while i < len(user_segments) and j < len(model_segments):
        user_start, user_end = user_segments[i]
        model_start, model_end = model_segments[j]
        start = max(user_start, model_start)
        end = min(user_end, model_end)
        if end > start:
            raw.append((start, end))
        if user_end < model_end:
            i += 1
        else:
            j += 1

    best = {}
    for start, end in raw:
        key = int(round(end * 1000))
        if key not in best or (end - start) < (best[key][1] - best[key][0]):
            best[key] = (start, end)
    return [[round(start, 3), round(end, 3)] for start, end in sorted(best.values(), key=lambda item: item[1])]


def timing_response_gaps(user_segments: list[tuple[float, float]], model_segments: list[tuple[float, float]]) -> list[list[float]]:
    model_starts = [start for start, _ in model_segments]
    by_start = {}
    for _, user_end in user_segments:
        next_model_start = next((start for start in model_starts if start > user_end), None)
        if next_model_start is None:
            continue
        key = int(round(next_model_start * 1000))
        candidate = [round(user_end, 3), round(next_model_start, 3)]
        if key not in by_start or candidate[0] > by_start[key][0]:
            by_start[key] = candidate
    return [interval for _, interval in sorted(by_start.items(), key=lambda item: item[1][1])]


class LocalTimingHelpers:
    SR = 16000
    USER_MERGE_GAP = 0.6
    MODEL_MERGE_GAP = 0.5

    def __init__(self) -> None:
        from silero_vad import get_speech_timestamps, load_silero_vad

        self.model = load_silero_vad()
        self.get_speech_timestamps = get_speech_timestamps

    def load_wav(self, path: Path):
        import numpy as np
        import soundfile as sf
        import torch
        from scipy.signal import resample_poly

        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != self.SR:
            gcd = np.gcd(sr, self.SR)
            mono = resample_poly(mono, self.SR // gcd, sr // gcd).astype("float32")
        return torch.from_numpy(np.ascontiguousarray(mono))

    def speech_segments(self, wav, gap_threshold: float) -> list[tuple[float, float]]:
        timestamps = self.get_speech_timestamps(wav, self.model, sampling_rate=self.SR)
        return merge_timing_segments(
            [(item["start"] / self.SR, item["end"] / self.SR) for item in timestamps],
            gap_threshold,
        )

    def process_file_pair(self, user_wav: Path, model_wav: Path) -> dict[str, Any]:
        user_segments = self.speech_segments(self.load_wav(user_wav), self.USER_MERGE_GAP)
        model_segments = self.speech_segments(self.load_wav(model_wav), self.MODEL_MERGE_GAP)
        return {
            "latency_stop_list": timing_overlaps(user_segments, model_segments),
            "latency_resp_list": timing_response_gaps(user_segments, model_segments),
        }

    def calculate_average_latency(self, results: list[dict[str, Any]]) -> dict[str, float]:
        total_stop = 0.0
        count_stop = 0
        total_resp = 0.0
        count_resp = 0
        for result in results:
            for start, end in result["latency_stop_list"]:
                total_stop += end - start
                count_stop += 1
            for start, end in result["latency_resp_list"]:
                total_resp += end - start
                count_resp += 1
        return {
            "avg_latency_stop": total_stop / count_stop if count_stop else 0.0,
            "avg_latency_resp": total_resp / count_resp if count_resp else 0.0,
        }


def collect_groups(folder: Path) -> dict[str, dict[str, Path]]:
    helper = load_eval_helpers()
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for file_name in os.listdir(folder):
        if not file_name.endswith(".json"):
            continue
        prefix, file_type = helper.get_file_group(file_name)
        if prefix and file_type:
            groups[prefix][file_type] = folder / file_name
    return groups


def call_deepseek(client: Any, system_msg: str, user_msg: str, retries: int = 4) -> dict[str, Any]:
    helper = load_eval_helpers()
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                stream=False,
            )
            prediction = response.choices[0].message.content.strip().replace("\n", "")
            return helper.parse_eval(prediction)
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"DeepSeek error attempt {attempt}/{retries}: {exc}. Retrying...", flush=True)
            time.sleep(5)
    raise RuntimeError("unreachable")


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda item: item["key"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def build_judge_row(category: str, prefix: str, files: dict[str, Path], client: Any, instruction: str) -> dict[str, Any]:
    helper = load_eval_helpers()
    data = {}
    for file_type, path in files.items():
        with path.open("r", encoding="utf-8") as f:
            data[file_type] = json.load(f)
    final_input = {
        "input_clean": helper.json_dict_to_compact_text(data["input_clean"]),
        "input_noisy": helper.json_dict_to_compact_text(data["input_noisy"]),
        "output_clean": helper.json_dict_to_compact_text(data["output_clean"]),
        "output_noisy": helper.json_dict_to_compact_text(data["output_noisy"]),
    }
    result = call_deepseek(
        client=client,
        system_msg=instruction,
        user_msg=json.dumps(final_input, separators=(",", ":"), ensure_ascii=False),
    )
    return {"key": f"{category}_{prefix}", "behaviour": result.get("behaviour", [])}


def judge_folder(eval_root: Path, category: str, folder: Path, client: Any, workers: int = 1) -> dict[str, Any]:
    instruction = read_instruction()
    output_dir = eval_root / "json_group" / category
    output_dir.mkdir(parents=True, exist_ok=True)
    content_path = output_dir / f"{category}_content_tags.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if content_path.exists():
        with content_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                done[item["key"]] = item

    groups = collect_groups(folder)
    required = {"input_clean", "input_noisy", "output_clean", "output_noisy"}
    rows_by_key: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, dict[str, Path]]] = []
    for prefix, files in sorted(groups.items()):
        key = f"{category}_{prefix}"
        if key in done:
            rows_by_key[key] = done[key]
            continue
        if not required.issubset(files):
            print(f"Skipping {category}/{prefix}: missing {sorted(required - set(files))}")
            continue
        pending.append((prefix, files))

    if pending:
        print(f"[DeepSeek] {category}: queued {len(pending)} items with workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(build_judge_row, category, prefix, files, client, instruction): prefix
                for prefix, files in pending
            }
            for future in as_completed(futures):
                row = future.result()
                rows_by_key[row["key"]] = row
                write_jsonl_atomic(content_path, list(rows_by_key.values()))
                print(f"{row['key']}: {row['behaviour']}", flush=True)

    rows = sorted(rows_by_key.values(), key=lambda item: item["key"])
    if rows:
        write_jsonl_atomic(content_path, rows)

    counts = Counter(tag for row in rows for tag in row.get("behaviour", []))
    return {
        "category": category,
        "n": len(rows),
        "counts": dict(counts),
        "respond_rate": counts["C_RESPOND"] / len(rows) if rows else 0.0,
        "resume_rate": counts["C_RESUME"] / len(rows) if rows else 0.0,
        "content_tags": str(content_path),
    }


def judge_specs(eval_root: Path) -> list[tuple[str, Path]]:
    return [
        *[(category, data_dir(eval_root, category)) for category in ["ask", "deny", "repeat", "shift", "wait", "backchannel", "others_talk_to_user_after"]],
        ("talk_to_others", eval_root / "data" / "talk_to_others_second"),
    ]


def summarize_judge_rows(category: str, rows: list[dict[str, Any]], content_path: Path) -> dict[str, Any]:
    counts = Counter(tag for row in rows for tag in row.get("behaviour", []))
    return {
        "category": category,
        "n": len(rows),
        "counts": dict(counts),
        "respond_rate": counts["C_RESPOND"] / len(rows) if rows else 0.0,
        "resume_rate": counts["C_RESUME"] / len(rows) if rows else 0.0,
        "content_tags": str(content_path),
    }


def judge_ready(eval_root: Path, client: Any, workers: int = 1) -> list[dict[str, Any]]:
    instruction = read_instruction()
    required = {"input_clean", "input_noisy", "output_clean", "output_noisy"}
    rows_by_category: dict[str, dict[str, dict[str, Any]]] = {}
    paths_by_category: dict[str, Path] = {}
    pending: list[tuple[str, str, dict[str, Path]]] = []

    for category, folder in judge_specs(eval_root):
        output_dir = eval_root / "json_group" / category
        output_dir.mkdir(parents=True, exist_ok=True)
        content_path = output_dir / f"{category}_content_tags.jsonl"
        paths_by_category[category] = content_path
        rows_by_category[category] = {}
        if content_path.exists():
            with content_path.open("r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    rows_by_category[category][item["key"]] = item

        if not folder.exists():
            continue
        groups = collect_groups(folder)
        for prefix, files in sorted(groups.items()):
            key = f"{category}_{prefix}"
            if key in rows_by_category[category]:
                continue
            if required.issubset(files):
                pending.append((category, prefix, files))

    if pending:
        print(f"[DeepSeek] queued {len(pending)} ready items globally with workers={workers}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(build_judge_row, category, prefix, files, client, instruction): category
                for category, prefix, files in pending
            }
            for future in as_completed(futures):
                category = futures[future]
                row = future.result()
                rows_by_category[category][row["key"]] = row
                write_jsonl_atomic(paths_by_category[category], list(rows_by_category[category].values()))
                print(f"{row['key']}: {row['behaviour']}", flush=True)

    results = []
    for category, _ in judge_specs(eval_root):
        rows = sorted(rows_by_category.get(category, {}).values(), key=lambda item: item["key"])
        if rows:
            write_jsonl_atomic(paths_by_category[category], rows)
        results.append(summarize_judge_rows(category, rows, paths_by_category[category]))
    return results


def rejection_rate(folder: Path, before: bool = False) -> dict[str, Any]:
    total = 0
    ahead = 0
    index = 1 if before else 0
    for origin in sorted(folder.glob("*_sentence.json")):
        prefix = origin.name.removesuffix("_sentence.json")
        output = folder / f"{prefix}_output.json"
        if not output.exists():
            continue
        origin_data = json.load(origin.open("r", encoding="utf-8"))
        output_data = json.load(output.open("r", encoding="utf-8"))
        if not output_data.get("chunks"):
            continue
        total += 1
        diff = output_data["chunks"][0]["timestamp"][0] - origin_data["speech_segments"][index]["xmax"]
        if diff > 0:
            ahead += 1
    return {"total": total, "ahead": ahead, "ratio": ahead / total if total else 0.0}


def first_response_delay(folder: Path, before: bool = False, nonnegative: bool = False) -> dict[str, Any]:
    diffs: list[float] = []
    index = 1 if before else 0
    for origin in sorted(folder.glob("*_sentence.json")):
        prefix = origin.name.removesuffix("_sentence.json")
        output = folder / f"{prefix}_output.json"
        if not output.exists():
            continue
        origin_data = json.load(origin.open("r", encoding="utf-8"))
        output_data = json.load(output.open("r", encoding="utf-8"))
        if not output_data.get("chunks"):
            continue
        diff = output_data["chunks"][0]["timestamp"][0] - origin_data["speech_segments"][index]["xmax"]
        if diff > 0 or (nonnegative and diff >= 0):
            diffs.append(diff)
    return {
        "n": len(diffs),
        "avg": sum(diffs) / len(diffs) if diffs else 0.0,
        "values": diffs,
    }


def first_response_delay_for_prefixes(
    folder: Path,
    prefixes: list[str],
    *,
    before: bool = False,
    nonnegative: bool = False,
) -> dict[str, Any]:
    diffs: list[float] = []
    index = 1 if before else 0
    seen = 0
    for prefix in sorted(prefixes):
        origin = folder / f"{prefix}_sentence.json"
        output = folder / f"{prefix}_output.json"
        if not origin.exists() or not output.exists():
            continue
        origin_data = json.load(origin.open("r", encoding="utf-8"))
        output_data = json.load(output.open("r", encoding="utf-8"))
        if len(origin_data.get("speech_segments", [])) <= index:
            continue
        seen += 1
        if not output_data.get("chunks"):
            continue
        diff = output_data["chunks"][0]["timestamp"][0] - origin_data["speech_segments"][index]["xmax"]
        if diff > 0 or (nonnegative and diff >= 0):
            diffs.append(diff)
    return {
        "prefix_count": len(prefixes),
        "seen": seen,
        "n": len(diffs),
        "avg": sum(diffs) / len(diffs) if diffs else 0.0,
        "values": diffs,
    }


def manifest_rows(eval_root: Path) -> list[dict[str, Any]]:
    path = eval_root / "manifest_eval.jsonl"
    if path.exists():
        return load_jsonl(path)
    rows: list[dict[str, Any]] = []
    for category_dir in sorted((eval_root / "data").glob("*")):
        if not category_dir.is_dir() or category_dir.name.endswith("_second"):
            continue
        for wav in sorted(category_dir.glob("*.wav")):
            if wav.stem.endswith("_output") or wav.stem.startswith("clean_"):
                continue
            rows.append(
                {
                    "category": category_dir.name,
                    "prefix": wav.stem,
                    "lang_short": "unknown",
                }
            )
    return rows


def prefixes_by_category_lang(eval_root: Path) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in manifest_rows(eval_root):
        category = row.get("category")
        prefix = row.get("prefix")
        lang = row.get("lang_short") or lang_short(row.get("lang", "unknown"))
        if category and prefix:
            grouped[category][lang].append(prefix)
    return {
        category: {lang: sorted(set(prefixes)) for lang, prefixes in langs.items()}
        for category, langs in grouped.items()
    }


def timing_current(path: Path, prefixes: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        data = json.load(path.open("r", encoding="utf-8"))
    except Exception:
        return False
    return sorted(data.get("prefixes", [])) == sorted(prefixes)


def timing_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "samples": len(results),
        "stop_intervals": sum(len(item.get("latency_stop_list", [])) for item in results),
        "resp_intervals": sum(len(item.get("latency_resp_list", [])) for item in results),
    }


def run_timing_subset(
    *,
    helper: Any,
    folder: Path,
    prefixes: list[str],
    category: str,
    lang: str,
    output_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    prefixes = sorted(set(prefixes))
    if not force and timing_current(output_path, prefixes):
        return json.load(output_path.open("r", encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    processed: list[str] = []
    for prefix in prefixes:
        user_wav = folder / f"{prefix}.wav"
        model_wav = folder / f"{prefix}_output.wav"
        if not user_wav.exists() or not model_wav.exists():
            continue
        result = helper.process_file_pair(user_wav, model_wav)
        results.append(
            {
                "key": f"{category}_{prefix}",
                "latency_stop_list": result["latency_stop_list"],
                "latency_resp_list": result["latency_resp_list"],
            }
        )
        processed.append(prefix)

    final_output = {
        "category": category,
        "lang": lang,
        "prefixes": prefixes,
        "processed_prefixes": processed,
        "results": results,
        "average_latency": helper.calculate_average_latency(results),
        "counts": timing_counts(results),
    }
    output_path.write_text(json.dumps(final_output, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_output


def compute_interruption_latency(eval_root: Path, force: bool = False) -> dict[str, Any]:
    grouped = prefixes_by_category_lang(eval_root)
    helper = None
    by_category_lang: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}
    stop_values: list[float] = []
    resp_values: list[float] = []

    for category in INTERRUPTION_CATEGORIES:
        folder = data_dir(eval_root, category)
        lang_rows = grouped.get(category, {})
        category_stop: list[float] = []
        category_resp: list[float] = []
        category_counts = {"samples": 0, "stop_intervals": 0, "resp_intervals": 0}
        for lang, prefixes in sorted(lang_rows.items()):
            if not prefixes:
                continue
            output_path = eval_root / "json_group" / category / f"{category}_{lang}_latency_results.json"
            if force or not timing_current(output_path, prefixes):
                if helper is None:
                    helper = load_timing_helpers()
                print(f"[timing] {category}/{lang}: {len(prefixes)} samples", flush=True)
                timing = run_timing_subset(
                    helper=helper,
                    folder=folder,
                    prefixes=prefixes,
                    category=category,
                    lang=lang,
                    output_path=output_path,
                    force=force,
                )
            else:
                timing = json.load(output_path.open("r", encoding="utf-8"))

            avg = timing.get("average_latency", {})
            counts = timing.get("counts", {})
            stop = float(avg.get("avg_latency_stop", 0.0) or 0.0)
            resp = float(avg.get("avg_latency_resp", 0.0) or 0.0)
            key = f"{category}:{lang}"
            by_category_lang[key] = {
                "category": category,
                "lang": lang,
                "samples": counts.get("samples", 0),
                "stop_intervals": counts.get("stop_intervals", 0),
                "resp_intervals": counts.get("resp_intervals", 0),
                "avg_latency_stop": round(stop, 3),
                "avg_latency_resp": round(resp, 3),
                "path": str(output_path),
            }
            stop_values.append(stop)
            resp_values.append(resp)
            category_stop.append(stop)
            category_resp.append(resp)
            for count_key in category_counts:
                category_counts[count_key] += int(counts.get(count_key, 0) or 0)

        by_category[category] = {
            **category_counts,
            "avg_latency_stop": round(sum(category_stop) / len(category_stop), 3) if category_stop else 0.0,
            "avg_latency_resp": round(sum(category_resp) / len(category_resp), 3) if category_resp else 0.0,
        }

    return {
        "by_category_lang": by_category_lang,
        "by_category": by_category,
        "avg_latency_stop": round(sum(stop_values) / len(stop_values), 3) if stop_values else 0.0,
        "avg_latency_resp": round(sum(resp_values) / len(resp_values), 3) if resp_values else 0.0,
    }


def compute_first_response_delay_by_lang(eval_root: Path) -> dict[str, Any]:
    grouped = prefixes_by_category_lang(eval_root)
    by_category_lang: dict[str, dict[str, Any]] = {}
    values: list[float] = []

    for category in FIRST_RESPONSE_CATEGORIES:
        folder = data_dir(eval_root, category)
        for lang, prefixes in sorted(grouped.get(category, {}).items()):
            if not prefixes:
                continue
            result = first_response_delay_for_prefixes(
                folder,
                prefixes,
                before=category == "others_talk_to_user_before",
                nonnegative=category == "backchannel",
            )
            avg = float(result["avg"])
            by_category_lang[f"{category}:{lang}"] = {
                "category": category,
                "lang": lang,
                "prefix_count": result["prefix_count"],
                "seen": result["seen"],
                "n": result["n"],
                "avg": round(avg, 3),
            }
            values.append(avg)

    return {
        "by_category_lang": by_category_lang,
        "global_avg": round(sum(values) / len(values), 3) if values else 0.0,
    }


def existing_judge_results(eval_root: Path) -> list[dict[str, Any]]:
    results = []
    for category, _ in judge_specs(eval_root):
        content_path = eval_root / "json_group" / category / f"{category}_content_tags.jsonl"
        rows: list[dict[str, Any]] = []
        if content_path.exists():
            with content_path.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        results.append(summarize_judge_rows(category, rows, content_path))
    return results


def build_summary(
    eval_root: Path,
    deepseek_results: list[dict[str, Any]],
    *,
    include_timing: bool = True,
    force_timing: bool = False,
) -> dict[str, Any]:
    rates = {
        "pause": rejection_rate(data_dir(eval_root, "pause"), before=False),
        "others_talk_to_user_before": rejection_rate(
            data_dir(eval_root, "others_talk_to_user_before"),
            before=True,
        ),
    }
    delays = {}
    for category in ["ask", "deny", "repeat", "shift", "wait", "pause", "talk_to_others", "others_talk_to_user_after"]:
        delays[category] = first_response_delay(data_dir(eval_root, category))
    delays["backchannel"] = first_response_delay(data_dir(eval_root, "backchannel"), nonnegative=True)
    delays["others_talk_to_user_before"] = first_response_delay(
        data_dir(eval_root, "others_talk_to_user_before"),
        before=True,
    )

    by_category = {row["category"]: row for row in deepseek_results}
    interruption_scores = {
        category: by_category.get(category, {}).get("respond_rate", 0.0)
        for category in INTERRUPTION_CATEGORIES
    }
    rejection_scores = {
        "pause": rates["pause"]["ratio"],
        "backchannel": by_category.get("backchannel", {}).get("resume_rate", 0.0),
        "talk_to_others": by_category.get("talk_to_others", {}).get("resume_rate", 0.0),
        "third_party_speech": (
            by_category.get("others_talk_to_user_after", {}).get("resume_rate", 0.0)
            + rates["others_talk_to_user_before"]["ratio"]
        )
        / 2,
    }
    first_response_by_lang = compute_first_response_delay_by_lang(eval_root)
    interruption_latency = compute_interruption_latency(eval_root, force=force_timing) if include_timing else {}
    interruption_avg = sum(interruption_scores.values()) / len(interruption_scores)
    rejection_avg = sum(rejection_scores.values()) / len(rejection_scores)
    first_response_global = float(first_response_by_lang["global_avg"])
    score_summary = {
        "Interruption Total Score": round(interruption_avg * 100, 2),
        "Rejection Total Score": round(rejection_avg * 100, 2),
        "Overall Score": round(((interruption_avg + rejection_avg) / 2) * 100, 2),
        "First Response Delay": round(first_response_global, 3),
    }
    if include_timing:
        total_delay = (
            interruption_latency["avg_latency_stop"]
            + interruption_latency["avg_latency_resp"]
            + first_response_global
        ) / 3
        score_summary.update(
            {
                "avg_latency_stop": interruption_latency["avg_latency_stop"],
                "avg_latency_resp": interruption_latency["avg_latency_resp"],
                "Total Delay": round(total_delay, 3),
            }
        )

    summary = {
        "deepseek_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "deepseek_results": deepseek_results,
        "rejection_rates": rates,
        "first_response_delay": {
            key: {"n": value["n"], "avg": round(value["avg"], 3)}
            for key, value in delays.items()
        },
        "first_response_delay_by_category_lang": first_response_by_lang["by_category_lang"],
        "interruption_scores": {
            key: round(value, 3) for key, value in interruption_scores.items()
        },
        "interruption_respond_avg": round(interruption_avg, 3),
        "rejection_scores": {
            key: round(value, 3) for key, value in rejection_scores.items()
        },
        "rejection_resume_avg": round(rejection_avg, 3),
        "score_summary": score_summary,
    }
    if include_timing:
        summary["interruption_latency"] = interruption_latency
    return summary


def write_summary(
    eval_root: Path,
    deepseek_results: list[dict[str, Any]],
    *,
    include_timing: bool = True,
    force_timing: bool = False,
) -> dict[str, Any]:
    summary = build_summary(
        eval_root,
        deepseek_results,
        include_timing=include_timing,
        force_timing=force_timing,
    )
    output = eval_root / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")
    return summary


def score_existing(eval_root: Path, *, force_timing: bool = False) -> dict[str, Any]:
    return write_summary(eval_root, existing_judge_results(eval_root), include_timing=True, force_timing=force_timing)


def judge(
    eval_root: Path,
    workers: int = 1,
    *,
    include_timing: bool = True,
    force_timing: bool = False,
) -> None:
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    for proxy_var in ["all_proxy", "ALL_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(proxy_var, None)
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "120")),
    )

    deepseek_results = judge_ready(eval_root, client, workers=workers)
    write_summary(
        eval_root,
        deepseek_results,
        include_timing=include_timing,
        force_timing=force_timing,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--asr-workers", type=int, default=int(os.getenv("ASR_WORKERS", "8")))
    parser.add_argument("--asr-threads", type=int, default=int(os.getenv("ASR_THREADS", "4")))
    parser.add_argument("--asr-engine", choices=["fast", "folder"], default=os.getenv("ASR_ENGINE", "fast"))
    parser.add_argument("--asr-device", choices=["cpu", "cuda"], default=os.getenv("ASR_DEVICE", "cpu"))
    parser.add_argument("--asr-batch-size", type=int, default=int(os.getenv("ASR_BATCH_SIZE", "8")))
    parser.add_argument("--judge-workers", type=int, default=int(os.getenv("DEEPSEEK_WORKERS", "16")))
    parser.add_argument("--incremental-second", action="store_true")
    parser.add_argument("--force-timing", action="store_true")
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "sync",
            "transcribe",
            "prepare-second",
            "transcribe-second",
            "judge",
            "score",
            "all-no-judge",
        ],
    )
    args = parser.parse_args()
    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        prepare(eval_root)
    elif args.command == "sync":
        stats = sync_eval_layout(eval_root, require_clean=False)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    elif args.command == "transcribe":
        run_transcribe(
            eval_root,
            workers=args.asr_workers,
            asr_threads=args.asr_threads,
            asr_engine=args.asr_engine,
            asr_device=args.asr_device,
            asr_batch_size=args.asr_batch_size,
        )
    elif args.command == "prepare-second":
        prepare_speech_directed_second(eval_root, incremental=args.incremental_second)
    elif args.command == "transcribe-second":
        run_second_transcribe(
            eval_root,
            workers=args.asr_workers,
            asr_threads=args.asr_threads,
            asr_engine=args.asr_engine,
            asr_device=args.asr_device,
            asr_batch_size=args.asr_batch_size,
        )
    elif args.command == "judge":
        judge(eval_root, workers=args.judge_workers, force_timing=args.force_timing)
    elif args.command == "score":
        score_existing(eval_root, force_timing=args.force_timing)
    elif args.command == "all-no-judge":
        prepare(eval_root)
        run_transcribe(
            eval_root,
            workers=args.asr_workers,
            asr_threads=args.asr_threads,
            asr_engine=args.asr_engine,
            asr_device=args.asr_device,
            asr_batch_size=args.asr_batch_size,
        )
        prepare_speech_directed_second(eval_root)
        run_second_transcribe(
            eval_root,
            workers=args.asr_workers,
            asr_threads=args.asr_threads,
            asr_engine=args.asr_engine,
            asr_device=args.asr_device,
            asr_batch_size=args.asr_batch_size,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
