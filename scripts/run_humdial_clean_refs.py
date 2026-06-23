#!/usr/bin/env python3
"""Run clean-reference counterparts for a HumDial manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import yaml

from run_humdial_batch import Sample, run_samples, safe_id, wav_duration, write_summary


DEFAULT_SKIP_CATEGORIES = {"pause", "others_talk_to_user_before"}


def iter_clean_samples(manifest_path: Path, skip_categories: set[str]) -> list[Sample]:
    samples: list[Sample] = []
    test_root = Path("data/HumDial-FDBench/extracted/test")
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            category = str(item["category"])
            if category in skip_categories:
                continue
            noisy_path = Path(item["path"])
            clean_path = noisy_path.with_name(f"clean_{noisy_path.name}")
            if not clean_path.exists():
                raise FileNotFoundError(f"Missing clean counterpart: {clean_path}")
            rel = clean_path.relative_to(test_root)
            stem = safe_id("_".join(rel.with_suffix("").parts))
            samples.append(
                Sample(
                    sample_id=f"clean_{item['sample_id']}",
                    path=clean_path,
                    relpath=str(rel),
                    lang=str(item["lang"]),
                    category=category,
                    duration=wav_duration(clean_path),
                )
            )
    return samples


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/config.yaml")
    parser.add_argument("--manifest", default="logs/humdial_100/manifest.jsonl")
    parser.add_argument("--out-dir", default="logs/humdial_100_clean_refs")
    parser.add_argument("--exp", default="humdial-100-clean")
    parser.add_argument("--lang", default="batch")
    parser.add_argument("--trailing-silence", type=float, default=2.0)
    parser.add_argument("--post-send-wait", type=float, default=8.0)
    parser.add_argument("--sample-timeout", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.getenv("FDBC_BATCH_WORKERS", "1")))
    parser.add_argument(
        "--skip-category",
        action="append",
        default=sorted(DEFAULT_SKIP_CATEGORIES),
        help="Category that does not need a clean reference run. Can be repeated.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ws_url = f"ws://127.0.0.1:{cfg['server']['port']}/realtime"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = iter_clean_samples(Path(args.manifest), set(args.skip_category))
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.__dict__ | {"path": str(sample.path)}, ensure_ascii=False) + "\n")

    summary_path = out_dir / "summary.csv"
    rows = await run_samples(
        samples=samples,
        ws_url=ws_url,
        out_dir=out_dir,
        summary_path=summary_path,
        exp=args.exp,
        lang_mode=args.lang,
        trailing_silence=args.trailing_silence,
        post_send_wait=args.post_send_wait,
        sample_timeout=args.sample_timeout,
        resume=args.resume,
        workers=args.workers,
    )

    ok = sum(1 for row in rows if row.get("status") == "ok")
    with_tts = sum(
        1
        for row in rows
        if row.get("status") == "ok" and int(row.get("tts_count", 0)) > 0
    )
    errors = sum(1 for row in rows if row.get("status") != "ok")
    print(f"done total={len(rows)} ok={ok} with_tts={with_tts} errors={errors}", flush=True)
    print(f"manifest={out_dir / 'manifest.jsonl'}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
