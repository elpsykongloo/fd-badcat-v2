#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1 concurrent batch replay (吞吐轨 runner — see AGENTS.md 评测并发策略).

Runs many wavs through the ACTOR engine concurrently with perfect isolation:
one engine + VAD + trace + output dir per session. Modes:

  realtime  — real model fns, wall-paced frames. Latency-faithful; use
              --concurrency 1 for official latency numbers (实时轨).
  injected  — recorded decision scripts (golden traces); faster than realtime,
              deterministic. Any concurrency.
  mock      — deterministic policy mocks (CI / smoke).

With real models and the vLLM audio deploy config (max_num_seqs: 1) the server
serializes anyway; the text-only deploy config supports true batching (W2).

Usage:
  python scripts/batch_replay.py --glob 'data/HumDial-FDBench/extracted/test/**/*.wav' \
      --mode mock --concurrency 8 --out-root exp/batch_smoke
"""
import argparse
import asyncio
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402


async def run_one(wav, args, cfg, sem, results):
    from replay_session import MockModels, RecordedScript, load_vad_script, run_actor
    async with sem:
        t0 = time.perf_counter()
        stem = Path(wav).stem
        out_dir = Path(args.out_root) / stem
        mocks = None
        script = None
        mode = args.mode
        if mode == "mock":
            mocks = MockModels(cfg["prompts"], llm_sleep=args.llm_sleep)
            mode = "realtime" if args.paced_mock else "oracle"
            if mode == "oracle":
                # oracle needs a decision script, build from policy mocks
                n = {"k": 0}

                def script(kind, meta, _m=mocks):
                    n["k"] += 1
                    if kind == "asr":
                        return {"text": _m.asr("x"), "infer": 0.0}
                    if kind == "tts":
                        return {"infer": 0.0, "dur_audio": 0.5}
                    return {"text": _m.llm([{"content": meta.get("prompt", "")}]), "infer": 0.0}
        elif mode == "injected":
            golden = [json.loads(l) for l in open(Path(args.golden_dir) / f"{stem}.jsonl",
                                                  encoding="utf-8") if l.strip()]
            script = RecordedScript(golden)

        vad_script = load_vad_script(wav) if args.vad_scripts else None
        try:
            trace, eng = await run_actor(wav, mocks, out_dir, mode=mode, script=script,
                                         vad_script=vad_script)
            trace_path = Path(args.out_root) / "traces" / f"{stem}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            with trace_path.open("w", encoding="utf-8") as f:
                for ev in trace:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            results.append({"wav": str(wav), "events": len(trace),
                            "wall_s": round(time.perf_counter() - t0, 2), "ok": True})
        except Exception as exc:
            results.append({"wav": str(wav), "error": str(exc), "ok": False})
        n_done = len(results)
        print(f"[{n_done}] {stem}: {'ok' if results[-1]['ok'] else 'FAIL'} "
              f"({results[-1].get('wall_s', '-')}s)", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--mode", choices=["realtime", "injected", "mock"], default="mock")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-root", default="exp/batch_replay")
    ap.add_argument("--golden-dir", default="traces/golden")
    ap.add_argument("--vad-scripts", action="store_true",
                    help="use precomputed VAD scripts (skip torch)")
    ap.add_argument("--paced-mock", action="store_true")
    ap.add_argument("--llm-sleep", type=float, default=0.0)
    args = ap.parse_args()

    if args.vad_scripts:
        from replay_session import install_light_stubs
        install_light_stubs()

    with open(ROOT / "src/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    wavs = sorted(glob.glob(args.glob, recursive=True))
    if args.limit:
        wavs = wavs[:args.limit]
    print(f"{len(wavs)} wavs, mode={args.mode}, concurrency={args.concurrency}")
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    t0 = time.perf_counter()
    await asyncio.gather(*[run_one(w, args, cfg, sem, results) for w in wavs])
    ok = sum(1 for r in results if r["ok"])
    report = {"total": len(results), "ok": ok, "wall_s": round(time.perf_counter() - t0, 1),
              "mode": args.mode, "concurrency": args.concurrency, "results": results}
    out = Path(args.out_root) / "batch_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"done: {ok}/{len(results)} ok in {report['wall_s']}s -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
