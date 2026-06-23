#!/usr/bin/env python3
"""
tact/offline_runner.py
======================
Score TACT on FDB-v3 WITHOUT LiveKit.

The official harness streams audio through LiveKit Cloud, records the agent's
audio, and NeMo-ASRs it. That is needed for the *real-time latency / turn-take*
numbers (Track 3). But the three evaluators only read `actual_tool_calls` and
`transcript` from each result_{provider}.json on disk. So for fast iteration on
the transaction logic + Pass@1 / tool-F1 + the 21-scenario rollback subset, we can
emit those JSONs directly from a refactored, callable engine.

What this runner does for each example folder {example_id}_{speaker_id}/:
  1. load input.wav (resample to 16 kHz)
  2. ask the decider (your qwen3-omni-flash) for tool ops over the audio
  3. apply ops to a deterministic Transaction  (launch/patch/cancel/commit)
  4. write result_{provider}.json with:
        actual_tool_calls  = tx.to_actual_tool_calls()    <- scored for Pass@1 / F1
        transcript         = the model's spoken text       <- scored for response acc
        latency            = a MODEL-COMPUTE proxy (NOT the headline latency)
        status             = "completed"

IMPORTANT (honesty): offline mode gives you Pass@1 / F1 / rollback-subset and a
compute-time proxy. The HEADLINE latency + turn-take numbers require the real-time
path (LiveKit agent or the live fd-badcat loop) — see INTEGRATION.md, Track 3.
Offline mode with whole-utterance audio ≈ a cascade (accurate, slow): it validates
the plumbing and the self-correction prompt; the latency WIN shows up only in the
streaming regime.

Run:
  # real model (needs DASHSCOPE_API_KEY and tact/ on the same path as src/module.py)
  python -m tact.offline_runner --data fdb_v3_data_released --provider tact_blocking --mode blocking
  python -m tact.offline_runner --data fdb_v3_data_released --provider tact_async   --mode async

  # structural dry-run (no API, emits empty tool calls): verifies wiring/schema
  python -m tact.offline_runner --data fdb_v3_data_released --provider dryrun --dry-run

Then score with the official evaluators (no LiveKit needed):
  cd Full-Duplex-Bench/v3
  python evaluate_pass_rate.py  --benchmark benchmark_data_v2.json --results-dir <data> --provider tact_blocking --use-llm
  python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir <data> --provider tact_blocking --use-llm
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import torchaudio
    import torch
    _HAVE_TA = True
except Exception:
    _HAVE_TA = False

# tact package imports (run as module: python -m tact.offline_runner)
from .transaction import Transaction
from .tools import ToolRegistry
from .decider import decide_and_apply
from .act_executor import make_act_track
from .module_adapter import llm_text
from .paths import configure_external_paths, default_data_dir

_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def _load_audio_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        if _HAVE_TA:
            data = torchaudio.functional.resample(
                torch.from_numpy(data).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        else:
            # crude linear resample fallback
            n = int(len(data) * 16000 / sr)
            data = np.interp(np.linspace(0, len(data), n, endpoint=False),
                             np.arange(len(data)), data).astype(np.float32)
    return data


def _get_llm_call(dry_run: bool):
    if dry_run:
        # emit a valid JSON object with no ops
        return lambda msgs: '{"dialogue":"speak","ops":[],"say":""}'
    return llm_text


def process_example(folder: Path, provider: str, llm_call, mode: str,
                    latency_profile: str = "normal", force: bool = False) -> dict | None:
    m = _FOLDER_RE.match(folder.name)
    if not m:
        return None
    example_id, speaker_id = m.group(1), m.group(2)
    input_path = folder / "input.wav"
    meta_path = folder / "metadata.json"
    if not input_path.exists():
        return None
    result_path = folder / f"result_{provider}.json"
    if result_path.exists() and not force:
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    # use the scenario's own latency profile if present (so 'occupied silence' is realistic)
    lat = meta.get("latency_profile", latency_profile)
    history = []  # FDB-v3 examples are single-input; keep empty (model hears full audio)

    audio = _load_audio_16k(input_path)
    reg = ToolRegistry(latency_profile=lat, room=f"offline-{example_id}",
                       telemetry_path=f"/tmp/tact_tool_calls_{provider}.log")
    tx = Transaction()
    track = make_act_track(mode)

    t0 = time.time()
    decision = decide_and_apply(
        tx, reg.executor, llm_call, state="LISTEN",
        user_text=None, audio=audio,            # feed the disfluent audio directly
        history=history, t=0.0,
        blocking=(mode == "blocking"),
        async_launcher=(track if mode == "async" else None),
    )
    # in async mode, flush the act track and commit whatever finished
    if mode == "async":
        async def _flush():
            await track.drain()
            for op in track.ready_ops():
                if op.op_id in tx.pending:
                    tx.commit(op.op_id, reg.executor, t=time.time() - t0)
        asyncio.run(_flush())
    model_compute_s = round(time.time() - t0, 3)

    result = {
        "pid": speaker_id,
        "example_id": example_id,
        "category": meta.get("domain", "unknown"),
        "title": meta.get("title", ""),
        "provider": provider,
        "evaluated_at": datetime.datetime.now().isoformat(),
        "mode": mode,
        "actual_tool_calls": tx.to_actual_tool_calls(),   # <- scored
        "transcript": decision.get("say", ""),            # <- scored (agent response)
        "latency": {"model_compute_s": model_compute_s},  # proxy only; see header note
        "tx_log": tx.log,                                  # full audit trail (for case studies)
        "status": "completed",
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default=None,
        help="path to fdb_v3_data_released (default: FDB_V3_DATA_DIR or sibling FDBench_v3/v3 data)",
    )
    ap.add_argument("--provider", default="tact_blocking")
    ap.add_argument("--mode", choices=["blocking", "async"], default="blocking")
    ap.add_argument("--latency-profile", default="normal")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="no API; emit empty tool calls (wiring test)")
    ap.add_argument("--limit", type=int, default=0, help="process only first N folders")
    ap.add_argument("--fd-badcat-root", help="override fd-badcat checkout path")
    ap.add_argument("--fd-badcat-src", help="override fd-badcat src path")
    ap.add_argument("--fdb-v3-dir", help="override FDBench_v3/v3 path")
    args = ap.parse_args()

    if args.fd_badcat_root:
        os.environ["FDBC_REPO_ROOT"] = args.fd_badcat_root
    if args.fd_badcat_src:
        os.environ["FDBC_SRC_DIR"] = args.fd_badcat_src
    if args.fdb_v3_dir:
        os.environ["FDB_V3_DIR"] = args.fdb_v3_dir
    paths = configure_external_paths()

    if args.data is None:
        discovered_data = default_data_dir()
        args.data = str(discovered_data) if discovered_data else None
    if args.data is None:
        print("❌ data dir not configured. Set --data or FDB_V3_DATA_DIR.")
        sys.exit(1)
    root = Path(args.data)
    if not root.exists():
        print(f"❌ data dir not found: {root}")
        sys.exit(1)

    llm_call = _get_llm_call(args.dry_run)
    folders = sorted([f for f in root.iterdir() if f.is_dir() and _FOLDER_RE.match(f.name)])
    if args.limit:
        folders = folders[: args.limit]
    print(f"🚀 offline FDB-v3 | provider={args.provider} mode={args.mode} | {len(folders)} examples")
    print(f"🔗 paths: fd_badcat_src={paths['fd_badcat_src']} fdb_v3_dir={paths['fdb_v3_dir']}")

    n_ok = n_skip = 0
    for i, folder in enumerate(folders, 1):
        try:
            r = process_example(folder, args.provider, llm_call, args.mode,
                                latency_profile=args.latency_profile, force=args.force)
            if r is None:
                n_skip += 1
            else:
                n_ok += 1
                print(f"[{i}/{len(folders)}] {folder.name}: "
                      f"{len(r['actual_tool_calls'])} calls, {r['latency']['model_compute_s']}s")
        except Exception as e:
            print(f"[{i}/{len(folders)}] {folder.name}: ERROR {e}")
    print(f"🏁 done: {n_ok} written, {n_skip} skipped. "
          f"Now run evaluate_pass_rate.py --provider {args.provider}")


if __name__ == "__main__":
    main()
