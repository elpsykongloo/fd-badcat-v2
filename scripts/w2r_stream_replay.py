#!/usr/bin/env python3
"""
w2r_stream_replay.py — W3 D1: the REPLAY DRIVER of the unified TACT engine.

As of W3 this script is no longer a parallel implementation: every semantic rule
(objection window on the silence clock, commit barrier, decision-op application,
parse/repair, ack fallback, prompt/snapshot v2) lives in src/tact_core.py and is
shared verbatim with the live engine (src/engine_b.py TactEngine). This file only
supplies the offline perception + clock:

  * silero VAD (offline, retroactive) segments the recorded input; an EoU fires at
    seg_end + HOLD (0.64 s) iff the following silence lasts through HOLD.
  * At each EoU the REAL decider (Qwen3-Omni via :10004, T=0/seed=42) hears the
    cumulative audio prefix + a PENDING_OPS snapshot and emits launch/patch/cancel.
  * launch -> Transaction.pending; the objection window burns DELTA seconds of
    *silent* audio-clock time (user speech freezes the countdown; patch restarts it).
  * COMMIT BARRIER (default ON — the W3 §一 ruling): expiries landing while a
    decision holding the op in its snapshot is in flight are deferred (dual-stamped),
    the decision's ops apply first, then the sweep commits the unrescued at their
    NOMINAL deadlines. --commit-barrier off is the continuous-clock ABLATION arm:
    expiries commit the instant they fire; late patches drop (patch_after_commit).
  * delta=0 reproduces the eager/dirty regime; blocking = one decision at the end.

Two driver modes:
  --engine core   (default) EoU-granular driver of tact_core — fast path, and the
                  BIT-PARITY path: barrier=on + the W2 decision cache reproduces the
                  W2 grid v1 result files (actual_tool_calls / latency / tx_log).
  --engine full   frames -> the ACTUAL TactEngine (injected replay mode, streaming
                  VADIterator perception, cache-backed decisions). This is the H6
                  instrument: same semantics core, live perception path. tact only.

Latency accounting (audio clock, TTS excluded on both arms by design) and the
decision cache (sha256 over messages) are unchanged from W2.

Usage:
  python scripts/w2r_stream_replay.py --delta 1.5 --provider w3_tact_d150 \
      [--mode tact|blocking] [--commit-barrier on|off] [--engine core|full] \
      [--only-rollback] [--limit N] [--force] [--workers N] [--infer-nominal S] \
      [--cache exp/w2_rerun/decision_cache.json] \
      [--latency-profile official|realistic] [--dag on|off] \
      [--speculative on|off] [--prompt v2|v3|v3.1] [--ids-file PATH]

W3 D4-D6 knobs (ALL default off / official — frozen-grid parity):
  --latency-profile realistic  attach latency_realistic accounting (additive
                               fields; decisions unchanged => full cache reuse)
  --dag on                     arm OpDag patch propagation + compensation plans
  --speculative on             dispatch each EoU decision at VAD SEG END; the
                               result applies no earlier than the EoU
                               (t_dec = seg_end + max(HOLD, infer))
  --prompt v3                  five-target prompt batch (INVALIDATES the cache
                               keys — new decisions; use a fresh provider tag)
  --prompt v3.1                v3 with rules 10/11 tightened + rule 15 added
                               (repairs eco23/eco25/fin19; provider w3p31_*)
  --ids-file                   newline/JSON list of example_ids to run (e.g.
                               exp/w3/tuning30.json — the preregistered subset)
"""
import argparse
import asyncio
import json
import math
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/src")

import tact_core                                            # noqa: E402
from tact_core import (WindowLedger, apply_decision_ops, advance_over,     # noqa: E402
                       build_msgs, decide_from_msgs, ack_fallback, HOLD, SR)
from tact.transaction import Transaction                    # noqa: E402
from tact.tools import ToolRegistry                         # noqa: E402
import latency_realistic                                    # noqa: E402

import os                                                   # noqa: E402

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
BENCH = Path("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json")

_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def load_16k(path):
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        import torch
        import torchaudio
        a = torchaudio.functional.resample(
            torch.from_numpy(a).unsqueeze(0), sr, SR).squeeze(0).numpy()
    return a


_TLS = threading.local()


def vad_segments(audio):
    from silero_vad import load_silero_vad, get_speech_timestamps
    if getattr(_TLS, "vad", None) is None:      # per-thread model: silero is stateful
        _TLS.vad = load_silero_vad()
    ts = get_speech_timestamps(audio, _TLS.vad, sampling_rate=SR,
                               min_silence_duration_ms=400, speech_pad_ms=30)
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def _llm_call(messages):
    """Thread-safe replica of module.llm_qwen3o (identical payload, incl. max_tokens
    256), with a per-thread requests.Session — module.py's shared session is the
    documented concurrency trap. T=0/seed fixed => same outputs, same cache keys."""
    if getattr(_TLS, "http", None) is None:
        s = requests.Session()
        s.trust_env = False
        _TLS.http = s
    payload = {
        "model": os.getenv("FDBC_QWEN_MODEL", "Qwen3-Omni-30B-A3B-Instruct"),
        "temperature": float(os.getenv("FDBC_QWEN_TEMPERATURE", "0")),
        "top_p": float(os.getenv("FDBC_QWEN_TOP_P", "0.7")),
        "top_k": int(os.getenv("FDBC_QWEN_TOP_K", "40")),
        "presence_penalty": float(os.getenv("FDBC_QWEN_PRESENCE_PENALTY", "1.2")),
        "frequency_penalty": float(os.getenv("FDBC_QWEN_FREQUENCY_PENALTY", "0.8")),
        "max_tokens": int(os.getenv("FDBC_QWEN_MAX_TOKENS", "256")),
        "seed": int(os.getenv("FDBC_QWEN_SEED", "42")),
        "modalities": ["text"],
        "messages": messages,
    }
    r = _TLS.http.post(
        os.getenv("FDBC_QWEN_URL", "http://127.0.0.1:10004/v1/chat/completions"),
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=int(os.getenv("FDBC_QWEN_TIMEOUT", "300")),
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


class DecisionCache:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        self.hits = self.misses = 0
        self._lock = threading.Lock()

    def key(self, msgs):
        import hashlib
        return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()

    def call(self, msgs):
        k = self.key(msgs)
        with self._lock:
            if k in self.data:
                self.hits += 1
                e = self.data[k]
                return e["raw"], e["infer"]
        t0 = time.time()
        raw = _llm_call(msgs)                    # network outside the lock
        infer = round(time.time() - t0, 3)
        with self._lock:
            self.misses += 1
            self.data[k] = {"raw": raw, "infer": infer}
        return raw, infer

    def save(self):
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))


# ---------------------------------------------------------------------------
# Shared latency accounting (text-ready anchors, TTS excluded on both arms).
# Formulas are FROZEN from W2 (favor blocking; see w2_rerun_report §3.2).
# ---------------------------------------------------------------------------
def assemble_latency(commits, say_events, t_user_end, mode, n_eou, infer_nominal):
    result_ready = (max(c["t_commit"] + (c.get("tool_wall_s") or 0.0) for c in commits)
                    if commits else None)
    # Wall-free completion anchor (nominal commit stamps only): deterministic under
    # --workers (the measured tool_wall_s is not — global seeded RNG interleaves
    # across threads; see AGENTS). Additive field, W3+; absent from W2 v1 archives.
    nominal_ready = max((c["t_commit"] for c in commits), default=None)
    final_says = [t for t, _ in say_events if t >= t_user_end]
    if mode == "blocking":
        first_response_s = (round(max(0.0, result_ready - t_user_end), 3)
                            if result_ready is not None else None)
        ack_emitted = False
    else:
        if final_says:
            first_response_s = round(min(final_says) - t_user_end, 3)
            ack_emitted = True
        else:
            first_response_s = (round(max(0.0, result_ready - t_user_end), 3)
                                if result_ready is not None else None)
            ack_emitted = False
    task_completion_s = (round(max(0.0, result_ready - t_user_end), 3)
                         if result_ready is not None else None)
    return {"first_response_s": first_response_s,
            "ack_emitted": ack_emitted,
            "task_completion_s": task_completion_s,
            "completion_nominal_s": (round(max(0.0, nominal_ready - t_user_end), 3)
                                     if nominal_ready is not None else None),
            "n_eou": n_eou,
            "infer_mode": ("nominal" if infer_nominal is not None else "live")}


# ---------------------------------------------------------------------------
# core mode: EoU-granular driver of tact_core (bit-parity path)
# ---------------------------------------------------------------------------
def run_example(folder, provider, delta, cache, mode="tact", force=False,
                infer_nominal=None, barrier=True, engine="core",
                speculative=False, dag_on=False, lat_profile="official"):
    m = _FOLDER_RE.match(folder.name)
    if not m:
        return None
    example_id, speaker_id = m.group(1), m.group(2)
    meta_p = folder / "metadata.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    out_p = folder / f"result_{provider}.json"
    if out_p.exists() and not force:
        return json.loads(out_p.read_text())

    audio = load_16k(folder / "input.wav")
    segs = vad_segments(audio)
    if not segs:
        return None
    t_user_end = segs[-1][1]

    if engine == "full":
        return _run_example_full(folder, provider, delta, cache, mode, meta,
                                 audio, segs, out_p, infer_nominal, barrier,
                                 example_id, speaker_id, t_user_end,
                                 speculative=speculative, dag_on=dag_on,
                                 lat_profile=lat_profile)

    random.seed(42)  # seeded sandbox latency (W2 D1-2 discipline)
    reg = ToolRegistry(latency_profile=meta.get("latency_profile", "normal"),
                       room=f"w2r-{example_id}",
                       telemetry_path=f"/tmp/w2r_tools_{provider}.log")
    tx = Transaction()
    ledger = WindowLedger(delta, barrier=barrier)
    dag = comp_reg = None
    if dag_on:
        from tact_dag import OpDag, CompensationRegistry
        dag = OpDag(ledger)
        comp_reg = CompensationRegistry()

    # EoU points: seg ends whose following silence lasts through HOLD (+ final seg)
    eous = []
    for i, (s, e) in enumerate(segs):
        nxt = segs[i + 1][0] if i + 1 < len(segs) else None
        if nxt is None or (nxt - e) >= HOLD:
            eous.append((i, e + HOLD))
    if mode == "blocking":
        eous = eous[-1:]                     # decide once, after the full input

    trace = {"segs": segs, "eous": eous, "delta": delta, "mode": mode,
             "decisions": [], "ops": {}, "commits": []}
    say_events = []                           # (t_audio, text)

    def do_commit(op_id, t_nominal, t_actual):
        if op_id not in tx.pending:
            return
        ledger.close(op_id)                   # defensive; commit paths pre-remove
        t0 = time.time()
        tx.commit(op_id, reg.executor, t=t_nominal)
        w = round(time.time() - t0, 3)
        trace["commits"].append({"op_id": op_id, "t_commit": round(t_nominal, 3),
                                 "tool_wall_s": w,
                                 "actual_commit": round(t_actual, 3),
                                 "deferred_s": round(max(0.0, t_actual - t_nominal), 3)})

    cursor = 0.0
    for seg_idx, t_eou in eous:
        # Speculative dispatch (W3 D6): the LLM call starts at VAD SEG END, so
        # inference overlaps the 0.64 hold; the result stays inert until the
        # EoU is confirmed => t_dec = seg_end + max(HOLD, infer). Non-spec:
        # dispatch at the EoU itself => t_dec = t_eou + infer (frozen v1 path).
        t_disp = segs[seg_idx][1] if speculative else t_eou
        # windows expiring up to the dispatch point close first (deadline order)
        advance_over(ledger, cursor, t_disp, segs, do_commit)
        cursor = max(cursor, t_disp)
        prefix = audio[: int(segs[seg_idx][1] * SR)]
        msgs = build_msgs(tx, prefix)
        # BARRIER GUARD registered at dispatch with the snapshot's pending set;
        # expiries inside (t_disp, t_dec] defer under the barrier, commit under
        # the continuous-clock ablation.
        ledger.begin_decision(seg_idx, set(tx.pending))
        dec, infer_live = decide_from_msgs(cache.call, msgs)
        # Audio-clock advance: live wall infer (latency track, serial only) or a
        # fixed nominal (throughput track — makes the sim independent of server
        # contention, so concurrent runs are bit-reproducible).
        infer = infer_nominal if infer_nominal is not None else infer_live
        t_dec = max(t_eou, t_disp + infer)   # spec results gate on EoU confirm
        advance_over(ledger, cursor, t_dec, segs, do_commit)
        cursor = max(cursor, t_dec)

        say = dec.get("say", "") if mode == "blocking" else ack_fallback(dec)
        if say:
            say_events.append((t_dec, say))
        # DecisionDone order (spec §一-2): ops FIRST (patch rescues, window
        # restarts), THEN sweep at the current clock.
        applied = apply_decision_ops(tx, ledger, dec, t_dec,
                                     immediate=(mode == "blocking" or delta <= 0),
                                     commit_cb=do_commit,
                                     dag=dag, comp_registry=comp_reg)
        ledger.end_decision(seg_idx)
        ledger.sweep(t_dec, do_commit, cause="decision_done")
        dec_entry = {"seg_idx": seg_idx, "t_eou": round(t_eou, 3),
                     "infer_s": infer, "say": dec.get("say", ""),
                     "ops": applied,
                     "repaired": bool(dec.get("_repaired"))}
        if speculative:
            dec_entry["t_disp"] = round(t_disp, 3)
            dec_entry["t_dec"] = round(t_dec, 3)
        trace["decisions"].append(dec_entry)

    # end of audio: remaining windows expire at their deadlines on the tail silence
    advance_over(ledger, cursor, math.inf, segs, do_commit)
    ledger.sweep(cursor, do_commit, cause="finalize")

    latency = assemble_latency(trace["commits"], say_events, t_user_end,
                               mode, len(eous), infer_nominal)
    result = {
        "pid": speaker_id, "example_id": example_id,
        "category": meta.get("domain", "unknown"), "title": meta.get("title", ""),
        "provider": provider, "mode": mode, "delta": delta,
        "actual_tool_calls": tx.to_actual_tool_calls(),
        "transcript": say_events[-1][1] if say_events else "",
        "latency": latency,
        "tx_log": tx.log, "trace": trace, "status": "completed",
        "commit_barrier": barrier, "engine": "core",
        "ledger": ledger.export(),
    }
    if speculative:
        result["speculative_dispatch"] = True
    if dag is not None:
        result["dag"] = dag.export()
        if comp_reg is not None and comp_reg.plans:
            result["comp_plans"] = list(comp_reg.plans)
    if lat_profile == "realistic":
        latency_realistic.attach(result, t_user_end,
                                 edges=dag.edges if dag is not None else None)
    out_p.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


# ---------------------------------------------------------------------------
# full mode: frames -> the actual TactEngine (H6 instrument; tact mode only)
# ---------------------------------------------------------------------------
def _run_example_full(folder, provider, delta, cache, mode, meta, audio, segs,
                      out_p, infer_nominal, barrier, example_id, speaker_id,
                      t_user_end, speculative=False, dag_on=False,
                      lat_profile="official"):
    if mode != "tact":
        raise SystemExit("--engine full supports --mode tact only (a live engine "
                         "cannot know an EoU is the last one; blocking is an "
                         "offline evaluation construct — use --engine core).")
    from engine import frames_from_array
    from engine_b import TactEngine

    random.seed(42)
    reg = ToolRegistry(latency_profile=meta.get("latency_profile", "normal"),
                       room=f"w2r-{example_id}",
                       telemetry_path=f"/tmp/w2r_tools_{provider}.log")
    nominal = 1.0 if infer_nominal is None else infer_nominal

    def script(kind, meta_d):
        if kind == "tact":
            dec, infer_live = decide_from_msgs(cache.call, meta_d["messages"])
            return {"decision": dec,
                    "infer": nominal if infer_nominal is not None else infer_live}
        return {"text": "", "infer": 0.0, "wav_path": "", "dur_audio": 0.0}

    vad_model = getattr(_TLS, "vad", None)
    if vad_model is None:
        from silero_vad import load_silero_vad
        _TLS.vad = vad_model = load_silero_vad()

    eng = TactEngine(
        prompts={}, delay={"end_hold_frame": HOLD, "after_continue_time": 2.5},
        llm_cfg={"decision_timeout_s": 300},
        engine_cfg={"phase": "b", "mode": "tact", "delta": delta,
                    "commit_barrier": barrier, "tool_sync": True,
                    "tts_enabled": False, "asr_enabled": False,
                    "speculative_dispatch": speculative, "dag": dag_on},
        llm_fn=lambda m: "", asr_fn=lambda p: "",   # never called: injected replay
        tts_fn=lambda t, **k: ("", 0.0),
        replay_mode="injected", decision_script=script,
        vad_model=vad_model, tool_executor=reg.executor)

    asyncio.run(eng.run_offline(frames_from_array(audio)))
    eng.finalize_windows()

    # latency anchors use OFFLINE VAD segs for cross-mode comparability
    latency = assemble_latency(eng.commit_records, eng.say_events, t_user_end,
                               mode, len(eng.tact_eous), infer_nominal)
    result = {
        "pid": speaker_id, "example_id": example_id,
        "category": meta.get("domain", "unknown"), "title": meta.get("title", ""),
        "provider": provider, "mode": mode, "delta": delta,
        "actual_tool_calls": eng.tx.to_actual_tool_calls(),
        "transcript": eng.say_events[-1][1] if eng.say_events else "",
        "latency": latency,
        "tx_log": eng.tx.log,
        "trace": {"segs": segs, "eous": eng.tact_eous, "delta": delta,
                  "mode": mode, "decisions": eng.tact_decisions, "ops": {},
                  "commits": eng.commit_records},
        "status": "completed",
        "commit_barrier": barrier, "engine": "full",
        "ledger": eng.ledger.export(),
        "engine_trace_events": len(eng.trace),
        "engine_trace_event_counts": dict(Counter(
            ev.get("event") for ev in eng.trace)),
    }
    if speculative:
        result["speculative_dispatch"] = True
    if eng.dag is not None:
        result["dag"] = eng.dag.export()
        if eng.comp_registry is not None and eng.comp_registry.plans:
            result["comp_plans"] = list(eng.comp_registry.plans)
    if lat_profile == "realistic":
        latency_realistic.attach(
            result, t_user_end,
            edges=eng.dag.edges if eng.dag is not None else None)
    out_p.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--mode", choices=["tact", "blocking"], default="tact")
    ap.add_argument("--commit-barrier", choices=["on", "off"], default="on",
                    help="on = decision-atomic commit (W3 §一 ruling, the paper "
                         "semantics); off = continuous-clock ablation arm.")
    ap.add_argument("--engine", choices=["core", "full"], default="core",
                    help="core = EoU-granular tact_core driver (bit-parity path); "
                         "full = frames through the actual TactEngine (H6 probe).")
    ap.add_argument("--only-rollback", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="Thread-pool size for throughput runs. Forces nominal infer "
                         "(see --infer-nominal) so results are bit-reproducible. "
                         "Latency numbers from concurrent runs are NOT authoritative; "
                         "use --workers 1 (A-profile, live infer) for paper numbers.")
    ap.add_argument("--infer-nominal", type=float, default=None,
                    help="Advance the audio clock by this fixed decision time instead "
                         "of measured wall time. Auto-set to 1.0 when --workers>1.")
    ap.add_argument("--cache", default="/root/autodl-tmp/fd-badcat/exp/w2_rerun/decision_cache.json")
    ap.add_argument("--latency-profile", choices=["official", "realistic"],
                    default="official",
                    help="realistic = attach latency_realistic accounting "
                         "(additive fields; decisions/cache unchanged).")
    ap.add_argument("--dag", choices=["on", "off"], default="off",
                    help="on = arm OpDag patch propagation + compensation "
                         "planning (W3 D5; changes window-restart timing).")
    ap.add_argument("--speculative", choices=["on", "off"], default="off",
                    help="on = dispatch decisions at VAD seg end (W3 D6); "
                         "t_dec = seg_end + max(HOLD, infer). Snapshots can "
                         "differ from the frozen grid — use a fresh provider.")
    ap.add_argument("--prompt", choices=["v2", "v3", "v3.1"], default="v2",
                    help="v3 = five-target batch; v3.1 = word-level repair of "
                         "the three v3 regressions (docs/prompt_v3_five_targets"
                         ".md §v3.1). Both INVALIDATE cache keys; new decisions.")
    ap.add_argument("--ids-file", default=None,
                    help="Path to a JSON array or newline list of example_ids "
                         "to run (e.g. exp/w3/tuning30.json).")
    args = ap.parse_args()

    if args.prompt == "v3":
        tact_core.install_prompt_v3()
        if not (args.provider.startswith("w3p3_") or "_p3" in args.provider):
            print("WARNING: --prompt v3 without a p3-marked provider tag; "
                  "recommend w3p3_* so grids stay separable.")
    elif args.prompt == "v3.1":
        tact_core.install_prompt_v31()
        if not (args.provider.startswith("w3p31") or "_p31" in args.provider):
            print("WARNING: --prompt v3.1 without a p31-marked provider tag; "
                  "recommend w3p31_* / w3p31r_* so grids stay separable.")

    barrier = (args.commit_barrier == "on")
    frozen_risk = (not barrier or args.engine != "core" or args.speculative == "on"
                   or args.dag == "on" or args.prompt != "v2")
    if args.provider.startswith("w2r_") and frozen_risk:
        print("WARNING: provider namespace w2r_* is the frozen W2 grid; writing "
              "barrier-off / engine-full / speculative / dag / prompt-v3 results "
              "into it will confuse the archive. Prefer a fresh provider tag "
              "(e.g. w3p_*).")

    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench
    rollback_ids = {x["id"] for x in items if x.get("state_rollback_test")}

    folders = sorted(f for f in DATA.iterdir() if f.is_dir() and _FOLDER_RE.match(f.name))
    if args.only_rollback:
        folders = [f for f in folders
                   if _FOLDER_RE.match(f.name).group(1) in rollback_ids]
    if args.ids_file:
        raw = Path(args.ids_file).read_text()
        try:
            wanted = set(json.loads(raw))
        except json.JSONDecodeError:
            wanted = {ln.strip() for ln in raw.splitlines() if ln.strip()}
        folders = [f for f in folders
                   if _FOLDER_RE.match(f.name).group(1) in wanted]
    if args.limit:
        folders = folders[: args.limit]

    cache = DecisionCache(args.cache)
    infer_nominal = args.infer_nominal
    if args.workers > 1 and infer_nominal is None:
        infer_nominal = 1.0
        print("NOTE: --workers>1 forces nominal infer=1.0s (throughput track; "
              "latency fields not authoritative)")
    print(f"streaming replay | provider={args.provider} mode={args.mode} "
          f"delta={args.delta} barrier={args.commit_barrier} engine={args.engine} "
          f"spec={args.speculative} dag={args.dag} prompt={args.prompt} "
          f"latprof={args.latency_profile} workers={args.workers} "
          f"infer={'nominal %.2f' % infer_nominal if infer_nominal is not None else 'live'} "
          f"| {len(folders)} examples")

    done_count = [0]
    count_lock = threading.Lock()

    def _one(f):
        try:
            r = run_example(f, args.provider, args.delta, cache,
                            mode=args.mode, force=args.force,
                            infer_nominal=infer_nominal,
                            barrier=barrier, engine=args.engine,
                            speculative=(args.speculative == "on"),
                            dag_on=(args.dag == "on"),
                            lat_profile=args.latency_profile)
            with count_lock:
                done_count[0] += 1
                i = done_count[0]
            if r:
                lat = r["latency"]
                print(f"[{i}/{len(folders)}] {f.name}: "
                      f"{len(r['actual_tool_calls'])} calls, "
                      f"resp={lat.get('first_response_s')} done={lat['task_completion_s']} "
                      f"eou={lat['n_eou']} "
                      f"defer={len(r.get('ledger', {}).get('deferrals', []))}")
            if i % 10 == 0:
                cache.save()
        except Exception as e:
            print(f"{f.name}: ERROR {type(e).__name__} {e}")

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(_one, folders))
    else:
        for f in folders:
            _one(f)
    cache.save()
    print(f"cache: {cache.hits} hits / {cache.misses} misses")


if __name__ == "__main__":
    main()
