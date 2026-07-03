#!/usr/bin/env python3
"""
w2r_stream_replay.py — W2 streaming replay: TACT objection-window (Phase-B v0) on FDB-v3.

Faithful-but-minimal streaming simulation on the recorded audio timeline (audio clock):

  * silero VAD segments the input; an EoU fires at seg_end + HOLD (0.64 s) iff the
    following silence really lasts through HOLD (else the user resumed: no EoU).
  * At each EoU the REAL decider (Qwen3-Omni via :10004, T=0/seed=42) hears the
    cumulative audio so far + a PENDING_OPS snapshot, and emits launch/patch/cancel
    ops (model-emitted `commit` ops are ignored: commit timing belongs to the harness).
  * launch -> Transaction.pending; its objection window closes after delta seconds of
    *silent* audio-clock time (user speech pauses the countdown, per W2 plan: switch-type
    speech cancels the timer; the next EoU's decision may patch/cancel the pending op).
  * window expiry -> commit -> executor (official mock APIs, latency profile from the
    scenario metadata, random.seed(42) per example: seeded sandbox, W2 D1-2).
  * delta=0 reproduces the eager/dirty regime; large delta converges to clean trajectories.

Latency accounting (audio clock, TTS excluded on both arms by design — text-ready anchors):
  first_ack_s        : first decision with non-empty `say` after the FINAL user segment end
                       (blocking analog: final decision infer + tool wall time)
  task_completion_s  : last commit completion (commit_t + tool wall) − final user seg end

Decisions are cached on sha256(messages) (T=0 + fixed seed => deterministic), so
identical prefixes across the delta grid are not re-queried; cache hits are logged.

Usage:
  python scripts/w2r_stream_replay.py --delta 0.6 --provider w2r_tact_d060 \
      [--only-rollback] [--limit N] [--fresh] [--cache exp/w2_rerun/decision_cache.json]
"""
import argparse
import glob
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/root/autodl-tmp/tact")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/src")

from tact.transaction import Transaction, Reversibility  # noqa: E402
from tact.tools import ToolRegistry, REVERSIBILITY        # noqa: E402
import tact.decider as _decider                           # noqa: E402
from tact.decider import build_decider_messages, parse_decision  # noqa: E402
from tact.module_adapter import llm_text                  # noqa: E402

# ---------------------------------------------------------------------------
# Prompt v2 (W2 rerun): two documented addenda over tact/decider.py SYSTEM_PROMPT.
# Applied IDENTICALLY to both arms (blocking + TACT) => fair A/B.
# Rationale (rollback-subset failure analysis, see docs/w2_rerun_report.md):
#   R8: housing_21 — model emitted `patch` for the revised filter but DROPPED the
#       new `calculate_commute` launch arriving in the same utterance.
#   R9: 41% of launch decisions returned empty `say`, destroying the ack anchor.
# ---------------------------------------------------------------------------
PROMPT_V2_ADDENDUM = """
ADDITIONAL RULES:
8. If the user BOTH revises an existing pending op AND adds a new request in the
   same utterance, emit the `patch` AND the new `launch` together in the same ops list.
9. `say` must NEVER be empty when ops is non-empty: briefly announce what you are
   doing (e.g. "Updating that to Chicago and checking the commute now.").
"""
_decider.SYSTEM_PROMPT = _decider.SYSTEM_PROMPT + PROMPT_V2_ADDENDUM

# ---------------------------------------------------------------------------
# Snapshot fix (W2 rerun, harness-side, both arms): tact's snapshot_for_prompt
# only lists PENDING ops. In the streaming regime an op committed at window
# expiry vanishes from the snapshot, so at the NEXT EoU the model re-launches
# the same intent => duplicate calls => official precision kill (ecommerce_13).
# Include committed ops as ALREADY-EXECUTED context.
# ---------------------------------------------------------------------------
_orig_snapshot = Transaction.snapshot_for_prompt


def _snapshot_v2(self):
    parts = []
    if self.committed:
        parts.append("ALREADY EXECUTED (do NOT launch these again):")
        for op in self.committed:
            parts.append(f"  - fn={op.fn} args={json.dumps(op.args, ensure_ascii=False)}")
    pend = _orig_snapshot(self)
    parts.append("PENDING (not yet executed, patch/cancel by id):")
    parts.append(pend if pend != "(none)" else "  (none)")
    return "\n".join(parts)


Transaction.snapshot_for_prompt = _snapshot_v2

# Schema-typed argument coercion at the executor boundary (BOTH arms).
# The mock-API schema types these fields as numbers/booleans; the decider sometimes
# emits digit-strings on partial audio ("1800" vs 1800), which the official
# normalizer does not coerce. Standard tool-schema validation, applied uniformly.
NUMERIC_FIELDS = {"amount", "max_price", "bedrooms", "quantity"}
POLY_FIELDS = {"value"}      # update_search_filter.value: bool | number | string


def _coerce_args(args):
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and (k in NUMERIC_FIELDS or k in POLY_FIELDS):
            s = v.strip()
            if s.lower() in ("true", "false"):
                v = s.lower() == "true"
            else:
                try:
                    v = int(s) if re.fullmatch(r"-?\d+", s) else float(s)
                except ValueError:
                    pass
        out[k] = v
    return out

# ack-v0 (W2 plan D4, TACT arm only): template + slot filling when the model
# launched ops but said nothing — the announce IS the objection-window opener.
ACK_TEMPLATES = {
    "search_flights": "Let me search those flights for you.",
    "book_flight": "Booking that flight now.",
    "update_identity_doc": "Updating that document now.",
    "get_card_benefits": "Let me pull up those card benefits.",
    "get_exchange_rate": "Let me get that exchange rate.",
    "modify_autopay": "Setting up that autopay change.",
    "search_apartments": "Searching apartments for you now.",
    "calculate_commute": "Let me check that commute.",
    "update_search_filter": "Updating your search filter.",
    "track_order": "Let me track that order.",
    "search_products": "Searching for that now.",
    "add_to_cart": "Adding that to your cart.",
}

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
BENCH = Path("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json")
HOLD = 0.64          # frozen engine END_HOLD (iron rule 1: value reused, not changed)
SR = 16000

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


_VAD_MODEL = None


def vad_segments(audio):
    global _VAD_MODEL
    from silero_vad import load_silero_vad, get_speech_timestamps
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad()
    ts = get_speech_timestamps(audio, _VAD_MODEL, sampling_rate=SR,
                               min_silence_duration_ms=400, speech_pad_ms=30)
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


class DecisionCache:
    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        self.hits = self.misses = 0

    def key(self, msgs):
        return hashlib.sha256(json.dumps(msgs, sort_keys=True).encode()).hexdigest()

    def call(self, msgs):
        k = self.key(msgs)
        if k in self.data:
            self.hits += 1
            e = self.data[k]
            return e["raw"], e["infer"]
        t0 = time.time()
        raw = llm_text(msgs)
        infer = round(time.time() - t0, 3)
        self.misses += 1
        self.data[k] = {"raw": raw, "infer": infer}
        return raw, infer

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data))


def _salvage(raw):
    """Last-resort tolerant extraction when strict JSON parsing fails: pull op
    objects and the say string out of malformed output (e.g. a missing `]` before
    `"say"`). Only accepts ops whose own JSON parses; never invents fields."""
    ops = []
    for m in re.finditer(r'\{\s*"type"\s*:\s*"(launch|patch|cancel|commit|noop)".*?\}(?=\s*[,\]\}])',
                         raw, re.S):
        frag = m.group(0)
        # extend to balance braces (args objects nest one level)
        extra, depth = 0, frag.count("{") - frag.count("}")
        end = m.end()
        while depth > 0 and end + extra < len(raw):
            ch = raw[end + extra]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            extra += 1
        frag = raw[m.start(): end + extra]
        try:
            ops.append(json.loads(frag))
        except Exception:
            continue
    say = ""
    ms = re.search(r'"say"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if ms:
        try:
            say = json.loads('"' + ms.group(1) + '"')
        except Exception:
            say = ms.group(1)
    if ops or say:
        return {"dialogue": "stay", "ops": ops, "say": say, "_salvaged": True}
    return None


def decide(cache, tx, audio_prefix):
    msgs = build_decider_messages(tx, "LISTEN", audio=audio_prefix)
    raw, infer = cache.call(msgs)
    dec = parse_decision(raw)
    if "_parse_error" in dec:                      # one repair retry (W2 plan D4)
        msgs2 = msgs + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": [{"type": "text", "text":
                "Your previous output was NOT valid JSON (check bracket balance — "
                "the \"ops\" array must be closed with ] before \"say\"). "
                "Reply with ONLY the corrected JSON object, nothing else."}]}]
        raw2, infer2 = cache.call(msgs2)
        dec = parse_decision(raw2)
        infer += infer2
        dec["_repaired"] = True
        if "_parse_error" in dec:                  # salvage before conservative fallback
            sal = _salvage(raw2) or _salvage(raw)
            if sal is not None:
                sal["_repaired"] = True
                return sal, infer
    return dec, infer


def silent_deadline(t_open, delta, segs):
    """Audio-clock time when `delta` seconds of SILENCE have elapsed after t_open
    (user speech pauses the countdown)."""
    t, remaining = t_open, delta
    while True:
        nxt = next(((s, e) for s, e in segs if e > t and s < t + remaining), None)
        if nxt is None:
            return t + remaining
        s, e = nxt
        if s > t:
            gained = s - t
            remaining -= gained
        t = e  # countdown frozen while user speaks; resume after segment
        if remaining <= 0:
            return s


def run_example(folder, provider, delta, cache, mode="tact", force=False):
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

    random.seed(42)  # seeded sandbox latency (W2 D1-2 discipline)
    reg = ToolRegistry(latency_profile=meta.get("latency_profile", "normal"),
                       room=f"w2r-{example_id}",
                       telemetry_path=f"/tmp/w2r_tools_{provider}.log")
    tx = Transaction()

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
    windows = {}                              # op_id -> commit deadline (audio clock)
    exec_wall = {}                            # op_id -> tool wall seconds
    say_events = []                           # (t_audio, text)

    def do_commit(op_id, t_commit):
        if op_id not in tx.pending:
            return
        t0 = time.time()
        tx.commit(op_id, reg.executor, t=t_commit)
        w = round(time.time() - t0, 3)
        exec_wall[op_id] = w
        trace["commits"].append({"op_id": op_id, "t_commit": round(t_commit, 3),
                                 "tool_wall_s": w})
        windows.pop(op_id, None)

    for seg_idx, t_eou in eous:
        # windows expiring strictly before this EoU close first (time order)
        for op_id, dl in sorted(windows.items(), key=lambda kv: kv[1]):
            if dl <= t_eou:
                do_commit(op_id, dl)
        prefix = audio[: int(segs[seg_idx][1] * SR)]
        dec, infer = decide(cache, tx, prefix)
        t_dec = t_eou + infer                # decision lands after real infer time
        say = dec.get("say", "")
        launched_fns = [op.get("fn", "") for op in dec.get("ops", [])
                        if op.get("type") == "launch"]
        if mode != "blocking" and not say and launched_fns:
            say = ACK_TEMPLATES.get(launched_fns[0], "I'm on it.")   # ack-v0 fallback
            dec["_ack_template"] = True
        if say:
            say_events.append((t_dec, say))
        applied = []
        for op in dec.get("ops", []):
            typ = op.get("type", "noop")
            if typ == "launch":
                fn = op.get("fn", "")
                args = op.get("args", {}) or {}
                if "args" in args and isinstance(args["args"], dict):
                    args = {**args["args"], **{k: v for k, v in args.items() if k != "args"}}
                args = _coerce_args(args)
                # PendingSet idempotence: an identical intent (fn+args) already
                # pending or committed is NOT a new intent — drop the duplicate.
                dup = (any(p.fn == fn and p.args == args for p in tx.pending.values())
                       or any(c.fn == fn and c.args == args for c in tx.committed))
                if dup:
                    applied.append({"type": "launch_dedup", "fn": fn})
                    continue
                rev = REVERSIBILITY.get(fn, Reversibility.IRR)
                p = tx.launch(fn, args, rev, t=t_dec)
                if mode == "blocking" or delta <= 0:
                    do_commit(p.op_id, t_dec)
                else:
                    windows[p.op_id] = silent_deadline(t_dec, delta, segs)
                applied.append({"type": "launch", "fn": fn, "op_id": p.op_id})
            elif typ == "patch":
                oid = _resolve(tx, op)
                if oid is not None and oid in tx.pending:
                    diff = op.get("diff", {}) or {}
                    if set(diff.keys()) == {"args"} and isinstance(diff["args"], dict):
                        diff = diff["args"]      # unwrap model's nested-args habit
                    diff = _coerce_args(diff)
                    tx.patch(oid, diff, t=t_dec)
                    windows[oid] = silent_deadline(t_dec, delta, segs)  # window restarts
                    applied.append({"type": "patch", "op_id": oid, "diff": diff})
            elif typ == "cancel":
                oid = _resolve(tx, op)
                if oid is not None and oid in tx.pending:
                    tx.cancel(oid, t=t_dec, executor=reg.executor)
                    windows.pop(oid, None)
                    applied.append({"type": "cancel", "op_id": oid})
            # model-emitted commit / noop: ignored (harness owns commit timing)
        trace["decisions"].append({"seg_idx": seg_idx, "t_eou": round(t_eou, 3),
                                   "infer_s": infer, "say": dec.get("say", ""),
                                   "ops": applied,
                                   "repaired": bool(dec.get("_repaired"))})

    # end of audio: remaining windows expire at their deadlines
    for op_id, dl in sorted(windows.items(), key=lambda kv: kv[1]):
        do_commit(op_id, dl)

    # ---- latency accounting (text-ready anchors, TTS excluded on both arms) ----
    # first response after the FINAL user segment end:
    #   TACT     : the final decision's say (ack/announce) — available at t_dec,
    #              BEFORE tool execution.  Fallback (empty say): result-ready time.
    #   blocking : must wait for the tool — infer + tool wall (conservative:
    #              answer-generation time counted as 0, which FAVOURS blocking).
    result_ready = (max(c["t_commit"] + c["tool_wall_s"] for c in trace["commits"])
                    if trace["commits"] else None)
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

    result = {
        "pid": speaker_id, "example_id": example_id,
        "category": meta.get("domain", "unknown"), "title": meta.get("title", ""),
        "provider": provider, "mode": mode, "delta": delta,
        "actual_tool_calls": tx.to_actual_tool_calls(),
        "transcript": say_events[-1][1] if say_events else "",
        "latency": {"first_response_s": first_response_s,
                    "ack_emitted": ack_emitted,
                    "task_completion_s": task_completion_s,
                    "n_eou": len(eous)},
        "tx_log": tx.log, "trace": trace, "status": "completed",
    }
    out_p.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def _resolve(tx, op):
    if op.get("op_id") is not None:
        try:
            return int(op["op_id"])
        except Exception:
            return None
    if "fn" in op:
        p = tx.find_pending_by_fn(op["fn"])
        return p.op_id if p else None
    p = tx.latest_pending()
    return p.op_id if p else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--mode", choices=["tact", "blocking"], default="tact")
    ap.add_argument("--only-rollback", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cache", default="/root/autodl-tmp/fd-badcat/exp/w2_rerun/decision_cache.json")
    args = ap.parse_args()

    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench
    rollback_ids = {x["id"] for x in items if x.get("state_rollback_test")}

    folders = sorted(f for f in DATA.iterdir() if f.is_dir() and _FOLDER_RE.match(f.name))
    if args.only_rollback:
        folders = [f for f in folders
                   if _FOLDER_RE.match(f.name).group(1) in rollback_ids]
    if args.limit:
        folders = folders[: args.limit]

    cache = DecisionCache(args.cache)
    print(f"streaming replay | provider={args.provider} mode={args.mode} "
          f"delta={args.delta} | {len(folders)} examples")
    for i, f in enumerate(folders, 1):
        try:
            r = run_example(f, args.provider, args.delta, cache,
                            mode=args.mode, force=args.force)
            if r:
                lat = r["latency"]
                print(f"[{i}/{len(folders)}] {f.name}: "
                      f"{len(r['actual_tool_calls'])} calls, "
                      f"resp={lat.get('first_response_s')} done={lat['task_completion_s']} "
                      f"eou={lat['n_eou']}")
        except Exception as e:
            print(f"[{i}/{len(folders)}] {f.name}: ERROR {type(e).__name__} {e}")
        if i % 10 == 0:
            cache.save()
    cache.save()
    print(f"cache: {cache.hits} hits / {cache.misses} misses")


if __name__ == "__main__":
    main()
