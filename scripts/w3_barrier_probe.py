#!/usr/bin/env python3
"""
w3_barrier_probe.py — D1 acceptance probe for the commit-barrier ruling (06 §一).

Hard expectations (nominal infer 1.0s, W2 decision cache, official exact scorer):

  barrier ON  : eco19@d1.0 P | hou25@d1.0 P | fin12b@d1.5 P | travel_10@d1.5 F
                rollback-17 @ d1.5 => 12/17
  barrier OFF : hou25 and fin12b F at the same deltas; travel_10 F;
                **eco19 stays P** (see correction below)
                rollback-17 @ d1.5 => 10/17

PRE-REGISTERED CORRECTION to the 06 §一 prediction (9/17, all three flips dead):
re-derivation before running showed eco19's rescue patch is a VALUE-NEUTRAL diff
({query:"tablet"} onto an op already carrying tablet — the first revision was
hold-merged into the launch). Its delta-flip is a SNAPSHOT effect: the window
keeps the op PENDING at EoU2 (expiry 19.91 > 19.55 for any delta > 0.64), the
pending snapshot elicits the add_to_cart launch, and commit-vs-patch ORDER never
matters. So the barrier's causal weight is exactly 2 clips (hou25: patch 3500;
fin12b: patch 150), and barrier-off @1.5 = 10/17. Recorded here BEFORE the run
(prediction-first discipline); if the run contradicts THIS table, that is a
finding, not a tweak target.

plus the BIT-PARITY contract: barrier-on results reproduce the frozen W2 grid v1
files on every DETERMINISTIC field — actual_tool_calls exactly (functions, args,
nominal launch/commit stamps), latency structural fields exactly, first_response_s
exactly on the ack path. Wall-derived latency values (task_completion_s) are
compared informationally only: the v1 archives were produced with --workers>1
where the global seeded RNG interleaves across threads, so their tool_wall_s are
scheduling artifacts (AGENTS: non-authoritative), bounded here by a 1.0s sanity
gate against structural regressions (e.g. wrong latency profile).

Zero-GPU when every decision is cache-hit; barrier-off trajectories mostly reuse
on-arm snapshots (expiries land after the EoU snapshots), so misses should be
rare — if the probe reports cache misses with the vLLM stack down, start it and
re-run (results are resumable).

Usage:
  python scripts/w3_barrier_probe.py [--force] [--engine core|full]
      [--cache exp/w2_rerun/decision_cache.json] [--out exp/w3/barrier_probe.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "/root/autodl-tmp/FDBench_v3/v3")

from w2r_stream_replay import (DATA, BENCH, DecisionCache, run_example)  # noqa: E402
from evaluate_pass_rate import evaluate_scenario_pass                    # noqa: E402

WALL_SANITY_S = 1.0   # sanity gate for wall-derived latency drift (see latency_drift)

# The 17 rollback clips (released dataset folder names; two scripts have 2 takes).
ROLLBACK_FOLDERS = [
    "ecommerce_09_695bd157114f0d2317f88617",
    "ecommerce_11_62a885d5b6af18b3d4579e1b",
    "ecommerce_19_66f59c766e7e22e1f90d08f6",
    "finance_12_65e8cf8f4c7424fa062e54a3",
    "finance_12_69a9cf80f4d7668d5c815038",
    "finance_15_66c4f3cb14cbfc4db836bd4e",
    "finance_19_66f59c766e7e22e1f90d08f6",
    "finance_23_66f59c766e7e22e1f90d08f6",
    "housing_09_695bd157114f0d2317f88617",
    "housing_11_62a885d5b6af18b3d4579e1b",
    "housing_17_65e8cf8f4c7424fa062e54a3",
    "housing_17_69a9cf80f4d7668d5c815038",
    "housing_19_62a885d5b6af18b3d4579e1b",
    "housing_21_66c4f3cb14cbfc4db836bd4e",
    "housing_25_66f59c766e7e22e1f90d08f6",
    "travel_10_5f4a4da1575d605c43bef871",
    "travel_19_695bd157114f0d2317f88617",
]

PROBE_FOLDERS = [   # the four mechanism clips from the W3 ledger
    "ecommerce_19_66f59c766e7e22e1f90d08f6",
    "housing_25_66f59c766e7e22e1f90d08f6",
    "finance_12_69a9cf80f4d7668d5c815038",
    "travel_10_5f4a4da1575d605c43bef871",
]

# Expected official-exact verdicts per (folder, delta, barrier) — from the W3
# ledger (docs/w3_ledger.md §2/§4A). d150 covers all 17; d100 covers the probes.
EXPECT_D150_ON = {
    "ecommerce_09_695bd157114f0d2317f88617": True,
    "ecommerce_11_62a885d5b6af18b3d4579e1b": True,
    "ecommerce_19_66f59c766e7e22e1f90d08f6": True,
    "finance_12_65e8cf8f4c7424fa062e54a3": True,
    "finance_12_69a9cf80f4d7668d5c815038": True,
    "finance_15_66c4f3cb14cbfc4db836bd4e": True,
    "finance_19_66f59c766e7e22e1f90d08f6": True,
    "finance_23_66f59c766e7e22e1f90d08f6": False,
    "housing_09_695bd157114f0d2317f88617": True,
    "housing_11_62a885d5b6af18b3d4579e1b": False,
    "housing_17_65e8cf8f4c7424fa062e54a3": True,
    "housing_17_69a9cf80f4d7668d5c815038": False,
    "housing_19_62a885d5b6af18b3d4579e1b": True,
    "housing_21_66c4f3cb14cbfc4db836bd4e": True,
    "housing_25_66f59c766e7e22e1f90d08f6": True,
    "travel_10_5f4a4da1575d605c43bef871": False,
    "travel_19_695bd157114f0d2317f88617": False,
}
# Barrier-dependent flips (rescue patch carries a REAL value diff). eco19 is NOT
# here: its patch is value-neutral, its flip is snapshot-driven (see module doc).
_FLIPS = {"housing_25_66f59c766e7e22e1f90d08f6",
          "finance_12_69a9cf80f4d7668d5c815038"}
EXPECT_D150_OFF = {f: (v and f not in _FLIPS) for f, v in EXPECT_D150_ON.items()}
EXPECT_D100 = {  # folder -> (on, off)
    "ecommerce_19_66f59c766e7e22e1f90d08f6": (True, True),   # snapshot effect, barrier-free
    "housing_25_66f59c766e7e22e1f90d08f6": (True, False),
    "finance_12_69a9cf80f4d7668d5c815038": (False, False),   # threshold 1.12 > 1.0
    "travel_10_5f4a4da1575d605c43bef871": (False, False),    # threshold 3.91
}


def sid_of(folder_name):
    return "_".join(folder_name.split("_")[:2])


def score_result(by_id, res):
    sid = res["example_id"]
    return evaluate_scenario_pass(by_id[sid], res.get("actual_tool_calls", []),
                                  res.get("transcript", ""), res)["passed"]


def run_set(folders, provider, delta, barrier, cache, force, engine):
    out = {}
    for name in folders:
        folder = DATA / name
        if not folder.is_dir():
            print(f"  MISSING FOLDER: {name}")
            continue
        try:
            res = run_example(folder, provider, delta, cache, mode="tact",
                              force=force, infer_nominal=1.0,
                              barrier=barrier, engine=engine)
        except Exception as e:
            print(f"  {name}: ERROR {type(e).__name__} {e}")
            res = None
        if res is not None:
            out[name] = res
    return out


def calls_signature(res):
    """Order-sensitive (function, args) sequence + commit stamps — the parity unit.
    Fully deterministic (nominal infer + nominal commit stamps) => compared exactly."""
    return [(c.get("function"), json.dumps(c.get("args", {}), sort_keys=True),
             c.get("timestamp_start"), c.get("timestamp_end"))
            for c in res.get("actual_tool_calls", [])]


def latency_drift(ref, new):
    """Latency parity, decomposed by determinism.

    GATING (bit-exact): ack_emitted, n_eou, infer_mode, and first_response_s on
    the ack path (= a say-event stamp = t_eou + nominal infer; fully deterministic).

    INFORMATIONAL (never gates, reported): wall-derived values — task_completion_s
    and fallback first_response_s include measured tool_wall_s. The W2 v1 archives
    were produced with --workers>1 where the per-example `random.seed(42)` hits a
    GLOBAL RNG interleaved across threads (AGENTS: latency fields from concurrent
    runs are NOT authoritative), so archived tool_wall_s values are scheduling
    artifacts, not seed-42 truth — no finite tolerance is principled. The
    deterministic completion component (nominal commit stamps) is already compared
    bit-exactly via calls_signature (timestamp_end).

    A loose sanity bound still gates structural-scale wall drift (e.g. a wrong
    latency profile would shift every clip by ~p50).

    Returns (gating_drift | None, wall_info_list | None).
    """
    a, b = ref.get("latency", {}) or {}, new.get("latency", {}) or {}
    for k in ("ack_emitted", "n_eou", "infer_mode"):
        if a.get(k) != b.get(k):
            return f"latency.{k} ({a.get(k)} vs {b.get(k)})", None
    info = []
    for k in ("first_response_s", "task_completion_s"):
        va, vb = a.get(k), b.get(k)
        if (va is None) != (vb is None):
            return f"latency.{k} (None mismatch: {va} vs {vb})", None
        if va is None:
            continue
        d = abs(va - vb)
        if k == "first_response_s" and a.get("ack_emitted"):
            if d > 1e-9:                       # deterministic say-anchor
                return f"latency.{k} ({va} vs {vb})", None
        else:                                   # wall component present
            if d > WALL_SANITY_S:
                return (f"latency.{k} ({va} vs {vb}, exceeds sanity bound "
                        f"{WALL_SANITY_S}s)"), None
            if d > 1e-9:
                info.append(f"{k} {va} vs {vb} (Δ{d:.3f}, wall component)")
    return None, (info or None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-run even if result_<provider>.json exists (use after "
                         "any code change; without it stale results are reused)")
    ap.add_argument("--engine", choices=["core", "full"], default="core")
    ap.add_argument("--cache",
                    default="/root/autodl-tmp/fd-badcat/exp/w2_rerun/decision_cache.json")
    ap.add_argument("--out", default="/root/autodl-tmp/fd-badcat/exp/w3/barrier_probe.json")
    args = ap.parse_args()

    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench
    by_id = {x["id"]: x for x in items}

    cache = DecisionCache(args.cache)
    tag = "" if args.engine == "core" else "f"
    report = {"engine": args.engine, "checks": [], "parity": [], "deferrals": {}}
    ok_all = True

    def check(label, got, want):
        nonlocal ok_all
        good = (got == want)
        ok_all &= good
        report["checks"].append({"label": label, "got": got, "want": want, "ok": good})
        print(f"  [{'OK ' if good else 'FAIL'}] {label}: got {got}, want {want}")
        return good

    # ---- rollback-17 @ delta 1.5, both semantics -------------------------
    combos = [(f"w3p{tag}_on_d150", 1.5, True, ROLLBACK_FOLDERS, EXPECT_D150_ON),
              (f"w3p{tag}_off_d150", 1.5, False, ROLLBACK_FOLDERS, EXPECT_D150_OFF),
              (f"w3p{tag}_on_d100", 1.0, True, PROBE_FOLDERS,
               {f: v[0] for f, v in EXPECT_D100.items()}),
              (f"w3p{tag}_off_d100", 1.0, False, PROBE_FOLDERS,
               {f: v[1] for f, v in EXPECT_D100.items()})]

    results = {}
    for provider, delta, barrier, folders, expect in combos:
        print(f"\n== {provider} (delta={delta}, barrier={'on' if barrier else 'off'}, "
              f"engine={args.engine}) ==")
        rs = run_set(folders, provider, delta, barrier, cache, args.force, args.engine)
        results[provider] = rs
        passes = {}
        for name, res in rs.items():
            p = score_result(by_id, res)
            passes[name] = p
            mark = "P" if p else "F"
            exp_mark = "P" if expect.get(name) else "F"
            flag = "" if p == expect.get(name) else "   <-- MISMATCH"
            print(f"  {mark} (want {exp_mark})  {name}{flag}")
        for name in folders:
            if name in passes:
                check(f"{provider}:{sid_of(name)}:{name[-8:]}", passes[name],
                      expect[name])
        want_total = sum(1 for n2 in folders if n2 in passes and expect[n2])
        check(f"{provider}:TOTAL", sum(passes.values()), want_total)
        # deferral stats (dual-stamp evidence; barrier-on only produces them)
        defs = [d for r in rs.values() for d in r.get("ledger", {}).get("deferrals", [])]
        committed = [d for d in defs if d.get("outcome") == "committed"]
        stat = {"n": len(defs), "committed": len(committed),
                "rescued": sum(1 for d in defs if d.get("outcome") == "rescued_patch"),
                "max_deferred_s": max((d.get("deferred_s") or 0.0 for d in committed),
                                      default=0.0)}
        report["deferrals"][provider] = stat
        print(f"  deferrals: {stat}")

    # ---- bit-parity vs the frozen W2 grid v1 (barrier-on, engine core) ----
    if args.engine == "core":
        print("\n== bit-parity vs W2 grid v1 (barrier-on) ==")
        for provider, ref_provider, folders in [
                (f"w3p{tag}_on_d150", "w2r_tact_d150", ROLLBACK_FOLDERS),
                (f"w3p{tag}_on_d100", "w2r_tact_d100", PROBE_FOLDERS)]:
            for name in folders:
                ref_p = DATA / name / f"result_{ref_provider}.json"
                new = results.get(provider, {}).get(name)
                if new is None or not ref_p.exists():
                    continue
                ref = json.loads(ref_p.read_text())
                drift = []
                if calls_signature(ref) != calls_signature(new):
                    drift.append("actual_tool_calls")
                ld, wall_info = latency_drift(ref, new)
                if ld:
                    drift.append(ld)
                report["parity"].append({"folder": name, "vs": ref_provider,
                                         "drift": drift, "wall_info": wall_info})
                if drift:
                    ok_all = False
                    print(f"  DRIFT {name} vs {ref_provider}: {drift}")
                elif wall_info:
                    print(f"  info  {name}: {'; '.join(wall_info)}")
        clean = sum(1 for p in report["parity"] if not p["drift"])
        n = len(report["parity"])
        walls = sum(1 for p in report["parity"] if p.get("wall_info"))
        print(f"  parity: {clean}/{n} bit-exact on deterministic fields "
              f"({walls} with non-gating wall-component deltas — expected: "
              f"v1 archives carry workers-scrambled tool walls)")

    cache.save()
    print(f"\ncache: {cache.hits} hits / {cache.misses} misses"
          + ("  (misses need the vLLM stack at :10004 — if it was down, "
             "start it and re-run)" if cache.misses else ""))
    verdict = "ALL EXPECTATIONS MET — semantics pinned (E-probe green)" if ok_all \
        else "MISMATCHES FOUND — see FAIL/DRIFT lines above"
    print(f"\nVERDICT: {verdict}")
    report["verdict"] = ok_all

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"report -> {out_p}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
