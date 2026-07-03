#!/usr/bin/env python3
"""
w2r_score_grid.py — score every provider tag on: official exact / state track / latency.
Emits the W2 delta-scan three-curve table + JSON.
"""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/FDBench_v3/v3")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/scripts")
from evaluate_pass_rate import evaluate_scenario_pass          # noqa: E402
from w2r_state_track import state_track_pass                   # noqa: E402

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
BENCH = Path("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json")
_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return round(v[min(len(v) - 1, int(p * len(v)))], 3)


def score(provider, only_rollback=False):
    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench
    by_id = {x["id"]: x for x in items}
    rb = {x["id"] for x in items if x.get("state_rollback_test")}

    rows = []
    for folder in sorted(DATA.iterdir()):
        m = _FOLDER_RE.match(folder.name) if folder.is_dir() else None
        if not m or m.group(1) not in by_id:
            continue
        sid = m.group(1)
        if only_rollback and sid not in rb:
            continue
        rp = folder / f"result_{provider}.json"
        if not rp.exists():
            continue
        res = json.loads(rp.read_text())
        actual = res.get("actual_tool_calls", [])
        exact = evaluate_scenario_pass(by_id[sid], actual,
                                       res.get("transcript", ""), res)["passed"]
        state, _ = state_track_pass(by_id[sid]["expected_tool_calls"], actual)
        lat = res.get("latency", {})
        rows.append({"id": sid, "rollback": sid in rb, "exact": exact, "state": state,
                     "first": lat.get("first_response_s"),
                     "done": lat.get("task_completion_s"),
                     "ack": lat.get("ack_emitted")})
    n = len(rows)
    firsts = [r["first"] for r in rows if r["first"] is not None]
    dones = [r["done"] for r in rows if r["done"] is not None]
    return {
        "provider": provider, "n": n,
        "exact": round(sum(r["exact"] for r in rows) / max(n, 1), 3),
        "state": round(sum(r["state"] for r in rows) / max(n, 1), 3),
        "first_p50": pct(firsts, .50), "first_p90": pct(firsts, .90),
        "done_p50": pct(dones, .50), "done_p90": pct(dones, .90),
        "ack_rate": round(sum(1 for r in rows if r.get("ack")) / max(n, 1), 2),
        "rows": rows,
    }


if __name__ == "__main__":
    providers = sys.argv[1:] or [
        "w2r_blocking",
        "w2r_tact_d000", "w2r_tact_d030", "w2r_tact_d060",
        "w2r_tact_d100", "w2r_tact_d150", "w2r_tact_d250",
    ]
    only_rb = "--full" not in sys.argv
    providers = [p for p in providers if not p.startswith("--")]
    out = []
    print(f"{'provider':18s} {'n':>3s} {'exact':>6s} {'state':>6s} "
          f"{'first p50/p90':>14s} {'done p50/p90':>14s} {'ack':>4s}")
    for p in providers:
        s = score(p, only_rollback=only_rb)
        out.append(s)
        print(f"{p:18s} {s['n']:3d} {s['exact']:6.3f} {s['state']:6.3f} "
              f"{str(s['first_p50']):>6s}/{str(s['first_p90']):<7s} "
              f"{str(s['done_p50']):>6s}/{str(s['done_p90']):<7s} {s['ack_rate']:4.2f}")
    tag = "rollback" if only_rb else "full"
    Path(f"/root/autodl-tmp/fd-badcat/exp/w2_rerun/grid_{tag}.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
