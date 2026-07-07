#!/usr/bin/env python3
"""
w3_state_classify.py — state-track failure classification ledger (W3 D5, 裁断 C).

Input : exp/w2_rerun/state_track_{provider}.json (w2r_state_track.py output)
Output: per-failure classification into the 裁断-C taxonomy, printed + written to
        exp/w3/state_classify_{provider}.{json,md}

Classes (priority order — first match wins):
  benchmark_attr   housing_11 / ecommerce_12: gold argument lives in dialogue
                   context the benchmark never plays to ANY compliant system
                   (D2 audit: official SUT hears only the final turn)
  dynref           expected value/key is a $RESULT reference (chained scenario;
                   the model cannot emit the resolved value in the mock world —
                   DAG territory, not an arg error)
  missing_call     the call never happened (paralysis / avoidance class)
  extra_call       an unexpected write remained in the terminal state
  format           values equal under the closed norm-v1 rule set
                   (=> rescued by the normalized report, no model change needed)
  canonicalization the model echoed the user's surface form instead of the
                   canonical entity (alias table / token-subset DETECTION ONLY —
                   never used for scoring; prompt v3 target #5)
  asr_mishear      normalized edit distance <= 2 or a digit-substitution
                   (audio-native re-listen candidate — 蓝图情况矩阵 #8)
  true_param_error everything else (genuinely wrong value)

The 蓝图 #8 pilot trigger (裁断 C: >=3 asr_mishear cases) is evaluated and
printed at the end.

Usage:
  python scripts/w3_state_classify.py --provider w2r_tact_full
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import normalize_entity as ne  # noqa: E402

BENCHMARK_ATTR_IDS = {"housing_11", "ecommerce_12"}   # D2 audit-closed class

# DETECTION-ONLY alias pairs (canonicalization failures the MODEL should fix;
# deliberately NOT in the scorer's normalizer)
ALIASES = {("vegas", "las vegas"), ("nyc", "new york city"),
           ("la", "los angeles"), ("philly", "philadelphia")}


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _digit_swap(a, b):
    """Same shape, only digits differ (e.g. p88990011 vs p99990011): the classic
    ASR digit confusion."""
    if len(a) != len(b):
        return False
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    return bool(diffs) and all(x.isdigit() and y.isdigit() for x, y in diffs)


def classify_pair(gold, actual):
    """Classify one mismatching (gold, actual) value pair."""
    g, a = str(gold), str(actual)
    if ne.values_equal(gold, actual):
        return "format"
    ng, na = ne.normalize_value(g), ne.normalize_value(a)
    if (na, ng) in ALIASES or (ng, na) in ALIASES:
        return "canonicalization"
    tg, ta = set(ng.split()), set(na.split())
    if tg and ta and (tg < ta or ta < tg):
        return "canonicalization"        # added/dropped words around the entity
    if _digit_swap(ng.replace(" ", ""), na.replace(" ", "")):
        return "asr_mishear"
    if min(len(ng), len(na)) >= 2 and edit_distance(ng, na) <= 2:
        return "asr_mishear"
    return "true_param_error"


def classify_diff(sid, diff, all_diffs):
    if sid in BENCHMARK_ATTR_IDS:
        return "benchmark_attr", []
    kind = diff.get("kind", "")
    exp = diff.get("expected") or {}
    act = diff.get("actual")
    key = diff.get("key", ["", ""])
    if kind == "write" and len(key) > 1 and isinstance(key[1], str) \
            and key[1].strip().startswith("$"):
        return "dynref", []      # dynamic write key: chained scenario (DAG territory)
    if kind == "write_extra":
        return "extra_call", []
    if kind == "read_missing":
        return "missing_call", []
    if kind == "write" and act is None:
        # pair with a write_extra of the same fn: same call, different key form
        mate = next((d for d in all_diffs if d.get("kind") == "write_extra"
                     and d.get("key", [""])[0] == key[0]), None)
        if mate is None:
            return "missing_call", []
        act = mate.get("actual") or {}
    fields = []
    for k, gv in exp.items():
        if isinstance(gv, str) and gv.strip().startswith("$"):
            continue
        av = (act or {}).get(k)
        if av is None:
            fields.append((k, gv, None, "missing_field"))
        elif str(gv) != str(av):
            fields.append((k, gv, av, classify_pair(gv, av)))
    if not fields:
        return "format", []      # everything matched under norm rules
    ranking = ["true_param_error", "asr_mishear", "canonicalization",
               "missing_field", "format"]
    worst = min((f[3] for f in fields), key=lambda c: ranking.index(c)
                if c in ranking else 0)
    return worst, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    args = ap.parse_args()

    src = Path(f"/root/autodl-tmp/fd-badcat/exp/w2_rerun/state_track_{args.provider}.json")
    data = json.loads(src.read_text())
    counts, ledger = {}, []
    for row in data["rows"]:
        if row["state_pass"]:
            continue
        seen = set()
        for diff in row["diffs"]:
            cls, fields = classify_diff(row["id"], diff, row["diffs"])
            # write/write_extra pairs collapse into one ledger line
            sig = (row["id"], diff.get("key", [diff.get("fn", "")])[0], cls)
            if cls == "extra_call" and any(s[0] == row["id"] and s[2] != "extra_call"
                                           and s[1] == diff.get("key", [""])[0]
                                           for s in seen):
                continue
            if sig in seen:
                continue
            seen.add(sig)
            counts[cls] = counts.get(cls, 0) + 1
            ledger.append({"id": row["id"], "kind": diff.get("kind"),
                           "fn": diff.get("key", [diff.get("fn", "?")])[0],
                           "class": cls,
                           "fields": [{"field": f[0], "gold": f[1], "actual": f[2],
                                       "pair_class": f[3]} for f in fields],
                           "rescued_by_norm": bool(row.get("state_pass_norm"))})

    total = sum(counts.values())
    print(f"state-track failure classification | provider={args.provider} | "
          f"{total} failure items over "
          f"{sum(1 for r in data['rows'] if not r['state_pass'])} failing scenarios")
    for cls in sorted(counts, key=counts.get, reverse=True):
        print(f"  {cls:18s} {counts[cls]}")
    asr_n = counts.get("asr_mishear", 0)
    print(f"\n蓝图#8 pilot trigger (asr_mishear >= 3): "
          f"{'TRIGGERED' if asr_n >= 3 else 'not triggered'} ({asr_n} cases)")
    if "n_pass_norm" in data:
        print(f"dual report: verbatim {data['n_pass']}/{data['n']} -> "
              f"normalized {data['n_pass_norm']}/{data['n']} "
              f"(+{data['n_pass_norm'] - data['n_pass']} rescued by "
              f"{data.get('normalizer', 'norm-v1')})")

    outdir = Path("/root/autodl-tmp/fd-badcat/exp/w3")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"state_classify_{args.provider}.json").write_text(
        json.dumps({"provider": args.provider, "counts": counts,
                    "asr_pilot_triggered": asr_n >= 3, "ledger": ledger},
                   indent=1, ensure_ascii=False))
    md = ["| id | fn | class | field | gold | actual |", "|---|---|---|---|---|---|"]
    for e in ledger:
        if e["fields"]:
            for f in e["fields"]:
                md.append(f"| {e['id']} | {e['fn']} | {e['class']} | "
                          f"{f['field']} | {f['gold']} | {f['actual']} |")
        else:
            md.append(f"| {e['id']} | {e['fn']} | {e['class']} | — | — | — |")
    (outdir / f"state_classify_{args.provider}.md").write_text("\n".join(md) + "\n")
    print(f"ledger -> {outdir}/state_classify_{args.provider}.{{json,md}}")


if __name__ == "__main__":
    main()
