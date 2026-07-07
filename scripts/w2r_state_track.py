#!/usr/bin/env python3
"""
w2r_state_track.py — W2 D3: terminal-state (state-track) scorer + calibration matrix.

State-track semantics (blueprint §3.2 / oracle W2 plan D3):
  * The sandbox state is what matters, not the call trajectory.
  * expected final state = fold(expected_tool_calls) over the initial state.
  * actual   final state = fold(actual_tool_calls)   over the initial state.
  * READ tools (search_*, get_*, track_order, calculate_commute) do not mutate
    state; extra/duplicate READ calls change nothing terminally.
  * Write tools fold by semantic key:
        update_search_filter(filter_name=X, value=V)   -> filters[X] = V   (last wins)
        add_to_cart(product_id=P, quantity=Q)          -> cart[P] += Q
        book_flight(passenger_name=N, ...)             -> bookings[N] = args
        modify_autopay(bill_type=B, ...)               -> autopay[B] = args
        update_identity_doc(doc_type=T, doc_number=D)  -> docs[T] = D
  * READ terminal contract: the LAST call per (fn, semantic-key) must argument-match
    the expected one (a wrong search that was later corrected is forgiven; a wrong
    final search is not).

Calibration (D3 GPU item): run over the blocking-100 results and cross-tab
state-track verdict vs official exact verdict.

W3 D5 (裁断 C): DUAL REPORT — every invocation now also scores a NORMALIZED
variant (src/normalize_entity.py, closed public rule set norm-v1) alongside the
verbatim verdict. The verbatim path is byte-identical to W2; normalized fields
are additive. The official exact score is never touched.

Usage:
  python scripts/w2r_state_track.py --provider w2r_blocking [--only-rollback]
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import normalize_entity as ne  # noqa: E402

DATA = Path("/root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released")
BENCH = Path("/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json")

READ_TOOLS = {"search_flights", "get_card_benefits", "get_exchange_rate",
              "search_apartments", "calculate_commute", "track_order",
              "search_products"}

# semantic fold key per write tool (None -> whole-args replacement under fn)
WRITE_KEY = {
    "update_search_filter": "filter_name",
    "add_to_cart": "product_id",
    "book_flight": "passenger_name",
    "modify_autopay": "bill_type",
    "update_identity_doc": "doc_type",
}

_FOLDER_RE = re.compile(r"^(.+)_([0-9a-f]{24})$")


def _norm(v):
    """EXACTLY official exact_match_args.normalize (lower/strip/underscore->space).
    Deliberately no extra tolerance: the state track must differ from the exact
    track ONLY in trajectory semantics (order/duplicates/compensation), so the
    calibration matrix isolates structural forgiveness, not string forgiveness."""
    if isinstance(v, str):
        return v.lower().strip().replace("_", " ")
    return v


def _is_dynref(v):
    return isinstance(v, str) and v.strip().startswith("$")


def _norm_args(args, vnorm=_norm):
    return {k: vnorm(v) for k, v in sorted((args or {}).items())}


def _default_eq(a, b):
    return _norm(a) == _norm(b)


def _args_match(expected, actual, eq=_default_eq):
    """Official exact_match_args semantics: every expected key must be present and
    match (normalized); EXTRA actual keys are allowed; $-references skipped."""
    if actual is None:
        return False
    for k, ev in (expected or {}).items():
        if _is_dynref(ev):
            continue
        if k not in actual or not eq(ev, actual.get(k)):
            return False
    return True


def fold(calls, vnorm=_norm):
    """Reduce a call list to (write_state, read_finals)."""
    state, read_finals = {}, {}
    for c in calls:
        fn = c.get("function", "")
        args = c.get("args", {}) or {}
        if fn in WRITE_KEY:
            key = vnorm(args.get(WRITE_KEY[fn], ""))
            state[(fn, key)] = _norm_args(args, vnorm)
        elif fn in READ_TOOLS:
            read_finals[fn] = _norm_args(args, vnorm)   # last call per read fn wins
        else:  # unknown tool: treat as write keyed by fn (conservative)
            state[(fn, "")] = _norm_args(args, vnorm)
    return state, read_finals


def state_track_pass(expected_calls, actual_calls, vnorm=_norm, eq=_default_eq):
    es, er = fold(expected_calls, vnorm)
    as_, ar = fold(actual_calls, vnorm)
    diffs = []
    # every expected write entry must be terminally satisfied (subset match);
    # extra write entries in actual are terminal-state violations too
    for k in es:
        if not _args_match(es[k], as_.get(k), eq):
            # $-reference key (e.g. add_to_cart product_id="$RESULT_1...."): the key
            # itself is dynamic — match ANY actual entry of the same fn that arg-matches
            fn = k[0]
            if _is_dynref(k[1]):
                if any(k2[0] == fn and _args_match(es[k], as_[k2], eq) for k2 in as_):
                    continue
            diffs.append({"kind": "write", "key": list(k),
                          "expected": es.get(k), "actual": as_.get(k)})
    matched_extra = set()
    for k in as_:
        if k not in es:
            fn = k[0]
            dyn = [k2 for k2 in es if k2[0] == fn and _is_dynref(k2[1])
                   and _args_match(es[k2], as_[k], eq)]
            if dyn or _is_dynref(k[1]):
                matched_extra.add(k)
                continue
            diffs.append({"kind": "write_extra", "key": list(k), "actual": as_[k]})
    for fn in er:
        if fn not in ar:
            diffs.append({"kind": "read_missing", "fn": fn, "expected": er[fn]})
        elif not _args_match(er[fn], ar[fn], eq):
            diffs.append({"kind": "read_final", "fn": fn,
                          "expected": er[fn], "actual": ar[fn]})
    return len(diffs) == 0, diffs


def state_track_pass_normalized(expected_calls, actual_calls):
    """The norm-v1 variant (dual report). Same structural semantics, values
    compared under the closed public rule set — symmetric on both sides."""
    return state_track_pass(expected_calls, actual_calls,
                            vnorm=ne.normalize_value, eq=ne.values_equal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--only-rollback", action="store_true")
    ap.add_argument("--exact-report", help="official pass_rate report JSON for calibration")
    args = ap.parse_args()

    bench = json.load(open(BENCH))
    items = bench["scenarios"] if isinstance(bench, dict) else bench
    by_id = {x["id"]: x for x in items}
    rollback_ids = {x["id"] for x in items if x.get("state_rollback_test")}

    exact = {}
    if args.exact_report and Path(args.exact_report).exists():
        rep = json.load(open(args.exact_report))
        for r in rep.get("results", rep.get("scenario_results", [])):
            sid = r.get("scenario_id") or r.get("id")
            key = (sid, r.get("pid") or r.get("speaker_id") or "")
            exact[key] = bool(r.get("passed"))

    rows, n_pass, n_pass_norm = [], 0, 0
    sys.path.insert(0, "/root/autodl-tmp/FDBench_v3/v3")
    from evaluate_pass_rate import evaluate_scenario_pass  # official verdict, per folder
    mat = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
    mat_norm = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
    folders = sorted(DATA.iterdir())
    for folder in folders:
        m = _FOLDER_RE.match(folder.name) if folder.is_dir() else None
        if not m:
            continue
        sid = m.group(1)
        if args.only_rollback and sid not in rollback_ids:
            continue
        rp = folder / f"result_{args.provider}.json"
        if not rp.exists() or sid not in by_id:
            continue
        res = json.loads(rp.read_text())
        actual = res.get("actual_tool_calls", [])
        ok, diffs = state_track_pass(by_id[sid]["expected_tool_calls"], actual)
        ok_n, diffs_n = state_track_pass_normalized(
            by_id[sid]["expected_tool_calls"], actual)
        official = evaluate_scenario_pass(by_id[sid], actual,
                                          res.get("transcript", ""), res)["passed"]
        mat[("T" if ok else "F") + ("T" if official else "F")] += 1
        mat_norm[("T" if ok_n else "F") + ("T" if official else "F")] += 1
        n_pass += ok
        n_pass_norm += ok_n
        rows.append({"id": sid, "folder": folder.name, "state_pass": ok,
                     "state_pass_norm": ok_n, "exact_pass": official,
                     "diffs": diffs, "diffs_norm": diffs_n})
    n = len(rows)
    print(f"state-track | provider={args.provider} | verbatim {n_pass}/{n} "
          f"({100.0*n_pass/max(n,1):.1f}%) | normalized({ne.RULES_VERSION}) "
          f"{n_pass_norm}/{n} ({100.0*n_pass_norm/max(n,1):.1f}%)")
    exact_n = sum(1 for r in rows if r["exact_pass"])
    print(f"official exact (recomputed per folder): {exact_n}/{n}")
    print(f"calibration verbatim (state, exact): {mat}   "
          f"[TF = state-pass but exact-fail = trajectory-dirty / order-forgiven]")
    print(f"calibration normalized (state, exact): {mat_norm}")
    for r in rows:
        if r["state_pass"] != r["state_pass_norm"]:
            print(f"  NORM-RESCUED {r['folder']}: "
                  f"{json.dumps(r['diffs'], ensure_ascii=False)[:160]}")
        if r["state_pass"] != r["exact_pass"]:
            tag = "S+E-" if r["state_pass"] else "S-E+"
            print(f"  {tag} {r['folder']}: {json.dumps(r['diffs'], ensure_ascii=False)[:200]}")

    out = Path(f"/root/autodl-tmp/fd-badcat/exp/w2_rerun/state_track_{args.provider}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"provider": args.provider, "n": n, "n_pass": n_pass,
                               "n_pass_norm": n_pass_norm,
                               "normalizer": ne.RULES_VERSION,
                               "exact_pass": exact_n, "calibration": mat,
                               "calibration_norm": mat_norm,
                               "rows": rows}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
