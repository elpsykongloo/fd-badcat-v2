#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rb_commit_judge.py — the RB v2.4 commitment-judge OVERLAY (analysis layer;
rb_design §17.0 item 3, rb_test_protocol §11).

Main arms speak FREE TEXT (FC off), so the marker path of the commitment-
repair track reads zero by construction. This script re-scores archived rows'
say_events with the FROZEN COMMIT_JUDGE_PROMPT (rb/scorer.py, freeze v5;
design rb_design §17.0 item 3)
through DeepSeek `deepseek-v4-flash`, and writes a judged overlay NEXT TO the
archive — it never modifies rows, reports, or caches. Deterministic replay
via an on-disk prompt-keyed cache.

  $PY scripts/rb_commit_judge.py --build exp/rb/build_v24 \
      --provider rbt24_b_tact_d150 [--layers L14] [--judge-cache PATH]
  $PY scripts/rb_commit_judge.py --selftest        # stub judge, no network

Judge discipline (the fdb_pass_judge_strict lessons): proxy env vars cleared,
temperature 0, max_tokens 512, up to 5 retries per call, HARD FAIL after —
silent fallback is how the W1/W2 judge numbers went bad. The overlay's own
sha256 is pinned in scorer freeze v5: changing this file after the freeze is
a version bump, same as the scorer.  Rows are judged concurrently with
thread-local SDK clients; the prompt-keyed cache is single-flight and atomically
persisted, so identical prompts are issued once and completion order cannot
change the overlay."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rb.scorer import (COMMIT_JUDGE_PROMPT, make_llm_judge,   # noqa: E402
                       commitment_repair, episode_claim_forms)

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
MAX_TOKENS = 512
RETRIES = 5
DEFAULT_WORKERS = int(os.getenv("DEEPSEEK_WORKERS", "100"))
USER_ID_PREFIX = "fd-badcat-rb-commitjudge"


def clear_proxy_env():
    """The direct DeepSeek endpoint must not inherit the local SOCKS proxy."""
    for v in ("all_proxy", "ALL_PROXY", "http_proxy", "https_proxy",
              "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)


class JudgeCache:
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        self._inflight = {}
        self._errors = {}

    def key(self, prompt):
        return hashlib.sha256(prompt.encode()).hexdigest()

    def call(self, raw_call, prompt):
        k = self.key(prompt)
        with self._lock:
            if k in self.d:
                self.hits += 1
                return self.d[k]
            ready = self._inflight.get(k)
            if ready is None:
                ready = threading.Event()
                self._inflight[k] = ready
                self._errors.pop(k, None)
                self.misses += 1
                owner = True
            else:
                # A duplicate request joins the first one instead of racing a
                # second API call or a cache-file write.
                self.hits += 1
                owner = False
        if not owner:
            ready.wait()
            with self._lock:
                if k in self.d:
                    return self.d[k]
                err = self._errors.get(k, RuntimeError("cache owner failed"))
            raise RuntimeError(f"judge cache producer failed for {k[:12]}") from err

        try:
            out = raw_call(prompt)
            with self._lock:
                self.d[k] = out
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.d, ensure_ascii=False, sort_keys=True))
                tmp.replace(self.path)
        except BaseException as exc:             # wake duplicate waiters too
            with self._lock:
                self._errors[k] = exc
                self._inflight.pop(k, None)
                ready.set()
            raise
        with self._lock:
            self._inflight.pop(k, None)
            ready.set()
        return out


def deepseek_call(prompt):
    from openai import OpenAI
    # A client belongs to this call/worker only.  Do not share a failing
    # connection across concurrent rows.
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL,
                    timeout=120, max_retries=0)
    last = None
    user_id = f"{USER_ID_PREFIX}-{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
    for attempt in range(RETRIES):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"user_id": user_id})
            out = r.choices[0].message.content or ""
            if out.strip():
                return out
            last = "empty content"
        except Exception as e:                      # noqa: BLE001
            last = repr(e)
        if attempt + 1 < RETRIES:
            time.sleep(min(4.0, 0.25 * (2 ** attempt)))
    raise RuntimeError(f"commit judge hard-fail after {RETRIES} tries: {last}")


def judge_provider(build, provider, layers, cache, workers=1):
    resdir = Path(build) / "results" / provider
    epdir = Path(build) / "episodes"
    rows = []
    for f in sorted(resdir.glob("*.json")):
        r = json.loads(f.read_text())
        if layers and r.get("layer") not in layers:
            continue
        rows.append(r)
    judge = make_llm_judge(lambda p: cache.call(deepseek_call, p))

    def judge_row(idx, r):
        ep = json.loads((epdir / f"{r['id']}.json").read_text())
        # same spoken+canonical claim forms as the scorer (one source)
        gold_vals, superseded = episode_claim_forms(ep)
        cr = commitment_repair([tuple(s) for s in r.get("say_events", [])],
                               ep["lang"], gold_vals, superseded, judge)
        return idx, {"id": r["id"], "layer": r["layer"],
                     "arm": r["arm"], "exact": r["exact"],
                     "marker": r["commit_repair"], "judged": cr}

    out_rows = [None] * len(rows)
    worker_count = max(1, min(int(workers), len(rows))) if rows else 1
    if worker_count == 1:
        for idx, r in enumerate(rows):
            _, out_rows[idx] = judge_row(idx, r)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(judge_row, idx, r): idx
                       for idx, r in enumerate(rows)}
            try:
                for future in as_completed(futures):
                    idx, row = future.result()
                    out_rows[idx] = row
            except Exception:
                for future in futures:
                    future.cancel()
                raise
    n = len(out_rows)
    agg = {"n": n,
           "episodes_with_commit": sum(1 for x in out_rows
                                       if x["judged"]["n_commits"]),
           "commit_emission_rate": round(sum(
               1 for x in out_rows if x["judged"]["n_commits"]) / n, 4)
           if n else None,
           "wrong_commits": sum(x["judged"]["wrong_commits"] for x in out_rows),
           "repaired": sum(x["judged"]["repaired"] for x in out_rows),
           "unrepaired": sum(x["judged"]["unrepaired"] for x in out_rows),
           "marker_wrong_commits": sum(x["marker"]["wrong_commits"]
                                       for x in out_rows)}
    return {"schema": "rb-commit-judge-overlay-v1", "provider": provider,
            "model": MODEL, "layers": sorted(layers) if layers else "all",
            "judge_prompt_sha256": hashlib.sha256(
                COMMIT_JUDGE_PROMPT.encode()).hexdigest(),
            "aggregate": agg, "rows": out_rows}


def selftest():
    ck = {}
    calls = {"n": 0}

    def stub(prompt):
        calls["n"] += 1
        text = prompt.rsplit("): ", 1)[1] if "): " in prompt else prompt
        if "订好了" in text or "booked" in text.lower():
            val = "五月三号" if "五月三号" in text else "May 8"
            return json.dumps({"commit": True, "repair": False, "claim": val},
                              ensure_ascii=False)
        if "改成" in text or "correction" in text.lower():
            return json.dumps({"commit": False, "repair": True,
                               "claim": "五月八号"}, ensure_ascii=False)
        return '{"commit": false, "repair": false, "claim": ""}'

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cache = JudgeCache(Path(td) / "jc.json")
        judge = make_llm_judge(lambda p: cache.call(stub, p))
        says = [(1.0, "好的，马上办。"),
                (5.0, "订好了，五月三号入住。"),          # commit on OLD value
                (9.0, "改成八号了。")]                    # repair (judge path)
        cr = commitment_repair(says, "zh", ["五月八号"], ["五月三号"], judge)
        ck["judge_wrong_commit_detected"] = cr["wrong_commits"] == 1
        # cache: identical rerun = all hits, byte-identical result
        m0 = cache.misses
        cr2 = commitment_repair(says, "zh", ["五月八号"], ["五月三号"], judge)
        ck["judge_cache_replay"] = (cache.misses == m0
                                    and json.dumps(cr) == json.dumps(cr2))
        # marker path short-circuits the judge (no extra calls)
        n0 = calls["n"]
        cr3 = commitment_repair([(2.0, "已确认：五月三号")], "zh",
                                ["五月八号"], ["五月三号"], judge)
        ck["marker_shortcircuits_judge"] = (calls["n"] == n0
                                            and cr3["wrong_commits"] == 1)
        # cache file survives a reload
        cache2 = JudgeCache(cache.path)
        ck["cache_persisted"] = cache2.d == cache.d
        # Concurrent identical prompts must produce one API-style call, with
        # all duplicate consumers receiving the same cached bytes.
        concurrent_calls = {"n": 0}
        concurrent_lock = threading.Lock()

        def slow_stub(_prompt):
            with concurrent_lock:
                concurrent_calls["n"] += 1
            time.sleep(0.02)
            return '{"commit": false, "repair": false, "claim": ""}'

        concurrent_cache = JudgeCache(Path(td) / "concurrent.json")
        with ThreadPoolExecutor(max_workers=8) as pool:
            replies = list(pool.map(
                lambda _n: concurrent_cache.call(slow_stub, "same-prompt"),
                range(8)))
        ck["cache_singleflight_parallel"] = (
            concurrent_calls["n"] == 1 and concurrent_cache.misses == 1
            and concurrent_cache.hits == 7 and len(set(replies)) == 1)
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", default="exp/rb/build_v24")
    ap.add_argument("--provider", action="append", default=[])
    ap.add_argument("--layers", default="L14",
                    help="comma list; empty string = all layers")
    ap.add_argument("--judge-cache", default=None,
                    help="default: <build>/commit_judge_cache.json")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="concurrent isolated DeepSeek requests "
                         "(default: DEEPSEEK_WORKERS or 100)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.provider:
        ap.error("need at least one --provider")
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    clear_proxy_env()
    layers = set(x for x in args.layers.split(",") if x) or None
    cache = JudgeCache(args.judge_cache or
                       Path(args.build) / "commit_judge_cache.json")
    for prov in args.provider:
        out = judge_provider(args.build, prov, layers, cache, args.workers)
        dst = Path(args.build) / f"rb_commitjudge_{prov}.json"
        dst.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"{prov}: n={out['aggregate']['n']} "
              f"emission={out['aggregate']['commit_emission_rate']} "
              f"wrong={out['aggregate']['wrong_commits']} "
              f"unrepaired={out['aggregate']['unrepaired']} -> {dst}")
    print(f"judge cache: {cache.hits} hits / {cache.misses} misses; "
          f"workers={args.workers}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
