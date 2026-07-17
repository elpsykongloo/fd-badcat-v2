#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rb_content_gen.py — build the FROZEN RB content bank with DeepSeek v4-flash
(rb_design §15 item 5: LLM-sampled construction diversity instead of pure
rules — user ruling 2026-07-16).

  $PY scripts/rb_content_gen.py                # real API -> exp/rb/content_bank.json
  $PY scripts/rb_content_gen.py --workers 100  # bounded concurrent requests
  $PY scripts/rb_content_gen.py --selftest     # stub generator, no network

Architecture: generation happens OFFLINE, ONCE; the bank is reviewed, then
committed — its sha256 enters the generator config_hash, so builds stay fully
deterministic given the bank file. The build itself never calls an API.

API per https://api-docs.deepseek.com/zh-cn/ : base_url https://api.deepseek.com,
model deepseek-v4-flash (non-thinking chat mode), key from DEEPSEEK_API_KEY
(configs/eval.env, gitignored). Proxy vars are cleared for the call the same
way the judge discipline requires.

Every generated template is VALIDATED before entering the bank: required
placeholders intact, no other braces, TTS-safe charset, length bounds, and
language sanity. Rejected samples are counted in the provenance block.

Generation tasks are independent.  They run concurrently but are assembled by
their stable job index only after *all* calls succeed: an API failure can never
leave a partially rewritten bank, and completion order cannot change its
layout/provenance.  Every request uses a workload-only DeepSeek ``user_id``
for provider-side cache/scheduling isolation (no user data is sent there)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rb.registry import SCENARIOS                                # noqa: E402
from rb.grammar import (REV_UTT, BYSTANDER, PROGRESS_QUERY,      # noqa: E402
                        CONFIRM_QUERY,
                        DISFLUENCY_FALLBACK, DISFLUENCY_FAMILIES)

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
N_PER = 6                     # paraphrases requested per (category, kind, lang)
OUT = ROOT / "exp/rb/content_bank.json"
RETRIES = 5
DEFAULT_WORKERS = int(os.getenv("DEEPSEEK_WORKERS", "100"))
USER_ID = "fd-badcat-rb-content"

GEN_PROMPT = (
    "You write natural SPOKEN {lang_name} utterances for a phone-call speech "
    "benchmark. Produce {n} distinct paraphrases of the EXAMPLE below. Rules: "
    "keep EVERY placeholder token EXACTLY as written (e.g. {placeholders}); "
    "colloquial, speakable, one sentence, no emoji, no quotes, no digits "
    "unless the example has them, length within 2x of the example. "
    "{position_rule}Reply as "
    "a JSON array of {n} strings, nothing else.\n"
    "Category: {cat}\nEXAMPLE: {example}")

LANG_NAME = {"zh": "Chinese (Mandarin)", "en": "English"}
_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def _placeholders(t):
    """v2.4: MULTISET of placeholders (sorted list, duplicates kept). The
    v2.3 set-compare let DeepSeek turn the double-{new} value_first template
    into contrastive contradictions ("Change that from X to X") — the L4
    artifact's second root cause (rb_test_protocol §10.7)."""
    return sorted(_PLACEHOLDER_RE.findall(t))


def validate(cand, example, lang, kind=None):
    """Template gate: placeholder MULTISET identical, no stray braces,
    speakable; belt-and-suspenders: at most one {new} ever (a revision
    utterance carries its value exactly once)."""
    if not isinstance(cand, str) or not cand.strip():
        return None
    c = cand.strip()
    if _placeholders(c) != _placeholders(example):
        return None
    if c.count("{new}") > 1:
        return None
    # L4 is not merely contrastive: its perturbation must begin with the NEW
    # value, or it stops testing the value-first auditory cue.
    if kind == "value_first" and not c.startswith("{new}"):
        return None
    if c.count("{") != len(_PLACEHOLDER_RE.findall(c)) or \
            c.count("}") != c.count("{"):
        return None
    if len(c) > 2 * max(24, len(example)) or "\n" in c:
        return None
    if lang == "zh" and not re.search(r"[一-鿿]", c):
        return None
    if lang == "en" and re.search(r"[一-鿿]", c):
        return None
    return c


def clear_proxy_env():
    """The direct DeepSeek endpoint must not inherit the local SOCKS proxy."""
    for v in ("all_proxy", "ALL_PROXY", "http_proxy", "https_proxy",
              "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)


def deepseek_call(prompt):
    """One isolated, retrying DeepSeek request; safe to call from a worker."""
    from openai import OpenAI
    # Do not share SDK/http clients across workers.  It keeps a failed or
    # closed connection local to exactly one generation task.
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL,
                    timeout=120, max_retries=0)
    user_id = f"{USER_ID}-{hashlib.sha256(prompt.encode()).hexdigest()[:16]}"
    last = None
    for attempt in range(RETRIES):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=1.3, max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"user_id": user_id})
            out = r.choices[0].message.content or ""
            if out.strip():
                return out
            last = "empty content"
        except Exception as exc:                # noqa: BLE001 - hard fail below
            last = repr(exc)
        if attempt + 1 < RETRIES:
            time.sleep(min(4.0, 0.25 * (2 ** attempt)))
    raise RuntimeError(f"DeepSeek hard-fail after {RETRIES} attempts: {last}")


def parse_array(raw):
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [x for x in arr if isinstance(x, str)]
    except json.JSONDecodeError:
        return []


def gen_category(call, cat, example, lang, n=N_PER, kind=None):
    ph = ", ".join(_placeholders(example)) or "(none)"
    position_rule = (
        "For this VALUE-FIRST category, EVERY string must literally begin "
        "with {new}, and contrast it with {old} later. "
        if kind == "value_first" else "")
    raw = call(GEN_PROMPT.format(lang_name=LANG_NAME[lang], n=n,
                                 placeholders=ph, cat=cat, example=example,
                                 position_rule=position_rule))
    got, ok = parse_array(raw), []
    for c in got:
        v = validate(c, example, lang, kind=kind)
        if v and v not in ok and v != example:
            ok.append(v)
    return ok, len(got)


def generation_jobs():
    """Return the immutable job list in the canonical bank traversal order."""
    jobs = []

    def add(section, key_path, example, lang, cat):
        jobs.append((section, key_path, example, lang, cat))

    for lang in ("zh", "en"):
        for kind, tpl in REV_UTT[lang].items():
            add("revision", (lang, kind), tpl, lang,
                f"user revising a request mid-call ({kind})")
        for kind, tpl in BYSTANDER[lang].items():
            add("bystander", (lang, kind), tpl, lang,
                f"a third person speaking near the phone ({kind})")
        add("progress", (lang,), PROGRESS_QUERY[lang], lang,
            "user asking whether the task is done yet")
        add("confirm", (lang,), CONFIRM_QUERY[lang], lang,
            "user asking the assistant to repeat back what it just set "
            "(no cancel words, no specific value)")
        for fam in DISFLUENCY_FAMILIES:
            add("disfluency", (lang, fam), DISFLUENCY_FALLBACK[lang][fam],
                lang, f"disfluent wrapper around a correction ({fam}; keep {{body}})")
    for sid, s in sorted(SCENARIOS.items()):
        for lang in ("zh", "en"):
            add("intent", (sid, lang), s["utt"][lang], lang,
                f"user asking for the task ({sid})")
    return jobs


def build_bank(call, workers=1, progress=False):
    bank = {"revision": {}, "bystander": {}, "progress": {}, "intent": {},
            "disfluency": {}, "confirm": {}}
    prov = {"model": MODEL, "n_requested": 0, "n_raw": 0, "n_accepted": 0}
    jobs = generation_jobs()
    results = [None] * len(jobs)

    def run_one(idx, job):
        section, key_path, example, lang, cat = job
        try:
            ok, n_raw = gen_category(
                call, cat, example, lang,
                kind=key_path[-1] if section == "revision" else None)
        except Exception as exc:                 # retain the canonical job id
            raise RuntimeError(
                f"content job {idx + 1}/{len(jobs)} ({section}/{'.'.join(key_path)}) "
                f"failed: {exc}") from exc
        return idx, section, key_path, example, ok, n_raw

    worker_count = max(1, min(int(workers), len(jobs)))
    if worker_count == 1:
        for idx, job in enumerate(jobs):
            results[idx] = run_one(idx, job)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(run_one, idx, job): idx
                       for idx, job in enumerate(jobs)}
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    item = future.result()       # hard-fail; no partial output write
                    results[item[0]] = item
                    if progress:
                        print(f"DeepSeek content: {done}/{len(jobs)} complete", flush=True)
            except Exception:
                for future in futures:
                    future.cancel()
                raise

    # Only the canonical input ordering controls output ordering/provenance.
    for _, section, key_path, example, ok, n_raw in results:
        prov["n_requested"] += N_PER
        prov["n_raw"] += n_raw
        prov["n_accepted"] += len(ok)
        node = bank[section]
        for k in key_path[:-1]:
            node = node.setdefault(k, {})
        node[key_path[-1]] = [example] + ok      # original stays variant 0
    bank["_provenance"] = prov
    return bank


def stub_call(prompt):
    """Deterministic offline stub for --selftest: echoes valid variants."""
    ex = prompt.split("EXAMPLE: ", 1)[1]
    return json.dumps([f"{ex}", f"{ex} "], ensure_ascii=False)


def selftest():
    ck = {}
    ok, _ = gen_category(stub_call, "t", "等等，改成{new}。", "zh")
    ck["stub_roundtrip_dedup"] = ok == []          # echoes == example -> rejected
    ck["validate_placeholders"] = (
        validate("好的换成{new}吧。", "等等，改成{new}。", "zh") is not None
        and validate("好的换成{news}吧。", "等等，改成{new}。", "zh") is None
        and validate("Wait make it {new}.", "等等，改成{new}。", "zh") is None
        and validate("等等，改成{new}。{x}", "等等，改成{new}。", "zh") is None)
    # v2.4 multiset gate: the L4-killer contrastive rewrite (one {new} in the
    # example, two in the candidate) is rejected; {old} must survive; and a
    # double-{new} example can never admit a double-{new} candidate again
    # (the >1 belt catches it even at multiset parity).
    ck["validate_multiset_v24"] = (
        validate("从{new}改成{new}。", "等等，改成{new}。", "zh") is None
        and validate("{new}才对，不要{old}了。", "{new}，不是{old}。", "zh")
        is not None
        and validate("要{new}，别弄{new}的。", "{new}，不是{old}。", "zh") is None
        and validate("{new}，改回{new}。", "{new}，改成{new}。", "zh") is None)
    ck["validate_value_first_position"] = (
        validate("{new}，不是{old}。", "{new}，不是{old}。", "zh",
                 kind="value_first") is not None
        and validate("我想要{new}，不是{old}。", "{new}，不是{old}。", "zh",
                     kind="value_first") is None)
    bank = build_bank(stub_call)
    ck["bank_shape"] = all(k in bank for k in
                           ("revision", "bystander", "progress", "intent",
                            "disfluency", "confirm"))
    ck["originals_kept"] = bank["revision"]["zh"]["default"][0] == \
        REV_UTT["zh"]["default"]
    # Completion order may differ, but the bank must be byte-identical to the
    # serial traversal.  The slow first default job forces an out-of-order
    # completion in the parallel path.
    import threading
    completed = []
    completed_lock = threading.Lock()

    def delayed_stub(prompt):
        if "(default)" in prompt:
            time.sleep(0.02)
        with completed_lock:
            completed.append(prompt)
        return "[]"

    serial = build_bank(lambda _p: "[]", workers=1)
    parallel = build_bank(delayed_stub, workers=8)
    ck["parallel_stable_assembly"] = (
        serial == parallel and completed and "(default)" not in completed[0])

    def failing_stub(prompt):
        if "(default)" in prompt:
            raise RuntimeError("intentional isolated failure")
        return "[]"

    try:
        build_bank(failing_stub, workers=8)
        ck["parallel_hard_fail"] = False
    except RuntimeError as exc:
        ck["parallel_hard_fail"] = "content job" in str(exc)
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help="concurrent isolated DeepSeek requests "
                         "(default: DEEPSEEK_WORKERS or 100)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    clear_proxy_env()
    bank = build_bank(deepseek_call, workers=args.workers, progress=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(bank, ensure_ascii=False, indent=1, sort_keys=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(blob)
    tmp.replace(out)
    print(f"bank -> {out}  sha256 {hashlib.sha256(blob.encode()).hexdigest()[:12]}")
    print(json.dumps(bank["_provenance"], indent=1))
    print(f"DeepSeek workers: {min(args.workers, len(generation_jobs()))} "
          f"(requested {args.workers}; task-isolated user_id={USER_ID})")
    print("REVIEW the bank (TTS-ability + content), then COMMIT it — its hash "
          "enters config_hash; regenerating = a new bench version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
