#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rb_content_gen.py — build the FROZEN RB content bank with DeepSeek v4-flash
(rb_design §15 item 5: LLM-sampled construction diversity instead of pure
rules — user ruling 2026-07-16).

  $PY scripts/rb_content_gen.py                # real API -> exp/rb/content_bank.json
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
language sanity. Rejected samples are counted in the provenance block."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rb.registry import SCENARIOS                                # noqa: E402
from rb.grammar import (REV_UTT, BYSTANDER, PROGRESS_QUERY,      # noqa: E402
                        DISFLUENCY_FALLBACK, DISFLUENCY_FAMILIES)

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
N_PER = 6                     # paraphrases requested per (category, kind, lang)
OUT = ROOT / "exp/rb/content_bank.json"

GEN_PROMPT = (
    "You write natural SPOKEN {lang_name} utterances for a phone-call speech "
    "benchmark. Produce {n} distinct paraphrases of the EXAMPLE below. Rules: "
    "keep EVERY placeholder token EXACTLY as written (e.g. {placeholders}); "
    "colloquial, speakable, one sentence, no emoji, no quotes, no digits "
    "unless the example has them, length within 2x of the example. Reply as "
    "a JSON array of {n} strings, nothing else.\n"
    "Category: {cat}\nEXAMPLE: {example}")

LANG_NAME = {"zh": "Chinese (Mandarin)", "en": "English"}
_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def _placeholders(t):
    return sorted(set(_PLACEHOLDER_RE.findall(t)))


def validate(cand, example, lang):
    """Template gate: placeholders identical, no stray braces, speakable."""
    if not isinstance(cand, str) or not cand.strip():
        return None
    c = cand.strip()
    if _placeholders(c) != _placeholders(example):
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


def deepseek_call(prompt):
    from openai import OpenAI
    for v in ("all_proxy", "ALL_PROXY", "http_proxy", "https_proxy",
              "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE_URL)
    r = client.chat.completions.create(
        model=MODEL, temperature=1.3, max_tokens=1024,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""


def parse_array(raw):
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        return [x for x in arr if isinstance(x, str)]
    except json.JSONDecodeError:
        return []


def gen_category(call, cat, example, lang, n=N_PER):
    ph = ", ".join(_placeholders(example)) or "(none)"
    raw = call(GEN_PROMPT.format(lang_name=LANG_NAME[lang], n=n,
                                 placeholders=ph, cat=cat, example=example))
    got, ok = parse_array(raw), []
    for c in got:
        v = validate(c, example, lang)
        if v and v not in ok and v != example:
            ok.append(v)
    return ok, len(got)


def build_bank(call):
    bank = {"revision": {}, "bystander": {}, "progress": {}, "intent": {},
            "disfluency": {}}
    prov = {"model": MODEL, "n_requested": 0, "n_raw": 0, "n_accepted": 0}

    def add(dst, key_path, example, lang, cat):
        ok, n_raw = gen_category(call, cat, example, lang)
        prov["n_requested"] += N_PER
        prov["n_raw"] += n_raw
        prov["n_accepted"] += len(ok)
        node = dst
        for k in key_path[:-1]:
            node = node.setdefault(k, {})
        node[key_path[-1]] = [example] + ok      # original stays variant 0

    for lang in ("zh", "en"):
        for kind, tpl in REV_UTT[lang].items():
            add(bank["revision"], (lang, kind), tpl, lang,
                f"user revising a request mid-call ({kind})")
        for kind, tpl in BYSTANDER[lang].items():
            add(bank["bystander"], (lang, kind), tpl, lang,
                f"a third person speaking near the phone ({kind})")
        add(bank["progress"], (lang,), PROGRESS_QUERY[lang], lang,
            "user asking whether the task is done yet")
        for fam in DISFLUENCY_FAMILIES:
            add(bank["disfluency"], (lang, fam),
                DISFLUENCY_FALLBACK[lang][fam], lang,
                f"disfluent wrapper around a correction ({fam}; keep {{body}})")
    for sid, s in sorted(SCENARIOS.items()):
        for lang in ("zh", "en"):
            add(bank["intent"], (sid, lang), s["utt"][lang], lang,
                f"user asking for the task ({sid})")
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
    bank = build_bank(stub_call)
    ck["bank_shape"] = all(k in bank for k in
                           ("revision", "bystander", "progress", "intent",
                            "disfluency"))
    ck["originals_kept"] = bank["revision"]["zh"]["default"][0] == \
        REV_UTT["zh"]["default"]
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    bank = build_bank(deepseek_call)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(bank, ensure_ascii=False, indent=1, sort_keys=True)
    out.write_text(blob)
    print(f"bank -> {out}  sha256 {hashlib.sha256(blob.encode()).hexdigest()[:12]}")
    print(json.dumps(bank["_provenance"], indent=1))
    print("REVIEW the bank (TTS-ability + content), then COMMIT it — its hash "
          "enters config_hash; regenerating = a new bench version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
