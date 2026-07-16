# -*- coding: utf-8 -*-
"""rb/scorer.py — RB v2 three scoring tracks (docs/rb_design.md v2 §5;
constants = the v2.1 numeric freeze).

1. official-compatible: exact = canonical-sort multiset equality of NET
   committed calls (compensated-away pairs net out of the multiset but their
   fees stay in the transactional track) vs gold; state = final live sandbox
   state vs gold (verbatim + normalized).
2. transactional: commitment-repair — commitments/repairs are detected from
   the machine markers the FC templates carry (trace-metadata mode); an
   optional llm_judge callable((text, lang)) -> {"commit": bool, "claim": str,
   "repair": bool} handles free-text systems. WrongCommit@t = commitments
   asserting a superseded value; unrepaired ones are the utility penalty.
3. joint utility: U = 1[pass] * GAMMA**done_s - W_R*unrepaired - W_C*sum(C_k).

FROZEN constants (v2.1): GAMMA=0.95/s; W_R=0.25; W_C=0.05;
C_KAPPA = {READ:1, REV:2, COMP:4, IRR:8}; sensitivity scan x{0.5,1,2} is
reported alongside, never used for selection."""
from __future__ import annotations

import json
import re

from .registry import TOOLS
from .sandbox import canonical_calls

GAMMA = 0.95
W_R = 0.25
W_C = 0.05
C_KAPPA = {"READ": 1.0, "REV": 2.0, "COMP": 4.0, "IRR": 8.0}
SENS = (0.5, 1.0, 2.0)

COMMIT_MARKERS = {"zh": "已确认：", "en": "Confirmed: "}
REPAIR_MARKERS = {"zh": "抱歉，刚才说错了：", "en": "Sorry, correction: "}

# FROZEN commitment judge (v2.3): lets the commitment-repair track score
# FREE-TEXT systems (external SUTs speak without our FC markers). One frozen
# prompt + strict JSON; parse failure -> None verdict (marker mode already
# ran). Changing this prompt = scorer version bump, same as the markers.
COMMIT_JUDGE_PROMPT = (
    "You will read ONE utterance an assistant spoke during a task call. "
    "Classify it:\n"
    "- commit: the assistant ASSERTS a task fact or outcome as settled "
    "(booked/paid/set to VALUE, 'done', a definite confirmation of a "
    "parameter value). Questions, fillers, progress reports, and intentions "
    "('I will...') are NOT commits.\n"
    "- repair: the assistant retracts or corrects an earlier assertion.\n"
    "Reply with EXACTLY one JSON object, no prose: "
    '{"commit": true|false, "repair": true|false, "claim": "<the asserted '
    'value/fact verbatim, or empty>"}\n'
    "Utterance ({lang}): {text}")


def make_llm_judge(llm_call):
    """Adapt a raw text-completion callable into the extract_commitments
    llm_judge interface. llm_call(prompt_str) -> str. Deterministic given a
    deterministic backend (T=0 / cached)."""
    def judge(text, lang):
        raw = llm_call(COMMIT_JUDGE_PROMPT.format(lang=lang, text=text))
        try:
            m = re.search(r"\{.*\}", raw or "", re.S)
            j = json.loads(m.group(0)) if m else None
            if isinstance(j, dict):
                return {"commit": bool(j.get("commit")),
                        "repair": bool(j.get("repair")),
                        "claim": str(j.get("claim", ""))}
        except Exception:
            pass
        return None
    return judge


def net_calls(calls, state):
    """Calls whose effect survives (not compensated away)."""
    live = {k for k, v in state.items() if not v.get("void")}
    return [c for c in calls if f"{c['fn']}#{c['rid']}" in live] \
        if calls and "rid" in calls[0] else calls


def score_exact(sys_calls, gold_calls):
    return canonical_calls(sys_calls) == canonical_calls(gold_calls)


def _norm(v):
    s = str(v).casefold()
    return re.sub(r"[\s\.\,\-—，。、]+", "", s)


SANDBOX_META_KEYS = ("void", "fee", "completes_at", "aborted_at")


def score_state(live_state, gold_state, normalized=False):
    def strip(d):
        out = {}
        for k, v in d.items():
            args = {a: (_norm(x) if normalized else x) for a, x in v.items()
                    if a not in SANDBOX_META_KEYS}
            key = k.split("#")[0]              # ids are sandbox-minted: compare by fn
            out.setdefault(key, []).append(json.dumps(args, sort_keys=True,
                                                      ensure_ascii=False))
        return {k: sorted(v) for k, v in out.items()}
    return strip(live_state) == strip(gold_state)


def extract_commitments(say_events, lang, llm_judge=None):
    """say_events: [(t, text)] -> [{'t', 'kind': commit|repair, 'claim'}]."""
    out = []
    cm, rm = COMMIT_MARKERS[lang], REPAIR_MARKERS[lang]
    for t, text in say_events:
        if not text:
            continue
        if cm in text:
            out.append({"t": t, "kind": "commit",
                        "claim": text.split(cm, 1)[1].strip("。. ")})
        elif rm in text:
            out.append({"t": t, "kind": "repair",
                        "claim": text.split(rm, 1)[1].strip("。. ")})
        elif llm_judge is not None:
            j = llm_judge(text, lang) or {}
            if j.get("commit"):
                out.append({"t": t, "kind": "commit", "claim": j.get("claim", text)})
            elif j.get("repair"):
                out.append({"t": t, "kind": "repair", "claim": j.get("claim", text)})
    return out


def commitment_repair(say_events, lang, gold_values, superseded_values,
                      llm_judge=None):
    """WrongCommit@t / repair accounting. A commitment is WRONG iff its claim
    contains a superseded value and no gold value; it is REPAIRED iff a later
    repair (or later correct commit) asserts a gold value."""
    marks = extract_commitments(say_events, lang, llm_judge)
    wrong = []
    for i, m in enumerate(marks):
        if m["kind"] != "commit":
            continue
        has_gold = any(str(v) in m["claim"] for v in gold_values)
        has_old = any(str(v) in m["claim"] for v in superseded_values)
        if has_old and not has_gold:
            repaired = any(
                mm["kind"] in ("repair", "commit") and
                any(str(v) in mm["claim"] for v in gold_values)
                for mm in marks[i + 1:])
            wrong.append({"t": m["t"], "repaired": repaired})
    return {"n_commits": sum(1 for m in marks if m["kind"] == "commit"),
            "wrong_commits": len(wrong),
            "repaired": sum(1 for w in wrong if w["repaired"]),
            "unrepaired": sum(1 for w in wrong if not w["repaired"]),
            "wrong_at": [w["t"] for w in wrong]}


def comp_cost(state):
    """Sum C_kappa over COMPENSATED entries (the fee side of the ledger).
    Aborted-while-executing entries (v2.3 `aborted_at`) are free — the def2
    cost of an abort is the lost wall time, priced by gamma**done, not a
    fee."""
    tot = 0.0
    for k, v in state.items():
        if v.get("void") and "aborted_at" not in v:
            fn = k.split("#")[0]
            tot += C_KAPPA.get(TOOLS.get(fn, {}).get("kappa", "IRR"), 8.0)
    return tot


def utility(passed, done_s, unrepaired, c_sum, gamma=GAMMA, w_r=W_R, w_c=W_C):
    return round((1.0 if passed else 0.0) * (gamma ** max(0.0, done_s))
                 - w_r * unrepaired - w_c * c_sum, 4)


def score_episode(episode, sys_calls, live_state, say_events, done_s,
                  llm_judge=None):
    """One-episode scorecard (the runner aggregates)."""
    gold_calls = episode["gold_calls"]
    gold_state = episode["gold_state"]
    passed = score_exact(net_calls(sys_calls, live_state), gold_calls)
    gold_vals = list(episode["slots_final"].values())
    superseded = [r["old"] for r in episode.get("revisions", []) if r["by"] == "user"]
    if episode.get("bystander"):
        superseded.append(episode["bystander"].get("other"))
    cr = commitment_repair(say_events, episode["lang"], gold_vals,
                           [s for s in superseded if s], llm_judge)
    c_sum = comp_cost(live_state)
    row = {"id": episode["id"], "layer": episode["layer"], "arm": episode["arm"],
           "exact": passed,
           "state_verbatim": score_state(live_state, gold_state, False),
           "state_normalized": score_state(live_state, gold_state, True),
           "commit_repair": cr, "comp_cost": c_sum, "done_s": done_s,
           "U": utility(passed, done_s, cr["unrepaired"], c_sum),
           "U_sens": {str(m): utility(passed, done_s, cr["unrepaired"], c_sum,
                                      w_r=W_R * m, w_c=W_C * m) for m in SENS}}
    return row
