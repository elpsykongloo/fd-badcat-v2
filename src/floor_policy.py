# -*- coding: utf-8 -*-
"""
src/floor_policy.py — floor-holding rule v0 (W3 D4; agent mode only, default off).

When the user barges in while the agent is speaking, the agent must decide what
happens to the REST of its utterance. v0 is a fixed public rule (no learning):

  inputs   : kind of the current utterance
                 "narration"    result narration / chit-chat
                 "ack"          objection-window opening announce
                 "confirmation" announce of an imminent IRR/COMP commit
             window_remaining_s  min remaining objection-window budget over
                                 pending ops (None if none pending)
             eta_prior_s         registry ETA prior for the op class(es) about
                                 to execute (latency_realistic p50; None ok)
  output   : one of three tiers
                 "yield"            stop feeding further sentences NOW
                 "finish_clause"    at most ONE more queued sentence, then stop
                 "finish_utterance" reserved — v0 NEVER emits it (a full-duplex
                                    agent that talks over its user needs more
                                    justification than a fixed rule can give)

  rule     : narration and ack are UNCONDITIONALLY interruptible -> "yield".
             confirmation may finish its clause iff a commit is imminent
             (window_remaining_s <= CLAUSE_THRESHOLD_S): cutting the announce
             mid-clause would leave the user objecting to an unnamed action.

Note on actuation granularity: playback is sentence-granular (sentence-split
TTS sends bytes per sentence). "yield" therefore means "no further sentences";
audio already at the client cannot be recalled. The user's speech itself is
ALWAYS processed as a potential revision regardless of tier (Phase-B v1
behavior, unchanged) — and while the user speaks, every window countdown is
frozen by the silence clock, so floor-holding never races the commit barrier.
"""

from __future__ import annotations

from latency_realistic import CLASS_PARAMS, tool_class
import math

CLAUSE_THRESHOLD_S = 1.0
TIERS = ("yield", "finish_clause", "finish_utterance")


def eta_prior(fns):
    """Registry ETA prior: max class-p50 over the tool fns about to execute."""
    if not fns:
        return None
    return max(round(math.exp(CLASS_PARAMS[tool_class(fn)][0]), 3) for fn in fns)


def decide(kind, window_remaining_s=None, eta_prior_s=None):
    """The v0 floor-holding rule. Pure; total; returns a TIERS member."""
    if kind in ("narration", "ack"):
        return "yield"
    if kind == "confirmation":
        if window_remaining_s is not None and window_remaining_s <= CLAUSE_THRESHOLD_S:
            return "finish_clause"
        return "yield"
    return "yield"          # unknown kinds fail open (interruptible)


# ---------------------------------------------------------------------------
# W5-FC — commitment-tier policy v1 (rb_design.md v2 §6; default-off flag
# `floor_commit_tiers` in engine_b). v0 `decide()` above governs what happens
# to an utterance ALREADY being spoken when the user barges in; commit_tier()
# governs a different variable: HOW MUCH the agent verbally commits to while a
# tool is still running. Five tiers, ordered by commitment level:
#
#   silence   say nothing (short waits: the result lands before words help)
#   filler    contentless hold ("稍等，我看一下。")
#   progress  content progress, names the task + ETA (no outcome claims)
#   hedge     partial answer with an explicit uncertainty marker (only when a
#             partial result exists) — cheap to repair if falsified
#   commit    asserts the outcome (only when the result is known & confident)
#             — a wrong commit must later be REPAIRED (scored by RB's
#             commitment-repair track: WrongCommit@t / repair rate)
#
# v1 rule (threshold policy on the ETA prior; the theory note in rb_design v2
# §6 gives its P2-family optimality reading — waiting cost vs repair cost):
#   result known             -> commit if conf >= CONF_COMMIT else hedge
#   ETA unknown              -> silence until T_SILENCE_S elapsed, then filler
#   ETA <= T_SILENCE_S       -> silence
#   ETA <= T_FILLER_S        -> filler
#   ETA >  T_FILLER_S        -> progress
#   elapsed > ETA*ESCALATE   -> progress (overdue: user deserves content)
#
# Templates are machine-marked: hedge carries an uncertainty marker, commit
# carries "已确认/Confirmed" — the RB scorer detects commitments and repairs
# from these markers (trace-metadata mode) without an LLM judge.
# ---------------------------------------------------------------------------

COMMIT_TIERS = ("silence", "filler", "progress", "hedge", "commit")
T_SILENCE_S = 1.0
T_FILLER_S = 5.0
CONF_COMMIT = 0.9
ESCALATE_FACTOR = 1.5
_KAPPA_ORDER = {"READ": 0, "REV": 1, "COMP": 2, "IRR": 3}

TIER_TEMPLATES = {
    "zh": {"filler": "稍等，我看一下。",
           "progress": "正在处理{task}，大约还需要{eta}秒。",
           "hedge": "初步来看{claim}，我再确认一下。",
           "commit": "已确认：{claim}。",
           "repair": "抱歉，刚才说错了：{claim}。"},
    "en": {"filler": "One moment, let me check.",
           "progress": "Working on {task}, about {eta} more seconds.",
           "hedge": "It looks like {claim} — let me double-check.",
           "commit": "Confirmed: {claim}.",
           "repair": "Sorry, correction: {claim}."},
}


def worst_kappa(fns):
    """Highest reversibility grade over the tool fns (name form: READ..IRR)."""
    from tact.tools import REVERSIBILITY
    worst = "READ"
    for fn in fns or []:
        r = REVERSIBILITY.get(fn)
        name = r.name if r is not None else "IRR"
        if _KAPPA_ORDER[name] > _KAPPA_ORDER[worst]:
            worst = name
    return worst


def commit_tier(eta_s=None, result_known=False, result_conf=0.0,
                kappa="IRR", elapsed_s=0.0):
    """The v1 commitment-tier rule. Pure; total; returns a COMMIT_TIERS member.
    kappa is accepted for signature stability (v1 thresholds are kappa-flat;
    a kappa-conditional variant would be a NEW preregistration)."""
    if result_known:
        return "commit" if result_conf >= CONF_COMMIT else "hedge"
    if eta_s is None:
        return "filler" if elapsed_s >= T_SILENCE_S else "silence"
    if elapsed_s > eta_s * ESCALATE_FACTOR:
        return "progress"
    if eta_s <= T_SILENCE_S:
        return "silence"
    if eta_s <= T_FILLER_S:
        return "filler"
    return "progress"


def tier_utterance(tier, lang="zh", fns=None, eta_s=None, claim=None):
    """Template utterance for a tier ('' for silence). Slots: task = humanized
    tool names; eta = rounded ETA seconds; claim = the asserted content
    (hedge/commit only; caller supplies)."""
    if tier == "silence":
        return ""
    t = TIER_TEMPLATES.get(lang, TIER_TEMPLATES["zh"])
    task = "、".join(f.replace("_", " ") for f in (fns or [])) or (
        "任务" if lang == "zh" else "the task")
    if lang == "en" and fns:
        task = ", ".join(f.replace("_", " ") for f in fns)
    eta = int(round(eta_s)) if eta_s else 1
    c = claim or ("正在办理" if lang == "zh" else "it is in progress")
    if tier == "filler":
        return t["filler"]
    if tier == "progress":
        return t["progress"].format(task=task, eta=max(1, eta))
    if tier == "hedge":
        return t["hedge"].format(claim=c)
    if tier == "commit":
        return t["commit"].format(claim=c)
    return ""
