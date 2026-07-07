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
