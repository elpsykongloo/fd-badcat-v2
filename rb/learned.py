# -*- coding: utf-8 -*-
"""rb/learned.py — RB adaptation layer for the FROZEN W4 stopping-time heads
(test-897 batch 2, docs/rb_test_protocol.md §六).

The heads (v2 / C0 / C1 pi points) are EVALUATION-ONLY artifacts here: weights,
theta, normalization, grid, horizon all come from the archived model JSONs
(exp/w4/stophead_v2.json, exp/w4v3/stophead_v3c*_pi*.json) and are never
refit, rescaled, or threshold-tuned on RB (firewall: RB observations must not
flow back into training or selection).

What the head needs that was FDB-shaped, and what this module supplies:

  * stophead.REQUIRED_ARGS      — per-tool required-argument lists, used by
    slots_missing_from_args (FEATS_V2 load-bearing feature). FDB's 12 tools are
    hardcoded there; RB's 28 tools live in rb.registry.TOOLS[fn]["required"]
    (the SAME frozen source the catalog prompt is built from). Tool names are
    disjoint from FDB's by construction (rb_design v2 §4), so a dict update is
    collision-free and leaves FDB behavior untouched.
  * kappa lookup                — stophead.kappa_name reads
    tact.tools.REVERSIBILITY, which rb_run.install_rb() already seeds with the
    RB kappa registry. Nothing to do here (twostage policies are kappa-flat
    anyway; only v0/v1 hazard-window policies would consume it).
  * finality one-hots           — judged at runtime by the frozen Omni finality
    prompt on the audio tail (src/delta_policy.py FINALITY_PROMPT/TAIL_S),
    exactly the W4 FDB convention; the runner owns that call (it needs the
    episode audio). Text-input smoke runs use FINALITY_FALLBACK, declared.
  * domain one-hots             — RB uses the same four domain names as FDB;
    FEATS_V2 excludes domain either way.

Nothing in this module touches rb/scorer.py, rb/sandbox.py, rb/registry.py
(scorer freeze v3) or any frozen decision path: with --delta-policy fixed
(the default) rb_run never imports this file.
"""
from __future__ import annotations

from rb.registry import TOOLS

# Per-tool required args from the frozen registry — the single source shared
# with the catalog prompt (never duplicate the lists by hand).
RB_REQUIRED_ARGS = {fn: list(spec["required"]) for fn, spec in TOOLS.items()}


def install_stophead_rb():
    """Extend the stophead module's REQUIRED_ARGS with the RB toolset (idempotent).
    Returns the number of RB entries installed. Must run before any
    make_learned_delta_fn call on RB episodes; FDB keys are left untouched."""
    import stophead
    overlap = set(stophead.REQUIRED_ARGS) & set(RB_REQUIRED_ARGS)
    assert not overlap, f"RB/FDB tool-name overlap breaks the adapter: {overlap}"
    stophead.REQUIRED_ARGS.update(RB_REQUIRED_ARGS)
    return len(RB_REQUIRED_ARGS)


def load_head(path, expect="learned:v2"):
    """Load a frozen stophead JSON and validate it against the CLI spec the
    same way the FDB harness does (w2r_stream_replay main): learned:v2 requires
    a policy=twostage model with a frozen theta."""
    import stophead
    model = stophead.StopHead.load(path)
    twostage = model.d.get("policy") == "twostage"
    if expect == "learned:v2":
        assert twostage, "learned:v2 requires a policy=twostage model JSON"
        assert model.d.get("theta") is not None, \
            "twostage model has no theta (train/remap step missing)"
    else:
        assert not twostage, f"{expect} given a twostage model — use learned:v2"
        assert model.d.get("c_w") is not None, "hazard model has no c_w"
    return model


def head_summary(model):
    d = model.d
    return {"version": d.get("version"), "policy": d.get("policy"),
            "theta": d.get("theta"), "w_protect": d.get("w_protect"),
            "risk_horizon": d.get("risk_horizon"), "feats": list(model.feats),
            **({"armc": d["armc"]} if "armc" in d else {})}
