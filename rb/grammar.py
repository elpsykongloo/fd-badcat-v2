# -*- coding: utf-8 -*-
"""rb/grammar.py — RB v2 layer definitions, event grammar, and utterance
templates (docs/rb_design.md v2 §2/§3).

Layers L1–L10: gap bins are on the SILENCE CLOCK (seconds between the end of
the intent utterance and the start of the revision utterance). L4 revision
text is value-first (the new value is the first word — travel_10 lesson).

Arm-B event grammar: a rule = (lifecycle_state, offset_spec, action, content
kind). The reactive user simulator (rb/simulator.py) tracks the engine's
lifecycle from trace events and fires rules deterministically per episode
seed. Content is template-filled; `content_hook` (an optional callable
(kind, lang, slots) -> str) is the LLM slot interface — the default is
template-only so builds are fully deterministic with no API.
"""

HOLD_S = 0.64
NOMINAL_INFER_S = 1.0     # nominal decision latency (arm-A lifecycle projection)

# gap bins per layer (uniform in-bin unless a pause prior narrows them)
LAYER_GAP = {
    "L2": (0.20, 0.55),
    "L3": (0.64, 0.80),
    "L4": (0.68, 1.14),
    "L5": (1.00, 4.00),
    "L6": [(0.70, 1.50), (1.00, 3.00)],   # two revisions
    "L7": (1.00, 2.50),
    "L8": None,                            # timed to the in-flight window
    "L9": None,                            # no revision by default (latency layer)
    "L10": None,                           # adversarial: lifecycle-timed injection
}
LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10")

# scenario kind preference per layer (chain exercises DAG; multi = independent
# calls; single = COMP/IRR-terminal short tasks)
LAYER_KIND = {"L1": "single", "L2": "single", "L3": "chain", "L4": "chain",
              "L5": "chain", "L6": "chain", "L7": "multi", "L8": "chain",
              "L9": "single", "L10": "chain"}

REV_UTT = {
    "zh": {"default": "等等，改成{new}。",
           "value_first": "{new}，改成{new}。",
           "inline": "，哦不对，改成{new}",
           "second": "还有，再改一下，要{new}。",
           "cancel": "算了，先别办了。"},
    "en": {"default": "Wait — make it {new}.",
           "value_first": "{new} — change it to {new}.",
           "inline": ", oh no, make that {new}",
           "second": "Also, one more change — {new}.",
           "cancel": "Actually, hold off, don't do it yet."},
}
PROGRESS_QUERY = {"zh": "好了没？", "en": "Any progress?"}
BYSTANDER = {
    "zh": {"command": "改成{other}吧。", "cancel": "别买了别买了。",
           "irrelevant": "今晚想吃什么？我们点外卖吧。"},
    "en": {"command": "Change it to {other}.", "cancel": "Don't buy it, cancel that.",
           "irrelevant": "What do you want for dinner? Let's order in."},
}

# Arm-B event rules per layer: (state, offset_spec, action, content_kind).
# state ∈ {eou, inflight, committed, tts}; offset_spec = (lo, hi) seconds
# after the state's anchor event — except inflight, where it is a FRACTION of
# the tool's wall time (lands inside the execution window by construction).
ARM_B_RULES = {
    "L4": [("eou", (0.04, 0.50), "revise", "value_first")],
    "L5": [("eou", (0.36, 3.36), "revise", "default")],
    "L6": [("eou", (0.06, 0.86), "revise", "default"),
           ("committed", (0.30, 1.50), "revise", "second")],
    "L8": [("inflight", (0.20, 0.80), "revise", "default"),
           ("inflight", (0.20, 0.80), "cancel", "cancel"),
           ("inflight", (0.30, 0.90), "progress_query", "progress")],
    "L9": [("inflight", (0.50, 0.70), "progress_query", "progress")],
    "L10": [("inflight", (0.10, 0.80), "bystander", "command"),
            ("tts", (0.10, 0.60), "bystander", "command"),
            ("inflight", (0.10, 0.80), "bystander", "irrelevant"),
            ("eou", (0.30, 1.20), "benign_control", "default")],
}


def gap_for_layer(layer, rng, pause_prior=None, which=0):
    """Sample a revision gap for a layer. With a pause prior (w5sg census
    histogram), sample from the prior RESTRICTED to the layer bin; else
    uniform in-bin. Deterministic given rng state."""
    spec = LAYER_GAP.get(layer)
    if spec is None:
        return None
    lo, hi = spec[which] if isinstance(spec, list) else spec
    if pause_prior:
        edges = pause_prior["hist_edges_s"]
        counts = pause_prior["hist_counts"]
        cells = [(edges[i], edges[i + 1], counts[i]) for i in range(len(counts))
                 if edges[i + 1] > lo and edges[i] < hi and counts[i] > 0]
        tot = sum(c for _a, _b, c in cells)
        if tot > 0:
            x = rng.random() * tot
            for a, b, c in cells:
                if x < c:
                    return round(max(lo, a) + rng.random() * (min(hi, b) - max(lo, a)), 3)
                x -= c
    return round(lo + rng.random() * (hi - lo), 3)


def revision_text(lang, kind, new, content_hook=None):
    if content_hook:
        out = content_hook(kind, lang, {"new": new})
        if out:
            return out
    t = REV_UTT[lang].get(kind, REV_UTT[lang]["default"])
    return t.format(new=new)


def bystander_text(lang, kind, other=None, content_hook=None):
    if content_hook:
        out = content_hook("bystander_" + kind, lang, {"other": other})
        if out:
            return out
    t = BYSTANDER[lang][kind]
    return t.format(other=other) if "{other}" in t else t
