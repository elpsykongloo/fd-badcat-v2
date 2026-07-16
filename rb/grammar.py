# -*- coding: utf-8 -*-
"""rb/grammar.py — RB layer definitions, event grammar, and utterance
templates (docs/rb_design.md v2 §2/§3; v2.3 = review-driven revision).

Layers (v2.3):
  L1  inline revision            L2  short-gap revision
  L3  epsilon-band revision      L4  race-region revision (value-first)
  L5  chain revision             L6  chain + double revision
  L7  COMPENSATION arena: the revision is TIMED PAST the reference commit
      horizon (hold + infer + delta* + tool wall + margin), so under the
      reference window the forward op is already committed — the only path
      to the gold terminal state is reverse-tool + relaunch (or, for systems
      holding longer windows, an in-window patch; both net to the same call
      multiset — the fee/time difference lands in the transactional track).
  L8  in-flight events           L9  latency long-tail
  L10 adversarial bystander      L11 TTS barge-in revision (arm B; the user
      interrupts the agent's own speech to revise — the signature full-duplex
      revision cell)
  L12 ATTRIBUTION arena (the old L7-multi construction): the revision names a
      STEP-2-ONLY slot while step 1's window is open — probes revision-target
      binding (the v2.2.1 anti-window cells' mechanism).

Arm-B event grammar: a rule = (lifecycle_state, offset_spec, action, content
kind). v2.3: arm-B revision content is delivered ONLY through events (the
v2.2 script-piece + event double delivery is removed); the `inflight` anchor
is the first COMP/IRR launch (was: first launch of any kappa — always a READ
in every chain, so nothing ever landed in a transactional execution window).

Content: templates below are the deterministic fallback. When
exp/rb/content_bank.json exists (built by scripts/rb_content_gen.py from
DeepSeek v4-flash samples, then FROZEN — its hash enters config_hash), text
functions draw seeded variants from the bank and optionally inject one
disfluency (five families: false_start, filler, repetition, elongation,
self_repair). `content_hook` remains the external override interface.
"""
import hashlib
import json
from pathlib import Path

HOLD_S = 0.64
NOMINAL_INFER_S = 1.0     # nominal decision latency (arm-A lifecycle projection)
DELTA_REF_S = 1.5         # reference objection window (L7 commit-horizon calc)
L7_MARGIN_S = 0.30
DISFLUENCY_P = 0.30       # per-revision injection probability (bank builds)

# gap bins per layer (uniform in-bin unless a pause prior narrows them)
LAYER_GAP = {
    "L2": (0.20, 0.55),
    "L3": (0.64, 0.80),
    "L4": (0.68, 1.14),
    "L5": (1.00, 4.00),
    "L6": [(0.70, 1.50), (1.00, 3.00)],   # two revisions
    "L7": None,                            # computed past the commit horizon
    "L8": None,                            # timed to the in-flight window
    "L9": None,                            # no revision by default (latency layer)
    "L10": None,                           # adversarial: lifecycle-timed injection
    "L11": None,                           # reactive: anchored to agent TTS onset
    "L12": (1.00, 2.50),                   # attribution arena (old L7 bin)
}
LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10",
          "L11", "L12")

# scenario kind preference per layer (chain exercises DAG; multi = independent
# calls; single = COMP/IRR-terminal short tasks)
LAYER_KIND = {"L1": "single", "L2": "single", "L3": "chain", "L4": "chain",
              "L5": "chain", "L6": "chain", "L7": "single", "L8": "chain",
              "L9": "single", "L10": "chain", "L11": "single", "L12": "multi"}

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
# state in {eou, inflight, committed, tts}; offset_spec = (lo, hi) seconds
# after the state's anchor event — except inflight, where it is a FRACTION of
# the anchoring tool's wall time. v2.3 anchors: inflight = first COMP/IRR
# launch; committed = first commit (live since the feed_sim watermark fix);
# tts = first agent audio onset.
ARM_B_RULES = {
    "L4": [("eou", (0.04, 0.50), "revise", "value_first")],
    "L5": [("eou", (0.36, 3.36), "revise", "default")],
    "L6": [("eou", (0.06, 0.86), "revise", "default"),
           ("committed", (0.30, 1.50), "revise", "second")],
    "L7": [("committed", (0.30, 1.20), "revise", "default")],
    "L8": [("inflight", (0.20, 0.80), "revise", "default"),
           ("inflight", (0.20, 0.80), "cancel", "cancel"),
           ("inflight", (0.30, 0.90), "progress_query", "progress")],
    "L9": [("inflight", (0.50, 0.70), "progress_query", "progress")],
    "L10": [("inflight", (0.10, 0.80), "bystander", "command"),
            ("tts", (0.10, 0.60), "bystander", "command"),
            ("inflight", (0.10, 0.80), "bystander", "irrelevant"),
            ("eou", (0.30, 1.20), "benign_control", "default")],
    "L11": [("tts", (0.05, 0.40), "revise", "default")],
}
# layers whose arm-B revision/cancel content arrives ONLY via events (v2.3:
# the generator emits no script piece for these on arm B — de-duplication)
ARM_B_EVENT_ONLY = ("L4", "L5", "L6", "L7", "L11")

# ---------------------------------------------------------------------------
# content bank (optional, frozen artifact) + disfluency injection
# ---------------------------------------------------------------------------
BANK_PATH = Path(__file__).resolve().parents[1] / "exp/rb/content_bank.json"
_BANK = None
_BANK_HASH = "none"
if BANK_PATH.exists():
    _raw = BANK_PATH.read_bytes()
    _BANK = json.loads(_raw)
    _BANK_HASH = hashlib.sha256(_raw).hexdigest()[:12]

DISFLUENCY_FAMILIES = ("false_start", "filler", "repetition", "elongation",
                       "self_repair")
# deterministic fallback disfluency patterns ({body} = the clean utterance,
# {new} available for self_repair)
DISFLUENCY_FALLBACK = {
    "zh": {"false_start": "那个我先——{body}",
           "filler": "嗯……{body}",
           "repetition": "改成，改成，{body}",
           "elongation": "呃——{body}",
           "self_repair": "改成那个……不对，{body}"},
    "en": {"false_start": "So I was— {body}",
           "filler": "Um... {body}",
           "repetition": "Make it, make it, {body}",
           "elongation": "Uh— {body}",
           "self_repair": "Change it to the... no wait, {body}"},
}


def bank_hash():
    return _BANK_HASH


def _bank_pick(rng, path_keys, fallback):
    """Seeded pick from a bank list at bank[k0][k1]...; falls back."""
    node = _BANK
    for k in path_keys:
        if not isinstance(node, dict) or k not in node:
            return fallback
        node = node[k]
    if isinstance(node, list) and node:
        return node[rng.randrange(len(node))]
    return fallback


def maybe_disfluent(rng, lang, text, new=None):
    """With DISFLUENCY_P (bank builds only), wrap `text` in one seeded
    disfluency family. Inline-kind texts are never wrapped (they splice into
    the intent sentence)."""
    if _BANK is None or rng.random() >= DISFLUENCY_P:
        return text
    fam = DISFLUENCY_FAMILIES[rng.randrange(len(DISFLUENCY_FAMILIES))]
    pat = _bank_pick(rng, ("disfluency", lang, fam),
                     DISFLUENCY_FALLBACK[lang][fam])
    try:
        return pat.format(body=text, new=new if new is not None else "")
    except (KeyError, IndexError):
        return text


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


def l7_gap(rng, tool_wall_s):
    """L7 compensation gap: past the reference commit horizon by construction
    (EoU hold + nominal infer + delta* + the forward tool's wall + margin)."""
    lo = HOLD_S + NOMINAL_INFER_S + DELTA_REF_S + float(tool_wall_s) + L7_MARGIN_S
    return round(lo + rng.random() * 1.5, 3)


def intent_text(lang, scenario_id, default_template, rng=None):
    """Scenario intent template: seeded bank paraphrase when available."""
    if rng is None:
        return default_template
    return _bank_pick(rng, ("intent", scenario_id, lang), default_template)


def revision_text(lang, kind, new, content_hook=None, rng=None):
    if content_hook:
        out = content_hook(kind, lang, {"new": new})
        if out:
            return out
    t = REV_UTT[lang].get(kind, REV_UTT[lang]["default"])
    if rng is not None:
        t = _bank_pick(rng, ("revision", lang, kind), t)
    try:
        text = t.format(new=new)
    except (KeyError, IndexError):
        text = REV_UTT[lang].get(kind, REV_UTT[lang]["default"]).format(new=new)
    if rng is not None and kind not in ("inline", "cancel"):
        text = maybe_disfluent(rng, lang, text, new=new)
    return text


def bystander_text(lang, kind, other=None, content_hook=None, rng=None):
    if content_hook:
        out = content_hook("bystander_" + kind, lang, {"other": other})
        if out:
            return out
    t = BYSTANDER[lang][kind]
    if rng is not None:
        t = _bank_pick(rng, ("bystander", lang, kind), t)
    try:
        return t.format(other=other) if "{other}" in t else t
    except (KeyError, IndexError):
        t = BYSTANDER[lang][kind]
        return t.format(other=other) if "{other}" in t else t


def progress_text(lang, rng=None):
    if rng is None:
        return PROGRESS_QUERY[lang]
    return _bank_pick(rng, ("progress", lang), PROGRESS_QUERY[lang])
