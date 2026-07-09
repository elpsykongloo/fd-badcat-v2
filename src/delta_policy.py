"""delta_policy.py — W4 adaptive-ladder objection-window policies (rungs 2-3).

Preregistered per-op window policies for the adaptive ladder
(docs/w4_ladder_design.md; 09 §三):

  rung 1  fixed delta*            — --delta-policy fixed   (default, frozen path)
  rung 2  kappa-conditional rule  — --delta-policy kappa:{v0|safe|rev}
  rung 3  prompted finality       — --delta-policy prompted:v0
  rung 4  learned hazard head     — W4 D5+, not here

Design invariants:
  * The policy chooses ONLY the objection-window length per op. It never touches
    the Phase-B messages, so decision cache keys and ops semantics stay v3.1.
    (Trajectories may still diverge through the snapshot channel once commit
    times differ — same legitimate effect as moving along the fixed-delta grid.)
  * kappa source = tact.tools.REVERSIBILITY — the same symbol apply_decision_ops
    uses (12/12 FDB-v3 tools mapped; unmapped tools fall back to IRR).
  * Tables are PREREGISTERED constants. Do not tune them post-hoc; a new table
    is a new preregistered rung.
"""
import re

from tact.tools import REVERSIBILITY  # single source, same as tact_core


def kappa_name(fn):
    r = REVERSIBILITY.get(fn)
    return r.name if r is not None else "IRR"


# -- rung 2: kappa-conditional rule tables (preregistered 2026-07-09) --------
# v0   aggressive on reads (the recovery bet: 62% of FDB ops are READ)
# safe conservative variant (lower bound on rule-arm recovery)
# rev  adversarial control: anti-monotone assignment. If rev matches v0 on the
#      exact/premium Pareto, kappa-conditioning is vacuous. (Caveat: FDB's op
#      mix is READ-skewed, so rev's MEAN window is higher than v0's — the
#      control tests alignment, not level; documented in the design doc.)
KAPPA_TABLES = {
    "v0":   {"READ": 0.64, "REV": 1.0,  "COMP": 1.5, "IRR": 2.0},
    "safe": {"READ": 1.0,  "REV": 1.5,  "COMP": 2.0, "IRR": 2.0},
    "rev":  {"READ": 2.0,  "REV": 1.5,  "COMP": 1.0, "IRR": 0.64},
}

# -- rung 3: prompted-finality table (preregistered 2026-07-09) --------------
# delta[finality][kappa]. Rows: monotone in kappa; columns: final <= hesitant
# <= unfinished. final/READ = 0.0 is the deliberate aggressive corner: if the
# judge is right, no revision comes and the read commits at the decision point.
FINALITY_TABLE = {
    "final":      {"READ": 0.0, "REV": 0.64, "COMP": 1.0, "IRR": 1.5},
    "hesitant":   {"READ": 1.0, "REV": 1.5,  "COMP": 1.5, "IRR": 2.0},
    "unfinished": {"READ": 2.0, "REV": 2.0,  "COMP": 2.0, "IRR": 2.5},
}
FINALITY_LABELS = ("final", "hesitant", "unfinished")
FINALITY_FALLBACK = "hesitant"   # neutral row when the judge output is unparseable

# Frozen one-word finality instruction (T=0/seed=42 through the normal LLM path).
FINALITY_PROMPT = (
    "You will hear the tail of one speaker turn from a task-oriented phone call. "
    "Judge ONLY how the turn ends prosodically and syntactically - ignore what is "
    "being asked. Answer with EXACTLY one word:\n"
    "final - the speaker sounds done: falling intonation, complete phrase, no "
    "trailing hesitation.\n"
    "hesitant - possibly done but uncertain: trailing filler, an elongated last "
    "syllable, rising or suspended intonation, or a hedge like 'hmm' or 'let me "
    "think'.\n"
    "unfinished - clearly mid-thought: a cut-off phrase, an incomplete clause, or "
    "an item list left hanging."
)
FINALITY_TAIL_S = 8.0            # audio tail fed to the judge (preregistered)

_LABEL_RE = re.compile(r"\b(unfinished|hesitant|final)\b", re.I)


def parse_finality(raw):
    """-> (label, parsed_ok). First recognized label wins; fallback = hesitant."""
    m = _LABEL_RE.search(raw or "")
    if m:
        return m.group(1).lower(), True
    return FINALITY_FALLBACK, False


def parse_spec(spec):
    """'fixed' | 'kappa:NAME' | 'prompted:v0' -> (kind, table_name). Raises on junk."""
    if spec == "fixed":
        return "fixed", None
    kind, _, name = spec.partition(":")
    if kind == "kappa" and name in KAPPA_TABLES:
        return "kappa", name
    if kind == "prompted" and name == "v0":
        return "prompted", name
    raise ValueError(f"unknown --delta-policy {spec!r}")


def make_delta_fn(spec, finality=None):
    """Per-EoU factory: returns delta_fn(fn)->float for apply_decision_ops,
    or None for the fixed policy (frozen path: ledger uses its own delta)."""
    kind, name = parse_spec(spec)
    if kind == "fixed":
        return None
    if kind == "kappa":
        table = KAPPA_TABLES[name]
        return lambda fn: table[kappa_name(fn)]
    table = FINALITY_TABLE[finality if finality in FINALITY_TABLE else FINALITY_FALLBACK]
    return lambda fn: table[kappa_name(fn)]


def build_finality_msgs(audio_block):
    """Message list for the finality judge. audio_block = a ready content block
    (decider_b._audio_block(tail)) so the wire format matches the decider's."""
    return [{"role": "system", "content": FINALITY_PROMPT},
            {"role": "user", "content": [audio_block]}]
