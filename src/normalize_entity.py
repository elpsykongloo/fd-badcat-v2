# -*- coding: utf-8 -*-
"""
src/normalize_entity.py — the PUBLIC, DETERMINISTIC normalizer behind the
state track's "normalized" report (W3 D5, 裁断 C).

Design contract (defense against "you built yourself a softer scorer"):
  * The rule set below is CLOSED, ordered, and published verbatim in the paper
    appendix. No fuzzy matching, no edit distance, no aliases, no gazetteer.
  * Applied SYMMETRICALLY to gold and actual values.
  * The official exact score is never touched; the state track REPORTS BOTH
    verbatim and normalized verdicts (双报), so the delta each rule buys is
    itself an auditable number.
  * Semantic canonicalization (Vegas -> Las Vegas) is deliberately EXCLUDED:
    that is the MODEL's job (prompt v3 target #5), not the scorer's.

Rules (v1), each motivated by an observed shared-failure class in the W2
blocking/tact state-track diffs (docs/w3_state_classify output):
  N1 official base     lower / strip / underscore->space  (= official normalize)
  N2 whitespace        collapse runs of spaces
  N3 possessive        "driver's license" -> "driver license"; drop other apostrophes
  N4 ordinal suffix    "june 3rd" -> "june 3"  (digit + st|nd|rd|th, word-final)
  N5 hyphen fold       "k-2" -> "k2", "d-l-5-5-5" -> "dl555"  (all hyphens dropped)
  N6 leading article   "the gym" -> "gym"  (the|a|an at string start only)
  N7 plural fold       word-final single "s" dropped on words of len>=4 that do
                       not end in "ss" ("keyboards" -> "keyboard", "address" kept)
  N8 numeric equal     "7.0" and "7" compare equal (float equality when both
                       sides parse as numbers)
"""

from __future__ import annotations

import re

RULES_VERSION = "norm-v1"

_ORD = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_WS = re.compile(r"\s+")
_PLURAL = re.compile(r"\b([a-z]{3,}[^s\s])s\b")


def normalize_value(v):
    """N1–N7 pipeline for strings; non-strings pass through (N8 lives in
    values_equal). Idempotent: normalize(normalize(x)) == normalize(x)."""
    if not isinstance(v, str):
        return v
    s = v.lower().strip().replace("_", " ")          # N1
    s = s.replace("'s", "").replace("’s", "")        # N3 possessive
    s = s.replace("'", "").replace("’", "")          # N3 stray apostrophes
    s = _ORD.sub(r"\1", s)                           # N4
    s = s.replace("-", "")                           # N5
    s = _WS.sub(" ", s).strip()                      # N2
    for art in ("the ", "a ", "an "):                # N6
        if s.startswith(art):
            s = s[len(art):]
            break
    s = _PLURAL.sub(r"\1", s)                        # N7
    return s


def _as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def values_equal(a, b):
    """Normalized equality: N8 numeric compare first, then N1–N7 strings."""
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return na == nb
    return normalize_value(a) == normalize_value(b)


if __name__ == "__main__":
    cases = [("June 3rd", "june 3"), ("K-2", "k2"), ("driver's license", "driver license"),
             ("the gym", "gym"), ("mechanical keyboards", "mechanical keyboard"),
             ("D-L-5-5-5", "dl555"), ("7.0", 7), ("address", "address"),
             ("august 20th", "august 20"), ("V-4-4", "v44")]
    for a, b in cases:
        assert values_equal(a, b), (a, b)
    assert not values_equal("vegas", "las vegas")      # canonicalization excluded
    assert not values_equal("bob", "pop")              # no fuzz
    print(f"{RULES_VERSION}: all self-checks pass")
