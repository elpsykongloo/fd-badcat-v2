# -*- coding: utf-8 -*-
"""
src/tts_sentence.py — sentence splitting for incremental (per-sentence) TTS
(W3 D4 "分句 TTS 接完成锚"; agent mode only, default off).

Whole-utterance TTS synthesizes the full reply before the first byte plays;
per-sentence synthesis moves the first-audio anchor to the first SENTENCE and
— on barge-in — lets the engine drop the not-yet-played tail (the floor policy
decides how much). The completion anchor gain (E5) is first_audio(sentence 1)
vs first_audio(full utterance), measured live.

Pure text utility here; dispatch lives in engine_b.TactEngine.
"""

from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[.!?;])\s+")
MIN_CHARS = 12          # fragments shorter than this merge forward ("Done. ")


def split_sentences(text, min_chars=MIN_CHARS):
    """Split into sentence units, merging fragments < min_chars into their
    successor (trailing short fragment merges backward). Whitespace-preserving
    enough for TTS; never returns empty strings."""
    parts = [p.strip() for p in _SPLIT.split(text or "") if p.strip()]
    if not parts:
        return []
    merged, buf = [], ""
    for p in parts:
        buf = f"{buf} {p}".strip() if buf else p
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = f"{merged[-1]} {buf}"
        else:
            merged.append(buf)
    return merged
