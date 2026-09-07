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


class StreamingSentenceBuffer:
    """Lossless incremental splitter for ActorEngine; legacy splitter unchanged.

    One-character lookahead keeps quotes/decimals intact. Ambiguous English
    abbreviations stay with their successor. Long clauses split at whitespace
    or comma, never inside a word. Joining feed() + flush() reproduces the input.
    """

    _abbreviations = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
                      "vs", "etc", "e.g", "i.e", "a.m", "p.m"}
    _closers = '”’\"\'）)]】》'

    def __init__(self, max_chars=160):
        self.buffer = ""
        self.max_chars = max_chars

    def feed(self, delta):
        self.buffer += delta
        return self._drain(final=False)

    def flush(self):
        return self._drain(final=True)

    def _period_boundary(self, i):
        text = self.buffer
        if i + 1 < len(text) and not text[i + 1].isspace() and text[i + 1] not in self._closers:
            return False
        token = re.search(r"([\w.]+)\.$", text[:i + 1])
        stem = token.group(1) if token else ""
        if stem.lower() in self._abbreviations:
            return False
        if re.fullmatch(r"(?:[A-Za-z]\.)*[A-Za-z]", stem):
            return False
        return True

    def _drain(self, final):
        result = []
        while self.buffer:
            cut = None
            for i, char in enumerate(self.buffer):
                boundary = char in "。！？!?；;\n"
                if char == ".":
                    boundary = self._period_boundary(i)
                numeric_separator = (char in ",:" and i > 0 and self.buffer[i - 1].isdigit()
                                     and (i + 1 == len(self.buffer) or self.buffer[i + 1].isdigit()))
                if i >= self.max_chars and (char.isspace() or (char in "，,、：:" and not numeric_separator)):
                    boundary = True
                if not boundary:
                    continue
                j = i + 1
                while j < len(self.buffer) and self.buffer[j] in self._closers + "。！？!?;；.":
                    j += 1
                if j == len(self.buffer) and not final:
                    break
                cut = j
                break
            if cut is None:
                if final:
                    result.append(self.buffer)
                    self.buffer = ""
                break
            result.append(self.buffer[:cut])
            self.buffer = self.buffer[cut:]
        return result
