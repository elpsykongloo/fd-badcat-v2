"""Bounded reference text, separate from completed response/history.

Sentence boundaries are synthesized PCM offsets, not word alignment or proof
of physical audibility. Only the actor feeds published, current-utterance events.
"""
import re
from collections import deque

REFERENCE_CHARS = 512
# Seconds per non-space character used to bound a unit whose END is not sent yet:
# ASCII letters, then everything else (CJK, digits, symbols, punctuation pauses).
# On 178 captured demo TTS units the bound is at least 1.1x the synthesized length.
UNIT_MAX_S_PER_CHAR = (0.1, 0.3)
# A heard partial ends at a clause mark; ASCII marks only before a space or the end.
CLAUSE_MARKS = "，。？！；：…"
ASCII_CLAUSE_MARKS = ",.?!;:"


QUESTION_CUES_ZH = ("？", "吗", "呢", "还是", "是否", "要不要", "想不想", "能不能", "可不可以",
                    "是不是", "好不好", "对不对", "行不行", "有没有", "什么", "怎么", "哪", "多少")
QUESTION_LEADS_EN = ("would", "do", "does", "did", "shall", "should", "can", "could", "will",
                     "are", "is", "what", "which", "how", "why", "when", "where", "who")


def looks_like_question(text):
    """Cheap lexical test: does a heard fragment read as a question to the user?"""
    text = (text or "").strip()
    if not text:
        return False
    if any(cue in text for cue in QUESTION_CUES_ZH) or "?" in text:
        return True
    lowered = text.lower()
    for clause in re.split(r"[,.;:!]+", lowered):
        words = clause.split()
        if words and words[0] in QUESTION_LEADS_EN:
            return True
    return " or " in f" {lowered} "


def completed_context(sentences, played):
    """Last three fully ACKed sentences, bounded without cutting a sentence.

    sentence_end offsets already exist for heard-history tracking. Never infer
    completion from the next sentence's start or from generated text.
    """
    completed = [text for end, text in sentences if end <= played]
    tail = []
    for text in reversed(completed[-3:]):
        if len(text) + sum(map(len, tail)) > REFERENCE_CHARS:
            break
        tail.append(text)
    return "".join(reversed(tail))


def unit_seconds_bound(text):
    """Upper bound on the synthesized length of `text`, for a unit without an END."""
    return sum(UNIT_MAX_S_PER_CHAR[not (ch.isascii() and ch.isalpha())]
               for ch in text if not ch.isspace())


def clause_prefix(text, count):
    """Longest prefix of text[:count] that ends at a clause mark, else ""."""
    for i in range(min(count, len(text)), 0, -1):
        ch = text[i - 1]
        if ch in CLAUSE_MARKS or (ch in ASCII_CLAUSE_MARKS and (i == len(text) or text[i].isspace())):
            return text[:i]
    return ""


class SpeechReference:
    def __init__(self):
        self.generated_tail = ""
        self.sentences = deque(maxlen=64)

    def observe(self, kind, data):
        if kind == "text_delta":
            self.generated_tail = (self.generated_tail + data["text"])[-REFERENCE_CHARS:]
        elif kind == "text_done":
            self.generated_tail = data["text"][-REFERENCE_CHARS:]
        elif kind == "sentence":
            start = data.get("start_sample")
            if type(start) is int and start >= 0:
                self.sentences.append((start, data["text"][-REFERENCE_CHARS:]))

    def heard_window(self, *, played, sent, min_heard):
        """Units already audible at `played`, for reply review before any unit ends.

        A unit counts only once playback has passed its start; the newest one must
        also have been heard for at least `min_heard` samples, else it is dropped.
        Units starting at or after `played` were never heard and never count.
        """
        eligible = [(start, text) for start, text in self.sentences
                    if start < played and start < sent]
        if eligible and played - eligible[-1][0] < min_heard:
            eligible = eligible[:-1]
        tail = []
        for _, text in reversed(eligible[-3:]):
            if len(text) + sum(map(len, tail)) > REFERENCE_CHARS:
                break
            tail.append(text)
        return "".join(reversed(tail))

    def heard_prefix(self, *, played, sent, ends, min_heard, rate, limit=REFERENCE_CHARS):
        """Heard clauses of the unit still playing at `played`, else "".

        Same unit as heard_window's newest one (started, heard for `min_heard`).
        Its characters count in proportion to the samples already played: over its
        END when sent, otherwise over a duration bound from its text and never
        `sent` alone, which runs ahead of playback; without an END the whole unit
        never counts. At most `limit` characters, cut back to the last clause mark
        so a half-heard phrase never reads as a different question. A unit ended
        by `played` is completed context, not a partial; a tail-cut unit is skipped.
        """
        eligible = [(start, text) for start, text in self.sentences
                    if start < played and start < sent]
        if not eligible:
            return ""
        start, text = eligible[-1]
        heard = played - start
        if heard < max(min_heard, 1) or len(text) >= REFERENCE_CHARS:
            return ""
        end = min((end for end in ends if end > start), default=None)
        if end is not None and end <= played:
            return ""
        if end is None:
            end = start + max(sent - start, round(rate * unit_seconds_bound(text)))
        count = min(len(text) * heard // (end - start), len(text) - 1, limit)
        return clause_prefix(text, count)

    def snapshot(self, *, started, played, sent):
        if not started:
            return self.generated_tail, "generated_unplayed"
        # Future TTS sentences can be prepared before current audio is played.
        # Select by the last valid browser ACK, not the latest generated token.
        eligible = [(start, text) for start, text in self.sentences
                    if start <= played and start < sent]
        if not eligible:
            return "", "unavailable"
        return "".join(text for _, text in eligible[-3:])[-REFERENCE_CHARS:], "playback_sentence_window"
