"""Bounded reference text, separate from completed response/history.

Sentence boundaries are synthesized PCM offsets, not word alignment or proof
of physical audibility. Only the actor feeds published, current-utterance events.
"""
from collections import deque

REFERENCE_CHARS = 512


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
