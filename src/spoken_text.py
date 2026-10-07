"""Demo-only Markdown -> canonical spoken text, before sentence TTS.

Raw model deltas remain in model cases/native continuation prefixes. The UI,
sentence references, TTS literals and played history all use this one stream.
Plain/closed inline sentences can stream immediately; incomplete markup waits
for its closing delimiter or physical line end. No TTS grammar is weakened.
"""
import re

from markdown_it import MarkdownIt

from tts_sentence import StreamingSentenceBuffer

VERSION = "spoken-text-v1"
_MD = MarkdownIt("commonmark").enable("strikethrough")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LIST = re.compile(r"^\s*(?:[-+*]|\d{1,9}[.)])[ \t]+")
_QUOTE = re.compile(r"^ {0,3}>[ \t]?")
_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+")
_AMBIGUOUS_PREFIX = re.compile(r"^\s*(?:\d{1,9}[.)]?|[-+*]|#{1,6}|>*)$")
_ONLY_MARKUP = re.compile(r"^[\s*_~`#>|:\-+=.]*$")
# Visual emoji decoration is not a spoken word. In particular an emoji-only
# sentence can make Talker generate tens of seconds of near-silence. Restrict
# stripping to prose; inline/fenced code remains literal data. Ordinary math,
# CJK, accented letters and numeric operators are outside these ranges.
_EMOJI = re.compile(r"[\U0001f000-\U0001faff\u2600-\u27bf\u200d\ufe0e\ufe0f\u20e3\U000e0020-\U000e007f]")


class SpeechTextError(RuntimeError):
    code = "speech_text_invalid"
    terminal = True


def _inline(text):
    """Return plain text and whether inline markup is safe to publish early."""
    output = []
    stable = True
    def visit(tokens):
        nonlocal stable
        for token in tokens:
            if token.type == "text":
                value = _EMOJI.sub("", token.content.translate(str.maketrans({"➕": "+", "➖": "-", "➗": "/", "✖": "*"})))
                # Unclosed markup is parsed as literal text by CommonMark. Do
                # not expose it before a future delta can close that construct.
                if re.search(r"[*_`~\[\]<>]", value):
                    stable = False
                # Drop only orphan decoration at text edges, not identifiers,
                # arithmetic operators in prose, or literal inline/fenced code.
                value = re.sub(r"^(\s*)\*{1,3}(?=\w)", r"\1", value)
                value = re.sub(r"(?<=\w)\*{1,3}(\s*)$", r"\1", value)
                output.append(value)
            elif token.type == "code_inline":
                output.append(token.content)
            elif token.type in {"softbreak", "hardbreak"}:
                output.append(" ")
            elif token.type == "image":
                visit(token.children or [])
            # Links retain their visible label; markup/HTML tags are not speech.
    # Numerical multiplication is data, even though CommonMark would otherwise
    # interpret the middle factor in 2*3*4 as emphasis.
    text = re.sub(r"(?<=\d)\*(?=\d)", r"\\*", text)
    visit(_MD.parseInline(text)[0].children or [])
    result = "".join(output)
    return ("" if _ONLY_MARKUP.fullmatch(result) else result), stable


class SpokenTextBuffer:
    def __init__(self, max_buffer=4096):
        self.buffer = ""
        self.line_start = True
        self.fence = None
        self.max_buffer = max_buffer
        self.raw_chars = self.spoken_chars = 0

    def feed(self, delta):
        self.raw_chars += len(delta)
        self.buffer += delta
        return self._drain(False)

    def flush(self):
        return self._drain(True)

    def _prefix(self, value, final_line):
        if not self.line_start:
            return value, 0
        if not final_line and _AMBIGUOUS_PREFIX.fullmatch(value):
            return None, 0
        original = value
        while _QUOTE.match(value):
            value = _QUOTE.sub("", value, count=1)
        value = _HEADING.sub("", value, count=1)
        value = _LIST.sub("", value, count=1)
        return value, len(original) - len(value)

    def _drain(self, final):
        output = []
        while self.buffer:
            end = self.buffer.find("\n")
            complete = end >= 0 or final
            line = self.buffer[:end] if end >= 0 else self.buffer
            line = line.rstrip("\r") if complete else line
            if self.line_start:
                fence = _FENCE.match(line)
                if self.fence or fence:
                    if not complete:
                        break
                    if self.fence:
                        if fence and fence[1][0] == self.fence[0] and len(fence[1]) >= self.fence[1] and not fence[2].strip():
                            self.fence = None
                        else:
                            # Code is data: preserve it, remove only fence syntax.
                            output.append(line + ("\n" if end >= 0 else ""))
                    else:
                        self.fence = (fence[1][0], len(fence[1]))
                    self.buffer = self.buffer[end + 1:] if end >= 0 else ""
                    continue
            body, prefix = self._prefix(line, complete)
            if body is None:
                break
            if complete:
                # CommonMark table delimiter / thematic break has no speech.
                if re.fullmatch(r"\s*[|:\-*_ =]+\s*", body):
                    rendered = ""
                else:
                    body = re.sub(r"\s+#+\s*$", "", body) if self.line_start and _HEADING.match(line) else body
                    rendered, _ = _inline(body)
                    if body.strip().startswith("|") and body.strip().endswith("|"):
                        rendered = "；".join(p.strip() for p in rendered.split("|") if p.strip())
                rendered = rendered.rstrip(" \t")
                if rendered:
                    output.append(rendered + ("\n" if end >= 0 else ""))
                elif end >= 0 and not self.line_start:
                    # A streamed sentence may leave only decoration on this
                    # physical line. Keep its separator even when that tail is
                    # removed; chunk boundaries must not concatenate words.
                    output.append("\n")
                self.buffer = self.buffer[end + 1:] if end >= 0 else ""
                self.line_start = True
                continue
            # Plain prose and closed inline markup retain sentence-level first
            # audio latency. A list number is removed BEFORE period splitting.
            splitter = StreamingSentenceBuffer()
            units = splitter.feed(body)
            pending = ""
            used = 0
            for unit in units:
                pending += unit
                # The sentence splitter sees the leading ! of an image as
                # punctuation. Retain it until the following [alt](url) closes.
                if pending.endswith("!") and body[used + len(pending):].startswith("["):
                    continue
                rendered, stable = _inline(pending)
                if not stable:
                    continue
                output.append(rendered)
                used += len(pending)
                pending = ""
            if not used:
                break
            self.buffer = self.buffer[prefix + used:]
            self.line_start = False
        if len(self.buffer) > self.max_buffer:
            raise SpeechTextError("Unfinished speech markup exceeded its bounded buffer")
        text = "".join(output)
        self.spoken_chars += len(text)
        return text
