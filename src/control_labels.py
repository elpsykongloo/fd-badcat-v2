"""Actor control protocols: validated result -> one repair -> safe fallback."""
import asyncio
import json
import time
from async_utils import cancellable_wait

LABELS = {"judge": ("continue", "switch"), "interrupt": ("continue", "switch"),
          "shift": ("no", "yes"), "input_route": ("keep", "stop_only", "yield_wait", "yield_ready"),
          "input_reply": ("keep", "yield_ready")}
FALLBACK = {"judge": "continue", "interrupt": "continue", "shift": "no", "input_route": "keep"}
ROUTE_PROTOCOL = "transcript-first-v1"
ROUTE_PROTOCOL_V2 = "transcript-first-v2"
# v2: the assistant turn is prefilled with this exact prefix (continue_final_message),
# so the model only generates `<transcript>","label":"<label>"}`.
ROUTE_PREFIX = '{"transcript":"'
ROUTE_MAX_OUTPUT = 4096
REPLY_PROTOCOL = "played-reply-v2"
_V1_FENCE = "```json\n"


def route_protocol(prefill=False):
    return ROUTE_PROTOCOL_V2 if prefill else ROUTE_PROTOCOL


def route_prefill_message():
    return {"role": "assistant", "content": ROUTE_PREFIX}


def with_route_prefill(messages, prefill=True):
    """The prefix is always the LAST message (also after a repair user message)."""
    return list(messages) + [route_prefill_message()] if prefill else messages


class RouteScanner:
    """Incremental lexer that finds the closing quote of the label value.

    Tracks only string boundaries (`\\"`, `\\\\` and `\\uXXXX` are skipped as
    escapes; hex digits can never close a string) and brace depth, character
    by character, so any delta split is equivalent to one delta. It never
    decides anything itself: the caller must run the full parse_route on
    `candidate()`; a structurally different output fails that parse.
    """

    def __init__(self, prefilled=False):
        self.prefilled = bool(prefilled)
        self._text = []
        self._size = 0
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._strings = 0
        self.closed_at = None  # offset in GENERATED text just after the label's closing quote
        if self.prefilled:
            self._scan(ROUTE_PREFIX, 0)

    def _scan(self, chunk, offset):
        for i, ch in enumerate(chunk):
            if self._in_string:
                if self._escape:
                    self._escape = False
                elif ch == "\\":
                    self._escape = True
                elif ch == '"':
                    self._in_string = False
                    if self._depth == 1:
                        self._strings += 1
                        # transcript key, transcript value, label key, label value
                        if self._strings == 4:
                            return offset + i + 1
            elif ch == '"':
                self._in_string = True
            elif ch in "{[":
                self._depth += 1
            elif ch in "}]":
                self._depth -= 1
        return None

    def feed(self, delta):
        """Return True exactly once: when the label value's closing quote arrives."""
        if self.closed_at is not None or not delta:
            if delta:
                self._text.append(delta)
                self._size += len(delta)
            return False
        offset = self._size
        self._text.append(delta)
        self._size += len(delta)
        closed = self._scan(delta, offset)
        if closed is None:
            return False
        self.closed_at = closed
        return True

    def text(self):
        return "".join(self._text)

    def head(self):
        return self.text()[:self.closed_at] if self.closed_at is not None else None

    def trailing(self):
        return self.text()[self.closed_at:] if self.closed_at is not None else ""

    def candidate(self):
        """Generated text up to the label close, completed as parse_route expects."""
        head = self.head()
        if head is None:
            return None
        if not self.prefilled and head.lstrip().startswith(_V1_FENCE):
            return head + "\n}\n```"
        return head + "}"


def input_route_token_budget(audio_samples):
    """Bound protocol tails while allowing normal speech plus JSON overhead."""
    if type(audio_samples) is not int or audio_samples < 0:
        raise ValueError("Audio sample count must be a nonnegative integer")
    return min(256, max(64, 48 + (audio_samples + 999) // 1000))


def reply_messages(prompt, transcript, context):
    # Data stays out of the instruction role. No audio and no generated future
    # text: this call cannot rewrite the independently obtained transcription.
    return [{"role": "system", "content": prompt}, {"role": "user", "content":
        json.dumps({"played_assistant": context, "user_transcript": transcript}, ensure_ascii=False)}]


async def decide_input_route(call, messages, timeout, *, playing=False, closed=True,
                             reply_prompt=None, reply_context="", on_review=None,
                             route_prefill=False):
    """Audio decision + optional text-only reply check share ONE deadline.

    call(messages, stage) includes queue acquisition. The review is one attempt
    on a valid nonempty KEEP or STOP_ONLY decision; it can only promote to READY
    and never authorizes STOP/WAIT. WAIT is never reviewed: the user is still
    talking, so an unfinished fragment must not be answered. The audio stage
    decides stopping and waiting without assistant text; the played context
    decides whether the heard words call for a reply. on_review(label) runs just
    before the review so a STOP can take effect without waiting for it.
    """
    deadline = time.perf_counter() + timeout
    label, audit = await decide_control(lambda msgs: call(msgs, "input_route"),
                                        messages, "input_route", timeout,
                                        route_prefill=route_prefill)
    audit["base_label"] = label
    if not (reply_prompt and playing and closed and reply_context and label in ("keep", "stop_only")
            and not audit["fallback"] and audit.get("transcript", "").strip()):
        return label, audit
    review = {"protocol": REPLY_PROTOCOL, "attempts": 0, "fallback": False,
              "timed_out": False, "label": "keep"}
    audit["reply_review"] = review
    if on_review is not None:
        on_review(label)
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        review.update(timed_out=True, fallback=True)
        return label, audit
    started = time.perf_counter()
    try:
        review["attempts"] = 1
        raw = await cancellable_wait(call(reply_messages(reply_prompt, audit["transcript"],
                                                        reply_context), "input_reply"), remaining)
        review["raw"] = str(raw)[:128]
        # Deliberately no repair, label extraction, or extra action authority.
        checked = parse_label("input_reply", raw)
        if checked is None:
            review["fallback"] = True
        else:
            review["label"] = checked
            # A review KEEP leaves the audio decision (keep/stop) untouched.
            if checked == "yield_ready":
                label = checked
    except asyncio.TimeoutError:
        review.update(timed_out=True, fallback=True)
    except Exception as exc:
        review.update(error=type(exc).__name__, fallback=True)
    review["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return label, audit


def parse_route(raw, prefilled=False):
    """No label extraction, duplicate keys, reversed order or empty evidence.

    Fences are harmless formatting. The transcript is diagnostic evidence, not
    a second independent vote or a replacement for the user's audio/history.
    prefilled=True (transcript-first-v2): `raw` is the continuation after the
    prefilled ROUTE_PREFIX; the prefix is joined back and every check is the same.
    """
    if not isinstance(raw, str) or len(raw) > ROUTE_MAX_OUTPUT:
        return None
    if prefilled:
        body = (ROUTE_PREFIX + raw).strip()
    else:
        body = raw.strip()
        if body.startswith(_V1_FENCE) and body.endswith("\n```"):
            body = body[8:-4].strip()
    try:
        pairs = json.loads(body, object_pairs_hook=lambda pairs: pairs)
        if (not isinstance(pairs, list) or len(pairs) != 2
                or any(not isinstance(p, tuple) or len(p) != 2 for p in pairs)
                or [p[0] for p in pairs] != ["transcript", "label"]):
            return None
        result = dict(pairs)
        transcript, label = result["transcript"], result["label"]
        if (not isinstance(transcript, str) or len(transcript) > 512
                or not isinstance(label, str) or label not in LABELS["input_route"]
                or (label != "keep" and not transcript.strip())):
            return None
        return result
    except (ValueError, TypeError, RecursionError):
        return None


def parse_label(kind, raw, *, legacy_route=False, prefilled=False):
    if kind == "input_reply":
        return raw.strip() if isinstance(raw, str) and raw.strip() in LABELS[kind] else None
    if kind == "input_route" and not legacy_route:
        result = parse_route(raw, prefilled=prefilled)
        return result["label"] if result is not None else None
    if not isinstance(raw, str):
        return None
    text = raw.strip().lower()
    # Only harmless formatting; never extract a label from an explanation.
    text = text.removesuffix("。").removesuffix(".").strip().strip("\"'").strip()
    return text if text in LABELS[kind] else None


async def decide_control(call, messages, kind, timeout, *, route_prefill=False):
    """route_prefill (input_route only): `messages` is the UNprefilled request; the
    v2 prefix is appended as the last message of the first AND the repair request."""
    deadline = time.perf_counter() + timeout
    audit = {"kind": kind, "attempts": 0, "raw": [], "errors": [],
             "repaired": False, "fallback": False, "timed_out": False}
    prefilled = bool(route_prefill and kind == "input_route")
    request = with_route_prefill(messages, prefilled)
    for attempt in range(2):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            audit["timed_out"] = True
            break
        audit["attempts"] += 1
        try:
            raw = await cancellable_wait(call(request), remaining)
        except asyncio.TimeoutError:
            audit["timed_out"] = True
            break
        except Exception as exc:
            raw = ""
            audit["errors"].append(type(exc).__name__)
        audit["raw"].append(str(raw)[:512])
        label = parse_label(kind, raw, prefilled=prefilled)
        if label is not None:
            audit["repaired"] = attempt == 1
            if kind == "input_route":
                audit.update(protocol=route_protocol(prefilled),
                             transcript=parse_route(raw, prefilled=prefilled)["transcript"])
            return label, audit
        if kind == "input_route":
            # Do not echo a malformed/hallucinated transcript back as evidence.
            request = with_route_prefill(messages + [{"role": "user", "content":
                'The previous result was invalid. Transcribe the original audio and classify it '
                'under the original rules. Return exactly one JSON object, with the string field '
                '"transcript" FIRST and "label" SECOND. label must be keep, stop_only, '
                'yield_wait or yield_ready. With no clear speech use an empty transcript and keep. '
                'No additional fields or explanation.'}], prefilled)
            continue
        request = messages + ([{"role": "assistant", "content": raw[:512]}]
                              if isinstance(raw, str) and raw.strip() else []) + [
            {"role": "user", "content": "The previous classification was missing or invalid. "
             "Classify the original audio under the original rules. Reply with exactly ONE label: "
             + " or ".join(LABELS[kind]) + ". No explanation or other text."}]
    audit["fallback"] = True
    return FALLBACK[kind], audit
