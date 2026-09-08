"""Actor control protocols: validated result -> one repair -> safe fallback."""
import asyncio
import json
import time
from async_utils import cancellable_wait

LABELS = {"judge": ("continue", "switch"), "interrupt": ("continue", "switch"),
          "shift": ("no", "yes"), "input_route": ("keep", "stop_only", "yield_wait", "yield_ready")}
FALLBACK = {"judge": "continue", "interrupt": "continue", "shift": "no", "input_route": "keep"}
ROUTE_PROTOCOL = "transcript-first-v1"
ROUTE_MAX_OUTPUT = 4096


def parse_route(raw):
    """No label extraction, duplicate keys, reversed order or empty evidence.

    Fences are harmless formatting. The transcript is diagnostic evidence, not
    a second independent vote or a replacement for the user's audio/history.
    """
    if not isinstance(raw, str) or len(raw) > ROUTE_MAX_OUTPUT:
        return None
    body = raw.strip()
    if body.startswith("```json\n") and body.endswith("\n```"):
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


def parse_label(kind, raw, *, legacy_route=False):
    if kind == "input_route" and not legacy_route:
        result = parse_route(raw)
        return result["label"] if result is not None else None
    if not isinstance(raw, str):
        return None
    text = raw.strip().lower()
    # Only harmless formatting; never extract a label from an explanation.
    text = text.removesuffix("。").removesuffix(".").strip().strip("\"'").strip()
    return text if text in LABELS[kind] else None


async def decide_control(call, messages, kind, timeout):
    deadline = time.perf_counter() + timeout
    audit = {"kind": kind, "attempts": 0, "raw": [], "errors": [],
             "repaired": False, "fallback": False, "timed_out": False}
    request = messages
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
        label = parse_label(kind, raw)
        if label is not None:
            audit["repaired"] = attempt == 1
            if kind == "input_route":
                audit.update(protocol=ROUTE_PROTOCOL, transcript=parse_route(raw)["transcript"])
            return label, audit
        if kind == "input_route":
            # Do not echo a malformed/hallucinated transcript back as evidence.
            request = messages + [{"role": "user", "content":
                'The previous result was invalid. Transcribe the original audio and classify it '
                'under the original rules. Return exactly one JSON object, with the string field '
                '"transcript" FIRST and "label" SECOND. label must be keep, stop_only, '
                'yield_wait or yield_ready. With no clear speech use an empty transcript and keep. '
                'No additional fields or explanation.'}]
            continue
        request = messages + ([{"role": "assistant", "content": raw[:512]}]
                              if isinstance(raw, str) and raw.strip() else []) + [
            {"role": "user", "content": "The previous classification was missing or invalid. "
             "Classify the original audio under the original rules. Reply with exactly ONE label: "
             + " or ".join(LABELS[kind]) + ". No explanation or other text."}]
    audit["fallback"] = True
    return FALLBACK[kind], audit
