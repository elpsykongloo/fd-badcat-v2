"""Actor binary control protocol: exact label -> one repair -> safe fallback."""
import asyncio
import time
from async_utils import cancellable_wait

LABELS = {"judge": ("continue", "switch"), "interrupt": ("continue", "switch"),
          "shift": ("no", "yes"), "input_route": ("keep", "stop_only", "yield_wait", "yield_ready")}
FALLBACK = {"judge": "continue", "interrupt": "continue", "shift": "no", "input_route": "keep"}


def parse_label(kind, raw):
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
            return label, audit
        request = messages + ([{"role": "assistant", "content": raw[:512]}]
                              if isinstance(raw, str) and raw.strip() else []) + [
            {"role": "user", "content": "The previous classification was missing or invalid. "
             "Classify the original audio under the original rules. Reply with exactly ONE label: "
             + " or ".join(LABELS[kind]) + ". No explanation or other text."}]
    audit["fallback"] = True
    return FALLBACK[kind], audit
