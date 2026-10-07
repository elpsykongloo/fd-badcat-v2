"""Bounded, content-free HTTP failure evidence shared by demo and proxy.

Never serialize exception strings, arbitrary response bodies, headers, URL
credentials or query strings: those can contain conversation data or secrets.
"""
import asyncio
import re
from urllib.parse import urlsplit

import aiohttp

VERSION = "model-transport-v1"


class TruncatedStreamError(RuntimeError):
    """Transport ended without its SSE terminator, not a text-proof failure."""


def safe_request_id(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else None


def safe_endpoint(url):
    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if ":" in host:
            host = "[" + host + "]"
        return f"{parsed.scheme}://{host}" + (f":{parsed.port}" if parsed.port else "") + parsed.path[:160]
    except (TypeError, ValueError):
        return "invalid-endpoint"


def failure_details(exc, state=None):
    result = {"version": VERSION, **(state or {})}
    result.update(error_type=type(exc).__name__)
    if isinstance(exc, aiohttp.ClientResponseError):
        result["http_status"] = exc.status
    cause = getattr(exc, "os_error", None) or exc.__cause__
    if cause is not None:
        result["cause_type"] = type(cause).__name__
    errno = getattr(cause or exc, "errno", None)
    if type(errno) is int:
        result["errno"] = errno
    return result


def retryable_transport_error(exc):
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in {502, 503, 504}
    if isinstance(exc, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError,
                        asyncio.TimeoutError, TruncatedStreamError)):
        return True
    return isinstance(exc, RecordedTransportError) and exc.retryable


class RecordedTransportError(RuntimeError):
    """Replay the recorded failure category without contacting an upstream."""
    def __init__(self, evidence, retryable=False):
        super().__init__("Recorded model transport failure")
        self.transport_failure = dict(evidence or {})
        self.retryable = bool(retryable)


class SpeechSynthesisError(RuntimeError):
    code = "tts_unavailable"
    terminal = True

    def __init__(self, operation_id, attempts, output_samples, cause):
        super().__init__("Speech synthesis failed; inspect the correlated model calls")
        self.operation_id = operation_id
        self.attempts = attempts
        self.output_samples = output_samples
        self.transport_failure = getattr(cause, "transport_failure", None)
