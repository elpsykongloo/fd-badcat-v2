"""Cancellable SSE transport for the installed vLLM-Omni chat endpoint.

Audio SSE uses modality=audio + delta.content=base64(WAV chunk), not the
non-streaming message.audio.data dialect. Decode in memory to native-rate PCM16.
"""
import base64
import io
import json
from contextlib import aclosing
from dataclasses import dataclass

import aiohttp
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class PCMChunk:
    pcm: bytes
    sample_rate: int


async def sse_json(url, payload, timeout=60):
    """Parse fragmented SSE; cancellation closes HTTP; incomplete EOF fails."""
    limits = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=timeout)
    async with aiohttp.ClientSession(trust_env=False, timeout=limits) as client:
        async with client.post(url, json={**payload, "stream": True}) as response:
            response.raise_for_status()
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise RuntimeError("Upstream did not return SSE for stream=true")
            pending = bytearray()
            fields = []
            event_bytes = 0
            async for chunk in response.content.iter_any():
                pending.extend(chunk)
                if len(pending) > 16 * 1024 * 1024:
                    raise RuntimeError("Oversized SSE record")
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    line = line.rstrip(b"\r")
                    if line.startswith(b"data:"):
                        event_bytes += len(line)
                        if event_bytes > 16 * 1024 * 1024:
                            raise RuntimeError("Oversized SSE event")
                        fields.append(line[5:].lstrip(b" "))
                    elif not line and fields:
                        data = b"\n".join(fields).decode("utf-8")
                        fields.clear()
                        event_bytes = 0
                        if data == "[DONE]":
                            return
                        obj = json.loads(data)
                        if obj.get("error"):
                            raise RuntimeError(f"Upstream stream error: {obj['error']}")
                        yield obj
            raise RuntimeError("Truncated SSE: missing [DONE]")


async def text_stream(url, payload, timeout=60):
    async with aclosing(sse_json(url, payload, timeout)) as records:
        async for obj in records:
            if obj.get("modality", "text") != "text":
                continue
            for choice in obj.get("choices", []):
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if not isinstance(content, str):
                        raise RuntimeError("Unexpected text delta")
                    yield content


async def audio_stream(url, payload, timeout=60):
    count = 0
    async with aclosing(sse_json(url, payload, timeout)) as records:
        async for obj in records:
            if obj.get("modality") != "audio":
                continue
            for choice in obj.get("choices", []):
                data = (choice.get("delta") or {}).get("content")
                if not data:
                    continue
                raw = base64.b64decode(data, validate=True)
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
                if sr < 8000 or sr > 96000 or not np.isfinite(audio).all():
                    raise RuntimeError("Invalid streaming audio")
                pcm = (np.clip(audio.mean(axis=1), -1, 32767 / 32768) * 32768).round().astype("<i2")
                if pcm.size:
                    count += 1
                    yield PCMChunk(pcm.tobytes(), sr)
    if not count:
        raise RuntimeError("TTS stream produced no audio")
