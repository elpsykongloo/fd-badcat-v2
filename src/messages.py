# -*- coding: utf-8 -*-
"""Shared LLM message helpers: audio content-block formats + log scrubbing.

Two wire formats exist for putting audio into an OpenAI-compatible chat payload:

- ``audio_url``          : vLLM dialect, data-URI inside ``audio_url.url``  (local default)
- ``input_audio``        : OpenAI-strict, BARE base64 inside ``input_audio.data``
- ``input_audio_datauri``: DashScope dialect, data-URI inside ``input_audio.data``
                           (matches cloud origin/main `f4ff2b1`)

The scrubber strips ALL formats unconditionally, so switching formats can never
leak base64 audio into `llm_done` control traces.
"""
import base64
import copy
import io

import soundfile as sf

AUDIO_BLOCK_STYLES = ("audio_url", "input_audio", "input_audio_datauri")


def encode_wav_base64(user_audio, sample_rate):
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, user_audio, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    return base64.b64encode(wav_buffer.read()).decode("utf-8")


def build_audio_content(user_audio, sample_rate, style="audio_url"):
    audio_base64 = encode_wav_base64(user_audio, sample_rate)
    if style == "audio_url":
        return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
    if style == "input_audio":
        return {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}
    if style == "input_audio_datauri":
        return {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{audio_base64}", "format": "wav"}}
    raise ValueError(f"Unsupported llm.audio_block={style!r}; expected one of {AUDIO_BLOCK_STYLES}")


def scrub_audio_blocks(messages):
    """Deep-copied messages with audio payloads replaced by a placeholder (for traces)."""
    messages_clean = copy.deepcopy(messages)
    for msg in messages_clean:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "audio_url":
                    block["audio_url"]["url"] = "<AUDIO_BASE64_OMITTED>"
                elif block.get("type") == "input_audio":
                    block["input_audio"]["data"] = "<AUDIO_BASE64_OMITTED>"
    return messages_clean
