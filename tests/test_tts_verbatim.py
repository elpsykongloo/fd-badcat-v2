"""Verbatim synthesis: model-free boundary tests, never accept free-form fallback."""
import asyncio
import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import module
import stream_transport
from demo_startup import verify_spoken_text


def test_payload_constrains_literal_not_regex_and_preserves_legacy_request(monkeypatch):
    monkeypatch.setenv("FDBC_OMNI_TTS_MAX_TOKENS", "1")
    text = '“你好吗？” Dr. Li paid $3.14 (not $5). [a-z]*'
    legacy = module.omni_tts_payload(text)
    payload = module.verbatim_tts_payload(text)
    assert payload["structured_outputs"] == {"grammar": "root ::= " + json.dumps(text, ensure_ascii=False)}
    assert payload["modalities"] == ["text", "audio"]
    assert payload["max_tokens"] == len(text.encode("utf-8")) + 8
    assert payload["messages"] == legacy["messages"]
    assert "structured_outputs" not in legacy and legacy["max_tokens"] == 1


@pytest.mark.parametrize("text", [
    "那我给你讲一个奇幻小故事吧：\n", "Hello.\r\n\tNext.",
    'He said "Hi". Path: C:\\notes. [a-z]* $3.14.',
    "中😀\u2028\u2029", "\n\r\tHello.\n\r\t",
])
def test_literal_escapes_controls_without_changing_text(text):
    payload = module.verbatim_tts_payload(text)
    literal = payload["structured_outputs"]["grammar"].removeprefix("root ::= ")
    assert json.loads(literal) == text
    assert not any(chr(i) in literal for i in range(32))
    assert payload["messages"][-1]["content"] == text
    assert "choice" not in payload["structured_outputs"]


def test_production_core_pins_one_grammar_backend():
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/qwen3_omni_audio_single_gpu.yaml").read_text())
    assert config["stages"][0]["structured_outputs_config"] == {"backend": "xgrammar"}


@pytest.mark.parametrize("text", [None, "", "  ", "中" * 342, "<|im_end|>", "x\0y", "x\by", "x\fy", "x\x7fy"])
def test_unsafe_or_unbounded_literal_rejected(text):
    with pytest.raises(ValueError): module.verbatim_tts_payload(text)


def text_event(text, reason=None):
    result = {"modality": "text", "choices": [{"index": 0, "delta": {"content": text}}]}
    if reason is not None: result["fd_text_finish_reason"] = reason
    return result


def audio_event(samples=240):
    wav = io.BytesIO()
    sf.write(wav, np.zeros(samples), 24000, format="WAV", subtype="PCM_16")
    return {"modality": "audio", "choices": [{"index": 0, "delta": {
        "content": base64.b64encode(wav.getvalue()).decode()}}]}


async def test_audio_quarantined_until_full_literal_and_successful_text_end(monkeypatch):
    verified = False
    async def records(*args):
        nonlocal verified
        yield audio_event()
        yield text_event("你好吗？")
        yield audio_event()
        verified = True
        yield text_event("", "stop")
        yield audio_event()
    monkeypatch.setattr(stream_transport, "sse_json", records)
    chunks = []
    async for chunk in stream_transport.audio_stream("unused", {}, expected_text="你好吗？"):
        assert verified
        chunks.append(chunk)
    assert len(chunks) == 3 and sum(len(c.pcm) for c in chunks) == 1440


@pytest.mark.parametrize("events", [
    [audio_event(), text_event("我很好。", "stop")],
    [audio_event(), text_event("你好吗？")],  # ignored contract / no completion proof
    [audio_event(), text_event("你好吗？", "length")],
    [audio_event(), text_event("你好", "stop")],
    [audio_event()],
    [audio_event(300000)],  # bounded pre-verification quarantine
])
async def test_wrong_missing_truncated_or_unproven_text_never_leaks_audio(monkeypatch, events):
    closed = False
    async def records(*args):
        nonlocal closed
        try:
            for event in events: yield event
        finally: closed = True
    monkeypatch.setattr(stream_transport, "sse_json", records)
    chunks = []
    with pytest.raises(RuntimeError):
        async for chunk in stream_transport.audio_stream("unused", {}, expected_text="你好吗？"):
            chunks.append(chunk)
    assert not chunks and closed


async def test_cancel_during_verification_closes_upstream(monkeypatch):
    entered, closed = asyncio.Event(), asyncio.Event()
    async def records(*args):
        try:
            yield audio_event()
            entered.set()
            await asyncio.Event().wait()
        finally: closed.set()
    monkeypatch.setattr(stream_transport, "sse_json", records)
    async def collect():
        return [c async for c in stream_transport.audio_stream("unused", {}, expected_text="你好吗？")]
    task = asyncio.create_task(collect())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert closed.is_set()


def test_startup_readback_does_not_accept_a_new_answer():
    verify_spoken_text("你好。Hello, how are you?", "你好，hello ,how are you.")
    with pytest.raises(RuntimeError, match="readback mismatch"):
        verify_spoken_text("你那边怎么样？", "我这边一切都好，谢谢关心。你呢？")


def test_server_adapter_is_idempotent_and_refuses_unknown_layout():
    path = Path(__file__).resolve().parents[1] / "scripts/patch_omni_verbatim_tts.py"
    spec = importlib.util.spec_from_file_location("omni_patch", path)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    source = ("def overrides():\n    if True:\n" + patch.OVERRIDE_ANCHOR + "        pass\n"
              + "def stream():\n    if True:\n        if True:\n            if True:\n"
              + "                if final_output_type == 'text':\n                    if True:\n"
              + patch.STREAM_ANCHOR + "                    pass\n")
    updated = patch.patched_source(source)
    assert patch.patched_source(updated) == updated
    assert "params.structured_outputs = deepcopy(request.structured_outputs)" in updated
    assert 'proof["fd_text_finish_reason"] = output.finish_reason' in updated
    with pytest.raises(RuntimeError): patch.patched_source("unexpected source")
