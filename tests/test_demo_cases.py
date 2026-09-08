"""Private actual-call capture/review/replay contracts; no GPU or network."""
import asyncio
import importlib.util
import json
from pathlib import Path
import queue
import threading
import wave

import numpy as np
import pytest

from test_actor_candidate import actor
from demo_cases import CaseArchive, MAX_OUTPUT, pack_audio, restore_request, load_case
from messages import build_audio_content
from stream_transport import PCMChunk

spec = importlib.util.spec_from_file_location("case_cli", Path(__file__).resolve().parents[1] / "scripts/demo_cases.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def request(style="audio_url"):
    return {"messages": [{"role": "system", "content": "classify"},
        {"role": "user", "content": [{"type": "text", "text": "assistant reference"},
        build_audio_content(np.array([.1, -.2, .3], dtype=np.float32), 16000, style)]}],
        "seed": 42, "stream": True, "model": "test-only"}


async def saved(tmp_path, kind="input_route", style="audio_url", **kwargs):
    writer = CaseArchive(tmp_path, **kwargs)
    call = writer.begin(kind, request(style), {"session_id": "synthetic", "turn": 3})
    assert call
    call.feed("keep")
    assert call.finish("completed")
    await writer.close()
    return writer, tmp_path / "captures" / call.case["case_id"]


@pytest.mark.parametrize("style", ["audio_url", "input_audio", "input_audio_datauri"])
async def test_exact_input_context_sampling_roundtrip_and_private_permissions(tmp_path, style):
    writer, path = await saved(tmp_path, style=style)
    assert writer.saved == 1 and not writer.dropped and not writer.error
    assert restore_request(path) == request(style)
    case = load_case(path)
    assert case["expected"] is None and case["outcome"]["text"] == "keep"
    assert case["context"] == {"session_id": "synthetic", "turn": 3}
    assert (path.stat().st_mode & 0o777) == 0o700
    assert all((p.stat().st_mode & 0o777) == 0o600 for p in path.iterdir())
    with wave.open(str(path / "input-00.wav")) as wav:
        assert wav.getnframes() == 3 and wav.getframerate() == 16000


async def test_output_prefix_is_explicitly_bounded_and_partial(tmp_path):
    writer = CaseArchive(tmp_path)
    call = writer.begin("tts", {"messages": []}, {}, "你好。")
    call.feed(PCMChunk(b"\0\0" * 20, 24000))
    call.feed(PCMChunk(b"\0" * MAX_OUTPUT, 24000))
    call.feed(PCMChunk(b"\0\0", 24000))
    assert call.finish("cancelled")
    await writer.close()
    path = tmp_path / "captures" / call.case["case_id"]
    case = load_case(path)
    assert case["outcome"]["output_truncated"] is True
    assert case["outcome"]["status"] == "cancelled"
    assert case["outcome"]["saved_samples"] == 20
    with wave.open(str(path / "output.wav")) as wav:
        assert wav.getnframes() == 20


@pytest.mark.parametrize("mode", ["completed", "cancelled", "error"])
async def test_real_engine_wrapper_preserves_output_error_cancel_and_capacity(tmp_path, monkeypatch, mode):
    import module
    e, _, _ = actor()
    e.demo_cases = CaseArchive(tmp_path)
    e.demo_session_id = "synthetic"
    async def fake(messages):
        yield "keep"
        if mode == "error":
            raise RuntimeError("never archive exception text or secrets")
    monkeypatch.setattr(module, "llm_qwen3o_stream", fake)
    stream = e._capacity_stream("judge", fake, request()["messages"], case_context={"input_id": 9})
    assert await anext(stream) == "keep"
    if mode == "cancelled":
        await stream.aclose()
    elif mode == "error":
        with pytest.raises(RuntimeError):
            await anext(stream)
    else:
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    assert e.request_capacity.snapshot()["active_total"] == 0
    await e.demo_cases.close()
    case = load_case(next((tmp_path / "captures").iterdir()))
    assert case["outcome"]["status"] == mode
    assert case["context"]["input_id"] == 9
    assert case["request"]["stream"] is True
    assert "never archive" not in json.dumps(case)


async def test_disk_failure_does_not_change_stream_results(tmp_path, monkeypatch):
    import module
    root = tmp_path / "not-a-directory"
    root.touch()
    e, _, _ = actor()
    e.demo_cases = CaseArchive(root)
    await asyncio.to_thread(e.demo_cases.thread.join, 1)
    async def fake(messages):
        yield "keep"
    monkeypatch.setattr(module, "llm_qwen3o_stream", fake)
    assert [p async for p in e._capacity_stream("judge", fake, request()["messages"])] == ["keep"]
    assert e.demo_cases.error and e.demo_cases.dropped == 1
    await e.demo_cases.close()


async def test_quota_preserves_previous_case_and_does_not_silently_delete(tmp_path):
    writer, path = await saved(tmp_path)
    original = (path / "case.json").read_bytes()
    second = CaseArchive(tmp_path, max_cases=1)
    call = second.begin("input_route", request(), {})
    if call:
        call.finish("completed")
    await second.close()
    assert second.saved == 0 and second.dropped == 1
    assert second.error == "quota_exceeded"
    assert (path / "case.json").read_bytes() == original


def test_queue_overflow_is_nonblocking_and_counted():
    writer = CaseArchive.__new__(CaseArchive)
    writer.queue = queue.Queue(maxsize=1)
    writer.closed = threading.Event()
    writer.error, writer.dropped = None, 0
    assert writer.submit({}, b"")
    assert not writer.submit({}, b"")
    assert writer.dropped == 1
    writer.closed.set()
    assert not writer.submit({}, b"")


async def test_manual_promotion_filters_and_no_overwrite(tmp_path):
    _, path = await saved(tmp_path)
    assert cli.cases(tmp_path, reviewed=True) == []
    assert len(cli.cases(tmp_path, kind="input_route", session="synthetic", status="completed")) == 1
    assert cli.cases(tmp_path, session="other") == []
    with pytest.raises(ValueError):
        cli.label(path, "switch", "wrong label space")
    with pytest.raises(ValueError):
        cli.label(path, "keep", " ")
    cli.label(path, "yield_ready", "Human listened: a clear request; observed keep was incorrect")
    with pytest.raises(FileExistsError):
        cli.label(path, "keep", "cannot overwrite")
    assert len(cli.cases(tmp_path, reviewed=True)) == 1
    assert load_case(path)["expected"] is None


async def test_replay_reconstructs_saved_payload_and_fails_wrong_gold(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import stream_transport
    _, path = await saved(tmp_path)
    cli.label(path, "yield_ready", "self-authored wrong-label regression")
    got = []
    async def fake(url, payload, timeout):
        got.append(payload)
        yield "keep"
    monkeypatch.setattr(stream_transport, "text_stream", fake)
    args = SimpleNamespace(root=tmp_path, current_prompt=False, timeout=1, url="unused")
    assert await cli.replay(args, cli.cases(tmp_path, reviewed=True)) == 1
    assert got == [request()]
    with pytest.raises(ValueError, match="empty suites"):
        await cli.replay(args, [])


async def test_replay_rejects_escape_and_remote_audio(tmp_path):
    _, path = await saved(tmp_path)
    case = load_case(path)
    case["request"]["messages"][1]["content"][1]["audio_url"]["url"]["case_file"] = "../input-secret.wav"
    with pytest.raises(ValueError):
        restore_request(path, case)
    with pytest.raises(ValueError):
        cli.case_path(tmp_path, "../escape")
    bad = request()
    bad["messages"][1]["content"][1]["audio_url"]["url"] = "http://private-network/secret"
    with pytest.raises(ValueError):
        pack_audio(bad)


@pytest.mark.parametrize("current", [False, True])
async def test_route_replay_versions_keep_saved_baseline_and_strip_current_reference(tmp_path, monkeypatch, current):
    from types import SimpleNamespace
    import stream_transport
    _, path = await saved(tmp_path)
    cli.label(path, "keep", "Synthetic no-action replay contract")
    got = []
    async def fake(url, payload, timeout):
        got.append(payload)
        yield '{"transcript":"","label":"keep"}' if current else "keep"
    monkeypatch.setattr(stream_transport, "text_stream", fake)
    args = SimpleNamespace(root=tmp_path, current_prompt=current, timeout=1, url="unused")
    assert await cli.replay(args, cli.cases(tmp_path, reviewed=True)) == 0
    if current:
        user = got[0]["messages"][1]["content"]
        assert "assistant reference" not in user[0]["text"]
        assert "assistant_reference_text" not in user[0]["text"]
        assert user[1] == request()["messages"][1]["content"][1]
        assert got[0]["seed"] == 42
    else:
        assert got == [request()]
