"""Bounded ahead-of-play synthesis, publication, cancellation and config isolation."""
import asyncio
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speech_stream import MAX_PREFETCH_CHUNK_BYTES, PCM_HEADER, SpeechPipeline
from stream_transport import PCMChunk


async def drain(pipeline, queue, events, credit):
    while True:
        event = await queue.get()
        events.append(event)
        if event.kind == "audio" and credit.is_set():
            pipeline.progress(pipeline.sent)
        if event.delivered is not None and not event.delivered.done():
            event.delivered.set_result(None)
        if event.kind == "finished":
            return


async def test_one_sentence_ahead_is_bounded_and_not_published_early():
    calls, events = [], []
    second_started, credit = asyncio.Event(), asyncio.Event()
    async def text(_):
        yield "第一句。第二句。第三句。"
    async def tts(sentence):
        calls.append(sentence)
        if len(calls) == 2:
            second_started.set()
        yield PCMChunk(b"\1\0" * 24000, 24000)
    queue = asyncio.Queue()
    pipeline = SpeechPipeline(1, queue, [], text, tts, buffer_ms=80,
                              prefetch_ms=2000, diagnostics=True, track_sentences=True)
    consumer = asyncio.create_task(drain(pipeline, queue, events, credit))
    await asyncio.wait_for(second_started.wait(), 1)
    await asyncio.sleep(.01)
    assert calls == ["第一句。", "第二句。"]
    assert [e.data['text'] for e in events if e.kind == 'sentence'] == ["第一句。"]
    assert pipeline.sent <= 1920
    assert pipeline.ahead.peak_bytes <= 24000 * 2 * 2
    credit.set()
    pipeline.progress(pipeline.sent)
    await asyncio.wait_for(consumer, 2)
    await pipeline.task
    assert calls == ["第一句。", "第二句。", "第三句。"]
    assert [e.data['start_sample'] for e in events if e.kind == 'sentence'] == [0, 24000, 48000]
    assert [e.data['end_sample'] for e in events if e.kind == 'sentence_end'] == [24000, 48000, 72000]
    audio = [e.data['wire'] for e in events if e.kind == 'audio']
    assert [PCM_HEADER.unpack(x[:16])[2] for x in audio] == list(range(len(audio)))
    assert b''.join(x[16:] for x in audio) == b"\1\0" * 72000
    assert sum(e.kind == 'audio_end' for e in events) == 1
    assert not pipeline.ahead.items and pipeline.ahead.bytes == 0


async def test_unconfirmed_candidate_cannot_synthesize_next_sentence():
    gate, credit = asyncio.Event(), asyncio.Event()
    credit.set()
    calls, events = [], []
    async def text(_):
        yield "一。二。"
    async def tts(sentence):
        calls.append(sentence)
        yield PCMChunk(bytes(1920), 24000)
    queue = asyncio.Queue()
    pipeline = SpeechPipeline(1, queue, [], text, tts, prefetch_ms=600, precompute_gate=gate)
    consumer = asyncio.create_task(drain(pipeline, queue, events, credit))
    while not any(e.kind == 'audio' for e in events):
        await asyncio.sleep(.001)
    assert calls == ["一。"]
    gate.set()
    await asyncio.wait_for(consumer, 1)
    await pipeline.task
    assert calls == ["一。", "二。"]


async def test_cancel_closes_source_and_discards_full_prefetch():
    closed, credit = asyncio.Event(), asyncio.Event()
    events = []
    async def text(_):
        yield "必须取消。"
    async def tts(sentence):
        try:
            for _ in range(20):
                yield PCMChunk(bytes(48000), 24000)
        finally:
            closed.set()
    queue = asyncio.Queue()
    pipeline = SpeechPipeline(1, queue, [], text, tts, buffer_ms=80, prefetch_ms=200)
    consumer = asyncio.create_task(drain(pipeline, queue, events, credit))
    while pipeline.ahead.peak_bytes < 9600:
        await asyncio.sleep(.001)
    pipeline.cancel()
    await asyncio.gather(pipeline.task, return_exceptions=True)
    await asyncio.wait_for(consumer, 1)
    assert closed.is_set()
    assert not pipeline.ahead.items and pipeline.ahead.bytes == 0
    assert not any(e.kind == 'audio_end' for e in events)


@pytest.mark.parametrize('failure', ['size', 'rate', 'source'])
async def test_prefetch_errors_cancel_without_fake_audio_end(failure):
    async def text(_):
        yield "检查错误。"
    async def tts(_):
        if failure == 'size':
            yield PCMChunk(bytes(MAX_PREFETCH_CHUNK_BYTES + 2), 24000)
        elif failure == 'rate':
            yield PCMChunk(bytes(1920), 24000)
            yield PCMChunk(bytes(1920), 16000)
        else:
            raise RuntimeError('source failed')
            yield
    events, queue, credit = [], asyncio.Queue(), asyncio.Event()
    credit.set()
    pipeline = SpeechPipeline(1, queue, [], text, tts, prefetch_ms=200)
    await asyncio.wait_for(drain(pipeline, queue, events, credit), 1)
    await pipeline.task
    assert sum(e.kind == 'error' for e in events) == 1
    assert not any(e.kind == 'audio_end' for e in events)


def test_demo_config_changes_only_chunk_fields(tmp_path):
    spec = importlib.util.spec_from_file_location('demo_stream_config', ROOT / 'scripts/demo_stream_config.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source = ROOT / 'configs/qwen3_omni_audio_single_gpu.yaml'
    original = source.read_text()
    target = tmp_path / 'demo.yaml'
    for profile in ('4:25', '8:25', '4:8', '4:12'):
        helper.write_config(source, target, profile)
        config, expected = yaml.safe_load(target.read_text()), yaml.safe_load(original)
        initial, steady = map(int, profile.split(':'))
        expected['connectors']['connector_of_shared_memory']['extra'].update(
            initial_codec_chunk_frames=initial, codec_chunk_frames=steady)
        assert config == expected
    assert source.read_text() == original
    for profile in ('8', '0:8', '4:0', '4:26', '4:8:8'):
        with pytest.raises(ValueError):
            helper.write_config(source, target, profile)
    with pytest.raises(ValueError, match='overwrite'):
        helper.write_config(source, source, '4:8')
