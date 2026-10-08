"""Capacity liveness, bounded read-ahead and causal clocks; no model required."""
import asyncio
import json
import sys
from contextlib import aclosing
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from speech_stream import SpeechPipeline, drain_text, MAX_RESPONSE_CHARS
from request_capacity import RequestCapacity, NORMAL
from stream_transport import PCMChunk
from demo_trace import DemoTrace


async def consume(pipeline, queue, events):
    while True:
        event = await queue.get()
        events.append(event)
        if event.kind == "audio":
            pipeline.progress(pipeline.sent)
        if event.delivered is not None and not event.delivered.done():
            event.delivered.set_result(None)
        if event.kind == "finished":
            return


async def test_overlapping_work_releases_response_slots_before_playback_backpressure():
    work_count = 3
    capacity = RequestCapacity(4, 3)
    started = 0
    all_first = asyncio.Event()
    pipelines, consumers, collected = [], [], []
    async def text(_):
        async with capacity.slot(NORMAL):
            for i in range(12):
                yield f"第{i}句。"
                await asyncio.sleep(0)
    async def tts(sentence):
        nonlocal started
        async with capacity.slot(NORMAL):
            started += 1
            if started >= work_count:
                all_first.set()
            yield PCMChunk(bytes(1920), 24000)
    try:
        for sid in range(work_count):
            queue, events = asyncio.Queue(), []
            pipeline = SpeechPipeline(sid, queue, [], text, tts, text_read_ahead=True)
            pipelines.append(pipeline)
            collected.append(events)
            consumers.append(asyncio.create_task(consume(pipeline, queue, events)))
        await asyncio.wait_for(all_first.wait(), 2)
        await asyncio.wait_for(asyncio.gather(*consumers), 3)
        await asyncio.gather(*(p.task for p in pipelines))
        assert all(sum(e.kind == "sentence" for e in events) == 12 for events in collected)
        assert all(not any(e.kind == "error" for e in events) for events in collected)
        assert capacity.snapshot()["active_total"] == 0
        assert capacity.snapshot()["peak_normal"] <= 3
        assert capacity.snapshot()["peak_total"] <= 4
    finally:
        for p in pipelines:
            p.cancel()
        for c in consumers:
            c.cancel()
        await asyncio.gather(*(p.task for p in pipelines), *consumers, return_exceptions=True)


async def test_text_reader_finishes_while_consumer_is_blocked_and_closes_on_cancel():
    closed = asyncio.Event()
    async def source():
        try:
            for i in range(100):
                yield str(i)
        finally:
            closed.set()
    async with aclosing(drain_text(source(), 1)) as output:
        assert await output.__anext__() == "0"
        await asyncio.wait_for(closed.wait(), 1)
        # No downstream reads are necessary to release the upstream stream.

    entered, closed = asyncio.Event(), asyncio.Event()
    async def slow():
        try:
            yield "first"
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()
    async with aclosing(drain_text(slow(), 1)) as output:
        assert await output.__anext__() == "first"
        await entered.wait()
    assert closed.is_set()
    assert not any(t.get_name() == "speech-text-reader" and not t.done()
                   for t in asyncio.all_tasks())


async def test_text_inference_clock_excludes_downstream_delivery_wait():
    closed, release = asyncio.Event(), asyncio.Event()
    async def text(_):
        try:
            for i in range(12):
                yield f"第{i}句。"
        finally:
            closed.set()
    async def tts(_):
        await release.wait()
        yield PCMChunk(bytes(1920), 24000)
    queue, events = asyncio.Queue(), []
    pipeline = SpeechPipeline(1, queue, [], text, tts, text_read_ahead=True)
    consumer = asyncio.create_task(consume(pipeline, queue, events))
    try:
        await asyncio.wait_for(closed.wait(), 1)
        # The true RPC has ended, while sentence delivery still awaits TTS.
        assert not any(e.kind == "text_done" for e in events)
        await asyncio.sleep(.06)
        release.set()
        await asyncio.wait_for(consumer, 2)
        await pipeline.task
        done = next(e.data for e in events if e.kind == "text_done")
        assert done['delivery_ms'] - done['infer'] * 1000 >= 40
    finally:
        pipeline.cancel()
        consumer.cancel()
        await asyncio.gather(pipeline.task, consumer, return_exceptions=True)


async def test_empty_deltas_cannot_hold_a_read_ahead_lease_forever(monkeypatch):
    import speech_stream
    monkeypatch.setattr(speech_stream, 'MAX_RESPONSE_CHARS', 4)
    closed = asyncio.Event()
    async def source():
        try:
            for _ in range(100):yield ''
        finally:closed.set()
    with pytest.raises(RuntimeError, match='streaming text limit'):
        async for _ in speech_stream.drain_text(source(), 1):pass
    assert closed.is_set()


async def test_reader_limit_and_errors_follow_received_text():
    async def source():
        yield "received"
        yield "x" * MAX_RESPONSE_CHARS
    async with aclosing(drain_text(source(), 1)) as output:
        assert await output.__anext__() == "received"
        with pytest.raises(RuntimeError, match="text limit"):
            await output.__anext__()
    async def failing():
        yield "partial"
        raise LookupError("upstream failure")
    async with aclosing(drain_text(failing(), 1)) as output:
        assert await output.__anext__() == "partial"
        with pytest.raises(LookupError):
            await output.__anext__()


async def test_causal_anchor_survives_epoch_change_and_newer_input(tmp_path):
    clock = [10.]
    trace = DemoTrace(tmp_path / "events.jsonl", clock=lambda: clock[0])
    def event(t, name, epoch=1, generation=0, **data):
        clock[0] = 10 + t
        return trace.observe(name, data, generation=generation, epoch=epoch)
    event(0, "vad_done", input_id=1)
    event(.01, "input_dispatch", input_id=1, parent_id="route-1")
    event(.2, "candidate_created", epoch=2, candidate_id=7, parent_id="route-1")
    event(.3, "vad_done", epoch=2, input_id=2)
    event(.64, "vad_640_done", epoch=2, candidate_id=7, parent_id="route-1")
    event(.8, "speech_start", epoch=2, candidate_id=7, parent_id="route-1", utterance_id=3)
    summary = event(1., "speech_first_audio", epoch=2, utterance_id=3)
    assert summary["vad_to_audio_ms"] == 1000
    assert summary["hold_ms"] == 640 and summary["decision_ms"] == 160
    event(2, "speech_start", epoch=2, candidate_id=99, utterance_id=4)
    assert event(2.1, "speech_first_audio", epoch=2, utterance_id=4)["vad_to_audio_ms"] is None
    event(3, "speech_start", epoch=2, candidate_id=7, utterance_id=5)
    assert event(3.1, "speech_first_audio", epoch=2, generation=1, utterance_id=5) is None
    for i in range(200):
        event(4+i/1000, "vad_done", input_id=i+10)
        event(4+i/1000, "input_dispatch", input_id=i+10, parent_id=f"r{i}")
        event(4+i/1000, "candidate_created", candidate_id=i+10, parent_id=f"r{i}")
    assert max(len(trace.inputs), len(trace.operations), len(trace.candidates)) <= 128
    await trace.close()


@pytest.mark.parametrize("enabled", [False, True])
async def test_preplay_input_has_reserved_capacity_when_enabled(enabled):
    from test_guarded_turns import guarded, new_input, cleanup
    from test_actor_candidate import pump
    e, models, _ = guarded()
    e.engine_cfg["input_control_reserve"] = enabled
    release = asyncio.Event()
    async def occupy():
        async with e.request_capacity.slot(NORMAL):
            await release.wait()
    slots = [asyncio.create_task(occupy()) for _ in range(3)]
    try:
        while e.request_capacity.snapshot()["active_normal"] < 3:
            await asyncio.sleep(0)
        await new_input(e)
        if enabled:
            await pump(e, lambda: e._guard_input.decided)
            assert any(kind == "input_route" for kind, _ in models.calls)
            assert e.request_capacity.snapshot()["peak_total"] == 4
        else:
            await asyncio.sleep(.02)
            assert not models.calls
            assert e.request_capacity.snapshot()["waiting_normal"] == 1
    finally:
        release.set()
        await asyncio.gather(*slots)
        await cleanup(e)


@pytest.mark.parametrize('label', ['no', 'yes'])
@pytest.mark.parametrize('prefetch', [0, 2000])
async def test_parallel_shift_prepares_privately_and_preserves_third_party_gate(label, prefetch):
    import struct
    import numpy as np
    from test_actor_candidate import actor, Models, pump, cleanup
    models = Models(shift=label, blocked='shift')
    e, _, sock = actor(models)
    e.engine_cfg.update(input_route_parallel_shift=True,stream_prefetch_ms=prefetch,
                        stream_text_read_ahead=True,stream_pcm_read_ahead=True)
    e.TURN_IDX = 1
    try:
        c = e._begin_candidate(np.ones(256,dtype=np.float32),stage='shift')
        await pump(e, lambda: c.pipeline is not None and c.pipeline.sent > 0)
        private_sid = c.pipeline.sid
        assert c.shift_pending and models.entered['shift'].is_set()
        await e._confirm_candidate()
        assert not c.published and not sock.audio
        assert not any(x['event']=='speech_start' for x in sock.events)
        assert not e._assistants_by_turn and not e.asr_calls
        models.gate.set()
        await pump(e, lambda: c.published and bool(sock.audio))
        assert not c.shift_pending
        if label=='yes':
            assert c.meta.kind=='shift_re' and c.pipeline.sid!=private_sid
            assert all(struct.unpack_from('<4sIII',p)[1]!=private_sid for p in sock.audio)
            assert not c.meta.add_to_history
        else:
            assert c.meta.kind=='response' and c.pipeline.sid==private_sid
        assert sum(x['event']=='speech_start' for x in sock.events)==1
    finally:
        await cleanup(e)


async def test_cancel_pending_parallel_shift_closes_both_requests_without_publication():
    import numpy as np
    from test_actor_candidate import actor, Models, pump, cleanup
    models=Models(blocked='shift')
    e,_,sock=actor(models)
    e.engine_cfg.update(input_route_parallel_shift=True,stream_text_read_ahead=True,
                        stream_pcm_read_ahead=True)
    e.TURN_IDX=1
    c=e._begin_candidate(np.ones(256,dtype=np.float32),stage='shift')
    await pump(e,lambda:c.pipeline is not None and c.pipeline.sent>0)
    await cleanup(e)
    assert 'shift' in models.closed and 'response' in models.closed and not sock.audio


@pytest.mark.parametrize('label', ['yes', 'no'])
async def test_pending_shift_settles_private_response_errors_after_its_decision(label):
    import numpy as np
    from test_actor_candidate import actor, Models, pump, cleanup
    models=Models(shift=label,blocked='shift',fail_first=True)
    e,_,sock=actor(models)
    e.engine_cfg.update(input_route_parallel_shift=True,stream_text_read_ahead=True)
    e.TURN_IDX=1
    try:
        c=e._begin_candidate(np.ones(256,dtype=np.float32),stage='shift')
        await pump(e,lambda:c.error is not None)
        await e._confirm_candidate()
        assert not c.published and not any(x['event']=='speech_error' for x in sock.events)
        models.gate.set()
        await pump(e,lambda:bool(sock.audio))
        assert c.meta.kind==('shift_re' if label=='yes' else 'response')
        assert not any(x['event']=='speech_error' for x in sock.events)
        assert '不能泄露的失败草稿' not in json.dumps(sock.events,ensure_ascii=False)
    finally:
        await cleanup(e)






@pytest.mark.parametrize("samples,expected", [(0,64), (16000,64), (48000,96), (160000,208), (320000,256)])
def test_short_route_budget_has_overhead_and_preserves_global_ceiling(samples, expected):
    from control_labels import input_route_token_budget
    assert input_route_token_budget(samples) == expected


def test_engineering_options_are_demo_only():
    import yaml
    from engine import ActorEngine
    from test_engine import ScriptedVAD
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / 'configs/demo_chat.yaml').read_text())
    assert cfg['engine']['stream_startup_ms'] == 350
    e = ActorEngine(engine_cfg={'chat_demo':False, 'stream_text_read_ahead':True},
        vad_iterator=ScriptedVAD({}), llm_fn=lambda _:"", asr_fn=lambda _:"", tts_fn=lambda *_:None)
    assert e._speech_stream_options() == {}


@pytest.mark.parametrize('prefetch', [0,600,2000])
async def test_cancel_credit_stalled_work_releases_all_streams_and_keeps_control_live(prefetch):
    from request_capacity import CONTROL
    capacity=RequestCapacity(4,3)
    entered=0
    barrier=asyncio.Event()
    text_closed,tts_closed=[],[]
    pipelines,consumers,events=[],[],[]
    async def text(messages):
        nonlocal entered
        sid=messages[0]
        try:
            async with capacity.slot(NORMAL):
                entered+=1
                if entered==3:barrier.set()
                await barrier.wait()
                for i in range(20):yield f'{i}。'
        finally:text_closed.append(sid)
    async def tts(sentence):
        try:
            async with capacity.slot(NORMAL):
                for _ in range(100):yield PCMChunk(bytes(1920),24000)
        finally:tts_closed.append(sentence)
    async def acknowledge(queue,record):
        while True:
            event=await queue.get();record.append(event)
            if event.delivered is not None and not event.delivered.done():event.delivered.set_result(None)
            if event.kind=='finished':return
    try:
        for sid in range(3):
            queue,record=asyncio.Queue(),[]
            p=SpeechPipeline(sid,queue,[sid],text,tts,buffer_ms=80,
                prefetch_ms=prefetch,text_read_ahead=True)
            pipelines.append(p);events.append(record)
            consumers.append(asyncio.create_task(acknowledge(queue,record)))
        async def ready():
            while len(text_closed)<3 or not all(p.sent>0 for p in pipelines):await asyncio.sleep(0)
        await asyncio.wait_for(ready(),1)
        assert capacity.snapshot()['active_normal']==3
        async with capacity.slot(CONTROL):assert capacity.snapshot()['active_total']==4
        assert all(p.sent<=1920 for p in pipelines)
        assert all(p.ahead is None or p.ahead.peak_bytes<=24000*2*prefetch/1000 for p in pipelines)
    finally:
        for p in pipelines:p.cancel()
        await asyncio.gather(*(p.task for p in pipelines),return_exceptions=True)
        await asyncio.wait_for(asyncio.gather(*consumers),1)
    assert len(tts_closed)>=3 and capacity.snapshot()['active_total']==0
    assert capacity.snapshot()['waiting_normal']==0


@pytest.mark.parametrize('text,first', [
    ('我是一个自然的语音助手，可以和你交流。后一句，保持原来分句。','我是一个自然的语音助手，'),
    ('Cats can rotate their ears independently, which helps them locate sounds.','Cats can rotate their ears independently,'),
    ('I\'m a helpful voice assistant, and I can answer questions.',"I'm a helpful voice assistant,"),
    ('这段话引用“红色，蓝色，绿色”，随后解释颜色。','这段话引用“红色，蓝色，绿色”，'),
    ('价格总共是1,000元，剩下的留作预算。','价格总共是1,000元，'),
    ('Dr. Li paid 1,000 dollars, then went home.','Dr. Li paid 1,000 dollars,'),
    ('参数（红色，蓝色）已经确定，可以开始。','参数（红色，蓝色）已经确定，'),
    ('Alice greeted the new visitor,"hello, Bob", then went home.','Alice greeted the new visitor,'),
    ("Read this quotation 'we're here, hello', then continue.","Read this quotation 'we're here, hello',"),
])
@pytest.mark.parametrize('width', [1,2,7,1000])
def test_first_clause_partition_protects_quotes_numbers_and_token_boundaries(text,first,width):
    from tts_sentence import StreamingSentenceBuffer
    splitter=StreamingSentenceBuffer(first_clause_chars=8)
    parts=[]
    for i in range(0,len(text),width):parts.extend(splitter.feed(text[i:i+width]))
    parts.extend(splitter.flush())
    assert parts[0]==first
    assert ''.join(parts)==text
    assert not any(not part for part in parts)


def test_new_comma_partition_is_disabled_for_legacy_and_does_not_split_urls():
    from tts_sentence import StreamingSentenceBuffer
    text='Cats can rotate their ears independently, which helps them locate sounds.'
    legacy=StreamingSentenceBuffer();parts=legacy.feed(text)+legacy.flush()
    assert parts==[text]
    url='The full address is https://example.com/path,another and it stays intact.'
    enabled=StreamingSentenceBuffer(first_clause_chars=8)
    assert enabled.feed(url)+enabled.flush()==[url]
    mixed='请保留链接https://example.com/path,another。'
    enabled=StreamingSentenceBuffer(first_clause_chars=8)
    assert enabled.feed(mixed)+enabled.flush()==[mixed]


async def test_early_tts_clause_is_not_a_completed_request_for_reply_review():
    from test_guarded_turns import guarded, playing, new_input, cleanup
    from test_actor_candidate import pump
    from speech_reference import completed_context
    e, models, _ = guarded()
    try:
        await playing(e)
        sid = e._speech.sid
        e._guard_outputs[sid] = {'turn': e._speech_meta.turn, 'sentences': []}
        e._guard_sentence_end(sid, {'end_sample': 100, 'text': '你想听这个故事，', 'reply_complete': False})
        record = e._guard_outputs[sid]
        assert completed_context(record['reply_sentences'], 100) == ''
        e._speech.played = 100
        models.labels['input_route'] = 'keep'
        await new_input(e)
        await pump(e, lambda: e._guard_input.decided)
        assert e._guard_input.reply_context == ''
        e._guard_sentence_end(sid, {'end_sample': 200, 'text': '还是换一个？', 'reply_complete': True})
        assert completed_context(record['reply_sentences'], 199) == ''
        assert completed_context(record['reply_sentences'], 200) == '你想听这个故事，还是换一个？'
        assert completed_context(record['sentences'], 100) == '你想听这个故事，'
    finally:
        await cleanup(e)


@pytest.mark.parametrize('prefetch', [0, 2000])
async def test_first_clause_marks_its_complete_reply_boundary_after_ack(prefetch):
    async def text(_):yield '你可以先听这个故事，或者选择另一个。'
    async def tts(_):yield PCMChunk(bytes(1920),24000)
    queue, events = asyncio.Queue(), []
    p = SpeechPipeline(1,queue,[],text,tts,first_clause_chars=8,
        track_sentences=True,prefetch_ms=prefetch)
    await asyncio.wait_for(consume(p,queue,events),2)
    await p.task
    ends=[e.data for e in events if e.kind=='sentence_end']
    assert [e['reply_complete'] for e in ends]==[False,True]
    assert ends[0]['end_sample']<ends[1]['end_sample']


@pytest.mark.parametrize('prefetch',[0,600,2000])
async def test_pcm_reader_releases_model_slots_before_late_playback_and_credit_stall(prefetch):
    capacity=RequestCapacity(4,3)
    closed=[]
    async def text(_):yield '一。'
    async def tts(_):
        async with capacity.slot(NORMAL):
            try:
                for _ in range(100):yield PCMChunk(bytes(1920),24000)
            finally:closed.append(True)
    pipelines,consumers=[],[]
    async def acknowledge(queue):
        while True:
            event=await queue.get()
            if event.delivered is not None and not event.delivered.done():event.delivered.set_result(None)
            if event.kind=='finished':return
    try:
        for sid in range(3):
            queue=asyncio.Queue()
            p=SpeechPipeline(sid,queue,[],text,tts,buffer_ms=80,prefetch_ms=prefetch,pcm_read_ahead=True)
            pipelines.append(p);consumers.append(asyncio.create_task(acknowledge(queue)))
        async def ready():
            while len(closed)<3 or not all(p.sent>0 for p in pipelines):await asyncio.sleep(0)
        await asyncio.wait_for(ready(),1)
        assert all(p.sent<=1920 for p in pipelines)
        assert capacity.snapshot()['active_normal']==0
        # New work must not depend on a stopped playback ACK.
        async with capacity.slot(NORMAL):assert capacity.snapshot()['active_normal']==1
    finally:
        for p in pipelines:p.cancel()
        await asyncio.gather(*(p.task for p in pipelines),return_exceptions=True)
        await asyncio.wait_for(asyncio.gather(*consumers),1)
    assert capacity.snapshot()['active_total']==0
    assert not any(t.get_name()=='speech-pcm-reader' and not t.done() for t in asyncio.all_tasks())


@pytest.mark.parametrize('kind',['bytes','chunks','rate','failure'])
async def test_pcm_reader_limits_and_errors_are_ordered_and_close_upstream(kind):
    from speech_stream import drain_pcm
    closed=asyncio.Event()
    async def source():
        try:
            yield PCMChunk(bytes(8),24000)
            if kind=='failure':raise LookupError('source failure')
            yield PCMChunk(bytes(8),16000 if kind=='rate' else 24000)
            yield PCMChunk(bytes(8),24000)
        finally:closed.set()
    kwargs={'max_bytes':12} if kind=='bytes' else {'max_chunks':1} if kind=='chunks' else {}
    async with aclosing(drain_pcm(source(),1,**kwargs)) as output:
        chunk=await output.__anext__();assert chunk.sample_rate==24000 and len(chunk.pcm)==8
        with pytest.raises((RuntimeError,LookupError)):await output.__anext__()
    assert closed.is_set()


async def test_preplay_stop_is_admitted_during_blocked_tts_and_releases_old_work():
    from test_guarded_turns import guarded, new_input, cleanup
    from test_actor_candidate import pump, frame
    import numpy as np

    e, models, sock = guarded(route="stop_only", blocked="tts")
    e.engine_cfg.update(input_control_reserve=True, stream_text_read_ahead=True,
                        stream_pcm_read_ahead=True)
    release = asyncio.Event()

    async def occupy():
        async with e.request_capacity.slot(NORMAL):
            await release.wait()

    holders = [asyncio.create_task(occupy()) for _ in range(2)]
    try:
        while e.request_capacity.snapshot()["active_normal"] < 2:
            await asyncio.sleep(0)
        old = e._begin_candidate(np.ones(256, dtype=np.float32), stage="response")
        await pump(e, lambda: old.pipeline is not None
                   and any(event.kind == "sentence" for event in old.stash))
        await asyncio.wait_for(models.entered["tts"].wait(), 1)
        assert e.request_capacity.snapshot()["active_normal"] == 3
        await new_input(e, .2)
        await pump(e, lambda: e._guard_input.decided)
        assert e._guard_input.route == "stop_only"
        assert e._candidate is None and e._speech is None and not sock.audio
        await asyncio.gather(old.pipeline.task, return_exceptions=True)
        assert "tts" in models.closed
        assert e.request_capacity.snapshot()["active_normal"] == 2
        await frame(e, 10.)
        assert e._candidate is None and models.responses == 1
    finally:
        release.set()
        await asyncio.gather(*holders)
        await cleanup(e)


async def test_session_release_fences_pending_shift_before_new_input_can_publish():
    from actor_candidate import CandidateResult
    from test_actor_candidate import actor, Models, pump, cleanup
    import numpy as np
    import struct

    models = Models(blocked="shift")
    e, _, sock = actor(models)
    e.engine_cfg.update(input_route_parallel_shift=True, stream_text_read_ahead=True,
                        stream_pcm_read_ahead=True)
    e.TURN_IDX = 1
    old = e._begin_candidate(np.ones(256, dtype=np.float32), stage="shift")
    await pump(e, lambda: old.pipeline is not None and old.pipeline.sent > 0)
    old_sid = old.pipeline.sid
    await cleanup(e)
    assert old.pipeline.task.done() and not sock.audio
    try:
        new = e._begin_candidate(np.ones(256, dtype=np.float32), stage="response", confirmed=True)
        await e._process_event(CandidateResult(old.cid, "shift", "yes", accounted=False))
        await pump(e, lambda: bool(sock.audio))
        assert e._speech_candidate is new and new.meta.kind == "response"
        assert all(struct.unpack_from("<4sIII", packet)[1] != old_sid for packet in sock.audio)
        assert not any(kind == "shift_re" for kind, _ in models.calls)
        assert sum(item["event"] == "speech_start" for item in sock.events) == 1
    finally:
        await cleanup(e)


@pytest.mark.parametrize("reader", ["text", "pcm"])
async def test_stalled_source_timeout_closes_upstream_before_terminal_error(reader):
    from speech_stream import drain_pcm
    closed = asyncio.Event()

    async def source():
        try:
            yield "first" if reader == "text" else PCMChunk(bytes(8), 24000)
            await asyncio.Event().wait()
        finally:
            closed.set()

    output = drain_text(source(), .02) if reader == "text" else drain_pcm(source(), .02)
    async with aclosing(output):
        await output.__anext__()
        with pytest.raises(asyncio.TimeoutError):
            await output.__anext__()
        assert closed.is_set()


async def test_cancelling_pcm_consumer_joins_the_live_source():
    from speech_stream import drain_pcm
    entered, closed = asyncio.Event(), asyncio.Event()

    async def source():
        try:
            yield PCMChunk(bytes(8), 24000)
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    async with aclosing(drain_pcm(source(), 1)) as output:
        await output.__anext__()
        await entered.wait()
    assert closed.is_set()
    assert not any(task.get_name() == "speech-pcm-reader" and not task.done()
                   for task in asyncio.all_tasks())


@pytest.mark.parametrize("samples", [-1, 1.5, True, None])
def test_route_budget_rejects_invalid_sample_counts(samples):
    from control_labels import input_route_token_budget
    with pytest.raises(ValueError):
        input_route_token_budget(samples)


@pytest.mark.parametrize("late_event", ["speech_first_audio", "speech_cancelled"])
async def test_old_generation_event_does_not_consume_new_reply_clock(tmp_path, late_event):
    clock = [10.]
    trace = DemoTrace(tmp_path / "events.jsonl", clock=lambda: clock[0])
    try:
        trace.observe("vad_done", {"input_id": 2}, generation=1, epoch=1)
        clock[0] = 10.5
        trace.observe("speech_start", {"utterance_id": 7, "input_id": 2}, generation=1, epoch=1)
        clock[0] = 10.6
        assert trace.observe(late_event, {"utterance_id": 7}, generation=0, epoch=1) is None
        clock[0] = 10.8
        summary = trace.observe("speech_first_audio", {"utterance_id": 7}, generation=1, epoch=1)
        assert summary["generation_ms"] == 300 and summary["vad_to_audio_ms"] == 800
        assert 7 not in trace.replies
    finally:
        await trace.close()


async def test_missing_dispatch_ids_do_not_create_a_causal_mapping(tmp_path):
    trace = DemoTrace(tmp_path / "events.jsonl", clock=lambda: 10.)
    try:
        trace.observe("vad_done", {"input_id": 1}, generation=0, epoch=1)
        trace.observe("input_dispatch", {"input_id": 1, "parent_id": None}, generation=0, epoch=1)
        trace.observe("input_dispatch", {"input_id": None, "parent_id": "missing"}, generation=0, epoch=1)
        assert not trace.operations
        trace.observe("candidate_created", {"candidate_id": 8, "parent_id": None}, generation=0, epoch=1)
        trace.observe("speech_start", {"candidate_id": 8, "utterance_id": 9}, generation=0, epoch=1)
        summary = trace.observe("speech_first_audio", {"utterance_id": 9}, generation=0, epoch=1)
        assert summary["vad_to_audio_ms"] is None
    finally:
        await trace.close()


@pytest.mark.parametrize("reader", ["text", "pcm"])
async def test_upstream_cancellation_follows_received_items_and_closes_source(reader):
    from speech_stream import drain_pcm
    closed = asyncio.Event()

    async def source():
        try:
            yield "first" if reader == "text" else PCMChunk(bytes(8), 24000)
            raise asyncio.CancelledError()
        finally:
            closed.set()

    output = drain_text(source(), 1) if reader == "text" else drain_pcm(source(), 1)
    async with aclosing(output):
        await asyncio.wait_for(output.__anext__(), .5)
        with pytest.raises(RuntimeError, match="cancelled"):
            await asyncio.wait_for(output.__anext__(), .5)
        assert closed.is_set()
    assert not any(task.get_name() in {"speech-text-reader", "speech-pcm-reader"}
                   and not task.done() for task in asyncio.all_tasks())


async def test_private_no_pcm_failure_retries_once_and_publishes_only_one_terminal_error():
    import numpy as np
    from test_actor_candidate import actor, Models, pump, cleanup
    models = Models()
    e, _, sock = actor(models)
    e.engine_cfg.update(stream_text_read_ahead=True, stream_pcm_read_ahead=True)

    async def failed_tts(_):
        raise RuntimeError("upstream unavailable")
        yield

    e.tts_stream_fn = failed_tts
    try:
        candidate = e._begin_candidate(np.ones(256, dtype=np.float32), stage="response")
        await pump(e, lambda: candidate.error is not None)
        assert not sock.audio and not candidate.published
        await e._confirm_candidate()
        await pump(e, lambda: any(item["event"] == "speech_error" for item in sock.events))
        assert models.responses == 2 and not sock.audio
        assert sum(item["event"] == "speech_error" for item in sock.events) == 1
    finally:
        await cleanup(e)


@pytest.mark.parametrize("cancelled_source", ["text", "pcm"])
async def test_upstream_self_cancellation_is_an_explicit_pipeline_failure(cancelled_source):
    capacity = RequestCapacity(4, 3)
    source_closed = asyncio.Event()

    async def text(_):
        async with capacity.slot(NORMAL):
            try:
                yield "第一句。第二句。"
                if cancelled_source == "text":
                    raise asyncio.CancelledError()
            finally:
                if cancelled_source == "text":
                    source_closed.set()

    async def tts(_):
        async with capacity.slot(NORMAL):
            try:
                yield PCMChunk(bytes(1920), 24000)
                if cancelled_source == "pcm":
                    raise asyncio.CancelledError()
            finally:
                if cancelled_source == "pcm":
                    source_closed.set()

    queue, events = asyncio.Queue(), []
    pipeline = SpeechPipeline(1, queue, [], text, tts,
                              text_read_ahead=True, pcm_read_ahead=True)
    try:
        await asyncio.wait_for(consume(pipeline, queue, events), 1)
        await pipeline.task
        assert source_closed.is_set()
        assert sum(event.kind == "error" for event in events) == 1
        assert events[-1].kind == "finished"
        assert not any(event.kind == "audio_end" for event in events)
        assert capacity.snapshot()["active_total"] == 0
    finally:
        pipeline.cancel()
        await asyncio.gather(pipeline.task, return_exceptions=True)
