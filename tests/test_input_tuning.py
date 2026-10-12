"""Input timing and route-only decoding contracts, without weights/network."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_guarded_turns import guarded, cleanup
from test_actor_candidate import frame, Models, pump
from guarded_turns import input_timing
from demo_cases import CaseArchive, load_case


@pytest.mark.parametrize("config", [
    {"input_preroll_ms": float("nan")}, {"input_preroll_ms": -16},
    {"input_preroll_ms": 321},
    {"input_preroll_ms": 976}, {"input_preroll_ms": True},
    {"input_vad_silence_ms": 95}, {"input_vad_silence_ms": 0}, {"input_vad_silence_ms": 512},
    {"input_vad_silence_ms": 96.0}, {"input_vad_silence_ms": True}, {"input_vad_silence_ms": "96"},
    {"input_vad_silence_ms_playing": 150}, {"input_vad_silence_ms_playing": 16},
])
def test_input_timing_rejects_invalid_values(config):
    with pytest.raises(ValueError): input_timing(config)


@pytest.mark.parametrize("ms,windows,effective", [
    (32, 1, 64), (64, 2, 96), (96, 3, 128), (100, 4, 160), (128, 4, 160), (480, 15, 512)])
def test_vad_end_windows_and_effective_trailing_silence(ms, windows, effective):
    timing = input_timing({"input_vad_silence_ms": ms, "input_vad_silence_ms_playing": ms})
    assert (timing["vad_silence_ms"], timing["vad_end_windows"], timing["vad_effective_end_ms"]) == (
        ms, windows, effective)
    assert (timing["vad_end_windows_playing"], timing["vad_effective_end_ms_playing"]) == (windows, effective)


def test_default_input_timing_is_the_evaluated_baseline():
    assert input_timing({}) == {"vad_threshold": .5, "vad_silence_ms": 100, "preroll_ms": 160,
        "vad_end_windows": 4, "vad_effective_end_ms": 160, "vad_silence_ms_playing": 100,
        "vad_end_windows_playing": 4, "vad_effective_end_ms_playing": 160}


@pytest.mark.parametrize("enabled", [True, False])
def test_real_vad_keeps_threshold_and_end_silence_while_preroll_is_demo_only(monkeypatch, enabled):
    import engine
    seen = []
    monkeypatch.setattr(engine, "VADIterator", lambda model, **kw: seen.append(kw))
    m = Models()
    e = engine.ActorEngine(prompts={"input_route": "route"},
        engine_cfg={"stream_response": True, "speculative_response": True,
            "guarded_turns": enabled, "input_preroll_ms": 320, "input_vad_silence_ms": 96},
        vad_model=object(), llm_fn=lambda _: "continue", asr_fn=lambda _: "",
        tts_fn=lambda _: None, text_stream_fn=m.text, tts_stream_fn=m.tts)
    # The guarded demo may tune the end silence; the non-guarded baseline stays at 100.
    assert seen == [{"sampling_rate": 16000, "threshold": .5,
                     "min_silence_duration_ms": 96 if enabled else 100}]
    assert e._guard_preroll.maxlen == (20 if enabled else 10)
    assert e.input_timing["vad_end_windows"] == (3 if enabled else 4)


class RecordingVAD:
    """Silero stand-in exposing min_silence_samples like utils_vad.VADIterator."""
    def __init__(self, ms=100):
        self._value = 16000 * ms / 1000
        self.writes = []
    @property
    def min_silence_samples(self):
        return self._value
    @min_silence_samples.setter
    def min_silence_samples(self, value):
        self.writes.append(value)
        self._value = value
    def __call__(self, *args, **kwargs):
        return None
    def reset_states(self):
        pass


async def test_vad_end_silence_follows_listening_preplay_playing_state():
    from test_guarded_turns import playing as start_playing
    e, _, sock = guarded()
    e.engine_cfg.update(input_vad_silence_ms=96, input_vad_silence_ms_playing=160)
    e._init_guarded()
    vad = e.vad_iterator = RecordingVAD()
    try:
        await frame(e, .008)
        assert e._guard_vad_mode == "listening" and vad.min_silence_samples == 1536
        await start_playing(e)  # END -> private candidate (preplay) -> published -> started
        done = next(x["data"] for x in sock.events if x["event"] == "vad_done")
        assert (done["vad_state"], done["vad_silence_ms"], done["vad_effective_end_ms"]) == ("listening", 96, 128)
        await frame(e, .72)
        assert e._guard_vad_mode == "playing" and vad.min_silence_samples == 2560
        assert vad.writes == [1536, 2560]
        e._reset_session()
        await frame(e, 1.)
        assert e._guard_vad_mode == "listening" and vad.min_silence_samples == 1536
    finally:
        await cleanup(e)


async def test_vad_preplay_uses_listening_value_until_playback_starts():
    e, m, _ = guarded()
    e.engine_cfg.update(input_vad_silence_ms=64, input_vad_silence_ms_playing=128)
    e._init_guarded()
    vad = e.vad_iterator = RecordingVAD()
    try:
        await frame(e, .016, {"start": 0}, .1)
        await frame(e, .032, {"end": .032}, .2)
        await pump(e, lambda: e._candidate is not None)
        await frame(e, .1)
        assert e._guard_vad_mode == "preplay" and vad.min_silence_samples == 1024
        await frame(e, .704)  # hold expires: published, still waiting for the playback ACK
        await pump(e, lambda: e._speech is not None)
        await frame(e, .72)
        assert e._guard_vad_mode == "preplay" and vad.min_silence_samples == 1024
    finally:
        await cleanup(e)


async def test_default_vad_silence_never_writes_the_iterator():
    from test_guarded_turns import playing as start_playing
    e, _, _ = guarded()
    vad = e.vad_iterator = RecordingVAD()
    try:
        await start_playing(e)
        await frame(e, .72)
        assert e._guard_vad_mode == "playing" and vad.writes == []
    finally:
        await cleanup(e)


async def test_preroll_retains_actual_pretrigger_samples_and_resets_without_waiting():
    e, _, _ = guarded()
    e.engine_cfg["input_preroll_ms"] = 320
    e._init_guarded()
    try:
        for i in range(25): await frame(e, (i+1)*.016, value=(i+1)/100)
        assert e._guard_input is None
        await frame(e, .416, {"start": .416}, value=.8)
        span = e._guard_input
        assert span.start == .416 and span.preroll_samples == 5120
        assert len(span.frames) == 21
        np.testing.assert_array_equal(span.frames[0], np.full(256,.06,dtype=np.float32))
        np.testing.assert_array_equal(span.frames[-1], np.full(256,.8,dtype=np.float32))
        assert e.END_HOLD == .64 and e._inflight == 0
        e._reset_session()
        assert not e._guard_preroll and e._guard_preroll.maxlen == 20
    finally:
        await cleanup(e)


async def test_route_wire_and_archive_use_same_zero_penalties_without_changing_chat(tmp_path, monkeypatch):
    import module, stream_transport
    monkeypatch.setenv("FDBC_QWEN_PRESENCE_PENALTY", "1.2")
    monkeypatch.setenv("FDBC_QWEN_FREQUENCY_PENALTY", "0.8")
    requests=[]
    async def stream(url, payload, timeout, **options):
        assert options["transport_context"]["request_id"].startswith("call-")
        assert options["report_finish"] is False
        requests.append(payload)
        yield '{"transcript":"停","label":"stop_only"}'
    monkeypatch.setattr(stream_transport, "text_stream", stream)
    e, _, _ = guarded()
    e.demo_cases = CaseArchive(tmp_path)
    e.demo_session_id = "synthetic-input-tuning"
    messages=[{"role":"system","content":e.prompts["input_route"]}]
    try:
        for route in [True, False]:
            assert [p async for p in e._capacity_stream("interrupt" if route else "response",
                module.llm_qwen3o_stream, messages, route=route)]
        await e.demo_cases.close()
        cases=[load_case(p) for p in (tmp_path/'captures').iterdir()]
        assert len(cases)==2
        wire=sorted(requests,key=lambda p:p['presence_penalty'])
        saved=sorted((c['request'] for c in cases),key=lambda p:p['presence_penalty'])
        assert saved==[{**p,"stream":True} for p in wire]
        assert [(p['presence_penalty'],p['frequency_penalty']) for p in wire]==[(0.,0.),(1.2,.8)]
        assert {k:v for k,v in wire[0].items() if k not in ['presence_penalty','frequency_penalty']} == {
            k:v for k,v in wire[1].items() if k not in ['presence_penalty','frequency_penalty']}
    finally:
        await cleanup(e)


def test_synchronous_warmup_adapter_uses_same_route_payload(monkeypatch):
    import module
    posted=[]
    class HTTP:
        def post(self, url, **kwargs):
            posted.append(json.loads(kwargs['data']))
            return SimpleNamespace(raise_for_status=lambda:None,
                json=lambda:{"choices":[{"message":{"content":"ok"}}]})
    monkeypatch.setattr(module,"_http",lambda:HTTP())
    module.llm_qwen3o_strict([],route=True)
    assert posted==[module.qwen_text_payload([],route=True)]


async def test_saved_replay_is_unchanged_and_current_route_replay_opts_in(tmp_path, monkeypatch):
    import stream_transport
    from test_demo_cases import cli, saved
    from demo_cases import restore_request
    _, path = await saved(tmp_path)
    original = restore_request(path)
    requests = []
    async def stream(url, payload, timeout):
        requests.append(payload)
        yield '{"transcript":"","label":"keep"}'
    monkeypatch.setattr(stream_transport, "text_stream", stream)
    for current in [False, True]:
        args = SimpleNamespace(root=tmp_path, current_prompt=current, timeout=2, url="unused")
        assert await cli.replay(args, [(path, load_case(path))]) == 0
    assert requests[0] == original == restore_request(path)
    assert requests[1]["presence_penalty"] == requests[1]["frequency_penalty"] == 0
    assert requests[1]["seed"] == original["seed"]
    assert requests[1]["model"] == original["model"]
    assert requests[1]["messages"][1]["content"][-1] == original["messages"][1]["content"][-1]
