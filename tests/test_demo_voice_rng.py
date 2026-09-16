"""Opt-in boundary and actual installed runner's request/row RNG isolation."""
import ast
import asyncio
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import module
import omni_demo_voice as helper
import patch_omni_demo_voice as patcher
import stream_transport

OMNI = Path("/root/autodl-tmp/conda-envs/fdbc-qwen3o-vllm/lib/python3.12/site-packages/vllm_omni")


def test_demo_opt_in_leaves_old_payload_unchanged():
    before = module.omni_tts_payload("你好。")
    assert module.demo_voice_config({}) is None
    assert module.demo_voice_config({"chat_demo": True}) is None
    control = module.demo_voice_config({"chat_demo": True, "demo_voice_control": True})
    payload = module.verbatim_tts_payload("你好。", voice_control=control)
    assert payload["voice"] == "chelsie" and payload["seed"] == 42
    assert payload["vllm_xargs"] == {"fd_demo_tts_rng": helper.CONTRACT}
    assert module.omni_tts_payload("你好。") == before
    assert "voice" not in module.verbatim_tts_payload("你好。")
    with pytest.raises(ValueError):
        module.demo_voice_config({"demo_voice_control": True})


@pytest.mark.parametrize("voice", [
    {"speaker": "ethan", "seed": 42}, {"speaker": "invalid", "seed": 42},
    {"speaker": "chelsie", "seed": True}, {"speaker": "chelsie", "seed": -1},
    {"speaker": "chelsie", "seed": 2**63}, {"speaker": "chelsie", "seed": "42"},
])
def test_bad_voice_rejected_before_http(voice):
    with pytest.raises(ValueError):
        module.verbatim_tts_payload("你好。", voice_control=voice)


@pytest.mark.parametrize("enabled", [False, True])
async def test_actor_uses_and_records_the_same_voice_payload(monkeypatch, enabled):
    from test_chat_demo import actor
    observed, recorded = [], []
    async def tts(text, **options):
        observed.append(module.verbatim_tts_payload(text, **options))
        yield stream_transport.PCMChunk(b"\0\0", 24000)
    monkeypatch.setattr(module, "tts_omni_stream", tts)
    e = actor(stream_response=True, demo_voice_control=enabled)
    e.demo_cases = NS(begin=lambda role, payload, context, text: recorded.append(payload))
    chunks = [c async for c in e._response_tts_stream("你好。")]
    assert chunks
    assert recorded == [{**observed[0], "stream": True}]
    assert ("voice" in observed[0]) == enabled
    e = actor(stream_response=False, demo_voice_control=True)
    assert e.tts_voice_control is None


def request(seed=42):
    return NS(vllm_xargs={"fd_demo_tts_rng": helper.CONTRACT}, voice="chelsie", seed=seed,
              modalities=["text", "audio"], structured_outputs=NS(grammar='root ::= "hi"'))


def test_only_opted_in_talker_params_change():
    params = [NS(seed=42, temperature=t, extra_args=None) for t in (0, .5, 0)]
    before = deepcopy(params)
    helper.configure_sampling(NS(vllm_xargs=None), params, "other_model")
    assert params == before
    helper.configure_sampling(request(7), params, "qwen3_omni_moe")
    assert params[0] == before[0] and params[2] == before[2]
    assert params[1].seed == 7 and params[1].temperature == .5
    assert params[1].extra_args == {helper.SEED_KEY: 7}
    bad = request(); bad.voice = "bogus"
    with pytest.raises(ValueError): helper.configure_sampling(bad, params, "qwen3_omni_moe")


def test_missing_voice_ack_releases_no_audio(monkeypatch):
    from test_tts_verbatim import audio_event, text_event
    async def events(*args):
        yield audio_event()
        yield text_event("你好。", "stop")
    monkeypatch.setattr(stream_transport, "sse_json", events)
    async def run():
        seen = []
        with pytest.raises(RuntimeError, match="acknowledge"):
            async for chunk in stream_transport.audio_stream("unused", {}, expected_text="你好。",
                    expected_voice={"contract": helper.CONTRACT, "speaker": "chelsie", "seed": 42}):
                seen.append(chunk)
        assert not seen
    asyncio.run(run())


def installed_sources():
    if not OMNI.is_dir():
        pytest.skip("Installed vLLM-Omni layout not available")
    return {f: (OMNI / f).read_text() for f in patcher.FILES}


def test_trace_is_noop_without_explicit_directory(monkeypatch):
    monkeypatch.delenv("FDBC_DEMO_VOICE_TRACE_DIR", raising=False)
    # No access to tensors/runner, CPU copies or files in the normal path.
    helper.trace_mtp(None, None, None, None, None, None)


def test_native_numerics_requires_explicit_worker_opt_in(monkeypatch):
    fake = NS(backends=NS(cuda=NS(preferred_blas_library=lambda **kwargs: None, matmul=NS(
        allow_bf16_reduced_precision_reduction=True,
        allow_bf16_reduced_precision_reduction_split_k=True))))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.delenv("FDBC_DEMO_TALKER_NATIVE_NUMERICS", raising=False)
    helper.configure_native_numerics("talker")
    assert fake.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    monkeypatch.setenv("FDBC_DEMO_TALKER_NATIVE_NUMERICS", "1")
    helper.configure_native_numerics("thinker")
    assert fake.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    helper.configure_native_numerics("talker")
    assert fake.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == (False, False)


def test_trace_excludes_uncontrolled_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("FDBC_DEMO_VOICE_TRACE_DIR", str(tmp_path))
    runner = NS(requests={"old": NS(sampling_params=NS(extra_args=None))})
    helper.trace_mtp(runner, ["old"], None, None, None, None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mode,key", [("native", "FDBC_DEMO_TALKER_NATIVE_NUMERICS"),
                                      ("invariant", "VLLM_BATCH_INVARIANT")])
def test_talker_mode_derives_config_without_changing_frozen_source(tmp_path, mode, key):
    import yaml
    config = yaml.safe_load((ROOT / "configs/qwen3_omni_audio_single_gpu.yaml").read_text())
    config["stages"][1]["env"] = {"EXISTING_SETTING": "kept"}
    source, target = tmp_path / "base.yaml", tmp_path / "demo.yaml"
    source.write_text(yaml.safe_dump(config))
    original = source.read_text()
    with pytest.raises(ValueError, match="overwrite"):
        patcher.write_talker_numerics_deploy(source, source, mode)
    patcher.write_talker_numerics_deploy(source, target, mode)
    actual = yaml.safe_load(target.read_text())
    assert source.read_text() == original
    assert actual["stages"][1]["env"].pop(key) == "1"
    assert actual == config
    config["stages"] = config["stages"][:1]
    source.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="three-stage"):
        patcher.write_talker_numerics_deploy(source, target, mode)


def test_patch_idempotence_and_unknown_layout():
    sources = installed_sources()
    patched = patcher.patch_sources(sources)
    assert patcher.patch_sources(patched) == patched
    partial = dict(patched)
    partial[patcher.FILES[0]] = partial[patcher.FILES[0]].replace(patcher.MARKER, "")
    with pytest.raises(RuntimeError): patcher.patch_sources(partial)
    with pytest.raises(RuntimeError): patcher.patch_sources(dict.fromkeys(patcher.FILES, "unknown"))


def test_actual_runner_isolates_seeded_rows_across_reordering_and_global_draws(monkeypatch):
    """Execute the patched upstream runner method with a stochastic CPU MTP.

    Uses real torch.Generator/multinomial. Model math is a test double; this
    proves scheduling/RNG plumbing, not GPU model or perceptual determinism.
    """
    import torch
    patched = patcher.patch_sources(installed_sources())
    tree = ast.parse(patched["worker/gpu_model_runner.py"])
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_talker_mtp_forward")
    method.decorator_list = []
    monkeypatch.setitem(sys.modules, "vllm_omni.fd_demo_voice", helper)
    modes = []
    class Wrapper:
        def __call__(self, ids, embeds, hidden, text, **kwargs):
            g = kwargs.get("generator")
            if g is not None:
                assert ids.shape[0] == 1 and modes[-1] == "none"
            codes = torch.multinomial(torch.ones((ids.shape[0], 64)), 8, replacement=True, generator=g)
            return embeds, codes
    def forward_context(*args, **kwargs):
        modes.append(kwargs["cudagraph_runtime_mode"])
        return nullcontext()
    ns = {"torch": torch, "np": np, "CUDAGraphMode": NS(NONE="none"),
          "current_omni_platform": NS(get_graph_wrapper_cls=lambda: Wrapper, set_forward_context=forward_context),
          "logger": NS(info=lambda *a: None)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "upstream_runner", "exec"), ns)
    class Runner:
        _talker_mtp_forward = ns["_talker_mtp_forward"]
        def __init__(self):
            self.requests = {rid: NS(sampling_params=NS(extra_args={helper.SEED_KEY: seed} if seed is not None else None))
                             for rid, seed in (("a", 42), ("b", 7), ("old", None))}
            self.vllm_config = NS(model_config=NS(subtalker_sampling_params=None))
            self.model = NS()
            self.talker_mtp = Wrapper()
            for field in ("talker_mtp_input_ids", "talker_mtp_inputs_embeds", "last_talker_hidden", "text_step"):
                setattr(self, field, NS(gpu=torch.zeros((4, 1))))
            self.outputs = {k: [] for k in self.requests}
        def _determine_batch_execution_and_padding(self, **kwargs):
            return "full", NS(num_tokens=4), None, None, None
        def _merge_additional_information_update(self, rid, update):
            self.outputs[rid].append(update["codes"]["audio"].clone())
        def step(self, ids):
            self._talker_mtp_forward(ids, torch.zeros((4, 1)), list(range(len(ids))))
    solo, mixed = Runner(), Runner()
    global_before = torch.random.get_rng_state().clone()
    for _ in range(4): solo.step(["a"])
    assert torch.equal(global_before, torch.random.get_rng_state())
    for ids in (["b", "a"], ["a"], ["old", "a", "b"], ["b", "a"]):
        torch.rand(100)  # unrelated calls cannot advance a's stream
        mixed.step(ids)
    assert all(torch.equal(a, b) for a, b in zip(solo.outputs["a"], mixed.outputs["a"], strict=True))
    assert not torch.equal(solo.outputs["a"][0], solo.outputs["a"][1])
    # The request object, not its batch position, owns the persistent stream.
    assert set(mixed._talker_mtp_generators) == {"a", "b"}
    assert 'self._talker_mtp_generators.pop(req_id, None)' in patched["worker/gpu_model_runner.py"]
