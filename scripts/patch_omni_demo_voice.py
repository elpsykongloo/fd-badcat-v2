#!/usr/bin/env python3
"""Opt-in Qwen3-Omni voice/RNG adapter; validate all layouts before writing.

Reuse the runner's existing per-request generator lifetime and seeded-row
isolation. Only opted-in Omni MTP calls bypass CUDA graph replay. Never reseed
the process-global RNG. No sampling-distribution changes for residual codes.
"""
import ast
import argparse
import importlib.util
from pathlib import Path

MARKER = "# fd-badcat: demo-voice-rng-v1"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Unknown vLLM-Omni source layout at {old[:100]!r}")
    return source.replace(old, new, 1)


def patch_sources(sources):
    result = dict(sources)
    marked = [MARKER in s for s in sources.values()]
    if any(marked):
        if not all(marked):
            raise RuntimeError("Partially patched Omni voice adapter; refusing to proceed")
        return patch_trace(result)
    f = "entrypoints/openai/serving_chat.py"
    s = sources[f]
    s = replace_once(s, "        return sampling_params_list\n\n    def _log_inputs(",
        f"""        {MARKER}
        from vllm_omni.fd_demo_voice import configure_sampling
        configure_sampling(request, sampling_params_list, self.model_config.hf_config.model_type)
        return sampling_params_list

    def _log_inputs(""")
    s = replace_once(s, '                            proof["fd_text_finish_reason"] = output.finish_reason\n',
        '''                            proof["fd_text_finish_reason"] = output.finish_reason
                            from vllm_omni.fd_demo_voice import voice_proof
                            voice = voice_proof(request)
                            if voice is not None:
                                proof["fd_tts_voice"] = voice
''')
    result[f] = s

    f = "worker/gpu_model_runner.py"
    s = sources[f]
    s = replace_once(s,
        '        if not isinstance(self.talker_mtp, current_omni_platform.get_graph_wrapper_cls()):\n',
        f'''        {MARKER}
        from vllm_omni.fd_demo_voice import request_seed
        demo_seeded = any(request_seed(self, rid) is not None for rid in decode_req_ids)
        if demo_seeded or not isinstance(self.talker_mtp, current_omni_platform.get_graph_wrapper_cls()):
''')
    s = replace_once(s,
        '            seed = extra_args.get("qwen3_tts_request_seed") if isinstance(extra_args, dict) else None\n',
        '''            seed = request_seed(self, req_id)
            if seed is None:
                seed = extra_args.get("qwen3_tts_request_seed") if isinstance(extra_args, dict) else None
''')
    s = replace_once(s, '                    generators[first_req_id] = generator\n',
        '''                    generators[first_req_id] = generator
                    if request_seed(self, first_req_id) is not None:
                        logger.info("fd_demo_tts_rng request=%s seed=%d residual_batch=1 eager=true",
                                    first_req_id, seed)
''')
    # These are necessary lifetime/isolation invariants, not optional optimizations.
    for anchor in ('self._talker_mtp_generators.pop(req_id, None)',
                   'self._talker_mtp_forward([req_id], inputs_embeds, row_offsets)',
                   'talker_kwargs["generator"] = generator'):
        if anchor not in s:
            raise RuntimeError(f"Missing upstream RNG lifetime/isolation support: {anchor}")
    result[f] = s

    f = "model_executor/models/qwen3_omni/qwen3_omni.py"
    result[f] = replace_once(sources[f],
        '            input_ids, inputs_embeds, last_talker_hidden=last_talker_hidden\n',
        f'''            {MARKER}
            input_ids, inputs_embeds, last_talker_hidden=last_talker_hidden,
            **({{"generator": kwargs["generator"]}} if "generator" in kwargs else {{}}),
''')
    f = "model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py"
    s = sources[f]
    s = replace_once(s, '        last_talker_hidden: torch.Tensor | None = None,\n        **_: object,\n',
        f'''        {MARKER}
        last_talker_hidden: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        **_: object,
''')
    s = replace_once(s, '                last_talker_hidden,\n            )\n',
        '''                last_talker_hidden,
                **({"generator": generator} if generator is not None else {}),
            )
''')
    result[f] = s
    for s in result.values():
        ast.parse(s)
    return patch_trace(result)


def patch_trace(sources):
    result = dict(sources)
    f = "worker/gpu_model_runner.py"
    marker = "# fd-badcat: demo-voice-trace-v1"
    if marker not in result[f]:
        result[f] = replace_once(result[f],
            "        # update the inputs_embeds and code_predictor_codes\n",
            f"""        {marker}
        from vllm_omni.fd_demo_voice import trace_mtp
        trace_mtp(self, decode_req_ids, req_input_ids, last_talker_hidden, text_step, code_predictor_codes)
        # update the inputs_embeds and code_predictor_codes
""")
        ast.parse(result[f])
    marker = "# fd-badcat: demo-native-numerics-v1"
    if marker not in result[f]:
        result[f] = replace_once(result[f],
            "        super().load_model(*args, **kwargs)\n",
            f"""        {marker}
        from vllm_omni.fd_demo_voice import configure_native_numerics
        configure_native_numerics(self.model_config.model_stage)
        super().load_model(*args, **kwargs)
""")
        ast.parse(result[f])
    return result


FILES = ("entrypoints/openai/serving_chat.py", "worker/gpu_model_runner.py",
         "model_executor/models/qwen3_omni/qwen3_omni.py",
         "model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py")


def write_talker_numerics_deploy(source, target, mode):
    """Derive a local config; keep the shared/frozen YAML untouched."""
    import yaml
    if Path(source).resolve() == Path(target).resolve():
        raise ValueError("Demo deployment must not overwrite its source config")
    if mode not in ("native", "invariant"):
        raise ValueError("Demo Talker numerical mode must be native or invariant")
    config = yaml.safe_load(Path(source).read_text())
    stages = config.get("stages", [])
    if [stage.get("stage_id") for stage in stages] != [0, 1, 2]:
        raise ValueError("Demo Talker mode requires the three-stage Omni deployment")
    env = dict(stages[1].get("env") or {})
    env.pop("VLLM_BATCH_INVARIANT", None)
    env.pop("FDBC_DEMO_TALKER_NATIVE_NUMERICS", None)
    env["FDBC_DEMO_TALKER_NATIVE_NUMERICS" if mode == "native" else "VLLM_BATCH_INVARIANT"] = "1"
    stages[1]["env"] = env
    Path(target).write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talker-numerics-deploy", nargs=3, metavar=("MODE", "SOURCE", "TARGET"))
    args = parser.parse_args()
    spec = importlib.util.find_spec("vllm_omni")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm_omni not installed")
    root = Path(next(iter(spec.submodule_search_locations)))
    sources = {f: (root / f).read_text() for f in FILES}
    patched = patch_sources(sources)
    helper = Path(__file__).with_name("omni_demo_voice.py").read_text()
    ast.parse(helper)
    (root / "fd_demo_voice.py").write_text(helper)
    for f, source in patched.items():
        if source != sources[f]:
            (root / f).write_text(source)
    if args.talker_numerics_deploy:
        mode, source, target = args.talker_numerics_deploy
        write_talker_numerics_deploy(source, target, mode)
    print("Omni demo voice adapter ready (request RNG opt-in; optional Talker process numerics)")


if __name__ == "__main__":
    main()
