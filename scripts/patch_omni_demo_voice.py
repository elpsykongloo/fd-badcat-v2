#!/usr/bin/env python3
"""Opt-in Qwen3-Omni voice/RNG adapter; validate all layouts before writing.

Reuse the runner's existing per-request generator lifetime and seeded-row
isolation. Only opted-in Omni MTP calls bypass CUDA graph replay. Never reseed
the process-global RNG. No sampling-distribution changes for residual codes.
"""
import ast
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
        return result
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
    return result


FILES = ("entrypoints/openai/serving_chat.py", "worker/gpu_model_runner.py",
         "model_executor/models/qwen3_omni/qwen3_omni.py",
         "model_executor/models/qwen3_omni/qwen3_omni_moe_talker.py")


def main():
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
    print("Omni demo voice adapter ready (request opt-in only)")


if __name__ == "__main__":
    main()
