"""Installed alongside vLLM-Omni by patch_omni_demo_voice.py; no torch import.

Voice/RNG hooks require the explicit request contract. Optional numerical
settings apply to the demo Talker process; other deployments default off.
Ordinary requests retain their sampling parameters and speaker defaults.
"""
CONTRACT = "demo-voice-rng-v1"
SEED_KEY = "fd_omni_request_seed"


def voice_proof(request):
    args = getattr(request, "vllm_xargs", None) or {}
    if "fd_demo_tts_rng" not in args:
        return None
    if args["fd_demo_tts_rng"] != CONTRACT:
        raise ValueError("Unsupported fd_demo_tts_rng contract")
    voice = getattr(request, "voice", None) or getattr(request, "speaker", None)
    if voice != "chelsie":
        raise ValueError("demo-voice-rng-v1 requires explicit voice=chelsie")
    seed = getattr(request, "seed", None)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Demo voice requires an integer seed in [0, 2**63)")
    if set(getattr(request, "modalities", None) or []) != {"text", "audio"}:
        raise ValueError("Demo voice requires text and audio modalities")
    structured = getattr(request, "structured_outputs", None)
    grammar = structured.get("grammar") if isinstance(structured, dict) else getattr(structured, "grammar", None)
    if not grammar:
        raise ValueError("Demo voice requires the verbatim grammar contract")
    return {"contract": CONTRACT, "speaker": voice, "seed": seed}


def configure_sampling(request, params, model_type):
    proof = voice_proof(request)
    if proof is None:
        return
    if model_type != "qwen3_omni_moe" or len(params) != 3:
        raise ValueError("Demo voice requires the three-stage Qwen3-Omni pipeline")
    # Stage 0 already receives request.seed. Match the primary-code sampler's
    # seed and use an independent generator with this seed for residual codes.
    params[1].seed = proof["seed"]
    params[1].extra_args = {**(params[1].extra_args or {}), SEED_KEY: proof["seed"]}


def request_seed(runner, req_id):
    params = getattr(runner.requests[req_id], "sampling_params", None)
    extra = getattr(params, "extra_args", None) or {}
    return extra.get(SEED_KEY)


def trace_mtp(runner, req_ids, primary, hidden, text, codes):
    """Explicit offline diagnostic; CPU copies perturb timing, never a latency run.

    Full tensor values permit direct comparisons without integrity hashes.
    Only opted-in synthetic test requests should be sent to a traced server.
    """
    import os
    directory = os.environ.get("FDBC_DEMO_VOICE_TRACE_DIR")
    if not directory:
        return
    import json
    from pathlib import Path
    import re
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for row, req_id in enumerate(req_ids):
        if request_seed(runner, req_id) is None:
            continue
        info = runner.model_intermediate_buffer.get(req_id, {})
        def values(tensor):
            return tensor.detach().float().cpu().reshape(-1).tolist()
        def shape(value):
            return list(value.shape) if hasattr(value, "shape") else None
        meta = info.get("meta", {})
        state = runner.requests[req_id]
        generator = getattr(state, "generator", None)
        record = {
            "request_id": req_id, "seed": request_seed(runner, req_id),
            "scheduled_ids": list(runner.input_batch.req_ids),
            "computed_tokens": state.num_computed_tokens,
            "primary_rng_offset": generator.get_offset() if generator is not None else None,
            "residual_rng_offset": runner._talker_mtp_generators[req_id].get_offset(),
            "primary": values(primary[row]), "codes": values(codes[row]),
            "hidden": values(hidden[row]), "text": values(text[row]),
            "meta": {key: value for key, value in meta.items()
                     if isinstance(value, (str, int, float, bool, type(None)))},
            "decode_shape": shape(info.get("embed", {}).get("decode")),
            "cached_decode_shape": shape(info.get("embed", {}).get("cached_decode")),
        }
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", req_id) + ".jsonl"
        with (root / filename).open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def configure_native_numerics(model_stage):
    """Demo Talker: retain native kernels with FP32, unsplit BF16 accumulation."""
    import os
    if model_stage != "talker" or os.environ.get("FDBC_DEMO_TALKER_NATIVE_NUMERICS") != "1":
        return
    import torch
    # PyTorch 2.11 rejects allow_splitk=False on its cuBLAS backend.
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    import logging
    logging.getLogger(__name__).info(
        "fd_demo_talker_native bf16_reduced=%s bf16_split_k=%s",
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
