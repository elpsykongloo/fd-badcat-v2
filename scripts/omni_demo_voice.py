"""Installed alongside vLLM-Omni by patch_omni_demo_voice.py; no torch import.

Only requests bearing the explicit demo contract opt in. Ordinary requests keep
their original sampling parameters, speaker defaults and execution path.
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
