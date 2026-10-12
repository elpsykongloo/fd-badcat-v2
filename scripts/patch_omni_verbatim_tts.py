#!/usr/bin/env python3
"""Narrow, idempotent adapter for the installed vLLM-Omni chat server.

Forward structured decoding to the comprehension stage only. For constrained
streams expose its finish reason separately: Omni suppresses the ordinary text
finish_reason while audio is still running. The client needs this proof before
releasing audio, without waiting for the whole audio response.
Unconstrained requests and all downstream sampling defaults remain unchanged.

Supported layouts: vLLM-Omni 0.22 and 0.30. Both lack these two behaviours
upstream; 0.30 only dropped the GLM-Image block that 0.22 used as the anchor
inside _apply_request_overrides. Exactly one known layout must match.
"""
import ast
import importlib.util
from pathlib import Path

FORWARD_MARKER = "# fd-badcat: forward structured decoding to comprehension only"
FINISH_MARKER = "# fd-badcat: constrained text completion proof before audio EOF"
FORWARD_BLOCK = f"""        {FORWARD_MARKER}
        if "structured_outputs" in explicit_fields and request.structured_outputs is not None:
            from copy import deepcopy
            params.structured_outputs = deepcopy(request.structured_outputs)

"""
# vLLM-Omni 0.22: insert before the GLM-Image budgeting block.
OVERRIDE_ANCHOR = """        # For GLM-Image: compute max_tokens from height/width with mode-aware
"""
# vLLM-Omni 0.30: no GLM-Image block; insert after the explicit-field loop.
OVERRIDE_ANCHOR_030 = """                setattr(params, field_name, value)

        return params

    @staticmethod
    def _set_if_supported("""
OVERRIDE_LAYOUTS = (
    (OVERRIDE_ANCHOR, FORWARD_BLOCK + OVERRIDE_ANCHOR),
    (OVERRIDE_ANCHOR_030, """                setattr(params, field_name, value)

""" + FORWARD_BLOCK + """        return params

    @staticmethod
    def _set_if_supported("""),
)
STREAM_ANCHOR = """                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\\n\\n"

                elif final_output_type == "audio":
"""


def patched_source(source):
    if FORWARD_MARKER in source and FINISH_MARKER in source:
        return source
    counts = [source.count(old) for old, _ in OVERRIDE_LAYOUTS]
    if (FORWARD_MARKER in source or FINISH_MARKER in source
            or sorted(counts) != [0] * (len(counts) - 1) + [1] or source.count(STREAM_ANCHOR) != 1):
        raise RuntimeError("Unknown/partially patched vLLM-Omni serving_chat; refusing an unsafe patch")
    old, new = OVERRIDE_LAYOUTS[counts.index(1)]
    source = source.replace(old, new, 1)
    source = source.replace(STREAM_ANCHOR, f"""                        data = chunk.model_dump_json(exclude_unset=True)
                        {FINISH_MARKER}
                        if request.structured_outputs is not None and output.finish_reason is not None:
                            proof = json.loads(data)
                            proof["fd_text_finish_reason"] = output.finish_reason
                            data = json.dumps(proof, ensure_ascii=False)
                        yield f"data: {{data}}\\n\\n"

                elif final_output_type == "audio":
""", 1)
    ast.parse(source)
    return source


def main():
    spec = importlib.util.find_spec("vllm_omni")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm_omni is not installed in this interpreter")
    path = Path(next(iter(spec.submodule_search_locations))) / "entrypoints/openai/serving_chat.py"
    original = path.read_text(encoding="utf-8")
    patched = patched_source(original)
    if patched != original:
        path.write_text(patched, encoding="utf-8")
    print(f"Omni verbatim TTS adapter ready: {path}")


if __name__ == "__main__":
    main()
