#!/usr/bin/env python3
"""Narrow, idempotent adapter for the installed vLLM-Omni chat server.

Forward structured decoding to the comprehension stage only. For constrained
streams expose its finish reason separately: Omni suppresses the ordinary text
finish_reason while audio is still running. The client needs this proof before
releasing audio, without waiting for the whole audio response.
Unconstrained requests and all downstream sampling defaults remain unchanged.
"""
import ast
import importlib.util
from pathlib import Path

FORWARD_MARKER = "# fd-badcat: forward structured decoding to comprehension only"
FINISH_MARKER = "# fd-badcat: constrained text completion proof before audio EOF"
OVERRIDE_ANCHOR = """        # For GLM-Image: compute max_tokens from height/width with mode-aware
"""
STREAM_ANCHOR = """                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\\n\\n"

                elif final_output_type == "audio":
"""


def patched_source(source):
    if FORWARD_MARKER in source and FINISH_MARKER in source:
        return source
    if (FORWARD_MARKER in source or FINISH_MARKER in source
            or source.count(OVERRIDE_ANCHOR) != 1 or source.count(STREAM_ANCHOR) != 1):
        raise RuntimeError("Unknown/partially patched vLLM-Omni serving_chat; refusing an unsafe patch")
    source = source.replace(OVERRIDE_ANCHOR, f"""        {FORWARD_MARKER}
        if "structured_outputs" in explicit_fields and request.structured_outputs is not None:
            from copy import deepcopy
            params.structured_outputs = deepcopy(request.structured_outputs)

""" + OVERRIDE_ANCHOR, 1)
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
