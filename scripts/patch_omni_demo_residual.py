"""Install opt-in residual Transformer graphs; keep request-local sampling eager.

Version-aware: supports the vLLM-Omni 0.22 CodePredictorWrapper layout
(unchanged rollback path) and the 0.30 layout (packed projections, Gumbel-max
sampling, ``generators``/``sample_uniforms`` arguments). Unknown layouts are
refused before anything is written.

Seed mapping on 0.30: upstream 0.30 replaced the 0.22 residual sampler
(softmax + torch.multinomial) with Gumbel-max. Both draw from the same
distribution, but one request seed gives different residual codes, so the
0.22 seed-to-voice mapping (e.g. the chosen demo seed) is not kept by default.
On 0.30 this installer therefore also adds FDBC_DEMO_RESIDUAL_SAMPLER:
"gumbel" (default, upstream 0.30) or "multinomial" (the 0.22 sampler on the
same request Generator, for scalar-Generator calls only). The 0.22 path has
no such switch because it always uses the 0.22 sampler.
"""
import ast
import importlib.util
from pathlib import Path

MARKER = "# fd-demo-residual-inner-graphs-v1"
HELPER = '''
    # fd-demo-residual-inner-graphs-v1
    def _demo_inner_graphs(self):
        """Capture deterministic batch-one forward without sampling or RNG.

        Seeded requests already execute separately per row in the demo runner.
        The private pool does not alias outer Talker graphs. Each graph output
        is consumed on the same stream before the next residual step replays.
        """
        cache = getattr(self, "_fd_demo_inner_graph_cache", None)
        if cache is not None:
            return cache
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Seeded residual inner graphs cannot initialize inside an outer capture")
        lengths = [self._num_groups + 1]
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        cache = {}
        with torch.cuda.stream(stream):
            for length in lengths:
                positions = torch.arange(length, device=self._proj_buf.device, dtype=torch.long).unsqueeze(0)
                inputs = self._proj_buf[:1, :length, :]
                for _ in range(3):
                    self._compiled_model_fwd(inputs, positions)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    output = self._compiled_model_fwd(inputs, positions)
                cache[length] = (graph, output, positions)
        torch.cuda.current_stream().wait_stream(stream)
        self._fd_demo_inner_graph_cache = cache
        self._fd_demo_inner_graph_pool = pool
        logger.info("fd_demo_residual_inner_graphs mode=graph lengths=%s sampling=eager", lengths)
        return cache

'''

# Shared by both layouts: the forward signature anchor and the demo-mode
# selection block (inserted before the padded-buffer zeroing).
FORWARD_ANCHOR = "    @torch.inference_mode()\n    def forward(\n        self,\n        layer0_code:"
ZERO_ANCHOR = "        # Zero the padded region of the buffer\n"
MODE_BLOCK = '''        # Capture only the deterministic inner model. Preserve sampling and RNG.
        import os
        demo_graphs = None
        demo_mode = os.environ.get("FDBC_DEMO_RESIDUAL_OPT", "off")
        if demo_mode not in ("off", "graph"):
            raise ValueError("FDBC_DEMO_RESIDUAL_OPT must be off or graph")
        if generator is not None and demo_mode == "graph":
            if bsz != 1:
                raise ValueError("Seeded residual graph path requires the existing per-request row execution")
            demo_graphs = self._demo_inner_graphs()

        # Zero the padded region of the buffer
'''

# vLLM-Omni 0.22: transformer dispatch in the residual loop.
RUN_ANCHOR_022 = "            # Run transformer (device graph replay or compiled forward)\n            if device_graph_entry is not None:\n"
# vLLM-Omni 0.30: same dispatch, plus the NPU outer-capture guard.
RUN_ANCHOR_030 = (
    "            # Let the outer graph record the regular forward during capture;\n"
    "            # normal inference still uses the inner graph replay fast path.\n"
    "            if device_graph_entry is not None and not is_npu_capturing:\n"
)
# Layout fingerprints (must all be present for the 0.30 path).
FINGERPRINT_030 = (
    "    def _sample_codes_gumbel(\n",
    "        generators: Sequence[torch.Generator | None] | None = None,\n",
    "        sample_uniforms: torch.Tensor | None = None,\n",
    "            hidden_out = model_fwd(proj_buf[:padded_bsz, :seq_len, :], pos_ids)\n",
    "import torch.nn.functional as F\n",
)


# vLLM-Omni 0.30 only: residual sampler selection, appended to MODE_BLOCK.
SAMPLER_BLOCK_030 = '''        # Residual sampler. "gumbel" is upstream 0.30 (Gumbel-max). "multinomial"
        # restores the vLLM-Omni 0.22 sampler (softmax + multinomial on the same
        # request Generator) so a request seed keeps its 0.22 residual codes.
        # Same distribution either way; only scalar-Generator calls switch.
        demo_sampler = os.environ.get("FDBC_DEMO_RESIDUAL_SAMPLER", "gumbel")
        if demo_sampler not in ("gumbel", "multinomial"):
            raise ValueError("FDBC_DEMO_RESIDUAL_SAMPLER must be gumbel or multinomial")
        demo_multinomial = (demo_sampler == "multinomial" and generator is not None
                            and generators is None and sample_uniforms is None)
        if demo_multinomial and not getattr(self, "_fd_demo_sampler_logged", False):
            self._fd_demo_sampler_logged = True
            logger.info("fd_demo_residual_sampler=multinomial seed_mapping=vllm-omni-0.22")

'''
# vLLM-Omni 0.30: upstream Gumbel-max calls (stored mode, then per_call mode).
STORED_SAMPLE_030 = (
    "                step_uniforms = sample_uniforms[:, step - 1, :] if sample_uniforms is not None else None\n"
    "                code = self._sample_codes_gumbel(logits, generator=sample_generator, uniforms=step_uniforms)\n"
)
STORED_SAMPLE_030_NEW = '''                if demo_multinomial:
                    # vLLM-Omni 0.22 stored-mode sampler on the same request Generator.
                    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
                    code = torch.multinomial(probs, num_samples=1, generator=generator)
                else:
                    step_uniforms = sample_uniforms[:, step - 1, :] if sample_uniforms is not None else None
                    code = self._sample_codes_gumbel(logits, generator=sample_generator, uniforms=step_uniforms)
'''
PER_CALL_SAMPLE_030 = (
    "                    step_uniforms = sample_uniforms[:, step - 1, :] if sample_uniforms is not None else None\n"
    "                    code = self._sample_codes_gumbel(scaled, generator=sample_generator, uniforms=step_uniforms)\n"
)
PER_CALL_SAMPLE_030_NEW = '''                    if demo_multinomial:
                        # vLLM-Omni 0.22 per_call sampler on the same request Generator.
                        probs = F.softmax(scaled, dim=-1, dtype=torch.float32)
                        code = torch.multinomial(probs, num_samples=1, generator=generator)
                    else:
                        step_uniforms = sample_uniforms[:, step - 1, :] if sample_uniforms is not None else None
                        code = self._sample_codes_gumbel(scaled, generator=sample_generator, uniforms=step_uniforms)
'''


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise RuntimeError("Unsupported CodePredictorWrapper source layout: " + anchor.splitlines()[0])
    return source.replace(anchor, replacement, 1)


def detect_layout(source):
    """Return "0.22" or "0.30"; refuse anything else before writing."""
    is_022 = source.count(RUN_ANCHOR_022) == 1 and "def _sample_codes_gumbel(" not in source
    is_030 = (source.count(RUN_ANCHOR_030) == 1 and all(source.count(f) == 1 for f in FINGERPRINT_030)
              and source.count(STORED_SAMPLE_030) == 1 and source.count(PER_CALL_SAMPLE_030) == 1)
    if is_022 == is_030:
        raise RuntimeError("Unsupported CodePredictorWrapper source layout: neither the vLLM-Omni 0.22 nor the 0.30 layout")
    return "0.22" if is_022 else "0.30"


def patch_source_022(source):
    """Original 0.22 patch; output must stay identical to the deployed installer."""
    anchor = "    @torch.inference_mode()\n    def forward(\n        self,\n        layer0_code:"
    source = replace_once(source, anchor, HELPER + anchor)
    source = replace_once(source, "        # Zero the padded region of the buffer\n", '''        # Capture only the deterministic inner model. Preserve sampling and RNG.
        import os
        demo_graphs = None
        demo_mode = os.environ.get("FDBC_DEMO_RESIDUAL_OPT", "off")
        if demo_mode not in ("off", "graph"):
            raise ValueError("FDBC_DEMO_RESIDUAL_OPT must be off or graph")
        if generator is not None and demo_mode == "graph":
            if bsz != 1:
                raise ValueError("Seeded residual graph path requires the existing per-request row execution")
            demo_graphs = self._demo_inner_graphs()

        # Zero the padded region of the buffer
''')
    source = replace_once(source, "            # Run transformer (device graph replay or compiled forward)\n            if device_graph_entry is not None:\n", '''            # Run transformer; sampling below stays on the original Generator.
            if demo_graphs is not None:
                graph, hidden_out, _positions = demo_graphs[max_seq]
                graph.replay()
            elif device_graph_entry is not None:
''')
    ast.parse(source)
    return source


def patch_source_030(source):
    """0.30: same full-length batch-one inner graph, same gating.

    Only ``generator`` (the scalar request Generator the runner passes on its
    per-row seeded path) selects the graph; ``generators``/``sample_uniforms``
    callers keep upstream behaviour. The graph replaces only the transformer
    forward, so the selected sampler still consumes the original Generator
    (or the supplied uniforms) eagerly. The full-length graph
    deliberately ignores prefix re-prefill buckets, as on 0.22.

    FDBC_DEMO_RESIDUAL_SAMPLER=multinomial (default gumbel) swaps only the
    final sampling op of scalar-Generator calls back to the 0.22 sampler, in
    both stored and per_call mode; the filtering before it is unchanged.
    """
    source = replace_once(source, FORWARD_ANCHOR, HELPER + FORWARD_ANCHOR)
    source = replace_once(source, ZERO_ANCHOR, MODE_BLOCK.replace(ZERO_ANCHOR, SAMPLER_BLOCK_030 + ZERO_ANCHOR))
    source = replace_once(source, RUN_ANCHOR_030, '''            # Let the outer graph record the regular forward during capture;
            # normal inference still uses the inner graph replay fast path.
            # Demo residual graph first; sampling below stays on the original Generator.
            if demo_graphs is not None:
                graph, hidden_out, _positions = demo_graphs[max_seq]
                graph.replay()
            elif device_graph_entry is not None and not is_npu_capturing:
''')
    source = replace_once(source, STORED_SAMPLE_030, STORED_SAMPLE_030_NEW)
    source = replace_once(source, PER_CALL_SAMPLE_030, PER_CALL_SAMPLE_030_NEW)
    ast.parse(source)
    return source


def patch_source(source):
    if MARKER in source:
        return None
    if detect_layout(source) == "0.22":
        return patch_source_022(source)
    return patch_source_030(source)


def main():
    spec = importlib.util.find_spec("vllm_omni")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_omni is not installed in this Python environment")
    root = Path(next(iter(spec.submodule_search_locations)))
    target = root / "model_executor/models/common/qwen3_code_predictor.py"
    patched = patch_source(target.read_text(encoding="utf-8"))
    if patched is None:
        print("Demo residual inner graph patch already installed")
        return
    target.write_text(patched, encoding="utf-8")
    print("Installed demo residual inner graphs (default off):", target)


if __name__ == "__main__":
    main()
