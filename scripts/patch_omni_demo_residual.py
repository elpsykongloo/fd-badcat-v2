"""Install opt-in residual Transformer graphs; keep request-local sampling eager."""
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


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise RuntimeError("Unsupported CodePredictorWrapper source layout: " + anchor.splitlines()[0])
    return source.replace(anchor, replacement, 1)


def patch_source(source):
    if MARKER in source:
        return None
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
