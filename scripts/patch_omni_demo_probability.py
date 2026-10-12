"""Opt-in deterministic probability graphs; request sampling remains eager."""
import ast
from pathlib import Path

HELPER = '''
    # fd-demo-residual-probability-graphs-v1
    def _demo_probability_graphs(self, hidden_out):
        cache = getattr(self, "_fd_demo_probability_graph_cache", None)
        if cache is not None:
            return cache
        if self._wrapper_config.sampling_mode != "stored":
            raise ValueError("Probability graphs require stored sampling parameters")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Probability graphs cannot initialize in an outer capture")
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        cache = {}
        # Original operations and shapes, including both softmax operations.
        # The request Generator and multinomial stay outside these graphs.
        def probabilities(step):
            logits = self._lm_heads_list[step - 1](hidden_out[:1, step, :])
            if self._top_k > 0:
                topk_vals, _ = logits.topk(self._top_k, dim=-1)
                logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
            if self._top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1, dtype=torch.float32)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                remove_mask = (cumulative_probs - sorted_probs) >= self._top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            return F.softmax(logits, dim=-1, dtype=torch.float32)
        with torch.cuda.stream(stream):
            for step in range(1, self._num_groups):
                for _ in range(3):
                    probabilities(step)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    output = probabilities(step)
                cache[step] = (graph, output)
        torch.cuda.current_stream().wait_stream(stream)
        self._fd_demo_probability_graph_cache = cache
        self._fd_demo_probability_graph_pool = pool
        logger.info("fd_demo_residual_probability_graphs steps=%s sampling=eager", len(cache))
        return cache

'''


def replace(source, anchor, value):
    if source.count(anchor) != 1:
        raise RuntimeError("Unsupported layout: " + anchor[:100])
    return source.replace(anchor, value, 1)


def patch_source(source):
    if '# fd-demo-residual-probability-graphs-v1' in source:
        return None
    source = replace(source, '    # fd-demo-residual-inner-graphs-v1\n', HELPER + '    # fd-demo-residual-inner-graphs-v1\n')
    source = replace(source, '        # Zero the padded region of the buffer\n', '''        demo_prob_graphs = None
        if demo_graphs is not None and os.environ.get("FDBC_DEMO_RESIDUAL_PROBS", "0") == "1":
            demo_prob_graphs = self._demo_probability_graphs(demo_graphs[max_seq][1])

        # Zero the padded region of the buffer
''')
    source = replace(source, '            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])\n', '''            if demo_prob_graphs is not None:
                probability_graph, probs = demo_prob_graphs[step]
                probability_graph.replay()
            else:
                logits = lm_heads[step - 1](hidden_out[:bsz, step, :])
''')
    source = replace(source, '            # Sample next code\n            if stored_mode:\n', '''            # Sample next code with the original request Generator.
            if demo_prob_graphs is not None:
                code = torch.multinomial(probs, num_samples=1, generator=generator)
            elif stored_mode:
''')
    ast.parse(source)
    return source


def main():
    import importlib.util
    spec = importlib.util.find_spec('vllm_omni')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('vllm_omni is not installed')
    target = Path(next(iter(spec.submodule_search_locations))) / 'model_executor/models/common/qwen3_code_predictor.py'
    result = patch_source(target.read_text(encoding='utf-8'))
    if result is None:
        print('Demo probability graph patch already installed')
        return
    target.write_text(result, encoding='utf-8')
    print('Installed demo probability graphs (default off):', target)


if __name__ == '__main__':
    main()
