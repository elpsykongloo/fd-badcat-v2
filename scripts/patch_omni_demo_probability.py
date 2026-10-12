"""Opt-in deterministic probability graphs; request sampling remains eager.

Version-aware: the vLLM-Omni 0.22 path (softmax + multinomial) is unchanged.
On the 0.30 layout upstream samples filtered logits with Gumbel-max, so the
graphs capture the same deterministic ops up to the filtered logits and the
sampler stays eager: upstream Gumbel-max (request Generator / precomputed
uniforms), or the 0.22 softmax + multinomial when the residual installer's
FDBC_DEMO_RESIDUAL_SAMPLER=multinomial is selected (``demo_multinomial``), so
turning graphs on or off never changes which sampler a request seed uses.
Requires patch_omni_demo_residual.py to be installed first.
"""
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

# vLLM-Omni 0.30: the stored-mode path ends in Gumbel-max over filtered logits
# (no softmax/multinomial). Capture exactly the deterministic part (lm head,
# top-k, top-p with its softmax); no random op may appear in this helper.
# The sampler (Gumbel-max, or 0.22 multinomial when selected) runs eagerly after replay.
HELPER_030 = '''
    # fd-demo-residual-probability-graphs-v1 (vllm-omni 0.30: filtered logits; sampler eager)
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
        # Original stored-mode operations and shapes up to the filtered logits.
        # The sampler (Gumbel-max or 0.22 multinomial; Generator / uniforms) stays outside these graphs.
        def filtered_logits(step):
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
            return logits
        with torch.cuda.stream(stream):
            for step in range(1, self._num_groups):
                for _ in range(3):
                    filtered_logits(step)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    output = filtered_logits(step)
                cache[step] = (graph, output)
        torch.cuda.current_stream().wait_stream(stream)
        self._fd_demo_probability_graph_cache = cache
        self._fd_demo_probability_graph_pool = pool
        logger.info("fd_demo_residual_probability_graphs steps=%s sampling=eager", len(cache))
        return cache

'''

INNER_MARKER = '    # fd-demo-residual-inner-graphs-v1\n'
ZERO_ANCHOR = '        # Zero the padded region of the buffer\n'
PROB_BLOCK = '''        demo_prob_graphs = None
        if demo_graphs is not None and os.environ.get("FDBC_DEMO_RESIDUAL_PROBS", "0") == "1":
            demo_prob_graphs = self._demo_probability_graphs(demo_graphs[max_seq][1])

        # Zero the padded region of the buffer
'''
LOGITS_ANCHOR = '            logits = lm_heads[step - 1](hidden_out[:bsz, step, :])\n'
SAMPLE_ANCHOR_022 = '            # Sample next code\n            if stored_mode:\n'
SAMPLE_ANCHOR_030 = '            if stored_mode:\n                # "stored" mode: top-k -> top-p -> Gumbel-max\n'
# The residual installer's 0.30 dispatch and sampler switch must already be present.
RESIDUAL_030 = '            elif device_graph_entry is not None and not is_npu_capturing:\n'
SAMPLER_030 = '        demo_multinomial = (demo_sampler == "multinomial" and generator is not None\n'
STORED_SAMPLE_030 = ('                if demo_multinomial:\n'
                     '                    # vLLM-Omni 0.22 stored-mode sampler on the same request Generator.\n'
                     '                    probs = F.softmax(logits, dim=-1, dtype=torch.float32)\n'
                     '                    code = torch.multinomial(probs, num_samples=1, generator=generator)\n'
                     '                else:\n'
                     '                    step_uniforms = sample_uniforms[:, step - 1, :] if sample_uniforms is not None else None\n'
                     '                    code = self._sample_codes_gumbel(logits, generator=sample_generator, uniforms=step_uniforms)\n')


def replace(source, anchor, value):
    if source.count(anchor) != 1:
        raise RuntimeError("Unsupported layout: " + anchor[:100])
    return source.replace(anchor, value, 1)


def detect_layout(source):
    """Return "0.22" or "0.30"; refuse anything else before writing."""
    is_022 = source.count(SAMPLE_ANCHOR_022) == 1 and 'def _sample_codes_gumbel(' not in source
    is_030 = (source.count(SAMPLE_ANCHOR_030) == 1 and source.count('    def _sample_codes_gumbel(\n') == 1
              and source.count(RESIDUAL_030) == 1 and source.count(SAMPLER_030) == 1
              and source.count(STORED_SAMPLE_030) == 1)
    if is_022 == is_030:
        raise RuntimeError('Unsupported layout: neither the vLLM-Omni 0.22 nor the 0.30 layout '
                           '(is patch_omni_demo_residual.py installed?)')
    return '0.22' if is_022 else '0.30'


def patch_source_022(source):
    """Original 0.22 patch; output must stay identical to the deployed installer."""
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


def patch_source_030(source):
    source = replace(source, INNER_MARKER, HELPER_030 + INNER_MARKER)
    source = replace(source, ZERO_ANCHOR, PROB_BLOCK)
    source = replace(source, LOGITS_ANCHOR, '''            if demo_prob_graphs is not None:
                probability_graph, logits = demo_prob_graphs[step]
                probability_graph.replay()
            else:
                logits = lm_heads[step - 1](hidden_out[:bsz, step, :])
''')
    source = replace(source, SAMPLE_ANCHOR_030, '''            if demo_prob_graphs is not None:
                # Filtered logits came from the graph; the selected sampler stays eager
                # on the original request Generator (or the supplied uniforms).
                if demo_multinomial:
                    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
                    code = torch.multinomial(probs, num_samples=1, generator=generator)
                else:
                    step_uniforms = sample_uniforms[:, step - 1, :] if sample_uniforms is not None else None
                    code = self._sample_codes_gumbel(logits, generator=sample_generator, uniforms=step_uniforms)
            elif stored_mode:
                # "stored" mode: top-k -> top-p -> Gumbel-max
''')
    ast.parse(source)
    return source


def patch_source(source):
    if '# fd-demo-residual-probability-graphs-v1' in source:
        return None
    if detect_layout(source) == '0.22':
        return patch_source_022(source)
    return patch_source_030(source)


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
