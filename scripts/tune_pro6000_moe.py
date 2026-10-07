#!/usr/bin/env python3
"""Bounded serial BF16 tuning using the matching vLLM upstream benchmark.

Run with the demo stopped. Validation seeds are separate from search seeds;
unproven shapes retain the installed version's default configuration.
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import time

import torch
import vllm
from vllm.triton_utils import triton


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def candidates(default, tokens):
    values = [default]
    # Small, explicit space; no split-K or precision changes.
    ms = (16, 32) if tokens <= 16 else ((16, 32, 64) if tokens <= 256 else (32, 64, 128))
    for m in ms:
        for n, k in ((32, 64), (64, 64), (64, 128), (128, 64)):
            for stages in (2, 3):
                value = dict(BLOCK_SIZE_M=m, BLOCK_SIZE_N=n, BLOCK_SIZE_K=k,
                             GROUP_SIZE_M=1, num_warps=4, num_stages=stages,
                             SPLIT_K=1)
                if value not in values:
                    values.append(value)
    if tokens >= 256:
        for warps in (4, 8):
            value = dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                         GROUP_SIZE_M=4, num_warps=warps, num_stages=3, SPLIT_K=1)
            if value not in values:
                values.append(value)
    return values


def main(args):
    if vllm.__version__ != '0.22.0':
        raise RuntimeError('Retune protocol must match vLLM 0.22.0')
    # Keep the upstream benchmark function unchanged; its Ray scheduler is
    # unnecessary for one GPU and is deliberately excluded from this harness.
    import types
    upstream = types.SimpleNamespace()
    source = ast.parse(args.upstream.read_text())
    nodes = []
    for node in source.body:
        if isinstance(node, ast.Import) and all(a.name != 'ray' for a in node.names):
            nodes.append(node)
        elif isinstance(node, ast.ImportFrom) and not node.module.startswith('ray'):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FP8_DTYPE' for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'benchmark_config':
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == 'BenchmarkConfig':
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(args.upstream), 'exec'), upstream.__dict__)
    torch.set_default_device('cuda')
    args.output.mkdir(parents=True, exist_ok=False)
    args.config_dir.mkdir(parents=True, exist_ok=True)
    receipt = {'protocol':'pro6000-bf16-bounded-v1', 'vllm':vllm.__version__,
               'torch':torch.__version__, 'triton':triton.__version__,
               'device':torch.cuda.get_device_name(), 'serial_exclusive_gpu':True,
               'dtype':'bfloat16', 'split_k':1, 'search_seed':42,
               'validation_seeds':[101, 202, 303], 'guard_seeds':[404,505,606], 'rows':[]}
    started = time.monotonic()
    for model, hidden, intermediate, topk in [('thinker',2048,768,8), ('talker',1024,384,6)]:
        chosen = {}
        for tokens in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
            default = upstream.get_default_config(tokens,128,2*intermediate,hidden,topk,None)
            default['SPLIT_K'] = 1
            row = {'model':model,'tokens':tokens,'experts':128,'hidden':hidden,
                   'intermediate':intermediate,'topk':topk,'default':default,
                   'search':[], 'validation':[]}
            receipt['rows'].append(row)
            def measure(config, seed, iterations):
                upstream.set_random_seed(seed)
                return upstream.benchmark_config(config, tokens,128,2*intermediate,
                                                hidden,topk,torch.bfloat16,False,False,
                                                num_iters=iterations)
            for config in candidates(default,tokens):
                try:
                    timing = measure(config,42,6)
                    row['search'].append({'config':config,'us':timing})
                except Exception as exc:
                    row['search'].append({'config':config,'error':type(exc).__name__+': '+str(exc)[:300]})
                save(args.output/'receipt.json', receipt)
            valid = sorted([r for r in row['search'] if 'us' in r], key=lambda r:r['us'])[:2]
            finalist = [default]
            for entry in valid:
                if entry['config'] not in finalist:
                    finalist.append(entry['config'])
            for seed in (101,202,303):
                # Alternate order to reduce time-order bias.
                ordered = finalist if seed != 202 else list(reversed(finalist))
                for config in ordered:
                    row['validation'].append({'seed':seed,'config':config,'us':measure(config,seed,30)})
            def median(config):
                return statistics.median(r['us'] for r in row['validation'] if r['config']==config)
            best = min(finalist,key=median)
            base = median(default)
            paired = [next(r['us'] for r in row['validation'] if r['seed']==seed and r['config']==default)/
                      next(r['us'] for r in row['validation'] if r['seed']==seed and r['config']==best)
                      for seed in (101,202,303)]
            accepted = base/median(best) >= 1.03 and min(paired) >= 1.0
            selected = best if accepted else default
            row.update(selected=selected, default_us=base, selected_us=median(selected),
                       speedup=base/median(selected), optimized=accepted and selected!=default,
                       paired_speedups=paired)
            chosen[str(tokens)] = selected
            save(args.output/'receipt.json',receipt)
            print(model,tokens,'default_us',round(base,2),'selected_us',round(median(selected),2),
                  'speedup',round(row['speedup'],3),flush=True)
        # Check the gaps in the nearest-batch lookup on fresh routing seeds.
        # Keep an explicit default at a gap unless the measured benefit holds.
        tuned_grid=dict(chosen)
        for tokens in (3,6,12,24,48,96,192,384,768,1536,3072,6144):
            default=upstream.get_default_config(tokens,128,2*intermediate,hidden,topk,None)
            default['SPLIT_K']=1
            nearest=min(tuned_grid,key=lambda key:abs(int(key)-tokens))
            candidate=tuned_grid[nearest]
            row={'model':model,'tokens':tokens,'experts':128,'hidden':hidden,
                 'intermediate':intermediate,'topk':topk,'default':default,
                 'guard_only':True,'validation':[]}
            receipt['rows'].append(row)
            if candidate==default:
                selected=default
                row.update(selected=selected,optimized=False,speedup=1.0)
            else:
                for seed in (404,505,606):
                    for config in ([default,candidate] if seed!=505 else [candidate,default]):
                        upstream.set_random_seed(seed)
                        us=upstream.benchmark_config(config,tokens,128,2*intermediate,hidden,
                            topk,torch.bfloat16,False,False,num_iters=30)
                        row['validation'].append({'seed':seed,'config':config,'us':us})
                def median(config):
                    return statistics.median(r['us'] for r in row['validation'] if r['config']==config)
                paired=[next(r['us'] for r in row['validation'] if r['seed']==seed and r['config']==default)/
                        next(r['us'] for r in row['validation'] if r['seed']==seed and r['config']==candidate)
                        for seed in (404,505,606)]
                accepted=median(default)/median(candidate)>=1.03 and min(paired)>=1.0
                selected=candidate if accepted else default
                row.update(selected=selected,optimized=accepted,default_us=median(default),
                    selected_us=median(selected),speedup=median(default)/median(selected),paired_speedups=paired)
            chosen[str(tokens)]=selected
            save(args.output/'receipt.json',receipt)
            print(model,tokens,'guard speedup',round(row['speedup'],3),flush=True)
        filename=upstream.get_config_file_name(128,intermediate,None)
        save(args.config_dir/filename,chosen)
    receipt['elapsed_s']=time.monotonic()-started
    save(args.output/'receipt.json',receipt)
    save(args.output/'summary.json',{k:v for k,v in receipt.items() if k!='rows'} | {
        'rows':[{k:v for k,v in r.items() if k not in ('search','validation')} for r in receipt['rows']]})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config-dir',type=Path,required=True)
    main(parser.parse_args())
