# Migrated to c3v2; no artifact integrity hashing.
"""Paired analysis of the f3_heard SUPPLEMENT (see supplement/supplement_prereg.json)."""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_f3 import load, compare, majority, jitter, valid
YIELD_KEYS = {'other', 'why', 'repeat'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=HERE / 'results_sup' / 'live')
    p.add_argument('--dir', type=Path, default=HERE / 'supplement')
    p.add_argument('--arms', default='off,on050,cf_trueprefix,cf_fullsent')
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    lock = {}
    data_bytes = (args.dir / 'manifest.jsonl').read_bytes()
    prereg_path = args.dir / 'supplement_prereg.json'
    pre = json.loads(prereg_path.read_text())
    manifest = {r['id']: r for r in map(json.loads, data_bytes.decode().splitlines())}
    arms = [a for a in args.arms.split(',') if (args.results / a).exists()]
    data = load(args.results, arms)
    cluster_of = {i: r['scenario'] for (i, r) in manifest.items()}
    boot = {'B': 10000, 'seed': 20261010}
    ids = list(manifest)
    harm = [i for i in ids if manifest[i]['expected'] == 'yield_ready']
    keep = [i for i in ids if manifest[i]['expected'] == 'keep']
    out = {'results': str(args.results), 'arms': arms, 'per_arm': {}, 'comparisons': {}}
    for arm in arms:
        acc = collections.defaultdict(lambda : [0, 0])
        treat = collections.defaultdict(lambda : [0, 0])
        rel = collections.Counter()
        for i in ids:
            rows = data[arm].get(i, {})
            m = majority(rows)
            if m is not None:
                k = manifest[i]['category']
                acc[k][0] += m == manifest[i]['expected']
                acc[k][1] += 1
            for r in rows.values():
                if valid(r):
                    treat[manifest[i]['position']][0] += (r.get('partial_chars') or 0) > 0
                    treat[manifest[i]['position']][1] += 1
                    rel['reviews'] += bool(r.get('review_called'))
                    rel['timeout_or_fallback'] += bool(r.get('review_timed_out') or r.get('review_fallback'))
                else:
                    rel['harness_errors'] += 1
        out['per_arm'][arm] = {'accuracy': {k: {'correct': c, 'n': n, 'acc': c / n} for (k, (c, n)) in sorted(acc.items())}, 'treatment': {k: {'with_partial': a, 'n': n} for (k, (a, n)) in sorted(treat.items())}, 'reliability': dict(rel), 'jitter': jitter(data, arm, ids)}
    for arm in arms:
        if arm == 'off':
            continue
        c = {'harm_all': compare(data, arm, 'off', harm, cluster_of, boot), 'keep_all': compare(data, arm, 'off', keep, cluster_of, boot), 'by_category': {}, 'by_position': {}}
        for cat in sorted({manifest[i]['category'] for i in ids}):
            c['by_category'][cat] = compare(data, arm, 'off', [i for i in ids if manifest[i]['category'] == cat], cluster_of, boot)
        for pos in ('Q1', 'Q2', 'Q3'):
            c['by_position'][pos] = {'harm': compare(data, arm, 'off', [i for i in harm if manifest[i]['position'] == pos], cluster_of, boot), 'keep': compare(data, arm, 'off', [i for i in keep if manifest[i]['position'] == pos], cluster_of, boot)}
        flips = collections.defaultdict(lambda : {'yield_to_keep': [], 'keep_to_yield': []})
        for i in ids:
            if i in data[arm] and i in data['off']:
                (x, y) = (majority(data['off'][i]), majority(data[arm][i]))
                if x == 'yield_ready' and y == 'keep':
                    flips[manifest[i]['category']]['yield_to_keep'].append(i)
                elif x == 'keep' and y == 'yield_ready':
                    flips[manifest[i]['category']]['keep_to_yield'].append(i)
        c['flips'] = {k: {**v, 'net_yield_to_keep': len(v['yield_to_keep']) - len(v['keep_to_yield'])} for (k, v) in sorted(flips.items())}
        crit = pre['criteria']['SUP_HARM']
        h = c['harm_all']
        harm_cats = {k for k in c['flips'] if k.split('/')[-1] in YIELD_KEYS}
        c['verdict'] = {'SUP_HARM': bool(h.get('n') and h['diff'] >= -crit['max_point_decrease'] and all((c['flips'][k]['net_yield_to_keep'] <= crit['max_net_yield_to_keep_flips_per_category'] for k in harm_cats))), 'role': 'primary' if arm == 'on050' else 'counterfactual'}
        out['comparisons'][arm] = c
    path = args.out or args.results / 'summary_sup.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')

    def fmt(x):
        if not x.get('n'):
            return 'n=0'
        return f"n={x['n']} off={x['off_rate']:.3f} arm={x['arm_rate']:.3f} d={x['diff']:+.3f} CI=[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]"
    for (arm, c) in out['comparisons'].items():
        print(f"== {arm} vs off: {c['verdict']}")
        print('   harm_all', fmt(c['harm_all']))
        print('   keep_all', fmt(c['keep_all']))
        for (k, v) in c['by_category'].items():
            print(f'   {k:44s} {fmt(v)}')
        for (k, v) in c['by_position'].items():
            print(f"   {k} harm {fmt(v['harm'])} | keep {fmt(v['keep'])}")
        print('   flips', {k: (len(v['yield_to_keep']), len(v['keep_to_yield'])) for (k, v) in c['flips'].items()})
    print('wrote', path)
if __name__ == '__main__':
    main()
