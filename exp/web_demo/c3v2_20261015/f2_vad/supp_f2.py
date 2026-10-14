# Migrated to c3v2; no artifact integrity hashing.
"""Supplementary (reported, not judged) f2 analyses on top of analyze_f2.load():
1) dose-response table from summary.json; 2) latency cost restricted to units where neither arm
fragmented nor acted prematurely (pure deferral cost); 3) merge/deferral rates; 4) end-window check."""
import json, sys, collections
import numpy as np
sys.path.insert(0, '.')
import analyze_f2 as A
run = sys.argv[1]
units = A.load([run])
summ = json.load(open(f'{run}/summary.json', encoding='utf-8'))
out = {'dose_response': {}, 'pure_latency': {}, 'rates': {}, 'end_window': {}}
keys = ['H_hole.clean_success', 'H_hole.frag_or_premature', 'H_hole.fragmented', 'H_hole.premature_action', 'H_hole.stale', 'H_hole.route_correct', 'H_nohole.clean_success', 'P.clean_success', 'Q.behavior_ok', 'HP_all.stop_cancelled', 'HP_all.stop_latency', 'HP_all.response_latency', 'HP_all.decision_latency', 'HP_all.timeout_or_fallback', 'HP_all.calls_route']
for (arm, res) in summ['by_arm'].items():
    out['dose_response'][arm] = {k: {kk: res[k][kk] for kk in ('base', 'arm', 'diff', 'ci', 'per_rep_diff', 'unit_diff_p90', 'n_items')} for k in keys if k in res}
    out['dose_response'][arm]['L.identity'] = res['L.identity']
    out['dose_response'][arm]['Q.net_flips'] = res['Q.net_flips']
for arm in sorted(summ['by_arm']):
    base = A.base_of(arm)
    for (key, exp) in (('stop_latency', 'stop_only'), ('response_latency', 'yield_ready')):
        (roots, diffs, raw) = ([], [], collections.defaultdict(list))
        for r in units.values():
            if r['set'] not in ('H', 'P') or r['expected'] != exp or (not r.get('valid')):
                continue
            (a, b) = (A.m(r, arm), A.m(r, base))
            if a is None or b is None:
                continue
            if any((x['fragmented'] or x['premature_action'] for x in (a, b))):
                continue
            if key == 'response_latency' and (not a.get('response_via') == b.get('response_via') == 'decision'):
                continue
            if a.get(key) is None or b.get(key) is None:
                continue
            raw[r['uid'], r['hole'], r['root']].append(a[key] - b[key])
        if not raw:
            continue
        ks = sorted(raw)
        d = np.array([np.mean(raw[k]) for k in ks])
        (lo, hi) = A.cluster_ci([k[2] for k in ks], list(d), 10000, 20261010)
        allraw = np.concatenate([raw[k] for k in ks])
        out['pure_latency'][f'{arm}.{key}'] = {'n_items': len(ks), 'n_units': int(allraw.size), 'mean_ms': round(float(d.mean()) * 1000, 1), 'ci_ms': [round(lo * 1000, 1), round(hi * 1000, 1)], 'unit_p50_ms': round(float(np.quantile(allraw, 0.5)) * 1000, 1), 'unit_p90_ms': round(float(np.quantile(allraw, 0.9)) * 1000, 1), 'unit_max_ms': round(float(allraw.max()) * 1000, 1)}
for arm in sorted({a for r in units.values() for a in r['arms']}):
    c = collections.Counter()
    for r in units.values():
        x = r['arms'][arm]
        mt = x['metrics']
        if not mt.get('valid'):
            continue
        s = A.stratum(r)
        c[s, 'n'] += 1
        c[s, 'merged'] += mt['merged'] > 0
        c[s, 'deferred'] += mt['deferred'] > 0
        ew = x['timing'].get('end_window') or {}
        c['all', 'hold_open'] += bool(ew.get('hold_open'))
        c['all', 'deferred_left'] += bool(ew.get('deferred_left'))
        c['all', 'no_field'] += not ew
        c['all', 'episodes'] += 1
    out['rates'][arm] = {f'{s}.{k}': round(c[s, k] / c[s, 'n'], 4) for s in ('H_hole', 'H_nohole', 'P', 'Q', 'L') for k in ('merged', 'deferred') if c[s, 'n']}
    out['end_window'][arm] = {k: c['all', k] for k in ('episodes', 'hold_open', 'deferred_left', 'no_field')}
json.dump(out, open(f'{run}/supp.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('== dose-response (arm vs W0 same latency): diff [ci]')
for k in keys + ['L.identity']:
    print(' ', k)
    for arm in sorted(out['dose_response']):
        v = out['dose_response'][arm].get(k)
        if v is None:
            continue
        if k == 'L.identity':
            print(f'    {arm:10s} {v}')
        else:
            print(f"    {arm:10s} base={v['base']:.4f} arm={v['arm']:.4f} diff={v['diff']:+.4f} ci={v['ci']} reps={v['per_rep_diff']} p90={v['unit_diff_p90']} n={v['n_items']}")
print('== pure latency (neither arm fragmented/premature)')
for (k, v) in out['pure_latency'].items():
    print(' ', k, v)
print('== rates')
[print(' ', a, v) for (a, v) in out['rates'].items()]
print('== end window')
[print(' ', a, v) for (a, v) in out['end_window'].items()]
