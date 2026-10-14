# Migrated to c3v2; no artifact integrity hashing.
"""Paired analysis of run_f2.py results against prereg.json.

usage: python analyze_f2.py RUN_DIR [RUN_DIR ...] [--prereg prereg.json] [--out summary.json] [--B 10000]

Unit = (item, hole, rep). Every arm is paired with the W0 arm of the same latency model
(W400_pool vs W0_pool, W400_idle vs W0_idle) on the same unit. An item-hole score is the
mean over the repeats valid in both arms; 95% CIs come from a bootstrap that resamples
whole roots (all holes / variants / gaps of one source utterance move together).
"""
import argparse
import collections
import glob
import json
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent

def load(dirs):
    """Latest valid record per unit; metrics are recomputed from the stored rows with the
    runner's summarize() so a metric definition never depends on the run that produced it."""
    import run_f2
    (manifest, _) = run_f2.load_manifest()
    by_uid = {it['uid']: it for it in manifest['items']}
    units = {}
    for d in dirs:
        for path in sorted(glob.glob(str(Path(d) / 'units.w*.jsonl'))):
            with open(path, encoding='utf-8') as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    prev = units.get(r['unit'])
                    if prev is None or not prev.get('valid') or r.get('valid'):
                        units[r['unit']] = r
    for r in units.values():
        for a in r['arms'].values():
            rec = {k: a[k] for k in ('status', 'timing', 'calls', 'begun')}
            a['metrics'] = run_f2.summarize(by_uid[r['uid']], rec, a['rows'], a['controls'])
    return units

def stratum(r):
    if r['set'] == 'H':
        return 'H_hole' if r['hole'] else 'H_nohole'
    return r['set']

def base_of(arm):
    return 'W0_' + arm.split('_', 1)[1]

def m(r, arm):
    a = r['arms'].get(arm)
    return None if a is None or not a['metrics'].get('valid') else a['metrics']

def decision_seq(r, arm):
    return [(ev, t, d.get('route'), d.get('closed'), d.get('transcript')) for (ev, t, d) in r['arms'][arm]['rows'] if ev in ('input_decision', 'input_decision_stale')]

def frag_or_premature(x, r):
    return float(x['fragmented'] or x['premature_action']) if r['set'] in ('H', 'P') else None
BINARY = {'clean_success': lambda x, r: None if x.get('clean_success') is None else float(x['clean_success']), 'route_correct': lambda x, r: None if x.get('route_correct') is None else float(x['route_correct']), 'behavior_ok': lambda x, r: float(x['behavior_ok']), 'fragmented': lambda x, r: float(x['fragmented']) if r['set'] in ('H', 'P') else None, 'premature_action': lambda x, r: float(x['premature_action']) if r['set'] in ('H', 'P') else None, 'frag_or_premature': frag_or_premature, 'stale_any': lambda x, r: float(x['stale'] > 0), 'timeout_or_fallback': lambda x, r: float(bool(x['calls_timeout'] or x['fallback'])), 'stop_cancelled': lambda x, r: float(x['cancel_t'] is not None) if r['expected'] == 'stop_only' else None, 'merged_any': lambda x, r: float(x['merged'] > 0)}
COUNT = {'stale': lambda x, r: float(x['stale']), 'n_inputs': lambda x, r: float(x['n_inputs']), 'calls_route': lambda x, r: float(x['calls_route'])}
LATENCY = {'stop_latency': lambda x, r: x['stop_latency'] if r['expected'] == 'stop_only' else None, 'response_latency': lambda x, r: x['response_latency'] if r['expected'] == 'yield_ready' and x.get('response_via') == 'decision' else None, 'decision_latency': lambda x, r: x['decision_latency']}

def paired(units, arm, metric_fn, select):
    """-> dict item_key -> (root, [(rep, on, off)])"""
    base = base_of(arm)
    out = collections.defaultdict(lambda : [None, []])
    for r in units.values():
        if not select(r):
            continue
        (a, b) = (m(r, arm), m(r, base))
        if a is None or b is None or (not r.get('valid')):
            continue
        (va, vb) = (metric_fn(a, r), metric_fn(b, r))
        if va is None or vb is None:
            continue
        key = f"{r['uid']}|{r['hole']}"
        out[key][0] = r['root']
        out[key][1].append((r['rep'], va, vb))
    return out

def cluster_ci(roots, diffs, B, seed, level=0.95):
    if not diffs:
        return (None, None)
    groups = collections.defaultdict(list)
    for (root, d) in zip(roots, diffs):
        groups[root].append(d)
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), size=(B, len(sums)))
    stats = sums[pick].sum(1) / counts[pick].sum(1)
    (lo, hi) = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (float(lo), float(hi))

def compare(units, arm, fn, select, B, seed, level=0.95):
    data = paired(units, arm, fn, select)
    if not data:
        return {'n_items': 0}
    keys = sorted(data)
    roots = [data[k][0] for k in keys]
    on = np.array([np.mean([v for (_, v, _) in data[k][1]]) for k in keys])
    off = np.array([np.mean([v for (_, _, v) in data[k][1]]) for k in keys])
    diff = on - off
    (lo, hi) = cluster_ci(roots, list(diff), B, seed, level)
    per_rep = {}
    for rep in sorted({rp for k in keys for (rp, *_) in data[k][1]}):
        d = [v - w for k in keys for (rp, v, w) in data[k][1] if rp == rep]
        per_rep[str(rep)] = round(float(np.mean(d)), 4) if d else None
    raw = np.array([v - w for k in keys for (_, v, w) in data[k][1]])
    return {'n_items': len(keys), 'n_roots': len(set(roots)), 'n_units': int(raw.size), 'arm': round(float(on.mean()), 4), 'base': round(float(off.mean()), 4), 'diff': round(float(diff.mean()), 4), 'ci': [None if lo is None else round(lo, 4), None if hi is None else round(hi, 4)], 'ci_level': level, 'per_rep_diff': per_rep, 'unit_diff_p50': round(float(np.quantile(raw, 0.5)), 4), 'unit_diff_p90': round(float(np.quantile(raw, 0.9)), 4), 'unit_diff_max': round(float(raw.max()), 4)}

def net_flips(units, arm, select, fn=BINARY['behavior_ok']):
    """Per category: items whose majority result went good->bad minus bad->good."""
    data = paired(units, arm, fn, select)
    cats = collections.Counter()
    worse = collections.Counter()
    cat_of = {f"{r['uid']}|{r['hole']}": r['category'] for r in units.values()}
    for (k, (_, rows)) in data.items():
        on = np.mean([v for (_, v, _) in rows]) > 0.5
        off = np.mean([v for (_, _, v) in rows]) > 0.5
        if on != off:
            cats[cat_of[k]] += 1 if off and (not on) else -1
            if off and (not on):
                worse[cat_of[k]] += 1
    return {'net_worse': dict(cats), 'items_worse': dict(worse)}

def jitter(units, arm, fn, select):
    """Share of item-holes whose repeats disagree (arm alone), and mean pairwise rep disagreement."""
    per = collections.defaultdict(list)
    for r in units.values():
        x = m(r, arm)
        if x is None or not select(r) or (not r.get('valid')):
            continue
        v = fn(x, r)
        if v is not None:
            per[f"{r['uid']}|{r['hole']}"].append(v)
    multi = [v for v in per.values() if len(v) >= 2]
    if not multi:
        return {'n_items': 0}
    disagree = [len(set(v)) > 1 for v in multi]
    pairs = [np.mean([a != b for (i, a) in enumerate(v) for b in v[i + 1:]]) for v in multi]
    return {'n_items': len(multi), 'items_disagree': round(float(np.mean(disagree)), 4), 'pairwise_disagree': round(float(np.mean(pairs)), 4)}

def listening_identity(units, arm):
    same = total = 0
    for r in units.values():
        if r['set'] != 'L' or m(r, arm) is None or m(r, base_of(arm)) is None:
            continue
        total += 1
        same += decision_seq(r, arm) == decision_seq(r, base_of(arm))
    return {'units': total, 'identical': same, 'rate': round(same / total, 4) if total else None}

def harness_failures(units):
    bad = [u for (u, r) in units.items() if not r.get('valid')]
    return {'count': len(bad), 'units': bad[:50]}
SELECT = {'H_hole': lambda r: stratum(r) == 'H_hole', 'H_nohole': lambda r: stratum(r) == 'H_nohole', 'P': lambda r: r['set'] == 'P', 'Q': lambda r: r['set'] == 'Q', 'L': lambda r: r['set'] == 'L', 'HP_all': lambda r: r['set'] in ('H', 'P'), 'H_all': lambda r: r['set'] == 'H'}

def check(crit, res):
    """Evaluate one pre-registered criterion against computed results -> (passed, detail)."""
    kind = crit['kind']
    r = res.get(crit['result']) or {}
    if not r or not (r.get('units') if kind == 'identity' else r.get('n_items')):
        return (None, 'no data')
    if kind == 'gain':
        ok = r['diff'] >= crit['min_point_gain'] and r['ci'][0] is not None and (r['ci'][0] > crit['ci_lower_gt']) and all((v is not None and v > 0 for v in r['per_rep_diff'].values()))
    elif kind == 'decrease':
        ok = r['diff'] <= -crit['min_point_drop'] and r['ci'][1] is not None and (r['ci'][1] < 0)
    elif kind == 'noninferior':
        ok = r['diff'] >= -crit['max_point_drop'] and r['ci'][1] is not None and (r['ci'][1] >= 0)
    elif kind == 'noninferior_bad':
        ok = r['diff'] <= crit['max_point_rise'] and r['ci'][0] is not None and (r['ci'][0] <= 0)
    elif kind == 'latency':
        ok = r['ci'][1] is not None and r['ci'][1] <= crit['max_mean_increase_s'] and (r['unit_diff_p90'] <= crit['max_p90_increase_s'])
    elif kind == 'identity':
        ok = r.get('rate') is not None and r['rate'] >= crit['min_rate']
    elif kind == 'flips':
        ok = all((v <= crit['max_net_worse_per_category'] for v in r.get('net_worse', {}).values()))
    elif kind == 'zero':
        ok = r.get('count', 0) == 0
    else:
        raise ValueError(kind)
    return (bool(ok), r)

def analyse_arm(units, arm, B, seed, level=0.95):
    res = {}
    for (name, sel) in SELECT.items():
        for (metric, fn) in {**BINARY, **COUNT, **LATENCY}.items():
            r = compare(units, arm, fn, sel, B, seed, level)
            if r.get('n_items'):
                res[f'{name}.{metric}'] = r
    res['Q.net_flips'] = {**net_flips(units, arm, SELECT['Q']), 'n_items': 1}
    res['HP_all.net_flips_clean'] = {**net_flips(units, arm, SELECT['HP_all'], BINARY['clean_success']), 'n_items': 1}
    res['L.identity'] = listening_identity(units, arm)
    engine_errors = sum((m(r, arm)['engine_errors'] for r in units.values() if m(r, arm) is not None))
    res['engine_errors'] = {'count': engine_errors, 'n_items': 1}
    by_cat = {}
    for r in units.values():
        if stratum(r) in ('H_hole', 'Q', 'H_nohole'):
            by_cat.setdefault(f"{stratum(r)}:{r['category']}:{r['hole']}", None)
    for key in sorted(by_cat):
        (st, cat, hole) = key.split(':')
        sel = (lambda st, cat, hole: lambda r: stratum(r) == st and r['category'] == cat and (str(r['hole']) == hole))(st, cat, hole)
        fn = BINARY['behavior_ok'] if st == 'Q' else BINARY['clean_success']
        by_cat[key] = compare(units, arm, fn, sel, 2000, seed, level)
    res['by_category'] = by_cat
    return res

def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--prereg', default=str(HERE / 'prereg.json'))
    p.add_argument('--out')
    p.add_argument('--B', type=int, default=10000)
    args = p.parse_args()
    prereg = json.loads(Path(args.prereg).read_text(encoding='utf-8'))
    seed = prereg['bootstrap']['seed']
    units = load(args.dirs)
    arms = sorted({a for r in units.values() for a in r['arms']})
    summary = {'units': len(units), 'arms': arms, 'harness_failures': harness_failures(units), 'by_arm': {}, 'jitter': {}, 'criteria': {}}
    for arm in arms:
        if arm.startswith('W0_'):
            summary['jitter'][arm] = {f'{s}.{k}': jitter(units, arm, BINARY[k], SELECT[s]) for s in ('H_hole', 'H_nohole', 'P', 'Q') for k in ('clean_success', 'route_correct', 'behavior_ok')}
            continue
        if base_of(arm) not in arms:
            continue
        level = prereg['arms'].get(arm, {}).get('ci_level', 0.95)
        summary['by_arm'][arm] = analyse_arm(units, arm, args.B, seed, level)
    for (arm, spec) in prereg['arms'].items():
        if spec.get('role') not in ('primary', 'alternate') or arm not in summary['by_arm']:
            continue
        res = summary['by_arm'][arm]
        verdicts = {}
        for (cid, crit) in prereg['criteria'].items():
            source = arm if not crit.get('latency') else arm.split('_')[0] + '_' + crit['latency']
            if source not in summary['by_arm']:
                verdicts[cid] = {'passed': None, 'detail': f'{source} missing'}
                continue
            (ok, detail) = check(crit, summary['by_arm'][source])
            verdicts[cid] = {'passed': ok, 'source_arm': source, 'detail': detail}
        decided = [v['passed'] for v in verdicts.values()]
        verdicts['OVERALL'] = None if any((v is None for v in decided)) else all(decided)
        summary['criteria'][arm] = verdicts
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    print(f"units={len(units)} arms={arms} harness_failures={summary['harness_failures']['count']}")
    keys = ['H_hole.clean_success', 'H_hole.frag_or_premature', 'H_hole.stale', 'H_hole.route_correct', 'H_nohole.clean_success', 'P.clean_success', 'Q.behavior_ok', 'HP_all.stop_latency', 'HP_all.response_latency', 'HP_all.timeout_or_fallback', 'HP_all.calls_route']
    for (arm, res) in summary['by_arm'].items():
        print(f"== {arm} vs {base_of(arm)}  L.identity={res['L.identity']}")
        for k in keys:
            r = res.get(k)
            if r:
                print(f"  {k:30s} n={r['n_items']:5d} base={r['base']:.4f} arm={r['arm']:.4f} diff={r['diff']:+.4f} ci={r['ci']} reps={r['per_rep_diff']}")
    for (arm, v) in summary['criteria'].items():
        print(arm, {k: x if isinstance(x, (bool, type(None))) else x['passed'] for (k, x) in v.items()})
if __name__ == '__main__':
    main()
