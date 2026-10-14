# Migrated to c3v2; no artifact integrity hashing.
"""Paired analysis of run_f3.py results against the pre-registered criteria.

Item score = share of repeats whose review label is yield_ready (a timeout/fallback
keeps the production label keep). Arms are compared item by item against `off`;
95% CIs come from a bootstrap that resamples whole scenarios. Writes summary.json.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TARGET = {'target_generic', 'target_specific'}

def load(results, arms):
    data = {}
    for arm in arms:
        per = collections.defaultdict(dict)
        for f in sorted((results / arm).glob('r*.jsonl')):
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                prev = per[r['id']].get(r['repeat'])
                if prev is None or prev.get('harness_error') or (not r.get('harness_error')):
                    per[r['id']][r['repeat']] = r
        data[arm] = per
    return data

def yes(r):
    return 1.0 if r.get('review_label') == 'yield_ready' else 0.0

def valid(r):
    return r is not None and (not r.get('harness_error'))

def paired(data, arm, base, ids):
    """ids with every common repeat valid in both arms -> (ids, on scores, off scores, repeats)."""
    (keep, on, off, reps) = ([], [], [], None)
    for i in ids:
        (a, b) = (data[arm].get(i, {}), data[base].get(i, {}))
        common = sorted(set(a) & set(b))
        if not common or not all((valid(a[k]) and valid(b[k]) for k in common)):
            continue
        reps = common if reps is None else sorted(set(reps) & set(common))
        keep.append(i)
        on.append(np.mean([yes(a[k]) for k in common]))
        off.append(np.mean([yes(b[k]) for k in common]))
    return (keep, np.array(on), np.array(off), reps or [])

def cluster_ci(ids, diff, cluster_of, B, seed, level=0.95):
    if not len(ids):
        return (None, None)
    groups = collections.defaultdict(list)
    for (i, d) in zip(ids, diff):
        groups[cluster_of[i]].append(d)
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), size=(B, len(sums)))
    stats = sums[pick].sum(1) / counts[pick].sum(1)
    (lo, hi) = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return (float(lo), float(hi))

def majority(rows):
    vals = [yes(r) for r in rows.values() if valid(r)]
    return None if not vals else 'yield_ready' if sum(vals) * 2 > len(vals) else 'keep'

def mcnemar(data, arm, base, ids):
    b = c = 0
    for i in ids:
        (x, y) = (majority(data[base][i]), majority(data[arm][i]))
        b += x == 'keep' and y == 'yield_ready'
        c += x == 'yield_ready' and y == 'keep'
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum((math.comb(n, k) for k in range(0, min(b, c) + 1))) / 2 ** n)
    return {'keep_to_yield': b, 'yield_to_keep': c, 'net': b - c, 'p_two_sided': p}

def compare(data, arm, base, ids, cluster_of, boot, level=0.95):
    (ids, on, off, reps) = paired(data, arm, base, ids)
    if not ids:
        return {'n': 0}
    diff = on - off
    (lo, hi) = cluster_ci(ids, diff, cluster_of, boot['B'], boot['seed'], level)
    per_repeat = {}
    for k in reps:
        d = [yes(data[arm][i][k]) - yes(data[base][i][k]) for i in ids]
        per_repeat[k] = float(np.mean(d))
    return {'n': len(ids), 'clusters': len({cluster_of[i] for i in ids}), 'repeats': reps, 'off_rate': float(off.mean()), 'arm_rate': float(on.mean()), 'diff': float(diff.mean()), 'ci': [lo, hi], 'ci_level': level, 'per_repeat_diff': per_repeat, 'mcnemar_majority': mcnemar(data, arm, base, ids)}

def flips_by_category(data, arm, base, ids, cat_of):
    out = collections.defaultdict(lambda : {'keep_to_yield': [], 'yield_to_keep': []})
    for i in ids:
        if i not in data[arm] or i not in data[base]:
            continue
        (x, y) = (majority(data[base][i]), majority(data[arm][i]))
        if x == 'keep' and y == 'yield_ready':
            out[cat_of[i]]['keep_to_yield'].append(i)
        elif x == 'yield_ready' and y == 'keep':
            out[cat_of[i]]['yield_to_keep'].append(i)
    return {k: {**v, 'net': len(v['keep_to_yield']) - len(v['yield_to_keep'])} for (k, v) in out.items()}

def invariants(data, arms, base, ids):
    route_same = route_n = called_same = 0
    for i in ids:
        for (k, b) in data[base].get(i, {}).items():
            if not valid(b):
                continue
            others = [data[a].get(i, {}).get(k) for a in arms if a != base]
            others = [o for o in others if valid(o)]
            route_n += 1
            route_same += all((o.get('route_payload') == b.get('route_payload') for o in others))
            called_same += all((o.get('review_called') == b.get('review_called') for o in others))
    return {'pairs': route_n, 'route_identity': route_same / route_n if route_n else None, 'review_called_identity': called_same / route_n if route_n else None}

def reliability(data, arm, ids):
    rows = [r for i in ids for r in data[arm].get(i, {}).values() if valid(r) and r.get('review_called')]
    lat = [r['review_elapsed_ms'] for r in rows if r.get('review_elapsed_ms') is not None]
    bad = sum((bool(r.get('review_timed_out') or r.get('review_fallback')) for r in rows))
    return {'reviews': len(rows), 'timeout_or_fallback': bad / len(rows) if rows else None, 'timeouts': sum((bool(r.get('review_timed_out')) for r in rows)), 'p50_ms': float(np.percentile(lat, 50)) if lat else None, 'p95_ms': float(np.percentile(lat, 95)) if lat else None, 'harness_errors': sum((1 for i in ids for r in data[arm].get(i, {}).values() if not valid(r))), 'omni_requests': sum((r.get('omni_requests', 0) for i in ids for r in data[arm].get(i, {}).values()))}

def jitter(data, arm, ids):
    unstable = n = 0
    for i in ids:
        vals = [yes(r) for r in data[arm].get(i, {}).values() if valid(r)]
        if len(vals) >= 2:
            n += 1
            unstable += len(set(vals)) > 1
    return {'items_with_repeats': n, 'unstable_items': unstable, 'unstable_rate': unstable / n if n else None}

def accuracy(data, arm, ids, manifest):
    by = collections.defaultdict(lambda : [0, 0])
    for i in ids:
        if manifest[i]['expected'] not in ('keep', 'yield_ready') or i not in data[arm]:
            continue
        m = majority(data[arm][i])
        if m is None:
            continue
        key = f"{manifest[i]['dataset']}/{manifest[i]['category']}"
        by[key][0] += m == manifest[i]['expected']
        by[key][1] += 1
    return {k: {'correct': c, 'n': n, 'acc': c / n} for (k, (c, n)) in sorted(by.items())}

def treatment(data, arm, ids, manifest):
    by = collections.defaultdict(lambda : [0, 0])
    for i in ids:
        for r in data[arm].get(i, {}).values():
            if valid(r):
                key = f"{manifest[i]['dataset']}/{manifest[i]['position']}"
                by[key][0] += (r.get('partial_chars') or 0) > 0
                by[key][1] += 1
    return {k: {'with_partial': a, 'n': n, 'rate': a / n} for (k, (a, n)) in sorted(by.items())}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=HERE / 'results' / 'live')
    p.add_argument('--manifest', type=Path, default=HERE / 'manifest.jsonl')
    p.add_argument('--prereg', type=Path, default=HERE / 'prereg.json')
    p.add_argument('--arms', default='off,on050,on100,cf_trueprefix,cf_fullsent')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--fresh-only', action='store_true', help='sensitivity: D1 scenarios never seen during prompt tuning or the f3 smoke')
    args = p.parse_args()
    pre = json.loads(args.prereg.read_text())
    lock = {}
    manifest = {r['id']: r for r in map(json.loads, args.manifest.read_text().splitlines())}
    arms = [a for a in args.arms.split(',') if (args.results / a).exists()]
    base = pre['baseline_arm']
    if base not in arms:
        raise SystemExit(f'no results for baseline arm {base}')
    data = load(args.results, arms)
    cluster_of = {i: f"{r['dataset']}:{r['scenario']}" for (i, r) in manifest.items()}
    cat_of = {i: r['category'] for (i, r) in manifest.items()}
    ids = [i for i in manifest if any((i in data[a] for a in arms))]
    if args.fresh_only:
        ids = [i for i in ids if manifest[i]['dataset'] != 'D1_followup_mid' or manifest[i]['source'].startswith('self-authored')]

    def sel(pred):
        return [i for i in ids if pred(manifest[i])]
    d1_target = sel(lambda r: r['dataset'] == 'D1_followup_mid' and r['category'] in TARGET)
    d1_specific = sel(lambda r: r['dataset'] == 'D1_followup_mid' and r['category'] == 'target_specific')
    d1_answer = sel(lambda r: r['dataset'] == 'D1_followup_mid' and r['category'] == 'answer_in_partial')
    d1_control = sel(lambda r: r['dataset'] == 'D1_followup_mid' and r['category'].startswith('control_'))
    d1_amb = sel(lambda r: r['category'] == 'ambiguous_cut_question')
    d1_early = sel(lambda r: r['category'] == 'target_generic_early')
    d2 = sel(lambda r: r['dataset'] == 'D2_frozen_text')
    d3_keep = sel(lambda r: r['dataset'] == 'D3_frozen_late' and r['expected'] == 'keep')
    d3_yield = sel(lambda r: r['dataset'] == 'D3_frozen_late' and r['expected'] == 'yield_ready')
    boot = pre['bootstrap']
    summary = {'results': str(args.results), 'arms': arms, 'items_seen': len(ids), 'invariants': invariants(data, arms, base, ids), 'per_arm': {}, 'comparisons': {}}
    for arm in arms:
        summary['per_arm'][arm] = {'reliability': reliability(data, arm, ids), 'jitter': jitter(data, arm, ids), 'accuracy': accuracy(data, arm, ids, manifest), 'treatment': treatment(data, arm, ids, manifest)}
    off_unstable = summary['per_arm'][base]['jitter']['unstable_items']
    for arm in arms:
        if arm == base:
            continue
        level = 0.95
        c = {'primary_target': compare(data, arm, base, d1_target, cluster_of, boot, level), 'target_specific': compare(data, arm, base, d1_specific, cluster_of, boot), 'answer_in_partial': compare(data, arm, base, d1_answer, cluster_of, boot), 'target_generic_early': compare(data, arm, base, d1_early, cluster_of, boot), 'd1_controls': compare(data, arm, base, d1_control, cluster_of, boot), 'd1_control_flips': flips_by_category(data, arm, base, d1_control, cat_of), 'd3_keep': compare(data, arm, base, d3_keep, cluster_of, boot), 'd3_yield': compare(data, arm, base, d3_yield, cluster_of, boot), 'ambiguous': compare(data, arm, base, d1_amb, cluster_of, boot), 'd2_compare': compare(data, arm, base, d2, cluster_of, boot)}
        same = total = check = 0
        for i in d2:
            for (k, b) in data[base].get(i, {}).items():
                a = data[arm].get(i, {}).get(k)
                if valid(a) and valid(b):
                    total += 1
                    same += a.get('review_payload') == b.get('review_payload')
                    check += bool(a.get('matches_check_script_payload'))
        c['d2_identity'] = {'pairs': total, 'identity': same / total if total else None, 'check_script_match': check / total if total else None, 'majority_changed': c['d2_compare'].get('mcnemar_majority', {}).get('keep_to_yield', 0) + c['d2_compare'].get('mcnemar_majority', {}).get('yield_to_keep', 0), 'off_unstable_items': off_unstable}
        (rel_on, rel_off) = (summary['per_arm'][arm]['reliability'], summary['per_arm'][base]['reliability'])
        crit = pre['criteria']
        pt = c['primary_target']
        verdict = {}
        if pt.get('n'):
            verdict['PRIMARY'] = pt['diff'] >= crit['PRIMARY']['min_point_gain'] and pt['ci'][0] > crit['PRIMARY']['ci_lower_gt'] and (len(pt['per_repeat_diff']) == pre['repeats']) and all((v > 0 for v in pt['per_repeat_diff'].values()))
        ct = c['d1_controls']
        if ct.get('n'):
            verdict['S1_D1_CONTROLS'] = ct['diff'] <= crit['S1_D1_CONTROLS']['max_point_increase'] and ct['ci'][0] <= crit['S1_D1_CONTROLS']['ci_lower_le'] and all((v['net'] <= crit['S1_D1_CONTROLS']['max_net_flips_per_category'] for v in c['d1_control_flips'].values()))
        di = c['d2_identity']
        if di['pairs']:
            verdict['S2_D2_FROZEN'] = di['identity'] >= 1.0 and di['check_script_match'] >= 1.0 and (di['majority_changed'] <= off_unstable)
        k3 = c['d3_keep']
        if k3.get('n'):
            verdict['S3_D3_KEEP'] = k3['diff'] <= crit['S3_D3_KEEP']['max_point_increase'] and k3['ci'][0] <= crit['S3_D3_KEEP']['ci_lower_le']
        inv = summary['invariants']
        if rel_on['reviews'] and rel_off['reviews']:
            verdict['S4_INVARIANTS'] = bool(inv['route_identity'] == 1.0 and inv['review_called_identity'] == 1.0 and (rel_on['timeout_or_fallback'] - rel_off['timeout_or_fallback'] <= crit['S4_INVARIANTS']['max_timeout_fallback_increase']) and ((rel_on['p95_ms'] or 0) - (rel_off['p95_ms'] or 0) <= crit['S4_INVARIANTS']['max_p95_latency_increase_ms']))
        need = ['PRIMARY', 'S1_D1_CONTROLS', 'S2_D2_FROZEN', 'S3_D3_KEEP', 'S4_INVARIANTS']
        verdict['OVERALL'] = all((verdict.get(k) for k in need)) if all((k in verdict for k in need)) else None
        verdict['role'] = 'primary' if arm == pre['primary_arm'] else 'ablation'
        c['verdict'] = verdict
        summary['comparisons'][arm] = c
    out = args.out or args.results / 'summary.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    for (arm, c) in summary['comparisons'].items():

        def fmt(x):
            if not x.get('n'):
                return 'n=0'
            return f"n={x['n']} off={x['off_rate']:.3f} arm={x['arm_rate']:.3f} d={x['diff']:+.3f} CI=[{x['ci'][0]:+.3f},{x['ci'][1]:+.3f}]"
        print(f"== {arm} vs {base}: {c['verdict']}")
        for k in ('primary_target', 'target_specific', 'answer_in_partial', 'd1_controls', 'd3_keep', 'd3_yield', 'ambiguous'):
            print(f'   {k:18s} {fmt(c[k])}')
        print(f"   d2_identity        {c['d2_identity']}")
    print('invariants', summary['invariants'])
    print('wrote', out)
if __name__ == '__main__':
    main()
