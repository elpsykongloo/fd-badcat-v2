# Migrated to c3v2; no artifact integrity hashing.
"""Analyse f1_wait A/B runs against the pre-registered criteria (prereg.json).

Paired unit = (trial_id, rep) between two arms; CIs are cluster bootstraps over
items (all reps/scenario rows of one audio item move together), B=10000, seed 0.
Only status=="ok" rows enter metrics; harness errors are reported separately.

usage: python f1_analyze.py RUN_DIR [RUN_DIR ...] [--control off] [--treatment on_max1] [--out FILE]
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
EXP = Path(__file__).resolve().parent
NEGATIVE = ('N1_resume', 'N2a_third_natural', 'N3_stop_fault', 'N4_echo_fault', 'N5_question_natural', 'N5_backchannel_natural', 'N5_wait_natural')
POSITIVE = ('P_timeout', 'P_error')

def load(run_dirs):
    out = []
    for d in run_dirs:
        p = Path(d) / 'results_resummarized.jsonl'
        if not p.exists():
            p = Path(d) / 'results.jsonl'
        if p.exists():
            out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    best = {}
    for r in out:
        k = (r['arm'], r['rep'], r['trial_id'])
        if r.get('status') == 'ok' or k not in best:
            best[k] = r
    return (list(best.values()), out)

def boot_ci(pairs, stat=np.mean, b=10000, seed=0):
    """pairs: {item: [values]} -> (point, lo, hi) of stat over item means."""
    items = [k for (k, v) in pairs.items() if v]
    if not items:
        return (None, None, None)
    means = np.array([np.mean(pairs[k]) for k in items], dtype=float)
    point = float(stat(means))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(items), size=(b, len(items)))
    dist = stat(means[idx], axis=1) if stat is np.mean else np.array([stat(means[i]) for i in idx])
    return (point, float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5)))

def upper95(k, n):
    """One-sided 95% Clopper-Pearson upper bound for k/n."""
    if n == 0:
        return None
    if k == n:
        return 1.0
    (lo, hi) = (k / n, 1.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum((math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1)))
        (lo, hi) = (mid, hi) if cdf > 0.05 else (lo, mid)
    return hi

def signature(r):
    return (tuple(r.get('test_closed_routes') or []), bool(r.get('test_admitted')), bool(r.get('cancelled_setup')), (r.get('answer_playbacks') or 0) > 0, tuple(sorted(set(r.get('waiting') or []))), (r.get('released') or 0) > 0, (r.get('recover_playbacks') or 0) > 0)

def bad_outcome(r):
    """Scenario-specific user-visible failure, judged identically in every arm."""
    s = r['scenario']
    if s == 'N1_resume':
        return not (r.get('answer_playbacks') or 0) > 0
    if s == 'N5_question_natural':
        return not (r.get('answer_playbacks') or 0) > 0
    if s in ('N2a_third_natural', 'N5_backchannel_natural', 'N4_echo_fault'):
        return bool(r.get('cancelled_setup'))
    if s == 'N3_stop_fault':
        return (r.get('answer_playbacks') or 0) > 0 or (r.get('recover_playbacks') or 0) > 0
    if s == 'N5_wait_natural':
        return not (r.get('answer_playbacks') or 0) > 0
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--control', default='off')
    ap.add_argument('--treatment', default='on_max1')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    prereg = json.loads((EXP / 'prereg.json').read_text())
    (rows, raw) = load(args.runs)
    ok = [r for r in rows if r.get('status') == 'ok']
    by = {(r['arm'], r['rep'], r['trial_id']): r for r in ok}
    (C, T) = (args.control, args.treatment)
    rep_ids = sorted({r['rep'] for r in ok})
    report = {'control': C, 'treatment': T, 'rows': len(raw), 'ok_rows': len(ok), 'harness_errors': defaultdict(int), 'criteria': {}}
    for r in rows:
        if r.get('status') != 'ok':
            report['harness_errors'][r['arm']] += 1
    excl = defaultdict(lambda : defaultdict(int))
    for r in rows:
        if r.get('status') != 'ok':
            excl[r['arm']][str(r.get('status'))] += 1
    attempts = defaultdict(int)
    for r in raw:
        attempts[r['arm'], r['rep'], r['trial_id']] += 1
    report['exclusions_by_arm'] = {a: dict(v) for (a, v) in excl.items()}
    report['retries'] = sum((v - 1 for v in attempts.values() if v > 1))
    report['units_over_two_attempts'] = [list(k) for (k, v) in attempts.items() if v > 2]
    report['ok_units_by_arm'] = {a: sum((1 for r in ok if r['arm'] == a)) for a in sorted({r['arm'] for r in rows})}

    def paired(scenarios, fn, need=None):
        """item -> list of fn(T) - fn(C) over (trial, rep) pairs present in both arms."""
        out = defaultdict(list)
        for ((arm, rep, tid), rc) in by.items():
            if arm != C or rc['scenario'] not in scenarios:
                continue
            rt = by.get((T, rep, tid))
            if rt is None or (need and (not (need(rc) and need(rt)))):
                continue
            (a, b) = (fn(rc), fn(rt))
            if a is None or b is None:
                continue
            out[rc['item']].append(float(b) - float(a))
        return out

    def rate(arm, scenarios, fn, need=None):
        xs = [fn(r) for r in ok if r['arm'] == arm and r['scenario'] in scenarios and (need is None or need(r))]
        xs = [x for x in xs if x is not None]
        return (sum(map(bool, xs)), len(xs))
    valid = lambda r: (r.get('uay') or 0) > 0
    dead = lambda r: bool(r.get('dead_silence'))
    d = paired(POSITIVE, dead, need=valid)
    (pt, lo, hi) = boot_ci(d)
    m1 = {'control_rate': rate(C, POSITIVE, dead, valid), 'treatment_rate': rate(T, POSITIVE, dead, valid), 'paired_items': len(d), 'paired_units': sum(map(len, d.values())), 'improvement': None if pt is None else -pt, 'ci95': None if pt is None else [-hi, -lo]}
    m1['validity_rate'] = {a: rate(a, POSITIVE, valid) for a in (C, T)}
    itt = paired(POSITIVE, lambda r: valid(r) and dead(r))
    (p2, l2, h2) = boot_ci(itt)
    m1['itt_improvement'] = None if p2 is None else [-p2, -h2, -l2]
    report['M1_dead_silence'] = m1
    report['M1_by_scenario'] = {s: {'control': rate(C, (s,), dead, valid), 'treatment': rate(T, (s,), dead, valid)} for s in POSITIVE}
    lat = [r['cut_to_first_playback_ms'] for r in ok if r['arm'] == T and r['scenario'] in POSITIVE and r.get('first_play_after_cut_is_recover') and (r.get('cut_to_first_playback_ms') is not None)]
    report['recovery_latency_ms'] = {'n': len(lat), 'median': float(np.median(lat)) if lat else None, 'p90': float(np.percentile(lat, 90)) if lat else None}
    fu = paired(('P_timeout',), lambda r: r.get('followup_answered'), need=valid)
    (p3, l3, h3) = boot_ci(fu)
    report['followup_answered'] = {'control': rate(C, ('P_timeout',), lambda r: r.get('followup_answered'), valid), 'treatment': rate(T, ('P_timeout',), lambda r: r.get('followup_answered'), valid), 'paired_diff': p3, 'ci95': [l3, h3] if p3 is not None else None}
    fl = paired(('P_timeout',), lambda r: r.get('followup_latency_ms'), need=valid)
    (p4, l4, h4) = boot_ci(fl)
    report['followup_latency_diff_ms'] = {'mean': p4, 'ci95': [l4, h4] if p4 is not None else None}
    fr = lambda r: (r.get('recover_playbacks') or 0) > 0 or bool(r.get('recovery_events'))
    s1 = {}
    for arm in (C, T):
        (k, n) = rate(arm, NEGATIVE, fr)
        s1[arm] = {'k': k, 'n': n, 'upper95': upper95(k, n)}
    s1['by_scenario'] = {s: rate(T, (s,), fr) for s in NEGATIVE}
    report['S1_false_recovery'] = s1
    s2 = {}
    for s in NEGATIVE:
        dd = paired((s,), bad_outcome)
        (p, l, h) = boot_ci(dd)
        s2[s] = {'control': rate(C, (s,), bad_outcome), 'treatment': rate(T, (s,), bad_outcome), 'paired_diff': p, 'ci95': [l, h] if p is not None else None}
    allneg = paired(NEGATIVE, bad_outcome)
    (p, l, h) = boot_ci(allneg)
    s2['pooled'] = {'paired_diff': p, 'ci95': [l, h] if p is not None else None, 'items': len(allneg)}
    report['S2_bad_outcome'] = s2
    jitter = {}
    for arm in (C, T):
        diff = n = 0
        for ((a, rep, tid), r) in by.items():
            if a != arm or r['scenario'] not in NEGATIVE or rep != rep_ids[0]:
                continue
            for other in rep_ids[1:]:
                r2 = by.get((arm, other, tid))
                if r2 is not None:
                    n += 1
                    diff += signature(r) != signature(r2)
        jitter[arm] = {'disagree': diff, 'pairs': n}
    cross = sum((signature(r) != signature(by[T, rep, tid]) for ((a, rep, tid), r) in by.items() if a == C and r['scenario'] in NEGATIVE and ((T, rep, tid) in by)))
    crossn = sum((1 for ((a, rep, tid), r) in by.items() if a == C and r['scenario'] in NEGATIVE and ((T, rep, tid) in by)))
    report['S2_signature_jitter'] = {'within_arm': jitter, 'cross_arm': {'disagree': cross, 'pairs': crossn}}
    la = paired(('N5_question_natural', 'N1_resume', 'N5_wait_natural'), lambda r: r.get('resume_end_to_first_answer_ms') if r['scenario'] == 'N1_resume' else r.get('test_end_to_first_answer_ms'))
    (p, l, h) = boot_ci(la)
    report['S3_latency_diff_ms'] = {'mean': p, 'ci95': [l, h] if p is not None else None, 'items': len(la)}
    err = lambda r: bool(r.get('errors') or r.get('page_errors'))
    report['S4_errors'] = {arm: rate(arm, tuple({r['scenario'] for r in ok}), err) for arm in (C, T)}
    rp = lambda r: (r.get('recover_playbacks') or 0) > 0
    report['S5_resume_collision'] = {arm: rate(arm, ('N1_resume',), rp) for arm in (C, T)}
    report['S5_resume_recovery_events'] = {arm: rate(arm, ('N1_resume',), fr) for arm in (C, T)}
    adm = lambda r: bool(r.get('test_admitted'))
    report['R_third_fault_conditional'] = {arm: {'admitted': rate(arm, ('N2b_third_fault',), adm), 'recovered_given_admitted': rate(arm, ('N2b_third_fault',), fr, adm)} for arm in (C, T)}
    report['D_resume_late'] = {arm: {'recovered': rate(arm, ('D_resume_late',), fr), 'resume_answered': rate(arm, ('D_resume_late',), lambda r: (r.get('answer_playbacks') or 0) > 0)} for arm in sorted({r['arm'] for r in ok})}
    report['S_double_cut'] = {arm: {'recover_playbacks': [r.get('recover_playbacks') for r in ok if r['arm'] == arm and r['scenario'] == 'S_double_cut']} for arm in sorted({r['arm'] for r in ok})}
    report['requests'] = {arm: {'omni': sum((r.get('omni_requests') or 0 for r in ok if r['arm'] == arm)), 'decision_timeouts': sum((r.get('decisions_timed_out') or 0 for r in ok if r['arm'] == arm)), 'decision_fallbacks': sum((r.get('decisions_fallback') or 0 for r in ok if r['arm'] == arm)), 'injected': sum((len(r.get('injected_faults') or []) for r in ok if r['arm'] == arm))} for arm in sorted({r['arm'] for r in ok})}
    c = report['criteria']
    nn = report['criteria_n'] = {}
    g = prereg['primary']
    (vc, vt) = (m1['validity_rate'][C], m1['validity_rate'][T])
    nn['M1_validity_guard'] = {'control_P': vc[1], 'treatment_P': vt[1]}
    if vc[1] == 0 or vt[1] == 0:
        c['M1_validity_guard'] = None
    else:
        (rc_, rt_) = (vc[0] / vc[1], vt[0] / vt[1])
        c['M1_validity_guard'] = rc_ >= 0.6 and rt_ >= 0.6 and (abs(rc_ - rt_) <= 0.15)
    nn['M1'] = {'paired_units': m1['paired_units'], 'paired_items': m1['paired_items'], 'treatment_valid_trials': m1['treatment_rate'][1]}
    if not m1['ci95'] or m1['treatment_rate'][1] == 0:
        c['M1'] = None
    else:
        c['M1'] = bool(m1['ci95'][0] > g['min_ci_low'] and m1['improvement'] >= g['min_improvement'] and (m1['paired_units'] >= g['min_valid_units']) and (m1['treatment_rate'][0] / m1['treatment_rate'][1] <= g['max_treatment_dead_rate']))
        if c['M1_validity_guard'] is not True:
            c['M1'] = None if c['M1_validity_guard'] is None else False
    s1_missing = [k for (k, v) in s1['by_scenario'].items() if v[1] == 0]
    nn['S1'] = {'n': s1[T]['n'], 'missing_scenarios': s1_missing}
    c['S1'] = None if s1[T]['n'] == 0 or s1_missing else s1[T]['k'] <= prereg['side_effects']['S1_false_recovery_max']
    lim = prereg['side_effects']['S2_max_pooled_increase']
    pooled_n = sum(map(len, allneg.values()))
    nn['S2'] = {'pooled_units': pooled_n, 'pooled_items': len(allneg)}
    if s2['pooled']['ci95'] is None:
        c['S2'] = None
    else:
        c['S2'] = bool(s2['pooled']['ci95'][0] <= 0 <= s2['pooled']['ci95'][1] + 1e-12 and s2['pooled']['paired_diff'] <= lim)
        if s2['pooled']['ci95'][1] < 0:
            s2['pooled']['note'] = 'CI 整段 < 0：按 prereg 判 S2 失败，但方向是改善（坏结果减少）而非退化'
    per_n = {k: sum(map(len, paired((k,), bad_outcome).values())) for k in NEGATIVE}
    nn['S2_per_scenario'] = per_n
    per_missing = [k for (k, v) in per_n.items() if v == 0]
    c['S2_per_scenario'] = None if per_missing else all((s2[k]['ci95'][0] <= 0 for k in NEGATIVE))
    within = report['S2_signature_jitter']['within_arm'][C]
    cross_ = report['S2_signature_jitter']['cross_arm']
    nn['S2_jitter'] = {'within_pairs': within['pairs'], 'cross_pairs': cross_['pairs']}
    c['S2_jitter'] = None if not (cross_['pairs'] and within['pairs']) else bool(cross_['disagree'] / cross_['pairs'] <= within['disagree'] / within['pairs'] + 0.05)
    nn['S3'] = {'paired_units': sum(map(len, la.values())), 'items': len(la)}
    c['S3'] = None if report['S3_latency_diff_ms']['ci95'] is None else bool(report['S3_latency_diff_ms']['ci95'][1] <= prereg['side_effects']['S3_latency_ci_high_ms'])
    nn['S4'] = {arm: report['S4_errors'][arm][1] for arm in (C, T)}
    c['S4'] = None if report['S4_errors'][T][1] == 0 or report['S4_errors'][C][1] == 0 else report['S4_errors'][T][0] <= report['S4_errors'][C][0]
    nn['S5'] = {'n': report['S5_resume_collision'][T][1]}
    c['S5'] = None if report['S5_resume_collision'][T][1] == 0 else report['S5_resume_collision'][T][0] == 0
    nn['followup'] = {'paired_units': sum(map(len, fu.values())), 'items': len(fu)}
    c['followup'] = None if report['followup_answered']['ci95'] is None else bool(report['followup_answered']['ci95'][0] >= -prereg['side_effects']['followup_max_drop'])
    gating = ('M1', 'S1', 'S2', 'S2_per_scenario', 'S2_jitter', 'S3', 'S4', 'S5', 'followup')
    c['no_data'] = [k for k in gating if c[k] is None]
    c['failed'] = [k for k in gating if c[k] is False]
    c['PASS'] = all((c[k] is True for k in gating))
    report['harness_errors'] = dict(report['harness_errors'])
    text = json.dumps(report, ensure_ascii=False, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)
if __name__ == '__main__':
    main()
