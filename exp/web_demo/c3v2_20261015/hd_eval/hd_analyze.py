"""Behaviour scoring of HumDial runs (train/dev) per scenario x language x arm, paired vs base.

usage: python hd_analyze.py RUN_DIR [RUN_DIR...] --manifest manifest.json [--base base] [--out summary.json]

Behaviour per scenario (decided from engine events, not from the route label alone):
  respond  (interruption scenarios): a response started by the decision, covering >= 85 % of
           the utterance, with no premature action.
  stop     (Silence or Termination): playback cancelled; 'stop_silent' also requires that no
           response started within the settle window (a wait release that answers counts as fail).
  keep     (backchannel, third-party, speech to others): no cancel, no response, no recovery prompt.
  pause    (listening): one response after the whole question, no premature action.
  tp_before (listening): nothing started before the user's own question, a response after it.
"""
import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

KIND = {'Follow-up Questions': 'respond', 'Negation or Dissatisfaction': 'respond',
        'Repetition Requests': 'respond', 'Topic Switching': 'respond',
        'Silence or Termination': 'stop', 'User Real-time Backchannels': 'keep',
        'Third-party Speech': 'keep', 'Third-party Speech_after': 'keep',
        'Speech Directed at Others': 'keep', 'Pause Handling': 'pause',
        'Third-party Speech_before': 'tp_before'}


def score(item, arm):
    mt = arm['metrics']
    if not mt.get('valid'):
        return None
    t_u = arm['timing']['t_u']
    begun = [b for b in arm['begun'] if b['t'] >= t_u - 1e-6]
    resp = [b for b in begun if b['stage'] == 'response']
    recover = [b for b in begun if b['stage'] == 'recover']
    cancel = mt['cancel_t'] is not None
    kind = KIND[item['category']]
    v1 = arm['timing']['voiced'][1]
    # A response that starts before the user has finished answers a fragment (premature).
    early_resp = bool(resp) and resp[0]['t'] < v1 - 0.1
    out = {'recover': float(bool(recover)), 'cancel': float(cancel), 'responded': float(bool(resp)),
           'route': mt['final_route'], 'resp_lat': mt['response_latency'], 'stop_lat': mt['stop_latency'],
           'premature': float(early_resp), 'via_release': float(mt['response_via'] == 'release'),
           'multi_resp': float(len(resp) >= 2)}
    if kind in ('respond', 'pause'):
        ok = bool(resp) and not early_resp and (mt['coverage'] or 0) >= 0.85
    elif kind == 'stop':
        ok = cancel
        out['stop_silent'] = float(cancel and not resp)
    elif kind == 'keep':
        ok = not cancel and not resp and not recover
    else:
        second = t_u + item['second_start_rel']
        early = [b for b in begun if b['t'] < second - 0.05]
        late = [b for b in resp if b['t'] >= second - 0.05]
        ok = not early and bool(late)
    out['ok'] = float(ok)
    return out


def load(dirs):
    units = {}
    for d in dirs:
        for path in sorted(glob.glob(str(Path(d) / 'units.w*.jsonl'))):
            for line in open(path, encoding='utf-8'):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                prev = units.get(r['unit'])
                if prev is None or not prev.get('valid') or r.get('valid'):
                    units[r['unit']] = r
    return units


def boot(diffs, B=2000, seed=7):
    if not diffs:
        return [None, None]
    a = np.array(diffs)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (B, len(a)))].mean(axis=1)
    return [round(float(np.quantile(means, .025)), 4), round(float(np.quantile(means, .975)), 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='+')
    p.add_argument('--manifest', required=True)
    p.add_argument('--base', default='base')
    p.add_argument('--out')
    args = p.parse_args()
    items = {it['uid']: it for it in json.load(open(args.manifest))['items']}
    units = load(args.dirs)
    arms = sorted({a for r in units.values() for a in r['arms']})
    # per item: arm -> list of score dicts over reps
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    invalid = collections.Counter()
    for r in units.values():
        it = items[r['uid']]
        for a, v in r['arms'].items():
            s = score(it, v)
            if s is None:
                invalid[a] += 1
                continue
            per[r['uid']][a].append(s)
    cells = collections.defaultdict(list)
    for uid in per:
        it = items[uid]
        cells[(it['category'], it['lang'])].append(uid)
        cells[(it['category'], 'all')].append(uid)
        cells[(KIND[it['category']], 'ALL')].append(uid)
    summary = {'units': len(units), 'invalid': dict(invalid), 'arms': arms, 'cells': {}}

    def mean(uid, a, key):
        xs = [s[key] for s in per[uid].get(a, []) if s.get(key) is not None]
        return float(np.mean(xs)) if xs else None
    for (cat, lang), uids in sorted(cells.items()):
        row = {'n': len(uids)}
        for a in arms:
            vals = [mean(u, a, 'ok') for u in uids]
            vals = [v for v in vals if v is not None]
            entry = {'ok': round(float(np.mean(vals)), 4) if vals else None}
            for key in ('recover', 'premature', 'via_release', 'multi_resp', 'cancel', 'responded'):
                entry[key] = round(float(np.mean([mean(u, a, key) or 0 for u in uids])), 4)
            if KIND.get(cat, cat) == 'stop':
                sv = [mean(u, a, 'stop_silent') for u in uids]
                sv = [v for v in sv if v is not None]
                entry['stop_silent'] = round(float(np.mean(sv)), 4) if sv else None
            lat = [s['resp_lat'] for u in uids for s in per[u].get(a, []) if s['resp_lat'] is not None]
            st = [s['stop_lat'] for u in uids for s in per[u].get(a, []) if s['stop_lat'] is not None]
            entry['resp_lat_p50'] = round(float(np.median(lat)), 3) if lat else None
            entry['stop_lat_p50'] = round(float(np.median(st)), 3) if st else None
            if a != args.base:
                d = [mean(u, a, 'ok') - mean(u, args.base, 'ok') for u in uids
                     if mean(u, a, 'ok') is not None and mean(u, args.base, 'ok') is not None]
                entry['diff'] = round(float(np.mean(d)), 4) if d else None
                entry['ci'] = boot(d)
                entry['flips'] = [sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)]
            row[a] = entry
        summary['cells'][f'{cat}|{lang}'] = row
    text = json.dumps(summary, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding='utf-8')
    print('units', len(units), 'invalid', dict(invalid))
    show = [a for a in arms if a != args.base]
    print(f"{'cell':52s} {'n':>4s} {'base':>6s} " + ' '.join(f'{a:>22s}' for a in show))
    for key, row in summary['cells'].items():
        b = row[args.base]
        cols = []
        for a in show:
            e = row[a]
            cols.append(f"{e['ok']:.3f}({e['diff']:+.3f})" + (f" r{e['recover']:.2f}" if e['recover'] else '') if e['ok'] is not None else '-')
        extra = f" silent={b.get('stop_silent')}" if 'stop_silent' in b else ''
        print(f"{key:52s} {row['n']:4d} {b['ok']:.3f} " + ' '.join(f'{c:>22s}' for c in cols) + extra)


if __name__ == '__main__':
    main()
