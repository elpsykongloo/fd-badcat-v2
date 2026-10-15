"""Compare arms on replayed real browser sessions: counts and per-input differences."""
import collections
import glob
import json
import sys

run = sys.argv[1]
base = 'base'
units = {}
for p in glob.glob(run + '/units.w*.jsonl'):
    for line in open(p, encoding='utf-8'):
        u = json.loads(line)
        if u.get('valid'):
            units[u['unit']] = u


def closed(arm):
    out = []
    for ev, t, d in arm['rows']:
        if ev == 'input_decision' and d.get('closed'):
            out.append((t, d['route'], d.get('review'), d.get('transcript') or ''))
    return out


def speech(arm):
    resp = sorted((c for c in arm['calls'] if c['role'] == 'response' and not c.get('cancelled')), key=lambda c: c['due'])
    out = []
    ends = {}
    for ev, t, d in arm['controls']:
        if ev in ('speech_cancelled', 'speech_played') and d.get('utterance_id') not in ends:
            ends[d.get('utterance_id')] = (t, d.get('reason') or 'played')
    for ev, t, d in arm['controls']:
        if ev == 'speech_start':
            txt = None
            for c in resp:
                if c['due'] <= t + 1e-6:
                    txt = c.get('out')
            out.append((t, d.get('utterance_id'), ends.get(d.get('utterance_id')), (txt or '')[:90]))
    return out


tot = collections.defaultdict(collections.Counter)
for uid, u in sorted(units.items(), key=lambda kv: kv[1]['duration']):
    print('=' * 110)
    print(uid, 'dur', u['duration'])
    per = {}
    for name, arm in u['arms'].items():
        c = collections.Counter()
        for t, route, rev, tr in closed(arm):
            c['dec_' + route] += 1
        for ev, t, d in arm['rows']:
            if ev in ('input_wait_released', 'input_recover', 'input_wait_recover', 'input_rejected', 'speech_error', 'engine_error'):
                c[ev] += 1
        for ev, t, d in arm['controls']:
            if ev in ('speech_start', 'speech_cancelled', 'input_waiting', 'input_ignored'):
                c[ev] += 1
            if ev == 'speech_cancelled':
                c['cancel_' + str(d.get('reason'))] += 1
        per[name] = c
        tot[name] += c
        print(f'  {name:8s} ' + ' '.join(f'{k}={v}' for k, v in sorted(c.items())))
    b = u['arms'][base]
    bd = {round(t, 1): (r, tr) for t, r, rev, tr in closed(b)}
    for name, arm in u['arms'].items():
        if name == base:
            continue
        ad = closed(arm)
        diffs = []
        for t, r, rev, tr in ad:
            key = min(bd, key=lambda x: abs(x - t)) if bd else None
            if key is None or abs(key - t) > 0.6 or bd[key][0] != r:
                diffs.append((t, r, tr, bd.get(key) if key is not None and abs(key - t) <= 3 else None))
        if diffs:
            print(f'  -- {name} decision diffs vs base ({len(diffs)}):')
            for t, r, tr, bb in diffs[:25]:
                print(f'     {t:7.2f} {r:12s} [{tr[:60]}]  base: {bb}')
        bs = [(round(t, 1), x[:40]) for t, _, _, x in speech(b)]
        asp = [(round(t, 1), x[:40]) for t, _, _, x in speech(arm)]
        if bs != asp:
            print(f'  -- {name} speech differs: base {len(bs)} starts, arm {len(asp)} starts')
            sb = set(bs)
            for s in speech(arm):
                if (round(s[0], 1), s[3][:40]) not in sb:
                    print(f'     +{s[0]:7.2f} end={s[2]} {s[3]}')
            sa = set(asp)
            for s in speech(b):
                if (round(s[0], 1), s[3][:40]) not in sa:
                    print(f'     -{s[0]:7.2f} end={s[2]} {s[3]}')
print('=' * 110)
for name, c in tot.items():
    print('TOTAL', name, ' '.join(f'{k}={v}' for k, v in sorted(c.items())))
