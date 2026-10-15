"""Rule scoring for the HumDial test whole-dialogue run (full_harness.py output).

Per arm and unit, from the event log: speech starts (system replies), cancels, segment
times (+0.64 s mic offset). Category rules (seg k = k-th speech segment of the json):
  pause                         one reply, first start >= s1.end-0.1, none during the break
  ask/deny/repeat/shift         if speaking at s2.start: cancelled by s2.end+1.5; reply after s2.end-0.1
  wait                          if speaking at s2.start: cancelled by s2.end+1.5; no new reply after s2.start
  backchannel/others_..._after  the reply active at s2.start is not cancelled; no new reply after s2.start
  others_talk_to_user_before    no reply in [s1.start, s2.start); reply after s2.end-0.1
  talk_to_others                s2 as ask; s3 as backchannel
usage: python hdt_analyze.py RUN_DIR [--base base] [--out summary.json] [--diffs diffs.jsonl]
"""
import argparse
import collections
import glob
import json
import random

OFF = 0.64
INTERRUPT = {'ask', 'deny', 'repeat', 'shift'}
IGNORE = {'backchannel', 'others_talk_to_user_after'}


def speeches(arm):
    """[(sid, start, end, end_reason, text)] in start order."""
    starts, ends = [], {}
    for ev, t, d in arm['controls']:
        if t is None:
            continue
        if ev == 'speech_start':
            starts.append((d.get('utterance_id'), t))
        elif ev in ('speech_cancelled', 'speech_played') and d.get('utterance_id') not in ends:
            ends[d.get('utterance_id')] = (t, 'cancel:' + str(d.get('reason')) if ev == 'speech_cancelled' else 'played')
    resp = sorted((c for c in arm['calls'] if c['role'] == 'response' and not c.get('cancelled') and not c.get('timeout')),
                  key=lambda c: c['due'])
    out = []
    for sid, t in starts:
        text = None
        for c in resp:
            if c['due'] <= t + 1e-6:
                text = c.get('out')
        end, why = ends.get(sid, (None, 'open'))
        out.append({'sid': sid, 'start': t, 'end': end, 'why': why, 'text': text})
    return out


def score(unit, arm):
    cat = unit['category']
    segs = [(s + OFF, e + OFF, txt) for s, e, txt in unit['segments']]
    sp = speeches(arm)
    t_last = arm['timing']['t_last']

    def speaking_at(x):
        return next((s for s in sp if s['start'] <= x and (s['end'] is None or s['end'] > x)), None)

    def starts_in(a, b):
        return [s for s in sp if a <= s['start'] < b]

    def stop_check(seg):
        active = speaking_at(seg[0] + 0.05)
        if active is None:
            return {'speaking': False, 'stopped': None, 'stop_lat': None}
        ok = active['end'] is not None and active['why'].startswith('cancel') and active['end'] <= seg[1] + 1.5
        return {'speaking': True, 'stopped': ok, 'stop_lat': round(active['end'] - seg[0], 3) if ok else None,
                'stop_reason': active['why']}

    def ignore_check(seg, until):
        active = speaking_at(seg[0] + 0.05)
        cut = active is not None and active['end'] is not None and active['why'].startswith('cancel') and active['end'] <= until
        new = starts_in(seg[0], until)
        return {'speaking': active is not None, 'cut': cut, 'new_reply': bool(new),
                'ok': not cut and not new}

    r = {'cat': cat, 'lang': unit['lang'], 'n_speech': len(sp)}
    s1 = segs[0]
    first = starts_in(s1[0], segs[1][0] if len(segs) > 1 else t_last + 1)
    r['talkover'] = sum(1 for s in sp for g in segs if g[0] + 0.2 < s['start'] < g[1] - 0.05)
    if cat == 'others_talk_to_user_before':
        s2 = segs[1]
        r['reply_to_other'] = bool(first)
        after = starts_in(s2[1] - 0.1, t_last + 1)
        r['reply'] = bool(after)
        r['reply_lat'] = round(after[0]['start'] - s2[1], 3) if after else None
        r['ok'] = not first and bool(after)
        return r, sp
    r['r1'] = bool([s for s in first if s['start'] >= s1[1] - 0.1])
    r['r1_lat'] = round(first[0]['start'] - s1[1], 3) if first else None
    r['r1_early'] = bool([s for s in first if s['start'] < s1[1] - 0.1])
    if cat == 'pause':
        allr = starts_in(s1[0], t_last + 1)
        r['ok'] = bool(allr) and allr[0]['start'] >= s1[1] - 0.1
        r['multi'] = len(allr) > 1
        return r, sp
    s2 = segs[1]
    end2 = segs[2][0] if len(segs) > 2 else t_last + 1
    if cat in INTERRUPT or cat == 'talk_to_others':
        st = stop_check(s2)
        rep = starts_in(s2[1] - 0.1, end2)
        r.update(st)
        r['reply'] = bool(rep)
        r['reply_lat'] = round(rep[0]['start'] - s2[1], 3) if rep else None
        r['reply_text'] = rep[0]['text'] if rep else None
        r['ok'] = (not st['speaking'] or st['stopped']) and bool(rep)
        if cat == 'talk_to_others':
            ig = ignore_check(segs[2], t_last + 1)
            r['s3'] = ig
            r['ok2'] = r['ok']
            r['ok'] = r['ok'] and ig['ok']
    elif cat == 'wait':
        st = stop_check(s2)
        new = starts_in(s2[0], t_last + 1)
        r.update(st)
        r['new_reply'] = bool(new)
        r['ok'] = (not st['speaking'] or st['stopped']) and not new
    elif cat in IGNORE:
        ig = ignore_check(s2, t_last + 1)
        r.update(ig)
    return r, sp


def boot(diffs, n=2000, seed=7):
    if not diffs:
        return None
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n)], 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run')
    p.add_argument('--base', default='base')
    p.add_argument('--out')
    p.add_argument('--diffs')
    a = p.parse_args()
    units = {}
    for path in glob.glob(a.run + '/units.w*.jsonl'):
        for line in open(path, encoding='utf-8'):
            try:
                u = json.loads(line)
            except ValueError:
                continue
            if u.get('valid'):
                units[u['unit']] = u
    arms = None
    rows = collections.defaultdict(dict)
    for uid, u in units.items():
        arms = arms or list(u['arms'])
        for name, arm in u['arms'].items():
            rows[uid][name] = score(u, arm)
    keys = sorted({(u['lang'], u['category']) for u in units.values()})
    summary = {'units': len(units), 'arms': arms, 'cells': {}}
    print(f'units {len(units)}  arms {arms}')
    hdr = f"{'lang/cat':34s} {'n':>4s} " + ' '.join(f'{x:>16s}' for x in arms)
    print(hdr)
    for lang, cat in keys + [('all', 'all')]:
        ids = [uid for uid, u in units.items() if (lang == 'all' or (u['lang'], u['category']) == (lang, cat))]
        cell = {'n': len(ids)}
        line = f'{lang + "/" + cat:34s} {len(ids):4d} '
        for name in arms:
            ok = [1 if rows[uid][name][0]['ok'] else 0 for uid in ids]
            rate = sum(ok) / len(ok)
            d = [rows[uid][name][0]['ok'] - rows[uid][a.base][0]['ok'] for uid in ids]
            ci = boot(d) if name != a.base else None
            cell[name] = {'ok': round(rate, 4), 'diff': round(sum(d) / len(d), 4), 'ci': ci,
                          'better': sum(1 for x in d if x > 0), 'worse': sum(1 for x in d if x < 0)}
            line += f"{rate * 100:6.1f}" + (f" {sum(d) / len(d) * 100:+5.1f}({cell[name]['better']}/{cell[name]['worse']})" if name != a.base else ' ' * 10)
        print(line)
        summary['cells'][f'{lang}/{cat}'] = cell
    # failure modes and latencies per category (all langs pooled per lang)
    print('\nfailure modes / latency medians')
    for lang, cat in keys:
        ids = [uid for uid, u in units.items() if (u['lang'], u['category']) == (lang, cat)]
        for name in arms:
            rs = [rows[uid][name][0] for uid in ids]
            fm = collections.Counter()
            for r in rs:
                if r['ok']:
                    continue
                if cat == 'pause':
                    fm['no_reply' if r['r1_lat'] is None else 'early' if r['r1_early'] else 'other'] += 1
                elif cat == 'others_talk_to_user_before':
                    fm['replied_other' if r['reply_to_other'] else ''] += 1
                    fm['no_reply' if not r['reply'] else ''] += 1
                elif cat in INTERRUPT or cat == 'talk_to_others':
                    if r['speaking'] and not r['stopped']:
                        fm['no_stop'] += 1
                    if not r['reply']:
                        fm['no_reply'] += 1
                    if cat == 'talk_to_others' and not r['s3']['ok']:
                        fm['s3_cut' if r['s3']['cut'] else ''] += 1
                        fm['s3_reply' if r['s3']['new_reply'] else ''] += 1
                elif cat == 'wait':
                    fm['no_stop' if r['speaking'] and not r['stopped'] else ''] += 1
                    fm['replied' if r['new_reply'] else ''] += 1
                elif cat in IGNORE:
                    fm['cut' if r['cut'] else ''] += 1
                    fm['replied' if r['new_reply'] else ''] += 1
            fm.pop('', None)
            lat = sorted(r['r1_lat'] for r in rs if r.get('r1_lat') is not None)
            st = sorted(r['stop_lat'] for r in rs if r.get('stop_lat') is not None)
            rl = sorted(r['reply_lat'] for r in rs if r.get('reply_lat') is not None)
            spk = sum(1 for r in rs if r.get('speaking'))
            med = lambda v: round(v[len(v) // 2], 2) if v else None
            print(f'  {lang}/{cat:28s} {name:8s} fail={dict(fm)} speaking@s2={spk} r1_lat={med(lat)} stop_lat={med(st)} reply_lat={med(rl)} talkover={sum(r["talkover"] for r in rs)}')
    if a.out:
        json.dump(summary, open(a.out, 'w'), ensure_ascii=False, indent=1)
    if a.diffs:
        with open(a.diffs, 'w', encoding='utf-8') as fh:
            for uid, u in units.items():
                base = rows[uid][a.base]
                for name in arms:
                    if name == a.base:
                        continue
                    r, sp = rows[uid][name]
                    texts_differ = [s['text'] for s in sp] != [s['text'] for s in base[1]]
                    if r['ok'] != base[0]['ok'] or texts_differ:
                        fh.write(json.dumps({'unit': uid, 'arm': name, 'segments': u['segments'],
                                             'ok': r['ok'], 'base_ok': base[0]['ok'],
                                             'arm_speech': sp, 'base_speech': base[1]}, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
