# Migrated to c3v2; no artifact integrity hashing.
"""End-of-episode window check: hold_until closed, no deferred left, last end + window <= t_last."""
import json, glob, sys, collections
arms = json.load(open('arms.json'))
win = {k: (v['engine'].get('input_vad_end_ms_playing') or 100) / 1000 for (k, v) in arms.items()}
stat = collections.defaultdict(collections.Counter)
margins = collections.defaultdict(list)
for d in sys.argv[1:]:
    for p in glob.glob(d + '/units.w*.jsonl'):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line)
            for (a, x) in r['arms'].items():
                tm = x['timing']
                ew = tm.get('end_window') or {}
                c = stat[a]
                c['episodes'] += 1
                c['hold_open'] += bool(ew.get('hold_open'))
                c['deferred_left'] += bool(ew.get('deferred_left'))
                c['span_undecided'] += bool(ew.get('span_undecided'))
                c['no_end_window_field'] += not ew
                c['status_not_ok'] += x['status'] != 'ok'
                ends = [t for (ev, t, dd) in x['controls'] if ev == 'vad_done' and t is not None]
                if ends:
                    margins[a].append(tm['t_last'] - (max(ends) + win[a] - 0.1))
                dfr = sum((1 for (ev, *_) in x['rows'] if ev == 'input_decision_deferred'))
                mg = sum((1 for (ev, *_) in x['rows'] if ev == 'input_vad_end_merged'))
                c['deferred_events'] += dfr
                c['merged_events'] += mg
for a in sorted(stat):
    mm = sorted(margins[a])
    print(a, dict(stat[a]), 'min_margin_s=%.3f' % (mm[0] if mm else float('nan')))
