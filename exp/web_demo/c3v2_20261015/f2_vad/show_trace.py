# Migrated to c3v2; no artifact integrity hashing.
"""Print compact traces of one unit for given arms."""
import json, glob, sys
(d, unit) = (sys.argv[1], sys.argv[2])
arms = sys.argv[3].split(',')
for p in glob.glob(d + '/units.w*.jsonl'):
    for line in open(p, encoding='utf-8'):
        u = json.loads(line)
        if u['unit'] != unit:
            continue
        print(u['unit'], u['category'], 'expected', u['expected'])
        for a in arms:
            x = u['arms'][a]
            tm = x['timing']
            print(f"--- {a} t_u={tm['t_u']} voiced={tm['voiced']} hole={tm['hole']} pair={tm['pair']} end_window={tm.get('end_window')}")
            ev = [(t, 'C:' + e, dd.get('input_id'), dd.get('reason')) for (e, t, dd) in x['controls'] if e in ('vad_start', 'vad_done', 'speech_cancelled', 'input_waiting')]
            for (e, t, dd) in x['rows']:
                if e == 'input_dispatch':
                    ev.append((t, 'dispatch', dd.get('input_id'), f"rev{dd.get('revision')} closed={dd.get('closed')} win={dd.get('vad_end_silence_ms')} merges={dd.get('vad_end_merges')}"))
                elif e in ('input_decision', 'input_decision_stale', 'input_decision_deferred', 'input_vad_end_merged'):
                    ev.append((t, e.replace('input_', ''), dd.get('input_id'), f"rev{dd.get('revision')} closed={dd.get('closed')} route={dd.get('route')} tr={dd.get('transcript', '')!s:.40} dfr={dd.get('deferred_from')}"))
            for b in x['begun']:
                if b['stage'] == 'response':
                    ev.append((b['t'], 'RESPONSE', None, b['samples']))
            for row in sorted(ev, key=lambda r: r[0] or 0):
                if row[0] is not None and row[0] >= tm['t_u'] - 0.05:
                    print('   ', row)
            m = x['metrics']
            print('    metrics:', {k: m.get(k) for k in ('clean_success', 'route_correct', 'behavior_ok', 'fragmented', 'premature_action', 'stale', 'final_route', 'stop_latency', 'response_latency', 'merged', 'deferred')})
