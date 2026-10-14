# Migrated to c3v2; no artifact integrity hashing.
import json, sys
for path in sys.argv[1:]:
    for l in open(path):
        r = json.loads(l)
        print(r['unit'], r['valid'])
        for (a, v) in r['arms'].items():
            m = v['metrics']
            print(' ', a, v['status'], {k: m.get(k) for k in ('n_inputs', 'fragmented', 'stale', 'premature_action', 'final_route', 'cancel_t', 'response_t', 'response_via', 'coverage', 'deferred', 'merged', 'clean_success', 'behavior_ok', 'engine_errors', 'calls_route', 'calls_reply', 'calls_timeout', 'stop_latency')})
            print('    timing', {k: v['timing'][k] for k in ('t_play', 't_u', 'voiced', 'hole', 'pair', 'wall_s')})
            print('    ctl', [(c[0], c[1]) for c in v['controls'] if c[1] and c[1] > 0.1][:12])
            print('    dec', [(x[0][6:], x[1], x[2].get('route'), x[2].get('closed'), x[2].get('transcript')) for x in v['rows'] if x[0] in ('input_decision', 'input_decision_deferred', 'input_vad_end_merged')][:8])
