import json, glob, sys
out = sys.argv[1]
arms_want = sys.argv[2].split(',') if len(sys.argv) > 2 else None
for path in sorted(glob.glob(out + '/units.w*.jsonl')):
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        print('=' * 100)
        print(r['unit'], 'dur', r['duration'], 'valid', r['valid'])
        for s in r['segments']:
            print(f'   SEG {s[0] + 0.64:6.2f}-{s[1] + 0.64:6.2f} {s[2][:90]}')
        for name, a in r['arms'].items():
            if arms_want and name not in arms_want:
                continue
            print(f'  -- {name} status={a["status"]} wall={a["timing"]["wall_s"]} kinds={a.get("client_kinds")}')
            ev = []
            for e, t, d in a['controls']:
                if e in ('vad_start', 'vad_done'):
                    continue
                ev.append((t or 0, 'C', e, json.dumps(d, ensure_ascii=False)[:100]))
            for e, t, d in a['rows']:
                if e == 'input_decision':
                    ev.append((t, 'D', f"{d['route']}{'' if d['closed'] else '(open)'}", f"[{d.get('base_label')}/{d.get('review')}] {d['transcript'][:70]}"))
                elif e in ('input_wait_released', 'input_recover', 'input_wait_recover', 'speech_error', 'engine_error', 'input_rejected'):
                    ev.append((t, 'R', e, json.dumps(d, ensure_ascii=False)[:100]))
            for c in a['calls']:
                if c['role'] not in ('input_route', 'input_reply'):
                    ev.append((c['t'], 'M', c['role'] + (' X' if c.get('cancelled') else ''), (c.get('out') or '').replace('\n', ' ')[:110]))
            vs = [(t, e) for e, t, d in a['controls'] if e in ('vad_start', 'vad_done')]
            print('     VAD', ' '.join(f"{'S' if e == 'vad_start' else 'E'}{t:.2f}" for t, e in vs if t is not None))
            for t, k, e, d in sorted(ev, key=lambda x: x[0]):
                print(f'     {t:6.2f} {k} {e:22s} {d}')
