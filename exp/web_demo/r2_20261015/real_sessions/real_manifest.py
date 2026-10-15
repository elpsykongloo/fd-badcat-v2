"""Build the manifest of genuine human demo sessions on casecade-demo (all trees), with captures per session."""
import collections, glob, json, os, re
from pathlib import Path

P = Path('/root/autodl-tmp/fd-badcat-r2/exp/web_demo/r2_20261015/real_sessions')
inv = [json.loads(l) for l in open(P / 'inventory.jsonl')]
cand = {r['dir']: r for r in json.load(open(P / 'cand.json'))}

# Human = main-tree session reached over a real network (RTT >= 5 ms with jitter, not the 81 ms bench injector,
# not the 10-07/10-08 scripted remote browser runs), plus the two 10-10 sessions at ~7 ms RTT reviewed by content.
SCRIPTED_DAYS = ('2026-10-07', '2026-10-08')
human = []
for r in inv:
    if r['tree'] != 'fd-badcat' or r['n_vad'] == 0 or r['rtt_med'] is None or r['rtt_med'] < 5:
        continue
    if (r['utc'] or '')[:10] in SCRIPTED_DAYS:
        continue
    human.append(r)

caps = collections.defaultdict(list)
for f in glob.glob('/root/autodl-tmp/*/exp/demo_cases/captures/*/case.json'):
    try:
        d = json.load(open(f))
    except ValueError:
        continue
    sid = (d.get('context') or {}).get('session_id')
    caps[sid].append({'case': str(Path(f).parent), 'kind': d.get('kind'), 'utc': d.get('utc'),
                      'wavs': len(glob.glob(str(Path(f).parent / '*.wav')))})

def transcripts(d):
    out = []
    for line in open(Path(d) / 'events.jsonl', errors='replace'):
        if '"input_decision"' in line:
            m = re.search(r'"transcript"\s*:\s*"([^"]*)"', line)
            l = re.search(r'"label"\s*:\s*"(\w+)"', line)
            if m and m.group(1):
                out.append([l.group(1) if l else None, m.group(1)])
    return out

rows = []
for r in sorted(human, key=lambda x: x['utc']):
    d = r['dir']
    size = sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file())
    c = caps.get(r['session'], [])
    rows.append({'session': r['session'], 'dir': d, 'utc': r['utc'], 'dur_s': r['dur_s'], 'rtt_med': r['rtt_med'],
                 'rtt_sd': cand.get(d, {}).get('sd'), 'vad_starts': r['n_vad'], 'full_mic': r['mic'],
                 'turn_wavs': r['turn_wavs'], 'aec': (r['settings'] or {}).get('echoCancellation'),
                 'captures': len(c), 'capture_kinds': dict(collections.Counter(x['kind'] for x in c)),
                 'capture_wavs': sum(x['wavs'] for x in c), 'bytes': size,
                 'transcripts': transcripts(d)})
(P / 'human_sessions.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print(len(rows), 'human sessions')
for x in rows:
    print(x['utc'][:16], x['session'][9:17], f"dur={x['dur_s']:7.1f}s rtt={x['rtt_med']:6.1f} vad={x['vad_starts']:3d} mic={int(x['full_mic'])} "
          f"turn_wavs={x['turn_wavs']:2d} caps={x['captures']:3d}/{x['capture_wavs']:3d}wav {x['bytes']/1e6:6.1f}MB aec={x['aec']}")
print('total dur min', round(sum(x['dur_s'] for x in rows) / 60, 1), 'MB', round(sum(x['bytes'] for x in rows) / 1e6, 1),
      'caps', sum(x['captures'] for x in rows), 'utterances', sum(len(x['transcripts']) for x in rows))
