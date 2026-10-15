"""Re-run every input_route call captured from the 30 human demo sessions with the base and wr3 route prompts.

Only the route prompt is swapped (any continuation suffix the engine appended is kept); user content and audio are
identical across arms. Output: per-call labels and a transition summary.
"""
import collections, concurrent.futures as cf, glob, json, re, sys, urllib.request
from pathlib import Path

ARCH = Path('/root/autodl-tmp/real_sessions_archive_20261015')
D = Path('/root/autodl-tmp/fd-badcat-r2/exp/web_demo/r2_20261015')
OUT = D / 'real_sessions/capture_ab'
OUT.mkdir(parents=True, exist_ok=True)
arms = json.loads((D / 'hd_eval/arms_wr34.json').read_text())
import yaml
base = yaml.safe_load(open('/root/autodl-tmp/fd-badcat-lat/configs/demo_chat.yaml', encoding='utf-8'))['prompts']['input_route']
PROMPTS = {'base': base, 'wr3': arms['wr3']['prompts']['input_route']}
assert PROMPTS['wr3'] != base and PROMPTS['wr3'].startswith(base.split('stop_only：')[0])
SUFFIX_MARK = '\n本次音频包含两部分'
URL = 'http://127.0.0.1:10003/v1/chat/completions'

cases = []
for f in sorted(glob.glob(str(ARCH / 'captures/*/case.json'))):
    d = json.loads(Path(f).read_text())
    if d.get('kind') != 'input_route':
        continue
    msgs = d['request']['messages']
    if not msgs or msgs[0].get('role') != 'system' or not isinstance(msgs[0].get('content'), str):
        continue
    cases.append((Path(f).parent.name, d))

def label(text):
    m = re.search(r'"label"\s*:\s*"(\w+)"', text or '')
    t = re.search(r'"transcript"\s*:\s*"([^"]*)"', text or '')
    return (m.group(1) if m else None), (t.group(1) if t else '')

import base64
MODEL = json.loads(urllib.request.urlopen('http://127.0.0.1:10003/v1/models', timeout=10).read())['data'][0]['id']

def resolve(content, case_dir):
    if not isinstance(content, list):
        return content
    out = []
    for c in content:
        u = (c.get('audio_url') or {}).get('url') if isinstance(c, dict) else None
        if isinstance(u, dict) and 'case_file' in u:
            data = base64.b64encode((case_dir / u['case_file']).read_bytes()).decode()
            c = {'type': 'audio_url', 'audio_url': {'url': u.get('data_prefix', 'data:audio/wav;base64,') + data}}
        out.append(c)
    return out

def call(item):
    cid, d, arm = item
    req = dict(d['request'])
    req['model'] = MODEL
    req['messages'] = [dict(m, content=resolve(m.get('content'), ARCH / 'captures' / cid)) for m in req['messages']]
    sysmsg = req['messages'][0]['content']
    i = sysmsg.find(SUFFIX_MARK)
    suffix = sysmsg[i:] if i >= 0 else ''
    req['messages'] = [{'role': 'system', 'content': PROMPTS[arm] + suffix}] + req['messages'][1:]
    req['stream'] = False
    body = json.dumps(req).encode()
    for attempt in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL, body, {'Content-Type': 'application/json'}), timeout=120)
            out = json.loads(r.read())['choices'][0]['message']['content']
            return cid, arm, out
        except Exception as e:  # noqa: BLE001
            err = repr(e) + ((' ' + e.read().decode(errors='replace')[:300]) if hasattr(e, 'read') else '')
    return cid, arm, 'ERROR ' + err

done = {}
res_path = OUT / 'results.jsonl'
if res_path.exists():
    for line in res_path.open():
        r = json.loads(line); done[(r['case'], r['arm'])] = r
jobs = [(cid, d, arm) for cid, d in cases for arm in PROMPTS if (cid, arm) not in done]
print('cases', len(cases), 'jobs', len(jobs), flush=True)
with res_path.open('a') as fh, cf.ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as ex:
    for n, (cid, arm, out) in enumerate(ex.map(call, jobs), 1):
        r = {'case': cid, 'arm': arm, 'out': out}
        done[(cid, arm)] = r
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        if n % 200 == 0:
            print(n, flush=True)

sess = {c: (d['context'] or {}).get('session_id') for c, d in cases}
trans = collections.Counter(); flips = []; errors = 0
for cid, d in cases:
    a, b = done.get((cid, 'base')), done.get((cid, 'wr3'))
    if not a or not b or a['out'].startswith('ERROR') or b['out'].startswith('ERROR'):
        errors += 1; continue
    (la, ta), (lb, tb) = label(a['out']), label(b['out'])
    trans[(la, lb)] += 1
    if la != lb:
        flips.append({'case': cid, 'session': sess[cid], 'playing': (d['context'] or {}).get('playing'),
                      'base': [la, ta], 'wr3': [lb, tb], 'orig': label(d.get('outcome', {}).get('text'))})
summary = {'cases': len(cases), 'errors': errors,
           'transitions': {f'{k[0]}->{k[1]}': v for k, v in sorted(trans.items(), key=lambda x: -x[1])},
           'labels_base': dict(collections.Counter(k[0] for k in trans.elements())),
           'labels_wr3': dict(collections.Counter(k[1] for k in trans.elements())), 'flips': len(flips)}
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
(OUT / 'flips.json').write_text(json.dumps(flips, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False, indent=1))
for f in flips:
    print(f"{f['base'][0]:>11} -> {f['wr3'][0]:<11} play={int(bool(f['playing']))} | base: {f['base'][1][:40]} | wr3: {f['wr3'][1][:40]}")
