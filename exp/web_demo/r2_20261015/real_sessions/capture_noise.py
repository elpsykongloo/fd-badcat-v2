"""Noise floor for capture A/B: rerun the base prompt once more and count base vs base2 label flips."""
import collections, json, re, sys
sys.argv = [sys.argv[0], '12']
src = open('/root/autodl-tmp/fd-badcat-r2/exp/web_demo/r2_20261015/real_sessions/capture_ab.py', encoding='utf-8').read()
src = src.replace("PROMPTS = {'base': base, 'wr3': arms['wr3']['prompts']['input_route']}",
                  "PROMPTS = {'base': base, 'wr3': arms['wr3']['prompts']['input_route'], 'base2': base}")
cut = src.index('sess = {c:')
exec(compile(src[:cut], 'capture_ab', 'exec'))
trans = collections.Counter(); flips = []
for cid, d in cases:
    a, b = done.get((cid, 'base')), done.get((cid, 'base2'))
    if not a or not b or a['out'].startswith('ERROR') or b['out'].startswith('ERROR'):
        continue
    la, lb = label(a['out'])[0], label(b['out'])[0]
    trans[(la, lb)] += 1
    if la != lb:
        flips.append([cid, la, lb, label(a['out'])[1][:40]])
s = {'pairs': sum(trans.values()), 'flips': len(flips),
     'transitions': {f'{k[0]}->{k[1]}': v for k, v in sorted(trans.items(), key=lambda x: -x[1]) if k[0] != k[1]}}
(OUT / 'noise_base2.json').write_text(json.dumps(s, ensure_ascii=False, indent=1))
print(json.dumps(s, ensure_ascii=False, indent=1))
