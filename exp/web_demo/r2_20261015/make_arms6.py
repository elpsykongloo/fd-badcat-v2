"""Build wr6 (wr3 + 'wait followed by a question/repeat/correction is not stop_only') arms for train and dev."""
import json
from pathlib import Path
D = Path('/root/autodl-tmp/fd-badcat-r2/exp/web_demo/r2_20261015')
a34 = json.loads((D / 'hd_eval/arms_wr34.json').read_text())
wr3 = a34['wr3']['prompts']['input_route']
S = "也是stop_only：用户回来后会再开口。"
assert wr3.count(S) == 1
wr6 = wr3.replace(S, S + "但“等等”“稍等”后面接着提出问题、要求重复或纠正时，不是stop_only，按后面的内容判断。")
base = a34['base']
mk = lambda **k: {n: {"engine": {}, "prompts": {"input_route": p}, "latency": "pool"} for n, p in k.items()}
(D / 'hd_eval/arms_wr6.json').write_text(json.dumps({"base": base, **mk(wr6=wr6)}, ensure_ascii=False, indent=1))
(D / 'hd_eval/arms_wr36.json').write_text(json.dumps({"base": base, **mk(wr3=wr3, wr6=wr6)}, ensure_ascii=False, indent=1))
(D / 'route_wr6.txt').write_text(wr6)
print('ok', len(wr3), len(wr6))