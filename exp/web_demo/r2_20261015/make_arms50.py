"""Build hd_eval/arms_wr50.json: wr5 (wr3 broadened: noting things down) and wr0 (neutral perturbation control)."""
import json, yaml
from pathlib import Path
R = Path('/root/autodl-tmp/fd-badcat-r2')
D = R / 'exp/web_demo/r2_20261015'
p = yaml.safe_load((R / 'configs/demo_chat.yaml').read_text(encoding='utf-8'))['prompts']['input_route']
A = "stop_only：明确要求停止或别说了，没有同时提出新的问题、纠正或要求。停止本身不需要回答。"
assert p.count(A) == 1
wr5 = p.replace(A, A + "用户让助手先停一下，好让自己去做别的事（如记下来、开个门、接个电话、缓一缓），也是stop_only：用户做完后会再开口。例如“稍等，我把这个记下来”“等一下，我接个电话”“Hold on, let me write that down”。")
wr0 = p.replace(A, A + "（以上定义同样适用于中文和英文。）")
arms = {"base": {"engine": {}, "latency": "pool", "note": "main 69f49bf8 route prompt"},
        "wr5": {"engine": {}, "prompts": {"input_route": wr5}, "latency": "pool", "note": "wr3 broadened: pause-to-do-something incl. noting down -> stop_only"},
        "wr0": {"engine": {}, "prompts": {"input_route": wr0}, "latency": "pool", "note": "neutral perturbation control"}}
(D / 'hd_eval/arms_wr50.json').write_text(json.dumps(arms, ensure_ascii=False, indent=1))
(D / 'route_wr5.txt').write_text(wr5); (D / 'route_wr0.txt').write_text(wr0)
print('ok', len(p), len(wr5), len(wr0))