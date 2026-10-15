"""Build hd_eval/arms_wr.json: base (current route prompt) vs wr1 (pause requests -> stop_only)."""
import json, yaml
from pathlib import Path
R = Path('/root/autodl-tmp/fd-badcat-r2')
p = yaml.safe_load((R / 'configs/demo_chat.yaml').read_text(encoding='utf-8'))['prompts']['input_route']
REPL = [
    ("stop_only：明确要求停止或别说了，没有同时提出新的问题、纠正或要求。停止本身不需要回答。",
     "stop_only：明确要求停止、别说了，或只让助手先等一下、稍等、暂停，包括说明自己要暂时去做别的事（如开门、接电话、查东西），没有同时提出新的问题、纠正或要求。停止本身不需要回答，用户之后会再开口。"),
    ("yield_wait：用户只是要求先等一下或表示自己还没说完，并且没有给出任何可以回应的完整问题或请求；",
     "yield_wait：用户表示自己还没说完、正在想或正在组织语言，并且没有给出任何可以回应的完整问题或请求；"),
    ("或可以确认用户正在表达一个尚未完成的请求/纠正。\n",
     "或可以确认用户正在表达一个尚未完成的请求/纠正。只让助手等一下、稍等，而没有表示自己还要接着说，是stop_only，不是yield_wait。\n"),
    ("只有最后表达的是“我还没说完”之类的等待要求时才是yield_wait。",
     "只有最后表达的是“我还没说完”之类的等待要求时才是yield_wait。“等等”“等一下”后面跟着完整的异议、纠正或评价（如“等等，你说得不对”“Wait, that's not right”）也是yield_ready。"),
    ("“停”“别说了”“stop talking”→stop_only；",
     "“停”“别说了”“stop talking”“等一下”“稍等”“稍等，我去开个门”“等一下，我接个电话”“Hold on, someone's at the door”→stop_only；"),
]
q = p
for a, b in REPL:
    assert q.count(a) == 1, a
    q = q.replace(a, b)
arms = {"base": {"engine": {}, "latency": "pool", "note": "main 69f49bf8 route prompt"},
        "wr1": {"engine": {}, "prompts": {"input_route": q}, "latency": "pool",
                "note": "pause/leave requests -> stop_only; complete objections after 等等 -> yield_ready"}}
out = R / 'exp/web_demo/r2_20261015/hd_eval/arms_wr.json'
out.write_text(json.dumps(arms, ensure_ascii=False, indent=1))
(R / 'exp/web_demo/r2_20261015/route_wr1.txt').write_text(q)
print(len(p), len(q))
