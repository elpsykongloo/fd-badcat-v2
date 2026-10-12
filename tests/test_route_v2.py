"""transcript-first-v2: prefilled parse, payload, repair order, label-close scanner, early close."""
import asyncio
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from control_labels import (parse_route, parse_label, decide_control, decide_input_route, RouteScanner,
                            route_prefill_message, with_route_prefill, route_protocol,
                            ROUTE_PREFIX, ROUTE_PROTOCOL, ROUTE_PROTOCOL_V2)
from guarded_turns import route_options, route_prompt, ROUTE_FORMAT_LINE
import replay_route_v2_cpu as replay

LABELS = ["keep", "stop_only", "yield_wait", "yield_ready"]


def v2(transcript, label, *, ascii=False):
    return json.dumps(transcript, ensure_ascii=ascii)[1:] + ',"label":' + json.dumps(label) + "}"


# ---------- parse ----------

@pytest.mark.parametrize("cont,expected", [
    ('你是谁","label":"yield_ready"}', {"transcript": "你是谁", "label": "yield_ready"}),
    ('","label":"keep"}', {"transcript": "", "label": "keep"}),
    ('停", "label" : "stop_only" }\n', {"transcript": "停", "label": "stop_only"}),
    ('a\\"b\\\\c\\u4f60","label":"yield_wait"}', {"transcript": 'a"b\\c你', "label": "yield_wait"}),
    ('  ","label":"keep"}\n\n', {"transcript": "  ", "label": "keep"}),
])
def test_v2_parse_joins_prefix_and_tolerates_whitespace(cont, expected):
    result = parse_route(cont, prefilled=True)
    assert result == expected and list(result) == ["transcript", "label"]
    assert parse_label("input_route", cont, prefilled=True) == expected["label"]
    # The same continuation without the prefix is not a valid v1 object.
    assert parse_route(cont) is None


V1_INVALID = [
    '{"transcript":"停","label":"stop_only","label":"keep"}',
    '{"transcript":"停","label":"stop_only","extra":1}',
    '{"transcript":"","label":"stop_only"}',
    '{"transcript":"  ","label":"yield_ready"}',
    '{"transcript":"停","label":true}',
    '{"transcript":"停","label":"STOP_ONLY"}',
    '{"transcript":"停","label":"stop_only"',
    '{"transcript":"停","label":"stop_only"} trailing',
    json.dumps({"transcript": "a" * 513, "label": "yield_ready"}, separators=(",", ":")),
]


@pytest.mark.parametrize("raw", V1_INVALID)
def test_v2_keeps_every_v1_constraint(raw):
    assert raw.startswith(ROUTE_PREFIX)
    assert parse_route(raw) is None
    assert parse_route(raw[len(ROUTE_PREFIX):], prefilled=True) is None


@pytest.mark.parametrize("cont", [
    "", "keep", 'yield_ready"}', '停"}', '停","label":"stop_only"', '停","extra":"x","label":"keep"}',
    '```json\n{"transcript":"停","label":"stop_only"}\n```', '停","label":"keep"}}',
    'x' * 4097, None,
])
def test_v2_rejects_malformed_continuations(cont):
    assert parse_route(cont, prefilled=True) is None


def test_v1_fence_branch_is_kept():
    raw = '```json\n{\n  "transcript": "你是谁",\n  "label": "yield_ready"\n}\n```'
    assert parse_route(raw) == {"transcript": "你是谁", "label": "yield_ready"}
    assert parse_route(raw, prefilled=True) is None


@pytest.mark.parametrize("label", LABELS)
def test_v2_matches_v1_for_compact_outputs(label):
    for text in ["你好", "keep stop_only yield_wait yield_ready", '","label":"yield_ready"}', "\\", "😀"]:
        if label != "keep" or text:
            raw = json.dumps({"transcript": text, "label": label}, ensure_ascii=False, separators=(",", ":"))
            assert parse_route(raw[len(ROUTE_PREFIX):], prefilled=True) == parse_route(raw)


# ---------- payload / prefill / repair ----------

def test_prefill_payload_uses_continue_final_message():
    import module
    base = [{"role": "system", "content": "route"}, {"role": "user", "content": "audio"}]
    messages = with_route_prefill(base)
    assert messages[-1] == route_prefill_message() == {"role": "assistant", "content": '{"transcript":"'}
    assert base == [{"role": "system", "content": "route"}, {"role": "user", "content": "audio"}]
    payload = module.qwen_text_payload(messages, route=True, max_tokens=64, continue_final_message=True)
    assert payload["continue_final_message"] is True and payload["add_generation_prompt"] is False
    assert payload["messages"][-1]["content"] == ROUTE_PREFIX and payload["max_tokens"] == 64
    assert (payload["presence_penalty"], payload["frequency_penalty"]) == (0.0, 0.0)
    assert with_route_prefill(base, False) is base


def test_strict_warmup_adapter_forwards_continuation(monkeypatch):
    import module
    posted = []
    class HTTP:
        def post(self, url, **kwargs):
            posted.append(json.loads(kwargs["data"]))
            return SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"choices": [{"message": {"content": 'ok","label":"yield_ready"}'}}]})
    monkeypatch.setattr(module, "_http", lambda: HTTP())
    messages = with_route_prefill([{"role": "user", "content": "x"}])
    raw = module.llm_qwen3o_strict(messages, route=True, continue_final_message=True)
    assert posted == [module.qwen_text_payload(messages, route=True, continue_final_message=True)]
    assert parse_label("input_route", raw, prefilled=True) == "yield_ready"


async def test_v2_valid_result_one_call_prefix_last():
    requests = []
    async def call(messages):
        requests.append(messages)
        return '你是谁","label":"yield_ready"}'
    original = [{"role": "user", "content": "audio placeholder"}]
    label, audit = await decide_control(call, original, "input_route", 2, route_prefill=True)
    assert label == "yield_ready" and len(requests) == 1
    assert requests[0] == original + [route_prefill_message()]
    assert audit["protocol"] == ROUTE_PROTOCOL_V2 and audit["transcript"] == "你是谁"
    assert original == [{"role": "user", "content": "audio placeholder"}]


async def test_v2_repair_prefills_after_the_repair_user_message():
    requests = []
    async def call(messages):
        requests.append(messages)
        return 'invented stop' if len(requests) == 1 else '","label":"keep"}'
    original = [{"role": "user", "content": "audio placeholder"}]
    label, audit = await decide_control(call, original, "input_route", 2, route_prefill=True)
    assert label == "keep" and audit["repaired"] and len(requests) == 2
    first, second = requests
    assert first[-1] == route_prefill_message() and first[:-1] == original
    assert second[:-2] == original and second[-2]["role"] == "user" and "JSON" in second[-2]["content"]
    assert second[-1] == route_prefill_message()
    assert "invented stop" not in str(second)


async def test_prefill_ignored_for_non_route_controls():
    requests = []
    async def call(messages):
        requests.append(messages)
        return "switch"
    label, _ = await decide_control(call, [], "judge", 2, route_prefill=True)
    assert label == "switch" and requests == [[]]


async def test_v2_reply_review_still_uses_transcript():
    calls = []
    async def call(messages, stage):
        calls.append((stage, messages))
        return '对，就这个","label":"keep"}' if stage == "input_route" else "yield_ready"
    label, audit = await decide_input_route(call, [], 2, playing=True, reply_prompt="review",
                                            reply_context="你想听哪个？", route_prefill=True)
    assert label == "yield_ready" and audit["base_label"] == "keep"
    assert [s for s, _ in calls] == ["input_route", "input_reply"]
    assert "对，就这个" in calls[1][1][1]["content"] and calls[1][1][-1]["role"] == "user"


def test_route_protocol_names():
    assert route_protocol(False) == ROUTE_PROTOCOL == "transcript-first-v1"
    assert route_protocol(True) == ROUTE_PROTOCOL_V2 == "transcript-first-v2"


# ---------- config / compact prompt ----------

def test_route_options_defaults_and_validation():
    assert route_options({}) == {"prefill": False, "compact_prompt": False, "early_close": True}
    with pytest.raises(ValueError):
        route_options({"input_route_prefill": "yes"})


def test_compact_prompt_changes_only_the_format_line():
    import yaml
    prompts = yaml.safe_load((ROOT / "configs/demo_chat.yaml").read_text(encoding="utf-8"))["prompts"]
    engine = yaml.safe_load((ROOT / "configs/demo_chat.yaml").read_text(encoding="utf-8"))["engine"]
    assert engine["input_route_prefill"] is False and engine["input_route_compact_prompt"] is False
    assert engine["input_route_early_close"] is True
    original = route_prompt(prompts, {})
    assert original == prompts["input_route"]
    compact = route_prompt(prompts, {"input_route_compact_prompt": True})
    old_lines, new_lines = original.split("\n"), compact.split("\n")
    assert len(old_lines) == len(new_lines)
    changed = [(a, b) for a, b in zip(old_lines, new_lines) if a != b]
    assert changed == [(ROUTE_FORMAT_LINE, prompts["input_route_compact_format"].strip())]
    assert "紧凑" in changed[0][1] and "代码围栏" in changed[0][1]
    with pytest.raises(ValueError):
        route_prompt({"input_route": original}, {"input_route_compact_prompt": True})
    with pytest.raises(ValueError):
        route_prompt({"input_route": "no format line", "input_route_compact_format": "x"},
                     {"input_route_compact_prompt": True})


# ---------- scanner ----------

def feed_all(pieces, prefilled):
    scanner = RouteScanner(prefilled=prefilled)
    fired = []
    seen = 0
    for piece in pieces:
        seen += len(piece)
        if scanner.feed(piece):
            fired.append(seen)
    return scanner, fired


def splits(text, rng):
    if len(text) < 2:
        return [text]
    cuts = sorted(rng.sample(range(1, len(text)), rng.randint(0, min(12, len(text) - 1))))
    edges = [0, *cuts, len(text)]
    return [text[a:b] for a, b in zip(edges, edges[1:])]


TRICKY = ['", "label": "yield_ready"}', '\\', '\\"', '\\\\"', '"}', "{", "}", "[", "]", "\\u0022",
          '","label":"stop_only"', "😀", " ", "停", " ", "\n", "\t", "label", "```"]


def layouts(transcript, label, rng):
    """Equivalent v1 serializations the model can produce, plus the v2 continuation."""
    obj = {"transcript": transcript, "label": label}
    ascii = rng.random() < .3
    compact = json.dumps(obj, ensure_ascii=ascii, separators=(",", ":"))
    pretty = json.dumps(obj, ensure_ascii=ascii, indent=2)
    return [(compact, False), (pretty, False), ("```json\n" + pretty + "\n```", False),
            (compact[len(ROUTE_PREFIX):], True),
            (json.dumps(transcript, ensure_ascii=ascii)[1:] + ', "label": ' + json.dumps(label) + " }", True)]


def random_transcript(rng):
    pool = TRICKY + list("你好吗abc,:{}[]\"\\ ") + ["\\u", "u0022", "퟿", "label\":\""]
    return "".join(rng.choice(pool) for _ in range(rng.randint(0, 12)))


def test_scanner_handles_escapes_across_every_split():
    for transcript in ['a"b', "a\\b", 'x\\"y', "\\u0022", '","label":"yield_ready"}', "😀停"]:
        for text, prefilled in layouts(transcript, "yield_wait", random.Random(1)):
            expected = replay.label_close_offset(text, prefilled)
            for cut in range(len(text) + 1):
                scanner, fired = feed_all([text[:cut], text[cut:]], prefilled)
                assert scanner.closed_at == expected
                assert len(fired) == 1 and fired[0] >= expected
                assert parse_route(scanner.candidate(), prefilled=prefilled) == parse_route(
                    text, prefilled=prefilled)


def test_scanner_unicode_escape_never_closes_string():
    # " is a quote CHARACTER inside the transcript, not a delimiter.
    cont = '\\u0022label\\u0022:\\u0022keep\\u0022","label":"yield_ready"}'
    for pieces in (list(cont), [cont], ['\\', 'u0022label\\u', '0022:\\u0022keep\\u0022",', '"label":"yield_ready"}']):
        scanner, fired = feed_all(pieces, True)
        assert scanner.closed_at == len(cont) - 1
        assert parse_route(scanner.candidate(), prefilled=True) == {
            "transcript": '"label":"keep"', "label": "yield_ready"}


def test_scanner_fuzz_zero_false_triggers():
    rng = random.Random(20261013)
    checked = 0
    for _ in range(1500):
        label = rng.choice(LABELS)
        transcript = random_transcript(rng)
        if label != "keep" and not transcript.strip():
            transcript = "停" + transcript
        for text, prefilled in layouts(transcript, label, rng):
            full = parse_route(text, prefilled=prefilled)
            assert full == {"transcript": transcript, "label": label}
            expected = replay.label_close_offset(text, prefilled)
            scanner, fired = feed_all(splits(text, rng), prefilled)
            # Never before the label closes, exactly once, same decision as the full parse.
            assert scanner.closed_at == expected and len(fired) == 1 and fired[0] >= expected
            assert parse_route(scanner.candidate(), prefilled=prefilled) == full
            # Any strict prefix ending before the label close never fires.
            cut = rng.randint(0, expected - 1) if expected else 0
            assert feed_all(splits(text[:cut], rng), prefilled)[1] == []
            checked += 1
    assert checked == 1500 * 5


def test_scanner_fuzz_garbage_never_contradicts_full_parse():
    """Random/truncated/garbled outputs: whenever early close would accept, a full
    parse of the same output either agrees or (trailing junk) rejects; never differs."""
    rng = random.Random(7)
    pool = ['{', '}', '"', '\\', ',', ':', ' ', '\n', 'transcript', 'label', 'keep', 'yield_ready',
            'stop_only', '停', '```json\n', '"transcript":"', '","label":"', '"}']
    accepted = 0
    for _ in range(20000):
        prefilled = rng.random() < .5
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 14)))
        scanner, fired = feed_all(splits(text, rng), prefilled)
        if not fired:
            continue
        early = parse_route(scanner.candidate(), prefilled=prefilled)
        full = parse_route(text, prefilled=prefilled)
        if early is not None:
            accepted += 1
            assert full is None or full == early
    assert accepted > 0


# ---------- CPU replay helper ----------

@pytest.mark.parametrize("raw", [
    '```json\n{\n  "transcript": "你是谁",\n  "label": "yield_ready"\n}\n```',
    '```json\n{\n  "transcript": "",\n  "label": "keep"\n}\n```',
    '{"transcript":"a\\"b\\u4f60","label":"stop_only"}',
    '```json\n{"transcript": "Wait, \\\\ let me", "label": "yield_wait"}\n```',
])
def test_replay_rewrite_preserves_decision(raw):
    cont = replay.v1_to_v2(raw)
    assert not cont.startswith('{') and cont.endswith("}")
    assert parse_route(cont, prefilled=True) == parse_route(raw) is not None
    assert replay.check_early(raw, False, parse_route(raw), random.Random(0), 10)
    assert replay.check_early(cont, True, parse_route(raw), random.Random(0), 10)


def test_replay_cli_on_synthetic_archive(tmp_path, capsys):
    outputs = {"a": ('```json\n{\n  "transcript": "你是谁",\n  "label": "yield_ready"\n}\n```', "completed"),
               "b": ('```json\n{\n  "transcript": "",\n  "label": "keep"\n}\n```', "completed"),
               "c": ('```json\n{\n  "transcript": "停', "cancelled"),
               "d": ('nonsense', "completed")}
    for name, (text, status) in outputs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "case.json").write_text(json.dumps(
            {"case_id": name, "kind": "input_route", "outcome": {"status": status, "text": text}}))
    assert replay.main(["--captures", str(tmp_path), "--trials", "5"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["completed_parsed"] == 2 and report["v2_parse_equal"] == 2
    assert report["early_v1_equal"] == report["early_v2_equal"] == 2
    assert report["completed_unparsed"] == 1 and report["mismatches"] == 0
    assert report["input_route_status"] == {"completed": 3, "cancelled": 1}
