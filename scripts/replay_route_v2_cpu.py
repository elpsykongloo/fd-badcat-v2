"""CPU-only replay of captured input_route outputs under transcript-first-v2.

For every COMPLETED input_route case in the capture archive, the real model
output (v1, usually fenced and pretty-printed) is de-fenced and rewritten as the
v2 continuation that follows the prefilled `{"transcript":"` prefix, keeping the
exact JSON string literals the model produced (escapes included). Checks:

  1. parse_route(v2, prefilled=True) equals the original parse_route(raw);
  2. the early-close scanner, fed random delta splits of the original raw output
     and of the v2 continuation, closes exactly at the label's closing quote and
     its full parse equals the original parse (never earlier, never different).

No model call, no GPU. Prints a JSON summary; exits 1 on any mismatch.
"""
import argparse
import json
import random
import sys
from collections import Counter
from json.decoder import scanstring
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from control_labels import parse_route, RouteScanner  # noqa: E402

_WS = " \t\n\r"


def _skip_ws(text, i):
    while i < len(text) and text[i] in _WS:
        i += 1
    return i


def _literal(text, i):
    """Return (raw JSON string literal, end index) starting at text[i] == '"'."""
    if i >= len(text) or text[i] != '"':
        raise ValueError("expected string literal")
    _, end = scanstring(text, i + 1, True)
    return text[i:end], end


def v1_to_v2(raw):
    """Rewrite one valid v1 output as the v2 continuation (raw literals preserved)."""
    body = raw.strip()
    if body.startswith("```json\n") and body.endswith("\n```"):
        body = body[8:-4].strip()
    i = _skip_ws(body, 0)
    if body[i:i + 1] != "{":
        raise ValueError("expected object")
    values = []
    for key, sep in (("transcript", ","), ("label", "}")):
        name, i = _literal(body, _skip_ws(body, i + 1))
        if json.loads(name) != key:
            raise ValueError("unexpected key")
        i = _skip_ws(body, i)
        if body[i:i + 1] != ":":
            raise ValueError("expected colon")
        value, i = _literal(body, _skip_ws(body, i + 1))
        values.append(value)
        i = _skip_ws(body, i)
        if body[i:i + 1] != sep:
            raise ValueError("expected separator")
    if body[i + 1:].strip():
        raise ValueError("trailing text")
    transcript, label = values
    return transcript[1:] + ',"label":' + label + "}"


def label_close_offset(text, prefilled):
    """Reference (non-incremental) offset just after the label value literal."""
    body = ('{"transcript":"' + text) if prefilled else text
    shift = len('{"transcript":"') if prefilled else 0
    i = body.index("{")
    for _ in range(4):  # transcript key/value, label key/value
        i = body.index('"', i)
        _, i = scanstring(body, i + 1, True)
    return i - shift


def random_splits(text, rng):
    cuts = sorted(rng.sample(range(1, len(text)), min(len(text) - 1, rng.randint(0, 8)))) if len(text) > 1 else []
    edges = [0, *cuts, len(text)]
    return [text[a:b] for a, b in zip(edges, edges[1:])]


def scan(pieces, prefilled):
    scanner = RouteScanner(prefilled=prefilled)
    consumed = 0
    for piece in pieces:
        consumed += len(piece)
        if scanner.feed(piece):
            return scanner, consumed
    return scanner, None


def check_early(text, prefilled, expected, rng, trials):
    want = label_close_offset(text, prefilled)
    for trial in range(trials):
        # First trial: one character per delta; then random cut points.
        pieces = list(text) if trial == 0 else random_splits(text, rng)
        scanner, consumed = scan(pieces, prefilled)
        if scanner.closed_at != want or consumed is None or consumed < want:
            return False
        result = parse_route(scanner.candidate(), prefilled=prefilled)
        if result is None or list(result.items()) != list(expected.items()):
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--captures", default="exp/demo_cases/captures")
    parser.add_argument("--trials", type=int, default=20, help="random delta splits per output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="optional JSON summary path")
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    status = Counter()
    summary = Counter()
    mismatches = []
    chars = {"v1": 0, "v2": 0}
    for path in sorted(Path(args.captures).glob("*/case.json")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary["unreadable"] += 1
            continue
        if case.get("kind") != "input_route":
            continue
        outcome = case.get("outcome") or {}
        status[outcome.get("status")] += 1
        if outcome.get("status") != "completed":
            continue
        raw = outcome.get("text", "")
        case_id = case.get("case_id", path.parent.name)
        original = parse_route(raw)
        if original is None:
            summary["completed_unparsed"] += 1
            continue
        summary["completed_parsed"] += 1
        try:
            v2 = v1_to_v2(raw)
        except ValueError as exc:
            mismatches.append({"case_id": case_id, "stage": "rewrite", "error": str(exc)})
            continue
        chars["v1"] += len(raw)
        chars["v2"] += len(v2)
        rewritten = parse_route(v2, prefilled=True)
        if rewritten is None or list(rewritten.items()) != list(original.items()):
            mismatches.append({"case_id": case_id, "stage": "v2_parse", "raw": raw[:200], "v2": v2[:200]})
            continue
        summary["v2_parse_equal"] += 1
        if not check_early(raw, False, original, rng, args.trials):
            mismatches.append({"case_id": case_id, "stage": "early_v1", "raw": raw[:200]})
            continue
        summary["early_v1_equal"] += 1
        if not check_early(v2, True, original, rng, args.trials):
            mismatches.append({"case_id": case_id, "stage": "early_v2", "v2": v2[:200]})
            continue
        summary["early_v2_equal"] += 1
        summary["labels_" + original["label"]] += 1
    parsed = summary["completed_parsed"] or 1
    report = {"captures": str(args.captures), "input_route_status": dict(status),
              **dict(summary), "mismatches": len(mismatches), "mismatch_examples": mismatches[:10],
              "mean_chars_v1": round(chars["v1"] / parsed, 2),
              "mean_chars_v2": round(chars["v2"] / parsed, 2),
              "trials_per_output": args.trials, "seed": args.seed}
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
