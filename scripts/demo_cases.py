#!/usr/bin/env python3
"""List private actual-call cases, human-label them, and replay against local Omni.

No model is called by list/label. Replay is serial, explicit and writes a NEW
private report, never modifies captures or infers gold labels from observations.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from demo_cases import load_case, restore_request, private_write, json_bytes
from control_labels import LABELS, parse_label, ROUTE_PROTOCOL


def case_path(root, case_id):
    path = (root / "captures" / case_id).resolve()
    if path.parent != (root / "captures").resolve() or case_id != path.name:
        raise ValueError("Expected a case ID, not a path")
    return path


def cases(root, *, kind=None, session=None, status=None, reviewed=False):
    result = []
    for path in sorted((root / "captures").glob("*/case.json"), reverse=True):
        if path.parent.name.startswith("."):
            continue
        case = load_case(path.parent)
        if kind and case["kind"] != kind:
            continue
        if session and case["context"].get("session_id") != session:
            continue
        if status and case["outcome"]["status"] != status:
            continue
        if reviewed and not (path.parent / "review.json").exists():
            continue
        result.append((path.parent, case))
    return result


def label(path, expected, note, source="human"):
    case = load_case(path)
    if expected not in LABELS.get(case["kind"], ()):
        raise ValueError("Only control-call cases accept gold labels, matching that control's label set")
    if not note.strip():
        raise ValueError("A human review note is required")
    if source not in ("human", "synthetic_fixture"):
        raise ValueError("Unknown review source")
    review = {"version": "demo-case-review-v1", "case_id": case["case_id"],
              "expected": expected, "note": note, "source": source,
              "utc": datetime.now(timezone.utc).isoformat()}
    # Exclusive creation: cannot silently overwrite an existing human decision.
    private_write(path / "review.json", json_bytes(review))
    return review


async def replay(args, selected):
    from stream_transport import text_stream, audio_stream
    if not selected:
        raise ValueError("No cases selected; empty suites cannot pass")
    # Do not load today's environment/model payload. Saved sampling parameters,
    # full prompts, history and audio are the default reproduction contract.
    replacement = None
    if args.current_prompt:
        import yaml
        base = yaml.safe_load((ROOT / "src/config.yaml").read_text())
        demo = yaml.safe_load((ROOT / "configs/demo_chat.yaml").read_text())
        replacement = {**base["prompts"], **demo["prompts"]}
    report_dir = args.root / "replays" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8])
    report_dir.mkdir(parents=True, mode=0o700)
    rows = []
    for path, case in selected:
        row = {"case_id": case["case_id"], "kind": case["kind"], "pass": None,
               "original_status": case["outcome"]["status"], "prompt": "current" if replacement else "saved"}
        try:
            payload = restore_request(path, case)
            if replacement:
                if case["kind"] not in LABELS or case["kind"] not in replacement:
                    raise ValueError("--current-prompt is only for named control calls")
                payload["messages"][0]["content"] = replacement[case["kind"]]
                if case["kind"] == "input_route":
                    from guarded_turns import route_messages
                    audio = [c for c in payload["messages"][1]["content"] if c.get("type") != "text"]
                    if len(audio) != 1:
                        raise ValueError("Expected exactly one captured routing audio block")
                    payload["messages"] = route_messages(replacement["input_route"], audio[0],
                        playing=case["context"].get("playing", False))
                    from module import qwen_text_payload
                    current = qwen_text_payload(payload["messages"], route=True)
                    for key in ("presence_penalty", "frequency_penalty"):
                        payload[key] = current[key]
                    row["sampling"] = "current_route_penalties; other saved parameters retained"
            review_path = path / "review.json"
            if review_path.exists():
                review = json.loads(review_path.read_text())
                if (review.get("version") != "demo-case-review-v1"
                        or review.get("case_id") != case["case_id"]
                        or review.get("expected") not in LABELS.get(case["kind"], ())):
                    raise ValueError("Invalid human review")
                row["expected"] = review["expected"]
                row["review_source"] = review.get("source")
            if case["kind"] == "tts":
                # Same strict text proof as production; does not prove acoustic
                # pronunciation fidelity. Never compare stochastic PCM bytes.
                count = 0
                async def audio_call():
                    nonlocal count
                    async for chunk in audio_stream(args.url, payload, args.timeout,
                                                    expected_text=case["tts_expected_text"]):
                        count += len(chunk.pcm) // 2
                await asyncio.wait_for(audio_call(), args.timeout)
                if not count:
                    raise ValueError("Empty audio")
                row.update(saved_contract_pass=True, audio_samples=count)
            else:
                async def text_call():
                    parts = []
                    async for part in text_stream(args.url, payload, args.timeout):
                        parts.append(part)
                        if sum(map(len, parts)) > 32768:
                            raise ValueError("Oversized replay response")
                    return "".join(parts)
                raw = await asyncio.wait_for(text_call(), args.timeout)
                row["observed"] = raw
                if "expected" in row:
                    row["pass"] = parse_label(case["kind"], raw, legacy_route=(
                        not replacement and case["kind"] == "input_route"
                        and case["context"].get("route_protocol") != ROUTE_PROTOCOL)) == row["expected"]
        except Exception as exc:
            row.update(error_type=type(exc).__name__)
            row["pass"] = False
        rows.append(row)
        private_write(report_dir / (case["case_id"] + ".json"), json_bytes(row))
        # Output may contain conversation text; use only in the private terminal.
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = {"version": "demo-case-replay-v1", "scope": "single-call; not end-to-end turn or acoustic regression",
              "cases": rows, "labelled": sum("expected" in r for r in rows),
              "failures": sum(r["pass"] is False for r in rows)}
    private_write(report_dir / "report.json", json_bytes(report))
    print(f"Report: {report_dir / 'report.json'}")
    return 1 if report["failures"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "exp/demo_cases")
    subs = parser.add_subparsers(dest="command", required=True)
    listing = subs.add_parser("list", help="Newest first; no inference; bounded display")
    listing.add_argument("--kind", choices=[*LABELS, "response", "tts", "shift_re"])
    listing.add_argument("--session")
    listing.add_argument("--status", choices=["completed", "cancelled", "error"])
    listing.add_argument("--reviewed", action="store_true")
    listing.add_argument("--latest", type=int, default=20)
    marking = subs.add_parser("label", help="Promote a control call by attaching a human gold label")
    marking.add_argument("case_id")
    marking.add_argument("--expected", required=True)
    marking.add_argument("--note", required=True)
    marking.add_argument("--source", choices=["human", "synthetic_fixture"], default="human",
                         help="Use synthetic_fixture only for a known, self-authored test input")
    running = subs.add_parser("replay", help="Explicit real inference, sequential; does not change labels")
    selection = running.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", dest="case_id")
    selection.add_argument("--reviewed", action="store_true")
    running.add_argument("--url", default="http://127.0.0.1:10004/v1/chat/completions")
    running.add_argument("--current-prompt", action="store_true",
                         help="Use current control prompt; input_route also uses current route penalties")
    running.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.command == "list":
        if args.latest < 1:
            parser.error("--latest must be positive")
        for path, case in cases(args.root, kind=args.kind, session=args.session,
                                status=args.status, reviewed=args.reviewed)[:args.latest]:
            print(json.dumps({"id": case["case_id"], "kind": case["kind"],
                "session": case["context"].get("session_id"), "turn": case["context"].get("turn"),
                "status": case["outcome"]["status"], "observed": case["outcome"]["text"][:120],
                "tts_input": (case.get("tts_expected_text") or "")[:120],
                "reviewed": (path / "review.json").exists()}, ensure_ascii=False))
        return 0
    if args.command == "label":
        print(json.dumps(label(case_path(args.root, args.case_id), args.expected, args.note, args.source), ensure_ascii=False))
        return 0
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.case_id:
        path = case_path(args.root, args.case_id)
        selected = [(path, load_case(path))]
    else:
        selected = cases(args.root, reviewed=True)
    return asyncio.run(replay(args, selected))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"Case operation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
