#!/usr/bin/env python3
"""Manage private demo session diagnostics and injected replay."""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demo_diagnostics import DiagnosticStore, finalize_session
from demo_session_replay import replay_session


def config():
    base = yaml.safe_load((ROOT / "src/config.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load((ROOT / "configs/demo_chat.yaml").read_text(encoding="utf-8"))
    for section, values in overlay.items():
        base.setdefault(section, {}).update(values)
    return base


def main():
    parser = argparse.ArgumentParser(description="Demo session diagnostics")
    parser.add_argument("--exp-root", type=Path, default=ROOT / "exp")
    parser.add_argument("--cases-root", type=Path, default=ROOT / "exp/demo_cases")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--anomalies", action="store_true")
    show = sub.add_parser("show"); show.add_argument("session_id")
    audit = sub.add_parser("audit"); audit.add_argument("session_id")
    review = sub.add_parser("review"); review.add_argument("session_id"); review.add_argument("turn_id")
    review.add_argument("--label", action="append", required=True)
    review.add_argument("--reviewer", required=True); review.add_argument("--note", required=True)
    replay = sub.add_parser("replay"); replay.add_argument("session_id")
    replay.add_argument("--out", type=Path); replay.add_argument("--speed", type=float, default=1)
    prune = sub.add_parser("prune"); prune.add_argument("--older-than-days", "--days",
                                                        dest="older_than_days", type=int, required=True)
    delete = sub.add_parser("delete"); delete.add_argument("session_id")
    args = parser.parse_args()
    store = DiagnosticStore(args.exp_root, args.cases_root)

    if args.command == "list":
        value = store.list_sessions(anomalies_only=args.anomalies)
    elif args.command == "show":
        value = store.session_detail(args.session_id)
    elif args.command == "audit":
        diag = store.session_dir(args.session_id)
        value = finalize_session(diag, diag.parent / "events.jsonl", args.cases_root)
    elif args.command == "review":
        value = store.add_review(args.session_id, args.turn_id, labels=args.label,
                                 reviewer=args.reviewer, note=args.note)
    elif args.command == "replay":
        if args.speed <= 0:
            parser.error("--speed must be positive")
        output = args.out or (store.session_dir(args.session_id) / "replays" /
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        value = asyncio.run(replay_session(args.session_id, store.session_dir(args.session_id),
                            args.cases_root, config(), output, speed=args.speed))
    elif args.command == "prune":
        value = {"removed": store.prune(args.older_than_days)}
    elif args.command == "delete":
        store.delete_session(args.session_id)
        value = {"deleted": args.session_id}
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
