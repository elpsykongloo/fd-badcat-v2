#!/usr/bin/env python3
"""Derive demo-only codec chunk sizing without mutating frozen configurations."""
import argparse
from pathlib import Path

import yaml


def write_config(source, target, chunks):
    if Path(source).resolve() == Path(target).resolve():
        raise ValueError("Demo deployment must not overwrite its source config")
    try:
        initial, steady = map(int, chunks.split(":"))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Demo codec chunks must be INITIAL:STEADY") from exc
    if not 1 <= initial <= 25 or not 4 <= steady <= 25:
        raise ValueError("Demo codec chunks require initial 1..25 and steady 4..25")
    config = yaml.safe_load(Path(source).read_text())
    if [stage.get("stage_id") for stage in config.get("stages", [])] != [0, 1, 2]:
        raise ValueError("Demo codec chunks require the three-stage audio deployment")
    talker = config["stages"][1]
    name = talker["output_connectors"]["to_stage_2"]
    extra = config["connectors"][name].setdefault("extra", {})
    extra.update(initial_codec_chunk_frames=initial, codec_chunk_frames=steady)
    Path(target).write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("chunks")
    args = parser.parse_args()
    write_config(args.source, args.target, args.chunks)
