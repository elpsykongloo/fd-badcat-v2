#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rb_golden_manifest.py — the human golden-subset recording sheet
(docs/rb_design.md v2 §4: 100–150 clips, TTS<->human consistency anchor).

Selects 144 arm-A episodes across the key cells (L3/L4/L5/L10 x zh/en x 18)
and emits a recording sheet: what to read, the coarse timing instruction, the
target gap bin, and the episode linkage (the recorded clip REPLACES that
episode's TTS audio in the golden arm). Recording protocol = v1 §3
oversample-and-re-bin: 3 takes per item, offline VAD measures the real gap,
takes are binned; missing bins get re-takes.

  $PY scripts/rb_golden_manifest.py --build exp/rb/build_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GOLDEN_LAYERS = ("L3", "L4", "L5", "L10")
PER_CELL = 18

INSTR = {
    "L3": {"zh": "念完第一句后【很短地】停顿（不到一秒），马上念更正句。",
           "en": "After the first sentence, pause VERY briefly (<1s), then read the correction."},
    "L4": {"zh": "念完第一句，停顿约一秒，然后【开口第一个词就是新值】地念更正句。",
           "en": "Pause about one second, then read the correction with the NEW VALUE as the very first word."},
    "L5": {"zh": "念完第一句后停顿一到三秒（自然犹豫），再念更正句。",
           "en": "Pause one to three seconds (a natural hesitation), then read the correction."},
    "L10": {"zh": "两人录制：主说话人念任务句；另一人（旁观者声）在其后随意时刻念干扰句。",
            "en": "Two speakers: the main speaker reads the task; the other (bystander) reads the interference line at a casual moment after."},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", default="exp/rb/build_v2")
    ap.add_argument("--out", default="exp/rb/golden")
    args = ap.parse_args()
    epdir = Path(args.build) / "episodes"
    rows = []
    for layer in GOLDEN_LAYERS:
        for lang in ("zh", "en"):
            picked = 0
            for p in sorted(epdir.glob(f"A_{layer}_*.json")):
                e = json.loads(p.read_text())
                if e["lang"] != lang or e["split"] != "test":
                    continue
                pieces = [pp for pp in e["pieces"] if pp["role"] == "user"]
                by = [pp for pp in e["pieces"] if pp["role"] == "bystander"]
                rows.append({
                    "item": f"G_{layer}_{lang}_{picked:02d}", "episode": e["id"],
                    "layer": layer, "lang": lang,
                    "lines_main": [pp["text"] for pp in pieces],
                    "line_bystander": by[0]["text"] if by else None,
                    "target_gap_bin": e["revisions"][0]["gap"] if e["revisions"]
                    and e["revisions"][0].get("gap") else None,
                    "instruction": INSTR[layer][lang], "takes": 3})
                picked += 1
                if picked >= PER_CELL:
                    break
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "golden_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    md = ["# RB 金子集录制清单（3 takes/条；先粗档指导语，事后 VAD 量 gap 归箱）", ""]
    for r in rows:
        md.append(f"## {r['item']}  （episode {r['episode']}）")
        md.append(f"- 指导语：{r['instruction']}")
        for i, ln in enumerate(r["lines_main"]):
            md.append(f"- 主说话人第{i + 1}句：{ln}")
        if r["line_bystander"]:
            md.append(f"- 旁观者句：{r['line_bystander']}")
        md.append("")
    (out / "golden_manifest.md").write_text("\n".join(md))
    print(f"{len(rows)} items -> {out}/golden_manifest.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
