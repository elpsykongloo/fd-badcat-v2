#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rb_build.py — build the RB v2 episode set (docs/rb_design.md v2 §7/§8).

  $PY scripts/rb_build.py                       # episodes + manifest (no audio)
  $PY scripts/rb_build.py --audio stub          # + placeholder wavs (pipeline dry run)
  $PY scripts/rb_build.py --audio qwen          # + real Qwen3-TTS wavs (user wiring)
  $PY scripts/rb_build.py --verify              # determinism: rebuild == manifest
  $PY scripts/rb_build.py --selftest            # tiny-quota structural checks

Outputs under --out: episodes/<id>.json, audio/<id>.wav (+ cues in episode),
manifest.json (config_hash / ids_hash / content_hash / split counts). The
pause prior (exp/w5sg/pause_prior.json, shared W5-SG census) refines L5-class
gap sampling when present — build WITHOUT it is valid but must be declared."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rb.generator import build_all, manifest, ARM_A_QUOTA, ARM_B_QUOTA  # noqa: E402


def build(out_dir, audio=None, pause_prior_path=None, quota_a=None, quota_b=None):
    pause_prior = None
    if pause_prior_path and Path(pause_prior_path).exists():
        pause_prior = json.loads(Path(pause_prior_path).read_text())
        print(f"pause prior: {pause_prior_path} (n={pause_prior.get('n')})")
    else:
        print("pause prior: NONE (uniform in-bin sampling; declare in the report)")
    ch, eps = build_all(pause_prior=pause_prior, quota_a=quota_a, quota_b=quota_b)
    out = Path(out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    backend = None
    if audio == "stub":
        from rb.audio import SilenceStub
        backend = SilenceStub()
    elif audio == "qwen":
        from rb.audio import QwenTTSBackend
        backend = QwenTTSBackend()
    for e in eps:
        if backend is not None and e["arm"] == "A":
            from rb.audio import assemble_episode, measured_gaps
            cues = assemble_episode(e, backend, out / "audio" / f"{e['id']}.wav")
            e["cues"] = cues
            e["measured_gaps"] = measured_gaps(cues)
        (out / "episodes" / f"{e['id']}.json").write_text(
            json.dumps(e, ensure_ascii=False, indent=1))
    m = manifest(ch, eps)
    m["audio"] = audio or "none"
    m["pause_prior"] = bool(pause_prior)
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))
    return m


def verify(out_dir):
    old = json.loads((Path(out_dir) / "manifest.json").read_text())
    ch, eps = build_all(pause_prior=None)
    m = manifest(ch, eps)
    same = (m["config_hash"] == old["config_hash"]
            and m["ids_hash"] == old["ids_hash"])
    note = ("content_hash matches too" if m["content_hash"] == old["content_hash"]
            else "content differs — was the original built WITH a pause prior? "
                 "(prior changes gap draws by design)")
    print(f"config_hash {'OK' if same else 'MISMATCH'}; {note}")
    return 0 if same else 1


def selftest():
    from rb.generator import make_episode, config_hash
    from rb.registry import canon_value
    from rb.sandbox import Sandbox, canonical_calls
    ck = {}
    ch = config_hash()
    qa = {k: 2 for k in ARM_A_QUOTA}
    qb = {k: 2 for k in ARM_B_QUOTA}
    ch1, eps1 = build_all(quota_a=qa, quota_b=qb)
    ch2, eps2 = build_all(quota_a=qa, quota_b=qb)
    ck["deterministic_rebuild"] = (
        json.dumps(eps1, sort_keys=True) == json.dumps(eps2, sort_keys=True))
    ck["episode_count"] = len(eps1) == 2 * len(ARM_A_QUOTA) + 2 * len(ARM_B_QUOTA)
    e = make_episode("A", "L4", 0, ch)
    ck["l4_value_first"] = e["revisions"][0]["kind"] == "value_first" and \
        e["pieces"][1]["text"].startswith(e["revisions"][0]["new"])
    ck["l4_gap_in_bin"] = 0.68 <= e["revisions"][0]["gap"] <= 1.14
    revised = e["revisions"][0]
    new_canon = canon_value(revised["slot"], revised["new"])
    ck["gold_uses_new_value"] = any(
        value == new_canon
        for call in e["gold_calls"]
        for value in call["args"].values())
    e10 = next(x for x in eps1 if x["layer"] == "L10" and x["bystander"])
    gold_arg_values = [v for call in e10["gold_calls"]
                       for v in call["args"].values()]
    ck["bystander_not_in_gold"] = e10["bystander"]["other"] not in gold_arg_values or \
        e10["bystander"]["other"] in [v for v in e10["slots_final"].values()]
    e8c = make_episode("A", "L8", 1, ch)          # idx 1 -> cancel
    ck["l8_cancel_gold_empty"] = e8c["cancelled"] and e8c["gold_calls"] == []
    eb = make_episode("B", "L8", 0, ch)
    ck["armb_has_events"] = len(eb["events"]) == 1
    # sandbox: idempotency + compensation fee
    sb = Sandbox("t1")
    r1 = sb.execute("reserve_hotel", {"city": "杭州", "checkin": "五月三号",
                                      "nights": "两"}, idem_key="k1")
    r2 = sb.execute("reserve_hotel", {"city": "杭州", "checkin": "五月三号",
                                      "nights": "两"}, idem_key="k1")
    ck["idempotent_retry"] = r1 == r2 and len(sb.calls) == 1
    rid = r1["result"]["id"]
    c = sb.compensate("reserve_hotel", rid)
    ck["comp_fee"] = c["status"] == "success" and sb.fees == 1 and \
        sb.live_state() == {}
    ck["canonical_sort"] = canonical_calls(
        [{"fn": "a", "args": {"x": 2}}, {"fn": "a", "args": {"x": 1}}]) == \
        canonical_calls([{"fn": "a", "args": {"x": 1}}, {"fn": "a", "args": {"x": 2}}])
    ck["split_present"] = {e["split"] for e in eps1} <= {"dev", "test"}
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="exp/rb/build_v2")
    ap.add_argument("--audio", choices=["stub", "qwen"])
    ap.add_argument("--pause-prior", default="exp/w5sg/pause_prior.json")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.verify:
        return verify(args.out)
    build(args.out, audio=args.audio, pause_prior_path=args.pause_prior)
    return 0


if __name__ == "__main__":
    sys.exit(main())
