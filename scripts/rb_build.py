#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rb_build.py — build the RB v2 episode set (docs/rb_design.md v2 §7/§8).

  $PY scripts/rb_build.py                       # episodes + manifest (no audio)
  $PY scripts/rb_build.py --audio stub          # + placeholder wavs (pipeline dry run)
  $PY scripts/rb_build.py --audio qwen --tts-workers 16
                                                # + parallel real Qwen3-TTS prewarm
  $PY scripts/rb_build.py --out BUILD --prewarm-arm B --tts-workers 16
                                                # cache all reactive B pieces/events
  $PY scripts/rb_build.py --verify              # determinism: rebuild == manifest
  $PY scripts/rb_build.py --selftest            # tiny-quota structural checks

Outputs under --out: episodes/<id>.json, audio/<id>.wav (+ cues in episode),
manifest.json (config_hash / ids_hash / content_hash / split counts). The
pause prior (exp/w5sg/pause_prior.json, shared W5-SG census) refines L5-class
gap sampling when present — build WITHOUT it is valid but must be declared."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rb.generator import build_all, manifest, ARM_A_QUOTA, ARM_B_QUOTA  # noqa: E402


def _default_tts_workers():
    try:
        return int(os.getenv("RB_TTS_WORKERS", "16"))
    except ValueError:
        return 16


def audit_templates():
    """v2.4 build-time hard gate (the L4 erratum's regression net,
    rb_test_protocol §10.7): every revision template — frozen AND every bank
    variant — carries {new} EXACTLY once (cancel: zero); value_first also
    starts with {new} and carries {old}; confirm templates stay free of cancel lexemes (the oracle
    cancel fallback must never misfire on the L14 probe). Returns a list of
    violations; the build refuses to proceed on any."""
    import rb.grammar as g
    bad = []
    bank = g._BANK or {}
    for lang in ("zh", "en"):
        for kind, tpl in g.REV_UTT[lang].items():
            variants = [tpl] + [v for v in (bank.get("revision", {})
                                            .get(lang, {}).get(kind, []))
                                if v != tpl]
            for v in variants:
                want = 0 if kind == "cancel" else 1
                if v.count("{new}") != want:
                    bad.append(("revision", lang, kind, v))
                if kind == "value_first" and v.count("{old}") != 1:
                    bad.append(("revision_old", lang, kind, v))
                if kind == "value_first" and not v.startswith("{new}"):
                    bad.append(("revision_value_first_position", lang, kind, v))
        for v in [g.CONFIRM_QUERY[lang]] + list(bank.get("confirm", {})
                                                .get(lang, [])):
            if any(tok in v for tok in ("别办", "先别", "hold off")):
                bad.append(("confirm_cancel_lexeme", lang, "confirm", v))
    return bad


def build(out_dir, audio=None, pause_prior_path=None, quota_a=None, quota_b=None,
          tts_workers=16):
    bad = audit_templates()
    if bad:
        for b in bad:
            print("TEMPLATE AUDIT FAIL:", b)
        raise SystemExit("template audit failed - refusing to build "
                         "(regenerate/fix the content bank first)")
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
        from rb.audio import QwenTTSBackend, episode_tts_requests
        backend = QwenTTSBackend()
        # The GPU work runs in parallel, but only after deduplicating cache
        # keys.  Assembly deliberately remains serial and cache-only so cue
        # ordering and final WAV bytes cannot depend on completion order.
        a_requests = episode_tts_requests(
            (e for e in eps if e["arm"] == "A"), include_events=False)
        tts_stats = backend.prewarm(a_requests, workers=tts_workers)
        print("qwen TTS prewarm:", json.dumps(tts_stats, sort_keys=True))
        backend.cache_only = True
    else:
        tts_stats = None
    try:
        for e in eps:
            if backend is not None and e["arm"] == "A":
                from rb.audio import assemble_episode, measured_gaps
                cues = assemble_episode(e, backend, out / "audio" / f"{e['id']}.wav")
                e["cues"] = cues
                e["measured_gaps"] = measured_gaps(cues)
            (out / "episodes" / f"{e['id']}.json").write_text(
                json.dumps(e, ensure_ascii=False, indent=1))
    finally:
        if audio == "qwen":
            backend.cache_only = False
    m = manifest(ch, eps)
    m["audio"] = audio or "none"
    m["pause_prior"] = bool(pause_prior)
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    if tts_stats is not None:
        print("qwen TTS assembly: cache-only PASS (0 serial synthesis fallback)")
    print(json.dumps(m, indent=2))
    return m


def prewarm_existing(out_dir, arm, tts_workers):
    """Warm every scripted and reactive segment of an already-built arm."""
    from rb.audio import QwenTTSBackend, episode_tts_requests
    epdir = Path(out_dir) / "episodes"
    paths = sorted(epdir.glob(f"{arm}_*.json"))
    if not paths:
        raise SystemExit(f"no arm-{arm} episodes under {epdir}")
    episodes = [json.loads(path.read_text()) for path in paths]
    requests = episode_tts_requests(episodes, include_events=True)
    backend = QwenTTSBackend()
    stats = backend.prewarm(requests, workers=tts_workers)
    receipt = {"arm": arm, "episodes": len(episodes),
               "includes_events": True, **stats}
    print("qwen TTS prewarm:", json.dumps(receipt, sort_keys=True))
    return receipt


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
    # v2.4: frozen templates alone are clean; a poisoned double-{new} bank
    # variant IS caught; and with the CURRENTLY installed bank, any
    # violations are confined to the v2.3 value_first residue (which the
    # build gate refuses until the v2.4 bank is regenerated).
    import rb.grammar as _g
    saved = _g._BANK
    _g._BANK = None
    viol_frozen = audit_templates()
    _g._BANK = {"revision": {"zh": {"default": ["等等，改成{new}。",
                                               "从{new}换到{new}。"]}}}
    viol_poison = audit_templates()
    _g._BANK = {"revision": {"zh": {"value_first": ["{new}，不是{old}。",
                                                        "我想要{new}，不是{old}。"]}}}
    viol_position = audit_templates()
    _g._BANK = saved
    viol_now = audit_templates()
    ck["v24_template_audit"] = (
        viol_frozen == [] and len(viol_poison) == 1
        and len(viol_position) == 1
        and all(v[2] == "value_first" for v in viol_now))
    # v2.4: L13 octuple structure + stratified split floors survive the build
    e13 = make_episode("B", "L13", 0, ch)
    ck["v24_l13_cell"] = (e13["pair"]["who"] == "user"
                          and e13["pair"]["state"] == "eou"
                          and e13["pair"]["family"] == "BF000")
    e13b = make_episode("B", "L13", 4, ch)
    ck["v24_l13_byst_cell"] = (e13b["pair"]["who"] == "bystander"
                               and e13b["pair"]["state"] == "eou"
                               and e13b["revisions"] == []
                               and bool(e13b["bystander"]["other"]))
    ck["v24_caps_present"] = all(
        e.get("caps", {}).get("abort_on_cancel") for e in eps1)
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
    # TTS concurrency safety without touching the live service: duplicate
    # cache keys coalesce before submission, later calls hit atomically
    # published WAVs, and cache-only mode refuses a silent serial fallback.
    from rb.audio import QwenTTSBackend
    import tempfile
    import threading
    import wave
    with tempfile.TemporaryDirectory() as tmpdir:
        class _FakeQwen(QwenTTSBackend):
            def __init__(self):
                super().__init__(tts_dir=tmpdir, cache_dir=tmpdir,
                                 voice_map_path=Path(tmpdir) / "missing.json")
                self.calls = []
                self.calls_lock = threading.Lock()

            def _synthesize_to_path(self, text, preset, lang, tmp):
                with self.calls_lock:
                    self.calls.append((text, preset, lang))
                with wave.open(str(tmp), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(b"\x00\x00")

        fake = _FakeQwen()
        warm = fake.prewarm([
            {"text": "alpha", "voice": "cv01", "lang": "en"},
            {"text": "alpha", "voice": "cv01", "lang": "en"},
            {"text": "beta", "voice": "cv02", "lang": "en"},
        ], workers=4)
        hit = fake.prewarm([
            {"text": "alpha", "voice": "cv01", "lang": "en"},
            {"text": "beta", "voice": "cv02", "lang": "en"},
        ], workers=4)
        fake.cache_only = True
        try:
            fake._ensure_wav("uncached", "cv01", "en")
        except RuntimeError:
            cache_only_guard = True
        else:
            cache_only_guard = False
        ck["tts_parallel_cache_isolation"] = (
            warm["requested"] == 3 and warm["unique"] == 2
            and warm["synthesized"] == 2 and warm["workers_used"] == 2
            and hit["initial_cache_hits"] == 2 and hit["synthesized"] == 0
            and len(fake.calls) == 2 and cache_only_guard)
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="exp/rb/build_v2")
    ap.add_argument("--audio", choices=["stub", "qwen"])
    ap.add_argument("--tts-workers", type=int, default=_default_tts_workers(),
                    help="parallel Qwen cache-warm requests (default: RB_TTS_WORKERS or 16)")
    ap.add_argument("--prewarm-arm", choices=["A", "B"],
                    help="warm existing arm pieces and reactive events only")
    ap.add_argument("--pause-prior", default="exp/w5sg/pause_prior.json")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="print the current content-template audit without building")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.tts_workers < 1:
        ap.error("--tts-workers must be >= 1")
    if args.selftest:
        return selftest()
    if args.audit:
        bad = audit_templates()
        for row in bad:
            print("TEMPLATE AUDIT FAIL:", row)
        print(f"template audit: {len(bad)} violations")
        return 0 if not bad else 1
    if args.verify:
        return verify(args.out)
    if args.prewarm_arm:
        if args.audio:
            ap.error("--prewarm-arm is a standalone cache operation; omit --audio")
        prewarm_existing(args.out, args.prewarm_arm, args.tts_workers)
        return 0
    build(args.out, audio=args.audio, pause_prior_path=args.pause_prior,
          tts_workers=args.tts_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
