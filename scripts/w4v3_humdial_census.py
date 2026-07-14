#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w4v3_humdial_census.py — HumDial HD-Track2 revision-phenomena census (Phase 1
of docs/w4v3_design.md; zero GPU, stdlib-only unless --vad).

What it measures (all aggregates; NO transcript text ever reaches disk):
  counts      per split/lang/scene sample + segment + audio-hour totals
  utterance   duration / char-length / char-rate quantiles per (lang, role)
  gaps        train inter-segment silences: user->assistant (response latency),
              assistant->user (reaction gap = interruption arrival), and
              rev_arrival = first-user-EoU -> next user speech (the revision-
              arrival analog; includes the model answer in between — flagged)
  pause       Pause Handling: [break] coverage, prefix/suffix lengths, T1
              lexicon hit-rates on the pause prefix vs the full utterance;
              with --vad: measured in-utterance pause durations (= the real
              sigma_pre distribution) using the SAME silero call as the
              throughput-track perception (get_speech_timestamps 400ms/30ms)
  dev_gaps    dev JSON inter-segment silences (pure silence — dev has no model
              audio): the cleanest revision-arrival gap analog + reserved tail
  anomalies   parse failures, template mismatches, timeline overruns (§11.8),
              missing [break] (§11.6), unescaped-quote text lines (§11.5)
  ranges_anchor  measured quantiles side by side with w4_synth_gen.RANGES

Cut-point export (--emit-cuts, for the Omni readout + probe duration features):
  kind=utt_end    cut at the user utterance end (label 0, no continuation)
  kind=break_mid  cut at the onset of the largest measured in-utterance pause
                  (label 1, continuation follows) — requires --vad

Run (server):
  $PY scripts/w4v3_humdial_census.py --root /root/autodl-tmp/HumDial_train \
      --strict-counts --out exp/w4v3/humdial_census.json
  $PY scripts/w4v3_humdial_census.py --root /root/autodl-tmp/HumDial_train \
      --vad --emit-cuts exp/w4v3/humdial_cuts.jsonl --splits train,dev
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w4v3_common import (BREAK, INTERRUPT_SCENES, SR, TRAIN_ROLES,  # noqa: E402
                         assert_no_text, derive_roles, iter_dev_samples,
                         iter_train_samples, quantiles, resolve_root,
                         read_textgrid_segments, strip_break, t1_features,
                         wav_duration)

MIN_UTT_FOR_RATE = 0.2   # s; below this char-rate is meaningless
OVERRUN_WARN = 0.1       # s; annotation end past the real WAV tail (§11.8)


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def collect_train(root, langs, limit=None):
    recs, anomalies = [], defaultdict(int)
    for i, s in enumerate(iter_train_samples(root, langs)):
        if limit and i >= limit:
            break
        if s["textgrid"] is None:
            anomalies["train_missing_textgrid"] += 1
            continue
        segs = read_textgrid_segments(s["textgrid"])
        if not segs:
            anomalies["train_textgrid_parse_fail"] += 1
            continue
        wdur = wav_duration(s["wav"])
        if wdur is None:
            anomalies["train_wav_unreadable"] += 1
            continue
        roles = derive_roles(s["scene"], len(segs))
        if len(segs) != len(TRAIN_ROLES.get(s["scene"], [])):
            anomalies["train_template_mismatch"] += 1
        if any(seg["xmax"] > wdur + OVERRUN_WARN for seg in segs):
            anomalies["train_overrun_gt_100ms"] += 1
            if any(seg["xmax"] > wdur + 1.0 for seg in segs):
                anomalies["train_overrun_gt_1s"] += 1
        if any('"' in seg["text"] for seg in segs):
            anomalies["train_quote_text_samples"] += 1
        recs.append({**s, "segs": segs, "roles": roles, "wav_dur": wdur})
    return recs, anomalies


def seg_bounds(seg, wdur):
    """Clamp annotation times to the real audio tail (never trust xmax; §11.8)."""
    a = max(0.0, min(seg["xmin"], wdur))
    b = max(a, min(seg["xmax"], wdur))
    return a, b


def utterance_stats(recs):
    dur, clen, rate = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in recs:
        for seg, role in zip(r["segs"], r["roles"]):
            a, b = seg_bounds(seg, r["wav_dur"])
            d = b - a
            n = len(strip_break(seg["text"]).strip())
            key = (r["language"], role)
            dur[key].append(d)
            clen[key].append(n)
            if d >= MIN_UTT_FOR_RATE:
                rate[key].append(n / d)
    out = {}
    for key in sorted(dur):
        lang, role = key
        out[f"{lang}/{role}"] = {"dur_s": quantiles(dur[key]),
                                 "char_len": quantiles(clen[key]),
                                 "char_rate": quantiles(rate[key])}
    return out


def gap_stats(recs):
    gaps = defaultdict(lambda: defaultdict(list))
    for r in recs:
        key = (r["language"], r["scene"])
        segs, roles, wdur = r["segs"], r["roles"], r["wav_dur"]
        for i in range(len(segs) - 1):
            _, e0 = seg_bounds(segs[i], wdur)
            s1, _ = seg_bounds(segs[i + 1], wdur)
            g = max(0.0, s1 - e0)
            pair = (roles[i], roles[i + 1])
            if pair == ("user", "assistant"):
                gaps[key]["u2a"].append(g)
            elif pair == ("assistant", "user"):
                gaps[key]["a2u"].append(g)
            elif pair == ("assistant", "third"):
                gaps[key]["a2third"].append(g)
        u_idx = [i for i, ro in enumerate(roles) if ro == "user"]
        if len(u_idx) >= 2:
            _, e0 = seg_bounds(segs[u_idx[0]], wdur)
            s1, _ = seg_bounds(segs[u_idx[1]], wdur)
            gaps[key]["rev_arrival"].append(max(0.0, s1 - e0))
    return {f"{k[0]}/{k[1]}": {kk: quantiles(vv) for kk, vv in sorted(d.items())}
            for k, d in sorted(gaps.items())}


def pause_stats(recs, anomalies):
    """Pause Handling text-layer stats + the break-sample list for VAD/cuts."""
    out, break_samples = {}, []
    by_lang = defaultdict(list)
    for r in recs:
        if r["scene"] == "Pause Handling" and r["roles"] and r["roles"][0] == "user":
            by_lang[r["language"]].append(r)
    for lang, rs in sorted(by_lang.items()):
        pre_len, suf_len, t1_pre, t1_full, n_break, n_multi = [], [], [], [], 0, 0
        for r in rs:
            text = r["segs"][0]["text"]
            if BREAK not in text:
                anomalies[f"pause_no_break_{lang}"] += 1
                continue
            n_break += 1
            n_multi += text.count(BREAK) > 1
            prefix = text.split(BREAK)[0]
            suffix = strip_break(text.split(BREAK, 1)[1])
            pre_len.append(len(prefix.strip()))
            suf_len.append(len(suffix.strip()))
            t1_pre.append(t1_features(prefix, lang))
            t1_full.append(t1_features(strip_break(text), lang))
            break_samples.append({**r, "prefix_char_len": len(prefix.strip())})
        def rates(rows):
            if not rows:
                return None
            return {k: round(sum(f[k] for f in rows) / len(rows), 4)
                    for k in rows[0]}
        out[lang] = {"n_scene": len(rs), "n_with_break": n_break,
                     "n_multi_break": n_multi,
                     "prefix_char_len": quantiles(pre_len),
                     "suffix_char_len": quantiles(suf_len),
                     "t1_rates_prefix": rates(t1_pre),
                     "t1_rates_full": rates(t1_full)}
    return out, break_samples


def run_vad(break_samples, min_gap=0.05):
    """Measured in-utterance pauses via the throughput-track perception call
    (silero get_speech_timestamps, min_silence_duration_ms=400, speech_pad_ms=30
    — identical to the replay harness). Annotates each break sample in place."""
    import numpy as np                                     # lazy: GPU-day deps
    import soundfile as sf
    from silero_vad import load_silero_vad, get_speech_timestamps
    model = load_silero_vad()
    pauses_all, max_pause, n_none = defaultdict(list), defaultdict(list), 0
    for i, r in enumerate(break_samples):
        data, sr = sf.read(str(r["wav"]), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        assert sr == SR, f"{r['wav']}: sr={sr}"
        a, b = seg_bounds(r["segs"][0], r["wav_dur"])
        clip = data[int(a * SR):int(b * SR)]
        ts = get_speech_timestamps(clip, model, sampling_rate=SR,
                                   min_silence_duration_ms=400, speech_pad_ms=30)
        spans = [(t["start"] / SR, t["end"] / SR) for t in ts]
        sil = [(spans[j][1], spans[j + 1][0]) for j in range(len(spans) - 1)
               if spans[j + 1][0] - spans[j][1] >= min_gap]
        r["vad_spans"] = len(spans)
        if sil:
            durs = [e - s for s, e in sil]
            k = max(range(len(durs)), key=durs.__getitem__)
            r["pause_start_abs"] = round(a + sil[k][0], 3)
            r["pause_dur"] = round(durs[k], 3)
            pauses_all[r["language"]].extend(durs)
            max_pause[r["language"]].append(durs[k])
        else:
            n_none += 1
        if (i + 1) % 100 == 0:
            print(f"  vad {i + 1}/{len(break_samples)}", flush=True)
    return {lang: {"pause_dur_max_per_utt": quantiles(max_pause[lang]),
                   "pause_dur_all_internal": quantiles(pauses_all[lang])}
            for lang in sorted(max_pause)} | {"n_no_internal_pause": n_none}


def dev_stats(root, langs, limit=None):
    gaps = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    anomalies = defaultdict(int)
    for i, s in enumerate(iter_dev_samples(root, langs)):
        if limit and i >= limit:
            break
        if not s["annotation"]:
            anomalies["dev_json_parse_fail"] += 1
            continue
        counts[(s["language"], s["scene"])] += 1
        segs = s["annotation"]["speech_segments"]
        fdur = s["annotation"]["final_duration"]
        key = (s["language"], s["scene"])
        for j in range(len(segs) - 1):
            gaps[key]["seg_gap"].append(
                max(0.0, segs[j + 1]["xmin"] - segs[j]["xmax"]))
        gaps[key]["tail_reserved"].append(max(0.0, fdur - segs[-1]["xmax"]))
        wdur = wav_duration(s["wav"]) if s["wav"] else None
        if wdur is not None and abs(fdur - wdur) > 0.011:
            anomalies["dev_duration_mismatch"] += 1
    return ({f"{k[0]}/{k[1]}": {kk: quantiles(vv) for kk, vv in sorted(d.items())}
             for k, d in sorted(gaps.items())},
            {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())}, anomalies)


def emit_cuts(break_samples, path, with_vad):
    n_mid = n_end = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in break_samples:
            a, b = seg_bounds(r["segs"][0], r["wav_dur"])
            base = {"wav": r["rel_wav"], "textgrid": r["rel_grid"],
                    "seg_idx": 0, "seg": [round(a, 3), round(b, 3)],
                    "language": r["language"], "split": r["split"]}
            rec = {"key": f"{r['key']}#end", "kind": "utt_end",
                   "cut_t": round(b, 3), "label": 0, **base}
            assert_no_text(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_end += 1
            if with_vad and r.get("pause_start_abs") is not None:
                rec = {"key": f"{r['key']}#break0", "kind": "break_mid",
                       "cut_t": r["pause_start_abs"],
                       "pause_dur": r["pause_dur"], "label": 1, **base}
                assert_no_text(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_mid += 1
    if not with_vad:
        print("WARNING: --emit-cuts without --vad -> utt_end cuts only "
              "(break_mid cut times need measured pauses)")
    print(f"cuts -> {path}  (utt_end {n_end} / break_mid {n_mid})")


def ranges_anchor(vad_block, utt, gaps_train, dev_gaps):
    """Measured quantiles next to the synthetic-generator support (v2 RANGES),
    for the w4v3_design.md §4 backfill table."""
    anchor = {"measured": {
        "sigma_pre_pause_dur": {lang: (vad_block or {}).get(lang, {}).get(
            "pause_dur_max_per_utt") for lang in ("zh", "en")} if vad_block else None,
        "utt_dur_user": {k: v["dur_s"] for k, v in utt.items()
                         if k.endswith("/user")},
        "rev_arrival_train": {k: v.get("rev_arrival") for k, v in
                              gaps_train.items() if v.get("rev_arrival")},
        "dev_seg_gap": {k: v.get("seg_gap") for k, v in (dev_gaps or {}).items()
                        if v.get("seg_gap")},
    }}
    try:
        from w4_synth_gen import RANGES                    # noqa: E402
        anchor["synthetic_v2_ranges"] = {
            k: RANGES[k] for k in RANGES
            if any(t in k for t in ("sig", "utt", "inter_req"))}
    except Exception as e:                                 # pragma: no cover
        anchor["synthetic_v2_ranges"] = f"unavailable: {e}"
    return anchor


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/HumDial_train")
    ap.add_argument("--langs", default="zh,en")
    ap.add_argument("--splits", default="train",
                    help="comma set from {train,dev}")
    ap.add_argument("--vad", action="store_true",
                    help="measure in-utterance pauses on Pause Handling "
                         "(needs torch+silero+soundfile; ~30-45 min CPU)")
    ap.add_argument("--emit-cuts", metavar="JSONL",
                    help="write probe/readout cut points (break_mid needs --vad)")
    ap.add_argument("--out", default="exp/w4v3/humdial_census.json")
    ap.add_argument("--limit", type=int, help="debug: cap samples per split")
    ap.add_argument("--strict-counts", action="store_true",
                    help="verify totals against the 2026-07-15 format report")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()

    root = resolve_root(args.root)
    langs = tuple(args.langs.split(","))
    splits = set(args.splits.split(","))
    report = {"root": str(root), "langs": list(langs),
              "splits": sorted(splits)}
    anomalies = defaultdict(int)

    if "train" in splits:
        recs, anom = collect_train(root, langs, args.limit)
        anomalies.update(anom)
        counts = defaultdict(int)
        hours = defaultdict(float)
        for r in recs:
            counts[(r["language"], r["scene"])] += 1
            hours[r["language"]] += r["wav_dur"] / 3600
        report["counts_train"] = {f"{k[0]}/{k[1]}": v
                                  for k, v in sorted(counts.items())}
        report["train_total"] = len(recs)
        report["audio_hours_train"] = {k: round(v, 3)
                                       for k, v in sorted(hours.items())}
        report["utterance"] = utterance_stats(recs)
        report["gaps_train"] = gap_stats(recs)
        pause, break_samples = pause_stats(recs, anomalies)
        report["pause"] = pause
        if args.vad:
            report["vad_pause"] = run_vad(break_samples)
        if args.emit_cuts:
            Path(args.emit_cuts).parent.mkdir(parents=True, exist_ok=True)
            emit_cuts(break_samples, args.emit_cuts, args.vad)

    dev_gaps = None
    if "dev" in splits:
        dev_gaps, dev_counts, anom = dev_stats(root, langs, args.limit)
        anomalies.update(anom)
        report["gaps_dev"] = dev_gaps
        report["counts_dev"] = dev_counts
        report["dev_total"] = sum(dev_counts.values())

    report["anomalies"] = dict(sorted(anomalies.items()))
    report["ranges_anchor"] = ranges_anchor(
        report.get("vad_pause"), report.get("utterance", {}),
        report.get("gaps_train", {}), dev_gaps)

    ok = True
    if args.strict_counts:
        want = {"train_total": 9988 if set(langs) == {"zh", "en"} else None,
                "dev_total": 1800 if "dev" in splits else None}
        for k, v in want.items():
            if v is not None and report.get(k) is not None:
                good = report[k] == v
                ok &= good
                print(f"strict-counts {k}: {report[k]} vs {v} "
                      f"{'PASS' if good else 'FAIL'}")

    assert_no_text(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in
                      ("train_total", "dev_total", "anomalies")
                      if k in report}, ensure_ascii=False, indent=1))
    print(f"-> {out}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# selftest: synthesize a miniature HD-Track2 tree (CRLF grids, unescaped quote,
# [break], 4-seg interrupt scene, timeline overrun, dev JSON) and verify every
# census stage end to end. Stdlib only.
# ---------------------------------------------------------------------------
def _mk_wav(path, dur):
    import wave as _w
    path.parent.mkdir(parents=True, exist_ok=True)
    with _w.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"\x00\x00" * int(dur * SR))


def _mk_grid(path, segs, total, crlf=True):
    nl = "\r\n" if crlf else "\n"
    lines = ['File type = "ooTextFile"', 'Object class = "TextGrid"',
             "xmin = 0 ", f"xmax = {total} ", "tiers? <exists> ", "size = 3 ",
             "item []: ", "    item [1]:", '        class = "IntervalTier" ',
             '        name = "文本" ', "        xmin = 0 ",
             f"        xmax = {total} ",
             f"        intervals: size = {len(segs)} "]
    for i, (a, b, t) in enumerate(segs, 1):
        lines += [f"        intervals [{i}]:", f"            xmin = {a} ",
                  f"            xmax = {b} ", f'            text = "{t}" ']
    for k, name in ((2, "事件"), (3, "情绪")):
        lines += [f"    item [{k}]:", '        class = "IntervalTier" ',
                  f'        name = "{name}" ', "        xmin = 0 ",
                  f"        xmax = {total} ", "        intervals: size = 1 ",
                  "        intervals [1]:", "            xmin = 0 ",
                  f"            xmax = {total} ", '            text = "" ']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((nl.join(lines) + nl).encode("utf-8"))


def selftest():
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="w4v3_census_st_"))
    (root / "Data_Protocol.md").write_text("stub")
    tz = root / "HD-Track2-train/HD-Track2-train-zh"
    te = root / "HD-Track2-train/HD-Track2-train-en"
    dz = root / "HD-Track2-dev/HD-Track2-dev-zh"

    p = tz / "Pause Handling/0001_0001"
    _mk_wav(p.with_suffix(".wav"), 12.0)
    _mk_grid(p.with_suffix(".TextGrid"),
             [(0.5, 4.5, "请问你们周末的[break]营业时间是几点"),
              (6.0, 10.0, "营业时间是九点到六点")], 12.0)
    p = tz / "Pause Handling/0001_0002"
    _mk_wav(p.with_suffix(".wav"), 8.0)
    _mk_grid(p.with_suffix(".TextGrid"),
             [(0.5, 3.0, "帮我查下天气"), (4.0, 6.0, "今天晴")], 8.0)
    p = te / "Pause Handling/0001_0001"          # same basename, other dir
    _mk_wav(p.with_suffix(".wav"), 10.0)
    _mk_grid(p.with_suffix(".TextGrid"),
             [(0.2, 5.0, "what is the opening time [break] on weekends and"),
              (6.0, 9.0, "nine to six")], 10.0)
    p = tz / "Follow-up Questions/0002_0001"     # overrun + unescaped quote
    _mk_wav(p.with_suffix(".wav"), 18.0)
    _mk_grid(p.with_suffix(".TextGrid"),
             [(0.5, 3.0, "帮我订一张去北京的票"),
              (4.0, 10.0, '他说 "好的" 已经订好'),
              (12.0, 14.0, "对了再帮我加一份保险的"),
              (15.0, 20.0, "保险已加")], 20.0)
    p = dz / "Follow-up Questions/0001_0001"
    _mk_wav(p.with_suffix(".wav"), 30.0)
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.parent / "0001_0001_sentence.json").write_text(json.dumps(
        {"final_duration": 30.0, "speech_segments": [
            {"xmin": 0.5, "xmax": 4.0, "text": "帮我查个航班"},
            {"xmin": 9.0, "xmax": 12.0, "text": "改成明天的"}]},
        ensure_ascii=False))

    out = root / "census.json"
    cuts = root / "cuts.jsonl"
    rc = main(["--root", str(root), "--splits", "train,dev",
               "--emit-cuts", str(cuts), "--out", str(out)])
    rep = json.loads(out.read_text())
    ck = {"train_total": rep["train_total"] == 4,
          "counts": rep["counts_train"] == {"en/Pause Handling": 1,
                                            "zh/Follow-up Questions": 1,
                                            "zh/Pause Handling": 2},
          "pause_zh": rep["pause"]["zh"]["n_with_break"] == 1
          and rep["pause"]["zh"]["n_scene"] == 2,
          "pause_en_t1": rep["pause"]["en"]["t1_rates_prefix"]["ends_conn"] == 0.0
          and rep["pause"]["en"]["t1_rates_full"]["ends_conn"] == 1.0,
          "gap_u2a": rep["gaps_train"]["zh/Follow-up Questions"]["u2a"]["p50"] == 1.0,
          "gap_a2u": rep["gaps_train"]["zh/Follow-up Questions"]["a2u"]["p50"] == 2.0,
          "rev_arrival": rep["gaps_train"]["zh/Follow-up Questions"]
          ["rev_arrival"]["p50"] == 9.0,
          "overrun": rep["anomalies"].get("train_overrun_gt_100ms") == 1
          and rep["anomalies"].get("train_overrun_gt_1s") == 1,
          "quote": rep["anomalies"].get("train_quote_text_samples") == 1,
          "no_break": rep["anomalies"].get("pause_no_break_zh") == 1,
          "dev_gap": rep["gaps_dev"]["zh/Follow-up Questions"]["seg_gap"]["p50"] == 5.0,
          "dev_tail": rep["gaps_dev"]["zh/Follow-up Questions"]
          ["tail_reserved"]["p50"] == 18.0,
          "cuts": sum(1 for _ in open(cuts)) == 2,
          "exit": rc == 0}
    # census artifacts must never carry transcript text
    ck["no_text_in_artifacts"] = ('"text"' not in out.read_text()
                                  and '"text"' not in cuts.read_text())
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL", f"(tree: {root})")
    return 0 if all(ck.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
