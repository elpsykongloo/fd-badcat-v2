#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w4v3_text_probe.py — feature-information probe on real HumDial pauses
(Phase 1 of docs/w4v3_design.md; CPU-only, numpy).

Question it answers (the condition-(ii) question, on CLEAN non-test data): do
text-pragmatic features carry continuation/revision signal BEYOND structural
length features? The answer gates whether v3 gets a feature upgrade (Arm F) or
stays calibration-only (Arm C).

Task (PAIRED-CONTINUATION): every Pause Handling utterance with a [break] yields
two instances sharing one group:
    prefix (text before the first [break])  -> label 1  (continuation follows)
    full utterance ([break] markers removed) -> label 0  (turn really ends)
The pair shares topic/speaker/recording, so confounds cancel; both instances
always land in the same CV fold (group = full-text hash, which also collapses
duplicated prompts across samples — the probe report §11.11 leakage guard).

Feature families:
    S   structural: log char length, language flag (+ measured duration/rate
        when census --vad cuts are supplied) — the FDB-side S7 analog available
        on HumDial (slots_missing etc. have no HumDial counterpart; the probe
        gates only the TEXT increment, S7 keeps its FDB evidence regardless)
    T1  hand pragmatic markers (dangling connective/preposition/particle/
        filler/auxiliary/number tails; bilingual lexicons in w4v3_common)
    T2  hashed char n-grams of the utterance tail (256 buckets — the cheap
        "learned representation" tier)
    P   annotator punctuation (LEAKAGE-SUSPECT: transcribers punctuate knowing
        whether the sentence ended; diagnostic only, never in the gate)
    X   external per-key features (--extra-jsonl; e.g. Omni readout labels)

PREREGISTERED GATE (frozen in w4v3_design.md §5 before the real run):
    dAUC = AUC_cv(S+T1+T2) - AUC_cv(S) >= 0.05
    AND DeLong one-sided p < 0.01 on the pooled out-of-fold scores.
Secondary (reported, not gating): pair-flip permutation p for the T-only
paired accuracy (exact null: acc = 0.5 under 50% pair flips).

Run (server):
  $PY scripts/w4v3_text_probe.py --root /root/autodl-tmp/HumDial_train \
      --splits train --cuts exp/w4v3/humdial_cuts.jsonl \
      --out exp/w4v3/text_probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w4v3_common import (BREAK, P_NAMES, T1_NAMES, assert_no_text,  # noqa: E402
                         auc, delong_test, fit_lr, group_fold,
                         iter_dev_samples, iter_train_samples, predict_lr,
                         p_features, quantiles, read_textgrid_segments,
                         resolve_root, strip_break, t1_features, t2_features,
                         text_hash, wav_duration)

T2_BUCKETS = 256
GATE_DAUC = 0.05        # frozen
GATE_P = 0.01           # frozen
K_FOLDS = 5


# ---------------------------------------------------------------------------
# instance construction
# ---------------------------------------------------------------------------
def collect_pairs(root, langs, splits, limit=None):
    """-> list of {key, group, lang, label, text}; text stays in memory only."""
    pairs, seen, n_dup = [], set(), 0

    def add_pair(key, lang, text):
        nonlocal n_dup
        if BREAK not in text:
            return
        full = strip_break(text).strip()
        prefix = text.split(BREAK)[0].strip()
        if not prefix or not full:
            return
        g = text_hash(full)
        if g in seen:                       # duplicated prompt -> keep first
            n_dup += 1
            return
        seen.add(g)
        pairs.append({"key": f"{key}#prefix", "group": g, "lang": lang,
                      "label": 1, "text": prefix, "base": key})
        pairs.append({"key": f"{key}#full", "group": g, "lang": lang,
                      "label": 0, "text": full, "base": key})

    n_seen = 0
    if "train" in splits:
        for s in iter_train_samples(root, langs):
            if s["scene"] != "Pause Handling" or s["textgrid"] is None:
                continue
            if limit and n_seen >= limit:
                break
            segs = read_textgrid_segments(s["textgrid"])
            if not segs:
                continue
            n_seen += 1
            add_pair(s["key"], s["language"], segs[0]["text"])
    if "dev" in splits:
        for s in iter_dev_samples(root, langs):
            if s["scene"] != "Pause Handling" or not s["annotation"]:
                continue
            segs = s["annotation"]["speech_segments"]
            if segs:
                add_pair(s["key"], s["language"], segs[0]["text"])
    return pairs, n_dup


def collect_extra_negs(root, langs, scenes, limit=None):
    """Secondary analysis: full first-user utterances from other scenes as
    additional label-0 instances (their own groups)."""
    out, seen = [], set()
    for s in iter_train_samples(root, langs):
        if s["scene"] not in scenes or s["textgrid"] is None:
            continue
        segs = read_textgrid_segments(s["textgrid"])
        if not segs:
            continue
        text = strip_break(segs[0]["text"]).strip()
        g = text_hash(text)
        if not text or g in seen:
            continue
        seen.add(g)
        out.append({"key": f"{s['key']}#u1", "group": g,
                    "lang": s["language"], "label": 0, "text": text})
        if limit and len(out) >= limit:
            break
    return out


def load_cut_durations(cuts_path):
    """census --vad cuts -> {base_key: {'prefix_dur':…, 'full_dur':…}}."""
    durs = defaultdict(dict)
    for line in Path(cuts_path).read_text().splitlines():
        r = json.loads(line)
        base = r["key"].rsplit("#", 1)[0]
        if r["kind"] == "break_mid":
            durs[base]["prefix_dur"] = max(0.05, r["cut_t"] - r["seg"][0])
        elif r["kind"] == "utt_end":
            durs[base]["full_dur"] = max(0.05, r["cut_t"] - r["seg"][0])
    return durs


def external_common_support(insts, extra):
    """Keep whole paired groups only when every side has external features.

    VAD can miss a prefix cut while the full-turn cut is always available.
    Zero-imputing that label-dependent absence would create a missingness
    shortcut, so the X probe is defined on strict paired common support.
    """
    by_group = defaultdict(list)
    for inst in insts:
        by_group[inst["group"]].append(inst)

    def covered(inst):
        return inst["key"] in extra or inst.get("base") in extra

    keep = {group for group, rows in by_group.items()
            if len(rows) == 2 and all(covered(row) for row in rows)}
    out = [inst for inst in insts if inst["group"] in keep]
    return out, {"pairs_before": len(by_group), "pairs_after": len(keep),
                 "pairs_dropped": len(by_group) - len(keep),
                 "instances_after": len(out)}


# ---------------------------------------------------------------------------
# feature matrices
# ---------------------------------------------------------------------------
def build_matrices(insts, durs=None, extra=None):
    fams = {}
    s_cols = ["log_len", "is_zh"]
    use_dur = False
    if durs:
        cov = sum(1 for i in insts if _dur_of(i, durs) is not None) / len(insts)
        use_dur = cov >= 0.95
        print(f"duration features: coverage {cov:.1%} -> "
              f"{'ENABLED' if use_dur else 'DISABLED (<95%)'}")
        if use_dur:
            s_cols += ["dur", "char_rate"]
    S = []
    for i in insts:
        body = i["text"]
        row = [math.log(1 + len(body)), 1.0 if i["lang"] == "zh" else 0.0]
        if use_dur:
            d = _dur_of(i, durs) or 1.0
            row += [d, len(body) / d]
        S.append(row)
    fams["S"] = (np.array(S), s_cols)
    fams["T1"] = (np.array([[t1_features(i["text"], i["lang"])[k]
                             for k in T1_NAMES] for i in insts]), list(T1_NAMES))
    fams["T2"] = (np.array([t2_features(i["text"], i["lang"], T2_BUCKETS)
                            for i in insts]),
                  [f"t2_{j}" for j in range(T2_BUCKETS)])
    fams["P"] = (np.array([[p_features(i["text"])[k] for k in P_NAMES]
                           for i in insts]), list(P_NAMES))
    if extra:
        names = sorted({k for v in extra.values() for k in v})
        cov = sum(1 for i in insts if i["key"] in extra or i["base"] in extra)
        print(f"external features: {len(names)} cols, "
              f"coverage {cov}/{len(insts)}")
        X = [[float((extra.get(i["key"]) or extra.get(i.get("base"), {})
                     ).get(k, 0.0)) for k in names] for i in insts]
        fams["X"] = (np.array(X), names)
    return fams


def _dur_of(inst, durs):
    d = durs.get(inst.get("base", ""), {})
    return d.get("prefix_dur" if inst["label"] == 1 else "full_dur")


def stack(fams, names):
    X = np.hstack([fams[n][0] for n in names])
    cols = [c for n in names for c in fams[n][1]]
    return X, cols


def cv_oof(X, y, folds):
    oof = np.zeros(len(y))
    for f in sorted(set(folds)):
        tr = folds != f
        oof[~tr] = predict_lr(fit_lr(X[tr], y[tr]), X[~tr])
    return oof


def paired_accuracy(insts, scores):
    by_group = defaultdict(dict)
    for i, s in zip(insts, scores):
        by_group[i["group"]][i["label"]] = s
    a = [1.0 if d[1] > d[0] else (0.5 if d[1] == d[0] else 0.0)
         for d in by_group.values() if len(d) == 2]
    return (float(np.mean(a)), len(a), np.array(a)) if a else (None, 0, None)


def flip_perm_p(a, n_perm, seed):
    """Exact pair-flip null for paired accuracy: each pair swaps labels w.p. .5
    -> its contribution becomes 1-a_i. One-sided p for acc > 0.5."""
    rng = np.random.default_rng(seed)
    obs = a.mean()
    hits = 1
    for _ in range(n_perm):
        flip = rng.random(len(a)) < 0.5
        if np.where(flip, 1 - a, a).mean() >= obs:
            hits += 1
    return hits / (n_perm + 1)


# ---------------------------------------------------------------------------
def run_probe(insts, durs, extra, n_perm, seed, tag=""):
    y = np.array([i["label"] for i in insts], float)
    folds = np.array([group_fold(i["group"], K_FOLDS) for i in insts])
    fams = build_matrices(insts, durs, extra)
    combos = [("S",), ("T1",), ("T2",), ("S", "T1"), ("S", "T2"),
              ("S", "T1", "T2"), ("S", "T1", "T2", "P")]
    if "X" in fams:
        combos += [("S", "X"), ("S", "T1", "T2", "X")]
    oofs, res = {}, {}
    for combo in combos:
        X, cols = stack(fams, combo)
        oof = cv_oof(X, y, folds)
        oofs[combo] = oof
        res["+".join(combo)] = {"auc": round(auc(y, oof), 4), "n_cols": len(cols)}
    def gate(contender, baseline):
        a1, a0, z, p = delong_test(y, oofs[contender], oofs[baseline])
        d_auc = a1 - a0
        return {"delta_auc": round(d_auc, 4),
                "auc_gate": round(a1, 4), "auc_base": round(a0, 4),
                "delong_z": round(z, 3), "delong_p_one_sided": p,
                "thresholds": {"dAUC": GATE_DAUC, "p": GATE_P},
                "pass": bool(d_auc >= GATE_DAUC and p < GATE_P)}

    # Frozen main gate: text increment over structure.  X is independently
    # gated by the same rule (design §5), never substituted into this result.
    gate_main = gate(("S", "T1", "T2"), ("S",))
    gate_x = gate(("S", "X"), ("S",)) if "X" in fams else None
    # secondary: T-only paired accuracy + exact flip permutation
    Xt, _ = stack(fams, ("T1", "T2"))
    oof_t = cv_oof(Xt, y, folds)
    acc, n_pairs, a_vec = paired_accuracy(insts, oof_t)
    acc_p = flip_perm_p(a_vec, n_perm, seed) if a_vec is not None else None
    out = {"tag": tag, "n_instances": len(insts), "n_pairs": n_pairs,
           "combos": res,
           "gate": gate_main,
           "secondary_T_only": {"paired_acc": None if acc is None
                                else round(acc, 4),
                                "flip_perm_p": acc_p, "n_perm": n_perm}}
    if gate_x is not None:
        out["gate_x"] = gate_x
    lang_n = defaultdict(int)
    for i in insts:
        lang_n[i["lang"]] += 1
    out["by_lang"] = dict(lang_n)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="/root/autodl-tmp/HumDial_train")
    ap.add_argument("--langs", default="zh,en")
    ap.add_argument("--splits", default="train")
    ap.add_argument("--cuts", help="census --vad cuts jsonl (adds duration/rate to S)")
    ap.add_argument("--extra-jsonl",
                    help="external per-key features ({key, feats:{...}} lines; "
                         "e.g. w4v3_omni_readout.py output)")
    ap.add_argument("--neg-scenes", default="",
                    help="comma list of extra scenes whose first user turns "
                         "join as additional negatives (secondary analysis)")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default="exp/w4v3/text_probe.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.extra_jsonl and args.neg_scenes:
        raise SystemExit("--extra-jsonl and --neg-scenes cannot be combined: "
                         "external X requires strict paired common support")

    root = resolve_root(args.root)
    langs = tuple(args.langs.split(","))
    splits = set(args.splits.split(","))
    pairs, n_dup = collect_pairs(root, langs, splits, args.limit)
    if not pairs:
        raise SystemExit("no [break] pairs found — check --root/--splits")
    print(f"pairs: {len(pairs) // 2} ({len(pairs)} instances), "
          f"dropped {n_dup} duplicate prompts")
    durs = load_cut_durations(args.cuts) if args.cuts else None
    extra = None
    support = None
    if args.extra_jsonl:
        extra = {}
        for line in Path(args.extra_jsonl).read_text().splitlines():
            r = json.loads(line)
            extra[r["key"]] = r["feats"]
        pairs, support = external_common_support(pairs, extra)
        print("external paired common support: "
              f"{support['pairs_after']}/{support['pairs_before']} pairs "
              f"({support['pairs_dropped']} dropped)")
        if not pairs:
            raise SystemExit("no paired external common support")

    report = {"primary": run_probe(pairs, durs, extra, args.perms,
                                   args.seed, tag="paired")}
    if support is not None:
        report["external_common_support"] = support
    if args.neg_scenes:
        scenes = [s.strip() for s in args.neg_scenes.split(",") if s.strip()]
        negs = collect_extra_negs(root, langs, scenes, args.limit)
        print(f"extra negatives from {scenes}: {len(negs)}")
        report["with_extra_negs"] = run_probe(pairs + negs, durs, extra,
                                              args.perms, args.seed,
                                              tag="paired+negs")
    report["prefix_len_quantiles"] = quantiles(
        [len(i["text"]) for i in pairs if i["label"] == 1])

    assert_no_text(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    g = report["primary"]["gate"]
    print(json.dumps(report["primary"]["combos"], indent=1))
    print(f"GATE dAUC={g['delta_auc']} (>= {GATE_DAUC}) "
          f"DeLong p={g['delong_p_one_sided']:.2e} (< {GATE_P}) "
          f"-> {'PASS: Arm F enabled' if g['pass'] else 'FAIL: Arm C only'}")
    if "gate_x" in report["primary"]:
        gx = report["primary"]["gate_x"]
        print(f"GATE_X dAUC={gx['delta_auc']} (>= {GATE_DAUC}) "
              f"DeLong p={gx['delong_p_one_sided']:.2e} (< {GATE_P}) "
              f"-> {'PASS: X enabled' if gx['pass'] else 'FAIL: X disabled'}")
    print(f"-> {out}")
    return 0


# ---------------------------------------------------------------------------
# selftest: two synthetic tasks. (1) signal: prefixes end with dangling
# function words, fulls end with content tokens -> gate must PASS and T-only
# paired accuracy must be high. (2) null: both sides end with content tokens
# -> gate must FAIL (text adds nothing beyond length).
# ---------------------------------------------------------------------------
def _mk_insts(rng, n, lang, signal):
    zh_content = "票房间价格钱天航班期店餐周末点单号"
    zh_hang = ["的", "和", "或者", "然后", "因为", "到", "把", "嗯"]
    en_content = ["ticket", "room", "price", "flight", "hotel", "table",
                  "monday", "order", "seat", "menu"]
    en_hang = ["and", "or", "the", "to", "of", "um", "because", "with"]
    out = []
    for k in range(n):
        g = f"{lang}{signal}{k}"
        if lang == "zh":
            base = "".join(rng.choice(zh_content) for _ in range(rng.randint(6, 16)))
            full = base + rng.choice(zh_content)
            prefix = (base[:rng.randint(4, len(base))]
                      + (rng.choice(zh_hang) if signal else rng.choice(zh_content)))
        else:
            base = " ".join(rng.choice(en_content)
                            for _ in range(rng.randint(4, 9)))
            full = base + " " + rng.choice(en_content)
            words = base.split()
            cut = rng.randint(3, len(words))
            prefix = (" ".join(words[:cut]) + " "
                      + (rng.choice(en_hang) if signal else rng.choice(en_content)))
        out.append({"key": f"{g}#prefix", "group": g, "lang": lang,
                    "label": 1, "text": prefix, "base": g})
        out.append({"key": f"{g}#full", "group": g, "lang": lang,
                    "label": 0, "text": full, "base": g})
    return out


def selftest():
    import random
    rng = random.Random(0)
    sig = _mk_insts(rng, 300, "zh", True) + _mk_insts(rng, 200, "en", True)
    nul = _mk_insts(rng, 300, "zh", False) + _mk_insts(rng, 200, "en", False)
    r_sig = run_probe(sig, None, None, n_perm=500, seed=0, tag="sig")
    r_nul = run_probe(nul, None, None, n_perm=500, seed=0, tag="nul")
    xinst, xextra = [], {}
    for k in range(200):
        group = f"x{k}"
        for label in (1, 0):
            key = f"{group}#{label}"
            xinst.append({"key": key, "group": group, "lang": "en",
                          "label": label, "text": "same item", "base": group})
            # Strong but non-perfect signal keeps DeLong variance non-zero.
            xval = 1 - label if k % 10 == 0 else label
            xextra[key] = {"readout_signal": float(xval)}
    r_x = run_probe(xinst, None, xextra, n_perm=100, seed=0, tag="x")
    xmissing = dict(xextra)
    xmissing.pop(xinst[0]["key"])
    xkept, xsupport = external_common_support(xinst, xmissing)
    y_tie = np.array([0, 1, 0, 1], float)
    p_tie = np.ones(4)
    ck = {"signal_gate_pass": r_sig["gate"]["pass"] is True,
          "signal_paired_acc": (r_sig["secondary_T_only"]["paired_acc"] or 0) > 0.85,
          "signal_perm_p": (r_sig["secondary_T_only"]["flip_perm_p"] or 1) < 0.01,
          "null_gate_fail": r_nul["gate"]["pass"] is False,
          "null_paired_acc_mid": abs((r_nul["secondary_T_only"]["paired_acc"]
                                      or 0.5) - 0.5) < 0.08,
          "x_gate_independent": r_x["gate"]["pass"] is False
          and r_x["gate_x"]["pass"] is True
          and r_x["gate_x"]["auc_gate"] == r_x["combos"]["S+X"]["auc"]
          and r_x["gate_x"]["auc_base"] == r_x["combos"]["S"]["auc"],
          "x_common_support": len(xkept) == 398
          and xsupport["pairs_after"] == 199,
          "auc_ties_midrank": auc(y_tie, p_tie) == 0.5
          and auc(y_tie[::-1], p_tie[::-1]) == 0.5}
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print(f"  (signal gate dAUC={r_sig['gate']['delta_auc']}, "
          f"null gate dAUC={r_nul['gate']['delta_auc']})")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
