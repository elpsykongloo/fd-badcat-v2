#!/usr/bin/env python3
"""
w4_synth_gen.py — synthetic dialogue generator for the rung-4 stopping head (v1).

Generates EVENT TIMELINES (no audio, no TTS). v1 emits the EXPLICIT silence
structure so the trainer can charge a PREMIUM-FAITHFUL cost:
  - `eous`   : per-EoU records (utt_dur, gap_prev, finality, style)
  - `sigmas` : sigmas[k] = silence-clock seconds between decision k and k+1
  - each op  : launch `eou_idx`, and `rev_eou` (always launch+1) with
               gap_silence = sigmas[launch] when a revision is scheduled
so "window w survives the revision iff w > gap_silence" and "an op's leftover
budget past the FINAL decision delays completion" both hold exactly as in the
WindowLedger semantics.

Learnable structure (preregistered; the head must discover it):
  missing slot -> completion patch likely, small gap; cutoff/hesitant ->
  continuation/revision likely; complete&final -> low afterthought rate with
  WIDE gaps (the round-1 blind-spot class, kept at real mass). The finality
  LABEL is a noisy observation of style (confusion priors ~ the 217 archived
  w4pf calls). Gap mixtures span [0.3, 4.0]s and are NOT fitted to FDB.

Deterministic: one random.Random(seed); config hash printed; commit the recipe,
not the corpus.

Usage: w4_synth_gen.py [--n 8000] [--seed 42] [--tag v1]
Output: exp/w4/synth/dialogues_{tag}.jsonl + validator summary.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp")
sys.path.insert(0, "/root/autodl-tmp/fd-badcat/src")
from stophead import kappa_name, REQUIRED_ARGS   # noqa: E402

# -- preregistered grammar constants (frozen; changes = new tag) --------------
CATALOG = {  # domain -> [(tool, chained_followup|None)]
    "travel":    [("search_flights", "book_flight"), ("update_identity_doc", None)],
    "finance":   [("get_card_benefits", None), ("get_exchange_rate", None),
                  ("modify_autopay", None)],
    "housing":   [("search_apartments", "calculate_commute"),
                  ("update_search_filter", None)],
    "ecommerce": [("track_order", None), ("search_products", "add_to_cart")],
}
STYLE_P = [("complete", 0.70), ("hesitant", 0.18), ("cutoff", 0.12)]
FINALITY_CONFUSION = {  # style -> label distribution (priors ~ w4pf archive)
    "complete": [("final", 0.80), ("hesitant", 0.15), ("unfinished", 0.05)],
    "hesitant": [("final", 0.25), ("hesitant", 0.60), ("unfinished", 0.15)],
    "cutoff":   [("final", 0.10), ("hesitant", 0.20), ("unfinished", 0.70)],
}
P_SLOT_MISSING = 0.25         # multi-slot tools: one required slot omitted
P_REV = {"slot_completion": 0.85, "cutoff_continuation": 0.90,
         "hesitant_revision": 0.45, "afterthought": 0.10}
P_CHAIN = 0.6                 # chained followup launched same EoU via $RESULT
P_UPSTREAM_HIT = 0.30         # a revision on the parent also invalidates child
INTER_REQ_SIL = (1.0, 3.0)    # silence between independent requests (uniform)
REV_UTT_DUR = (1.0, 3.0)      # revising utterances are short plain statements


def gap_small(rng):           # completion / continuation: revision comes fast
    return min(2.5, max(0.3, rng.lognormvariate(-0.22, 0.5)))   # median ~0.8


def gap_wide(rng):            # afterthought: broad support, incl. long tail
    u = rng.random()
    if u < 0.4:
        return rng.uniform(0.5, 1.5)
    if u < 0.8:
        return rng.uniform(1.5, 3.0)
    return rng.uniform(3.0, 4.0)


def pick(rng, table):
    x, acc = rng.random(), 0.0
    for v, p in table:
        acc += p
        if x < acc:
            return v
    return table[-1][0]


def gen_dialogue(rng, did):
    domain = rng.choice(list(CATALOG))
    n_intents = rng.randint(1, 3)
    eous, sigmas, ops = [], [], []
    n_prior = 0
    for it in range(n_intents):
        tool, chain = rng.choice(CATALOG[domain])
        style = pick(rng, STYLE_P)
        finality = pick(rng, FINALITY_CONFUSION[style])
        multi = len(REQUIRED_ARGS.get(tool, [])) >= 2
        slots_missing = 1 if (multi and rng.random() < P_SLOT_MISSING) else 0
        utt_dur = rng.uniform(1.5, 6.0)
        gap_prev = (round(sigmas[-1] + 0.64, 2) if sigmas
                    else round(rng.uniform(0.5, 2.0), 2))
        launch = len(eous)
        eous.append({"utt_dur": round(utt_dur, 2), "gap_prev": gap_prev,
                     "finality": finality, "style": style})

        # revision determination — causally tied to STYLE/slots (ground truth);
        # `finality` is only the noisy observation the head sees.
        gap = kind = None
        if slots_missing and rng.random() < P_REV["slot_completion"]:
            gap, kind = gap_small(rng), "slot_completion"
        elif style == "cutoff" and rng.random() < P_REV["cutoff_continuation"]:
            gap, kind = gap_small(rng), "cutoff_continuation"
        elif style == "hesitant" and rng.random() < P_REV["hesitant_revision"]:
            gap, kind = (gap_small(rng) + gap_wide(rng)) / 2, "hesitant_revision"
        elif rng.random() < P_REV["afterthought"]:
            gap, kind = gap_wide(rng), "afterthought"
        rev_eou = len(eous) if gap is not None else None

        base = {"eou_idx": launch, "utt_dur": round(utt_dur, 2),
                "gap_prev": gap_prev, "n_prior_ops": n_prior,
                "domain": domain, "finality": finality, "style": style}
        ops.append({**base, "fn": tool, "kappa": kappa_name(tool),
                    "slots_missing": slots_missing, "chain_dep": 0,
                    "gap_silence": round(gap, 3) if gap else None,
                    "rev_eou": rev_eou, "revision_kind": kind})
        n_prior += 1
        if chain and rng.random() < P_CHAIN:
            hit = gap is not None and rng.random() < P_UPSTREAM_HIT
            ops.append({**base, "fn": chain, "kappa": kappa_name(chain),
                        "slots_missing": 0, "chain_dep": 1,
                        "n_prior_ops": n_prior,
                        "gap_silence": round(gap, 3) if hit else None,
                        "rev_eou": rev_eou if hit else None,
                        "revision_kind": "upstream" if hit else None})
            n_prior += 1

        if gap is not None:      # the revising utterance is itself an EoU
            sigmas.append(round(gap, 3))          # silence launch -> revision
            eous.append({"utt_dur": round(rng.uniform(*REV_UTT_DUR), 2),
                         "gap_prev": round(gap + 0.64, 2),
                         "finality": pick(rng, FINALITY_CONFUSION["complete"]),
                         "style": "complete"})
        if it < n_intents - 1:   # silence before the next independent request
            sigmas.append(round(rng.uniform(*INTER_REQ_SIL), 3))
    return {"did": did, "domain": domain, "version": "v1",
            "eous": eous, "sigmas": sigmas, "ops": ops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    cfg = {k: v for k, v in globals().items()
           if k.isupper() and isinstance(v, (dict, list, tuple, float, int))}
    cfg_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
    rng = random.Random(args.seed)
    out = Path(f"/root/autodl-tmp/fd-badcat/exp/w4/synth/dialogues_{args.tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    kinds, doms, fin_by_style = Counter(), Counter(), Counter()
    n_ops = n_rev = 0
    with out.open("w") as fh:
        for i in range(args.n):
            d = gen_dialogue(rng, i)
            assert len(d["sigmas"]) == len(d["eous"]) - 1
            doms[d["domain"]] += 1
            for o in d["ops"]:
                n_ops += 1
                fin_by_style[(o["style"], o["finality"])] += 1
                if o["gap_silence"] is not None:
                    assert 0.25 <= o["gap_silence"] <= 4.5
                    assert o["rev_eou"] == o["eou_idx"] + 1
                    assert abs(d["sigmas"][o["eou_idx"]] - o["gap_silence"]) < 1e-6
                    n_rev += 1
                    kinds[o["revision_kind"]] += 1
            fh.write(json.dumps(d) + "\n")

    print(f"config_hash={cfg_hash} seed={args.seed} n={args.n}")
    print(f"dialogues={args.n} ops={n_ops} revised={n_rev} ({n_rev / n_ops:.1%})")
    print("revision kinds:", dict(kinds))
    print("domains:", dict(doms))
    print("finality|style confusion:", {f"{s}->{f}": c for (s, f), c
                                        in sorted(fin_by_style.items())})
    print("->", out)


if __name__ == "__main__":
    main()
