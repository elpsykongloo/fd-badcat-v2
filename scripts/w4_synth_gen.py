#!/usr/bin/env python3
"""
w4_synth_gen.py — synthetic dialogue generator for the rung-4 stopping head.

Generates EVENT TIMELINES (no audio, no TTS): each dialogue is a sequence of
EoUs; each EoU launches ops (with slot structure) and may schedule a future
revision landing on an op after `gap_silence` seconds of SILENCE-CLOCK time —
the same clock the WindowLedger burns, so "window w survives a revision iff
gap_silence < w" holds by construction on both sides.

The generator is, deliberately and explicitly, a sampler of (op-features, gap)
pairs with structured correlations + a thin cosmetic timeline. The learnable
structure (preregistered; the head must discover it, we never feed it labels):
  - missing required slot        -> completion patch very likely, small gap
  - cutoff / hesitant utterance  -> continuation revision likely
  - complete & final             -> low afterthought rate, WIDE gaps  <- the
    blind-spot class the ladder round-1 identified; kept with real mass
  - finality LABEL is a noisy observation of utterance style (confusion table
    with priors set from the 217 archived w4pf calls), never the ground truth
Revision gap mixtures span [0.3, 4.0]s and are NOT fitted to FDB (leakage
firewall; FDB gap stats are used only post-hoc to check coverage).

Deterministic: one random.Random(seed); config hash printed; corpus is
regenerable from (seed, N) — commit the recipe, not the corpus.

Usage: w4_synth_gen.py [--n 8000] [--seed 42] [--tag v0]
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
from stophead import kappa_name, REQUIRED_ARGS, DOMAINS   # noqa: E402

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
    ops, eou_idx, n_prior = [], 0, 0
    for _ in range(n_intents):
        tool, chain = rng.choice(CATALOG[domain])
        style = pick(rng, STYLE_P)
        finality = pick(rng, FINALITY_CONFUSION[style])
        multi = len(REQUIRED_ARGS.get(tool, [])) >= 2
        slots_missing = 1 if (multi and rng.random() < P_SLOT_MISSING) else 0
        utt_dur = rng.uniform(1.5, 6.0)
        gap_prev = rng.uniform(0.8, 3.0) if eou_idx else rng.uniform(0.5, 2.0)

        # revision determination — causally tied to STYLE/slots (ground truth),
        # while `finality` is only the noisy observation the head will see.
        gap = kind = None
        if slots_missing and rng.random() < P_REV["slot_completion"]:
            gap, kind = gap_small(rng), "slot_completion"
        elif style == "cutoff" and rng.random() < P_REV["cutoff_continuation"]:
            gap, kind = gap_small(rng), "cutoff_continuation"
        elif style == "hesitant" and rng.random() < P_REV["hesitant_revision"]:
            gap, kind = (gap_small(rng) + gap_wide(rng)) / 2, "hesitant_revision"
        elif rng.random() < P_REV["afterthought"]:
            gap, kind = gap_wide(rng), "afterthought"

        base = {"eou_idx": eou_idx, "utt_dur": round(utt_dur, 2),
                "gap_prev": round(gap_prev, 2), "n_prior_ops": n_prior,
                "domain": domain, "finality": finality, "style": style}
        ops.append({**base, "fn": tool, "kappa": kappa_name(tool),
                    "slots_missing": slots_missing, "chain_dep": 0,
                    "gap_silence": round(gap, 3) if gap else None,
                    "revision_kind": kind})
        n_prior += 1
        if chain and rng.random() < P_CHAIN:
            cgap = ckind = None
            if gap is not None and rng.random() < P_UPSTREAM_HIT:
                cgap, ckind = gap, "upstream"          # same silence event
            ops.append({**base, "fn": chain, "kappa": kappa_name(chain),
                        "slots_missing": 0, "chain_dep": 1,
                        "n_prior_ops": n_prior,
                        "gap_silence": round(cgap, 3) if cgap else None,
                        "revision_kind": ckind})
            n_prior += 1
        eou_idx += 1
        if gap is not None:
            eou_idx += 1        # the revising utterance is itself an EoU
    return {"did": did, "domain": domain, "n_eou": eou_idx, "ops": ops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="v0")
    args = ap.parse_args()

    cfg = {k: v for k, v in globals().items()
           if k.isupper() and isinstance(v, (dict, list, float, int))}
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
            doms[d["domain"]] += 1
            for o in d["ops"]:
                n_ops += 1
                fin_by_style[(o["style"], o["finality"])] += 1
                assert o["gap_silence"] is None or 0.25 <= o["gap_silence"] <= 4.5
                if o["gap_silence"] is not None:
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
