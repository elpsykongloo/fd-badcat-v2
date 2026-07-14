#!/usr/bin/env python3
"""
w4_synth_gen.py — synthetic dialogue generator for the rung-4 stopping head (v2).

Generates EVENT TIMELINES (no audio, no TTS) with the v1 explicit silence
structure (`eous` / `sigmas` / per-op `rev_eou`, gap_silence == sigmas[launch])
so the trainer can replay a premium-faithful cost.

v2 = DOMAIN RANDOMIZATION (docs/w4_ladder_design.md §12, lever a). The §11
ceiling diagnostic localized the v0/v1 failures to sim-to-real RANKING
transfer: a single hand-set grammar config is one point in distribution space
and the head overfits its marginals (rev prior 32.5%, unfitted gap shapes).
v2 samples a fresh config PER DIALOGUE from preregistered RANGES — the head
can no longer rely on any fixed marginal and must learn rank-stable structure
(missing slot / cutoff-unfinished / short utterance -> revision risk).

Two structural corrections (engine/track knowledge, NOT FDB statistics):
  * GAP_FLOOR = 0.64 hold + 1.0 nominal infer: inter-decision silence in the
    nominal throughput track cannot be smaller (the revising utterance itself
    passes EoU hold + decision infer). gap = GAP_FLOOR + sigma_pre, where
    sigma_pre (user hesitation before revising) is the randomized quantity.
    v0/v1 gaps in [0.3, 4.0] were physically unrealizable.
  * utt_dur is style-conditioned (a cutoff utterance IS shorter) — v0/v1 drew
    it style-independent, leaving a live deployment feature untrained.

Deterministic: one random.Random(seed) drives configs AND content; config_hash
covers the RANGES table + fixed grammar. Commit the recipe, not the corpus.

Usage: w4_synth_gen.py [--n 8000] [--seed 42] [--tag v2]
Output: exp/w4/synth/dialogues_{tag}.jsonl + validator summary.
"""
import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from stophead import kappa_name, GAP_FLOOR, REQUIRED_ARGS   # noqa: E402

# -- fixed grammar (unchanged from v1) ----------------------------------------
CATALOG = {  # domain -> [(tool, chained_followup|None)]
    "travel":    [("search_flights", "book_flight"), ("update_identity_doc", None)],
    "finance":   [("get_card_benefits", None), ("get_exchange_rate", None),
                  ("modify_autopay", None)],
    "housing":   [("search_apartments", "calculate_commute"),
                  ("update_search_filter", None)],
    "ecommerce": [("track_order", None), ("search_products", "add_to_cart")],
}
FINALITY_BASE = {  # style -> label distribution (priors ~ the 217 w4pf calls)
    "complete": [("final", 0.80), ("hesitant", 0.15), ("unfinished", 0.05)],
    "hesitant": [("final", 0.25), ("hesitant", 0.60), ("unfinished", 0.15)],
    "cutoff":   [("final", 0.10), ("hesitant", 0.20), ("unfinished", 0.70)],
}
REV_UTT_DUR = (1.0, 3.0)      # revising utterances are short plain statements

# -- preregistered randomization RANGES (per-dialogue config; changes = new tag)
RANGES = {
    # style mix: raw weights, normalized (complete share spans ~0.32..0.88)
    "style_w_complete": (2.0, 6.0),
    "style_w_hesitant": (0.5, 2.5),
    "style_w_cutoff":   (0.3, 1.8),
    # revision channels (pre-intensity)
    "p_slot_missing":     (0.08, 0.40),
    "p_rev_slot":         (0.45, 0.95),
    "p_rev_cutoff":       (0.45, 0.95),
    "p_rev_hesitant":     (0.15, 0.70),
    "p_rev_afterthought": (0.02, 0.25),
    # global revision intensity, LOG-uniform, multiplies all four channels:
    # stretches the per-dialogue op-level rev prior down to ~3% so that
    # deployment-like low-revision regimes are INSIDE the training support
    # (v1's single 32.5% prior was the declared death cause).
    "rev_intensity":      (0.15, 1.50),
    "first_intent_mult":  (1.0, 2.0),   # generic "first request underspecified"
    # chains
    "p_chain":        (0.30, 0.80),
    "p_upstream_hit": (0.10, 0.50),
    # sigma_pre (user hesitation before the revising utterance): gap = floor + it
    "sig_fast_mu": (-1.5, -0.5),        # lognormal median ~0.22..0.61 s
    "sig_fast_s":  (0.4, 0.8),
    "sig_wide_p_lo": (0.35, 0.65),      # afterthought: P(low band)
    # inter-request silence (tail-premium economics)
    "inter_req_lo": (0.6, 1.5),
    "inter_req_hi": (2.0, 4.0),
    # style-conditioned utterance duration: (lo range, hi range) per style
    "utt_complete": ((1.2, 2.0), (4.5, 6.5)),
    "utt_hesitant": ((1.0, 1.6), (3.5, 5.5)),
    "utt_cutoff":   ((0.6, 1.2), (2.5, 4.0)),
}
FIN_JITTER = 0.6              # log-multiplicative jitter on confusion rows
SIG_FAST_CLAMP = (0.03, 1.5)
SIG_WIDE_LO = (0.10, 1.0)     # afterthought sigma_pre bands
SIG_WIDE_HI = (1.0, 2.80)
GAP_MAX = GAP_FLOOR + SIG_WIDE_HI[1] + 0.01   # validator bound (~4.45)

# -- C1 world re-anchor (w4v3_design.md §10.2/§11.3; measured HumDial values).
# Default OFF: the v2 path stays byte-identical (this dict is EXCLUDED from the
# default config_hash so historical b62a069cd900 keeps reproducing). --anchor
# v3c1 swaps ONLY the three named world quantities; grammar, chains and the
# revision channels stay v2.
ANCHOR_V3C1 = {
    # sigma_pre <- empirical resample of the 748 measured max-pauses,
    # clamped to the observed support (census vad_pause receipts)
    "sig_clamp": (0.38, 3.60),
    # finality emission <- measured Omni confusion rows (readout receipts):
    # complete <- the true-end row, cutoff <- the continuation row; the
    # hesitant style row is UNMEASURED on HumDial (the judge never emitted
    # 'hesitant' there) and keeps the v2 base row. FIN_JITTER still applies.
    "fin_base": {
        "complete": [("final", 0.700), ("hesitant", 0.0), ("unfinished", 0.300)],
        "cutoff":   [("final", 0.147), ("hesitant", 0.0), ("unfinished", 0.853)],
    },
    # utt_dur bands <- measured bilingual user-utterance p10-p90 envelope
    # ([[2.0,4.5],[5.0,9.0]] frozen in §10.2), style multipliers 1.0/0.8/0.5
    "utt": {"utt_complete": ((2.0, 4.5), (5.0, 9.0)),
            "utt_hesitant": ((1.6, 3.6), (4.0, 7.2)),
            "utt_cutoff":   ((1.0, 2.25), (2.5, 4.5))},
}


def load_pause_pool(path):
    """Measured break_mid pause durations -> sorted, clamped resampling pool."""
    lo, hi = ANCHOR_V3C1["sig_clamp"]
    pool = sorted(min(hi, max(lo, json.loads(line)["pause_dur"]))
                  for line in Path(path).read_text().splitlines()
                  if json.loads(line).get("kind") == "break_mid")
    if len(pool) < 300:                     # §4 kill threshold as a load guard
        raise SystemExit(f"pause pool too small: {len(pool)} < 300")
    return pool


def U(rng, key):
    return rng.uniform(*RANGES[key])


def jitter_row(rng, row):
    w = [(lab, p * math.exp(rng.uniform(-FIN_JITTER, FIN_JITTER))) for lab, p in row]
    tot = sum(p for _, p in w)
    return [(lab, p / tot) for lab, p in w]


def sample_config(rng, anchor=None):
    wc, wh, wk = (U(rng, "style_w_complete"), U(rng, "style_w_hesitant"),
                  U(rng, "style_w_cutoff"))
    tot = wc + wh + wk
    utt = {}
    for style in ("complete", "hesitant", "cutoff"):
        (lo_r, hi_r) = ((anchor or {}).get("utt", {}).get("utt_" + style)
                        or RANGES["utt_" + style])
        utt[style] = (rng.uniform(*lo_r), rng.uniform(*hi_r))
    lo, hi = RANGES["rev_intensity"]
    rev_m = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    fin_base = {**FINALITY_BASE, **(anchor or {}).get("fin_base", {})}
    return {
        "style_p": [("complete", wc / tot), ("hesitant", wh / tot),
                    ("cutoff", wk / tot)],
        "fin": {s: jitter_row(rng, row) for s, row in fin_base.items()},
        "p_slot_missing": U(rng, "p_slot_missing"),
        "rev_intensity": rev_m,
        "p_rev": {"slot_completion": min(0.97, U(rng, "p_rev_slot") * rev_m),
                  "cutoff_continuation": min(0.97, U(rng, "p_rev_cutoff") * rev_m),
                  "hesitant_revision": min(0.97, U(rng, "p_rev_hesitant") * rev_m),
                  "afterthought": min(0.97, U(rng, "p_rev_afterthought") * rev_m)},
        "m0": U(rng, "first_intent_mult"),
        "p_chain": U(rng, "p_chain"),
        "p_upstream_hit": U(rng, "p_upstream_hit"),
        "sig_fast": (U(rng, "sig_fast_mu"), U(rng, "sig_fast_s")),
        "sig_wide_p_lo": U(rng, "sig_wide_p_lo"),
        "inter_req": (U(rng, "inter_req_lo"), U(rng, "inter_req_hi")),
        "utt": utt,
    }


def sig_fast(rng, cfg):
    mu, s = cfg["sig_fast"]
    return min(SIG_FAST_CLAMP[1], max(SIG_FAST_CLAMP[0], rng.lognormvariate(mu, s)))


def sig_wide(rng, cfg):
    if rng.random() < cfg["sig_wide_p_lo"]:
        return rng.uniform(*SIG_WIDE_LO)
    return rng.uniform(*SIG_WIDE_HI)


def pick(rng, table):
    x, acc = rng.random(), 0.0
    for v, p in table:
        acc += p
        if x < acc:
            return v
    return table[-1][0]


def rev_prior(cfg, domain):
    """Exact per-op expected revision prob under cfg (first-intent multiplier
    ignored — informational, for the validator's spread readout)."""
    multi = [len(REQUIRED_ARGS.get(t, [])) >= 2 for t, _ in CATALOG[domain]]
    a = (sum(multi) / len(multi)) * cfg["p_slot_missing"] * cfg["p_rev"]["slot_completion"]
    pr, pat = cfg["p_rev"], cfg["p_rev"]["afterthought"]
    q = {"complete": pat,
         "hesitant": pr["hesitant_revision"] + (1 - pr["hesitant_revision"]) * pat,
         "cutoff": pr["cutoff_continuation"] + (1 - pr["cutoff_continuation"]) * pat}
    style = sum(p * q[s] for s, p in cfg["style_p"])
    return a + (1 - a) * style


def gen_dialogue(rng, did, anchor=None, pool=None):
    cfg = sample_config(rng, anchor)
    emp = (lambda: pool[rng.randrange(len(pool))]) if pool else None
    domain = rng.choice(list(CATALOG))
    n_intents = rng.randint(1, 3)
    eous, sigmas, ops = [], [], []
    n_prior = 0
    for it in range(n_intents):
        tool, chain = rng.choice(CATALOG[domain])
        style = pick(rng, cfg["style_p"])
        finality = pick(rng, cfg["fin"][style])
        multi = len(REQUIRED_ARGS.get(tool, [])) >= 2
        slots_missing = 1 if (multi and rng.random() < cfg["p_slot_missing"]) else 0
        utt_dur = rng.uniform(*cfg["utt"][style])
        gap_prev = (round(sigmas[-1] + 0.64, 2) if sigmas
                    else round(rng.uniform(0.5, 2.0), 2))
        launch = len(eous)
        eous.append({"utt_dur": round(utt_dur, 2), "gap_prev": gap_prev,
                     "finality": finality, "style": style})

        # revision determination — causally tied to STYLE/slots (ground truth);
        # `finality` is only the noisy observation the head sees. gap =
        # GAP_FLOOR + sigma_pre (structural floor of the nominal track).
        m = cfg["m0"] if it == 0 else 1.0
        p_eff = lambda k: min(0.97, cfg["p_rev"][k] * m)
        sig = kind = None
        if slots_missing and rng.random() < p_eff("slot_completion"):
            sig, kind = emp() if emp else sig_fast(rng, cfg), "slot_completion"
        elif style == "cutoff" and rng.random() < p_eff("cutoff_continuation"):
            sig, kind = emp() if emp else sig_fast(rng, cfg), "cutoff_continuation"
        elif style == "hesitant" and rng.random() < p_eff("hesitant_revision"):
            sig, kind = (emp() if emp else
                         (sig_fast(rng, cfg) + sig_wide(rng, cfg)) / 2), "hesitant_revision"
        elif rng.random() < p_eff("afterthought"):
            sig, kind = emp() if emp else sig_wide(rng, cfg), "afterthought"
        gap = None if sig is None else round(GAP_FLOOR + sig, 3)
        rev_eou = len(eous) if gap is not None else None

        base = {"eou_idx": launch, "utt_dur": round(utt_dur, 2),
                "gap_prev": gap_prev, "n_prior_ops": n_prior,
                "domain": domain, "finality": finality, "style": style}
        ops.append({**base, "fn": tool, "kappa": kappa_name(tool),
                    "slots_missing": slots_missing, "chain_dep": 0,
                    "gap_silence": gap,
                    "rev_eou": rev_eou, "revision_kind": kind})
        n_prior += 1
        if chain and rng.random() < cfg["p_chain"]:
            hit = gap is not None and rng.random() < cfg["p_upstream_hit"]
            ops.append({**base, "fn": chain, "kappa": kappa_name(chain),
                        "slots_missing": 0, "chain_dep": 1,
                        "n_prior_ops": n_prior,
                        "gap_silence": gap if hit else None,
                        "rev_eou": rev_eou if hit else None,
                        "revision_kind": "upstream" if hit else None})
            n_prior += 1

        if gap is not None:      # the revising utterance is itself an EoU
            sigmas.append(gap)                    # silence launch -> revision
            eous.append({"utt_dur": round(rng.uniform(*REV_UTT_DUR), 2),
                         "gap_prev": round(gap + 0.64, 2),
                         "finality": pick(rng, cfg["fin"]["complete"]),
                         "style": "complete"})
        if it < n_intents - 1:   # silence before the next independent request
            sigmas.append(round(rng.uniform(*cfg["inter_req"]), 3))
    rec = {"did": did, "domain": domain,
           "version": "v3c1" if anchor else "v2",
           "eous": eous, "sigmas": sigmas, "ops": ops,
           "rev_prior": round(rev_prior(cfg, domain), 4),
           "cfg": json.loads(json.dumps(cfg), parse_float=lambda x: round(float(x), 4))}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--anchor", choices=["v3c1"],
                    help="C1 world re-anchor (w4v3_design.md §11.3); default "
                         "off = frozen v2 path, byte-identical")
    ap.add_argument("--pauses", default=str(ROOT / "exp/w4v3/humdial_cuts.jsonl"),
                    help="census cuts jsonl for the empirical sigma_pre pool")
    ap.add_argument("--outdir",
                    default="/root/autodl-tmp/fd-badcat/exp/w4/synth")
    args = ap.parse_args()

    anchor = ANCHOR_V3C1 if args.anchor == "v3c1" else None
    pool = load_pause_pool(args.pauses) if anchor else None
    # ANCHOR_V3C1 is excluded from the default hash payload so the frozen v2
    # config_hash (b62a069cd900) keeps reproducing; anchor runs extend it.
    cfg_const = {k: v for k, v in globals().items()
                 if k.isupper() and k != "ANCHOR_V3C1"
                 and isinstance(v, (dict, list, tuple, float, int))}
    if anchor:
        cfg_const["ANCHOR"] = {
            "mode": args.anchor, **ANCHOR_V3C1,
            "pool_n": len(pool),
            "pool_sha": hashlib.sha256(
                json.dumps(pool).encode()).hexdigest()[:12]}
    cfg_hash = hashlib.sha256(
        json.dumps(cfg_const, sort_keys=True, default=str).encode()).hexdigest()[:12]
    rng = random.Random(args.seed)
    out = Path(args.outdir) / f"dialogues_{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    gap_max = (GAP_FLOOR + ANCHOR_V3C1["sig_clamp"][1] + 0.01) if anchor else GAP_MAX

    kinds, doms, fin_by_style = Counter(), Counter(), Counter()
    n_ops = n_rev = 0
    priors, gaps = [], []
    with out.open("w") as fh:
        for i in range(args.n):
            d = gen_dialogue(rng, i, anchor, pool)
            assert len(d["sigmas"]) == len(d["eous"]) - 1
            doms[d["domain"]] += 1
            priors.append(d["rev_prior"])
            for o in d["ops"]:
                n_ops += 1
                fin_by_style[(o["style"], o["finality"])] += 1
                if o["gap_silence"] is not None:
                    assert GAP_FLOOR - 1e-9 <= o["gap_silence"] <= gap_max
                    assert o["rev_eou"] == o["eou_idx"] + 1
                    assert abs(d["sigmas"][o["eou_idx"]] - o["gap_silence"]) < 1e-6
                    n_rev += 1
                    gaps.append(o["gap_silence"])
                    kinds[o["revision_kind"]] += 1
            fh.write(json.dumps(d) + "\n")

    priors.sort()
    gaps.sort()
    qt = lambda v, q: v[min(len(v) - 1, int(q * len(v)))] if v else None
    print(f"config_hash={cfg_hash} seed={args.seed} n={args.n}")
    print(f"dialogues={args.n} ops={n_ops} revised={n_rev} ({n_rev / n_ops:.1%})")
    print(f"rev_prior spread (per-dialogue expected op rev prob) "
          f"p10/p50/p90 = {qt(priors, .1):.3f}/{qt(priors, .5):.3f}/{qt(priors, .9):.3f}")
    print(f"gap_silence p10/p50/p90 = {qt(gaps, .1):.2f}/{qt(gaps, .5):.2f}/"
          f"{qt(gaps, .9):.2f} (floor {GAP_FLOOR}, rescuable<="
          f"{GAP_FLOOR + 0.86:.2f} share {sum(g < 2.5 for g in gaps) / max(1, len(gaps)):.1%})")
    print("revision kinds:", dict(kinds))
    print("domains:", dict(doms))
    print("finality|style confusion:", {f"{s}->{f}": c for (s, f), c
                                        in sorted(fin_by_style.items())})
    print("->", out)


if __name__ == "__main__":
    main()
