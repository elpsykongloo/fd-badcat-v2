"""stophead.py — W4 rung-4 stopping-time head v0 (shared by labeler, trainer, runtime).

The head estimates the discrete-time revision hazard
    lambda_hat(t) = P(a revision hits this op in silence-clock (t, t+H] | alive at t)
from dialogue-state/positional features (+ the Omni zero-shot finality label).
The policy on top is P2's closed-form threshold rule — NOT learned:
    window(op) = min{ t in T_GRID : lambda_hat(t) * C_kappa <= c_w },  clamped to W_CAP.

Feature vector (FEATS order) is the single source of truth for BOTH the synthetic
labeler and the runtime plugin — never let the two drift.
Preregistered constants: C_KAPPA, T_GRID/H, W_CAP (docs/w4_ladder_design.md §8).
"""
import json
import math
from pathlib import Path

from tact.tools import REVERSIBILITY

KAPPAS = ("READ", "REV", "COMP", "IRR")
C_KAPPA = {"READ": 1.0, "REV": 2.0, "COMP": 4.0, "IRR": 8.0}   # cost weights (prereg)
T_GRID = [round(i * 0.25, 2) for i in range(13)]               # 0 .. 3.0 s silence clock
H = 0.25                                                       # hazard step horizon
W_CAP = 2.5                                                    # window clamp (prereg)
DOMAINS = ("travel", "finance", "housing", "ecommerce")
FINALITY = ("final", "hesitant", "unfinished")

# Required args per FDB tool (from mock_apis.py signatures) — runtime slots_missing.
REQUIRED_ARGS = {
    "search_flights": ["destination", "date"],
    "book_flight": ["passenger_name"],
    "update_identity_doc": ["doc_type", "doc_number"],
    "get_card_benefits": ["card_type"],
    "get_exchange_rate": ["amount", "from_currency", "to_currency"],
    "modify_autopay": ["bill_type", "source_account"],
    "search_apartments": ["city", "bedrooms", "max_price"],
    "calculate_commute": ["origin_address", "destination_address"],
    "update_search_filter": ["filter_name", "value"],
    "track_order": ["order_id"],
    "search_products": ["query"],
    "add_to_cart": ["product_id", "quantity"],
}

FEATS = (["t", "eou_idx", "utt_dur", "gap_prev", "n_prior_ops",
          "slots_missing", "chain_dep"]
         + ["k_" + k for k in KAPPAS]
         + ["f_" + f for f in FINALITY]
         + ["d_" + d for d in DOMAINS])


def kappa_name(fn):
    r = REVERSIBILITY.get(fn)
    return r.name if r is not None else "IRR"


def slots_missing_from_args(fn, args):
    args = args or {}
    miss = 0
    for a in REQUIRED_ARGS.get(fn, []):
        v = args.get(a)
        if v is None or (isinstance(v, str) and not v.strip()):
            miss += 1
    return miss


def chain_dep_from_args(args):
    return int(any(isinstance(v, str) and "$RESULT" in v
                   for v in (args or {}).values()))


def featurize(s, t):
    """s: op-context dict {eou_idx, utt_dur, gap_prev, n_prior_ops, slots_missing,
    chain_dep, kappa, finality, domain} -> feature row matching FEATS."""
    row = [float(t), float(s["eou_idx"]), float(s["utt_dur"]), float(s["gap_prev"]),
           float(s["n_prior_ops"]), float(s["slots_missing"]), float(s["chain_dep"])]
    row += [1.0 if s["kappa"] == k else 0.0 for k in KAPPAS]
    f = s.get("finality") or "hesitant"
    row += [1.0 if f == x else 0.0 for x in FINALITY]
    d = s.get("domain") if s.get("domain") in DOMAINS else None
    row += [1.0 if d == x else 0.0 for x in DOMAINS]
    return row


class StopHead:
    """Logistic hazard model, stored as plain JSON (no runtime sklearn/torch)."""

    def __init__(self, d):
        self.d = d

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def hazard(self, row):
        z = self.d["b"]
        for wi, xi, mu, sd in zip(self.d["w"], row, self.d["mean"], self.d["std"]):
            z += wi * ((xi - mu) / (sd if sd > 1e-9 else 1.0))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def window(self, s, c_w=None):
        """P2 threshold rule: first grid time where hazard * C_kappa <= c_w."""
        cw = self.d["c_w"] if c_w is None else c_w
        ck = C_KAPPA[s["kappa"]]
        for t in self.d.get("t_grid", T_GRID):
            if self.hazard(featurize(s, t)) * ck <= cw:
                return min(float(t), W_CAP)
        return W_CAP


def make_learned_delta_fn(model, ctx):
    """Runtime per-EoU factory. ctx = {eou_idx, utt_dur, gap_prev, n_prior_ops,
    finality, domain}; returns delta_fn(fn, args)->window for apply_decision_ops."""
    def fn(fname, args=None):
        s = dict(ctx)
        s["kappa"] = kappa_name(fname)
        s["slots_missing"] = slots_missing_from_args(fname, args)
        s["chain_dep"] = chain_dep_from_args(args)
        return model.window(s)
    return fn
