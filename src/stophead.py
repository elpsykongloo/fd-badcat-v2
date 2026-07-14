"""stophead.py — W4 rung-4 stopping-time head (shared by labeler, trainer, runtime).

Two head/policy generations live here; model JSONs are SELF-DESCRIBING
(feature list, grid, policy form), so old archives stay loadable bit-for-bit.

v0/v1 — logistic HAZARD head + P2 threshold rule (continuous window):
    lambda_hat(t) = P(a revision hits this op in silence-clock (t, t+H] | alive at t)
    window(op)    = min{ t in T_GRID : lambda_hat(t) * C_kappa <= c_w },  clamp W_CAP.

v2 — TWO-STAGE policy (docs/w4_ladder_design.md §12). The ceiling diagnostic
(§11) showed (i) the bottleneck is sim-to-real RANKING transfer, not capacity
or window shape, and (ii) with the commit-barrier grace accounted for, the
fixed W=1.5 window covers every load-bearing FDB rescue. v2 therefore
collapses the policy to a preregistered two-point set and makes ranking the
only free variable:
    risk(op)   = 1 - prod_{t in T_GRID, t < RISK_HORIZON} (1 - lambda_hat(t))
    window(op) = W_PROTECT if risk(op) >= theta else 0.0
Downside is structurally bounded: protect-all == the fixed delta*=1.5 arm.

Feature rows are built BY NAME from the model's own `feats` list — the single
source shared by labeler/trainer/runtime (never let them drift). FEATS is the
18-dim v0/v1 set; FEATS_V2 is the §12 signal core (shrunk drift surface).

Preregistered constants: C_KAPPA, T_GRID/H, W_CAP (§8/§10); W_PROTECT, GRACE,
RISK_HORIZON, GAP_FLOOR (§12). GRACE and GAP_FLOOR are STRUCTURAL constants of
the nominal throughput track (0.64 EoU hold + 1.0s nominal decision infer;
expiries inside the next decision's guard defer under the commit barrier and
get patch-rescued) — engine/track knowledge, not FDB statistics.
"""
import json
import math
from pathlib import Path

from tact.tools import REVERSIBILITY

KAPPAS = ("READ", "REV", "COMP", "IRR")
C_KAPPA = {"READ": 1.0, "REV": 2.0, "COMP": 4.0, "IRR": 8.0}   # cost weights (prereg)
# v1: grid/cap extended to cover the generator's own gap support (<=4.0s) —
# v0's 2.5 cap couldn't even express rescues its training distribution contains
# (design bug, w4_ladder_design §10). v0 model JSONs carry their own t_grid/w_cap.
T_GRID = [round(i * 0.25, 2) for i in range(18)]               # 0 .. 4.25 s silence clock
H = 0.25                                                       # hazard step horizon
W_CAP = 4.0                                                    # window clamp (prereg v1)
DOMAINS = ("travel", "finance", "housing", "ecommerce")
FINALITY = ("final", "hesitant", "unfinished")

# -- v2 structural constants (prereg §12; nominal-track engine knowledge) -----
HOLD_S = 0.64                 # frozen EoU hold
GRACE = 1.0                   # nominal decision infer: an expiry inside the next
                              # decision's guard (t_disp, t_dec] defers under the
                              # commit barrier and is patch-rescued
                              # => rescued iff window > gap - GRACE.
GAP_FLOOR = HOLD_S + GRACE    # minimum realizable inter-decision silence (1.64s)
W_PROTECT = 1.5               # two-stage protect window (= delta*)
RISK_HORIZON = W_PROTECT + GRACE   # rescuable-gap support: risk integrates [0, 2.5)

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

# §12 signal core (ceiling diagnostic §11 loadings: slots_missing 0.67 /
# f_unfinished 0.62 / short utt_dur / early eou_idx). Deliberately EXCLUDED:
# kappa one-hots (single-feature AUC ~= chance 0.48-0.53, and the two-stage
# economics on FDB binary pass are kappa-flat), gap_prev / n_prior_ops /
# chain_dep (~= chance), domain one-hots (pure synthetic-artifact carriers).
# `t` stays: the hazard shape over t places mass inside vs beyond RISK_HORIZON.
FEATS_V2 = ["t", "eou_idx", "utt_dur", "slots_missing",
            "f_final", "f_hesitant", "f_unfinished"]


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


def featurize(s, t, feats=FEATS):
    """s: op-context dict {eou_idx, utt_dur, gap_prev, n_prior_ops, slots_missing,
    chain_dep, kappa, finality, domain} -> feature row matching `feats` BY NAME
    (default = the 18-dim v0/v1 order, bit-identical to the old builder)."""
    f = s.get("finality") or "hesitant"
    d = s.get("domain") if s.get("domain") in DOMAINS else None
    vals = {"t": float(t), "eou_idx": float(s["eou_idx"]),
            "utt_dur": float(s["utt_dur"]),
            "gap_prev": float(s.get("gap_prev") or 0.0),
            "n_prior_ops": float(s.get("n_prior_ops") or 0),
            "slots_missing": float(s["slots_missing"]),
            "chain_dep": float(s.get("chain_dep") or 0)}
    for k in KAPPAS:
        vals["k_" + k] = 1.0 if s["kappa"] == k else 0.0
    for x in FINALITY:
        vals["f_" + x] = 1.0 if f == x else 0.0
    for x in DOMAINS:
        vals["d_" + x] = 1.0 if d == x else 0.0
    return [vals[k] for k in feats]


class StopHead:
    """Hazard model stored as plain JSON (no runtime sklearn/torch).
    arch absent = logistic regression (v0/v1/v2-lr); arch == "mlp" = one
    tanh hidden layer (v2 ablation; W1 stored as hidden x dim rows)."""

    def __init__(self, d):
        self.d = d
        self.feats = list(d.get("feats", FEATS))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def featurize(self, s, t):
        return featurize(s, t, self.feats)

    def _logit(self, row):
        xn = [(x - m) / (sd if sd > 1e-9 else 1.0)
              for x, m, sd in zip(row, self.d["mean"], self.d["std"])]
        if self.d.get("arch") == "mlp":
            h = [math.tanh(sum(w * x for w, x in zip(wr, xn)) + br)
                 for wr, br in zip(self.d["W1"], self.d["b1"])]
            return sum(w * x for w, x in zip(self.d["W2"], h)) + self.d["b2"]
        return self.d["b"] + sum(w * x for w, x in zip(self.d["w"], xn))

    def hazard(self, row):
        z = max(-30.0, min(30.0, self._logit(row)))
        return 1.0 / (1.0 + math.exp(-z))

    def risk(self, s):
        """P(a revision arrives within RISK_HORIZON of silence) — the two-stage
        score: 1 - prod over grid steps below the horizon of (1 - hazard)."""
        horizon = self.d.get("risk_horizon", RISK_HORIZON)
        surv = 1.0
        for t in self.d.get("t_grid", T_GRID):
            if t >= horizon:
                break
            surv *= 1.0 - self.hazard(self.featurize(s, t))
        return 1.0 - surv

    def window(self, s, c_w=None):
        """v2 twostage: W_PROTECT if risk >= theta else 0 (c_w arg doubles as a
        theta override for the trainer sweep). v0/v1: P2 threshold rule."""
        if self.d.get("policy") == "twostage":
            th = self.d.get("theta") if c_w is None else c_w
            wp = float(self.d.get("w_protect", W_PROTECT))
            return wp if self.risk(s) >= th else 0.0
        cw = self.d["c_w"] if c_w is None else c_w
        ck = C_KAPPA[s["kappa"]]
        cap = self.d.get("w_cap", W_CAP)
        for t in self.d.get("t_grid", T_GRID):
            if self.hazard(self.featurize(s, t)) * ck <= cw:
                return min(float(t), cap)
        return cap


def make_learned_delta_fn(model, ctx):
    """Runtime per-EoU factory. ctx = {eou_idx, utt_dur, gap_prev, n_prior_ops,
    finality, domain}; returns delta_fn(fn, args)->window for apply_decision_ops.
    Dispatches on the model's own policy field (v0/v1 hazard-window or v2
    twostage) — the harness wiring is identical."""
    def fn(fname, args=None):
        s = dict(ctx)
        s["kappa"] = kappa_name(fname)
        s["slots_missing"] = slots_missing_from_args(fname, args)
        s["chain_dep"] = chain_dep_from_args(args)
        return model.window(s)
    return fn
