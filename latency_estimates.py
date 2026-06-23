"""
tact/latency_estimates.py
=========================
Per-tool latency priors (seconds), used by the async act track to estimate how
much longer an in-flight call will take -> drives floor-holding (silence vs
backchannel vs progress-narration) in Week 3 (blueprint §3.4).

These mirror the spirit of Full-Duplex-Bench/v3/latency_injector.py:
  - read-only lookups are fast
  - mutations (book/autopay) are slow
You will later REPLACE these point priors with an online estimator that conditions
on the live `latency_profile` and the call index, but a fixed prior is enough to
get floor-holding working and to ablate it.
"""

# midpoint-ish seconds per tool under the 'normal' API profile
_PRIOR = {
    "search_flights":       0.5,
    "book_flight":          2.0,   # slow
    "update_identity_doc":  0.2,
    "get_card_benefits":    0.2,
    "get_exchange_rate":    0.2,
    "modify_autopay":       2.0,   # slow
    "search_apartments":    0.5,
    "calculate_commute":    0.5,
    "update_search_filter": 0.2,
    "track_order":          0.2,
    "search_products":      0.5,
    "add_to_cart":          0.2,
}

_DEFAULT = 0.5


def estimate_seconds(fn: str) -> float:
    return _PRIOR.get(fn, _DEFAULT)
