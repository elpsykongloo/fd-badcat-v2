"""
tact/tools.py
=============
The 12 FDB-v3 tools, the reversibility classification (blueprint §2.3 / §5.3),
and a tool-call telemetry writer compatible with FDB-v3's evaluator.

We mirror Full-Duplex-Bench/v3/mock_apis.py EXACTLY (same function names, same
argument names) so that the `args` we emit line up with each scenario's
`expected_tool_calls` for Pass@1 / F1 scoring.

Two ways to obtain the tool backend:
  (A) Reuse the official mock_apis.py (import it). Recommended — guarantees parity.
  (B) Fall back to the inline definitions below (kept identical) if the import path
      is unavailable.

`REVERSIBILITY` is our contribution: it annotates each tool with READ / REV / COMP / IRR.
This is the reversibility-annotated tool schema we release (§5.3). For the *blocking
baseline* it is unused; for async speculation (Week-2+) it gates what may be pre-executed.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from .paths import add_to_syspath, fdb_v3_dir
from .transaction import Reversibility   # use `from transaction import ...` if running flat


# ---------------------------------------------------------------------------
# Reversibility-annotated tool schema  (§5.3)
# ---------------------------------------------------------------------------
# READ : pure lookups (always safe to pre-execute)
# REV  : cheap exact inverse (e.g. a filter you can reset, a cart line you can remove)
# COMP : real mutation, undoable via a compensating call (booking -> cancel, autopay -> revert)
# IRR  : no inverse (we keep update_identity_doc IRR: identity writes shouldn't be speculated)
REVERSIBILITY = {
    # ── Travel & Identity ──
    "search_flights":       Reversibility.READ,
    "book_flight":          Reversibility.COMP,   # compensator: cancel_booking
    "update_identity_doc":  Reversibility.IRR,
    # ── Finance & Billing ──
    "get_card_benefits":    Reversibility.READ,
    "get_exchange_rate":    Reversibility.READ,
    "modify_autopay":       Reversibility.COMP,   # compensator: revert_autopay
    # ── Housing & Location ──
    "search_apartments":    Reversibility.READ,
    "calculate_commute":    Reversibility.READ,
    "update_search_filter": Reversibility.REV,    # inverse: set previous value
    # ── E-Commerce ──
    "track_order":          Reversibility.READ,
    "search_products":      Reversibility.READ,
    "add_to_cart":          Reversibility.REV,    # inverse: remove_from_cart
}

# Optional compensators (only needed once you turn on speculation / post-commit rollback)
COMPENSATORS = {
    "book_flight": "cancel_booking",
    "modify_autopay": "revert_autopay",
}


# ---------------------------------------------------------------------------
# Tool backend
# ---------------------------------------------------------------------------
def _load_official_registry(latency_profile: str = "instant"):
    """Try to import the official FDB-v3 mock_apis.MockAPIRegistry."""
    add_to_syspath(fdb_v3_dir())
    try:
        from mock_apis import MockAPIRegistry  # requires v3/ on sys.path
        return MockAPIRegistry(latency_profile=latency_profile)
    except Exception:
        return None


# Inline fallback — identical signatures to v3/mock_apis.py
def _search_flights(destination, date, **k):  return {"status": "success", "flights": [{"flight_id": "FL123", "destination": destination, "date": date, "price": 450.0}]}
def _book_flight(passenger_name, flight_id="FL123", **k): return {"status": "success", "booking_ref": "B789", "passenger": passenger_name}
def _update_identity_doc(doc_type, doc_number, **k): return {"status": "success", "updated_doc": doc_type, "masked_number": str(doc_number)[-4:]}
def _get_card_benefits(card_type, **k): return {"status": "success", "card_type": card_type, "benefits": ["2% Cashback", "No Foreign Transaction Fee"]}
def _get_exchange_rate(amount, from_currency, to_currency, **k): r = 1.1 if from_currency == "EUR" else 0.9; return {"status": "success", "converted_amount": float(amount) * r, "rate": r}
def _modify_autopay(bill_type, source_account, **k): return {"status": "success", "autopay_enabled": True, "bill": bill_type, "source": source_account}
def _search_apartments(city, bedrooms, max_price, **k): return {"status": "success", "city": city, "results": [{"id": "APT1", "price": max_price - 100, "beds": bedrooms}]}
def _calculate_commute(origin_address, destination_address, mode="driving", **k): return {"status": "success", "duration_mins": 25, "mode": mode}
def _update_search_filter(filter_name, value, **k): return {"status": "success", "filter_updated": filter_name, "new_value": value}
def _track_order(order_id, **k): return {"status": "success", "order_id": order_id, "shipping_status": "Out for delivery"}
def _search_products(query, max_price=None, **k): p = max_price - 10 if max_price else 99.99; return {"status": "success", "products": [{"product_id": "PROD1", "name": f"{query} Premium", "price": p}]}
def _add_to_cart(product_id, quantity=1, **k): return {"status": "success", "product_id": product_id, "quantity": quantity, "cart_total": 99.99 * (quantity or 1)}

_INLINE = {
    "search_flights": _search_flights, "book_flight": _book_flight, "update_identity_doc": _update_identity_doc,
    "get_card_benefits": _get_card_benefits, "get_exchange_rate": _get_exchange_rate, "modify_autopay": _modify_autopay,
    "search_apartments": _search_apartments, "calculate_commute": _calculate_commute, "update_search_filter": _update_search_filter,
    "track_order": _track_order, "search_products": _search_products, "add_to_cart": _add_to_cart,
    # trivial compensators for the COMP/REV tools (no-op success in the mock world)
    "cancel_booking": lambda **k: {"status": "success", "cancelled": True},
    "revert_autopay": lambda **k: {"status": "success", "reverted": True},
    "remove_from_cart": lambda **k: {"status": "success", "removed": True},
}


# ---------------------------------------------------------------------------
# ToolRegistry — what the engine actually calls
# ---------------------------------------------------------------------------
class ToolRegistry:
    """A thin wrapper that (a) executes a named tool with kwargs, (b) injects the
    scenario's API latency (so the 'occupied silence' is real), and (c) appends a
    telemetry line in FDB-v3 format so the official pipeline can pick it up."""

    def __init__(self, latency_profile: str = "instant", room: str = "offline",
                 telemetry_path: str = "/tmp/agent_tool_calls.log"):
        self.room = room
        self.telemetry_path = telemetry_path
        self._official = _load_official_registry(latency_profile)
        # latency injector (reuse official if present, else import the module directly)
        if self._official is None:
            try:
                from latency_injector import LatencyInjector
                self._lat = LatencyInjector(profile=latency_profile)
            except Exception:
                self._lat = None

    def call(self, fn: str, **kwargs) -> dict:
        """Execute a tool. Latency is injected to emulate real backends."""
        if self._official is not None:
            return self._official.call(fn, **kwargs)
        if self._lat is not None:
            self._lat.inject(fn)
        func = _INLINE.get(fn)
        if func is None:
            return {"status": "error", "message": f"unknown tool {fn}"}
        return func(**kwargs)

    # --- the executor signature the Transaction expects: executor(fn, args) -> dict ---
    def executor(self, fn: str, args: dict) -> dict:
        t0 = time.time()
        res = self.call(fn, **args)
        t1 = time.time()
        self._log(fn, args, t0, t1)
        return res

    def _log(self, fn: str, args: dict, t0: float, t1: float):
        """Append in the exact format process_single() expects in /tmp/agent_tool_calls.log."""
        try:
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "room": self.room,
                    "call": {"function": fn, "args": args,
                             "timestamp_start": t0, "timestamp_end": t1},
                }) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# A compact tool catalog string for the decider prompt
# ---------------------------------------------------------------------------
TOOL_CATALOG = """\
Travel & Identity:
  search_flights(destination, date)
  book_flight(passenger_name)
  update_identity_doc(doc_type, doc_number)
Finance & Billing:
  get_card_benefits(card_type)
  get_exchange_rate(amount, from_currency, to_currency)
  modify_autopay(bill_type, source_account)
Housing & Location:
  search_apartments(city, bedrooms, max_price, pets_allowed)
  calculate_commute(origin_address, destination_address, mode)
  update_search_filter(filter_name, value)
E-Commerce:
  track_order(order_id)
  search_products(query, max_price, category)
  add_to_cart(product_id, quantity)"""


if __name__ == "__main__":
    reg = ToolRegistry(latency_profile="instant", room="selftest")
    print(reg.executor("search_flights", {"destination": "Tokyo", "date": "July 15"}))
    print("reversibility(book_flight) =", REVERSIBILITY["book_flight"].name)
    print("catalog:\n", TOOL_CATALOG[:120], "...")
