"""
src/tools_registry.py
=====================
Tool executor registry for Phase-B.

Wraps FDB-v3 mock APIs with telemetry and reversibility annotations.
Compatible with both local mock and official FDB-v3 mock_apis.py.
"""

import json
import time
from typing import Callable, Optional

from transaction import Reversibility


# ---------------------------------------------------------------------------
# Reversibility map (FDB-v3 tools)
# ---------------------------------------------------------------------------
REVERSIBILITY = {
    # Travel & Identity
    "search_flights":       Reversibility.READ,
    "book_flight":          Reversibility.COMP,
    "update_identity_doc":  Reversibility.IRR,
    # Finance & Billing
    "get_card_benefits":    Reversibility.READ,
    "get_exchange_rate":    Reversibility.READ,
    "modify_autopay":       Reversibility.COMP,
    # Housing & Location
    "search_apartments":    Reversibility.READ,
    "calculate_commute":    Reversibility.READ,
    "update_search_filter": Reversibility.REV,
    # E-Commerce
    "track_order":          Reversibility.READ,
    "search_products":      Reversibility.READ,
    "add_to_cart":          Reversibility.REV,
}

COMPENSATORS = {
    "book_flight": "cancel_booking",
    "modify_autopay": "revert_autopay",
}


# ---------------------------------------------------------------------------
# Mock tool implementations (inline fallback)
# ---------------------------------------------------------------------------
def _search_flights(destination, date, **k):
    return {"status": "success", "flights": [{"flight_id": "FL123", "destination": destination, "date": date, "price": 450.0}]}

def _book_flight(passenger_name, flight_id="FL123", **k):
    return {"status": "success", "booking_ref": "B789", "passenger": passenger_name, "flight_id": flight_id}

def _update_identity_doc(doc_type, doc_number, **k):
    return {"status": "success", "updated_doc": doc_type, "masked_number": str(doc_number)[-4:]}

def _get_card_benefits(card_type, **k):
    return {"status": "success", "card_type": card_type, "benefits": ["2% Cashback", "No Foreign Transaction Fee"]}

def _get_exchange_rate(amount, from_currency, to_currency, **k):
    r = 1.1 if from_currency == "EUR" else 0.9
    return {"status": "success", "converted_amount": float(amount) * r, "rate": r, "from": from_currency, "to": to_currency}

def _modify_autopay(bill_type, source_account, **k):
    return {"status": "success", "autopay_enabled": True, "bill": bill_type, "source": source_account}

def _search_apartments(city, bedrooms, max_price, **k):
    return {"status": "success", "city": city, "results": [{"id": "APT1", "price": max_price - 100, "beds": bedrooms}]}

def _calculate_commute(origin_address, destination_address, mode="driving", **k):
    return {"status": "success", "duration_mins": 25, "mode": mode, "origin": origin_address, "destination": destination_address}

def _update_search_filter(filter_name, value, **k):
    return {"status": "success", "filter_updated": filter_name, "new_value": value}

def _track_order(order_id, **k):
    return {"status": "success", "order_id": order_id, "shipping_status": "Out for delivery"}

def _search_products(query, max_price=None, **k):
    p = max_price - 10 if max_price else 99.99
    return {"status": "success", "products": [{"product_id": "PROD1", "name": f"{query} Premium", "price": p}]}

def _add_to_cart(product_id, quantity=1, **k):
    return {"status": "success", "product_id": product_id, "quantity": quantity, "cart_total": 99.99 * (quantity or 1)}

def _cancel_booking(**k):
    return {"status": "success", "cancelled": True}

def _revert_autopay(**k):
    return {"status": "success", "autopay_disabled": True}


MOCK_TOOLS = {
    "search_flights": _search_flights,
    "book_flight": _book_flight,
    "update_identity_doc": _update_identity_doc,
    "get_card_benefits": _get_card_benefits,
    "get_exchange_rate": _get_exchange_rate,
    "modify_autopay": _modify_autopay,
    "search_apartments": _search_apartments,
    "calculate_commute": _calculate_commute,
    "update_search_filter": _update_search_filter,
    "track_order": _track_order,
    "search_products": _search_products,
    "add_to_cart": _add_to_cart,
    "cancel_booking": _cancel_booking,
    "revert_autopay": _revert_autopay,
}


# ---------------------------------------------------------------------------
# Tool registry with telemetry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """
    Tool executor with telemetry and reversibility annotations.

    Usage:
        registry = ToolRegistry(latency_profile="instant")
        result = registry.executor("search_flights", {"destination": "NYC", "date": "July 15"})
    """

    def __init__(self, latency_profile: str = "instant", telemetry_path: Optional[str] = None,
                 use_official: bool = False):
        """
        latency_profile: "instant" | "realistic" (adds simulated delay)
        telemetry_path: path to write tool call log (FDB-v3 format)
        use_official: try to import FDB-v3 mock_apis.py (requires v3/ on sys.path)
        """
        self.latency_profile = latency_profile
        self.telemetry_path = telemetry_path
        self.tool_calls = []

        # Try to load official FDB-v3 mock APIs
        if use_official:
            try:
                import sys
                from pathlib import Path
                fdb_v3 = Path("/root/autodl-tmp/FDBench_v3/v3")
                if fdb_v3.exists() and str(fdb_v3) not in sys.path:
                    sys.path.insert(0, str(fdb_v3))
                from mock_apis import MockAPIRegistry
                self._backend = MockAPIRegistry(latency_profile=latency_profile)
                self._call = self._backend.call_api
            except Exception:
                self._call = self._mock_call
        else:
            self._call = self._mock_call

    def _mock_call(self, fn: str, **args) -> dict:
        """Call mock tool implementation."""
        if fn not in MOCK_TOOLS:
            return {"status": "error", "message": f"Unknown tool: {fn}"}
        return MOCK_TOOLS[fn](**args)

    def executor(self, fn: str, args: dict) -> dict:
        """
        Execute a tool call with telemetry.

        This is the callable passed to Transaction.commit() / speculate().
        """
        t0 = time.time()
        result = self._call(fn, **args)
        t1 = time.time()

        call_record = {
            "function": fn,
            "args": args,
            "result": result,
            "timestamp_start": t0,
            "timestamp_end": t1,
        }
        self.tool_calls.append(call_record)

        if self.telemetry_path:
            self._log_telemetry(call_record)

        return result

    def _log_telemetry(self, record: dict):
        """Append to telemetry log (FDB-v3 format)."""
        try:
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "call": {
                        "function": record["function"],
                        "args": record["args"],
                        "timestamp_start": record["timestamp_start"],
                        "timestamp_end": record["timestamp_end"],
                    }
                }) + "\n")
        except Exception:
            pass

    def get_reversibility(self, fn: str) -> Reversibility:
        """Get reversibility annotation for a tool."""
        return REVERSIBILITY.get(fn, Reversibility.IRR)

    def export_tool_calls_fdb(self) -> list:
        """Export tool calls in FDB-v3 actual_tool_calls format."""
        return [
            {
                "function": rec["function"],
                "args": rec["args"],
                "timestamp_start": rec["timestamp_start"],
                "timestamp_end": rec["timestamp_end"],
            }
            for rec in self.tool_calls
        ]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    registry = ToolRegistry(latency_profile="instant")

    result = registry.executor("search_flights", {"destination": "Tokyo", "date": "July 15"})
    print("search_flights result:", result)

    result = registry.executor("book_flight", {"passenger_name": "Alice Chen", "flight_id": "FL123"})
    print("book_flight result:", result)

    print("\nReversibility annotations:")
    for fn in ["search_flights", "book_flight", "update_identity_doc"]:
        print(f"  {fn}: {registry.get_reversibility(fn).name}")

    print("\nExported tool calls (FDB-v3 format):")
    print(json.dumps(registry.export_tool_calls_fdb(), indent=2))
