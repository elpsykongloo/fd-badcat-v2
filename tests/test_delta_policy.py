"""W4 adaptive-ladder unit tests — pure CPU, no model, no network.

Covers: preregistered table shape invariants, spec parsing, per-op delta
resolution, finality parsing, WindowLedger per-op override (incl. frozen
default), and apply_decision_ops delta_fn wiring (launch + patch restart).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import delta_policy as dp                                   # noqa: E402
from tact_core import WindowLedger, apply_decision_ops      # noqa: E402
from tact.transaction import Transaction                    # noqa: E402

ORDER = ("READ", "REV", "COMP", "IRR")


# -- preregistered table invariants ------------------------------------------
def test_kappa_tables_monotone():
    for name in ("v0", "safe"):
        t = dp.KAPPA_TABLES[name]
        vals = [t[k] for k in ORDER]
        assert vals == sorted(vals), name
    rev = [dp.KAPPA_TABLES["rev"][k] for k in ORDER]
    assert rev == sorted(rev, reverse=True)


def test_finality_table_monotone_both_axes():
    for lbl in dp.FINALITY_LABELS:
        vals = [dp.FINALITY_TABLE[lbl][k] for k in ORDER]
        assert vals == sorted(vals), lbl
    for k in ORDER:
        col = [dp.FINALITY_TABLE[lbl][k] for lbl in dp.FINALITY_LABELS]
        assert col == sorted(col), k


# -- spec parsing / delta resolution -----------------------------------------
def test_parse_spec():
    assert dp.parse_spec("fixed") == ("fixed", None)
    assert dp.parse_spec("kappa:v0") == ("kappa", "v0")
    assert dp.parse_spec("prompted:v0") == ("prompted", "v0")
    with pytest.raises(ValueError):
        dp.parse_spec("kappa:nope")


def test_make_delta_fn_kappa():
    assert dp.make_delta_fn("fixed") is None
    fn = dp.make_delta_fn("kappa:v0")
    assert fn("search_flights") == 0.64        # READ
    assert fn("add_to_cart") == 1.0            # REV
    assert fn("book_flight") == 1.5            # COMP
    assert fn("update_identity_doc") == 2.0    # IRR
    assert fn("no_such_tool") == 2.0           # unmapped -> IRR bucket


def test_make_delta_fn_prompted_rows():
    assert dp.make_delta_fn("prompted:v0", "final")("search_flights") == 0.0
    assert dp.make_delta_fn("prompted:v0", "unfinished")("search_flights") == 2.0
    # unknown/None finality falls back to the neutral hesitant row
    assert (dp.make_delta_fn("prompted:v0", None)("book_flight")
            == dp.FINALITY_TABLE["hesitant"]["COMP"])


def test_parse_finality():
    assert dp.parse_finality("final") == ("final", True)
    assert dp.parse_finality("  Final.") == ("final", True)
    assert dp.parse_finality("I think: unfinished") == ("unfinished", True)
    assert dp.parse_finality("HESITANT\n") == ("hesitant", True)
    assert dp.parse_finality("no idea") == ("hesitant", False)
    assert dp.parse_finality("") == ("hesitant", False)


# -- WindowLedger per-op override ---------------------------------------------
def test_ledger_open_restart_override_and_frozen_default():
    led = WindowLedger(1.5, barrier=True)
    led.open("a")                 # frozen default
    led.open("b", delta=0.2)      # override
    assert led.remaining("a") == 1.5 and led.remaining("b") == 0.2
    commits = []
    led.advance_silence(0.0, 1.0, lambda oid, tn, ta: commits.append((oid, tn)))
    assert commits == [("b", 0.2)]            # deadline order, override honored
    assert abs(led.remaining("a") - 0.5) < 1e-9
    led.restart("a", delta=0.3)
    assert led.remaining("a") == 0.3
    led.restart("a")                          # default restores ledger budget
    assert led.remaining("a") == 1.5


# -- apply_decision_ops wiring -------------------------------------------------
def test_apply_decision_ops_delta_fn_launch_and_patch():
    tx = Transaction()
    led = WindowLedger(1.5, barrier=True)
    fn = dp.make_delta_fn("kappa:v0")
    dec = {"ops": [{"type": "launch", "fn": "search_flights",
                    "args": {"destination": "Milan"}}]}
    applied = apply_decision_ops(tx, led, dec, 1.0, immediate=False,
                                 commit_cb=lambda *a: None, delta_fn=fn)
    (launch,) = [a for a in applied if a["type"] == "launch"]
    oid = launch["op_id"]
    assert led.remaining(oid) == 0.64                     # READ window
    dec2 = {"ops": [{"type": "patch", "op_id": oid,
                     "diff": {"date": "June 3"}}]}
    tx._localmap = {oid: oid}
    apply_decision_ops(tx, led, dec2, 2.0, immediate=False,
                       commit_cb=lambda *a: None, delta_fn=fn)
    assert led.remaining(oid) == 0.64                     # restart uses policy
    # frozen path: no delta_fn -> ledger-wide budget
    dec3 = {"ops": [{"type": "launch", "fn": "book_flight",
                     "args": {"passenger_name": "A"}}]}
    applied3 = apply_decision_ops(tx, led, dec3, 3.0, immediate=False,
                                  commit_cb=lambda *a: None)
    (l3,) = [a for a in applied3 if a["type"] == "launch"]
    assert led.remaining(l3["op_id"]) == 1.5
