# -*- coding: utf-8 -*-
"""W5 batch tests — specgate / floor commitment tiers / RB machinery.
Pure CPU, stdlib only. Run: python -m pytest tests/test_w5.py -q"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


# ---- W5-SG ------------------------------------------------------------------

def test_sg_tracker_and_labels():
    from specgate import SGTracker, events_to_rows, FEATS_SG
    tr = SGTracker()
    tr.on_start(0.0); tr.on_end(2.0)
    assert len(tr.features()) == len(FEATS_SG)
    rows = events_to_rows([(0.0, 2.0), (2.3, 3.0), (4.5, 6.0)])
    assert [r[1] for r in rows] == [0, 1, 1]


def test_sg_gate_threshold():
    from specgate import SpecGate
    g = SpecGate({"feats": ["utt_dur", "gap1", "gap2", "gap3", "n_segs_10s",
                            "speech_ratio_5s"], "mean": [0] * 6, "std": [1] * 6,
                  "w": [1, 0, 0, 0, 0, 0], "b": 0.0, "theta": 0.5})
    assert g.allow([1.0, 0, 0, 0, 0, 0]) and not g.allow([-1.0, 0, 0, 0, 0, 0])


# ---- W5-FC ------------------------------------------------------------------

def test_fc_tier_rule():
    from floor_policy import commit_tier
    assert commit_tier(eta_s=0.5) == "silence"
    assert commit_tier(eta_s=3.0) == "filler"
    assert commit_tier(eta_s=20.0) == "progress"
    assert commit_tier(eta_s=3.0, elapsed_s=10.0) == "progress"     # overdue
    assert commit_tier(result_known=True, result_conf=0.95) == "commit"
    assert commit_tier(result_known=True, result_conf=0.5) == "hedge"
    assert commit_tier(eta_s=None, elapsed_s=0.0) == "silence"
    assert commit_tier(eta_s=None, elapsed_s=2.0) == "filler"


def test_fc_templates_carry_markers():
    from floor_policy import tier_utterance
    assert tier_utterance("silence") == ""
    assert "已确认" in tier_utterance("commit", "zh", claim="改到杭州")
    assert "Confirmed" in tier_utterance("commit", "en", claim="Denver")
    assert "再确认" in tier_utterance("hedge", "zh", claim="有票")
    assert tier_utterance("progress", "zh", fns=["reserve_hotel"], eta_s=12)


def test_fc_v0_unchanged():
    from floor_policy import decide
    assert decide("narration") == "yield"
    assert decide("confirmation", window_remaining_s=0.5) == "finish_clause"
    assert decide("confirmation", window_remaining_s=5.0) == "yield"


# ---- RB ---------------------------------------------------------------------

def test_rb_generator_deterministic():
    from rb.generator import build_all
    qa = {"L4": 2, "L10": 3}
    qb = {"L8": 2}
    _, e1 = build_all(quota_a=qa, quota_b=qb)
    _, e2 = build_all(quota_a=qa, quota_b=qb)
    assert json.dumps(e1, sort_keys=True) == json.dumps(e2, sort_keys=True)


def test_rb_gold_and_layers():
    from rb.generator import make_episode, config_hash
    ch = config_hash()
    e = make_episode("A", "L5", 3, ch)
    assert e["revisions"] and 1.0 <= e["revisions"][0]["gap"] <= 4.0
    assert e["revisions"][0]["new"] in json.dumps(e["gold_calls"], ensure_ascii=False)
    assert e["revisions"][0]["old"] not in [
        v for v in e["slots_final"].values()] or \
        e["revisions"][0]["old"] != e["slots_final"][e["revisions"][0]["slot"]]


def test_rb_sandbox_semantics():
    from rb.sandbox import Sandbox
    sb = Sandbox("ep1")
    r = sb.execute("transfer_funds", {"from_acct": "checking",
                                      "to_acct": "savings", "amount": "500"})
    assert r["status"] == "success"
    bad = sb.execute("transfer_funds", {"from_acct": "checking"})
    assert bad["status"] == "error"
    rid = r["result"]["id"]
    assert sb.compensate("transfer_funds", rid)["status"] == "success"
    assert sb.fees == 1 and sb.live_state() == {}
    assert sb.compensate("transfer_funds", rid)["status"] == "error"


def test_rb_scorer_tracks():
    from rb.scorer import (score_exact, score_state, commitment_repair,
                           utility, score_episode)
    gold = [{"fn": "hold_seat", "args": {"train_id": "T1", "seat_class": "一等座"}}]
    assert score_exact(gold, list(reversed(gold)))
    assert not score_exact([], gold)
    st = {"hold_seat#X1": {"train_id": "T1", "seat_class": "一等座"}}
    gd = {"hold_seat#Y9": {"train_id": "T1", "seat_class": "一等座"}}
    assert score_state(st, gd)                       # id-free comparison by fn
    says = [(1.0, "已确认：去成都。"), (2.0, "抱歉，刚才说错了：去杭州。")]
    cr = commitment_repair(says, "zh", ["杭州"], ["成都"])
    assert cr["wrong_commits"] == 1 and cr["repaired"] == 1 and cr["unrepaired"] == 0
    cr2 = commitment_repair([(1.0, "已确认：去成都。")], "zh", ["杭州"], ["成都"])
    assert cr2["unrepaired"] == 1
    assert utility(True, 0.0, 0, 0.0) == 1.0
    assert utility(True, 2.0, 0, 0.0) == round(0.95 ** 2, 4)
    assert utility(True, 0.0, 1, 4.0) == round(1.0 - 0.25 - 0.05 * 4.0, 4)


def test_rb_simulator_lifecycle():
    from rb.simulator import ReactiveUser
    ep = {"id": "B_L8_0001", "lang": "en", "step_latencies": [10.0],
          "events": [{"state": "inflight", "offset": 0.3, "action": "revise",
                      "role": "user", "voice": "cv02", "text": "Wait — make it Denver."}]}
    u = ReactiveUser(ep)
    acts = u.on_event({"event": "tact_op_applied", "t": 4.0,
                       "data": {"t_audio": 4.0,
                                "op": {"type": "launch", "fn": "reserve_hotel"}}})
    assert acts and abs(acts[0]["at"] - 7.0) < 1e-9


def test_rb_audio_assembly_stub():
    import tempfile
    from rb.audio import SilenceStub, assemble_episode, measured_gaps
    ep = {"pieces": [
        {"role": "user", "voice": "cv01", "lang": "zh", "text": "帮我订酒店。",
         "gap_before": 0.5},
        {"role": "user", "voice": "cv01", "lang": "zh", "text": "等等，改成杭州。",
         "gap_before": 1.2},
        {"role": "bystander", "voice": "cv02", "lang": "zh", "text": "改成上海吧。",
         "at_after_eou": 1.0}]}
    out = Path(tempfile.mkdtemp()) / "a.wav"
    cues = assemble_episode(ep, SilenceStub(), out)
    assert out.exists() and len(cues) == 3
    gaps = measured_gaps(cues)
    assert len(gaps) == 1 and abs(gaps[0] - 1.2) < 0.01
    assert cues[2]["t_start"] > cues[1]["t_end"]     # placed after seq end + hold
