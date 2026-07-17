# -*- coding: utf-8 -*-
"""rb/simulator.py — RB arm-B reactive user (docs/rb_design.md v2 §2;
v2.3 anchor revision).

The user simulator subscribes to the engine's trace/control event stream and
fires the episode's event bindings when their lifecycle anchor first occurs:

  anchor    fired on                              offset semantics
  eou       first confirmed EoU                   seconds after the anchor
  inflight  first COMP/IRR launch (v2.3; was:     FRACTION of THAT tool's wall
            first launch of any kappa — every
            chain starts with a READ, so nothing
            ever landed in a transactional
            execution window)
  committed first commit                          seconds after the anchor
  executing first non-READ COMMIT (v2.4, L15):    FRACTION of THAT tool's wall
            the event lands INSIDE the sandbox
            execution window [t, t+wall] —
            strictly post-window by construction
  tts       first agent audio onset               seconds after the anchor

All offsets were pre-sampled at build time (episode["events"]) — the
simulator adds NO randomness, so a run is bit-reproducible given the engine's
own determinism. Output actions are {"at": t_audio, "piece": {...}}; the
live driver turns them into injected audio (same synth pipeline as arm A).
This module is the grammar's executable semantics + its selftest."""
from __future__ import annotations

try:
    from .registry import TOOLS
except ImportError:                      # `python rb/simulator.py` selftest path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rb.registry import TOOLS

EVENT_ANCHORS = {
    "eou": ("tact_eou", "eou_confirmed", "vad_hold_expired"),
    "inflight": ("tool_launched", "tact_op_applied_launch", "act_launched"),
    "committed": ("op_committed", "act_committed"),
    "tts": ("tts_start", "tts_sent_start", "agent_audio_start"),
}
# v2.3: the anchor is the first TRANSACTIONAL launch (any non-READ kappa).
# The review asked for "first COMP/IRR"; REV is included because REV ops have
# real execution windows too and three domains' single-step scenarios are
# REV-terminal (schedule_payment/add_item/save_listing) — a COMP/IRR-only
# rule would orphan their in-flight events entirely. Deviation documented in
# rb_design §15.
INFLIGHT_KAPPAS = ("REV", "COMP", "IRR")


def normalize_event(ev):
    """Map an engine trace/control event dict to (anchor_kind, t, extra).
    Accepts both simplified test events {'event','t'} and engine shapes
    {'event','data':{'t_audio':...}}."""
    name = ev.get("event", "")
    t = ev.get("t")
    if t is None:
        t = (ev.get("data") or {}).get("t_audio")
    if t is None:
        return None
    if name in ("tact_op_applied",):
        op = (ev.get("data") or {}).get("op") or {}
        if op.get("type") == "launch":
            return ("inflight", float(t), {"fn": op.get("fn")})
        if op.get("type") == "commit":
            return ("committed", float(t), {"fn": op.get("fn")})
        return None
    for kind, names in EVENT_ANCHORS.items():
        if name in names:
            return (kind, float(t), (ev.get("data") or {}))
    return None


class ReactiveUser:
    def __init__(self, episode):
        self.episode = episode
        self.pending = list(episode.get("events", []))
        self.fired = []
        self.anchors = {}

    def _wall_of(self, fn):
        """Wall time of the anchoring tool = the episode's precomputed latency
        for ITS step (occurrence-keyed in v2.3, so step order is safe)."""
        scn_steps = self.episode.get("scenario_steps") or []
        lats = self.episode.get("step_latencies") or []
        for i, sfn in enumerate(scn_steps):
            if sfn == fn and i < len(lats):
                return lats[i]
        return lats[0] if lats else 1.0

    def on_event(self, ev):
        """Feed one engine event; returns newly-scheduled actions. A commit
        event may satisfy TWO anchors: `committed` (seconds offset) and — for
        non-READ tools — `executing` (v2.4: fraction of that tool's wall,
        landing inside the sandbox execution window)."""
        norm = normalize_event(ev)
        if norm is None:
            return []
        kind, t, extra = norm
        kinds = [kind]
        if kind == "committed":
            fn = (extra or {}).get("fn")
            if fn is None or TOOLS.get(fn, {}).get("kappa") != "READ":
                kinds.append("executing")
        out = []
        for k in kinds:
            if k == "inflight":
                fn = (extra or {}).get("fn")
                if fn is not None and \
                        TOOLS.get(fn, {}).get("kappa") not in INFLIGHT_KAPPAS:
                    continue                 # v2.3: anchor = first COMP/IRR
            if k in self.anchors:
                continue
            self.anchors[k] = t
            keep = []
            for e in self.pending:
                if e["state"] != k:
                    keep.append(e)
                    continue
                if k in ("inflight", "executing"):
                    wall = self._wall_of((extra or {}).get("fn"))
                    at = t + float(e["offset"]) * float(wall)
                else:
                    at = t + float(e["offset"])
                out.append({"at": round(at, 3),
                            "piece": {"role": e["role"], "voice": e["voice"],
                                      "lang": self.episode["lang"],
                                      "text": e["text"], "action": e["action"]}})
            self.pending = keep
        self.fired.extend(out)
        return out


def selftest():
    ep = {"id": "B_L8_0000", "lang": "zh", "step_latencies": [4.0],
          "events": [{"state": "inflight", "offset": 0.5, "action": "revise",
                      "role": "user", "voice": "cv01", "text": "等等，改成成都。"},
                     {"state": "eou", "offset": 0.3, "action": "progress_query",
                      "role": "user", "voice": "cv01", "text": "好了没？"}]}
    ck = {}
    u = ReactiveUser(ep)
    a1 = u.on_event({"event": "tact_eou", "t": 5.0})
    ck["eou_fires"] = len(a1) == 1 and a1[0]["at"] == 5.3
    a2 = u.on_event({"event": "tact_op_applied", "t": 6.2,
                     "data": {"t_audio": 6.2, "op": {"type": "launch", "fn": "reserve_hotel"}}})
    ck["inflight_fraction_of_wall"] = len(a2) == 1 and a2[0]["at"] == 6.2 + 0.5 * 4.0
    a3 = u.on_event({"event": "tact_eou", "t": 9.0})
    ck["anchor_fires_once"] = a3 == []
    u2 = ReactiveUser(ep)
    b1 = u2.on_event({"event": "tact_eou", "t": 5.0})
    ck["deterministic"] = b1 == a1
    ck["normalize_ignores_noise"] = normalize_event({"event": "vad_start", "t": 1}) is None
    # v2.4: a non-READ commit fires BOTH `committed` (seconds) and
    # `executing` (fraction of wall) anchors, each at most once.
    ep2 = {"id": "B_L15_0000", "lang": "zh",
           "scenario_steps": ["reserve_hotel"], "step_latencies": [10.0],
           "events": [{"state": "executing", "offset": 0.2, "action": "revise",
                       "role": "user", "voice": "cv01", "text": "等等，改成成都。"},
                      {"state": "committed", "offset": 0.5, "action": "revise",
                       "role": "user", "voice": "cv01", "text": "x"}]}
    u3 = ReactiveUser(ep2)
    c1 = u3.on_event({"event": "tact_op_applied", "t": 8.0,
                      "data": {"t_audio": 8.0,
                               "op": {"type": "commit", "fn": "reserve_hotel"}}})
    ck["executing_frac_of_wall"] = (
        len(c1) == 2 and
        sorted(a["at"] for a in c1) == [8.5, 8.0 + 0.2 * 10.0])
    c2 = u3.on_event({"event": "tact_op_applied", "t": 9.0,
                      "data": {"t_audio": 9.0,
                               "op": {"type": "commit", "fn": "reserve_hotel"}}})
    ck["executing_fires_once"] = c2 == []
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
