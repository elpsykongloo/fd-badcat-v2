# -*- coding: utf-8 -*-
"""rb/simulator.py — RB v2 arm-B reactive user (docs/rb_design.md v2 §2).

The user simulator subscribes to the engine's trace/control event stream and
fires the episode's event bindings when their lifecycle anchor first occurs:

  anchor   fired on                            offset semantics
  eou      first confirmed EoU                 seconds after the anchor
  inflight first tool launch                   FRACTION of that tool's wall
  committed first commit                       seconds after the anchor
  tts      first agent audio onset             seconds after the anchor

All offsets were pre-sampled at build time (episode["events"]) — the
simulator adds NO randomness, so a run is bit-reproducible given the engine's
own determinism. Output actions are {"at": t_audio, "piece": {...}}; the
Phase-2 live driver turns them into injected audio (same synth pipeline as
arm A). This module is the grammar's executable semantics + its selftest;
websocket wiring is the Phase-2 integration item."""
from __future__ import annotations

EVENT_ANCHORS = {
    "eou": ("tact_eou", "eou_confirmed", "vad_hold_expired"),
    "inflight": ("tool_launched", "tact_op_applied_launch", "act_launched"),
    "committed": ("op_committed", "act_committed"),
    "tts": ("tts_start", "tts_sent_start", "agent_audio_start"),
}


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
            return ("committed", float(t), {})
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

    def on_event(self, ev):
        """Feed one engine event; returns newly-scheduled actions."""
        norm = normalize_event(ev)
        if norm is None:
            return []
        kind, t, extra = norm
        if kind in self.anchors:
            return []
        self.anchors[kind] = t
        out = []
        keep = []
        for e in self.pending:
            if e["state"] != kind:
                keep.append(e)
                continue
            if kind == "inflight":
                wall = (self.episode.get("step_latencies") or [1.0])[0]
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
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
