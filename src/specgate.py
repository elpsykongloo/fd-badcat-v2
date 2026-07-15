# -*- coding: utf-8 -*-
"""src/specgate.py — W5-SG speculative-dispatch gate (docs/w5_specgate_design.md).

At every VAD-end the engine may speculatively dispatch the EoU decision
(engine_b `speculative_dispatch`). 67.4% of live dispatches are voided by
speech resuming inside the 0.64s hold (W3 D6: 604/197/407). The gate predicts

    y = 1[t_next_speech_start >= t_vad_end + HOLD_S]   (dispatch will confirm)

and skips the dispatch when P(y=1) < theta. Gating only skips PRECOMPUTATION:
a gated-out EoU that confirms takes the ordinary non-speculative path with the
same snapshot, so decision CONTENT is bitwise unchanged by construction — the
gate trades first-response latency (missed confirms lose the 0.64 floor)
against wasted decision-LLM calls.

Feature parity is the load-bearing contract of this module: `SGTracker` is the
ONE featurizer, fed with (speech_start, speech_end) times by all three
consumers — the HumDial census (training labels), the FDB replay accounting,
and the live engine hook. All features are stream-position-invariant (rolling
windows, no turn/session anchors) so per-sample training streams and the long
live stream featurize identically. F_time only; F_text/F_pros enter solely via
the preregistered probe/availability gates (design §4).

Model JSON follows the stophead convention (mean/std standardization; LR
`w`/`b` or MLP `W1/b1/W2/b2`) plus `theta` and a `specgate` provenance block.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HOLD_S = 0.64                 # frozen EoU hold (engine constant)
GAP_CAP_S = 10.0              # gap features capped here (log-tail irrelevant)
RATIO_WIN_S = 5.0             # speech-ratio rolling window
SEGS_WIN_S = 10.0             # segment-count rolling window
FEATS_SG = ("utt_dur", "gap1", "gap2", "gap3", "n_segs_10s", "speech_ratio_5s")


class SGTracker:
    """Rolling VAD-segment tracker. Feed monotone times via on_start/on_end;
    features() describes the segment that just ended. Pure; no engine deps."""

    def __init__(self):
        self.segs = []            # closed segments [(t_start, t_end), ...]
        self._open = None

    def reset(self):
        self.segs = []
        self._open = None

    def on_start(self, t):
        self._open = float(t)

    def on_end(self, t):
        t = float(t)
        s = self._open if self._open is not None else t
        self._open = None
        self.segs.append((s, t))
        if len(self.segs) > 64:                      # bound memory; windows are short
            self.segs = self.segs[-64:]

    def features(self):
        """Feature row (list, FEATS_SG order) for the most recent on_end."""
        if not self.segs:
            return [0.0] * len(FEATS_SG)
        t = self.segs[-1][1]
        utt_dur = self.segs[-1][1] - self.segs[-1][0]
        gaps = []
        for (s0, e0), (s1, _e1) in zip(self.segs[:-1], self.segs[1:]):
            gaps.append(min(max(s1 - e0, 0.0), GAP_CAP_S))
        g = list(reversed(gaps[-3:])) + [0.0, 0.0, 0.0]
        n10 = sum(1 for (_s, e) in self.segs if t - SEGS_WIN_S < e <= t)
        lo = t - RATIO_WIN_S
        sp = sum(max(0.0, min(e, t) - max(s, lo)) for (s, e) in self.segs)
        return [round(utt_dur, 4), round(g[0], 4), round(g[1], 4), round(g[2], 4),
                float(n10), round(min(sp / RATIO_WIN_S, 1.0), 4)]


def events_to_rows(seg_list, tail_confirms=True):
    """(t_start, t_end) list for ONE stream -> [(features, y, gap_next)].
    y = 1 iff the silence after this vad-end reaches HOLD_S (dispatch confirms).
    The stream-final vad-end has no next start: y = tail_confirms (a census
    sample ends in silence => the hold would expire => confirm)."""
    tr = SGTracker()
    rows = []
    for i, (s, e) in enumerate(seg_list):
        tr.on_start(s)
        tr.on_end(e)
        if i + 1 < len(seg_list):
            gap = max(seg_list[i + 1][0] - e, 0.0)
            y = 1 if gap >= HOLD_S else 0
        else:
            gap, y = None, (1 if tail_confirms else 0)
        rows.append((tr.features(), y, None if gap is None else round(gap, 4)))
    return rows


class SpecGate:
    """Runtime gate: stophead-convention JSON, LR or one-hidden-layer MLP."""

    def __init__(self, d):
        self.d = d
        self.feats = list(d.get("feats", FEATS_SG))
        self.theta = float(d["theta"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def prob(self, row):
        xn = [(x - m) / (sd if sd > 1e-9 else 1.0)
              for x, m, sd in zip(row, self.d["mean"], self.d["std"])]
        if self.d.get("arch") == "mlp":
            h = [math.tanh(sum(w * x for w, x in zip(wr, xn)) + br)
                 for wr, br in zip(self.d["W1"], self.d["b1"])]
            z = sum(w * x for w, x in zip(self.d["W2"], h)) + self.d["b2"]
        else:
            z = self.d["b"] + sum(w * x for w, x in zip(self.d["w"], xn))
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def allow(self, row):
        return self.prob(row) >= self.theta


def selftest():
    ck = {}
    tr = SGTracker()
    tr.on_start(1.0); tr.on_end(3.0)                 # seg A 2.0s
    f1 = tr.features()
    ck["utt_dur"] = abs(f1[0] - 2.0) < 1e-9
    ck["gaps_zero_padded"] = f1[1:4] == [0.0, 0.0, 0.0]
    ck["n_segs"] = f1[4] == 1.0
    ck["ratio"] = abs(f1[5] - 2.0 / 5.0) < 1e-9      # 2s speech in (−2,3]
    tr.on_start(3.5); tr.on_end(4.0)                 # gap 0.5
    tr.on_start(5.2); tr.on_end(9.4)                 # gap 1.2; seg 4.2s
    f3 = tr.features()
    ck["gap_order_recent_first"] = f3[1] == 1.2 and f3[2] == 0.5
    ck["ratio_caps_at_window"] = f3[5] == round(min((9.4 - 5.2 + 0.0) / 5.0, 1.0), 4)
    ck["n_segs_10s"] = f3[4] == 3.0
    rows = events_to_rows([(1.0, 3.0), (3.5, 4.0), (5.2, 9.4)])
    ck["labels"] = [r[1] for r in rows] == [0, 1, 1]          # 0.5<hold, 1.2>=hold, tail
    ck["gap_next"] = rows[0][2] == 0.5 and rows[2][2] is None
    # position invariance: same relative stream shifted by +1000s
    rows2 = events_to_rows([(1001.0, 1003.0), (1003.5, 1004.0), (1005.2, 1009.4)])
    ck["shift_invariant"] = all(a[0] == b[0] for a, b in zip(rows, rows2))
    gate = SpecGate({"feats": list(FEATS_SG), "mean": [0] * 6, "std": [1] * 6,
                     "w": [0, 5.0, 0, 0, 0, 0], "b": -3.0, "theta": 0.5})
    ck["gate_directional"] = (gate.allow([0, 1.2, 0, 0, 0, 0])
                              and not gate.allow([0, 0.1, 0, 0, 0, 0]))
    for k, v in ck.items():
        print(f"  selftest {k}: {'PASS' if v else 'FAIL'}")
    print("SELFTEST", "PASS" if all(ck.values()) else "FAIL")
    return 0 if all(ck.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
