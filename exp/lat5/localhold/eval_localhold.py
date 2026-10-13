#!/usr/bin/env python
"""Offline calibration of the client-hold-v1 OnsetDetector on the lat5 real episodes (CPU).

Replays the bench's microphone feed of one candidate run (each item placed at its receipt
``start_T``, digital silence elsewhere, 256-sample frames, RMS of int16/32768 exactly as
``bench_guarded_latency.mic_loop``) through ``scripts/bench_guarded_latency.OnsetDetector``
(imported, not re-implemented) and scores a parameter set on

* events window: ``[first_packet_at, counterfactual play start)`` of every utterance the
  client received PCM for (the real client-hold-v1 opportunity; server speech_hold
  intervals are blocked, a local hold that happened in the run is undone by using
  ``initial_play_at``).  A trigger is ``match`` when a truth segment starts within
  +-100 ms, ``late`` when it falls inside a truth segment (user already speaking),
  otherwise ``misfire``.
* approx window: ``[stop + 0.3 s, stop + 1.6 s)`` after every truth segment end
  (run independent).
* detection lag: for every truth segment whose start is not echo-gated, first onset
  frame at or after ``start - 100 ms`` and before ``end``; ``missed`` when none,
  ``pre`` when the detector was already on before ``start - 100 ms``.
* the run's locally caught continuations (local hold with a truth start in
  ``(stop_T, hold + 50 ms]``, the ana.py rule) must still trigger.

usage:
  eval_localhold.py RUN --check                 # reproduce the run's logged local holds
  eval_localhold.py RUN --params '{"marginDb": 15}' [--detail]
  eval_localhold.py RUN --grid [--out DIR]      # grid scan + Pareto table
"""
import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_guarded_latency as bench  # noqa: E402
from bench_guarded_latency import FRAME, FRAME_S, LOCAL_HOLD_DEFAULTS, SR, OnsetDetector  # noqa: E402

OUTPUT_LATENCY_S = 27.0 / 1000  # bench --output-latency-ms default used by the cand runs
MATCH_S = 0.100
APPROX_WIN = (0.3, 1.6)


def jrows(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def pct(v, q):
    v = [x for x in v if x is not None]
    return round(float(np.percentile(v, q)), 1) if v else None


class Trial:
    def __init__(self, tdir, ends):
        r = json.loads((tdir / "receipt.json").read_text(encoding="utf-8"))
        self.name, self.items, self.utts = r["name"], r["items"], r.get("utterances", [])
        t_end = max([it["wait_done_T"] for it in self.items]
                    + [u.get(k) or 0 for u in self.utts
                       for k in ("cancelled_at", "ended_at", "play_start_at", "initial_play_at")]) + 1.0
        n = int(math.ceil(t_end / FRAME_S))
        pcm = np.zeros(n * FRAME, np.int16)
        for it in self.items:
            x = bench.read16k(ROOT / it["path"])
            s0 = int(round(it["start_T"] * SR))
            assert s0 % FRAME == 0, (self.name, it["label"], s0)
            seg = np.clip(np.round(x * 32768), -32768, 32767).astype("<i2")
            pcm[s0:s0 + len(seg)] = seg[:max(0, len(pcm) - s0)]
        f = pcm.reshape(n, FRAME).astype(np.float64) / 32768
        self.rms = np.sqrt(np.mean(f * f, axis=1))
        self.db = 20 * np.log10(np.maximum(self.rms, 1e-9))
        self.T = (np.arange(n) + 1) * FRAME_S  # bench: frame k is processed at t0 + (k+1)*FRAME_S
        self.segs = [(it["start_T"] + s["start"], it["start_T"] + s["end"], i, j, it["label"])
                     for i, it in enumerate(self.items) for j, s in enumerate(it.get("segments", []))]
        # output (echo) intervals: play start .. first of reply_stop / audio end / cancel
        self.out = []
        for u in self.utts:
            ps = u.get("play_start_at")
            if ps is None:
                continue
            stops = [u[k] for k in ("reply_stop_at", "audio_end_at", "ended_at", "cancelled_at")
                     if u.get(k) is not None and u[k] >= ps]
            self.out.append((ps, min(stops) if stops else ps + 30.0))
        logged = {}
        for e in jrows(tdir / "events.jsonl"):
            d = e.get("data") or {}
            if e.get("event") == "client_hold" and d.get("action") == "hold":
                logged.setdefault(d.get("utterance_id"), d)
        rows = {}
        for x in ends:
            if x.get("trial") == self.name and x.get("utterance_id") is not None:
                y = rows.get(x["utterance_id"])
                if y is None or (x.get("stop_T") or 0) > (y.get("stop_T") or 0):
                    rows[x["utterance_id"]] = x
        self.windows = []
        for u in self.utts:
            w0 = u.get("first_packet_at")
            if w0 is None:
                continue
            local_at = (u.get("local_hold_at") or [None])[0]
            cands = [u.get("cancelled_at"),
                     u.get("initial_play_at") if local_at is not None
                     else (u.get("play_start_at") or u.get("initial_play_at"))]
            cands = [c for c in cands if c is not None and c > w0]
            if not cands:
                continue
            w1 = min(cands)
            blocked, open_at = [], None
            for h in u.get("holds", []):
                if not isinstance(h, dict) or h.get("source") != "server":
                    continue
                if h.get("held") and open_at is None:
                    open_at = h["at"]
                elif not h.get("held") and open_at is not None:
                    blocked.append((open_at, h["at"]))
                    open_at = None
            if open_at is not None:
                blocked.append((open_at, w1))
            row = rows.get(u["utterance_id"]) or {}
            self.windows.append(dict(
                uid=u["utterance_id"], w0=w0, w1=w1, blocked=blocked, local_hold_at=local_at,
                logged=logged.get(u["utterance_id"]), stop_T=row.get("stop_T"),
                label=row.get("item_label"), transcript=(row.get("transcript") or "")[:12]))
        self.n = n

    def echo_flags(self, window_s):
        e = np.zeros(self.n, bool)
        for a, b in self.out:
            k0 = int(np.searchsorted(self.T, a, "left"))
            k1 = int(np.searchsorted(self.T, b + window_s, "left"))
            e[k0:k1] = True
        return e

    def run(self, params):
        d = OnsetDetector(params)
        echo = self.echo_flags(OUTPUT_LATENCY_S + d.p["echoTailMs"] / 1000).tolist()
        rms = self.rms.tolist()
        onset = np.zeros(self.n, bool)
        floor = np.empty(self.n)
        thr = np.empty(self.n)
        frame = d.frame
        for k in range(self.n):
            r = frame(rms[k], 0.0, echo[k])
            onset[k], floor[k], thr[k] = r["onset"], r["floor_db"], r["threshold_db"]
        return onset, floor, thr, np.array(echo, bool)

    # ---- scoring
    def classify(self, t):
        for s, e, i, j, lab in self.segs:
            if s - MATCH_S <= t <= s + MATCH_S:
                return "match", t - s, lab, j
        for s, e, i, j, lab in self.segs:
            if s < t <= e:
                return "late", t - s, lab, j
        return "misfire", None, None, None

    def ana_ok(self, stop_T, t):
        return stop_T is not None and any(stop_T < s <= t + .05 for s, e, i, j, lab in self.segs)

    def first_trigger(self, onset, w0, w1, blocked=()):
        k0 = int(np.searchsorted(self.T, w0, "left"))
        k1 = int(np.searchsorted(self.T, w1, "left"))
        for k in range(k0, k1):
            if onset[k] and not any(a <= self.T[k] < b for a, b in blocked):
                return k
        return None

    def score(self, onset, floor, thr, echo):
        res = dict(events=[], approx=[], lag=[])
        for w in self.windows:
            k = self.first_trigger(onset, w["w0"], w["w1"], w["blocked"])
            rec = dict(trial=self.name, uid=w["uid"], label=w["label"], transcript=w["transcript"],
                       stop_T=w["stop_T"], w0=w["w0"], w1=w["w1"], live_hold_at=w["local_hold_at"],
                       live_ok=(w["local_hold_at"] is not None and self.ana_ok(w["stop_T"], w["local_hold_at"])),
                       logged=w["logged"])
            if k is None:
                rec.update(kind="none", t=None)
            else:
                t = float(self.T[k])
                kind, lag, lab, j = self.classify(t)
                rec.update(kind=kind, t=t, lag_ms=None if lag is None else round(lag * 1000, 1), seg=lab,
                           db=round(float(self.db[k]), 1), floor_db=round(float(floor[k]), 1),
                           thr_db=round(float(thr[k]), 1), rms=float(self.rms[k]),
                           ana_ok=self.ana_ok(w["stop_T"], t),
                           after_stop_ms=None if w["stop_T"] is None else round((t - w["stop_T"]) * 1000))
            res["events"].append(rec)
        for s, e, i, j, lab in self.segs:
            k = self.first_trigger(onset, e + APPROX_WIN[0], e + APPROX_WIN[1])
            rec = dict(trial=self.name, label=lab, seg=j, stop_T=e)
            if k is None:
                rec.update(kind="none")
            else:
                t = float(self.T[k])
                kind, lag, lab2, j2 = self.classify(t)
                rec.update(kind=kind, t=t, lag_ms=None if lag is None else round(lag * 1000, 1),
                           db=round(float(self.db[k]), 1), floor_db=round(float(floor[k]), 1),
                           thr_db=round(float(thr[k]), 1))
            res["approx"].append(rec)
        for s, e, i, j, lab in self.segs:
            ks = int(np.searchsorted(self.T, s - MATCH_S, "left"))
            if ks >= self.n or echo[min(self.n - 1, int(np.searchsorted(self.T, s, "left")))]:
                continue  # start during playback/echo tail: not a pre-playback case
            rec = dict(trial=self.name, label=lab, seg=j, cont=j > 0, start=s, pre=bool(ks > 0 and onset[ks - 1]))
            k = self.first_trigger(onset, s - MATCH_S, e)
            if k is None:
                rec.update(kind="missed", lag_ms=None)
            else:
                rec.update(kind="hit", lag_ms=round((float(self.T[k]) - s) * 1000, 1),
                           db=round(float(self.db[k]), 1), thr_db=round(float(thr[k]), 1))
            res["lag"].append(rec)
        return res


def load_run(run):
    run = Path(run)
    ends = jrows(run / "ends.jsonl")
    trials = []
    for tdir in sorted((run / "trials").glob("real_*")):
        if (tdir / "receipt.json").exists():
            trials.append(Trial(tdir, ends))
    return trials


CAUGHT = set()  # (trial, uid) of the run's live local holds that are true detections under the task rule


def evaluate(trials, params):
    ev, ap, lag = [], [], []
    for tr in trials:
        r = tr.score(*tr.run(params))
        ev += r["events"]
        ap += r["approx"]
        lag += r["lag"]
    kinds = lambda rows: {k: sum(1 for x in rows if x["kind"] == k) for k in ("match", "late", "misfire", "none")}
    ek, ak = kinds(ev), kinds(ap)
    caught = [x for x in ev if (x["trial"], x["uid"]) in CAUGHT]
    cont = [x for x in lag if x["cont"]]
    hit_c = [x["lag_ms"] for x in cont if x["kind"] == "hit"]
    hit_a = [x["lag_ms"] for x in lag if x["kind"] == "hit"]
    ev_lags = [x["lag_ms"] for x in ev if x["kind"] in ("match", "late")]
    m = dict(
        params={k: v for k, v in params.items()},
        ev_n=len(ev), ev_match=ek["match"], ev_late=ek["late"], ev_misfire=ek["misfire"],
        ev_misfire_pct=round(100 * ek["misfire"] / max(1, len(ev)), 1),
        ev_misfire_strict_pct=round(100 * (ek["misfire"] + ek["late"]) / max(1, len(ev)), 1),
        ev_lag_p50=pct(ev_lags, 50), ev_lag_p90=pct(ev_lags, 90),
        caught_n=len(caught), caught_kept=sum(1 for x in caught if x["kind"] in ("match", "late")),
        caught_lags=[x.get("lag_ms") for x in caught],
        ap_n=len(ap), ap_match=ak["match"], ap_late=ak["late"], ap_misfire=ak["misfire"],
        ap_misfire_pct=round(100 * ak["misfire"] / max(1, len(ap)), 1),
        cont_n=len(cont), cont_missed=sum(1 for x in cont if x["kind"] == "missed"),
        cont_pre=sum(1 for x in cont if x["pre"]), cont_p50=pct(hit_c, 50), cont_p90=pct(hit_c, 90),
        all_n=len(lag), all_missed=sum(1 for x in lag if x["kind"] == "missed"),
        all_pre=sum(1 for x in lag if x["pre"]), all_p50=pct(hit_a, 50), all_p90=pct(hit_a, 90),
    )
    return m, ev, ap, lag


def fmt_params(p):
    return " ".join(f"{k}={v}" for k, v in p.items())


def check(trials):
    """Reproduce the run's logged local holds with the run's own params (defaults)."""
    print("## replay check: logged hold vs offline trigger (events window, default params)")
    print("trial uid label | live_hold_T offline_T dT_ms | rms live/offline | floor live/offline dB | thr live/offline dB | kind ana_ok after_stop_ms")
    n_live = n_same = 0
    extra = []
    for tr in trials:
        onset, floor, thr, echo = tr.run({})
        r = tr.score(onset, floor, thr, echo)
        for x in r["events"]:
            if x["live_hold_at"] is None and x["kind"] == "none":
                continue
            if x["live_hold_at"] is None:
                extra.append(x)
                continue
            n_live += 1
            lg = x["logged"] or {}
            same = x["t"] is not None and abs(x["t"] - x["live_hold_at"]) <= 0.02
            n_same += same
            dt = None if x["t"] is None else round((x["t"] - x["live_hold_at"]) * 1000, 1)
            todb = lambda v: None if v is None else round(20 * math.log10(max(v, 1e-9)), 1)
            print(f"{tr.name} {x['uid']} {x['label']} | {x['live_hold_at']} {x['t']} {dt} | "
                  f"{todb(lg.get('frame_rms'))}/{x.get('db')} | {todb(lg.get('noise_floor_rms'))}/{x.get('floor_db')} | "
                  f"{todb(lg.get('threshold_rms'))}/{x.get('thr_db')} | {x['kind']} live_ok={x['live_ok']} "
                  f"offline_ok={x.get('ana_ok')} {x.get('after_stop_ms')}")
    print(f"live local holds: {n_live}, offline trigger within 20 ms: {n_same}")
    for x in extra:
        print("offline-only trigger:", {k: x[k] for k in ("trial", "uid", "label", "t", "kind", "db", "floor_db", "thr_db", "w0", "w1")})


def probe(trials, params, miss_params):
    """Why the default-parameter triggers fire, and what the segments missed under miss_params look like."""
    print(f"## probe: events-window triggers under {params or 'defaults'}: pre-noise (0.5 s before, outside truth) median/P90 dB, learned floor, trigger dB, run length above thr (frames), peak dB in next 0.3 s")
    for tr in trials:
        onset, floor, thr, echo = tr.run(params)
        r = tr.score(onset, floor, thr, echo)
        for x in r["events"]:
            if x["kind"] == "none":
                continue
            k = int(round(x["t"] / FRAME_S)) - 1
            k0 = max(0, k - 31)
            pre = [tr.db[i] for i in range(k0, k) if not any(s_ <= tr.T[i] <= e_ for s_, e_, *_ in tr.segs)]
            run = 0
            while k + run < tr.n and tr.db[k + run] > thr[k + run]:
                run += 1
            peak = float(np.max(tr.db[k:min(tr.n, k + 19)]))
            print(f"  {tr.name} {x['uid']} {x['label']} {x['kind']:7s} after_stop {x['after_stop_ms']} | pre {pct(pre, 50)}/{pct(pre, 90)} | floor {x['floor_db']} thr {x['thr_db']} | trig {x['db']} run {run} peak {round(peak, 1)}")
    print(f"## probe: continuation segments missed under {miss_params}: duration, max dB, P90 dB, first 6 frames dB, learned floor at start")
    for tr in trials:
        onset, floor, thr, echo = tr.run(miss_params)
        r = tr.score(onset, floor, thr, echo)
        for x in r["lag"]:
            if x["kind"] != "missed":
                continue
            s_ = x["start"]
            e_ = next(e for a, e, i, j, lab in tr.segs if abs(a - s_) < 1e-6)
            ks, ke = int(np.searchsorted(tr.T, s_)), int(np.searchsorted(tr.T, e_))
            seg = tr.db[ks:max(ke, ks + 1)]
            print(f"  {tr.name} {x['label']} seg{x['seg']} cont={x['cont']} dur {round(e_ - s_, 2)} s | max {round(float(seg.max()), 1)} P90 {pct(seg.tolist(), 90)} | first {[round(float(v), 1) for v in seg[:6]]} | floor {round(float(floor[ks]), 1)} thr {round(float(thr[ks]), 1)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--grid2", action="store_true", help="riseRate x fine thresholds")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--params", action="append", default=[])
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args()
    t0 = time.time()
    trials = load_run(a.run)
    print(f"loaded {len(trials)} trials, {sum(t.n for t in trials)} frames, {sum(len(t.segs) for t in trials)} truth segments, "
          f"{sum(len(t.windows) for t in trials)} event windows ({time.time() - t0:.1f}s)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = evaluate(trials, {})[1]
    CAUGHT.update((x["trial"], x["uid"]) for x in base if x["live_ok"] and x["kind"] in ("match", "late"))
    print(f"must-keep live catches (default params, task rule): {sorted(CAUGHT)}")
    if a.check:
        check(trials)
    if a.probe:
        probe(trials, {}, {"marginDb": 15, "absMinDb": -45})
    for ps in a.params:
        p = json.loads(ps)
        m, ev, apx, lag = evaluate(trials, p)
        print(json.dumps(m, ensure_ascii=False))
        if a.detail:
            print("-- events window triggers")
            for x in ev:
                if x["kind"] != "none":
                    print("  ", {k: x.get(k) for k in ("trial", "uid", "label", "transcript", "kind", "lag_ms", "after_stop_ms", "db", "floor_db", "thr_db", "live_ok")})
            print("-- continuation segments missed/pre")
            for x in lag:
                if x["cont"] and (x["kind"] == "missed" or x["pre"]):
                    print("  ", x)
            (out / "detail.json").write_text(json.dumps(dict(metrics=m, events=ev, approx=apx, lag=lag), ensure_ascii=False, indent=1), encoding="utf-8")
    if a.grid or a.grid2:
        grid = (dict(marginDb=[12, 15, 18, 21], absMinDb=[-50, -45, -40, -35], onsetFrames=[2, 3, 4], floorInitDb=[-60, -50])
                if a.grid else
                dict(marginDb=[12, 14, 15, 16], absMinDb=[-50, -47, -45, -43, -40], onsetFrames=[2, 3], riseRate=[0.02, 0.05, 0.1]))
        keys = list(grid)
        rows = []
        with (out / ("grid.jsonl" if a.grid else "grid2.jsonl")).open("w", encoding="utf-8") as f:
            for combo in itertools.product(*(grid[k] for k in keys)):
                p = dict(zip(keys, combo))
                m = evaluate(trials, p)[0]
                rows.append(m)
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
                print(f"{fmt_params(p)} | ev_mis {m['ev_misfire']}/{m['ev_n']} ({m['ev_misfire_pct']}%) late {m['ev_late']} | "
                      f"caught {m['caught_kept']}/{m['caught_n']} | ap_mis {m['ap_misfire_pct']}% | "
                      f"cont P50/P90 {m['cont_p50']}/{m['cont_p90']} missed {m['cont_missed']} pre {m['cont_pre']} | "
                      f"all P50/P90 {m['all_p50']}/{m['all_p90']} missed {m['all_missed']}", flush=True)
        # Pareto: minimise ev_misfire_pct and cont_p90 (ties by cont_p50)
        pareto = []
        for m in rows:
            dom = any((o["ev_misfire_pct"] <= m["ev_misfire_pct"] and (o["cont_p90"] or 1e9) <= (m["cont_p90"] or 1e9)
                       and (o["ev_misfire_pct"], o["cont_p90"] or 1e9) != (m["ev_misfire_pct"], m["cont_p90"] or 1e9))
                      for o in rows)
            if not dom:
                pareto.append(m)
        pareto.sort(key=lambda m: (m["ev_misfire_pct"], m["cont_p90"] or 1e9))
        lines = ["| 参数 | 误触（事件窗） | late | 拦住 7 次保留 | 误触（近似窗） | 续说滞后 P50/P90 ms | 漏检/预触发 | 全段 P50/P90 | 全段漏检 |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for m in pareto:
            p = m["params"]
            lines.append(f"| {fmt_params(p)} | "
                         f"{m['ev_misfire']}/{m['ev_n']}（{m['ev_misfire_pct']}%） | {m['ev_late']} | {m['caught_kept']}/{m['caught_n']} | "
                         f"{m['ap_misfire']}/{m['ap_n']}（{m['ap_misfire_pct']}%） | {m['cont_p50']}/{m['cont_p90']} | "
                         f"{m['cont_missed']}/{m['cont_pre']} | {m['all_p50']}/{m['all_p90']} | {m['all_missed']} |")
        (out / ("pareto.md" if a.grid else "pareto2.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
