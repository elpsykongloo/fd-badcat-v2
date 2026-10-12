"""The bench's simulated client must run the SAME client-hold-v1 rule as speech-player.js."""
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_guarded_latency as bench  # noqa: E402

PLAYWRIGHT_NODE = "/root/miniconda3/envs/fd-sds/lib/python3.10/site-packages/playwright/driver/node"


def _node():
    found = shutil.which("node")
    if found:
        return found
    return PLAYWRIGHT_NODE if os.path.exists(PLAYWRIGHT_NODE) else None


def _sequence(seed=7, n=3000):
    rng = random.Random(seed)
    rows, level = [], 1e-4
    for i in range(n):
        phase = (i // 150) % 6
        if phase in (1, 4):
            level = 10 ** rng.uniform(-2.5, -1)       # speech-like bursts
        elif phase == 3:
            level = 10 ** rng.uniform(-3.2, -2.6)     # louder noise step
        else:
            level = 10 ** rng.uniform(-4.5, -3.5)     # quiet floor
        ref = 10 ** rng.uniform(-2.5, -1.5) if rng.random() < .03 else 0.0
        rows.append([level, ref, rng.random() < .02])
    return rows


@pytest.mark.parametrize("params", [{}, {"marginDb": 9, "onsetFrames": 3, "absMinDb": -55, "refHangFrames": 4}])
def test_python_onset_detector_matches_speech_player_js(tmp_path, params):
    node = _node()
    if node is None:
        pytest.skip("node not available")
    rows = _sequence()
    script = tmp_path / "onset.cjs"
    script.write_text("""
const fs = require("node:fs"), vm = require("node:vm");
const [src, data, params] = process.argv.slice(2);
const scope = vm.createContext({performance: {now: () => 0}, Float32Array, DataView});
vm.runInContext(fs.readFileSync(src, "utf8").replace(/^export /gm, ""), scope);
const OnsetDetector = vm.runInContext("OnsetDetector", scope);
const d = new OnsetDetector(JSON.parse(params));
const out = JSON.parse(fs.readFileSync(data, "utf8")).map(([m, r, o]) => {
  const x = d.frame(m, r, o);
  return [x.onset, x.echo, x.floorDb, x.thresholdDb];
});
process.stdout.write(JSON.stringify(out));
""", encoding="utf-8")
    data = tmp_path / "rows.json"
    data.write_text(json.dumps(rows), encoding="utf-8")
    js = json.loads(subprocess.run([node, str(script), str(ROOT / "src/static/speech-player.js"), str(data),
                                    json.dumps(params)], capture_output=True, text=True, check=True).stdout)
    detector = bench.OnsetDetector(params)
    py = []
    for mic, ref, recent in rows:
        r = detector.frame(mic, ref, recent)
        py.append([r["onset"], r["echo"], r["floor_db"], r["threshold_db"]])
    assert len(js) == len(py)
    onsets = 0
    for i, (a, b) in enumerate(zip(js, py)):
        assert a[0] == b[0] and a[1] == b[1], i
        assert math.isclose(a[2], b[2], abs_tol=1e-9) and math.isclose(a[3], b[3], abs_tol=1e-9), i
        onsets += a[0]
    assert onsets > 0  # the sequence actually exercises the onset path


class _Link:
    def __init__(self):
        self.sent = []

    def send_json(self, event, data):
        self.sent.append((event, data))


def _browser(client_hold=None, local_pause=False, release_lead_ms=None):
    args = SimpleNamespace(local_pause=local_pause, release_lead_ms=release_lead_ms, output_latency_ms=0.0,
                           net_delay_ms=0.0, client_hold=client_hold)
    br = bench.Browser(_Link(), args, [], [bench.now()])
    br.telemetry = True
    return br


def _arm(br, startup_ms=350):
    br.start({"utterance_id": 1, "startup_ms": startup_ms, "buffer_ms": 600})
    s = br.speech
    s.received, s.rate, s.first_audio = 2400, 24000, 0
    s.next = s.play_at = bench.now() + 10  # far in the future: not started
    br.schedule(s, 0, 2400)
    return s


def test_bench_reads_client_hold_like_demo_js():
    off = _browser({"local_hold": "off", "resume_lead_ms": 350})
    assert off.local_mode == "off" and off.resume_lead_ms == 350
    assert off.local_pause(1.0) is None  # detector not even run
    on = _browser({"local_hold": "on", "resume_lead_ms": 80, "params": {"onsetFrames": 1}})
    assert on.local_mode == "on" and on.resume_lead_ms == 80 and on.detector.p["onsetFrames"] == 1
    assert _browser({}, local_pause=True).local_mode == "on"
    assert _browser(None).resume_lead_ms is None  # older backend: re-wait startup
    assert _browser({"resume_lead_ms": 80}, release_lead_ms=120).resume_lead_ms == 120


def test_bench_local_hold_once_per_utterance_and_server_wins():
    br = _browser({"local_hold": "on", "resume_lead_ms": 80})
    s = _arm(br)
    for _ in range(3):
        br.local_pause(1e-5)
    assert not s.held
    br.local_pause(0.1)
    br.local_pause(0.1)  # onsetFrames=2
    assert s.held and s.local_hold and not s.local_hold["done"]
    holds = [d for e, d in br.link.sent if e == "demo_telemetry" and d["kind"] == "client_hold"]
    assert [d["action"] for d in holds] == ["hold"] and holds[0]["timer_ms"] == 500  # max(300, 2*150+200)
    br.local_server(False)  # server releases: timer cleared, resume with 80 ms lead
    assert not s.held and s.local_hold["done"]
    assert abs((s.play_at - bench.now()) - .08) < .05
    s.play_at = s.next = bench.now() + 10
    for _ in range(4):
        br.local_pause(0.1)
    assert not s.held  # at most one local hold per utterance
    actions = [d["action"] for e, d in br.link.sent if e == "demo_telemetry" and d["kind"] == "client_hold"]
    assert actions == ["hold", "release"]


def test_bench_local_hold_timer_and_shadow():
    br = _browser({"local_hold": "on"})
    s = _arm(br)
    br.local_pause(0.1)
    br.local_pause(0.1)
    assert s.held
    s.local_hold["timer_at"] = bench.now() - 1
    br.tick()
    assert not s.held and s.local_hold["done"]
    assert abs((s.play_at - bench.now()) - .35) < .05  # no resume lead from the backend: startup
    shadow = _browser({"local_hold": "shadow"})
    s = _arm(shadow)
    shadow.local_pause(0.1)
    shadow.local_pause(0.1)
    assert not s.held and s.local_hold["shadow"]
    actions = [d["action"] for e, d in shadow.link.sent if e == "demo_telemetry" and d["kind"] == "client_hold"]
    assert actions == ["shadow"]
