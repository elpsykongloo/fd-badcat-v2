"""--reply-stop-s in the bench's simulated client: drain/close semantics, fencing and resume lead."""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_guarded_latency as bench  # noqa: E402


class _Link:
    def __init__(self):
        self.sent = []

    def send_json(self, event, data):
        self.sent.append((event, data))


def _browser(reply_stop_s=None, mode="drain", client_hold=None):
    args = SimpleNamespace(local_pause=False, release_lead_ms=None, output_latency_ms=0.0, net_delay_ms=0.0,
                           client_hold=client_hold, reply_stop_s=reply_stop_s, reply_stop_mode=mode)
    br = bench.Browser(_Link(), args, [], [bench.now()])
    br.telemetry = True
    return br


def _packet(uid, seq, n=2400):
    return bench.OUTPUT_HEADER.pack(b"FDS1", uid, seq, 24000) + bytes(2 * n)


def _playing(br, played_s=0.2, received=4800):
    """Utterance 1 that started on the playback clock `played_s` ago and is still playing."""
    br.start({"utterance_id": 1, "startup_ms": 350, "buffer_ms": 600})
    s = br.speech
    for k in range(received // 2400):
        br.packet(_packet(1, k))
    t0 = bench.now() - played_s
    s.next = t0
    s.nodes = []
    for k in range(received // 2400):
        br.schedule(s, 2400 * k, 2400)
    s.play_at = t0
    s.started = True
    return s


def _events(br, name):
    return [d for e, d in br.link.sent if e == name]


def test_drain_acks_everything_and_ends_naturally():
    br = _browser(reply_stop_s=0.15)
    s = _playing(br, played_s=0.2, received=7200)  # 0.3 s received, ~0.2 s rendered
    br.tick()
    assert not s.drain  # not armed: the item is still being injected
    br.stop_armed = True
    br.tick()
    assert s.drain and s.played == s.received == 7200 and not s.nodes
    rec = br.utt[1]
    assert rec["reply_stop_mode"] == "drain" and 0.15 * 24000 <= rec["reply_stop_played_samples"] < 7200
    br.packet(_packet(1, 3))  # later packets are acknowledged on arrival
    assert s.played == s.received == 9600
    br.control({"event": "speech_audio_end", "data": {"utterance_id": 1, "samples": 9600}})
    last = _events(br, "playback_progress")[-1]
    assert last["ended"] is True and last["played_samples"] == 9600 and not br.busy()
    assert not _events(br, "playback_stopped")  # no client-side stop the server would ignore
    assert [d["kind"] for d in _events(br, "demo_telemetry")].count("playback_end") == 1


def test_no_stop_before_x_seconds_or_before_start():
    br = _browser(reply_stop_s=1.0)
    br.stop_armed = True
    s = _playing(br, played_s=0.2)
    br.tick()
    assert not s.drain and "reply_stop_at" not in br.utt[1]
    br2 = _browser(reply_stop_s=0.01)
    br2.stop_armed = True
    br2.start({"utterance_id": 1, "startup_ms": 350, "buffer_ms": 600})
    br2.packet(_packet(1, 0))  # playAt is 350 ms away: not started
    br2.tick()
    assert not br2.speech.drain
    assert not _browser(reply_stop_s=None).reply_stop_s


def test_close_is_the_demo_stop_button():
    br = _browser(reply_stop_s=0.15, mode="close")
    br.stop_armed = True
    s = _playing(br, played_s=0.2)
    br.tick()
    assert br.closed_by_stop and not s.drain
    event, last = br.link.sent[-1]
    assert event == "demo_telemetry" and last["kind"] == "stop"
    assert last["utterance_id"] == 1 and last["received_samples"] == s.received
    n = len(br.link.sent)
    # active = null: nothing more is sent, not even player.cancel()'s playback_stopped.
    br.cancel("replaced")
    br.emit("ping", seq=1)
    br.progress(s)
    br.tick()
    assert len(br.link.sent) == n and not _events(br, "playback_stopped")


def test_resume_lead_is_recorded_and_reported():
    br = _browser(client_hold={"local_hold": "on", "resume_lead_ms": 80})
    br.start({"utterance_id": 1, "startup_ms": 300, "buffer_ms": 600})
    br.packet(_packet(1, 0))
    s = br.speech
    assert s.scheduled_lead_ms == 300
    s.next = s.play_at = bench.now() + 10
    br.local_pause(0.1)
    br.local_pause(0.1)
    assert s.held
    s.local_hold["timer_at"] = bench.now() - 1
    br.tick()
    assert not s.held and br.utt[1]["release_lead_ms"] == [80] and s.scheduled_lead_ms == 80
    timeout = [d for d in _events(br, "demo_telemetry") if d["kind"] == "client_hold" and d["action"] == "timeout"]
    assert timeout and timeout[0]["scheduled_lead_ms"] == 80


def test_link_delays_client_to_server():
    class WS:
        def __init__(self):
            self.at = []

        async def send(self, payload):
            self.at.append(bench.now())

    async def go():
        ws = WS()
        link = bench.Link(ws, 0.05)
        task = asyncio.create_task(link.sender())
        t0 = bench.now()
        link.send_bytes(b"x")
        link.send_json("e", {})
        link.out.put_nowait((bench.now(), None))
        await task
        return t0, ws.at

    t0, at = asyncio.run(go())
    assert len(at) == 2 and all(t - t0 >= 0.045 for t in at)
