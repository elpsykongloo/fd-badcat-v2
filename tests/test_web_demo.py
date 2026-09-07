"""Browser endpoint/launcher contracts; no model service, corpus or GPU calls."""
import importlib.util
from pathlib import Path
import socket
import sys

from fastapi.testclient import TestClient
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from backend import create_app
import engine


def test_demo_static_and_config_do_not_claim_upstream_health():
    for cfg, expected in [({}, False), ({"stream_response": True}, True),
                          ({"arch": "legacy", "stream_response": True}, False),
                          ({"phase": "b", "stream_response": True}, False)]:
        with TestClient(create_app({}, {}, engine_cfg=cfg)) as client:
            assert client.get("/api/demo/info").json() == {"protocol": "pcm16.v1", "streaming": expected}
            page = client.get("/demo/")
            assert page.status_code == 200
            assert "自然接话" in page.text
            for file in ("demo.js", "demo.css", "speech-player.js", "mic-worklet.js", "favicon.svg"):
                assert client.get("/demo/" + file).status_code == 200


def test_browser_handshake_assigns_path_and_waits_for_engine(monkeypatch, tmp_path):
    made = []
    class Engine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            made.append(self)
        async def run_realtime(self, ws):
            await ws.close()
    monkeypatch.setattr(engine, "ActorEngine", Engine)
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app({}, {}, engine_cfg={"stream_response": True})) as client:
        with client.websocket_connect("/realtime", headers={"origin": "http://testserver"}) as ws:
            ws.send_json({"event": "config", "data": {"client": "humdial-web", "audio_protocol": "pcm16.v1",
                          "exp": "../../escape", "lang": "../../escape"}})
            ready = ws.receive_json()
            assert ready["event"] == "demo_ready"
            assert ready["data"]["session_id"].startswith("web-demo-")
            assert made[0].output_dir == Path("exp") / ready["data"]["session_id"] / "realtimeout_live"
            assert made[0].kwargs["engine_cfg"]["stream_response"] is True
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("origin,protocol,enabled", [
    ("http://attacker.example", "pcm16.v1", True),
    ("http://testserver", "wav", True),
    ("http://testserver", "pcm16.v1", False),
    ("", "pcm16.v1", True),
])
def test_browser_bad_handshakes_are_rejected(origin, protocol, enabled, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app({}, {}, engine_cfg={"stream_response": enabled})) as client:
        with client.websocket_connect("/realtime", headers={"origin": origin}) as ws:
            ws.send_json({"event": "config", "data": {"client": "humdial-web", "audio_protocol": protocol}})
            assert ws.receive_json()["event"] == "error"
            assert ws.receive()["code"] == 1008
    assert not (tmp_path / "exp").exists()


def test_old_client_keeps_handshake_path_and_whole_wav(monkeypatch, tmp_path):
    made = []
    class Engine:
        def __init__(self, **kwargs):
            made.append(self)
            self.kwargs = kwargs
        async def run_realtime(self, ws):
            await ws.send_json({"event": "legacy-stub"})
            await ws.close()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "ActorEngine", Engine)
    with TestClient(create_app({}, {}, engine_cfg={"stream_response": True})) as client:
        with client.websocket_connect("/realtime") as ws:
            ws.send_json({"event": "config", "data": {"exp": "old-session", "lang": "zh"}})
            assert ws.receive_json()["event"] == "legacy-stub"
    assert made[0].output_dir == Path("exp/old-session/realtimeout_zh")
    assert made[0].kwargs["engine_cfg"]["stream_response"] is False


def load_launcher():
    spec = importlib.util.spec_from_file_location("serve_demo", ROOT / "scripts/serve_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_refuses_occupied_port_without_touching_owner():
    launcher = load_launcher()
    with socket.socket() as existing:
        existing.bind(("127.0.0.1", 0))
        existing.listen()
        with pytest.raises(RuntimeError, match="不会停止或接管"):
            launcher.require_free(existing.getsockname()[1])
        assert existing.fileno() >= 0


def test_launcher_shutdown_only_signals_its_owned_process_groups(monkeypatch):
    import io
    launcher = load_launcher()
    signals = []
    class Process:
        pid = 43210
        def poll(self):
            return 0
        def wait(self):
            return 0
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    log = io.StringIO()
    launcher.stop_owned([("test-child", Process(), log)])
    assert [pid for pid, _ in signals] == [43210, 43210]
    assert log.closed
