"""Unified private session/turn/call diagnostics; no model or network."""
import json
from pathlib import Path
import sys
import wave

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from demo_diagnostics import (AudioRingCapture, DiagnosticStore, build_manifest,
    build_spans, build_turns, finalize_session, private_json, private_jsonl)
from demo_session_replay import replay_session, signature


def manifest(tmp_path, session="web-demo-a1"):
    return build_manifest(session_id=session, repository_root=tmp_path,
        profile="chat-demo-v1",
        engine_cfg={"guarded_turns": True, "input_preroll_ms": 320,
                    "secret_option": "must-not-leak"},
        delay={"end_hold_frame": .64, "after_continue_time": 2.5},
        llm_cfg={"model": "local", "api_key": "must-not-leak"},
        asr_cfg={"backend": "sensevoice", "provider": "cpu", "password": "must-not-leak"})


def test_manifest_is_allowlisted_and_contains_protocol_versions(tmp_path):
    value = manifest(tmp_path)
    text = json.dumps(value)
    assert value["protocols"]["trace"] == "demo-trace-v2"
    assert value["effective"]["engine"]["input_preroll_ms"] == 320
    assert value["effective"]["input_timing"] == {
        "vad_threshold": .5, "vad_silence_ms": 100, "preroll_ms": 320}
    assert value["effective"]["time"]["long_interrupt_s"] == 1.5
    assert "must-not-leak" not in text and "api_key" not in text and "password" not in text


def test_lifecycle_only_session_does_not_create_fake_turns():
    rows = [
        {"event":"disconnect","generation":0,"turn":0,"data":{}},
        {"event":"session_final","generation":1,"turn":0,"data":{"state":"LISTEN"}},
    ]
    assert build_turns(rows, {}) == []


def test_opt_in_audio_ring_is_bounded_aligned_and_private(tmp_path):
    capture = AudioRingCapture(tmp_path / "capture", max_seconds=1)
    for seq in range(1, 66):
        mic = np.full(256, seq / 100, dtype=np.float32)
        capture.raw(seq, seq * .016, mic, -mic)
        capture.clean(seq, mic / 2)
    capture.input_settings({"echoCancellation": True, "sampleRate": 48000})
    meta = capture.close()
    assert meta["truncated"] and meta["samples"] <= 16000
    assert meta["first_seq"] > 1 and meta["last_seq"] == 65
    assert (tmp_path / "capture/frames.jsonl").stat().st_mode & 0o777 == 0o600
    for name in ("browser_mic.wav", "render_reference.wav", "engine_clean.wav"):
        with wave.open(str(tmp_path / "capture" / name)) as wav:
            assert wav.getnframes() == meta["samples"] and wav.getframerate() == 16000


def make_session(tmp_path, *, findings=True):
    session = "web-demo-a1"
    diag = tmp_path / "exp" / session / "realtimeout_live" / "diagnostics"
    diag.mkdir(parents=True)
    private_json(diag / "manifest.json", manifest(tmp_path, session))
    rows = [
        {"event":"session_start","seq":1,"server_ms":0,"generation":0,"turn":0,"data":{}},
        {"event":"model_call_dispatch","seq":2,"server_ms":1,"generation":0,"turn":0,
         "data":{"call_id":"call-1","kind":"input_route"}},
        {"event":"model_case_started","seq":3,"server_ms":2,"generation":0,"turn":0,
         "data":{"case_id":"case-1","call_id":"call-1","input_id":1}},
        {"event":"input_decision","seq":4,"server_ms":3,"generation":0,"turn":0,
         "data":{"input_id":1,"revision":1,"route":"yield_ready","call_ids":["call-1"]}},
        {"event":"speech_start","seq":5,"server_ms":4,"generation":0,"turn":0,
         "data":{"utterance_id":1,"candidate_id":9}},
        {"event":"model_case_queued","seq":6,"server_ms":5,"generation":0,"turn":0,
         "data":{"case_id":"case-1","queued":True,"status":"completed"}},
        {"event":"model_call_done","seq":7,"server_ms":6,"generation":0,"turn":0,
         "data":{"call_id":"call-1","status":"completed"}},
        {"event":"session_final","seq":8,"server_ms":7,"generation":0,"turn":0,
         "data":{"state":"LISTEN","pending_tasks":0,"active_capacity":0}},
        {"event":"trace_closed","dropped":0},
    ]
    if not findings:
        rows.insert(4, {"event":"candidate_confirmed","seq":5,"server_ms":3.5,
            "generation":0,"turn":0,"data":{"candidate_id":9}})
        for index, row in enumerate(rows[:-1], 1): row["seq"] = index
    private_jsonl(diag.parent / "events.jsonl", rows)
    case_dir = tmp_path / "cases/captures/case-1"; case_dir.mkdir(parents=True)
    private_json(case_dir / "case.json", {"version":"demo-case-v2","case_id":"case-1",
        "kind":"input_route","context":{"session_id":session,"turn":0,"call_id":"call-1"},
        "outcome":{"status":"completed","text":"yield_ready","elapsed_ms":12}})
    return session, diag


def test_finalize_joins_turn_call_case_and_builds_anomaly_queue(tmp_path):
    session, diag = make_session(tmp_path)
    summary = finalize_session(diag, diag.parent / "events.jsonl", tmp_path / "cases")
    turns = [json.loads(line) for line in (diag / "turns.jsonl").read_text().splitlines()]
    assert turns[0]["turn_id"] == "g0-t0"
    assert turns[0]["call_ids"] == ["call-1"] and turns[0]["case_ids"] == ["case-1"]
    assert any(f["code"] == "private_candidate_published" for f in summary["anomalies"])
    store = DiagnosticStore(tmp_path / "exp", tmp_path / "cases")
    assert store.list_sessions(anomalies_only=True)[0]["session_id"] == session


def test_review_is_revisioned_and_delete_covers_linked_cases(tmp_path):
    session, diag = make_session(tmp_path, findings=False)
    finalize_session(diag, diag.parent / "events.jsonl", tmp_path / "cases")
    store = DiagnosticStore(tmp_path / "exp", tmp_path / "cases")
    one = store.add_review(session, "g0-t0", labels=["route"], note="route was correct",
                           reviewer="tester")
    two = store.add_review(session, "g0-t0", labels=["acceptable"], note="second pass",
                           reviewer="tester2")
    assert (one["revision"], two["revision"]) == (1, 2)
    assert len(store.session_detail(session)["reviews"]) == 2
    store.delete_session(session)
    assert not diag.exists() and not (tmp_path / "cases/captures/case-1").exists()


def test_retention_prunes_session_and_its_linked_cases(tmp_path):
    session, diag = make_session(tmp_path, findings=False)
    value = json.loads((diag / "manifest.json").read_text())
    value["created_utc"] = "2000-01-01T00:00:00+00:00"
    private_json(diag / "manifest.json", value)
    finalize_session(diag, diag.parent / "events.jsonl", tmp_path / "cases")
    store = DiagnosticStore(tmp_path / "exp", tmp_path / "cases")
    assert store.prune(30) == [session]
    assert not diag.exists() and not (tmp_path / "cases/captures/case-1").exists()


def test_local_review_api_lists_detail_and_appends_revision(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend import create_app
    session, diag = make_session(tmp_path, findings=False)
    finalize_session(diag, diag.parent / "events.jsonl", tmp_path / "cases")
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app({}, {}, engine_cfg={"diagnostics_review": True,
                                                   "case_capture_dir": "cases"})) as client:
        listed = client.get("/api/demo/diagnostics/sessions").json()["sessions"]
        assert listed[0]["session_id"] == session
        detail = client.get(f"/api/demo/diagnostics/sessions/{session}").json()
        assert detail["turns"][0]["turn_id"] == "g0-t0" and "case-1" in detail["cases"]
        response = client.post(f"/api/demo/diagnostics/sessions/{session}/turns/g0-t0/reviews",
            json={"labels":["acceptable"],"note":"reviewed through local UI API","reviewer":"tester"})
        assert response.status_code == 200 and response.json()["revision"] == 1
        assert client.post(f"/api/demo/diagnostics/sessions/{session}/turns/g0-t0/reviews",
            json={"labels":["unknown"],"note":"bad","reviewer":"tester"}).status_code == 400


def test_replay_signature_ignores_timing_and_keeps_actions():
    rows = [{"event":"input_decision","server_ms":10,"data":{"route":"keep","input_id":1,"audit":{"x":1}}},
            {"event":"speech_start","server_ms":20,"data":{"utterance_id":2,"turn":0}}]
    assert signature(rows) == [{"event":"input_decision","route":"keep","input_id":1},
                               {"event":"speech_start","turn":0,"utterance_id":2}]


def test_cross_layer_spans_join_call_utterance_and_clock_interval():
    rows = [
        {"event":"model_call_dispatch","server_ms":10,"data":{"call_id":"c1","parent_id":"u1","kind":"tts","transport":"stream"}},
        {"event":"capacity_acquired","server_ms":15,"data":{"call_id":"c1","wait_ms":5,"capacity":{"active_total":1}}},
        {"event":"model_call_first_output","server_ms":40,"data":{"call_id":"c1","elapsed_ms":30,"request_id":"upstream-1"}},
        {"event":"model_call_done","server_ms":60,"data":{"call_id":"c1","status":"completed","elapsed_ms":50}},
        {"event":"speech_start","server_ms":12,"data":{"utterance_id":7,"parent_id":"u1"}},
        {"event":"speech_timing","server_ms":42,"data":{"utterance_id":7,"phase":"tts_chunk","decode_ms":2}},
        {"event":"speech_played","server_ms":90,"data":{"utterance_id":7,"played_samples":100}},
        {"event":"client_ping","server_ms":100,"data":{"seq":1,"client_ms":1000}},
        {"event":"demo_pong","server_ms":102,"data":{"seq":1}},
        {"event":"client_rtt","server_ms":105,"data":{"seq":1,"client_ms":1010,"rtt_ms":10}},
    ]
    spans = build_spans(rows)
    call = next(s for s in spans if s["span_type"] == "model")
    utterance = next(s for s in spans if s["span_type"] == "utterance")
    clock = next(s for s in spans if s["span_type"] == "clock_sync")
    assert call["capacity_wait_ms"] == 5 and call["upstream_request_id"] == "upstream-1"
    assert utterance["milestones"]["playback_done_ack"] == 90 and utterance["tts"][0]["decode_ms"] == 2
    assert clock["uncertainty_ms"] == 4


async def test_opted_in_session_replay_drives_current_actor_without_models(tmp_path):
    session = "web-demo-replay1"
    diag = tmp_path / "exp" / session / "realtimeout_live" / "diagnostics"
    value = build_manifest(session_id=session, repository_root=tmp_path,
        profile="chat-demo-v1", engine_cfg={"chat_demo":True,"stream_response":True,
        "guarded_turns":True,"speculative_response":True,"cancellable_response":True,
        "input_protocol":"pcm16.ref.v1"}, delay={"end_hold_frame":.64,"after_continue_time":2.5},
        llm_cfg={}, asr_cfg={}, capture_enabled=True, capture_seconds=2)
    private_json(diag / "manifest.json", value)
    capture = AudioRingCapture(diag / "capture", max_seconds=2)
    for seq in range(1, 5):
        capture.raw(seq, seq * .016, np.zeros(256, dtype=np.float32), np.zeros(256, dtype=np.float32))
        capture.clean(seq, np.zeros(256, dtype=np.float32))
    capture.close()
    private_jsonl(diag.parent / "events.jsonl", [
        {"event":"session_start","seq":1,"server_ms":0,"t_audio":0,"data":{}},
        {"event":"session_final","seq":2,"server_ms":64,"t_audio":.064,
         "data":{"state":"LISTEN","pending_tasks":0,"active_capacity":0}},
        {"event":"trace_closed","dropped":0}])
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "src/config.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load((root / "configs/demo_chat.yaml").read_text(encoding="utf-8"))
    for section, values in overlay.items(): cfg.setdefault(section, {}).update(values)
    report = await replay_session(session, diag, tmp_path / "cases", cfg,
                                  tmp_path / "replay", speed=20)
    assert report["mismatch_count"] == 0 and report["remaining_model_cases"] == {}
    assert report["timing_comparable"] is False
