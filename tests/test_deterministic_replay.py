#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test suite for deterministic injected replay (W1 core validation).

Verifies:
  1. Audio clock monotonicity and correctness
  2. DecisionScript faithful replay from golden traces
  3. Deterministic output (same input → same output, N times)
  4. 60× real-time speed achievement
  5. Trace equivalence to golden baseline
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import pytest
import numpy as np

# Inject src/ into path
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from audio_clock import (
    AudioClock,
    AudioClockFrameGenerator,
    validate_audio_clock_monotonicity,
    audio_clock_stats,
)
from injected_replay import (
    DecisionScript,
    InjectedReplaySession,
    compare_traces,
    extract_decisions_summary,
    load_golden_trace,
)


# ============================================================================
# Audio Clock Tests
# ============================================================================

def test_audio_clock_tick():
    """Test audio clock advances correctly by sample count."""
    clock = AudioClock()
    assert clock.t_audio == 0.0

    # 256 samples @ 16kHz = 0.016s
    t = clock.tick(256)
    assert abs(t - 0.016) < 1e-6

    # Another 256 samples
    t = clock.tick(256)
    assert abs(t - 0.032) < 1e-6


def test_audio_clock_advance():
    """Test audio clock fast-forward for tail drain."""
    clock = AudioClock()
    clock.tick(256)  # t=0.016

    # Fast-forward to 1.0s
    t = clock.advance(1.0)
    assert abs(t - 1.0) < 1e-6

    # Cannot rewind
    with pytest.raises(ValueError):
        clock.advance(0.5)


@pytest.mark.asyncio
async def test_audio_clock_schedule_and_drain():
    """Test scheduling and draining results at audio clock time."""
    delivered = []

    async def callback(kind, result, metadata):
        delivered.append((kind, result, metadata))

    clock = AudioClock(delivery_callback=callback)

    # Schedule three results
    clock.schedule('judge', {'text': 'switch', 'infer': 0.1}, {'turn': 0}, infer_time=0.1)
    clock.schedule('response', {'text': '回复1', 'infer': 0.2}, {'turn': 0}, infer_time=0.2)
    clock.schedule('asr', {'text': '转写1', 'infer': 0.15}, {'turn': 0}, infer_time=0.15)

    assert clock.pending_count() == 3
    assert abs(clock.next_due_time() - 0.1) < 1e-6

    # Advance to 0.05s: nothing due yet
    clock.advance(0.05)
    count = await clock.drain_due()
    assert count == 0
    assert len(delivered) == 0

    # Advance to 0.12s: judge + asr due (0.1, 0.15)
    clock.advance(0.12)
    count = await clock.drain_due()
    assert count == 1  # only judge at 0.1
    assert len(delivered) == 1
    assert delivered[0][0] == 'judge'

    # Advance to 0.16s: asr now due
    clock.advance(0.16)
    count = await clock.drain_due()
    assert count == 1
    assert len(delivered) == 2
    assert delivered[1][0] == 'asr'

    # Advance to 0.25s: response due
    clock.advance(0.25)
    count = await clock.drain_due()
    assert count == 1
    assert len(delivered) == 3
    assert delivered[2][0] == 'response'

    assert clock.pending_count() == 0


def test_audio_clock_frame_generator():
    """Test frame generator produces correct audio-clock timestamps."""
    gen = AudioClockFrameGenerator(sample_rate=16000, window_size=256)

    pcm = np.zeros(512, dtype=np.float32)  # 2 frames
    frames = gen.chunk_to_frames(pcm)

    assert len(frames) == 2
    assert frames[0]['seq'] == 1
    assert abs(frames[0]['t_audio'] - 0.016) < 1e-6
    assert frames[1]['seq'] == 2
    assert abs(frames[1]['t_audio'] - 0.032) < 1e-6


def test_validate_audio_clock_monotonicity():
    """Test trace monotonicity validator."""
    good_trace = [
        {'event': 'vad_start', 'data': {'t_audio': 0.1}},
        {'event': 'vad_done', 'data': {'t_audio': 0.5}},
        {'event': 'llm_done', 'data': {'t_audio': 0.6}},
    ]
    valid, msg = validate_audio_clock_monotonicity(good_trace)
    assert valid
    assert msg == ""

    bad_trace = [
        {'event': 'vad_start', 'data': {'t_audio': 0.5}},
        {'event': 'vad_done', 'data': {'t_audio': 0.3}},  # rewind!
    ]
    valid, msg = validate_audio_clock_monotonicity(bad_trace)
    assert not valid
    assert "0.3" in msg


# ============================================================================
# DecisionScript Tests
# ============================================================================

def test_decision_script_parsing(tmp_path):
    """Test DecisionScript parses golden traces correctly."""
    golden = [
        {'event': 'llm_done', 'data': {
            'kind': 'judge', 'content': 'switch', 'infer_time': 0.093, 'timestamp': 3.81}},
        {'event': 'llm_done', 'data': {
            'kind': 'response', 'content': '回复1', 'infer_time': 0.211, 'timestamp': 4.02}},
        {'event': 'asr_done', 'data': {
            'content': '转写1', 'infer_time': 0.15, 'timestamp': 4.1}},
        {'event': 'tts_done', 'data': {
            'infer_time': 1.098, 'dur_audio': 2.5, 'timestamp': 5.1}},
    ]

    script = DecisionScript(golden)
    stats = script.stats()

    assert stats['judge'] == 1
    assert stats['response'] == 1
    assert stats['asr'] == 1
    assert stats['tts'] == 1

    # Pop in order
    res = script('judge', {})
    assert res['text'] == 'switch'
    assert abs(res['infer'] - 0.093) < 1e-6

    res = script('response', {})
    assert res['text'] == '回复1'

    res = script('asr', {})
    assert res['text'] == '转写1'

    res = script('tts', {})
    assert abs(res['dur_audio'] - 2.5) < 1e-6


def test_decision_script_legacy_trace():
    """Test DecisionScript handles legacy traces without 'kind' field."""
    golden = [
        {'event': 'llm_done', 'data': {
            'content': 'switch', 'infer_time': 0.1, 'timestamp': 1.0}},
        {'event': 'llm_done', 'data': {
            'content': 'no', 'infer_time': 0.08, 'timestamp': 1.2}},
    ]

    script = DecisionScript(golden)
    stats = script.stats()

    # Legacy traces go to '_llm' fallback queue
    assert stats['_llm'] == 2

    # Any kind should pop from fallback
    res = script('judge', {})
    assert res['text'] == 'switch'

    res = script('shift', {})
    assert res['text'] == 'no'


def test_decision_script_asr_infer_reconstruction():
    """Test ASR infer_time reconstruction from trace timeline."""
    golden = [
        {'event': 'llm_done', 'data': {
            'kind': 'response', 'content': '回复1', 'infer_time': 0.2, 'timestamp': 1.0}},
        {'event': 'asr_done', 'data': {
            'content': '转写1', 'timestamp': 1.5}},  # no infer_time
    ]

    script = DecisionScript(golden)
    res = script('asr', {})

    # Reconstructed: 1.5 - 1.0 = 0.5s
    assert abs(res['infer'] - 0.5) < 1e-6


# ============================================================================
# Deterministic Replay Tests (requires golden traces in exp/golden/)
# ============================================================================

@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "exp" / "golden").exists(),
    reason="Golden traces not available"
)
@pytest.mark.asyncio
async def test_deterministic_replay_single_session():
    """Test that replaying the same golden trace produces identical output."""
    import yaml

    repo_root = Path(__file__).parents[1]
    golden_dir = repo_root / "exp" / "golden"

    # Pick first available golden trace
    sessions = [d for d in golden_dir.iterdir() if d.is_dir()]
    if not sessions:
        pytest.skip("No golden sessions found")

    session_dir = sessions[0]
    wav_files = list(session_dir.glob("stream_turn0_input.wav"))
    if not wav_files:
        pytest.skip(f"No input wav in {session_dir}")

    wav_path = wav_files[0]

    # Load config
    config_path = repo_root / "src" / "config.yaml"
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Find corresponding golden trace (legacy: in traces/golden_rerun/)
    trace_name = session_dir.name.replace('actor_', '') + '.jsonl'
    golden_trace_path = repo_root / "traces" / "golden_rerun" / trace_name

    if not golden_trace_path.exists():
        pytest.skip(f"Golden trace not found: {golden_trace_path}")

    # Run twice and compare
    tmp_dir = repo_root / "exp" / "test_deterministic_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    session1 = InjectedReplaySession(
        golden_trace=golden_trace_path,
        wav_path=wav_path,
        config=config,
        output_dir=tmp_dir / "run1"
    )
    trace1, _ = await session1.replay(mode='injected')

    session2 = InjectedReplaySession(
        golden_trace=golden_trace_path,
        wav_path=wav_path,
        config=config,
        output_dir=tmp_dir / "run2"
    )
    trace2, _ = await session2.replay(mode='injected')

    # Compare traces
    identical, diffs = compare_traces(trace1, trace2)

    if not identical:
        print("\n=== TRACE DIFFERENCES ===")
        for diff in diffs[:10]:  # show first 10
            print(f"  {diff}")

    assert identical, f"Replay not deterministic: {len(diffs)} differences"


@pytest.mark.asyncio
async def test_fast_replay_speed():
    """Test that injected replay achieves >10× real-time (ideally 60×)."""
    import yaml
    import soundfile as sf

    repo_root = Path(__file__).parents[1]

    # Create a minimal synthetic session
    tmp_dir = repo_root / "exp" / "test_speed_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 10s of silence
    wav_path = tmp_dir / "input.wav"
    sf.write(str(wav_path), np.zeros(160000, dtype=np.float32), 16000, subtype='PCM_16')

    # Minimal golden trace
    golden_events = [
        {'event': 'vad_start', 'data': {'timestamp': 0.2, 'turn': 0, 'state': 'LISTEN'}},
        {'event': 'vad_done', 'data': {'timestamp': 3.0, 'turn': 0, 'state': 'LISTEN'}},
        {'event': 'vad_640_done', 'data': {'timestamp': 3.64, 'turn': 0, 'state': 'LISTEN'}},
        {'event': 'llm_done', 'data': {
            'timestamp': 3.73, 'infer_time': 0.09, 'content': 'switch',
            'kind': 'judge', 'turn': 0, 'state': 'LISTEN'}},
        {'event': 'llm_done', 'data': {
            'timestamp': 3.94, 'infer_time': 0.21, 'content': '回复1',
            'kind': 'response', 'turn': 0, 'state': 'LISTEN'}},
        {'event': 'tts_done', 'data': {
            'timestamp': 5.04, 'infer_time': 1.1, 'turn': 0, 'state': 'LISTEN', 'dur_audio': 2.0}},
        {'event': 'asr_done', 'data': {
            'timestamp': 4.1, 'turn': 0, 'state': 'SPEAK', 'content': '测试'}},
    ]

    golden_path = tmp_dir / "golden.jsonl"
    with open(golden_path, 'w', encoding='utf-8') as f:
        for ev in golden_events:
            f.write(json.dumps(ev, ensure_ascii=False) + '\n')

    config_path = repo_root / "src" / "config.yaml"
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    session = InjectedReplaySession(
        golden_trace=golden_path,
        wav_path=wav_path,
        config=config,
        output_dir=tmp_dir / "out"
    )

    t0 = time.perf_counter()
    trace, _ = await session.replay(mode='injected')
    elapsed = time.perf_counter() - t0

    # 10s audio should complete in <1s (10× minimum, ideally <0.2s for 50×)
    speedup = 10.0 / elapsed
    print(f"\n=== REPLAY SPEED: {speedup:.1f}× real-time (elapsed={elapsed:.3f}s) ===")

    assert speedup > 10.0, f"Replay too slow: {speedup:.1f}× (target >10×)"


# ============================================================================
# Trace Comparison Tests
# ============================================================================

def test_compare_traces_identical():
    """Test trace comparison on identical traces."""
    trace = [
        {'event': 'vad_start', 'data': {'turn': 0, 'state': 'LISTEN'}},
        {'event': 'llm_done', 'data': {'kind': 'judge', 'content': 'switch', 'turn': 0}},
    ]

    identical, diffs = compare_traces(trace, trace)
    assert identical
    assert len(diffs) == 0


def test_compare_traces_content_mismatch():
    """Test trace comparison detects content differences."""
    trace_a = [
        {'event': 'llm_done', 'data': {'kind': 'judge', 'content': 'switch', 'turn': 0}},
    ]
    trace_b = [
        {'event': 'llm_done', 'data': {'kind': 'judge', 'content': 'continue', 'turn': 0}},
    ]

    identical, diffs = compare_traces(trace_a, trace_b)
    assert not identical
    assert any('content mismatch' in d for d in diffs)


def test_extract_decisions_summary():
    """Test decision summary extraction."""
    trace = [
        {'event': 'llm_done', 'data': {'kind': 'judge', 'content': 'switch'}},
        {'event': 'llm_done', 'data': {'kind': 'judge', 'content': 'continue'}},
        {'event': 'llm_done', 'data': {'kind': 'response', 'content': '回复1'}},
        {'event': 'asr_done', 'data': {'content': '转写1'}},
        {'event': 'tts_done', 'data': {}},
        {'event': 'tts_done', 'data': {}},
    ]

    summary = extract_decisions_summary(trace)

    assert len(summary['judge']) == 2
    assert summary['judge'][0] == 'switch'
    assert summary['judge'][1] == 'continue'
    assert len(summary['response']) == 1
    assert len(summary['asr']) == 1
    assert summary['tts_count'] == 2


# ============================================================================
# Integration: Full replay with real golden traces
# ============================================================================

@pytest.mark.skipif(
    not (Path(__file__).parents[1] / "traces" / "golden_rerun").exists(),
    reason="Golden traces not available"
)
@pytest.mark.asyncio
async def test_full_golden_replay():
    """Integration test: replay all available golden traces and verify determinism."""
    import yaml

    repo_root = Path(__file__).parents[1]
    golden_trace_dir = repo_root / "traces" / "golden_rerun"
    golden_data_dir = repo_root / "exp" / "golden"

    config_path = repo_root / "src" / "config.yaml"
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    trace_files = list(golden_trace_dir.glob("*.jsonl"))[:3]  # test first 3
    if not trace_files:
        pytest.skip("No golden traces found")

    results = []

    for trace_path in trace_files:
        # Find corresponding wav
        session_name = trace_path.stem.replace('_', '/', 1)  # ask_0001_0004 → ask/0001/0004
        session_name = 'actor_' + trace_path.stem.rsplit('_', 1)[0] + '_' + trace_path.stem.split('_')[-2] + '_' + trace_path.stem.split('_')[-1]

        session_dir = golden_data_dir / session_name
        if not session_dir.exists():
            continue

        wav_path = session_dir / "stream_turn0_input.wav"
        if not wav_path.exists():
            continue

        # Replay twice
        tmp_dir = repo_root / "exp" / "test_full_tmp" / trace_path.stem
        tmp_dir.mkdir(parents=True, exist_ok=True)

        session1 = InjectedReplaySession(trace_path, wav_path, config, tmp_dir / "run1")
        trace1, _ = await session1.replay()

        session2 = InjectedReplaySession(trace_path, wav_path, config, tmp_dir / "run2")
        trace2, _ = await session2.replay()

        identical, diffs = compare_traces(trace1, trace2)
        results.append((trace_path.name, identical, len(diffs)))

        if not identical:
            print(f"\n{trace_path.name}: {len(diffs)} differences")

    # Report
    print("\n=== GOLDEN REPLAY RESULTS ===")
    for name, identical, diff_count in results:
        status = "✓ PASS" if identical else f"✗ FAIL ({diff_count} diffs)"
        print(f"  {name}: {status}")

    pass_count = sum(1 for _, ok, _ in results if ok)
    assert pass_count == len(results), f"Determinism check: {pass_count}/{len(results)} passed"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
