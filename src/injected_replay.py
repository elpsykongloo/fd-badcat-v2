#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Injected replay system for deterministic evaluation (W1 fast track).

Reads golden traces and replays them through the engine with recorded inference
times, achieving:
  - Deterministic output (same input trace → same output, always)
  - 60× real-time speed (no actual model execution)
  - Perfect concurrency isolation (each session independent)

Architecture:
  Golden trace → DecisionScript → AudioClock schedules results →
  Engine processes at audio-clock pace → New trace output

The decision script supplies (text, infer_time, dur_audio) from the golden
trace; the audio clock ensures results arrive exactly when t_audio advances
by infer_time, preserving all timing-dependent state transitions.
"""
import json
from pathlib import Path
from typing import Optional, Union


class DecisionScript:
    """Replays golden trace decisions by kind and order.

    Parses a golden trace (jsonl) and builds per-kind queues of decisions.
    When the engine calls the script (via dispatch_llm/asr/tts in injected mode),
    the script returns the next recorded result for that kind.
    """

    def __init__(self, golden_trace: Union[str, Path, list],
                 tts_dur_default: float = 0.5):
        """
        Args:
            golden_trace: path to .jsonl golden trace, or list of parsed events
            tts_dur_default: fallback TTS duration when not in trace
        """
        if isinstance(golden_trace, (str, Path)):
            golden_trace = self._load_trace(golden_trace)

        self.queues = {}
        self.tts_dur_default = tts_dur_default
        self._parse_trace(golden_trace)

    def _load_trace(self, path: Union[str, Path]) -> list:
        """Load a jsonl trace file."""
        events = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def _parse_trace(self, events: list) -> None:
        """Parse golden trace events into per-kind decision queues."""
        last_llm_ts = None

        for ev in events:
            event_type = ev.get('event')
            data = ev.get('data', {})
            ts = data.get('timestamp')

            if event_type == 'llm_done':
                kind = data.get('kind')  # present in new-engine traces
                # legacy traces lack 'kind': all go to '_llm' fallback queue
                queue_key = kind if kind else '_llm'
                self.queues.setdefault(queue_key, []).append({
                    'text': data.get('content', ''),
                    'infer': data.get('infer_time', 0.0),
                    'prompt': data.get('prompt', []),
                })
                last_llm_ts = ts

            elif event_type == 'asr_done':
                # Reconstruct infer_time from trace timeline for legacy traces
                # (dispatch ≈ last_llm_ts; completion = current ts)
                infer = data.get('infer_time')
                if infer is None and ts is not None and last_llm_ts is not None:
                    infer = max(0.05, round(ts - last_llm_ts, 3))
                self.queues.setdefault('asr', []).append({
                    'text': data.get('content', ''),
                    'infer': infer if infer is not None else 0.1,
                })

            elif event_type == 'tts_done':
                self.queues.setdefault('tts', []).append({
                    'infer': data.get('infer_time', 0.2),
                    'dur_audio': data.get('dur_audio', self.tts_dur_default),
                    'wav_path': '',  # injected mode doesn't need actual files
                })

    def __call__(self, kind: str, meta: dict) -> dict:
        """Called by engine dispatch in injected mode.

        Args:
            kind: decision kind (judge, shift, interrupt, response, shift_re, asr, tts)
            meta: engine metadata (turn, t_audio, epoch, prompt, text, ...)

        Returns:
            dict with {text, infer, dur_audio, wav_path, ...}
        """
        # Try kind-specific queue first, then fallback to '_llm' for legacy traces
        queue = self.queues.get(kind) or self.queues.get('_llm') or []

        if queue:
            return queue.pop(0)

        # Exhausted: return safe defaults
        if kind == 'tts':
            return {'text': '', 'infer': 0.0, 'dur_audio': self.tts_dur_default, 'wav_path': ''}
        else:
            return {'text': '', 'infer': 0.0}

    def stats(self) -> dict:
        """Return statistics about the loaded script."""
        return {k: len(v) for k, v in self.queues.items()}


class InjectedReplaySession:
    """High-level injected replay driver (convenience wrapper).

    Combines DecisionScript + ActorEngine + offline driving for one-shot replay.
    """

    def __init__(self, golden_trace: Union[str, Path, list],
                 wav_path: Union[str, Path],
                 config: Optional[dict] = None,
                 output_dir: Optional[Union[str, Path]] = None):
        """
        Args:
            golden_trace: path to golden .jsonl or parsed events
            wav_path: input audio file path
            config: engine config dict (prompts, delay, llm, engine sections)
            output_dir: where to write trace.jsonl and audio artifacts
        """
        self.golden_trace = golden_trace
        self.wav_path = Path(wav_path)
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else None

        self.script = DecisionScript(golden_trace)
        self.engine = None
        self.trace = []

    async def replay(self, mode: str = 'injected') -> tuple[list, object]:
        """Run the replay and return (trace_events, engine_instance).

        Args:
            mode: 'injected' (recorded infer times) or 'oracle' (zero latency)

        Returns:
            (trace_events, engine)
        """
        from engine import ActorEngine, frames_from_wav

        prompts = self.config.get('prompts', {})
        delay = self.config.get('delay', {})
        llm_cfg = self.config.get('llm', {})
        engine_cfg = self.config.get('engine', {})

        self.engine = ActorEngine(
            prompts=prompts,
            delay=delay,
            llm_cfg=llm_cfg,
            engine_cfg=engine_cfg,
            replay_mode=mode,
            decision_script=self.script,
        )

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.engine.output_dir = self.output_dir

        frames = frames_from_wav(self.wav_path)
        await self.engine.run_offline(frames, paced=False)

        self.trace = self.engine.trace
        return self.trace, self.engine

    def save_trace(self, path: Union[str, Path]) -> None:
        """Write the replay trace to a jsonl file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            for ev in self.trace:
                f.write(json.dumps(ev, ensure_ascii=False) + '\n')


def compare_traces(trace_a: list, trace_b: list,
                   ignore_timestamps: bool = True,
                   ignore_prompts: bool = True) -> tuple[bool, list]:
    """Compare two traces for determinism verification.

    Args:
        trace_a, trace_b: lists of trace events
        ignore_timestamps: skip wall-clock timestamp comparisons
        ignore_prompts: skip prompt snapshot comparisons (audio blocks differ)

    Returns:
        (is_identical, differences)
    """
    if len(trace_a) != len(trace_b):
        return False, [f"Length mismatch: {len(trace_a)} vs {len(trace_b)}"]

    diffs = []

    for i, (a, b) in enumerate(zip(trace_a, trace_b)):
        ev_type = a.get('event')
        if ev_type != b.get('event'):
            diffs.append(f"Event {i}: type mismatch {ev_type} vs {b.get('event')}")
            continue

        data_a = a.get('data', {})
        data_b = b.get('data', {})

        # Compare critical fields per event type
        if ev_type == 'llm_done':
            if data_a.get('content') != data_b.get('content'):
                diffs.append(f"Event {i} (llm_done): content mismatch")
            if data_a.get('turn') != data_b.get('turn'):
                diffs.append(f"Event {i} (llm_done): turn mismatch")
            if data_a.get('state') != data_b.get('state'):
                diffs.append(f"Event {i} (llm_done): state mismatch")

        elif ev_type == 'asr_done':
            if data_a.get('content') != data_b.get('content'):
                diffs.append(f"Event {i} (asr_done): content mismatch")

        elif ev_type == 'tts_done':
            if data_a.get('turn') != data_b.get('turn'):
                diffs.append(f"Event {i} (tts_done): turn mismatch")

        elif ev_type in ('vad_start', 'vad_done', 'vad_640_done',
                         'long_interrupt', 'shot_interrupt'):
            if data_a.get('turn') != data_b.get('turn'):
                diffs.append(f"Event {i} ({ev_type}): turn mismatch")
            if data_a.get('state') != data_b.get('state'):
                diffs.append(f"Event {i} ({ev_type}): state mismatch")

        # t_audio must be identical in injected mode (audio clock determinism)
        if not ignore_timestamps:
            if abs(data_a.get('t_audio', 0) - data_b.get('t_audio', 0)) > 0.001:
                diffs.append(f"Event {i}: t_audio mismatch")

    return len(diffs) == 0, diffs


def load_golden_trace(path: Union[str, Path]) -> list:
    """Utility: load a golden trace .jsonl file."""
    events = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def extract_decisions_summary(trace: list) -> dict:
    """Extract a summary of decisions from a trace for quick comparison."""
    summary = {
        'judge': [],
        'shift': [],
        'interrupt': [],
        'response': [],
        'shift_re': [],
        'asr': [],
        'tts_count': 0,
    }

    for ev in trace:
        if ev.get('event') == 'llm_done':
            data = ev.get('data', {})
            kind = data.get('kind')
            content = data.get('content', '')

            if kind in summary:
                summary[kind].append(content)
            elif kind is None:
                # legacy trace: infer kind from content
                low = content.lower()
                if 'switch' in low or 'continue' in low:
                    summary['judge'].append(content)
                elif 'yes' in low or 'no' in low:
                    summary['shift'].append(content)
                else:
                    summary['response'].append(content)

        elif ev.get('event') == 'asr_done':
            summary['asr'].append(ev.get('data', {}).get('content', ''))

        elif ev.get('event') == 'tts_done':
            summary['tts_count'] += 1

    return summary
