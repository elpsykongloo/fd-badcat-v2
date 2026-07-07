#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audio clock driver for deterministic replay (W1 injected mode).

The audio clock is the single source of time for all interval logic:
  t_audio = cumulative_samples / SAMPLE_RATE

In injected mode, model inference times come from golden traces, and results
are delivered when the audio clock advances by the recorded infer_time.
This enables:
  - Deterministic replay (same trace → same output, always)
  - Fast evaluation (60× real-time: no actual model waits)
  - Perfect isolation for concurrent evaluation
"""
import heapq
from dataclasses import dataclass
from typing import Optional, Callable, Any


SAMPLE_RATE = 16000
WINDOW_SIZE = 256


@dataclass
class ScheduledResult:
    """A model result scheduled for delivery at a specific audio clock time."""
    due_t_audio: float
    seq: int  # tie-breaker for heap ordering
    kind: str  # judge|shift|interrupt|response|shift_re|asr|tts
    result: dict  # {text, infer, dur_audio, wav_path, ...}
    metadata: dict  # {gen, epoch, turn, ...}


class AudioClock:
    """Audio clock driver for deterministic replay.

    Maintains the current audio clock time and a priority queue of scheduled
    model results. When the clock advances (via tick or advance), due results
    are delivered via the callback.
    """

    def __init__(self, delivery_callback: Optional[Callable] = None):
        """
        Args:
            delivery_callback: async function(kind, result, metadata) called when
                             a scheduled result becomes due. If None, results
                             accumulate in .delivered for inspection.
        """
        self.t_audio = 0.0
        self._scheduled = []  # min-heap of (due_t, seq, ScheduledResult)
        self._seq = 0
        self._delivery_callback = delivery_callback
        self.delivered = []  # fallback collector when no callback

    def tick(self, samples: int) -> float:
        """Advance the audio clock by the given number of samples.

        Args:
            samples: number of audio samples (e.g., 256 for one frame)

        Returns:
            new t_audio value
        """
        self.t_audio += samples / SAMPLE_RATE
        return self.t_audio

    def advance(self, t_audio: float) -> float:
        """Jump the audio clock to an absolute time (fast-forward for tail drain).

        Args:
            t_audio: target audio clock time (must be >= current)

        Returns:
            new t_audio value
        """
        if t_audio < self.t_audio:
            raise ValueError(f"Cannot rewind audio clock: {self.t_audio} -> {t_audio}")
        self.t_audio = t_audio
        return self.t_audio

    def schedule(self, kind: str, result: dict, metadata: dict,
                 infer_time: float) -> None:
        """Schedule a model result for delivery after infer_time elapses on the audio clock.

        Args:
            kind: decision kind (judge, shift, interrupt, response, shift_re, asr, tts)
            result: result dict with text, infer, dur_audio, wav_path, etc.
            metadata: engine metadata (gen, epoch, turn, add_to_history, etc.)
            infer_time: model inference time in seconds (from golden trace)
        """
        due_t = self.t_audio + infer_time
        self._seq += 1
        item = ScheduledResult(
            due_t_audio=due_t,
            seq=self._seq,
            kind=kind,
            result=result,
            metadata=metadata
        )
        heapq.heappush(self._scheduled, (due_t, self._seq, item))

    async def drain_due(self) -> int:
        """Deliver all results whose due time has been reached.

        Returns:
            number of results delivered
        """
        count = 0
        while self._scheduled and self._scheduled[0][0] <= self.t_audio:
            _, _, item = heapq.heappop(self._scheduled)
            if self._delivery_callback:
                await self._delivery_callback(item.kind, item.result, item.metadata)
            else:
                self.delivered.append((item.kind, item.result, item.metadata))
            count += 1
        return count

    def pending_count(self) -> int:
        """Return the number of scheduled results not yet due."""
        return len(self._scheduled)

    def next_due_time(self) -> Optional[float]:
        """Return the audio clock time of the next scheduled result, or None."""
        if not self._scheduled:
            return None
        return self._scheduled[0][0]

    def reset(self) -> None:
        """Reset the clock and clear all scheduled results."""
        self.t_audio = 0.0
        self._scheduled = []
        self._seq = 0
        self.delivered = []


class AudioClockFrameGenerator:
    """Generates audio frames with audio-clock timestamps for replay."""

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 window_size: int = WINDOW_SIZE):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.cumulative_samples = 0
        self.seq = 0

    def chunk_to_frames(self, pcm_array) -> list:
        """Convert a PCM array into audio-clock-stamped frames.

        Args:
            pcm_array: numpy array of float32 samples

        Returns:
            list of dicts with {seq, t_audio, pcm}
        """
        import numpy as np
        frames = []
        for i in range(0, len(pcm_array), self.window_size):
            chunk = pcm_array[i:i + self.window_size]
            if len(chunk) < self.window_size:
                chunk = np.pad(chunk, (0, self.window_size - len(chunk)))
            self.seq += 1
            self.cumulative_samples += len(chunk)
            frames.append({
                'seq': self.seq,
                't_audio': self.cumulative_samples / self.sample_rate,
                'pcm': chunk
            })
        return frames

    def reset(self) -> None:
        """Reset the frame generator state."""
        self.cumulative_samples = 0
        self.seq = 0


def validate_audio_clock_monotonicity(events: list) -> tuple[bool, str]:
    """Validate that a trace has monotonically increasing audio clock times.

    Args:
        events: list of trace events (dicts with 'data': {'t_audio': ...})

    Returns:
        (is_valid, error_message)
    """
    last_t = -1.0
    for i, ev in enumerate(events):
        t = ev.get('data', {}).get('t_audio')
        if t is None:
            continue
        if t < last_t:
            return False, f"Event {i}: t_audio {t} < previous {last_t}"
        last_t = t
    return True, ""


def audio_clock_stats(events: list) -> dict:
    """Extract audio clock statistics from a trace.

    Args:
        events: list of trace events

    Returns:
        dict with duration, frame_count, event_count, etc.
    """
    audio_times = []
    for ev in events:
        t = ev.get('data', {}).get('t_audio')
        if t is not None:
            audio_times.append(t)

    if not audio_times:
        return {'duration': 0.0, 'events': 0}

    return {
        'duration': max(audio_times),
        'events': len(audio_times),
        'min_t': min(audio_times),
        'max_t': max(audio_times),
    }
