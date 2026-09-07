"""Negotiated, bounded mic/render-reference packets and conservative echo rejection.

The browser records both channels in the same AudioWorklet callback. This is not
a replacement for browser AEC or speaker verification: only strongly correlated,
echo-dominated windows are rejected; double-talk/uncertain windows pass through.
"""
import struct
from collections import deque

import numpy as np

INPUT_PROTOCOL = "pcm16.ref.v1"
INPUT_HEADER = struct.Struct("<4sIII")
INPUT_SAMPLES = 256


def decode_input_packet(raw, expected_seq):
    if len(raw) != INPUT_HEADER.size + INPUT_SAMPLES * 4:
        raise ValueError("Invalid microphone/reference packet size")
    magic, seq, count, rate = INPUT_HEADER.unpack_from(raw)
    if magic != b"FDM1" or seq != expected_seq or count != INPUT_SAMPLES or rate != 16000:
        raise ValueError("Invalid microphone/reference packet header or sequence")
    data = np.frombuffer(raw, dtype="<i2", offset=INPUT_HEADER.size).reshape(-1, 2)
    return data[:, 0].astype(np.float32) / 32768, data[:, 1].astype(np.float32) / 32768


class EchoEvidence:
    """128 ms waveform evidence; up to 480 ms acoustic/processing delay.

    Correlation alone is insufficient: rejection additionally requires little
    unexplained energy. Thresholds are conservative engineering defaults, not a
    calibrated probability or a guarantee for every speaker/microphone pair.
    """
    def __init__(self):
        self.mic = deque(maxlen=8)
        self.ref = deque(maxlen=38)
        self.frames = 0
        self.last = {"echo_only": False, "correlation": 0., "residual_ratio": 1., "delay_ms": 0.}

    def process(self, mic, reference):
        self.frames += 1
        # 4 kHz is sufficient for this conservative waveform test. Original
        # 16 kHz mic audio is retained for the recognizers.
        self.mic.append(np.asarray(mic).reshape(-1, 4).mean(axis=1))
        self.ref.append(np.asarray(reference).reshape(-1, 4).mean(axis=1))
        if len(self.mic) < 8 or len(self.ref) < 8:
            return dict(self.last)
        raw_m, raw_r = np.concatenate(self.mic), np.concatenate(self.ref)
        # Differencing removes DC/slow gain variation and reduces spurious
        # matches between unrelated voiced segments.
        m, r = np.diff(raw_m), np.diff(raw_r)
        me = float(m @ m)
        if me < 1e-10 or float(r @ r) < 1e-10:
            self.last = {"echo_only": False, "correlation": 0., "residual_ratio": 1., "delay_ms": 0.}
            return dict(self.last)
        dots = np.correlate(r, m, mode="valid")
        energy = np.concatenate(([0.], np.cumsum(r.astype(np.float64) ** 2)))
        re = energy[len(m):] - energy[:-len(m)]
        corr = np.abs(dots) / np.sqrt(np.maximum(re * me, 1e-20))
        best = int(np.argmax(corr))
        score = min(1., float(corr[best]))
        # Verify residual in the ORIGINAL (DC-centred) band, not 1-corr**2:
        # differencing can conceal low-frequency near-end/double-talk energy.
        aligned = raw_r[best:best + len(raw_m)].astype(np.float64)
        aligned -= aligned.mean()
        observed = raw_m.astype(np.float64) - raw_m.mean()
        gain = float(observed @ aligned) / max(float(aligned @ aligned), 1e-20)
        error = observed - gain * aligned
        residual = float(error @ error) / max(float(observed @ observed), 1e-20)
        self.last = {"echo_only": score >= .92 and residual <= .16,
                     "correlation": round(score, 4), "residual_ratio": round(residual, 4),
                     "delay_ms": round((len(dots) - 1 - best) / 4, 1)}
        return dict(self.last)
