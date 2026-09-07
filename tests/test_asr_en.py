"""Fast SenseVoice adapter tests; no model weights or synthetic benchmark."""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import module as audio_module  # noqa: E402


class _FakeStream:
    def __init__(self, text):
        self.result = type("Result", (), {"text": text})()
        self.accepted = None

    def accept_waveform(self, sample_rate, audio):
        self.accepted = (sample_rate, audio)


class _FakeRecognizer:
    def __init__(self, text):
        self.stream = _FakeStream(text)
        self.decoded = False

    def create_stream(self):
        return self.stream

    def decode_stream(self, stream):
        assert stream is self.stream
        self.decoded = True


def test_sensevoice_asr_adapter_strips_tags(tmp_path, monkeypatch):
    """Exercise WAV input, recognizer wiring and bilingual tag cleanup."""
    wav = tmp_path / "input.wav"
    sf.write(wav, np.zeros(160, dtype=np.float32), 16000, subtype="PCM_16")
    recognizer = _FakeRecognizer("<|en|><|HAPPY|> hello world <|nospeech|>")

    monkeypatch.setattr(audio_module, "ASR_BACKEND", "sensevoice")
    monkeypatch.setattr(audio_module, "_get_asr_model", lambda: recognizer)
    monkeypatch.setattr(
        audio_module, "_mono_16k", lambda audio, sample_rate: np.asarray(audio))

    assert audio_module.asr(wav) == "hello world"
    assert recognizer.decoded
    sample_rate, accepted = recognizer.stream.accepted
    assert sample_rate == 16000
    assert accepted.shape == (160,)


def test_sensevoice_tag_stripping_contract():
    cases = [
        ("<|zh|> 你好世界", "你好世界"),
        ("<|en|> hello world <|nospeech|>", "hello world"),
        ("no tags here", "no tags here"),
        ("<|zh|><|HAPPY|> 开心 <|nospeech|>", "开心"),
    ]
    for raw, expected in cases:
        assert audio_module.strip_sensevoice_tags(raw) == expected
