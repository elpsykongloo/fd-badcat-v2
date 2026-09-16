"""Offline scoring guardrails; no model download or GPU required."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_demo_speaker_embeddings import load_manifest, normalize_embedding, read_audio, score_pairs


@pytest.mark.parametrize("value", [np.zeros(256), np.ones(128),
                                  np.full(256, np.nan), np.full(256, np.inf)])
def test_invalid_embeddings_cannot_produce_scores(value):
    with pytest.raises(ValueError, match="embedding"):
        normalize_embedding(value)


def test_cosine_direction_and_dependent_denominators():
    x = np.zeros(256); x[0] = 1
    y = np.zeros(256); y[1] = 1
    audio = {key: {"text": "same text"} for key in "abcd"}
    vectors = dict(zip("abcd", map(normalize_embedding, [x, x * 4, y, -x])))
    pairs = [{"group": "repeat", "reference": "a", "candidate": k} for k in "bcd"]
    scored, groups = score_pairs(audio, pairs, vectors)
    assert [p["cosine"] for p in scored] == [1, 0, -1]
    assert [p["distance"] for p in scored] == [0, 1, 2]
    assert groups["repeat"]["pairs"] == 3
    assert groups["repeat"]["unique_texts"] == 1
    assert groups["repeat"]["unique_audio_ids"] == 4


@pytest.mark.parametrize("bad", ["duplicate", "unknown", "self", "empty"])
def test_invalid_manifest_fails_before_inference(tmp_path, bad):
    manifest = {"audio": [{"id": "a", "path": "a.wav"}, {"id": "b", "path": "b.wav"}],
                "pairs": [{"group": "repeat", "reference": "a", "candidate": "b"}]}
    if bad == "duplicate":
        manifest["audio"][1]["id"] = "a"
    elif bad == "unknown":
        manifest["pairs"][0]["candidate"] = "missing"
    elif bad == "self":
        manifest["pairs"][0]["candidate"] = "a"
    else:
        manifest["pairs"] = []
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_resampling_preserves_duration_and_pitch(tmp_path):
    t = np.arange(24000, dtype=np.float32) / 24000
    path = tmp_path / "tone.wav"
    sf.write(path, .1 * np.sin(2 * np.pi * 200 * t), 24000)
    output = read_audio(path)
    assert len(output) == 16000
    assert np.argmax(abs(np.fft.rfft(output))) == 200


@pytest.mark.parametrize("shape", [(16000,), (4000,), (16000, 2)])
def test_silence_short_or_stereo_audio_is_not_silently_scored(tmp_path, shape):
    path = tmp_path / "invalid.wav"
    sf.write(path, np.zeros(shape), 16000)
    with pytest.raises(ValueError):
        read_audio(path)
