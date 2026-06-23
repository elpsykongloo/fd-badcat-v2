#!/usr/bin/env python3
"""Patch the vendored index-tts-vllm checkout for Blackwell startup."""

from __future__ import annotations

import sys
from pathlib import Path


MARKER = "# fd-badcat: configurable eager mode for Blackwell CUDA graph startup"
AUDIO_MARKER = "# fd-badcat: torchaudio fallback without torchcodec"


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: patch_index_tts_vllm.py [index-tts-vllm-dir]")

    repo_dir = Path(sys.argv[1]) if len(sys.argv) == 2 else Path("model/index-tts-vllm")
    target = repo_dir / "indextts" / "infer_vllm.py"
    if not target.is_file():
        raise SystemExit(f"Index-TTS file not found: {target}")

    text = target.read_text(encoding="utf-8")
    changed = False

    if MARKER not in text:
        old_var = """        vllm_dir = os.path.join(model_dir, "gpt")
        engine_args = AsyncEngineArgs(
"""
        new_var = f"""        {MARKER}
        enforce_eager = os.getenv("INDEX_TTS_ENFORCE_EAGER", "0").strip().lower() in ("1", "true", "yes", "on")

        vllm_dir = os.path.join(model_dir, "gpt")
        engine_args = AsyncEngineArgs(
"""
        old_arg = """            gpu_memory_utilization=gpu_memory_utilization,
            # enforce_eager=True,
            async_scheduling=True,
"""
        new_arg = """            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            async_scheduling=True,
"""
        if old_var not in text or old_arg not in text:
            raise SystemExit(f"eager patch target not found in {target}")
        text = text.replace(old_var, new_var, 1).replace(old_arg, new_arg, 1)
        changed = True

    if AUDIO_MARKER not in text:
        old_import = "import torchaudio\n"
        new_import = "import torchaudio\nimport soundfile as sf\n"
        helper_anchor = "\n\ndef trim_and_pad_silence"
        helper = f"""\n\n{AUDIO_MARKER}
def _load_audio(path):
    try:
        return torchaudio.load(path)
    except ImportError as exc:
        if "TorchCodec" not in str(exc):
            raise
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(audio.T.copy()), sr
"""
        if old_import not in text or helper_anchor not in text:
            raise SystemExit(f"audio fallback patch target not found in {target}")
        text = text.replace(old_import, new_import, 1)
        text = text.replace(helper_anchor, helper + helper_anchor, 1)
        text = text.replace("torchaudio.load(ap_)", "_load_audio(ap_)")
        changed = True

    if changed:
        target.write_text(text, encoding="utf-8")
        print(f"patched: {target}")
    else:
        print(f"already patched: {target}")


if __name__ == "__main__":
    main()
