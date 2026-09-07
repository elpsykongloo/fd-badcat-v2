"""One startup warmup in an isolated context; no user session/history is created."""
import asyncio
import io
import json
import time
from pathlib import Path


async def warmup(prompts):
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import load_silero_vad, VADIterator
    import module
    from messages import build_audio_content
    from control_labels import parse_label

    started = time.perf_counter()
    fixture = Path(__file__).resolve().parents[1] / "exp/streaming_demo/synthetic_question.wav"

    def warm_cpu():
        audio, rate = sf.read(fixture, dtype="float32")
        vad = VADIterator(load_silero_vad(), sampling_rate=16000)
        vad(torch.zeros(512), return_seconds=True)
        transcript = module.asr(str(fixture))
        if not transcript:
            raise RuntimeError("ASR warmup returned no transcript")
        return audio, rate, transcript

    async def run():
        audio, rate, transcript = await asyncio.to_thread(warm_cpu)
        content = build_audio_content(audio, rate, "audio_url")
        judge = await asyncio.to_thread(module.llm_qwen3o_strict, [
            {"role": "system", "content": prompts["judge"]}, {"role": "user", "content": [content]}])
        if parse_label("judge", judge) is None:
            raise RuntimeError("Control model warmup returned an invalid label")
        # Exercise the same SSE text and native PCM decoder as the real demo.
        pieces = [p async for p in module.llm_qwen3o_stream([
            {"role": "system", "content": prompts["response"]},
            {"role": "user", "content": "用中文和英文各打一个简短招呼。"}])]
        if not "".join(pieces).strip():
            raise RuntimeError("Text SSE warmup returned no text")
        chunks = [p async for p in module.tts_omni_stream("你好。Hello, how are you?")]
        if not chunks or len({p.sample_rate for p in chunks}) != 1:
            raise RuntimeError("TTS warmup returned invalid PCM")
        # Decode bilingual speech with the selected ASR, not just model loading.
        pcm = np.frombuffer(b"".join(p.pcm for p in chunks), dtype="<i2").astype(np.float32) / 32768
        wav = io.BytesIO()
        sf.write(wav, pcm, chunks[0].sample_rate, format="WAV", subtype="PCM_16")
        wav.seek(0)
        bilingual = await asyncio.to_thread(module.asr, wav)
        if not bilingual:
            raise RuntimeError("Bilingual ASR warmup returned no text")
        return transcript, bilingual, chunks
    transcript, bilingual, chunks = await asyncio.wait_for(run(), 90)
    result = {"event": "demo_warmup_done", "asr": module.ASR_BACKEND,
              "asr_provider": module.ASR_PROVIDER, "transcript": transcript,
              "bilingual_transcript": bilingual, "tts_chunks": len(chunks),
              "elapsed_s": round(time.perf_counter() - started, 3)}
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result
