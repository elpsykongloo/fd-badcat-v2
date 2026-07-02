#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W1 replay driver: run one wav through either engine, offline, and emit a trace.

Arches:
  actor  — src/engine.py ActorEngine.run_offline (modes: realtime/injected/oracle;
           realtime here means model fns actually run + frames are wall-paced)
  legacy — src/backend_legacy.py ConversationEngine behind a FakeWebSocket
           (always wall-paced: its interval logic is wall-clock)

Models:
  --mock policy   deterministic scripted models (tonight, no GPU):
                  judge/interrupt/shift answers fixed by --judge/--interrupt/--shift,
                  response = "回复N", asr = "转写N", tts writes 0.5s of silence;
                  each with configurable sleep to emulate latency.
  --mock none     real module.py models (needs vLLM etc. — golden recording on GPU day)

The driver never sends the "end" control message: it waits for quiescence then
disconnects, so tail decision chains complete identically in both arches
(production keeps frontend semantics; see docs/w1_equivalence.md).
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import yaml  # noqa: E402


# ---------------------------------------------------------------------------
# deterministic mock models
# ---------------------------------------------------------------------------
class MockModels:
    """Kind is recovered from the system prompt (works for both engines)."""

    def __init__(self, prompts, judge="switch", interrupt="continue", shift="no",
                 llm_sleep=0.3, asr_sleep=0.1, tts_sleep=0.2):
        self._by_prompt = {
            prompts.get("judge", ""): ("judge", judge),
            prompts.get("interrupt", ""): ("interrupt", interrupt),
            prompts.get("shift", ""): ("shift", shift),
            prompts.get("response", ""): ("response", None),
            prompts.get("shift_s", ""): ("shift_re", None),
        }
        self.llm_sleep = llm_sleep
        self.asr_sleep = asr_sleep
        self.tts_sleep = tts_sleep
        self.counters = {}

    def _kind_of(self, messages):
        sys_prompt = messages[0]["content"] if messages else ""
        kind, fixed = self._by_prompt.get(sys_prompt, ("response", None))
        return kind, fixed

    def _n(self, kind):
        self.counters[kind] = self.counters.get(kind, 0) + 1
        return self.counters[kind]

    def llm(self, messages):
        kind, fixed = self._kind_of(messages)
        n = self._n(kind)
        time.sleep(self.llm_sleep)
        if fixed is not None:
            return fixed
        return f"回复{n}"

    def asr(self, path):
        n = self._n("asr")
        time.sleep(self.asr_sleep)
        return f"转写{n}"

    def tts(self, text, path):
        self._n("tts")
        time.sleep(self.tts_sleep)
        sf.write(str(path), np.zeros(8000, dtype=np.float32), 16000, subtype="PCM_16")
        return str(path)


# ---------------------------------------------------------------------------
# scripted VAD (precomputed by scripts/extract_vad_events.py; lets equivalence
# and freeze runs skip torch entirely — the dev container has a 2GB limit)
# ---------------------------------------------------------------------------
def install_light_stubs():
    """Stub torch/silero_vad BEFORE engine imports. Only valid together with
    a scripted VAD (apply_vad_script)."""
    import types
    if "torch" not in sys.modules:
        t = types.ModuleType("torch")
        t.from_numpy = lambda x: x
        sys.modules["torch"] = t
    if "silero_vad" not in sys.modules:
        sv = types.ModuleType("silero_vad")

        class _DummyIter:
            def __init__(self, *a, **k):
                pass

            def reset_states(self):
                pass

            def __call__(self, *a, **k):
                return None

        sv.load_silero_vad = lambda *a, **k: None
        sv.VADIterator = _DummyIter
        sys.modules["silero_vad"] = sv


def load_vad_script(wav_path):
    p = Path("traces/vad_scripts") / f"{Path(wav_path).stem}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {int(k): v for k, v in d["events"].items()}


def apply_vad_script(engine_obj, events, probe=None):
    """Replace an engine instance's per-frame VAD with the recorded event script.
    probe: optional list collecting perf_counter() at every VAD call — the
    symmetric probe point for the perception-freeze A/B (D3.4)."""
    state = {"i": 0}

    def detect(chunk):
        if probe is not None:
            probe.append(time.perf_counter())
        state["i"] += 1
        return events.get(state["i"])

    engine_obj.detect_vad_frame = detect


# ---------------------------------------------------------------------------
# legacy engine driving (FakeWebSocket + monkeypatch)
# ---------------------------------------------------------------------------
class FakeWebSocket:
    """Feeds paced frames like uvicorn would; records everything sent back."""

    def __init__(self, wav_path, chunk=256, paced=True):
        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        assert sr == 16000, f"{wav_path}: {sr} != 16000"
        self.frames = [data[i:i + chunk] for i in range(0, len(data), chunk)]
        if len(self.frames[-1]) < chunk:
            self.frames[-1] = np.pad(self.frames[-1], (0, chunk - len(self.frames[-1])))
        self.paced = paced
        self.chunk_sec = chunk / sr
        self._i = 0
        self.trace = []
        self.bytes_out = 0
        self.start_wall = None
        self._drained = asyncio.Event()

    async def receive(self):
        if self.start_wall is None:
            self.start_wall = time.time()
            self._t0 = time.perf_counter()
        if self._i < len(self.frames):
            frame = self.frames[self._i]
            self._i += 1
            if self.paced:
                # absolute schedule: no per-frame sleep-overhead accumulation,
                # matches production arrival timing (uvicorn receives in realtime)
                target = self._t0 + self._i * self.chunk_sec
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            return {"type": "websocket.receive", "bytes": frame.astype(np.float32).tobytes()}
        # quiesce instead of "end": wait so tail decision chains can finish
        self._drained.set()
        await asyncio.sleep(3600)  # driver cancels us via disconnect timeout
        return {"type": "websocket.disconnect"}

    async def send_text(self, s):
        self.trace.append(json.loads(s))

    async def send_bytes(self, b):
        self.bytes_out += len(b)


async def run_legacy(wav, mocks, out_dir, grace=3.0, vad_script=None, vad_probe=None):
    if mocks is not None and "module" not in sys.modules:
        # inject mocks as the `module` module BEFORE backend_legacy imports it:
        # skips sherpa_onnx/torchaudio entirely (matters on the 2GB dev container)
        import types
        fake = types.ModuleType("module")
        fake.asr, fake.llm_qwen3o, fake.tts = mocks.asr, mocks.llm, mocks.tts
        sys.modules["module"] = fake
    import backend_legacy
    if mocks is not None:
        backend_legacy.llm_qwen3o = mocks.llm
        backend_legacy.asr = mocks.asr
        backend_legacy.tts = mocks.tts

    with open(SRC / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ws = FakeWebSocket(wav)
    eng = backend_legacy.ConversationEngine(websocket=ws, prompts=cfg["prompts"], delay=cfg["time"])
    if vad_script is not None:
        apply_vad_script(eng, vad_script, probe=vad_probe)
    eng.output_dir = Path(out_dir)
    eng.output_dir.mkdir(parents=True, exist_ok=True)

    task = asyncio.create_task(eng.run_realtime(ws))
    await ws._drained.wait()
    # quiescence: no new trace events for `grace` seconds
    last_len, quiet_since = len(ws.trace), time.time()
    while time.time() - quiet_since < grace:
        await asyncio.sleep(0.1)
        if len(ws.trace) != last_len:
            last_len, quiet_since = len(ws.trace), time.time()
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
    return ws.trace, eng


async def run_actor(wav, mocks, out_dir, mode="realtime", script=None, vad_script=None,
                    vad_probe=None):
    from engine import ActorEngine, frames_from_wav
    with open(SRC / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    eng = ActorEngine(
        prompts=cfg["prompts"], delay=cfg["time"],
        llm_cfg=cfg.get("llm", {}), engine_cfg=cfg.get("engine", {}),
        llm_fn=mocks.llm if mocks else None,
        asr_fn=mocks.asr if mocks else None,
        tts_fn=mocks.tts if mocks else None,
        replay_mode=mode, decision_script=script,
    )
    if vad_script is not None:
        apply_vad_script(eng, vad_script, probe=vad_probe)
    eng.output_dir = Path(out_dir)
    eng.output_dir.mkdir(parents=True, exist_ok=True)
    frames = frames_from_wav(wav)
    await eng.run_offline(frames, paced=(mode == "realtime"))
    return eng.trace, eng


class RecordedScript:
    """Replays a golden trace's llm/asr/tts results by per-kind order (injected mode)."""

    def __init__(self, golden_events, tts_dur_default=0.5):
        self.q = {}
        last_llm_ts = None
        for ev in golden_events:
            data = ev.get("data", {})
            ts = data.get("timestamp")
            if ev.get("event") == "llm_done":
                kind = data.get("kind")  # present in new-engine traces
                self.q.setdefault(kind or "_llm", []).append(
                    {"text": data.get("content", ""), "infer": data.get("infer_time", 0.0)})
                last_llm_ts = ts
            elif ev.get("event") == "asr_done":
                # legacy traces carry no infer_time for asr; reconstruct the real
                # duration from the trace timeline (dispatch ≈ the llm_done that
                # triggered the answer chain) so concurrent completion ORDER is
                # faithfully reproduced in injected replay
                infer = data.get("infer_time")
                if infer is None and ts is not None and last_llm_ts is not None:
                    infer = max(0.05, round(ts - last_llm_ts, 3))
                self.q.setdefault("asr", []).append(
                    {"text": data.get("content", ""), "infer": infer if infer is not None else 0.1})
            elif ev.get("event") == "tts_done":
                self.q.setdefault("tts", []).append(
                    {"infer": data.get("infer_time", 0.2),
                     "dur_audio": data.get("dur_audio", tts_dur_default)})
        self.tts_dur_default = tts_dur_default

    def __call__(self, kind, meta):
        seq = self.q.get(kind) or self.q.get("_llm") or []
        if seq:
            return seq.pop(0)
        return {"text": "", "infer": 0.0, "dur_audio": self.tts_dur_default}


def save_trace(trace, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in trace:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--arch", choices=["actor", "legacy"], default="actor")
    ap.add_argument("--mode", choices=["realtime", "injected", "oracle"], default="realtime")
    ap.add_argument("--mock", choices=["policy", "none"], default="policy")
    ap.add_argument("--golden", help="golden trace jsonl for injected mode script")
    ap.add_argument("--trace", required=True, help="output trace jsonl")
    ap.add_argument("--out-dir", default="exp/replay_tmp")
    ap.add_argument("--judge", default="switch")
    ap.add_argument("--interrupt", default="continue")
    ap.add_argument("--shift", default="no")
    ap.add_argument("--llm-sleep", type=float, default=0.3)
    args = ap.parse_args()

    with open(SRC / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mocks = None
    if args.mock == "policy":
        mocks = MockModels(cfg["prompts"], judge=args.judge, interrupt=args.interrupt,
                           shift=args.shift, llm_sleep=args.llm_sleep)

    if args.arch == "legacy":
        trace, eng = asyncio.run(run_legacy(args.wav, mocks, args.out_dir))
    else:
        script = None
        if args.mode in ("injected", "oracle"):
            golden = [json.loads(l) for l in open(args.golden, encoding="utf-8") if l.strip()]
            script = RecordedScript(golden)
        trace, eng = asyncio.run(run_actor(args.wav, mocks, args.out_dir,
                                           mode=args.mode, script=script))
        stats = eng.freeze_stats()
        if stats:
            print(f"[freeze] {stats}")

    save_trace(trace, args.trace)
    print(f"saved {len(trace)} events -> {args.trace}")


if __name__ == "__main__":
    main()
