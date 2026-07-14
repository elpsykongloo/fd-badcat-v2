# TACT × FDB-v3 — Week-1 Integration Guide

This is the concrete how-to for the first experiment: hook `fd-badcat` into the
FDB-v3 harness, reproduce the cascaded trilemma numbers, stand up the **blocking
tool-call baseline**, and land the **async act track + extended decision space
`{launch/patch/cancel/commit}`** with a **deterministic pending-set**.

Everything here is grounded in the actual code of both repos
(`yu-haoyuan/fd-badcat` and `DanielLin94144/Full-Duplex-Bench`, folder `v3/`).

---

## 0. The three things to know before you start

1. **The harness runs on LiveKit Cloud + OpenAI keys.** Inference streams each
   `input.wav` through a LiveKit room (`run_tool_benchmark.py` → `livekit_inference.py`),
   records the agent's audio, NeMo-ASRs it (`nvidia/parakeet-tdt-0.6b-v2`), and
   captures tool calls. **But the three evaluators do NOT need LiveKit** — they read
   `result_{provider}.json` off disk. That gap is what we exploit for fast iteration.

2. **Your real (local) system is wired for Chinese in two places — use the
   pre-April-1 local commit as the reference, not the April-8 "代码检查" cloud build.**
   The stable local tree (e.g. `6a8751f`, 2026-02-02) runs everything locally:
   - **Decision center**: vLLM-served **Qwen3-Omni-30B-A3B-Instruct** via the
     `qwen3_api.py` proxy on `:10004` (non-streamed, `temperature=0, seed=42` →
     deterministic). It is **audio-native and multilingual**, so you feed the English
     `input.wav` straight to it — **no ASR is needed for the tool decision**. The
     audio message block for this endpoint is `audio_url` (the cloud build used
     `input_audio`); `tact/decider.py` now defaults to `audio_url`.
   - **Auxiliary ASR**: **sherpa-onnx `paraformer-zh-2024-03-09`** — *Chinese-only,
     no LID*. It is used only to fill `user_history` text. For English you must swap
     it (faster-whisper, a sherpa-onnx Whisper/English model, or reuse FDB-v3's
     `nvidia/parakeet-tdt-0.6b-v2`). This is a small, contained change — it does **not**
     touch the decision path.
   - **TTS**: local **IndexTTS-1.5** (`http://127.0.0.1:19000/tts`, `character:"ht"`).
     It is bilingual, so English text renders intelligibly (the benchmark ASRs your
     audio for scoring anyway); optionally supply an English voice prompt.
   - **Prompts**: all Chinese in `config.yaml`. Swap in English — the `tact/decider.py`
     prompt is already English and tool-eager.

   Net: the LLM handles English audio directly; only the *history* ASR needs swapping;
   TTS is fine. (My earlier "LID auto-detects English / no ASR swap" note came from the
   cloud `module.py` and does **not** apply to your local system.)

3. **The integration contract is one schema.** Whatever path you take, you must emit
   `result_{provider}.json` with these load-bearing fields (everything else is
   optional metadata):
   ```json
   {
     "example_id": "travel_01",
     "provider": "tact_blocking",
     "actual_tool_calls": [{"function": "...", "args": {...},
                            "timestamp_start": 0.0, "timestamp_end": 0.0}],
     "transcript": "<agent's spoken response text>",
     "status": "completed"
   }
   ```
   `evaluate_pass_rate.py` pairs `example_id` → scenario `expected_tool_calls`, and
   breaks results down **by `state_rollback_test`** — so your 21-scenario rollback
   subset is scored for free. **PASS** = all expected tools called, no extras, all
   args correct.

> Proven already: a `result_*.json` emitted by `tact/offline_runner.py` scores
> **100% Pass Rate** on the official `evaluate_pass_rate.py`. The plumbing is sound.

---

## Track 1 — Reproduce the cascaded trilemma (the official path)

This validates your harness setup and gives you the baseline numbers
(latency / turn-take / Pass@1) you are trying to beat. It does **not** touch
`fd-badcat`.

### 1.1 Environment + keys + data
```bash
cd Full-Duplex-Bench/v3
conda create -n fdb python=3.10 -y && conda activate fdb
pip install "livekit-agents[openai]~=1.3" "livekit-plugins-silero" \
            "livekit-plugins-openai" "livekit[crypto]~=1.0" numpy \
            "nemo_toolkit[asr]" pydub ffmpeg-python openai python-dotenv
apt install -y ffmpeg    # or brew install ffmpeg

# .env.local in v3/
cat > .env.local <<'ENV'
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=<...>
LIVEKIT_API_SECRET=<...>
OPENAI_API_KEY=sk-...
ENV
```
Download the data from the Drive link in `v3/README.md` and extract so you have
`v3/fdb_v3_data_released/{example_id}_{speaker_id}/{input.wav,metadata.json}`
(100 examples).

> A LiveKit Cloud free-tier account is enough. If you cannot get LiveKit at all,
> skip to Track 2 and treat the cascade numbers from the FDB-v3 paper
> (GPT-Realtime Pass@1 0.600; Cascade perfect turn-take but ~10.12 s latency;
> Gemini Live fastest ~4.25 s but 78% turn-take) as the reference baseline.

### 1.2 Run cascaded inference + evaluate
```bash
# Terminal A: start the cascaded agent (Silero VAD + Whisper + gpt-4o + OpenAI TTS)
cd v3 && python cascaded_agent.py start

# Terminal B: batch inference, then the 3 evaluators
cd v3
python run_tool_benchmark_all_released.py --provider cascaded
python evaluate_pass_rate.py  --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released --provider cascaded --use-llm
python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released --provider cascaded --use-llm
python analyze_tool_latency.py --results-dir fdb_v3_data_released --provider cascaded
```
Record from the reports: **Pass@1** (`*_pass_rate_report.json`), **tool-F1 +
response acc** (`*_evaluation_report.json`), and **first-response / tool-call /
task-completion latency** (`*_latency_report.json`). That is your trilemma anchor.

---

## Track 2 — TACT integration (the research code)

The code lives in `tact/` (drop the folder next to `src/`). Modules:

| file | what it is |
|---|---|
| `transaction.py` | **deterministic pending-set** + algebra `launch/patch/cancel/commit/speculate/compensate` + reversibility lattice `READ⪯REV⪯COMP⪯IRR`. Pure data, no I/O. |
| `tools.py` | the 12 FDB-v3 tools (mirror `mock_apis.py` exactly), reversibility map, FDB-v3-format telemetry. |
| `decider.py` | the prompt-engineered **extended decision space** → JSON ops, with a robust parser. Decoupled via `llm_call`. |
| `act_executor.py` | **async act track** (`asyncio.to_thread`) + a `BlockingActTrack` baseline behind one interface. |
| `latency_estimates.py` | per-tool latency priors for floor-holding. |
| `offline_runner.py` | emits scoreable `result_*.json` **without LiveKit**. |

### 2.1 First, get a Pass@1 number offline (no LiveKit, ~minutes)
The offline runner feeds the audio straight to Qwen3-Omni and uses the model's text
as the response, so for offline scoring you only need the **decision-center server up**
— *not* IndexTTS and *not* the sherpa ASR. Start the local vLLM Qwen3-Omni + proxy
(README Terminals 1–2), then:
```bash
# Terminal 1: vLLM serve Qwen3-Omni  (per README)
conda activate fdbc-qwen3o-vllm
vllm serve model/Qwen3-Omni-30B-A3B-Instruct --port 10003 --host 0.0.0.0 \
     --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp 4
# Terminal 2: the OpenAI-compatible proxy on :10004
conda activate fdbc-qwen3o-vllm && python src/qwen3_api.py

# Terminal 3: run the offline runner (local model; no DASHSCOPE key needed)
export PYTHONPATH="$PWD/src:$PWD"     # so `from module import llm_qwen3o` resolves
# BLOCKING baseline
python -m tact.offline_runner --data Full-Duplex-Bench/v3/fdb_v3_data_released \
       --provider tact_blocking --mode blocking
# ASYNC variant (same decider, tools fired on the act track)
python -m tact.offline_runner --data Full-Duplex-Bench/v3/fdb_v3_data_released \
       --provider tact_async --mode async

# score with the official evaluators (run from v3/, point --results-dir at the data)
cd Full-Duplex-Bench/v3
python evaluate_pass_rate.py  --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released --provider tact_blocking --use-llm
python evaluate_tool_calls.py --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released --provider tact_blocking --use-llm
```
(If you instead use the April-8 cloud `module.py`, drop the vLLM terminals, set
`DASHSCOPE_API_KEY`, and flip `tact/decider.AUDIO_BLOCK_FORMAT = "input_audio"`.)
This validates tools + decider + transaction export and gives Pass@1 / F1 / the
21-scenario rollback subset. Iterate on the `decider.py` prompt here — it is the
fastest loop you have.

> **Streaming vs offline (read this).** Offline mode hands the model the *whole*
> utterance, so it resolves "New York … actually Boston" by emitting the final
> arg directly — i.e. it behaves like a cascade: accurate but with no latency
> advantage. That is exactly what you want for the **blocking baseline** and for
> validating Pass@1. The **latency win of async is only observable in the
> real-time/streaming regime** (Track 3), where a system that commits early must
> `patch`/`cancel`, and where the dialogue loop can keep moving while a tool runs.

### 2.2 English handling for the local system (FDB-v3 is English)
- **Decision/tool head**: nothing to do — the decider sends the English audio
  directly to Qwen3-Omni (`audio_url` block) with the English `tact/decider.SYSTEM_PROMPT`.
  Qwen3-Omni-30B is multilingual, so it understands the disfluent English audio.
- **History ASR**: replace the Chinese-only sherpa Paraformer in `src/module.py:asr()`.
  Cheapest swap that keeps the same `asr(path)->str` signature:
  ```python
  # faster-whisper (English), drop-in for the sherpa recognizer
  from faster_whisper import WhisperModel
  _W = WhisperModel("small.en", device="cuda", compute_type="float16")
  def asr(path):
      segs, _ = _W.transcribe(str(path), language="en")
      return " ".join(s.text for s in segs).strip()
  ```
  (Or reuse FDB-v3's `nvidia/parakeet-tdt-0.6b-v2` for exact parity with the scorer.)
- **TTS**: `src/module.py:tts()` already posts text to local IndexTTS-1.5. English text
  renders intelligibly; if the `"ht"` voice sounds too accented, point `character` at an
  English voice prompt. The benchmark NeMo-ASRs your audio, so intelligibility (not
  accent) is what matters.
- **Prompts in `config.yaml`** (`judge` / `interrupt` / `shift` / `response`): translate
  to English so the floor-control decisions also work on English audio. Keep them terse;
  they are binary classifiers.

### 2.3 Wire blocking-vs-async into `backend.py` (the real-time path)

Today, in `ConversationEngine.handle_listen`, when the user finishes
(`judge` → `switch`, then `shift` → `no`) you do:
```python
asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
decision = await self.async_llm(self.RESPONSE_PROMPT, user_audio, self.TURN_IDX, add_to_history=True)
asyncio.create_task(self.async_tts(decision, self.TURN_IDX))
```
TACT replaces the `RESPONSE_PROMPT` step with a **tool-decision step**. Minimal diff:

**(a) In `__init__`, create the transactional machinery once:**
```python
from tact.tools import ToolRegistry
from tact.transaction import Transaction
from tact.act_executor import make_act_track
from tact.decider import decide_and_apply
from tact.module_adapter import llm_text  # thin wrapper over your llm_qwen3o

self.ACT_MODE = "blocking"            # "blocking" (baseline) | "async" (TACT)
self.tools = ToolRegistry(latency_profile="normal",
                          room=getattr(self, "room_name", "fdbadcat"),
                          telemetry_path="/tmp/agent_tool_calls.log")
self.tx = Transaction()
self.act = make_act_track(self.ACT_MODE)
```

**(b) Add one method** that turns a finished user turn into ops + a spoken line:
```python
async def handle_tool_turn(self, user_audio, turn_id):
    t = round(time.time() - self.start_wall, 3)   # audio-relative-ish timestamp
    # run the decider in a thread (it calls qwen3-omni-flash); apply ops to self.tx
    decision = await asyncio.to_thread(
        decide_and_apply,
        self.tx, self.tools.executor, llm_text,    # llm_text(messages)->str
        "LISTEN", None, user_audio, None, t,
        self.ACT_MODE == "blocking",               # blocking flag
        self.act if self.ACT_MODE == "async" else None,
    )
    say = decision.get("say", "")
    if say:
        asyncio.create_task(self.async_tts(say, turn_id))
    return decision
```

**(c) Call it** where `RESPONSE_PROMPT` was:
```python
# was: decision = await self.async_llm(self.RESPONSE_PROMPT, ...); async_tts(decision)
await self.handle_tool_turn(user_audio, self.TURN_IDX)
```

**(d) In async mode, drain ready ops each frame** so finished tool calls get
committed (cheap — results already computed) and you can narrate progress:
```python
# near the top of handle_listen / handle_speak, once per tick:
if self.ACT_MODE == "async":
    now = round(time.time() - self.start_wall, 3)
    for op in self.act.ready_ops():
        if op.op_id in self.tx.pending:
            self.tx.commit(op.op_id, self.tools.executor, t=now)
    # optional floor-holding (Week 3): self.act.max_remaining_estimate() -> narrate/backchannel
```

**The whole blocking↔async ablation is then `self.ACT_MODE`.** Blocking awaits each
tool synchronously inside `decide_and_apply` (the dialogue loop stalls = "occupied
silence" = the cascade failure mode). Async fires tools on `AsyncActTrack` and the
loop keeps running.

### 2.4 Self-correction during a turn (the rollback subset)
In the real-time loop, the decider is called as the turn unfolds. If it already
launched `search_flights(destination="New York")` and the user then says
"actually Boston", the decider returns `{"type":"patch","fn":"search_flights",
"diff":{"destination":"Boston"}}` — the pending-set rewrites the arg in place, and
**only the corrected call is ever committed** (verified: `tact/transaction.py`
self-test nets NYC→Boston to a single committed Boston call). False starts →
`cancel`. This is the mechanism the 21 `state_rollback_test` scenarios probe.

---

## Track 3 — Real-time numbers (Week 2–3, for the paper's headline)

Offline gives Pass@1/F1/rollback. The **latency + turn-take** headline needs the
real-time path. Two options, same `tact/` modules underneath:

- **Option A (recommended): a LiveKit agent wrapping the engine.** Write
  `tact_agent.py` modeled on `cascaded_agent.py`: subscribe to the user track via
  `rtc.AudioStream`, feed frames into a refactored `ConversationEngine`, publish the
  engine's TTS to the room, and let `ToolRegistry` append to
  `/tmp/agent_tool_calls.log` (already in the right format). Then it runs under the
  *unmodified* `run_tool_benchmark_all_released.py --provider tact_blocking|tact_async`
  and you get the harness's own latency/turn-take measurement.
- **Option B (no LiveKit): drive `fd-badcat`'s own frontend/backend in real time**
  (frontend feeds `input.wav` at 1× speed) and have the engine write
  `result_{provider}.json` per example (reuse the writer in `offline_runner.py`),
  then run the 3 evaluators. You lose the harness's native turn-take metric but keep
  true wall-clock latency from your `send_control` timestamps.

The key MVP figure is **blocking vs async latency at equal Pass@1**: same decider,
same tools, flip `ACT_MODE`, plot first-response + task-completion latency.

---

## Go / No-Go gate (end of MVP)

Proceed to the full method (learned policy, floor-holding, optimistic execution)
only if, on the real-time path:

- **Pass@1** (incl. the 21-scenario rollback subset) **≥ GPT-Realtime** (≈0.60 overall;
  beat its rollback subset), AND
- **task-completion latency ≤ ½ of the cascade** (cascade ≈10 s → target ≲5 s), AND
- **turn-take rate ≥ 95%** (cascade is ~perfect; do not regress it for speed).

If async beats blocking on latency **without** dropping Pass@1 below the blocking
baseline, the core TACT hypothesis holds and the rest of the blueprint is worth building.

---

## Week-1 checklist

- [ ] Track 1: cascade reproduced; trilemma numbers recorded.
- [ ] `tact/` dropped into the repo; `python -m tact.transaction` and
      `python -m tact.offline_runner --dry-run` both pass.
- [ ] Offline `tact_blocking` scored on the official `evaluate_pass_rate.py` (real model).
- [ ] English prompt + English TTS swapped in.
- [ ] `ACT_MODE` wired into `backend.py`; blocking baseline runs end-to-end on a few examples.
- [ ] `tact_async` offline run produces identical Pass@1 to blocking (correctness invariant).
- [ ] One real-time blocking-vs-async latency comparison on ≥10 examples (smoke test for Track 3).
```
