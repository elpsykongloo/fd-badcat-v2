# Phase-B Integration Guide

## Overview

Phase-B extends the Phase-A actor engine with **transactional semantics** for tool-calling voice agents. This implementation realizes the TACT (Transactional Agent Control Theory) architecture for the fd-badcat system.

**Status**: Phase-B v0 (blocking mode) implemented and tested. Ready for FDB-v3 evaluation.

## Architecture

### Core Components

1. **`src/transaction.py`** - Transaction algebra and PendingSet
   - `Transaction` class: deterministic pending-set management
   - `PendingOp` class: single tentative operation
   - Operations: `launch`, `patch`, `cancel`, `commit`, `speculate`, `compensate`
   - Reversibility lattice: `READ ⪯ REV ⪯ COMP ⪯ IRR`

2. **`src/decider_b.py`** - Extended decision space
   - Prompt-engineered transactional controller (train-free regime)
   - Emits JSON programs: `{"dialogue": "speak|listen|stay", "ops": [...], "say": "..."}`
   - Self-correction via `patch` operations
   - Tool-eager policy (execute immediately, no confirmation)

3. **`src/engine_b.py`** - Phase-B engine (TactEngine)
   - Extends `ActorEngine` (Phase-A) with transaction management
   - Dissent window mechanism (δ-parameterized)
   - Single-writer principle maintained
   - Audio clock discipline preserved

4. **`src/tools_registry.py`** - Tool executor with telemetry
   - Wraps FDB-v3 mock APIs
   - Reversibility annotations
   - FDB-v3 compatible output format

### Key Mechanisms

#### Transaction Algebra

```python
from transaction import Transaction, Reversibility

tx = Transaction()

# User: "Book a flight to New York... actually, Boston"
op = tx.launch("search_flights", 
               {"destination": "New York", "date": "July 15"},
               Reversibility.READ, t=2.0)

tx.patch(op.op_id, {"destination": "Boston"}, t=2.5)  # Self-correction!
tx.commit(op.op_id, executor, t=3.0)

# Result: committed call has destination="Boston"
```

#### Dissent Window

After committing an irreversible (IRR) or compensable (COMP) operation, a **dissent window** of duration δ (default 2.0s) opens. During this window:
- User can object ("wait, cancel that")
- Engine can `patch` or `cancel` the operation
- After δ seconds with no dissent, operation is finalized

**Audio clock discipline**: δ is measured on the audio timeline (not wall clock), ensuring deterministic replay.

#### Self-Correction

The signature TACT mechanism. Instead of launching a new operation when the user corrects themselves:

```python
# Phase-A (blocking): 
#   search_flights(destination="NYC")  # executed
#   search_flights(destination="Boston")  # duplicate!

# Phase-B (transactional):
#   launch search_flights(destination="NYC")
#   patch op_id=N diff={"destination": "Boston"}  # single corrected call
#   commit op_id=N  # executes with destination="Boston"
```

### Reversibility Lattice

Tools are annotated with reversibility classes (blueprint §2.3):

| Class | Semantics | Examples | Speculative Execution |
|-------|-----------|----------|----------------------|
| **READ** | Pure, no side effect | `search_flights`, `get_exchange_rate` | Always safe |
| **REV** | Cheap exact inverse | `add_to_cart`, `update_search_filter` | Safe (with inverse) |
| **COMP** | Compensating action | `book_flight`, `modify_autopay` | Safe (with compensator) |
| **IRR** | No inverse | `update_identity_doc` | Never speculate |

## Configuration

### Engine Config (engine_cfg)

```yaml
engine:
  arch: "tact"           # Use Phase-B engine
  phase: "b"             # Enable transactional mode
  blocking: true         # v0: launch+commit immediate (false = async speculation)
  delta: 2.0             # Dissent window duration (seconds, audio clock)
```

### LLM Config (llm_cfg)

```yaml
llm:
  audio_block: "audio_url"        # Local vLLM Qwen3-Omni format
  decision_timeout_s: 30          # Decision timeout
```

## Usage

### Integration with fd-badcat

1. **Replace engine import** in `src/backend.py`:
   ```python
   # Old: from engine import ActorEngine
   from engine_b import TactEngine as ActorEngine
   ```

2. **Update config** (`src/config.yaml`):
   ```yaml
   engine:
     arch: "tact"
     phase: "b"
     blocking: true
     delta: 2.0
   ```

3. **Provide tool executor**:
   ```python
   from tools_registry import ToolRegistry
   
   registry = ToolRegistry(latency_profile="instant")
   engine = TactEngine(
       websocket=websocket,
       prompts=prompts,
       delay=delay,
       llm_cfg=llm_cfg,
       engine_cfg=engine_cfg,
       tool_executor=registry.executor
   )
   ```

### Standalone Testing

```bash
# Run integration tests
/root/miniconda3/envs/fd-sds/bin/python tests/test_phase_b.py

# Test transaction algebra
/root/miniconda3/envs/fd-sds/bin/python -m src.transaction

# Test decider
/root/miniconda3/envs/fd-sds/bin/python -m src.decider_b

# Test tools registry
/root/miniconda3/envs/fd-sds/bin/python -m src.tools_registry
```

## FDB-v3 Evaluation

### Export Format

Phase-B engines export results in FDB-v3 compatible format:

```python
result = engine.export_fdb_result(example_id="travel_01", provider="tact_b_v0")

# Result structure:
{
    "example_id": "travel_01",
    "provider": "tact_b_v0",
    "actual_tool_calls": [
        {
            "function": "search_flights",
            "args": {"destination": "Boston", "date": "July 15"},
            "timestamp_start": 2.0,
            "timestamp_end": 3.0
        }
    ],
    "transcript": "Searching flights to Boston.",
    "status": "completed",
    "transaction_log": [...],  # Full audit trail
    "dissent_windows_used": false
}
```

### Evaluation Metrics

Phase-B should be evaluated on:

1. **Pass@1** (FDB-v3 `evaluate_pass_rate.py`) - correctness
2. **Tool-F1** (FDB-v3 `evaluate_tool_calls.py`) - precision/recall
3. **First-response latency** (`analyze_tool_latency.py`)
4. **Task-completion latency**
5. **Self-correction rate** (patch ops / total ops)
6. **Dissent window utilization** (windows used / windows opened)

## Implementation Status

### Completed (Phase-B v0)

- ✅ Transaction algebra (`launch/patch/cancel/commit`)
- ✅ Reversibility annotations for 12 FDB-v3 tools
- ✅ Extended decision space (prompt-engineered)
- ✅ Dissent window mechanism (δ-parameterized)
- ✅ Audio clock discipline (deterministic replay)
- ✅ Single-writer principle maintained
- ✅ FDB-v3 export format
- ✅ Integration tests
- ✅ Blocking mode (launch+commit immediate)

### Not Yet Implemented (Future)

- ⏸️ Async speculation (`blocking=false`, W3+)
- ⏸️ Compensating actions (post-commit rollback)
- ⏸️ Live dissent detection (user says "wait, cancel that")
- ⏸️ Learned policy (SFT / time-shaped GRPO, W4+)
- ⏸️ Structured output via guided decoding (vLLM feature gating)

## Conformance with W1 Iron Laws

1. **Behavior preservation**: Phase-B is behind `phase="b"` flag. Phase-A behavior unchanged when `phase="a"` (default).

2. **Single writer principle**: All transaction state (`tx`, `dissent_windows`) modified only by engine loop. Decision workers communicate via queue events.

3. **Audio clock discipline**: All timestamps (`launched_at`, `committed_at`, dissent window expiry) use audio-relative time (`t_audio = samples/16000`), not wall clock.

## Troubleshooting

### JSON Parse Errors

If the LLM emits malformed JSON:
- Fallback: `{"dialogue": "stay", "ops": [], "say": ""}`
- Check `parse_error` field in decision result
- Consider adding one-retry logic with error feedback

### Tool Not Found

If tool executor fails:
- Check tool name spelling in decider prompt
- Verify `REVERSIBILITY` map includes the tool
- Check `MOCK_TOOLS` or FDB-v3 `mock_apis.py` has the tool

### Dissent Window Not Triggering

- Ensure `delta > 0` in engine config
- Check control events: `dissent_window_open` / `dissent_window_closed`
- Verify audio clock advancing (replay mode issue?)

## References

- Blueprint: `/root/autodl-tmp/fd-badcat/手工文档/神谕/00_系统蓝图.md`
- W2 Plan: `/root/autodl-tmp/fd-badcat/手工文档/神谕/03_W2 完整计划.md`
- Original tact prototype: `/root/autodl-tmp/tact/`
- AGENTS.md: Project memory and iron laws

## Next Steps (W2 Schedule)

1. **Day 1**: Validate with real LLM (Qwen3-Omni audio decisions)
2. **Day 2**: δ-scan experiment (dissent window grid search)
3. **Day 3-4**: FDB-v3 offline evaluation (Pass@1, latency)
4. **Day 5**: Live vs offline gap analysis
5. **Gate G1'**: Phase-B v0 Pass@1 ≥ blocking baseline

---

**Implementation**: 2026-07-03  
**Status**: Ready for W2 Day 1 validation  
**Contact**: See AGENTS.md for project memory
