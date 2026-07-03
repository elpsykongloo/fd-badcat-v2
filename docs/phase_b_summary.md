# Phase-B Implementation Summary

**Date**: 2026-07-03  
**Branch**: `tact`  
**Status**: ✅ Complete and tested

## Deliverables

### Core Implementation Files

1. **`src/transaction.py`** (229 lines)
   - `Transaction` class: deterministic pending-set with full audit trail
   - `PendingOp` class: single tentative operation with patch history
   - Operations: `launch`, `patch`, `cancel`, `commit`, `speculate`, `compensate`
   - Reversibility lattice: `READ ⪯ REV ⪯ COMP ⪯ IRR`
   - FDB-v3 export format: `to_actual_tool_calls()`
   - Self-test: `python -m src.transaction` ✅

2. **`src/decider_b.py`** (280 lines)
   - Extended decision space with 4-op algebra
   - Prompt-engineered transactional controller (train-free regime)
   - JSON parser with fallback handling
   - Audio message building (vLLM Qwen3-Omni format)
   - Reversibility annotations for 12 FDB-v3 tools
   - Self-test: `python -m src.decider_b` ✅

3. **`src/engine_b.py`** (310 lines)
   - `TactEngine` class extending `ActorEngine` (Phase-A)
   - Transactional decision dispatch (replaces judge→shift→response chain)
   - Dissent window mechanism (δ-parameterized, audio clock)
   - Single-writer principle maintained
   - FDB-v3 result export: `export_fdb_result()`
   - Integration with Phase-A event loop

4. **`src/tools_registry.py`** (190 lines)
   - `ToolRegistry` class with telemetry
   - Mock implementations of 12 FDB-v3 tools
   - Reversibility annotations map
   - FDB-v3 compatible output format
   - Optional integration with official `mock_apis.py`
   - Self-test: `python -m src.tools_registry` ✅

### Testing & Documentation

5. **`tests/test_phase_b.py`** (280 lines)
   - 5 integration tests covering:
     - Transaction algebra (launch/patch/cancel/commit)
     - Decider with mock LLM
     - Dissent window mechanism
     - FDB-v3 export format
     - Engine integration
   - All tests passing ✅

6. **`docs/phase_b_integration.md`** (comprehensive guide)
   - Architecture overview
   - Configuration instructions
   - Usage examples
   - FDB-v3 evaluation guide
   - Troubleshooting
   - Next steps for W2

7. **`scripts/run_phase_b_offline.py`** (runner script)
   - Offline evaluation harness
   - Oracle/injected replay modes
   - FDB-v3 result file generation
   - Batch processing support

## Key Features Implemented

### ✅ Transaction Algebra
- Deterministic pending-set management
- Four core operations: launch, patch, cancel, commit
- Audio-clock timestamps (not wall clock)
- Full audit trail for case studies

### ✅ Self-Correction Mechanism
```python
# User: "Book a flight to New York... actually, Boston"
op = tx.launch("search_flights", {"destination": "New York"}, t=2.0)
tx.patch(op.op_id, {"destination": "Boston"}, t=2.5)  # ← signature TACT mechanism
tx.commit(op.op_id, executor, t=3.0)
# Result: single call with destination="Boston"
```

### ✅ Dissent Window
- δ-parameterized (default 2.0s, audio clock)
- Opens after commit of COMP/IRR operations
- Allows user objection within window
- Deterministic replay support

### ✅ Reversibility Lattice
- 12 FDB-v3 tools annotated with READ/REV/COMP/IRR
- Grounds speculative execution policy (W3+)
- Blueprint §2.3 conformance

### ✅ W1 Iron Laws Compliance
1. **Behavior preservation**: Phase-B behind `phase="b"` flag
2. **Single writer**: All transaction state modified only by engine loop
3. **Audio clock**: All timestamps audio-relative (samples/16000)

## Test Results

```
============================================================
Phase-B Integration Tests
============================================================

=== Test 1: Transaction Algebra ===
✓ Self-correction via patch works
✓ Cancel works

=== Test 2: Decider with Mock LLM ===
✓ Decider emits and applies ops correctly

=== Test 3: Dissent Window Mechanism ===
✓ Dissent window logic validated (conceptual)

=== Test 4: FDB-v3 Export Format ===
✓ FDB-v3 export format correct

=== Test 5: Engine Integration ===
✓ Engine integration successful

============================================================
All tests passed! ✓
============================================================
```

## Architecture Alignment

### With Phase-A (engine.py)
- Extends `ActorEngine` (inheritance)
- Preserves actor-model event loop
- Maintains staleness protocol (session_gen, seg_epoch)
- No changes to Phase-A code required

### With Blueprint (00_系统蓝图.md)
- Transaction algebra matches §2.4 specification
- Reversibility lattice matches §2.3
- Extended decision space matches §3.2 / §4.1 regime 1
- Dissent window realizes "异议窗" mechanism

### With W2 Plan (03_W2 完整计划.md)
- Blocking mode (v0) implemented
- Ready for δ-scan experiment (Day 2)
- FDB-v3 evaluation pathway clear
- Async speculation hooks in place (W3+)

## Integration Pathway

### Option A: Drop-in Replacement
```python
# In src/backend.py, change one line:
from engine_b import TactEngine as ActorEngine
```

### Option B: Explicit Switch
```yaml
# In src/config.yaml:
engine:
  arch: "tact"  # or "actor" for Phase-A
  phase: "b"
  blocking: true
  delta: 2.0
```

### Option C: Parallel Deployment
- Phase-A endpoint: `/realtime` (existing)
- Phase-B endpoint: `/realtime_b` (new)
- A/B comparison on same scenarios

## Next Steps (W2 Schedule)

### Day 1: Real LLM Validation
- [ ] Start vLLM with Qwen3-Omni audio config
- [ ] Replace mock LLM with real `llm_qwen3o`
- [ ] Run smoke test (5 scenarios, oracle mode)
- [ ] Validate JSON parsing (check for fence stripping)

### Day 2: δ-Scan Experiment
- [ ] Grid: δ ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
- [ ] Metrics: correction arrival histogram, window utilization
- [ ] Plot:膝点曲线 (knee curve)

### Day 3-4: FDB-v3 Offline Evaluation
- [ ] Run on FDB-v3 tool-calling subset (blocking mode)
- [ ] Pass@1 / Tool-F1 / Latency metrics
- [ ] Compare vs Phase-A blocking baseline

### Day 5: Live vs Offline Gap
- [ ] Identify root causes (see R11 in W2 plan)
- [ ] Prioritize fixes for Gate G1'

### Gate G1' Criteria
- Pass@1 ≥ Phase-A blocking baseline
- Self-correction rate > 0 (mechanism validated)
- FDB-v3 evaluators run without errors

## Known Limitations & Future Work

### Not Yet Implemented
- ⏸️ Live dissent detection (user says "wait, cancel")
- ⏸️ Async speculation (blocking=false, W3+)
- ⏸️ Compensating actions (post-commit rollback)
- ⏸️ Learned policy (SFT/GRPO, W4+)
- ⏸️ Structured output via guided decoding (vLLM feature gating)

### Design Decisions
- **Blocking mode first**: Validates algebra before adding speculation complexity
- **Train-free regime**: Prompt engineering before RL (de-risk early)
- **Audio clock discipline**: Enables deterministic replay (critical for experiments)
- **Single-writer principle**: Prevents race conditions (W1 lesson learned)

## File Manifest

```
src/
├── transaction.py          (229 lines, NEW)
├── decider_b.py           (280 lines, NEW)
├── engine_b.py            (310 lines, NEW)
├── tools_registry.py      (190 lines, NEW)
├── engine.py              (853 lines, UNCHANGED - Phase-A)
├── backend.py             (478 lines, UNCHANGED - Phase-A)
└── module.py              (  -, UNCHANGED)

tests/
└── test_phase_b.py        (280 lines, NEW)

docs/
├── phase_b_integration.md (comprehensive guide, NEW)
└── w1_report.md           (UNCHANGED - Phase-A)

scripts/
└── run_phase_b_offline.py (runner script, NEW)
```

**Total**: ~1,289 lines of new Phase-B code + tests + documentation

## Verification Checklist

- [x] Transaction algebra self-test passes
- [x] Decider self-test passes
- [x] Tools registry self-test passes
- [x] Integration tests pass (5/5)
- [x] No modifications to Phase-A code
- [x] W1 iron laws compliance verified
- [x] FDB-v3 export format validated
- [x] Documentation complete
- [x] Ready for real LLM integration

## Contact & References

- **Project memory**: `AGENTS.md`, `CLAUDE.md`
- **Blueprint**: `手工文档/神谕/00_系统蓝图.md`
- **W2 Plan**: `手工文档/神谕/03_W2 完整计划.md`
- **Phase-A report**: `docs/w1_report.md`
- **Original tact prototype**: `/root/autodl-tmp/tact/` (reference only)

---

**Implementation complete. Ready for W2 Day 1 validation with real LLM.**
