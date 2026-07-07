#!/bin/bash
# Validation script for audio clock-driven deterministic replay system

set -e

PYTHON=/root/miniconda3/envs/fd-sds/bin/python
REPO_ROOT=/root/autodl-tmp/fd-badcat

cd $REPO_ROOT

echo "=== Audio Clock System Validation ==="
echo ""

# 1. Unit tests
echo "1. Running unit tests..."
$PYTHON -c "
import sys
sys.path.insert(0, 'src')

from audio_clock import AudioClock, AudioClockFrameGenerator
from injected_replay import DecisionScript, compare_traces

# Audio clock
clock = AudioClock()
clock.tick(256)
assert abs(clock.t_audio - 0.016) < 1e-6

# Frame generator
gen = AudioClockFrameGenerator()
import numpy as np
frames = gen.chunk_to_frames(np.zeros(512, dtype=np.float32))
assert len(frames) == 2

# DecisionScript
golden = [
    {'event': 'llm_done', 'data': {'kind': 'judge', 'content': 'switch', 'infer_time': 0.1}},
]
script = DecisionScript(golden)
assert script.stats()['judge'] == 1

# Trace comparison
trace = [{'event': 'vad_start', 'data': {'turn': 0}}]
identical, _ = compare_traces(trace, trace)
assert identical

print('✓ All unit tests passed')
"

# 2. Real golden trace parsing
echo ""
echo "2. Testing real golden trace parsing..."
$PYTHON -c "
import sys
sys.path.insert(0, 'src')
from injected_replay import load_golden_trace, DecisionScript

trace = load_golden_trace('traces/golden_rerun/ask_0001_0004.jsonl')
print(f'  Loaded {len(trace)} events')

script = DecisionScript(trace)
stats = script.stats()
print(f'  Decision queues: {stats}')
print('✓ Golden trace parsing works')
"

# 3. Deterministic replay
echo ""
echo "3. Testing deterministic replay..."
$PYTHON << 'PYEOF'
import sys
import asyncio
sys.path.insert(0, 'src')

from injected_replay import InjectedReplaySession, compare_traces
import yaml

async def test():
    with open('src/config.yaml') as f:
        config = yaml.safe_load(f)
    
    session1 = InjectedReplaySession(
        'traces/golden_rerun/ask_0001_0004.jsonl',
        'exp/golden/actor_ask_0001_0004/stream_turn0_input.wav',
        config,
        'exp/validate_tmp/run1'
    )
    trace1, _ = await session1.replay()
    
    session2 = InjectedReplaySession(
        'traces/golden_rerun/ask_0001_0004.jsonl',
        'exp/golden/actor_ask_0001_0004/stream_turn0_input.wav',
        config,
        'exp/validate_tmp/run2'
    )
    trace2, _ = await session2.replay()
    
    identical, diffs = compare_traces(trace1, trace2)
    
    if identical:
        print(f'  Run 1: {len(trace1)} events')
        print(f'  Run 2: {len(trace2)} events')
        print('  ✓ Traces identical (deterministic)')
        return True
    else:
        print(f'  ✗ NOT deterministic: {len(diffs)} differences')
        return False

result = asyncio.run(test())
sys.exit(0 if result else 1)
PYEOF

# 4. CLI runner
echo ""
echo "4. Testing CLI runner..."
$PYTHON scripts/run_injected_replay.py \
    --golden traces/golden_rerun/ask_0001_0004.jsonl \
    --wav exp/golden/actor_ask_0001_0004/stream_turn0_input.wav \
    --out exp/validate_tmp/cli_test \
    > /tmp/cli_test.log 2>&1

if [ -f exp/validate_tmp/cli_test/trace.jsonl ]; then
    event_count=$(wc -l < exp/validate_tmp/cli_test/trace.jsonl)
    echo "  Generated trace with $event_count events"
    echo "✓ CLI runner works"
else
    echo "✗ CLI runner failed"
    exit 1
fi

echo ""
echo "=== All Validation Tests PASSED ==="
echo ""
echo "Deliverables ready:"
echo "  - src/audio_clock.py"
echo "  - src/injected_replay.py"
echo "  - tests/test_deterministic_replay.py"
echo "  - scripts/run_injected_replay.py"
echo "  - docs/audio_clock_replay.md"
echo "  - docs/QUICKSTART.md"
