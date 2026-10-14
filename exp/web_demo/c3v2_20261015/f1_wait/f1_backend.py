# Migrated to c3v2; no artifact integrity hashing.
"""Isolated f1_wait backend: the worktree's own src/backend.py, unchanged, plus two
experiment-only shims around ActorEngine._capacity_stream:

1. Fault injection for the CLOSED input_route call of a span that was already
   admitted (playback already cut). "timeout" never returns, so the production
   decision deadline fires and produces its own timed_out/fallback audit; "error"
   raises inside the call, so production error handling runs. Arming is read per
   call from F1_FAULT_FILE ({"trial", "fault", "max_spans"}), written by the runner
   only after the setup answer is playing. Injected calls never reach Omni.
2. An Omni request budget (F1_OMNI_BUDGET, 0 = unlimited) counted across restarts in
   F1_CALL_LOG; every real Omni request is logged with its kind and trial.

usage: F1_ROOT=... F1_FAULT_FILE=... F1_CALL_LOG=... python f1_backend.py CONFIG PORT
"""
import asyncio
import json
import os
import sys
import time
from contextlib import aclosing
from pathlib import Path
ROOT = Path(os.environ['F1_ROOT'])
sys.path.insert(0, str(ROOT / 'src'))
os.chdir(ROOT)
import backend
import engine
import module
FAULT_FILE = Path(os.environ['F1_FAULT_FILE'])
CALL_LOG = Path(os.environ['F1_CALL_LOG'])
BUDGET = int(os.environ.get('F1_OMNI_BUDGET', '0') or 0)
_original = engine.ActorEngine._capacity_stream
_faulted = {}
_omni = [0]
if CALL_LOG.exists():
    _omni[0] = sum((1 for line in CALL_LOG.read_text().splitlines() if '"omni": true' in line))

def _log(row):
    with CALL_LOG.open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')

def _spec():
    try:
        return json.loads(FAULT_FILE.read_text())
    except (OSError, ValueError):
        return {}

async def _capacity_stream(self, kind, stream_fn, arg, *, case_context=None, route=False, call_id=None, parent_id=None, response_options=None, stream_state=None):
    ctx = case_context or {}
    spec = _spec()
    (fault, trial) = (spec.get('fault', 'none'), spec.get('trial'))
    span = getattr(self, '_guard_input', None)
    if fault in ('timeout', 'error') and route and ctx.get('closed') and (ctx.get('decision_stage') == 'input_route') and (span is not None) and (span.sid == ctx.get('input_id')) and span.admitted and ('input_ids' not in spec or span.sid in spec['input_ids']):
        ids = _faulted.setdefault(trial, [])
        if span.sid in ids or len(ids) < int(spec.get('max_spans', 1)):
            if span.sid not in ids:
                ids.append(span.sid)
            _log({'wall': time.time(), 'trial': trial, 'session': getattr(self, 'demo_session_id', None), 'kind': kind, 'injected': fault, 'input_id': span.sid, 'revision': ctx.get('revision'), 'omni': False})
            if fault == 'timeout':
                await asyncio.sleep(3600)
            raise RuntimeError('f1_injected_route_error')
    is_omni = stream_fn in (module.llm_qwen3o_stream, module.tts_omni_stream)
    if is_omni:
        if BUDGET and _omni[0] >= BUDGET:
            _log({'wall': time.time(), 'trial': trial, 'kind': kind, 'refused': 'budget', 'omni': False})
            raise RuntimeError('f1_omni_budget_exhausted')
        _omni[0] += 1
        _log({'wall': time.time(), 'trial': trial, 'session': getattr(self, 'demo_session_id', None), 'kind': kind, 'call_id': call_id, 'closed': ctx.get('closed'), 'stage': ctx.get('decision_stage'), 'omni': True, 'n': _omni[0]})
    async with aclosing(_original(self, kind, stream_fn, arg, case_context=case_context, route=route, call_id=call_id, parent_id=parent_id, response_options=response_options, stream_state=stream_state)) as source:
        async for item in source:
            yield item
engine.ActorEngine._capacity_stream = _capacity_stream
if __name__ == '__main__':
    (config, port) = (sys.argv[1], sys.argv[2])
    sys.argv = ['backend.py', '--config', config, '--streaming', '--host', '127.0.0.1', '--port', port]
    backend.main()
