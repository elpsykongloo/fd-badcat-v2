# Migrated to c3v2; no artifact integrity hashing.
"""Audio-clock harness for f2_vad: production ActorEngine/GuardedTurns, production Silero VAD.

One episode = one engine session: a scripted lead-in request starts assistant playback
(canned answer text, silent TTS PCM, browser playback acks every 48 ms), then the
labelled user audio is mixed into the microphone stream (pink floor; browser AEC leaves no audible
echo, the render reference is still attached) and fed frame by frame (256 samples).

Production code under test, unmodified: ActorEngine._process_event/_on_frame,
detect_vad_frame + VADIterator(input_timing), GuardedTurns admission/deferral/reopen,
route_messages, decide_input_route/decide_control/reply_messages, parse_route, the
payload builder (module.qwen_text_payload) and the real Omni call
(module.llm_qwen3o_stream).

Virtualised: model latency runs on the audio clock. A route/reply call registers at
dispatch with a sampled latency; its Omni request is issued only when the audio clock
reaches the due time (a cancelled/stale call never reaches Omni) and the result is
delivered on that frame. control_labels' perf_counter is the audio clock and its
cancellable_wait deadline is enforced on the audio clock, so the production 2 s decision
budget (route + review share one deadline) applies to simulated latency, not to harness
wall time. Identical payloads within one unit share one Omni result (common random
numbers across arms); reps use separate caches.
"""
import asyncio
import contextvars
import hashlib
import json
import os
import sys
import time as _real_time
from copy import deepcopy
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('F2_ROOT', '/root/autodl-tmp/fd-badcat-c3v2'))
os.environ.setdefault('FDBC_QWEN_URL', 'http://127.0.0.1:10003/v1/chat/completions')
os.environ.setdefault('FDBC_QWEN_TIMEOUT', '60')
os.environ.setdefault('NO_PROXY', '127.0.0.1,localhost')
for _k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
    os.environ.pop(_k, None)
sys.path.insert(0, str(ROOT / 'src'))
import soundfile as sf
import yaml
from scipy.signal import lfilter, resample_poly
import module
import control_labels
from engine import ActorEngine, ControlMsg, FrameEvent
from request_capacity import RequestCapacity
from stream_transport import PCMChunk, TextDelta
SR = 16000
FRAME = 256
_GATE = contextvars.ContextVar('f2_gate', default=None)
_DEADLINE = contextvars.ContextVar('f2_deadline', default=None)
_ORIG_STREAM = module.llm_qwen3o_stream
_ORIG_WAIT = control_labels.cancellable_wait

class _AudioClockTime:
    """control_labels.time: perf_counter is the episode's audio clock."""

    def __getattr__(self, name):
        return getattr(_real_time, name)

    def perf_counter(self):
        gate = _GATE.get()
        return gate.now if gate is not None else _real_time.perf_counter()

async def _audio_clock_wait(awaitable, timeout):
    gate = _GATE.get()
    if gate is None:
        return await _ORIG_WAIT(awaitable, timeout)
    token = _DEADLINE.set(gate.now + max(0.0, timeout))
    try:
        return await _ORIG_WAIT(awaitable, None)
    finally:
        _DEADLINE.reset(token)

async def _dispatch_stream(messages, **kwargs):
    gate = _GATE.get()
    if gate is None:
        async for part in _ORIG_STREAM(messages, **kwargs):
            yield part
        return
    async for part in gate.stream(messages, **kwargs):
        yield part
control_labels.time = _AudioClockTime()
control_labels.cancellable_wait = _audio_clock_wait
module.llm_qwen3o_stream = _dispatch_stream
import async_utils as _au
_ORIG_AU_WAIT = _au.cancellable_wait

async def _no_wall_timeout(awaitable, timeout):
    """Drop wall-clock timeouts (speech_stream credit wait 5 s, text/TTS reads) while a gate
    is active: the harness can block on a real Omni call with the audio clock frozen."""
    return await _ORIG_AU_WAIT(awaitable, None if _GATE.get() is not None else timeout)
for _m in list(sys.modules.values()):
    if _m is not None and _m is not control_labels and getattr(_m, 'cancellable_wait', None) is _ORIG_AU_WAIT:
        _m.cancellable_wait = _no_wall_timeout

def load_config():
    base = yaml.safe_load((ROOT / 'src/config.yaml').read_text(encoding='utf-8'))
    demo = yaml.safe_load((ROOT / 'configs/demo_chat.yaml').read_text(encoding='utf-8'))
    cfg = deepcopy(base)
    for (key, value) in demo.items():
        if isinstance(value, dict):
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    return cfg

class LatencyModel:
    """'pool' = empirical production pool (problem_size/all/latency_pool.json);
    'fixed:ROUTE_MS,REPLY_MS'. Draws are keyed on (rep, unit, payload, role), so the same
    payload gets the same latency in every arm of a unit."""

    def __init__(self, spec, pool_path):
        self.spec = spec
        if spec == 'pool':
            pool = json.loads(Path(pool_path).read_text())
            self.pool = {k: np.array(sorted(pool[k]), dtype=float) / 1000 for k in ('input_route', 'input_reply')}
        elif spec.startswith('fixed:'):
            (route, reply) = (float(v) / 1000 for v in spec[6:].split(','))
            self.fixed = {'input_route': route, 'input_reply': reply}
        else:
            raise ValueError(spec)

    def __call__(self, role, key, salt):
        if self.spec != 'pool':
            return self.fixed[role]
        u = int(hashlib.sha256(f'{salt}|{key}|{role}'.encode()).hexdigest()[:15], 16) / 16 ** 15
        values = self.pool[role]
        return float(values[min(len(values) - 1, int(u * len(values)))])

class OmniCache:
    """Per-unit result cache shared by every arm of the unit; persisted with the unit."""

    def __init__(self, entries=None, *, sem, budget, dry=None, log=None):
        self.entries = dict(entries or {})
        (self.sem, self.budget, self.dry, self.log) = (sem, budget, dry, log)

    async def get(self, key, messages, kwargs, role, unit):
        hit = self.entries.get(key)
        if hit is not None and (not hit.get('error')):
            return (hit, True)
        if self.dry is not None:
            entry = {'text': self.dry(messages, role), 'wall_ms': 0.0, 'attempts': 0, 'dry': True}
            self.entries[key] = entry
            return (entry, False)
        self.budget.take()
        entry = {'attempts': 0}
        async with self.sem:
            for attempt in range(3):
                entry['attempts'] = attempt + 1
                began = _real_time.perf_counter()
                try:
                    parts = [p async for p in _ORIG_STREAM(messages, route=kwargs.get('route', False), max_tokens=kwargs.get('max_tokens'))]
                    entry.update(text=''.join((p for p in parts if isinstance(p, str))), wall_ms=round((_real_time.perf_counter() - began) * 1000, 1))
                    entry.pop('error', None)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    entry.update(error=type(exc).__name__, wall_ms=round((_real_time.perf_counter() - began) * 1000, 1))
        self.entries[key] = entry
        if self.log is not None:
            self.log({'unit': unit, 'key': key[:16], 'role': role, 'attempts': entry['attempts'], 'wall_ms': entry.get('wall_ms'), 'error': entry.get('error'), 'http_timeout_s': int(os.environ['FDBC_QWEN_TIMEOUT'])})
        return (entry, False)

class Budget:

    def __init__(self, limit=None):
        (self.limit, self.used) = (limit, 0)

    def take(self):
        if self.limit is not None and self.used >= self.limit:
            raise RuntimeError(f'Omni request cap reached ({self.limit})')
        self.used += 1

class Gate:

    def __init__(self, *, prompts, latency, salt, cache, unit, leadin):
        self.now = 0.0
        self.prompts = prompts
        self.by_prompt = {v: k for (k, v) in prompts.items() if isinstance(v, str)}
        (self.latency, self.salt, self.cache, self.unit) = (latency, salt, cache, unit)
        self.leadin = leadin
        self.phase = 'leadin'
        self.pending = []
        self.calls = []

    def role(self, messages):
        system = messages[0].get('content') if messages and messages[0].get('role') == 'system' else None
        if isinstance(system, str) and system.startswith(self.prompts['input_route']):
            return 'input_route'
        if system == self.prompts.get('input_reply'):
            return 'input_reply'
        return self.by_prompt.get(system, 'response')

    def canned(self, role):
        if role == 'input_route':
            return json.dumps({'transcript': self.leadin['previous_user'] or '你好', 'label': 'yield_ready'}, ensure_ascii=False)
        if role in ('judge', 'interrupt'):
            return 'switch'
        if role == 'shift':
            return 'no'
        if role == 'input_reply':
            return 'keep'
        return self.leadin['assistant_text'] if self.phase == 'leadin' else '好的。'

    async def stream(self, messages, **kwargs):
        role = self.role(messages)
        if self.phase == 'leadin' or role not in ('input_route', 'input_reply'):
            yield self.canned(role)
            if kwargs.get('report_finish'):
                yield TextDelta('', finish_reason='stop')
            return
        payload = module.qwen_text_payload(messages, route=kwargs.get('route', False), max_tokens=kwargs.get('max_tokens'))
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        latency = self.latency(role, key, self.salt)
        deadline = _DEADLINE.get()
        due = self.now + latency
        timeout = deadline is not None and due > deadline + 1e-09
        entry = {'role': role, 'key': key, 't': round(self.now, 3), 'latency': round(latency, 3), 'due': round(deadline if timeout else due, 3), 'timeout': timeout, 'fut': asyncio.get_running_loop().create_future(), 'messages': messages, 'kwargs': kwargs}
        self.pending.append(entry)
        self.calls.append(entry)
        try:
            result = await entry['fut']
        except asyncio.CancelledError:
            entry['cancelled'] = True
            raise
        if timeout:
            raise asyncio.TimeoutError()
        if result.get('error'):
            raise RuntimeError('omni transport failure: ' + result['error'])
        yield result['text']

    def next_due(self):
        live = [e['due'] for e in self.pending if not e['fut'].done()]
        return min(live) if live else None

    async def release(self, t):
        """Deliver every live call due at or before audio time t (oldest first)."""
        for entry in sorted(self.pending, key=lambda e: e['due']):
            if entry['fut'].done() or entry['due'] > t + 1e-09:
                continue
            if entry['timeout']:
                entry['fut'].set_result({})
                continue
            (result, hit) = await self.cache.get(entry['key'], entry['messages'], entry['kwargs'], entry['role'], self.unit)
            entry.update(cache_hit=hit, wall_ms=result.get('wall_ms'), error=result.get('error'), out=(result.get('text') or '')[:200])
            if not entry['fut'].done():
                entry['fut'].set_result(result)
        self.pending = [e for e in self.pending if not e['fut'].done()]

class Socket:

    def __init__(self):
        self.events = []

    async def send_text(self, value):
        self.events.append(json.loads(value))

    async def send_bytes(self, value):
        pass

def speech_seconds(text):
    cjk = sum((1 for ch in text if '一' <= ch <= '\u9fff'))
    words = len([w for w in text.replace('，', ' ').split() if any((c.isalpha() for c in w))])
    stops = sum((text.count(p) for p in '，。？！,.?!;；：:'))
    return max(0.3, 0.21 * cjk + 0.3 * words + 0.15 * stops)

async def silent_tts(text, **_):
    samples = int(24000 * speech_seconds(text))
    step = int(24000 * 0.12)
    for start in range(0, samples, step):
        yield PCMChunk(b'\x00\x00' * min(step, samples - start), 24000)
_AUDIO = {}

def read_audio(path, seg=None):
    key = (path, tuple(seg) if seg else None)
    if key not in _AUDIO:
        (x, sr) = sf.read(path, dtype='float32', always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            from math import gcd
            g = gcd(sr, SR)
            x = resample_poly(x, SR // g, sr // g).astype('float32')
        if seg:
            x = x[max(0, int(seg[0] * SR)):min(len(x), int(seg[1] * SR))]
        _AUDIO[key] = x
    return _AUDIO[key]

def fade(x, ms):
    n = min(len(x) // 2, int(SR * ms / 1000))
    if n:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n, dtype=np.float32))
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x

def utterance(item, hole, header):
    """User audio (float32 @16 kHz), voiced span and the hole span, all utterance-relative."""
    syn = header['synthesis']
    if item['set'] == 'Q':
        (a, b) = item['parts']
        xa = fade(read_audio(a['path'])[int(a['cut'][0] * SR):int(a['cut'][1] * SR)].copy(), syn['pair_fade_ms'])
        xb = fade(read_audio(b['path'])[int(b['cut'][0] * SR):int(b['cut'][1] * SR)].copy(), syn['pair_fade_ms'])
        pad = np.zeros(int(0.2 * SR), np.float32)
        x = np.concatenate([pad, xa, np.zeros(int(item['gap_s'] * SR), np.float32), xb, pad])
        voiced = (0.2, len(x) / SR - 0.2)
        rms = float(np.sqrt(np.mean(np.concatenate([xa, xb]) ** 2)))
        x *= min(10 ** (syn['voiced_rms_dbfs'] / 20) / max(rms, 1e-06), 0.95 / max(float(np.max(np.abs(x))), 1e-06))
        second = 0.2 + len(xa) / SR + item['gap_s']
        return (x, voiced, None, {'first_end': 0.2 + len(xa) / SR, 'second_start': second})
    part = item['parts'][0]
    x = read_audio(part['path'], part.get('seg')).copy() * item['gain']
    span = None
    if hole:
        start = item['holes'][str(hole)]
        (a, b) = (int(start * SR), int((start + hole) * SR))
        ramp = int(SR * syn['hole_ramp_ms'] / 1000)
        depth = 10 ** (syn['hole_depth_db'] / 20)
        gain = np.ones(len(x), np.float32)
        gain[a:b] = depth
        down = np.linspace(1, depth, ramp, dtype=np.float32)
        gain[max(0, a - ramp):a] = down[-len(gain[max(0, a - ramp):a]):]
        gain[b:b + ramp] = down[::-1][:len(gain[b:b + ramp])]
        x *= gain
        span = (start, start + hole)
    return (x, tuple(item['voiced']), span, {})

def pink(n, seed, dbfs):
    z = np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    z = lfilter([1], [1, -0.96], z).astype(np.float32)
    return z * (10 ** (dbfs / 20) / max(float(np.sqrt(np.mean(z * z))), 1e-09))

def scaled(path, dbfs):
    x = read_audio(path)
    return x * (10 ** (dbfs / 20) / max(float(np.sqrt(np.mean(x * x))), 1e-09))

def build_engine(cfg, arm_overrides, vad_model, socket):
    engine_cfg = dict(cfg['engine'])
    engine_cfg.update(stream_response=True, input_protocol='pcm16.ref.v1', guarded_turns=True)
    engine_cfg.update(arm_overrides)
    e = ActorEngine(websocket=socket, prompts=cfg['prompts'], delay=cfg['time'], llm_cfg=cfg.get('llm') or {}, engine_cfg=engine_cfg, llm_fn=lambda _m: '', asr_fn=lambda *_a: '', tts_fn=lambda *a: None, text_stream_fn=module.llm_qwen3o_stream, tts_stream_fn=silent_tts, vad_model=vad_model, request_capacity=RequestCapacity(int(engine_cfg.get('request_total_limit', 4)), int(engine_cfg.get('normal_request_limit', 3))))
    e.vad_iterator.reset_states()
    e.dispatch_asr = lambda *a, **k: None
    return e

async def drain(e, idle_target=8):
    idle = 0
    while idle < idle_target:
        await asyncio.sleep(0)
        if e.q.empty():
            idle += 1
            continue
        idle = 0
        while not e.q.empty():
            await e._process_event(e.q.get_nowait())

def first_sentence_end(e):
    speech = e._speech
    if speech is None:
        return None
    record = e._guard_outputs.get(speech.sid, {})
    for (end, text) in record.get('sentences', []):
        if text.strip()[-1:] in '。？！.?!':
            return end / (speech.rate or 24000)
    return None

async def run_episode(item, hole, arm, *, cfg, header, vad_model, latency, salt, cache, unit, settle_s=3.5):
    """Returns (record, rows). Never raises for model/transport problems: they are recorded."""
    gate = Gate(prompts=cfg['prompts'], latency=latency, salt=salt, cache=cache, unit=unit, leadin=item['context'])
    token = _GATE.set(gate)
    sock = Socket()
    e = build_engine(cfg, arm['engine'], vad_model, sock)
    rows = []
    observe = e._observe

    def traced(event, data=None):
        rows.append([event, round(e.t_audio, 3), dict(data or {})])
        return observe(event, data)
    e._observe = traced
    begun = []
    begin = e._begin_candidate

    def begin_traced(audio, **kw):
        begun.append({'t': round(e.t_audio, 3), 'stage': kw.get('stage'), 'samples': len(audio)})
        return begin(audio, **kw)
    e._begin_candidate = begin_traced
    syn = header['synthesis']
    floor = pink(SR * 90, int(hashlib.sha256(unit.encode()).hexdigest()[:8], 16), syn['floor']['dbfs'])
    echo_src = syn['echo_residual']['path']
    echo = scaled(echo_src, syn['echo_residual']['dbfs']) if syn['echo_residual']['dbfs'] is not None else None
    ref = scaled(echo_src, syn['echo_residual']['reference_rms_dbfs'])
    delay = int(SR * syn['echo_residual']['delay_ms'] / 1000)
    (user, voiced, hole_span, pair) = utterance(item, hole, header)
    (seq, t) = (0, 0.0)
    t_play = None
    t_u = None
    status = 'ok'
    wall0 = _real_time.perf_counter()

    def mic_frame(i):
        a = i * FRAME
        x = floor[a % len(floor):a % len(floor) + FRAME].copy()
        if len(x) < FRAME:
            x = np.resize(x, FRAME)
        r = np.zeros(FRAME, np.float32)
        if t_play is not None:
            k = a - int(t_play * SR)
            if k >= 0:
                r = np.take(ref, np.arange(k, k + FRAME) % len(ref)).astype(np.float32)
            if echo is not None and k - delay >= 0:
                x += np.take(echo, np.arange(k - delay, k - delay + FRAME) % len(echo)).astype(np.float32)
        if t_u is not None:
            k = a - int(round(t_u * SR))
            if k + FRAME > 0 and k < len(user):
                (lo, hi) = (max(0, k), min(len(user), k + FRAME))
                x[lo - k:hi - k] += user[lo:hi]
        return (np.clip(x, -1, 1), r)

    async def step():
        nonlocal seq, t
        seq += 1
        t = round(seq * FRAME / SR, 6)
        gate.now = t
        await gate.release(t)
        await drain(e)
        (x, r) = mic_frame(seq - 1)
        await e._process_event(FrameEvent(seq, t, None, x, r))
        await drain(e)

    async def ack():
        speech = e._speech
        if item['playing'] and t_play is not None and (seq % 3 == 0) and (speech is not None) and speech.rate:
            played = min(speech.sent, max(speech.played, int(speech.rate * (t - t_play))))
            await e._process_event(ControlMsg('playback_progress', {'utterance_id': speech.sid, 'played_samples': played, 'started': True}))
    try:
        if item['playing']:
            real_detect = e.detect_vad_frame
            script = {1: {'start': FRAME / SR}, 2: {'end': 2 * FRAME / SR}}
            e.detect_vad_frame = lambda chunk: (real_detect(chunk), script.get(seq))[1]
            while not (e._speech is not None and e.STATE == 'SPEAK' and (e._speech.sent > 0)):
                await step()
                if t > 4:
                    raise RuntimeError('lead-in never reached playback')
            await e._process_event(ControlMsg('playback_progress', {'utterance_id': e._speech.sid, 'played_samples': 0, 'started': True}))
            await drain(e)
            t_play = t
            del e.detect_vad_frame
            e._users_by_turn[0] = item['context']['previous_user'] or ('你好' if item['lang'] == 'zh' else 'Hello')
            gate.phase = 'test'
            while True:
                first = first_sentence_end(e)
                target = t_play + (first + 0.3 if first is not None and first < 8 else 2.0)
                if (first is not None or t - t_play >= 8) and t >= target - 1e-09:
                    break
                await ack()
                await step()
            t_u = round(t + FRAME / SR, 6)
        else:
            gate.phase = 'test'
            t_u = 0.64
        t_end = t_u + len(user) / SR + settle_s
        while t < t_end:
            await ack()
            await step()
        for _ in range(3):
            await gate.release(t)
            await drain(e)
    except RuntimeError as exc:
        status = 'error:' + str(exc)[:120]
    finally:
        n_trace = len(e.trace)
        span = getattr(e, '_guard_input', None)
        end_window = {'hold_open': bool(span is not None and getattr(span, 'hold_until', None) is not None), 'deferred_left': len(getattr(span, 'deferred', None) or []) if span is not None else 0, 'span_undecided': bool(span is not None and (not span.decided))}
        e._reset_session()
        await asyncio.gather(*e._guard_tasks, *e._speech_tasks, *e._candidate_tasks, return_exceptions=True)
        for entry in gate.pending:
            if not entry['fut'].done():
                entry['fut'].cancel()
        await drain(e, 4)
        _GATE.reset(token)
    controls = [[ev.get('event'), (ev.get('data') or {}).get('t_audio'), ev.get('data') or {}] for ev in e.trace[:n_trace]]
    calls = [{k: c.get(k) for k in ('role', 't', 'latency', 'due', 'timeout', 'cancelled', 'cache_hit', 'wall_ms', 'error', 'out')} | {'key': c['key'][:16]} for c in gate.calls]
    timing = {'t_play': t_play, 't_u': t_u, 'voiced': [round(t_u + voiced[0], 3), round(t_u + voiced[1], 3)], 'hole': [round(t_u + hole_span[0], 3), round(t_u + hole_span[1], 3)] if hole_span else None, 'pair': {k: round(t_u + v, 3) for (k, v) in pair.items()}, 't_last': t, 'end_window': end_window, 'wall_s': round(_real_time.perf_counter() - wall0, 3)}
    return ({'status': status, 'timing': timing, 'calls': calls, 'begun': begun}, rows, controls)
