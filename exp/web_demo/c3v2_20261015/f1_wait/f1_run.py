# Migrated to c3v2; no artifact integrity hashing.
"""f1_wait A/B runner: serial headless-Chromium trials against an isolated backend
started from THIS worktree (production VAD, GuardedTurns, route/reply prompts, Omni,
TTS, websocket, playback ACKs). Only f1_backend.py's fault/budget shims are added.

Arms are config overrides read from arms.json. Results append one JSON row per
(arm, rep, trial) to RUN/results.jsonl; finished rows are skipped on restart, so
the run can be killed and resumed. Each trial's full server trace and diagnostics
are moved into RUN/trials/<arm>/r<rep>/<trial_id>/.

usage: python f1_run.py --run RUN_DIR --arms off,on_max1 [--reps 2] [--scenarios ...]
       [--trials id,id] [--limit-per-scenario N] [--budget N] [--port 18131]
"""
import argparse
import asyncio
import base64
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
import httpx
import yaml
from playwright.async_api import async_playwright
ROOT = Path('/root/autodl-tmp/fd-badcat-c3v2')
EXP = ROOT / 'exp/web_demo/c3v2_20261015/f1_wait'
PY = '/root/miniconda3/envs/fd-sds/bin/python'
CHROME = '/root/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome'
OBSERVE_S = 8.0
DEAD_S = 6.0
FOLLOWUP_WAIT_S = 15.0
RECOVER_MARKS = {'zh': ('再说一遍',), 'en': ("didn't catch", 'say it again')}

def recover_lang(text):
    """Which fixed recovery sentence (if any) a text contains."""
    text = text or ''
    for (lang, marks) in RECOVER_MARKS.items():
        if any((m in text for m in marks)):
            return lang
    return None

def prod_connections():
    """harness_fixes.md H7: established TCP connections to the production backend
    (127.0.0.1:18000). Real users reach it through an SSH tunnel, so >0 means the
    shared Omni may be serving a person during this trial. Read-only."""
    try:
        out = subprocess.run(['ss', '-Htn', 'state', 'established', '( sport = :18000 )'], capture_output=True, text=True, timeout=5).stdout
        return sum((1 for line in out.splitlines() if line.strip()))
    except Exception:
        return None
INIT = "(() => {\n  window.__f1 = {echo: false};\n  window.__tracks = [];\n  navigator.mediaDevices.getUserMedia = async () => {\n    const ctx = new AudioContext({sampleRate: 16000}); await ctx.resume();\n    const dst = ctx.createMediaStreamDestination();\n    window.__micContext = ctx; window.__micDestination = dst;\n    window.__tracks.push(...dst.stream.getTracks()); return dst.stream;\n  };\n  window.__inject = async (encoded) => {\n    const data = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));\n    const buffer = await window.__micContext.decodeAudioData(data.buffer);\n    const source = window.__micContext.createBufferSource(); source.buffer = buffer;\n    source.connect(window.__micDestination); await window.__micContext.resume();\n    const start = window.__micContext.currentTime + .1;\n    const startClient = performance.now() + 100;\n    source.start(start);\n    return {duration: buffer.duration, end_client_ms: startClient + buffer.duration * 1000};\n  };\n  // Echo-only stimulus: mix a 96 ms delayed copy of the real render reference into\n  // the uploaded mic channel (same construction as scripts/check_guarded_interactions.py).\n  const Native = WebSocket;\n  window.WebSocket = class extends Native {\n    constructor(...args) { super(...args); this.__refs = []; }\n    send(data) {\n      if (typeof data === 'string') {\n        const v = JSON.parse(data);\n        if (v.event === 'config') { v.data.diagnostic_capture = true; data = JSON.stringify(v); }\n      } else if (data instanceof ArrayBuffer && data.byteLength > 16) {\n        const view = new DataView(data);\n        if (view.getUint32(0, true) === 0x314d4446) {\n          const n = (data.byteLength - 16) / 4, ref = new Int16Array(n);\n          for (let i = 0; i < n; i++) ref[i] = view.getInt16(18 + i * 4, true);\n          this.__refs.push(ref);\n          if (this.__refs.length > 6) {\n            const d = this.__refs.shift();\n            if (window.__f1.echo) for (let i = 0; i < Math.min(n, d.length); i++)\n              view.setInt16(16 + i * 4, Math.max(-32768, Math.min(32767, view.getInt16(16 + i * 4, true) + d[i] * .35)), true);\n          }\n        }\n      }\n      return super.send(data);\n    }\n  };\n})();"

def rows(path):
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out

def append(path, row):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')

def load_manifest():
    manifest = json.loads((EXP / 'manifest.json').read_text())
    for clip in manifest['clips'].values():
        if not Path(clip['path']).is_file():
            raise SystemExit(f"missing audio: {clip['id']}")
    return (manifest, None)

def build_config(arm_overrides, run_dir, arm):
    cfg = yaml.safe_load((ROOT / 'src/config.yaml').read_text())
    for (section, values) in yaml.safe_load((ROOT / 'configs/demo_chat.yaml').read_text()).items():
        cfg.setdefault(section, {}).update(values)
    cfg['engine'].update(stream_response=True, warmup=False, diagnostics_retention_days=None, case_capture_dir=str(run_dir / 'cases'))
    cfg['engine'].update(arm_overrides)
    path = run_dir / f'config_{arm}.yaml'
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    path.chmod(384)
    return path

def resume_latency(after, plays, resume_ms):
    """Resumed input's final VAD end -> first non-recovery playback after it (ms)."""
    if resume_ms is None:
        return None
    sid = next((d.get('input_id') for (e, d, t) in after if e == 'vad_start' and t >= resume_ms), None)
    ends = [t for (e, d, t) in after if e == 'vad_done' and d.get('input_id') == sid]
    if not ends:
        return None
    play = next((p for p in plays if p['ms'] > ends[-1] and (not p['recover'])), None)
    return None if play is None else round(play['ms'] - ends[-1], 1)

def summarize(trace, client, trial, marks, calls):
    """Authoritative metrics from the server trace of one trial."""
    ev = [(r.get('event'), r.get('data') or {}, r.get('server_ms', 0.0)) for r in trace]
    armed = marks.get('armed_server_ms')
    after = [(e, d, t) for (e, d, t) in ev if armed is None or t >= armed]
    vad_done = {}
    for (e, d, t) in after:
        if e == 'vad_done':
            vad_done[d.get('input_id')] = t
    admitted = {d.get('input_id') for (e, d, t) in after if e == 'input_admitted'}
    decisions = [d for (e, d, t) in after if e == 'input_decision']
    uay = []
    for (e, d, t) in after:
        if e == 'input_decision' and d.get('closed') and (d.get('route') == 'keep') and (d.get('input_id') in admitted):
            route = (d.get('audit') or {}).get('route') or {}
            uay.append({'input_id': d['input_id'], 'decision_ms': t, 'end_ms': vad_done.get(d['input_id'], t), 'timed_out': route.get('timed_out'), 'fallback': route.get('fallback'), 'error': (d.get('audit') or {}).get('error'), 'transcript': route.get('transcript')})
    (recover_cids, last_recovery) = (set(), None)
    for (e, d, t) in after:
        if e == 'input_cut_recovery':
            last_recovery = t
        elif e == 'candidate_created' and last_recovery is not None and (d.get('audio_samples') == 0) and (t - last_recovery < 1000):
            recover_cids.add(d.get('candidate_id'))
            last_recovery = None
    plays = [{'ms': t, 'candidate_id': d.get('candidate_id'), 'utterance_id': d.get('utterance_id'), 'recover': d.get('candidate_id') in recover_cids} for (e, d, t) in after if e == 'speech_playback_started']
    cut = uay[0] if uay else None
    first_play_after_cut = None
    if cut:
        first_play_after_cut = next((p for p in plays if p['ms'] > cut['end_ms']), None)
    test_sid = marks.get('test_input_id')
    test_end = vad_done.get(test_sid)
    followup_ms = marks.get('followup_server_ms')
    follow_play = None
    if followup_ms is not None:
        follow_play = next((p for p in plays if p['ms'] > followup_ms and (not p['recover'])), None)
    route_audits = [(d.get('audit') or {}).get('route') or {} for d in decisions]
    follow_sid = None
    if followup_ms is not None:
        follow_sid = next((d.get('input_id') for (e, d, t) in after if e == 'vad_start' and t >= followup_ms), None)
    follow_dispatch = [d for (e, d, t) in after if e == 'input_dispatch' and d.get('input_id') == follow_sid]
    setup_utt = marks.get('setup_utterance_id')
    res = {'uay': len(uay), 'uay_detail': uay, 'recovery_events': [d for (e, d, t) in after if e == 'input_cut_recovery'], 'recovery_skipped': [d for (e, d, t) in after if e == 'input_cut_recovery_skipped'], 'recover_playbacks': sum((p['recover'] for p in plays)), 'answer_playbacks': sum((not p['recover'] for p in plays)), 'dead_silence': None if not cut else first_play_after_cut is None or first_play_after_cut['ms'] - cut['end_ms'] > DEAD_S * 1000, 'cut_to_first_playback_ms': None if not (cut and first_play_after_cut) else round(first_play_after_cut['ms'] - cut['end_ms'], 1), 'first_play_after_cut_is_recover': None if not first_play_after_cut else first_play_after_cut['recover'], 'test_input_id': test_sid, 'test_closed_routes': [d.get('route') for d in decisions if d.get('input_id') == test_sid and d.get('closed')], 'test_interim_routes': [d.get('route') for d in decisions if d.get('input_id') == test_sid and (not d.get('closed'))], 'test_admitted': test_sid in admitted, 'test_end_to_first_answer_ms': next((round(p['ms'] - test_end, 1) for p in plays if test_end is not None and p['ms'] > test_end and (not p['recover'])), None), 'resume_end_to_first_answer_ms': resume_latency(after, plays, marks.get('resume_server_ms')), 'cancelled_setup': any((e == 'speech_cancelled' and d.get('utterance_id') == setup_utt for (e, d, t) in after)), 'waiting': [d.get('reason') for (e, d, t) in after if e == 'input_waiting'], 'released': sum((e == 'input_wait_released' for (e, d, t) in after)), 'rejected': [d.get('reason') for (e, d, t) in after if e == 'input_rejected'], 'followup_answered': None if followup_ms is None else follow_play is not None, 'followup_latency_ms': None if not follow_play else round(follow_play['ms'] - followup_ms, 1), 'followup_context_source': follow_dispatch[-1].get('context_source') if follow_dispatch else None, 'followup_context_samples': follow_dispatch[-1].get('context_audio_samples') if follow_dispatch else None, 'followup_transcript': next((((d.get('audit') or {}).get('route') or {}).get('transcript') for d in reversed(decisions) if d.get('input_id') == follow_sid and d.get('closed')), None), 'budget_refused': sum((1 for c in calls if c.get('refused'))), 'errors': [e for (e, d, t) in ev if e in ('speech_error', 'engine_error', 'model_call_error')], 'decisions_timed_out': sum((bool(a.get('timed_out')) for a in route_audits)), 'decisions_fallback': sum((bool(a.get('fallback')) for a in route_audits)), 'decisions_error': sum((bool((d.get('audit') or {}).get('error')) for d in decisions)), 'stale_decisions': sum((e == 'input_decision_stale' for (e, d, t) in after)), 'model_calls': sum((e == 'model_call_dispatch' for (e, d, t) in ev)), 'omni_requests': sum((1 for c in calls if c.get('omni'))), 'injected_faults': [c for c in calls if c.get('injected')]}
    texts = {}
    for m in client:
        if m.get('event') == 'speech_text_delta':
            dd = m.get('data') or {}
            texts[dd.get('utterance_id')] = texts.get(dd.get('utterance_id'), '') + (dd.get('text') or '')
    rec_texts = [t for t in texts.values() if recover_lang(t)]
    res['client_recover_text'] = rec_texts[:2]
    res['client_recover_lang'] = [recover_lang(t) for t in rec_texts]
    allowed = [x for x in (marks.get('test_input_id'), marks.get('second_input_id')) if x is not None]
    res['cut_is_test_span'] = None if not cut else cut['input_id'] == test_sid
    res['injected_outside_test'] = None if trial.get('echo') else [c.get('input_id') for c in res['injected_faults'] if c.get('input_id') not in allowed]
    res['item_lang'] = 'en' if '_en_' in trial.get('item', '') else 'zh' if '_zh_' in trial.get('item', '') else None
    return res

def budget_status(row, calls):
    """A refused Omni call looks like a model fallback, so it must never be scored.
    Refusals that only hit the follow-up void the follow-up fields; earlier ones void
    the whole trial (status budget_cut, retried on resume)."""
    refused = [c['wall'] for c in calls if c.get('refused')]
    if row.get('status') != 'ok' or not refused:
        return
    marks = row.get('marks') or {}
    follow = marks.get('followup_wall')
    if follow is None and marks.get('followup_server_ms') is not None:
        follow = row['started_wall'] + marks['followup_server_ms'] / 1000
    if follow is not None and min(refused) >= follow:
        row.update(followup_budget_cut=True, followup_answered=None, followup_latency_ms=None, followup_transcript=None)
    else:
        row['status'] = 'budget_cut'

class Trial:

    def __init__(self, browser, url, manifest, t, arm, rep, run_dir, fault_file, call_log):
        (self.browser, self.url, self.m, self.t) = (browser, url, manifest, t)
        (self.arm, self.rep, self.run_dir) = (arm, rep, run_dir)
        (self.fault_file, self.call_log) = (fault_file, call_log)
        self.key = f"{arm}|r{rep}|{t['trial_id']}"
        self.out = run_dir / 'trials' / arm / f'r{rep}' / t['trial_id']
        (self.events, self.client, self.errors) = ([], [], [])

    def fault(self, fault='none', max_spans=1, input_ids=None):
        spec = {'trial': self.key, 'fault': fault, 'max_spans': max_spans}
        if input_ids is not None:
            spec['input_ids'] = list(input_ids)
        self.fault_file.write_text(json.dumps(spec))

    async def next_input_id(self, start, seconds=3.0):
        await self.until(lambda : any((e.get('event') == 'vad_start' for e in self.events[start:])), seconds)
        starts = [e for e in self.events[start:] if e.get('event') == 'vad_start']
        return starts[0]['data'].get('input_id') if starts else None

    def of(self, kind, start=0):
        return [e for e in self.events[start:] if e.get('event') == kind]

    async def until(self, fn, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if fn():
                return True
            if self.errors:
                raise RuntimeError('pageerror: ' + self.errors[0])
            await asyncio.sleep(0.04)
        return False

    async def inject(self, clip_id):
        data = Path(self.m['clips'][clip_id]['path']).read_bytes()
        return await self.page.evaluate('x => window.__inject(x)', base64.b64encode(data).decode())

    def trace(self, session):
        return rows(ROOT / 'exp' / session / 'realtimeout_live/events.jsonl')

    def server_ms_now(self, session):
        tr = self.trace(session)
        return tr[-1].get('server_ms') if tr else None

    async def run(self):
        t = self.t
        self.out.mkdir(parents=True, exist_ok=True)
        row = {'arm': self.arm, 'rep': self.rep, 'trial_id': t['trial_id'], 'scenario': t['scenario'], 'group': t['group'], 'item': t['item'], 'fault': t['fault'], 'expect': t['expect'], 'resume_delay_s': t['resume_delay_s'], 'started_wall': time.time(), 'status': 'error', 'prod_conn_start': prod_connections()}
        marks = {}
        context = await self.browser.new_context(permissions=['microphone'])
        await context.add_init_script(INIT)
        self.page = page = await context.new_page()
        page.on('pageerror', lambda e: self.errors.append(str(e)))

        def ws_open(ws):

            def received(v):
                if isinstance(v, str):
                    self.events.append(dict(observed_ms=time.monotonic() * 1000, **json.loads(v)))

            def sent(v):
                if isinstance(v, str):
                    x = json.loads(v)
                    if x.get('event') in ('playback_progress', 'playback_stopped', 'demo_telemetry'):
                        self.client.append(dict(observed_ms=time.monotonic() * 1000, **x))
            ws.on('framereceived', received)
            ws.on('framesent', sent)
        page.on('websocket', ws_open)
        session = None
        try:
            self.fault()
            await page.goto(self.url + '/demo/')
            await page.click('#start')
            if not await self.until(lambda : bool(self.of('demo_ready')), 30):
                raise TimeoutError('demo_ready')
            session = row['session_id'] = self.of('demo_ready')[0]['data']['session_id']
            await self.inject(t['setup'])
            if not await self.until(lambda : bool(self.of('speech_playback_started')), 45):
                raise TimeoutError('setup playback')
            await asyncio.sleep(0.8)
            marks['setup_utterance_id'] = self.of('speech_playback_started')[-1]['data']['utterance_id']
            if self.of('speech_played'):
                row['setup_ended_before_test'] = True
            before = len(self.events)
            marks['armed_server_ms'] = self.server_ms_now(session)
            if t['echo']:
                self.fault(t['fault'], t.get('max_spans', 1))
                await page.evaluate('window.__f1.echo = true')
                await asyncio.sleep(6.0)
                await page.evaluate('window.__f1.echo = false')
                end_wall = time.monotonic()
                marks['test_input_id'] = await self.next_input_id(before, 0)
            else:
                self.fault(t['fault'], t.get('max_spans', 1), input_ids=[])
                stamp = await self.inject(t['item'])
                end_wall = time.monotonic() + 0.1 + stamp['duration']
                row['test_audio_s'] = round(stamp['duration'], 3)
                sid = marks['test_input_id'] = await self.next_input_id(before)
                allowed = [sid] if sid is not None else []
                self.fault(t['fault'], t.get('max_spans', 1), input_ids=allowed)
                delay = t['resume_delay_s']
                if isinstance(delay, (int, float)):
                    await asyncio.sleep(max(0, end_wall - time.monotonic()) + delay)
                    marks['resume_server_ms'] = self.server_ms_now(session)
                    self.fault()
                    stamp2 = await self.inject(t['resume_item'])
                    end_wall = time.monotonic() + 0.1 + stamp2['duration']
                elif delay == 'on_recover':
                    n0 = len(self.of('speech_playback_started'))
                    await asyncio.sleep(max(0, end_wall - time.monotonic()))
                    await self.until(lambda : len(self.of('speech_playback_started')) > n0, 3.0)
                    await asyncio.sleep(0.3)
                    marks['second_cut_server_ms'] = self.server_ms_now(session)
                    mark = len(self.events)
                    stamp2 = await self.inject(t['resume_item'])
                    end_wall = time.monotonic() + 0.1 + stamp2['duration']
                    sid2 = marks['second_input_id'] = await self.next_input_id(mark)
                    if sid2 is not None:
                        allowed.append(sid2)
                        self.fault(t['fault'], t.get('max_spans', 1), input_ids=allowed)
            await asyncio.sleep(max(0, end_wall + OBSERVE_S - time.monotonic()))
            self.fault()
            if t['followup']:
                marks['followup_server_ms'] = self.server_ms_now(session)
                marks['followup_wall'] = time.time()
                n0 = len(self.of('speech_playback_started'))
                await self.inject(t['item'])
                await self.until(lambda : len(self.of('speech_playback_started')) > n0, FOLLOWUP_WAIT_S)
                await asyncio.sleep(1.0)
            self.fault()
            try:
                page_text = await page.inner_text('#messages', timeout=3000)
            except Exception as exc:
                page_text = f'<<inner_text failed: {exc!r}>>'
            (self.out / 'page_text.txt').write_text(page_text)
            row['page_recover_lang'] = sorted({recover_lang(line) for line in page_text.splitlines() if recover_lang(line)})
            await page.click('#stop')
            await asyncio.sleep(0.8)
            row['status'] = 'ok'
        except Exception as exc:
            row['harness_error'] = repr(exc)
        finally:
            self.fault()
            try:
                await context.close()
            except Exception:
                pass
        if session:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if any(r.get('event') == 'trace_closed' for r in self.trace(session)):
                    break
                await asyncio.sleep(.1)
        trace = self.trace(session) if session else []
        calls = [c for c in rows(self.call_log) if c.get('trial') == self.key]
        row['marks'] = marks
        row['page_errors'] = self.errors
        row['trace_closed'] = any((r.get('event') == 'trace_closed' for r in trace))
        row.update(summarize(trace, self.events, t, marks, calls))
        row['finished_wall'] = time.time()
        row['prod_conn_end'] = prod_connections()
        budget_status(row, calls)
        if session and (ROOT / 'exp' / session).exists():
            shutil.move(str(ROOT / 'exp' / session), str(self.out / 'session'))
        (self.out / 'client_events.json').write_text(json.dumps({'events': self.events, 'client': self.client}, ensure_ascii=False))
        (self.out / 'receipt.json').write_text(json.dumps(row, ensure_ascii=False, indent=1))
        return row

async def start_backend(cfg_path, port, run_dir, budget, fault_file, call_log):
    env = dict(os.environ)
    for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY'):
        env.pop(k, None)
    env.update(QWEN_HOST='127.0.0.1', QWEN_PORT='10003', FDBC_PROXY_HOST='127.0.0.1', FDBC_PROXY_PORT='10004', FDBC_VLLM_URL='http://127.0.0.1:10003/v1/chat/completions', FDBC_DEMO_VOICE_ADAPTER='1', PYTHONUNBUFFERED='1', FDBC_DEMO_TALKER_NUMERICS='native', FDBC_DEMO_CODEC_CHUNKS='4:12', FDBC_QWEN_URL='http://127.0.0.1:10003/v1/chat/completions', FDBC_OMNI_TTS_URL='http://127.0.0.1:10003/v1/chat/completions', FDBC_DEMO_INFERENCE_MODE='direct', NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost', F1_ROOT=str(ROOT), F1_FAULT_FILE=str(fault_file), F1_CALL_LOG=str(call_log), F1_OMNI_BUDGET=str(budget))
    log = (run_dir / 'backend.log').open('a')
    proc = subprocess.Popen([PY, str(EXP / 'f1_backend.py'), str(cfg_path), str(port)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(120):
            try:
                r = await client.get(f'http://127.0.0.1:{port}/demo/', timeout=2)
                if r.status_code == 200:
                    return (proc, log)
            except Exception:
                pass
            if proc.poll() is not None:
                break
            await asyncio.sleep(1)
    stop_backend(proc, log)
    raise RuntimeError('isolated backend failed to start; see backend.log')

def stop_backend(proc, log):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    log.close()

def resummarize(run):
    """Recompute metrics from saved trial artefacts (no backend, no Omni)."""
    calls = rows(run / 'calls.jsonl')
    out = run / 'results_resummarized.jsonl'
    out.write_text('')
    manifest = {t['trial_id']: t for t in json.loads((EXP / 'manifest.json').read_text())['trials']}
    for r in rows(run / 'results.jsonl'):
        d = run / 'trials' / r['arm'] / f"r{r['rep']}" / r['trial_id']
        trace = []
        for p in d.glob('session/*/events.jsonl'):
            trace = rows(p)
        client = json.loads((d / 'client_events.json').read_text())['events'] if (d / 'client_events.json').exists() else []
        key = f"{r['arm']}|r{r['rep']}|{r['trial_id']}"
        r.update(summarize(trace, client, manifest[r['trial_id']], r.get('marks') or {}, [c for c in calls if c.get('trial') == key]))
        budget_status(r, [c for c in calls if c.get('trial') == key])
        append(out, r)
    print(f'resummarized -> {out}')

async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--arms', default='off,on_max1')
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--scenarios', default='')
    ap.add_argument('--trials', default='')
    ap.add_argument('--limit-per-scenario', type=int, default=0)
    ap.add_argument('--budget', type=int, default=0, help='hard cap on real Omni requests for this run dir')
    ap.add_argument('--port', type=int, default=18131)
    ap.add_argument('--max-attempts', type=int, default=2)
    ap.add_argument('--resummarize', action='store_true', help='recompute metrics offline and exit')
    args = ap.parse_args()
    if args.resummarize:
        return resummarize(args.run)
    (manifest, digest) = load_manifest()
    arms_cfg = json.loads((EXP / 'arms.json').read_text())
    arms = [a for a in args.arms.split(',') if a]
    for a in arms:
        if a not in arms_cfg:
            raise SystemExit(f'unknown arm {a}')
    trials = manifest['trials']
    if args.scenarios:
        keep = set(args.scenarios.split(','))
        trials = [t for t in trials if t['scenario'] in keep]
    if args.trials:
        keep = set(args.trials.split(','))
        trials = [t for t in trials if t['trial_id'] in keep]
    if args.limit_per_scenario:
        seen = {}
        trials = [t for t in trials if seen.setdefault(t['scenario'], []).append(t) or len(seen[t['scenario']]) <= args.limit_per_scenario]
    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    (run / 'run_args.json').write_text(json.dumps({**vars(args), 'run': str(run), 'trials': [t['trial_id'] for t in trials]}, indent=1))
    results = run / 'results.jsonl'
    (fault_file, call_log) = (run / 'fault.json', run / 'calls.jsonl')
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, executable_path=CHROME, args=['--no-sandbox', '--autoplay-policy=no-user-gesture-required', '--no-proxy-server'])
        try:
            for rep in range(1, args.reps + 1):
                order = arms if rep % 2 else list(reversed(arms))
                for arm in order:
                    done = {}
                    for r in rows(results):
                        k = (r['arm'], r['rep'], r['trial_id'])
                        done[k] = done.get(k, 0) + 1 if r.get('status') != 'ok' else 10 ** 6
                    pending = [t for t in trials if done.get((arm, rep, t['trial_id']), 0) < args.max_attempts]
                    if not pending:
                        continue
                    (block_t0, block_rows) = (time.time(), [])
                    cfg = build_config(arms_cfg[arm], run, arm)
                    (proc, log) = await start_backend(cfg, args.port, run, args.budget, fault_file, call_log)
                    try:
                        for t in pending:
                            row = await Trial(browser, f'http://127.0.0.1:{args.port}', manifest, t, arm, rep, run, fault_file, call_log).run()
                            append(results, row)
                            block_rows.append(row)
                            print(json.dumps({k: row.get(k) for k in ('arm', 'rep', 'trial_id', 'status', 'uay', 'dead_silence', 'recover_playbacks', 'answer_playbacks', 'omni_requests', 'harness_error')}, ensure_ascii=False), flush=True)
                            if any(('budget' in json.dumps(c) for c in rows(call_log)[-3:])):
                                raise SystemExit('Omni budget exhausted; stopping')
                    finally:
                        stop_backend(proc, log)
                        print('[block] ' + json.dumps({'arm': arm, 'rep': rep, 'trials': len(block_rows), 'ok': sum((r.get('status') == 'ok' for r in block_rows)), 'not_ok': sum((r.get('status') != 'ok' for r in block_rows)), 'elapsed_s': round(time.time() - block_t0, 1), 'block_omni': sum((r.get('omni_requests') or 0 for r in block_rows)), 'cum_omni': sum((1 for c in rows(call_log) if c.get('omni'))), 'prod_conn_trials': sum((bool(r.get('prod_conn_start') or r.get('prod_conn_end')) for r in block_rows)), 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}), flush=True)
        finally:
            await browser.close()
if __name__ == '__main__':
    asyncio.run(main())
