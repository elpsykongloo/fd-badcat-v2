"""HumDial test (FDBench nondev, noisy) whole-dialogue run on the audio clock.

One unit = one test recording. The whole recording is the microphone from t=0.64 s;
the engine starts in listening mode and every model role is the real Omni call:
input_route / input_reply latency from the FP8 pool, every other role (response,
completion repair, ...) a fixed latency (response 0.45 s). TTS is silent PCM with a
text-length duration; a simulated browser acks playback every 3 frames
(played = min(sent, 24 kHz x elapsed)) and reports `ended` when the clip is done, so
playback_autoend closes the turn as in the demo. Browser AEC is assumed perfect (no
echo in the mic; the render reference is attached while playing) and not reported,
so `input_vad_end_ms_playing_aec_only` arms do not extend (production default).
Arms of one unit share one Omni cache (identical payload -> identical answer).

usage: python full_harness.py --out DIR [--langs cn,en] [--cats ...] [--limit N]
                              [--arms arms.json] [--procs 8 --conc 3] [--uids ...]
"""
import argparse
import asyncio
import glob
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = Path('/root/autodl-tmp/fd-badcat/data/HumDial-FDBench/extracted/test')
FIXED = {'response': 0.45}
OTHER_S = 0.30
KEEP_ROWS = {'input_dispatch', 'input_decision', 'input_decision_stale', 'input_decision_deferred',
             'input_vad_end_merged', 'input_admitted', 'input_rejected', 'input_wait_released',
             'input_context_expired', 'speech_error', 'engine_error', 'response_completion_repair',
             'input_recover', 'input_wait_recover'}
KEEP_CONTROLS = {'vad_start', 'vad_done', 'speech_cancelled', 'input_waiting', 'input_ignored',
                 'input_notice', 'speech_start', 'turn_finished', 'speech_played'}


def build_manifest(langs, cats):
    items = []
    for lang in langs:
        for cat_dir in sorted((DATA / f'{lang}_test_nondev').iterdir()):
            if cats and cat_dir.name not in cats:
                continue
            for meta in sorted(cat_dir.glob('*.json')):
                if meta.name.startswith('clean') or meta.name.endswith('_timestamp.json'):
                    continue
                wav = meta.with_suffix('.wav')
                if not wav.is_file():
                    continue
                d = json.loads(meta.read_text(encoding='utf-8'))
                items.append({'uid': f'{lang}/{cat_dir.name}/{meta.stem}', 'lang': lang,
                              'category': cat_dir.name, 'wav': str(wav),
                              'duration': d['final_duration'],
                              'segments': [[s['xmin'], s['xmax'], s['text']] for s in d['speech_segments']]})
    return items


def load_wav(path):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype('float32')
    return x


def install(H):
    """Full-dialogue Gate/Cache on top of the f2 harness primitives."""
    from stream_transport import TextDelta
    import async_utils
    # Wall-clock timeouts outside control_labels (speech_stream credit wait 5 s, text/TTS
    # read 15 s, ...) would fire while the harness blocks on a real Omni call with the
    # audio clock frozen. Under an active gate they are dropped (everything they guard is
    # driven by the audio clock here); control_labels keeps its audio-clock deadline.
    orig = async_utils.cancellable_wait

    async def no_wall_timeout(awaitable, timeout):
        return await orig(awaitable, None if H._GATE.get() is not None else timeout)
    for mod in list(sys.modules.values()):
        if mod is not None and mod is not H.control_labels and getattr(mod, 'cancellable_wait', None) is orig:
            mod.cancellable_wait = no_wall_timeout
            print('no-wall-timeout patched', mod.__name__, flush=True)

    class FullCache(H.OmniCache):
        async def get(self, key, messages, kwargs, role, unit):
            hit = self.entries.get(key)
            if hit is not None and not hit.get('error'):
                return hit, True
            self.budget.take()
            entry = {'attempts': 0}
            async with self.sem:
                for attempt in range(3):
                    entry['attempts'] = attempt + 1
                    began = H._real_time.perf_counter()
                    try:
                        text, finish = [], None
                        async for p in H._ORIG_STREAM(messages, route=kwargs.get('route', False),
                                                      max_tokens=kwargs.get('max_tokens'),
                                                      continue_final_message=kwargs.get('continue_final_message', False),
                                                      report_finish=True):
                            if getattr(p, 'finish_reason', None) is not None:
                                finish = p.finish_reason
                            if p:
                                text.append(str(p))
                        entry.update(text=''.join(text), finish=finish,
                                     wall_ms=round((H._real_time.perf_counter() - began) * 1000, 1))
                        entry.pop('error', None)
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        entry.update(error=type(exc).__name__,
                                     wall_ms=round((H._real_time.perf_counter() - began) * 1000, 1))
            self.entries[key] = entry
            if self.log is not None:
                self.log({'unit': unit, 'key': key[:16], 'role': role, 'attempts': entry['attempts'],
                          'wall_ms': entry.get('wall_ms'), 'error': entry.get('error')})
            return entry, False

    class FullGate(H.Gate):
        async def stream(self, messages, **kwargs):
            role = self.role(messages)
            payload = H.module.qwen_text_payload(messages, route=kwargs.get('route', False),
                                                 max_tokens=kwargs.get('max_tokens'),
                                                 continue_final_message=kwargs.get('continue_final_message', False))
            key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if role in ('input_route', 'input_reply'):
                latency = self.latency(role, key, self.salt)
            else:
                latency = FIXED.get(role, OTHER_S)
            deadline = H._DEADLINE.get()
            due = self.now + latency
            timeout = deadline is not None and due > deadline + 1e-09
            entry = {'role': role, 'key': key, 't': round(self.now, 3), 'latency': round(latency, 3),
                     'due': round(deadline if timeout else due, 3), 'timeout': timeout,
                     'fut': asyncio.get_running_loop().create_future(), 'messages': messages, 'kwargs': kwargs}
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
            if result.get('text'):
                yield result['text']
            if kwargs.get('report_finish'):
                yield TextDelta('', finish_reason=result.get('finish') or 'stop')

        async def release(self, t):
            for entry in sorted(self.pending, key=lambda e: e['due']):
                if entry['fut'].done() or entry['due'] > t + 1e-09:
                    continue
                if entry['timeout']:
                    entry['fut'].set_result({})
                    continue
                result, hit = await self.cache.get(entry['key'], entry['messages'], entry['kwargs'],
                                                   entry['role'], self.unit)
                text = result.get('text') or ''
                entry.update(cache_hit=hit, wall_ms=result.get('wall_ms'), error=result.get('error'),
                             finish=result.get('finish'),
                             out=text if entry['role'] not in ('input_route', 'input_reply') else text[:300])
                if not entry['fut'].done():
                    entry['fut'].set_result(result)
            self.pending = [e for e in self.pending if not e['fut'].done()]

    return FullGate, FullCache


async def run_dialogue(H, FullGate, item, user, arm, *, cfg, header, vad_model, latency, salt, cache, unit,
                       tail_s=2.0):
    import numpy as np
    from engine import ControlMsg, FrameEvent
    SR, FRAME = H.SR, H.FRAME
    if arm.get('prompts'):
        cfg = dict(cfg, prompts={**cfg['prompts'], **arm['prompts']})
    gate = FullGate(prompts=cfg['prompts'], latency=latency, salt=salt, cache=cache, unit=unit,
                    leadin={'previous_user': '', 'assistant_text': ''})
    gate.phase = 'test'
    token = H._GATE.set(gate)
    sock = H.Socket()
    e = H.build_engine(cfg, arm['engine'], vad_model, sock)
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
    floor = H.pink(SR * 90, int(hashlib.sha256(unit.encode()).hexdigest()[:8], 16), syn['floor']['dbfs'])
    ref = H.scaled(syn['echo_residual']['path'], syn['echo_residual']['reference_rms_dbfs'])
    # replay mode: the recorded browser mic and the recorded render reference are fed
    # exactly as the live engine received them (frame 1 = sample 0), no synthetic floor
    replay = item.get('ref') is not None
    rec_ref = load_wav(item['ref']) if replay else None
    t_u = 0.0 if replay else 0.64
    seq, t = 0, 0.0
    starts, acks = {}, []
    status = 'ok'
    wall0 = H._real_time.perf_counter()

    def mic_frame(i):
        a = i * FRAME
        if replay:
            x = np.zeros(FRAME, np.float32)
            seg = user[a:a + FRAME]
            x[:len(seg)] = seg
            r = np.zeros(FRAME, np.float32)
            seg = rec_ref[a:a + FRAME]
            r[:len(seg)] = seg
            return np.clip(x, -1, 1), r
        x = np.resize(floor[a % len(floor):a % len(floor) + FRAME], FRAME).astype(np.float32)
        r = np.zeros(FRAME, np.float32)
        sp = e._speech
        if sp is not None and e.STATE == 'SPEAK' and sp.sid in starts and sp.played < sp.sent:
            k = a - int(starts[sp.sid] * SR)
            if k >= 0:
                r = np.take(ref, np.arange(k, k + FRAME) % len(ref)).astype(np.float32)
        k = a - int(round(t_u * SR))
        if k + FRAME > 0 and k < len(user):
            lo, hi = max(0, k), min(len(user), k + FRAME)
            x[lo - k:hi - k] += user[lo:hi]
        return np.clip(x, -1, 1), r

    async def step():
        nonlocal seq, t
        seq += 1
        t = round(seq * FRAME / SR, 6)
        gate.now = t
        await gate.release(t)
        await H.drain(e)
        x, r = mic_frame(seq - 1)
        await e._process_event(FrameEvent(seq, t, None, x, r))
        await H.drain(e)

    async def ack():
        sp = e._speech
        if seq % 3 or sp is None or not sp.rate or sp.sent <= 0:
            return
        if sp.sid not in starts:
            starts[sp.sid] = t
        played = min(sp.sent, max(sp.played, int(sp.rate * (t - starts[sp.sid]))))
        data = {'utterance_id': sp.sid, 'played_samples': played, 'started': True}
        if played == sp.sent and e._speech_audio_done:
            data['ended'] = True
        await e._process_event(ControlMsg('playback_progress', data))

    try:
        if item.get('input_settings'):
            await e._process_event(ControlMsg('input_settings', dict(item['input_settings'])))
            await H.drain(e)
        t_end = t_u + len(user) / SR + tail_s
        while t < t_end:
            await ack()
            await step()
        for _ in range(3):
            await gate.release(t)
            await H.drain(e)
    except RuntimeError as exc:
        status = 'error:' + str(exc)[:120]
    finally:
        n_trace = len(e.trace)
        e._reset_session()
        await asyncio.gather(*e._guard_tasks, *e._speech_tasks, *e._candidate_tasks, return_exceptions=True)
        for entry in gate.pending:
            if not entry['fut'].done():
                entry['fut'].cancel()
        await H.drain(e, 4)
        H._GATE.reset(token)
    controls = [[ev.get('event'), (ev.get('data') or {}).get('t_audio'), ev.get('data') or {}]
                for ev in e.trace[:n_trace]]
    calls = [{k: c.get(k) for k in ('role', 't', 'latency', 'due', 'timeout', 'cancelled', 'cache_hit',
                                     'wall_ms', 'error', 'finish', 'out')} | {'key': c['key'][:16]}
             for c in gate.calls]
    timing = {'t_u': t_u, 't_last': t, 'starts': {str(k): v for k, v in starts.items()},
              'wall_s': round(H._real_time.perf_counter() - wall0, 3)}
    return {'status': status, 'timing': timing, 'calls': calls, 'begun': begun,
            'client_kinds': sorted({ev.get('type') or ev.get('event') or '?' for ev in sock.events})}, rows, controls


def compact(rows, controls):
    keep = []
    for ev, t, d in rows:
        if ev not in KEEP_ROWS:
            continue
        if ev == 'input_dispatch':
            d = {k: d.get(k) for k in ('input_id', 'revision', 'closed', 'playing', 'current_audio_samples',
                                       'reply_context_kind', 'vad_end_silence_ms')}
        elif ev == 'input_decision':
            audit = d.get('audit') or {}
            route = audit.get('route') or {}
            review = route.get('reply_review') or {}
            d = {'input_id': d.get('input_id'), 'revision': d.get('revision'), 'closed': d.get('closed'),
                 'route': d.get('route'), 'acoustic_reject': audit.get('acoustic_reject'),
                 'base_label': route.get('base_label'), 'transcript': (route.get('transcript') or '')[:200],
                 'fallback': route.get('fallback'), 'timed_out': route.get('timed_out'),
                 'interim_stop': route.get('interim_stop'), 'review': review.get('label'),
                 'deferred_from': audit.get('deferred_from')}
        keep.append([ev, t, d])
    ctl = [[ev, t, {k: d.get(k) for k in ('input_id', 'reason', 'utterance_id', 'text', 'turn')
                    if d.get(k) is not None}] for ev, t, d in controls if ev in KEEP_CONTROLS]
    return keep, ctl


def worker(index, jobs, args, arms, header):
    os.chdir(HERE)
    import torch
    torch.set_num_threads(1)
    import f2_harness as H
    from silero_vad import load_silero_vad
    FullGate, FullCache = install(H)
    cfg = H.load_config()
    out = Path(args.out)
    units_path, req_path = out / f'units.w{index}.jsonl', out / f'requests.w{index}.jsonl'

    def log_request(row):
        with req_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')

    async def main():
        models = asyncio.Queue()
        for _ in range(args.conc):
            models.put_nowait(load_silero_vad())
        sem = asyncio.Semaphore(args.conc)
        budget = H.Budget(None)
        queue = asyncio.Queue()
        for job in jobs:
            queue.put_nowait(job)
        done = 0

        async def consume():
            nonlocal done
            while not queue.empty():
                item, cached = queue.get_nowait()
                vad = await models.get()
                try:
                    user = load_wav(item['wav'])
                    cache = FullCache(cached, sem=sem, budget=budget, log=log_request)
                    res = {'unit': item['uid'], **{k: item[k] for k in ('lang', 'category', 'duration', 'segments')},
                           'arms': {}}
                    for name, arm in arms.items():
                        latency = H.LatencyModel('pool', Path(os.environ['F2_POOL']))
                        rec, rows, controls = await run_dialogue(
                            H, FullGate, item, user, arm, cfg=cfg, header=header, vad_model=vad,
                            latency=latency, salt=f"r0|{item['uid']}", cache=cache, unit=item['uid'])
                        keep, ctl = compact(rows, controls)
                        res['arms'][name] = {'status': rec['status'], 'timing': rec['timing'], 'calls': rec['calls'],
                                             'begun': rec['begun'], 'rows': keep, 'controls': ctl,
                                             'client_kinds': rec['client_kinds']}
                    res['cache'] = {k: {kk: v.get(kk) for kk in ('text', 'finish', 'wall_ms', 'attempts', 'error')}
                                    for k, v in cache.entries.items()}
                    res['valid'] = all(a['status'] == 'ok' and not any(c.get('error') for c in a['calls'])
                                       for a in res['arms'].values())
                finally:
                    models.put_nowait(vad)
                with units_path.open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps(res, ensure_ascii=False) + '\n')
                done += 1
                if done % 10 == 0:
                    with (out / 'progress.log').open('a') as fh:
                        fh.write(f"{time.strftime('%H:%M:%S')} w{index} {done}/{len(jobs)} omni={budget.used}\n")
        await asyncio.gather(*(consume() for _ in range(args.conc)))
        print(f'[w{index}] finished {done} units, omni requests {budget.used}', flush=True)
    asyncio.run(main())


def finished(out, arms):
    done, cache = set(), {}
    for path in glob.glob(str(Path(out) / 'seed.w*.jsonl')):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                try:
                    res = json.loads(line)
                except ValueError:
                    continue
                cache[res['unit']] = res.get('cache') or {}
    for path in glob.glob(str(Path(out) / 'units.w*.jsonl')):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                try:
                    res = json.loads(line)
                except ValueError:
                    continue
                cache[res['unit']] = res.get('cache') or {}
                if res.get('valid') and set(arms) <= set(res['arms']):
                    done.add(res['unit'])
    return done, cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--langs', default='cn,en')
    p.add_argument('--cats')
    p.add_argument('--arms', default=str(HERE / 'arms.json'))
    p.add_argument('--procs', type=int, default=8)
    p.add_argument('--conc', type=int, default=3)
    p.add_argument('--limit', type=int)
    p.add_argument('--per-cat', type=int, help='first N items of every (lang, category)')
    p.add_argument('--uids')
    p.add_argument('--replay', help='JSON list of recorded browser sessions (replay mode)')
    args = p.parse_args()
    if args.procs * args.conc > 24:
        p.error('procs * conc must stay <= 24')
    args.out = str(Path(args.out).resolve())
    arms = json.loads(Path(args.arms).read_text())
    header = json.loads((HERE.parent / 'hd_eval/train/manifest.json').read_text())['header']
    if args.replay:
        # [{uid, wav (browser_mic), ref (render_reference), input_settings, duration}]
        items = [{'lang': 'real', 'category': 'replay', 'segments': [], **it}
                 for it in json.loads(Path(args.replay).read_text())]
    else:
        items = build_manifest(args.langs.split(','), set(args.cats.split(',')) if args.cats else None)
    if args.per_cat:
        seen = {}
        items = [it for it in items if seen.setdefault((it['lang'], it['category']), []).append(1)
                 or len(seen[(it['lang'], it['category'])]) <= args.per_cat]
    if args.uids:
        want = set(args.uids.split(','))
        items = [it for it in items if it['uid'] in want]
    if args.limit:
        items = items[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done, cache = finished(out, arms)
    todo = [(it, cache.get(it['uid'], {})) for it in items if it['uid'] not in done]
    root = Path('/root/autodl-tmp/fd-badcat-r2')
    meta = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'argv': sys.argv, 'arms': arms,
            'units_total': len(items), 'units_todo': len(todo), 'latency': {'route_reply': os.environ.get('F2_POOL'),
            'response_s': FIXED['response'], 'other_s': OTHER_S},
            'git_head': subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True,
                                       text=True).stdout.strip()}
    with (out / 'run_meta.jsonl').open('a') as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + '\n')
    print(json.dumps({k: meta[k] for k in ('units_total', 'units_todo', 'git_head')}), flush=True)
    if not todo:
        return
    # longest first so the tail of the run is not one slow worker
    todo.sort(key=lambda j: -j[0]['duration'])
    procs = max(1, min(args.procs, len(todo)))
    chunks = [todo[i::procs] for i in range(procs)]
    if procs == 1:
        worker(0, chunks[0], args, arms, header)
        return
    ctx = mp.get_context('spawn')
    ps = [ctx.Process(target=worker, args=(i, chunks[i], args, arms, header)) for i in range(procs)]
    for proc in ps:
        proc.start()
    for proc in ps:
        proc.join()


if __name__ == '__main__':
    main()
