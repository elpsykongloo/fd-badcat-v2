# Migrated to c3v2; no artifact integrity hashing.
"""f2_vad experiment runner (resumable, per-unit JSONL). See f2_harness.py for the method.

A unit = (item, hole, rep). Every arm of a unit replays the identical microphone stream
through a fresh production engine; arms differ only by engine config overrides and the
latency model. Omni results are cached per unit, so identical payloads across arms get
the identical answer; each rep has its own cache (rep-to-rep jitter is measured).

usage:
  python run_f2.py --out RUN_DIR [--sets H,P,Q,L] [--holes 0,0.3,0.4,0.5] [--reps 0,1,2]
                   [--arms arms.json] [--procs 6 --conc 4] [--limit N] [--uids a,b]
                   [--max-omni N] [--dry]
arms.json: {"W0_pool": {"engine": {}, "latency": "pool"},
            "W400_pool": {"engine": {"input_vad_end_ms_playing": 400}, "latency": "pool"}, ...}
Outputs in RUN_DIR: run_meta.json, units.wN.jsonl (one line per finished unit),
requests.wN.jsonl (every real Omni request: attempts, wall ms, transport error), progress.log.
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
DEFAULT_ARMS = {f'W{w}_{lat}': {'engine': {'input_vad_end_ms_playing': w} if w else {}, 'latency': spec} for w in (0, 200, 300, 400, 500) for (lat, spec) in (('pool', 'pool'), ('idle', 'fixed:260,80'))}
KEEP_ROWS = {'input_dispatch', 'input_decision', 'input_decision_stale', 'input_decision_deferred', 'input_vad_end_merged', 'input_admitted', 'input_rejected', 'input_wait_released', 'input_context_expired', 'speech_error', 'engine_error'}
KEEP_CONTROLS = {'vad_start', 'vad_done', 'speech_cancelled', 'input_waiting', 'input_ignored', 'input_notice', 'speech_start', 'turn_finished'}

def load_manifest(run_meta=None):
    return (json.loads((HERE / 'manifest.json').read_text()), {})

def check_sources(manifest):
    for item in manifest['items']:
        for part in item['parts']:
            if not Path(part['path']).is_file():
                raise SystemExit('missing source audio: ' + part['path'])

def units_for(manifest, sets, holes, reps, uids=None):
    out = []
    for it in manifest['items']:
        if it['set'] not in sets or (uids and it['uid'] not in uids):
            continue
        if it['set'] == 'H':
            hs = [h for h in holes if h == 0 or it['holes'].get(str(h)) is not None]
        elif it['set'] == 'L':
            hs = [0.3]
        else:
            hs = [0]
        for h in hs:
            for r in reps:
                out.append((f"{it['uid']}|h{h}|r{r}", it, h, r))
    return out

def compact(rows, controls):
    keep = []
    for (ev, t, d) in rows:
        if ev not in KEEP_ROWS:
            continue
        if ev == 'input_dispatch':
            d = {k: d.get(k) for k in ('input_id', 'revision', 'closed', 'playing', 'current_audio_samples', 'context_audio_samples', 'context_source', 'reply_context_kind', 'vad_end_silence_ms', 'vad_end_merges')}
        elif ev == 'input_decision':
            audit = d.get('audit') or {}
            route = audit.get('route') or {}
            review = route.get('reply_review') or {}
            d = {'input_id': d.get('input_id'), 'revision': d.get('revision'), 'closed': d.get('closed'), 'route': d.get('route'), 'acoustic_reject': audit.get('acoustic_reject'), 'base_label': route.get('base_label'), 'transcript': (route.get('transcript') or '')[:120], 'fallback': route.get('fallback'), 'timed_out': route.get('timed_out'), 'interim_stop': route.get('interim_stop'), 'review': review.get('label'), 'review_timed_out': review.get('timed_out'), 'deferred_from': audit.get('deferred_from')}
        keep.append([ev, t, d])
    ctl = [[ev, t, {k: d.get(k) for k in ('input_id', 'reason', 'utterance_id')}] for (ev, t, d) in controls if ev in KEEP_CONTROLS]
    return (keep, ctl)

def summarize(item, rec, rows, ctl):
    tm = rec['timing']
    if rec['status'] != 'ok' or tm['t_u'] is None:
        return {'valid': False, 'status': rec['status']}
    (t_u, (v0, v1)) = (tm['t_u'], tm['voiced'])
    starts = [(t, d['input_id']) for (ev, t, d) in ctl if ev == 'vad_start' and t is not None and (t >= t_u - 0.05)]
    utt = []
    for (t, sid) in starts:
        if t <= v1 + 0.05 and sid not in utt:
            utt.append(sid)
    after = {sid for (_, sid) in starts}
    reopens = sum((1 for (i, (t, sid)) in enumerate(starts) if t <= v1 + 0.05 and sid in [s for (_, s) in starts[:i]]))
    vad_ends = sum((1 for (ev, t, d) in ctl if ev == 'vad_done' and t is not None and (t_u <= t <= v1 + 0.05)))
    dec = [(t, d) for (ev, t, d) in rows if ev == 'input_decision' and d['input_id'] in after]
    stale = sum((1 for (ev, t, d) in rows if ev == 'input_decision_stale' and d.get('input_id') in utt))
    stale_closed = sum((1 for (ev, t, d) in rows if ev == 'input_decision_stale' and d.get('input_id') in utt and d.get('closed')))
    premature = [(t, d) for (t, d) in dec if t < v1 - 1e-06 and (d['closed'] or d['route'] == 'stop_only')]
    exp = item['expected']
    premature_action = [(t, d) for (t, d) in premature if d['route'] != 'keep' and (not (exp == 'stop_only' and d['route'] == 'stop_only')) and (d['closed'] or exp == 'keep')]
    closed = [(t, d) for (t, d) in dec if d['closed'] and d['input_id'] in utt]
    final_route = closed[-1][1]['route'] if closed else 'none'
    final_t = closed[-1][0] if closed else None
    cancel = [t for (ev, t, d) in ctl if ev == 'speech_cancelled' and t is not None and (t >= t_u)]
    cancel_t = cancel[0] if cancel else None
    released = [t for (ev, t, d) in rows if ev == 'input_wait_released']
    resp = [b for b in rec['begun'] if b['stage'] == 'response' and b['t'] >= t_u]
    first = resp[0] if resp else None
    via = None if first is None else 'release' if released and first['t'] >= released[0] else 'decision'
    span = max(v1 - v0, 0.05)
    coverage = round(first['samples'] / (span * 16000), 3) if first else None
    calls = rec['calls']
    out = {'valid': True, 'n_inputs': len(utt), 'fragmented': len(utt) >= 2, 'reopens': reopens, 'vad_ends': vad_ends, 'stale': stale, 'stale_closed': stale_closed, 'premature_any': bool(premature), 'premature_action': bool(premature_action), 'premature_routes': [d['route'] for (_, d) in premature], 'final_route': final_route, 'final_t': final_t, 'cancel_t': cancel_t, 'response_t': first['t'] if first else None, 'response_via': via, 'coverage': coverage, 'released': bool(released), 'deferred': sum((1 for (ev, *_) in rows if ev == 'input_decision_deferred')), 'merged': sum((1 for (ev, *_) in rows if ev == 'input_vad_end_merged')), 'calls_route': sum((1 for c in calls if c['role'] == 'input_route')), 'calls_reply': sum((1 for c in calls if c['role'] == 'input_reply')), 'calls_cancelled': sum((1 for c in calls if c.get('cancelled'))), 'calls_timeout': sum((1 for c in calls if c['timeout'] and (not c.get('cancelled')))), 'calls_error': sum((1 for c in calls if c.get('error'))), 'fallback': sum((1 for (_, d) in dec if d.get('fallback'))), 'engine_errors': sum((1 for (ev, *_) in rows if ev in ('speech_error', 'engine_error')))}
    if isinstance(exp, dict):
        second = tm['pair']['second_start']
        responded = first is not None and first['t'] >= second - 0.05
        cov2 = first['samples'] / (max(v1 - second, 0.05) * 16000) if first else 0
        out['second_coverage'] = round(cov2, 3)
        out['behavior_ok'] = bool(cancel) == exp['cancel'] and responded == exp['respond'] and (not exp['respond'] or cov2 >= 0.85)
        out['clean_success'] = out['behavior_ok']
        out['route_correct'] = None
    else:
        if exp == 'keep':
            ok = cancel_t is None and first is None
        elif exp == 'stop_only':
            ok = cancel_t is not None and first is None
        elif exp == 'yield_wait':
            ok = cancel_t is not None and (first is None or via == 'release')
        else:
            ok = first is not None and via == 'decision' and (coverage >= 0.85)
        out['behavior_ok'] = ok
        out['route_correct'] = final_route == exp
        out['clean_success'] = bool(out['route_correct'] and ok and (not premature_action))
    out['stop_latency'] = round(cancel_t - v1, 3) if cancel_t is not None else None
    out['response_latency'] = round(first['t'] - v1, 3) if first else None
    out['decision_latency'] = round(final_t - v1, 3) if final_t is not None else None
    return out

async def run_unit(unit_id, item, hole, rep, arms, *, cfg, header, models, H, cache_entries, sem, budget, dry, log_request):
    vad_model = await models.get()
    try:
        cache = H.OmniCache(cache_entries, sem=sem, budget=budget, dry=dry, log=log_request)
        result = {'unit': unit_id, 'uid': item['uid'], 'set': item['set'], 'category': item['category'], 'root': item['root'], 'lang': item['lang'], 'hole': hole, 'rep': rep, 'expected': item['expected'], 'arms': {}}
        for (name, arm) in arms.items():
            latency = H.LatencyModel(arm['latency'], Path(os.environ.get('F2_POOL', str(HERE / 'problem_size/all/latency_pool.json'))))
            (rec, rows, controls) = await H.run_episode(item, hole, arm, cfg=cfg, header=header, vad_model=vad_model, latency=latency, salt=f'r{rep}|{unit_id}', cache=cache, unit=unit_id)
            (keep, ctl) = compact(rows, controls)
            result['arms'][name] = {'metrics': summarize(item, rec, keep, ctl), 'timing': rec['timing'], 'status': rec['status'], 'calls': rec['calls'], 'rows': keep, 'controls': ctl, 'begun': rec['begun']}
        result['cache'] = {k: {kk: v.get(kk) for kk in ('text', 'wall_ms', 'attempts', 'error', 'dry')} for (k, v) in cache.entries.items()}
        result['valid'] = all((a['metrics'].get('valid') and (not a['metrics'].get('calls_error')) for a in result['arms'].values()))
        return result
    finally:
        models.put_nowait(vad_model)

def worker(index, jobs, args, arms):
    os.chdir(HERE)
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(HERE))
    import f2_harness as H
    from silero_vad import load_silero_vad
    (manifest, _) = load_manifest()
    by_uid = {it['uid']: it for it in manifest['items']}
    cfg = H.load_config()
    dry = None
    if args.dry:

        def dry(messages, role):
            if role == 'input_reply':
                return 'keep'
            return json.dumps({'transcript': '测试', 'label': args.dry}, ensure_ascii=False)
    out = Path(args.out)
    (units_path, req_path) = (out / f'units.w{index}.jsonl', out / f'requests.w{index}.jsonl')

    def log_request(row):
        with req_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')

    async def main():
        models = asyncio.Queue()
        for _ in range(args.conc):
            models.put_nowait(load_silero_vad())
        sem = asyncio.Semaphore(args.conc)
        budget = H.Budget(args.max_omni)
        queue = asyncio.Queue()
        for job in jobs:
            queue.put_nowait(job)
        done = 0

        async def consume():
            nonlocal done
            while not queue.empty():
                (unit_id, uid, hole, rep, cached) = queue.get_nowait()
                try:
                    res = await run_unit(unit_id, by_uid[uid], hole, rep, arms, cfg=cfg, header=manifest['header'], models=models, H=H, cache_entries=cached, sem=sem, budget=budget, dry=dry, log_request=log_request)
                except RuntimeError as exc:
                    if 'cap reached' in str(exc):
                        print(f'[w{index}] {exc}; stopping', flush=True)
                        while not queue.empty():
                            queue.get_nowait()
                        return
                    raise
                with units_path.open('a', encoding='utf-8') as fh:
                    fh.write(json.dumps(res, ensure_ascii=False) + '\n')
                done += 1
                if done % 20 == 0:
                    with (out / 'progress.log').open('a') as fh:
                        fh.write(f"{time.strftime('%H:%M:%S')} w{index} {done}/{len(jobs)} omni={budget.used}\n")
        await asyncio.gather(*(consume() for _ in range(args.conc)))
        print(f'[w{index}] finished {done} units, omni requests {budget.used}', flush=True)
    asyncio.run(main())

def finished(out, arms):
    (done, cache) = (set(), {})
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
    return (done, cache)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--sets', default='H,P,Q,L')
    p.add_argument('--holes', default='0,0.3,0.4,0.5')
    p.add_argument('--reps', default='0,1,2')
    p.add_argument('--arms', help='JSON file of arm configs (default: ./arms.json = W in 0/200/300/400/500 x pool/idle)')
    p.add_argument('--only-arms', help='comma list subset of arm names')
    p.add_argument('--procs', type=int, default=6)
    p.add_argument('--conc', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--uids')
    p.add_argument('--max-omni', type=int, help='hard cap on real Omni requests per worker')
    p.add_argument('--dry', help='no Omni: every route call returns this label (harness debugging only)')
    args = p.parse_args()
    args.out = str(Path(args.out).resolve())
    if args.procs < 1 or args.conc < 1 or args.procs * args.conc > 24:
        p.error('procs * conc must be in [1, 24]')
    if args.procs * args.conc > 24 and (not args.dry):
        raise SystemExit('Omni concurrency procs*conc must stay <= 24')
    if not args.arms and (HERE / 'arms.json').exists():
        args.arms = str(HERE / 'arms.json')
    arms = json.loads(Path(args.arms).read_text()) if args.arms else DEFAULT_ARMS
    if args.only_arms:
        arms = {k: arms[k] for k in args.only_arms.split(',')}
    (manifest, freeze) = load_manifest()
    check_sources(manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    holes = [float(h) if float(h) else 0 for h in args.holes.split(',')]
    reps = [int(r) for r in args.reps.split(',')]
    units = units_for(manifest, set(args.sets.split(',')), holes, reps, set(args.uids.split(',')) if args.uids else None)
    if args.limit:
        units = units[:args.limit]
    (done, cache) = finished(out, arms)
    todo = [(u, it['uid'], h, r, cache.get(u, {})) for (u, it, h, r) in units if u not in done]
    root = Path(os.environ.get('F2_ROOT', '/root/autodl-tmp/fd-badcat-c3v2'))
    meta = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'argv': sys.argv, 'arms': arms, 'units_total': len(units), 'units_todo': len(todo), 'code_root': str(root), 'git_head': subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(), 'omni_http_timeout_s': 60, 'decision_budget': 'production input_decision_timeout_s on audio clock'}
    with (out / 'run_meta.jsonl').open('a') as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + '\n')
    print(json.dumps({k: meta[k] for k in ('units_total', 'units_todo', 'git_head')}), flush=True)
    if not todo:
        return
    procs = max(1, min(args.procs, len(todo)))
    chunks = [todo[i::procs] for i in range(procs)]
    if procs == 1:
        worker(0, chunks[0], args, arms)
        return
    ctx = mp.get_context('spawn')
    ps = [ctx.Process(target=worker, args=(i, chunks[i], args, arms)) for i in range(procs)]
    for proc in ps:
        proc.start()
    for proc in ps:
        proc.join()
if __name__ == '__main__':
    main()
