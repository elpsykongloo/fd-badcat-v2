# Migrated to c3v2; no artifact integrity hashing.
"""f3_heard paired runner: one frozen manifest item -> the production guarded path.

Per item and arm this builds a real ActorEngine (GuardedTurns, Actor, queues) with the
test-suite harness, swaps in the PRODUCTION prompts and `input_*` engine settings from
configs/demo_chat.yaml plus the arm's overrides, plays the item's assistant sentences
through the production sentence/END handlers, and opens a closed user input at the
item's heard position. The audio route call is stubbed to the item's transcript and
base label (keep / stop_only): its request is byte-identical across arms and is
hashed to prove it. The input_reply review is the REAL Omni call made by the
production decide_input_route -> reply_messages -> llm_qwen3o_stream path under the
production 2 s shared deadline (or a constant "keep" with --dry: zero requests).

Resumable: rows already present in results/<arm>/r<repeat>.jsonl are skipped.
"""
import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
os.environ.setdefault('FDBC_QWEN_URL', 'http://127.0.0.1:10003/v1/chat/completions')
os.environ.setdefault('FDBC_QWEN_MODEL', 'Qwen3-Omni-30B-A3B-Instruct')
import yaml
import module
from control_labels import reply_messages
from speech_reference import SpeechReference
from speech_stream import SpeechEvent
from test_actor_candidate import frame
from test_guarded_turns import guarded, playing, cleanup
RATE = 24000
CREDIT = int(0.6 * RATE)
BASE = yaml.safe_load((ROOT / 'src/config.yaml').read_text())
DEMO = yaml.safe_load((ROOT / 'configs/demo_chat.yaml').read_text())
PROMPTS = {**BASE['prompts'], **DEMO['prompts']}
PROD_INPUT_CFG = {k: v for (k, v) in DEMO['engine'].items() if k == 'guarded_turns' or k.startswith('input_') or k.startswith('hold_') or (k == 'end_hold_tiers')}

def decision_context(messages):
    return [{'role': m['role'], 'content': m['content'] if isinstance(m['content'], str)
             else [b['text'] for b in m['content'] if isinstance(b, dict) and b.get('type') == 'text']}
            for m in messages]


class CounterfactualReference(SpeechReference):
    """Ablation only: same eligibility as heard_prefix, different representation."""

    def __init__(self, mode, true_chars):
        super().__init__()
        (self.mode, self.true_chars) = (mode, true_chars)

    def heard_prefix(self, **kw):
        got = super().heard_prefix(**kw)
        if not got:
            return got
        text = self.sentences[-1][1]
        return text if self.mode == 'full_sentence' else text[:self.true_chars]

async def drive(e, predicate, timeout):

    async def run():
        while not predicate():
            await e._process_event(await e.q.get())
    await asyncio.wait_for(run(), timeout)

class Budget:

    def __init__(self, limit):
        (self.limit, self.used) = (limit, 0)

    def take(self):
        if self.limit is not None and self.used >= self.limit:
            raise RuntimeError('omni request budget exhausted')
        self.used += 1

async def run_item(row, arm, *, dry, budget, item_timeout):
    out = {'id': row['id'], 'dataset': row['dataset'], 'category': row['category'], 'scenario': row['scenario'], 'lang': row['lang'], 'position': row['position'], 'expected': row['expected'], 'base_label': row['base_label'], 'omni_requests': 0}
    (e, m, _) = guarded(route='yield_ready')
    e.engine_cfg.update(stream_prefetch_ms=0, stream_startup_ms=DEMO['engine'].get('stream_startup_ms', 300), stream_diagnostics=True)
    e.engine_cfg.update(PROD_INPUT_CFG)
    e.engine_cfg.update(arm.get('engine', {}))
    e.prompts.update(input_route=PROMPTS['input_route'], input_reply=PROMPTS['input_reply'])
    e._init_guarded()
    phase = {'measure': False}
    calls = []
    fake_text = m.text

    async def review(messages):
        if dry:
            yield 'keep'
            return
        try:
            budget.take()
        except RuntimeError:
            out['budget_blocked'] = True
            raise
        out['omni_requests'] += 1
        async for part in module.llm_qwen3o_stream(messages, route=True):
            yield part

    async def text(messages, **_):
        head = messages[0]['content']
        if head.startswith(PROMPTS['input_route']):
            calls.append(('input_route', copy.deepcopy(messages)))
            if phase['measure']:
                (label, transcript) = (row['base_label'], row['user'])
            else:
                (label, transcript) = ('yield_ready', row['request'] or '讲个故事')
            yield json.dumps({'transcript': transcript, 'label': label}, ensure_ascii=False)
        elif head == PROMPTS['input_reply']:
            calls.append(('input_reply', copy.deepcopy(messages)))
            async for part in review(messages):
                yield part
        else:
            async for part in fake_text(messages):
                yield part
    e.text_stream_fn = text
    events = []
    try:
        await playing(e)
        await drive(e, lambda : len(e._guard_outputs[e._speech.sid]['sentences']) >= 2, 5)
        e._observe = lambda event, data=None: events.append((event, dict(data or {})))
        sid = e._speech.sid
        record = e._guard_outputs[sid]
        for key in ('reference', 'reply_sentences', 'reply_pending'):
            record.pop(key, None)
        record['sentences'].clear()
        if arm.get('cf'):
            record['reference'] = CounterfactualReference(arm['cf'], row['heard_chars'] or 0)
        per_char = row['rate_s_per_char'] * RATE
        t = e._speech.sent
        units = []
        for text_ in row['completed']:
            units.append((t, t + round(len(text_) * per_char), text_))
            t = units[-1][1]
        if row['playing']:
            start = t
            units.append((start, start + round(len(row['playing']) * per_char), row['playing']))
            played = start + round(row['heard_chars'] * per_char)
        else:
            played = t + RATE // 10
        sent = played + CREDIT
        for (start_, end_, text_) in units:
            if start_ < sent:
                e._guard_reference_event(SpeechEvent(sid, 'sentence', {'start_sample': start_, 'text': text_}))
        for (start_, end_, text_) in units:
            if end_ <= sent:
                e._guard_sentence_end(sid, {'end_sample': end_, 'text': text_, 'reply_complete': True})
        (e._speech.played, e._speech.sent) = (played, sent)
        e._users_by_turn[0] = row['request']
        phase['measure'] = True
        before = len(calls)
        await frame(e, 1, {'start': 1}, 0.2)
        span = e._guard_input
        await frame(e, 1.1, {'end': 1.1}, 0.2)
        await drive(e, lambda : span.decided, item_timeout)
        out['history_assistant'] = dict(e._assistants_by_turn)
        out['partial_history_enabled'] = bool(e._guard_partial_history)
        measured = calls[before:]
        routes = [msgs for (kind, msgs) in measured if kind == 'input_route']
        reviews = [msgs for (kind, msgs) in measured if kind == 'input_reply']
        dispatch = [d for (ev, d) in events if ev == 'input_dispatch' and d.get('closed')]
        decision = [d for (ev, d) in events if ev == 'input_decision' and d.get('closed')]
        audit = (decision[-1].get('audit') or {}).get('route') or {} if decision else {}
        rr = audit.get('reply_review') or {}
        out.update(unit_end_known=bool(row['playing']) and units[-1][1] <= sent, played_samples=played, sent_samples=sent, n_route_calls=len(routes), n_review_calls=len(reviews), route_payload=decision_context(routes[-1]) if routes else None, review_payload=decision_context(reviews[-1]) if reviews else None, played_assistant=json.loads(reviews[-1][1]['content'])['played_assistant'] if reviews else None, context_kind=dispatch[-1].get('reply_context_kind') if dispatch else None, reply_context_chars=dispatch[-1].get('reply_context_chars') if dispatch else None, partial_chars=span.reply_partial_chars, reply_context=span.reply_context, final_route=decision[-1]['route'] if decision else None, audio_base_label=audit.get('base_label'), review_called=bool(rr), review_label=rr.get('label'), review_raw=rr.get('raw'), review_timed_out=rr.get('timed_out'), review_fallback=rr.get('fallback'), review_error=rr.get('error'), review_elapsed_ms=rr.get('elapsed_ms'))
        if row['dataset'] == 'D2_frozen_text' and reviews:
            expect = reply_messages(PROMPTS['input_reply'], row['user'], ''.join(row['completed']))
            out['matches_check_script_payload'] = reviews[-1] == expect
        if out.get('budget_blocked'):
            out['harness_error'] = 'budget exhausted before the review request'
    except Exception as exc:
        out['harness_error'] = f'{type(exc).__name__}: {exc}'
        out['traceback'] = traceback.format_exc(limit=4)
    finally:
        try:
            await cleanup(e)
        except Exception as exc:
            out['cleanup_error'] = type(exc).__name__
    return out

def load_manifest(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def done_ids(path, retry_errors):
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not (retry_errors and r.get('harness_error')):
                seen.add(r['id'])
    return seen

async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=HERE / 'manifest.jsonl')
    p.add_argument('--arms', default='off,on050', help='comma list of names in arms.json')
    p.add_argument('--arms-file', type=Path, default=HERE / 'arms.json')
    p.add_argument('--set', action='append', default=[], metavar='engine.KEY=JSON', help='extra engine override applied to every arm (ad hoc ablation)')
    p.add_argument('--repeats', default='0', help='comma list of repeat indices, e.g. 0,1,2')
    p.add_argument('--results', type=Path, default=HERE / 'results')
    p.add_argument('--datasets', default='', help='comma list filter, e.g. D1_followup_mid')
    p.add_argument('--ids', default='', help='comma list of item ids (smoke)')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--max-requests', type=int, default=None, help='hard cap on real Omni requests')
    p.add_argument('--item-timeout', type=float, default=60.0)
    p.add_argument('--dry', action='store_true', help='review returns keep; zero Omni requests')
    p.add_argument('--retry-errors', action='store_true')
    args = p.parse_args()
    if not 1 <= args.concurrency <= 24:
        p.error('concurrency must be in [1, 24]')
    arms_all = json.loads(args.arms_file.read_text())
    extra = {}
    for item in args.set:
        (key, value) = item.split('=', 1)
        (section, name) = key.split('.', 1)
        if section != 'engine':
            raise SystemExit('only engine.* overrides are supported')
        extra[name] = json.loads(value)
    arms = {name: {**arms_all[name], 'engine': {**arms_all[name].get('engine', {}), **extra}} for name in args.arms.split(',')}
    rows = load_manifest(args.manifest)
    if args.datasets:
        keep = set(args.datasets.split(','))
        rows = [r for r in rows if r['dataset'] in keep]
    if args.ids:
        keep = set(args.ids.split(','))
        rows = [r for r in rows if r['id'] in keep]
    if args.limit:
        rows = rows[:args.limit]
    tag = 'dry' if args.dry else 'live'
    budget = Budget(args.max_requests)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    meta = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'mode': tag, 'arms': arms, 'prod_input_cfg': PROD_INPUT_CFG, 'qwen_url': module.QWEN_URL, 'items': len(rows), 'concurrency': args.concurrency, 'argv': sys.argv[1:]}
    for repeat in map(int, args.repeats.split(',')):
        files = {a: args.results / tag / a / f'r{repeat}.jsonl' for a in arms}
        for f in files.values():
            f.parent.mkdir(parents=True, exist_ok=True)
        (args.results / tag / f'run_meta_r{repeat}_{int(time.time())}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        done = {a: done_ids(f, args.retry_errors) for (a, f) in files.items()}
        order = list(rows)
        random.Random(repeat).shuffle(order)

        async def one(row):
            names = [a for a in arms if row['id'] not in done[a]]
            random.Random(f"{repeat}:{row['id']}").shuffle(names)
            for a in names:
                if budget.limit is not None and budget.used >= budget.limit:
                    return
                async with sem:
                    started = time.perf_counter()
                    try:
                        result = await asyncio.wait_for(run_item(row, arms[a], dry=args.dry, budget=budget, item_timeout=args.item_timeout), args.item_timeout + 10)
                    except Exception as exc:
                        result = {'id': row['id'], 'harness_error': f'{type(exc).__name__}: {exc}'}
                result.update(arm=a, repeat=repeat, mode=tag, wall_ms=round((time.perf_counter() - started) * 1000, 1))
                async with lock:
                    with files[a].open('a') as fh:
                        fh.write(json.dumps(result, ensure_ascii=False) + '\n')
        await asyncio.gather(*(one(r) for r in order))
        print(json.dumps({'repeat': repeat, 'omni_requests': budget.used, 'files': {a: str(f) for (a, f) in files.items()}}), flush=True)
if __name__ == '__main__':
    asyncio.run(main())
