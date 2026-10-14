# Migrated to c3v2; no artifact integrity hashing.
"""f3_heard end-to-end check (descriptive only, see supplement_prereg.json e2e_check).

For D1 answer_in_partial items that on050 released (majority yield_ready in results/live),
replay the item in the production guarded harness under on050 with the review stubbed to
yield_ready, capture the response request the engine itself issues after the interruption
(production build_messages: history = what _guard_history_progress wrote), put in the
production response prompt and a real spoken user turn (Omni TTS, chelsie seed 42,
resampled to 16 kHz like browser input), and generate one answer with the production
response call. A contrast answer adds the heard partial to the assistant history.
"""
import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_f3 as rf
import module
from messages import build_audio_content
from test_actor_candidate import frame
from test_guarded_turns import guarded, playing, cleanup
from speech_stream import SpeechEvent
from analyze_f3 import load, majority
ARM = {'engine': {'input_reply_partial_heard_s': 0.5}}
MAX_TOKENS = int(rf.DEMO['engine'].get('response_max_tokens', 512))

async def capture_response_request(row, history=False):
    """Production harness under on050; review forced to yield_ready; returns engine's response request."""
    (e, m, _) = guarded(route='yield_ready')
    e.engine_cfg.update(stream_prefetch_ms=0, stream_startup_ms=rf.DEMO['engine'].get('stream_startup_ms', 300), stream_diagnostics=True)
    e.engine_cfg.update(rf.PROD_INPUT_CFG)
    e.engine_cfg.update(ARM['engine'])
    e.prompts.update(input_route=rf.PROMPTS['input_route'], input_reply=rf.PROMPTS['input_reply'])
    e.engine_cfg["input_reply_partial_heard_history"] = history
    e._init_guarded()
    phase = {'measure': False}
    responses = []
    fake_text = m.text

    async def text(messages, **kw):
        head = messages[0]['content']
        if isinstance(head, str) and head.startswith(rf.PROMPTS['input_route']):
            (label, transcript) = (row['base_label'], row['user']) if phase['measure'] else ('yield_ready', row['request'])
            yield json.dumps({'transcript': transcript, 'label': label}, ensure_ascii=False)
        elif head == rf.PROMPTS['input_reply']:
            yield 'yield_ready'
        else:
            if phase['measure']:
                responses.append(copy.deepcopy(messages))
            async for part in fake_text(messages, **kw):
                yield part
    e.text_stream_fn = text
    try:
        await playing(e)
        await rf.drive(e, lambda : len(e._guard_outputs[e._speech.sid]['sentences']) >= 2, 5)
        sid = e._speech.sid
        record = e._guard_outputs[sid]
        for key in ('reference', 'reply_sentences', 'reply_pending'):
            record.pop(key, None)
        record['sentences'].clear()
        per_char = row['rate_s_per_char'] * rf.RATE
        t = e._speech.sent
        units = []
        for text_ in row['completed']:
            units.append((t, t + round(len(text_) * per_char), text_))
            t = units[-1][1]
        start = t
        units.append((start, start + round(len(row['playing']) * per_char), row['playing']))
        played = start + round(row['heard_chars'] * per_char)
        sent = played + rf.CREDIT
        for (start_, end_, text_) in units:
            if start_ < sent:
                e._guard_reference_event(SpeechEvent(sid, 'sentence', {'start_sample': start_, 'text': text_}))
        for (start_, end_, text_) in units:
            if end_ <= sent:
                e._guard_sentence_end(sid, {'end_sample': end_, 'text': text_, 'reply_complete': True})
        (e._speech.played, e._speech.sent) = (played, sent)
        e._users_by_turn[0] = row['request']
        phase['measure'] = True
        await frame(e, 1, {'start': 1}, 0.2)
        span = e._guard_input
        await frame(e, 1.1, {'end': 1.1}, 0.2)
        await rf.drive(e, lambda : span.decided, 10)
        await rf.drive(e, lambda : bool(responses), 10)
        return {'messages': responses[0], 'review_context': span.reply_review_context or span.reply_context, 'partial_chars': span.reply_partial_chars, 'audio_block': e.AUDIO_BLOCK, 'history_assistant': dict(e._assistants_by_turn), 'history_user': dict(e._users_by_turn)}
    finally:
        await cleanup(e)

async def tts16k(text):
    chunks = [c async for c in module.tts_omni_stream(text, voice_control={'speaker': 'chelsie', 'seed': 42})]
    rate = chunks[0].sample_rate
    pcm = np.frombuffer(b''.join((c.pcm for c in chunks)), dtype='<i2').astype(np.float32) / 32768
    return (resample_poly(pcm, 16000, rate).astype(np.float32), len(pcm) / rate)

async def answer(messages):
    return ''.join([p async for p in module.llm_qwen3o_stream(messages, max_tokens=MAX_TOKENS)])

async def one(row, offmaj, onmaj, sem, out_f, lock):
    async with sem:
        rec = {'id': row['id'], 'user': row['user'], 'request': row['request'], 'completed': row['completed'], 'playing': row['playing'], 'heard_chars': row['heard_chars'], 'position': row['position'], 'off_majority': offmaj, 'on050_majority': onmaj}
        try:
            cap = await capture_response_request(row)
            msgs = cap['messages']
            (audio, secs) = await tts16k(row['user'])
            prod = copy.deepcopy(msgs)
            prod[0] = {'role': 'system', 'content': rf.PROMPTS['response']}
            assert prod[-1]['role'] == 'user'
            prod[-1] = {'role': 'user', 'content': [build_audio_content(audio, 16000, cap['audio_block'])]}
            completed = ''.join(row['completed'])
            partial = cap['review_context'][len(completed):] if cap['review_context'].startswith(completed) else ''
            cap_history = await capture_response_request(row, history=True)
            cf = copy.deepcopy(cap_history['messages'])
            cf[0] = {'role': 'system', 'content': rf.PROMPTS['response']}
            cf[-1] = copy.deepcopy(prod[-1])
            hist = [(m_['role'], m_['content'] if isinstance(m_['content'], str) else [c.get('text', '<audio>') if isinstance(c, dict) else c for c in m_['content']]) for m_ in prod[1:-1]]
            rec.update(partial=partial, partial_chars=cap['partial_chars'], tts_seconds=round(secs, 2), history_in_prod_request=hist, history_assistant=cap['history_assistant'], history_assistant_variant=cap_history['history_assistant'], answer_prod=await answer(prod), answer_with_partial_history=await answer(cf))
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}: {exc}'
        async with lock:
            with out_f.open('a') as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + '\n')

async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=HERE / 'results' / 'live')
    p.add_argument('--out', type=Path, default=HERE / 'e2e' / 'answers.jsonl')
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    if args.concurrency > 2:
        raise SystemExit('e2e concurrency must be <= 2')
    rows = {r['id']: r for r in rf.load_manifest(HERE / 'manifest.jsonl')}
    data = load(args.results, ['off', 'on050'])
    picks = []
    for (i, r) in rows.items():
        if r['category'] != 'answer_in_partial' or i not in data['on050']:
            continue
        onmaj = majority(data['on050'][i])
        if onmaj == 'yield_ready':
            picks.append((r, majority(data['off'].get(i, {})), onmaj))
    if args.limit:
        picks = picks[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)['id'] for l in args.out.read_text().splitlines() if l.strip() and 'error' not in json.loads(l)}
    (sem, lock) = (asyncio.Semaphore(args.concurrency), asyncio.Lock())
    todo = [x for x in picks if x[0]['id'] not in done]
    print(json.dumps({'released': len(picks), 'todo': len(todo)}), flush=True)
    await asyncio.gather(*(one(r, a, b, sem, args.out, lock) for (r, a, b) in todo))
    print('done', flush=True)
if __name__ == '__main__':
    asyncio.run(main())
