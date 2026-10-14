# Migrated to c3v2; no artifact integrity hashing.
import asyncio, json, sys
sys.path.insert(0, '.')
import f2_harness as H
import run_f2 as R
from silero_vad import load_silero_vad
(m, _) = R.load_manifest()
it = [x for x in m['items'] if x['uid'] == 'p1:tts_stop_1_s42'][0]
cfg = H.load_config()
orig = H.first_sentence_end

def fse(e):
    s = e._speech
    rec = e._guard_outputs.get(s.sid, {}) if s else {}
    print('speech', s and s.sid, 'sent', s and s.sent, 'rate', s and s.rate, 'keys', list(rec), 'sent', rec.get('sentences'))
    return orig(e)
H.first_sentence_end = fse

async def main():
    lat = H.LatencyModel('fixed:260,80', None)
    cache = H.OmniCache({}, sem=asyncio.Semaphore(1), budget=H.Budget(0), dry=lambda m, r: json.dumps({'transcript': '测试', 'label': 'yield_ready'}, ensure_ascii=False) if r == 'input_route' else 'keep')
    (rec, rows, ctl) = await H.run_episode(it, 0, {'engine': {}}, cfg=cfg, header=m['header'], vad_model=load_silero_vad(), latency=lat, salt='x', cache=cache, unit='u')
    for (ev, t, d) in rows:
        if ev in ('speech_error', 'engine_error', 'speech_timing') or 'error' in ev:
            print(ev, t, json.dumps(d, ensure_ascii=False)[:600])
    print([(ev, t) for (ev, t, d) in rows if ev.startswith('speech') or ev.startswith('candidate')][:40])
asyncio.run(main())
