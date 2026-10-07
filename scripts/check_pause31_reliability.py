"""Private selected-training pause regression. Never save corpus text/audio.

Exact historical prefix calls plus serial, real-time complete-utterance replay
through current Chromium UI/Worklet/Player, Actor, VAD, Omni and TTS. Only the
diagnostic transport/storage adapter differs: in-memory trace and ASR WAV IO.
"""
import argparse
import asyncio
import base64
import copy
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from collections import Counter

import numpy as np
import soundfile as sf
import torch
from scipy.signal import correlate, resample_poly
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import module
from backend import load_runtime_config
from engine import ActorEngine, ModelDone
from demo_trace import DemoTrace
from control_labels import decide_input_route
from guarded_turns import route_messages
from messages import build_audio_content
from w4v3_common import read_textgrid_segments
from check_demo_continuity_live import MIC

OUT = None
CHROME = None


class MemoryTrace(DemoTrace):
    def __init__(self):
        self.clock = time.perf_counter
        self.origin = self.clock()
        self.records = []
        self.anchor = None
        self.replies = {}
        self.client_tokens = 20.
        self.last_client = self.clock()
        self.last_health = -math.inf
        self.diagnostics_dir = None
        self.manifest = {}

    def record(self, event, data=None, **context):
        self.records.append({'event': event, 'server_ms': round((self.clock()-self.origin)*1000, 3),
                             'data': copy.deepcopy(data or {}), **context})


class MemoryActor(ActorEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mic_frames = []
        self.candidate_pcm = {}

    def _guard_detect(self, ev):
        self.mic_frames.append(ev.pcm.copy())
        return super()._guard_detect(ev)

    def _begin_candidate(self, audio, **kwargs):
        candidate = super()._begin_candidate(audio, **kwargs)
        self.candidate_pcm[candidate.cid] = audio.copy()
        return candidate

    def dispatch_asr(self, user_audio, turn, answer_id=0):
        # Same asynchronous ASR/history contract, with BytesIO instead of the
        # production temporary WAV. This avoids persisting licensed corpus audio.
        gen, epoch = self.session_gen, self.seg_epoch
        parent = self._diagnostic_id('asr')
        async def run():
            started = time.perf_counter()
            call_id = self._diagnostic_id('call')
            self._observe('model_call_dispatch', {'kind':'asr', 'call_id':call_id, 'parent_id':parent, 'transport':'thread'})
            text, error, error_type = '', '', None
            def work():
                wav = io.BytesIO()
                sf.write(wav, user_audio, 16000, format='WAV', subtype='PCM_16')
                wav.seek(0)
                return self.asr_fn(wav)
            try:
                text = await asyncio.to_thread(work)
                self._observe('model_call_first_output', {'kind':'asr', 'call_id':call_id, 'parent_id':parent,
                    'elapsed_ms': round((time.perf_counter()-started)*1000,3)})
            except Exception as exc:
                error, error_type = str(exc), type(exc).__name__
            infer = round(time.perf_counter()-started,3)
            self._observe('model_call_done', {'kind':'asr','call_id':call_id,'parent_id':parent,
                'status':'error' if error else 'completed','error_type':error_type,'elapsed_ms':infer*1000})
            self.q.put_nowait(ModelDone(kind='asr', gen=gen, epoch=epoch, turn=turn, text=str(text),
                infer=infer, error=error, answer_id=answer_id, call_id=call_id, parent_id=parent))
        self._inflight += 1
        asyncio.create_task(run())


def crop(row, end):
    with sf.SoundFile(row['file']) as handle:
        sr=handle.samplerate
        start=max(0,int(row['start']*sr))
        stop=min(len(handle),int(end*sr))
        handle.seek(start)
        audio=handle.read(stop-start,dtype='float32',always_2d=True).mean(axis=1)
    if sr!=16000:
        audio=resample_poly(audio,16000,sr).astype(np.float32)
    return audio


def save(name, value):
    path=OUT/name
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600),'w') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2)
        handle.write('\n')


SAFE_KEYS = set('input_id revision closed playing audio_samples preroll_samples route reason utterance_id '
    'candidate_id turn samples rate played_samples started ended underruns prepared_audio_ms '
    'speculative confirmed published stage code status kind error_type elapsed_ms call_id parent_id '
    'fallback timed_out repaired attempts finish_reason continuations promise_repairs t_audio '
    'pending_tasks active_capacity queued_events held context_audio_samples current_audio_samples '
    'continuation_protocol context_source output_samples output_chunks tts_operation_id'.split())
SAFE_EVENTS = set('vad_start vad_done input_dispatch input_decision input_decision_stale input_admitted '
    'input_rejected input_waiting input_ignored candidate_created candidate_cancelled candidate_confirmed '
    'candidate_dispatch candidate_control_done candidate_speech_started speech_start speech_hold '
    'speech_playback_started speech_played speech_cancelled speech_error speech_audio_end speech_first_audio '
    'playback_stopped turn_finished response_completion_repair model_call_dispatch model_call_done '
    'engine_error session_final tts_recovery input_context_expired'.split())


def sanitized(records):
    out=[]
    for record in records:
        if record['event'] not in SAFE_EVENTS:
            continue
        data={k:v for k,v in record['data'].items() if k in SAFE_KEYS and (v is None or isinstance(v,(str,int,float,bool)))}
        if record['event']=='input_decision':
            audit=record['data'].get('audit',{}).get('route',{})
            data['route_audit']={k:audit[k] for k in ('base_label','attempts','fallback','timed_out','repaired') if k in audit}
        out.append({k:record[k] for k in ('event','server_ms','t_audio','turn','state') if k in record} | {'data':data})
    return out


def inspect_session(engine, item, prefix, source, texts, error, stop_reason):
    trace=engine.demo_trace.records
    def events(kind):return [r for r in trace if r['event']==kind]
    mic=np.concatenate(engine.mic_frames) if engine.mic_frames else np.zeros(0)
    probe=source[:min(len(source),48000)]
    offset=int(np.argmax(correlate(mic,probe,mode='valid',method='fft'))) if len(mic)>=len(probe) else 0
    overlap=min(len(source),len(mic)-offset)
    corr=float(np.corrcoef(mic[offset:offset+overlap],source[:overlap])[0,1]) if overlap>0 else 0.
    clip_start=offset/16000
    clip_end=clip_start+len(source)/16000
    starts=events('speech_playback_started')
    ends=events('vad_done')
    final_vad=max((r['t_audio'] for r in ends if r['t_audio']<clip_end+.5),default=clip_end)
    audible=[]
    for row in starts:
        sid=row['data']['utterance_id']
        terminal=next((r for r in trace if r['event'] in ('speech_cancelled','speech_played') and r['data'].get('utterance_id')==sid and r['server_ms']>=row['server_ms']),None)
        audible.append({'utterance_id':sid,'start_audio_s':round(row['t_audio']-clip_start,3),
            'before_final_vad':row['t_audio']<final_vad-.032,
            'end_audio_s':round(terminal['t_audio']-clip_start,3) if terminal else None,
            'end_event':terminal['event'] if terminal else 'observation_end'})
    accepted=events('input_decision')
    candidate_cancels=events('candidate_cancelled')
    result={'id':item['id'],'language':item['language'],'old_label':item['label'],
        'prefix_current':prefix,'source_duration_s':round(len(source)/16000,3),
        'alignment_correlation':round(corr,6),'clip_start_audio_s':round(clip_start,4),
        'final_vad_relative_s':round(final_vad-clip_start,3),'audible':audible,
        'early_playback':any(r['before_final_vad'] for r in audible),
        'playback_starts':len(starts),'completed_playbacks':len(events('speech_played')),
        'input_labels':[r['data']['route'] for r in accepted],
        'cancelled_candidates':len(candidate_cancels),
        'speech_cancellations':[r['data'].get('reason') for r in events('speech_cancelled')],
        'route_fallbacks':sum(bool(r['data'].get('audit',{}).get('route',{}).get('fallback')) for r in accepted),
        'model_calls':dict(Counter(r['data']['kind'] for r in events('model_call_dispatch'))),
        'speech_errors':[r['data'].get('code') for r in events('speech_error')],
        'engine_errors':len(events('engine_error')),'browser_errors':error,
        'stop_reason':stop_reason,'last_wait_reason':engine._guard_wait_reason,
        'tts_health':[{k:v for k,v in r['data'].items() if k in {
            'sentence_index','sentence_chars','audio_ms','consume_ms','low_energy_ms',
            'longest_low_energy_ms','analyzed_ms','credit_wait_ms','prefetch_wait_ms'}}
            for r in events('speech_timing') if r['data'].get('phase')=='tts_complete'],
        'normalization':[{k:v for k,v in r['data'].items() if k in {'protocol','raw_chars','spoken_chars'}}
            for r in events('speech_timing') if r['data'].get('phase')=='spoken_text'],
        'public_has_bold_markup':any('**' in r['data'].get('text','') for r in events('speech_text_done')),
        'response_completed':any(r['data'].get('stage')=='completed' for r in events('response_completion_repair')),
        'speech_state_at_end':({'sent_samples':engine._speech.sent,'played_samples':engine._speech.played,
            'sample_rate':engine._speech.rate} if engine._speech else None),
        'events':sanitized(trace)}
    memory={'id':item['id'],'annotation':texts,'prefix_transcript':prefix.pop('_transcript',None),
        'decisions':[{'t_audio':r['t_audio']-clip_start,**r['data']} for r in accepted],
        'asr':[r['data'] for r in events('asr_done')],
        'sentences':[r['data'] for r in events('speech_sentence')],
        'responses':[r['data'] for r in events('speech_text_done')],
        'public_text':[r['data'] for r in events('speech_text_delta')],
        'history':[r['data'] for r in events('history_write')],
        'result':result}
    engine.mic_frames.clear()
    engine.candidate_pcm.clear()
    return result,memory


async def main():
    global OUT, CHROME
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chromium',type=Path,required=True)
    parser.add_argument('--rows',type=Path,default=ROOT/'exp/demo_cases/diagnostics/transcript_release_prefix/rows.jsonl')
    parser.add_argument('--url',default='http://127.0.0.1:10003/v1/chat/completions')
    parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--review-seconds',type=int,default=0)
    parser.add_argument('--tail-seconds',type=float,default=90,
                        help='Observation only, not a runtime reply limit; default matches historical audit')
    args=parser.parse_args()
    if not 8 <= args.tail_seconds <= 600:
        parser.error('--tail-seconds must be in [8, 600]')
    OUT,CHROME=args.output,str(args.chromium)
    module.QWEN_URL=module.OMNI_TTS_URL=args.url
    torch.set_num_threads(2)
    OUT.mkdir(mode=0o700,parents=True,exist_ok=False)
    rows=[json.loads(line) for line in args.rows.read_text().splitlines()]
    items=[row for row in rows if row['arm']=='new' and row['grade']=='pause_prefix']
    assert len(items)==31 and len({r['id'] for r in items})==31
    cfg=load_runtime_config(ROOT/'src/config.yaml',True)
    module.configure_asr(cfg['asr'])
    await asyncio.to_thread(module.asr,ROOT/'exp/streaming_demo/synthetic_question.wav')
    settings={**cfg['engine'],'stream_response':True,'input_protocol':'pcm16.ref.v1',
              'case_capture':False,'warmup':False,'diagnostics_retention_days':None}
    info={'protocol':'pcm16.v1','streaming':True,'input_protocol':'pcm16.ref.v1',
          'guarded_turns':True,'diagnostics':{'audio_capture_allowed':False},
          'speech_text':'spoken-text-v1' if settings.get('spoken_text_normalization') else 'literal',
          'input_continuation_context':settings.get('input_continuation_context',False),
          'tts_transport_retries':settings.get('tts_transport_retries',0)}
    selected=[i for i in range(31) if args.indices is None or i in args.indices]
    if not selected or (args.indices and any(i not in range(31) for i in args.indices)):
        raise ValueError('indices must select original 0-based samples in 0..30')
    save('protocol.json',{'version':'pause31-reliability-v1','source_base_revision':'886a4e5','worktree_reliability_changes':True,
        'n':len(selected),'indices':selected,'sample_ids':[items[i]['id'] for i in selected],'current_info':info,
        'serial_sessions':True,'model_backend':'existing production seq4; no historical latency comparison',
        'source':'same 31 historical train sources; NOT independent new holdout',
        'prefix_arm':'exact historical crop; current route=True payload; no assistant playback',
        'full_arm':'full first user interval, original internal pauses; 640ms leading silence; no recorded assistant',
        'real_components':['Chromium','production frontend/Worklet/Player','ActorEngine','Silero','Omni route/response/shift','TTS','SenseVoice history'],
        'adapters':['isolated loopback websocket handshake','memory-only trace','BytesIO ASR WAV; otherwise same contract'],
        'physical_audio':False,'source_and_model_transcripts_persisted':False,
        'stop_rule':f'after source ends: idle and inflight=0 for 3s, at least 8s tail; maximum {args.tail_seconds:g}s tail, then censored',
        'observation_tail_s':args.tail_seconds,
        'outcomes':'early playback is descriptive, NOT automatically failure; separately review continuation/context/final response',
        'no_within_run_prompt_or_policy_tuning':True})
    state={'current':None,'engine':None,'trace':None,'completed':False,'rows':[],'memory':{},'closed':None}
    exit_event=asyncio.Event()
    app=FastAPI()
    @app.get('/api/demo/info')
    async def get_info():return info
    @app.get('/audit/status')
    async def status():return {'current':state['current'],'completed':state['completed'],'rows':state['rows']}
    @app.get('/audit/memory/{index}')
    async def memory(index:int):return state['memory'].get(index,{'pending':True})
    @app.post('/audit/close')
    async def close():exit_event.set();return {'closing':True}
    @app.websocket('/realtime')
    async def realtime(ws:WebSocket):
        await ws.accept()
        msg=await ws.receive_json()
        assert msg['data']['input_protocol']=='pcm16.ref.v1'
        engine=MemoryActor(websocket=ws,prompts=cfg['prompts'],delay=cfg['time'],llm_cfg=cfg['llm'],engine_cfg=settings)
        engine.demo_trace=MemoryTrace()
        state['engine']=engine
        state['trace']=engine.demo_trace
        await ws.send_json({'event':'demo_ready','data':{'session_id':f'pause31-{state["current"]}',
            'protocol':'pcm16.v1','input_protocol':'pcm16.ref.v1','observability':'demo-trace-v2',
            'guarded_turns':True,'speculative_response':True,'cancellable_response':True,
            'profile':'chat-demo-v1','tts_contract':'verbatim-grammar-v2','case_capture':False,'diagnostic_capture':False}})
        try:await engine.run_realtime(ws)
        finally:state['closed'].set()
    app.mount('/demo',StaticFiles(directory=ROOT/'src/static',html=True),name='demo')
    listener=socket.socket();listener.bind(('127.0.0.1',0))
    port=listener.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='warning',access_log=False))
    server_task=asyncio.create_task(server.serve(sockets=[listener]))
    while not server.started:await asyncio.sleep(.05)
    print(json.dumps({'audit_port':port,'output':str(OUT)}),flush=True)
    try:
        async with async_playwright() as pw:
            browser=await pw.chromium.launch(headless=True,executable_path=CHROME,
                args=['--no-sandbox','--autoplay-policy=no-user-gesture-required','--no-proxy-server'])
            for index,item in enumerate(items):
                if index not in selected:continue
                state['current']=index
                state['closed']=asyncio.Event()
                target=read_textgrid_segments(Path(item['file']).with_suffix('.TextGrid'))[0]
                source=crop(item,target['xmax']+.08)
                old_len=round((item['end']-item['start'])*16000)
                messages=route_messages(cfg['prompts']['input_route'],build_audio_content(source[:old_len],16000),playing=False)
                async def call(msgs,stage):return ''.join([p async for p in module.llm_qwen3o_stream(msgs,route=True)])
                label,audit=await decide_input_route(call,messages,2,playing=False,closed=True)
                prefix={'label':label,'attempts':audit['attempts'],'fallback':audit['fallback'],
                        'timed_out':audit['timed_out'],'_transcript':audit.get('transcript','')}
                context=await browser.new_context(permissions=['microphone'])
                await context.add_init_script(MIC)
                page=await context.new_page()
                errors=[]
                page.on('pageerror',lambda error:errors.append(type(error).__name__))
                await page.goto(f'http://127.0.0.1:{port}/demo/')
                await page.click('#start')
                await page.wait_for_function("document.getElementById('connection').dataset.connected==='true'",timeout=30000)
                # Existing production Omni/TTS is warm; local CPU ASR was warmed
                # once on a self-authored fixture before any corpus input.
                wav=io.BytesIO()
                sf.write(wav,np.concatenate([np.zeros(10240,dtype=np.float32),source]),16000,format='WAV',subtype='PCM_16')
                duration=await page.evaluate('s=>window.__inject(s)',base64.b64encode(wav.getvalue()).decode())
                source_end=time.perf_counter()+duration+.1
                idle_since=None
                stop_reason='tail_limit'
                while time.perf_counter()<source_end+args.tail_seconds:
                    await asyncio.sleep(.1)
                    engine=state['engine']
                    idle=(engine._inflight==0 and engine._candidate is None and engine._speech is None)
                    if idle:
                        idle_since=idle_since or time.perf_counter()
                    else:idle_since=None
                    if time.perf_counter()>source_end+8 and idle_since and time.perf_counter()-idle_since>=3:
                        stop_reason='quiescent'
                        break
                    if errors:
                        stop_reason='browser_error'
                        break
                engine=state['engine']
                result,memory=inspect_session(engine,item,prefix,source,target['text'],errors,stop_reason)
                await page.click('#stop')
                await page.wait_for_function("window.__tracks.every(t=>t.readyState==='ended')")
                await page.evaluate('window.__micContext.close()')
                await context.close()
                await asyncio.wait_for(state['closed'].wait(),10)
                final=[r for r in engine.demo_trace.records if r['event']=='session_final']
                result['cleanup']=final[-1]['data'] if final else None
                state['memory'][index]=memory
                state['rows'].append(result)
                save(f'case-{index:02d}.json',result)
                save('rows.json',state['rows'])
                print(json.dumps({'completed':index+1,'id':item['id'],'old':item['label'],'prefix_now':label,
                    'early_playback':result['early_playback'],'playback_starts':result['playback_starts'],
                    'completed_playbacks':result['completed_playbacks'],'stop':stop_reason},ensure_ascii=False),flush=True)
            await browser.close()
        state['completed']=True
        save('summary.json',{'n':len(state['rows']),
            'old_prefix_labels':dict(Counter(r['old_label'] for r in state['rows'])),
            'current_prefix_labels':dict(Counter(r['prefix_current']['label'] for r in state['rows'])),
            'early_playback':sum(r['early_playback'] for r in state['rows']),
            'no_playback':sum(r['playback_starts']==0 for r in state['rows']),
            'completed_any':sum(r['completed_playbacks']>0 for r in state['rows']),
            'tail_limit':sum(r['stop_reason']=='tail_limit' for r in state['rows']),
            'unfinished_playback':sum(bool(r['audible']) and r['audible'][-1]['end_event']=='observation_end' for r in state['rows']),
            'speech_errors':sum(bool(r['speech_errors']) for r in state['rows']),
            'tts_long_low_energy_cases':sum(any(h.get('longest_low_energy_ms',0)>=5000 for h in r['tts_health']) for r in state['rows']),
            'physical_audio':False,'semantic_review':'pending; source/response text only in memory'})
        print('ALL_COMPLETED_MEMORY_REVIEW_AVAILABLE',flush=True)
        if args.review_seconds:
            try:await asyncio.wait_for(exit_event.wait(),args.review_seconds)
            except asyncio.TimeoutError:pass
    finally:
        server.should_exit=True
        await server_task
        listener.close()


if __name__=='__main__':asyncio.run(main())
