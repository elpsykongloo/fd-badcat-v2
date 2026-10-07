#!/usr/bin/env python3
"""Serial demo acceptance and latency probe, synthetic audio, no physical device."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from playwright.async_api import async_playwright

ROOT = Path('/root/autodl-tmp/fd-badcat')
sys.path.insert(0, str(ROOT / 'src'))
import module

INPUTS = {
    'normal_zh': '请用一到两句话介绍太阳系。',
    'normal_en': 'Please tell me one interesting fact about cats in one sentence.',
    'long': '请讲一个小狐狸在森林里帮助朋友的童话故事，大约一百五十个字，要有完整的结尾。',
    'stop': '请停止说话，现在先不要继续讲了。',
    'resume': '现在请只说一句，你好，很高兴认识你。',
    'wait': '我想问一个旅行问题，我还没有说完，请先等一下。',
    'continue': '去杭州玩两天，应该怎么安排？请简短地回答。',
}
MIC = r'''(() => {
  window.__tracks=[];
  navigator.mediaDevices.getUserMedia=async()=>{
    const ctx=new AudioContext({sampleRate:16000}); await ctx.resume();
    const dst=ctx.createMediaStreamDestination();
    window.__micContext=ctx;window.__micDestination=dst;
    window.__tracks.push(...dst.stream.getTracks());return dst.stream;
  };
  window.__inject=async(encoded)=>{
    const data=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
    const buffer=await window.__micContext.decodeAudioData(data.buffer);
    const source=window.__micContext.createBufferSource();source.buffer=buffer;
    source.connect(window.__micDestination);await window.__micContext.resume();
    const start=window.__micContext.currentTime+.1;
    const startClient=performance.now()+100;
    source.start(start);
    return {duration:buffer.duration,end_client_ms:startClient+buffer.duration*1000};
  };
})();'''

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def percentile(values, pct):
    return float(np.percentile(values, pct)) if values else None

async def sample_resources(folder):
    while True:
        proc=await asyncio.create_subprocess_exec('nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE)
        stdout,_=await proc.communicate()
        values=[float(v.strip()) for v in stdout.decode().split(',')]
        mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
        row=dict(wall_s=time.time(),gpu_memory_mib=values[0],gpu_total_mib=values[1],gpu_util_pct=values[2],temperature_c=values[3],power_w=values[4],system_used_mib=(mem['MemTotal']-mem['MemAvailable'])/1024)
        with (folder/'resources.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
        await asyncio.sleep(1)

async def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in INPUTS.items():
        path = folder / (name+'.wav')
        if path.exists():
            continue
        chunks = [c async for c in module.tts_omni_stream(text, voice_control={'speaker':'chelsie','seed':42})]
        sf.write(path, np.frombuffer(b''.join(c.pcm for c in chunks), dtype='<i2'), chunks[0].sample_rate, subtype='PCM_16')
        print('prepared', name, flush=True)
    save(folder/'inputs.json', INPUTS)

async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    await prepare(args.fixtures)
    encoded = {name:base64.b64encode((args.fixtures/(name+'.wav')).read_bytes()).decode() for name in INPUTS}
    resource_task=asyncio.create_task(sample_resources(args.output))
    result={'version':'pro6000-p0-v1','condition':args.condition,'scope':{'serial':True,'physical_audio':False,'synthetic_inputs':True,'production_seq4':True,'same_fixtures_across_conditions':True,'not_formal_paper_latency':True},'cases':[],'tts_probes':[]}
    # Timely SSE consumption: this RTF has no browser playback backpressure.
    for name,text in list(INPUTS.items())[:3]:
        for repeat in range(2):
            start=time.perf_counter(); first=None; samples=0; rate=None; energy=0.0; clipped=0; peak=0
            try:
                async for chunk in module.tts_omni_stream(text, voice_control={'speaker':'chelsie','seed':42}):
                    if first is None: first=(time.perf_counter()-start)*1000
                    samples+=len(chunk.pcm)//2;rate=chunk.sample_rate
                    pcm=np.frombuffer(chunk.pcm,dtype="<i2").astype(np.float64)/32768
                    energy+=float(np.dot(pcm,pcm)); clipped+=int(np.count_nonzero(np.abs(pcm)>=.999)); peak=max(peak,float(np.max(np.abs(pcm))))
                elapsed=time.perf_counter()-start
                result['tts_probes'].append({'fixture':name,'repeat':repeat,'first_pcm_ms':first,'elapsed_ms':elapsed*1000,'audio_ms':samples/rate*1000,'rtf':elapsed/(samples/rate),'rms':(energy/samples)**.5,'clip_fraction':clipped/samples,'peak':peak,'passed':samples>0 and energy>0})
            except Exception as exc:
                result['tts_probes'].append({'fixture':name,'repeat':repeat,'passed':False,'error':type(exc).__name__+': '+str(exc)})
            save(args.output/'receipt.json',result)
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,executable_path=args.chromium,args=['--no-sandbox','--autoplay-policy=no-user-gesture-required','--no-proxy-server'])
        try:
            for repeat in range(args.repeats):
                for scenario in ('normal_zh','normal_en','long_complete','stop_resume','wait_continue'):
                    row={'scenario':scenario,'repeat':repeat,'passed':False,'errors':[],'latency_ms':[],'events':[],'client':[]}
                    result['cases'].append(row)
                    context=await browser.new_context(permissions=['microphone'])
                    await context.add_init_script(MIC)
                    page=await context.new_page()
                    page.on('pageerror',lambda error:row['errors'].append(str(error)))
                    def connected(ws):
                        def recv(payload):
                            if isinstance(payload,str):
                                row['events'].append({'received_ms':time.monotonic()*1000,**json.loads(payload)})
                        def sent(payload):
                            if isinstance(payload,str):
                                value=json.loads(payload)
                                if value.get('event') in ('demo_telemetry','playback_stopped','playback_progress'):
                                    row['client'].append({'observed_ms':time.monotonic()*1000,**value})
                        ws.on('framereceived',recv);ws.on('framesent',sent)
                    page.on('websocket',connected)
                    def events(kind): return [e['data'] for e in row['events'] if e['event']==kind]
                    async def until(predicate,seconds=150):
                        deadline=time.monotonic()+seconds
                        while not predicate():
                            if row['errors']: raise RuntimeError('page error')
                            if events('speech_error'): raise RuntimeError('speech_error: '+json.dumps(events('speech_error')))
                            if time.monotonic()>deadline: raise TimeoutError('scenario timeout')
                            await asyncio.sleep(.04)
                    async def inject(name):
                        stamp=await page.evaluate('s=>window.__inject(s)',encoded[name])
                        row.setdefault('inputs',[]).append({'fixture':name,**stamp})
                        return stamp
                    async def completed_input(name):
                        before=len(events('speech_played'))
                        stamp=await inject(name)
                        await until(lambda:len(events('speech_played'))>before)
                        sid=events('speech_played')[-1]['utterance_id']
                        ends=[e for e in events('speech_text_done') if e.get('utterance_id')==sid]
                        if not ends or ends[-1].get('timed_out') or ends[-1].get('incomplete') or not ends[-1].get('text'):
                            raise RuntimeError('played response lacks completed text')
                        starts=[e['data'] for e in row['client'] if e['event']=='demo_telemetry' and e['data'].get('kind')=='playback_start' and e['data'].get('utterance_id')==sid]
                        if starts: row['latency_ms'].append(starts[0]['client_ms']-stamp['end_client_ms'])
                        return sid
                    try:
                        await page.goto(args.url+'/demo/');await page.click('#start')
                        await until(lambda:bool(events('demo_ready')),30)
                        row['session_id']=events('demo_ready')[0]['session_id']
                        if scenario in ('normal_zh','normal_en'):
                            await completed_input(scenario)
                        elif scenario=='long_complete':
                            await completed_input('long')
                        elif scenario=='stop_resume':
                            await inject('long');await until(lambda:bool(events('speech_playback_started')))
                            sid=events('speech_playback_started')[0]['utterance_id']
                            await asyncio.sleep(1)
                            stamp=await inject('stop')
                            await until(lambda:any(x.get('utterance_id')==sid for x in events('speech_cancelled')),35)
                            await until(lambda:any(x.get('reason')=='stop_only' for x in events('input_waiting')),10)
                            await until(lambda:any(e['event']=='playback_stopped' and e['data'].get('utterance_id')==sid for e in row['client']),10)
                            stopped=[e for e in row['client'] if e['event']=='playback_stopped' and e['data'].get('utterance_id')==sid][-1]
                            # Host-observed stop delay is separately labelled; no fabricated speaker timing.
                            row['stop_ms']=stopped['observed_ms']-(time.monotonic()*1000-(await page.evaluate('performance.now()'))+stamp['end_client_ms'])
                            before=len(events('speech_start'));await asyncio.sleep(3)
                            if len(events('speech_start'))!=before: raise RuntimeError('stop_only produced unsolicited speech')
                            await completed_input('resume')
                        else:
                            await inject('wait');await until(lambda:bool(events('input_waiting')),35)
                            before=len(events('speech_start'));await asyncio.sleep(1)
                            if len(events('speech_start'))!=before: raise RuntimeError('wait produced unsolicited speech')
                            await completed_input('continue')
                        row['completed']=len(events('speech_played'))
                        row['underruns']=sum(x.get('underruns',0) for x in events('speech_played'))
                        row['max_underrun_ms']=max([e['data'].get('max_underrun_ms',0) for e in row['client'] if e['event']=='demo_telemetry'] or [0])
                        row['passed']=True
                    except Exception as exc:
                        row['failure']=type(exc).__name__+': '+str(exc)
                    finally:
                        try: await page.click('#stop');await asyncio.sleep(.5)
                        except Exception: pass
                        await context.close()
                        if row.get('session_id'):
                            path=ROOT/'exp'/row['session_id']/'realtimeout_live/events.jsonl'
                            records=[json.loads(line) for line in path.read_text().splitlines()]
                            row['trace_closed']=bool(records and records[-1]['event']=='trace_closed')
                            row['trace_dropped']=records[-1].get('data',{}).get('dropped',records[-1].get('dropped')) if records else None
                            row['model_spans']=[{k:v for k,v in e.get('data',{}).items() if k in ('kind','status','elapsed_ms','first_output_ms','capacity_wait_ms','error_type','finish_reason')} for e in records if e['event']=='model_call_done']
                        if row.get('trace_closed') is not True or row.get('trace_dropped')!=0:
                            row['passed']=False
                            row.setdefault('failure','trace_not_closed_or_dropped')
                        save(args.output/(f'{repeat}-{scenario}.json'),row)
                        save(args.output/'receipt.json',result)
                    print(args.condition,repeat,scenario,'PASS' if row['passed'] else row.get('failure'),row['latency_ms'],flush=True)
        finally: await browser.close()
    resource_task.cancel()
    try: await resource_task
    except asyncio.CancelledError: pass
    samples=[json.loads(line) for line in (args.output/'resources.jsonl').read_text().splitlines()]
    resource_summary={'sample_interval_s':1,'samples':len(samples),'peak_gpu_memory_mib':max(s['gpu_memory_mib'] for s in samples),'minimum_gpu_free_mib':min(s['gpu_total_mib']-s['gpu_memory_mib'] for s in samples),'peak_gpu_util_pct':max(s['gpu_util_pct'] for s in samples),'peak_system_used_mib':max(s['system_used_mib'] for s in samples),'peak_temperature_c':max(s['temperature_c'] for s in samples),'peak_power_w':max(s['power_w'] for s in samples)}
    save(args.output/'resources_summary.json',resource_summary)
    lat=[v for row in result['cases'] for v in row['latency_ms']]
    warm=[r for r in result['tts_probes'] if r['passed'] and r['repeat']==1]
    summary={'condition':args.condition,'cases':len(result['cases']),'passed':sum(r['passed'] for r in result['cases']),'completed_replies':sum(r.get('completed',0) for r in result['cases']),'browser_first_playback_ms':{'n':len(lat),'p50':percentile(lat,50),'p95':percentile(lat,95),'max':max(lat) if lat else None},'underruns':sum(r.get('underruns',0) for r in result['cases']),'max_underrun_ms':max(r.get('max_underrun_ms',0) for r in result['cases']),'warm_tts':{'n':len(warm),'first_pcm_p50_ms':percentile([r['first_pcm_ms'] for r in warm],50),'rtf_p50':percentile([r['rtf'] for r in warm],50)},'traces_closed':sum(r.get('trace_closed',False) for r in result['cases']),'trace_dropped':sum(r.get('trace_dropped') or 0 for r in result['cases']),'stop_host_observed_ms':[r['stop_ms'] for r in result['cases'] if 'stop_ms' in r],'tts_probe_failures':sum(not r['passed'] for r in result['tts_probes']),'failures':[{'scenario':r['scenario'],'repeat':r['repeat'],'failure':r.get('failure')} for r in result['cases'] if not r['passed']]}
    save(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--condition',required=True)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--chromium',default='/root/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome')
    parser.add_argument('--url',default='http://127.0.0.1:18000')
    asyncio.run(main(parser.parse_args()))
