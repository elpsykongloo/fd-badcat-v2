#!/usr/bin/env python3
"""Paired, serial Actor route evaluation; NEVER writes HumDial audio/transcripts.

    python scripts/evaluate_actor_routes.py --suite humdial --output NEW_DIR \
      --baseline-plan exp/demo_cases/diagnostics/asr_first_robust_v1/plan.json

This is model-boundary diagnosis, not an official HumDial benchmark, human
listening test, live VAD/echo test, or permission to use test data for training.
"""
import argparse
import asyncio
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import unicodedata

import numpy as np
from scipy.signal import resample_poly, resample
import soundfile as sf
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from control_labels import parse_label, decide_control, ROUTE_PROTOCOL
from guarded_turns import route_messages
from messages import build_audio_content
from module import qwen_text_payload
from stream_transport import text_stream
from demo_cases import private_write, json_bytes, restore_request
from w4v3_common import iter_train_samples, read_textgrid_segments, resolve_root


def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())


def old_messages(prompt,content,playing,reference):
    return [{'role':'system','content':prompt},{'role':'user','content':[
        {'type':'text','text':'情境资料：'+json.dumps({'assistant_playing':bool(playing),
          'assistant_reference_text':reference[:512]},ensure_ascii=False)+'\n接下来的音频块是本次待判断的麦克风采样，请分类。'},content]}]


def crop(path,start,end):
    with sf.SoundFile(path) as f:
        sr=f.samplerate;a=max(0,int(start*sr));b=min(len(f),int(end*sr))
        if b<=a:raise ValueError('Empty crop')
        f.seek(a);x=f.read(b-a,dtype='float32',always_2d=True).mean(axis=1)
    if sr!=16000:x=resample_poly(x,16000,sr).astype(np.float32)
    return x


def humdial_inputs(root,per_cell):
    groups=defaultdict(list)
    for r in iter_train_samples(resolve_root(root)):
        if r['textgrid']:groups[(r['language'],r['scene'])].append(r)
    rng=random.Random(84019);out=[]
    for (lang,scene),pool in sorted(groups.items()):
        # Frozen seed/sample order; semantic exceptions below were reviewed
        # BEFORE inference for the 20-per-cell selection, not fitted to outputs.
        for i,r in enumerate(rng.sample(pool,per_cell)):
            segs=read_textgrid_segments(r['textgrid']);index=0 if scene=='Pause Handling' else 2
            if not segs or len(segs)<=index:raise ValueError('Missing required text interval')
            target=segs[index];expected='yield_ready';grade='semantic'
            if scene=='User Real-time Backchannels':expected='keep'
            elif scene=='Third-party Speech':expected='keep';grade='source_attribution_stress'
            elif scene=='Silence or Termination':
                if per_cell!=20:raise ValueError('Reviewed termination map requires 20-per-cell; review a new map first')
                if lang=='zh':expected='stop_only' if i==17 else 'yield_wait'
                elif i in [3,15]:expected=None;grade='conditional_ambiguous'
                elif i==14:expected='yield_ready'
                elif i in [0,1,4,5,8,12,17]:expected='yield_wait'
                else:expected='stop_only'
            x=crop(r['wav'],target['xmin']-.08,target['xmax']+.08)
            row={'id':r['key'],'group':scene,'language':lang,'grade':grade,'expected':expected,
                'playing':scene!='Pause Handling','audio':x,'annotation':target['text'].replace('[break]',''),
                'reference':segs[1]['text'] if index==2 else '',
                'file':str(r['wav']),'start':target['xmin']-.08,'end':target['xmax']+.08}
            out.append(row)
    return out


def case_inputs(root):
    out=[]
    for p in sorted((root/'captures').glob('*/case.json')):
        d=json.loads(p.read_text())
        if d['kind']!='input_route':continue
        expected={'20260908T050431242881Z-a9bcf294702a':'yield_ready',
                  '20260908T043937525718Z-dceee18b4bae':'stop_only'}.get(p.parent.name)
        req=restore_request(p.parent)
        content=next(c for c in req['messages'][1]['content'] if c['type']!='text')
        metadata=json.loads(req['messages'][1]['content'][0]['text'].split('\n')[0].removeprefix('情境资料：'))
        out.append({'id':p.parent.name,'group':'captured','grade':'confirmed' if expected else 'unlabeled',
            'language':'unknown','expected':expected,'content':content,'saved_request':req,
            'reference':metadata.get('assistant_reference_text',''),'playing':metadata['assistant_playing'],
            'closed':d['context'].get('closed'),'source_status':d['outcome']['status']})
    return out


def pause_prefixes(root,per_cell):
    import torch
    from silero_vad import load_silero_vad,VADIterator
    torch.set_num_threads(2)
    model=load_silero_vad();out=[]
    for row in humdial_inputs(root,per_cell):
        if row['group']!='Pause Handling':continue
        iterator=VADIterator(model,sampling_rate=16000)
        x=row['audio'];events=[]
        padded=np.concatenate([x,np.zeros(8000,dtype=np.float32)])
        for start in range(0,len(padded)-511,512):
            event=iterator(torch.from_numpy(padded[start:start+512]),return_seconds=True)
            if event:events.append(event)
        ends=[e['end'] for e in events if 'end' in e and e['end']<len(x)/16000-.25]
        if not ends:continue  # Explicit coverage count in the summary/plan.
        end=ends[0]
        target=read_textgrid_segments(Path(row['file']).with_suffix('.TextGrid'))[0]['text']
        if '[break]' not in target:continue
        prefix=target.split('[break]')[0].strip()
        # A dataset pause is not proof of semantic incompleteness: these
        # prefixes already form a question; report them outside the hard gate.
        ambiguous=row['id'].endswith(('en/Pause Handling/0110_0006','zh/Pause Handling/0111_0001','zh/Pause Handling/0168_0003'))
        row.update(id=row['id']+'/prefix',audio=x[:round(end*16000)],annotation=prefix,
            expected=None if ambiguous else 'yield_wait',grade='prefix_ambiguous' if ambiguous else 'pause_prefix',
            end=row['start']+end)
        out.append(row)
    return out


def synthetic_inputs(root):
    assets=json.loads((root/'assets.json').read_text());out=[]
    def add(sid,group,x,expected,ref,**kw):
        out.append(dict(id=sid,group=group,audio=x,expected=expected,reference=ref,
                        playing=True,grade='synthetic_stress',language='mixed',**kw))
    ref='从前有一个老木匠，住在山脚下。他每天给村民修理桌椅。'
    for i,a in enumerate(assets):
        x=crop(root/a['file'],0,60);rms=max(float(np.sqrt(np.mean(x*x))),1e-6)
        rng=np.random.default_rng(792+i)
        for form,y in [('clean',x),('gain',x*.35),('noise',np.clip(x+rng.normal(0,rms/10**.4,len(x)),-1,1).astype(np.float32)),
                       ('speed',resample(x,round(len(x)/1.15)).astype(np.float32))]:
            add(a['id']+'_'+form,'speech_'+form,y,a['label'],ref,verified_source=a['asr_match'])
    rng=np.random.default_rng(940);base=rng.normal(0,1,16000).astype(np.float32)
    colored=np.convolve(base,np.ones(65)/65,mode='same').astype(np.float32);colored/=np.std(colored)
    impulses=np.zeros(16000,dtype=np.float32);impulses[::1600]=.3;t=np.arange(16000)/16000
    signals={'silence':np.zeros(16000,dtype=np.float32),'white_low':base*.0001,'white_mid':base*.001,
      'white':base*.01,'colored_low':colored*.003,'colored':colored*.03,
      'hum':(.03*np.sin(2*np.pi*180*t)).astype(np.float32),
      'sweep':(.03*np.sin(2*np.pi*(180*t+400*t*t))).astype(np.float32),'clicks':impulses}
    for name,x in signals.items():
        for playing in [True,False]:
            for ri,r in enumerate(['',ref,'你是谁？','停。']):
                add(f'{name}_{playing}_{ri}','nonspeech',x,'keep',r)
                out[-1]['playing']=playing
    return out


async def evaluate(args):
    old=json.loads(args.baseline_plan.read_text())['prompts']['direct']
    new=yaml.safe_load((ROOT/'configs/demo_chat.yaml').read_text())['prompts']['input_route']
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    items=(humdial_inputs(args.dataset,args.per_cell) if args.suite=='humdial' else
           pause_prefixes(args.dataset,args.per_cell) if args.suite=='humdial-prefix' else
           case_inputs(args.cases) if args.suite=='cases' else synthetic_inputs(args.synthetic))
    if not items:raise ValueError('Empty suite')
    # Whitelisted metadata only. Never serialize annotation/reference/content,
    # model transcripts, raw outputs, or encoded audio from HumDial.
    allowed=['id','group','language','grade','expected','playing','file','start','end','closed','source_status','verified_source']
    private_write(args.output/'plan.json',json_bytes({'version':ROUTE_PROTOCOL,'suite':args.suite,
       'old_prompt':old,'new_prompt':new,'items':[{k:r[k] for k in allowed if k in r} for r in items],
       'scope':'serial model-boundary pairs; same clip, old context vs no reference; not official benchmark; no transcript/audio artifacts',
       'old_repair':'legacy exact label parser; new uses production decide_control including at-most-one repair/2s',
       'acknowledgment':'Uses data provided by the Hum-Dial Challenge for non-commercial research; no data redistribution.'}))
    rows=[]
    with os.fdopen(os.open(args.output/'rows.jsonl',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'w',buffering=1) as f:
        for i,item in enumerate(items):
            content=item.get('content') or build_audio_content(item['audio'],16000)
            for arm in (['old','new'] if i%2==0 else ['new','old']):
                messages=(old_messages(old,content,item['playing'],item['reference']) if arm=='old' else
                          route_messages(new,content,playing=item['playing'],reference=item['reference']))
                request_base=item.get('saved_request')
                async def call(msgs):
                    payload=copy.deepcopy(request_base) if request_base else qwen_text_payload(msgs)
                    payload['messages']=msgs
                    raw=''
                    async for p in text_stream(args.url,payload,2):
                        raw+=p
                        if len(raw)>4096:raise ValueError('Oversized routing output')
                    return raw
                t=time.perf_counter();row={k:item[k] for k in allowed if k in item};row['arm']=arm
                try:
                    if arm=='new':label,audit=await decide_control(call,messages,'input_route',2)
                    else:
                        raw=await asyncio.wait_for(call(messages),2)
                        label=parse_label('input_route',raw,legacy_route=True)
                        audit={'attempts':1,'fallback':label is None,'repaired':False,'timed_out':False}
                        label=label or 'keep'
                    row.update(label=label,attempts=audit['attempts'],fallback=audit['fallback'],
                               repaired=audit['repaired'],timed_out=audit['timed_out'])
                    if 'annotation' in item and arm=='new':
                        row['transcript_normalized_equal']=norm(audit.get('transcript',''))==norm(item['annotation'])
                    if item['expected'] is not None:
                        row['correct']=(label in ('keep','yield_wait') if item['grade']=='pause_prefix' else label==item['expected'])
                except Exception as exc:row.update(error=type(exc).__name__,correct=False)
                row['elapsed_ms']=round((time.perf_counter()-t)*1000,1)
                rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if i%10==0 or i==len(items)-1:print('PROGRESS',i+1,len(items),flush=True)
    stats={}
    for grade in sorted({r['grade'] for r in rows})+['ALL']:
        subset=[r for r in rows if grade=='ALL' or r['grade']==grade];stats[grade]={}
        for arm in ['old','new']:
            a=[r for r in subset if r['arm']==arm];times=sorted(r['elapsed_ms'] for r in a)
            stats[grade][arm]={'n':len(a),'scored':sum('correct' in r for r in a),
              'correct':sum(r.get('correct',False) for r in a),'labels':dict(Counter(r.get('label','error') for r in a)),
              'fallback':sum(r.get('fallback',False) for r in a),'errors':sum('error' in r for r in a),
              'repairs':sum(r.get('repaired',False) for r in a),'p50_ms':statistics.median(times),'p90_ms':times[int((len(times)-1)*.9)]}
        baseline={r['id']:r for r in subset if r['arm']=='old'}
        pairs=[(baseline[r['id']],r) for r in subset if r['arm']=='new']
        stats[grade]['paired']={'changed':sum(a.get('label')!=b.get('label') for a,b in pairs),
          'gain':sum(not a.get('correct') and b.get('correct',False) for a,b in pairs),
          'loss':sum(a.get('correct',False) and not b.get('correct') for a,b in pairs)}
    private_write(args.output/'summary.json',json_bytes(stats));print(json.dumps(stats,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['humdial','humdial-prefix','cases','synthetic'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline-plan',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=Path('/root/autodl-tmp/HumDial_train'))
    p.add_argument('--cases',type=Path,default=ROOT/'exp/demo_cases')
    p.add_argument('--synthetic',type=Path,default=ROOT/'exp/demo_cases/diagnostics/asr_first_robust_v1')
    p.add_argument('--per-cell',type=int,default=20)
    p.add_argument('--url',default='http://127.0.0.1:10004/v1/chat/completions')
    asyncio.run(evaluate(p.parse_args()))


if __name__=='__main__':main()
