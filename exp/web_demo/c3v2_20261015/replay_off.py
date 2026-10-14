import asyncio,sys,json
from pathlib import Path
sys.path[:0]=[sys.argv[1],str(Path.cwd()/'tests')]
from test_guarded_turns import guarded,playing,cleanup,new_input,cut_then_keep,wait_awaiting_user
from test_actor_candidate import frame,pump
async def one(case,explicit):
    e,m,sock=guarded()
    e.engine_cfg.update(input_wait_release_s=2,hold_tier_rules=True,
        end_hold_tiers={'low':.4,'default':.48,'keep':.64},input_route_conversation=True)
    if explicit:e.engine_cfg.update(input_wait_uncertain_recover=False,input_vad_end_ms_playing=0,
        input_reply_partial_heard_s=0,input_reply_partial_heard_history=False)
    e._init_guarded()
    try:
        await playing(e)
        if case=='uncertain':since=await cut_then_keep(e,m)
        elif case=='wait_release':since=await wait_awaiting_user(e,m)
        else:
            m.labels['input_route']=case;await new_input(e);await pump(e,lambda:e._guard_input.decided);since=1.032
        await frame(e,since+2.001)
        c=e._candidate
        return {'case':case,'state':e.STATE,'turn':e.TURN_IDX,'wait':e._guard_wait_reason,
          'responses':m.responses,'asr_calls':len(e.asr_calls),'speech':e._speech is not None,
          'candidate':None if c is None else {'stage':c.stage,'anchor':c.anchor,'hold_s':c.hold_s,
                                           'hold_tier':c.hold_tier,'audio_samples':len(c.audio)},
          'history':dict(e._assistants_by_turn),
          'model_calls':[kind for kind,_ in m.calls]}
    finally:await cleanup(e)
async def main():
    rows=[await one(case,sys.argv[2]=='explicit') for case in ['keep','stop_only','yield_wait','yield_ready','uncertain','wait_release']]
    Path(sys.argv[3]).write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
asyncio.run(main())
