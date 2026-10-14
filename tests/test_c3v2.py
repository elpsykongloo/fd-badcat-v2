"""Port integration: real actor, onset history and END/recovery ownership."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastapi.testclient import TestClient
from backend import create_app
from guarded_turns import input_timing
from engine import ControlMsg
from test_input_reply import prepare, onset_after, STORY, STORY_BASE
from test_guarded_turns import guarded, cleanup, playing, cut_then_keep, wait_awaiting_user
from test_actor_candidate import frame, pump

@pytest.mark.parametrize('silence', [96, 320, 480])
def test_f2_has_one_effective_end_entry(silence):
    with pytest.raises(ValueError, match='cannot both'):
        input_timing({'input_vad_end_ms_playing': 400, 'input_vad_silence_ms_playing': silence})
    assert input_timing({'input_vad_end_ms_playing': 0,
                         'input_vad_silence_ms_playing': silence})['vad_silence_ms_playing'] == silence

@pytest.mark.parametrize('cfg', [{}, {'browser_auto_gain_control': False}, {'browser_auto_gain_control': True}])
def test_agc_default_is_exposed(cfg):
    with TestClient(create_app({}, {}, engine_cfg=cfg)) as client:
        assert client.get('/api/demo/info').json()['browser_auto_gain_control'] is cfg.get('browser_auto_gain_control', True)

@pytest.mark.parametrize('recover', [False, True])
@pytest.mark.parametrize('reason', ['awaiting_user', 'uncertain_after_yield'])
async def test_release_and_recovery_are_exclusive_policies(recover, reason):
    e,m,_ = guarded()
    e.engine_cfg.update(input_wait_release_s=2, input_wait_uncertain_recover=recover,
                        hold_tier_rules=True, end_hold_tiers={'low': .4, 'default': .48, 'keep': .64})
    e._init_guarded()
    seen=[]; e._observe=lambda kind,data=None: seen.append(kind)
    try:
        await playing(e)
        since=await (wait_awaiting_user(e,m) if reason=='awaiting_user' else cut_then_keep(e,m))
        await frame(e,since+2.001)
        if reason=='awaiting_user':
            assert e._candidate.stage=='response' and e._candidate.hold_s in (.4,.48,.64)
            assert seen.count('input_wait_released')==1 and 'input_cut_recovery' not in seen
        elif recover:
            assert e._candidate.stage=='recover' and e._candidate.hold_s==.48
            assert seen.count('input_cut_recovery')==1 and 'input_wait_released' not in seen
        else:
            assert e._candidate is None
            assert 'input_cut_recovery' not in seen and 'input_wait_released' not in seen
    finally: await cleanup(e)

@pytest.mark.parametrize('cfg,has_partial', [
    ({},False),
    ({'input_reply_partial_heard_history':True},False),
    ({'input_reply_partial_heard_s':.5},False),
    ({'input_reply_partial_heard_s':.5,'input_reply_partial_heard_history':True},True),
    ({'input_reply_partial_heard_s':.5,'input_reply_partial_heard_history':True,
      'input_wait_uncertain_recover':True,'input_wait_release_s':2,'input_vad_end_ms_playing':400,
      'hold_tier_rules':True,'end_hold_tiers':{'low':.4,'default':.48,'keep':.64}},True)])
@pytest.mark.parametrize('prefetch',[0,2000])
async def test_partial_history_is_in_response_snapshot_and_survives_late_ack(cfg,has_partial,prefetch):
    e,m,_=await prepare(**cfg)
    e.engine_cfg['stream_prefetch_ms']=prefetch
    try:
        sid=e._speech.sid; record=e._guard_outputs[sid]; turn=record['turn']
        e._users_by_turn[turn]='讲个故事'
        start=record['sentences'][-1][0]
        record['reference'].observe('sentence',{'start_sample':start,'text':STORY})
        record['sentences'].append((start+72000,STORY))
        e._speech.sent=start+80000
        span=await onset_after(e,start+48000)
        await frame(e,1.1,{'end':1.1},.2)
        if e._guard_end_hold:
            await pump(e,lambda: bool(span.deferred))
            assert e._speech is not None
            await frame(e,1.1+e._guard_end_hold+.001)
        await pump(e,lambda:span.decided)
        expected=STORY_BASE+(STORY[:8] if has_partial else '')
        assert e._assistants_by_turn[turn]==expected
        assert [x['content'] for x in e._candidate.messages['response'] if x['role']=='assistant']==[expected]
        await e._process_event(ControlMsg('playback_stopped',{'utterance_id':sid,'played_samples':start+60000}))
        assert e._assistants_by_turn[turn]==expected
        assert not e._guard_recover_since
    finally: await cleanup(e)

@pytest.mark.parametrize('label',['keep','stop_only','yield_wait'])
async def test_history_variant_cannot_commit_when_no_answer_is_authorized(label):
    e,m,_=await prepare(input_reply_partial_heard_s=.5,input_reply_partial_heard_history=True)
    m.labels['input_reply']='keep';m.labels['input_route']=label
    try:
        record=e._guard_outputs[e._speech.sid];start=record['sentences'][-1][0]
        record['reference'].observe('sentence',{'start_sample':start,'text':STORY})
        record['sentences'].append((start+72000,STORY));e._speech.sent=start+80000
        span=await onset_after(e,start+48000)
        await frame(e,1.1,{'end':1.1},.2);await pump(e,lambda:span.decided)
        assert 'partial_history_prefix' not in record
    finally: await cleanup(e)
