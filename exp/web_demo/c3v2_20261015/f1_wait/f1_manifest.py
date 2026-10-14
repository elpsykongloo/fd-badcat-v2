# Migrated to c3v2; no artifact integrity hashing.
"""Build and freeze the f1_wait trial manifest BEFORE any inference.

Sources (all pre-existing, none is live-user audio):
  - p1 dev set: HumDial dev originals + self-authored TTS clips (variants.json)
  - p0 self-authored fixtures (setup prompt, stop/wait/continue clips)
Selected audio is resampled to 16 kHz mono PCM16 and copied under OUT/audio so the
set cannot drift; the manifest records sha1/size/duration of every file and the
manifest's own sha1 is written next to it. Re-running refuses to overwrite.

usage: python f1_manifest.py OUT_DIR
"""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
P1 = Path('/root/autodl-tmp/fd-badcat/exp/web_demo/p1_input_diagnosis_20261008')
FIX = Path('/root/autodl-tmp/fd-badcat/exp/web_demo/p0_pro6000_20261007/fixtures')
FIX_TEXT = json.loads((FIX / 'inputs.json').read_text())

def load16(path):
    (x, sr) = sf.read(path, dtype='float32')
    x = x.mean(axis=1) if x.ndim == 2 else x
    if sr != 16000:
        g = math.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype('float32')
    return x

def active(x):
    """(first, last) sample of frames within 35 dB of the loudest 20 ms frame."""
    n = 320
    frames = x[:len(x) // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-09)
    on = np.flatnonzero(db > db.max() - 35)
    return (int(on[0] * n), int((on[-1] + 1) * n))

def sha1(path):
    return None

def main():
    out = Path(sys.argv[1])
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        raise SystemExit(f'{manifest_path} is frozen; refusing to overwrite')
    audio_dir = out / 'audio'
    audio_dir.mkdir(parents=True, exist_ok=True)
    variants = {v['id']: v for v in json.loads((P1 / 'variants.json').read_text())}
    variants.update({v['id']: v for v in json.loads((P1 / 'pause_manifest.json').read_text())})
    clips = {}

    def put(cid, x, source, text, category, language, origin):
        path = audio_dir / f'{cid}.wav'
        sf.write(path, x, 16000, subtype='PCM_16')
        (a, b) = active(x)
        clips[cid] = {'id': cid, 'path': str(path), 'source': source, 'origin': origin, 'text': text, 'category': category, 'language': language, 'duration_s': round(len(x) / 16000, 3), 'speech_s': round((b - a) / 16000, 3), 'bytes': path.stat().st_size}

    def from_variant(vid):
        v = variants[vid]
        put(vid, load16(v['path']), v.get('source'), v.get('text'), v.get('category'), v.get('language') or ('en' if '_en_' in vid else 'zh'), v['path'])

    def from_fixture(name, category):
        put('fixture_' + name, load16(FIX / f'{name}.wav'), 'self_authored_fixture', FIX_TEXT[name], category, 'zh', str(FIX / f'{name}.wav'))

    def trimmed(vid):
        x = load16(variants[vid]['path'])
        (a, b) = active(x)
        return (x[a:b], variants[vid])
    for (name, cat) in [('long', 'setup'), ('stop', 'stop'), ('wait', 'wait'), ('continue', 'continuation'), ('resume', 'continuation')]:
        from_fixture(name, cat)
    pool = [vid for (vid, v) in variants.items() if vid.endswith('_clean_n0') and vid.startswith('humdial_') and (v.get('category') in ('followup', 'short_question', 'short_request'))]
    for vid in sorted(pool):
        from_variant(vid)
    questions = sorted([c for c in clips.values() if c['category'] in ('followup', 'short_question', 'short_request') and c['speech_s'] >= 2.0], key=lambda c: c['id'])
    for c in [c for c in clips.values() if c['category'] in ('followup', 'short_question', 'short_request') and c['speech_s'] < 2.0]:
        Path(c['path']).unlink()
        del clips[c['id']]
    third = []
    for vid in sorted((v for v in variants if v.endswith('_clean_n0') and variants[v].get('category') == 'third_party')):
        from_variant(vid)
        if clips[vid]['speech_s'] >= 2.0:
            third.append(vid)
        else:
            Path(clips[vid]['path']).unlink()
            del clips[vid]
    backchannels = ['humdial_zh_backchannel_01_clean_n0', 'humdial_zh_backchannel_04_clean_n0', 'humdial_en_backchannel_01_clean_n0', 'humdial_en_backchannel_03_clean_n0', 'recorded_fixture_4_clean_n0', 'tts_backchannel_2_s42_clean_n0']
    for vid in backchannels:
        from_variant(vid)
    stops = []
    gap = np.zeros(960, dtype='float32')
    for (i, (sv, fv)) in enumerate([('humdial_zh_stop_06_clean_n0', 'tts_fragment_1_s202_clean_n0'), ('humdial_zh_stop_07_clean_n0', 'tts_fragment_1_s202_clean_n0'), ('humdial_zh_stop_05_clean_n0', 'tts_fragment_1_s101_clean_n0'), ('humdial_en_stop_03_clean_n0', 'tts_fragment_2_s42_clean_n0'), ('humdial_en_stop_06_clean_n0', 'tts_fragment_2_s42_clean_n0')]):
        ((s, sm), (f, fm)) = (trimmed(sv), trimmed(fv))
        pad = np.zeros(4800, dtype='float32')
        cid = f'stopfrag_{i}'
        put(cid, np.concatenate([pad, s, gap, f, pad]), 'composite_dev_set', sm['text'] + fm['text'], 'stop_then_fragment', 'en' if '_en_' in sv else 'zh', f"{sm['path']}+{fm['path']}")
        stops.append(cid)
    stops.append('fixture_stop')
    pauses = [vid for vid in variants if vid.startswith('pause_')]
    for vid in sorted(pauses):
        from_variant(vid)
    p_items = ['humdial_en_followup_00_clean_n0', 'humdial_en_followup_01_clean_n0', 'humdial_en_followup_02_clean_n0', 'humdial_en_followup_03_clean_n0', 'humdial_en_question_01_clean_n0', 'humdial_en_question_02_clean_n0', 'humdial_zh_short_request_00_clean_n0', 'humdial_zh_short_request_01_clean_n0', 'humdial_zh_short_request_02_clean_n0', 'humdial_zh_short_request_03_clean_n0', 'humdial_zh_short_request_04_clean_n0', 'humdial_zh_short_request_05_clean_n0']
    n5_items = ['humdial_zh_question_04_clean_n0', 'humdial_zh_question_05_clean_n0', 'humdial_zh_question_07_clean_n0', 'humdial_zh_followup_07_clean_n0', 'humdial_en_followup_04_clean_n0', 'humdial_en_followup_05_clean_n0', 'humdial_en_question_03_clean_n0', 'humdial_en_question_04_clean_n0']
    assert all((i in clips for i in p_items + n5_items))
    cont = ['fixture_continue', 'fixture_resume']
    trials = []

    def add(scenario, item, *, fault='none', resume=None, delay=None, followup=False, echo=False, max_spans=1, expect, group):
        tid = f'{scenario}__{item}' + (f'__d{delay}' if delay is not None else '')
        trials.append({'trial_id': tid, 'scenario': scenario, 'group': group, 'item': item, 'fault': fault, 'max_spans': max_spans, 'resume_item': resume, 'resume_delay_s': delay, 'followup': followup, 'echo': echo, 'expect': expect, 'setup': 'fixture_long'})
    for it in p_items:
        add('P_timeout', it, fault='timeout', followup=True, expect='recover_once', group='positive')
        add('P_error', it, fault='error', expect='recover_once', group='positive')
        for (i, d) in enumerate((0.8, 1.6)):
            add('N1_resume', it, fault='timeout', resume=cont[i % 2], delay=d, expect='no_recover_answer', group='negative')
        add('D_resume_late', it, fault='timeout', resume=cont[0], delay=3.0, expect='descriptive', group='descriptive')
        add('S_double_cut', it, fault='timeout', resume=it, delay='on_recover', max_spans=2, expect='max_dependent', group='ablation')
    for it in third:
        add('N2a_third_natural', it, expect='no_recover', group='negative')
        add('N2b_third_fault', it, fault='timeout', expect='report_conditional', group='risk')
    for it in stops:
        add('N3_stop_fault', it, fault='timeout', expect='no_recover', group='negative')
    for (i, it) in enumerate(['none_echo_a', 'none_echo_b', 'none_echo_c', 'none_echo_d']):
        add('N4_echo_fault', it, fault='timeout', echo=True, expect='no_recover', group='negative')
    for it in n5_items:
        add('N5_question_natural', it, expect='no_recover_identical', group='negative')
    for it in backchannels:
        add('N5_backchannel_natural', it, expect='no_recover_identical', group='negative')
    for it in ['fixture_wait'] + sorted(pauses):
        add('N5_wait_natural', it, expect='no_recover_identical', group='negative')
    manifest = {'version': 'f1-wait-manifest-v1', 'frozen_before_inference': True, 'clip_count': len(clips), 'trial_count': len(trials), 'clips': dict(sorted(clips.items())), 'trials': trials, 'labels': {'P_*': 'cut playback + closed decision fails (injected) → should ask once', 'N1_resume': 'user resumes before release → must not ask', 'N2a': 'third party, no fault', 'N2b': 'third party + fault (risk, conditional)', 'N3': 'STOP then fragment + fault → must not ask', 'N4': 'echo only + fault armed → must not ask', 'N5_*': 'natural interactions, no fault → identical behaviour'}, 'privacy': 'no live-user audio; HumDial dev originals + self-authored TTS/fixtures only'}
    text = json.dumps(manifest, ensure_ascii=False, indent=1)
    manifest_path.write_text(text + '\n')
    digest = None
    by = {}
    for t in trials:
        by[t['scenario']] = by.get(t['scenario'], 0) + 1
    print(json.dumps({'clips': len(clips), 'trials': len(trials), 'by_scenario': by, 'p_items': p_items, 'n5_items': n5_items, 'third': third, 'stops': stops}, ensure_ascii=False, indent=1))
if __name__ == '__main__':
    main()
