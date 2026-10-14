# Migrated to c3v2; no artifact integrity hashing.
"""Freeze the f3_heard SUPPLEMENT manifest before any inference on it.

Separately registered after the main prereg (the main manifest/prereg stay untouched).
Covers two side-effect control classes the frozen D1 does not contain: the completed
sentence is a question to this user, and the sentence still playing either announces
the assistant's own choice / next step (SA_*), or starts answering a teaching question
itself (SE_*). The input_reply prompt says both require keep for agreement / an answer
to the teaching question, while a contrary choice, a follow-up question or a repeat
request still need a reply. Off sees only the question; on050 may also see the heard
clause of the announcement/explanation, so it can flip labels in either direction.

Positions (heard characters of the playing sentence, same timeline as run_f3.py):
  Q1  first clause just heard (+1 char): true prefix has it, the conservative bound
      usually does not;
  Q2  the first heard count at which production heard_prefix (on050) returns the first
      clause (computed here with the production function, before any inference);
  Q3  all but the last 2 characters heard (END known).
Writes supplement/manifest.jsonl + supplement/manifest.lock.json (read-only).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from speech_reference import SpeechReference, clause_prefix
RATE = {'zh': 0.2, 'en': 0.065}
(SR, CREDIT) = (24000, int(0.6 * 24000))
SOURCE = 'self-authored cheap3 f3 supplement (2026-10-10)'
LABEL = 'author-assigned from input_reply Situations A/B and its final rule; frozen before inference'
ANNOUNCE = [('zh_a_story', '再讲一个故事', ['故事讲完了。', '你想听童话还是科幻故事？'], '那我先给你讲一个童话吧，从前有一只小兔子住在森林边上。', {'agree': '好啊', 'same': '童话', 'either': '都可以', 'other': '我想听科幻的'}), ('zh_a_fox', '讲个动物故事', ['我这里有两个故事。', '你想先听小狐狸的还是小熊的？'], '我先给你讲小狐狸的故事，小狐狸阿火住在一棵大树下面。', {'agree': '行', 'same': '小狐狸', 'either': '随便', 'other': '我想先听小熊的'}), ('zh_a_food', '教我做个菜', ['今天我们聊聊做菜。', '你想学番茄炒蛋还是红烧肉？'], '那我们先从番茄炒蛋开始吧，它最简单也最快学会。', {'agree': '好的', 'same': '番茄炒蛋', 'either': '都行', 'other': '我想学红烧肉'}), ('zh_a_city', '帮我规划旅行', ['我来帮你规划一下旅行。', '你想去北京还是上海？'], '我先按北京给你排一下行程，第一天可以去故宫和天安门看看。', {'agree': '可以', 'same': '北京', 'either': '都可以', 'other': '我想去上海'}), ('zh_a_music', '聊聊音乐', ['我们来聊点音乐知识。', '你喜欢古典乐还是流行乐？'], '我先给你介绍一下古典乐，贝多芬是最有名的作曲家之一。', {'agree': '嗯好', 'same': '古典乐', 'either': '都喜欢', 'other': '我喜欢流行乐'}), ('zh_a_game', '陪我玩个游戏', ['我们来玩个游戏吧。', '你想玩猜谜还是成语接龙？'], '那就先玩猜谜吧，我来说谜面，你来猜谜底。', {'agree': '好呀', 'same': '猜谜', 'either': '随便', 'other': '我想玩成语接龙'}), ('zh_a_sport', '我想开始运动', ['运动方面我可以给你一些建议。', '你想练跑步还是游泳？'], '我们先说说跑步吧，跑步对场地和装备的要求最低。', {'agree': '好', 'same': '跑步', 'either': '都行', 'other': '我想练游泳'}), ('zh_a_book', '推荐几本书', ['我可以给你推荐几本书。', '你想看小说还是历史书？'], '我先给你推荐一本小说，余华的《活着》写得非常感人。', {'agree': '好的', 'same': '小说', 'either': '都可以', 'other': '我想看历史书'}), ('en_a_story', 'Tell me another story', ['That was the end of the story.', 'Would you like a fairy tale or a science fiction story?'], 'I will start with a fairy tale, about a little rabbit who lived by the forest.', {'agree': 'Okay.', 'same': 'A fairy tale.', 'either': 'Either is fine.', 'other': 'I want science fiction.'}), ('en_a_drink', 'Help me plan my morning', ["Let's plan your morning.", 'Do you prefer tea or coffee?'], 'I will go with coffee for now, since it pairs well with a quick breakfast.', {'agree': 'Sure.', 'same': 'Coffee.', 'either': 'Whatever.', 'other': "I'd rather have tea."}), ('en_a_trip', 'Plan a trip for me', ['Let me help you plan a trip.', 'Would you rather visit Paris or Rome?'], "Let's begin with Paris, where the first day could start at the Louvre.", {'agree': 'Sounds good.', 'same': 'Paris.', 'either': 'Either one.', 'other': 'I want to go to Rome.'}), ('en_a_sport', 'I want to get fit', ['I can suggest a workout.', 'Do you want to try running or swimming?'], 'I will start with running, because it needs the least equipment.', {'agree': 'Okay.', 'same': 'Running.', 'either': 'Both are fine.', 'other': "I'd prefer swimming."})]
ANNOUNCE_EXPECTED = {'agree': 'keep', 'same': 'keep', 'either': 'keep', 'other': 'yield_ready'}
EXPLAIN = [('zh_e_sky', '聊聊天空', ['我们来聊聊天空。', '你知道天空为什么是蓝色的吗？'], '这是因为阳光里的蓝光波长短，更容易被空气分子散射到各个方向。', {'dunno': '不知道', 'guess': '因为大海吧', 'why': '为什么蓝光更容易散射？', 'repeat': '没听清，再说一遍'}), ('zh_e_leaves', '秋天的树叶', ['秋天到了。', '你知道树叶为什么会变黄吗？'], '因为天冷以后叶绿素慢慢分解，叶子里原本的黄色素就显露出来了。', {'dunno': '不知道诶', 'guess': '因为缺水吧', 'why': '叶绿素为什么会分解？', 'repeat': '你再说一遍'}), ('zh_e_moon', '聊聊月亮', ['今晚月亮很圆。', '你知道月亮为什么会有圆缺吗？'], '其实月亮自己不发光，我们看到的是它被太阳照亮的那一部分。', {'dunno': '不知道', 'guess': '因为云挡住了吧', 'why': '那为什么照亮的部分会变？', 'repeat': '能再说一遍吗'}), ('zh_e_salt', '聊聊大海', ['说到大海。', '你知道海水为什么是咸的吗？'], '主要是因为河流把陆地上的盐分带进了大海，水蒸发后盐却留了下来。', {'dunno': '不清楚', 'guess': '因为鱼的眼泪', 'why': '河水为什么不咸？', 'repeat': '没听清'}), ('zh_e_rainbow', '彩虹是什么', ['雨后经常能看到彩虹。', '你知道彩虹是怎么形成的吗？'], '阳光照进小水滴时会发生折射，不同颜色的光弯折的角度不一样，就分开了。', {'dunno': '不知道', 'guess': '是水汽反光吧', 'why': '为什么角度不一样？', 'repeat': '再讲一遍吧'}), ('zh_e_ice', '冬天的湖', ['冬天湖面会结冰。', '你知道冰为什么会浮在水面上吗？'], '因为水结冰时体积会变大，同样重量的冰比水占的地方更多，所以更轻。', {'dunno': '不知道', 'guess': '因为冰是空心的', 'why': '为什么结冰体积会变大？', 'repeat': '你刚才说什么'}), ('en_e_sky', 'Tell me about the sky', ["Let's talk about the sky.", 'Do you know why the sky is blue?'], 'It is because blue light has a short wavelength, so the air scatters it in every direction.', {'dunno': 'No idea.', 'guess': 'Because of the ocean.', 'why': 'Why does blue light scatter more?', 'repeat': 'Sorry, say that again.'}), ('en_e_leaves', 'Why do leaves change', ['Autumn is here.', 'Do you know why leaves turn yellow?'], 'When it gets cold, the green chlorophyll breaks down and the yellow pigments underneath show through.', {'dunno': "I don't know.", 'guess': 'Because they get dry.', 'why': 'Why does chlorophyll break down?', 'repeat': 'Can you repeat that?'}), ('en_e_ice', 'Tell me about ice', ['Lakes freeze in winter.', 'Do you know why ice floats on water?'], 'Water expands when it freezes, so a piece of ice takes up more space than the same weight of water.', {'dunno': 'Not really.', 'guess': 'Because it is cold.', 'why': 'Why does water expand?', 'repeat': 'What did you say?'})]
EXPLAIN_EXPECTED = {'dunno': 'keep', 'guess': 'keep', 'why': 'yield_ready', 'repeat': 'yield_ready'}

def lang_of(sid):
    return sid.split('_', 1)[0]

def on050_prefix(playing, heard, lang):
    """Production heard_prefix on run_f3.py's timeline (on050: min_heard 0.5 s)."""
    per = RATE[lang] * SR
    end = round(len(playing) * per)
    played = round(heard * per)
    sent = played + CREDIT
    ref = SpeechReference()
    ref.observe('sentence', {'start_sample': 0, 'text': playing})
    ends = [end] if end <= sent else []
    return ref.heard_prefix(played=played, sent=sent, ends=ends, min_heard=int(SR * 0.5), rate=SR)

def first_clause(playing):
    for n in range(1, len(playing)):
        got = clause_prefix(playing, n)
        if got:
            return got
    raise ValueError(playing)

def positions(playing, lang):
    clause = first_clause(playing)
    q1 = len(clause) + 1
    q2 = next((h for h in range(1, len(playing) - 1) if on050_prefix(playing, h, lang)), None)
    q3 = len(playing) - 2
    out = [('Q1', q1)]
    if q2 is not None and q2 != q1 and (q2 < q3):
        out.append(('Q2', q2))
    out.append(('Q3', q3))
    return (clause, out)

def main():
    out_dir = HERE / 'supplement'
    out_dir.mkdir(exist_ok=True)
    (manifest, lock_path) = {}
    if lock_path.exists():
        raise SystemExit('supplement lock exists; refusing to rebuild a frozen manifest')
    rows = []
    for (family, scenarios, expected_of, category) in (('announce', ANNOUNCE, ANNOUNCE_EXPECTED, 'sup_question_then_announce'), ('explain', EXPLAIN, EXPLAIN_EXPECTED, 'sup_question_then_explain')):
        for (sid, request, completed, playing, users) in scenarios:
            lang = lang_of(sid)
            (clause, pos) = positions(playing, lang)
            for (key, text) in users.items():
                for (name, heard) in pos:
                    rows.append({'id': f'{sid}.{key}.{name}', 'dataset': 'S_supplement', 'scenario': sid, 'lang': lang, 'family': family, 'category': f'{category}/{key}', 'user': text, 'utt_key': key, 'expected': expected_of[key], 'base_label': 'keep', 'request': request, 'completed': completed, 'playing': playing, 'first_clause': clause, 'heard_chars': heard, 'position': name, 'rate_s_per_char': RATE[lang], 'on050_prefix_dry': on050_prefix(playing, heard, lang), 'source': SOURCE, 'label_method': LABEL})
    data = ''.join((json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows)).encode()
    manifest.write_bytes(data)
    prereg = out_dir / 'supplement_prereg.json'
    analyzer = HERE / 'analyze_sup.py'
    counts = {}
    for r in rows:
        counts[r['category']] = counts.get(r['category'], 0) + 1
    lock = {}
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n')
    for p in (manifest, lock_path, prereg):
        os.chmod(p, 292)
    print(json.dumps(lock, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
