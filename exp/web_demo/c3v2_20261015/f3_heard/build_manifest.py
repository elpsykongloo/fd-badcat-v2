# Migrated to c3v2; no artifact integrity hashing.
"""Freeze the f3_heard evaluation manifest BEFORE any inference.

Three datasets, all text-only and self-authored or taken from existing frozen sets:
  D1 followup_mid  new: long assistant sentences still playing at user onset, with
                   follow-ups / answers / controls at three heard positions.
  D2 frozen_text   scripts/check_input_reply.py CASES/CONFIRM/HOLDOUT/FOLLOWUP as-is
                   (completed sentences only; f3 must not change their payload).
  D3 frozen_late   the same four groups with their LAST sentence still playing and all
                   but its last two characters heard (END known), behind a completed lead.

Labels are author-assigned from the input_reply prompt's Situations A/B and frozen here;
D2/D3 reuse the frozen sets' own labels. Writes manifest.jsonl + manifest.lock.json.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import check_input_reply as frozen
RATE = {'zh': 0.2, 'en': 0.065}
SOURCE_NEW = 'self-authored cheap3 f3 (2026-10-10)'
ZH_THIRD = ['妈妈，我的书包在哪？', '老公，帮我拿一下快递。', '你先去写作业吧。', '小张，会议几点开始？']
ZH_STOP = ['别说了', '先别讲了']
EN_THIRD = ['Mom, where are my keys?', 'Honey, can you grab the mail?', 'Go do your homework first.', 'Hey Sam, what time is the meeting?']
EN_STOP = ['Stop talking.', 'Please stop.']
ZH_NARRATION = [('zh_fox', 'check_input_reply FOLLOWUP zh_why context, extended', '给我讲个小狐狸的故事', ['好的，我给你讲个小故事。'], '小狐狸阿火在森林里发现小鸟摔伤了翅膀，就小心地把它抱回了自己的树洞。', '小鸟摔伤了翅膀', '哪只小鸟？', '阿火是谁？', '不对，是小兔子', '哇，好可怜', '小鸟摔伤了翅膀'), ('zh_photosynthesis', 'exp/f3/f3_smoke.py zh_mean_mid, extended', '光合作用是什么', ['我们来聊聊光合作用。'], '植物利用叶绿素把光能转化为化学能，并把它储存在葡萄糖这样的有机物里。', '叶绿素把光能转化为化学能', '哪种能量？', '叶绿素是什么？', '不对吧，应该是转化成热能', '哇，好神奇', '光能转化为化学能'), ('zh_balloon', 'check_input_reply FOLLOWUP zh_then context, extended', '为什么热气球能飞起来', ['这是个好问题。'], '热会让气球里的空气膨胀变轻，于是热气球就能被周围更重的冷空气托起来。', '空气膨胀变轻', '哪部分空气？', '膨胀是什么意思？', '不对，是空气变重了', '哇，真厉害', '空气膨胀变轻'), ('zh_camping', 'check_input_reply FOLLOWUP zh_wait_repeat context, extended', '露营要准备什么', ['露营前要提前看好天气。'], '露营时建议带上防水帐篷、保暖睡袋和头灯，晚上山里的温度会降得很快。', '防水帐篷、保暖睡袋和头灯', '哪种睡袋？', '头灯是什么？', '不对，睡袋不用带', '哇，听起来好好玩', '防水帐篷和头灯'), ('zh_tomato_egg', SOURCE_NEW, '番茄炒蛋怎么做', ['番茄炒蛋很简单。'], '先把鸡蛋打散炒到半熟盛出来，再用热油把番茄炒出汁，最后把鸡蛋倒回去翻炒均匀。', '炒到半熟盛出来', '哪个先下锅？', '半熟是什么意思？', '不对，应该先炒番茄', '哇，好香啊', '鸡蛋打散炒到半熟'), ('zh_great_wall', SOURCE_NEW, '介绍一下长城', ['长城是中国古代的军事防御工程。'], '现在我们看到的大部分长城修建于明朝，它东起山海关，西到嘉峪关，全长六千多公里。', '修建于明朝', '哪个朝代？', '明朝是什么时候？', '不对，是秦朝修的', '哇，好壮观', '大部分长城修建于明朝'), ('zh_sleep', SOURCE_NEW, '怎么才能睡得好', ['想睡得好，作息很重要。'], '睡前一小时尽量别看手机，因为屏幕的蓝光会抑制褪黑素分泌，让人更难入睡。', '屏幕的蓝光会抑制褪黑素', '哪种光？', '褪黑素是什么？', '不对，蓝光不影响睡眠', '哎呀，好难做到', '睡前一小时别看手机'), ('zh_moon', SOURCE_NEW, '月亮为什么有阴晴圆缺', ['月亮本身不会发光。'], '我们看到的月光其实是太阳光照在月球表面再反射回来的，所以月相会随着位置变化。', '太阳光照在月球表面再反射回来', '哪一面会反光？', '反射是什么意思？', '不对，月亮自己会发光', '哇，原来是这样', '太阳光反射回来'), ('zh_hare', SOURCE_NEW, '讲龟兔赛跑', ['从前，兔子和乌龟要比赛跑步。'], '兔子跑到半路觉得乌龟太慢，就躺在大树下睡了一觉，结果醒来时乌龟已经快到终点了。', '躺在大树下睡了一觉', '哪棵大树？', '兔子为什么要睡觉？', '不对，兔子没睡觉', '哈哈，兔子好笨', '兔子睡了一觉'), ('zh_vaccine', SOURCE_NEW, '疫苗是怎么起作用的', ['疫苗的原理其实不复杂。'], '疫苗会把减毒或灭活的病原体送进身体，让免疫系统提前认识它并产生抗体，以后真遇到就能快速应对。', '减毒或灭活的病原体', '哪种病原体？', '灭活是什么意思？', '不对，疫苗里没有病原体', '哇，好厉害', '减毒或灭活的病原体'), ('zh_panda', SOURCE_NEW, '说说大熊猫', ['大熊猫是中国特有的动物。'], '大熊猫虽然属于食肉目，但它每天要吃十几公斤竹子，一天有一大半时间都在吃东西和睡觉。', '属于食肉目', '哪一目？', '食肉目是什么？', '不对，熊猫是吃草的', '哈哈，好可爱', '大熊猫属于食肉目'), ('zh_volcano', SOURCE_NEW, '火山为什么会喷发', ['火山喷发的力量非常大。'], '地下深处的岩浆温度很高，里面溶解了大量气体，当压力积累到一定程度就会冲破地壳喷出来。', '溶解了大量气体', '哪种气体？', '岩浆是什么？', '不对，岩浆里没有气体', '哇，好吓人', '岩浆温度很高'), ('zh_interest', SOURCE_NEW, '什么是复利', ['复利常被叫作利滚利。'], '就是把每年赚到的利息也算进本金里继续生息，时间越长，钱增长得就越快。', '利息也算进本金里', '哪部分算进本金？', '本金是什么？', '不对，利息不能算进本金', '哇，听起来不错', '利息算进本金'), ('zh_prince', SOURCE_NEW, '讲一个小王子的故事', ['小王子住在一颗很小的星球上。'], '他每天都要给一朵骄傲的玫瑰浇水，还要清理星球上的三座小火山，免得它们突然喷发。', '一朵骄傲的玫瑰', '哪朵玫瑰？', '玫瑰为什么骄傲？', '不对，是一朵郁金香', '好温馨啊', '给玫瑰浇水'), ('zh_running', SOURCE_NEW, '跑步前要注意什么', ['跑步前一定要热身。'], '可以先做五分钟的动态拉伸，比如高抬腿和弓步走，让关节和肌肉慢慢活动开，再开始慢跑。', '五分钟的动态拉伸', '哪种拉伸？', '动态拉伸是什么？', '不对，跑步前应该静态拉伸', '听起来有道理', '五分钟的动态拉伸'), ('zh_sea_salt', SOURCE_NEW, '海水为什么是咸的', ['海水的盐主要来自陆地。'], '雨水会冲刷岩石里的矿物质，河流再把这些盐分带进大海，水蒸发了，盐却留了下来。', '冲刷岩石里的矿物质', '哪些矿物质？', '矿物质是什么？', '不对，盐是海底冒出来的', '哇，好神奇', '冲刷岩石里的矿物质')]
EN_NARRATION = [('en_fox', 'check_input_reply FOLLOWUP en_why context, extended', 'Tell me a story about a fox', ['Here is a short story.'], 'Little fox Ahuo found a bird with a broken wing in the forest and carried it carefully back to his den.', 'a bird with a broken wing', 'Which bird?', 'Who is Ahuo?', 'No, it was a rabbit.', 'Aw, poor bird.', 'a bird with a broken wing'), ('en_photosynthesis', SOURCE_NEW, 'What is photosynthesis?', ["Let's talk about photosynthesis."], 'Plants use chlorophyll to turn sunlight into chemical energy, which they store in sugars like glucose for later use.', 'use chlorophyll to turn sunlight into chemical energy', 'Which kind of energy?', 'What is chlorophyll?', 'No, they turn it into heat.', "Wow, that's amazing.", 'sunlight into chemical energy'), ('en_balloon', SOURCE_NEW, 'How does a hot air balloon fly?', ['Good question.'], 'Heating the air inside the balloon makes it expand and become lighter, so the cooler air around it pushes the balloon upward.', 'makes it expand and become lighter', 'Which air?', 'What does expand mean?', 'No, hot air is heavier.', 'Oh, cool.', 'expand and become lighter'), ('en_camping', SOURCE_NEW, 'What should I pack for camping?', ['Check the weather before you go.'], 'Bring a waterproof tent, a warm sleeping bag and a headlamp, because nights in the mountains get cold very quickly.', 'a waterproof tent, a warm sleeping bag and a headlamp', 'Which sleeping bag?', 'What is a headlamp?', "No, I don't need a tent.", 'Sounds fun.', 'a tent, a sleeping bag and a headlamp'), ('en_great_wall', SOURCE_NEW, 'Tell me about the Great Wall', ['The Great Wall is an ancient defense project.'], 'Most of the wall we see today was built during the Ming dynasty, stretching from Shanhai Pass in the east to Jiayu Pass in the west.', 'built during the Ming dynasty', 'Which dynasty?', 'When was the Ming dynasty?', 'No, it was built by the Qin.', "Wow, that's huge.", 'built during the Ming dynasty'), ('en_sleep', SOURCE_NEW, 'How can I sleep better?', ['A steady routine helps a lot.'], 'Avoid your phone for an hour before bed, because blue light from the screen suppresses melatonin and makes it harder to fall asleep.', 'blue light from the screen suppresses melatonin', 'Which light?', 'What is melatonin?', "No, blue light doesn't matter.", "Ugh, that's hard.", 'avoid your phone before bed'), ('en_hare', SOURCE_NEW, 'Tell me the tortoise and the hare', ['The hare and the tortoise agreed to race.'], 'Halfway through, the hare thought the tortoise was too slow and took a nap under a big tree, but when he woke up the tortoise was almost at the finish line.', 'took a nap under a big tree', 'Which tree?', 'Why did he nap?', 'No, the hare never slept.', 'Haha, silly hare.', 'a nap under a big tree'), ('en_volcano', SOURCE_NEW, 'Why do volcanoes erupt?', ['Volcanic eruptions are very powerful.'], 'Deep underground the magma is extremely hot and full of dissolved gases, and when the pressure builds up it bursts through the crust.', 'full of dissolved gases', 'Which gases?', 'What is magma?', 'No, magma has no gas in it.', 'Wow, scary.', 'full of dissolved gases')]
ZH_QUESTION = [('zh_q_story', 'exp/f3/f3_smoke.py zh_answer_mid / CASES choice_question, extended', '再讲一个故事吧', ['故事讲完了。'], '接下来你想听童话还是科幻故事，两种我都可以给你讲。', '童话还是科幻故事', '接下来你想听童话', ['科幻', '都可以']), ('zh_q_color', SOURCE_NEW, '我们玩个游戏吧', ['我们来做个小游戏。'], '你最喜欢的颜色是红色还是蓝色，告诉我之后我来猜你的性格。', '红色还是蓝色', '你最喜欢的颜色是红色', ['蓝色', '红色吧']), ('zh_q_continue', 'check_input_reply CASES permission, rephrased', '继续讲故事', ['第一章讲完了。'], '你想让我接着讲第二章吗，如果累了我们也可以明天再讲。', '接着讲第二章吗', '你想让我接着讲', ['想', '不用了']), ('zh_q_food', SOURCE_NEW, '推荐点家常菜', ['今天我们聊聊吃的。'], '你平时更喜欢吃米饭还是面条，我可以根据你的口味推荐几道家常菜。', '米饭还是面条', '你平时更喜欢吃米饭', ['面条', '都喜欢']), ('zh_q_city', SOURCE_NEW, '帮我规划旅行', ['我来帮你规划一下旅行。'], '这次你打算去北京还是上海玩，确定之后我再帮你列一份行程。', '北京还是上海玩', '这次你打算去北京', ['上海', '北京吧']), ('zh_q_check', 'check_input_reply CONFIRM zh_teaching_check, rephrased', '冰为什么会融化', ['冰吸热以后会变成水。'], '这个过程你听明白了吗，没明白的话我可以换个例子再讲一遍。', '你听明白了吗', '这个过程你听', ['明白了', '没听懂'])]
EN_QUESTION = [('en_q_story', 'exp/f3/f3_smoke.py en_answer_mid, extended', 'Tell me another story', ['That was the end of the story.'], 'Would you like a fairy tale or a science fiction story next, since I can tell you either one?', 'a fairy tale or a science fiction story', 'Would you like a fairy tale', ['Science fiction.', 'Either is fine.']), ('en_q_drink', 'check_input_reply CASES en_imperative, rephrased', 'Help me plan breakfast', ["Let's plan your morning."], 'Do you prefer tea or coffee in the morning, so I can suggest a good breakfast to go with it?', 'tea or coffee in the morning', 'Do you prefer tea', ['Coffee.', 'Tea, please.']), ('en_q_continue', 'check_input_reply CASES en_permission, rephrased', 'Continue the story', ['That was chapter one.'], 'Shall I go on with chapter two now, because the next part is where the real adventure begins?', 'go on with chapter two now', 'Shall I go on', ['Yes, go on.', 'No, not now.'])]
ZH_Q_CUT = ('好', '嗯')
EN_Q_CUT = ('Okay.', 'Mm.')
ZH_KEEP = [('zh_k_quote_ticket', 'exp/f3/f3_smoke.py zh_quoted_mid / input_reply prompt example, extended', '讲个故事', ['我给你讲个车站里的故事。'], '售货员笑着问旅客：你要靠窗还是靠过道的座位呢，旅客想了想说都行。', '靠过道的座位呢', ['随便', '靠窗', '嗯嗯']), ('zh_k_quote_teacher', 'check_input_reply CONFIRM zh_quote, extended', '讲个上课的故事', ['有一天上课的时候。'], '老师问小明：这道题你到底会不会做啊，小明低着头不敢说话。', '会不会做啊', ['会', '不会', '嗯']), ('zh_k_rhetorical_sunset', 'check_input_reply CONFIRM zh_rhetorical, extended', '讲讲海边的傍晚', ['傍晚我们来到了海边。'], '这么漂亮的晚霞，谁看了能不心动呢，大家都忍不住拿出手机拍照。', '谁看了能不心动呢', ['是啊', '对呀，太美了', '嗯嗯']), ('zh_k_rhetorical_effort', 'check_input_reply HOLDOUT zh_rhetorical, extended', '讲个练琴的故事', ['他每天都坚持练琴。'], '付出了这么多努力，难道不值得骄傲吗，最后他终于站上了大舞台。', '难道不值得骄傲吗', ['当然值得', '是啊', '嗯'])]
EN_KEEP = [('en_k_quote_ticket', 'check_input_reply CASES en_quote, extended', 'Tell me a story', ['Here is a story from a train station.'], 'The clerk smiled and asked the traveler: would you like a window seat or an aisle seat, and the traveler said either was fine.', 'a window seat or an aisle seat', ['Either is fine.', 'Window.', 'Uh-huh.']), ('en_k_rhetorical', 'check_input_reply CASES en_rhetorical, extended', 'Describe the beach at sunset', ['We arrived at the beach at sunset.'], "With a sky as beautiful as this, who wouldn't fall in love with it, and everyone took out their phones to take pictures.", "who wouldn't fall in love with it", ['Absolutely.', 'So true.', 'Mm-hmm.'])]
EARLY_CHARS = {'zh': 4, 'en': 12}
EARLY_KEYS = {'why', 'bc1', 'bc2', 'third', 'stop', 'bc'}

def positions(text, anchor_end):
    """True heard characters at user onset: right after the anchor, 40% further, END-2."""
    n = len(text)
    p1 = anchor_end + 1
    p2 = p1 + round(0.4 * (n - p1))
    p3 = n - 2
    if not p1 < p2 < p3:
        raise ValueError(f'positions collapse for {text!r}: {p1} {p2} {p3}')
    return {'P1': p1, 'P2': p2, 'P3': p3}

def anchor_end(text, anchor):
    at = text.find(anchor)
    if at < 0:
        raise ValueError(f'anchor {anchor!r} not in {text!r}')
    return at + len(anchor)

def item(**kw):
    kw.setdefault('label_method', 'author-assigned from input_reply Situations A/B; frozen before inference')
    return kw

def d1():
    rows = []

    def scenario_base(lang, sid, source, request, completed, playing, kind):
        return dict(dataset='D1_followup_mid', scenario=sid, lang=lang, kind=kind, source=source, request=request, completed=completed, playing=playing, rate_s_per_char=RATE[lang])
    for (lang, table, third, stop, generic) in (('zh', ZH_NARRATION, ZH_THIRD, ZH_STOP, (('why', '为什么？'), ('mean', '这是什么意思？'), ('bc1', '嗯嗯'), ('bc2', '对'))), ('en', EN_NARRATION, EN_THIRD, EN_STOP, (('why', 'Why?'), ('mean', 'What do you mean?'), ('bc1', 'Uh-huh.'), ('bc2', 'Right.')))):
        for (k, (sid, source, request, completed, playing, anchor, which, what, corr, emo, echo)) in enumerate(table):
            base = scenario_base(lang, sid, source, request, completed, playing, 'narration')
            g = dict(generic)
            users = [('why', g['why'], 'target_generic', 'yield_ready', 'keep'), ('mean', g['mean'], 'target_generic', 'yield_ready', 'keep'), ('which', which, 'target_specific', 'yield_ready', 'keep'), ('what', what, 'target_specific', 'yield_ready', 'keep'), ('correction', corr, 'target_specific', 'yield_ready', 'keep'), ('bc1', g['bc1'], 'control_backchannel', 'keep', 'keep'), ('bc2', g['bc2'], 'control_backchannel', 'keep', 'keep'), ('emotion', emo, 'control_emotion', 'keep', 'keep'), ('third', third[k % len(third)], 'control_third_party', 'keep', 'keep'), ('stop', stop[k % len(stop)], 'control_stop', 'keep', 'stop_only'), ('echo', echo, 'control_echo', 'keep', 'keep')]
            for (pos, heard) in positions(playing, anchor_end(playing, anchor)).items():
                for (key, user, cat, expected, base_label) in users:
                    rows.append(item(id=f'{sid}.{key}.{pos}', **base, anchor=anchor, utt_key=key, user=user, category=cat, expected=expected, base_label=base_label, position=pos, heard_chars=heard))
            for (key, user, cat, expected, base_label) in users:
                if key in EARLY_KEYS:
                    rows.append(item(id=f'{sid}.{key}.P0', **base, anchor=anchor, utt_key=key, user=user, category='target_generic_early' if key == 'why' else cat, expected=expected, base_label=base_label, position='P0', heard_chars=EARLY_CHARS[lang]))
    for (lang, table, cut_users, third) in (('zh', ZH_QUESTION, ZH_Q_CUT, ZH_THIRD), ('en', EN_QUESTION, EN_Q_CUT, EN_THIRD)):
        for (k, (sid, source, request, completed, playing, qend, cut, answers)) in enumerate(table):
            base = scenario_base(lang, sid, source, request, completed, playing, 'question')
            users = [('answer1', answers[0], 'answer_in_partial', 'yield_ready'), ('answer2', answers[1], 'answer_in_partial', 'yield_ready'), ('bc', cut_users[1], 'control_backchannel', 'keep'), ('third', third[(k + 1) % len(third)], 'control_third_party', 'keep')]
            for (pos, heard) in positions(playing, anchor_end(playing, qend)).items():
                for (key, user, cat, expected) in users:
                    rows.append(item(id=f'{sid}.{key}.{pos}', **base, anchor=qend, utt_key=key, user=user, category=cat, expected=expected, base_label='keep', position=pos, heard_chars=heard))
            rows.append(item(id=f'{sid}.bc.P0', **base, anchor=qend, utt_key='bc', user=cut_users[1], category='control_backchannel', expected='keep', base_label='keep', position='P0', heard_chars=EARLY_CHARS[lang]))
            heard = anchor_end(playing, cut)
            rows.append(item(id=f'{sid}.cut_agree.Pcut', **base, anchor=cut, utt_key='cut_agree', user=cut_users[0], category='ambiguous_cut_question', expected='ambiguous', base_label='keep', position='Pcut', heard_chars=heard, label_method='ambiguous by design: question cut before its options; reported, never scored'))
            rows.append(item(id=f'{sid}.cut_bc.Pcut', **base, anchor=cut, utt_key='cut_bc', user=cut_users[1], category='control_cut_question_backchannel', expected='keep', base_label='keep', position='Pcut', heard_chars=heard))
    for (lang, table) in (('zh', ZH_KEEP), ('en', EN_KEEP)):
        for (sid, source, request, completed, playing, anchor, users) in table:
            base = scenario_base(lang, sid, source, request, completed, playing, 'keep')
            for (pos, heard) in positions(playing, anchor_end(playing, anchor)).items():
                for (j, user) in enumerate(users):
                    rows.append(item(id=f'{sid}.u{j}.{pos}', **base, anchor=anchor, utt_key=f'u{j}', user=user, category='control_quoted_or_rhetorical', expected='keep', base_label='keep', position=pos, heard_chars=heard))
    return rows
SENTENCE = re.compile('.*?(?:[。？！?!]|\\.(?=\\s|$)|$)', re.S)

def split_sentences(text):
    parts = [p for p in SENTENCE.findall(text) if p]
    assert ''.join(parts) == text, text
    return parts

def lang_of(text):
    return 'zh' if re.search('[一-\u9fff]', text) else 'en'

def frozen_groups():
    for (group, cases) in (('cases', frozen.CASES), ('confirm', frozen.CONFIRM), ('holdout', frozen.HOLDOUT), ('followup', frozen.FOLLOWUP)):
        for (name, context, transcript, expected) in cases:
            yield (group, name, context, transcript, expected)

def d2_d3():
    rows = []
    for (group, name, context, transcript, expected) in frozen_groups():
        lang = lang_of(context + transcript)
        source = f'scripts/check_input_reply.py {group.upper()} {name}'
        rows.append(item(id=f'D2.{group}.{name}', dataset='D2_frozen_text', group=group, scenario=f'{group}.{name}', lang=lang, kind='frozen', source=source, request='', completed=split_sentences(context), playing='', anchor=None, utt_key=name, user=transcript, category=f'frozen_{group}', expected=expected, base_label='keep', position='none', heard_chars=None, rate_s_per_char=RATE[lang], label_method='frozen set label (unchanged)'))
        parts = split_sentences(context)
        lead = parts[:-1] or (['好的。'] if lang == 'zh' else ['Okay.'])
        playing = parts[-1]
        if len(playing) < 5:
            continue
        rows.append(item(id=f'D3.{group}.{name}', dataset='D3_frozen_late', group=group, scenario=f'{group}.{name}', lang=lang, kind='frozen_late', source=source + ' (last sentence playing, END-2 heard)', request='', completed=lead, playing=playing, anchor=None, utt_key=name, user=transcript, category=f'frozen_late_{expected}', expected=expected, base_label='keep', position='P3', heard_chars=len(playing) - 2, rate_s_per_char=RATE[lang], label_method='frozen set label carried over to an almost fully heard last sentence'))
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    manifest = args.out / 'manifest.jsonl'
    lock = {}
    if lock.exists():
        raise SystemExit(f'{lock} exists: the manifest is frozen; never rebuild after inference')
    rows = d1() + d2_d3()
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate ids'
    data = ''.join((json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows)).encode()
    manifest.write_bytes(data)
    counts = {}
    for r in rows:
        key = f"{r['dataset']}/{r['category']}"
        counts[key] = counts.get(key, 0) + 1
    lock.write_text(json.dumps({'manifest': manifest.name, 'bytes': len(data), 'items': len(rows), 'scenarios': len({(r['dataset'], r['scenario']) for r in rows}), 'counts': dict(sorted(counts.items())), 'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'frozen_before_inference': True}, ensure_ascii=False, indent=2) + '\n')
    print(lock.read_text())
if __name__ == '__main__':
    main()
