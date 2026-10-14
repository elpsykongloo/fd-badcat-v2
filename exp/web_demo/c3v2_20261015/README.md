# c3v2 评估准备（2026-10-15）

本目录迁自 cheap3 10-10 实验，运行源码为 `1601c983` 上的 c3v2。没有运行全量 GPU 评估。旧阈值只作历史参考；分析需报告效果、区间、有效分母和噪声，不以预登记 PASS 作为丢弃结果的理由。

服务器工作树 `/root/autodl-tmp/fd-badcat-c3v2`；Python `/root/miniconda3/envs/fd-sds/bin/python`。Omni 复用 `:10003`。每个一键驱动立即转后台，持 `/tmp/c3v2_gpu.lock`，进度写各实验的 `progress.txt`、`driver.log`、`driver.pid`。外部实验未必遵守这个锁，运行前仍需检查 GPU 和线上连接。不要与走生产 2 秒预算的验收重叠。

```bash
cd /root/autodl-tmp/fd-badcat-c3v2
bash exp/web_demo/c3v2_20261015/f1_wait/run_eval.sh
# 等前一项结束后再启动下一项
bash exp/web_demo/c3v2_20261015/f2_vad/run_eval.sh
bash exp/web_demo/c3v2_20261015/f3_heard/run_eval.sh
```

| 实验 | 规模及输出 | 预计单卡占用时间（推断） |
|---|---|---|
| f1 | 135 条，主比较 2 臂×2 重复=540 trial；max2 48、rel3 114，名义共702；最多重试1次，总请求硬上限12000。`runs/full`、`runs/abl_max2`、`runs/abl_rel3` | 5～9 小时。历史约25秒/trial，新线冒烟27.6～39.1秒/trial；702条约4.9～7.6小时，再加启动/重试。 |
| f2 | 1293 item、10608 配对单元、10臂=106080虚拟音频 episode；Omni结果按单元缓存。`full`，`summary.json` | 2～4 小时。10-10进度覆盖约1小时52分钟；FP8吞吐及主线时序可能改变请求量。 |
| f3 | 主集1279×6臂×3=23022 trial；补集252×5臂×3=3780，共26802；另生成有限 answer_in_partial 回答对 | 30～60分钟，加端到端回答约5～15分钟。旧5/4臂主/补集约23分钟；新增history臂增加请求。 |

这些是占用共享 GPU 的墙钟预算，不是 GPU 忙碌时间，也不是新线正式实测。先导完成后按每 trial 秒数和实际请求量修正。`CONC` 可覆盖 f2/f3 驱动的并发；f2 默认 `PROCS=8 CONC=3`，乘积不得超过24；f3默认2，最大24。离线 HTTP/条目超时60秒；生产决策预算仍2秒。

f2 **两种机制不同**：旧 `input_vad_end_ms_playing` 保持 Silero100ms、路由即发，决策延迟 `(W-100)/1000` 秒；主线 `input_vad_silence_ms_playing` 改Silero本身。旧f2打开时两个Silero键必须100，否则启动拒绝，只有一处延长END。`arms.json` 保留W0/200/300/400/500×pool/idle。若另评主线键，需独立臂文件（96/128/…/480），旧键必须0。不要把W400当成Silero400，它后者不是合法32ms倍数。

`f2_vad/problem_size/all/latency_pool.json` 是旧10-10时延池，不能当FP8校准池。可用 `F2_POOL=/绝对路径/新池.json bash .../f2_vad/run_eval.sh` 指定新池；schema为input_route/input_reply毫秒数组。idle反事实仍260/80ms。

f3新增 `on050_history` 臂：半句复核 + 真正的生产回答历史写入。`e2e_answers.py` 分别捕获 history关/开两条实际引擎请求，使用相同合成用户音频生成回答，不手工拼接半句。真实answer结果需要下一步人工核对衔接、重复、虚构尾句。`cf_trueprefix`、`cf_fullsent`仍仅为不可部署反事实。

f1包含修订版的三项修复：n=0判None且PASS只接纳全True；M1样本门用有效paired_units；按utterance_id拼文本并识别中英文恢复句。另外适配主线 `_capacity_stream(stream_state=...)`，归档前等trace_closed，防止仍在关闭的录音被搬走。故障在续说前解除，保留H10修复；不改故障范围、语义阈值、负例集合。

音频源保留原始绝对只读路径（旧三个工作树及原数据目录）；未复制大音频。不要删除旧实验源目录。builders保留供明确创建新数据版本时使用，本次驱动只读已迁移manifest，**不要就地重建冻结集**。脚本不计算/比对文件完整性哈希；f2的确定性随机种子和请求缓存键属于功能用途。

`checks/`为本次未提交原始检查产物；`replay_off.py`用于模型无关行为回放。复核当前分支可执行全量pytest及Node的两个cjs测试；本机不得打开浏览器，浏览器验证仅用服务器headless Chromium1243。
