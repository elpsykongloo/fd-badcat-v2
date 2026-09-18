# Demo 连续播放：demo-continuity-v1

本机制解决 TTS 整句生成速度足够、但音频块到达不均匀造成的播放中断。仅独立 demo 配置启用；输入准入、音频钟、640 ms hold、逐字 TTS、固定声音和控制请求预留容量沿用原契约。

## 管线与边界

```text
文本增量 → 原分句器 → 顺序 TTS / PCM读取 → 有界PCM队列 → 40ms包 → 浏览器播放信用
                                           ↑              ↓
                                        队列回压      最多600ms未播音频
```

`configs/demo_chat.yaml` 设置：

- `stream_startup_ms: 350`：通过 `speech_start.startup_ms` 下发，首次排程与开播前 hold 释放后使用同一值。字段缺省为80 ms，范围0–1000 ms；断流恢复仍为80 ms，避免将已发生的停顿再增加350 ms。
- `stream_prefetch_ms: 2000`：将模型消费与发送解耦；PCM队列按实际采样率限制字节数。另有一个不超过512 KiB的已解码源块和一个发送包。发送端仍独立限制600 ms未确认播放样本。
- `stream_diagnostics: true`：通过引擎事件队列记录分块与回压；不会从模型任务直接修改引擎状态。

预取最多允许当前发送句与下一句两个未发送完的句子；TTS请求仍串行。未确认候选只允许首句，第二句仍等待原确认门。只有发送端发布 `sentence/start_sample` 和 `sentence_end/end_sample`；生成完成或预取完成不等于已播放，不提前更新已播参考/历史。

取消会结束文本、TTS读取、发送三个任务，关闭HTTP源并清空预取。原utterance ID隔离、播放进度单调检查、5秒无进度超时、单次结束结算和保留控制槽继续生效。

## Demo 分块配置

`FDBC_DEMO_CODEC_CHUNKS=INITIAL:STEADY` 由正常启动链生成临时部署YAML；demo默认 `4:12`：

```bash
FDBC_DEMO_CODEC_CHUNKS=4:12 bash setup/start_demo.sh
```

`off` 使用调用者原配置；通用Omni启动默认off。只有显式启用demo voice adapter才允许派生，校验首块1–25帧、后续4–25帧；只改Talker→Code2Wav连接器的两个分块字段。`codec_left_context_frames`、三个阶段容量、数值模式与冻结源YAML不变。`--backend-only` 不会修改已运行的Omni分块。

当前版本从连接器部署配置读取这些参数；不是运行时逐请求覆盖。改变它们需要重启Omni。回退时将demo的起播/预取/诊断配置设为 `80/0/false`，用 `FDBC_DEMO_CODEC_CHUNKS=off` 正常重启（基础配置为4:25）。

## 指标口径

- `speech_timing.phase=tts_request/tts_chunk/tts_complete/sentence_sent`：句子索引、块样本数、消费时间、SSE解析时刻、文本证明完成时刻、解码时间、预取等待、发送信用等待和PCM队列峰值。所有后端时间都来自单调墙钟，仅诊断使用。
- `sse_audio_ms` 是消费者解析到该SSE记录时的相对时间；若消费者已被队列回压，不能冒充模型或网络真实产出时刻。
- `tts_complete.consume_ms` 和私有案例 `elapsed_ms` 包含消费者等待。不能直接除以音频长度作为模型纯RTF。无播放回压RTF由独立及时消费SSE的串行探针给出，仍包含本机HTTP与解码。
- `client_playback_start` 记录AudioContext排程起点及观测时刻；`client_underrun` 记录同一时钟的排程尾端、包序号、样本偏移、到包间隔、缺音和80 ms恢复余量。结束/取消快照记录 `underrun_ms/max_underrun_ms`；详细事件受既有遥测速率限制，快照仍保留累计数。
- 浏览器指标不是物理扬声器录音或主观听感评分，内部韵律静音不能与缓冲耗尽混算。首包到达时间不包含新加的起播余量，应同时报告首次排程播放的成本。

## 验证入口

```bash
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python -m pytest \
  tests/test_speech_continuity.py tests/test_speech_stream.py \
  tests/test_actor_candidate.py tests/test_guarded_turns.py \
  tests/test_speech_reference.py tests/test_demo_trace.py tests/test_tts_verbatim.py -q
node tests/test_stream_demo.cjs

# 使用装有playwright的隔离环境；每次必须使用新目录。
python scripts/check_demo_continuity.py --output NEW_PRIVATE_DIR --profile INITIAL:STEADY \
  --chromium EXISTING_CHROMIUM
python scripts/check_demo_continuity_live.py --output ANOTHER_PRIVATE_DIR \
  --chromium EXISTING_CHROMIUM
```

第一项包含6条自造中英文本的串行重复、ASR严格归一回读、真实Chromium播放生产管线的短/长回答及取消、路由请求重叠。浏览器固定回答夹没有经过Actor输入决策，不能称为完整真实对话。第二项通过展示页和合成麦克风调用真实Actor/Omni，检查长回答被明确停止、STOP_ONLY静默及后续新回复；没有真实人类音频和物理设备。

## 验收结果

2026-09-18完成，采用 **4:12 + 350 ms起播 + 2000 ms预取**，已通过正常启动链部署到原demo端口。实际部署YAML与基础配置的结构比较只出现两个分块字段及既有Talker native环境设置；三阶段 `max_num_seqs=4`、stage0 xgrammar、声音/seed及控制槽不变。可归仓的自造夹逐块数据与聚合收据在 [speech_continuity_v1](../exp/web_demo/speech_continuity_v1/validation.json)。原真实会话和其重放留在私有目录。

### 分块选择

同一块RTX PRO 6000 Blackwell，生产seq4/native配置；条件按4:25→8:25→4:8→4:12串行运行，每组6条自造中/英文文本、每条重复2次，共48次完整TTS。每组暖启动列仅取第2次的6条；全部12条（包括首次较慢调用）保留，最大缓冲需求使用全部12条。重复与跨配置共享文本不算独立新样本；这些不是历史seq1正式延迟成绩。

| 首块/后续帧 | 暖首PCM中位数 | 暖RTF中位数 | 全部样本最大连续播放起播需求 | 暖样本块数范围 |
|---|---:|---:|---:|---:|
| 4/25 原配置 | 260.8 ms | 0.3426 | 352.4 ms | 2–4 |
| 8/25 | 356.5 ms | 0.3412 | 9.2 ms | 2–3 |
| 4/8 | 249.3 ms | 0.3606 | 0 ms | 4–8 |
| **4/12 采用** | **254.9 ms** | **0.3426** | **14.7 ms** | **3–6** |

这里的“起播需求”是以首块到达为起点，逐块比较到达时刻和此前音频累计时长得到的理想下界，不含用户网络抖动、浏览器调度或句间空档。4/12在本组中保持原先首块/RTF量级，同时减少大后续块的等待；8/25增加约96 ms首块时间，4/8增加解码分块且RTF约高5%。本次不是参数全局最优证明。

逐字协议/声音证明48/48通过；独立SenseVoice严格归一回读每组均10/12相同，失败均为混合语言文本“它表示”回读成同音的“他表示”，未删除失败或放松评分。不能据此宣称48/48声学逐字正确。不同分块下PCM时长有变化：固定长故事4/25为43.957 s、4/12为43.448 s、4/8为42.963 s、8/25为44.003 s；本次没有主观听感/MOS、逐词时长或跨文本音色验收。

### 实际浏览器播放

Chromium 151，真实WebAudio + 生产SpeechPipeline/TTS；固定回复文本不经过Actor输入判定。共14个串行用例：10个完整播放、4个按计划取消；含48次TTS请求（包括取消请求）及12次播放中音频路由调用。这些数字与48次无回压TTS探针分开统计。

| 固定文本 | 原4/25、80 ms、无预取 | 新4/12、350 ms、2 s预取 |
|---|---|---|
| 短句 | 1次断流，累计312.6 ms | 0次 |
| 7句长故事 | 7次断流，累计1069.4 ms | 0次 |
| 取消与旧包注入 | — | 播放停止，旧包未复活 |

原长故事7次断流均落在每句起点后7125样本（296.875 ms）处，与4帧首块被消耗、后续25帧块尚未到达的位置一致。这给出了受控夹的机制证据，不能据此把所有真实会话卡顿归于同一原因。

4/25仅启用新起播/预取，以及8/25、4/8、4/12的新管线短/长用例，也均记录0次断流（8个完整用例）。这说明播放器/预取组合已对这些用例有效；未做完整因子消融，不能把收益全部归给某一个改动。各组长用例均验证3次音频路由请求完成，容量峰值total=2/normal=1；最终4/12路由墙钟约339/186/193 ms，不是正式决策延迟评测。客户端未确认样本均未超过600 ms，PCM队列未超过96000字节（24 kHz、2 s）。

**首响代价**：固定起播余量80→350 ms，增加270 ms。最终短/长夹首包约290/264 ms，排程首响约640/614 ms；这不含VAD/输入判定、物理设备或SSH链路，实际Actor投机生成还能与停顿窗重叠。不能将“首包更早”写成“实际开口更早”。原优化管线长夹的首次首包1301 ms受到夹具延迟加载重采样器影响，已排除首包对比，保留连续播放/控制请求观测；后续夹具在计时前准备输入。

### 完整Actor与兼容性

- 正式demo页面以自造合成麦克风输入真实Omni：讲长故事→明确语音叫停→收到停止ACK→STOP_ONLY静默3秒→新指令→新回复播完，通过；被取消段和新回复均0次断流。
- 另一独立会话连续两轮正常输入，2/2回复播完、0次断流；上述两会话trace dropped=0，无引擎/页面错误。均是虚拟音频设备，不是人类或物理扬声器测试。
- 188项定向pytest通过，覆盖预取容量、顺序/边界、取消/错误关闭源、候选确认门、已播参考、guarded turns、控制容量、原引擎/回归、逐字证明先于PCM释放，以及demo隔离；Node播放/麦克风/遥测测试通过。
- 独立浏览器契约检查通过，包括权限、重连、结束清理、旧包隔离。最初两次浏览器启动因可执行文件路径失败，未产生浏览器模型调用；失败保留，后用已安装Chromium显式路径完成。

本轮四项改动分别是：起播缓冲、模型分块、有界生成预取、逐块/断流观测。没有改输入准入或停止词规则，没有消耗FDB/RB冻结测试窗口。验收支持当前受控样本的连续性改善；真实网络、设备调度、韵律停顿和长时间多会话负载仍需以新增遥测区分。
