# ActorEngine 流式语音（可选）

用途：HumDial 半级联全双工 demo。默认关闭，不改比赛提示词、15 字限制、轮次/打断判据、音频钟阈值；`backend_legacy.py` 和 TactEngine 的旧分句函数不动。启用后播放提前、分句韵律及取消行为会变化，不能主张声学输出或所有交互轨迹逐字节等价。

## 启动

笔记本展示推荐 `bash setup/start_demo.sh` + SSH 转发。完整的服务器/笔记本分步命令、页面说明与关停方法见 [web_demo.md](web_demo.md)。

如果要手动启动，在仓库根目录的**三个独立终端**分别启动原有 Omni 音频服务、更新后的代理、backend：

```bash
bash setup/start_qwen3omni_audio.sh
bash setup/start_qwen3_proxy.sh
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python src/backend.py --streaming
```

浏览器打开 `http://localhost:18000/demo/`，点击“开始对话”。远程使用 HTTPS/WSS 或 SSH 端口转发到 localhost，否则浏览器不开放麦克风。推荐耳机；页面请求浏览器的回声消除，但这不等于经过扬声器回声评测。不要直接无鉴权暴露公网：认证、限流、日志隐私和多租户隔离不是本补丁的范围。

也可设置 `engine.stream_response: true`。服务器开关 **与**客户端握手 `audio_protocol: "pcm16.v1"` 必须同时存在。旧客户端仍得到 `tts_done + 完整 WAV`，原评测脚本不用改。仅支持 ActorEngine phase a、realtime、Omni 原生 TTS；不将流式链伪装成 IndexTTS 或离线 injected 模式。

## 连续管线与边界

`SSE 文本增量 → 中英分句 → SSE TTS 音频块 → 内存 WAV 解码 → 原采样率 PCM16 → 浏览器播放`

分句保留原文，不在中文句号处等待空格；保护小数、常见缩写、引号，长句优先在空白/逗号处软切。它是启发式边界检测，不是语言学句法模型。新分句器独立于 Phase-B 的 `split_sentences()`。

TTS 请求逐句有序执行。文本生产和 TTS 消费是独立任务，最多排队两句；文本总长设防御性上限 16,384 字符，超限报错而不静默截断。保留一个上游音频块在内存，切为默认 40 ms 的网络包；不在首音频路径写 WAV、读 WAV、落盘重采样。输入 ASR 归档仍沿用旧路径。

客户端默认预留 80 ms 起播余量。`stream_buffer_ms: 600` 限制**已发送但未确认播放**的样本，而不是仅限制 WebSocket 队列长度。浏览器按 AudioContext 播放完成回报信用；服务端在信用不足时停止发送。此上限不包括模型内部缓冲、一个上游 WAV 块或操作系统音频输出延迟。5 秒无有效播放进展/发送阻塞会取消或断开；后台挂起的页面不应继续无限合成。

判定为短打断、长打断、回复替换或会话重置时，取消文本/TTS HTTP 流、丢弃待合成句子和旧音频包；浏览器按 utterance ID 停掉已经排程的旧音频。普通 `continue` 不取消，也不引入新的 VAD 即停策略。代理使用异步 HTTP 和 SSE 透传，上游取消会关闭实际连接；GPU 已在执行的单个 kernel 不保证瞬时抢占。

独立 `--demo-chat` 配置另外启用候选投机和 `guarded-turns-v1` 输入准入：原始 VAD 只暂缓未开播候选，确认真实续说才撤销；播放期间按 keep/stop_only/yield_wait/yield_ready 分开停止与回答，噪声、含糊输入和附和不直接打断。此为独立策略版本，详见 [web_demo.md](web_demo.md) §9；基础 HumDial 开关仍默认关闭。

生成过程中出错则显式 `speech_error` 并停止该回复；没有输出任何文本前保留 response 的一次超时重试与道歉回退，有部分输出后不重试，避免重复播报。TTS 失败不会伪造完成。未开启输入准入时历史仍使用完整生成文本；开启后对已取消回复使用确认播完的完整句子与打断标记，审计史保留全文。仍不是词级或物理“听到前缀”对齐。

流式 TTS 已从提示词式朗读升级为 `verbatim-choice-v1`：Thinker 单一原文解码约束 + 文本全文/结束证明校验，防止 Omni 把待读疑问句重新回答；早到音频在校验前隔离（512KiB硬上限），没有证明不放行。需要启动器安装的窄范围服务适配，机制、旧整 WAV 兼容边界和音频回读验收见 [web_demo.md](web_demo.md) §8。

当前单用户生产部署为三阶段 `max_num_seqs: 4`、stage 0 上下文 4096、FCFS；backend 全局最多放行三个普通请求，给 judge/interrupt 留一个总容量槽。这样流式文本与逐句 TTS 可以重叠，同时不会让普通合成队列堵死控制判定。它不是 GPU 抢占：已经运行的 kernel 不会被 judge 中断。完整准入语义、历史预算和真机收据见 `docs/production_capacity.md`；旧 `seq=1` 测量由独立串行配置保留。

## 协议

握手仍为 `/realtime` 的首条 JSON：

```json
{"event":"config","data":{"exp":"unique-session-id","lang":"live","audio_protocol":"pcm16.v1"}}
```

客户端上传 mono float32 / 16 kHz，每包 256 样本。浏览器 worklet 在设备不接受 16 kHz 时做带跨块累积的重采样，保持输入音频钟不丢余数。

新展示页若 `/api/demo/info` 声明 `guarded_turns: true`，握手另外发送 `input_protocol: "pcm16.ref.v1"`。此时上传改成1040字节：小端头 `<4sIII>` = `FDM1`、从0递增序号、256、16000；后面256对交错 `int16 mic, int16 render_reference`。两个通道来自同一 worklet 音频钟；参考是实际播放排程的输出而非任意刚收包音频。静音只清 mic，保留参考。缺协商仍用上述旧 float32；未知协议或错序/尺寸拒绝，不自动猜格式。

服务器下发：

- `speech_start`：`utterance_id`、`protocol`、`buffer_ms`。
- `speech_text_delta` / `speech_text_done` / `speech_sentence`：携带原文增量/全文/合成句子；现有 `llm_done` 全文日志保留且只写一次历史。
- `speech_first_audio`：首包交给发送队列的时间，不是扬声器出声时间。
- 二进制：16 字节小端头 `<4sIII>`，依次为 `FDS1`、utterance ID、从 0 递增的包序号、采样率；后面为 mono little-endian PCM16。**不是 WAV**，不能交给旧客户端解码。
- `speech_audio_end`：总 `samples`、`rate`、`packets`；合成发送结束不等于播放结束。
- `speech_cancelled` / `speech_error`：清空对应 ID，不接受旧包复活播放。
- `speech_hold {utterance_id,held}`：新准入专用，只暂缓尚未起播的排程，保留有界 PCM；已起播则忽略 hold 并回报 started，不因原始 VAD 暂停旧回答。它不是取消后恢复播放。
- `input_waiting / input_ignored / input_notice`：明确停止、等待续说、输入被忽略或长度限制；不凭这些事件制造新回答。

客户端按累计播放样本回报（不是收包确认）：

```json
{"event":"playback_progress","data":{"utterance_id":1,"played_samples":1920,"started":true,"ended":false,"underruns":0}}
```

ID、单调性与 `played_samples <= sent_samples` 均检查；服务器收到真正的尾部播放确认后只记录一次 `speech_played`。`playback_autoend: false` 的 HumDial 默认状态语义继续保持。

`started` 是可选兼容字段：仅 WebAudio 时间轴到达首包排程时间后为 true，可以早于第一块的完成回执，此时 `played_samples` 仍为 0。到包、建缓冲或开始排程均不应报 true；旧客户端缺此字段时以首个正数播放样本回执确认开播。候选版据此关闭“开播前自动撤销”的窗口并开始 ASR/提交完整回复历史。回执过网有延迟，不代表物理声卡听检。

新页取消时另报 `playback_stopped {utterance_id,played_samples}`，计入当前包已经排程播放的部分，未开始的未来包不算。服务端仅对已取消、已知 ID 且样本数合法的记录更新句子前缀；重置后迟到回执无效。`input_settings` 只上报实际 echoCancellation/noiseSuppression/autoGainControl/sampleRate/channelCount，不上报设备 ID；“申请 AEC”不等于“实际声学验收通过”。

新版 HTML 页面使用附加握手 `{"event":"config","data":{"client":"humdial-web","audio_protocol":"pcm16.v1"}}`；服务端检查同源 Origin、生成归档会话 ID，并在 ActorEngine 初始化后发送 `demo_ready {session_id, protocol}`。浏览器收到此消息后才上传音频。只读 `/api/demo/info` 仅报告流式配置，不宣称 GPU 推理健康。其他旧客户端握手/归档命名不变。

## 验证与测量

```bash
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python -m pytest tests/test_speech_stream.py tests/test_engine.py tests/test_regression.py tests/test_w3_d456.py -q
node tests/test_stream_demo.cjs
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python scripts/smoke_streaming_ws.py --input OWN_MONO_16K.wav --repeats 3 --output exp/streaming_demo/new_receipt.json
```

探针串行交替全 WAV / PCM 流式请求；检查包序、采样率、尾计数与 600 ms 信用上限，保存可复查的合成 WAV 和 JSON。使用自造或本人有权使用的输入，不自动选择 HumDial/FDB/RB 测试集。

`post_hold_first_audio_ms` 从客户端收到 `vad_640_done` 到收到首音频包，包含 judge/response/TTS 和传输，但**不包含已经过去的 0.64 秒 hold**，也不是物理扬声器延迟。该 WS 探针的播放器是实时节拍模拟；浏览器逻辑另有 Node 契约测试，以及 `scripts/check_web_demo.py` 的真实 Chromium/WebAudio/AudioWorklet 检查（含虚拟麦克风接真实 Omni 的单输入烟测）。仍未做笔记本真实麦克风/声卡听检。短输入小样本收据只能证明链路与方向，不代替正式串行专机 P50/P95、长回复流畅度和打断恢复测量。流式 `llm_done.infer_time` 包含本地句子队列回压，不能直接当模型纯推理时长。新版页面详情区的计时起点则是 `speech_start`，不要和本段探针口径混用。

实跑事实、验收数量和后续状态以 `AGENTS.md` 的流式增量条目为准。
