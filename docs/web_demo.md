# HumDial 浏览器演示：服务器运行，笔记本展示

页面入口是 `src/static/index.html`，由 backend 的 `/demo/` 提供。纯 HTML/CSS/JavaScript，没有前端构建步骤、CDN、账号密钥或笔记本端 Python 依赖。

模型仍运行在 GPU 服务器；笔记本浏览器负责录音、播放与页面展示。使用 HumDial ActorEngine，展示启动器默认选择独立 `chat-demo-v1` 配置：自然中英聊天、双语 ASR、播放后轮次收尾与启动预热，并启用完整候选投机和开播前续说撤销。原 HumDial 评测配置及默认关闭的行为开关保留。

## 1. 在服务器启动

先用平时的 SSH 方式登录服务器。建议在 `tmux` 中运行，避免 SSH 断开后模型退出：

```bash
cd /root/autodl-tmp/fd-badcat
tmux new -s humdial-demo
bash setup/start_demo.sh
```

等待终端打印 `DEMO READY`。首次加载模型需要几分钟；期间会显示日志路径和就绪进度，不要反复启动。READY 前实际预热 VAD、CPU ASR、Omni 控制判定、流式文本和流式 TTS，并回读中英音频及易被误答的城市疑问句，要求内容匹配（忽略大小写/标点/空白）；失败会阻止启动。backend.log 的 `demo_warmup_done` 记录实际 ASR 后端、转写、`tts_readback_checks` 和耗时。**READY 不代表笔记本麦克风或物理扬声器已经验收**。

启动器按顺序启动：

- Omni 音频模型：服务器 `127.0.0.1:10003`，沿用生产 seq=4 / context=4096 / FCFS 配置。
- 流式代理：服务器 `127.0.0.1:10004`。
- ActorEngine 与页面：服务器 `127.0.0.1:18000`，显式启用 `--streaming --demo-chat`。

启动器不会停止或接管已有进程，端口占用会明确报错。如果 **10003 的 Omni 和 10004 的新版代理已经启动**、只是缺少 backend：

```bash
bash setup/start_demo.sh --backend-only
```

若已有 backend 正在 18000 运行，请直接复用它，或选择另一页面端口，例如 `--backend-only --port 18001`；笔记本 SSH 命令右侧的目标端口也要改成 18001。`--backend-only` 仅检查既有服务，不会改变它们原来的监听地址或模型配置。

如环境路径不同，设置 `FDBC_DEMO_PYTHON` 指向安装了 backend 依赖的 Python；Omni 的 conda 环境沿用现有启动脚本。启动器强制推理端口为本机 10003/10004，不支持任意远程推理拓扑。

## 2. 在笔记本建立 SSH 转发

这一步在**笔记本**的终端运行：macOS/Linux 用终端，Windows 用 PowerShell（需有 OpenSSH 客户端）。复制你平时登录服务器的主机、用户名和 SSH 端口：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18000:127.0.0.1:18000 -p SSH端口 用户名@服务器地址
```

`SSH端口` 是平台提供的 SSH 登录端口，**不是** 18000。若平台给出的连接命令类似 `ssh -p 12345 root@gpu.example.com`，则保留这两个连接参数，把上述 `-N ... -L ...` 参数加进去即可。密钥登录时沿用原来的 `-i` 参数，不要把私钥交给演示观众。

认证成功后终端保持无输出是正常现象。保持此窗口打开，然后在笔记本浏览器访问：

```text
http://localhost:18000/demo/
```

如果笔记本的 18000 已被占用，只改转发左侧端口即可；服务器配置不动：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18080:127.0.0.1:18000 -p SSH端口 用户名@服务器地址
```

此时打开 `http://localhost:18080/demo/`。页面会自动使用同源 WebSocket，无需填写服务器 IP、修改 JavaScript 或另转发 10003/10004。

浏览器的麦克风接口要求安全上下文，`localhost` 可用于本机开发；普通 `http://公网IP:端口` 通常不行。不要双击 HTML 或把静态文件单独放在另一个 HTTP 服务里，页面需要同源的 `/api/demo/info` 和 `/realtime`。[浏览器权限与安全上下文说明](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)，[OpenSSH 本地端口转发说明](https://man.openbsd.org/ssh.1)。

## 3. 现场操作

1. 戴耳机，打开页面。点击“开始对话”，允许麦克风访问；握手完成前不会上传录音。
2. 看到“我在听，你说。”后自然说话，不用按住按钮。页面显示输入音量、实时回复和稍晚到达的 ASR 转写。
3. 在回答前继续说话，会撤销尚未开播的旧回答、合并原问题与续说。已经开播后仍由原语义判定和长打断规则决定，不是所有声音都停。聊天配置通常回答一至三句，已不强制 15 字；正常换话题不会被提示词要求拒答。播放完成后自动进入下一轮，已经开始说的话会继续保留。
4. “静音麦克风”会关闭输入轨的声音并显式上传零值帧，已有回复仍可播放；持续发送静音帧是为了保持既有音频时钟，不是暂停服务器时间。
5. “结束”关闭 WebSocket、停止全部排程音频、释放麦克风；再次开始会分配新的服务器会话，清空上一段页面记录。已有记录也可以在断开后手动清空。
6. “全屏展示”隐藏浏览器外框；“连接与运行详情”显示当前会话、缓冲、计时口径和排错帮助。

每次回复的首文本/首音频时间从浏览器收到 `speech_start` 计时，不含此前的停顿/轮次判定，也不是麦克风到扬声器的端到端延迟。未确认播放缓冲按播放完成回执统计，起播余量仍为 80 ms、服务端信用上限仍为 600 ms。录音电平是真实输入；球体动画仅用于状态提示，不是音频分析仪。

## 4. 排错与关停

| 现象 | 检查 |
| --- | --- |
| 页面打不开 | 服务器终端是否 READY；笔记本 SSH 转发是否仍在；左右端口是否写对 |
| 麦克风按钮报权限错误 | 使用 localhost/HTTPS；地址栏麦克风权限；系统隐私设置；设备是否被占用 |
| 显示未启用流式 ActorEngine | 后端必须是新版，并显式 `--streaming`；不要把 10003/10004 当成页面端口 |
| 电平不动 | 选择正确输入设备；检查硬件静音；断开后可重新选择麦克风 |
| 电平有变化但不回应 | 查看本次 `backend.log`、`proxy.log`、`omni.log`；不要仅凭“连接成功”判断模型健康 |
| 听不到声音 | 确认系统输出设备和音量；网站未被静音；页面在前台；可结束后重连 |
| 背景/锁屏后停止 | 浏览器可能挂起 AudioContext；回到前台重新开始，不累计旧音频补播 |
| `Address already in use` / 端口占用 | 不要杀不明进程；已有推理服务用 `--backend-only`，已有 backend 复用或换端口 |

服务器日志：`exp/web_demo/<启动时间>-<pid>/{omni,proxy,backend}.log`。浏览器新会话的输入音频/转写归档沿用引擎路径：`exp/web-demo-<随机ID>/realtimeout_live/`。ID 由服务器生成；页面清空记录不会删除这些服务器文件。

在服务器启动终端按 `Ctrl+C`，启动器会关闭**本次创建的**服务并释放资源；`--backend-only` 不会停止既有 Omni/代理。`tmux` 分离是 `Ctrl+B` 后按 `D`；重新进入是 `tmux attach -t humdial-demo`。笔记本 SSH 转发窗口按 `Ctrl+C` 只关闭隧道，不会停止服务器模型。

## 5. 给外界展示的边界

这版适合你在笔记本上现场演示、共享屏幕，或受控的单用户体验。它不是已具备鉴权、配额、数据删除策略和多租户隔离的公开 SaaS。

如要让外界直接通过网址访问，需要另行部署同源 HTTPS/WSS 反向代理、身份验证/访问额度、WebSocket Origin 防护、会话并发治理、日志告知与保留策略；不要直接开放裸 backend 或推理端口。新 `humdial-web` 客户端已校验同源 Origin 并使用服务器分配的归档路径，但旧评测 WebSocket 协议为兼容仍然存在，**这些局部保护不能当作公网安全边界**。请勿共享 SSH 私钥或把服务器 IP 的 HTTP 链接直接发给观众。

## 6. 逐轮延迟观测（demo-trace-v1）

后端更新后需重启；笔记本 Ctrl+F5 刷新后重新开始。展开详情应看到“逐轮记录已启用”。旧后端握手不宣布版本时，新页面不会发送遥测，显示记录未启用。

每个浏览器会话写入 `exp/web-demo-<会话ID>/realtimeout_live/events.jsonl`。专用线程逐行落盘，4096 条有界队列，记录故障只报后端警告、不阻断收音；正常断线末行 `trace_closed.dropped` 应为 0。进程硬杀可能损失未排空队列；磁盘错误查看 backend.log 的 `Demo trace unavailable`，不要把缺记录当作零延迟。日志含回复、ASR 和脱敏后的模型消息快照，仍属敏感会话数据；不记录音频包内容或任意客户端字段。输入 WAV 归档规则不变，未设置自动删除。

| 记录 | 解释 |
| --- | --- |
| `vad_done` / `vad_640_done` / `speech_start` / `speech_first_audio` | 服务器三个阶段的边界；`server_ms` 是同一会话单调钟，`t_audio` 仍是驱动业务判据的音频钟 |
| `demo_latency` | ①停顿窗、②轮次判定/继续等待、③文本/分句/TTS/排队，以及 VAD 结束到首音频入发送链路的总和；按 generation/epoch 与 utterance ID 关联，缺锚或续说失配显示 null，不借前轮锚凑数 |
| `llm_dispatch` / `llm_done` / `llm_stale_dropped` / `capacity_acquired` | 请求类别、结果/过期、进程闸门等待；可查 continue / shift / interrupt 分支。容量时间包含在上述阶段内，不再次累加 |
| `candidate_created` / `candidate_dispatch` / `candidate_control_done` / `candidate_speech_started` | 私有预计算轨，仅服务器审计，不向浏览器展示未确认判断/文本；含候选 ID、段尾样本数、请求类别及阶段 |
| `candidate_confirmed` / `candidate_cancelled` / `candidate_result_discarded` | 确认时已准备音频量、预计算墙钟；作废原因、阶段、已准备样本及候选存活时间；存活时间不等于 GPU 计算费用 |
| `speech_playback_started` | 浏览器 WebAudio 开播回执被服务器接受；此后恢复原语义打断规则，不是物理扬声器测量 |
| `socket_first_audio_sent` / `socket_slow_send` | 首二进制包及总排队/发送超过 50ms 的发送；完成只表示服务器 socket 发送返回，不表示笔记本已收到 |
| `client_rtt` | 浏览器每 5 秒一次应用层 ping/pong，同一浏览器时钟差；包含网络、SSH、服务排队和浏览器调度，不是纯网络 RTT，不可除以 2 冒充单向延迟 |
| `client_first_audio` / `client_playback_end` / `client_cancel` / `client_stop` | 浏览器首包、播放完成、取消/结束；包含样本数、断流次数、上传 bufferedAmount。第一包的 `scheduled_lead_ms` 为 WebAudio 起播调度余量，设备 latency 为浏览器估计，均非物理扬声器听检 |
| `input_health` | 每秒一次 reader→actor 排队时长、音频钟和队列深度，帮助辨别服务内部积压 |

服务器 `server_ms` 与浏览器 `client_ms` 不同源，**不能直接相减**。旧的首文本/首音频到达指标仍从浏览器收到 `speech_start` 起算；完整三阶段用服务端指标另列。读完/生成完的文本不是用户实际听到的前缀。观测本身不改变判据；聊天配置的独立行为开关见下节，0.64/2.5/1.5s 阈值保持不变。

### 完整候选管线（actor-candidate-v1）

旧 Actor 异步收音不等于提前推理。新版只在流式 Actor 的显式开关下启用 `src/actor_candidate.py`：借用 TACT 的“候选→确认/作废”思想，不搬入事务工具判据，也不修改 TACT 冻结实现。

1. VAD END 当场复制只读音频、一次编码，并冻结 judge/shift/response 的消息及历史。停顿窗口内运行 judge→必要的 shift→所选回答流→首句 TTS；这些步骤相互仍有依赖，但整条链可以与 640ms hold 重叠。
2. 确认前不发送回复文本/音频、不展示候选、不写用户或助手历史。只预合成第一句，最多积压两句待合成文本，PCM 信用上限 600ms；确认后放开后续句。不是提前生成无限长音频。
3. 原 hold 满足且候选仍对应当前 generation/epoch/turn 时确认；直接释放已经准备好的首音频，不重新生成命中的回答。judge=`continue` 仍走原 2.5 秒继续等待，不绕过它抢答；shift=`yes` 只生成原 shift_re 分支，不泄漏普通回答或将旁路话语写进用户历史。
4. 新 VAD START 在开播回执之前到达，撤销 judge/shift/文本/TTS 的整条真实 SSE 请求与队列，保留原问题、间隙和续说音频，下一个 END 重新冻结。旧候选不升级轮次，不触发旧 ASR，不污染历史。重置/断线同样撤销；candidate ID、epoch 与 utterance ID 拦截迟到结果、已排队旧文本和旧音频。

所有未确认 judge/shift 也走 **normal** 槽，不侵占 interrupt/judge 的控制保留槽。每会话仅一个当前候选；取消传播到异步 HTTP，关闭后才释放容量。迁移另外修复了超时等待与取消同时发生时可能吞掉取消、导致管线卡在候选门上的竞态，以及取消尚未发出的文本仍在 socket 队列里的漏洞。私有流失败不会展示失败草稿：确认后最多重新启动一次选定回答链；已公开的失败流不重放部分文本。

助手历史和 ASR 延迟到浏览器确认开播；晚到 ASR 用回答版本号防覆盖。浏览器在既有麦克风帧发送前检查 WebAudio 时间轴并上报 `started`（旧客户端用首个正数 played_samples 回执）；**不是收到音频就算开播**。跨网络传递仍有时延，边界以服务器接受回执为准，不能声称精确物理“用户已听见”；开播后也仍保存完整生成文本，不是听到前缀对齐。

页面详情新增“投机”状态，展示确认时已经准备的音频毫秒数及重叠预计算时间。若理想无争用且 judge 接受结束，首音频从旧 `H+J+G` 调度形态变为 `max(H,J+G)`（J 含必要 shift，G 含文本与首句 TTS）；不是固定减去 640ms 或零成本保证，模型较慢时仍会在 hold 后等待。

**输入边界明确变化**：原 Actor 在 hold 后拼 BUFFER，包含窗口后的尾部音频且历史可能已更新；候选路径冻结在 VAD END。因此只承诺新路径投机开/关使用相同快照，不承诺与原路径输入等价或质量不变。可以在 WebSocket config 中指定 `speculative_response: false` 仅关闭提前派发，保留取消与段尾快照作对照；客户端不能反向开启服务器关闭的投机。彻底恢复旧路径需在独立 demo YAML 同时关闭两个开关并重启。实际收益、作废成本及听感仍需笔记本试用。

## 7. 独立聊天配置（chat-demo-v1）

`--demo-chat` 将 `configs/demo_chat.yaml` 覆盖到基础 YAML，限定 ActorEngine。展示启动器自动添加它；直接运行 `src/backend.py --streaming` 而不加该参数仍使用原 HumDial 配置。网页连接后会显示实际 profile，不能只凭页面版本判断后端已升级。

- **轮次收尾**：流式音频收到实际播放回执后只结算一次，递增轮次并回到 LISTEN；保留正在收取的插话音频。旧 interrupt 结果不能跨轮生效，已结束的插话片段改走新一轮 judge。迟到 ASR 通过轮次 ID 与回复配对，不再靠两个列表的到达顺序。流式生成失败也会清理并回到监听。非流式兼容路径只能用音频钟估计播放结束，不能声称是浏览器回执。
- **启动预热**：独立于用户会话，无用户历史或预热回复广播；总预算 90 秒。使用仓内自造样例，不读取用户录音。预热失败会退出启动流程。
- **双语 ASR**：选择已在仓内的 SenseVoice int8 / CPU / 2 线程，不占 Omni GPU。YAML 的 `asr` 段现在实际应用到延迟初始化的识别器；显式 `FDBC_ASR_BACKEND`、`FDBC_ASR_PROVIDER`、`FDBC_ASR_NUM_THREADS` 环境变量优先。切换后需重启 backend，已加载的模型不会在会话中热换。ASR 用于页面转写与后续文字历史，当前语音判定和回答仍由 Omni 直接听音频。
- **聊天提示词**：不再强制 15 字、不无条件附和；支持追问、纠正和中英切换，正常换话题不判 shift。仅明确对第三方说话才判 shift；这不是声纹识别。仍无工具和实时查询能力。
- **二分类容错**：judge/interrupt 仅接受 `continue|switch`，shift 仅接受 `no|yes`，拒绝空串、含糊解释及子串碰撞。首次无效或请求异常时最多修复一次，与首次请求共享 `llm.decision_timeout_s` 总预算；超时不继续重试。仍失败时 judge/interrupt 回退 `continue`，shift 回退 `no`。judge 的回退继续走已有等待超时机制，避免空输出立刻抢话；shift 不会再静默卡住。`control_validation` 记录原始标签、尝试次数、修复/回退/超时；不套用 TACT 工具 JSON 解析器。
- **完整投机与开播前撤销**：`speculative_response: true`、`cancellable_response: true` 仅在新流式协议下生效；前者必须伴随整链撤销保障，不能单独禁掉安全撤销。展示配置默认开启，基础 HumDial 配置默认关闭。机制和限制见上节。

历史仍保存完整生成文本，不是精确“用户已听到前缀”。预热不是持续健康检查，也不能消除每次会话的所有初始化耗时。提示词不是事实性保证：自造天气问题的功能烟测仍出现无依据的天气断言，不能据此宣称聊天质量已验收。这些变更没有修改 RB/TACT 评测配置或建立新的正式成绩。

## 8. 逐字合成修复（verbatim-choice-v1）

原实现把每个分句作为新的 user 消息交给 Omni，只用 system 提示“逐字读”，实际会重新回答问题。例如“你那边怎么样？”被读成“我这边一切都好……”；“你能告诉我你在哪个城市吗？”变成 14 秒的模型自我介绍。它不是分句乱序或投机误打断，而是音频内容背离页面文字。先前短输入/播放契约检查不能证明合成忠实性。

流式 Actor 现在用 `module.verbatim_tts_payload()`：通过 `structured_outputs.choice=[原文]` 对 Thinker 做单一字面值解码约束，再由同一个 Omni 的 Talker 合成；不更换模型、不增加第二套 GPU 服务。`choice` 不是正则拼接，问号、引号等不会被当作语法。[vLLM 结构化解码说明](https://docs.vllm.ai/en/latest/features/structured_outputs/)

- 当前 vLLM-Omni 版本的普通请求转换漏掉了 `structured_outputs`。启动器先执行仓内 `scripts/patch_omni_verbatim_tts.py`：仅向 comprehension stage 转发该参数，音频阶段仍用原采样配置；另给受约束的文本末帧增加 `fd_text_finish_reason`。适配幂等，未知或半修补的依赖源码布局会拒绝启动，不能静默忽略约束。普通无约束请求不改动。
- 请求同时返回文本和音频；`stream_transport.audio_stream(expected_text=...)` 逐块检查原文前缀，要求最终全文严格相同且文本以 `stop` 正常结束，才放行音频。Omni 通用 `finish_reason` 会等全部模态结束，所以使用上述单独文本完成证明；只等一个短句的文本，不等整段回答或整段音频。提前到达的 PCM 暂存有 512KiB 硬上限，取消仍关闭 HTTP。
- 上游不支持约束、回传缺失、文本不符、超长或截断均显式失败，不回退到自由聊天合成。单句 UTF-8 最多1024 bytes，禁止模型控制标记；文本生成预算按 byte-BPE 安全上界留足 EOS 余量，避免旧 max_tokens 环境值把指定原文截断。确认后的音频仍受原 600ms 播放信用和 utterance ID 栅栏约束。
- `tts_omni_stream()` 接入新契约；旧整 WAV `omni_tts_payload()` / `tts()` 仅为 legacy/RB/TACT 复现兼容保持原样，**不把它们称为已修复的可靠逐字 TTS**。展示页应通过新版流式 Actor 使用本修复。

启动预热与 `scripts/check_verbatim_tts.py` 同时核对真正合成音频的 ASR 回读；浏览器检查另保存它收到的逐 utterance PCM，供整段多句回读，避免只看页面文本就宣布语音通过。它仍不是每个线上回答的声学证明：专名、数字读法、韵律和设备听感要单独评估；ASR canary 是窄范围护栏，不能当成通用音质分数。

## 开发核验

```bash
node tests/test_stream_demo.cjs
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python -m pytest tests/test_tts_verbatim.py tests/test_actor_candidate.py tests/test_chat_demo.py tests/test_engine.py tests/test_speech_stream.py tests/test_web_demo.py -q
```

浏览器检查脚本 `scripts/check_web_demo.py` 使用 Playwright 和真实 Chromium 的 WebAudio/AudioWorklet；默认使用明确标注的合成协议测试服务器，不调用模型、不提供模拟回答给正式演示页。`--live-url` 可接真实 backend 与自有 mono PCM16 WAV 做链路烟测：默认播一次，`--turns 2` 在 20 秒静音后再播一次以检查跨轮收尾和 ASR 配对；不循环输入。工具的截图/机器检查不能代替笔记本真实耳机、麦克风、声卡及外放回声听检。Linux 截图环境需安装中文字体，否则系统字体缺字可能显示方框；页面不依赖在线字体服务。

`--no-speculation` 只在检查脚本的 live 握手关闭提前派发，适合与同输入的正常 live 烟测做顺序比较；不是正式延迟基准，不能和旧 seq1 数据相减。

协议细节见 [streaming_speech.md](streaming_speech.md)，生产容量见 [production_capacity.md](production_capacity.md)。
