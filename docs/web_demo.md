# HumDial 浏览器演示：服务器运行，笔记本展示

页面入口是 `src/static/index.html`，由 backend 的 `/demo/` 提供。纯 HTML/CSS/JavaScript，没有前端构建步骤、CDN、账号密钥或笔记本端 Python 依赖。

模型仍运行在 GPU 服务器；笔记本浏览器负责录音、播放与页面展示。默认 HumDial ActorEngine，不改提示词、15 字要求或打断规则。

## 1. 在服务器启动

先用平时的 SSH 方式登录服务器。建议在 `tmux` 中运行，避免 SSH 断开后模型退出：

```bash
cd /root/autodl-tmp/fd-badcat
tmux new -s humdial-demo
bash setup/start_demo.sh
```

等待终端打印 `DEMO READY`。首次加载模型需要几分钟；期间会显示日志路径和就绪进度，不要反复启动。看到 READY 表示模型列表、代理接口和流式后端配置的检查通过，**不代表麦克风、实际生成和扬声器已经验收**。

启动器按顺序启动：

- Omni 音频模型：服务器 `127.0.0.1:10003`，沿用生产 seq=4 / context=4096 / FCFS 配置。
- 流式代理：服务器 `127.0.0.1:10004`。
- ActorEngine 与页面：服务器 `127.0.0.1:18000`，显式启用 `--streaming`。

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
3. 在回答时继续说话可测试打断；是否打断沿用模型判据，不是“检测到声音就停”。当前 15 字短答较短，演示时不要把它误当成长答版模型。
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
| `socket_first_audio_sent` / `socket_slow_send` | 首二进制包及总排队/发送超过 50ms 的发送；完成只表示服务器 socket 发送返回，不表示笔记本已收到 |
| `client_rtt` | 浏览器每 5 秒一次应用层 ping/pong，同一浏览器时钟差；包含网络、SSH、服务排队和浏览器调度，不是纯网络 RTT，不可除以 2 冒充单向延迟 |
| `client_first_audio` / `client_playback_end` / `client_cancel` / `client_stop` | 浏览器首包、播放完成、取消/结束；包含样本数、断流次数、上传 bufferedAmount。第一包的 `scheduled_lead_ms` 为 WebAudio 起播调度余量，设备 latency 为浏览器估计，均非物理扬声器听检 |
| `input_health` | 每秒一次 reader→actor 排队时长、音频钟和队列深度，帮助辨别服务内部积压 |

服务器 `server_ms` 与浏览器 `client_ms` 不同源，**不能直接相减**。旧的首文本/首音频到达指标仍从浏览器收到 `speech_start` 起算；完整三阶段用服务端指标另列。读完/生成完的文本不是用户实际听到的前缀。本次观测不改 prompt、15字、0.64/2.5/1.5s、播放结束状态或投机开关。

### 为什么异步仍会出现阶段累加？

ActorEngine 的异步派发让收音不中断；它没有 `speculative_dispatch`。该开关在 `engine_b.py` 的 Phase-B TACT 路径，VAD END 时预计算决策，hold 确认前结果保持无效、续说则丢弃。TACT 普通与投机分支都通过 `_cumulative_prefix(anchor)` 截取相同段尾；Actor 目前在 hold 到期后拼 BUFFER（包含随后音频），不能直接提前调用并宣称输入逐字节等价。Phase-B 还把 judge→shift→response 换成一个事务决策，不能整体搬来当作 HumDial 性能开关。

可研究的独立方案：冻结候选输入/历史，投机 judge 或回答/TTS，在原判据确认前禁止播放、前端展示和写历史；续说、shift、取消、重置需整链作废。只提前 judge 主要重叠①②；要重叠③还须提前回答/TTS。若从 VAD END 跑判定→生成，在成功命中且无争用时，理想式从 `H+J+G` 变为 `max(H,J+G)`；若只投机判定则为 `max(H,J)+G`。这是调度上界，不是实测提升承诺，不能消除原 hold 门槛。

还须验证：少了真实尾部音频是否改变判定、历史快照是否一致、废弃率/取消是否真正释放 HTTP，以及 seq4 的 GPU 争用是否拖慢 judge。W3 已有 full-engine 投机记录为 604 派发/197 确认/407 作废（67.4%），仅说明成本确实存在，不是本 demo 的预测值。因此可保持对外提交规则，但不应在未验证时称为“无条件、无成本、逐位等价的纯提升”。本补丁只做观测，没有启用这项优化。

## 开发核验

```bash
node tests/test_stream_demo.cjs
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python -m pytest tests/test_web_demo.py tests/test_speech_stream.py -q
```

浏览器检查脚本 `scripts/check_web_demo.py` 使用 Playwright 和真实 Chromium 的 WebAudio/AudioWorklet；默认使用明确标注的合成协议测试服务器，不调用模型、不提供模拟回答给正式演示页。`--live-url` 可接真实 backend 与自有 mono PCM16 WAV 做单条链路烟测：探针为输入加前后静音并只播一次，避免 Chromium 默认循环文件导致永不满足停顿判定。工具的截图/机器检查不能代替笔记本真实耳机、麦克风、声卡及外放回声听检。Linux 截图环境需安装中文字体，否则系统字体缺字可能显示方框；页面不依赖在线字体服务。

协议细节见 [streaming_speech.md](streaming_speech.md)，生产容量见 [production_capacity.md](production_capacity.md)。
