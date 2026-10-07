# Demo 可靠性：传输恢复、口语文本、续说音频上下文

适用范围：`configs/demo_chat.yaml` / 流式 Actor。机制版本分别为 `model-transport-v1`、`spoken-text-v1`、`pending-user-audio-v1`。不修改 legacy、HumDial 或 TACT 的冻结配置，不引入第二 ASR、停止词白名单或新的强制回答计时器。

## 1. 传输与 502

### 拓扑

正常启动 `bash setup/start_demo.sh` 默认 `--inference-mode direct`：backend 的路由、回答和 TTS 全部直连 `127.0.0.1:10003/v1/chat/completions`。10004 原来只是 HTTP/SSE 代理，并不是另一套模型功能，所以可以从 demo 必经链路中移除。页面/backend 仍是18000；不把浏览器直接接到模型端口。

`--inference-mode proxy` 保留10004兼容路径；`--backend-only` 与两种模式可组合。直连模式不要求10004空闲或存活，也不接管独立启动的代理。启动器同时覆盖 `FDBC_QWEN_URL` / `FDBC_OMNI_TTS_URL`，防止继承旧环境后文本与TTS误走不同端口。`exp/web_demo/<启动批次>/topology.json` 保存选择结果；会话manifest保存有效端点。

通用适配器、旧评测脚本的缺省端口仍是10004。独立运行这些脚本且想复用直连demo时，显式设置上述两个环境变量为10003，不悄悄改变其旧契约。正常demo启动还开启vLLM的request-ID响应头。

### 有界恢复

- `engine.tts_transport_retries: 1`：只有当前句尚未向管线交付任何PCM，且发生502/503/504、连接/传输异常或网络超时，才允许150ms可取消退避后再试一次。
- 每次尝试使用新HTTP连接、独立容量租约、独立call/case ID，同一个 `tts_operation_id` 连接它们。重试不重新生成回答，也不重播前面已合成的句子。
- 已交付过PCM、格式/逐字证明错误、400/429等不会自动重试。持续失败明确终止为 `tts_unavailable`；私有候选也不能重启整个回答来重置重试预算。
- 取消/断线始终传播；取消中的生成器不恢复、不重发。错误通知不再被紧随其后的取消栅栏丢弃，页面保留“回答未完成”，不改写为“播放完成”。
- 兼容代理使用 `force_close=True`，不跨请求复用上游连接；透传状态码、关联ID和取消，保存结构化连接/流故障。代理自身不对所有POST做自动重试。

历史502只能定位为代理在取得上游响应头前的连接失败，当时日志未保存具体异常类型。旧keep-alive连接竞争是可能解释，不是已证实根因；本次移除必经代理并补足下一次故障的证据，不倒推“历史原因已确定”。

### 状态保存与排错

模型调用 `model_call_done.transport_state`、请求span及case `outcome.transport` 保存：安全端点、客户端call/request ID、上游request ID、connect/headers/stream/done阶段、HTTP状态、异常/底层原因类型及errno、超时设置、响应头/总耗时、接收字节/SSE事件数、是否收到DONE、输出PCM样本数。异常文本、URL凭据/查询串、任意响应体/headers不进入这组诊断字段。

`tts_recovery` 的retry/recovered/failed、case的attempt/operation ID和 `speech_error.pipeline_state` 保存已发送/ACK播放样本、采样率、包序号、待合成句数及预取字节。即使case配额已满，trace仍有调用与恢复状态；写盘失败和trace队列溢出必须查记录健康，不保证硬杀进程后零丢失。

排查顺序：

1. 页面详情取得session ID，打开 `/demo/diagnostics.html` 或运行 `python scripts/demo_diagnostics.py show SESSION_ID`。
2. 从 `speech_error` 找utterance/tts operation，再从request spans查看各attempt。`phase=headers,http_status=502` 与 `phase=connect,error_type=...` 不混为同一个原因。
3. 在对应启动批次的 `topology.json` 核对direct/proxy；用request ID关联 `omni.log`，只有proxy模式才有 `proxy.log`。
4. 确认 `trace_closed.dropped=0`、case入队/落盘与容量释放。录过完整opt-in多轨的会话可做Actor无模型重放；已保存的失败调用会重现原错误类别及已输出PCM前缀，零PCM失败不要求虚构一个output.wav。

原始会话、模型输入输出和可选音轨仍只保存在私有0700/0600目录，受原有配额与保留策略管理，不提交到仓库。

## 2. Markdown 与非口语装饰

仅靠“不要Markdown”的prompt不足以约束输出。`engine.spoken_text_normalization: true` 在原分句器之前建立一个共享的口语文本流：

`原始模型delta → 流式格式规范化 → 页面/参考/历史 + 原分句器 → 严格逐字TTS → PCM`

- 去除标题、引用、列表编号、粗斜体、删除线、链接语法、代码围栏/语言标签、表格分隔线及正文emoji装饰，保留可见文字、链接标签、图片alt和代码内容。
- 编号必须在分句之前移除，不能把`1.`作为独立句送TTS。纯格式/装饰片段不成为TTS请求；整轮只生成格式则明确报错，不假装完成。
- 代码块/inline code中的文字、下划线、普通数字、日期小数和算式保留；常见emoji算符映射为普通算符。不是对用户原话执行此清理，也不改音频输入。
- 普通句和闭合的行内格式仍可逐句起播；未闭合格式只等到闭合/行尾/流末，缓冲上限4096字符。原始输出总上限仍独立计算，不能靠删格式绕过限制。
- 页面增量、分句TTS、已播参考和历史共享同一规范文本，不允许页面显示一份而TTS偷偷删另一份。原始模型delta及长度续写prefix保留在原模型调用链，续写不会拿清理后文字替代模型原生前缀。
- `verbatim-grammar-v2` 仍对规范化后的每句严格证明，不放松grammar、不允许自由聊天回退。证明Thinker文本一致不等于证明Talker实际声音永远正确。

真实服务对照已发现Markdown加粗及纯emoji可引起几十秒低能量输出；清理这类输入比盲目剪掉PCM静音更可控。`tts_complete`额外记录100ms窗RMS阈值0.003下的低能量总时长/最长连续段；连续超过5秒产生 `tts_audio_warning`。这只是异常线索，不是听感/忠实性gold，也**不自动剪音频**。

90秒仍只是原31样本诊断的观察上限，不是运行时回答上限。自然长回答可以合法超过90秒；不能通过强行截短来把观察结果改成“全部完成”。

## 3. 续说路由必须听到已保存的用户音频

旧实现会为 `yield_wait` 保存前文，或为未开播候选保存原问题，但只把新片段发给路由；只有新片段先通过ready准入，前文才进入回答音频。孤立续说片段因此可能一直keep，保存的前文没有帮助判定。

`engine.input_continuation_context: true` 修正三条生命周期路径：

- 等待用户补完时，将已接纳的pending音频与新增片段一起发给Omni。
- 候选尚未开播时，路由也附带候选拥有的原问题；接纳后该前缀移交pending缓存，后续判定不重复拼接。未接纳的噪声不能提前撤销候选。
- 回答已经开播、用户很快继续补充时，在输入起点冻结近期原始用户音频，作为**仅供路由**的上下文。明确标注旧问题已在回答、只判断新增意图，附和仍应keep。旧音频不重复进入本轮ASR/回答，原来的已播历史和新音频契约不变。

播放期采用**同一次请求内两个独立音频块**，第一块明确为此前用户语音，第二块为新增输入；`transcript_scope=new_audio_only`只转写第二块，供既有 `played-reply-v1` 条件复核使用，不能把旧问题拼进“用户刚才的短答”。只在单块拼接音频上标注时间点的候选实现曾实际把附和误判ready并复制旧问题，已放弃，未部署。待完成/未开播上下文则仍转写完整拼接音频。两者的来源和样本边界都进trace/case；没有新增模型调用或ASR。

音频阶段仍不发送助手参考原文或ASR转写。消息只额外注明音频中新增片段的起点和来源，要求最新stop/wait优先、旧前文不能授权当前噪声/回声/第三方讲话。先对**新增片段**做原有声学拒绝，再附前文；关闭本开关恢复原始单片段输入。

pending与新增音频共享原20秒音频预算。pending以音频钟计算20秒间隔到期，仅在下一段输入开始时清理，不主动生成回复。播放期可选前文也只限20秒音频钟内；若前文与新输入相加超预算，只省略可选前文，不挤掉本次新输入。stop/ready/reset清理缓存；多段wait只累计一次。开播ACK竞态仍需按playing情境重新判定，不能用未开播决策强行打断已经播放的回复。

本修复不禁止用户续说前开口，也不把“早开口”统一算错。播放期准入/已播句复核仍保留；本机制不是无限历史音频或声源识别器，也不宣称转写、意图和第三方归属全部正确。

此前已被模型判为keep并丢弃的语音不自动复活：本轮仍有首段误听后被拒、后半句答偏的案例。它不是“已接纳前文没有进入路由”的缓存接线缺陷；不以恢复所有拒绝输入的方式绕过准入规则。

## 4. 验证与回退

自动回归：`tests/test_transport_recovery.py`、`tests/test_spoken_text.py`、`tests/test_continuation_context.py`，以及现有Actor/guarded/response completion/diagnostics/TTS/浏览器测试。

真实验证脚本（均每次新建私有输出目录、串行使用GPU）：

```bash
# 真实Omni + 自造样例 + 生产页面/Actor；私有TTS转发器注入故障
python scripts/check_demo_reliability.py --output NEW_PRIVATE_DIR --chromium CHROMIUM_PATH

# 原来的31条历史train样本；原录音/模型文字只在内存，落盘仅脱敏事件与计数
python scripts/check_pause31_reliability.py --output ANOTHER_PRIVATE_DIR \
  --chromium CHROMIUM_PATH --review-seconds 3600
```

31条不是独立holdout，浏览器虚拟麦克风不是物理设备；会话串行，但仍用生产seq4而非历史seq1延迟配置。观察后需分别核对无回答、错误、未播完、正常长内容及续说是否被理解。自造故障注入只证明恢复/不重播/可观测契约，不是线上502发生率。

### 2026-10-07 验收收据

脱敏聚合与自造夹：[`exp/web_demo/demo_reliability_v1/validation.json`](../exp/web_demo/demo_reliability_v1/validation.json)。完整私有轨迹不归仓。

| 同一31条历史输入，用户输入结束后最多观察90秒 | 修复前 | 首轮修复回放 |
| --- | ---: | ---: |
| 有回答且已播完 | 20 | 30 |
| 观察结束仍未播完 | 7 | 1 |
| TTS错误导致中断 | 1 | 0 |
| 完全没有开播 | 3 | 0 |

首轮包含传输、口语规范化、等待/未开播续说修复；随后才补充播放期双音频块。31条中199个完成TTS句子的最长连续低能量段为1.8秒，未见此前几十秒长尾；路由fallback、引擎/浏览器错误均0，关闭后任务/容量排空。最小源音频对齐相关0.852874，因此仍按虚拟麦克风受控回放解释，不当作物理设备成绩。

唯一90秒未播完的输入为自然长回答，逐句低能量最长1.6秒、无错误。最终版本把该条及播放期续说条另以180秒上限重放，2/2正常播完；播放期条先暂停旧答，再按续说启动新答并完成。长回答是重新生成，**不是对原次被截去尾部的直接观测**。两条与31条重叠，不增大独立样本分母，也不把不同实现阶段拼成一轮正式准确率。

自造中文加粗输入经规范化后，与本轮纯文本对照同为3.644375秒音频；另覆盖英文列表、中文列表与emoji尾段。旧诊断加粗/纯文本对照约29.80/3.88秒含前导空格，不能与本轮宣称逐样本PCM一致。没有人工听感评分。

最终9个上下文模型探针通过约定动作门：其中一个“先不要回答”的含糊探针允许stop/wait两种不回答动作，另外8个要求精确标签。早期单块拼接候选把附和误判ready，保留失败记录后改成两个音频块；没有把它写成成功上线。明确停止、等待、播放期附和、新问题和碎片续说分别覆盖。

5种真实浏览器/Actor/Omni场景均通过：正常、出音前单次502恢复、持续502、校验放行PCM后断流、不公开候选阶段持续失败。真正的断流失败调用已交付7125个PCM样本，未重试/重播；早期在校验前切流的探针不计入此证明。私有阶段夹仅为覆盖状态把hold设为3秒，其他浏览器夹及生产保持640ms。各夹trace正常闭合、无丢记录。401项定向pytest与Node播放/输入/遥测契约通过。

正常启动器已重启正式服务：只监听回环10003/18000，10004无监听，文本/路由/TTS端点一致。启动预热通过；正式页面的长回答→明确停止→新问题→播完通过，最终播放underrun为0，trace无丢记录、引擎/语音错误为0。该会话19条流式模型调用均记录传输状态，完成请求的客户端/HTTP request ID关联成立，SSE内部ID另存；包括真实播放期双音频上下文请求。仍未使用物理麦克风或扬声器。

必要回退：分别关闭 `spoken_text_normalization` / `input_continuation_context`，或设 `tts_transport_retries: 0`；代理兼容用 `--inference-mode proxy`。重启backend才加载行为配置，切换完整拓扑使用正常启动器；不要杀不明端口所有者。基础/冻结配置无须回退。
