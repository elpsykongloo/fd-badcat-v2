# AGENTS.md — fd-badcat 代理工作约束

> 本文件是仓库级代理的单一真相源；`CLAUDE.md` 指向这里。
> **只保存当前有效事实、长期约束和不可逆研究裁决。不要继续追加流水账。** 单次实验的 PID、逐调用数字、收据、失败过程、旧状态与详细判读放到 `docs/` / `exp/`；本文件只保留结论和入口。
> 若本文与旧文档/旧收据冲突，以本文“当前状态/永久规则”及更晚的明确裁决为准；需要审计历史时再读对应归档。

## 1. 使命与权威文档

- 目标：终极目标：把 fd-badcat（HumDial Challenge 比赛第二名，半级联的全双工语音系统）改造成 **TACT：事务性工具调用的全双工语音 agent**，面向 **ICLR 2027**（摘要 9/19、全文 9/24）；阶段性目标：把不接入工具调用、但是优化了真实对话体验demo版fd-badcat（暂定为fd-badcat v1.5），作为公司的展示产品进行实际部署，并且投稿ICASSP 2027（全文 9/23）。
- 总蓝图：`手工文档/神谕/00_系统蓝图.md`。它是论文命题、形式化、评测和时间线的最高层设计记录；允许基于实证偏离，但必须在正式文档中说明理由。
- W1 计划：`手工文档/神谕/01_W1 完整计划.md`；部分内容已过时，不能覆盖后续实证裁决。
- 人类评估/伦理流程当前不在工程主线；优先代码、模型能力和自动评测。
- 研究细节优先读对应专题文档，不要从本文件还原实验：`docs/w1_report.md`、`docs/w2_rerun_report.md`、`docs/w3_*`、`docs/w4_ladder_design.md`、`docs/w4v3_design.md`、`docs/w5_specgate_design.md`、`docs/rb_design.md`、`docs/rb_test_protocol.md`、`docs/web_demo.md`。

## 2. 永久工程纪律

1. **行为保持优先**：已有 judge/interrupt/shift 语义、0.64s hold、2.5s continue timeout、1.5s 长打断等基线行为，除非任务明确要求改变，否则不得偷偷改；行为实验必须放在显式 flag/独立配置后。
2. **单写者原则**：引擎状态只由引擎协程写；其他 task 通过 `asyncio.Queue`/事件传递结果，不直接改状态。
3. **音频钟是区间时钟**：VAD/hold/window 等语义时间使用音频钟；墙钟只用于推理耗时、网络耗时和 trace。
4. **决策保持 audio-grounded**：Qwen3-Omni 直接听音频；ASR 默认只用于 history/审计，不应悄悄成为关键决策授权信号。
5. **禁止完整性哈希检测**：日常测试、启动、实验、归档、提交、推送不得新计算/比对 SHA/MD5 等文件或产物完整性哈希，也不要把它们写进新收据。历史 freeze/文档已有哈希只作归档，不重算。允许功能性哈希：缓存键、确定性 RNG/数据切分、稳定 op/sample ID、Python 对象哈希。
6. 不把真实用户会话、录音、用户蓝图、私有诊断原始输出提交到仓库；只提交明确允许的自造夹、聚合统计和脱敏收据。
7. 不把一次烟测、单个真实案例、模拟链路或模型边界试验写成“根因已证实 / 零退化 / 普遍修复 / 正式准确率”。结论必须和证据口径匹配。
8. 多代理适合并行写代码/检查；**正式实验结论必须回到受控运行和原始产物逐项核验**。延迟、确定性、single-fire test 窗口按协议串行。

## 3. 当前 Demo / Actor 状态（2026-10-08）

运行时细节会过期：不要相信历史 PID；操作前先检查 `tmux humdial-demo`、监听端口和当前配置。demo 默认本机回环 `:10003` Omni ← `:18000` backend/demo；文本/路由/TTS共用10003，不再依赖10004。显式 `--inference-mode proxy` 保留10004兼容路径，页面仍供18000 SSH转发。

### 3.1 已部署的当前链路

- demo-only 输入准入：`guarded-turns-v1`，四态 `keep / stop_only / yield_wait / yield_ready`；旧 HumDial `interrupt` 基线仍保留用于关闭新路径时的对照。
- 音频路由：`transcript-first-v1`。一次 Omni 请求严格输出有序 `transcript,label` JSON；格式异常最多一次修复，失败保守回 `keep`。转写不替代聊天 ASR/history。
- 音频路由携带 playing 状态、用户音频及有界续说上下文标记，**不发送助手参考原文**。`played-reply-v1`：仅闭合播放期有效非空 keep，在同一2秒总预算内用同一模型纯文本复核，固定新增语音转写+输入起点已ACK播完的最多3句，仅可 keep→ready；空快照不后补、失败保持keep。配置入口 `prompts.input_reply`，机制与验收见 `docs/web_demo.md` §9、`exp/web_demo/played_reply_validation.json`。
- 续说上下文：`pending-user-audio-v1`。等待/未开播候选的已接纳前文进入路由；播放期近期用户音频在输入起点冻结，以同一次请求的独立前文/新增两个音频块区分，转写仅含新增块，不重复进入本轮ASR/回答。共用20秒音频预算/音频钟间隔界限；不复活已被keep拒绝的语音。机制、验收和模型边界见 `docs/demo_reliability.md`。
- 长回答参考：`speech-reference-v1`。只维护已公开且 ID 匹配的生成尾部/句子 PCM 起点；输入起点冻结当前和前两句已播放参考，避免用未来未播文本污染决策。取消回答的模型历史只保留 ACK 覆盖的完整句前缀；打断状态只存内部记录/trace，不注入 assistant 文本。
- 统一诊断线：`demo-diagnostics-v1` + `demo-trace-v2` + `demo-case-v2`，按 session/turn/call/parent/utterance 关联真实调用、请求 span、自动 invariant、逐轮修订式 review 和无模型 Actor replay；原始 mic/reference/clean 三轨仅逐会话明确 opt-in。私有目录、操作与边界见 `docs/demo_diagnostics.md`，观测不自动当 gold。
- demo 回答完成性：`response-completion-v2`。仅 demo 普通回答严格区分 stop/length，单次512 tokens、长度截断最多原生前缀续写3次，短承诺最多另修复一次；失败/预算耗尽明确标为未完成。原音频保留，续写可淘汰最旧完整历史轮对，调用与结束原因进入统一诊断；机制、回退和验收见 `docs/demo_response_completion.md`。
- 流式输出：PCM16 协议 + SSE 文本→`spoken-text-v1`口语格式规范化→分句→逐句TTS；页面/参考/已播历史共享规范文字，原模型delta/续写前缀保留。Markdown/正文emoji在分句前清理，不盲剪PCM；异常长低能量段仅诊断。`demo-continuity-v1`仍为350ms起播、2000ms有界预取、最多下一句、客户端信用600ms、codec `4:12`；基础/冻结配置不变。见 `docs/demo_speech_continuity.md`、`docs/demo_reliability.md`。
- 传输：`model-transport-v1`保存端点、上下游ID、HTTP/SSE阶段、异常类型、输出/播放状态及恢复轨迹；demo每句仅在零PCM且暂态故障时最多重试一次，已出PCM不重播，持续失败明确提示未完成。兼容代理不跨请求复用连接；历史502的具体异常未保存，不能倒推连接竞争已证实。操作与脱敏收据见 `docs/demo_reliability.md`。
- TTS：`verbatim-grammar-v2`。原文用 JSON 安全转义的单个 EBNF 字面量约束；LF/CR/TAB 保留，其他 C0/DEL 在 HTTP 前拒绝；生产 stage0 固定 xgrammar，不允许失败后退回自由聊天生成。
- demo 声音控制：`demo-voice-rng-v1`，`engine.demo_voice_control` 固定 chelsie/seed42，残差声码按请求维护 RNG；小样本串行重复可复现，并发波形仍有变化，不宣称跨文本音色稳定。机制与收据入口：`docs/demo_voice_rng.md`。
- demo 离线音色对比：`demo-speaker-embedding-v1`，作者 w2v-BERT2.0 + LoRA/Layer Adapter/MFA 最终 LMFT 权重，自动提取256维嵌入并计算 cosine；无人工评分依赖，不接线上决策，分数未校准为身份阈值。入口：`docs/demo_voice_rng.md`；跨文本评测按句长分组并补充固定短句聚合，保留逐句指标，见 `docs/demo_voice_cross_text.md`。
- demo 启动默认 `FDBC_DEMO_TALKER_NUMERICS=native`：仅 Talker 采用 cuBLASLt 并关闭 BF16 低精度归约/split-K，保留动态批处理；通用启动默认 off，完整 invariant 模式仅作较慢对照。受控样本并发声码一致，仍不宣称跨文本音色稳定或全链路 PCM 一致。机制与开销：`docs/demo_voice_concurrency.md`。
- 当前输入调参：`engine.input_preroll_ms=320`；VAD threshold `.5`、结束静音 `100ms`、`640ms` 确认及既有超时不变；`input_route` 和条件复核 `input_reply` 的 presence/frequency penalty 为0，聊天/shift/binary/TACT 的采样保持原值。取真实 deque 前缀，不补零伪造实录。

### 3.2 当前不能宣称的事

- 很短的“停/停下”等仍有模型转写/前截断/上下文交互型漏接证据；320ms pre-roll 是受控最小改动，**不能宣称短停根因已证实、召回普遍提高或零退化**。
- 不要加“停止词白名单”、第二 ASR、全局放松 VAD、ASR 非空即强停等未经新证据批准的机制。
- 问号门候选E未采用；短答使用上述语义复核。用户抢在问句播完前的含糊回答仍可能keep，这是接受的保守边界，不加词级对齐；复核不是通用声源识别或所有反问/引用均无误的保证。

## 4. 环境与服务

- 仓库：`/root/autodl-tmp/fd-badcat`。
- FDBench v3：`/root/autodl-tmp/FDBench_v3`（只使用 v3；旧代留作历史）。
- TACT 包已归仓到 `tact/`；运行环境的 `tact-fdb` editable 安装必须指向 `/root/autodl-tmp/fd-badcat/tact`。
- 常用环境：
  - backend：`/root/miniconda3/envs/fd-sds`
  - FDB：`/root/autodl-tmp/conda-envs/fdb_v3`
  - Omni/vLLM：`/root/autodl-tmp/conda-envs/fdbc-qwen3o-vllm`
  - Index/Qwen TTS 相关环境以当前脚本/部署为准。
- 无 GPU 小容器可能只有 1 核/2GB，直接起 torch 可能 OOM；优先轻量脚本/预抽取路径。GPU 日常见 RTX PRO 6000 Blackwell 96GB，但**启动前必须实查资源**。
- 通用/冻结生产 Omni 配置仍为 `configs/qwen3_omni_audio_single_gpu.yaml`，三阶段 `max_num_seqs=4`、stage0 `max_model_len=4096`、FCFS。Casecade-Demo 的 PRO 6000 使用 `setup/start_demo_pro6000.sh`：有界 KV/声码预热与独立 BF16 MoE 参数，四槽/上下文/引擎时序保持；声码图候选未采用。验收、测量边界与回退见 `docs/demo_pro6000_p0.md`。历史延迟可比测量使用 `qwen3_omni_audio_serial_eval.yaml`，不要拿 seq4 烟测直接减历史 seq1 延迟。
- backend 请求容量默认 total=4、normal=3，为 judge/interrupt 留槽；普通 response/shift/TTS 走 normal。
- 本地服务调用必须规避系统代理：使用 `trust_env=False` / `NO_PROXY`。shell 不要导出空值或 0 的 `OMP_NUM_THREADS`；必要时 `env -u OMP_NUM_THREADS`。
- DeepSeek key 在 `configs/eval.env`（gitignored，600 权限）。当前 judge 模型：`deepseek-v4-flash`。正式 FDB semantic judge 使用 `scripts/fdb_pass_judge_strict.py`；运行前清理 `all_proxy/ALL_PROXY/http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY`，不要回退到官方 200-token 的静默失败路径。

## 5. 仓库、Git 与发布

- 当前工作分支：`tact`；私仓发布目标为 `origin/main`。基线 tag：`golden-base`。
- `src/backend_legacy.py` 是旧引擎冻结对照件，除非任务明确要求，不要改。
- 本地代码是原上游的严格超集；不要从 `upstream/main` 反拷旧文件覆盖本地修复。
- 动 git 历史或准备推送前先 `git log --oneline -3`，确认没有踩到其他代理/用户的新提交。
- 实验运行不另建临时代理分支。完成真实运行和必要核验后，代码、允许提交的产物/报告以及必要的 AGENTS 事实更新一起提交并推到私人 `origin/main`；不需要 PR。
- 推送前做密钥/敏感文件扫描和语义/结构一致性检查；**不要做完整性哈希检测**。
- 原上游默认只读/NO_PUSH。已存在的唯一特殊授权是更新公开分支 `Rice`；不得写 `main`、其他分支、PR 或 issue。除非当前任务确实需要，不要主动触碰上游。
- 历史脱敏已经做过；不要恢复删除态明文凭据或历史 `.pyc`。

## 6. 架构事实与兼容性

- legacy `run_realtime` 的核心问题是 receive→VAD→决策串行 await 导致感知冻结；新架构用单写者+事件队列解决。
- VAD 事件时间按样本天然正确；会漂的是 hold/continue/长打断等墙钟区间，因此业务区间统一迁到音频钟。
- 旧代码的 `async_llm` / `async_tts` 曾直接改全局状态；新路径禁止恢复这种跨 task 写状态方式。
- 当前决策中枢是 Qwen3-Omni 音频判定；没有通用说话人验证模块。旧 shift 的话题连贯性只是代理，不等于 SV。
- HumDial 的 response/shift_s 含比赛型 prompt 约束（如短答/附和）；HumDial 模式保留，agent 模式不要把这些 hack 偷带进去。
- 音频消息格式已有三种开关：`audio_url`（本地 vLLM 默认）/ `input_audio`（OpenAI 裸 b64）/ `input_audio_datauri`（DashScope 方言），实现位于 `src/messages.py`。
- 流式 demo、legacy、RB/TACT 评测是不同契约；修改某一路时要证明未无意改变其它冻结路径。

## 7. 评测与实验口径

### 7.1 并发与延迟

- 准确性/决策质量回归可走 injected/并发吞吐轨；每会话独立 engine/VAD/output，worker 使用独立 session/模型实例，避免共享 `requests.Session` / sherpa 线程安全问题。
- 论文延迟、确定性和 single-fire 结果，必须走串行专机/串行配置；不要把并发 GPU 争用下的 infer/wall 数作为延迟主表。
- 决策缓存可用于 prompt/输入完全相同的 T=0 重放和 δ 网格省卡；缓存命中不是新的模型证据。

### 7.2 FDB 判分

- 始终区分 **exact** 与 **DeepSeek semantic-argument judge**；历史旧 judge（200 token、旧模型或 gpt-5.5）和当前 strict judge 不可直接相减。
- `evaluate_pass_rate.py` 与 `evaluate_tool_calls.py` 不是同一口径：前者是二元 pass，工具选择 precision=1；后者是 metrics 轨并带 `turn_take_success` 等门。
- 同名工具多次调用的官方历史对齐曾按位置 `pop(0)`；不要擅自把“更合理的语义对齐”混进冻结结果。
- semantic judge 存在约 ±2pt 的供应方噪声；≤2pt 差异单独不能当结论。

### 7.3 实验纪律

- W2 第一轮多代理并发实验结论已判无效并删除；**W2 有效结论只看 `docs/w2_rerun_report.md` 与对应神谕汇报**。
- test 单发/冻结协议一旦消耗，不得为了更好结果修改 generator/scorer/runner 后重跑；新机制必须升版本并开新预注册。
- “oracle”只表示替代相应决策内容，仍可能受窗口、屏障、执行期和可逆性天花板限制；不要把 oracle<1 自动判成 harness bug。
- 任何报告都要同时记录有效分母、排除项、是否独立样本、是否真实人类/真实设备、是否串行、是否可与历史口径比较。

## 8. 当前研究裁决（只保留会影响后续工作的部分）

### W1 / W2 / W3

- W1 已收口：感知冻结消灭、音频钟迁移、行为保持与 FDB 基础链路完成。
- W2 首轮无效；重跑版有效。核心经验：异议窗 δ≈1.5s 能救修订，但 eager 窗会留下脏轨迹；首响和完成时延必须分开报告。
- W3 已收口；最终 prompt 口径为 **v3.1**。事务机制主线成立：commit barrier、窗口、投机派发、单写者/音频钟是方法主体。投机可把首响压到 0.64s 地板，但会产生大量作废调用，收益必须和成本同报。

### W4 / 学习线

- W4 rung 4 v0/v1/v2 与 W4-V3 已按预注册收口；**G2“学习停时策略进入目标区”的核心证据没有建立**，不得用 FDB 结果继续反调或偷偷开新 rung。
- 当前论文叙事：**事务性执行模型是方法核心；学习组件转为“可学性/不可达性”的分析与负结果**。意图稳定性、部署时点 finality 和跨域风险标度均出现实证限制。
- SG v0/v1 两次探针均未过冻结门，**Speculative Gate 线永久收口**；没有 v2、没有 stophead 训练、没有 FDB 单发。

### RB

- RB 是独立事务/生命周期压力基准；详细版本、配额、冻结和 test 判读只看 `docs/rb_design.md` / `docs/rb_test_protocol.md` 及对应 `exp/rb/` 收据。
- v2.2.1 的 test 窗口已经全部消耗，不得改后重跑。
- v2.3 test-911 已收口：屏障与 L11 反应式窗口是主要正结果；补偿 reverse 的模型调用能力不足；旧 admission v1 因在解析前错误处理 local id 已退役。
- admission v1.1 把 schema 门移动到解析后并通过零损失门，但低基率下没有 headline 增益；不要恢复 v1。
- 最近归档状态：RB 主线已推进到 **v2.4 / freeze v5**；旧 freeze 中的哈希是历史记录，**不要因它们恢复哈希检测习惯**。如需继续 RB，先读最新协议确认当前合法窗口，不要仅凭 AGENTS 里的旧数字启动 test。

## 9. 操作清单

开始任务前：

- 读本文件相关章节 + 目标模块附近代码；涉及实验再读对应专题设计/协议。
- `git log --oneline -3`，检查工作树、当前 branch、运行中的 tmux/端口/GPU。
- 明确本次是在 legacy/HumDial、demo Actor、TACT/RB 还是评测工具链，避免跨路径误改。

修改后：

- 跑最小相关测试，再跑必要的集成/真实服务验证；不要为了“看起来完整”无条件跑全仓或昂贵 GPU。
- 若行为、公开接口、冻结配置或研究裁决改变，更新专题文档和本文件的**现状句**；不要追加新的长时间线。
- 新实验详细数字写收据/报告；AGENTS 最多保留一句“结论 + 指针”。
- 提交前检查敏感信息、真实用户数据和非预期大文件；不做完整性哈希。

## 10. AGENTS 维护规则（重要）

以后更新本文件必须遵守：

- **替换旧状态，不追加“覆盖下方”的新段落。**
- 单次 PID、session id、单条 trace、HTTP 调用数、pytest 项数、耗时、逐夹列表、cache hit/miss、实验矩阵、p 值等默认不进 AGENTS；放收据。
- 只有以下信息值得常驻：当前版本/开关、长期工程规则、不可逆裁决、关键路径、数据/发布红线、仍然开放的问题。
- 一项已完成工作若已有稳定 `docs/` 或 `exp/` 入口，AGENTS 只保留 1–3 行结论和路径。
- 若某段超过约 10 行，先问：代理在未来多数任务里是否必须读它？若否，迁出本文件。
