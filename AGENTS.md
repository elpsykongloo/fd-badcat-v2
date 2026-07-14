# AGENTS.md — fd-badcat 持久记忆（所有代理必读）

> 单一真相源。CLAUDE.md 指向本文件。有重大事实变更时**更新本文件**，不要另开新文档。
> 最后更新：2026-07-13 (v3.1 DeepSeek 三判闭合)

## 使命

把 fd-badcat（HumDial Challenge 半级联全双工语音系统）改造为 **TACT**（事务性工具调用的全双工语音 agent），投稿 **ICLR 2027**（摘要 9/19、全文 9/24）。

- **总蓝图（宪法）**：`手工文档/神谕/00_系统蓝图.md` — 论文命题、形式化、评测、时间线。尽量不违背；违背需有充分理由并记录。
- **W1 执行计划**：`手工文档/神谕/01_W1 完整计划.md` — 可微调（它基于云端旧代码写成，部分已过时）。
- 人类评估（伦理审批等）**暂时全部跳过**，只关心代码与模型能力。

## 环境事实

- 工作目录 `/root/autodl-tmp/fd-badcat`；FDBench 在 `/root/autodl-tmp/FDBench_v3`（v1_v1.5/v2/v3 三代，用 v3）；**TACT 原型包在 `/root/autodl-tmp/tact/`**（独立 git 仓库，6/23 构建：事务代数/工具注册/decider/offline_runner，decider 走文本路=现成消融基线；INTEGRATION.md 是集成指南）。
- **conda 环境**：`fd-sds`（`/root/miniconda3/envs/fd-sds`，backend 运行环境）；`/root/autodl-tmp/conda-envs/` 下有 `fdb_v3`、`fdbc-qwen3o-vllm`（vLLM 服务）、`index-tts-vllm`。直接用绝对路径 `/root/miniconda3/envs/fd-sds/bin/python` 最稳。
- **容器规格随租卡变化**：无 GPU 时 1 核/2GB（torch 进程会 OOM——用 `scripts/extract_vad_events.py` 预抽 VAD + `install_light_stubs()` 轻进程路；预压分配 trick 见该脚本）；GPU 日为 RTX PRO 6000 Blackwell 96GB + 208 核/118GB。
- 服务拓扑（GPU 在位时）：vLLM Qwen3-Omni-30B-A3B :10003（`setup/start_qwen3omni_audio.sh`，音频管线 max_num_seqs=1 确定性优先；文本配置 `qwen3_omni_text_only.yaml` 可高并发）→ 代理 `src/qwen3_api.py` :10004（`setup/start_qwen3_proxy.sh`）→ backend :18000。TTS 默认 **Omni 原生**。实测 Blackwell 上音频判定单次 ~0.26s。
- 网络：本机代理环境变量存在，本地服务必须 `trust_env=False`/`NO_PROXY`。**shell 里不要 export OMP_NUM_THREADS=空值**（libgomp 报 Invalid value）。
- DeepSeek judge key 已持久化：`configs/eval.env`（600 权限，gitignored，bashrc 自动 source）。judge 模型用 `deepseek-v4-flash`（`deepseek-chat` 已从 API 下线）。**延迟指标纪律：正式延迟数字必须串行专机跑（勿与 vLLM 争 max_num_seqs:1，勿并发 funasr GPU 加载）——W1 回归的 FRD 曾被此污染，勘误见 w1_report.md。**
- **judge 静默回退坑**：① 环境有 `all_proxy=socks5://…`，openai SDK(httpx) 缺 socksio 会静默初始化失败；② 官方 `evaluate_pass_rate.py` 给 argument judge 仅 `max_tokens=200`，`deepseek-v4-flash` 会把预算耗在 reasoning、`content` 为空后再次静默退化 exact。正式重判必须用 `scripts/fdb_pass_judge_strict.py`（当前冻结：1024 tokens、最多 5 retry、fallback 硬失败）并先 `unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`。
- **FDB 分数口径**：永远双轨报告 exact + deepseek-v4-flash judge。历史 blocking exact 0.570 / judge 0.700；但 7/13 审计发现 W1/W2 的 200-token judge 报告每份含 10–14 个明确 exact-fallback mismatch 签名，**历史 judge 绝对分在 strict 重判前只作旧口径记录，不得与 v3.1 strict judge 直接相减**。6/23 的 0.73 还是 gpt-5.5 口径，更不可比。**v3 有两条判分线**：`evaluate_pass_rate.py`（二元 pass，无 turn-take 门，工具选择 precision=1 一票否决——judge 模式也只接管参数比较）与 `evaluate_tool_calls.py`（metrics 轨，有 `turn_take_success` 门=转写非空，默认 metrics 只算 turn-taken 子集，tool F1 渐变）。同名工具多次调用按**位置 pop(0) 对齐**——顺序互换会双杀。
- **W1 汇报文件**：`手工文档/神谕/02_W1 执行汇报.md`（发给神谕求 W2 计划的汇总，含全部实验数字与开放问题）。

## 仓库拓扑与分叉事实（重要）

- **本地是云端的严格超集**：`origin/main` 顶端 `f4ff2b1` 即本地历史的 merge-base；本地在其上有 10+ 提交（vLLM 本地化、HumDial 批量评测、Omni TTS 端到端）。**不要从云端拷文件回来**（云端 `qwen3_api.py` 有 setdefault-after-post 的无效代码 + `/data/ptmodels` 路径；本地早已修好）。
- 音频块格式分叉已做成开关：`llm.audio_block: audio_url`（本地 vLLM 默认）| `input_audio`（OpenAI 严格裸 b64）| `input_audio_datauri`（云端 DashScope 方言）。实现在 `src/messages.py`。
- **工作分支 `tact`**，基线 tag `golden-base`（=420b539）。`main` 不动。`src/backend_legacy.py` 是旧引擎逐字节冻结件，W1 全程保持可运行（A/B 对照），别改它。
- 本仓库有多个代理活动过（git author 见 Codex 提交）。动 git 历史前先 `git log --oneline -3` 确认没踩到别人的新提交。
- **本仓库只在本地，永不在线提交**（用户 7/07 明示）：不做任何上游 PR / issue / push；06 计划里的"上游 PR ×2 文书"（latency_injector per-instance RNG、finance_14 pop(0) 对齐）**取消**，相关生态问题只在本地文档记录，不对神谕另行澄清。

## 架构事实（读码验证过，可直接引用）

1. **感知冻结**：旧引擎 `run_realtime` 单协程串行 receive→VAD→决策，决策 await 期间无法 receive，最坏 judge→shift→response 三连 LLM 冻结；SPEAK 态 interrupt 判定同病。这是论文 motivation 的第一实证（W1 修复 + before/after 测量）。
2. **漂移主体是墙钟**：VADIterator 事件时间按样本计天然正确；漂的是 END_HOLD/continue 超时/1.5s 长打断等墙钟区间量。修法 = 全部迁到音频钟（`t_audio = seq*256/16000`）。副产品：快于实时的确定性回放（模拟器地基）。
3. 旧代码竞态：`async_llm` 副作用写 `IN_SPEECH=False`（legacy:178）；`async_tts` 异步置 `STATE=SPEAK`（legacy:198）。新引擎单写者原则消灭之。
4. 决策中枢 audio-grounded（Qwen3-Omni 直接听音频，T=0/seed=42 确定性）；ASR 只进 history/日志，**不在决策关键路径**。
5. 无说话人验证模块；拒识靠 shift prompt 的话题连贯性代理（57.8 分根因）。W3 计划重新引入 SV 门控。
6. `response`/`shift_s` prompt 里有两条比赛 hack（只回 15 字；用户否定必须附和）——**humdial 模式保留，agent 模式的 prompt 集里必须去掉**（否定=patch 触发器）。
7. TTS 整句合成后一次性 send_bytes；无 playback 结束→LISTEN 的转换（HumDial 既定行为，`engine.playback_autoend` flag 默认 false 保持）。

## 三条铁律（W1 全程有效）

1. **行为保持优先于一切优化**：judge/interrupt/shift prompt、0.64s hold、2.5s continue 超时、1.5s 长打断——一个字符不动。行为变更藏在默认关闭的 flag 后。
2. **单写者原则**：引擎状态只由引擎协程写，其他 task 经 `asyncio.Queue` 投事件。
3. **音频钟为唯一区间时钟**；墙钟只测推理耗时与 trace 第二时间戳。

## 评测并发策略（算力昂贵，用户明示）

原始代码为延迟分数纯串行+实时流式，9k 样本要跑几天——**实验阶段允许完美隔离下的高并发**：

- **吞吐轨**（测准确性/决策质量）：`injected` 回放模式（音频钟推进=记录的 infer_time，不真等），每会话独立 engine/VAD/输出目录，vLLM 天然支持并发 batch，`--concurrency N` 压满 GPU。
- **实时轨**（测延迟指标）：串行、`realtime` 模式，只在最终出数时跑。
- 隔离要点：module.py 的共享 `requests.Session` 与 sherpa 模型是并发下的坑（thread-safety），并发 runner 需 per-worker 实例化。
- **W2 重跑追加的效率工具**（准确度回归用，出论文数字仍守 A 档串行）：
  - `scripts/w2r_stream_replay.py --workers N`：线程池并发（VAD/HTTP 均 thread-local，决策缓存加锁）。注意沙箱延迟 `random.seed` 是全局的——并发下 per-example 抖动序列不保逐位复现（吞吐轨可接受，P5 类确定性验证必须 --workers 1）。
  - **决策缓存**（sha256(messages)，T=0 合法）是最大的省卡时杠杆：δ 网格 6 个点里 4 个点几乎全缓存命中；改 harness 不改 prompt 时务必复用 `exp/w2_rerun/decision_cache.json`。
  - **DeepSeek judge 并发**：官方 `llm_judge.py` 原生支持 `FDB_LLM_WORKERS`；judge 是纯 API 调用。v3.1 正式三判采用 strict runner、workers=32，避免把并发/解析失败与 judge 判决混合。`scripts/run_fdb_with_deepseek.sh` 仍指向连续 metrics 轨，不可用于论文 binary judge-pass 主轨。
  - vLLM 侧：准确度回归可临时把 audio 配置 `max_num_seqs` 调高（重启分钟级）配合 --workers 压满；HumDial 管线自带 `--gen-workers/--clean-workers/--asr-workers/--judge-workers` 旋钮。

## 当前状态（W3 —— 全部收口，2026-07-09）

- **D6 标准卡串行复测已清（7/09，W3 最后一个 GPU 项）**——三臂全量 100、音频标准栈、`--workers 1`、每臂全新缓存（0 hit，锚干净）、A 档双跑决策逐位（pass/tool calls/signature/transcript diff 全 0）。**主表最终数字（v3.1 口径，live 串行档）**：
  - **准确性**：TACT d150 exact **0.640** / sblock **0.660**（差 −2pt 不变）。**7/13 DeepSeek semantic-argument 三判已闭合**：TACT **0.710±0.010**、spec **0.723±0.005**、sblock **0.730±0.010**（±=三次半极差；逐轮 TACT−sblock = −3/0/−3pt，均值仍 −2pt；spec−sblock = −2/0/0pt）。judge 语义宽恕把三臂各抬约 7–8pt，但不反转主结论；报告 `exp/w3/fdb_v31_deepseek_judge3_summary.json`。live state 同栈复算为 verbatim TACT/sblock **0.66/0.66**、normalized **0.77/0.79**；此前 0.79/0.82 是 text 栈，不得混入 live 主表。⚠️ live 音频栈 vs 前日 text 栈 exact 存在 **−1pt/臂 电平差**；主表引 live 档，δ 曲线保留 text 栈产物作形状主张。
  - **延迟**：TACT first 1.218/1.579、done 3.012/3.411 vs sblock 1.529/2.134（done 同）；infer_final p50/p90 = 0.561/0.767（v3.1 prompt 变长 vs W2 v2 0.444）；恒等式 first = 0.64+infer 残差 max 0.0（ack 路 83/100，fallback 17）。**live 完成保费 = 1.483 ≈ δ**。
  - **投机派发实测**：first p50 **1.218 → 0.640 = 地板精确命中**（Δ−0.578；P3 比 0.640/1.529 = **0.419 ≤ 0.50 过**）；exact 对基线 **0 翻转**；done 2.455（保费 0.926 = δ − infer 重叠 0.557 ≈ infer p50，算术自洽）；spec 恒等式 first = max(0.64, infer) 残差 0。**W2 的"P3 须 4B 头"结论正式作废——投机派发免训练拿下 P3**。
  - **投机真实成本（full engine live，informational）**：604 派发 / 197 确认 / 407 作废 = **浪费率 67.4%（≈2.8× LLM 调用）**；exact 0.630（ε 带 −1，D2 已知感知敏感）；first p50 0.786。论文写实测收益必须同页写此成本。
  - **权威现实档（干净锚重打分）**：P-1 ALL **0.62 ✗** / 写类 **0.249 ✓**；P-2 **1.494 ✓**（第 6 次落 δ±0.01）；P-3 27/34 ✗（7 miss 中 6 个方向利 TACT，前提失效非机制失效）；P-4 86/100（非同一 = fallback 夹）。产物 `exp/w3/realistic_compare_w3p31L_sblock__w3p31L_tact_d150.json`、`exp/w3/first_decomp_v31_{live,spec}.json`、`exp/w2_rerun/grid_full_v31_live*.json`。
- **全部 GPU 项已跑完并逐文件核验**（用户跑数 7/07 晚，单线程复核 7/08）。汇报：**`手工文档/神谕/08_W3 执行汇报.md`**（07 缓发内容已并入）。要点：
  - **现实档**：30 冒烟播种不变量 30/30 逐位；对表（冻结档 rescore 为权威）P-1 全量 0.592✗/写类 **0.237✓**、P-2 保费 **1.500 精确=δ\***、P-3 31/33✓、P-4 93✓；**全量 δ 曲线首次出**（exact 0.470/0.470/0.550/0.560/0.570 @ δ0–2.0，eager 惩罚 −10pt 全量复现）；d150 A 档双跑 100/100 逐位、与 W2 冻结档**决策内容位齐**（0 翻转/0 调用集差）。
  - **⚠️ 7/07 新跑臂（w3r_*/w3spec_*/w3p3_*）的 infer/墙钟性锚全部污染**（与并发 GPU 争抢）：决策内容有效，延迟锚不可用——现实档延迟一律引冻结档案 rescore（`exp/w3/realistic_*.json`）；网格新点 d000–d100/d200 锚干净。
  - **投机派发**：exact 56→57（+eco15 良性）；HumDial 门单判 off **63.38** / on **63.00** Δ−0.38 过；**三判加固已补**（7/08，复用同一生成输出重判）：off **63.54±2.07** / on **64.50±0.63**，均值 Δ=**+0.96**，配对 Δ=+1.00/−1.00/+2.88，gate Δ≥−1 维持通过且无回归证据；首响投影 1.141→**0.64 地板**（冻结档分解 recheck）。live 实测收益待 D6 标准卡串行复测。
  - **DAG**：0 翻转；11 夹 dag.events 非空；comp_plans=0（官方轨零链式修订与 D2 普查互证 → 补偿评测只能靠 RB L7）。
  - **prompt v3**：30 子集 12→18、rollback 12→14（+fin23/hou17/tr19 = 靶#4/#3/#5；−fin19）一轮过冻结门；全量 TACT 0.560→**0.620** / sblock 0.570→**0.650**——P1 差分门（vs 0.570）+5pt 过，**但 blocking 受益更大（+8 vs +6），同 prompt 差 −3pt**；verbatim 状态 57→64/65（规则 14 源头规范化实证）、normalized **反转** TACT 75 < sblock 79；v3 现实档 P-2 **再次精确 1.500**（定律 prompt 不变）、P-3 28/34 FAIL（前提失效非机制失效，5/6 miss 方向利 TACT）。TACT 三回归归因：eco23=规则10 过度 re-launch、eco25=动态引用退化、fin19=规则11 误伤。**v3.1 已落地（用户 7/08 裁定免神谕轮）**：规则 10/11 词条收紧 + 新规则 15（$RESULT 链完整性），`PROMPT_V31_ADDENDUM` 独立常量（v3 冻结为审计产物，进程内互斥 guard）；`--prompt v3.1`，provider 约定 `w3p31_*`；协议+预注册预期在 `docs/prompt_v3_five_targets.md` §3。**Round-1 已跑（7/09）：门全未中但全量净升——30 子集 17/30✗、rollback 13/17✗；全量 TACT 0.620→0.650（gain eco23/eco25/fin04/tr18/fin19 / loss fin23/tr19）、sblock 0.650→0.670（loss 仅 fin23）；δ 曲线 0.530/0.530/0.620/0.650/0.660 单调升→sblock 0.670；state normalized TACT 79 < sblock 82（verbatim 67 平）。误伤根因已定：fin23=规则11 例句吞掉规则13 替换构式（臂中立双臂同死）、tr19=规则10 诱发 relaunch 而非 patch pending（TACT 特异）；eco13 哨兵无害。栈一致性：30 夹跨栈（音频 vs text-only 8192）判决逐位一致；v3.1 全部延迟/infer 锚系 --workers 12 并发产物只作信息位（P-2=1.500 恒等式第 5 次精确命中，锚免疫可引用）。按预注册"不中不得再迭代"⇒ v3.2 封盘不做；主表口径建议 = v3.1 双臂同 prompt（−2pt 差距逐夹全归因），修法转 W4 训练素材（词条工程刀口零和 = 学习组件立项动机）。判决全文 §3.3。**
  - **E5 分句 TTS live**：机制链全验证（TtsSentDone/first_sentence/floor_decision yield/tts_sent_dropped，`exp/w3/e5_traces/`）；**勘误：同句首音频提前 p50 = 0.545s**（先前 10.324s 系跨句配对错误，作废）；只有长叙述句触发分句（3/10 runs）——分句收益域=长句场景，首响主战场在 ack/投机。
  - HumDial 门产物已入库 `exp/w3/humdial_gate_spec{off,on}_summary.json` 与三判聚合 `exp/w3/humdial_gate_spec_judge3_summary.json`；`exp/w2_rerun/grid_full.json` 已被 v3 网格覆写（v2 数字重算核验一致；scorer 后续加组名防覆写）。
- **待裁断**（08 §六）：① prompt 口径——**已结：用户 7/09 裁定 v3.1 = 最终版/主表口径（免神谕）**，v2 留作无 prompt 工程参照行 ② 蓝图#8 ASR 重听试点（TRIGGERED）③ P-1 写类分层口径 ④ G2'(ii) 自适应臂立项 ⑤ RB v1 批复。
- **下一步**：W3 GPU 项全清。**神谕请示稿已写：`手工文档/神谕/09_W3 终局与 W4 开工请示.md`**（终局数字表 / 学习组件收缩为"停时头"方案 + G2' 判据 / W4 逐日计划 / 裁断 ②–⑤+新增 A–D / 跑偏防火墙九条）。**W4 阶梯 rungs 2–3 已跑完（7/09 晚，近零 GPU：cache 214–217 hits/臂）**：四臂全败双门——v0 0.550/回收62.5%、safe 0.630/30%、rev 0.650/保费↑133.1（kill 判据未触发=κ 对齐有效）、prompted 0.610/**回收84%**。机制已钉死（probe 收据在 w4_ladder_design §7）：①早提交**关闭修订动作类**（fixed 靠 rescued_patch 赢的夹，短窗臂 patches=0，dropped 7/9 vs fixed 3）；②**afterthought 修订对尾韵律不可见**（rollback EoU 73% 标 final）——"韵律测话语完成，不测意图稳定"；③修订发生率是位置/话语结构变量非 κ 变量（rev 不掉 exact 的原因）；④fin12b/hou17b 间隙落 (1.0,1.5] 与 D3 预算阈值 1.12 咬合。**零 shot 前沿 = fixed(0,0)/safe(30%,−2)/pf(84%,−4)；rung 4 停时头目标 = 回收≥47% 且 ≥−1pt；特征以对话状态/位置为主，韵律降级**。零 shot 表封盘。**Rung 4 停时头 v0 代码已交付（7/10，预注册 = w4_ladder_design §8）**：`w4_synth_gen.py`（事件时间轴合成，无 TTS）→ `w4_hindsight_label.py`（hazard 目标=此刻风险非事后动作）→ `w4_train_stophead.py`（numpy LR+先验校正+合成集 c_w 扫描）→ `--delta-policy learned:v0 --stophead-model`（特征单源 `src/stophead.py`；learned 复用 finality_cache ⇒ FDB 评测近零 GPU）。冒烟 AUC 0.755/校准对齐；待用户按 §8 命令跑全量。scorer 已加 `--tag=` 防覆写（grid_{full,rollback}_TAG.json）。

### 既往（W3 D4–D6 代码批，2026-07-07）

- **D4–D6 代码/文书批已交付（7/07，单线程）**，全部机制藏默认关 flag 后，冻结路径逐位不动（37/37 + 新 89/89 单测 + 探针 VERDICT MET / 0 cache miss 复验）。清单与 GPU 出数命令：**`docs/w3_d456_handoff.md`**。
  - **现实档**：`src/latency_realistic.py`（κ 类对数正态，标定+预注册 `docs/latency_calibration.md`；per-instance sha256 播种=本地实现的 per-instance RNG；blocking 串行/TACT DAG 并行完成锚）。**纯记账层 ⇒ 决策缓存 100% 复用，δ 网格@现实档≈零 GPU**；`scripts/w3_realistic_rescore.py` 可对既有档案零 GPU 重打分。**预注册对表已出**（sblock vs tact 100 夹）：P-2 保费=**1.500 精确命中**（=δ*，档位无关定律实证）；P-3 链式条件式 31/33 恰过门（2 MISS 系两臂调用集不同）；P-4 齐 93/100（7=say 空 fallback）；P-1 写类 **0.237 ✓** 强达标 / 全量 **0.592 ✗**（>0.50，read_lookup 主导场景压比值——收益随工具时长放大的主张以**写类分层**形态成立，如实报）。
  - **状态轨双报 + 三分类**：`normalize_entity.py`（norm-v1 封闭规则集，无别名扩展）+ `w2r_state_track.py` 双报改造 + `w3_state_classify.py` 台账。**已出数**：TACT 57→**76**/100、sblock 57→75、blocking 57→74（P2' 差分制维持 TACT≥blocking；FT=0 无倒挂）。失败分类：format 24 主体 / missing_call TACT 7 < blocking 11 / **蓝图#8 试点 TRIGGERED**（asr_mishear 两臂≥3，立项待裁断）。
  - **DAG+补偿**：`src/tact_dag.py`（声明模板+值流证据边；patch 命中依赖字段→pending 下游重参数化/stale+窗重启、committed 下游→补偿计划；κ-忠实注册表 reverse 模板+幂等键 at-most-once）。`--dag on` / engine_cfg `dag:true`。
  - **speculative_dispatch**：core 算术（t_dec=seg_end+max(0.64,infer)）+ TactEngine 真实装（vad end 派发、EoU 确认前 inert、语音恢复作废、finalize 释放 guard——四组单测含屏障互作）。`--speculative on` / engine_cfg `speculative_dispatch:true`。
  - **分句 TTS + floor v0**：`tts_sentence.py`+`floor_policy.py`+TactEngine `tts_split`/`floor_holding`（TtsSentDone 事件流、first_sentence 完成锚仪器、barge-in 丢弃未播句；narration 无条件 yield，confirmation+commit 临近 finish_clause，粒度=句）。
  - **prompt v3 五靶**：`PROMPT_V3_ADDENDUM`（规则 10–14：后提交瘫痪/宣告即撤销/patch 对字段/自我打断替换/实体规范形态）`--prompt v3` 或 engine_cfg `prompt:"v3"` 显式装配；**30 调优子集已预注册** `exp/w3/tuning30.json`（21 rollback + 9 证据夹），协议 `docs/prompt_v3_five_targets.md`。驱动器新增 `--ids-file`。
  - **R18 缓存键审计闭合**：sha256(messages) 无 δ/档位/barrier 文本耦合；快照差异均属轨迹性合法臂差。
  - **剩余 GPU 项**（交用户，命令在 handoff §3）：30 子集现实档冒烟、现实档全量+δ 网格 A 档、投机 A/B+HumDial 门、DAG 链式冒烟、prompt v3 30 子集→冻结→全量+双档网格、分句 TTS live 增益（E5）。

### 既往（W3 D1–D3 + D1.5/D2 定界批，2026-07-06）

- **D3 提前交付**：三疑点逐数字核验 + 17×7 逐夹台账 `docs/w3_ledger.md`（真差分 = 赢 eco19/**hou21**/hou25、输 hou17b；travel_23 勘误已挂 w2_rerun_report；delta_hist **退役**；静默预算定律 δ>Σ静默(launch→救援EoU)，阈值 0.64/0.64/1.12/3.91；McNemar 修正 4 对 p=0.625）。神谕 06 批复：E2 闭合；**屏障裁断 = 显式设计（decision-atomic commit），grid v1 零重跑，永不重议**；prompt 批次定稿五靶；λ(t) v2 换静默钟重建。
- **D1 接线（代码交付，待手动验证）**：语义单源 **`src/tact_core.py`**（WindowLedger = 静默钟窗 + 提交屏障 + 双戳 + 三释放路径；决策应用/解析修复/prompt&snapshot v2 全部从 harness 迁入）；`src/engine_b.py` 重写为 Phase-B v1（v0 commit-后开窗语义退役；engine_cfg: phase/mode/delta/commit_barrier/stale_eou_policy/tool_sync/sv_alpha 占位）；`scripts/w2r_stream_replay.py` 降级为驱动器（`--commit-barrier on|off`、`--engine core|full`：core = 逐位契约路径，full = 帧→TactEngine 的 H6 仪器，仅 tact 模式）；backend.py `engine.phase: b` 分支接 live。规格 `docs/engine_b_spec.md`（五条款 + stale_eou_policy=apply 的 W2 逐位论证 + 感知差清单）。
- **验收工具**：`scripts/w3_barrier_probe.py`（四夹双语义 + rollback-17 on 12/17 / off **10/17** + 对 `result_w2r_tact_d100/d150` 全部确定性字段逐位对账——墙钟分量仅信息化：v1 档案 tool_wall 系 --workers 全局 RNG 打乱产物非权威，留 1.0s sanity 界；exit code 硬判；探针 provider 命名 w3p_*，不碰 w2r_* 冻结档）；`tests/test_commit_barrier.py`（纯 CPU、无模型无网络，37 项）。**跑前修正（预注册纪律）**：eco19 的救援 patch 是值中性 diff，其翻转为窗口保持 pending 的快照效应（barrier-off 下 δ≥1.0 仍过）——屏障因果权重收窄为 {hou25, fin12b}，06 预注册的 off=9/17 修正为 10/17；勘误挂 w3_ledger §4A 与 engine_b_spec §4。**已实测兑现**：机制判据全绿（on 12/17 / off 10/17 / d100 四夹全对、deferral 双戳 0.14/0.64 与台账算术吻合、80/80 缓存命中零 GPU）。W3 起 latency 增补 `completion_nominal_s`（墙钟免疫的完成锚）。
- **D1 验证收口（2026-07-06 当日完成）**：① 37/37 ✅ → ② core 探针全绿（VERDICT MET / exit 0 / on 12/17 off 10/17 / parity 21/21 逐位 + 10 条墙钟 info / deferral 双戳与台账吻合 / 80 缓存命中零 GPU）✅ → ③ S2 过（ack 层已 v1 重写 4/4；deterministic_replay 速度断言环境性挂，算力日复验）✅ → ④ full 仪器 13 偏差全归因：**ε 带 EoU 存在性效应**（间隙 0.67–0.708 三夹多 EoU 结构在流式感知下坍缩：eco19/hou25 转挂、hou17b 转过；ε 实测 0.03–0.13s）+ fin23 刀口翻转（prompt 五靶#4 证据）；serving 栈混杂（84g 临时配置）已记档 → ⑤ HumDial：三判 63.25±3.13 对 64.0 Δ=−0.75 门过、llm_timeout 0、`tact_*` 0、TTS 轮数 95/100 一致（5 夹差异疑 server 数值面非引擎）。完整记录 `docs/w3_d1_memo.md`（含 §6 补充实验）；神谕汇报稿 `手工文档/神谕/07_W3 D1 汇报.md`（**缓发**，并入 W3 总报告）。**D1.5 定界批已清账（7/06 晚）**：① HumDial 对称基线——W2 存档输出同日三判 61.17±1.94（7/03 单判 63.88 ⇒ **judge 供方跨日漂移 −2.7pt 实测**；新纪律：门槛比较必须同日同批对称判）；配对后 W3−W2 = **+2.08 正增益过门**（−0.75 系不对称伪差，勘销）。② server 漂移定界：10 缓存键绕缓存重调 **OPS-STABLE**（字节同 8/10，ops 同 10/10，漂移仅 say/dialogue 表面）⇒ fin23/hou17b 翻转主因 = ε 前缀效应。③ HumDial 生成栈已入档：`exp/w3/qwen3_omni_audio_84g.yaml`。④ **首响分解（裁断 B 回填，零 GPU）**：first = 0.64 hold + infer_final **恒等式零残差**；infer_final p50 0.501/p90 0.74；fallback（say 空）10/99；投机派发投影 first' p50 **0.64**（P3 比 0.786→**0.441 达标**）⇒ speculative_dispatch 维持 D6 实装。工具：`scripts/w3_server_drift_probe.py`、`scripts/w3_first_decomp.py`、`scripts/w3_d2_gap.py`。**⑤ D2 已完成（7/06 晚，H6 结案）**：full-100 `--engine full`（w3_tactfull_live，173 新决策）vs 离线 w2r_tact_full = **56 vs 56，差距 +0**；8 翻转逐例归因两两相消（ε 结构 4：eco19/eco25/hou25 挂、hou17b 过；前缀刀口决策 4：fin04/fin23/**travel_23** 过、fin22 挂）；感知敏感率 ≈8%（结构 4%+决策 4%），聚合不变。A 档双跑 run2 确定性字段 **100/100 逐位一致**（仅 2 夹 tool_wall 毫秒抖，completion_nominal_s 稳定）。延迟口径落定：pass=离线 core，延迟=W2 串行 live 档+D6 标准卡复测，full 轨管 H6/鲁棒性。E1 判据双满足（差 0 + 逐例归因）；H1/H2/H5 不再针对 pass 率启动（转真 websocket 演示链路问题）。**⑥ 零 GPU 并行批已清（7/06 晚）**：oracle 前沿三数（E4 ✓）= **固定 123.8s / oracle 20.9s / 回收上限 83.1%**（名义钟，96 可比场景；G2'(ii) 靶 = 自适应臂回收 ≥51.5s/百场景；travel_10 救援/撤销成本比 2.61×，cancel 规则零边际保费坐实 prompt 五靶#2）；hou11 契约审计闭合 = **官方 SUT 同样不喂前文（基准属性，不加 feed_history）**，该类恰 2/100（housing_11+ecommerce_12，全系统不可赢）；链式计数 = **28/100 无依赖 ≥2 调用**（A-iv：DAG claim 有据，链式修订覆盖为零→RB L7）；RB 设计文档 v1 `docs/rb_design.md`（80 规模/七层/ε 带 L3+竞态区 L4/post-EoU≥55%/屏障 held-out 消融）；04 勘误指针挂讫。**递延**：λ(t) v2（需 SenseVoice 多 cue 对齐工具）。工具：`scripts/w3_oracle_frontier.py`、`scripts/w3_hou11_census.py`。**下一步：D2（标准栈 live 全量重测 + 首响分解，ε 带发现列残差首查）+ 零 GPU 并行批（oracle 前沿三数 / hou11 契约审计与普查 / λ(t) v2 / RB 增补）。**

## 历史状态（W2 —— 重跑收口，2026-07-03）

**警示**：W2 首轮（17 代理并发，13:45–14:25）交付的实验结论**全部无效**（oracle 抄答案 / mock 数据 / 对照臂损坏），全部无效文档、伪 result 文件、mock 产物与失效脚本已**物理删除**（docs/ 只留 w1_*、fdbv3_memo、w2_rerun_report）。**有效结论只看 `docs/w2_rerun_report.md`（技术版）与 `手工文档/神谕/04_W2 执行汇报.md`（对神谕）。** 教训固化：多代理只用于写代码；实验结论必须单线程产出、逐数字对原始产物核验。

- G1'：P1 ✅（TACT@δ* exact 0.560 vs blocking 0.570；judge 0.693±0.015 vs 0.700）；P2 ⚠️（judge 轨 76.5% ✓ / 状态轨 70.6% < 85% ✗）；P3 ❌ 结构性（沙箱工具 p50 0.315s 无重叠空间=R9 兑现，双档评测预案待神谕裁断）；P4 ✅（HumDial 64.0 vs 63.88）；P5 ✅（三路缓存独立跑 100/100 工具序列逐位一致；解析末态失败 0，首试失败 1.6% 全部一次修复回收）。
- δ 扫描（rollback 17 夹，nominal infer）：δ*=1.5s，exact 0.706 **反超** sblock 0.588；eager δ=0 垫底 0.529（脏轨迹被 precision=1 处决）；**首响与 δ 解耦**（ack 占锚，平坦 1.64s），代价全在完成锚 ≈ +δ 线性。D2 直方图（VAD 段×SenseVoice token 真实对齐）：14/17 更正与意图同 VAD 段（EoU 粒度天然不暴露），跨段暴露间隙 0.42/1.12/1.16s——δ* 与间隙+决策延迟吻合，理论闭环第一圈成立。
- 全量 100（串行 live，独占）：TACT 首响 p50 1.141s / 完成 2.943s vs 流式 blocking 1.452 / 1.452。**决策延迟修正：整段工具决策 p50=0.444s**（此前 ~1.0s 是与 HumDial 管线共抢 vLLM 的污染值——延迟纪律教训第二次兑现）。首响物理地板 = 0.64 hold + 0.444 ≈ 1.08s；P3 要 ≤0.73s ⇒ 决策须 ≤0.09s ⇒ W3 增量决策/4B 头是唯一路径。
- **0.71<0.73 结案**（同 judge 重判 0.670 vs 0.690）：差距=2 场景——finance_14 同名调用顺序互换被官方按位置 pop(0) 对齐双杀（生态问题，提交时排序即修复）+ travel_16 实体逐字（Vegas/Las Vegas）。"急切调用被处决"证伪（0 例）。
- **R12 结案**：judge-pass 轨工具选择仍是二元 precision=1（严苛）；只有 evaluate_tool_calls 的 F1 渐变（宽恕，且带 turn_take_success=转写非空门）。双轨叙事按此分层。
- 状态轨判分器校准：blocking 上 TF=FT=0（≡exact，无虚增）；TACT 上 TF=1（ecommerce_15 先错后对=脏轨迹终态正确实例）。参数比较逐字沿用官方 exact_match_args。
- **judge 噪声带**：同文件三次重判 ±2pt（provider 侧非确定）。判读纪律：≤2pt judge 差不构成证据。
- ack-v0 实测（干净 provenance）：整句 TTS 0.933s → ack 0.429s，首音频提前 53.8%。
- 关键实现事实（W3 继承）：评测 harness=`scripts/w2r_stream_replay.py`（EoU=VAD 段尾+0.64s hold；异议窗音频钟、用户语音暂停倒计时、patch 重启窗；快照必须含已执行集否则跨 EoU 重复 launch；launch 幂等去重；参数 schema 矫正；salvage 解析器；**快照 op_id 必须局部编号**——全局计数器在并发下会使 prompt 漂移）。判分 `scripts/w2r_score_grid.py`；状态轨 `scripts/w2r_state_track.py`；直方图 `scripts/w2r_delta_hist.py`。
- rollback 21 中 6 场景无音频（released 只有 100 夹）；"回滚 21"物理上=15 场景/17 夹。
- **评测效率基建**（准确度回归轨；出数/确定性验证回 A 档串行）：harness `--workers 12` 全量 100 从 ~25min → 98s（thread-local VAD/HTTP、缓存加锁、nominal-infer 推进音频钟）；DeepSeek judge `FDB_LLM_WORKERS=100`（100 场景 ≈40s，run_fdb_with_deepseek.sh 已固化）；决策缓存跨 δ 网格复用（6 点 ≈1 点开销）。

**W3 入口弹药**：双档延迟评测裁断（R9）；增量决策/4B 头压首响地板（0.09s 目标）；live 差距重查（H6→H1→H2→H5→H3→H4）；状态轨 85% 缺口主体=与 blocking 共通的 ASR/实体逐字错误（非事务机制），实体规范化是否立项待裁断；patch 质量迭代（housing_17 打错字段，这次守 30 条子集纪律）；上游 PR 候选（latency_injector per-instance RNG、同名调用对齐语义）。

## 历史状态（W1 —— 已收口，2026-07-03 GPU 日）

**G0 达成**：感知冻结消灭 + 音频钟迁移 + 行为保持验证 + FDB 跑通。详见 `docs/w1_report.md`。

- [x] 全部 D0–D6 交付（引擎/工具/审计/双语/并发，提交历史 8dc0876..HEAD）
- [x] 真 LLM 等价性：分类决策 19/20 一致；legacy 自复现地板 2/3（墙钟抖动×T=0）；injected 回放保真 L1 20/20 @ 60× 实时
- [x] 冻结测量（真模型）：legacy 每 clip 停摆 0.5–0.9s vs actor 0（`docs/w1_freeze_real.json`）
- [x] HumDial 100 回归（seed 42 对齐 6/23）：TTS 轮数 94/100 一致；7 例为旧引擎丢 ASR 竞态（新引擎修复）；`llm_timeout` 0 触发。**分数级 judge 待 DEEPSEEK_API_KEY**
- [x] FDB-v3 blocking 冒烟 6/6 PASS（官方 scorer）；决策延迟基线：分类 p50 0.10s / response 0.15s / TTS 0.66s
- [x] 并发轨验证：24 会话 @ 并发 8 = 14s（mock）；SenseVoice 双语 ASR 验收（RTF 0.021, EN 带标点）

**W2 入口弹药**（蓝图 §4.2–4.3）：决策延迟基线已测；guided_json 可用性待查（vLLM 版本支持 structured output 大概率 ok，W2 Day1 确认）；PendingSet 原型已在 `/root/autodl-tmp/tact/transaction.py`；FDB 离线契约已验证。
