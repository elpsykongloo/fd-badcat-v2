# AGENTS.md — fd-badcat 持久记忆（所有代理必读）

> 单一真相源。CLAUDE.md 指向本文件。有重大事实变更时**更新本文件**，不要另开新文档。
> 最后更新：2026-07-03 (W2 重跑收口)

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
- **代理坑**：环境有 `all_proxy=socks5://…`，openai SDK(httpx) 缺 socksio 会静默初始化失败 → FDB evaluator 的 LLM judge 悄悄退化成 exact-match。跑 evaluator 前 `unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`（DeepSeek 国内直连）。
- **FDB 分数口径**：永远双轨报告 exact + deepseek-v4-flash judge。基线：blocking exact 0.570（W1 逐分复现 6/23）/ judge 0.700；6/23 的 0.73 是 gpt-5.5 judge 口径，不同 judge 不可直接比。**v3 有两条判分线**：`evaluate_pass_rate.py`（二元 pass，无 turn-take 门，工具选择 precision=1 一票否决——judge 模式也只接管参数比较）与 `evaluate_tool_calls.py`（metrics 轨，有 `turn_take_success` 门=转写非空，默认 metrics 只算 turn-taken 子集，tool F1 渐变）。同名工具多次调用按**位置 pop(0) 对齐**——顺序互换会双杀（0.71 之谜根因）。
- **W1 汇报文件**：`手工文档/神谕/02_W1 执行汇报.md`（发给神谕求 W2 计划的汇总，含全部实验数字与开放问题）。

## 仓库拓扑与分叉事实（重要）

- **本地是云端的严格超集**：`origin/main` 顶端 `f4ff2b1` 即本地历史的 merge-base；本地在其上有 10+ 提交（vLLM 本地化、HumDial 批量评测、Omni TTS 端到端）。**不要从云端拷文件回来**（云端 `qwen3_api.py` 有 setdefault-after-post 的无效代码 + `/data/ptmodels` 路径；本地早已修好）。
- 音频块格式分叉已做成开关：`llm.audio_block: audio_url`（本地 vLLM 默认）| `input_audio`（OpenAI 严格裸 b64）| `input_audio_datauri`（云端 DashScope 方言）。实现在 `src/messages.py`。
- **工作分支 `tact`**，基线 tag `golden-base`（=420b539）。`main` 不动。`src/backend_legacy.py` 是旧引擎逐字节冻结件，W1 全程保持可运行（A/B 对照），别改它。
- 本仓库有多个代理活动过（git author 见 Codex 提交）。动 git 历史前先 `git log --oneline -3` 确认没踩到别人的新提交。

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
  - **DeepSeek judge 并发**：官方 `llm_judge.py` 原生支持 `FDB_LLM_WORKERS`（默认 16 太保守），judge 是纯 API 调用，**直接 100**；`scripts/run_fdb_with_deepseek.sh` 已改默认 100。
  - vLLM 侧：准确度回归可临时把 audio 配置 `max_num_seqs` 调高（重启分钟级）配合 --workers 压满；HumDial 管线自带 `--gen-workers/--clean-workers/--asr-workers/--judge-workers` 旋钮。

## 当前状态（W2 —— 重跑收口，2026-07-03）

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

## GPU 到位后的快速启动

见 `docs/w1_report.md` 的「GPU 快速启动」节。核心顺序：起 vLLM（audio 配置）→ 起代理 → 冒烟 → 金标 trace（旧引擎）→ 新旧等价复验 → HumDial 回归（`run_humdial_100_pipeline.py --seed 42`，同种子=同样本集可与 6/23 产物逐样本对齐）→ FDB 冒烟（tact.offline_runner --limit 5）。
