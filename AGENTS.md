# AGENTS.md — fd-badcat 持久记忆（所有代理必读）

> 单一真相源。CLAUDE.md 指向本文件。有重大事实变更时**更新本文件**，不要另开新文档。
> 最后更新：2026-07-03 (W1 Day 0 夜间)

## 使命

把 fd-badcat（HumDial Challenge 半级联全双工语音系统）改造为 **TACT**（事务性工具调用的全双工语音 agent），投稿 **ICLR 2027**（摘要 9/19、全文 9/24）。

- **总蓝图（宪法）**：`手工文档/神谕/00_系统蓝图.md` — 论文命题、形式化、评测、时间线。尽量不违背；违背需有充分理由并记录。
- **W1 执行计划**：`手工文档/神谕/01_W1 完整计划.md` — 可微调（它基于云端旧代码写成，部分已过时）。
- 人类评估（伦理审批等）**暂时全部跳过**，只关心代码与模型能力。

## 环境事实

- 工作目录 `/root/autodl-tmp/fd-badcat`；FDBench 在 `/root/autodl-tmp/FDBench_v3`（v1_v1.5/v2/v3 三代，用 v3）。
- **conda 环境**：`fd-sds`（`/root/miniconda3/envs/fd-sds`，backend 运行环境：soundfile/silero_vad/fastapi/torch 全有）；`/root/autodl-tmp/conda-envs/` 下有 `fdb_v3`、`fdbc-qwen3o-vllm`（vLLM 服务）、`index-tts-vllm`。直接用绝对路径 `/root/miniconda3/envs/fd-sds/bin/python` 最稳。
- **GPU 常常不在位**（省钱策略：白天租卡）。无 GPU 时：写代码、mock 测试、CPU 级验证；一切需要 vLLM/真模型的验证列入「GPU 快速启动清单」（见 `docs/w1_report.md`）。
- 服务拓扑（GPU 在位时）：vLLM Qwen3-Omni-30B-A3B :10003 → 代理 `src/qwen3_api.py` :10004 → backend :18000。TTS 默认走 **Omni 原生**（`FDBC_TTS_PROVIDER=omni`，走同一个 vLLM），Index-TTS(:19000) 是可选回退。启动脚本在 `setup/`。
- 网络：本机代理环境变量存在，`_LOCAL_HTTP.trust_env=False` 与 `NO_PROXY=127.0.0.1` 是既定约定，新代码访问本地服务必须沿用。

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

## 当前状态（W1）

- [x] D0: `tact` 分支 + `golden-base` tag + `backend_legacy.py` 冻结 + `llm.audio_block` 开关（commit 8dc0876）
- [ ] D1–D3: `src/engine.py` actor 化（事件模型/音频钟/决策分叉/陈旧性协议）
- [ ] 单测 S2 十项 + trace_diff + 回放框架 + 冻结测量
- [ ] D4: ASR 工厂（sensevoice 双语，flag 后，默认 paraformer_zh 不动）
- [ ] D5: FDBench_v3 深审计 → `docs/fdbv3_memo.md`（Q1 判分语义 = 全文最大单点风险）
- [ ] D6: FDB adapter + agent_blocking 稻草人基线
- [ ] 真 LLM 金标录制与等价性复验（**需 GPU**，见快速启动清单）

## GPU 到位后的快速启动

见 `docs/w1_report.md` 的「GPU 快速启动」节（W1 收口时写就）。核心顺序：起 vLLM → 起代理 → 录金标 trace（旧引擎）→ 新旧引擎等价性复验 → HumDial 回归 → FDB 冒烟。
