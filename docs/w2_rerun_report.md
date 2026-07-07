# W2 重跑报告（最终版 · 2026-07-03）

> 本报告是 W2 唯一有效技术报告。全部数字来自真模型（Qwen3-Omni-30B-A3B，vLLM A 档 `max_num_seqs=1`，
> T=0/seed=42）+ 官方 FDB-v3 evaluator + deepseek-v4-flash judge；沙箱延迟按场景档位 `random.seed(42)` 播种。
> 复现：`scripts/w2r_stream_replay.py`（评测 harness）→ `scripts/w2r_score_grid.py`（三轨判分）；
> 原始产物在 `exp/w2_rerun/` 与 `FDBench_v3/v3/fdb_v3_data_released/*/result_w2r_*.json`。
> 神谕口径汇报见 `手工文档/神谕/04_W2 执行汇报.md`。

## 1. 实验设置

- **blocking 臂（基线）**：`tact.offline_runner --mode blocking` — 整段音频一次决策，与 W1/6-23 同路。
  复现钉死：exact **0.570** 逐分对齐 W1；judge **0.700** 对齐 W1。
- **流式双臂（公平 A/B，同 prompt/解析器/矫正器）**：`scripts/w2r_stream_replay.py`
  - silero VAD 分段；EoU = 段尾 + 0.64s hold（静音须真持续过 hold）；
  - 每 EoU 真 LLM 决策（累积音频前缀 + PendingSet 快照：已执行集 + 局部编号 pending 集）；
  - TACT：launch → 异议窗 δ（音频钟计时，用户语音暂停倒计时；patch 重启窗）→ 窗竭 commit → 官方 mock API；
  - 流式 blocking：只在最后一个 EoU 决策一次、立即执行；
  - launch 幂等去重、参数 schema 矫正、salvage 解析器对两臂一致。
- **决策缓存** sha256(messages)（T=0 合法）；确定性验证用全新缓存独立跑。

## 2. G1' 对账表（n=100；rollback 子集=15 场景/17 夹，6 场景 released 数据无音频）

| 判据 | 阈值 | 实测 | 判定 |
|---|---|---|---|
| P1 pass 持平 | exact ≥0.560 且 judge ≥0.690 | exact **0.560**（blocking 0.570）；judge 三次重判 0.68/0.71/0.69（均值 **0.693**，blocking 0.700，judge 噪声带 ±2pt） | ✅ |
| P2 回滚 | judge ≥70.6% 且状态轨 ≥85% | judge **76.5%**（sblock 70.6%）✓；状态轨 **70.6%** < 85% ✗ | ⚠️ 半过 |
| P3 延迟 | 首响 p50 ≤ blocking 50% 且 ≤1.2s；完成 ≤70% | 首响 **1.141s**（✓绝对值）= sblock 1.452s 的 **79%** ✗；完成 2.943s = **203%** ✗ | ❌ 结构性（§6）|
| P4 双工护栏 | HumDial Δ ≥ −1 | Overall **64.0** vs W1 63.88（Δ+0.12） | ✅ |
| P5 确定性 | 双跑逐位一致；解析失败 0 | **三路复证**：串行新缓存 vs 并发新缓存 vs 主缓存，工具序列 **100/100 逐位一致**；317 决策 5 次首试解析失败全部一次修复回收，末态失败 **0** | ✅ |

## 3. 核心数字

### 3.1 全量 100 三轨

| 臂 | exact | state | judge | 首响 p50/p90 | 完成 p50/p90 | ack率 |
|---|---|---|---|---|---|---|
| blocking（整段，W1 同路）| **0.570** | 0.570 | **0.700** | — | — | 0 |
| sblock（流式 blocking）| 0.570 | 0.570 | 0.713±0.006 | 1.452/1.967 | 1.452/1.967 | 0 |
| TACT @δ*=1.5s | 0.560 | 0.570 | 0.693±0.015 | **1.141/1.505** | 2.943/3.487 | 0.89 |

judge 为 3 次重判均值±半极差（同文件重判噪声 ±2pt，≤2pt 的 judge 差不是信号）。
延迟为串行 live-infer（独占服务器）。首响 p90 改善 23%；完成延迟 +1.49s = δ 窗构造性代价（§6）。

### 3.2 δ 扫描三曲线（rollback 17 夹；nominal infer=1.0s 隔离推理抖动，δ*=1.5 的 exact 已由串行 live 全量跑确认同值）

| δ (s) | exact | state | 首响 p50 | 完成 p50 |
|---|---|---|---|---|
| sblock | 0.588 | 0.588 | 1.955 | 1.955 |
| 0 (eager) | 0.529 | 0.529 | 1.640 | 1.868 |
| 0.3 | 0.529 | 0.529 | 1.640 | 2.166 |
| 0.6 | 0.529 | 0.529 | 1.640 | 2.517 |
| 1.0 | 0.647 | 0.647 | 1.640 | 2.955 |
| **1.5 = δ\*** | **0.706** | **0.706** | 1.640 | 3.455 |
| 2.5 | 0.706 | 0.706 | 1.640 | 4.455 |

三条结构性结论：
1. **准确性阶梯**：δ<暴露间隙 → 更正落窗外 → 脏轨迹被官方 precision=1 处决（eager 0.529 垫底）；
   δ 越过间隙分布（D2 实测 p50 1.12s，加决策延迟）→ patch 落窗内 → **0.706 反超 sblock 0.588**。
2. **首响与 δ 解耦**：ack/announce 在 launch 即刻发出，首响平坦 1.64s——异议窗的延迟代价全部落在完成锚，一点不落在首响锚。
3. **完成延迟线性 +δ**：更正大多发生在音频尾部之后（窗在尾静音中倒数），代价 ≈ δ 全额。δ*-完成曲线即 W3 的优化对象。

TACT@δ* 对 sblock 的 rollback 净胜场景：ecommerce_19、housing_25、travel_23（+finance_12 双夹之一）；
净负：housing_17（更正后模型 patch 了错误字段）。judge 轨 rollback：**13/17=76.5% vs 12/17=70.6%**。

> **勘误（2026-07-06，W3 逐数字核验）**：上行明细有误。真实差分 = 赢 ecommerce_19、**housing_21**、housing_25，
> 输 housing_17b；travel_23 不在 rollback 名册且任何 exact 口径都非 TACT 胜；finance_12b 属臂内窗阶梯而非臂间净胜。
> §3.1 结论 1 的"暴露间隙+决策延迟"算术同废，替换为静默预算定律。全部逐夹归因见 `docs/w3_ledger.md`。

### 3.3 D2 经验 Δt 直方图（真实对齐：silero VAD 段 × SenseVoice token 时间戳）

`exp/w2_rerun/delta_hist.json`。15/21 rollback 场景有音频（17 夹），更正提示词全部定位成功。
**14/17 的更正与初始意图在同一 VAD 段内**（EoU 粒度下决策器天然听到含更正的完整段——异议窗保护的
是剩余 3/17 的跨段更正 + 跨 EoU 场景）；跨段暴露间隙：0.42 / 1.12 / 1.16s。此分布是 λ(t) 的第一份
经验估计（W5 hazard 头标注雏形），也解释了话语中段抢跑在官方判分下价值为负的裁断。

### 3.4 状态轨判分器校准矩阵（效度证明）

| provider | TT | TF | FT | FF |
|---|---|---|---|---|
| blocking-100 | 57 | **0** | **0** | 43 |
| TACT-100 | 56 | **1** | 0 | 43 |

写类工具按语义键归约（last-wins/累加），READ 查终版参数，$-引用动态匹配；参数比较逐字沿用官方
`exact_match_args` → 矩阵只度量**结构宽恕**。blocking 上状态轨 ≡ exact（判分器无虚增）；TACT 唯一
TF = ecommerce_15（track_order 先错 id 后对 id：终态正确、轨迹脏——状态轨存在意义的实例）。

## 4. 两大悬案结案

### 4.1 0.71 < 0.73（6/23 async vs blocking）

同 judge（v4-flash）重判 6/23 原始产物：async **0.670** vs blocking **0.690**。逐场景差分——差距全部
来自 2 个场景：
1. **finance_14**：同名工具两次调用**顺序互换**。官方 evaluator 对同名调用按位置 `pop(0)` 对齐 →
   参数集合完全正确却双双记 0。evaluator 生态问题；提交时按规范序排序即免费修复。
2. **travel_16**："Vegas" vs "Las Vegas" + "August 8th" vs "August 8"（实体逐字性，judge 同样不宽恕）。

**"急切调用被 precision 处决"假设证伪：0 例。**

### 4.2 R12（judge 轨对多余调用的语义）

v3 有两条判分线：`evaluate_pass_rate --use-llm`（判 pass）的工具选择**仍是二元多重集 precision=1**，
LLM 只接管参数比较——多余调用一票否决（严苛分支）；`evaluate_tool_calls`（metrics 轨）tool F1 渐变
（宽恕分支），且有 `turn_take_success` 门（=转写非空）。补偿机制的展示阵地在 metrics/状态轨。

## 5. 修正与新事实

- **决策延迟修正**：整段音频工具决策 p50 = **0.444s** / p90 0.646s（串行、独占）。此前中间报告的
  ~1.0s 是与 HumDial 回归管线共抢 vLLM 的污染值——W1"延迟纪律"教训的再次实证，已修正。
  首响物理地板 = 0.64 hold + 0.44 决策 ≈ **1.08s**；要达成 P3 的"≤blocking 50%"（0.73s），
  决策须 ≤0.09s——恰是 W1 分类决策的量级（0.10s），指向 W3 增量决策/W4 4B 决策头。
- **judge 噪声带**：同文件三次重判波动 ±2pt（provider 侧非确定）。判读纪律：≤2pt 的 judge 差不构成证据。
- **ack-v0 实测**（重跑、干净 provenance，`analysis/ack_benchmark/`）：整句 TTS 0.933s → ack 短句
  0.429s，首音频提前 **53.8%**（5/5 例）。与首响锚数字自洽。
- **v3 指标终审**（D1-③）：pass 轨无 turn-take 门；metrics 轨有 `turn_take_success`（定义=转写非空），
  默认 metrics 只统计 turn-taken 子集。早期"turn-take 78%"情报出自 metrics 轨定义，非 pass 门。

## 6. P3 结构性失败分析（=风险册 R9 实现）

沙箱工具墙钟 p50 **0.315s** / p90 0.482s——重叠收益的理论上限被官方延迟档压死；异议窗在完成锚上是
纯加法 +δ。**机制没有输，是这套延迟档下没有可赢的时间。** 论文对策（神谕 R9 预案）：双报
"官方档 + 现实档（真实 API 数百 ms–数 s）"，δ*-完成曲线论证延迟档的生态效度；W3 增量决策压首响，
现实档放大重叠收益。

## 7. 评测效率基建（准确度回归轨；论文数字仍守 A 档串行）

- `w2r_stream_replay.py --workers N`：线程池并发（VAD/HTTP thread-local、缓存加锁、**nominal infer**
  推进音频钟消除争用不确定性）。全量 100：串行 ~25min → **12 workers 98s**。位级可复现已三路验证。
- 可复现性根修：PendingSet 快照的 op_id 从全局计数器改为**局部编号**（并发下全局取号交错 → prompt
  文本漂移 → 缓存键漂移 → 决策漂移；修后并发=串行逐位一致）。
- DeepSeek judge：官方 `llm_judge.py` 原生 `FDB_LLM_WORKERS`，默认 16 → **100**（`run_fdb_with_deepseek.sh`
  已固化）；100 场景 judge 约 40s。
- 决策缓存跨 δ 网格复用：6 点网格 ≈ 1 个点的模型开销。
- vLLM A 档 `max_num_seqs=1` 下并发收益来自重叠 VAD/音频/工具等待；如需更高吞吐走 T 档（text-only
  高并发配置），但**确定性验证与出数必须回 A 档串行**。

## 8. 复现索引

```
scripts/w2r_stream_replay.py     # 评测 harness（流式双臂 + 异议窗 + --workers）
scripts/w2r_score_grid.py        # exact/state/latency 三轨判分
scripts/w2r_state_track.py       # 状态轨判分器 + 校准矩阵
scripts/w2r_delta_hist.py        # D2 直方图（VAD×ASR 对齐）
exp/w2_rerun/delta_hist.json     # D2 逐场景明细
exp/w2_rerun/grid_rollback.json  # δ 网格三轨原始分（nominal）
exp/w2_rerun/grid_full.json      # 全量 100 三轨原始分（serial live）
exp/w2_rerun/state_track_*.json  # 状态轨明细 + 校准矩阵
exp/w2_rerun/decision_cache*.json # 三份独立决策缓存（P5 证据）
FDBench_v3/v3/w2r_*_pass_judge_report*.json  # judge 轨官方报告（rep1-3）
logs/humdial_w2rerun_eval/summary.json       # P4 护栏
analysis/ack_benchmark/ack_latency_improvement.json  # ack-v0 实测
```
