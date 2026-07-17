# Revision-Bench (RB) v2 —— 两段式预注册（W5-RB）

> 状态：**Phase-0 结构冻结（2026-07-15）**。数值（网格配额/判分常量/功效表/kill 阈）在 Phase-1 回填后冻结为 v2.1；冻结前不做任何系统评测。
> 用户裁定（7/15 chat，全部入册）：**混合语音制**（TTS 主网格 + 真人金子集）；规模 **~1,000 episodes**；**双臂制**（固定时间轴 + 反应式）；**承诺分层 floor policy（W5-FC）同批立项**；**对抗臂进第一批**。TTS 栈 = **Qwen3-TTS 开源版（CustomVoice 模型，9 个预设音色）**。
> 授权背景：预注册自由偏移 + 资源不设限（用户 7/15，AGENTS.md 入册）。
> 底稿关系：v1 原案（2026-07-06）逐字保留为附录 A；**v1 §5 "hazard 头训练正例 = L3–L6" 条款正式作废**——它先于防火墙成文，违反"FDB/RB=测试集，任何标签不得进训练"铁律。

---

## §0 一句话与站位

RB = **修订密集、时机受控、生命周期对齐**的事务性全双工工具调用诊断基准。反例声明（设计动机）：FDB-v3 仅 100 场景、预录非反应式、修订承重 op 仅 13 个、门与差距落在 1–2 夹刀口（±9pt 二项噪声）；RB 目标 = 千级规模、逐条件格统计功效、条件全参数化声明、播种确定性。

三轨定位：**(i) 官方兼容轨**（exact/state，与 FDB 口径可比）；**(ii) 事务轨**（承诺-修复、补偿计价、联合效用——TACT 代数中从未有考场的部分：inflight 代价模型、补偿注册表、L7 链式修订）；**(iii) 双工轨**（首响/完成/talkover/打断响应）。

## §1 相关基准差异化表（2026-07-15 摘要/README 级已核；全文核对 = Phase-1 义务 #1）

| 基准 | 用户侧 | 工具耦合 | 修订时机控制 | 规模 | 核验级 |
|---|---|---|---|---|---|
| FDB-v3 (arXiv:2604.04847) | **12 名真人预录**，5 类不流利标注 | 4 域链式 API，确定性沙盒 | 不流利自然采样，无时机分层 | 100 | 摘要+README ✓ |
| FDB-v2 (arXiv:2510.07838) | **闭环反应式考官**（WebRTC/WS + LLM Examiner） | **无工具/API 状态耦合**（README 级） | 无 | — | README ✓ |
| τ-Voice (arXiv:2603.13686，Sierra；AA S2S 指数占 33.3%) | LLM 模拟用户 + 真实音频条件（口音/噪声） | τ²-bench 任务（airline 50 / retail 114 / telecom 114） | 测任务完成率，不控修订时机 | 278 | 摘要+AA 页 ✓ |
| DuplexSLA-Bench (arXiv:2605.20755) | 静态案例（摘要级，待核） | 时序感知工具协议 | 无分层 | 2,100 | 摘要级（转述） |
| VoiceAgentBench / Audio2Tool / EVA-Bench (2605.13841) | TTS 合成查询，半双工为主 | 工具准确率轴 | 无 | 各异 | 转述，待核 |

**差异化主张（冻结措辞）**：RB 是（据我们 Phase-0 核验）**第一个用户事件以被测系统自身工具调用生命周期为触发条件、且修订时机按静默钟坐标受控分层**的全双工工具调用基准。若 Phase-1 全文核对发现先例，措辞降级为如实比较——设计价值（考场覆盖）独立于首创性，网格不因此改动。

## §2 双臂制（冻结）

- **臂 A 固定时间轴（可比轨）**：episode = 预生成音频时间轴（话语 wav + 帧级时刻表 + 修订 cue 多重标注），scripted-VAD 回放，跨系统**逐位可比**（FDB 口径兼容）。事件落点以"名义生命周期"（0.64 hold + nominal infer）预采样目标区间；评测后按被测系统**实际** gap 归箱报告（v1 §3 "过采样+事后分箱"原则的机器化——TTS 时间轴使落点可精确控制，真人念秒表问题在此轨消失）。
- **臂 B 反应式（闭环轨）**：用户模拟器订阅被测引擎 trace 事件流（`act_launched/inflight-ETA/act_committed/act_compensated/tts_start/tts_sent_done/floor_decision`），按**声明式事件语法**触发用户行为：barge-in、patch、cancel、进度询问、旁观者注入、沉默走开。语法 = (生命周期状态 × 偏移分布 × 内容模板) 三元组；LLM 仅在内容槽内生成话语文本（决策缓存 sha256、T=0、seed 固定）；话语音频 = Qwen3-TTS 预合成池 + 即时拼接。确定性口径：同 (系统, episode, seed) 逐位复现；**跨系统时间轴不同是特性**（生命周期对齐的定义），跨系统可比性由臂 A 承担——双臂制正面回应"反应式事件不可复现"的审稿攻击。
- 双臂共享场景网格与判分器。臂 B 首要预注册假设 **H-B1：现有系统的双工失败集中于 in-flight 窗口**（显著高于调用前窗口；迁移自外部报告 Idea 1-H1，属可证伪移植）。

## §3 场景网格（v1 七层保留 + 三层扩展；配额数值 Phase-1 冻结）

| 层 | 定义 | 动机（考什么） |
|---|---|---|
| L1 同段修订 | 修订与意图同 VAD 段 | 对照层（hold 合并） |
| L2 sub-hold 跨段 | gap < 0.64 | hold 层 |
| L3 ε 带 | gap ∈ [0.64, 0.80] | EoU 存在性翻转带（W3 实测 ε≈0.03–0.13） |
| L4 决策在途竞态区 | gap ∈ (0.64, 0.64+infer]，修订开口即含新值 | **提交屏障** held-out 消融主考场 |
| L5 post-EoU 长间隙 | gap ∈ (1.0, 4.0] | δ 窗阶梯区 |
| L6 多重跨 EoU 修订 | ≥2 处修订、≥1 处跨 EoU | 单 cue 盲区、cancel-on-announce、patch 链 |
| L7 链式修订 | ≥2 无依赖调用、修订命中其一 | DAG 传播/重参数化（FDB 零覆盖） |
| **L8 在飞事件（新）** | patch/cancel/进度询问落在 inflight 窗口 | def 2 代价模型（abort-relaunch vs wait-compensate）与**补偿注册表的首个考场**（官方轨 comp_plans=0） |
| **L9 时延长尾（新）** | 工具 T 三档 {0.3–1s / 1–5s / 10–60s}，κ 类对数正态+长尾 | floor/承诺策略主考场；DAG 并行收益随 T 放大（P-1 写类分层的受控版） |
| **L10 对抗（新，第一批）** | 第三方声注入：时机（生命周期偏移）× 内容（指令性 patch/cancel vs 无关语音）× 声源（异音色=第三方 / 同音色对照）+ 良性对照（用户本人修订不得误杀） | 蓝图情况 #14 / §4.6 **SV 门控复活**考场；误触发率 vs 良性穿透率 |
| 修订类合计占比 ≥55% 维持（v1 口径）。 | | |

**域**：4 个自建域（沿 FDB 域类型学：电商/金融/住房/出行——**场景与 API 全部原创**，题目零重叠）；每域工具注册表带 κ 分级 {R0–R3}、reverse 补偿模板、幂等键；沙盒 = 确定性状态机 + 播种延迟。**语言**：中英双语（比例 Phase-1 定，Qwen3-TTS 双语能力验证前不承诺）。

## §4 语音生产（混合制，用户裁定；冻结）

- **主网格 = Qwen3-TTS 开源版（CustomVoice，9 预设音色）**：修订话语构式模板库（自然改口词形："等等/不对/换成…"、"actually/wait/no—"、假起始、犹豫填充——五类不流利映射 FDB-v3 类型学）+ 扰动族（语速/音量/SNR 档，全部播种）。修订 cue 时刻由拼接时间轴**精确控制**——这是 TTS 轨相对真人录音的结构优势（时机受控性），也是其风险（韵律自然性），风险由金子集兜底。
- **真人金子集 = 100–150 夹自录**（2–4 名说话人，覆盖 L3/L4/L5/L10 关键条件格）：用途 (i) TTS↔真人**一致性验证**——系统排序 Spearman 相关 + 关键指标（pass/首响/修复率）逐格相关；不一致处如实报并把相应主张限定到金子集；(ii) 论文可信度锚（τ-Voice 先例证明合成用户可被接受，但真人锚更稳）。
- **红线**：RB 零 HumDial 内容（许可禁再分发）；HumDial 仅贡献时序统计先验（停顿分布 → 事件偏移采样分布校准，同 C1 世界锚用法，数值经 `assert_no_text` 类产物纪律）。
- TTS 伪影申报：RB 是评测集、永不进训练 ⇒ 蓝图"拼接捷径学习"风险不适用；评测侧风险=修订韵律不自然，处置见金子集条款与 §9 kill。

## §5 判分（三轨；常量 Phase-1 冻结）

1. **官方兼容轨**：exact + state（verbatim/normalized norm-v1）判分器移植；**同名调用对齐改 canonical-sort 多重集**（修正 pop(0) 位置对齐怪癖，差异文档化——生态勘误转化为设计改进）。
2. **事务轨**：承诺-修复计分——say 通道承诺抽取（LLM 判官，strict runner 纪律：1024 tokens / retry 5 / fallback 硬失败 / 三判均值制）对照沙盒终态与工具结果：`WrongCommit@t`、修复率、未修复承诺数、补偿计价 ΣC_κ、R3 违规数。
3. **联合效用**：`U = 1[终态对] · γ^{完成延迟} − w_r·未修复承诺 − w_c·ΣC_κ`（γ/w_r/w_c 数值 Phase-1 冻结 + 敏感性扫描随判分器发布）。
4. **双工轨**：首响/完成 p50/p90、talkover 率、被打断响应延迟、floor tier 分布。
- 统计口径：逐条件格 McNemar 配对 + 原始计数永远并报；judge 噪声带纪律沿用（≤2pt 不进结论句）。

## §6 W5-FC：承诺分层 floor policy（同批立项，机制侧预注册）

- **机制**：floor_policy v1 = 五档 `{silence, filler, progress, hedge(对冲部分答), commit}`；输入 = 注册表 ETA 先验 + inflight 结果置信 + κ + 窗口状态；输出 = 档位 + 模板话语；narration 无条件可打断不变（I1）。默认关 flag `floor.commit_tiers`，冻结路径逐位不动。
- **理论注**：档位选择 = 等待成本率 vs 期望修复代价的阈值策略（与 P2 停时同族：P2 管"何时提交动作"，FC 管"等待时把话说多满"）；最优性注记进理论节。
- **门（冻结）**：官方轨 flag-off 逐位不变（assert）；HumDial 三判配对 Δ≥−1；**H-FC1**：L9 长尾层上 progress+hedge 策略在 (修复率, 被打断率, U) 前沿支配恒填充与恒沉默（FDB-v3 实测 Ultravox 88% 恒填充、47.9% 撞用户 = 反例锚）；**H-FC2**：短时延档 silence 不劣于 filler。
- 评测臂：`{恒沉默, 恒填充, v0 三档, v1 五档}` × L9（+L8 进度询问格）。

## §7 规模与功效（~1,000；数值 Phase-1 冻结）

- 目标 ≈**1,000 episodes**：臂 A/B 配比 Phase-1 按生成与运行成本定（初值臂 A ~600 / 臂 B ~400）；每关键条件格 ≥30 配对（McNemar 功效：按 W2 实测翻转率 ~24%，格内 30 对 ⇒ 不和谐对 ~7，格间合并后主比较 ≥20 达标；精确功效表 Phase-1 出）。
- 生成全程序化：generator `config_hash` 纪律沿 `w4_synth_gen`；episode = 声明式 config 元组（层×域×κ×T 档×音色×扰动×seed）；A 档双跑逐位确定性为验收门。

## §8 防火墙与 dev/test 切分（自建自评的防御；冻结)

1. **RB = 评测 only**：系统侧任何训练/调参/θ 选点/prompt 迭代不得接触 RB 任何 split。
2. **RB-dev (~10%) / RB-test (~90%)**：harness/判分器调试只许用 dev；**test 上每系统版本单发**；判分器在首次 test 运行前冻结（代码+常量哈希入册）。
3. **设计冻结先于观测**：网格与判分器冻结于任何系统 test 表现观测之前；此后发现的缺陷只许勘误页申报（公开），不许改-重跑（判分不可用级缺陷除外，须公开记录版本递增）。
4. **释出**：生成器/判分器/TTS 配置/seeds/事件语法/金子集音频（自录，授权自有）全部随论文开源。
5. 模型动物园（评测对象，全部冻结版本）：blocking/sblock、TACT δ 网格 ± 屏障消融、spec、pf、v2/C0/C1 π 点、W5-SG 门控（若过门）、W5-FC 档位臂。oracle 前沿在 RB 上重算（RB 版回收上限）；λ(t) 在 RB 上**仅作事后诊断**，不进任何训练（v1 §5 作废条款的替代表述）。

## §9 Phase-1 清单（数值冻结前，零系统评测）

1. **相关基准全文核对**（§1 表升级为全文级）：τ-Voice 2603.13686、FDB-v2 2510.07838、FDB-v3 2604.04847、DuplexSLA-Bench 2605.20755、VoiceAgentBench、Audio2Tool、EVA-Bench 2605.13841；差异化主张表定稿。
2. **Qwen3-TTS 栈验证**：9 音色可用性、中英双语能力、修订构式合成质量人工听检（30 句/语言）、合成吞吐；不过检处置见 kill。
3. 域与工具注册表设计（4 域 × 工具 × κ 表 × reverse 模板 × 幂等键）+ 沙盒状态机。
4. 事件语法形式规格 + 反应式模拟器接口（trace 事件订阅协议）+ 决策缓存接线。
5. 网格配额表 + 功效表定稿；判分常量（γ/w_r/w_c）冻结；金子集录制脚本与分层表（交用户执行录制）。
6. **kill/降级判据裁定**：(K1) TTS 听检不自然率 ≥20% 且无替代栈 ⇒ 语音制降级纯自录 + 规模砍至 ~200，预注册如实改版；(K2) 金子集一致性 Spearman <0.7 ⇒ 全部主张限定金子集 + TTS 轨降为补充材料；(K3) 反应式臂确定性验收不过（双跑不逐位）⇒ 臂 B 降级为固定时间轴的条件化变体。

## §10 时序

Phase-1（普查/验证/规格，多数零 GPU）→ 数值冻结（v2.1）→ 生成与 dev 冒烟 → 判分器冻结 → 金子集录制（用户）→ test 全量（模型动物园单发）→ 论文素材。W5-SG / W5-FC / W5-RB 三线并行，SG 的 Phase-1（HumDial 引擎口径普查）与 RB 的时序先验校准共用一次数据扫描。

---

## §11 v2.1 数值冻结（第一批，2026-07-15）与实现台账

**K1 裁定（用户 2026-07-15）**：Qwen3-TTS 双语与修订构式听检**通过**——混合语音制确认，K1 降级分支不触发。

**冻结数值（第一批；与 census 无关的常量，代码即真相源）**：
- 配额（`rb/generator.py`）：臂 A 600 = L1 48/L2 42/L3 72/L4 72/L5 96/L6 48/L7 48/L8 60/L9 72/L10 42；臂 B 400 = L4 60/L5 60/L6 40/L8 120/L9 80/L10 40。实测全量构建：1,000 episodes、dev/test = 103/897（sha256(id)%10 规则）、修订占比 **0.673 ≥ 0.55** ✓；每臂每层 ≥40（配对功效下界满足）。
- 判分常量（`rb/scorer.py`）：**γ=0.95/s、w_r=0.25、w_c=0.05**、C_κ={READ 1, REV 2, COMP 4, IRR 8}；敏感性扫描 ×{0.5,1,2} 随行报告、永不用于选型。COMP 补偿审计残差费=1 单位（`rb/sandbox.py`）。
- 时延类（`rb/sandbox.py`）：short=LogN(ln0.5,0.4)、mid=LogN(ln2.0,0.5)、heavy=LogN(ln20,0.6)、cap 60s；L9 全 op（READ 除外）走 heavy。
- W5-FC 常量（`src/floor_policy.py`）：T_SILENCE=1.0s、T_FILLER=5.0s、CONF_COMMIT=0.9、ESCALATE=1.5×；commit/repair 机器标记 = "已确认：/Confirmed: " 与 "抱歉，刚才说错了：/Sorry, correction: "（scorer 的 trace-metadata 模式依赖此约定，模板改动=判分器版本递增）。
- **仍待冻结（第二批，census 后）**：pause_prior 校准的 L5 采样申报（无 prior 构建合法但须申报）；差异化主张表全文级核对；judge hook 用于外部系统时的 prompt 冻结。

**v2.1.1 勘误与 runner 交付（2026-07-16）**：runner 的 oracle 端到端 selftest 抓出生成器两个 bug——① L10 良性对照的修订 gap 取到 `LAYER_GAP["L10"]=None`，修订话语没进 pieces 而 gold 含新值（臂 A 14 夹结构性必挂）；② L1 inline 修订并入首句后又追加了一条独立修订句（语义重复）。两处已修，`GEN_VERSION=rb_v2.1.1`，**新官方构建目标 config_hash = `b30499ad9de7`**（n=1000、dev/test 103/897、修订占比 0.673 不变）；**旧构建 586da9e6a8c4 作废，须在合成任何音频之前重建**（尚未合成音频，零浪费）。同批交付 `scripts/rb_run.py`（臂 A 固定时间轴 + 臂 B 反应式 co-sim runner，core 语义与 w2r 驱动器位同源：同 WindowLedger/屏障/apply 路径/prompt v3.1/决策缓存/EoU 规则；RB 工具目录换装 + κ 注入；**沙盒 id 铸造改为按工具名内序**〔patch 重排提交序不再移位 id，本轮修复〕；`$RESULT_<op_id>` 提交期解析；OpDag 默认 on 可消融；oracle 决策器 = 零 LLM 的 gold-policy 健全性臂；FC 五档在 runner 内挂点；selftest 10/10 含 L4 patch 救援/L10 双向/臂 B 确定性/blocking 单决策）。live full-engine + websocket 确认批仍归 GPU 日。`rb/` 包（registry 4 域 28 工具 12 场景蓝本 / grammar 层定义+事件语法+双语模板 / sandbox 确定性沙盒 / generator 网格生成 / audio TTS 接口+时间轴装配 / simulator 臂 B 反应式用户 / scorer 三轨）+ `scripts/rb_build.py`（--selftest 12/12、千级干构建实测、--verify 决定性复核）+ `scripts/rb_golden_manifest.py`（144 条录制清单生成器）+ `tests/test_w5.py`（11/11）。**用户接口**（留待接线）：`rb/audio.py::QwenTTSBackend.synthesize()` + `VOICE_MAP`（9 音色映射）；`content_hook`（LLM 槽位内容，可选——默认模板全确定性）；`rb/scorer.py::llm_judge`（外部系统承诺判官，可选）。**Phase-2 件（下一轮）**：臂 A runner（RB episode → 引擎 core/full 驱动 + 沙盒对接）、臂 B live 驱动（simulator 动作 → 注入音频）、金子集录制执行。

## §12 判分器冻结（2026-07-16，§8-2 执行）

**触发**：oracle by_layer 门第一轮未过（A 的 L3 0.9231 / L4 0.8750、B 的 L4 0.8333）——根因 = OracleDecider 的 slot→(step,arg) 单值映射被共享槽覆盖（fin_transfer 的 `amount` 同时喂 get_fx_quote 与 transfer_funds，只 patch 了后者）。修复 = **multi-map（修订 patch 所有使用该槽的 pending op）** + 两条回归 selftest（A_L4 finance / B_L4_0005）。修复后本容器全量重验（同 config `b30499ad9de7` 的确定性重建，text 口径）：**A dev n=59 exact 0.7458，L1/L2/L3/L4/L7/L9/L10 全 1.0**；**B dev n=44 exact 0.7500，L4/L8/L9/L10 全 1.0**；失分仅剩窗口化 oracle 天花板层（A：L5 0.2143 / L6 0.25 / L8 0.6667；B：L5 0.1818 / L6 0.0）——**gate PASS**。

**冻结**：`exp/rb/scorer_freeze.json` 记录 `rb/scorer.py`、`rb/sandbox.py`、`rb/registry.py` 三文件 sha256 + gate 收据；`rb_run.py` 在 **test split 运行时强制校验哈希**（不匹配硬失败）——§8-2"判分器先冻结后单发"从此机械执行。此后任何判分器改动 = 版本递增 + 公开勘误页，不得改-重跑。

**窗口化 oracle 天花板读数（RB 版 oracle 前沿的第一笔）**：L5/L6 的失分 = 修订落在 δ=1.5 窗外（gap 上界 4.0s）+ L6 二次修订，L8-A 的失分 = 名义投影时机落后提交点——这些是**机制代价而非 harness 缺陷**，是 RB 要测的东西本身；oracle 行将作为各系统臂的上界参照行进入主表。

## §13 rb_v2.2：dev LLM 冒烟揭示的构建有效性修正（2026-07-16；判分器冻结 v2）

**触发**：判分器冻结 v1 后的首轮 LLM dev 冒烟（音频口径，177/177 HTTP 200、0 解析失败）双臂近零分（TACT 0.0508 / blocking 0.0339）。逐夹解剖决策缓存原始输出定死三类死因，**全部为 bench 侧缺陷、模型行为正确**：① 61/118 决策把听到的值规范化为 API 形态（"八百"→`800`、"dollars"→`"USD"`、"两居"→`2`）而 gold 存口语字符串——不可匹配是构造性的；② 35 处 `$RESULT` 引用全部猜测结果 schema（`$RESULT_0.trains[0].train_id`）且用批内 0 基索引，目录未声明返回结构、解析器只认全局 op_id ⇒ 链式全灭；③ 良性 L10 对照 gap 抽 L5 bin 可超窗——对照分不清"SV 误杀"与"窗口损失"。

**修正（rb_v2.2，config_hash `9f8f4ae9edc8`）**：① 注册表新增 `CANON`（口语→规范值映射）与 `ARG_FORMAT`（目录逐参数格式标注：integer / ISO code / 日期例）；话语保持口语形态、**gold 用目录声明的规范形态**；`canonical_calls` 标量按字符串比较（int 800 == "800"）；② 目录追加结果 schema 声明（每工具返回 `{"id"}`）与 `$RESULT_<n>.id` 约定（n = PENDING OPS 所示 op_id 或本次 ops 列表 0 基位置）；解析器支持批内引用 + 字段路径宽容（单值结果任意猜测路径均解析到 id；不可解析仍按字面提交=真实系统失败被判分）；③ 良性 L10 gap 改抽 L4 bin（构造性可救）。oracle 决策器同步用规范值。

**重验（本容器，text 口径 dev）**：A n=59 exact **0.7627**，门层 L1/L2/L3/L4/L7/L8/L9/L10 全 1.0；B n=44 exact **0.7500**，门层 L4/L8/L9/L10 全 1.0；天花板仅 L5/L6。**判分器冻结递增为 v2**（scorer_freeze.json 含勘误全文与新哈希；test split 运行时校验照旧）。**无任何 test 观测发生**——本修正完全落在 §8-3 允许的 dev 窗口内；此后再改 = 公开勘误页。

## §14 rb_v2.2.1：第二轮 dev 修正（2026-07-16；判分器冻结 v3）

dev22 LLM 冒烟（TACT 0.2542 / blocking 0.1695，层分化出现：L3 TACT 0.2308 vs blocking 0）后，用户 validity receipt 定位四项残留缺陷，全部 pre-test 修正：① **blocking 批内引用**——immediate commit 在 apply 期间发生而 batch_of 在 apply 后才登记，$RESULT_0 按字面提交；修复 = results_by_step 基址回退（本决策的提交按 launch 序落位）。② **RB 链零依赖边**——DAG_TEMPLATES 只有 FDB 工具；注入 5 条 RB 声明（hold_seat←search_trains 等），DAG 传播在 RB 链上首次真实生效（oracle L5 天花板 0.286→0.429 = 上游 patch 重启下游窗的合法救援）。③ 目录声明**字符串值用用户语言 verbatim**（成都≠Chengdu）+ 多词值完整（"first class"）+ seat_class 格式锚。④ en 词池去冠词（landlord/property office/university campus/tech park）——prompt 规则 14 去冠词，gold 不得带冠词。**rb_v2.2.1 config_hash `265c7cd8f485`**；oracle 门重验：A 0.7966 / B 0.7273，门层全 1.0。判分器冻结 **v3**（勘误全文入 scorer_freeze.json）。解析口径勘正入册：dev22 首试严格 JSON 94/118（TACT），salvage 23——**salvage 率是被测系统属性，照常判分**，先前"全部严格 JSON"的说法作废。

## §15 外部评审核验入账 + rb_v2.3.0（2026-07-16；用户裁定范围 = 评审 §五 items 1–5 + 归属门）

### 15.1 评审主张逐条核验（镜像容器，代码+归档双向；两处数字修正）

外部评审（2026-07-16，用户转交）的全部可数主张按"先验证后入账"处理，结论：**全部实证成立**，
其中两处数字修正：A_L8 实测对象为 57 个 test 夹（评审写 60 = 含 dev 的全量）；B-L6 dev 实为
**2** 夹（评审写 ≈4）。核验数字（工具 = 归档 `segs`/episodes/cues 重算）：

| # | 主张 | 实测 |
|---|---|---|
| 1 | 臂 B stub 排期 + 终点回填 ⇒ 自我重叠/分箱失效 | 重叠 **37/356**（L4 17/L6 10/L10 6/L5 4）；B_L4 gap −5.914~+3.387、in-bin **5/54**；B_L6 重叠 10/38 |
| 2 | 臂 B 修订双投放（脚本件+回声件） | **141/141** B_L4/L5/L6 test 行有注入回声段 |
| 3 | `trace_events[len(trace_events):]` 恒空 ⇒ committed 锚死 | B_L6 全部 38 行 = (3 kept, 2 events, 4 user segs)：第二事件 0 注入；伴生 `revisions[0]` 错引代码级确认 |
| 4 | 臂 A 投影双计 HOLD | 57/57 夹 placement−seq_end−declared ≡ **0.640**；在飞事件落入首工具名义执行窗 **0/57** |
| 5 | 链首步全 READ / 无 abort / 补偿零考核 | 四链首步全 READ；**从不进 gold 的 10/28 工具恰好全部是 reverse/撤销类**；gold 最深 3 调用；submit_application 不在任何场景；runner 不传 idem_key |
| 6 | 域×语言 100% 耦合 | ecommerce/housing 全 zh（251/249）、finance/travel 全 en（251/249） |
| 7 | blocking 单决策 + 无结果回流 | L3–L6 blocking 决策带 ops 直方图 {1:246, 0:3} |
| 8 | 内容表面积 | 唯一意图句 **296** / 唯一修订·事件句 **121** / 唯一语句 **417**；dev 意图句 **58/70=83%** 在 test 逐字复现 |
| 9 | 臂 A 无重叠（拼接器健康） | 0/600 ✓（短板确系臂 B 排期，不是 TTS/拼接） |
| 13 | dev 子格功效不足 | dev 每 (arm,layer) 多为个位数（B-L6=2、B-L10=2、A-L2=1） |

评审对已出结论的"安全/需勘误"二分照单采纳（勘误落 `rb_test_protocol.md` §八）。
评审 §五 item 6（外部系统接入面）按用户裁定**本轮不做**。

### 15.2 rb_v2.3.0 设计登记（GEN_VERSION `rb_v2.3.0`；本容器全套 selftest 34/34 + build 12/12 + W5 13/13）

**Item 1（零成本修复）**：① 臂 B 重分箱勘误表 = `exp/rb/errata_b_rebin_v221.json`（零重跑，
按 v1 §3 过采样-事后分箱原则：L4 in-bin 5/54、L5 27/49、L6 14/38；in-bin 子集两臂 exact 全 0 或
近 0——v2.2.1 的 B-L4/L5/L6 行标签不描述所考内容，正式作废）；② 语言改从 episode 哈希采样，
与域循环解耦（v2.3 manifest：每域两语 117–136）；③ `feed_sim` 水位线修复 + 末段收尾回灌
（committed 锚全链路激活）；④ 臂 B 事件文本按规则 content-kind 选修订（`revisions[0]` 错引修复）；
⑤ `at_after_eou` 改为真 EoU 相对（放置端只加一次 HOLD）——在飞事件按构造落入执行窗。

**Item 2（臂 B 调度器）**：先合成后排期（真实时长决定下一段起点 = 臂 A 拼接器同构）；
L4/L5/L6/L7/L11-B 修订内容**只经事件投放**（`ARM_B_EVENT_ONLY`，双投放废止）；注入段
落位有物理单嘴保证（不与既有段重叠）；每行 `armb_timing` 收据（overlaps 必须 0 + 实测 gap 表）
+ report 聚合。

**Item 3（事务考场）**：① 在飞锚 = 首个**非 READ** launch（评审写 COMP/IRR；纳入 REV 的偏离
理由：三个域的 single 场景以 REV 收尾〔schedule_payment/add_item/save_listing〕，严格 COMP/IRR
会使其在飞事件全体落空——已记为对评审的有据偏离）；② 沙盒执行期状态（`execute(t=…)` 记
`completes_at`）+ `abort()` 原语（执行中 REV/COMP 可中止、零费——def2 的另一半；IRR 不可中止；
abort 不计 C_κ，时间价由 γ^done 承担）；③ **reverse 工具一等公民**：目录里的撤销工具命中在册
forward（minted id 或值匹配）即净额对消（forward 作废 + COMP 计费 + reverse 调用不入净额）——
外部系统只要会调目录工具就能被补偿轨计分；④ **L7 重定义 = 补偿考场**：single 场景 + gap 按构造
落在参考提交视界之后（hold+infer+δ*+首工具 wall+0.3 margin），gold = forward(new) 净额——
补偿路（reverse+relaunch，计费）与长窗 patch 路（免费）净额相等，费/时差落事务轨与 U；oracle
决策器新增补偿路（目标 op 已提交 ⇒ reverse+带新值 relaunch）；⑤ runner 传 idem_key；
⑥ 时延 rng 改 (episode,fn,occurrence) 键控（mint_id 同课：交错免疫）。

**Item 4（L11 = TTS 打断改口层）**：臂 B、tts 锚 + (0.05,0.40)s 偏移的 revise 事件——用户在
agent 开口时插话改口 = 全双工修订标志格；屏障接力保证可救援（窗口过期落在修订决策 guard 内
⇒ 屏障延期 ⇒ patch 救回；blocking 结构性失败）。**L12 = 归属考场**：multi 场景 + 修订槽位约束
为 step-2-only（v2.2.1 反窗口夹机制的显式复刻）；配额重排（A：L3 72→60、L5 96→84、L9 72→60、
+L12 36；B：L4 60→50、L5 60→50、L8 120→100、L9 80→50、+L7 40、+L11 30；总量 600+400 不变）。

**Item 5（内容多样性 + judge 冻结）**：`scripts/rb_content_gen.py` = DeepSeek `deepseek-v4-flash`
（非思考 chat，api-docs.deepseek.com 口径，key 走 configs/eval.env）离线生成**冻结内容库**
`exp/rb/content_bank.json`（修订五构式 × 双语 + bystander/progress/意图逐场景 paraphrase +
五类不流利模板；占位符/字符集/长度验证器把关；bank 哈希进 config_hash ⇒ 构建仍逐位确定；
bank 缺席 = 模板回退，manifest 标注）；生成属一次性动作，重生成 = 新 bench 版本。
音频扰动族（rate 0.94–1.06 / gain −6~+2 dB / 场景 SNR {clean, 15–25 dB}）按 episode 播种、
合成后应用（TTS 缓存键不变）、臂 A/B 同一实现。判分器新增**冻结承诺 judge prompt**
（`COMMIT_JUDGE_PROMPT` + `make_llm_judge`）= 承诺-修复轨对自由文本外部系统可用；abort 不计
C_κ 的勘定同批。**scorer/sandbox/registry 三文件已变 ⇒ v2.2.1 冻结 v3 仍只对归档 test 生效；
v2.3 test 前按纪律另立冻结 v4。**

**归属门（W5 系统侧，非 bench）**：`PROMPT_RB_ATTR`（规则 16 修订目标绑定）+ `--attr on`
（默认 off = prompt 逐字节不变，selftest 钉死）；dev 迭代 ≤2 轮后冻结措辞。W6 缩编为 SV 门
单项记录（`docs/w6_admission_design.md`），当前仍在 W5。

**v2.3 未跑任何 test；dev/test 窗口纪律对 v2.3 重新起算（先 selftest/dev 有效性门，后冻结 v4，
后单发）。** 遗留台账（评审 §三，未在本轮范围）：感知层旁路（cue-based EoU）、结果无载荷/
无观察-行动循环、dev 子格功效、冻结哈希门扩到全链、外部系统接入面、口音/信道轴——
进 limitations 与 v2.4 候选。

## §16 v2.3 内容库、Qwen TTS 构建与 dev 首轮（2026-07-17）

完整机器收据：`exp/rb/build_v23/rb_dev23_receipt.json`。本节只登记已经实际运行并逐文件复核的
事实；**没有运行任何 test episode，也没有冻结 scorer v4**。

### 16.1 冻结内容库与最终 config

从 `origin/main` 快进到实现提交 `04ab6ff` 后，`rb_content_gen.py` 用
`deepseek-v4-flash` 完成 **52 次**离线调用（52 类 × 每类请求 6 条；不是按预估调用数凑次数）：
请求 312 条、返回 246 条、验证器接纳 244 条。39 类接纳 6 条、2 类接纳 5 条、11 类没有合格
新句而只保留原模板；共 296 个含原句的候选。逐类听写可用性审阅、占位符/语言/字符/全局重复/
密钥扫描均 0 问题，没有人工改句、补采或重试。冻结文件 SHA256 =
`309c7c0d03616d629a3e23752499fb576969c5b59c7cd44c66753748d1ff163f`，已单独提交
`3ebc337`。

评审实现时的 `3c36d0a9ae1b` 是 **bank=none 的代码 config**；设计本来就把 bank SHA 纳入
`config_hash`，所以冻结内容库后的正式构建 config 自动变为 **`e1a515c29b8a`**，不应强行保留
旧 hash。最终 manifest：1000 夹（A/B=600/400；dev/test=89/911）、修订率 .729、八个
域×语言格为 104–143/格、`ids_hash=acbb7984b03d`、`content_hash=2cc261e7e0fd`。

### 16.2 Qwen TTS 构建与运行中勘正

正式输出另置 `exp/rb/build_v23`，旧 `build_v2` 归档未改。Qwen3-TTS 首次正式构建命中
119 个已有键、新合成 995 个键；600/600 个 A 臂 WAV、1200 个 cue 齐全。WAV 全为
mono/PCM16/16 kHz，共 365,025,016 bytes，时长 min/p50/max =
11.050/18.152/50.449s，末 cue 后尾垫最短 5.9994s；九声音件数为
111/140/147/153/133/119/139/122/136。全量 cache-hit 重建后 manifest、1000 份 episode
与 600 个 WAV 逐字节相同。B 臂 oracle/dev 首轮按需补齐后，本地 TTS cache 为 3024 个
mono/PCM16/24 kHz WAV，0 临时文件、0 坏格式。

内容库改变了确定性抽样，因而暴露三处**测试/计分适配缺口**，均先修复并加回归再继续：

1. `rb_build` 的“新值进 gold”自测原先用原始字符串包含关系，`"two"` 对 canonical `2`
   假失败；改为 `canon_value` 后比较 canonical gold 参数，不改生成行为。
2. oracle 的 L8 cancel 只识别三个硬编码词组，漏掉冻结库中的合法改写；改为以 episode
   `l8_action` 为结构真值，旧词法仅作无标注 episode 的 fallback。
3. reverse/abort 后 sandbox 为费用审计保留 `void` 历史项，state scorer 却仍把它当活状态；
   state 轨现与 `net_calls()` 一样排除 `void`，并在 reverse 自测中锁定。

### 16.3 dev oracle 有效性门

音频口径、TACT δ=1.5、屏障 on。A 臂 54 夹：exact/state = **.8704/.8704**，除明确的
L5/L6 设计天花板（7/13、3/4）外，L1/L2/L3/L4/L7/L8/L9/L10/L12 全过。尤其：

- **L7 补偿路 4/4**：四夹均走“旧 forward 已提交 → reverse → 带新值 relaunch”，
  fee=[1,0,0,1]、`comp_cost`=[4,2,2,4]，净额 exact/state 全过。
- **L12 新归属层 3/3**。

B 臂 35 夹：exact/state = **.8286/.8286**，L4 4/4、L9 4/4、L10 2/2、**L11 新 TTS
打断层 3/3**，全部 user speech 物理重叠为 0。L5/L6 是预期天花板 4/8、1/2；L8 为
11/12。唯一 `B_L8_0057` 不是构建或排期故障：修订语音在首个事务 launch 后 1.557s 才开口，
比固定 δ=1.5 晚 57ms；transfer 被补偿重发，但旧 READ `get_fx_quote(amount=800)` 无 reverse，
故 exact 留下一条旧读调用。这是 v2.3 修正时序后真实显出的固定窗边界。

B-dev 没有 L7，故没有偷看 test 来凑数；补偿门由 A-dev 的 4 夹承担。有效性结论是
**新结构门 L7/L11/L12 全过，另保留一条可解释的固定窗天花板**，不是把 oracle 总分强行调成 1。

### 16.4 attr 门第一轮（冻结原词条，不回调）

本地 Qwen3-Omni 音频栈 `T=0/seed=42/max_num_seqs=1` 严格串行跑四个独立 provider；首轮
450 次决策均为 live cache miss、HTTP 200，0 retry/fallback：

| arm | baseline exact | attr-v1 exact | 配对净值 | 靶层/护栏 |
|---|---:|---:|---:|---|
| A（n=54） | 7/54 = .1296 | 5/54 = .0926 | −2 | L12 3/3→3/3；L3+L4 1/17→0/17 |
| B（n=35） | 5/35 = .1429 | 7/35 = .2000 | +2 | L11 1/3→1/3；增益落 L8 1/12→3/12 |
| 合并（n=89） | 12/89 = .1348 | 12/89 = .1348 | 0 | normalized state 同为 .1461 |

A 无 attr-only，通过丢失为 `A_L1_0010`、`A_L4_0055`。前者把 literal `LST12` 幻觉成
`$RESULT_0.listing_id`；后者把 `seat_class` 错绑到 `search_trains` 并保留旧的
`hold_seat` 值，恰是规则 16 本应阻止的护栏反例。B 的 attr-only 为 `B_L8_0029`、
`B_L8_0067`：前者只是把多词实体 `rent account` 从 baseline 的 `rent` 截断中救回，
不属于归属靶向收复；后者完整 launch 后又把三步全部 cancel，得到正确空净额。

所以首轮裁决是 **FAIL（靶层 0 增益 + A-L4 护栏损失；合并 exact 零和）**。按用户要求只跑
第一轮，本批没有改 `PROMPT_RB_ATTR`、没有运行第二轮；是否使用“最多两轮”的剩余额度由后续
裁定，不从本轮 dev 结果偷偷回调。

### 16.5 机器收官与纪律

Omni/代理/TTS 停服后确认 :10003/:10004/:8091 全关闭、GPU 0 残留；在无服务状态复播四臂，
分别 136/134/89/91 cache hit、全部 0 miss。四份决策 cache、四份 report 与
54+54+35+35 份逐夹结果在 live 前后逐字节一致。oracle 两份 report/逐夹目录也已记 SHA256。
`build_v23/audio/` 与密钥文件不入 git；**v2.3 test 仍为零次，scorer v4 仍待正式冻结**。

仓库完整性另记一条维护信息：所有可达 refs 的 missing object 为 0，针对 HEAD 与
`origin/main` 的 scoped fsck 通过；但 generic auto-gc/full fsck 仍会报告历史不可达 dangling
commit `3a1cb90…` 缺父 `d215c907…`。它不在当前可达历史中，本轮没有借实验之名改写/清理历史。

### 16.6 attr 门判定（判读层，2026-07-17）：第一轮 FAIL 成立，**封盘于一轮，剩余额度不动用**

镜像容器对四臂 178 份逐夹行独立重算：headline（7/5/5/7）、配对集
（A baseline-only = {A_L1_0010, A_L4_0055}、attr-only=∅；B attr-only =
{B_L8_0029, B_L8_0067}）、L12 3/3↔3/3、L3+L4 护栏 1/17→0/17——**与收据逐位一致**；
决策骨架扰动按 (type,fn) 粗口径复算为 ≥27/54（收据 34/54 为含参数口径，同向）。
`A_L4_0055` 逐轨迹复核把机制钉得比收据更干净：

- baseline：dec0 一次性 launch 全链（search+hold+purchase），dec1 把 seat_class
  **正确 patch 到 hold_seat（op 3）**——**没有规则时绑定就是对的**；
- attr：dec0 只 launch search（launch 节奏被规则改成增量式），dec1 把 seat_class
  **patch 进 search_trains（op 2）**——在规则明文禁止的方向上违反规则本身。

**结构性死因三联**（这不是措辞可修的形态）：① **靶层惰性**——L12 三夹决策骨架
0/3 改变，规则根本没有接通到它要治理的绑定行为；② **靶维反作用**——A_L4_0055 在
规则约束下反而产生规则禁止的错误绑定；③ **弥散副作用**——54 夹中 ≥27 夹决策骨架
被改（launch 节奏、引用形态 `$RESULT_0.listing_id` 幻觉），仅有的 +2 增益全部是
非归属性的偶然收益（多词实体保真、cancel 收敛）。三联合起来 = 词条到达不了绑定
机能、却到达了它不该动的一切——与 FDB 五靶 v3.1 的"词条工程刀口零和"同构，
且更强（那次至少靶层有翻转）。

**裁决**：attr 提示词门**封盘于第一轮**。冻结额度允许第二轮，但动用它需要一个
"新措辞为何能改变上述三联"的可信机制假说——本判读认为不存在：dev 靶层（L12
n=3）在天花板上（按 v2.2.1 实测 ~14% 误绑率，3/3 全过的概率 ≈64%，不构成"问题
已消失"的证据），第二轮在结果面上结构性无法显示靶向增益，只能显示"无害"，而
无害且无收益的规则不值得占用 test 单发面。若用户裁定仍要花掉第二轮，本判读
如实执行但预测记录在此：护栏中性可达、靶向增益不可测。

**WHICH 轴的正确机制类（记录，不在本轮建造）**：① 引擎级 patch 模式校验——
拒绝 diff 字段不属于目标 op 参数表的 patch（A_L4_0055 形态可机械拦截；v2.2.1
反窗口夹的"合法字段错语义"形态拦不住，覆盖为部分）；② 决策格式升级——decider
对每个修订显式输出目标动作槽（结构化绑定，非词条）。两者皆系统侧机制变更，
与 SV 门同属准入控制家族（W6 记录）。**L12 作为测量层保留**：v2.3 test 上
n=36 的基线误绑率 + 逐轨迹绑定审计 = 论文对该现象的定量刻画，机制修复
如实标注为 open problem。

### 16.7 admission v1 = patch 模式门（2026-07-17；用户裁定"方法必须应对 L12，不留 open problem"）

**先测量后建造**。全档案法医普查（归档 patch 自带 diff 字段，零重跑）：

- **非法字段 patch**（diff 键不在目标 op 参数表，如 `{seat_class}`→`search_trains`）：
  v2.2.1 test 全 provider **496/5,097 = 9.7%**；dev23 四臂 12/136 = 8.8%。主臂 25 夹
  含之，**0/25 通过**，层分布 L4 9/L10 5/L3 5/L7 2/L8 2/L6 2；重定向候选仅存在 7/25。
- **合法字段错语义**（`{destination: Thailand}`——字段合法、值类错）：六个历史反窗口
  夹中占 **4/6**（destination×4；item_id 非法形态 ×2）。此半边是模型语义问题，任何
  机械门拦不住——**保持 L12 测量层刻画，如实为 open half**。

**机制（`src/admission.py`，纯函数、harness 无关；runner `--admission schema`，默认
off 逐字节不变）**：拒绝 diff 中不属于目标 op 参数表的键（整条皆非法则丢弃整个 patch），
审计事件记录 rejected/kept/dropped 与重定向候选。**v1 只拒绝不重定向**——patch 值可能
本身是旧值（A_L4_0055-attr 的 diff 值就是旧值 "business class"），把它转投到正确 op
可能污染本来正确的参数；重定向只记数，作 v2 证据基础。

**可证明无害性**：gold 参数恰为工具声明参数集 ⇒ 提交调用里的非法键必然造成 canonical
失配；剥掉它只可能消除失配材料，exact/state 两轨的距离都不增。经验侧证 = 25/25 含非法
patch 的夹全是失败夹。selftest 五项：单元门（非法剥离/全非法丢弃/合法逐字节不动/未知
目标不碰/审计计数）、junk-arg-only 失败翻转（off False → on True）、审计键只在 flag 下
出现、默认路径双跑逐位、oracle 合法 patch 轨迹逐字节不变；**runner selftest 34→39**。

**与 attr 词条门的关系**：词条门死于"指令到达不了绑定机能"（§16.6）；模式门不经过
LLM——它在引擎侧把"永不可能正确"的那 9.7% 机械清除。二者合并后的论文表述：
misbinding 的机械可检半边由准入控制关闭（收据 = 本节），语义半边由 L12 定量刻画
（test n=36 基线误绑率），修复需要值域理解 = 前瞻。

**【勘误，2026-07-17，test-911 判读后（§十 R-ADM1）】本节"可证明无害"的主张对
所交付实现不成立，admission v1 臂已作废退役。** 死因 = id 命名空间分层错误：门在
决策**原始层**运行，把模型回显的**快照局部 id** 直查全局 `tx.pending`；而引擎的
`resolve_ref` 经 `tx._localmap` 把局部 id 翻译为真实 op（W2"快照 op_id 必须局部
编号"教义的另一半）。铁证 `A_L12_0001`：模型 `patch op_id=2 {threshold}` 在局部
编号下 = set_alert（**绑定完全正确**），引擎解析为真实 op 3 正确应用（主臂 PASS）；
门误读为 "threshold→get_balance 非法" 杀掉正确 patch（adm 臂 FAIL）。test-911 上
臂 A 因此损失 10 夹（另三夹是 `{'args':{…}}` 嵌套 wire 形态未建模）。**定理本身
（非法键必然 canonical 失配）在"解析后目标"上依然成立**——本节普查的 9.7%/9.9%
是解析后行层统计，人口真实（R-ADM3 复现 27/273）；错的是把解析前的流当成了解析后
的流。附带正面发现：裸决策层看似"非法字段"的形态多数是正确的局部 id 引用——
**真实误绑率低于裸层观感，引擎的局部 id 解析一直在静默纠正一部分归属歧义**。
v1.1 候选规格（未建造，须新系统版本单发）：门移到 `resolve_ref` 之后 + wire 形态
归一后；预期增益按 R-ADM2 精神标低（解析后非法人口 0/25 通过 = 多为共因失败）。

### 16.8 admission v1.1 = 解析后置模式门（2026-07-17；v1 勘误的机制修正）

同一条拒绝规则，搬到它被证明成立的那一层。与 v1 的全部差异（每条都镜像引擎语义）：

| | v1（已退役） | v1.1 |
|---|---|---|
| 运行层 | 决策原始层：拿模型回显的**快照局部 id** 直查全局 `tx.pending` | **解析后层**：用引擎自己的 `resolve_ref`（含 `_localmap` 局部→真实翻译）先定目标，再查模式 |
| wire 形态 | 不认识 `{"args": {...}}` 嵌套（当作非法键 'args' 整条杀） | 逐字节镜像引擎的解包规则（单键 args-dict → 解包后检查） |
| 解析失败/stale | 也可能误判 | **一律原样放行**（引擎对这些的丢弃行为与门无关，轨迹等价） |
| 无害性主张 | "可证明无害"（前提错误，已勘误） | **不做定理级主张**：剥离键仅限"引擎将以垃圾参数形式应用到已解析目标上"的键（提交调用必然 canonical 失配）；二阶轨迹效应（快照文本、dedup 交互）交给经验门——dev 零损失在前、test R-ADM1' 零损失硬门在后 |

实现 `src/admission.py::admit_decision_ops_v11` + runner `--admission schema11`
（`schema` 保留 = 仅供已退役 adm 臂的归档复放）。selftest 39→45，其中
`adm11_localmap_correct_patch_untouched` 显式复刻杀死 v1 的那个几何（per-episode
计数下真实 id 2/3 × 局部 id 1/2 的碰撞）：v1 在同一输入上误拒、v1.1 原样放行——
对照断言写死在测试里。预期收益维持 R-ADM2 口径的诚实低标（解析后非法 patch 人口
9.9%，所在夹多为共因失败）；核心价值 = 关闭 misbinding 的机械半边 + 审计轨 +
v1 教训的公开修正。运行协议 = `rb_test_protocol.md` §10.5（dev 零损失冒烟在前，
test 单发在后）。

运行已按该顺序收官，正式判定见 `rb_test_protocol.md` §10.6、机器收据
`exp/rb/build_v23/rb_test_receipt_v23_adm11.json`：dev A/B 与 test A/B 的
baseline/main-only loss 均为 0，test 两臂 exact/state/U 逐位不变，增益均为 0；
解析后实际拒绝 A/B = 27/14 个 patch，v1 假阳性对照 = **52/24 个 patch**。因此
v1.1 的经验安全门兑现，低增益预期也兑现；机制贡献按“关闭非法字段形态 + 审计”，
不按 accuracy gain 表述。

## §17 rb_v2.4.0 = 论文主线版本（2026-07-17；用户裁定"完全彻底实现 v2.4、作为主线非分支"，范围 = 判读侧五项建议 1–5 + 用户钉死的 L4 文本畸形修复；SV 门不动，仍属 W5）

> 实现批全部在判读容器完成并本地验证：runner selftest **62/62**、build selftest
> **16/16**、simulator 7/7、content-gen 5/5、commit-judge 4/4、tests/test_w5
> 13/13（其余失败均为容器缺 torch/sherpa/pytest 的存量环境噪声，干净树复验同样失败）。
> 全尺寸干构建 1364 夹双跑逐字节；bank-less oracle dev 冒烟 A **.9552** / B
> **.9737**，全部新层 oracle 天花板 = 1.0。正式 config_hash 待 v2.4 内容库
> 冻结后生成（bank hash 是 config 输入）。

### 17.0 触发问题清单（每项均已在归档上独立实证后才立项）

| # | 问题（v2.3 实证） | v2.4 机制 |
|---|---|---|
| 0 | **L4 修订文本系统性畸形**：`value_first` 模板双 `{new}`（grammar.py v2.3 :71/:76），DeepSeek 改写成对比构式后英文侧成真语义矛盾句；实测 **122/122** L4 夹带双值（A 72/72、B 50/50，比首报 121/122 还多 1；en 57 夹中 ~48 对比构式），test L4 = TACT 2/64 / blocking 0/64 / oracle 64/64（B 臂 1/46 / 0/46 / 46/46）——oracle 不读文本，oracle−LLM 差距在 L4 被构造伪影放大 | 模板改单 `{new}` 对比式 `"{new}，不是{old}。"` / `"{new} — not {old}."`；`revision_text` 增加 `old` 参数 + **单 new 运行时护栏**（bank 变体非恰一 `{new}` 即回退冻结模板）；验证器 `_placeholders` 改**多重集**比较 + `{new}`≤1 皮带；`rb_build.audit_templates()` 构建期硬门（冻结模板+全 bank 变体逐条审计，现 v2.3 bank 16 处违规被正确拒绝）；勘误 = `rb_test_protocol.md` §10.7 |
| 1 | WHO 轴无结论级数字（L10 双率格 n=12–19 < 30） | L10 配额 42/40 → **108/108**，dev 按构造格分层各抽 2 ⇒ test 每格恰 **34 ≥ 30**（实测六格全 34）；`rev_adopted`/`intruder_present` 升格为判分器行级一等字段（bystander 记录补 `slot`）；report 内置 `who_axis` |
| 2 | 生命周期因果主张缺配对设计（H-B1 死于无跨状态 pair id） | **L13 = 生命周期配对八元组**（臂 B 160 = 20 家族 × {user, bystander} × {eou, inflight, committed, tts}）：全部内容抽签走**家族 rng**（意图/槽位/修订/双声/文本/偏移固定消费序），8 格只差 who×state；家族共享时延命名空间（sandbox `lat_ns`）；`pair` 字段透传进行；dev/test **家族原子切分**；report 内置 `pair_axis` |
| 3 | 承诺-修复轨死仪器（wrong_commits 恒 0；冻结 judge 从未被消费——且实测 `.format` 遇字面 JSON 花括号必 KeyError = **潜伏死仪器**，无任何已判分结果受影响） | **L14 = 承诺考场**（臂 B 40）：`committed` 锚双事件 = 确认请求（+0.5–1.0s，新 `confirm_query` 动作 + `CONFIRM_QUERY` 类，词面刻意排除 cancel 词素）+ 迟修订（+5.0–7.0s）；judge 渲染改 `.replace`（渲染字节 = 设计意图）；新 `scripts/rb_commit_judge.py` 判官覆盖层（DeepSeek dv4-flash、T=0、512 tokens、5 重试硬失败、prompt 键控缓存、只写 overlay 不碰归档；脚本哈希入冻结 v5） |
| 4 | R6：L8 in-flight 不再区分系统、反应式主战场移至 L11（v2.3 n=27 过薄）；abort 通路存在但**决策层不可达**（全仓唯一调用点是 selftest） | **L8 减半 50 + L15 = 执行窗 abort 考场**（臂 B 50，heavy 时延、single 场景）：新 `executing` 锚（首个非 READ commit、frac-of-wall 落窗内、与 `committed` 同源事件双锚各触发一次）；**cancel→abort 映射**（runner 拦截 `op_id="X<gid>"` 的 cancel → `sandbox.abort`，成功零费；`_snapshot_v24` 给已执行 op 编 X 命名空间 id——与杀死 admission v1 的局部 id 几何**无碰撞**）；gold = forward(new) 净额**路由无关**（abort+relaunch 免费 / reverse+relaunch 计价，`abort_feasible` 构造标注，实测 46/50 可 abort）；oracle 单序列双分支路由（cancel-X + reverse + relaunch：abort 成则 reverse 无害报错，abort 败则 reverse 净额）；report 内置 `route_axis`；**L11 配额 30→60**（反应式主战场增厚，test n 27→54） |
| 5 | dev 功效不足（attr 门死因之一：靶层 L12 dev n=3 天花板） | **分层 dev 切分**（`_assign_splits` 构建后处理）：每 (arm,layer) 下限 6 / 率 8% / 上限半格；L10 按构造格各 2；L13 按家族 2 个整家族；实测 dev 143 / test 1221（A 67/599、B 76/622） |

### 17.1 版本边界与字节安全

- 全部 v2.4 决策语义由 episode `caps = {"abort_on_cancel": true, "snapshot": "v24"}` 门控；v2.3 构建（无 caps）在新 runner 下：cancel-X 不拦截（selftest `v24_abort_gated_off_without_caps`）、快照走类默认 `_snapshot_v2`（tact_core **零改动**，FDB 路径不受任何影响）、catalog 走 v2.3 字节（sha 钉死 `32e09323ed7e…`，selftest `v24_catalog_v23_frozen`；v2.4 catalog = v2.3 前缀 + abort/reverse/confirm 指引段）。`install_rb(v24=…)` 由 build manifest version 判定。
- scorer/sandbox 改动（行级新字段 `rev_adopted`/`intruder_present`/`aborted_ops`/`comp_calls`/`pair`、`lat_ns`、judge 渲染修复）⇒ **v2.3 归档行的字节复放义务终止于 freeze v4 时代**（同 v2.2.1→v2.3 先例："v2.2.1 归档不动"）；v2.3 判定全部由已入册收据背书，不受影响。v2.4 test 前另立 **freeze v5**（scorer/sandbox/registry 三文件 + `rb_commit_judge.py` + `COMMIT_JUDGE_PROMPT` 渲染路径）。
- 服务器侧 preflight 需补一次**缓存零 miss 平价探针**（本容器无音频无法执行）：复制 `decision_cache_rbdev23_tact_d150.json` 至探针 provider，对 build_v23 dev 复放，断言 0 miss 且 decisions 逐位同归档（证明 v2.3 prompt 字节在新代码下不变）。

### 17.2 def2 经济学的 v2.4 演化（诚实入册）

abort 一等公民后，commit-now 的下行不再是"必失或必付费"：执行窗未闭时 oracle 可走
abort+relaunch **免费且更快**（实测 L4 夹 commit-now done 4.701 < fixed 5.397——省掉
了 δ 等待；selftest `lh_commit_now_abort_escape` 显式断言该机制分裂）。窗口的不可替
代价值收缩到：**IRR op**（不可 abort 不可 reverse）、**短执行窗**（abort 物理不可
达）、**WHO 安全**（提前提交不改变旁观者拒绝问题）与**内容盲**（窗口不需要知道修订
绑定哪个 op；abort/reverse 路由要求归属能力——恰是 L12/H-COMP2 实测的 LLM 能力缺
口）。这不是机制削弱，是把"窗口 vs 补偿 vs abort"三路经济学摆上同一张桌子——论文的
def2 章从二元换算表升级为三路由前沿；LLM 主臂上 abort 使用率（H-ABORT2）按 L7
reverse 先例诚实预期 ≈ 0。

**完成锚计价规则（审查修正后冻结）**：commit 的完成贡献 = 存活效果的
`t_commit + wall`；**被 abort 的效果止于 `aborted_at`**（初版按幻影 completes_at
计价，把 abort 的时间收益整个抹掉——审查抓出 B_L15_0046 done_s 63.14 vs 真值
13.09、U 偏差 13×，已修）；**执行报错的调用不封门**（贡献 = `t_commit`，错误尝试
不产生用户等待的结局效果——abort 成功分支里必然报错的 reverse 调用因此不再把
heavy 幻影墙记入 done）。被 reverse 净掉的效果仍计全墙（用户真实等过它）。

### 17.3 判读侧五项之外的裁剪（记录不做之理由）

- **臂 A bystander 因子化**：被 L13 八元组取代（臂 A 固定时间轴的生命周期投影已被
  v2.2.1 §八证明是弱仪器；WHO×状态因子在反应式臂上才是真锚）。
- **干净音频子臂**（解混内容难度与机制效应）：一次构建双倍音频合成，收益可由
  oracle-passable 分解（§10.1 口径）近似获得，裁剪；留 v2.5 候选。
- **金子集**：`rb_golden_manifest.py` 不改（L3/L4/L5/L10 × 2 lang × 18 = 144；
  L4 文本修复后录制脚本自然继承单 new 对比式）；录制目标切到 v2.4 构建，用户执行。

### 17.4 容器验证记录（2026-07-17）

- selftest 全家：runner **62/62**（新增 17 项 v24_*，含 v2.3 语义门 2 项与两处既有
  selftest 的机制性改写：`v23_bank_cancel_oracle` 改按构造选预提交可救援格〔原 idx
  抽签敏感〕、`lh_commit_now_pays_or_loses` → `lh_commit_now_abort_escape`〔§17.2〕）；
  build 16/16（新增 4）；simulator 7/7（新增 2）；content-gen 5/5（新增 1）；
  commit-judge 4/4（新）；test_w5 13/13。
- 全尺寸干构建：1364 夹、dev/test 143/1221、双跑 `json.dumps` 逐字节同、config
  确定性；L10 test 六格全 34；L13 20 家族原子、dev 2 家族；L15 可 abort 46/50。
- oracle dev 冒烟（bank-less、text，审查修复后复跑）：A .9552（L5 .7143/L6 .8333
  为已知链/双修订时序天花板，其余全层 1.0）、B .9737（L5 .8333/L8 .8333 天花板，
  **L13 八格、L14、L15 全 1.0**）。L15 全体 50 夹 oracle 探针：**exact 50/50、abort
  实测 49/50**（`abort_feasible` 是保守单侧下界、只标 46——门以 `aborted ≥ feasible`
  判）；最长 done_s 63.14 系 relaunch 抽中 60s 封顶墙的合法 heavy 长尾。链式 reverse
  路由中出现机会性 abort（A 臂 1 次）= abort 语义全域可用的自然结果，路由轨照实计。
  who/pair/route 三轴 report 内置块全部就位（who_axis 对抗格 = command-only，
  irrelevant 独立对照格）。

**对抗审查收口（2026-07-17；5 维审查 + 逐发现证伪）**：7 项实锤全部修复——①
done_s abort 计价（上文）；② **臂 B L10 良性格双投放**（脚本 piece + benign_control
事件 = v2.2 双投放类在 L10 的漏网；v2.3 归档 **13/13** 良性格中招〔test 12+dev 1〕，
v2.2.1 同形；v2.4 改事件单投放，勘误挂 §10.7）；③ **L13 eou 偏移不可投放**（反应式
事件在决策后才投放，偏移 <1.0s 全被钳到 t_dec——12/20 家族名义偏移失真；eou 箱改
(1.00,1.95) 使采样=投放；legacy 臂 B eou 箱按 §八 re-binning 纪律保持名义值+实测
gap 双报）；④ who_axis 对抗分母混入 irrelevant 格（结构性零稀释一半，改
command-only + 独立对照格，scorer 行加 `bystander_kind`）；⑤ `rev_adopted`
槽盲（值落错 op/字段也计 adopted——L12 病灶反被记成收养；改**槽键控匹配**）；⑥
承诺轨只匹配口语形态（catalog 命令模型说数字/ISO 码——canonical 承诺全漏；
`episode_claim_forms` 双形态 + ≥2 字符护栏防 qty "1" 子串误报，scorer 与 judge
overlay 单源）；⑦ 杂项（split size-1 组、config_hash 补 REV_UTT/CONFIRM 文本、
manifest 版本比较改数值元组、三处 § 交叉引用）。**审查亦证伪 1 项**（L13 八格
perturb 不同不损配对——R-PAIR1 配对在同夹跨系统，扰动逐位相同）。

## 附录 A：v1 原案（2026-07-06，逐字保留；§5 已被 v2 §8-5 作废条款取代）

# Revision-Bench (RB) 设计文档 v1 — W3 定稿（录音执行在 W4）

> 依据：06 教义二/教义四 + W3 D1 实测（ε 带、屏障、静默预算定律）。W3 只定设计，不录音。

## 1. 规模与功效（教义四-iii）

- **规模 60–80 个修订场景**（蓝图 ~40 上调）。功效算术：McNemar 配对设计下要让当前效应量
  （rollback 实测翻转率 ≈4/17≈24%）显著，不和谐对需 ~20+ ⇒ n ≈ 20/0.24 ≈ **84 为稳妥上端**；
  60 为下限（单侧 α=0.05、power ~0.7）。**按 80 规划，预算不足再裁**。
- 全部场景双臂录制口径不变（同文本、同说话人、同信道），T=0 确定性系统里每次翻转都是可命名
  事件——统计主张永远配机制台账（教义四-i）。

## 2. 分层（按修订时机的静默钟坐标；post-EoU 总占比 ≥50%）

| 层 | 定义（gap = 修订段与前段的静默间隙） | 占比 | 动机 |
|---|---|---|---|
| L1 同段修订 | 修订与意图同 VAD 段 | ~20% | 对照层（hold 天然合并；官方数据主体 14/17） |
| L2 sub-hold 跨段 | gap < 0.64 | ~10% | hold 层验证 |
| L3 **ε 带**（新） | gap ∈ [0.64, 0.80] | ~15% | **EoU 存在性翻转带**（D1 full 实测 ε≈0.03–0.13，eco19/hou25/hou17b 全在 0.67–0.71）；live/离线分歧的富矿 |
| L4 **决策在途竞态区**（06 指定） | 修订落在 (t_eou, t_eou+决策时延]，即 gap ∈ (0.64, 0.64+~0.5] 且修订**开口即含新值** | ~15% | **专测提交屏障**：这正是 deferral 发生区（D1 实测 deferred_s ≤0.64）；无屏障系统在此区丢修订 |
| L5 post-EoU 长间隙 | gap ∈ (1.0, 4.0]，含 fin12b(1.12)/travel_10(3.91) 谱系 | ~25% | δ 网格的阶梯区；hazard 头正例主粮（官方数据仅 ~3-4 例） |
| L6 **多重跨 EoU 修订**（06 指定） | 单场景 ≥2 处修订、至少 1 处跨 EoU | ~10% | travel_10 型单 cue 盲区；测 cancel-on-announce 规则与多轮 patch 链 |
| L7 **链式修订**（06 D7 指定） | ≥2 个无依赖调用、修订命中其一 | ~5% | DAG 传播/重参数化专测；FDB 有 28 个链式场景但**零链式修订覆盖**（W3 普查） |

L3–L5 合计 ≥55% ⇒ post-EoU 分层 ≥50% 达标。

## 3. 录音脚本落点控制（竞态区/ε 带怎么录进目标区间）

修订时机无法靠说话人念秒表——**过采样+事后分箱**：每个 L3/L4 脚本录 3 个 take，指导语只给
粗档（"停顿约一秒后立刻改口"/"话音一落马上改"）；离线 VAD 量出真实 gap 后按 §2 区间归箱，
缺箱补录。L4 额外要求：修订**首词即含新值**（"actually SEVEN pm"，而非 "wait, hmm, ..."），
否则落点即使命中竞态区、值也在窗外（travel_10 教训：宣告与内容分离时 cancel 才是对的动作）。
每 take 交付：wav + 逐词时间戳 + 修订 cue 标注（多 cue 全登记——delta_hist 单 cue 盲区教训）。

## 4. 判分与基线

官方 exact 判分器一字不动；状态轨双报（verbatim + normalized，裁断 C）；每场景配 blocking 双臂。
预注册（沿用 G2' 框架）：TACT@δ* 对 blocking 的翻转集合按 §2 层内报告；L4 层追加屏障消融
（commit_barrier off 应在 L4 显著劣化——这是屏障价值的 held-out 复验，D1 只有 n=2）。

## 5. 与学习线的接口（W4–W5）〔**已作废**，见 v2 §8-5：RB=评测 only，标签不进训练〕

hazard 头训练正例 = L3–L6（≥65%）；oracle 前沿（W3 实测：固定 123.8s / oracle 20.9s /
回收上限 83.1%，`exp/w3/oracle_frontier.json`）给出 G2' 判据 (ii) 的靶：回收 ≥50% ⇒
自适应臂相对固定臂省 ≥51.5s/百场景。λ(t) v2（静默钟）在 RB 上重估。
