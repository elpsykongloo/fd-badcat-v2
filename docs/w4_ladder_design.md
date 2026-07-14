# W4 自适应阶梯 rungs 2–3：κ 规则臂 + prompted-finality 臂（预注册，2026-07-09）

> 09 §三的先行基线批。**表格与判读门在跑前冻结**；跑后不得改表重跑——新表 = 新预注册 rung。
> 学习臂（rung 4，停时头）必须逐级击穿本文两臂才有立足点。

## 1. 阶梯与每级回答的问题

| rung | 臂 | 提供 | 回答 |
|---|---|---|---|
| 1 | fixed δ*=1.5（已有 `w3p31_tact_d150`） | 基线 | 固定窗的帕累托墙 |
| 2 | κ 规则 `kappa:{v0,safe,rev}` | C_κ 侧 | 只按可逆性分级能回收多少保费（P2 预测 iii 第一测量） |
| 3 | prompted-finality `prompted:v0` | λ̂ 零 shot | 音频原生"说完了吗"判断加上去还能回收多少（P2 λ(t) 的免训练估计） |
| 4 | 学习停时头（W4 D5+，另文） | 校准 λ̂ | 学习增量 = G2 |

P2 阈值规则 t\* = inf{t: λ(t)·C_κ ≤ c_w} 有两个变量：rung 2 只用 C_κ，rung 3 加 zero-shot λ̂，rung 4 换校准 λ̂——阶梯即理论结构。

## 2. 预注册表（`src/delta_policy.py`，勿改）

κ 源 = `tact.tools.REVERSIBILITY`（与 `apply_decision_ops` 同符号；FDB v3 12/12 工具全映射，READ 占 62% 调用——规则臂的杠杆所在）。

```
KAPPA v0   READ 0.64  REV 1.0   COMP 1.5  IRR 2.0    # 激进读
KAPPA safe READ 1.0   REV 1.5   COMP 2.0  IRR 2.0    # 保守下界
KAPPA rev  READ 2.0   REV 1.5   COMP 1.0  IRR 0.64   # 反单调对照（控制臂）
FINALITY   final:      READ 0.0  REV 0.64  COMP 1.0  IRR 1.5
           hesitant:   READ 1.0  REV 1.5   COMP 1.5  IRR 2.0
           unfinished: READ 2.0  REV 2.0   COMP 2.0  IRR 2.5
```

- `rev` 控制臂检验**对齐**而非水平（FDB 调用混合偏 READ ⇒ rev 的平均窗 1.66 > v0 的 0.93，均值不配平——文档内声明此 caveat）。若 rev 与 v0 在 exact×保费平面上不可区分 ⇒ κ 分级空洞，规则臂叙事作废（诚实 kill 判据）。
- `final/READ = 0.0` 是故意的激进角：判官若准，读操作在决策点即提交；判官若错，复活 travel_19 类 relaunch 死法——这正是被测的赌注。
- finality 判官：独立小调用（`FINALITY_PROMPT` 冻结，一词输出 final/hesitant/unfinished，解析失败回退 hesitant 并计数）；输入 = 当段音频尾 ≤8s（`FINALITY_TAIL_S`）；**不进 Phase-B 消息**（ops 决策函数与缓存键保持 v3.1）；其墙钟**不推进音频钟**（部署下与 0.64 hold 重叠），infer 分布照录作诚实位（p90 若 > 0.64 需在论文里报重叠不完全）。

## 3. 实现契约（审计线）

- 策略只改**每 op 的异议窗长度**：launch 开窗与 patch 重启窗都走 `delta_fn(fn)`（`tact_core.WindowLedger.open/restart` 的 `delta=` 覆写；默认 `None` = 冻结路径逐位不动，141 项测试全绿含 37 项屏障回归）。
- 轨迹经快照通道分叉是合法臂差（与固定 δ 网格各点间的差异同类）；决策缓存高命中（提交时刻分叉后的新快照才 miss）。
- 全部对比在**同一 regime 内**：text-only stage-0 栈 + `--workers 12` + nominal infer 1.0，对照 = 同 regime 的 `w3p31_tact_d150` / `w3p31_sblock`。首响与阶梯无关（ack 路不动），本批只读保费/exact。

## 4. 运行（用户，GPU 不需独占；估计每臂分钟级）

```bash
PY=/root/miniconda3/envs/fd-sds/bin/python
# 服务：text-only stage-0 栈（QWEN_MAX_MODEL_LEN=8192，同 7/09 网格日）
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4k0_tact --prompt v3.1 --delta-policy kappa:v0    --workers 12
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4ks_tact --prompt v3.1 --delta-policy kappa:safe  --workers 12
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4kr_tact --prompt v3.1 --delta-policy kappa:rev   --workers 12
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4pf_tact --prompt v3.1 --delta-policy prompted:v0 --workers 12
$PY scripts/w4_ladder_report.py --arms w4k0_tact w4ks_tact w4kr_tact w4pf_tact
```

（决策缓存共用 `exp/w2_rerun/decision_cache.json`；finality 缓存自动落 `exp/w4/finality_cache.json`；报告落 `exp/w4/ladder_v0.json`。）

## 5. 判读门与预期（跑前锁定）

**门（决定臂是否可用）**：
- G2'(iii)：臂 exact ≥ fixed − 1pt（0.650 → ≥0.640，同 regime 比较）。
- 保费必须低于 fixed（premium_sum < 109.6s；fixed 自检已出）。

**预期（点预测，不中如实报）**：
- `v0`：回收 fixed 保费的 35–55%；exact 风险集中在"读操作被修订"夹（travel_19 类）。
- `safe`：回收 15–30%；exact ≈ 持平。
- `rev`：exact 掉 ≥3pt 且保费不降（严格劣角）——κ 对齐有效性的证据；若它不劣，规则臂叙事作废。
- `prompted`：回收 ≥ v0 且 exact 损失更小（判官对 rollback 话语应显著偏 hesitant/unfinished——报告的 finality 分布直接检验）；若 prompted ≤ v0，λ̂ 零 shot 无增量，学习臂只需击穿 v0。

**P2(iii) 读数**：fixed 臂 realized commit delay 按 κ 无结构（自检：3.2–4.3s 平坦）；κ 臂应显现单调结构（报告 `delay_monotone_in_kappa`）。

## 6. 汇报清单（跑完发回）

1. 四臂各自的 `cache: X hits / Y misses` 与 `finality cache` 行、任何 WARNING/ERROR；
2. `w4_ladder_report.py` 的整表输出（exact/state/done50/prem_sum/recov_s/recov%/dExact + 每臂 windows/delay/monotone 行 + finality 分布行 + flips 行）；
3. `exp/w4/ladder_v0.json` 与 `exp/w4/finality_cache.json` 生成确认。

## 7. Round-1 判决（2026-07-09 跑后追加；零 shot 阶梯封盘）

**门：四臂全败双门**（用户跑数，text-only 84g 栈 + workers 12，cache 215/217/217/214 hits——近零 GPU 兑现）：

| 臂 | exact(Δ) | 保费 | 回收 | 门 |
|---|---|---|---|---|
| v0 | 0.550（**−10pt**）| 41.8s | 62.5% | exact ✗ |
| safe | 0.630（−2pt）| 77.0s | 30.0% | exact ✗ |
| rev | 0.650（±0）| **133.1s** | −21.7% | 保费 ✗ |
| prompted | 0.610（−4pt）| 20.2s | **84.0%** | exact ✗ |

**预测对账（诚实清单）**：v0 回收超预测（62.5 vs 35–55）但 exact 远超风险带；safe 回收 30% 恰在带顶 ✓ 但 exact 未持平（−2pt ✗）；rev 保费↑ ✓ 但 **exact 未掉**（预测 ≥3pt drop ✗——机制见下）；prompted 回收 ≫ v0 ✓ 但 **rollback 话语 73% 被标 final**（预测"偏 hesitant/unfinished" ✗）。kill 判据未触发（rev 与 v0 在两轴上截然可分 ⇒ κ 对齐有效），但 κ 单独不可用。

**机制（零 GPU 探针，逐夹收据）**：

1. **救援通道关闭**：fixed 赢下这些夹靠 `rescued_patch`（k0 的 10 个丢失夹里 8 个在 fixed 档 rescued=1）；短窗臂下修订到来时 op 已 EXECUTED，模型**大多根本不再发 patch**（丢失夹 patches=0——快照显示已执行，patch 无靶），偶发的 patch 成为 dropped（arm-wide patch_after_commit：k0=7 / pf=9 / fixed=3）。早提交不只是"补丁被丢弃"，是**关闭了修订这个动作类**。
2. **事后修订对韵律不可见**：FDB rollback 修订是**afterthought**——修订前话语韵律上完全收尾（说话人当时真心的），新话语才改口。故 pf 在 eco19/fin12/hou25 的修订前 EoU 全标 `final`（labels=['final','final']），这不是判官弱，是**信号不在尾韵律里**。结构性结论：**韵律终结性检测的是"话语说完"，不是"意图稳定"**。
3. rev 不掉 exact 的原因：修订几乎全打**中段的读操作**，写操作晚发（常在末 EoU 后再无修订）⇒ 短写窗几乎免费、长读窗恰好护住修订多发区。**修订发生率是位置/话语结构变量，不是 κ 变量**——κ 只管代价侧，这正是 P2 双变量结构的实证。
4. fin12(1)/hou17(1) 在 READ=1.0 仍死、1.5 活——修订静默间隙落在 (1.0,1.5]，与 D3 静默预算台账的 1.12s 阈值咬合。

**零 shot 前沿（学习臂的标高）**：fixed (0%,0) / safe (30%,−2pt) / prompted (84%,−4pt)；v0 被 prompted 支配。**rung 4 目标（重述）**：回收 ≥~47%（≈ frontier 记账 51.5s）且 exact ≥ −1pt——没有任何零 shot 臂到达的区域。特征结论：停时头输入**必须以对话状态/位置为主**（EoU 序数、pending/已执行结构、槽完整度、域先验），韵律降级为句内延续信号 + 投机门控用途。

**封盘**：零 shot 表不再迭代（新表=新预注册 rung，但信息收益已低）；直接进 rung 4。部署脚注：finality 调用 infer p50 0.672 > 0.64 hold（workers-12 名义 regime 墙钟，非权威；该调用与决策派发独立可并行）。

### 7.1 勘正（7/10，用户质询后补）：对固定 δ 前沿的公正对表

判读臂的正确参照不是"门"本身，而是**固定 δ 曲线自己的回收-精度前沿**（同记账零 GPU 补算）：

| 固定 δ | exact(Δ) | 保费 | 回收% |
|---|---|---|---|
| d000 | 0.530（−12）| −49.7s | 145% |
| d060 | 0.530（−12）| 8.4s | 92% |
| d100 | 0.620（−3）| 55.7s | 49% |
| d150 | 0.650（0）| 109.6s | 0% |
| d200 | 0.660（+1）| 154.5s | −41% |

叠上阶梯臂：**safe (30%,−2) 贴在固定线上**（内插 −2pt≈33%）——κ 分级在保守档位上等价于把固定 δ 调小到 ~1.15，无帕累托增量，这才是它"不行"的真正理由；**v0 (62.5%,−10) 在线下**（内插 −10pt≈82%）——被固定 δ 严格支配，死；**prompted (84%,−4) 在线上方 ~30 个百分点**（内插 −4pt≈54%）——**唯一击穿固定前沿的零 shot 臂**，是真实的强结果。其"不可采纳"仅指不能作为终局配置（−4pt 使对 sblock 总差距扩到 −6pt，砸 P1 平价主张；且损失集中在论文核心的修订夹=机制自噬），论文中应作为亮点行报告。

**韵律结论的勘正**：前文"韵律降级"过度。117 个 final 标签只造成 ~5 夹损失 ⇒ finality 是高精度的"可提前提交"负风险信号（84% 回收的主要来源），它只对 afterthought 类修订盲。P2(iv) 精化为：韵律终结性对**多数类**（无修订话语）授权提前 commit 成立；对少数类（改口）无信号。停时头设计修正：**保留 finality 作为特征**（可直接用 Omni 零 shot 标签作运行时特征=蒸馏即特征，无需韵律训练集），叠加对话状态/位置特征修盲区；数据方案相应精化 = 合成（结构/时序）+ Omni 零 shot 韵律特征 + 小样本真实校准（HumDial 许可待查），FDB/RB 仍全程隔离。

## 8. Rung 4：停时头 v0（预注册，2026-07-10；代码已交付）

**组件**（特征定义单源 `src/stophead.py` 的 `FEATS`，标注/训练/运行时共用，禁止漂移）：
`w4_synth_gen.py`（事件时间轴生成器，**无音频无 TTS**；语法常量冻结在文件头，config_hash 打印；修订间隙混合分布横跨 [0.3,4.0]s **不拟合 FDB**；afterthought 类保留 10% 质量=本轮盲区）→ `w4_hindsight_label.py`（闭式 w\*=gap+ε；hazard 目标 y(op,t)=1⟺gap∈(t,t+H]——头学"此刻风险"而非事后动作，hindsight 偏差不进目标）→ `w4_train_stophead.py`（numpy 逻辑回归、类均衡+先验校正、按对话切分 8/2、AUC+校准 bin；c_w 在**合成验证集**上按预注册代价 `Σw + 3.0·C_κ·miss` 扫描选定）→ `--delta-policy learned:v0`（`--stophead-model`；同 `--delta-policy` 机制/同打表脚本；**learned 同时跑 finality 调用作特征**——w4pf 的 finality_cache 全命中 ⇒ 近零 GPU）。

**预注册常量**：C_κ=(1,2,4,8)、T_GRID=0:0.25:3.0、W_CAP=2.5、EPS=0.05、MISS_PEN=3.0、CW_GRID 九点；生成器 N=8000/seed=42/语法表冻结（改=新 tag）。**泄漏防火墙**：训练与 c_w 选择只见合成数据；FDB/RB 只评测；FDB 间隙统计只作跑后覆盖检查。

**门与目标（跑前锁定）**：exact ≥ fixed−1pt（≥0.640）；目标区 = 回收 ≥47%（≈frontier 记账 51.5s）——零 shot 前沿 fixed(0,0)/safe(30,−2)/pf(84,−4) 之外的区域；同时对表 pf 作 context。**两条已知 sim-to-real 迁移风险（如实报）**：① finality 特征训练时按混淆表模拟、部署时是真 Omni 标签；② 状态特征训练时来自脚本化决策、部署时来自 Omni 实际决策（协变量偏移）。冒烟基线（N=300）：AUC 0.755、先验校正后校准逐 bin 对齐。

**运行**（前三步纯 CPU；第四步需 text 栈 + workers 12）：
```bash
$PY scripts/w4_synth_gen.py --n 8000 --tag v0
$PY scripts/w4_hindsight_label.py --tag v0
$PY scripts/w4_train_stophead.py --tag v0
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4lh_tact --prompt v3.1 \
    --delta-policy learned:v0 --workers 12
$PY scripts/w4_ladder_report.py --arms w4k0_tact w4ks_tact w4kr_tact w4pf_tact w4lh_tact
```
**汇报**：gen 的 config_hash/修订率/混淆表；label 的样本数/正例率；train 的 AUC/校准 bin/c_w 表与选点；FDB 的 cache 行（决策+finality 均应高命中）与打表整行（exact/回收/翻转/finality 分布）。

## 9. Rung 4 v0 全量结果（2026-07-14；预注册后只跑一次）

### 9.1 运行口径与产物完整性

- 代码锚：`e97dbbb`；开跑前工作树干净，GPU 为 RTX PRO 6000 Blackwell 96GB。
- 服务严格复用 7/09 的 text-only 84g 口径：`exp/w3/qwen3_omni_text_only_84g.yaml`、stage-0、`max_model_len=8192`、`max_num_batched_tokens=8192`、`max_num_seqs=1`、GPU utilization 0.78；误起的默认 8-seq 服务在零请求时即停掉，不进入实验。
- gen → label → train → FDB → ladder 严格按 §8 顺序，五条命令均 exit 0。FDB runner 100/100 夹完成，未打印 `WARNING`/`ERROR`；三次真实 cache miss 请求在 proxy/vLLM 两端均为 HTTP 200。结束后按 proxy → vLLM 顺序优雅停服，端口释放、GPU 回到 0 MiB。
- 完整性复核：`dialogues_v0.jsonl` 8000 行、`ops_v0.jsonl` 19531 行、hazard `X=(205242,18)` / `y=205242` / 6148 positives、模型 features/mean/std/weights 均 18 维、100 个 `result_w4lh_tact.json` 全部可解析。官方 `evaluate_pass_rate.py` 独立复算仍为 58/100。

### 9.2 合成与标注

- `config_hash=d48204c6582b`，seed 42，dialogues 8000，ops 19531。
- 修订 6315 / 19531 = **32.3%**。kinds（占修订）：slot_completion 2234（35.4%）、afterthought 1116（17.7%）、cutoff_continuation 1469（23.3%）、hesitant_revision 1096（17.4%）、upstream 400（6.3%）。
- domain：travel 2011、finance 2018、housing 1936、ecommerce 2035。
- style → observed finality 混淆计数（脚本 stdout 标题写作 `finality|style`，实际方向由生成器定义为 style → label）：

| style \\ finality | final | hesitant | unfinished | 合计 |
|---|---:|---:|---:|---:|
| complete | 10906 | 2102 | 688 | 13696 |
| cutoff | 214 | 444 | 1642 | 2300 |
| hesitant | 871 | 2190 | 474 | 3535 |

- hindsight label：ops 19531，hazard samples 205242，positives 6148（**3.00%**），dims 18。

### 9.3 训练

- train n=163500、positive=3.01%；val n=41742，**AUC=0.789**（高于 N=300 冒烟的 0.755）。
- 先验校正后的五个等频校准 bin：

| bin | pred | actual | n |
|---:|---:|---:|---:|
| 0 | 0.006 | 0.007 | 8349 |
| 1 | 0.010 | 0.011 | 8348 |
| 2 | 0.013 | 0.015 | 8348 |
| 3 | 0.023 | 0.021 | 8348 |
| 4 | 0.106 | 0.094 | 8349 |

- `c_w` 只在 3964 个合成验证 ops 上扫描，代价仍为 `sum(w) + 3.0*C_k*miss`：

| c_w | mean_w | miss / 1262 | cost |
|---:|---:|---:|---:|
| 0.02 | 1.454 | 249 | 6866 |
| 0.05 | 0.914 | 456 | 5654 |
| **0.08** | **0.638** | **627** | **5293** |
| 0.12 | 0.448 | 732 | 5302 |
| 0.20 | 0.322 | 872 | 5441 |
| 0.30 | 0.224 | 961 | 5974 |
| 0.50 | 0.142 | 1052 | 6311 |
| 0.80 | 0.091 | 1121 | 7108 |
| 1.20 | 0.036 | 1209 | 8666 |

选点为内点 **`c_w=0.08`**；模型落 `exp/w4/stophead_v0.json`。

### 9.4 FDB learned:v0 整行与门判读

缓存：决策 **215 hits / 3 misses**（cache 从 997 增至 1000，既有键 0 改写）；finality **217 hits / 0 misses**。

| arm | exact | state | done50 | prem_sum | recov_s | recov% | dExact |
|---|---:|---:|---:|---:|---:|---:|---:|
| w4lh_tact | **0.580** | 0.620 | 1.955 | **6.2s** | **106.8s** | **97.4%** | **−0.070** |

- windows(mean)：READ 0.628、REV 0.526、COMP 0.333、IRR 2.096。
- realized delay(mean)：READ 1.996、REV 2.596、COMP 1.149、IRR 4.034；`delay_monotone_in_kappa=false`。
- finality all：final 117 / unfinished 62 / hesitant 38；rollback clips：final 22 / unfinished 6 / hesitant 2；infer p50/p90=0.672/0.990（名义吞吐轨信息位），unparsed=0。
- 对 fixed 的逐夹翻转：gain **0**；loss **7** = `ecommerce_01#1`、`ecommerce_19#0`、`ecommerce_25#1`、`finance_02#0`、`finance_12#1`、`housing_17#1`、`housing_25#0`。7 个净 loss 与 0.650 → 0.580 完全对齐。
- 预注册门：exact ≥0.640 **失败（差 6pt）**；回收 ≥47% **通过**；AND 目标区 **失败**。因此 **G2 核心证据未建立**。

对零 shot 前沿的定位：相对 safe（0.630 / 30%）多回收 67.4pt、少 exact 5pt；相对 prompted（0.610 / 84%）多回收 13.4pt、少 exact 3pt。学习头把 READ/REV/COMP 窗进一步压短，却没有换回任何 fixed 的失败夹，落在更激进、更低精度的一端，未击穿目标区。

**记账口径审计**：预注册 report 的 97.4% 必须保留为主读数；其 recovery 分子按 fixed↔learned 可比的 99 夹求和，fixed-premium 分母按 fixed↔blocking 可比的 98 夹求和。差异来自 `housing_25#0`：fixed done=3.454、learned=0.108，而 blocking done=None，故该夹只进入分子。严格三臂共同 98 夹时 fixed premium=109.640s、learned premium=6.187s、recovery=103.453s，回收为 **94.4%**。两口径均远高于 47%，不改变门判决；这是 scorer 既有配对定义，不在跑后改脚本。

**迁移结论**：合成验证集内 AUC/校准良好，但 FDB exact 下落 7pt，构成明确的 sim-to-real 失败信号；本轮不能在两条预注册风险间归因——①训练 finality 是混淆表模拟、部署是真 Omni 标签；②训练状态来自脚本决策、部署来自 Omni 决策。不得用本轮 FDB 结果反调 `c_w`；下一轮需新预注册的跨域/消融证据再区分两者。

产物：`exp/w4/synth/{dialogues_v0.jsonl,hazard_v0.npz,ops_v0.jsonl}`、`exp/w4/stophead_v0.json`、`exp/w4/ladder_v0.json`、`exp/w2_rerun/decision_cache.json`；单夹结果在 FDBench 外部数据树的 `result_w4lh_tact.json`。

## 10. Rung 4 v1（预注册，2026-07-14；代价函数勘正批）

### 10.1 v0 死因判定（探针收据）

7 个丢失夹**全同构**：fixed 靠"早段 EoU 的 op 被下一 EoU patch 救回"取胜（6/7 在 eou0 launch、@eou1 被 patch），learned 给这些 op 的窗是 **0.0/0.5**——而保费由**末 EoU 窗口**贡献（70/98 夹末 EoU 有 launch；windows 均值 REV 0.526/COMP 0.333 = 策略把火力全用在压免费的中段窗上）。**头是好的**（AUC 0.789、校准贴合）；错在 v0 代价 `Σw + 3·C_κ·miss` 的两个常数：①等待被逐 op 均匀收费（真实保费只在完成关键路径=末段窗+早段溢出上产生）；②miss 定价 3s（FDB 上 miss≈二元场景死刑，汇率由 G2' 门自身给出：−1pt ↔ 51.5s ⇒ ~50s/夹）。

**防火墙声明**：两处修正均为评测记账规则的结构知识（done 锚算法、exact 二元性、G2' 汇率），从公开判分规则可先验推出，不含任何 FDB 内容/统计；v0 之误 = 未把已知记账想透，如实认。

### 10.2 v1 改动（代码已交付；特征/模型/FDB 门不变）

1. **生成器 v1**：显式时间轴——`eous`（逐 EoU 特征）+ `sigmas`（决策间静默）+ `rev_eou`；gap_silence ≡ sigmas[launch]，与 WindowLedger 语义逐位对齐。语法概率不变，新 config_hash。
2. **标注器 v1**：补 rescue 后状态的全负样本（教会"补丁已落 → 快提交"）。
3. **训练器 v1 代价（核心）**：在合成时间轴上**真实回放**策略窗——`cost = Σ(尾部溢出保费) + 50.0×miss`；rescue 后按修订 EoU 特征重估窗（与运行时 restart 路径同构）。CW_GRID 补 0.001/0.002；产出按 κ×位置的窗结构诊断。
4. **常数勘正**：T_GRID→0:0.25:4.25、W_CAP→4.0（v0 的 2.5 连自己训练分布的 gap 支持域 ≤4.0 都盖不住=设计 bug；v0 模型 JSON 自带 t_grid 不受影响）。KILL_PEN=50.0 预注册。
5. report 补严格三臂共同支撑口径（housing_25 审计），双口径并报。

### 10.3 门与诚实预测（跑前锁定）

门不变：exact ≥ 0.640 ∧ 回收 ≥47%。**点预测**：v0 的 7 个丢失夹应全部收复（其 gap ≤1.6 ≪ 新窗）⇒ exact 门大概率过；**回收门存疑且这正是本轮的信息产出**——冒烟（N=300）显示 50s 汇率 + 合成 32% 修订先验下最优解偏保守（选点 c_w=0.002、miss=0、窗均值 ~3.7 ⇒ 若原样迁移则保费高于 fixed、回收为负）。全量模型（AUC 预计 ~0.85+）能否在低危区分辨出 λ̂<c_w 的安全 op，决定回收侧成败。**已声明的先验错配风险**：合成修订率（~32%/op）远高于 FDB（~6%）⇒ λ̂ 系统性偏高 ⇒ 保守侧失守的概率不小；被批准的 v2 杠杆 = 非评测真实数据（HumDial，许可待查）做先验/强度校准——仍不触 FDB。

### 10.4 运行（同 §8 流程，tag=v1）

```bash
$PY scripts/w4_synth_gen.py --n 8000 --tag v1
$PY scripts/w4_hindsight_label.py --tag v1
$PY scripts/w4_train_stophead.py --tag v1
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4lh1_tact --prompt v3.1 \
    --delta-policy learned:v0 --stophead-model exp/w4/stophead_v1.json --workers 12
$PY scripts/w4_ladder_report.py --arms w4k0_tact w4ks_tact w4kr_tact w4pf_tact w4lh_tact w4lh1_tact
```

**汇报**：gen config_hash/修订率；label 样本数（含 rescue_states 行）；train AUC/校准/c_w 整表与选点/**κ×位置窗结构两行**；FDB cache 双行 + 打表整行（含 strict 口径）+ 对 v0 七夹的逐夹收复情况。
