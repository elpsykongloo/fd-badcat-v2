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

### 10.5 全量结果（2026-07-14；预注册后只跑一次）

#### 运行口径与完整性

- 代码锚：`09f2dda`（实现提交 `c5a1bc5`）；开跑前 `tact` 工作树干净、96GB RTX PRO 6000 空闲、`result_w4lh1_tact.json` 为 0 份。
- gen → label → train → FDB → ladder 严格按 §10.4 顺序执行，五条命令均 exit 0。服务严格复用 7/09 口径：`exp/w3/qwen3_omni_text_only_84g.yaml`、stage-0、8192、`max_num_seqs=1`、GPU utilization 0.78。
- replay 完成 100/100，stdout 无 `WARNING`/`ERROR`；8 次 cache miss 在 proxy/vLLM 两端均 HTTP 200。结束后按 proxy → vLLM 优雅停服，端口释放、GPU 回到 0 MiB。
- 独立完整性复核：dialogues 8000 行、ops 19402 行、hazard `X=(382869,18)` / `y=382869` / 6310 positives、模型 features/mean/std/weights 均 18 维、`t_grid` 18 点覆盖 0–4.25、100 个结果 JSON 全部可解析。官方 `evaluate_pass_rate.py` 独立复算为 64/100。

#### Gen / label

- `config_hash=497567920fc4`，seed 42，dialogues 8000，ops 19402。
- 修订 6310 / 19402 = **32.5%**。kinds（占修订）：slot_completion 2257（35.8%）、afterthought 1114（17.7%）、hesitant_revision 1097（17.4%）、cutoff_continuation 1459（23.1%）、upstream 383（6.1%）。
- domain：travel 1975、ecommerce 2011、finance 2028、housing 1986。
- style → observed finality：complete=(final 10888 / hesitant 2007 / unfinished 748)，cutoff=(227 / 458 / 1602)，hesitant=(895 / 2044 / 533)。
- label：ops 19402，**rescue_states 6310**，hazard samples 382869，positives 6310（**1.65%**），dims 18。

#### Train

- train n=305905、positive=1.65%；val n=76964，**AUC=0.859**（跑前预计 0.85+，命中）。
- 先验校正后五个等频校准 bin：

| bin | pred | actual | n |
|---:|---:|---:|---:|
| 0 | 0.001 | 0.001 | 15393 |
| 1 | 0.003 | 0.003 | 15393 |
| 2 | 0.006 | 0.005 | 15392 |
| 3 | 0.012 | 0.012 | 15393 |
| 4 | 0.074 | 0.061 | 15393 |

- `c_w` 在 1600 个合成验证 dialogues 上扫描；代价为 `sum(tail premium) + 50.0*miss`，共 1251 个 revised ops：

| c_w | premium | premium/dialogue | miss / 1251 | cost |
|---:|---:|---:|---:|---:|
| 0.001 | 6250s | 3.906 | 1 | 6300 |
| **0.002** | **5802s** | **3.626** | **6** | **6102** |
| 0.005 | 4552s | 2.845 | 47 | 6902 |
| 0.010 | 3282s | 2.052 | 121 | 9332 |
| 0.020 | 2042s | 1.276 | 255 | 14792 |
| 0.030 | 1432s | 0.895 | 349 | 18882 |
| 0.050 | 867s | 0.542 | 472 | 24467 |
| 0.080 | 517s | 0.323 | 597 | 30367 |
| 0.120 | 314s | 0.196 | 710 | 35814 |
| 0.200 | 172s | 0.107 | 860 | 43172 |
| 0.300 | 110s | 0.069 | 965 | 48360 |

选点为 **`c_w=0.002`**。结构诊断：

- mean window by kappa：READ 3.843、REV 3.841、COMP 3.993、IRR 4.000。
- mean window by position：**final_eou 3.820**、earlier 3.909。final_eou ≫1.5，跑前已明确预告回收侧危险。

#### FDB 整行、strict 口径与门

缓存：决策 **210 hits / 8 misses**（cache 1000→1008，新键 8、既有键 0 改写）；finality **217 hits / 0 misses**（217→217）。

| arm | exact | state | done50 | prem_sum | recov_s | recov% | dExact | strict common support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| w4lh1_tact | **0.640** | 0.640 | 5.955 | **318.6s** | **−211.6s** | **−193.1%** | **−0.010** | **−190.6% (n=98)** |

- FDB windows(mean)：READ 3.604、REV 3.815、COMP 3.983、IRR 4.000；commit delay(mean)：READ 6.103、REV 7.170、COMP 5.871、IRR 5.802；commit-delay κ 单调性 false。
- premium paired n=98，p50=3.970s。strict 共同支撑复算：fixed premium 109.640s、v1 premium 318.630s、recovery −208.990s（report 按 −209.0s / 109.640 = **−1.906**）；主口径额外包含 blocking done 为空的 `housing_25#0`，故为 −211.6s / 109.6 = **−1.931**。双口径结论一致。
- finality all：final 117 / unfinished 62 / hesitant 38；rollback：final 22 / unfinished 6 / hesitant 2；infer p50/p90=0.672/0.990（吞吐轨信息位），unparsed=0。
- 对 fixed 翻转：gain `ecommerce_15#0`；loss `ecommerce_23#0`、`ecommerce_25#1`，净 −1pt，与 0.650→0.640 对齐。
- 门：exact ≥0.640 **精确踩线通过**；回收 ≥47% **失败**（主/strict 均远低于门）；AND 目标区 **失败**，G2 核心证据仍未建立。

#### v0 七个 loss 的逐夹收复

以下均用固定目录 hash 直接调用官方 `evaluate_scenario_pass(use_llm=False)` 重判，避免 occurrence 漂移：

| v0 loss | fixed | v0 | v1 | 收复 |
|---|---:|---:|---:|---:|
| ecommerce_01#1 | ✓ | ✗ | ✓ | ✓ |
| ecommerce_19#0 | ✓ | ✗ | ✓ | ✓ |
| ecommerce_25#1 | ✓ | ✗ | ✗ | **✗** |
| finance_02#0 | ✓ | ✗ | ✓ | ✓ |
| finance_12#1 | ✓ | ✗ | ✓ | ✓ |
| housing_17#1 | ✓ | ✗ | ✓ | ✓ |
| housing_25#0 | ✓ | ✗ | ✓ | ✓ |

实际收复 **6/7**。唯一例外 `ecommerce_25#1` 仍缺 `add_to_cart`：v1 给早段 `track_order` / `search_products` 的窗只有 0.25 / 0.0s，并未得到平均意义上的长窗，因此跑前“七夹 gap 都小于新窗”预测在该夹失效。新增 loss `ecommerce_23#0` 是长窗轨迹下多发 `process_refund`；新增 gain `ecommerce_15#0` 则消除了 fixed/v0 的多余 `track_order(DL555)`。

#### 预测对账与结论

- **AUC 预测命中**：预计 0.85+，实际 0.859；合成域可学性与校准成立。
- **exact 预测基本命中但非全中**：七夹收复 6/7，加一个新 gain、一个新 loss，最终 exact 0.640 精确过门，而非预测的七夹全收复。
- **回收赌点落在保守失败分支**：训练 final_eou 3.820s，部署 READ/REV/COMP/IRR 窗均接近 4s cap，done p50 从 fixed 3.455s 拉到 5.955s，保费从 109.6s 增至 318.6s。全量头未能把足够多低危 op 分到 `lambda_hat < 0.002`。
- 这构成对“合成修订先验 32.5% 远高于部署域，概率系统性偏高”风险的强一致证据，但单轮 FDB 结果不能单独证明唯一因果。按预注册，不用 FDB 反调；v2 只允许用 HumDial 等非评测真实数据做先验/强度校准，并须另开预注册。

产物：`exp/w4/synth/{dialogues_v1.jsonl,hazard_v1.npz,ops_v1.jsonl}`、`exp/w4/stophead_v1.json`、更新后的 `exp/w4/ladder_v0.json` 与 `exp/w2_rerun/decision_cache.json`；100 个单夹结果在 FDBench 数据树的 `result_w4lh1_tact.json`。

## 11. 因果可达上限诊断（2026-07-14；`scripts/w4_ceiling_diag.py`，零 GPU）

**性质**：与 `w3_oracle_frontier` 同类的 hindsight 诊断——在 FDB 自己的特征分布上估计"承重修订可分性"的上限；LOO（留一夹）防内漏；**产物永不进训练/选点**（防火墙照旧）。

**数据**：fixed 臂 trace 重建 147 个 launch op（18 维同源特征，finality 从 w4pf 交叉引用——同音频同调用，缓存 217/0 佐证）；hindsight 修订标签 25（patch 21 / relaunch 4）；**承重正类 = patch 救回且该夹 pass = 13**。gap 分布揭示屏障效应：patch 类密集在 1.64s（=0.64 hold + 1.0 nominal infer）——fixed 1.5 能救靠的是 guard 期过期→延迟提交→patch 救回，**有效 gap = gap − 1.0**（此修正后 W=1.5 覆盖全部 13 个承重 op）。sanity：同一记账下 fixed-sim 保费 106.5s vs 实测 109.6s（差 3%）。

**三个判定数字（lost = 未保护承重夹 ≈ exact 损失；W=1.5 保护窗、屏障修正后）**：

| 排序来源 | AUC | lost=0 | lost=1 | lost=2 | lost=3 |
|---|---|---|---|---|---|
| **LOO 上限**（FDB 特征训练，诊断） | 0.753 | 32% | **44%** | 52% | 74% |
| stophead **v0** 迁移 | 0.679 | 8% | 18% | 22% | 34% |
| stophead **v1** 迁移 | 0.644 | 6% | 8% | 29% | 41% |
| oracle-features（完美判别） | — | **100%** | | | |

**结论**：① 信号**存在**（LOO 0.753；载荷特征 = slots_missing 0.668 / f_unfinished 0.625 / 短 utt_dur 0.642 / 早 eou_idx 0.59）——"afterthought 完全不可预测"的悲观假设被否定；② 目标区角 (≤1, ≥47%) 在 LOO 前沿**边缘**（44% vs 47%，且这是全局阈值+全局窗+LR 的下界）；③ **约束瓶颈被精确定位为 sim-to-real 排序迁移**：合成训练模型的 AUC 差距（0.75→0.64/0.68）在 lost≤1 处折损 26–36 个回收点。v1 的失败此前归因"先验偏高→保守"，本诊断进一步显示即便经济学修对，排序迁移不够也到不了目标区。

**决策含义**：v2 的唯一正当靶 = 缩小排序迁移差。防火墙内的杠杆（按安全性排序）：(a) **域随机化**——多套生成器配置混合训练，逼模型依赖秩稳定结构而非合成边缘分布；(b) 特征收缩到信号核（slots/finality/utt_dur/eou_idx/κ，5–7 维，减小漂移面）；(c) 非评测真实语料（HumDial，许可待查）校准特征边缘分布与修订先验；(d) 策略形态改为诊断验证过的"score→保护(W=1.5)/立即提交"二段式 + 训练回放加入屏障宽限（又一处 metric-structural 修正）。**v2 = 最后一枪**，预注册后不再迭代；未中则按 8/15 决策树落 ICASSP 分支，本诊断的上限曲线 + 迁移分解直接成为论文的分析节。

## 12. Rung 4 v2（预注册，2026-07-14；排序迁移批——最后一枪）

### 12.1 死因链条与本轮靶

v0 死于**代价函数错**（记账层，§10.1 已修）；v1 死于**世界错**（分布层：合成修订先验 32.5% ≫ 部署域，窗全饱和 4s cap，回收 −193%）。§11 上限诊断把剩余瓶颈**唯一定位在 sim-to-real 排序迁移**（同一 LR：域内 LOO 0.753 → 合成迁移 0.644/0.679，lost≤1 处折损 26–36 回收点），并证明策略形态 W=1.5（屏障修正后）覆盖全部 13 个承重 op。v2 不再赌"把世界拍对"（防火墙禁止看 FDB 调分布），改为**让学习对分布未知免疫、把策略下行用结构封死**：模型只需学跨配置稳定的排序，其余全部固定。

### 12.2 改动清单（五杠杆；代码已交付）

1. **(a) 域随机化**（`w4_synth_gen.py` v2 重写）：语法常量 → 预注册 `RANGES` 表，**每 dialogue** 从范围采样一套配置（style 混合、各修订通道概率、slot 缺失率、σ_pre 分布参数、finality 混淆行抖动 J=0.6、链概率、间隔静默）；配置与内容同一 `random.Random(seed)` 驱动，config_hash 覆盖范围表。validator 打印 per-dialogue 期望修订率 `rev_prior` 的 p10/p50/p90（预期横跨 ~5%–45%，把部署域先验包进支撑域）。
2. **结构勘正 ①（GAP_FLOOR）**：gap = **1.64 + σ_pre**（0.64 hold + 1.0 nominal infer = 吞吐轨决策间静默物理下界；v0/v1 的 [0.3,4.0] gap 物理不可实现）。σ_pre（用户开口修订前的犹豫）才是被随机化的量。**结构勘正 ②（utt_dur 因果链）**：utt_dur 按 style 条件采样（cutoff 话语本来就短）——v0/v1 style 无关均匀采样使该部署实信号在训练中是死重。两处均为引擎/语言结构知识，非 FDB 统计。
3. **(b) 特征收缩**（`src/stophead.py` `FEATS_V2`，7 维）：`t, eou_idx, utt_dur, slots_missing, f_final, f_hesitant, f_unfinished`。**显式排除**：κ one-hots（§11 单特征 AUC≈随机 0.48–0.53，且二段式经济学在 FDB 二元 pass 上 κ-平坦——miss=场景死刑、保费=1.5s/op 均与 κ 无关）、gap_prev/n_prior_ops/chain_dep（≈随机）、domain one-hots（纯合成伪影载体）。特征行按名字从模型自带 `feats` 构建（标注/训练/运行时单源）。
4. **(d) 二段式策略**（`StopHead.risk/window`）：`risk(op) = 1 − Π_{t<RISK_HORIZON}(1−λ̂(t))`（RISK_HORIZON=2.5=W_PROTECT+GRACE=可救 gap 支撑域）；`window = 1.5 if risk ≥ θ else 0`。**下行结构有界：protect-all ≡ fixed δ\*=1.5 臂**，v1 的 4s 窗/318.6s 保费形态不可表达；上界 = oracle（只保护承重 op）。排序质量成为唯一自由变量。
5. **(e) 屏障宽限入训练回放**（`w4_train_stophead.py` v2 重写）：救援判定 = `w>0 且 w > gap − GRACE`（窗在下一决策 guard 内过期 → 屏障推迟 → patch 救回；与 §11 对 FDB 的实证规则同构，fixed-sim 保费 sanity 差 3%）。保费记账沿用 v1 尾溢出定义，KILL_PEN=50 不变。
6. **模型类阶梯内消融**：LR（主臂，~10¹ 参数）+ 单隐层 tanh MLP h=16（~10² 参数，seed=0 确定性 Adam）；(model, θ) 在**合成 val** 上按回放代价**联合选择**，平手取 LR。产出 `stophead_v2.json`（选中者）+ `stophead_v2_{lr,mlp}.json` 审计件；FDB 只跑选中者一次。
7. **θ 选点切片（跑前冒烟发现的结构修正，预注册化）**：全混合 val 上代价最优解塌缩到 **protect-all**（池化修订率 ~25% 使任何提前提交的边际保费收益 ≪ 50s 死刑期望——v1 先验错配换位重现，且 protect-all 在 FDB 上 ≡ fixed 臂 = 回收门必败）。故 θ 选点在 **val 的 rev_intensity 下三分位切片**上做：随机化混合是**教学分布**，高修订域是刻意夸张的（教排序稳健），部署代表性经济学在低强度带（op 级先验 ~2%–14%）。mid/high/full 三档 θ\* 全部打印留档（预期 full 档 = protect-all，作为该机制的对照证据）。选点仍纯合成侧，FDB 不进入。
7. **(c) HumDial 校准：本轮不启用**（训练用途许可未核查完）。范围以宽先验代替锚定；若后续许可通过，HumDial 只得用于范围中心/边缘校准并须另开预注册增补，不改本轮判决。

### 12.3 预注册常量（跑前锁定；改动=新 tag）

- `RANGES`（见 `w4_synth_gen.py` 文件头，本表为正文）：style 权重 complete/hesitant/cutoff = (2.0,6.0)/(0.5,2.5)/(0.3,1.8) 归一；p_slot_missing (0.08,0.40)；p_rev slot/cutoff/hesitant/afterthought = (0.45,0.95)/(0.45,0.95)/(0.15,0.70)/(0.02,0.25)（**pre-intensity**）；**全局修订强度 rev_intensity = log-uniform (0.15,1.50)** 乘到四通道（cap 0.97）——把 per-dialogue op 级先验支撑域拉到 ~3%–55%，部署样低修订域**进入训练支撑**（v1 单点 32.5% 先验 = 已判死因）；首意图倍率 (1.0,2.0)；p_chain (0.30,0.80)；p_upstream_hit (0.10,0.50)；σ_pre fast=lognormal(μ∈(−1.5,−0.5),s∈(0.4,0.8)) 截 [0.03,1.5]，wide=双带 [0.1,1.0]/[1.0,2.8] 低带权 (0.35,0.65)；inter_req (0.6,1.5)–(2.0,4.0)；utt_dur 三 style 界范围见文件；FIN_JITTER=0.6。N=600 冒烟：池化修订率 25.4%、rev_prior p10/p50/p90 = 0.072/0.185/0.441、gap floor=1.64 精确、可救占比 65.8%。
- 结构常数：HOLD_S=0.64、GRACE=1.0（nominal infer）、GAP_FLOOR=1.64、W_PROTECT=1.5、RISK_HORIZON=2.5——全部为评测轨引擎知识，非 FDB 统计。
- 训练/选点：THETA_GRID = {0.002,0.005,0.01,0.02,0.03,0.05,0.08,0.12,0.20,0.35,0.50}；KILL_PEN=50.0；**选点切片 = val 按 cfg.rev_intensity 下三分位**（§12.2-7）；生成 N=8000/seed=42；hazard 目标定义不变（宽限只进代价回放，不进标签）；did%5 切分不变。
- 经济学 sanity 锚（跑前写下）：二段式盈亏平衡 θ\* ≈ 保费上界/KILL_PEN = 1.5/50 = 0.03——低带选点若落 0.01–0.08 区间为算术自洽（N=600 冒烟：LR 低带 θ\*=0.03 精确命中；full 档 θ\*=0.02=protect-all，对照机制成立）。
- 冒烟信息位（N=600，不构成门）：LR val AUC 0.715 / **MLP 0.860**——随机化使线性任务显著变难（预测 ② 兑现方向），MLP 的 finality×slots×utt 交互在混合分布下有真实增益，消融臂地位坐实。风险相应更新：若全量后 LR/MLP 排序在 FDB 上分化，选型规则（低带代价）已预注册，不得跑后改选。

### 12.4 门、诚实预测与容量判据（跑前锁定）

**门不变**：exact ≥ 0.640（净口径，gain 可抵 loss）∧ 回收 ≥47%（主口径；strict 共同支撑并报）。

**诚实预测**：① 结构上保费 ≤ fixed ⇒ **回收 ≥ 0 保证成立**（v1 式负回收不可表达），回收侧的真赌点变为"θ 处保护率足够低"；② exact 侧的真赌点回到排序迁移：LOO 上限在 lost=1 处 44%、lost=2 处 52%——**命中 AND 区需要迁移 AUC 从 0.64–0.68 收复到 ~0.72+**（域随机化+特征收缩+两处结构勘正合力），或 lost=2 时出现 ≥1 个 gain（v1 实证 gain 存在）。③ 已声明风险：域随机化可能不足以关闭迁移差（配置多样性 ≠ 真实分布覆盖）；GAP_FLOOR/utt_dur 结构建模若有误差会systematic 移动 σ_pre 支撑域。预计 val AUC 会**低于** v1 的 0.859（randomization 使任务变难——这是特性不是回归，迁移 AUC 才是靶）。

**容量升级判据（固化，回应"是否换 0.6B"）**：仅当 (i) v2 迁移后排序质量已逼近 LOO 上限（FDB 侧 lost≤1 前沿差 ≤5 回收点）仍不过门，且 (ii) 特征增补 LOO 探针（零 GPU，扩展 `w4_ceiling_diag.py` 特征列）证明**文本语义特征显著抬升域内上限**时，才允许引入文本编码器（特征提取器形态，0.6B 起）。二者缺一，容量升级永久排除——§11 已证明现瓶颈在数据轴不在容量轴，同时换容量+换数据将使 v2 归因失效。

**最后一枪声明**：v2 预注册后不再迭代 rung 4；未中 AND 区则按 8/15 决策树落 ICASSP 分支（§11 曲线 + 三代失败归因 = 论文分析节），G2 叙事收缩为"停时规则的 zero-shot 可实现性 + 学习组件的迁移瓶颈实证"。

### 12.5 运行（前三步纯 CPU；第四步同 84g text-only 口径 + workers 12）

```bash
$PY scripts/w4_synth_gen.py --n 8000 --seed 42 --tag v2
$PY scripts/w4_hindsight_label.py --tag v2 --feats v2
$PY scripts/w4_train_stophead.py --tag v2
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4lh2_tact --prompt v3.1 \
    --delta-policy learned:v2 --stophead-model exp/w4/stophead_v2.json --workers 12
$PY scripts/w4_ladder_report.py --arms w4k0_tact w4ks_tact w4kr_tact w4pf_tact \
    w4lh_tact w4lh1_tact w4lh2_tact
```

### 12.6 汇报清单（跑完发回）

gen：config_hash / 池化修订率 / **rev_prior p10/p50/p90** / gap 分位（floor 应=1.64）/ 混淆表。label：样本数 / 正例率 / dims=7 / rescue_states。train：**双头 val AUC** / LR 校准 bin / **双头 θ 扫描整表** / 联合选型行（winner+双成本）/ protect-by-{finality,style,position,kind} 四行。FDB：cache 双行（决策+finality 均应高命中）/ 打表整行（exact/state/done50/prem/recov 主+strict）/ **对 fixed 逐夹翻转**（v0 七夹逐夹核对）/ finality 分布 / windows 均值（应为 {0,1.5} 两点混合）。
