# W4-V3：真实数据校准/增强停时头（G2R）——设计与两段式预注册（v1 冻结）

> **状态：CLOSED（2026-07-15）——P5 裁决 = 世界锚承重（§12），G2R 门未立（双世界锚复现空带），v3 按冻结规则无条件收口，无 C2**。结构/判据/探针门于实跑前冻结（第一段）；Phase-1 收据 = §9；数值常量与预测于 §10 冻结为 v1（第二段）；C0/C1 原始收据 = §10.6/§11.5；判读 = §11（C0）/§12（C1 + 终局）。Arm F/P 经探针判读不启用（§10.1，终局）。
> **前提变更记录（2026-07-15）**：① 用户获得 HumDial HD-Track2 完整训练用途许可（含训练集，9,988 样本/106.9h）；② 原 8/15 决策树的判定输入已于 7/14 齐备，实际进度快于路线约一个月；③ 用户明示目标 = 尽可能强的 **ICLR 2027** 文章，不被内部时间预设束缚。**v2 判决（w4_ladder_design.md §12.7）一字不动**：合成-only 的 rung 4 已收枪。本文不是 rung-4 重开，而是 §12.2-7(c) 当时预留的"许可通过 → 另开预注册增补"路径的兑现——预注册纪律防的是看结果反调同一杠杆，不禁止用预先点名的新数据源开新预注册。

## 1. 主张与门

**G2R（G2-Real）**：以真实非评测数据（HumDial）做条件校准/特征增强的学习停时头，到达零 shot 前沿（fixed 0%/safe 30%,−2/pf 84%,−4）之外的 AND 目标区。

- **门不变**：exact ≥ 0.640 ∧ 回收 ≥ 47%（strict 共同支撑并报；同 84g text-only/workers 12 口径，对照 `w3p31_tact_d150`/`w3p31_sblock`）。
- **边际先验参数化**：HumDial 是场景定向采集（每场景 100% 发生该现象），**给不出无条件修订率**——它校准的是条件形态（间隙分布、语用信号强度、finality 混淆），不是边际率。故部署边际先验 π 作为**显式参数**处理：主报告 = π ∈ [0.02, 0.30] 的 recovery/exact 扫描曲线（纯政策层重放，零 GPU 复用决策缓存）。现行冻结值为 **π\*=0.10、Δπ\*=0.05**，均为与 FDB 无关的部署参数；完整定义见 §10.2。曲线口径对部署者的语义：“若你的域修订先验为 π，此系统给你这些数。”
- FDB 出数预算：π\* 冒烟 + 8 点各一次政策重放；冻结决策/Finality cache 应近全命中，因此预算约束在**新模型调用**而非进程运行次数。

## 2. v2 尸检 → 三臂（每臂对一个已测缺口）

v2 终局分解（§12.7–12.8 收据）：排序迁移已闭（transfer AUC 0.752 ≈ LOO 0.753）；标定 ~31pt（自身排序前沿 lost=2 处 56.2% vs 实跑 24.8%，protect 85.7% vs 需 ~60%）；类上限（LOO lost=1 仅 43.9% < 47%，且 W=1.5 二段式 gain-free ⇒ lost≤1 强制）。

| 臂 | 对应缺口 | 内容 | 预注册预期 |
|---|---|---|---|
| **C（校准-only）** | 标定 31pt | 特征/策略类不动（FEATS_V2 + 二段式 W=1.5），风险电平/θ 由 HumDial 条件分布 + π 参数校准 | **预测不过门**：回收上限 ≈ 38.4%@lost≤1（v2 自身排序前沿）。C 臂的价值 = 分解预测力的可证伪验证——预注册一个预期失败 |
| **F（特征升级，探针门控）** | 类上限 | S7 ⊕ 探针胜出特征族（T1/T2/X 中过门者）重训；抬排序上限 | 上限抬升幅度 = 探针 ΔAUC 的迁移折扣版；F 臂是 AND 区的主赌点 |
| **P（策略类，gain 通道）** | gain-free 锁死 lost≤1 | 三档窗 {0, 1.5, W3=3.0 封顶} —— 最高危 top 段给长窗，重开 v1 实证过的 gain 通道（eco15），下行仍有界（长窗仅限 top 段，段占比封顶待冻结） | 仅当 F 臂排序达标后叠加（层级消融）；单独不跑 |

## 3. 数据使用矩阵与合规（红线）

| 数据 | 用途 | 禁止 |
|---|---|---|
| HumDial HD-Track2 train | 普查/探针/训练/校准 | **任何产物（exp/、docs/、日志）不得含转写原文或可还原内容**——本仓历史会发布到公开分支 `Rice`。脚本层强制：`w4v3_common.assert_no_text()` 挡所有落盘 payload；census/probe 产物只有统计、路径、时间、哈希、标签 |
| HumDial dev | 普查统计（间隙锚）；不进训练 | 同上 |
| FDB v3 | 评测 only，冻结臂单次出数 | 训练/选参/先验估计一律禁止（防火墙不变）；输入侧无标签统计亦不用（π 参数化使其不必要） |
| RB | 暂缓（用户 7/15 指示） | — |

许可义务（Data_Protocol.md）：仅限赛事 + 赛后非商业学术研究；禁止再分发；论文须致谢 Hum-Dial Challenge，官方论文发布后按要求引用（**已记入论文义务清单**）。

## 4. Phase-1a 普查（`scripts/w4v3_humdial_census.py`，零 GPU；--vad 除外）

产出 `exp/w4v3/humdial_census.json` + `humdial_cuts.jsonl`。回填映射表（普查值 → v1 冻结常量）：

| 普查读数 | 回填目标 |
|---|---|
| `vad_pause.pause_dur_max_per_utt`（[break] 停顿实测分位，silero 400ms/30ms 与吞吐轨感知同参） | 合成 σ_pre 支撑域校验/重锚（v2 RANGES sig_fast/sig_wide 是拍的）；Arm C 的可救 gap 条件分布 |
| `utterance.*/user`（时长/字符率分位，双语） | RANGES utt_dur 三 style 带重锚 |
| `gaps_train.rev_arrival` 与 `gaps_dev.seg_gap`（改口到来间隙——train 夹模型答话如实标注；dev 为纯静默，最干净） | 修订到来时间轴先验；W_PROTECT/W3 档位的覆盖率核算 |
| `pause.t1_rates_prefix` vs `t1_rates_full` | T1 词表的真实分离度 sanity（探针的前置目测） |
| readout finality 混淆（真停顿点 vs 真结束点的 Omni 标签矩阵） | 替换合成 FIN_JITTER=0.6 的拍值；Arm C 校准的核心条件量 |
| `anomalies`（对照格式报告 §11 已知值：溢出 108/13、无标记 64 等） | 加载层正确性核验，不进模型 |

**kill 判据（Phase-1 后）**：可用 [break] 停顿事件 < 300，或 VAD 停顿分布与合成 v2 支撑无显著差且探针双输 ⇒ v3 整体取消，零 GPU 浪费，论文以 v2 分析节收尾。

## 5. Phase-1b 探针（`scripts/w4v3_text_probe.py`，CPU；门现在冻结）

- **任务**：PAIRED-CONTINUATION——同一话语的 [break] 前缀（label 1，有延续）vs 全句（label 0，真结束）；配对共组、按全文哈希去重分组（§11.11 泄漏防护）、5-fold 分组 CV。
- **特征族**：S（长度/语言，+census cuts 的实测时长）；T1（双语语用词表）；T2（尾部字符 n-gram 哈希 256）；P（标注员标点——**泄漏嫌疑，永不进门**，只作对照行）；X（Omni 读数：rung-3 同线 finality + 冻结 AMEND_PROMPT_V0，`w4v3_omni_readout.py`）。
- **门（冻结）**：ΔAUC = AUC_cv(S+T1+T2) − AUC_cv(S) ≥ **0.05** 且 DeLong 单侧 p < **0.01**（pooled OOF）。过 ⇒ Arm F 启用（特征族取胜出者）；不过 ⇒ Arm F 取消，v3 收缩为 Arm C 单臂。X 族同式独立评判（过则 F 臂可含 Omni 读数特征——"蒸馏即特征"扩展）。
- 次级报告（不门控）：T-only 配对准确率 + 翻转置换 p（精确零假设 0.5）；S 的配对膨胀效应如实标注（前缀恒短于全句——部署语义下当前时长本就是合法特征，且 ΔAUC 判据两侧同含 S，自动抵消）。
- **探针在 HumDial 上做，不碰 FDB**。FDB 侧 LOO 特征增补探针（§12.4 条件 (ii) 原式）仅在 F 臂过 HumDial 门后作为诊断复核跑一次，标注 diagnostic-only。

## 6. Phase-2 结构（历史冻结快照；不再生效）

> 本节保留两段式预注册第一段的原始占位，便于审计；**现行 Phase-2 规范仅见 §10**。其中 λ_mix 已由 §10.2 作废为 0，以下 `▢` 不代表遗漏待填。

- **训练数据** = 合成生成器 v2（域随机化不动）⊕ HumDial 真实事件样本（[break] 停顿 → hazard 正例，gap = VAD 实测；真结束 → 负例）。混合权 λ_mix = ▢；真实样本的 t 网格/标签定义与 `w4_hindsight_label.py` hazard 目标同构（宽限只进代价回放不进标签，纪律不变）。
- **校准层**：risk 电平重标定 = 条件 likelihood（HumDial）× 边际 π（参数）——实现为对 λ̂ 的先验比例修正（公式冻结时定，含推导）；θ 仍由盈亏锚 1.5/50 结构给出，扫 π 出曲线。
- **模型**：LR 主臂 + MLP h=16 消融（v2 选型纪律：合成+HumDial val 联合选型，平手取 LR）；F 臂特征 = FEATS_V2 ⊕ 探针胜出族（HumDial 可测子集在真实样本上有值，FDB 侧由转写/读数管道供给——供给链在冻结版写死）。
- **评测**：先 rollback-17 冒烟（预算门 = ▢），过则全量单次；ladder_report 增 π 曲线段。
- 预注册预测（冻结时点数）：Arm C 回收 ∈ [30%, 38.4%]@lost≤1、不过门；Arm F 过门当且仅当探针 ΔAUC 迁移后仍 ≥ ▢；P 臂 gain ≥ 1 的概率押注 = ▢。

## 7. 诚实风险

① reactive（HumDial 打断 = 对模型输出的反应）vs self-initiated（FDB rollback = 自发改口）的域差——普查把两类间隙分开报，训练只用自发类（[break] 停顿）为正例主体；② 中英混合 vs FDB 英文域；③ [break] 是标注约定，真实停顿谱可能比场景采集更宽（场景配额偏置已声明）；④ 探针过门 ≠ FDB 迁移成立（HumDial→FDB 仍是一次迁移，但比 合成→FDB 少拍一层世界）；⑤ π 参数化把"选 θ"变成"报曲线"，审稿人可能要单点主张——预注册保守 π\* 即为回应。

## 8. 运行命令（Phase-1；服务器）

```bash
PY=/root/miniconda3/envs/fd-sds/bin/python
# 1a 普查（文本层，~分钟级；--strict-counts 对表格式报告）
$PY scripts/w4v3_humdial_census.py --root /root/autodl-tmp/HumDial_train \
    --splits train,dev --strict-counts --out exp/w4v3/humdial_census.json
# 1a' VAD 停顿实测 + 切点导出（CPU ~30-45min 或 GPU 日更快）
$PY scripts/w4v3_humdial_census.py --root /root/autodl-tmp/HumDial_train \
    --splits train --vad --emit-cuts exp/w4v3/humdial_cuts.jsonl \
    --out exp/w4v3/humdial_census_vad.json
# 1b 文本探针（CPU 秒级；先跑 --selftest）
$PY scripts/w4v3_text_probe.py --selftest
$PY scripts/w4v3_text_probe.py --root /root/autodl-tmp/HumDial_train \
    --splits train --cuts exp/w4v3/humdial_cuts.jsonl --out exp/w4v3/text_probe.json
# 1b' Omni 读数（GPU 日，vLLM 栈在位；先 --dry-run --limit 3）
$PY scripts/w4v3_omni_readout.py --cuts exp/w4v3/humdial_cuts.jsonl \
    --root /root/autodl-tmp/HumDial_train --modes finality,amend \
    --out exp/w4v3/omni_readout.jsonl
# 1b'' 探针加 X 族复跑
$PY scripts/w4v3_text_probe.py --root /root/autodl-tmp/HumDial_train \
    --splits train --cuts exp/w4v3/humdial_cuts.jsonl \
    --extra-jsonl exp/w4v3/omni_readout.jsonl --out exp/w4v3/text_probe_x.json
```

回报清单：census 的 counts/strict-counts 行、vad_pause 分位、ranges_anchor 段、anomalies 对表；probe 的 combos 表 + GATE 行（主/X 两份）；readout 的 cache 行与 skip 数。拿到即冻结 v1（§6 填数 + 预测点数），然后才谈训练与 FDB。

## 9. Phase-1 实跑收据（2026-07-15；仅供 §6 数值冻结输入）

### 9.1 运行口径与评分前修正

- 数据根：`/root/autodl-tmp/HumDial_train`，加载器解析至其 `HD-Track2/` 包装层；train/dev 物理计数分别为 9,988/1,800。
- GPU：NVIDIA RTX 6000D 85,651 MiB。三阶段音频输出配置的 stage-0 `gpu_memory_utilization=0.72` 启动时无 KV block，且该任务只需音频理解→文本标签；正式读数故复用已冻结并在同类 84g 卡验证过的 `exp/w3/qwen3_omni_text_only_84g.yaml`（SHA-256 `793d1ef0…6677`；stage-0 only、8192、`max_num_seqs=1`、util 0.78）。模型、prompt、T=0/seed=42 和线格式均未改。
- 实跑中发现四个实现—预注册偏差，均在任何 X 探针评分前修正并补 selftest：① overrun anomaly 改按文本 tier 声明 `xmax`（保留非空段给角色/时长/切点），恢复格式报告 108/13；②离散分数 AUC ties 改用 midrank；③新增独立 `GATE_X = DeLong(S+X, S)`，主门原样保留；④ cache 只允许五个 canonical 标签或 `__unparsed__`，杜绝模型 raw 落盘。另因 prefix 缺 X 全由 VAD 未检出内部停顿导致、与 label 强相关，X 评分在看结果前冻结为严格 paired common support，避免补零形成 missingness 旁道。
- 扩展自测：census 15/15、probe 8/8、py_compile、cache 合规守卫均通过。未运行训练或 FDB。

### 9.2 counts / anomalies / VAD

`--strict-counts` 最终控制台：

```text
strict-counts train_total: 9988 vs 9988 PASS
strict-counts dev_total: 1800 vs 1800 PASS
strict-counts anomalies.train_overrun_gt_100ms: 108 vs 108 PASS
strict-counts anomalies.train_overrun_gt_1s: 13 vs 13 PASS
strict-counts anomalies.train_quote_text_samples: 9 vs 9 PASS
strict-counts anomalies.pause_no_break_zh: 42 vs 42 PASS
strict-counts anomalies.pause_no_break_en: 22 vs 22 PASS
strict-counts anomalies.dev_duration_mismatch: 12 vs 12 PASS
```

其余已知 anomaly：`dev_duration_mismatch=12`；无标记总数 `42+22=64`。训练音频小时 en/zh = 39.232/67.647（合计 106.879h）。VAD 在 1,147 个有 `[break]` 样本上导出 748 个 `break_mid`；399 个未检出内部停顿；另导出 1,147 个 `utt_end`，cuts 共 1,895 条。

| lang | 统计 | n | mean | min | p10 | p25 | p50 | p75 | p90 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zh | max pause/utt | 484 | 1.0171 | 0.388 | 0.484 | 0.644 | 0.900 | 1.284 | 1.6584 | 3.140 |
| zh | all internal | 500 | 0.9997 | 0.388 | 0.484 | 0.612 | 0.868 | 1.284 | 1.636 | 3.140 |
| en | max pause/utt | 264 | 1.2022 | 0.388 | 0.516 | 0.740 | 1.092 | 1.572 | 1.924 | 3.588 |
| en | all internal | 295 | 1.1381 | 0.388 | 0.484 | 0.660 | 0.996 | 1.508 | 1.924 | 3.588 |

### 9.3 `ranges_anchor`

用户话语时长锚：

| lang | n | mean | min | p10 | p25 | p50 | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zh/user | 12,347 | 3.8918 | 0.000 | 2.080 | 2.899 | 3.779 | 4.789 | 5.880 | 14.129 |
| en/user | 5,208 | 5.7125 | 0.4034 | 2.267 | 3.979 | 5.650 | 7.309 | 8.979 | 17.670 |

train `rev_arrival`（含中间 assistant 音频，故只作 reactive 锚；单元秒）：

| scene | en n/p50/p90 | zh n/p50/p90 |
|---|---:|---:|
| Follow-up Questions | 373 / 22.8684 / 33.2644 | 1,134 / 18.2084 / 24.8524 |
| Negation or Dissatisfaction | 372 / 20.0884 / 26.0024 | 839 / 14.7584 / 18.8736 |
| Repetition Requests | 373 / 25.2484 / 34.0744 | 840 / 16.9784 / 23.5484 |
| Silence or Termination | 372 / 24.2034 / 34.5964 | 840 / 19.6584 / 27.1114 |
| Topic Switching | 373 / 23.8484 / 33.9204 | 840 / 18.3184 / 26.3014 |
| User Real-time Backchannels | 371 / 26.4384 / 33.1084 | 840 / 18.2684 / 23.2224 |

dev `seg_gap` 是采集模板固定静默：18 个 lang/scene 组的 p50/p90 **全部 5.0/5.0s**（总 n=1,809 个 gap；每组 n=50–200）。合成 v2 对照支撑保持：`sig_fast_mu=[-1.5,-0.5]`、`sig_fast_s=[0.4,0.8]`、`sig_wide_p_lo=[0.35,0.65]`、`inter_req_lo=[0.6,1.5]`、`inter_req_hi=[2.0,4.0]`、`utt_complete=[[1.2,2.0],[4.5,6.5]]`、`utt_hesitant=[[1.0,1.6],[3.5,5.5]]`、`utt_cutoff=[[0.6,1.2],[2.5,4.0]]`。真实 max-pause p50 0.900/1.092s 高于合成 fast 分量的 config-median 上界约 0.607s；因此“双探针输且停顿分布与合成无显著差”的复合 kill 前提不能仅由本轮直接宣告成立（“显著差”检验尚未在 v0 草案中数值冻结）。

### 9.4 主文本探针与 X 共同支撑探针

主探针：1,143 对/2,286 实例，全文哈希去重 4 对；cuts duration 覆盖 82.6% <95%，故 S 只含 `log_len,is_zh`。X 探针在 745/1,143 对共同支撑上评分（398 对丢弃），1,490 实例 X 与 duration 均 100% 覆盖；S 含 4 列。

| combo | 主 AUC (ncol) | X-common AUC (ncol) |
|---|---:|---:|
| S | 0.9841 (2) | 0.9897 (4) |
| T1 | 0.6218 (7) | 0.6252 (7) |
| T2 | 0.7359 (256) | 0.7357 (256) |
| S+T1 | 0.9859 (9) | 0.9901 (11) |
| S+T2 | 0.9638 (258) | 0.9697 (260) |
| S+T1+T2 | 0.9662 (265) | 0.9688 (267) |
| S+T1+T2+P | 0.9979 (268) | 0.9991 (270) |
| S+X | — | 0.9916 (10) |
| S+T1+T2+X | — | 0.9719 (273) |

```text
MAIN GATE: AUC(S+T1+T2)=0.9662 vs AUC(S)=0.9841; ΔAUC=-0.0179;
           DeLong z=-6.378, one-sided p=0.99999999991 -> FAIL
X-common MAIN: 0.9688 vs 0.9897; ΔAUC=-0.0210;
               z=-6.863, p=0.999999999997 -> FAIL
GATE_X: AUC(S+X)=0.9916 vs AUC(S)=0.9897; ΔAUC=+0.0018;
        z=2.312, one-sided p=0.010381 -> FAIL
```

T-only 次级 paired accuracy 主/X-common = 0.8014/0.8000，flip-permutation p 均为 1/2001≈0.0004998；说明文本尾部有原始配对信号，但在近饱和 S 之上没有正增量。按冻结 §5，**T1/T2 与 X 双输，Arm F 不启用；当前只剩 Arm C，是否触发复合 kill 留给 §6 数值冻结轮裁定。**

### 9.5 Omni readout

正式首跑：`readout: 1895 labeled / 0 skipped`；`cache: 36 hits / 3754 misses`。同 cache 全量复放：`3790 hits / 0 misses`，输出与 cache SHA-256 均逐字节不变。

| truth kind | n | final | hesitant | unfinished | amend continue | amend done | unparsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `utt_end` | 1,147 | 803 | 0 | 344 | 1,142 | 5 | 0 |
| `break_mid` | 748 | 110 | 0 | 638 | 748 | 0 | 0 |

finality 对真实停顿的 unfinished 召回 638/748=85.3%，对真结束的 final 召回 803/1,147=70.0%，有 standalone 信号；但它在共同支撑的 S 上只增 0.0018 AUC。`amend` 几乎塌成 continue（1,890/1,895），无可用分离度。

### 9.6 产物哈希与裁决边界

| artifact | SHA-256 |
|---|---|
| `humdial_census.json` | `22531d1e5ce6e44bfd5111a066e450cf2568db373ee5dad83014a8f5dbd181df` |
| `humdial_census_vad.json` | `74d42f208613e732751c6f20cf9d54cfc4bfaeaf4310c9129235a47e063e5f16` |
| `humdial_cuts.jsonl` | `325f898b639d4cdf6dd5f3e19735799ef51d029f70cecda2d5873558dcbf1f93` |
| `omni_readout.jsonl` | `affe547d43a40060fc4357d10422baa8757fc9be926e7256ee2dc41a59b9190c` |
| `omni_readout_cache.json` | `2b9062ee4cf8fa42ca62e06912c1418d1e913f6d76aa888ff1b7ab8a4cde498f` |
| `text_probe.json` | `43d2386321a91e666ab3a80ced3db9ed982d2c3e9c94877e5284793548810945` |
| `text_probe_x.json` | `838439a9767162839e934902c8aff5e802861a77f76b55090bfb6f151bc2b94d` |

本节记录 Phase-1/§9 时点，当时尚未完成数值冻结、不得据此启动训练或 FDB；该状态现已由 §10 的 v1 冻结与 Phase-2 放行覆盖。

## 10. v1 数值冻结（2026-07-15；Phase-1 判读 + Arm C 定版——本节生效后方可跑 Phase-2）

### 10.1 Phase-1 判读（对 §9 收据）

- **Arm F：不启用（终局）**。主门 ΔAUC=−0.0179 / p≈1，X 门 +0.0018 / p=0.0104——双 FAIL。**门校准诚实注**：S=0.984 使 ΔAUC≥0.05 的可用头室只剩 ~1.6pt，门在饱和基线下数学上近不可过；但裁决对门重参数化**稳健**——主门方向为负（文本添噪而非增信），X 门幅度 0.0018 连头室折半的门也过不了。结论"结构/长度之外，文本语用与韵律读数在真实延续数据上无正增量"成立，且它本身是 FEATS_V2 特征收缩的真实数据背书（论文素材，正面引用）。
- **Arm P：不进入（终局）**。结构冻结的 F 门控生效；gain 通道议题归档 future work，v3 内不复活。
- **复合 kill：不触发**。事件数 748 ≥ 300；"停顿分布与合成 v2 支撑无显著差"的检验现冻结为：实测 max-pause p50 落入合成 fast 分量 config-median 带 [e^{−1.5}, e^{−0.5}] = [0.223, 0.607] ⇒ 无差。实测 0.900(zh) / 1.092(en) 均在带外（超上界 48%/80%）⇒ 有实质差 ⇒ 第二合取假。W 覆盖核算：W=1.5 覆盖实测停顿 ~p75–p85（zh p75 1.284 < 1.5 < p90 1.658；en p50 1.092 < 1.5 < p75 1.572）；risk-horizon 的闭右口径覆盖 **731/748 = 97.7%**，严格 operational rescue（`w > gap−grace`）为 **730/748**，只差 `gap=2.5` 的一个边界样本——W_PROTECT=1.5 维持不动的实测依据。
- 附带发现（记录，论文可用）：judge 在 HumDial 音频上 **hesitant 发射率 = 0**（FDB 上 38/217）——判官行为本身随域移动；amend v0 塌缩 continue 1,890/1,895 = **无效仪器**，不构成"语义信号不存在"的证据（承载信号的是 finality：85.3%/70.0% standalone，但对 S 无增量）。

### 10.2 Arm C 定版：C0 先验移位（零训练）+ C1 条件诊断支

设计收缩理由：探针证明真实数据在结构特征之上不给新特征 ⇒ HumDial 对 C 臂的数值贡献收缩为条件量（C0 用 q_H；C1 用 σ_pre/finality 混淆/utt 带）。**C0 不训练、不动 v2 权重**——校准轴被隔离成唯一自由变量，落点检验因果干净。

- **C0 公式（冻结）**：`risk′ = σ(logit(risk) + logit(π·q_H) − logit(m_train))`；数值等价实现 = θ 重映射 **`θ_eff(π) = σ(logit(θ) − logit(π·q_H) + logit(m_train))`**，θ=0.03 盈亏锚不动（JSON 阈值存 6 位小数）。实现 = `scripts/w4v3_make_armc.py` 产出 8 个 stophead JSON（v2 权重及归一化参数数值逐项不变 + theta 重写 + armc 溯源块），**零 harness 改动**（`--delta-policy learned:v2` + `--stophead-model` 原机制）。
- **常量（从已归档产物算死，脚本 load 时复验）**：
  - **q_H = 731/748 = 0.97727**（`humdial_cuts.jsonl`：P(pause ≤ RISK_HORIZON 2.5 | **VAD 检出的 `break_mid`**)——HumDial 对 C0 的唯一数值贡献，条件形态、非边际率；399 个 VAD 未检出内部停顿样本不在此条件分母）；
  - **m_train = 3307/19288 = 0.17145**（`ops_v2.jsonl`：完整 v2 合成语料的 horizon 事件边际；与 gen 收据 4744×69.7% 逐位自洽）；
  - π 语义 = 部署域 op 级**任意修订率**（用户可感参数）；horizon 事件率 = π·q_H。
- **π 网格 = {0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30}**（8 点各一次 replay，同 δ 网格先例）；**π\* = 0.10**（"每十请求一次改口"——保守整数默认，声明为任意且与 FDB 无关）；**Δπ\* = 0.05**。对照锚：v2 隐含部署先验 = m_train/q_H = **0.1754**（θ_eff 恒等点，网格括住它）。
- **θ_eff 表**（selftest 8/8 锁定）：

| π | 0.02 | 0.04 | 0.06 | 0.08 | **0.10** | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|---|
| θ_eff | 0.2430 | 0.1359 | 0.0932 | 0.0702 | **0.0558** | 0.0359 | 0.0257 | 0.0152 |

- **λ_mix = 0（删除）**：真实样本不进训练——slots_missing 等 FDB 侧特征在 HumDial 无对应，混样引入特征域错配；真实数据以"校准常量 + 世界锚"形态进入。§6 的 λ_mix 占位据此作废。
- **C1（条件诊断支，仅当 P4 落点检验失败时跑）**：生成器再锚定——σ_pre ← 748 实测 max-pause 经验重采样（截 [0.38, 3.60]）；finality 发射行 ← 实测混淆（延续行 .147/.000/.853，结束行 .700/.000/.300；hesitant-style 行无实测保留 v2 行并标注）；utt_dur 带 ← [[2.0,4.5],[5.0,9.0]]（覆盖实测双语 p10–p90）；tag `v3c1` 重生成/标注/训练后同 C0 公式校准，FDB 单次。
- **冒烟（π\* 点，30 子集）**：机制门 = windows 原始值 ∈ {0,1.5} ∧ 决策缓存命中 ≥90% ∧ 启动行打印 θ_eff=0.0558。无数字门。

### 10.3 预注册预测（跑前锁定）

- **P1**：AND 过门 π 带 = **空**（gain-free ⇒ lost≤1 强制；v2 排序前沿 lost=1 = 38.4% < 47%）——G2R 门判据预测失败；**C0 的成功判据是 P4**，不是门。
- **P2**：回收对 π 弱单调降、exact 弱单调升（允许 ≤2 个相邻网格点出现单夹缓存分叉违例）。
- **P3** π\* 点行：protect 分数 ∈ [0.55, 0.75]（v2@θ=0.03 的 85.7% 为参照）；回收 ∈ [26%, 42%]；exact ∈ [0.615, 0.645]；done50 ≤ 3.455。
- **P4 落点检验（C0 的真判据）**：对实现 lost ∈ [0,4] 的网格点，|部署回收 − v2 排序前沿@同 lost| ≤ 5 回收点的占比 ≥ **75%**。前沿参照 = `ceiling_diag.json` transfer.v2.w15_frontier = {lost0: 15.1, lost1: 38.4, lost2: 56.2, lost3: 61.7, lost4: 69.9}%（诊断模拟器口径；其 protect@θ=85.7% 与 live 84.0% 互证在案）。**过 ⇒ 校准轴闭合**：三轴各有实测收据（排序 = v2 迁移 0.752≈上限；校准 = C0 落点；类上限 = LOO 43.9<47 + gain-free 论证）。**不过 ⇒ 触发 C1** 定位世界锚 vs 校准公式。
- **P5** 若 C1 触发：预测 C1 ≈ C0（±2pt 回收 / ±1 夹）= 世界锚零假设；若实质分化 ⇒ 世界锚承重（surprising = informative），进归因节。

### 10.4 运行（Phase-2；服务器）

```bash
PY=/root/miniconda3/envs/fd-sds/bin/python
# 0. 生成 8 个 C0 头（零 GPU；常量对产物复验 + 打印 θ_eff 表）
$PY scripts/w4v3_make_armc.py --selftest && $PY scripts/w4v3_make_armc.py
# 1. 冒烟（π* 点，30 子集，机制门见 §10.2）
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4v3c_smoke --prompt v3.1 \
    --delta-policy learned:v2 --stophead-model exp/w4v3/stophead_v3c_pi100.json \
    --ids-file exp/w3/tuning30.json --workers 12
# 2. π 网格全量（8 点；text-only 84g 栈 + workers 12，决策缓存应 ~全命中）
for pi in 020 040 060 080 100 150 200 300; do
  $PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4v3c_pi${pi}_tact \
      --prompt v3.1 --delta-policy learned:v2 \
      --stophead-model exp/w4v3/stophead_v3c_pi${pi}.json --workers 12
done
# 3. 打表（π 曲线 = 8 臂行）
$PY scripts/w4_ladder_report.py --arms w4v3c_pi020_tact w4v3c_pi040_tact \
    w4v3c_pi060_tact w4v3c_pi080_tact w4v3c_pi100_tact w4v3c_pi150_tact \
    w4v3c_pi200_tact w4v3c_pi300_tact --out exp/w4v3/ladder_armc.json
```

回报清单：冒烟机制门三项；每臂 cache 双行；ladder 整表（exact/state/done50/premium/回收主+strict/flips/windows 两点计数）；θ_eff 表与每次运行启动行打印的 theta 对账。P4 落点表（lost、回收、前沿参照、差值）我来算。

### 10.5 v3 终局叙事（预写，两出口）

- **P4 过**：学习组件章 = 三轴定量解剖 + 真实数据把每轴钉上收据；π 曲线给出部署者语义的可用前沿（"你的域先验是 π，则得这些数"）；G2R 门失败按 P1 预注册如实报。
- **P4 不过**：C1 分叉归因（世界层 vs 公式层），仍是闭合的负结果链。
- 任一出口 v3 即收口，文稿装配线开工；RB 议题另起（用户已裁暂缓）。

### 10.6 Arm C0 Phase-2 实跑收据（2026-07-15；仅归档观测，不代做 P1–P5/P4 裁决）

运行口径：输入提交 `22b1610`；NVIDIA RTX 6000D 85,651 MiB；冻结
`qwen3_omni_text_only_84g.yaml`（SHA-256 `793d1ef0…6677`）；v3.1 / learned:v2 /
workers=12 / nominal infer=1.0s。全网格每臂 100/100 completed、217 decisions；四个
decision-cache miss 均由在线栈 200 成功补齐，Finality miss 为 0。历史
`exp/w4/ladder_v0.json` SHA-256 仍为 `2715e3e6…c87`，未被打表覆盖。

**冒烟机制门（π\*=0.10，30 夹）**：启动 `theta=0.055792`（四位对账
0.0558）；decision/Finality cache 均 58/0；窗口 `{0.0×31, 1.5×17}`；30/30
completed、58 decisions。三项机制门全部 PASS。

| π | 启动 θ | decision H/M | finality H/M |
|---:|---:|---:|---:|
| .02 | .243022 | 216/1 | 217/0 |
| .04 | .135935 | 217/0 | 217/0 |
| .06 | .093174 | 217/1 | 217/0 |
| .08 | .070166 | 217/1 | 217/0 |
| **.10** | **.055792** | **218/0** | **217/0** |
| .15 | .035921 | 217/1 | 217/0 |
| .20 | .025668 | 218/0 | 217/0 |
| .30 | .015195 | 218/0 | 217/0 |

合计 decision cache `1796H/4M`、Finality `1794H/0M`。部分行 decision 调用为
218 是 `housing_24#0` 的 parse-repair retry（全网格共 6 个额外调用），满足
`H+M = decisions + repair`，不是结果数异常。

Ladder 主表（fixed=`w3p31_tact_d150`：exact .650、state .670、done50 3.455、
premium sum 109.6s@n98；blocking=`w3p31_sblock`：exact .670、done50 1.955）：

| π | exact | state | done50 | premium sum | recovery s / 主 | strict (n=98) | Δexact | windows 0/1.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .02 | .570 | .610 | 1.955 | −44.1 | 157.1 / 143.3% | 140.2% | −.080 | 141 / 10 |
| .04 | .600 | .630 | 1.955 | −23.0 | 136.0 / 124.1% | 121.0% | −.050 | 115 / 42 |
| .06 | .600 | .630 | 1.955 | −7.7 | 120.7 / 110.1% | 107.1% | −.050 | 104 / 56 |
| .08 | .600 | .630 | 1.955 | −1.9 | 114.9 / 104.8% | 101.7% | −.050 | 100 / 60 |
| **.10** | **.600** | **.630** | **1.955** | **1.6** | **111.4 / 101.6%** | **98.5%** | **−.050** | **96 / 64** |
| .15 | .630 | .660 | 2.399 | 42.9 | 66.6 / 60.8% | 60.8% | −.020 | 58 / 109 |
| .20 | .640 | .660 | 3.455 | 93.4 | 16.1 / 14.7% | 14.8% | −.010 | 15 / 154 |
| .30 | .650 | .670 | 3.455 | 102.7 | 7.1 / 6.5% | 6.4% | .000 | 2 / 170 |

所有臂 gain 均为 0。完整 occurrence-aware losses：

- π=.02：`ecommerce_01#1, ecommerce_19#0, ecommerce_25#1, finance_02#0, finance_12#1, housing_17#1, housing_25#0, travel_07#1`；
- π=.04/.06/.08/.10：`ecommerce_01#1, ecommerce_19#0, ecommerce_25#1, finance_12#1, housing_25#0`；
- π=.15：`ecommerce_01#1, ecommerce_25#1`；π=.20：`ecommerce_25#1`；π=.30：无 loss。

机器收据：`exp/w4v3/armc_run_receipt.json`；完整 ladder（含 occurrence-aware flips、
窗口总体/分 κ 计数和 protect fraction）：`exp/w4v3/ladder_armc.json`。结果完整性、模型
字段、θ、窗口、score、单调性及历史 ladder 不变均经独立脚本断言通过。

## 11. Arm C0 判读：P1–P5 对账 + P4 落点表 ⇒ C1 触发（2026-07-15；对 §10.6/`ladder_armc.json` 逐位重算核验）

### 11.1 P1–P5 对账

- **P1 空带预测 → 兑现**。全网格无 AND 点：π=.20 exact .640 ✓ / 回收 14.7% ✗；π=.15 回收 60.8% ✓ / exact .630 ✗；π=.30 exact .650 ✓ / 6.5% ✗。G2R 门判据按预注册失败。
- **P2 单调性 → 兑现（0 违例，额度 2）**：回收 143.3→124.1→110.1→104.8→101.6→60.8→14.7→6.5 严格降；exact .570→.600(×4)→.630→.640→.650 弱升。
- **P3 π\* 点 → 3/4 子带 MISS**：protect .400（带 [.55,.75]）、回收 101.6%（带 [26,42]%）、exact .600（带 [.615,.645]）；done50 1.955 ✓。根因 = FDB 上 risk 读数的**质量悬崖**：θ ∈ (.0152, .0558] 区间 protect 从 98.8% 掉到 40.0%（~59% 的 op 挤在这条窄带，另有 .0558–.0932 的平台只放 5%）——预测带假设的平滑稀释不成立。
- **P4 落点检验 → FAIL（1/3 = 33.3% < 75%）⇒ 按 §10.3 冻结规则 C1 触发**：

| π | lost | 部署回收 | v2 前沿@lost | 差 | 判 |
|---|---|---|---|---|---|
| .15 | 2 | 60.8% | 56.2% | **+4.6** | PASS |
| .20 | 1 | 14.7% | 38.4% | **−23.7** | FAIL |
| .30 | 0 | 6.5% | 15.1% | **−8.6** | FAIL |

（.02/.04–.10 实现 lost = 8/5×4，超出前沿参照域 [0,4]，按冻结规则不计入分母。）

### 11.2 机制读数（判读所依，非新判据）

1. **lb op 的底部落位（loss 阶梯完全嵌套）**：0 loss → +eco25 → +eco01 → +{eco19, fin12, hou25} → +{fin02, hou17, travel_07}——v0 七夹的嵌套超集。由网格夹逼：**eco25#1 risk ∈ [.0152, .0257)**（要 98.8% protect 才护住——三代持久夹的定量定位）；**eco01#1 risk ∈ [.0257, .030)**（91.1% 护住；v2 的 θ=.03 恰好放行它——v2 loss 集的闭式解释）。lost=1→2 之间回收跳变 14.7→60.8：护住 eco01 要求 θ 压到 .0257，连带保护 ~44 个夹在中间的安全 op——**eco01 的保护成本 ≈ 46 个回收点**。
2. **P4 失败分解为两个成分**：(a) **网格粗化**——lost=1 只在 protect .911 处被采样，真实部署 lost=1 前沿在 θ ∈ (.0257, .0359) 内未测（上界 < 60.8%）；(b) **参照系误差**——诊断模拟器前沿在低 lost 端系统性高于 live 记账（+8.6 / +23.7pt）；protect@θ 两口径互证过（85.7 vs 84.0），但**中间工作点的保费映射不互证**。⇒ P4 失败**不构成**"校准公式失败"的充分证据；剥离世界锚成分正是 C1 的职责（P5）。
3. **正结果（零 shot 前沿更新，论文素材）**：**C0@π=.30 弱帕累托支配 fixed**（exact .650 相等、0 loss、premium 102.7 < 109.6、done50 同 3.455）——**W4 全程序第一个无损支配 fixed 的臂**；**C0@π=.15 (60.8%, −2) 严格支配零 shot safe 臂 (30%, −2)**（同 exact 代价、回收翻倍）。可用前沿更新为 **C0π.30 (6.5%, 0) / C0π.15 (60.8%, −2) / pf (84%, −4)**；AND 区（47% @ ≥−1）仍未达，与 P1 一致。机制侧：217/218 决策全缓存位齐 ⇒ 全部差异纯窗口政策效应，归因干净。

### 11.3 C1 执行规范（§10.2 冻结支的具体化；代码已交付）

- **生成器 `--anchor v3c1`**（`w4_synth_gen.py`；默认路径已验证**逐字节不动**：config_hash b62a069cd900 复现 + 本地重生成 8000 对话与归档 `dialogues_v2.jsonl` cmp 相同）：σ_pre ← 748 实测 max-pause 经验重采样（截 [0.38, 3.60]，四修订类同源；池加载守卫 ≥300）；finality 基行 ← 实测混淆（complete ← 结束行 .700/.000/.300、cutoff ← 延续行 .147/.000/.853、hesitant 保留 v2 行 = 无实测），FIN_JITTER 机制不变；utt 带 ← complete ((2.0,4.5),(5.0,9.0)) / hesitant 0.8× / cutoff 0.5×；GAP_MAX 相应 5.25；config_hash 覆盖 anchor 常量 + 停顿池哈希。**600 冒烟收据**：σ_pre p10/p50/p90 = 0.452/0.996/1.924（复现实测分布）；发射行落实测值 ± jitter（complete .690/.000/.310、cutoff .183/.000/.817）；**rescuable≤2.5 占比 42.6% vs v2 的 69.7%**——真实停顿更长 = 世界锚差异的实体，C1 的 m_train 将显著低于 v2。
- **训练**：同流水线 tag=v3c1（`w4_hindsight_label.py --tag v3c1 --feats v2` → `w4_train_stophead.py --tag v3c1`）；取 **`stophead_v3c1_lr.json`**（LR 主臂先例）；trainer 自选 θ 弃用——θ 一律由 make_armc 从 0.03 锚重映射（`--ops` 给出时豁免 base-theta 检查）。
- **make_armc `--variant v3c1 --ops exp/w4/synth/ops_v3c1.jsonl`**：m_train 从 C1 世界重算（精确分数入溯源块）；q_H 不变（同一 HumDial 测量）；π 网格同 8 点；providers `w4v3c1_pi{020..300}_tact`。selftest 已扩 11/11（含 m_train 重算 + variant 命名 + trainer-θ 豁免）。
- **判读 = P5**（C1 ≈ C0：±2pt 回收 / ±1 夹 = 世界锚零假设；实质分化 = 世界锚承重）+ 同式 P4 落点表（信息位，参照系误差成分预期同现）。**C1 之后 v3 无条件收口**——不再有 C2。

### 11.4 运行命令（C1；服务器）

```bash
PY=/root/miniconda3/envs/fd-sds/bin/python
# 0. 生成 C1 世界 + 标注 + 训练（纯 CPU）
$PY scripts/w4_synth_gen.py --n 8000 --seed 42 --tag v3c1 --anchor v3c1 \
    --pauses exp/w4v3/humdial_cuts.jsonl
$PY scripts/w4_hindsight_label.py --tag v3c1 --feats v2
$PY scripts/w4_train_stophead.py --tag v3c1
# 1. C1 头重映射（m_train 从 C1 ops 重算；selftest 先行）
$PY scripts/w4v3_make_armc.py --selftest
$PY scripts/w4v3_make_armc.py --stophead exp/w4/stophead_v3c1_lr.json \
    --variant v3c1 --ops exp/w4/synth/ops_v3c1.jsonl
# 2. π 网格（8 点；同 84g text-only + workers 12）
for pi in 020 040 060 080 100 150 200 300; do
  $PY scripts/w2r_stream_replay.py --delta 1.5 --provider w4v3c1_pi${pi}_tact \
      --prompt v3.1 --delta-policy learned:v2 \
      --stophead-model exp/w4v3/stophead_v3c1_pi${pi}.json --workers 12
done
# 3. 打表（显式输出，别覆盖 C0 收据）
$PY scripts/w4_ladder_report.py --arms w4v3c1_pi020_tact w4v3c1_pi040_tact \
    w4v3c1_pi060_tact w4v3c1_pi080_tact w4v3c1_pi100_tact w4v3c1_pi150_tact \
    w4v3c1_pi200_tact w4v3c1_pi300_tact --out exp/w4v3/ladder_armc1.json
```

回报清单：gen 行（config_hash / rescuable 占比 / 发射混淆）；train 行（val AUC / m_train 精确分数）；每臂 cache 双行；ladder 整表 + flips + windows。P5 对账与 C1 版 P4 落点表我来算；C1 收口后 v3 终局叙事按 §10.5 两出口装配。

### 11.5 Arm C1 Phase-2 实跑收据（2026-07-15；仅归档观测，不代做 P5/C1 版 P4 裁决）

输入提交 `8b9ae9a`；同 84g text-only stage-0、prompt v3.1、
`learned:v2`、workers=12、nominal infer=1.0s 口径。默认 v2 路径先独立
重生成 8,000 条：`config_hash=b62a069cd900`，与归档 `dialogues_v2.jsonl`
SHA256 同为 `428a5b74…04b` 且逐字节相等，冻结路径未扰动。

**C1 世界/训练**：`config_hash=cca0eb049af8`，8,000 dialogues / 19,312 ops /
4,675 revised。748 条经验 σ_pre 池 p10/p50/p90=`0.484/0.964/1.828s`；
生成后修订 `gap_silence` p10/p50/p90=`2.124/2.604/3.468s`；horizon 内
`2014/4675=43.0802%`，故 **m_train=`2014/19312=0.10428749`**。发射混淆
完整计数（final/hesitant/unfinished）：complete=`8151/0/3697`、
cutoff=`464/0/2585`、hesitant=`1152/2593/670`。标签层 400,887 hazard /
4,627 positives / dims=7；LR val AUC **0.74847195**，MLP stdout AUC **0.902**；
按冻结规范使用 LR，trainer θ=.02 被弃用，由 θ anchor=.03 重映射；selftest
**11/11 PASS**。

启动/缓存收据（每臂均 100/100 completed、217 decisions、Finality 217/0；
decision=218 表示同一条 parse-repair 多一次调用）：

| π | θ | decision H/M | Finality H/M |
|---:|---:|---:|---:|
| .02 | .152996 | 217/0 | 217/0 |
| .04 | .081318 | 217/0 | 217/0 |
| .06 | .054651 | 217/0 | 217/0 |
| .08 | .040728 | 218/0 | 217/0 |
| .10 | .032176 | 218/0 | 217/0 |
| .15 | .020533 | 218/0 | 217/0 |
| .20 | .014606 | 217/1 | 217/0 |
| .30 | .008607 | 218/0 | 217/0 |

π=.20 的服务关闭首轮在一个新轨迹键上得到 99/100；GPU 恢复后该键在线
HTTP 200，随后同 provider **全量 `--force` 重放**得到上表一致的 100 件收据。
合计 decision=`1740H/1M`、Finality=`1736H/0M`、repair=5；decision cache
1013→1014，既有键 0 改/0 删，Finality cache 217→217 完全不变。

Ladder 主表（fixed=`w3p31_tact_d150`：exact/state=.650/.670、done50=3.455、
premium=109.6s@n98；blocking exact=.670、done50=1.955）：

| π | exact | state | done50 | premium sum | recovery s / 主 | strict (n=98) | Δexact | protect | windows 0/1.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| .02 | .570 | .610 | 2.031 | −37.3 | 150.3 / 137.1% | 134.1% | −.080 | .072 | 141 / 11 |
| .04 | .570 | .620 | 2.033 | −18.4 | 131.3 / 119.8% | 116.7% | −.080 | .194 | 125 / 30 |
| .06 | .600 | .630 | 2.037 | −5.5 | 118.4 / 108.0% | 105.0% | −.050 | .331 | 105 / 52 |
| .08 | .600 | .630 | 2.037 | 3.4 | 109.5 / 99.9% | 97.0% | −.050 | .375 | 100 / 60 |
| **.10** | **.620** | **.650** | **2.048** | **31.8** | **81.2 / 74.1%** | **71.1%** | **−.030** | **.512** | **80 / 84** |
| .15 | .640 | .660 | 3.538 | 102.9 | 6.5 / 5.9% | 6.2% | −.010 | .905 | 16 / 153 |
| .20 | .650 | .660 | 3.455 | 103.0 | 6.8 / 6.2% | 6.1% | .000 | .948 | 9 / 163 |
| .30 | .650 | .670 | 3.536 | 113.9 | −4.4 / −4.0% | −3.8% | .000 | 1.000 | 0 / 172 |

所有臂 gain=0；occurrence-aware loss 集严格嵌套：

- π=.02/.04：`ecommerce_01#1, ecommerce_19#0, ecommerce_25#1, finance_02#0, finance_12#1, housing_17#1, housing_25#0, travel_07#1`；
- π=.06/.08：`ecommerce_01#1, ecommerce_19#0, ecommerce_25#1, finance_12#1, housing_25#0`；
- π=.10：`ecommerce_01#1, ecommerce_25#1, housing_25#0`；
- π=.15：`ecommerce_25#1`；π=.20/.30：无 loss。

八臂 finality 均为 final/unfinished/hesitant=`117/62/38`、unparsed=0；窗口
支持严格为 `{0,1.5}`。独立复算 exact/state/done/premium/recovery/flips 与
report 全部一致；strict 共同支撑若不在分子处先舍入 0.1s，个别行只差最多
0.1pt，不影响任何夹或判据。完整 ladder=`exp/w4v3/ladder_armc1.json`
（SHA256 `5b556ba7…dee4`），机器收据=`exp/w4v3/armc1_run_receipt.json`；C0
ladder、历史 v0 ladder、冻结 v2 产物哈希均未变化。HumDial 原文/敏感字段
扫描通过；服务已停、GPU 已释放。

**P5 对账与 C1 版 P4 落点表待用户裁决；本节不代判。C1 后无 C2。**

## 12. Arm C1 判读：P5 裁决 = 世界锚承重；v3 终局收口（2026-07-15；对 `ladder_armc1.json` 逐位重算核验）

### 12.1 P5 对账（C1 vs C0，冻结带 ±2pt 回收 / ±1 夹）

| π | 回收 C0→C1 | Δ回收 | Δexact（夹） | 带内？ |
|---|---|---|---|---|
| .02 | 143.3 → 137.1 | −6.2 | 0 | ✗ |
| .04 | 124.1 → 119.8 | −4.3 | **−3** | ✗✗ |
| .06 | 110.1 → 108.0 | −2.1 | 0 | ✗（边缘） |
| .08 | 104.8 → 99.9 | −4.9 | 0 | ✗ |
| .10 | 101.6 → 74.1 | **−27.5** | **+2** | ✗✗ |
| .15 | 60.8 → 5.9 | **−54.9** | +1 | ✗ |
| .20 | 14.7 → 6.2 | −8.5 | +1 | ✗ |
| .30 | 6.5 → −4.0 | −10.5 | 0 | ✗ |

**P5 零假设被决定性拒绝（8/8 点回收带外，2 点夹带外）⇒ 按冻结分支：世界锚承重（surprising = informative）**。P5 的预测本身（C1≈C0）未中，如实记账。

### 12.2 分化的两个成分（机制读数）

1. **公式成分（m_train 移位）**：实测停顿更长 ⇒ rescuable 69.7%→43.1% ⇒ m_train 0.17145→**0.10429** ⇒ 全网格 θ_eff 下移（odds ×~0.55）。副产品：C1 头的隐含部署先验 = m_train/q_H = **0.1067 ≈ π\*=0.10**——π\* 点上 C1 几乎以原生校准运行（θ_eff .0322 ≈ 锚 .03）。
2. **风险场成分（重训改变排序/电平的联合结构）**：同 protect ~.91 对比——C0 (lost=1, 14.7%) vs C1 (lost=1, 5.9%)：C1 的保护集保费吸收更差；同 π=.10 对比——C1 在 protect .512 就救回 eco19/fin12（C0 在 .400 丢 5 夹），**0-loss 点从 protect .988（C0）提前到 .948（C1）**：持久夹 eco25 的相对排名在真实世界锚下上升。两个成分对冲后：C1 在 −3 夹档新增前沿点 **(74.1%, −3)**，但在 −1/0 档全面劣于 C0。
3. **回溯修正 §11.2 的判读**：P4 失败此前归为"网格粗化 × 参照系误差、不构成公式失败证据"——C1 现在证明其中**确有真实的世界层成分**（风险场随世界锚实质移动）。校准轴的终局收据：**先验移位机制按设计工作（单调/嵌套/机械预测全中），但绝对落点被世界模型误差支配；关闭它需要联合 (状态, 修订时序) 的真实域内数据——恰是防火墙 + 域差所拒绝的东西**。

### 12.3 C1 版 P4 落点表（信息位；参照 = v2 排序前沿，对 C1 为双重外参）

.10/lost3：74.1 vs 61.7 = **+12.4** ✗；.15/lost1：5.9 vs 38.4 = −32.5 ✗；.20/lost0：6.2 vs 15.1 = −8.9 ✗；.30/lost0：−4.0 vs 15.1 = −19.1 ✗ ⇒ 0/4 带内。

### 12.4 门与前沿（v3 终局读数）

- **AND 门（exact ≥.640 ∧ 回收 ≥47%）在 C1 上同样空带**：exact 达标点 .15/.20/.30 的回收 = 5.9/6.2/−4.0%。**P1 的空带结论跨两个世界锚复现**——门位于该特征类 + 二段式策略类的可达域之上（与 LOO 43.9% < 47% + gain-free 论证三方互证）。G2R 未立。
- **双臂并集的部署前沿**（exact 档位 → 最佳回收）：0 夹 **6.5%**（C0π.30；C1π.20 6.2% 次之——**两个世界锚各自给出无损弱支配 fixed 的点，结论稳健**）/ −1 夹 14.7%（C0π.20）/ −2 夹 **60.8%**（C0π.15，严格支配零 shot safe）/ −3 夹 **74.1%**（C1π.10，新点）/ −4 夹 84%（pf）/ −5 夹 124.1%（C0π.04）。
- **归档注记（不影响判决）**：C1 π=.15/.30 两臂 premium p50 = 1.575/1.576（≠1.5）、π=.30 protect-all premium 113.9 = fixed+4.3s、done50 +0.081——**outcome 层与 fixed 完全一致（0 loss/0 gain/state 同），偏差全在完成锚层**。候选解释：① workers-12 名义 regime 下 tool_wall 全局 RNG 交错的已知非权威抖动（W3 纪律：并发档延迟锚只作信息位——W4 全部回收数字同属此 regime，臂间比较同规同权）；② housing_24 类 parse-repair 的单夹轨迹微移。量级 ≪ P5 判据的承重差（−27.5/−54.9pt），不改任何裁决；可选的服务器一行归因（对 fixed 档逐夹 diff done 锚）留作档案卫生项，不重跑。

### 12.5 v3 终局裁决与收口

按 §11.3 冻结："C1 之后 v3 无条件收口，不再有 C2。" **v3 关闭。** 终局叙事（§10.5 出口二的充实版）：

1. **G2R 门未立**（P1 预测兑现，双世界锚复现）；学习组件章 = **三轴定量解剖，每轴有实测收据**：排序轴（v2 迁移 0.752≈LOO 0.753，域随机化关闭 sim-to-real 排序差）；校准轴（C0/C1：先验移位机制全部机械预测兑现，绝对落点被世界模型误差支配 = 世界锚承重的 C1 实证）；类上限轴（LOO 43.9<47 + gain-free + 双世界空带复现）。
2. **正结果清单**：C0π.30 无损弱支配 fixed（W4 首例，且跨世界锚稳健）；C0π.15 严格支配零 shot safe；C1π.10 新增 (74.1%,−3) 前沿点；持久夹闭式定位（eco25/eco01 的 risk 带）；探针负结果 = FEATS_V2 收缩的真实数据背书；π 参数化曲线 = 部署者语义的可用前沿（双世界版本）。
3. **数据义务**：论文致谢 Hum-Dial Challenge + 官方论文后按要求引用；所有归档产物无转写原文（合规红线全程执行）。
4. **下一步 = 文稿装配线**（学习组件分析章素材已齐：w4_ladder_design §11–12 + 本文全部）；RB 议题（用户已认同优先）另起新预注册；神谕 10 号文（W4+V3 终局决策记录）发送时机由用户定。
