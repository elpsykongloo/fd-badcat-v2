# W4-V3：真实数据校准/增强停时头（G2R）——设计与预注册草案

> **状态：PHASE-1 COMPLETE / NUMERIC FREEZE PENDING（两段式预注册的第一段已实跑）**。本文结构、判据形式与探针门已在实跑前冻结；§6 数值常量仍待下一次明确冻结为 v1——**冻结前不跑任何 FDB**。Phase-1 完整收据见 §9。
> **前提变更记录（2026-07-15）**：① 用户获得 HumDial HD-Track2 完整训练用途许可（含训练集，9,988 样本/106.9h）；② 原 8/15 决策树的判定输入已于 7/14 齐备，实际进度快于路线约一个月；③ 用户明示目标 = 尽可能强的 **ICLR 2027** 文章，不被内部时间预设束缚。**v2 判决（w4_ladder_design.md §12.7）一字不动**：合成-only 的 rung 4 已收枪。本文不是 rung-4 重开，而是 §12.2-7(c) 当时预留的"许可通过 → 另开预注册增补"路径的兑现——预注册纪律防的是看结果反调同一杠杆，不禁止用预先点名的新数据源开新预注册。

## 1. 主张与门

**G2R（G2-Real）**：以真实非评测数据（HumDial）做条件校准/特征增强的学习停时头，到达零 shot 前沿（fixed 0%/safe 30%,−2/pf 84%,−4）之外的 AND 目标区。

- **门不变**：exact ≥ 0.640 ∧ 回收 ≥ 47%（strict 共同支撑并报；同 84g text-only/workers 12 口径，对照 `w3p31_tact_d150`/`w3p31_sblock`）。
- **边际先验参数化**：HumDial 是场景定向采集（每场景 100% 发生该现象），**给不出无条件修订率**——它校准的是条件形态（间隙分布、语用信号强度、finality 混淆），不是边际率。故部署边际先验 π 作为**显式参数**处理：主报告 = π ∈ [0.02, 0.30] 的 recovery/exact 扫描曲线（纯政策层重放，零 GPU 复用决策缓存）。**判据 = 过门 π 带非空且带宽 ≥ Δπ\*（常量待冻结）**；单点主行取预注册保守默认 π\*（待冻结，从服务型对话文献/HumDial 打断场景占比推导，推导过程记录在冻结版）。曲线口径对部署者的语义：“若你的域修订先验为 π，此系统给你这些数。”
- FDB 出数预算：每冻结臂**单次**，全程 ≤ 3 次新 FDB 运行（决策缓存应 ~全命中，近零 GPU）。

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

## 6. Phase-2 结构（常量占位，冻结时填数）

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

本节不是 §6 数值冻结：`λ_mix/π*/Δπ*/P top 段/rollback 冒烟门/点预测` 仍为空，不得据此启动训练或 FDB。
