# W4-V3：真实数据校准/增强停时头（G2R）——设计与预注册草案

> **状态：DRAFT（两段式预注册的第一段）**。本文结构、判据形式与探针门**现在冻结**；数值常量（§6 占位）等 Phase-1 普查/探针结果回来后冻结为 v1——**冻结前不跑任何 FDB**。
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
