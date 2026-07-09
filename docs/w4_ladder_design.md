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
