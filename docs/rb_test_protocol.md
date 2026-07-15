# RB test-897 单发协议（v1，2026-07-16 冻结）

> 前提：dev3 有效性门 PASS（`rb_dev3_validity_receipt.json`）。**dev 迭代窗口就此关闭**：
> 自本文冻结起，generator/scorer/runner 任何改动 = 公开勘误 + 版本递增，**不得改-重跑 test**。
> 基线：源码 `8312904`（+dev3 归档 `ddd1cb2`）；build = rb_v2.2.1 / config `265c7cd8f485`；
> 判分器冻结 v3（scorer_freeze.json，test split 运行时自动校验哈希）。

## 一、单发纪律

1. 每个 provider **一次**。重跑仅限基础设施失败（HTTP 连接死/进程崩/磁盘满），且须在收据中登记；分数不满意不是重跑理由。
2. 运行序按 §二（决策缓存复用最大化）；每臂跑完立即归档 report + 决策缓存。
3. 全部跑完后停服，离线复放全部 provider（0 miss、逐字节）作收官验证。
4. 收据：一份 `rb_test_receipt.json`（build 哈希/provider 清单/缓存计数/复放结论/异常登记）。

## 二、模型动物园与运行序（第一批 = 零训练臂）

**臂 A（test 541 夹，--input audio --split test）**，按序：

| # | provider | 命令要点 | 备注 |
|---|---|---|---|
| 1 | `rbtest_tact_d150` | `--system tact --delta 1.5` | 主臂，建缓存主本 |
| 2 | `rbtest_sblock` | `--system blocking` | 对照主臂 |
| 3 | `rbtest_tact_d150_nobar` | `--delta 1.5 --commit-barrier off` | 屏障 held-out 消融（L4 预期显著劣化） |
| 4–7 | `rbtest_tact_d{000,060,100,200}` | `--delta 0/0.6/1.0/2.0` | δ 网格（缓存大量复用） |
| 8 | `rbtest_tact_d150_nodag` | `--dag off` | DAG 消融（L7/L5 预期劣化） |
| 9–11 | `rbtest_tact_d150_fc{v1,filler,silent}` | `--floor-commit-tiers v1/always_filler/always_silent` | W5-FC 三档（决策全缓存命中，只动 say/承诺轨） |
| 12 | `rbtest_oracle_a` | `--decider oracle --input text` | 窗口化 oracle 上界参照行 |

**臂 B（test 356 夹，反应式）**：`rbtest_b_tact_d150`、`rbtest_b_sblock`（decide-at-quiescence 口径照 §2 声明）、`rbtest_b_oracle`。

**第二批（学习头臂，暂缓）**：v2/C0/C1 π 点需要 learned delta-policy 的 RB 适配层
（stophead 的 REQUIRED_ARGS/κ 面向 FDB 工具）——适配代码 + selftest 交付后另行单发，
不阻塞第一批。

## 三、预注册假设（test 上判定；均已在设计文档冻结）

- **H-B1**：臂 B 失败集中于 in-flight 窗口（对比调用前窗口，McNemar 配对）。
- **H-FC1/FC2**：L9 上 progress+hedge 支配恒填充与恒沉默；短时延档 silence 不劣于 filler。
- **屏障**：L4 层 barrier-off 显著劣化（RB 版 held-out 复验；FDB 上 n=2，这里 L4 test ≈65 夹）。
- **DAG**：dag-off 在 L7/链式 L5 劣化；dag-on 的 stale 传播计数随行报告。
- **L10（新发现，dev3 定量雏形）**：TACT 窗口开放期第三方命令穿透率 vs blocking 的
  结构性免疫——SV 门控（蓝图 §4.6）的动机定量，主表单列 benign-穿透/adversarial-拒绝双率。

## 四、主表模板

行 = 系统臂（含 oracle 上界行）；列 = exact / state_norm / U / first_p50 / done_p50 /
wrong_commits / unrepaired / comp_cost / by-layer L1–L10。臂 A 主表 + 臂 B 副表 +
消融差分表（±barrier、±dag、δ 网格曲线、FC 三档前沿）。判读纪律照旧：逐条件格
配对计数并报，n<30 的格不进结论句。
