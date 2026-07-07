# W3 D4–D6 交付与 GPU 出数交接（2026-07-07）

> **已兑现（2026-07-08）**：§3 全部 GPU 项已跑完并逐文件核验，结果与勘误见
> `手工文档/神谕/08_W3 执行汇报.md` 与 AGENTS.md 当前状态。本文档保留为跑前契约存档。

> 本批为**纯代码/文书交付**（单线程产出）。GPU 出数项在 §3，按序执行即可；
> 每条命令给出门槛与核验方式。零 GPU 能算的数字已在 §2 出讫。
> 上游 PR ×2 **已取消**（仓库仅本地，用户 7/07 明示）——per-instance RNG 已在
> `latency_realistic` 本地实现，finance_14 pop(0) 对齐语义只留文档记录。

## 1. 新增机制一览（全部默认关，冻结路径逐位不动）

| 模块/flag | 内容 | 验收 |
|---|---|---|
| `src/latency_realistic.py` | 现实延迟档：κ 类对数正态（标定+预注册 `docs/latency_calibration.md`）；per-instance sha256 播种；blocking 串行 / TACT DAG 并行完成锚记账。**纯记账层：决策与缓存 100% 复用** | `tests/test_w3_d456.py` §A（校准 ±15/20%） |
| `src/tact_dag.py` + `--dag on` | DAG 传播（patch 命中被依赖字段 → pending 下游重参数化/置 stale + 窗重启；committed 下游 → 补偿计划）+ 补偿注册表（κ-忠实 reverse 模板 + 幂等键 at-most-once） | §B（12 项） |
| `src/normalize_entity.py` | norm-v1 封闭规则集（N1–N8：序数/连字符/所有格/冠词/复数/数值），双侧对称，**不含别名扩展**（Vegas 是模型的活） | §C + 自检 |
| `w2r_state_track.py` 双报 | verbatim + normalized 双轨输出（额外字段，verbatim 路径逐字节不变） | 已跑，见 §2 |
| `scripts/w3_state_classify.py` | 状态轨失败分类台账（8 类 + 蓝图#8 触发判定） | 已跑，见 §2 |
| `engine_b` + `--speculative on` | 投机派发：vad end 即派发决策，EoU 确认前 inert（stash/作废/finalize 释放三路径全测）；t_dec = seg_end + max(0.64, infer) | §E（4 组），屏障互作/尾释放含 |
| `engine_b` `tts_split` + `floor_holding` | 分句 TTS（TtsSentDone 事件、first_sentence 锚点仪器）+ floor v0（narration 无条件 yield；confirmation 且 commit 临近 finish_clause；granularity=句） | §F |
| `tact_core` `--prompt v3` | 五靶批次（`docs/prompt_v3_five_targets.md`），engine_cfg `prompt:"v3"` live 同源 | 30 子集协议见该文档 |
| `--ids-file` | 子集驱动（`exp/w3/tuning30.json` 预注册 30 子集 = 21 rollback + 9 证据夹） | — |
| `scripts/w3_realistic_rescore.py` | 既有档案的现实档零 GPU 重打分 + 预注册对表（不改档案） | 已跑，见 §2 |

回归证明：`tests/test_commit_barrier.py` 37/37 ✅、`tests/test_w3_d456.py` 89/89 ✅、
`scripts/w3_barrier_probe.py` VERDICT MET（on 12/17 / off 10/17 / parity 21/21 逐位 / **0 cache miss**）✅。

## 2. 零 GPU 已出数字（本批新结论）

**状态轨双报（对 W2 冻结档，官方 exact 不动）：**

| 臂 | verbatim | normalized(norm-v1) | 救回 |
|---|---|---|---|
| w2r_tact_full | 57/100 | **76/100** | +19 |
| w2r_sblock_full | 57/100 | 75/100 | +18 |
| w2r_blocking | 57/100 | 74/100 | +17 |

P2'（差分制）normalized 下依然 TACT ≥ blocking（76 vs 74）。校准矩阵 FT=0 保持（无倒挂虚增）。
分类台账（`exp/w3/state_classify_*.{json,md}`）：TACT 失败 53 项 = format 24 / missing_call 7 /
canonicalization 6 / true_param_error 4 / dynref 4 / asr_mishear 4 / benchmark_attr 2 / missing_field 2。
blocking 的 missing_call 11 > TACT 7（瘫痪不是 TACT 特有，blocking 更重）。
**蓝图#8 试点触发条件 TRIGGERED**（asr_mishear ≥3 两臂皆是）——是否立项最小试点待裁断。

**现实档预注册对表**（`w3_realistic_rescore.py --compare w2r_sblock_full w2r_tact_full`，100 夹配对，跑后预测未动）：

| 预注册 | 结果 | 判 |
|---|---|---|
| P-1 首响比：全量 ≤0.50 / 写类 ≤0.30 | 全量 **0.592** ✗ / 写类 **0.237** ✓ | 全量差 9pt（read_lookup 主导场景压比值）；写类强达标——"收益随工具时长放大"主张以写类分层形态成立，如实报 |
| P-2 完成锚保费 = δ*（档位无关定律） | **1.500** | ✓ 精确命中 |
| P-3 链式条件式（TACT≤blocking iff Σ−max ≥ δ） | **31/33**，wrong=2=门上限 | ✓ 过；2 个 MISS（eco13/fin14）系两臂调用集不同（轨迹差异）非条件式失效 |
| P-4 TACT 首响档位不变性 | 93/100 | ✓（7 例 = say 空 fallback，与 D1.5 的 10/99 同源） |

**现实档单臂头条（rescorer，既有 A 档轨迹）**：TACT 首响 p50 **1.144**（写类子集 1.245）/
完成 3.329；sblock 首响=完成 **1.997**（写类子集 **5.566**）。写类子集比值 1.245/5.566 ≈ 0.224
与 P-1 配对 p50 0.237 相合。

**R18 缓存键审计**：key = sha256(messages)；messages = SYSTEM_PROMPT + 音频前缀 b64 + 快照文本。
δ、延迟档、barrier 均不以文本形式进 prompt ⇒ 无非法耦合。快照内容对 δ/investigative 臂的依赖是**轨迹性**的
（committed 集随窗到期变化），属合法臂间差异。现实档确证零污染（探针 0 miss）；v3/spec/dag 臂的键变化是设计内的。

## 3. GPU 出数清单（按依赖序）

环境提醒：跑 evaluator 前 `unset all_proxy ALL_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`；
正式延迟数字 A 档串行专机；`PY=/root/miniconda3/envs/fd-sds/bin/python`。

### 3.1 现实档 30 子集冒烟（双模式，验证 driver 路径 ≡ rescorer）

```bash
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w3r_tact_d150 \
    --latency-profile realistic --ids-file exp/w3/tuning30.json
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w3r_sblock \
    --mode blocking --latency-profile realistic --ids-file exp/w3/tuning30.json
```
预期：近全缓存命中（分钟级）。核验对象 = **播种不变量**：新跑 result 的
`latency_realistic.per_op[].lat_s` 与 `exp/w3/realistic_w2r_*.json` 中同夹同
(fn,args,occurrence) 的 lat_s **逐位相等**（per-instance 播种保证）。
注意完成锚**不要求**逐位等于 rescorer——缓存里记录的 infer 含 W2 并发跑的墙钟污染，
新跑轨迹的 t_commit 会漂（7/07 冒烟实测：resp 4.66 vs 档案 1.14 即此效应）；
统计核验用 P-2 保费 p50 ∈ [1.3,1.7] 复现即可。正式数字一律 A 档 --workers 1 出。

### 3.2 现实档全量 + δ 网格（A 档，确定性双跑）

```bash
for d in 0.0 0.6 1.0 1.5 2.0; do
  $PY scripts/w2r_stream_replay.py --delta $d --provider w3r_tact_d${d/./} \
      --latency-profile realistic --workers 12   # 准确度轨；缓存跨点复用
done
# 出数档：--workers 1 重跑 δ*=1.5 两遍，确定性字段逐位对账
```
门：P-1..P-4 对表结论与 §2 一致（同一算术）；双跑逐位一致（E7）。

### 3.3 投机派发 A/B + HumDial 门（D6）

```bash
# 离线 A/B（部分新决策：hold 期内到期 op 在快照中仍 pending）
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w3spec_tact_d150 --speculative on
# 对照差分：pass 不降（gate Δ≥−1 场景）+ 首响分解重算（w3_first_decomp 口径）
# 预期（D1.5 裁断 B 投影）：首响 p50 1.141 → ~0.64+ε；P3 比 0.786→0.44 级
# live 标准卡复测：configs 里 engine.speculative_dispatch: true 起 backend，W2 串行 live 流程
# HumDial 门：flag off 跑一遍确认零扰动（应逐字节同 W3 D1 基线）；flag on 跑 100，Δ≥−1（同日对称三判）
```

### 3.4 DAG 链式冒烟（28 链式场景）

```bash
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w3dag_tact_d150 --dag on \
    --latency-profile realistic
# 核验：result 的 dag.events 非空（链式夹）；官方 exact 差分 ≥−1；
# comp_plans 仅计划不执行（官方轨零补偿调用）
```

### 3.5 prompt v3（协议详见 docs/prompt_v3_five_targets.md）

```bash
$PY scripts/w2r_stream_replay.py --delta 1.5 --provider w3p3_tact_d150 \
    --prompt v3 --ids-file exp/w3/tuning30.json          # 全新决策，30 子集先行
# 门过 → 冻结 → 全量双臂 + 双档 δ 网格重生成 + HumDial 门（同日对称判）
```

### 3.6 分句 TTS 完成锚增益（E5，live）

backend 配置 `engine: {phase: b, tts_split: true, floor_holding: true}`，
真 websocket 走 10 条演示夹；测量 = trace `tts_sent_done{first_sentence:true}` 时戳 vs
整句 TTS 基线的 `tts_done`；barge-in 场景核验 `tts_sent_dropped`/`floor_decision` 事件链。

## 4. 已知边界（诚实清单）

- 投机派发 core 模式是算术模拟（离线不产生"浪费的投机调用"）；真实 GPU 浪费率只有 live A/B 能测。
- floor v0 的实际执行粒度=句（send_bytes 后无法召回客户端音频）；"yield"=不再发后续句。
- 现实档对 **w2r_blocking**（W1 旧管线档，无 trace）不可重打分——延迟对比臂一律用 sblock（同驱动器）。
- eco13/fin14 的 P-3 MISS 属两臂调用集差异，链式条件式对"同调用集"前提外的夹不设防。
- 分类台账的 canonicalization 检测用别名表/词元包含**仅作检测**，永不进判分器。
