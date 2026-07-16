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

---

## 五、第一批判定（2026-07-16 判读层；协议正文 §一–§四 未动一字）

> 输入：`c29c05f` 归档（15 provider 单发、0 重跑、离线复放 15/15 逐字节一致，
> 机器收据 `exp/rb/build_v2/rb_test_receipt.json`）。
> **独立核验声明**：headline+by-layer（15 provider × 8,656 逐夹行）、屏障/DAG/主臂配对计数、
> δ 曲线、FC 三档决策逐位同一性与 tier 计数、臂 B 分层、oracle gate 层、revision-only 切片
> 均已在镜像容器由归档行独立重算，**与 report/收据逐位一致**（p50 口径 = runner 的上中位数
> `v[n//2]`；用平均中位数会差 ±0.005–0.016，已对齐）。L10 参数流双率与 DAG trace 计数需要
> 音频/沙盒重放（音频不入仓），依收据的 0-miss 逐字节复放证明采信，未在容器内二次重算。

### 5.0 判定快照

| 冻结假设 | 判定 | 一句话 |
|---|---|---|
| 屏障（L4 off 显著劣化） | **支持** | L4 配对 6/0（精确二项 p=.031）；L3 9/0（p=.0039）；全体 44/2（McNemar 精确 p=3.1e-11）；且 nobar .1885 < blocking .2181 = **无屏障的窗口是净负资产** |
| DAG（L7/链式 L5 劣化） | **未获支持** | L5、L7 on/off 逐位同（配对 0/0）；净效应 +3/0 落在 L3×1/L4×2 = 预注册法庭选错层 |
| H-B1（失败集中 in-flight，McNemar） | **不可合法计算** | 生成器无跨状态配对键（仪器缺口）；描述率方向相反（失败质量集中 pre-call），但 **TACT−blocking 差分**集中 in-flight（armB L8 配对 34/0，p=1.2e-10，n=103 可进结论句） |
| H-FC1（L9 上 progress+hedge 支配两恒策） | **不可判**（仪器灵敏度空结果） | outcome 轨构造性档间不变（决策逐位同已核验）；hedge/commit 档在 launch-time-only 钩子下结构不可达；L9 上连 first 锚都档间不变（62/65 夹 post-user-end 零 say → first=done 回退） |
| H-FC2（短时延档 silence 不劣于 filler） | **平凡成立**（无信息量） | outcome 轨以构造等式成立；短 eta 档 first 代价上界 = eta ≤ 1.0s |
| L10 双率（协议要求单列） | **描述性方向证据**（n=12–14 < 30，按 §四纪律不进结论句） | armA 交换形态：TACT 良性穿透 .615 vs .077，第三方拒绝 .500 vs .929；armB TACT 双轴均不占优（.333/.154 vs .417/.692）——开放窗的安全代价跨臂一致，良性收益仅固定时间轴臂兑现；SV 门控（蓝图 §4.6）动机定量成立 |

### 5.1 主表（test；exact / state_norm / U / first50 / done50；wrong_commits、unrepaired、comp_cost、fees 全 provider 恒 0）

**臂 A（n=541）**

| 系统 | exact | state_norm | U | first50 | done50 |
|---|---|---|---|---|---|
| TACT δ1.5 | **.2662** | .2828 | .1996 | **1.640** | 3.889 |
| blocking | .2181 | .2440 | .1410 | 2.176 | **2.302** |
| TACT −barrier | .1885 | .2052 | .1348 | 1.640 | 3.637 |
| TACT −DAG | .2606 | .2773 | .1956 | 1.640 | 3.869 |
| oracle（窗口化上界） | .7763 | .7763 | .5943 | 1.640 | 4.092 |

**臂 B（n=356，反应式）**

| 系统 | exact | state_norm | U | first50 | done50 |
|---|---|---|---|---|---|
| TACT δ1.5 | **.2781** | .3146 | .2078 | **1.640** | 4.050 |
| blocking | .1910 | .2303 | .1342 | 1.967 | **2.268** |
| oracle | .7865 | .7865 | .5944 | 1.640 | 4.338 |

**δ 网格（臂 A TACT）**：exact @0/.6/1/1.5/2 = .1922/.1922/.2366/.2662/.2865 **单调不减**；
done50 = 2.150/2.790/3.377/3.889/4.503 ≈ 线性 +δ；**first50 恒 1.640 逐位平**（= 0.64 hold +
1.0 名义 infer 地板）。FDB 的两条定律在 RB 复现：**首响与 δ 解耦**、**完成保费 ≈ δ**
（A：3.889−2.302=1.587≈δ；B：4.050−2.268=1.782）。δ=0 (.1922) ≤ blocking (.2181)：
不开窗的 eager TACT 不优于 blocking，窗口才是收益来源——与 FDB eager 惩罚同构。

### 5.2 主臂配对差分（TACT-only / blocking-only；精确二项）

| 层 | armA | p | armB | p |
|---|---|---|---|---|
| ALL | 41/15 | **6.9e-4** | 36/5 | **7.8e-7** |
| L1 | 0/0（通过集逐位同） | — | — | — |
| L2 | 0/0（通过集逐位同） | — | — | — |
| L3 | 9/0 | **.0039** | — | — |
| L4 | 6/0 | **.031** | 1/0 | 1 |
| L5 | 1/0 | 1 | 0/3 | .25 |
| L6 | 1/0 | 1 | 0/1 | 1 |
| L7 | 11/0 | **9.8e-4** | — | — |
| L8 | 6/11 | .33 | **34/0** | **1.2e-10** |
| L9 | 5/1 | .22 | 0/0 | — |
| L10 | 2/3 | 1 | 1/1 | 1 |

读法：
- **L1/L2 通过集逐位同 = harness 对齐锚**：修订先于唯一 EoU ⇒ 两系统喂给决策器的
  消息逐位同 ⇒ 缓存决策逐位同。系统间一切差分都来自窗口机制，不来自 prompt/解析噪声。
- **armA 显著差分格 = L3/L4/L7**（窗口/竞态/补偿=事务机制层）；**armB = L8**（在飞
  事件层，34/0）。L8 在两臂**反向**：armA（脚本化在飞投影）blocking 反而 +11/−6，
  armB（真反应式）TACT 34/0——在飞处理的分化只有反应式基准测得出来，这正是臂 B
  存在的理由（也是"（据核验）第一个以 SUT 自身工具生命周期触发用户事件"差异化的实证）。
- armB 分层（描述性，action 混杂）：pre-call-eou n=153 上 TACT .0065 / blocking .0261
  （地板格，两系统近全灭）；in-flight n=203 上 .4828 / .3153。revision-only 单事件切片：
  pre n=103 = .0097/.0291，in-flight n=36 = .1944/.0000。

### 5.3 逐假设判读补充

**屏障**：预注册方向（L4 劣化）兑现；且劣化不限 L4——off 后 L7 .3864→.0682、
L8 .2982→.1228、L3 .1525→0（决策在飞期间窗口照常落地 ⇒ patch/补偿目标先行提交）。
结构结论：**窗口的全部价值以 decision-atomic commit 为前提**；W3 D3"屏障=显式设计"
裁断在 RB held-out 上定量兑现（FDB 上该消融只有 n=2 证据格，这里 44/2）。

**DAG**：机制活性有据（0-miss trace 重建：219 edge / 29 stale / 145 夹有事件；
3 个 on-only 翻转全落 L3/L4 = stale 重启窗救回竞态修订），但预注册法庭空转：
L7 生成为独立 multi-call（0 边可断）、L5 失败质量在 DAG 裕度之下（LLM .0122，
oracle 上界也只 .122——链的第一环已死，轮不到 stale 传播救）。**如实报 not supported**；
若未来有 v2.3（公开勘误+版本递增），设计教训 = 链式修订须显式注入 L5/L7 场景蓝本。

**H-B1**：预注册 McNemar 的合法输入不存在（修复属生成器版本递增项：为每个 revision
场景铸 pre/in-flight 孪生对）。可进结论句的替代事实：**armB L8 配对 34/0（n=103）**
——"in-flight 生命周期是系统间分化的主战场"以差分形态成立；"失败质量集中 in-flight"
的原表述在描述率上反向（pre-call 地板 .0065 更低），不得引用为已验证假设。

**H-FC1/FC2**：三档 provider 决策缓存哈希同一（2b1f7aa3…）+ 决策 ops 逐位同已核验 =
"FC 只动 say 层"的构造保证在 test 上成立（这是 W5-FC 的合格性结论）。判别失败的三重
仪器根因：① hedge/commit 需 `result_known`/`elapsed_s`，RB 钩子是 launch-time-only ⇒
结构不可达（v1 实选 tier：silence 377 / filler 228 / progress 64，L9 全 progress = 策略
按设计运转）；② 承诺-修复轨全程 0 事件 ⇒ U 的修复/费项惰性，U 实际退化为
1[终态]·γ^done；③ L9 的 first 锚在 62/65 夹上回退为 done（修订落在长工具等待期内 ⇒
post-user-end 无 say）。**事后描述性双工面（post-hoc，非预注册判定）**：
say 起点落入用户语音段的 talk-over 率 = filler 207/745=**27.8%** / v1 52/368=**14.1%** /
silent 0/76=**0%**（blocking 0/82=0%），对应 first50 = 1.640/3.352/3.818（/2.176）——
**(首响, talk-over) 平面上三档构成单调前沿，v1 为内点，无支配关系**；FDB-v3 实测
Ultravox 恒填充 47.9% 撞用户的现象形态在 RB 复现（27.8%，say 起点口径、系统句无时长、
脚本用户不回避——三条口径注记必须随引）。FC 判别轴的复活需要臂 B 反应式 FC providers
或双工轨（新预注册，不在本批范围）。

**L10**：双率定义 = 参数流（良性=本人修订标量进入已执行沙盒调用；对抗=旁观者命令
标量未进入任何已执行调用），非 whole-episode exact 代理。whole-episode L10 exact：
armA .100 vs .125（配对 2/3）、armB .0789 = .0789（1/1）——n≥38 但无差分信号；
信号在参数流层。跨臂一致的部分：**TACT 拒绝率两臂均大幅低于 blocking（.500/.154 vs
.929/.692）= 开放异议窗对第三方语音的结构性暴露**；不一致的部分：良性穿透收益
armA 兑现（.615 vs .077）、armB 反而 .333 vs .417。SV 门控是下一个机制层的
动机——描述级、n<30，论文措辞维持"定量雏形"。

### 5.4 附带发现与诚实清单

- **oracle 上界 vs LLM**：A .7763/.2662、B .7865/.2781（比值 ~.34/.35）——bench 对更强
  模型有巨大区分空间；oracle gate 层全 1.0 已核验（L1/L2/L3/L4/L7/L9/L10-A；
  L4/L8/L9/L10-B），天花板层 L5 .122/.082、L6 .227/.184 = 窗口约束下的机制天花板。
- **三条死仪器如实入册**（对标 w4v3 amend-v0 塌缩纪律）：承诺-修复轨（0 commit 事件）、
  补偿费轨（comp_cost/fees 全 0，dag_comp_plan=0）、L9-FC first 锚（回退 done）。
- 收据三条异常均为非 provider 级（TTS 预热进程生命周期/冻结命令 --tts 物化/客户端
  中断不落进程），单发纪律无破口；`--tts qwen` 物化由 runner 断言强制且 0 正式合成，
  合法。
- 臂 A n=541 的 ±1 夹 ≈ ±0.18pt；主臂差 +4.8pt（armA）/+8.7pt（armB）对应配对
  41/15、36/5，均越过精确检验阈——**RB 主结论不是 FDB n=100 那种刀口差**。

### 5.5 主叙事落点（进论文的五句话）

1. 事务窗口在固定与反应式两臂、全 δ 网格上单调正收益，且首响一分不付
   （first50 恒 1.640 = 地板）；完成保费 ≈ δ，与 FDB 跨基准复现。
2. 窗口价值以提交屏障为前提：去屏障后 TACT (.1885) 劣于 blocking (.2181)
   ——"可修订"若无原子性反而净伤。
3. 反应式在飞事件层（armB L8 34/0）是全双工工具调用的分化主战场，
   且只有以 SUT 生命周期触发事件的基准测得出来。
4. 开放异议窗有可定量的安全代价（第三方命令拒绝率 .500/.154 vs .929/.692），
   SV 门控为下一机制层。
5. 仪器诚实清单（H-B1 配对键缺失、DAG 法庭选层、FC 判别轴死仪器、L5/L6 近地板）
   全部进 limitations/design-lessons。

**第二批**：learned-head 臂（v2/C0/C1 π 点）待 RB 适配层 + selftest 交付后按 §一同纪律
单发（评测 only，任何 RB 观测不得回流训练/调参）。
