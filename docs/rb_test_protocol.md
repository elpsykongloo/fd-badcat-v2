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

---

## 六、第二批：学习头臂（协议冻结 2026-07-16；在任何 batch-2 test 观测之前写定）

> **runner v1.1（增量勘误，按 §一纪律登记）**：`scripts/rb_run.py` 新增
> `--delta-policy learned:v2 / --stophead-model / --finality-cache` 与
> `rb/learned.py` 适配模块（stophead.REQUIRED_ARGS ← rb.registry.TOOLS[fn]["required"]，
> 工具名与 FDB 零重叠断言；κ 经 install_rb 既有注入）。**默认路径逐位不动的证明**：
> 扩展后 runner 以 `--delta-policy fixed`（默认）重放两个 oracle test provider，
> **897/897 逐夹行与归档 `rbtest_oracle_a/b` 逐字节相同**、报告除 provider 名外相等；
> runner selftest 14→21（新增 7 项：protect-all ≡ 固定 δ1.5 结构等式、commit-now 丢
> L4 救援、确定性双跑、冻结路径无审计键等）。**batch-1 结果不受影响、不重跑。**
> scorer/sandbox/registry 三文件未动，freeze v3 哈希不变。

### 6.1 评测对象与红线

冻结模型 = W4 归档原件，**逐字节不改**：`exp/w4/stophead_v2.json`（θ=.03）、
`exp/w4v3/stophead_v3c_pi{020,040,060,080,100,150,200,300}.json`（C0 先验移位）、
`exp/w4v3/stophead_v3c1_pi{020..300}.json`（C1 真实停顿锚重训）。全部 twostage
（w_protect=1.5 / risk_horizon=2.5；θ 随 π 单调降已验）。**评测 only**：RB 任何观测
不得回流训练、θ/π 选择或模型挑选；不做 RB 侧点预测校准。finality = 冻结
FINALITY_PROMPT 在音频尾 8s 上的 Omni 判读（独立缓存，不进决策消息 ⇒ 决策缓存
键口径不变）。

### 6.2 provider 清单与运行序（各一次，§一单发纪律全承接）

臂 A（test 541，`--split test --arm A --system tact --delta 1.5 --decider llm
--input audio --delta-policy learned:v2`）；缓存种子 = 前一 provider 决策缓存副本
（首个种自 `decision_cache_rbtest_tact_d150.json`）；finality 缓存共用一份
`finality_cache_rb_a.json`（臂 A 音频 provider 不变式）：

| # | provider | --stophead-model |
|---|---|---|
| 1 | `rbtest_lh_v2_tact` | exp/w4/stophead_v2.json |
| 2–9 | `rbtest_lh_c0_pi{020,040,060,080,100,150,200,300}_tact` | exp/w4v3/stophead_v3c_pi*.json（π 升序） |
| 10–17 | `rbtest_lh_c1_pi{020,...,300}_tact` | exp/w4v3/stophead_v3c1_pi*.json（π 升序） |

预算许可下跑满 17；若须裁减，**在第一个 provider 启动前**声明前缀截断
（只允许按上表顺序截断，不允许挑点）。臂 B 学习头 provider 代码已支持但
**不在本批**（跑它 = 新预注册）。每份 report 自动携带
`config.stophead`（版本/θ）与 `learned_windows`（protect_rate）。

### 6.3 冻结判读（仅此三条进判定句；其余全部描述性）

- **H-LH1a（机械）**：同族内 protect_rate 随 π 单调不减（θ_eff 单调降的直接后果；
  违反 = 适配层 bug，不是科学结果）。
- **H-LH1b（经验）**：同族内 exact 随 π 弱单调升、趋向固定臂 .2662
  （protect-all 结构等式已由 selftest 钉死）。
- **H-LH2（W4 判决迁移，主判读）**：**没有任何学习头臂对固定 δ1.5 臂
  （`rbtest_tact_d150`）取得配对净优势**（learned-only vs fixed-only，
  精确二项 p<.05）。预测 = 不发生；若发生 = W4 "G2 未立"结论不迁移，
  属重大更新，如实入册并重启学习线讨论。
- **H-LH3（经验方向）**：逐 π 配点上 C1 protect_rate ≥ C0（FDB 上已现方向；
  真实停顿锚 ⇒ m_train 更小 ⇒ θ_eff 更低）。

已声明风险（引用即须随注）：RB 修订基率（.673 episode 级）远超 π 网格支撑
[.02,.30] 与两模型的教学/真实先验——头的风险标度在 RB 上先验错配是**预期内**
现象，任何"标度错了"读数都不构成重训理由（防火墙）。

### 6.4 收官

与 §一同：停服 → 全 provider 离线复放（决策+finality 双缓存 0 miss、报告/缓存/
逐夹目录逐字节）→ `rb_test_receipt_batch2.json`（provider 清单/缓存计数/
异常登记/复放结论）。判读（H-LH1–3 + π 前沿表 + 与 batch-1 主表合并）交回
判读层，写入本文件 §七。

---

## 七、第二批判定（2026-07-16 判读层；§六冻结判读的正式裁决）

> 输入：`329ae38` 归档（17 provider 单发、0 重跑、无前缀截断；427 决策 miss + 934
> finality miss = 1,361 次正式请求全 200；停服后 17/17 离线复放双缓存 0 miss、
> 报告/缓存/9,197 份逐夹结果逐字节一致；收据 `rb_test_receipt_batch2.json`）。
> **独立核验**：17 provider 的 headline/by-layer/protect_rate（由逐夹行的
> op_windows 重算）/对固定臂配对计数/精确二项 p 值，均在镜像容器由归档行独立
> 重算，**与收据逐位一致**；音频复放层沿批一惯例采信收据的逐字节证明。

### 7.1 四条冻结判读的裁决

| 判读 | 裁决 | 数字 |
|---|---|---|
| H-LH1a（机械单调） | **PASS** | 两族 protect_rate 随 π 弱单调（C0 .1054→1.0；C1 .1351→1.0）——适配层无 bug |
| H-LH1b（经验单调） | **PASS** | 两族 exact 弱单调收敛 .2662；protect=1.0 的 5 个点与固定臂**通过集逐位相同** = protect-all ≡ fixed 结构等式在 test 上端到端兑现 |
| **H-LH2（主判读）** | **预测兑现** | **0 个学习头臂对固定臂有显著配对净优势**；10 个显著更差（低/中 π，p≤7.2e-3）、7 个通过集逐位相同（v2 + 两族 π≥.15/.20）——**W4"G2 核心证据未建立"零训练迁移到第二基准成立** |
| H-LH3（跨族方向） | **未获逐点支持** | 含平 6/8；π=.04/.06 违例（C1 .1895/.2495 < C0 .2344/.2559）尽管 θ_C1 < θ_C0——**风险场（权重）差异盖过阈值排序**：C1 真实停顿锚重训改变了 RB 特征上的 risk 分位质量，跨族比较隐含的"同风险场"假设不成立。族内单调（H-LH1a）双 PASS 证明这是权重层信息，不是适配层缺陷 |

H-LH2 的完整语义：RB 是**为窗口价值最大化而设计**的基准（episode 级修订率 .673、
修订按可救援 horizon 分层投放），学习头在这里仍拿不到对固定窗口的 outcome 优势
——比 FDB 上的同判决更强。已声明风险按预注册兑现：标度错配读数不构成重训理由。

### 7.2 π 前沿主表（臂 A test 541；first50 全部恒 1.640 = 第三次复现首响与窗口政策解耦）

| 点 | protect | exact | done50 | U | 对固定配对 (lh-only/fx-only, p) |
|---|---|---|---|---|---|
| **fixed δ1.5（参照）** | 1.0（定义） | .2662 | 3.889 | .1996 | — |
| **δ0（参照）** | 0 | .1922 | 2.150 | .1493 | — |
| v2（θ=.03） | .9939 | .2662 | 3.874 | .1998 | 0/0（逐位同） |
| C0 π=.02 | .1054 | .1922 | 2.273 | .1493 | 6/46, 1.0e-8 |
| C0 π=.04–.10 | .2344–.2690 | .1941 | 2.75–2.95 | .1493–.1495 | 6/45, 1.8e-8 |
| **C0 π=.15** | **.7764** | **.2662** | **3.636** | **.2029** | **0/0（逐位同）** |
| C0 π=.20/.30 | 1.0 | .2662 | 3.889 | .1996 | 0/0 |
| C1 π=.02/.04 | .1351/.1895 | .1922 | 2.30/2.40 | .1493/.1492 | 6/46, 1.0e-8 |
| C1 π=.06/.08 | .2495/.2690 | .1941 | 2.80/2.95 | .1493 | 6/45, 1.8e-8 |
| C1 π=.10 | .4944 | .2421 | 3.378 | .1844 | 4/17, .0072 |
| C1 π≥.15 | 1.0 | .2662 | 3.889 | .1996 | 0/0 |

### 7.3 结构读数（判读层新增，全部由归档行重算）

- **价值捕获账本**（对 δ0 锚 +40 夹 = 固定窗口的全部可得收益）：protect .19–.27
  的五个点只捕获 **+1/40（2.5%）**；C1@π.10 protect .4944 捕获 **+27/40（67.5%）**；
  protect ≥.776 才收全。价值不随保护预算成比例——**集中在窄 risk 带**（C0 在
  θ∈[.0359,.0558) 之间 protect .269→.776、C1 在 [.0205,.0322) 之间 .494→1.0 =
  FDB"质量悬崖"同构第三现场），带内排序基本不载信息（顶部四分位只含 1/40）。
- **L7 反窗口夹（勘误版机制，2026-07-16 当日修正）**：低/中 π 点对固定臂的
  learned-only 通过**全部落在 L7**（π=.02 的 6/6、C1@π.10 的 4/4）；低 π 点与
  δ0 的通过集也非嵌套（±3 互换）。**初版判读把机制猜成"cancel-in-window 吞掉
  pending"——逐夹轨迹证伪，勘误如下**：6 夹形状完全一致（A_L7_0003/0019/0031/
  0039/0040/0044），修订语义指向**第二步的槽位**（country→check_visa_rule、
  item_id→check_stock），但固定窗口下决策器把它 **patch 进了仍开着窗的第一步
  search op**（逐夹 ops 可见 `patch op_id=2`）——**修订目标归属错误
  （revision-target misattribution）：开着的窗口成为歧义修订的错误吸附面**。
  commit-now 恰好移除了这个错误吸附面（无 pending 可 patch ⇒ 修订只能进新
  launch），故低 π 点"更对"；blocking 同样失败（单发合并时也归属错），说明归属
  错误是模型级缺陷、窗口只是放大器。这仍是全基准唯一"提交更快 = 更对"的格
  （16 个 window-helps 夹 = 归属正确的真救援，与 6 个 anti-window 夹同层互斥），
  但正确结论从"需要区分修订/撤销意图"修正为：**需要修订目标归属（哪个 op 的
  哪个字段）的准入控制**——已按用户 7/16 裁定升级为 W6 in-paper 机制层
  （`docs/w6_admission_design.md`），不再是 future work。
- **效率边际 = 学习头在 RB 上的唯一正贡献（描述性，点选择非预注册）**：
  **C0@π=.15 通过集与固定臂逐位相同，22.4% 的 op 立即提交，均值完成保费
  4.120 vs 4.560（−0.44s，p50 −0.253s），U .2029 > .1996** ——预注册效用度量上的
  严格弱帕累托改进。v2（θ=.03）是微型版：0.6% 立即提交、0 翻转、done50 −0.015。
  **与 FDB C0@π.30 无损弱支配 fixed 同构 = 效率边际的存在性跨基准复现**（无损点
  的 π 位置基准相关：FDB .30 / RB .15，引用时必须成对报告，不得单点引用）。
- **v2 读数迁移注记**：同一 θ=.03 在 FDB 读出 protect .84，在 RB 读出 .9939——
  RB 特征分布（更短 utt、更高 slots_missing 密度）把几乎全部 op 推过 .03 风险线；
  这与 7.1 的 H-LH3 违例同源（风险绝对标度不跨域），一并构成"标度迁移误差"的
  第二基准证据。

### 7.4 学习线终局（进论文的话）

批二把 W4 的判决升级为**双基准外部效度**：意图稳定性不可学（W4 五代）、
finality 仅受控离线可分（w4v3 探针 0.984）、部署时点时序/ASR 内容双不可达
（SG 双探针）、**且冻结头的排序在修订密度 8 倍于 FDB 的第二基准上仍无一点
胜过频率盲的固定窗口（0 显著优/10 显著劣/7 逐位同）**。幸存的全部正贡献 =
先验移位形态的**同 outcome 保费效率**（FDB C0@π.30 ↔ RB C0@π.15 跨基准复现）。
δ=1.5 固定窗口 + 提交屏障作为方法核心的选择，至此有了完整的否定侧证据链。
L7 反窗口夹（7.3 勘误版）是唯一被实证的"超越固定窗口需要什么"：不是更好的
时序风险排序，而是**修订目标归属**（这个修订绑到哪个 op 的哪个字段）的内容
理解——连同 L10 的说话人轴（谁有权修改），构成异议窗的**准入控制**问题，
已按用户 7/16 裁定立项 W6（in-paper，见 `docs/w6_admission_design.md`）。

**test-897 全部窗口（批一 15 + 批二 17 = 32 providers）已耗尽；RB v2.2.1 上不再
有任何合法 test 运行。后续任何评测 = 新系统版本或新构建版本 + 公开勘误纪律。**

---

## 八、v2.2.1 已发布结论勘误（2026-07-16；外部评审核验后，全部零重跑）

> 依据：外部评审逐条实证核验（`rb_design.md` §15.1，本容器代码+归档双向复算）。
> 勘误只重述已发布行的含义，**不改动任何归档产物**；修复全部落在 rb_v2.3.0
> （新 bench 版本，test 纪律重新起算）。

1. **臂 B L4/L5/L6 by-layer 行作废（含 §五 5.1 副表的分层引用）**：臂 B 音频按
   stub 时长排期、真实 TTS 只回填终点 ⇒ 37/356 行存在用户语音自我重叠、B_L4
   实测 gap 落声明 bin 仅 5/54。重分箱勘误表 = `exp/rb/errata_b_rebin_v221.json`
   （L4 in-bin 5、L5 27、L6 14；in-bin 子集两臂 exact ≈ 0——这些行考的不是标签
   声称的东西）。**臂 B headline（.2781/.1910）与 L8/L9/L10-B 行不受影响**
   （L8/L9/L10 内容仅经事件注入；重叠 6 夹 L10 行两臂同规同权，方向结论不变）。
2. **差异化措辞收窄**："修订时机受控分层"限定**臂 A**；"用户事件以 SUT 工具
   生命周期触发"限定 **L8/L9/L10-B**（L4/L5/L6-B 的修订实为固定脚本件+回声，
   committed 锚从未触发）。§五 5.2 中 armB L8 34/0 主结论**安全**（纯事件注入层）。
3. **blocking 口径注记**：blocking = 决到静默 + 无工具结果回流 ⇒ 修订上占便宜
   （决策前听得到全部修订）、链上吃死亏（L3–L6 必须一次性符号引用整批发出）。
   主臂差 +4.8pt(A) 含此约定成分；屏障主张不受影响（TACT 自体 ±barrier）。
   per-EoU blocking 变体列 v2.3+ 候选。
4. **"补偿/在飞代价模型考场"的说法从论文叙事中移除**：v2.2.1 交付物中结构性
   缺席（reverse 类 10 工具零 gold 覆盖、无执行期状态、comp/fees 32 provider
   恒 0 是构造后果）。v2.3 以 L7 补偿考场 + abort 原语 + reverse 净额语义补齐。
5. **域=语言混杂入 limitations**：v2.2.1 上任何按域差异不可与语言分离
   （ecommerce/housing 全 zh、finance/travel 全 en）；v2.3 已解耦。
6. **安全清单确认**（评审 §四采纳）：屏障主张（±barrier 自体消融）、armB L8
   34/0、δ 网格两定律、L1/L2 对齐锚、批二学习头判读（单调性+逐位同一性）
   ——均不依赖上述缺陷。

---

## 九、v2.3 test-911 单发协议（v2，2026-07-17 冻结；在任何 v2.3 test 观测之前写定）

> 前提：v2.3 dev 有效性门 PASS（`exp/rb/build_v23/rb_dev23_receipt.json`：oracle A/B
> .8704/.8286、新层门 L7 补偿路 4/4 / L11 3/3 / L12 3/3、armB 物理重叠 0）；attr 门已按
> §16.6 封盘于一轮（不进本批）；**判分器冻结 v4 已立**（`exp/rb/scorer_freeze.json`，
> `--split test` 运行时自动验哈希；v2.2.1 归档仍由收据内记录的 v3 哈希背书）。
> 构建钉死：`rb_v2.3.0 / config e1a515c29b8a`（bank `309c7c0d0361…` 已含）、
> manifest sha `eef243ee…`、dev/test **89/911**（A 546 / B 365）。
> 单发纪律、收据、停服复放要求 = §一逐字承接；收据名 `rb_test_receipt_v23.json`。

### 9.1 provider 清单与运行序（15 个，各一次）

**臂 A（test 546，`--build exp/rb/build_v23 --split test --input audio`）**：

| # | provider | 命令要点 |
|---|---|---|
| 1 | `rbt23_tact_d150` | `--arm A --system tact --delta 1.5`（缓存主本，全新建） |
| 2 | `rbt23_sblock` | `--arm A --system blocking` |
| 3 | `rbt23_tact_d150_nobar` | `--commit-barrier off`（种自 #1 副本） |
| 4–7 | `rbt23_tact_d{000,060,100,200}` | δ 网格（逐个链式接种） |
| 8 | `rbt23_tact_d150_nodag` | `--dag off` |
| 9–11 | `rbt23_tact_d150_fc{v1,filler,silent}` | FC 三档（决策应全缓存命中；**臂 A 限定**——臂 B 的 L10/L11 事件锚在 tts_start 上，改 say 策略会结构性改变考题） |
| 12 | `rbt23_oracle_a` | `--decider oracle --input text` |

**臂 B（test 365，`--tts qwen`，正式跑前仓外预热全部段文本）**：
13 `rbt23_b_tact_d150`、14 `rbt23_b_sblock`、15 `rbt23_b_oracle`。

**硬有效性门（任何臂 B provider 违反即中止入册，属基础设施级）**：report
`armb_timing.total_overlaps == 0`。学习头臂不进本批（双基准否定链已闭合，§7.4）；
attr 臂不进本批（§16.6 封盘）。

### 9.2 冻结判读（复现类 R1–R2、新考场类 R3–R5、既有续读 R6–R8）

- **R1 屏障复现**：L4/L3 配对 on-only 显著（精确二项 p<.05）且 nobar 全量 exact <
  blocking（v2.2.1"无屏障净负"在修复后基准上复现）。
- **R2 三定律复现**：δ 网格 exact 弱单调、first50 与 δ 逐位解耦、完成保费 ≈ δ。
- **R3 补偿经济学（L7 新考场，主读数）**：**H-COMP1** = blocking L7 exact ≥ TACT
  L7 exact（该层给"早提交"定价；blocking 决到静默从不早提交，预测其免费通过——
  这是对我方臂的诚实逆风预注册）；**H-COMP2** = TACT 的 reverse 工具调用发生率
  （v2.2.1 全 32 provider 为 0）——>0 即"LLM 会走补偿路"的第一个证据，=0 则
  "补偿是能力缺口而非机制缺口"成立，fees/comp_cost 轨首次有非零机会。
- **R4（L11 TTS 打断，臂 B）**：TACT vs blocking 配对计数 + first/done；方向不预设
  （single 场景 blocking 也能过 exact，差异预期落在延迟与打断响应，进双工轨）。
- **R5（L12 归属测量）**：n=36 基线误绑率 = 失败夹中含 wrong-target patch 轨迹证据
  的比例（逐轨迹审计口径 = §16.6；机制修复已封盘，本读数是论文对该现象的定量刻画）。
- **R6（armB L8 复现）**：修复后锚（首个非 READ launch）下 TACT-only/blocking-only
  配对——v2.2.1 34/0 形态是否在真事务在飞窗上复现。
- **R7（L10 双率）**：描述级照旧（n<30 不进结论句）。
- **R8（FC talk-over 前沿）**：批一 post-hoc 双工面本批**预注册化**（say 起点
  talk-over 率 × first50，三档 + blocking 参照；仍为描述性前沿主张，无支配预设）。
- **H-B1 不再重冻**：v2.3 仍无跨状态孪生对键，预注册 McNemar 依旧不可合法计算；
  armB 分层只作描述（诚实沿批一 §5.3）。
- 判读纪律照旧：逐格配对 + 精确二项、n<30 描述级、judge 噪声带 ≤2pt。

### 9.3 观测后流程

判读层写 §十（镜像容器独立重算纪律照批一/批二）；v2.3 主表 + v2.2.1 勘误后可引行
并置进论文（分箱失效行永不并置）。本批之后 RB v2.3 test 窗口即告耗尽，任何后续
运行 = 新系统版本单发或 v2.4 版本递增。

### 9.4 增补（2026-07-17，仍在任何 v2.3 test 观测之前）：admission 臂

用户裁定"方法必须应对 L12"后交付 admission v1（patch 模式门，`rb_design.md`
§16.7：拒绝非法字段 patch，可证明无害，v1 不重定向）。provider 清单增补两个
（总数 15→17），排在原 #12 之后、臂 B 三臂之前：

| # | provider | 命令要点 |
|---|---|---|
| 12a | `rbt23_tact_d150_adm` | `--arm A --system tact --delta 1.5 --admission schema`（种自 #1 副本） |
| 12b | `rbt23_b_tact_d150_adm` | `--arm B ... --admission schema --tts qwen`（种自 #13 副本，跑在 #13 之后） |

**冻结判读追加**：
- **R-ADM1（安全，硬判据）**：admission 臂对主臂配对 **admission-only-loss = 0**
  （可证明无害性的实测检验；任何损失 = 实现 bug，按基础设施级处理）。
- **R-ADM2（收益，诚实低预期）**：admission-only-gain = "junk-arg 是唯一死因"的夹数
  ——普查含非法 patch 的 25 夹全部另有失败共因，预期增益 0–5 夹；=0 也入册
  （门的价值在审计轨与安全证明，不以翻夹数论成败）。
- **R-ADM3（审计）**：rejected/dropped/redirect-candidate 计数随行报告
  （v2.2.1 基线 = 9.7% patch 非法率的复现检验）。

---

## 十、v2.3 test-911 判定（2026-07-17 判读层；§九 R1–R8 + §9.4 R-ADM1–3 的正式裁决）

> 输入：`f321601` 归档（17 providers 单发、0 重跑、freeze v4 全程自动验、armb_timing
> 全 0、离线复放 15 缓存 0 miss + 17 报告/8,558 逐夹逐字节；收据
> `rb_test_receipt_v23.json`，SHA `05875497…` 已核）。**独立核验**：17 臂 headline、
> 四主臂+双 oracle by-layer、全部配对计数、δ 曲线、FC 骨架同一性与 talk-over、
> reverse 调用普查、L12 误绑审计、R-ADM 配对与逐夹尸检，均由镜像容器从归档行重算，
> **0 失配**。

### 10.1 主表（A n=546 / B n=365）

| 臂 | exact | state_norm | U | first50 | done50 |
|---|---|---|---|---|---|
| A TACT δ1.5 | .1740 | .1813 | **.1334** | **1.640** | 3.789 |
| A blocking | **.1832** | .1923 | .1254 | 2.166 | **2.262** |
| A TACT −barrier | .1209 | — | — | 1.640 | — |
| A oracle | .9121 | — | — | — | — |
| B TACT δ1.5 | **.1425** | .1534 | .0973 | **1.640** | 3.730 |
| B blocking | .0959 | .1068 | .0648 | 1.859 | **2.071** |
| B oracle | .8740 | — | — | — | — |

绝对电平较 v2.2.1 全面下移（扰动族 + 内容多样化 + 新难层；对照层 L1 .511→.289、
L9 .831→.679 两系统同降），oracle 天花板反而升到 .91/.87（修复后 L7/L11/L12 全部
门层 1.0，L6-A .818）；两系统共同失败 429 夹中 **382（89.0%）oracle 可过** =
难度增量在内容/感知侧，不在机制侧。

### 10.2 判读裁决

| 判读 | 裁决 | 数字 |
|---|---|---|
| **R1 屏障复现** | **支持（历来最强）** | 配对 on-only/off-only = **29/0**（p=3.7e-9；L8 17/0、L12 5/0、L10 3/0、L4 2/0）；nobar .1209 < blocking .1832 = 无屏障净负第三次复现 |
| **R2 三定律复现** | **支持** | exact .1300/.1300/.1648/.1740/.1832 单调；first50 恒 1.640；保费 A 1.527≈δ / B 1.659；δ>1.5 的增量全落 L5（5 夹），**L7 在全部测试 δ 下恒 0 = 该层只向补偿能力低头（按设计）** |
| **R3 补偿经济学（主读数）** | **H-COMP1 兑现；H-COMP2 = 0** | L7-A：blocking **14/44 (.318)** vs TACT **0/44**（预注册逆风精确兑现）；**四个 LLM 主臂 reverse 工具调用 = 0 次**（oracle 44/44、计费 10/44、comp_cost 108 = 机制通路健康）⇒ **"补偿是能力缺口而非机制缺口"成立**——本批最锋利的负结果 |
| **R4 L11（TTS 打断改口）** | **TACT 独占** | **15/27 (.556) vs 0/27**，配对 15/0（p=6.1e-5）——用户打断 agent 播报改口的标志性全双工格，只有异议窗机制能救（blocking 立即提交后无 reverse 能力 ⇒ 全灭）。**本批最强正结果** |
| **R5 L12（归属测量）** | 已定量 | TACT 6/33 vs blocking 3/33；TACT 27 个失败夹中 **4 夹（15%）** 含解析层 wrong-target patch 证据。附带发现（来自 R-ADM 尸检）：裸决策层的"非法字段"形态多数实为**正确的局部 id 引用**（引擎 `_localmap` 解析后落对目标）——真实误绑率低于裸层观感 |
| **R6 armB L8 复现** | **未复现（诚实降级）** | 修复后真事务在飞窗上 3/88 vs 1/88（配对 2/0 n.s.，oracle 上界 .773）——v2.2.1 的 34/0 应部分归因于坏时间轴的双投放回声件；反应式臂的分化主战场**迁移到 L11** |
| **R7 L10 双率** | 描述级 | whole-episode：A 4/40 vs 2/40、B 2/38 vs 4/38（n<30 纪律不变；参数流双率待服务器侧重放审计，本容器无音频） |
| **R8 talk-over 前沿（预注册化）** | 已出 | say 起点重叠率：base/filler **25.3%**、v1 **16.2%**、silent **1.9%**；first50 = 1.640/1.640/3.749（v1 本批档位混合偏 filler，first 与 filler 齐平——前沿形状随 eta 分布移动，如实报） |
| **R-ADM1（安全硬门）** | **臂 A FAIL（1/10）→ 尸检定罪：实现层 bug** | 十个损失夹逐轨迹尸检同构：门在**决策原始层**拿模型回显的**快照局部 id** 直查全局 `tx.pending`——`A_L12_0001` 里模型 `patch op_id=2` 是局部编号（=set_alert），引擎 `resolve_ref` 经 `_localmap` 正确翻译为真实 op 3 并正确应用（主臂 PASS）；门误读为 "threshold→get_balance 非法" 杀掉了**完全正确的 patch**。另三夹（`rejected_keys=['args']`）是嵌套 wire 形态未建模。**admission v1 臂作废退役**；§16.7 "可证明无害" 勘误：定理对"命名目标=生效目标"的前提在本引擎不成立（引擎本来就在做局部 id→真实 op 的正确重路由） |
| **R-ADM2/3** | 增益如预期低 / 非法率复现 | adm-only gain：A 1（A_L8_0057）、B 1（B_L5_0005）；**解析层非法 patch 率 27/273 = 9.9%**（≈ 普查 9.7% ✓）——该人口真实存在，但 v1 门从未真正作用于它（它拦的是解析前的另一层） |

### 10.3 臂 A "反转"的正确读法（本批叙事主判）

臂 A headline（TACT .1740 < blocking .1832）**不是机制失败**：配对 17/22（p=.52）=
统计平手；逐层分解 = **L7 贡献 −14**、其余全部层合计 **+11**（L12 +3、L4/L10/L8 各
+2、L3 +1）。L7 是本版新设的补偿考场：它给"早提交后不会修复"如实定价，而 H-COMP1
预注册就预测了 blocking（决到静默、从不早提交）免费通过。**窗口机制在修复后的基准上
保住了全部旧阵地（屏障 29/0、竞态/归属/对抗层正差分），新账单只有一张：模型不会调
reverse 工具。**臂 B 维持显著优势（+4.7pt，19/2，p=2.2e-4），主承重从 L8 迁移到
L11——"打断 agent 播报改口"这一全双工修订的定义性场景，是窗口机制的独占领地。

进论文的三句话：① 提交屏障与异议窗的机制价值在修复后基准上复现且更强（29/0、
L11 15/0）；② 补偿轨是当前 30B 级音频 LLM 的能力空白（reverse 调用发生率恒 0，
oracle 通路 1.0）——RB v2.3 把这个空白第一次变成可计价的测量；③ 臂 A 合计平手 =
"窗口收益 ≈ 补偿缺口代价"的当前汇率，δ 网格与 L7 定价共同说明窗口长度是命题参数
而非万能旋钮。

### 10.4 收尾

test-911 十七臂窗口耗尽；v2.3 上后续合法运行 = 新系统版本单发（候选：admission
v1.1〔解析后置门 + wire 形态归一〕——预期增益按 R-ADM2 精神诚实标低；SV 门
〔W6 记录〕）或 v2.4 版本递增。**下一步 = 文稿装配线**（RB 章数字齐：v2.2.1 主表
+ §八勘误 + v2.3 主表 + 双批判读），金子集录制与四基准全文核对并行不变。

### 10.5 admission v1.1 单发规格（2026-07-17 冻结；新系统版本，v2.3 test 上合法单发）

**前置 dev 冒烟（零损失门，先于任何 test 接触）**：两个 dev provider
`rbdev23_adm11_tact_d150`（A，种自 `decision_cache_rbdev23_tact_d150.json` 副本）与
`rbdev23_b_adm11_tact_d150`（B，种自 `decision_cache_rbdev23_b_tact_d150.json` 副本），
`--admission schema11` 其余同各自 dev 基线。**门：对基线配对 adm11-only-loss = 0**
（A 54 夹 + B 35 夹）；违门 = 实现缺陷，修完重进 dev（dev 可迭代），不碰 test。

**test 单发（dev 门过后）**：两个 provider，各一次，§一纪律全承接：

| provider | 命令要点 | 缓存种子 |
|---|---|---|
| `rbt23_tact_d150_adm11` | `--arm A --system tact --delta 1.5 --admission schema11` | `decision_cache_rbt23_tact_d150.json` 副本 |
| `rbt23_b_tact_d150_adm11` | `--arm B ... --admission schema11 --tts qwen` | `decision_cache_rbt23_b_tact_d150.json` 副本 |

**冻结判读**：
- **R-ADM1'（硬门）**：对各自 v2.3 主臂配对 **adm11-only-loss = 0**；任何损失 =
  实现缺陷级（臂作废重修），不入正表。
- **R-ADM2'（收益，诚实低预期）**：adm11-only-gain 预期 0–5 夹（解析后非法 patch
  所在夹多为共因失败）；=0 照实入册，门的价值不以翻夹数论。
- **R-ADM3'（审计三数）**：① 解析后拒绝事件数（对照主臂解析层非法率 9.9%）；
  ② `wire_unwrapped` 计数；③ **v1 假阳性对照数** = 若按 v1 规则本会误拒、v1.1
  放行的 patch 数（≥ test-911 已知的 10 夹当量——量化 v1 勘误的错杀面）。
- 附带记录：admission 审计键仅出现在 adm11 决策行（冻结路径零新键，selftest 钉死）。

判读并入 §十尾注；跑完后 v2.3 上剩余合法运行 = SV 门（W6 记录）或 v2.4。

### 10.6 admission v1.1 单发判定（2026-07-17；§10.5 的观测后尾注）

> 来源提交 `1302ab54966fac701442522b3be422444d2100a5`；完整机器收据
> `exp/rb/build_v23/rb_test_receipt_v23_adm11.json`。runner selftest 45/45，
> freeze v4 三文件逐哈希预检通过。TTS 仓外缓存对 dev/test 臂 B 分别覆盖
> 73/725 unique 段、0 missing；正式 provider 期间 `:8091` 关闭，0 新合成。

**dev 零损失门 PASS**：

| 臂 | baseline exact | adm11 exact | adm11-only / baseline-only | both-pass / both-fail | 拒绝事件 / wire-unwrapped | v1 假阳性对照 |
|---|---:|---:|---:|---:|---:|---:|
| A（54） | 7 | 7 | 0 / **0** | 7 / 47 | 2 / 1 | 7 patches（7 夹） |
| B（35） | 5 | 5 | 0 / **0** | 5 / 30 | 2 / 2 | 2 patches（2 夹） |

两臂 `baseline-only=0` 后才复制 test 主臂缓存并进入单发；dev 决策缓存首跑
A/B = 136/0、89/0 hits/misses。

**test 两臂各 live 一次、0 重跑**：

| 臂 | exact | state verbatim / norm | U | done50 | adm11-only / main-only | both-pass / both-fail |
|---|---:|---:|---:|---:|---:|---:|
| A（546） | 95 = .1740 | 95 / 99 | .1334 | 3.767 | 0 / **0** | 95 / 451 |
| B（365） | 52 = .1425 | 52 / 56 | .0973 | 3.720 | 0 / **0** | 52 / 313 |

- **R-ADM1' PASS（双臂）**：两臂 `main-only-pass` 均为空列表；没有 v1 式错杀。
- **R-ADM2' = 0 / 0 gains**：如预注册低标所允许，41 个被机械拒绝的非法 patch
  全部仍有共同失败因，不翻 whole-episode exact；exact/state/U 与各自主臂逐位同，
  done50 仅 A −.022s、B −.010s。
- **R-ADM3'**：
  - A：解析后拒绝 **27/273 resolved-pending patches = 9.9%**，其中
    `wire_unwrapped=11`；26 个整 patch 丢弃，28 个非法键实例。
  - B：解析后拒绝 **14/174 = 8.0%**，其中 `wire_unwrapped=7`；14 个整 patch
    丢弃，14 个非法键实例。
  - v1 假阳性对照：A **52 patches / 51 episodes**，B
    **24 patches / 22 episodes**。这 76 个 patch 全部是 raw local id 与引擎
    resolved id 不同；其中 10/6 个还带 nested `{"args": {...}}` wire。该数直接量化
    v1 把解析前编号几何误当语义目标的错杀面，远大于 test-911 已经翻成 loss 的 10 夹。

第三个审计数不能从归档后的 applied ops 反推（局部 id 与 wire 已被规范化），故在
停服后的全缓存复放中用 `scripts/rb_adm11_counterfactual.py` 旁路同时执行 v1/v1.1
纯函数；旁路原样返回 v1.1 输出。标准离线复放 A/B = **1432/0、954/0
hits/misses**，report、decision cache、546+365 逐夹目录三类哈希均与 live
逐字节一致；旁路复放后四个 dev/test provider 仍保持同一三类哈希。臂 B
`armb_timing.total_overlaps=0`，四个正式 HTTP miss（A 1 + B 3）全部 200。

**裁决**：v1.1 把机械规则放回正确的解析后层，在 dev 与单发 test 上双重兑现
零损失硬门；收益为 0 符合冻结低预期。论文可写“解析后 schema admission 安全关闭了
非法字段形态并提供审计”，不可写 whole-episode accuracy 增益；合法字段错语义形态
仍由 L12 测量。v2.3 上 admission 系列窗口至此耗尽，剩余合法运行仍仅 SV 门或 v2.4。

**镜像核验（判读侧独立重算，2026-07-17）**：对 22c0f765 归档行（4 provider ×
1,000 逐夹 JSON）不经 report/收据独立重算，**全部逐位一致**——四组配对
（dev A 7/7、dev B 5/5、test A 95/95、test B 52/52，diff 集全空）、headline
（exact/state×2/U_mean/done50 上中位）、审计六计数（A 27 事件/28 键实例/26 整弃/
11 unwrapped/27 决策/27 夹；B 14/14/14/7/14/14）、armB overlaps=0、反事实四文件
详单（v1 假阳性 76 = A 52 patch/51 夹 + B 24/22，**逐条 raw≠resolved**，nested
10/6；代数自洽 v1_would_reject = both + fp：63=11+52、33=9+24）、收据全部 27 个
SHA256（含 freeze v4 三文件、旧 v23 收据未动、反事实脚本）。本容器 runner
selftest 45/45。四点补充证据入册：

1. **结局场行级恒等**：exact/state_verbatim/state_normalized/U 在全部 1,000 个
   配对行上逐行相同；时序差异只出现在 31 个行（done_s：dev 1+2、test 17+11；
   first：test 8+5），**且 31 行全部 exact=False**——机制即"拒掉垃圾 patch ⇒ 少一次
   开窗重启"（示例 `A_L4_0001` done 4.025→2.385 = 恰缩一个 δ 窗，U 双零不动：失败
   夹无 γ^delay 项，故 U 对时序场天然免疫）。零增益因此有完整的失败共因解释：
   41 个被拒事件所在的 41 个夹在两臂全部失败，无一在 adm11 臂通过。
2. **v1 错杀面的层分布**：52 个臂 A 假阳性落 L12 15/L4 12/L3 9/L8 5/L5 4/L6 4/
   L10 3——L12 最重，与 v1 实际翻损的 10 夹（L12 占 5）同构；归属考场正是局部 id
   几何咬人最狠的地方，反事实把 test-911 只显影了 10 夹的错杀面补全为 51 夹当量。
3. **v1 双向失准**：除 76 个假阳性外，反事实同时录得 **v11-only-reject 16(A)+5(B)**
   ——v1 按错误目标的模式查键，恰好合法就放行了真非法 patch。v1 既过杀又漏杀，
   v1.1 对两个方向同时修正。
4. **对账注记（非异常）**：行内记录的 patch op 数 = 引擎应用层计数，臂 B/dev-B 各比
   gate 时点 resolved 计数多 1（161 vs 174−14、12 vs 13−2）——即恰 1 个 gate 时点
   不可解析而放行的 patch 在应用时点已可解析（同决策 launch 先行入 pending）；
   放行路径按设计与无门轨迹等价，计数差属两层定义差，不触及任何判分量。

镜像侧维持 §10.6 裁决不变：**R-ADM1' 双 PASS / R-ADM2'=0 如低标 / R-ADM3' 三数
成立，admission 线在 v2.3 上收官**。论文用法补一句：反事实审计使"若按 v1 会错杀
51+22 夹当量、且漏放 21 个真非法 patch"成为可引用数字——解析后层不是实现细节，
是这类机械门成立与否的分界线。

### 10.7 v2.3 公开勘误：L4 层修订文本系统性畸形（2026-07-17；用户发现，判读容器逐条实证）

**根因链（三环全部代码级复核成立）**：① `rb/grammar.py` v2.3 `REV_UTT.value_first`
模板本身双 `{new}`（zh `"{new}，改成{new}。"` / en `"{new} — change it to {new}."`
——"开口即含新值"的实现方式把值写了两遍）；② DeepSeek 内容库改写把冗余换成对比构式
但只有 `{new}` 一个占位符可用（"Change that from {new} to {new}."、"Put {new} in
place of {new}." 等 8 个 en 双值形态全部入库）；③ 验证器 `rb_content_gen.validate`
的 `_placeholders` 用 `sorted(set(...))` 只比占位符**集合**，双 `{new}` 与单
`{new}` 不可区分；§16.1 的逐类听写审阅未抓到。

**影响面（判读容器对 build_v23 归档 episodes 全量普查）**：L4 两臂 **122/122 夹**
的修订句含新值 ≥2 次（A 72/72、B 50/50；首报 121/122 系口径差、实为全覆盖），kind
全部 `value_first`；en 57 夹中 ~48 条是真语义矛盾句（"Change that from three
thousand to three thousand."），zh 65 条冗余但可解（"美元，改成美元。"）；叠加句首
不流利后出现三重畸形（"Write it, write it, Put X in place of X."）。**test 读数**：
L4-A = TACT 2/64 / blocking 0/64 / oracle **64/64**；L4-B = 1/46 / 0/46 / 46/46。

**判定影响**：oracle 不读文本（直接吃结构化修订记录），故 §10.1 "共同失败 89%
oracle 可过 = 难度在内容侧"的叙事里，**L4 的贡献是构造伪影而非自然内容难度**——
oracle−LLM 差距在 L4 被人为放大，引用该分解时 L4 行须挂本勘误。**缓解**：畸形双臂
对称（TACT/blocking 听同一段音频），配对差分方向不受系统性偏置——R1 的屏障配对读数
（含 L4 2/0）与全体 29/0 结论**维持有效**；但 L4 的绝对电平与 oracle 天花板对比在
v2.3 上不可引用。修复 = v2.4（`rb_design.md` §17 item 0：单 `{new}` 对比模板 +
`{old}` 占位符 + 多重集验证器 + 构建期模板审计硬门 + 运行时单 new 护栏）。
**v2.3 test 窗口按纪律不重跑；v2.3 归档一字不动。**

附带入册（v2.4 构建期发现，非 v2.3 判分影响）：冻结的 `COMMIT_JUDGE_PROMPT` 因内嵌
字面 JSON 花括号，`make_llm_judge` 的 `.format` 渲染必然 KeyError——v2.3 的承诺
judge 是从未被任何已判分运行消费过的**潜伏死仪器**（"对外可用"仅在接口意义上成立）。
v2.4 改 `.replace` 渲染（渲染字节 = 设计意图），修复入冻结 v5。

**第二项勘误（v2.4 对抗审查发现，2026-07-17）：臂 B L10 良性格双投放**。
`ARM_B_EVENT_ONLY` 漏收 L10 的良性格：良性修订既作脚本 piece（L4 箱 gap）又作
`benign_control` 事件投放——即 v2.2 双投放 bug 类在 L10 的漏网。v2.3 归档实测
**13/13** 臂 B 良性格中招（test 12 + dev 1），v2.2.1 同构。影响：v2.2.1/v2.3 的
L10-B 良性采纳读数（均为 n<30 描述级、未进结论句）建立在"修订被说两遍"的音频上，
引用时须挂本勘误；臂 A 不受影响（固定时间轴无事件）。v2.4 修复 = 良性格并入事件
单投放。同批亦钉死 arm-B eou 偏移的投放钳制事实（反应式事件在决策后才可注入，
偏移 <infer 名义值全部钳到 t_dec）：L13 的 eou 箱改为可投放区间 (1.00,1.95)，
legacy 臂 B eou 箱（L4/L5/L6）维持名义箱 + `armb_timing.measured_gaps` 实测双报
（§八 re-binning 纪律的既有口径，非新勘误）。

## §十一 v2.4 test 协议（Phase-0 冻结于 2026-07-17，观测前；构建/判读规格 = `rb_design.md` §17）

> 单发纪律、缓存/复放/收据要求、异常登记与 §一/§九 完全同源；本节只列 v2.4 差异。
> **v2.4 = 论文主线版本**：本批产出论文主表；v2.3/v2.2.1 降为归档对照行（跨版本并
> 置引用须注明各自勘误）。

#### 11.1 前置链（顺序硬性；任何一步失败即停）

- **P0 平价探针**：复制 `exp/rb/build_v23/decision_cache_rbdev23_tact_d150.json`
  为 `decision_cache_rbparity_probe.json`，新代码对 build_v23 跑
  `--split dev --arm A --system tact --decider llm --input audio --provider
  rbparity_probe`，断言 **cache 0 miss** 且逐夹 `decisions` 数组与归档
  `rbdev23_tact_d150` 行逐位相同（行内新增判分字段不在断言面）。这证明 v2.3 prompt
  字节在新代码下不变。探针产物用后删除，不入归档目录。
- **P1 内容库 v2**：`rb_content_gen.py`（DEEPSEEK_API_KEY，默认 `--workers 100`，受
  category 数上限、独立 client/`user_id`/硬重试且稳定归并）重新生成全库（52+2 类：
  新增 confirm×2 lang；value_first 例句须以 `{new}` 开头且带 `{old}`）；
  `rb_build --selftest` 过 + `rb_build --audit` / `audit_templates()` **0 违规**（当前 v2.3 bank 16 处
  违规会被硬拒——这是门在工作）；按 §16.1 checklist 听检，**L4 专项**：value_first
  全部变体逐条读一遍，确认对比语义正确、无双值；commit bank（独立提交，SHA 入册）。
- **P2 正式构建**：`rb_build --out exp/rb/build_v24 --audio qwen`；记录
  config/ids/content 三哈希、A 臂 666/666 WAV 完整性（mono PCM16 16k/cues/tail）、
  TTS 仓外缓存计数。金子集清单随构建再生（144 项，录制目标切到 v2.4）。
- **P3 oracle dev 门**（text 或 audio 皆可，两臂）：**硬门** = A 臂 L1–L4/L7–L12
  全 1.0；B 臂 L4/L7/L9–L15 全 1.0，其中 **L13 pair_axis 八格全 1.0、L14 1.0、
  L15 exact 全 1.0 且 route_axis.l15.aborted ≥ feasible**（`abort_feasible` 是保守
  单侧下界——flag 未标的格也可能实际 abort 成功：全体 50 夹探针实测 49 次 abort、
  flag 仅 46；初稿的等式门是审查修正前的错误表述）、who_axis 良性 adopted=1.0 /
  对抗 intruded=0（对抗格 = command-only，irrelevant 为独立对照格）。**L5/L6/L8
  为已知窗口/时序天花板层**，不设 1.0 门，逐格死因分类入收据（同 v2.3
  `B_L8_0057` 先例；冒烟参考 A .9552〔L5 .7143/L6 .8333〕/ B .9737〔L5 .8333/
  L8 .8333〕）。armb overlaps=0。
- **P4 LLM dev 冒烟**（audio、两臂各一次）：只做构建有效性判读（解析健康、L4 新文
  本在 ASR 意义下可懂、整层零的死因分类），**不设分数门**；预期 L4 dev > 0。
- **P5 冻结 v5**：`exp/rb/scorer_freeze.json` version 5 = rb/scorer.py +
  rb/sandbox.py + rb/registry.py + **scripts/rb_commit_judge.py** 四文件哈希 +
  v2.4 勘误链（§10.7 两项）+ P3 收据 + build 钉死；runner test 守卫自动校验。

#### 11.2 单发 provider 清单（15 个，各 live 恰一次，顺序即清单序）

| # | provider | 关键参数 | 备注 |
|---|---|---|---|
| 1 | `rbt24_tact_d150` | `--arm A --system tact --delta 1.5` | 主臂 A |
| 2 | `rbt24_sblock` | `--arm A --system blocking` | 主对照 A |
| 3 | `rbt24_tact_nobar` | `--commit-barrier off` | R1'' 屏障 |
| 4 | `rbt24_tact_d000` | `--delta 0` | δ 网格 |
| 5 | `rbt24_tact_d060` | `--delta 0.6` | δ 网格 |
| 6 | `rbt24_tact_d100` | `--delta 1.0` | δ 网格 |
| 7 | `rbt24_tact_d200` | `--delta 2.0` | δ 网格 |
| 8 | `rbt24_tact_fc_v1` | `--floor-commit-tiers v1` | FC（臂 A 限定同 §九） |
| 9 | `rbt24_tact_fc_filler` | `--floor-commit-tiers always_filler` | FC |
| 10 | `rbt24_tact_fc_silent` | `--floor-commit-tiers always_silent` | FC |
| 11 | `rbt24_tact_d150_adm11` | `--admission schema11`（种自 #1 缓存） | R-ADM1'' 零损失重验（新人口） |
| 12 | `rbt24_oracle_a` | `--decider oracle` | 天花板 A |
| 13 | `rbt24_b_tact_d150` | `--arm B ... --tts qwen` | 主臂 B |
| 14 | `rbt24_b_sblock` | `--arm B --system blocking --tts qwen` | 主对照 B |
| 15 | `rbt24_b_oracle` | `--arm B --decider oracle --tts qwen` | 天花板 B |

DAG 消融臂裁撤（两版本稳定判定 = 机制活性有据、预注册层零翻转；引用 §五/§十 即可）。
缓存接种：#3–#11 链式种自 #1，同 §九惯例；臂 B 各自独立。**test 跑完后**：
`rb_commit_judge.py --build exp/rb/build_v24 --provider rbt24_b_tact_d150
--provider rbt24_b_sblock --provider rbt24_b_oracle`（L14 判官覆盖层，DeepSeek，
只写 overlay；oracle 行兼作阴性对照——脚本化 say 的 emission 应 ≈0）。

#### 11.3 预注册判读（R''，冻结于观测前）

- **R1'' 屏障三连**：nobar 配对（预期第三次复现 TACT>nobar，p 显著）；**L4 修复后
  屏障主考场的配对差应扩大**（v2.3 伪影下 L4 仅 2/0——方向预注册，量级不设门）。
- **R2'' 三定律**：δ 单调、first50 恒 1.640、完成保费 ≈ δ 于 v2.4 复现。
- **R-L4FIX**：L4-A LLM 绝对电平相对 v2.3 的 2/64 上移（方向预注册）；oracle L4
  维持 1.0；若 L4 仍 ≈0 ⇒ 内容修复不充分，走勘误页不回改本批。
- **R-WHO1（结论级，n=34/格）**：良性采纳率 TACT ≥ blocking；对抗拒绝率
  blocking ≥ TACT——v2.2.1 描述级双率的结论级复验（效用-安全交换定量）。
- **R-PAIR1（主判读）**：L13 user 半区 TACT vs blocking 配对 McNemar（池化 4 状态
  n=72 对，p<.05 预期 TACT 优）；分状态存活序预测 eou ≥ tts ≥ inflight ≥
  committed（每格 n=18，描述级）。
- **R-PAIR2**：L13 bystander 半区入侵率分状态；预期 TACT ≥ blocking（开窗 = 攻击
  面的生命周期分辨版）；两臂 armb overlaps=0 硬门。
- **R-COMMIT1/2**：L14 判官覆盖层 emission rate > 0 = 仪器活（若 =0 按能力发现入
  册，不算 bench 失败——死仪器风险已声明）；wrong-commit 修复率描述级；oracle 阴性
  对照 emission ≈0；judge 缓存复放逐字节。
- **R-ABORT1**：oracle L15 = 1.0（硬门，harness 健康）；LLM 主臂 L15 TACT vs
  blocking 描述级。
- **R-ABORT2**：LLM 主臂 abort 使用率——预注册预测 = **0**（H-COMP2 先例：模型不会
  调 reverse，也不会用 X-cancel）；非零即为值得报告的能力发现。
- **R-ADM1''**：#11 对 #1 配对 main-only-loss = 0 硬门（v1.1 在 v2.4 新人口上的
  零损失重验；PASS ⇒ 论文系统定义含解析后 admission，引 §10.6 连续两版本证据）。
- 判读写 §十二；单发窗口用毕后 generator/scorer/runner 任何改动走公开勘误 + 版本
  递增，不得回改本批。

#### 11.4 回报清单（用户 → 判读）

机器收据（`rb_test_receipt_v24.json`，结构同 v23 收据）+ 以下最小数字面：P0 探针
0-miss 证言；bank SHA/违规数/听检记录；build 三哈希与 WAV 完整性；P3 oracle 两臂
by_layer + pair/route/who 三轴；P4 冒烟两臂 headline 与死因分类；freeze v5 SHA；
15 provider 各自 headline（n/exact/state×2/U/first50/done50/cache 命中/armb
overlaps）；复放三类哈希证言；judge overlay 三 provider 聚合行。
