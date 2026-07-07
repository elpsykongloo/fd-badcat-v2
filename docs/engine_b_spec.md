# engine_b 规格 — 提交屏障（decision-atomic commit）与统一语义架构

> W3 D1 定案文档。裁断来源：`手工文档/神谕/06_W3.1 扩展版.md` §一（屏障升格为显式设计，
> 连续时钟被裁掉；"今后任何人若翻案到连续时钟，代价是全网格重跑 + δ* 上移 + 头条重写——现在定案，永不重议"）。
> 证据链：`docs/w3_ledger.md` §4A（三个翻转夹的救援 patch 全部落在名义到期之后，依赖屏障才生效）。

## 0. 架构：语义单源 + 两个驱动器

```
                 ┌──────────────────────────────┐
                 │  src/tact_core.py（唯一语义源）│
                 │  WindowLedger / apply_decision │
                 │  prompt&snapshot v2 / 解析修复 │
                 └──────┬───────────────┬───────┘
        live/因果驱动    │               │    离线/区间驱动
┌────────────────────────▼──┐   ┌────────▼───────────────────────┐
│ src/engine_b.py TactEngine │   │ scripts/w2r_stream_replay.py    │
│ VADIterator 流式感知        │   │ 离线 VAD（回溯 oracle EoU）      │
│ 逐帧烧静默 + 可重臂定时器    │   │ 逐 EoU 区间烧静默                │
│ backend :18000 / run_offline│   │ --engine core（逐位契约路径）    │
└────────────────────────────┘   │ --engine full（帧→TactEngine，  │
                                 │   H6 仪器：同语义、活感知）       │
                                 └─────────────────────────────────┘
```

W2 的 harness 语义（EoU 处懒惰 sweep、patch 先于 sweep 生效）不再是隐式实现细节：
它被形式化为**提交屏障**并在 core 里显式实现；replay 从此是 engine_b 语义核的驱动器，
不是平行实现（06 裁断 D / H6 的构造性消灭）。

## 1. 五条款（06 §一照抄 → 本实现的落点）

1. **提交调度器是定时器驱动的**。窗口是**静默钟预算**：launch 开窗 budget=δ；用户说话冻结
   倒数，静默烧预算；patch 重启（budget 回满 δ）；cancel 终止；δ≤0 或 blocking 即时提交。
   落点：`WindowLedger.advance_silence`（预算烧毁+到期）、TactEngine `_arm_timers`/`WindowTimer`
   （live 可重臂定时事件，audio clock 驱动；replay 的 end-of-audio 收尾即该定时器的离线代理）。
2. **屏障**：决策 dispatch 时登记快照 pending 集 S（`begin_decision(key, S)`）；S 内 op 的到期
   提交在决策在途期间**延迟不丢弃**（`expired[op]=nominal`）；DecisionDone 处理顺序 = 先
   `apply_decision_ops`（patch 救援即 `restart`、cancel 即 `close`），再 `sweep(t_now)` 以当前
   时钟提交未获救者、**时间戳记名义到期**。`commit_barrier: false` = 连续时钟消融臂：到期即提交
   （含决策在途期间），迟到 patch 丢弃并记 `patch_after_commit`。
3. **释放路径三条**（全部收敛到 `end_decision + sweep`，cause 入账）：
   | 路径 | 触发 | 行为 | trace |
   |---|---|---|---|
   | 正常返回 | DecisionDone | ops 先应用再 sweep | `tact_decision_done` |
   | staleness 作废 | gen 失配（恒）或 epoch 失配（`stale_eou_policy: drop` 时） | 弃 ops，立即 sweep | `tact_decision_stale` |
   | 决策超时 | `decision_timeout_s` 到期 | fail-open 立即 sweep | `tact_decision_timeout`（HumDial 与全量上应零触发） |
4. **双戳记账**：每次提交记 `{t_commit(=nominal), actual_commit, deferred_s}`（commit 记录与
   trace 事件 `act_commit`；deferred_s>0 时另发 `act_commit_deferred`）。屏障延后量的分布是一等
   指标，上界应 ≤ 决策时延 p99；审计明细在 `WindowLedger.deferrals`（含 rescued_patch/cancelled/
   committed 三种结局与 cause）。
5. **工程时延预算**：live EoU 检测带 VAD 帧级时延 ε（silero 32ms 粒度 + 帧界取整，预算 ≤100ms），
   真实救援阈值 ≈ Σ静默 + ε。**δ 选择准则**：δ ≥ max(暴露阈值) + ε + 抖动余量。
   δ*=1.5 对台账最大可救阈值 1.12 满足，余量 0.28s。

## 2. `stale_eou_policy` 默认 `apply` 的论证（W2 逐位契约）

W1 的 staleness 协议为 judge/shift/interrupt 而立：它们的证据是"用户说完了"，被新语音证伪。
Phase-B 的 EoU 决策不同：证据是"停顿本身"（≥0.64s 静默确已发生），其 ops 作用于 PendingSet 而非
话轮——语音恢复不证伪一个 patch/launch 的意图；重复 launch 由幂等去重兜底、过时参数由下一 EoU 的
patch 收敛。W2 全部网格在此语义下产生（例：housing_25 EoU0 决策在语音恢复 36ms 后仍然落地并成为
δ* 胜利的前提）。故默认 `apply` = 逐位契约；`drop` 保留为 W4 消融旋钮。gen 失配（session 重置）
恒丢弃，不受此策略影响。

## 3. 已知 live/replay 感知差（D2 残差排查清单，非语义差）

- EoU 定位：VADIterator 流式事件（min_silence 100ms 默认）+ 帧界取整 vs 离线
  `get_speech_timestamps(min_silence=400ms)` 回溯——分段与锚点可差到数百 ms（H6 残差首查项）。
  **实测（W3 D1 full 探针）**：end 事件迟滞 ε≈0.03–0.13s、逐边界随内容浮动；**[hold, hold+ε) 带内
  ε 决定 EoU 的存在性**——间隙 0.708/0.676/0.67 三夹（eco19/hou25/hou17b）的多 EoU 结构在流式感知下
  坍缩为单发，live 轨可救修订集合 = 间隙 > hold+ε 者（δ 准则第 5 条的存在性补注，待神谕核准措辞）。
- 决策输入音频：live 取累计前缀截到 end 锚（帧粒度），replay 截到离线段尾——cache key 可能不同
  （机制不变，命中率下降而已）。
- ASR 仅日志：live 按 VAD 段送 ASR（Phase-A 是整轮合并）；不在决策关键路径。
- t_dec：live 为决策事件落地帧时刻（≤1 帧滞后）；replay 为 t_eou+infer 精确值。

## 4. D1 验收探针（硬判据；`scripts/w3_barrier_probe.py`）

nominal infer=1.0s、W2 决策缓存、官方 exact 判分：

| 夹 | δ | barrier on | barrier off |
|---|---|---|---|
| ecommerce_19 | 1.0 | **P** | **P**（见下方修正） |
| housing_25 | 1.0 | **P** | F |
| finance_12b (69a9) | 1.5 | **P** | F |
| travel_10 | 1.5 | F | F |
| **rollback-17 @1.5 汇总** | | **12/17** | **10/17** |

**对 06 §一预注册数字（9/17、三夹全挂）的跑前修正**：eco19 的救援 patch 是**值中性 diff**
（`{query:"tablet"}` 打在已为 tablet 的 op 上——首个修订被 hold 合并进了 launch）。其 δ 翻转是
**快照效应**：窗口使 op 在 EoU2 时仍 pending（到期 19.91 > 19.55，对任意 δ>0.64 成立）→ pending
快照诱导模型补发 add_to_cart；commit 与 patch 的先后从不影响该夹的轨迹。故**屏障的因果权重恰为
2 夹**（hou25：patch 3500；fin12b：patch 150），消融差 = 12 vs 10。修正在运行前登记（预注册纪律）；
若实测与本表不符，按发现处理而非调表。台账 §4A 的"三个翻转全部依赖屏障"一句按此收窄——机制分层
从此是三级：**hold 合并（8 夹）→ 窗口保持 pending 的快照效应（eco19）→ 屏障保住迟到 patch
（hou25、fin12b）**。

外加**逐位对账**：barrier-on 与冻结的 `result_w2r_tact_d100/d150` 在**全部确定性字段**上逐位相等——
`actual_tool_calls`（函数/参数/名义 launch+commit 时间戳）、latency 结构字段（ack_emitted/n_eou/
infer_mode）、ack 路径的 first_response_s。**墙钟分量（task_completion_s 里的 tool_wall_s）仅信息化
不设门**：v1 档案由 `--workers>1` 产生，全局种子 RNG 被线程交错打乱（AGENTS 判"并发跑的延迟字段非
权威"），任何有限容差都不原则；仅留 1.0s sanity 界拦截结构性回归（如延迟档配错）。W3 起 latency 增
补确定性字段 `completion_nominal_s`（= max 名义 commit − t_user_end，对 --workers 免疫）。逐位对上
⇒ 语义钉死 ⇒ grid v1 零重跑存活；on/off 两列即论文的语义敏感性消融。

## 5. 单测清单（`tests/test_commit_barrier.py`，纯 CPU）

静默预算定律（冻结/烧毁/名义戳精确）｜屏障延迟+patch 救援（窗重启回满）｜屏障延迟+未救援
（名义戳提交、deferred_s 审计）｜连续时钟消融（在途提交+迟到 patch 丢弃入账）｜stale/timeout
fail-open sweep｜多决策重叠 guard｜δ≤0/blocking 即时提交｜launch 去重+嵌套 args 展开+schema
矫正｜eco19 形状端到端（on 过 / off 挂）｜TactEngine 注入式冒烟（EoU→launch→尾部到期提交；
引擎回路内的屏障救援；v0 构造面兼容）。

## 6. 配置面（`src/config.yaml` engine 节 / 驱动器 CLI）

engine: `phase: b`, `mode: tact|blocking`, `delta`, `commit_barrier`, `stale_eou_policy`,
`tool_sync`, `tts_enabled`, `asr_enabled`, `sv_alpha: null`（W4 占位；W3 必须保持 null——任何
非空值都得进 prompt，会打穿缓存与 prompt 冻结）。
驱动器：`--commit-barrier on|off`，`--engine core|full`，其余 CLI 与 W2 相同。
