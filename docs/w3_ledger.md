# W3 逐夹机制台账 — 三疑点逐数字核验（05 计划 §零 / §4.3 第一交付物）

> 2026-07-06。产出方式：纯读既有产物（`grid_rollback.json`、`grid_full.json`、`delta_hist.json`、
> 各夹 `result_w2r_*.json` 的 trace/tx_log、`w2r_stream_replay.py` 源码、`benchmark_data_v2.json` gold），
> 零 GPU、零重跑。所有数字可由上述文件逐位复核。

## 0. 三疑点裁定摘要

| 疑点 | 裁定 | 一句话 |
| --- | --- | --- |
| 一（travel_23） | **成立，且神谕的更正案也错一处** | travel_23 不在 rollback 名册（`state_rollback_test:False`）；真实差分 = 赢 {ecommerce_19, **housing_21**, housing_25}，输 {housing_17b}，净 +2 对账闭合。finance_12b **不是**对 blocking 的净胜（blocking 也过它）；被两版明细共同遗漏的是 housing_21 |
| 二（暴露定义） | **成立，且比神谕说的更严重** | 直方图三个"暴露"值 0.42/1.12/1.16 与系统真实暴露**三个全不对应**：0.42 夹从未失败、1.12 夹真实暴露 3.91s 任何 δ 不可救、只有 1.16 夹数值近似正确但机制不同。"p50 1.12+决策 0.5=1.62"算术作废，替换为**静默预算定律**（§3） |
| 三（增益归因） | **分解做实；神谕的"同段夹窗不可能是胜因"被证伪** | 臂内 +3 全部是窗层（δ 恒 prompt 翻转即因果证明）；直方图 delta_t=0 是单 cue 设计的伪像（第二处修订跨 EoU 被漏检，神谕猜测 (b) 方向正确）。臂间 +2 = 窗 +2（eco19、hou25，均叠加 blocking 独立自失）+ 架构 +1（hou21）− patch 质量 −1（hou17b）+ 0（fin12b 平局修复）。**R15 排除**：sblock 与 TACT 走同一 decide()/prompt（w2r_stream_replay.py L349-350 仅裁 eous[-1:]），不存在 prompt 混淆，差异全部是架构效应——W4"blocking 换 Phase-B prompt"消融**已由 sblock 构造性完成** |

**但核验挖出一个比三疑点更大的问题（§4 异常行 A）：δ*=1.5→0.706 这条头条依赖回放器一个未记档的语义
——"决策在途提交屏障"。三个翻转夹的救援 patch 全部落在名义窗口到期之后、靠该屏障才生效。engine_b 若按
连续时钟实现（不设屏障），nominal 决策时延下 δ=1.5 的 12/17 退化为 9/17。这必须在 D1 作为显式设计裁断。**

---

## 1. 回放器保护栈语义（读码钉死，`scripts/w2r_stream_replay.py`）

1. **hold 层（oracle 式）**：EoU 从全文件 VAD 回溯预计算（L343-348）：段尾后静默 ≥0.64s 才成为 EoU。
   间隙 <0.64 的修订**结构上**并入同一决策输入——回放的 hold 是无检测时延的理想 hold（真机有 silero 延迟，D1 对照点）。
2. **无 staleness 层**：语音在决策在途期间恢复不作废决策（回放无 epoch 协议）。教义三预言的 stale 层接住边缘夹
   在回放里不存在，其职能被下述屏障顶替。
3. **异议窗 = 静默钟倒计时**（`silent_deadline` L302-316）：用户说话冻结倒数，静默期消耗预算；patch 重启窗（L420）；
   δ≤0 或 blocking 模式即时提交（L407-408）。
4. **决策在途提交屏障（未记档的第四层）**：到期 sweep 只发生在两个处理点——每个 EoU（`dl <= t_eou`，L370-373）
   与音频结束（L434-436），提交时间戳按名义 deadline 回填。deadline 落在 `(t_eou, t_dec]` 区间的 op 在决策返回时
   **仍是 pending**，patch/cancel 先于下一次 sweep 生效 ⇒ 决策在途期间 pending 集实质冻结。

## 2. 17×7 结果矩阵（官方 exact；P=过 F=挂；源 `grid_rollback.json` 逐行）

| 夹 | sblock | δ=0 | 0.3 | 0.6 | 1.0 | 1.5 | 2.5 | 归因码 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecommerce_09 | P | P | P | P | P | P | P | hold 合并（0.42<0.64；直方图"暴露"是伪暴露）+ 桩段 EoU 判 WAIT |
| ecommerce_11 | P | P | P | P | P | P | P | 单段单 EoU，n/a |
| ecommerce_19 | **F** | F | F | F | **P** | P | P | **窗层**，阈 δ>0.64；sblock 自失（漏 add_to_cart） |
| finance_12a (65e8) | P | P | P | P | P | P | P | 单段单 EoU，n/a |
| finance_12b (69a9) | P | F | F | F | F | **P** | P | **窗层**，阈 δ>1.12（静默预算 0.48+0.64） |
| finance_15 | P | P | P | P | P | P | P | hold 合并（0.45） |
| finance_19 | P | P | P | P | P | P | P | hold 合并（0.42） |
| finance_23 | **F** | F | F | F | F | F | F | 模型错（共通）：双臂 narrate-no-op，modify_autopay 从不被调用 |
| housing_09 | P | P | P | P | P | P | P | hold 合并（6 个间隙全 <0.64） |
| housing_11 | **F** | F | F | F | F | F | F | 管线缺陷（共通）：gold 城市 Austin 在对话前文，回放不喂历史，双臂各幻觉城市 |
| housing_17a (65e8) | P | P | P | P | P | P | P | hold 合并（0.45/0.61/0.39/0.62 全 <0.64）→ 单决策 |
| housing_17b (69a9) | P | F | F | F | F | F | F | **patch 质量**：EoU2 patch origin='here'（应 'my house'），后续只 patch mode，永不改回 |
| housing_19 | P | P | P | P | P | P | P | hold 合并（0.49） |
| housing_21 | **F** | **P** | P | P | P | P | P | **架构效应**：TACT 增量决策保 verbatim "train station"；sblock 单发过度解析 "Penn Station"+幻觉 NYC |
| housing_25 | **F** | F | F | F | **P** | P | P | **窗层**，阈 δ>0.64；sblock 自失（零输出） |
| travel_10 | F | F | F | F | F | F | F | 暴露超窗：真实静默预算 3.91s > 全网格；EoU1 错失 cancel；sblock 亦自失（say 说 7th、args 写 5th） |
| travel_19 | F | F | F | F | F | F | F | 格式失配（共通）："June 3rd" ≠ gold "June 3"；机制侧 4 连 patch 全部成功（Rome→Milan、June 1→3rd） |

阶梯核对：δ≤0.6 通过集 = 9 ✓；δ=1.0 加 {eco19, hou25} = 11 ✓；δ=1.5 加 {fin12b} = 12 ✓；sblock = 10 ✓。
与 `grid_rollback.json` 汇总 0.529/0.529/0.529/0.647/0.706/0.706 与 0.588 逐位一致。

## 3. 静默预算定律（替换 §3.1 的"暴露间隙+决策延迟"算术）

屏障语义下，一个 pending op 能被最终 EoU 的决策救回，当且仅当

**δ > Σ silence(launch → 救援 EoU)**  （说话时间不消耗窗；决策时延不进阈值——屏障顶住）

四个系统视角真暴露夹的精确阈值（全部经 `silent_deadline` 复算并与 tx_log 时间戳吻合）：

| 夹 | launch (t_dec) | 静默构成 | 阈值 | 网格表现 |
| --- | --- | --- | --- | --- |
| ecommerce_19 | 12.358（说话中）| 尾 hold 0.64 | **0.64+** | 0.6 挂 / 1.0 过（真阈在 0.64–1.0 之间，被网格量化到 1.0）|
| housing_25 | 9.254（说话中）| 尾 hold 0.64 | **0.64+** | 同上 |
| finance_12b | 7.398（说话中）| 段间隙 0.48 + 尾 hold 0.64 | **1.12+** | 1.0 挂 / 1.5 过 ✓ |
| travel_10 | 7.110 | 段间隙 3.27 + 尾 hold 0.64 | **3.91+** | 全网格挂 ✓（d250 亦挂）|

推论：
- **平台期真实区间是 [1.13, 3.91)**，不是"1.5–2.5"；δ* 取 1.5 合法但当前解释文字（p50 1.12 + 决策 0.5）是数字巧合，须重写。
- 阈值下限恒为 0.64（尾 hold 必然存在）⇒ **δ 只需覆盖静默**，这是"保护对首响免费"之外的第二条结构优势，可进论文。
- 直方图三个"暴露"值与本表对照：0.42（eco09）系统视角不存在（hold 合并）；1.124（travel_10）严重低估（单 cue 锚在 seg1 的 "wait"，真更正 "October 7th" 在 seg2，真实预算 3.91）；1.156（fin12b）数值近似对但机制是"段间隙+尾 hold"而非"launch 暴露间隙"。**直方图与系统视角相关性 ≈ 0，只能作 λ(t) 素材，不能解释曲线**——疑点二从"混用"升级为"不可用于归因"。

## 4. 异常行（§4.3 规则 4：实际与应然不符 = bug 或洞察）

**A.（P0，D1 裁断项）三个翻转全部依赖决策屏障。** 救援 patch 时刻 vs 名义 deadline：
eco19@δ1.0：patch 20.55 > dl 19.91；hou25@δ1.0：patch 21.54 > dl 20.90；fin12b@δ1.5：patch 16.04 > dl 15.42。
无一例 patch 在名义到期前抵达。若 engine_b 按连续时钟提交（无屏障），阈值全体 +决策时延：
- nominal 1.0s（本次网格口径）：阈 1.64/1.64/2.12 ⇒ δ=1.5 时 12/17 → **9/17**（反超叙事消失）；δ=2.5 才回 12。
- 实测决策 p50 0.444s：阈 1.08/1.08/1.56 ⇒ δ=1.5 时 **11/17**（fin12b 刀口 1.56>1.5）；δ≥1.6 回 12。
**建议裁断：把屏障升格为显式设计**——"决策在途期间冻结其快照内 pending op 的提交"正是 W1 单写者/快照一致性
原则在 PendingSet 上的对称延伸（神谕裁断 B 已想要"决策本身也被事务化"这句话）；engine_b 照此实现则 δ*=1.5 头条
站得住，且有原则性辩护。若裁断为连续时钟，则 δ 网格须重跑、δ* 上移。**二选一必须在 D1 定案并写进论文语义节。**

> **勘误（2026-07-06，D1 接线自查，跑前登记）**："三个翻转全部依赖屏障"收窄为**两个**。eco19 的救援 patch 是
> 值中性 diff（`{query:"tablet"}` 打在已为 tablet 的 op 上），其翻转是**窗口保持 pending 的快照效应**（到期
> 19.91 > EoU 19.55 对任意 δ>0.64 成立，pending 快照诱导补发 add_to_cart），与 commit/patch 先后无关——
> barrier-off 下 eco19 在 δ≥1.0 仍过。屏障因果权重 = {hou25, fin12b}；消融预期 12/17 vs **10/17**（非 06
> 预注册的 9/17）。机制分层三级：hold 合并 → 快照效应 → 屏障。详见 `docs/engine_b_spec.md` §4 与探针文档。

**B. 后提交瘫痪 3/3。** 快照显示错参已提交时，Phase-B 一律"嘴上更新、手上无 op"（say "Updating…"、ops=[]），
且连无关的后续调用也不发：eco19@δ≤0.6 漏 add_to_cart、hou25@δ≤0.6 漏 search_apartments、fin12b@δ≤1.0 不重发。
低 δ 的失败因此是双重的（脏轨迹 + 瘫痪漏调）。prompt 迭代批次头号靶（教"继续未完成调用/显式补偿"；30 条纪律）。
诚实注记：修好瘫痪会抬高低 δ 曲线（漏调类可救），但官方轨 wrong-arg 已提交即死不受影响——窗必要性叙事无恙。

**C. travel_10 在 EoU1 错失 cancel。** "wait, my schedule just changed"（修订已宣告、新值未到）时决策 ops=[]；
此刻 op 尚 pending（屏障护着，dl 8.89 ∈ (8.03, 9.03]），cancel 可救、patch-restart 不够（10.53 < 10.66）。
"宣告即撤销"规则是 prompt 迭代第二靶，也是唯一能把 travel_10 从不可救区拉回的路径。

**D. sblock 臂三处独立自失**（hou25 全零输出、travel_10 say/args 自相矛盾、eco19 漏 add）。
0.588 的构成里有非结构性失分；McNemar 诚实口径：不和谐对 = 4（3 胜 1 负），精确 p = 0.625
（04 汇报写 ~5 对 / p≈0.375，一并勘误）。窗胜利的两夹（eco19、hou25）同时需要 blocking 自失才成为净胜——
统计主张必须按教义四转为机制主张。

**E. housing_11 揭示回放不喂对话前文。** gold 参数（Austin）在 dialogue 首轮，released 音频只含末轮。
共通压低双臂。W3 应清点全量 100 中多轮场景数并裁断是否喂历史（影响与官方口径的可比性，动前先问神谕）。

> **审计闭合（2026-07-06 零 GPU 批）**：官方 SUT（cascaded_agent/lk_agent_tool）instructions 为固定
> 字符串，对话前文从不入被测上下文——**官方同样只听末轮音频**。裁断③走"基准属性"分支：不喂历史；
> 本类场景经普查恰 2/100（housing_11、ecommerce_12），对一切合规系统不可赢，归因码由"管线缺陷(共通)"
> 改判 **"基准属性"**。链式计数（裁断 A-iv 顺带）：≥2 无依赖调用场景 = 28/100。

**F. 判分器 $RESULT 引用是宽容解析**：eco19 的 `$RESULT_1.product_id` vs gold `$RESULT_0.products[0].product_id`
判过。记录在案防误读（与"同名调用 pop(0) 双杀"的严苛形成对照）。

**G. finance_23 双臂从不调用 modify_autopay**，而 W1 legacy blocking（不同 prompt）在全量轨曾判过该场景
（grid_full：blocking T / sblock F / tact F）——W2 决策 prompt 对该句式（"check the balance on my— actually never mind…"）
存在系统性回避。归共通模型错，prompt 批次第三靶。

## 5. 疑点一勘误定稿（供 04/w2_rerun_report 引用）

- rollback 17 夹差分（exact，δ*=1.5 vs sblock）：**赢 ecommerce_19、housing_21、housing_25；输 housing_17b；净 +2 = 10/17 → 12/17** ✓。
- travel_23：非 rollback 场景（flag=False、无 rollback 音频）；全量 exact 任何口径都不是 TACT 胜
  （w2r_blocking T / sblock_full F / tact_full F —— 它是两个 W2 臂对 W1 legacy 的**共同回归**）。
  写进明细的可能来源：judge 轨差分或 sblock-回归误读（不可考，不影响勘误结论）。
- finance_12b 从明细移除（blocking 亦过；它属于臂内窗阶梯 +3 的第三夹，不属于臂间净胜）。

## 6. 窗效应 / 架构效应分解定稿（疑点三）

| 成分 | 夹 | 数值 | 因果证据 |
| --- | --- | --- | --- |
| 窗层（含屏障） | eco19、hou25（δ=1.0 翻）、fin12b（δ=1.5 翻） | 臂内 **+3**（0.529→0.706） | δ 恒定 prompt/缓存下的翻转，T=0 可复现 |
| 架构效应（增量决策上下文，同 prompt） | hou21（+1）、hou17b（−1）、fin12b@低δ（−1，急切 launch 自造暴露） | 臂间 δ=0 时净 **−1**（0.529 vs 0.588） | δ=0 全对照 |
| 臂间 @δ* | 上两行叠加 + blocking 三处自失（§4D） | 净 **+2** | 10→12 逐夹闭合 |

叙事定稿（替换 R15 预案）："**异议窗贡献全部臂内阶梯 +3；架构效应净 0（verbatim 增益 +1 与 patch 质量 −1 对消）；
prompt 混淆不存在（两臂同 decide()）**。窗胜利的两夹叠加了 blocking 独立失误，n=17 不显著（p=0.625），
主张以逐夹机制归因立足，不以统计立足。"

## 7. 回填 W3 计划

- **D1 新增最高优先裁断**：屏障语义（§4A 二选一）写进 engine_b 规格；接线冒烟里加"翻转三夹在两种语义下的预期差异"作为对照探针。
- **D3 台账 = 本文件**（E2 判据 17×7 全归因 ✓ 三疑点书面闭合 ✓ 分解成文 ✓）；时间线图（图 2b）素材 = §3 表 + tx_log，待画。
- **裁断 C 素材已现 4 例**：travel_19（3rd/3 格式）、hou21-blocking（Penn Station 过度解析）、hou11（前文缺失）、travel_16（全量 Vegas）。
- **prompt 批次三靶**（30 条纪律）：后提交瘫痪（§4B）、宣告即撤销（§4C）、hou17 patch 字段（既有）。
- **oracle 前沿（教义二）**：本台账给出暴露集 = {eco19, hou25, fin12b, travel_10}，其余 13 夹窗保费为纯损——重放算术可直接开算。
