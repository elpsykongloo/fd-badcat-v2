# W3 D1 收口备忘 — engine_b 接线验证全记录（2026-07-06）

> 执行：手动（用户）；分析与记录：本备忘。原始产物：`exp/w3/barrier_probe.json`（core）、
> `exp/w3/barrier_probe_full.json`（full）、`exp/w3/decision_cache_engine_full.json`、
> `exp/humdial_100_w3_phasea_20260706_160032`、`logs/humdial_w3_phasea_20260706_160032_eval`。

## 1. 判定总览

| 步 | 内容 | 结果 | 判定 |
|---|---|---|---|
| ① | tests/test_commit_barrier.py | 37/37 | ✅ |
| ② | 屏障探针 core（硬判据） | VERDICT 绿，exit 0 | ✅ **D1 语义钉死** |
| ③ | S2 回归 | phase_b/transaction/engine ✅；ack_integration 4/4（v1 重写后）；deterministic_replay 仅速度断言挂（1.8×<10×，无 GPU 容器环境性，非阻塞） | ✅（速度项算力日复验） |
| ④ | 屏障探针 full（仪器，无硬预期） | exit 1；13 处偏差全部归因（§3） | ✅ 读数有效 |
| ⑤ | HumDial 100 回归（phase a） | Δ=−0.75 ≥ −1 门 PASS；llm_timeout 0；tact_* 事件 0 | ✅ 有条件通过（§4 混杂注记） |

## 2. core 探针（硬判据，全绿）

- 机制：on 12/17 / off **10/17**（跑前修正的预注册值逐位兑现）；d100 四夹 on {eco19 P, hou25 P, fin12b F, tra10 F} / off {eco19 P, 其余 F} 全对。
- **屏障消融差 = 2 夹 {hou25, fin12b}**；eco19 为快照效应（值中性 patch），off 下仍过——机制三级分层
  （hold 合并 → 窗口保持 pending 的快照效应 → 屏障保住迟到 patch）实测成立。
- parity：21/21 确定性字段逐位（actual_tool_calls 含名义时间戳、结构 latency、ack 路 first）；
  10 条非门控 wall info = v1 档案 tool_wall 系 --workers 全局 RNG 打乱产物（AGENTS 已判非权威）。
- 双戳：on_d150 deferrals n=8（rescued 6 / committed 2，max_deferred 0.14 = hou25 pets 21.542−21.402）；
  on_d100 max 0.64（pets 21.542−20.902）——与台账算术逐位吻合。off 臂 deferral 恒 0 ✓。
- 缓存 80/80 命中，零 GPU。

## 3. full 仪器读数（帧→TactEngine，流式 VADIterator；13 处偏差逐一归因）

跑机注记：85GB 卡；`exp/w3/qwen3_omni_text_only_84g.yaml`（stage-0 only、max_model_len 8192、
max_num_seqs 1、gpu_util 0.78）；独立缓存 28 条新决策（live 前缀帧粒度切分，键位与 W2 缓存不通）。

| 夹 | full 表现 | 归因 |
|---|---|---|
| eco19（4 组合全 F） | eous [[0,19.68]] 单 EoU（core 2 个） | **ε 段合并**：载荷间隙 0.708 落入 [hold, hold+ε)，流式 end 检测迟滞使 hold 不到期——两 EoU 结构消失 → 单发决策复现 sblock 失误（漏 add_to_cart；且新前缀下模型加发 null 可选参数，违 prompt 规则 5，非致命） |
| hou25（on 两组合 F） | eous [[0,20.672]] 单 EoU（core 2 个；间隙 0.676） | **ε 段合并** + 单发理解错：全文含 "3500" 更正，单发仍写 3000——增量决策+patch 在 core 里答对、单发答错（利好叙事素材） |
| hou17b（两组合 P） | eous 2 个（core 3 个；0.67 间隙被合并） | **ε 段合并 + 决策漂移**：幸存的中途决策这次 ops=[]（不早发）→ 末端单发全文答对 = sblock 路径 = P，增量 patch 失败路径未触发 |
| fin23（两组合 P） | eous [[0,18.176]] 单 EoU（core 同为单，锚点 +0.13） | **纯决策翻转**：ε 平移的前缀字节 → 新键 → 新调用 → 这次真的发了 modify_autopay——narrate-no-op 回避是刀口行为，微扰即翻（prompt 五靶#4 的直接证据） |
| d100 两条 TOTAL 0 | eco19/hou25 同因 | 同上两行 |

**结构性发现（进 D2 与 δ 准则）**：ε 不只平移救援阈值，**在 [hold, hold+ε) 带内它决定 EoU 的存在性**
——本批三个翻夹的载荷间隙 0.67/0.676/0.708 全在带内，实测 ε（end 事件迟滞）≈0.03–0.13s、逐边界随
内容浮动。live 轨的窗口救援机会集合因此与离线轨天然不同，与 δ 取值无关；这是 H6 残差的第一构成项。

**混杂记录**：serving 栈非 W2 provenance（卡型/配置均不同）。ε 结构类归因（eco19/hou25/hou17b 的
EoU 数变化）与模型无关、不受此混杂；fin23 与 hou17b 的决策翻转不能排除 server 数值面贡献。
可选对照（下个 GPU 日，标准栈）：取 5 条 W2 缓存键绕缓存重调、diff 原文，即可定界 server 漂移。

## 4. HumDial 100 回归（⑤）

- judge 三判（deepseek-v4-flash）：60.75 / 62.00 / 67.00 → **63.25 ± 3.13**；对 W2 基线 64.0 Δ=−0.75，
  门 Δ≥−1 **PASS**。波动全宽 6.25 > 惯常 ±2 噪声带，主要来自 rejection 类标签。
- llm_timeout 0；`tact_*` 事件 **0**（phase 无泄漏，接线对 HumDial 路径零影响的直接证据）；
  noisy 100/100、clean 80/80；TTS 轮数与 W2 逐夹一致 95/100，5 个差异夹（0001/0005/0020/0036/0080）
  均为 W2 两轮 → 本次一轮。
- **混杂**：本次 serving 栈与 6/23 对齐配置不同（85GB 卡 + 临时配置）；引擎侧无罪证据充分
  （phase-a 代码路径未动 + tact_*=0），5 夹轮数差最可能是边界 judge 决策在不同数值面下翻转。
- **回填两项**：(a) 对称基线——把 W2 存档输出同样重判 3 次，均值对均值（零 GPU，纯 DeepSeek）；
  (b) 记录本次 HumDial 生成所用 vLLM 配置文件名入档。

## 5. 结论

D1 收口：语义单源接线完成且钉死（core 全绿 + grid v1 存活 + 消融两列 12/17 vs 10/17 到手），
S2 与 HumDial 门通过。full 仪器交付了 D2 的第一批弹药（ε 带 EoU 存在性、fin23 刀口敏感性）。
无需新裁断；按 06 执行序进 D2（live 全量重测 + 首响分解）与零 GPU 并行批。
07 汇报稿**缓发**（用户裁定：并入 W3 总报告）。

## 6. D1.5 补充实验（2026-07-06 晚：混杂定界批 + 裁断 B 回填）

**(a) provenance 回填**：⑤ HumDial 生成栈 = `exp/w3/qwen3_omni_audio_84g.yaml`（音频 stage2
max_model_len 65536→8192、max_tokens→4096、三段显存 0.78/0.12/0.08、max_num_seqs=1；TTS 冒烟过，
vLLM 实报 max_model_len=8192）。

**(b) HumDial 对称基线**（W2 存档输出 deepseek-v4-flash 三判；隔离副本 `logs/humdial_w2r_rejudge_rep1..3`，
存档根未动）：Overall **60.13 / 63.62 / 59.75 → 61.17±1.94**（Interruption 84/86/82，Rejection
36.25/41.25/37.5）。两个结论：
- **judge 供方跨日漂移实测**：同一批 W2 输出 7/03 单判 63.88 → 7/06 三判均值 61.17（−2.7pt），
  漂移集中在 Rejection 子分。**纪律增补：门槛比较必须同日同批对称判，跨日 judge 数字不可相减。**
- 配对口径：W3 phase-a 63.25±3.13 vs W2 61.17±1.94（同日同 judge）→ **Δ=+2.08**，门以正增益通过；
  此前 −0.75 系"今日均值 vs 旧日单判"的不对称伪差，勘销。

**(c) server 漂移定界**（`scripts/w3_server_drift_probe.py`：10 条 W2 缓存键在 84g 栈绕缓存重调、
逐字节比对；`exp/w3/server_drift_probe.json`）：字节相同 8/10，**ops 级 10/10 一致**（两条漂移仅
say 措辞 / dialogue 标签）→ **OPS-STABLE**。fin23/hou17b 的 full 翻转主因坐实 **ε 前缀效应**；
server 数值面对工具决策无观测贡献（n=10 定界，非全称）。附带：say 表面字段跨栈不保稳——重生成
网格 transcript 措辞可能微漂，exact 轨无感、judge 轨或有微扰。

**(d) 首响分解**（`scripts/w3_first_decomp.py`，零 GPU，读 w2r_tact_full live 档；
`exp/w3/first_decomp.json`）：**first = 0.64 hold + infer_final，恒等式逐例零残差**（89/99 ack 路）。
infer_final p50 **0.501** / p90 0.74（0.444 是全决策池 p50；首响锚定末决策）。fallback 10 例
（say 空→result-ready 锚，R9 残余）+ 1 例无 first。**裁断 B 回填**：决策确在首响关键路径（严格
可加）→ speculative_dispatch 维持 D6 实装不降级；投影 first' = max(0.64, infer_final)：
**p50 0.64 / p90 0.74**（19/89 例 infer>hold 吃不满）；P3 首响比 = 1.141/1.452 = **0.786 现状**
→ 投机后 0.64/1.452 = **0.441 ≤ 50% 达标**（06 预注册 ~48% 方向兑现，实测更优）。架构注记：
v1 为融合单决策，06 所虑 "judge+Phase-B 串行 0.844s" 不适用。

## 7. 零 GPU 并行批（2026-07-06 晚，06 §三指定项）

**(a) oracle 前沿三数（E4 ✓；`scripts/w3_oracle_frontier.py` → `exp/w3/oracle_frontier.json`）**：
名义钟（max t_commit − t_user_end，墙钟免疫）：全量 96 个可比场景
**固定臂保费 123.8s / oracle-A（rollback 旗标 15 场景开窗）20.9s / 回收上限 83.1%**
（墙钟口径 124.4s/83.2% 一致；均值保费 1.29s/场景 ≈ δ* 减语音重叠）。机制级 oracle-B
（只在真暴露夹开窗，可比 2 夹）3.2s → 回收 97.4%。落在 06 预测的 79–95% 带内。
**G2'(ii) 靶具体化：自适应臂须相对固定臂回收 ≥51.5s/百场景。**
travel_10 成本比：δ=3.91 救援路完成 4.96s vs 宣告即撤销路 2.55s，**保费比 2.61×**，
且救援路在固定臂上涟漪 +241s/百场景——cancel 规则是零边际保费救援，坐实进 prompt 五靶#2。
Caveat：4 场景单臂零提交被跳过（eco12/fin23/hou24/hou25）；oracle-B 因 hou25 blocking 零提交仅计 2 夹。

**(b) hou11 普查 + 官方输入契约审计（裁断③ 闭合；`scripts/w3_hou11_census.py`）**：
全量 100 全部为多轮对话；**gold 参数仅存在于前文轮的场景 = 2/100（housing_11、ecommerce_12）**；
契约审计：官方 agent instructions 为固定字符串、benchmark_data 只用于枚举判分——
**官方 SUT 同样拿不到前文** ⇒ 裁断③"基准属性"分支：不加 feed_history，两场景对一切合规系统
不可赢，评测节注记；台账 §4E 归因码改判。"nowhere" 类启发式命中 52 场景 = verbatim/格式
表面（裁断 C 素材池，非缺陷计数）。
**链式计数（裁断 A-iv）：≥2 无依赖调用 = 28/100（另 10 个带 $RESULT 依赖）**——远超"<10 则单薄"线，
DAG 并行 claim 有据；但链式**修订**覆盖为零 ⇒ 已写进 RB L7 类目。

**(c) RB 设计文档 v1**（`docs/rb_design.md`）：规模 80（功效算术 n≈84 为稳妥上端、60 下限）；
七层分层含 06 指定两类（决策在途竞态区 L4、多重跨 EoU L6）+ D1 实测催生的 **ε 带 L3** + 链式修订 L7；
post-EoU ≥55%；录音落点用过采样+事后分箱；L4 追加屏障消融的 held-out 复验设计。

**(d) 文书**：04 汇报净胜明细挂勘误指针（原文不改）；McNemar 4 对 p=0.625 随指针传播。

**(e) 递延**：λ(t) v2 + 图 2b 时间线渲染——需 SenseVoice 多 cue token 对齐工具（延展
w2r_delta_hist 机械），排下一批（GPU 日顺跑 ASR 更快）。

## 8. D2 live 全量重测（full-100 `--engine full`，w3_tactfull_live，nominal 1.0，
独立缓存，173 新决策 + 27 命中，84g 栈，~8 min）——**H6 结案**

- **live 56 / 100 vs 离线基准（w2r_tact_full）56 / 100，差距 +0**。E1 判据（差 ≤2 或逐例归因）
  双重满足：总差为零，且 8 个翻转逐例归因、两两相消（4 胜 4 负）：
  - **EoU 结构类 4 个**（ε 存在性带）：eco19 转挂（2→1 EoU）、ecommerce_25 转挂（6→5）、
    hou25 转挂（2→1）、hou17b 转过（3→2）——净 −2；eco25 是 ε 带在全量上的新实例。
  - **同结构决策类 4 个**（ε 平移前缀上的刀口决策；server 已定界 ops-stable）：fin04 转过、
    fin22 转挂、fin23 转过、**travel_23 转过**（疑点一之夹在 live 感知下通过）——净 +2。
  - 感知敏感率 ≈8%/夹（结构 4% + 决策 4%），聚合分不变——live 感知重排个体、不动总量。
- **A 档双跑（run2，201/201 缓存命中）：确定性字段 100/100 逐位一致**（调用/参数/名义戳/EoU/
  first/completion_nominal 全同）；仅 2 夹 task_completion_s 第三位小数抖（工具墙钟，非权威），
  E7 双跑纪律在 live 感知轨成立。
- **H6 裁定：结案**。旧"live 差距"经构造（同一引擎代码）+ 实测（总差 0）消灭；残差机制已具名
  （ε 带 + 前缀刀口），无系统性残差 ⇒ H1/H2/H5 排查不再针对 pass 率启动（其对象转为真 websocket
  路径的 TTS/回声问题，属演示链路不属评测轨）。
- **延迟口径落定（E1 收尾）**：pass = 离线 core 轨；延迟 = W2 串行 live 档（首响分解恒等式已闭合）
  + D6 终版 prompt 时标准卡复测；live 感知轨（full engine）负责 H6/鲁棒性叙事。本次 run 的延迟
  字段非权威（nominal + 临时卡）。
