# f1_wait harness 与分析修复登记

登记时间：2026-10-10T03:10Z（UTC），先导块开始之前。登记人：第二轮运行子代理。

被测代码：`/root/autodl-tmp/fd-badcat-c3-f1_wait` 的 `082eb02`（父提交 `56a4dbd` = cheap3-base），已核对 HEAD、`src` 下无改动。

`prereg.json`、`arms.json`、`manifest.json`（sha1 `87d645de…`）都没有改动。下列修改只涉及 harness 观测、进度日志和分析器口径，阈值、主臂、配对单位、场景集都不变。修改前的原文件保存为同目录下的 `*.orig_r1`。

此前唯一的真实运行是构建阶段的冒烟（`runs/smoke`，1 条 P_timeout 中文条目），不进入任何判定。本轮全部数据（先导块、全量、消融）都在这些修改落档之后产生，所以没有任何条目需要重跑。

## 修改清单

| 编号 | 文件 | 问题 | 证据 | 修改 | 受影响的臂与条目 | 是否重跑 |
|---|---|---|---|---|---|---|
| H1 | `f1_analyze.py` | M1 的最低样本量拿处理臂有效试验数 `treatment_rate[1]` 和 30 比较；prereg 写的是“有效配对单位 ≥ 30” | 审计合成数据 `syn_improve`：配对有效单位 29、处理臂有效试验 37，原脚本判 M1 通过 | 改用 `m1["paired_units"] >= min_valid_units` | M1，off 与 on_max1，P_timeout 与 P_error 全部条目 | 否（只改分析） |
| H2 | `f1_analyze.py` | S1、S2、S2_per_scenario、S2_jitter、S3、S4、S5、followup 在 n=0 或 CI 为 None 时判通过 | 审计合成数据 `syn_noN1`（S5 在 0/0 时通过）、`syn_lowpair`（followup 的 CI 为 None 时通过） | 没有数据时判定为 None（“无数据”），不算通过；`PASS` 要求每项都是 True；新增 `criteria_n` 报告每项 n，新增 `no_data` 与 `failed` 列表。S1 和 S2_per_scenario 在 prereg 范围内任一负样本场景 n=0 时也判“无数据”。S5 按 prereg 原文只计恢复语音播放次数（原脚本把“只有 `input_cut_recovery` 事件、没有播放”也算进去），恢复事件数另列为 `S5_resume_recovery_events`。S2 合并 CI 整段 < 0 时加注“因改善而失败”，判定本身不改 | 全部副作用准则，所有负样本条目 | 否（只改分析） |
| H3 | `f1_analyze.py` | 排除数只报总数，不分臂、不分状态；重试次数没有报告 | 审计 validity_threats 第 7 条 | 新增 `exclusions_by_arm`（臂 × 状态）、`retries`、`units_over_two_attempts`、`ok_units_by_arm` | 报告字段，不影响判定 | 否 |
| H4 | `f1_run.py` `summarize()` | `client_recover_text` 只匹配“再说一遍”，抓不到英文恢复句 “Sorry, I didn't catch that. Could you say it again?”；而且按单个文本增量匹配，如果句子被拆成几段会漏掉 | `src/guarded_turns.py:53` 的 `RECOVER_TEXT`；P 集 12 个条目中 6 个是英文 | 先按 `utterance_id` 拼接 `speech_text_delta`，再同时匹配中文和英文恢复句；新增 `client_recover_lang` | 观测字段，两臂对称；可以用 `--resummarize` 离线重算 | 否 |
| H5 | `f1_run.py` `Trial.run()` | 没有抓页面，“前端显示了恢复文本”无从核对 | 第一轮 harness_changes 第 3 条 | 点 `#stop` 前读取 `#messages` 的 `innerText`，存为 trial 目录下的 `page_text.txt`，并新增 `page_recover_lang` 字段。只读，不改变行为，两臂对称 | 所有臂、所有条目 | 否（本轮数据全部在此之后产生） |
| H6 | `f1_run.py` `main()` | 只按 trial 打印进度，没有每个 rep × arm 组合的汇总 | 任务要求“每个 rep×arm 组合记日志” | 每个组合结束时打印一行 `[block]`，内容包括臂、重复、试验数、ok 数、耗时、本块与累计 Omni 请求数、与生产连接重叠的试验数 | 只是日志 | 否 |
| H7 | `f1_run.py` | Omni 与生产共用，6～7 小时的运行可能和真人使用重叠，事后无法标出重叠的试验 | 审计 validity_threats 第 4 条 | 每条试验在开始和结束时只读采样一次生产 backend（127.0.0.1:18000）的已建立 TCP 连接数，写入 `prod_conn_start`/`prod_conn_end`。驱动脚本另起一个只读监控，每 30 s 记录一次 GPU 利用率和这个连接数，写入 `load_monitor.log`。只用于事后的敏感性分析，不参与判定 | 所有条目 | 否 |
| H8 | `f1_run.py` `summarize()` | 人工检查 trace 时需要核对“注入只作用于测试 span”和“作为 cut 的 uay 是否就是测试 span”（审计 validity_threats 第 6 条：uay 没有限定为 test_input_id） | 审计 validity_threats 第 6 条 | 新增描述字段 `cut_is_test_span`、`injected_outside_test`（非回声条目中被注入、却不在 {test_input_id, second_input_id} 里的 input_id）和 `item_lang`。uay 与死寂的定义保持不变 | 描述字段 | 否 |
| H9 | 驱动脚本 `full_driver.sh` | 主比较预算 9500 可能不够（估计 8100 ± 20%）；rep2 的顺序是 on→off，预算一旦耗尽，被截掉的是对照臂的 rep2 | 审计 validity_threats 第 5 条 | 先导块后按实测的每试验 Omni 请求数重算预算，主比较优先，总硬上限 12000 不变。主比较跑完一遍后再跑一遍，用来补跑出错的单元（prereg：每个单元最多重试 1 次，`--max-attempts 2`），然后才开始消融 | 预算分配和重试时机 | 否 |

## 独立交叉核对

审计者的独立脚本 `audit_f1.py`（只依据 prereg 编写）放在 `audit/`，全量结束后与修复后的 `f1_analyze.py` 并行运行。两者结论不一致时，以更保守的一方为准。

修复后的 `f1_analyze.py` 已在审计的 4 套合成数据上和 `audit_f1.py` 对过，每项判定都一致。唯一的例外是 `syn_noN1`：修复后的分析器因为 N1_resume 缺失，把 S1 和 S2_per_scenario 判为“无数据”；审计脚本判这两项通过。修复后的分析器更保守。

## 先导块后的补充登记（2026-10-10T03:21Z，全量开始之前）

先导块（`runs/pilot`：off 与 on_max1 各 15 条；`runs/pilot_abl`：on_max2 与 on_max1_rel3 各 2 条）在上面 H1～H9 落档之后运行。先导块数据不进入任何判定。

| 编号 | 文件 | 问题 | 证据 | 修改 | 受影响的臂与条目 | 是否重跑 |
|---|---|---|---|---|---|---|
| H10 | `f1_run.py` `Trial.run()` | runner 的本意是“续说和跟进输入永远不注入故障”，靠 `input_ids` 白名单实现。但生产在保持窗内会把续说并入**同一个** `input_id`，所以续说也被注入了 timeout。结果是两臂的续说都没有回答（bad_outcome 两臂都必然为真），on 臂因此必然触发恢复（S1/S5 被构造性判失败） | `runs/pilot` 两条 N1_resume（en d0.8、zh d1.6）：测试 span 与续说共用 input_id 3 / 2；`injected` 对同一个 id 记了 3 次；合并后的闭合决策 `timed_out=true`，随后 `insufficient_evidence`；off 臂 `answer_playbacks=0`，on 臂播出了恢复语 | 注入续说音频之前先解除故障（`self.fault()`）。测试 span 的闭合决策在它自己的 VAD 结束时就已发出（先导块中都比续说注入早 ≥ 0.5 s），所以仍然会被注入。S_double_cut（`on_recover`）的第二次截断照常注入，不受影响 | N1_resume（24 条）、D_resume_late（12 条），两臂对称；rel3 消融里的这两类条目 | 先导块中这两个场景用修复后的代码重跑（`runs/pilot2`）；全量全部在修复后运行 |

先导块中观察到、但**不修改**的现象（如实记录，不属于 harness 缺陷）：

1. P_error 在生产路径上确实变成闭合 keep，但走的是 `decide_input_route` 内部的异常兜底：`fallback=true`、`timed_out=false`、`audit.error=None`，恢复原因记为 `fallback`，不是 `error`。
2. 测试 span 只有在闭合之前已被临时决策准入，才会被注入（`f1_backend.py` 要求 `span.admitted`）。P_error zh_short_request_01 的临时决策全部过期，闭合时才准入，所以没有注入，`uay=0`，按 prereg 计为无效单元。
3. VAD 可能把测试句切成两段。例如 P_error en_followup_01 的句尾 “typically?” 成了新 span 3；它不在注入范围内，生产把被拒 span 的音频作为上下文带入，完整回答了问题，两臂都不死寂。这是条目加生产 VAD 的性质。如果为此扩大注入范围，就要改 `max_spans` 的语义，还要放宽 `admitted` 条件，所以不改；按条目报告。
4. N4_echo_fault（none_echo_a）两臂都没有准入任何 span，所以没有注入。这是回声被正确拒绝。
5. N3_stop_fault（stopfrag_0）两臂都被判为 `yield_ready` 并回答了，两臂对称，属于条目或生产的 stop 识别问题。
6. on_max1_rel3 的 D_resume_late：恢复语的文本已经推到页面（`page_recover_lang=['zh']`），但语音在开播前被续说取消（`recover_playbacks=0`）。这是“只显示文本、没有出声”的界面现象，记入消融描述。
7. 每条试验平均约 15 次 Omni 请求、约 25 s，生产连接数一直为 0，GPU 上没有其他负载。

H10 之后的文件 sha256：
```
2026-10-10T03:22:28Z
40a7389aca9a673425586ad45d3d602094a7e09a867879005d2b65b896b72abe  f1_run.py
09cd718712fc0e00c3090c35f48a8a400348caf8c08bc40408fa430993e5d01b  f1_analyze.py
7f5c5e278a30f9a182b3ac7fee06139fbb99f0308ce9007fdc31fe012a8e1dc0  f1_backend.py
6ef97ce161b4bfe946d7f014150a2b27a3dcf074dcde86333e3aec809565050b  prereg.json
efb8761a08e0c4d3fbebf68bb5ee8f177c138841819f30d31cb12fe41af98a72  arms.json
10176e751c68b6b11fd8451c24540aa64ea2f02e069ae886b1b7cdcf32892d03  manifest.json
```
