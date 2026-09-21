# Demo 诊断线审查结论与实施闭环

日期：2026-09-21。完整架构、数据契约和操作方式见 `docs/demo_diagnostics.md`。

## 1. 审查基线

实施前的只读审计发现，逐轮 trace、单模型调用 case 和离线声音分析分别有效，但缺少统一会话清单、稳定调用标识、轮次 outcome 和完整状态机重放。54 个本机会话 trace 均闭合且无 trace drop；案例库有 800 个实际调用、覆盖 35 个会话，但只有 1 个 case review。该快照只说明诊断资产彼此割裂和标注闭环不足，不是正式数据集规模或模型质量结论。

## 2. P0/P1 实施状态

| 优先级 | 原缺口 | 当前实现 | 入口 |
| --- | --- | --- | --- |
| P0 | 缺会话有效配置和因果键 | allowlist manifest，统一 session/turn/call/parent/case/utterance ID | `src/demo_diagnostics.py`、`demo-trace-v2`、`demo-case-v2` |
| P0 | 只能单调用 replay | 保存帧时序、控制/VAD 边界和真实模型输出，注入当前 ActorEngine | `src/demo_session_replay.py`、`scripts/demo_diagnostics.py replay` |
| P0 | 事故靠人工翻日志 | 会话关闭自动生成 turn/span/summary，机械不变量转异常队列 | `summary.json`、`list --anomalies` |
| P0 | case 级 CLI 无法形成轮次标注 | 本机逐轮 review 页面、多轴标签、追加式修订 | `/demo/diagnostics.html` |
| P1 | 短停止缺采集前证据 | 明确 opt-in、有界、对齐的 mic/reference/clean 三轨 | 页面采集勾选、`diagnostics/capture/` |
| P1 | case elapsed 混入回压 | call/utterance/clock-sync 请求级 span | `spans.jsonl` |
| P1 | 容量满即停、无保留删除 | kind 分区、成功抽样开关、30天保留、跨资产删除 | 配置、CLI/API prune/delete |

实现保持在线判据不变。诊断 ID 不使用共享递增计数器；工作协程生成 ID 时不会修改引擎状态。多轨采集默认关闭，普通会话不会保存原始浏览器音轨。

## 3. 结构化结果

新的审查单位是 session → turn，而非孤立 call。每个会话现在可以回答：

- 当时运行的代码、协议和有效配置是什么；
- 某个输入经过哪些模型调用、repair/fallback 和候选分支；
- 输出是否被确认、公开、取消、播放或写入历史；
- 延迟发生在容量等待、模型首输出、TTS、socket、播放信用还是浏览器 ACK；
- 机械不变量是否失败，人工 reviewer 如何判断且判断是否被修订；
- 修复后能否用同一输入帧和保存的模型结果重新走当前 Actor 状态机。

旧的 `scripts/demo_cases.py` 保留为单调用检查工具。它与会话 replay 的证据口径在文档和命令中分开，不再用单调用重放代表端到端复现。

## 4. 未纳入本次工作的内容

原审查的 P2 是实际多句回答的逐句/滑窗声音轨迹、主观听评和阈值校准，以及进一步的聚合趋势产品化。本次按明确裁决不实现 P2，也不新增未经人工校准的音色身份或自然度结论。现有 `demo-speaker-embedding-v1` 仍只用于离线客观对比，不接入在线判断。

## 5. 后续使用纪律

- 先看异常队列和完整轮次，再决定是否提升为人工 gold；自动输出永远不是标签。
- 需要定位采集前丢音时，在受控会话逐次勾选多轨采集，不长期默认开启。
- 版本比较使用固定已审集合；供应方噪声或样本不足时不写成普遍改善。
- 真实会话、录音和 review 留在私有 gitignored 目录；到期 prune，按会话 delete 可删除所有关联资产。
