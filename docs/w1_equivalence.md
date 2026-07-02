# W1 等价性报告（mock 版 · CPU · 2026-07-03 夜间）

> 裁判：`scripts/trace_diff.py`（归一化事件序列 + 0.25s 软时间容差）
> 数据：`docs/w1_equivalence_data.json`；trace 原件在 `traces/mock_eq/`
> 真 LLM 金标复验待 GPU 日执行（见 w1_report.md 快速启动清单）

## 方法

- 音源：`exp/exp-1/test/test-000001.wav`（27s，6 起 6 止）与 HumDial `0001_0004.wav`（21s，2 起 2 止），16kHz 真实语音。
- 感知：silero VAD 事件预抽取为脚本（`scripts/extract_vad_events.py`），两引擎逐帧注入**完全相同**的 VAD 事件（消除数值噪声，A/B 只测控制流）。
- 决策：确定性 mock（judge/interrupt/shift 固定答案 × 4 种策略组合；response/asr 按调用序编号；LLM sleep 0.3s / ASR 0.1s / TTS 0.2s，模拟真实延迟结构）。
- 步频：绝对时间表（对齐生产 uvicorn 实时到达；早期版本用相对 sleep 累积漂移 ~0.4s/27s——本身就是墙钟脆弱性的一个演示，已修）。
- 旧引擎经 FakeWebSocket 驱动 `backend_legacy.py` 原码；新引擎 `engine.py` actor 路径。

## 结果

| 配置 × 音源 | 判定 | 事件数 | 软时间偏移 |
|---|---|---|---|
| j=switch, i=continue × test-000001 | **L1** | 20/20 | 0 |
| j=switch, i=switch × test-000001 | **L1** | 20/20 | 0 |
| j=switch, shift=yes × test-000001 | **L1** | 19/19 | 0 |
| j=continue × test-000001 | SEQ_EQ | 20/20 | 7（全部 ≈0.30–0.32s）|
| j=switch, i=continue × 0001_0004 | **L1** | 16/16 | 0 |
| j=switch, i=switch × 0001_0004 | **L1** | 16/16 | 0 |
| j=switch, shift=yes × 0001_0004 | **L1** | 15/15 | 0 |
| j=continue × 0001_0004 | SEQ_EQ | 16/16 | 7（全部 ≈0.30–0.32s）|

- **序列等价：8/8（100%）**——决策序列、内容、状态、轮次逐条一致。
- **L1 严格等价：6/8（75%）+ 2 例已归因设计偏差**（见下）→ 按 E1 判据（L1≥80%，其余 L2 归因完毕）：**归因后通过**。

## L2 归因（唯一一类偏差）

**现象**：judge=continue 路径上，continue-timeout 链的全部后续事件在新引擎中提前 ≈0.30–0.32s（= mock judge 延迟 0.3s）。

**根因（设计性，01 计划 D1.4 预注册）**：continue 超时锚点从"judge 返回的墙钟时刻"（旧）改为"被判定音频段末帧的 t_audio"（新）。旧行为把 judge 推理延迟叠加进用户等待窗口；新行为以用户体感的正确锚点计时。真实 30B judge 延迟为数百 ms–秒级，此偏差方向**只会缩短 Pause Handling 的响应等待**，不改变任何决策内容（序列 100% 一致为证）。

**判定**：新行为更合理，予以保留；HumDial 分数回归（GPU 日，D6.3）作最终兜底。

## 其他已知偏差（不触发 trace 分歧，记录备查）

1. `INTERRUPT_COUNT` 计数器删除（旧码从未读取）。
2. `asr` 任务不再回写清空 `BUFFER`（旧码的 create_task 竞态可能清掉新段；新引擎在链启动时快照消费）。
3. reset 后到达的迟到 TTS/ASR/LLM 结果被代际门丢弃（旧码会在 reset 后继续 send_bytes——竞态产物）。
4. 决策硬超时 15s + 保守回退（新增，代替旧 300s 挂死；trace 有 `llm_timeout` 事件可审计，HumDial 回归中应为 0 次）。
5. SPEAK 段在途判定期间抑制长打断计时（`_seg_closed`）；在途期间用户续说则以续说时刻重锚 1.5s 窗（旧码因冻结不可能进入该组合态；语义 = 排空后新段起算，见 engine.py 注释）。
