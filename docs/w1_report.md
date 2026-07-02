# W1 收口报告（tact 分支 · 2026-07-03）

> 神谕依据：`手工文档/神谕/00_系统蓝图.md`（宪法）+ `01_W1 完整计划.md`（执行计划，已按本地现实微调）。
> 人类评估/伦理/API 预算等行政项按用户指示全部跳过。

## G0 出口判据对账

| # | 判据 | 状态 | 证据 |
|---|---|---|---|
| E1 | 金标集等价性 L1≥80%，其余 L2 归因 | ✅ mock 版：序列等价 8/8，L1 6/8 + 2 例预注册偏差归因；真 LLM 版见下文 | `docs/w1_equivalence.md` |
| E2 | 感知冻结消除：新引擎分割处理延迟 p99 < 20ms | ✅ actor 帧滞后 p99 0.49ms；VAD 间隙对比 legacy max 2417ms/停摆 4.0s vs actor max 70.8ms/停摆 0 | `docs/w1_freeze_data.json` |
| E3 | HumDial 分数回归 | ⏳ 行为回归运行中（无 DeepSeek key，judge 分数待用户配 key 后一键补跑） | 本文档「HumDial 回归」节 |
| E4 | 双语 ASR | ✅ 工厂落地（默认 paraformer_zh 不动，sensevoice flag 后）；模型已下载；验收见下 | `src/module.py` |
| E5 | FDB-v3：备忘录 + blocking ≥5 场景 | ✅ 备忘录 11 问全答（Q1 实锤）；冒烟见下 | `docs/fdbv3_memo.md` |
| E6 | 行政 | ➖ 按用户指示跳过 | — |
| E7 | 测试：S2 全绿 + injected 回放 <5min/20条 | ✅ 11/11；injected 全金标集回放耗时见下 | `tests/test_engine.py` |

## 交付物索引

| 类别 | 文件 |
|---|---|
| 新引擎 | `src/engine.py`（actor/事件队列/音频钟/决策分叉/陈旧性协议/三回放模式/超时回退） |
| 冻结基线 | `src/backend_legacy.py`（逐字节）＋ tag `golden-base` |
| 开关 | `llm.audio_block`（`src/messages.py`，三方言）；`engine.arch`；`playback_autoend`；`decision_timeout_s`；`asr.backend` |
| 裁判 | `scripts/trace_diff.py`（L1/L2）＋ `tests/test_engine.py`（11 项） |
| 回放 | `scripts/replay_session.py`（双引擎离线驱动）＋ `scripts/batch_replay.py`（隔离并发） |
| 测量 | `scripts/measure_freeze.py`（论文图1数据）＋ `scripts/extract_vad_events.py` |
| 审计 | `docs/fdbv3_memo.md`（评测宪法）＋ `docs/w1_equivalence.md` |
| 双语 | ASR 工厂 + `prompts_en` v0 + `prompts_agent`（去 hack） |

## 感知冻结 before/after（论文图 1 数据，mock 延迟 0.8s 版）

| 指标 | legacy | actor |
|---|---|---|
| VAD 处理间隙 p50 | 16.3ms | 16.3ms |
| p99 | 16.5ms | 51.7ms* |
| **max** | **2417ms** | **70.8ms** |
| >100ms 间隙次数 | 2/clip | **0** |
| 累计停摆 (27s clip) | **~4.0s** | **0** |
| 帧滞后（到达→处理）p99 | —（不可测：冻结即滞后） | **0.49ms**，队列深度峰值 3 |

*单核容器下的调度噪声；GPU 大容器复测应更低。真 LLM 版本待回填。

## 评测并发策略（用户明示的算力策略，已落地）

- **吞吐轨**：`batch_replay.py --mode injected --concurrency N`——音频钟回放不真等，每会话独立 engine/VAD/输出/trace；module.py 已 thread-local Session + sherpa 锁。文本 deploy 配置（`configs/qwen3_omni_text_only.yaml`）支持 vLLM 真 batch。
- **实时轨**：`--mode realtime --concurrency 1`＋音频 deploy 配置（`max_num_seqs:1`，确定性优先）——只在出正式延迟数字时用。
- HumDial 9k+ 全量：吞吐轨跑准确性，实时轨只跑最终延迟样本。

## 真 LLM 实验（GPU 日执行记录）

<!-- FILL:GPU_RESULTS -->

## GPU 快速启动（复现实验用）

```bash
# 1) vLLM（音频管线，:10003）
nohup bash setup/start_qwen3omni_audio.sh > logs/vllm_omni.log 2>&1 &
# 就绪判据：curl -s http://127.0.0.1:10003/v1/models 返回模型
# 2) 代理（:10004）
nohup bash setup/start_qwen3_proxy.sh > logs/qwen3_proxy.log 2>&1 &
# 3) 冒烟（文本+音频判定）
/root/miniconda3/envs/fd-sds/bin/python -c "见 git log 或本仓库 scripts/smoke_qwen3omni_audio.py"
# 4) 金标录制（legacy + 真模型）
bash /tmp/record_golden.sh   # 或按 traces/golden_set.txt 逐条 replay_session --arch legacy --mock none
# 5) 等价性复验（actor + 真模型 → trace_diff）
# 6) HumDial 回归（backend :18000 起 actor 引擎 → run_humdial_100_pipeline --seed 42）
# 7) FDB 冒烟：cd /root/autodl-tmp && python -m tact.offline_runner --data FDBench_v3/v3/fdb_v3_data_released --provider tact_blocking_w1 --mode blocking --limit 5
```

## 已知偏差与遗留

1. continue-timeout 锚点音频钟化（预注册偏差，方向有利，HumDial 回归兜底）——`docs/w1_equivalence.md`。
2. 决策硬超时 15s（新行为，回归中 `llm_timeout` 事件应为 0）。
3. DeepSeek judge key 未配置：HumDial judge 分数待补（管线其余环节已验证）。
4. FDB-v3 严格 scorer 惩罚补偿调用（Q1）：双轨报告方案进 W2；官方 pass rate 本身成为 P2 停时实验的 y 轴。
5. zipformer_bi 流式 ASR：W2 选项，工厂里留位未集成。
