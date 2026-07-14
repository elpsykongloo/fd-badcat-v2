# FDB-v3 深审计备忘录（W1 D5.1 — 评测设计的宪法）

> 依据：直读 `/root/autodl-tmp/FDBench_v3/v3/` 源码 + 既有 tact_* 实验报告 + `tact/INTEGRATION.md`（上一轮 sprint 的集成指南；原独立包已于 2026-07-14 归入本仓，其结论本备忘录已复核）。
> 状态标记：✅=源码实锤 ⚠️=需 GPU/在线验证 📌=行动项
> 更新：2026-07-03（W1 Day 0 夜间，无 GPU）

## 执行摘要（三句话）

1. **Q1 实锤：官方严格 pass rate 结构性惩罚补偿与多余调用**——`evaluate_pass_rate.py` 多重集匹配要求 recall=1 ∧ precision=1，任何 extra call（含同名重复）直接 FAIL；参数核对按 `pop(0)` 对齐首次调用，早发射的错误参数造成双重惩罚。**蓝图预判的全文最大单点风险成立**，双轨报告方案（终态正确性 + 官方分）必须执行。
2. 但这不是坏消息：judge 惩罚 extra call ⟺ 把 P2 的 C_κ 具体化了——eager launch（低延迟+脏轨迹）vs deferred commit（高延迟+净轨迹）的权衡在官方分上**可测量**，commit 停时策略成为直接优化官方分的杠杆。
3. 集成管道已被上一轮 sprint 验证：`tact/offline_runner.py` 产出的 result.json 在官方 scorer 上跑通（plumbing 100% pass 冒烟），已有实测：**tact_blocking 0.73 / tact_async 0.71 / tact_async_live 0.65**（2026-06-23，对比蓝图记载 GPT-Realtime ≈0.600）。

## 11 问逐答

**Q1 判分语义（最高优先）** ✅
- 两套评分：`evaluate_tool_calls.py`（连续分：tool F1 + argument acc + response acc）与 `evaluate_pass_rate.py`（严格二值）。
- 严格 pass = 名字多重集完全匹配（missing=[] ∧ unexpected=[]）∧ 全部参数正确。**`book→cancel→book(θ′)` 之类补偿序列必 FAIL**（cancel 是 unexpected；第二个 book 也是 unexpected——多重集只匹配一次）。
- 参数判定：LLM judge（默认 gpt-4o，可换 `FDB_LLM_*` 自定义端点）语义匹配，规则宽松（`$RESULT_0.x` 动态引用、格式/别名/±5% 数值容差、多余参数忽略）；无 `--use-llm` 时退化为精确匹配。
- 同名多次调用：argument 检查取**第一次**调用的参数（`actual_by_func[func].pop(0)`）→ Paris→Berlin 修订场景中先调 Paris 会被拿去对 Berlin 的期望参数。
- 📌 双轨报告：加一个我们自己的终态判分器（挂在 mock sandbox 状态上）+ 官方分并列；给上游的 issue/PR 草稿在 W2 写。
- 📌 论文侧：官方 scorer 的严格性 = C_κ 的操作化定义，P2 膝点实验直接以官方 pass rate 为 y 轴之一。

**Q2 用户音频投递语义** ✅
- 固定时间轴、非反应式：`livekit_inference.py` 把 `input.wav`（48kHz→重采样发布）按实时速率推进 LiveKit 房间，同时录制 agent 音轨；用户侧不因 agent 输出而改变。
- 修订事件的时刻 = 音频里说出更正的时刻（`state_rollback_details` 只给参数对，不给时间戳——修订时刻要自己从 wav/转写对齐，W4 hazard 标注时处理）。⚠️ 捕获窗口的确切尾长（输入结束后还录多久）需 GPU 日在线验证。
- 我方 fd-badcat 前端语义相同（`frontend.py` 固定回放）——离线回放框架与 harness 语义一致 ✓。

**Q3 Pass@1 定义** ✅ 就是严格二值 pass rate 的均值（100 例）；报告含 domain/difficulty/disfluency/rollback 分解。无域间加权——直接平均。

**Q4 turn-take / interruption 判定** — v3 不含这两项（那是 v1/v1.5/v2 的指标）。v3 只有工具三件套 + 延迟三件套 + response acc。HumDial 回归继续用仓库自带 `evaluation/`。✅

**Q5 延迟锚点** ✅ `analyze_tool_latency.py` + `measure_latency_from_audio`：
- First Response = 输入音频结束 → 输出音轨第一段非静音（-40dBFS 阈值）。
- Tool Call Latency = user speech end → 首次工具调用（时间戳相对 stream_start 归一）。
- Task Completion = user speech end → 回复中关键信息出现（LLM judge 定位）。
- 📌 注意：锚点是"输入音频结束"，不是"用户语句 EoU"——尾部静默计入我们的响应窗口，与 HumDial 的度量不同。

**Q6 工具沙箱** ✅ `mock_apis.py`：12 个纯本地函数（无网络、无状态持久——每次调用独立返回固定形状结果）+ `latency_injector.py` 延迟档（instant/fast/normal/slow/progressive…，场景带 `latency_profile` 字段）。
- ⚠️ 陷阱：延迟抖动用未播种的 `random.randint` → 不可复现。📌 我方运行时固定 seed 或用 fixed_ms 档；P2 实验必须如此。
- 📌 W4 模拟器可直接复用该沙箱 + 对数正态拟合替换延迟档。
- 沙箱无真实状态机（book_flight 永远成功返回 B789）→ "终态判分"需要我们给沙箱加一层状态记录（W2，小工程）。

**Q7 被测系统接口契约** ✅ 两层：
- 在线层：LiveKit room（LiveKit Cloud 账号 + .env.local 密钥）。agent 作为房间参与者。**上一轮已写好 `tact_livekit_agent.py`（614 行，常驻 agent，含 silero VAD 事件桥接）与 `tact_livekit_inference.py`**——fd-badcat 不必写新 adapter，改造点是把其决策回路换成新 actor engine（W2）。
- 离线层（快迭代主路）：评测器只读 `fdb_v3_data_released/{id}/result_{provider}.json`。**契约字段：`example_id / provider / actual_tool_calls[{function,args,timestamp_start,timestamp_end}] / transcript / status`**。`tact/offline_runner.py` 已验证此路（100% plumbing 冒烟）。
- 📌 W1 我方 blocking 冒烟走离线层即可，不需要 LiveKit/GPU 之外的任何云资源。

**Q8 21 个 rollback 场景** ✅ `state_rollback_test: true` 标记（100 例中 21 个），带 `state_rollback_details{original_param, corrected_param}`；`expected_tool_calls` 只含**修正后**的最终调用。pass-rate 报告自动按此分解。ID 清单可由 jq 一键导出（见文末命令）。

**Q9 不流利标注** ✅ 场景级 `disfluency_features` 枚举（SELF_CORRECTION 等 5 类）+ `dialogue[].user_annotated` 文本标注。⚠️ 无词级时间戳——hazard 头训练需要自己做强制对齐（W4-W5，可用 harness 自带 parakeet ASR 的对齐输出）。

**Q10 语言与音频规格** ✅ 纯英文、真人录音、48kHz（inference 侧重采样）、100 例=79 场景×12 说话人去重组合；4 域 × 3 难度（easy=1 call / medium=2 / hard=3）。

**Q11 许可/训练卫生** ⚠️ 仓库 README 未见显式许可条款；数据经 Google Drive 分发。📌 邮件问作者（daniel094144@gmail.com）dev/train 用途边界；hazard 训练优先用 HumDial + TTS 增强，FDB 数据默认仅评测。

## 既有资产盘点（上一轮 sprint 遗产，勿重复造轮子）

| 资产 | 位置 | 状态 |
|---|---|---|
| TACT 原型包（事务代数/工具注册/决策器/离线 runner） | `tact/`（原独立仓历史已归入本仓） | 可用；decider 走文本 LLM（module_adapter），未接 Qwen3-Omni 音频决策 |
| LiveKit 常驻 agent + 推流器 | `FDBench_v3/v3/tact_livekit_{agent,inference}.py` | 已跑通线上评测（0.65） |
| blocking / async 基线分数 | `v3/tact_*_report*.json` | blocking 0.73 / async 0.71 / async_live 0.65（6/23） |
| 集成指南 | `tact/INTEGRATION.md` | 结论已复核，audio_url 判断与本备忘录一致 |
| 调参子集 | `v3/tuning_subsets/` | 防过拟合的场景切分 |

📌 蓝图（00）对系统的核心要求是 **audio-native 决策中枢**（Qwen3-Omni 直接听音频）——现 tact 包 decider 是文本路（消融基线现成！），W2 的正主是把 fd-badcat 新引擎（今晚已 actor 化）+ Qwen3-Omni 音频决策接到这个已验证的 harness 管道上。

## 常用命令

```bash
# rollback 21 场景 ID 清单
python - <<'PY'
import json; d=json.load(open('/root/autodl-tmp/FDBench_v3/v3/benchmark_data_v2.json'))
print([s['id'] for s in d['scenarios'] if s.get('state_rollback_test')])
PY
# 离线评测（不需要 LiveKit；--use-llm 需 judge 端点）
cd /root/autodl-tmp/FDBench_v3/v3 && python evaluate_pass_rate.py \
  --benchmark benchmark_data_v2.json --results-dir fdb_v3_data_released \
  --provider tact_blocking --output tact_blocking_pass_rate_report.json
```
