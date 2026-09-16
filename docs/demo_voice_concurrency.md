# Demo 并发声音差异：Talker 批次数值路径

最终采用 `native` 模式：只在 Talker 工作进程优先使用 cuBLASLt，并关闭 BF16 的低精度归约与 split-K，保留原生算子和动态批处理。demo 启动器默认启用；通用服务默认关闭。

本轮沿用 `demo-voice-rng-v1` 和 `demo-speaker-embedding-v1`，不需要人工评分。完整协议与允许归仓的自造音频收据在 `exp/web_demo/voice_concurrency_v1/`；模型权重、原始音频、逐步张量和服务日志仅保留本机。

## 定位方式与证据边界

`scripts/check_demo_voice_concurrency.py` 固定6个目标文本（5中文、1英文）。每个条件34次完整合成：30次受控目标句、4次旧请求作为干扰。先逐句串行重复，再做两轮混合并发（每批3个受控请求＋1个旧请求；第二轮反转受控请求提交顺序），最后逐句串行恢复。每个条件同文比较为6对串行重复、12对并发、6对恢复；另有4对跨中文文本和1对跨语言比较。重复与共享参考不是独立文本样本。

服务为生产 seq4 配置，单张 RTX PRO 6000 Blackwell，条件之间串行运行，嵌入提取在 Omni 停止、GPU空闲后串行执行。这是工程诊断，不是正式 serial-eval 确定性/延迟基准，不能与历史 seq1 主表相减。无真实人类音频或物理设备，也没有异说话人负样本/校准阈值。

临时启用 `FDBC_DEMO_VOICE_TRACE_DIR=ABSOLUTE_PRIVATE_DIR`：只记录带 demo 声音契约的请求，在每一步残差声码预测后保留主声码、残差声码、输入 Talker 隐状态、文本向量、两条 RNG 的推进位置与活跃请求 ID。直接比较完整数值，不做完整性哈希。关闭变量即不读取张量、不写文件。CPU拷贝会扰动运行时序，因此必须另做关闭追踪的复核；不要将该模式的耗时用于性能结论，不要对真实会话启用。

基线追踪结果：12对并发比较均出现 Talker 隐状态差异，首次出现于残差解码第0–9步（从0计数）；最早差异处最大绝对值为0.125–0.5。残差声码随后或在同一步分歧，主声码随后分歧。所有共同解码步的文本向量与主/残差 RNG offset 一致；并发轨迹均观察到4个活跃请求。6对串行重复和6对恢复均 PCM 相等。

这些证据把此组样本的并发变化定位到 **Talker 批次相关数值计算先产生差异，再经采样与自回归反馈放大**。这是受控探针的机制解释，不是对所有真实会话的根因认定；尚未细分到单个算子，也不证明其他文本不会受 Thinker 或流式输入时序影响。

## 定位对照：完整批次不变模式

复用已安装 vLLM 的 `VLLM_BATCH_INVARIANT=1`，**只注入 stage 1（Talker）的原生 `env` 字段**。无需自写确定性算子、全局重置随机数、降低 temperature 或排队串行化请求。原有请求级 RNG 仍然必要。

启动开关：

```bash
FDBC_DEMO_VOICE_ADAPTER=1 FDBC_DEMO_TALKER_NUMERICS=invariant \
  QWEN_HOST=127.0.0.1 bash setup/start_qwen3omni_audio.sh
```

开关在 `setup/start_qwen3omni_vllm_omni.sh` 中从所选部署配置生成临时副本，仅给 stage1 添加：

```yaml
env:
  VLLM_BATCH_INVARIANT: "1"
```

配置派生由现有 `patch_omni_demo_voice.py` 完成，保留其他字段和已有 env，不维护第二套生产 YAML。当前上游版本的 `--stage-overrides` 不转发 runtime env，不能仅凭 CLI 参数存在就断言模式已经生效。

它是 **Talker 进程级计算模式**，对该部署中所有 Talker 请求生效，不是每请求动态切换。通用启动脚本默认关闭，必须同时启用 demo voice adapter；数值模式接受 off/native/invariant。原生产 YAML、stage0 Thinker、stage2 Code2Wav、legacy 源文件和 TACT/RB 配置不变。关闭启动开关恢复原计算模式；需要重启推理服务。

开启后，带追踪的12对并发比较中，Talker 隐状态、主声码和全部残差声码逐步完全一致，长度一致。最终 PCM 仍只有3/12对完全一致：差异已落到后续 Code2Wav/分块处理边界，不应谎报整条链路 bitwise 一致，也不应仅为消灭低幅度数值差异再扩大改动范围。说话人嵌入均值从0.895361升到0.999921，最低值从0.836878升到0.999771；简易自相关 F0 中位数和音频时长的同句并发差异全部为0。F0仍存在算法与八度误差边界。

ASR沿用原严格归一验证，不改打分规则：带追踪基线27/34，Talker invariant为29/34；两组均有5条中文数字与阿拉伯数字写法不匹配。基线另有2条并发结果回读多出一次“一步”，候选未出现。不能仅凭单个 ASR 的回读将这些结果写成普遍发音准确率。

跨句结果仍需单独面对：带追踪条件下，四对中文跨句均值从0.833601变为0.812867。该开关解决的是同文调度相关漂移，**没有建立跨文本身份一致性提升的证据**；分数不等于身份概率或主观自然度。

## 复现与自动迭代

服务启动后，backend 环境执行：

```bash
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python \
  scripts/check_demo_voice_concurrency.py --output NEW_DIR --tag UNIQUE_TAG --asr
# 若服务启用了追踪，再加 --trace-dir ABSOLUTE_PRIVATE_DIR。
```

停止该实验服务、确认GPU空闲后，用 `scripts/check_demo_speaker_embeddings.py --manifest NEW_DIR/manifest.json --output NEW_DIR/speaker.json` 提取嵌入；完整命令和权重说明见 `docs/demo_voice_rng.md`。脚本保留全部失败与分母，不按高分筛选音频；新目录排除原始 WAV/trace/log 后再归档。

首包与总完成时间由客户端 `perf_counter` 记录，覆盖真实 SSE 与文本契约核验，不包含事后音频统计/ASR。关闭追踪的耗时只作同机 seq4 工程诊断；暖启动串行汇总采用每句第二次合成，各条件6个样本，不作为正式论文延迟结果。

## 关闭追踪后的复核与代价

使用实际启动开关重新启动服务。核对三个运行中 worker 的环境：只有 stage1 的 `VLLM_BATCH_INVARIANT=1`，三个 worker 都未设置追踪目录，收据见 `runtime_validation.json`。

| 指标 | 默认数值路径 | Talker batch invariant |
| --- | --- | --- |
| 完整合成 | 34 | 34 |
| 同文并发对数 / 独立目标文本 | 12 / 6 | 12 / 6 |
| 并发 speaker cosine 均值 / 最小值 | 0.893413 / 0.863260 | 0.999942 / 0.999788 |
| 并发 PCM 完全相等 | 0/12 | 5/12 |
| ASR 严格归一匹配 | 28/34 | 29/34 |
| 暖启动串行首包中位数 | 0.226589s | 0.427893s |
| 暖启动串行完成时间中位数 | 1.046325s | 3.412376s |

串行重复与串行恢复，两条件均各6/6 PCM相等。候选所有12对并发的时长差和简易 F0 中位数差均为0。上述稳定性复核支持定位结论，但完成时间变为约3.26倍，单请求生成实时因子接近1；并发时还需共享计算资源。**因此不将完整 batch-invariant 模式设为 demo 默认**，仅保留显式诊断开关，不以稳定性分数掩盖交互速度代价。

四个完整条件共136次合成，全部保留。另有一次启动尚未监听端口时的连接失败，产生0段音频，原失败收据保留于 `untraced_invariant_startup_race/receipt.json`，不混入成功合成分母；不是删除低分结果后重跑。

## 最终采用：native 归约设置

进一步消融发现，仅设置 cuBLAS workspace / cuBLASLt preference 并不够：34次完整合成中，12对并发 cosine 均值为0.878051、最小0.854127，暖启动首包0.425130s、完成3.171419s，未采用。

最终只在 **Talker 模型加载前** 执行两项设置：

```python
torch.backends.cuda.preferred_blas_library(backend="cublaslt")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
```

保留 BF16 权重/乘法和现有原生算子；只禁用 BF16 低精度归约与 split-K。第二项在当前 PyTorch 2.11 上要求 cuBLASLt；缺少该后端选择的初始化曾失败，产生0次合成，摘要保存在 `startup_failures.json`。不限制 cuBLAS workspace，不注册完整模式的其他算子替换。

使用一个公开启动变量 `FDBC_DEMO_TALKER_NUMERICS`，取值 `off / native / invariant`。通用启动脚本默认 `off`；`setup/start_demo.sh` 的 Python 启动器默认 `native`，尊重用户显式覆盖。派生配置只给 stage1 注入内部 `FDBC_DEMO_TALKER_NATIVE_NUMERICS=1`；helper 还检查 `model_stage == "talker"`，避免误改 Thinker。`invariant` 是较慢的完整数值对照，不作为默认。

```bash
# 常规 demo 默认 native；关闭时显式覆盖并重启推理服务。
bash setup/start_demo.sh
FDBC_DEMO_TALKER_NUMERICS=off bash setup/start_demo.sh

# 单独启动 Omni，仍显式限制到 demo adapter。
FDBC_DEMO_VOICE_ADAPTER=1 FDBC_DEMO_TALKER_NUMERICS=native \
  QWEN_HOST=127.0.0.1 bash setup/start_qwen3omni_audio.sh
```

`--backend-only` 复用的既有 Omni 不会被自动改模式。原始生产/serial-eval YAML、采样参数和业务定时不变；新模式不是逐请求数值切换，在这个 demo 部署的 Talker 进程内生效。

无追踪 native 条件同样34次完整合成，12对并发 cosine **均值0.999942、最小0.999788**，PCM相等5/12；所有同文并发时长差和简易F0中位数差均为0，串行重复/恢复各6/6 PCM相等。暖启动串行首包中位数 **0.237342s**，比基线高约11ms；完成时间 **1.150418s**，比基线高约104ms。输出时长本身随数值模式发生变化，不能把总完成时间差全部解释成算子开销。

ASR严格匹配28/34，其中5条是固定数字写法差异，另1条是旧干扰请求的回读异常；30条受控目标中25条严格匹配，其余5条均为该数字写法差异。旧干扰请求的异常仍计入总分母。没有用低分筛除或人工判断修改结果。

这支持在本轮样本中用很小的数值设置修复同文并发漂移；不宣称所有硬件、所有文本、整条PCM链路 bitwise 一致，也不宣称跨句自然度或身份一致性已解决。

最终启动入口又完成一轮同样的34次合成（`final_native/`）：12对并发在全部解码步的文本向量、Talker 隐状态、主声码、残差声码、两条 RNG offset 均一致，且观察到4个活跃请求，证实未通过串行排队换取一致性。中英混合问候和包含 CR/LF/TAB 的两条实际 demo TTS 启动 canary 均首次 ASR 回读匹配；这是 TTS canary，不是完整浏览器/物理设备验收。

最终共7个完整条件、238次探针合成，另有2次独立TTS canary。失败连接与不兼容后端设置的启动失败均保留摘要，未生成音频，不混入完整合成分母。所有条件和开销汇总见 `exp/web_demo/voice_concurrency_v1/summary.json`；失败说明见 `startup_failures.json`。原始音频与逐步张量留在本机，提交仅含自造夹清单、聚合/逐夹指标和脱敏验证收据。
