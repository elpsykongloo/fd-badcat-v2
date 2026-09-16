# Demo 声音身份与请求级声码随机流（demo-voice-rng-v1）

## 范围与契约

本改动只处理 demo 的可诊断性与采样可控性，不承诺跨文本音色一致、自然度提升或整条 GPU 链路 bitwise 确定性。

- `configs/demo_chat.yaml`：`engine.demo_voice_control: true`，固定 `demo_voice_speaker: chelsie`，默认 `demo_voice_seed: 42`。关闭该开关恢复原流式 TTS 请求；legacy、HumDial 基础配置、TACT/RB 不启用。
- 只作用于协商成功的 Actor PCM 流式路径。原整 WAV `omni_tts_payload()` 不变；非 demo 配置显式启用开关会失败。
- v1 只允许 `chelsie`，seed 必须是 `[0, 2**63)` 内的整数；无效配置在请求前失败。
- 请求携带 `voice=chelsie`、`seed` 和 `vllm_xargs.fd_demo_tts_rng=demo-voice-rng-v1`。仍使用原逐字 grammar。
- 服务端把该 seed 接到 Talker 主声码采样以及独立的残差声码生成器。两者各自有随机状态，残差阶段保留原 top-k/top-p 分布，不改为 greedy。
- 原文本完成 SSE 额外返回 `fd_tts_voice={contract,speaker,seed}`，表示已接受的配置；客户端在释放 PCM 前必须核对。这不是感知音质或逐样本确定性证明。

## 底层接入

demo 启动器设置 `FDBC_DEMO_VOICE_ADAPTER=1`，使 `scripts/patch_omni_demo_voice.py` 在已有 verbatim adapter 之后执行；非 demo 启动脚本默认不安装此适配。手动部署 Omni 供 demo 使用时同样设置该环境变量；`--backend-only` 的既有服务若未安装，预热会因缺少配置确认而失败。

适配校验全部源码位置后安装 `scripts/omni_demo_voice.py` 为依赖内 helper，再修改四个接入点。不计算完整性哈希；未知/部分补丁布局拒绝启动。

复用当前 vLLM-Omni runner 已有的请求级生成器管理：

1. `SamplingParams.extra_args.fd_omni_request_seed` 标识 demo 请求，其他请求参数不变。
2. RNG 以内部 request ID 保存，批次位置变化不改变归属；生成器只初始化一次，继续解码时推进自己的状态。
3. 带显式 seed 的残差预测按请求逐行执行。含此类请求的 MTP 执行走 eager，避免 replay 已捕获的全局采样；Thinker/Talker 主网络仍使用原执行配置。
4. 完成或取消由原 runner 的 `finished_req_ids` 清理生成器；暂时未被调度不清理。
5. 将 generator 经 Omni `talker_mtp` → `code_predictor_forward` 传入残差采样。关闭开关时保留原调用路径与全局 RNG；从不重置全局 `torch.manual_seed()`。

隔离的是请求随机流，不是全部数值效应：主模型动态组批的浮点差异、Thinker→Talker 的异步文本/音频调度等仍可能影响最终音频。逐行残差执行也可能有吞吐代价，不能假设免费。

## 合成对照计划

执行脚本：`scripts/check_demo_voice_rng.py --output NEW_DIR --asr`，backend 环境，禁用本地代理继承。单进程逐调用 await；期间不运行浏览器或其他 GPU 实验。

在同一补丁后生产 seq4 服务上对照三臂：

- legacy：原 verbatim 请求，隐式默认 speaker、原残差 RNG；不是 `backend_legacy.py` 路径。
- speaker：仅显式固定 chelsie，原残差 RNG。
- controlled：显式 chelsie + 请求级残差 RNG；seed 固定42。

每臂使用两个自造中文目标句，各运行 `A,A,B,A,C,A`；B/C 为自造中/英文干扰句。每臂12次、总36次合成；24次目标句、12次干扰。每个目标以第一次为参考，三次后续比较；每臂6对、总18对。

记录 PCM 直接相等、样本长度、首个不同样本、公共前缀未对齐 RMSE、RMS、简易自相关 F0（含有效帧分母）、ASR 回读；不计算文件哈希。F0 可能有八度误差，RMSE 不代表音色相似度；重复句不是独立文本样本。有效独立目标文本分母为2，无真实人类录音/物理设备/听感验收。

生产 seq4 服务下的墙钟耗时仅作诊断，不可与历史 seq1 论文延迟比较。并发混批与取消另行烟测，不纳入该串行对照分母。

## 验证结果

2026-09-16，真实本地 Qwen3-Omni 服务，RTX PRO 6000 Blackwell；**请求串行、服务仍为生产 seq4**。这是固定文本的工程探针，不是按 serial-eval 专机配置建立的正式确定性或延迟结果，不与历史主表相减。

| 请求臂 | 与本臂首次合成逐样本一致 | 目标句0时长范围 | 目标句1时长范围 |
| --- | --- | --- | --- |
| 原请求 | 0/6 | 2.331–2.491s | 2.891–3.291s |
| 仅显式 chelsie | 0/6 | 2.331–2.971s | 2.971–3.691s |
| chelsie + 请求级残差 RNG | 6/6 | 2.331s（四次相同） | 3.051s（四次相同） |

36/36 请求完成、36/36 ASR 归一回读匹配；没有排除调用。每臂6对比较共享两个参考，不能当成6个独立文本样本。旧两臂的变化包含连续重复与穿插后的重复，不能单独据此估计“插入一个请求”的因果效应。结果支持隐藏残差随机流是此小样本波形变化的重要来源；不证明原用户会话的全部音色偏移都由它造成。

独立烟测 `isolation/receipt.json`：

- 同时提交2个受控请求+1个旧请求，再反转两个受控请求的提交顺序。四个受控结果与各自串行参考 **0/4 PCM相等**；变化未被隐藏或重跑筛选。此处证明服务接受并完成混合并发请求，不能从调用重叠直接证明每个 GPU step 的实际批次组成。
- seed43与seed42的音频不同，seed43两次串行结果逐样本相同。
- 收到首块 PCM 后关闭一个长句 SSE，服务日志确认 abort；随后新请求与串行参考逐样本相同。该检查验证取消后的恢复，不等于长期无内存泄漏证明。
- 非法 speaker、负seed、未知契约版本均400，随后正常合成与串行参考相同。
- 9次完整合成的原产物事后 CPU ASR 回读全部匹配；另有1次主动取消、3次预期400，不进入完整合成分母。

请求级 RNG 隔离单测执行**补丁后的真实 upstream runner 方法**，MTP 模型运算替换为 CPU 随机采样器：覆盖多步推进、批次重排、混入旧请求、外部全局随机调用、seed归属、eager执行与不修改全局RNG。真实GPU实验与该单测口径不同；并发音频仍有变化的原因没有通过本次实验细分到浮点组批、异步输入或其他路径。

相关132项pytest通过；Node播放/采样率/取消/ACK检查通过。真实Chromium + 虚拟麦克风/扬声器完成一个自造输入回合、三个TTS分句，声音配置正确、播放结束ACK成功、underrun=0、页面错误=0。不是物理设备或主观听感验收。新服务启动预热两条中英/控制字符 canary 首次回读匹配。

收据入口：`exp/web_demo/voice_rng_v1/serial/receipt.json`、`isolation/receipt.json`、`validation.json`。全部自造音频保留本机对应目录；归仓收据不包含真实用户数据或私有会话原始输出。浏览器原始收据和录音只在本机，归仓为聚合验证结果。
