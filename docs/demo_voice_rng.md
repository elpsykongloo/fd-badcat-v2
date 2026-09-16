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

## 自动说话人嵌入评测（demo-speaker-embedding-v1）

采用作者发布的 **VoxCeleb2 + VoxBlink2、完整未剪枝、最终 LMFT** 权重：
[`zl389/w2v-bert-2.0_SV/model_lmft_0.14.pth`](https://huggingface.co/zl389/w2v-bert-2.0_SV/blob/main/model_lmft_0.14.pth)。依据为[论文](https://arxiv.org/abs/2510.04213)、[作者模型表](https://github.com/ZXHY-82/w2v-BERT-2.0_SV#model-download)和 [WeSpeaker v2 说明](https://github.com/wenet-e2e/wespeaker/blob/master/examples/voxceleb/v2/README.md)。

| 候选/评分条件 | Vox1-O / E / H EER（%），上游报告 |
| --- | --- |
| WeSpeaker 自训，VoxCeleb only，LMFT，均值归一 cosine | 0.250 / 0.398 / 0.838 |
| 作者最终 LMFT，VoxCeleb2 + VoxBlink2 | 0.14 / 0.31 / 0.73 |
| WeSpeaker 加载同一作者 checkpoint，AS-Norm + QMF | 0.138 / 0.285 / 0.625 |

最后两行使用同一作者 checkpoint，不能解释为两个嵌入模型谁更强；训练数据、评分后处理也使第一行不能成为纯架构对照。选择作者较充分训练的最终权重，并采用与 WeSpeaker 对齐的推理结构。没有在本机复现上述 EER，也不宣称在所有中文短句场景全球最优。

LoRA 已在训练过程中合并；Layer Adapter、MFA/ASP、256 维投影包含在最终权重中。LMFT 是训练阶段，不是推理时再串一个模型。加载全部 586,652,736 个推理参数，`strict=True`，禁止未匹配权重静默随机初始化。训练分类头不参与嵌入提取。采用现有 Transformers backbone 和一个轻量的 Adapter/MFA/ASP 推理适配；没有引入作者整套训练环境或额外线上服务。

脚本 `scripts/check_demo_speaker_embeddings.py` 接受显式 JSON 比较清单，逐条提取整句嵌入、L2 归一化，再算 cosine。固定模型、FP32、batch=1；mono 音频按原采样率重采样到 16kHz；不做 VAD、去静音、补齐或最佳片段挑选。非法/静音/不足0.5s/超过60s输入报错，不静默排除。每轮末尾重新提取第一条，检查评测器自身是否受中间请求影响。无网络调用、无完整性哈希、无人工评分依赖。

**此任务先不加 AS-Norm/QMF。** 它们作用于配对分数，需要额外 cohort/校准数据，不能把 VoxCeleb EER 或阈值直接转成短句 TTS 的音色稳定门槛。当前只报告原始相似度和距离 `1-cosine`，没有伪造“同一人概率”或通过阈值。embedding 关注身份一致性，不能代替已有 F0/时长指标；自然的音调变化不应该被全部压平。

复现（模型目录由 `.gitignore` 排除；不需要另下载基础 backbone 权重）：

```bash
# 使用 Omni 环境现有 hf/torch/transformers/torchaudio；未修改共享环境依赖。
env -u ALL_PROXY -u all_proxy HF_HUB_DISABLE_XET=1 \
  /root/autodl-tmp/conda-envs/fdbc-qwen3o-vllm/bin/hf download \
  zl389/w2v-bert-2.0_SV model_lmft_0.14.pth config/v1/s3.yaml \
  --local-dir model/speaker-w2vbert-lmft
env -u ALL_PROXY -u all_proxy HF_HUB_DISABLE_XET=1 \
  /root/autodl-tmp/conda-envs/fdbc-qwen3o-vllm/bin/hf download \
  facebook/w2v-bert-2.0 config.json preprocessor_config.json \
  --local-dir model/speaker-w2vbert-lmft
env -u OMP_NUM_THREADS \
  /root/autodl-tmp/conda-envs/fdbc-qwen3o-vllm/bin/python \
  scripts/check_demo_speaker_embeddings.py \
  --manifest exp/web_demo/voice_rng_v1/speaker/manifest.json \
  --output exp/web_demo/voice_rng_v1/speaker/receipt.NEW.json
```

Manifest 包含 `audio: [{id,path,text?}]` 和 `pairs: [{group,reference,candidate}]`，路径相对 manifest；`design` 记录样本与实验口径。新候选使用新清单与新收据，同一个固定参考/文本配对规则，不以高分音频重新选择参考。权重与原始 WAV 保留本机。上游[作者代码许可](https://github.com/ZXHY-82/w2v-BERT-2.0_SV/blob/master/License.md)声明 CC BY-NC-SA 4.0；WeSpeaker 代码的 Apache-2.0 不构成该 checkpoint 的商业授权。本改动是离线研究评测工具，不分发权重。

### 2026-09-16 实测

对已有36次串行合成和9次独立烟测完整合成逐一提取，**45/45，无排除**；提取串行、GPU无其他服务。沿用原产物而非重新抽样；源服务仍是 seq4，不改变原实验口径。共4个自造文本，其中比较涉及2个独立中文目标文本；无真实人类录音、无物理设备。39对比较预先写入 manifest，存在共享参考与重复文本，不能当作39个独立身份试验；没有异说话人负样本，不输出身份识别准确率。

| 比较 | 对数 | cosine 均值 | 范围 |
| --- | --- | --- | --- |
| 原请求，同句重复 | 6 | 0.838305 | 0.808458–0.866347 |
| 仅固定 chelsie，同句重复 | 6 | 0.830146 | 0.791443–0.853776 |
| chelsie + 请求 RNG，同句重复 | 6 | 1.000000 | 1.000000–1.000000 |
| 受控并发 vs 串行同句参考 | 4 | 0.910488 | 0.891190–0.925637 |
| seed42 vs seed43，同句 | 2 | 0.831908 | 两次 seed43 互相重复 |
| seed43 连续重复 | 1 | 1.000000 | — |
| 取消/非法请求后的恢复 | 2 | 1.000000 | — |
| 原请求，跨两个中文句子 | 4 | 0.774349 | 0.729241–0.809461 |
| 仅固定 chelsie，跨句 | 4 | 0.806861 | 0.781108–0.834344 |
| chelsie + 请求 RNG，跨句 | 4 | 0.827397 | 同一音频对的4次重复 |

同句受控串行 cosine=1 与 PCM 完全一致相符；单纯固定 speaker 不消除本样本的嵌入波动。并发下不仅 PCM 不同，嵌入也发生变化；但这些分数不足以判定身份切换或主观不自然。跨句受控均值较高仅为该文本对的描述，**不能据两句、四次重复宣称跨文本音色问题已解决**。

后续已扩大到六个目标文本，并定位、修复该组样本中的 Talker 并发数值漂移，详见 [并发声音诊断与 native 修复](demo_voice_concurrency.md)。demo 启动器默认 `FDBC_DEMO_TALKER_NUMERICS=native`，普通部署默认关闭。仍保留 ASR、F0、时长约束；跨文本身份一致性与自然度不能由这组未校准 cosine 自动宣告解决，不据此修改线上决策或引入重训练。

实现核验：全部推理权重严格加载；本地适配与原 WeSpeaker `W2VBert_Adapter_MFA`/`ASP`（代码版本 `dfa7419`）及 Transformers 标准 backbone forward，在3段自造音频上 FP32 嵌入最大绝对差均为0；评测器穿插45条后的重复提取最大绝对差为0。13项输入/计分/重采样边界测试通过。这不是完整 VoxCeleb benchmark 复现。

收据：`exp/web_demo/voice_rng_v1/speaker/{manifest,receipt,implementation_validation}.json`。
