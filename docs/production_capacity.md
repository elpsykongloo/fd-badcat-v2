# 单用户生产容量配置

当前默认音频服务面向 RTX PRO 6000 Blackwell 96GB 的单用户连续语音交互：三阶段均为 `max_num_seqs=4`，stage 0 的 `max_model_len=4096`，服务调度显式固定为 `fcfs`。这不是新的模型优先级方案；紧急通路由应用侧准入控制保证。

## 请求准入

`src/request_capacity.py` 在 backend 进程内维护一个所有 ActorEngine/TactEngine 会话共享的双层 FIFO 闸门：

- 总请求上限为 4。
- 普通请求上限为 3，包括 response、shift、shift_re、TACT decider，以及整句/分句/流式/ack TTS。
- judge 和 interrupt 属于 control 请求，只受总上限 4 约束。因此三个普通请求占满时，仍有一个槽可立即送往 Omni。
- ASR 和工具执行不占 Omni 槽。

它只在请求入场前保留容量，不会抢占已经运行的 GPU kernel。不同类别之间有意存在保留关系，各类别内部 FIFO；请求进入 vLLM 后仍由 FCFS 调度。两个限制必须同时存在：若只有普通上限 3 而没有总上限，多个 control 请求仍可能越过服务的四槽容量；若先让第 4 个普通请求占总票再等待普通票，反而会吃掉保留槽。

模型调用超过 ActorEngine 的决策超时时，底层同步 HTTP 工作可能尚未结束。实现会让该工作继续持有容量票，直到真实线程退出，避免“表面超时、实际仍占 GPU，却提前放票”造成隐性超卖。取消中的 SSE 流则会关闭上游连接并在退出时放票。

这是 backend **进程内**的全局限制，不约束绕过 backend 直接访问 `:10003`/`:10004` 的外部脚本。当前只承诺单用户；多租户公平、每用户配额和跨进程协调仍不在本轮范围。

## 历史预算

ActorEngine 默认 `history_token_budget: 3000`。组装 prompt 时从最新轮开始，只保留能完整装入预算的 user/assistant 对，再恢复时间顺序；不会截断半轮，也不会为了填满预算跳过过大的最新轮。历史未被请求的原路径仍完全不带历史，预算以内的消息角色与文本结构和原实现相同。

计数采用 `UTF-8 字节数 + 每条消息 16 token 模板余量`。Qwen 的 byte-BPE 内容 token 数不会高于 UTF-8 字节数，因此这是无需在实时 backend 额外加载 tokenizer 的保守上界，不是宣称与 tokenizer 精确计数相等。预算只限制送入模型的完整文本历史，不删除内存中的审计历史，也不限制当前异常超长音频。TactEngine 的累积音频前缀不走这份文本历史；若以后支持长时 Phase-B 会话，应另做音频前缀滚动/摘要策略。

4096 是 stage 0 的整次模型上下文，不是纯历史额度。3000 的保守历史上界为系统提示、当前音频表示、模板和最多 256 个输出 token 留出空间；病态超长的当前音频仍应由后续输入时长护栏处理。

## 历史测量复现

生产入口：

```bash
bash setup/start_qwen3omni_audio.sh
```

旧的 `max_num_seqs=1 / stage0=8192` 串行测量口径已逐字节另存为 `configs/qwen3_omni_audio_serial_eval.yaml`：

```bash
QWEN_DEPLOY_CONFIG=configs/qwen3_omni_audio_serial_eval.yaml \
  bash setup/start_qwen3omni_audio.sh
```

旧 RB/FDB 延迟数字仍按各自冻结配置解释，不因生产默认改变而重标。新的生产吞吐或延迟数字也不得直接和 `seq=1` 历史数相减。

## 2026-09-07 真机收据摘要

硬件为 RTX PRO 6000 Blackwell Server Edition 97,887 MiB；vLLM/vLLM-Omni 均为 0.22.0。默认脚本无覆盖启动成功：

- `/v1/models` 报告 `max_model_len=4096`；stage 0 日志报告 KV cache 87,920 tokens、4096 token 理论并发 21.46×。
- 进程命令行明确包含 `--scheduling-policy fcfs`；三阶段均成功初始化，空闲实测 GPU used/free 约 95,851/1,400 MiB。
- 两轮真实“4 个普通 TTS + 1 个 judge”闸门探针峰值均为 total 4 / normal 3。第 4 个普通请求分别在 2.9212 s、2.3764 s 才入场，恰在首个普通请求结束之后；judge 在 0.0074 s、0.0120 s 入场并于 1.6964 s、1.1539 s 完成，返回 `OK` / `switch`，早于首批 TTS 完成。两轮总墙钟 4.2947 s / 3.5143 s，10/10 HTTP 请求成功，无 OOM 或引擎崩溃。
- 全仓 Python 测试 187/187 通过；其中新增容量/历史契约 7 项。4 条 pytest warning 是既有测试返回 bool 的警告。
- 服务已正常停止，`:10003/:10004/:18000` 与 GPU 占用均为 0。vLLM-Omni 停服时报告各 2 个 semaphore/shared-memory `resource_tracker` 清理警告；没有遗留进程或端口。

这些是单机烟测，不是正式 P50/P95，也不证明高并发、多用户公平或物理扬声器体验。完整结构化数据见 `exp/streaming_demo/capacity_seq4_receipt.json`。
