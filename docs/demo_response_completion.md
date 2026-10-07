# Demo 回答完成性：response-completion-v2

本机制补齐聊天回答到达生成长度上限后被当作正常完成的问题。只作用于独立 demo 的普通回答；基础 HumDial、legacy、TACT/RB、路由、shift 和 TTS 的生成预算及既有决策保持原值。

## 完成契约

- `response_length_repair: true` 开启严格结束原因检查。SSE 必须同时给出有效结束原因和 `[DONE]`；只收到文字、缺失结束原因、异常关闭、未知结束原因均不能证明回答完成。
- `stop` 表示本次模型生成正常结束。`length` 表示达到上限，需要继续；不是回答已经完成的证明。
- demo 回答每次默认最多512 tokens，最多3次长度续写。原短承诺保护最多另加一次；所有调用共享同一轮输出、取消栅栏和长度续写计数。
- 长度续写使用 vLLM 原生 `continue_final_message=true`、`add_generation_prompt=false`，把已有完整生成文本放在最后一条 assistant 消息中，直接生成后缀。原始用户音频和 system prompt 保留，不经 ASR 改写；不新增“请继续”用户轮次、不插入拼接空格、不重新播放已有前缀。
- 所有分段进入同一个分句器和 SpeechPipeline。一次模型调用结束不会 flush 未完句；只有最终正常结束才发送整轮 `text_done`，播放完成仍由原 ACK 契约决定。

## 有界性、取消与失败

长度续写复用初始请求的冻结历史。为给新增 assistant 前缀留空间，按原 `history_token_budget` 的 UTF-8 字节上界计算，必要时从最旧开始移除完整历史轮对；当前音频、system 和生成前缀不裁切。前缀本身超预算则明确结束为未完成，并记录原因。该预算不是模型 tokenizer 的精确上下文长度；模型拒绝超长请求也进入明确失败路径。

达到续写次数上限、`length` 无新文字、接口/传输失败、缺少结束证明、上下文不足，以及一次承诺修复后仍未给出内容，均产生 `ResponseIncompleteError`。浏览器标注“回答未完成”，提示用户重新提出请求；旧播放按现有错误取消协议停止，不把半句作为成功回答收尾。错误通知独立于被取消音频的发送栅栏，但仍由浏览器核对回复 ID；未公开候选仅发布终止通知，不泄漏失败草稿，也不通过候选重启重置预算。已播放历史仍按原完整句 ACK 规则处理，未播部分不补入历史。

打断、续说、reset、断线都沿用候选与 SpeechPipeline 的取消传播，关闭在飞 HTTP 并释放 normal 请求槽。取消不触发自动重试。没有引入跨 task 引擎状态写入，也没有改变输入路由、hold 或声学规则。

## 诊断和重放

`/api/demo/info.response_completion` 宣布协议、单次预算、最大长度续写次数与承诺修复次数。manifest 记录对应开关和配置。

每个真实调用仍独立记录 call/parent/case；请求档案保存实际 `max_tokens` 和原生续写参数，`outcome.finish_reason` 与 `model_call_done.finish_reason` 保存结束原因。`response_completion_repair` 记录派发、每次调用结束、整轮完成、失败或取消；失败进入异常队列。单调用重放保留结束原因，会话重放按档案重现 length→续写，不把旧无结束证明档案伪造成 v2 证据。

## 验证入口与边界

`tests/test_response_completion.py` 覆盖真实 HTTP SSE 终止证明、多次长度截断、半句跨调用的逐句 PCM 管线、上下文淘汰、预算耗尽、接口故障、取消释放、真实请求档案和无模型重放。旧路径相关测试另作回归。

```bash
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python scripts/check_response_completion.py --output NEW_DIR
# 安装有 Playwright 的环境；音频使用上一命令生成的自造 zh-input.wav。
python scripts/check_response_completion_browser.py --output NEW_BROWSER_DIR --audio NEW_DIR/zh-input.wav --chromium EXISTING_CHROMIUM
```

前者串行使用真实 Omni、合成用户音频，分别检查中英小预算强制截断、生产预算和故意耗尽；后者在独立回环 backend 上用真实 Chromium、VAD、模型、逐句 TTS 和播放 ACK 检查成功完成及可见的失败状态。隔离 backend 不清理历史会话，也不改变生产服务配置。

2026-10-07验收：259项定向测试与Node播放/输入测试通过。两个中英自造音频复用到6个串行条件，共9次真实回答请求；小预算分别经过2次/1次长度续写后正常stop，生产预算均正常stop，故意耗尽均明确失败。独立浏览器最终2个会话验证长度续写完整播完（0断流）及预算耗尽的可见提示；初次故障测试发现取消栅栏吞掉错误通知，保留失败记录，修复后重验通过。最终生产链另通过长答叫停、STOP_ONLY静默及下一轮恢复，trace无丢失、无引擎/页面错误。

验收收据入口：`exp/web_demo/response_completion_v2/validation.json`。同一输入跨条件复用，不当作独立新音频样本；上述数字是自造小样本的机制验证，不是自然会话准确率、物理设备听检或正式延迟成绩。`stop` 是传输/模型结束证明，不保证模型在语义上满足所有请求；有限上下文和预算也不能支持任意长度的无限续写。
