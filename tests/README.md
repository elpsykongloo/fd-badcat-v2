# tests/

默认套件覆盖 ActorEngine/TACT 状态机、事务代数、屏障/DAG/窗口、流式语音协议、并发容量、RB 结构以及历史 HumDial 回放。不请求 LLM/TTS/ASR 网络服务，SenseVoice 只测试适配器契约；当本地 golden 音频存在时，会跑一条本地 VAD 确定性回放。

运行：

```bash
env -u OMP_NUM_THREADS /root/miniconda3/envs/fd-sds/bin/python -m pytest -q
```

测试约束：

- 并发/取消测试以事件、队列空闲或虚拟时钟推进，不用固定长时间 `sleep` 猜测完成时点。
- 性能测试必须视察项目代码；不保留只证明测试自己的 `sleep`/计时器更慢的用例。
- 完整性验证用语义断言、结构计数或必要时的直接内容比较；不做文件/源码/产物哈希检测。缓存键和确定性 ID 等功能性 hash 不属于检测。

真模型、GPU、实时延迟、真浏览器/声卡验收仍由 `scripts/` 下的专用 smoke/eval 工具运行，不塞进日常 pytest。
