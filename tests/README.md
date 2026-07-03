# tests/

W1 引擎回归（test_regression / test_deterministic_replay / test_asr_en / test_concurrent_*）
+ W2 Phase-B 骨架单测（test_transaction / test_phase_b* / test_state_track / test_ack_integration / test_performance）。

跑法：`/root/miniconda3/envs/fd-sds/bin/python -m pytest tests/ -q`

注意：W2 评测的权威路径是 `scripts/w2r_stream_replay.py`（真 LLM 流式回放）+
`scripts/w2r_score_grid.py`（exact/state/latency 三轨判分）；`src/engine_b*.py` 是
Phase-B 引擎骨架（单测覆盖，尚未接入评测关键路径）。
