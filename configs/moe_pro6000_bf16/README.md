# RTX PRO 6000 BF16 MoE profile

Measured with vLLM 0.22.0, PyTorch 2.11.0+cu130 and NVIDIA driver 580.173.02.
These two JSON files target Qwen3-Omni Thinker (E128, H2048, I768, topk8)
and Talker (E128, H1024, I384, topk6). Weights, dtype and routing remain unchanged.
Use through `setup/start_demo_pro6000.sh`; the generic/frozen launcher stays unchanged.

Each file includes 26 token batch points. Candidate search uses seed42, then
three separate routing seeds (101/202/303); gap checks use 404/505/606.
Accept only >=1.03 median speedup and no slower validation seed. Other points
retain the matching vLLM default. SPLIT_K remains 1. The runtime selects the
nearest batch point using its standard config loader.

The serial tuner uses the unchanged core function from the official v0.22.0
benchmark, excluding Ray scheduling:
https://github.com/vllm-project/vllm/blob/v0.22.0/benchmarks/kernels/benchmark_moe.py

Retune on hardware/software/model changes. Do not use these measurements as
universal model accuracy, voice stability, multi-user throughput, or end-to-end
speedup evidence. Aggregates and the complete comparison are in
`docs/demo_pro6000_p0.md` and `exp/web_demo/pro6000_p0_20261007_summary.json`.

Private tuning source/receipts remain under `exp/web_demo/p0_pro6000_20261007/`.
The tuner accepts the downloaded upstream Python file with `--upstream` and
writes new output with `--output` and `--config-dir`; stop the owned demo first
and run on an otherwise idle GPU. No integrity hashes or PCM byte comparisons.
