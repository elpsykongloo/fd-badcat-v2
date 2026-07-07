#!/bin/bash
# Quick service health check for W2

echo "=== Service Health Check ==="

# Check vLLM
if curl -s http://localhost:10003/v1/models > /dev/null 2>&1; then
    echo "✓ vLLM (port 10003): Running"
else
    echo "✗ vLLM (port 10003): Not responding"
    exit 1
fi

# Check Qwen3 API proxy
if curl -s http://localhost:10004/ > /dev/null 2>&1; then
    echo "✓ Qwen3 API Proxy (port 10004): Running"
else
    echo "✗ Qwen3 API Proxy (port 10004): Not responding"
    exit 1
fi

# Check GPU
if nvidia-smi > /dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    echo "✓ GPU: $GPU_NAME ($GPU_MEM MiB)"
else
    echo "✗ GPU: Not available"
    exit 1
fi

echo "=== All Services Ready ==="
