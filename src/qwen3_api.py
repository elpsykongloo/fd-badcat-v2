#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
from fastapi import FastAPI, Request
import uvicorn

# =========================
# 配置
# =========================
VLLM_URL = os.getenv("FDBC_VLLM_URL", "http://127.0.0.1:10003/v1/chat/completions")
QWEN_MODEL = os.getenv("FDBC_QWEN_MODEL", "Qwen3-Omni-30B-A3B-Instruct")
REQUEST_TIMEOUT = int(os.getenv("FDBC_PROXY_TIMEOUT", "300"))
LOCAL_HTTP = requests.Session()
LOCAL_HTTP.trust_env = False

# =========================
# FastAPI 应用
# =========================
app = FastAPI(title="Simple vLLM Direct Proxy")

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    try:
        payload = await request.json()
        payload.setdefault("model", QWEN_MODEL)
        response = LOCAL_HTTP.post(VLLM_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"请求 vLLM 服务失败: {e}"}
    except Exception as e:
        return {"error": f"处理请求时发生未知错误: {e}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10004)
