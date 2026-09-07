#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
from contextlib import asynccontextmanager
import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

# =========================
# 配置
# =========================
VLLM_URL = os.getenv("FDBC_VLLM_URL", "http://127.0.0.1:10003/v1/chat/completions")
QWEN_MODEL = os.getenv("FDBC_QWEN_MODEL", "Qwen3-Omni-30B-A3B-Instruct")
REQUEST_TIMEOUT = int(os.getenv("FDBC_PROXY_TIMEOUT", "300"))

# =========================
# FastAPI 应用
# =========================
@asynccontextmanager
async def lifespan(app):
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(trust_env=False, timeout=timeout) as client:
        app.state.http = client
        yield


app = FastAPI(title="vLLM streaming proxy", lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    try:
        payload = await request.json()
        payload.setdefault("model", QWEN_MODEL)
        upstream = await request.app.state.http.post(VLLM_URL, json=payload)
        content_type = upstream.headers.get("Content-Type", "application/json")
        if upstream.status >= 400 or not payload.get("stream"):
            try:
                return Response(await upstream.read(), status_code=upstream.status,
                                headers={"Content-Type": content_type})
            finally:
                upstream.close()

        async def forward():
            try:
                async for chunk in upstream.content.iter_any():
                    yield chunk
            finally:
                upstream.close()

        return StreamingResponse(forward(), status_code=upstream.status,
                                 headers={"Content-Type": content_type,
                                          "Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return JSONResponse({"error": "vLLM upstream unavailable or timed out"}, status_code=502)
    except (ValueError, AttributeError):
        return JSONResponse({"error": "Invalid JSON request"}, status_code=400)

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("FDBC_PROXY_HOST", "0.0.0.0"),
                port=int(os.getenv("FDBC_PROXY_PORT", "10004")))
