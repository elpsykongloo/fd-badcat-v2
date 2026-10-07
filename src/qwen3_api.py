#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import os
import json
import logging
import time
from uuid import uuid4
from contextlib import asynccontextmanager
import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from transport_diagnostics import failure_details, safe_endpoint, safe_request_id

LOG = logging.getLogger("uvicorn.error")

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
    # Compatibility mode only. Do not retain upstream keep-alive sockets across
    # unrelated SSE requests; a cancelled stream cannot poison a later request.
    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(trust_env=False, timeout=timeout, connector=connector) as client:
        app.state.http = client
        yield


app = FastAPI(title="vLLM streaming proxy", lifespan=lifespan)

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    started = time.perf_counter()
    rid = safe_request_id(request.headers.get("x-request-id")) or "proxy-" + uuid4().hex
    state = {"request_id": rid, "endpoint": safe_endpoint(VLLM_URL), "phase": "request",
             "fresh_connection": True, "received_bytes": 0}
    upstream = None
    streaming = False
    def record(status, exc=None):
        data = {"event": "proxy_request_done", **state, "status": status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
        if exc is not None:
            data = failure_details(exc, data)
        LOG.info(json.dumps(data, ensure_ascii=False))
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Request must be an object")
        payload.setdefault("model", QWEN_MODEL)
        state["phase"] = "connect"
        upstream = await request.app.state.http.post(VLLM_URL, json=payload,
                                                     headers={"X-Request-Id": rid})
        state.update(phase="headers", http_status=upstream.status,
            upstream_request_id=safe_request_id(upstream.headers.get("x-request-id")),
            headers_ms=round((time.perf_counter() - started) * 1000, 3))
        content_type = upstream.headers.get("Content-Type", "application/json")
        headers = {"Content-Type": content_type, "X-Request-Id": rid}
        if upstream.status >= 400 or not payload.get("stream"):
            try:
                data = await upstream.read()
                state["received_bytes"] = len(data)
                record("upstream_http_error" if upstream.status >= 400 else "completed")
                return Response(data, status_code=upstream.status, headers=headers)
            finally:
                upstream.close()

        async def forward():
            status, error = "cancelled", None
            try:
                state["phase"] = "stream"
                async for chunk in upstream.content.iter_any():
                    state["received_bytes"] += len(chunk)
                    yield chunk
                status = "completed"
            except Exception as exc:
                status, error = "stream_error", exc
                raise
            finally:
                upstream.close()
                record(status, error)

        streaming = True
        return StreamingResponse(forward(), status_code=upstream.status,
                                 headers={**headers,
                                          "Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})
    except asyncio.CancelledError:
        record("cancelled")
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        record("connection_error", exc)
        return JSONResponse({"error": "vLLM upstream unavailable or timed out",
                             "code": "upstream_connection_error", "request_id": rid},
                            status_code=502, headers={"X-Request-Id": rid})
    except (ValueError, AttributeError):
        record("invalid_request")
        return JSONResponse({"error": "Invalid JSON request"}, status_code=400,
                            headers={"X-Request-Id": rid})
    finally:
        if upstream is not None and not streaming:
            upstream.close()

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("FDBC_PROXY_HOST", "0.0.0.0"),
                port=int(os.getenv("FDBC_PROXY_PORT", "10004")))
