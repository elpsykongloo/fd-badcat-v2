#!/usr/bin/env python3
"""Start the laptop demo stack on loopback; stop only processes we started.

No daemon, no global pkill, no model/config changes, no port reuse by accident.
Use --backend-only explicitly when the two inference services already exist.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parents[1]
HTTP = build_opener(ProxyHandler({}))


def get_json(url):
    with HTTP.open(url, timeout=2) as response:
        return json.load(response)


def require_free(port):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"端口 {port} 已被占用。不会停止或接管已有服务。"
                "已有 Omni/代理时可用 --backend-only；页面端口可用 --port 修改。") from exc


def wait_ready(url, process, logfile, timeout, check=None):
    deadline = time.monotonic() + timeout
    next_status = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"服务提前退出（exit={process.returncode}），日志：{logfile}")
        try:
            data = get_json(url)
            if check is None or check(data):
                return
        except (URLError, OSError, ValueError):
            pass
        if time.monotonic() >= next_status:
            print(f"仍在等待就绪：{url}；日志 {logfile}", flush=True)
            next_status = time.monotonic() + 20
        time.sleep(.5)
    raise RuntimeError(f"服务启动超时（{timeout}s），日志：{logfile}")


def stop_owned(children):
    # Each child owns its own process group. Descendants may outlive the leader.
    for name, process, _ in reversed(children):
        print(f"停止本启动器创建的 {name}（进程组 {process.pid}）", flush=True)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(p.poll() is None for _, p, _ in children):
        time.sleep(.2)
    for _, process, handle in reversed(children):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-only", action="store_true", help="Reuse existing local Omni :10003 and proxy :10004")
    parser.add_argument("--port", type=int, default=18000, help="Demo/backend loopback port (default: 18000)")
    parser.add_argument("--startup-timeout", type=int, default=900, help="Model startup timeout, seconds")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.port in (10003, 10004):
        parser.error("--port must be 1024..65535, excluding inference ports 10003/10004")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    children = []
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        require_free(args.port)
        if args.backend_only:
            # Non-mutating checks. The proxy OpenAPI check doesn't invoke a model.
            if not get_json("http://127.0.0.1:10003/v1/models").get("data"):
                raise RuntimeError("Omni 尚未就绪。")
            get_json("http://127.0.0.1:10004/openapi.json")
        else:
            require_free(10003)
            require_free(10004)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        logdir = ROOT / "exp" / "web_demo" / (stamp + f"-{os.getpid()}")
        logdir.mkdir(parents=True, exist_ok=False)
        print(f"启动 HumDial 笔记本演示。日志：{logdir}", flush=True)
        env = dict(os.environ, QWEN_HOST="127.0.0.1", QWEN_PORT="10003",
                   FDBC_PROXY_HOST="127.0.0.1", FDBC_PROXY_PORT="10004",
                   FDBC_VLLM_URL="http://127.0.0.1:10003/v1/chat/completions",
                   PYTHONUNBUFFERED="1")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            if env.get(key, "") in ("", "0"):
                env[key] = "8"
        for key in ("NO_PROXY", "no_proxy"):
            env[key] = ",".join(filter(None, [env.get(key), "127.0.0.1", "localhost"]))
        def start(name, command, url, timeout, check=None):
            logfile = logdir / (name + ".log")
            handle = logfile.open("w", encoding="utf-8")
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                handle.close()
                raise
            children.append((name, process, handle))
            print(f"启动 {name} → {logfile}", flush=True)
            wait_ready(url, process, logfile, timeout, check)
            print(f"{name} 已就绪", flush=True)
        if not args.backend_only:
            start("omni", ["bash", "setup/start_qwen3omni_audio.sh"],
                  "http://127.0.0.1:10003/v1/models", args.startup_timeout, lambda d: bool(d.get("data")))
            start("proxy", ["bash", "setup/start_qwen3_proxy.sh"],
                  "http://127.0.0.1:10004/openapi.json", 60)
        start("backend", [sys.executable, "src/backend.py", "--streaming", "--host", "127.0.0.1", "--port", str(args.port)],
              f"http://127.0.0.1:{args.port}/api/demo/info", 120, lambda d: d.get("streaming") is True)
        print(f"\nDEMO READY → http://localhost:{args.port}/demo/\n"
              "现在在笔记本建立 SSH 端口转发，详见 docs/web_demo.md。\n"
              "仅监听服务器 127.0.0.1；不需要开放公网推理端口。\n"
              "Ctrl+C 关闭本启动器创建的服务；--backend-only 不会停止已有 Omni/代理。", flush=True)
        while True:
            for name, process, _ in children:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} 已退出（exit={process.returncode}），停止本次演示。日志：{logdir}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在关闭演示…", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        # A second Ctrl+C must not strand inference children on the GPU.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop_owned(children)


if __name__ == "__main__":
    raise SystemExit(main())
