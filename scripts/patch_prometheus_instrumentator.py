#!/usr/bin/env python3
"""Patch prometheus-fastapi-instrumentator for vllm-omni included routers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER = "# fd-badcat: skip FastAPI included routers without .path"


def main() -> None:
    spec = importlib.util.find_spec("prometheus_fastapi_instrumentator.routing")
    if spec is None or spec.origin is None:
        raise SystemExit("prometheus_fastapi_instrumentator.routing not found")

    path = Path(spec.origin)
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"already patched: {path}")
        return

    old = """    for route in routes:
        match, child_scope = route.matches(scope)
"""
    new = f"""    for route in routes:
        {MARKER}
        if not hasattr(route, "path") or not hasattr(route, "matches"):
            continue
        match, child_scope = route.matches(scope)
"""
    if old not in text:
        raise SystemExit(f"patch target not found in {path}")

    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
