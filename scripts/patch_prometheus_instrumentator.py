#!/usr/bin/env python3
"""Patch prometheus-fastapi-instrumentator for vllm-omni included routers.

FastAPI >= 0.137 stores lazy ``_IncludedRouter`` objects (no ``.path``) in
``app.routes``. Supported layouts of ``prometheus_fastapi_instrumentator.routing``:

* <= 8.0.0 (vllm / vllm-omni 0.22 env): ``_get_route_name`` reads
  ``route.path`` on every route, so every request 500s in the metrics
  middleware. Insert a guard that skips such routes (unchanged fd-badcat patch).
* >= 8.1.0 (vllm / vllm-omni 0.30 env): upstream flattens included routers via
  ``effective_route_contexts()`` and tolerates path-less routes, so there is
  nothing to patch. The script says so and exits 0 without touching the file.

Any other layout is refused with an error.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path


MARKER = "# fd-badcat: skip FastAPI included routers without .path"

# Snippets that together identify the upstream fix (8.1.0 routing.py).
UPSTREAM_FIX_SIGNATURES = (
    "def _effective_routes(",
    'getattr(route, "effective_route_contexts", None)',
    "for route in _effective_routes(routes):",
    'path = getattr(route, "path", None)',
)


def package_version(path: Path) -> str:
    init = path.with_name("__init__.py")
    try:
        found = re.search(
            r"""^__version__\s*=\s*["']([^"']+)["']""",
            init.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    except OSError:
        found = None
    return found.group(1) if found else "unknown"


def find_routing() -> Path | None:
    # find_spec() on the dotted submodule name would import (execute) the
    # parent package; look up the top-level package only and resolve
    # routing.py from its search locations, so no package code runs.
    spec = importlib.util.find_spec("prometheus_fastapi_instrumentator")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "routing.py"
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    path = find_routing()
    if path is None:
        raise SystemExit("prometheus_fastapi_instrumentator.routing not found")

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
        version = package_version(path)
        if all(sig in text for sig in UPSTREAM_FIX_SIGNATURES):
            print(
                "not needed: prometheus-fastapi-instrumentator "
                f"{version} already skips/flattens FastAPI included routers "
                f"upstream (_effective_routes); left unchanged: {path}"
            )
            return
        raise SystemExit(
            f"patch target not found in {path} "
            f"(prometheus-fastapi-instrumentator {version}): neither the "
            "<= 8.0.0 route walk nor the 8.1.0 upstream fix was recognised; "
            "refusing to patch an unknown layout"
        )

    patched = text.replace(old, new, 1)
    ast.parse(patched, filename=str(path))
    path.write_text(patched, encoding="utf-8")
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
