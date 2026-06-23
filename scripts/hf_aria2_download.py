#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


def fetch_repo(repo_id: str, repo_type: str, revision: str, token: str | None):
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if repo_type == "dataset":
        api_url = f"{endpoint}/api/datasets/{repo_id}?revision={quote(revision)}&blobs=true"
    else:
        api_url = f"{endpoint}/api/models/{repo_id}?revision={quote(revision)}&blobs=true"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(api_url, headers=headers)
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_files(payload: dict, include: list[str], exclude: list[str]):
    siblings = payload.get("siblings") or []
    for item in siblings:
        filename = item.get("rfilename")
        if not filename or filename.endswith("/"):
            continue
        if include and not any(Path(filename).match(pattern) for pattern in include):
            continue
        if exclude and any(Path(filename).match(pattern) for pattern in exclude):
            continue
        yield filename


def write_aria2_input(
    repo_id: str,
    repo_type: str,
    revision: str,
    local_dir: Path,
    input_file: Path,
    files: list[str],
    token: str | None,
):
    repo_kind = "datasets" if repo_type == "dataset" else ""
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base = f"{endpoint}/{repo_kind + '/' if repo_kind else ''}{repo_id}/resolve/{quote(revision, safe='')}"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    with input_file.open("w", encoding="utf-8") as handle:
        for filename in files:
            remote = f"{base}/{quote(filename, safe='/')}"
            parent = filename.rsplit("/", 1)[0] if "/" in filename else ""
            out_name = filename.rsplit("/", 1)[-1]
            target_dir = local_dir / parent
            target_dir.mkdir(parents=True, exist_ok=True)

            handle.write(remote + "\n")
            handle.write(f"  dir={target_dir}\n")
            handle.write(f"  out={out_name}\n")
            if token:
                handle.write(f"  header=Authorization: Bearer {token}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate aria2 input for a Hugging Face repo.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--input-file", required=True, type=Path)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()

    token = os.getenv(args.token_env) or os.getenv("HUGGING_FACE_HUB_TOKEN")
    payload = fetch_repo(args.repo_id, args.repo_type, args.revision, token)
    files = sorted(iter_files(payload, args.include, args.exclude))
    if not files:
        raise SystemExit(f"No files matched for {args.repo_type} {args.repo_id}")

    write_aria2_input(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=args.local_dir,
        input_file=args.input_file,
        files=files,
        token=token,
    )
    print(f"Wrote {len(files)} files to {args.input_file}")


if __name__ == "__main__":
    main()
