# TACT FDB-v3 Integration

This directory contains the original TACT modules from `files.zip`. The package
was moved into the fd-badcat repository on 2026-07-14 so the transaction
primitives and the later W2-W4 engine work share one source-control boundary:

- package: `/root/autodl-tmp/fd-badcat/tact`
- application/engine: `/root/autodl-tmp/fd-badcat/src`
- benchmark checkout: `/root/autodl-tmp/FDBench_v3`

TACT discovers its model endpoint through environment variables and falls back
to the same OpenAI-compatible HTTP contract used by `fd-badcat/src/module.py`.

## Existing Environments

Use the already-created environments:

```bash
conda activate /root/miniconda3/envs/fd-sds        # fd-badcat-compatible runtime
conda activate /root/autodl-tmp/conda-envs/fdb_v3 # FDB-v3 evaluation runtime
```

Install the same in-repository package into both runtimes without pulling new
dependencies:

```bash
/root/miniconda3/envs/fd-sds/bin/pip install -e \
  /root/autodl-tmp/fd-badcat/tact --no-deps
/root/autodl-tmp/conda-envs/fdb_v3/bin/pip install -e \
  /root/autodl-tmp/fd-badcat/tact --no-deps
```

## Dry Run

The dry run verifies package imports, audio loading, result schema, and FDB-v3
tool/evaluator compatibility without calling Qwen3-Omni. It still writes a
`result_<provider>.json`, so use a disposable data copy for migration smoke
tests rather than the released benchmark directory:

```bash
/root/miniconda3/envs/fd-sds/bin/python -m tact.offline_runner \
  --data /tmp/tact-dryrun-data \
  --provider tact_dryrun \
  --dry-run \
  --limit 1 \
  --force
```

## Real Offline Run

Start the local Qwen3-Omni vLLM server and proxy as described in
`INTEGRATION.md`, then run:

```bash
MODE=blocking PROVIDER=tact_blocking scripts/run_offline_tact.sh --force
MODE=async PROVIDER=tact_async scripts/run_offline_tact.sh --force
```

Score results from `FDBench_v3/v3`:

```bash
cd /root/autodl-tmp/FDBench_v3/v3
PROVIDERS="tact_blocking tact_async" bash run_all_evaluations_released.sh
```

Useful path overrides:

- `FDBC_REPO_ROOT=/path/to/fd-badcat`
- `FDBC_SRC_DIR=/path/to/fd-badcat/src`
- `FDBENCH_REPO_ROOT=/path/to/FDBench_v3`
- `FDB_V3_DIR=/path/to/FDBench_v3/v3`
- `FDB_V3_DATA_DIR=/path/to/fdb_v3_data_released`
