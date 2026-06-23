# TACT FDB-v3 Integration

This repository contains the TACT modules from `files.zip`, configured to run
beside the existing local repositories:

- `/root/autodl-tmp/fd-badcat`
- `/root/autodl-tmp/FDBench_v3`

`fd-badcat` is treated as read-only by this setup. TACT discovers its model
endpoint through environment variables and falls back to the same OpenAI-compatible
HTTP contract used by `fd-badcat/src/module.py`.

## Existing Environments

Use the already-created environments:

```bash
conda activate /root/miniconda3/envs/fd-sds        # fd-badcat-compatible runtime
conda activate /root/autodl-tmp/conda-envs/fdb_v3 # FDB-v3 evaluation runtime
```

Install TACT into the fd-badcat runtime without pulling new dependencies:

```bash
/root/miniconda3/envs/fd-sds/bin/pip install -e /root/autodl-tmp/tact --no-deps
```

## Dry Run

The dry run verifies package imports, audio loading, result schema, and FDB-v3
tool/evaluator compatibility without calling Qwen3-Omni:

```bash
/root/miniconda3/envs/fd-sds/bin/python -m tact.offline_runner \
  --data /root/autodl-tmp/FDBench_v3/v3/fdb_v3_data_released \
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
