# zkDL GAT baseline-local adapter

This directory is self-contained.  It uses the restored CUDA/C++ zkDL core
and the GAT adapter under `gat_adapter/`; it does not import code, assets, or
runtime state from another baseline or from the main experiment.

## Build and inspect

```bash
cd /home/pzh/project/comparison/zkdl
make baseline-local-build
./gat_runner inspect --config configs/cora_secure.cfg
```

The short fixture checks live under `tests/` only.  They are not inputs to a
formal run.

## Formal preparation and runs

Preparation computes a content-addressed `asset_manifest.json` beside the
run and then writes the native relation plan:

```bash
scripts/prepare_local.sh configs/cora_secure.cfg
```

Full-dataset proving is intentionally authorization-gated.  After reviewing
the copied assets, SRS, and source identity, run one backend with:

```bash
ZKDL_ALLOW_FULL_DATASET_RUN=1 \
  scripts/run_single.sh configs/cora_secure.cfg cpu runs/cora_cpu

ZKDL_ALLOW_FULL_DATASET_RUN=1 \
  scripts/run_single.sh configs/cora_secure.cfg gpu runs/cora_gpu
```

The five-dataset runner uses the same explicit gate:

```bash
ZKDL_ALLOW_FULL_DATASET_RUN=1 scripts/run_five.sh cpu
ZKDL_ALLOW_FULL_DATASET_RUN=1 scripts/run_five.sh gpu
```

Verification reads the public statement, public output/proof artifacts, and
the configured verification boundary:

```bash
scripts/verify_single.sh configs/cora_secure.cfg runs/cora_cpu
```

Results are collected and then validated without inferring proof success from
timings:

```bash
python3 scripts/collect_results.py --runs runs --output results/collected.json
python3 scripts/final_validate.py --runs runs --require-pairs
```

`setup_register.sh` refuses to generate or register a replacement when the
configured authenticated SRS is absent.  It is therefore expected to exit
non-zero in the current restored tree; this is a real readiness blocker, not
a successful setup.

## Current readiness boundary

The native bridge is fail-closed until the official core supplies the complete
GAT operation-family verifier and authenticated SRS path.  A `BLOCKED` record
is not a proof and must not be renamed to `VERIFIED`.
