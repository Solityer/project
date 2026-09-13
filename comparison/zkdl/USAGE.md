# zkDL GAT command reference

All commands below run from `/home/pzh/project/comparison/zkdl` and use only
baseline-local paths.

| Purpose | Command |
| --- | --- |
| Build adapter/core link | `make baseline-local-build` |
| Inspect local config | `./gat_runner inspect --config configs/cora_secure.cfg` |
| Hash copied assets and prepare relation | `scripts/prepare_local.sh configs/cora_secure.cfg` |
| Setup/register trusted SRS | `scripts/setup_register.sh configs/cora_secure.cfg` |
| One CPU formal run | `ZKDL_ALLOW_FULL_DATASET_RUN=1 scripts/run_single.sh configs/cora_secure.cfg cpu runs/cora_cpu` |
| One GPU formal run | `ZKDL_ALLOW_FULL_DATASET_RUN=1 scripts/run_single.sh configs/cora_secure.cfg gpu runs/cora_gpu` |
| Five CPU runs | `ZKDL_ALLOW_FULL_DATASET_RUN=1 scripts/run_five.sh cpu` |
| Five GPU runs | `ZKDL_ALLOW_FULL_DATASET_RUN=1 scripts/run_five.sh gpu` |
| Verify one run | `scripts/verify_single.sh configs/cora_secure.cfg runs/cora_cpu` |
| Collect records | `python3 scripts/collect_results.py --runs runs --output results/collected.json` |
| Final matrix validator | `python3 scripts/final_validate.py --runs runs --require-pairs` |

`ZKDL_ALLOW_FULL_DATASET_RUN=1` is mandatory for the single/five runners and
is the explicit user authorization to read the full copied dataset and start
proving.  It is intentionally not set by tests or build targets.

The current SRS paths in the configs are absent.  `setup_register.sh` exits
with status 3 rather than creating an untrusted SRS.  Likewise the native
bridge returns `BLOCKED` while core verifier APIs for the complete GAT
relation are unavailable.  Neither outcome is a formal result.
