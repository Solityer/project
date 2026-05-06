# zkDL GAT Full Forward Baseline

This directory is the only zkDL comparison baseline entry point:

`comparison/baselines/zkdl`

It carries the original zkDL CUDA/header primitives in this directory and adds an independent C++ manifest runner, CUDA primitive probe, configs, scripts, manifests, and result collection around those primitives.

The baseline treats the main repository as a read-only source for full dataset caches, checkpoint bundles, original configs, and published main-run manifests. It never includes or executes the main repository `src`, `include`, or `scripts` trees.

## Build

CPU route:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

GPU route:

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DZKDL_GAT_ENABLE_CUDA=ON
cmake --build build_cuda -j
```

## Run

```bash
python3 scripts/run_full_baseline.py --route both
python3 scripts/validate_results.py
```

Equivalent wrappers:

```bash
scripts/build_cpu.sh
scripts/build_gpu.sh
scripts/run_cpu_full.sh
scripts/run_gpu_full.sh
scripts/validate_results.sh
```

## Outputs

Per-dataset manifests:

- `runs/<dataset>/cpu/run_manifest.json`
- `runs/<dataset>/gpu/run_manifest.json`
- `runs/<dataset>/<route>/full_gat_semantic_precheck.json`

Collected tables:

- `results/zkdl_cpu_full_results.json`
- `results/zkdl_cpu_full_results.csv`
- `results/zkdl_gpu_full_results.json`
- `results/zkdl_gpu_full_results.csv`
- `results/main_vs_zkdl_cpu_full_comparison.json`
- `results/main_vs_zkdl_cpu_full_comparison.csv`
- `results/main_vs_zkdl_gpu_full_comparison.json`
- `results/main_vs_zkdl_gpu_full_comparison.csv`

## Current zkDL Proof Boundary

The CPU and GPU routes first run a full-dataset gate against the main repository config, data cache, checkpoint bundle, and published main-run manifest. The gate records dataset id, full-dataset status, node/edge counts, feature/class dimensions, graph batch pointers, layer/head configuration, self-loop and edge-sort policy, source hashes, checkpoint hashes, public-input hash, witness-coverage hash, and final-logits hash.

The local full forward and relation checker are now recorded only as `semantic_precheck_passed`. That artifact is named `full_gat_semantic_precheck.json` and is explicitly marked `semantic_precheck_not_zkdl_proof`. It is not counted as `proof_size_bytes` and cannot make a row `verified`.

`status=verified` is reserved for a real zkDL full-GAT prover output plus a real zkDL verifier acceptance path with `proof_backend=zkDL`, `verifier_backend=zkDL`, `zkdl_proof_verified=true`, and a transferable zkDL proof object. The current composed full-GAT zkDL prover/verifier is still missing, so full-dataset rows are blocked after the gate/precheck stage instead of being falsely verified.

The CUDA probe only exercises the original zkDL CUDA tensor commitment and sumcheck primitive path. GPU full-GAT rows remain blocked unless a CUDA composed full-GAT prover/verifier exists without CPU fallback.
