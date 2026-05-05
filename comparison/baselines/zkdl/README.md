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
- `runs/<dataset>/cpu/zkdl_gat_composed_proof.json`

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

The CPU route now runs a composed full-dataset GAT proof pipeline inside this baseline. It loads the real full dataset and checkpoint bundle, constructs the normalized public sparse route, computes all hidden-head projection, attention score, activation, destination-group, normalization, aggregation, hidden concat, output attention, output bias/average, and final logits binding stages, writes `runs/<dataset>/cpu/zkdl_gat_composed_proof.json`, then invokes the baseline verifier on that artifact. A CPU row is marked `verified` only when the artifact is generated, the verifier is called, every full-GAT relation check passes, final logits are derived inside the proof pipeline, public inputs match the main manifest, and all four metrics are non-null.

The CUDA probe still exercises the original zkDL CUDA tensor commitment and sumcheck primitive path. GPU full-GAT rows remain blocked unless a CUDA composed full-GAT prover/verifier exists without CPU fallback; the current GPU failure step is `gpu_composed_gat_prover` after the primitive probe succeeds.

A GPU manifest is never marked verified when fallback or missing GPU composed support occurs.
