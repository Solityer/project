# zkDL GAT Full Forward Baseline

This directory is an independent zkDL comparison baseline. It builds from `comparison/zkdl`, carries its own C++ runner, configs, scripts, manifests, results, and copied zkDL primitive headers under `third_party/zkdl`.

The baseline treats the main repository as a read-only source for full dataset caches, checkpoint bundles, original configs, and published main-run manifests. It never includes or executes the main repository `src`, `include`, or `scripts` trees.

## Build

CPU capability route:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

GPU capability route:

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

Collected tables:

- `results/zkdl_cpu_full_results.json`
- `results/zkdl_cpu_full_results.csv`
- `results/zkdl_gpu_full_results.json`
- `results/zkdl_gpu_full_results.csv`
- `results/main_vs_zkdl_cpu_full_comparison.json`
- `results/main_vs_zkdl_cpu_full_comparison.csv`
- `results/main_vs_zkdl_gpu_full_comparison.json`
- `results/main_vs_zkdl_gpu_full_comparison.csv`

## Current zkDL Capability Boundary

The copied zkDL backend exposes CUDA tensor commitment, multilinear evaluation, inner-product and Hadamard-product sumcheck, binary sumcheck, matrix multiplication, product, and ReLU-style range machinery. It does not expose a CPU tensor commitment/proof backend, a verifier acceptance API, or the full GAT proof relations required here: destination-group max, inverse, exponent and ELU lookup arguments, rescale arguments, FAC4DNN sparse aggregation, and final public logits binding.

For that reason this baseline records formal blockers instead of marking any full GAT result as verified. GPU probing still runs a real zkDL CUDA primitive path and records whether a CUDA backend is active. A GPU manifest is never marked verified when fallback or missing GPU support occurs.
