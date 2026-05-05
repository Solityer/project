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

The C++ runner now follows the full-dataset proof pipeline through config load, dataset load, public-input equivalence, checkpoint tensor parsing, GAT proof-statement domain construction, destination-group boundary construction, and a real full-dataset hidden-layer projection prefix. The prefix reads the actual feature tensor and the first hidden-head projection kernel from the checkpoint bundle, computes `H' = H * W` for every node, and records a quantized checksum and timing evidence in each manifest before any blocker is recorded.

The zkDL backend in this directory exposes CUDA tensor commitment, multilinear evaluation, inner-product and Hadamard-product sumcheck, binary sumcheck, matrix multiplication, product, and ReLU-style range machinery. After the projection prefix, the first missing full-GAT relation is now the public sparse route sumcheck for `R_src`/`R_dst`; later missing relations include destination-group max, inverse, exponent and ELU lookup arguments, rescale arguments, FAC4DNN aggregation/slice consistency, final public logits binding, and a verifier acceptance API for the composed GAT statement.

For that reason this baseline records formal blockers instead of marking any full GAT result as verified. GPU probing still runs a real zkDL CUDA primitive path and records whether a CUDA backend is active. A GPU manifest is never marked verified when fallback or missing GPU support occurs.
