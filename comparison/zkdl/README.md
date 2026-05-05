# zkDL GAT Full Forward Baseline

This directory is the independent zkDL comparison workspace for the full-dataset
GAT forward statement.

It does not include or link against the main repository `src`, `include`, or
`scripts` directories. The only main-repository inputs used at runtime are the
declared read-only dataset caches and checkpoint bundles listed in
`configs/datasets/*.cfg`, each with a recorded SHA-256 hash.

## Layout

- `third_party/zkdl/`: copied zkDL CUDA tensor commitment and sumcheck primitive source.
- `include/zkdl_gat/`, `src/zkdl_gat/`: C++ experiment runner and CUDA primitive probe.
- `configs/datasets/`: full-dataset source manifests for Cora, Citeseer, Pubmed, PPI, and ogbn-arxiv.
- `scripts/`: build, run, collect, and validate commands.
- `runs/`: per-dataset CPU/GPU run manifests.
- `results/`: aggregate JSON/CSV result tables and main-vs-zkDL comparisons.

## Build

CPU runner:

```bash
cd /home/pzh/project/comparison/zkdl
scripts/build_cpu.sh
```

GPU runner and zkDL CUDA primitive probe:

```bash
cd /home/pzh/project/comparison/zkdl
scripts/build_gpu.sh
```

## Run

CPU full-dataset route:

```bash
cd /home/pzh/project/comparison/zkdl
scripts/run_cpu_full.sh
```

GPU full-dataset route:

```bash
cd /home/pzh/project/comparison/zkdl
scripts/run_gpu_full.sh
```

Both routes:

```bash
cd /home/pzh/project/comparison/zkdl
scripts/run_full_experiment.sh
```

## Result Files

CPU:

- `results/zkdl_cpu_full_results.json`
- `results/zkdl_cpu_full_results.csv`
- `results/main_vs_zkdl_cpu_full_comparison.json`
- `results/main_vs_zkdl_cpu_full_comparison.csv`

GPU:

- `results/zkdl_gpu_full_results.json`
- `results/zkdl_gpu_full_results.csv`
- `results/main_vs_zkdl_gpu_full_comparison.json`
- `results/main_vs_zkdl_gpu_full_comparison.csv`

Per-dataset manifests:

- `runs/<dataset>/cpu/run_manifest.json`
- `runs/<dataset>/gpu/run_manifest.json`

## Verification Rules

The validator enforces:

- `verified` requires real proof metrics, strict public input equivalence, full GAT proof coverage, complete final-logits derivation, and no precomputed logits bypass.
- GPU `verified` additionally requires an active GPU backend and `cpu_fallback_used=false`.
- Blocked or failed rows must carry a concrete `failure_step` and `block_reason`.
- All declared input hashes must match the read-only source files.

## Current zkDL Backend Boundary

The copied zkDL backend provides CUDA tensor commitment, multilinear opening,
inner-product sumcheck, hadamard-product sumcheck, binary sumcheck, and limited
matmul/product/ReLU proof helpers. It does not expose the full API needed to
encode the requested GAT full forward statement: grouped softmax max, inverse,
exp/ELU lookup arguments, range arguments, rescale constraints, FAC4DNN
aggregation, final logits public binding, and a verifier acceptance API for the
whole statement.

For that reason, this baseline records the CPU and GPU full-dataset attempts as
blocked rather than marking any row as verified. The GPU route still runs a real
zkDL CUDA primitive probe and records device/fallback metadata; it does not use
CPU proof output as a GPU result.
