# Experiment Notes

## Protocol Alignment

The baseline uses the protocol identifiers and field order required by
`/home/pzh/project/comparison/GAT-ZKML-zkGPT.md`:

- `M_pub`
- `Y`
- `Com_in = (com_T_H, com_W, com_base, com_comp, com_psq, com_qry)`
- `Pi_Lasso`
- `Pi_MatMul`
- `Pi_GKR`
- `Pi_combine`
- `Open_in`

The GAT forward path is sparse over the public edge list. It keeps `node_ptr` and
`edge_ptr` for single-graph datasets and PPI multi-graph batches. Hidden heads apply
ELU after attention aggregation and are concatenated before the next layer. The output
layer remains attention-head based; each output head adds its own bias and the final
logits are the arithmetic mean of post-bias output heads.

## Range Lookup

The formal range table is the 12-bit table `T_range = {0, ..., 2^B-1}`. Large full-graph
attention deltas, especially on `ogbn-arxiv`, are encoded through structured 12-bit limb
decomposition under `lasso_cfg_id = gat-zkgpt-lasso-lrelu-elu-exp-range-structured-limbs-v1`.
The proof records the original max delta, limb count, and max limb. This still uses the
same structured `T_range` table and avoids pretending that a multi-limb value fits in one
12-bit lookup.

## GPU Path

GPU rows require `backend/cuda/zkgpt_cuda_digest.cu` to build and execute. The runner
aborts if CUDA is unavailable or if the helper does not report `cuda_kernel_executed=true`.
The GPU status is written to each `results/gpu/<dataset>/gpu_backend.json`.

## Invalid Adapter Archive

The previous adapter-only run was moved to `archive/invalid_adapter_20260525/`. Those files
are intentionally excluded from `results/summary.csv`, `results/summary.json`,
`results/cpu_results.csv`, and `results/gpu_results.csv`.

