# Experiment Notes

## Current Validity Boundary

The previous Python digest/replay artifacts are invalid and have been moved out
of `results/`. They must not be included in official tables.

The native `zkgpt_hyrax_mle` backend now provides mcl BN254 Hyrax/MLE
commit/open/verify functionality for witness segments. The full Lasso, matrix
sumcheck, ML-friendly GKR, combine, CPU/GPU proof hot path, and negative-test
integration are still the active implementation work before any official
`VERIFY_OK` row can be regenerated.

## Protocol Alignment

The baseline must use the protocol identifiers and field order required by
`/home/pzh/project/comparison/GAT-ZKML-zkGPT.md`:

- `M_pub`
- `Y`
- `Com_in = (com_T_H, com_W, com_base, com_comp, com_psq, com_qry)`
- `Pi_Lasso`
- `Pi_MatMul`
- `Pi_GKR`
- `Pi_combine`
- `Open_in`

The GAT forward path must remain sparse over the public edge list. It keeps `node_ptr` and
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

The existing CUDA digest helper is not a valid GPU proof backend. GPU rows must
use a CUDA path that participates in witness/proof hot-path computation before
they can be written to official results.

## Invalid Adapter Archive

The previous adapter-only run was moved to `archive/invalid_adapter_20260525/`. Those files
are intentionally excluded from `results/summary.csv`, `results/summary.json`,
`results/cpu_results.csv`, and `results/gpu_results.csv`.
