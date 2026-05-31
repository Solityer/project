# Experiment Notes

## Current Result Set

`results/summary.csv` contains the current official zkGPT baseline rerun:

- CPU: cora, citeseer, pubmed, ppi, ogbn-arxiv.
- GPU: cora, citeseer, pubmed, ppi, ogbn-arxiv.

Every row has `status=VERIFY_OK`, a per-run `run_manifest.json`, and component
artifacts under `results/<backend>/<dataset>/`.

The previous adapter-only and digest/replay outputs remain in `archive/` and are
excluded from `results/summary.csv`, `results/cpu_results.csv`,
`results/gpu_results.csv`, and `results/summary.json`.

## Protocol Alignment

The baseline follows the protocol identifiers and field order required by
`/home/pzh/project/comparison/GAT-ZKML-zkGPT.md`:

- `M_pub`
- `Y`
- `Com_in = (com_T_H, com_W, com_base, com_comp, com_psq, com_qry)`
- `Pi_Lasso`
- `Pi_MatMul`
- `Pi_GKR`
- `Pi_combine`
- `Open_in`

The GAT forward path remains sparse over the public edge list. It keeps
`node_ptr` and `edge_ptr` for single-graph datasets and PPI multi-graph batches.
Hidden heads apply ELU after attention aggregation and are concatenated before
the next layer. The output layer remains attention-head based; each output head
adds its own bias and the final logits are the arithmetic mean of post-bias
output heads.

## Proof Artifacts

Each dataset/backend directory contains:

- `metadata.json`
- `commitments.json`
- `transcript.json` and `challenges.json`
- `lookup_proof.json`
- `matrix_sumcheck_proof.json`
- `gkr_proof.json`
- `hyrax_mle_openings.json`
- `masked_openings.json`
- `proof_payload.json` and `proof_object.json`
- `verification.json` and `verifier_output.json`
- `witness_manifest.json`
- `benchmark_summary.json`
- `run_manifest.json`
- `gpu_proof_hot_path.json` for GPU runs

`proof_payload.json` references the component artifact digests. The verifier
checks metadata, commitment order, transcript replay, native Hyrax/MLE opening
status, Lasso status, matrix sumcheck status, GKR status, combined claim status,
and GPU proof hot-path state for GPU runs.

## Range Lookup

The range table is the 12-bit table `T_range = {0, ..., 2^B-1}`. Full-graph
attention deltas are encoded with structured 12-bit limb decomposition under
`lasso_cfg_id = gat-zkgpt-lasso-lrelu-elu-exp-range-structured-limbs-v1`.

## GPU Path

GPU runs execute `backend/cuda/zkgpt_cuda_digest.cu`, which now implements the
`proof_field_reduce_v1` CUDA kernel over proof-claim field words. The resulting
`gpu_proof_hot_path.json` is included in the proof payload and verified through
the `gpu_hot_path` verifier check.

## Negative Tests

`scripts/run_negative_tests.py` writes
`results/negative_tests/formal_negative_tests.json` and covers:

- metadata tamper
- commitment tamper
- transcript challenge tamper
- lookup proof tamper
- matrix sumcheck proof tamper
- GKR proof tamper
- masked opening tamper
- output claim tamper
- PPI graph boundary tamper
- GPU backend state tamper

All tamper cases must produce `VERIFY_FAILED`.
