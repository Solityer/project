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

`proof_payload.json` carries the verifier-facing proof messages directly:
`Com_in`, `Pi_Lasso`, `Pi_MatMul`, `Pi_GKR`, `Pi_combine`, `Open_in`, and
`OpenMasked`. Component artifact references are retained only as audit pointers.
`proof_size_bytes` is the canonical serialized size of this payload.

Witness segment commitment no longer uses statistical summaries. Each segment is
flattened in protocol order with array headers and every encoded element, split
into contiguous chunks, and committed/opened with the BN254 Hyrax-style MLE
OpenMasked backend. `commitments.json` records per-segment vector length, chunk
count, full-array digests, chunk commitments, and masked opening metadata.

The verifier checks metadata completeness, commitment order, transcript replay,
Hyrax/MLE masked opening messages, Lasso grand-product lookup messages, matrix
sumcheck messages, GKR zero-sumcheck messages, the combined claim digest, and GPU
proof hot-path state for GPU runs. It does not accept an artifact `status` field
as a proof by itself.

## Range Lookup

The range table is the 12-bit table `T_range = {0, ..., 2^B-1}`. Full-graph
attention deltas are encoded with structured 12-bit limb decomposition under
`lasso_cfg_id = gat-zkgpt-lasso-lrelu-elu-exp-range-structured-limbs-v1`.
LeakyReLU, ELU, and Exp lookups use the same grand-product transcript shape:
query and table multisets are challenge-compressed, chunk-digested, and checked
through the serialized proof messages.

## Matrix Sumcheck and GKR

Projection and attention inner-product claims use full matrix random-evaluation
claims. The prover folds all rows and columns with transcript challenges and
uses product sumcheck over the complete inner dimension; single row/column
claims are not part of the official path.

The ML-friendly GKR artifact is serialized as zero-constraint sumcheck
transcripts over the GAT arithmetic constraints: score, max/sum state machines,
delta, inverse, alpha, PSQ compressed aggregation, concat, output bias, and
output head average.

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
- witness segment chunk tamper
- Hyrax opening proof tamper

All tamper cases must produce `VERIFY_FAILED`.
