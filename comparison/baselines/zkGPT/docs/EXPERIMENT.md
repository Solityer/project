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
- `fixed_lookup_tables.json`
- `lookup_proof.json`
- `matrix_sumcheck_proof.json`
- `gkr_proof.json`
- `combine_proof.json`
- `hyrax_mle_openings.json`
- `masked_openings.json`
- `proof_payload.json` and `proof_object.json`
- `verification.json` and `verifier_output.json`
- `witness_manifest.json`
- `benchmark_summary.json`
- `run_manifest.json`
- `gpu_proof_hot_path.json` for GPU runs

`proof_payload.json` carries the verifier-facing proof messages directly:
`Com_in`, `FixedLookupTables`, `Pi_Lasso`, `Pi_MatMul`, `Pi_GKR`,
`Pi_combine`, `Open_in`, and `OpenMasked`. GPU runs also include
`GPU_proof_contribution`. Component artifact references are retained only as
audit pointers. `proof_size_bytes` is the canonical serialized size of this
payload.

Witness segment commitment no longer uses statistical summaries. Each segment is
flattened in protocol order with array headers and every encoded element, split
into contiguous chunks, and committed/opened with the BN254 Hyrax-style MLE
OpenMasked backend. `commitments.json` records per-segment vector length, chunk
count, full-array digests, chunk commitments, and masked opening metadata.

The verifier checks metadata completeness, commitment order, transcript replay,
Hyrax/MLE masked opening messages, fixed-table Lasso lookup messages, matrix
sumcheck messages plus output/bias/rescale equations, matrix rescale range
bindings through Lasso, GKR layered relation sumcheck messages, the combined
claim digest, and GPU proof hot-path state for GPU runs. It does not accept an
artifact `status` field as a proof by itself.

## Fixed-Table Lasso Lookup

`fixed_lookup_tables.json` binds `T_LReLU`, `T_ELU`, `T_exp`, and `T_range` to
`quant_cfg_id`, `static_table_id`, `range_bits`, and `encoding_id`. Function
lookups use structured fixed-function table descriptors; range lookup uses a
materialized fixed table `T_range = {0, ..., 2^B-1}`. The generated configs keep
the main repository's `range_bits` value for each dataset; current PPI artifacts
therefore use `B=14`, while the other full configs use `B=12`.

Query multisets come from the committed `L_qry` witness segment. Function
families challenge-compress raw `(x,y)` query columns after `com_qry`. The
structured opening binds the table id, table digest, table commitment,
structured verification key, input-domain chunks, expected-y chunks, selected
fixed-table chunks, raw query digest, and Fiat-Shamir challenges. The verifier
replays deterministic structured-key samples against the fixed evaluation rule
and rejects tampered query values, expected values, selected opening messages,
table ids, commitments, domains, or challenges.

Range families are intentionally named `direct_limb_bounded_range_argument`
rather than Lasso selected-table products. They decompose full-graph attention
deltas and matrix rounding residual limbs, bind the fixed `T_range` table
commitment/digest, and verify limb chunk coverage, non-negativity,
`max_limb < 2^B`, the direct range accumulator, the full fixed table product,
and transcript-derived beta. The verifier no longer accepts
`table_product = query_product`, and matrix rescale limbs are not reused as a
fake selected table side.

## Matrix Sumcheck and GKR

Projection and attention inner-product claims use full matrix random-evaluation
claims. The prover folds all rows and columns with transcript challenges and
uses product sumcheck over the complete inner dimension; single row/column
claims are not part of the official path. The verifier checks the final field
equation for every claim:

- matrix multiplication: `claimed_sum - output_eval_q2 - committed_residual_eval = 0`
- matrix-vector attention: `claimed_sum + bias_eval_q2 - output_eval_q2 - committed_residual_eval = 0`

Each matrix claim also carries a committed per-entry rescale/rounding binding.
The prover materializes `*.rescale_residual` arrays into `com_qry`; the matrix
claim proves the transcript-weighted evaluation of that committed residual
oracle with a separate product sumcheck. Per-entry residuals are range-bounded
with an operand-L1-derived quantization bound and their limb chunks are
cross-bound to the `rescale_rounding_range` direct fixed-range argument. The
rescale field is therefore no longer accepted as an unconstrained aggregate
difference.

The ML-friendly GKR artifact is serialized as layered relation claims rather
than residual zero-checks. Each relation carries left/right oracle chunk
digests, two random-linear-evaluation sumchecks, transcript seeds, final oracle
claims, and committed-segment opening references. The verifier checks
Fiat-Shamir seed replay, sumcheck transitions, final products, equality of
left/right random-linear oracle claims, final oracle claim messages, committed
witness opening proof digests, and opening-reference digests. Left/right chunk
equality, `mismatch_count`, and max error fields are retained only as
debug/audit metadata, not as accepting conditions. Covered relations include public
edge/node/edge-pointer wiring, score, LeakyReLU lookup query consistency,
max/sum state machines, delta, Exp lookup query consistency, inverse, alpha,
PSQ compressed aggregation, ELU lookup query consistency, concat, output bias,
and output head average.

## GPU Path

GPU runs execute `backend/cuda/zkgpt_cuda_proof_hot_path.cu` as
`segment_lookup_matrix_gkr_hot_path_fold_v2` over field words extracted from
segment encoding/opening claims, fixed-table lookup query/direct-range
accumulators, matrix sumcheck folds plus committed residual range bindings, and
GKR layer oracle/opening-reference bindings. The resulting
`gpu_proof_hot_path.json` is included in the proof payload as
`GPU_proof_contribution`; the verifier recomputes the CUDA input digest from the
proof bundle, checks `cpu_fallback=false`, and rejects a mismatched GPU
contribution.

## Negative Tests

`scripts/run_negative_tests.py` writes
`results/negative_tests/formal_negative_tests.json` and covers:

- metadata tamper
- commitment tamper
- fixed lookup table commitment tamper
- fixed lookup table id tamper
- fixed lookup table domain tamper
- structured verification key tamper
- transcript challenge tamper
- lookup query tamper
- selected structured table opening tamper
- lookup challenge tamper
- range limb/direct range argument tamper
- range table product tamper
- lookup proof tamper
- matrix sumcheck proof tamper
- matrix output evaluation tamper
- matrix bias evaluation tamper
- matrix rescale advice tamper
- matrix rescale limb/range binding tamper
- matrix rounding bound tamper
- GKR proof tamper
- GKR final oracle claim tamper
- GKR Fiat-Shamir seed tamper
- GKR oracle opening reference tamper
- committed witness opening proof tamper
- masked opening tamper
- output claim tamper
- PPI graph boundary tamper
- GPU backend state tamper
- witness segment chunk tamper
- Hyrax opening proof tamper
- direct subproof relation tamper for lookup grand product/challenges/range
  selected table openings, fixed table commitment, matrix output/rescale/range
  equations, GKR rounds, GKR Fiat-Shamir seed, GKR opening references, and GKR
  final oracle claims

All tamper cases must produce `VERIFY_FAILED`.

`scripts/audit_formal_proof_semantics.py` writes
`results/formal_proof_semantics_audit.json`. It audits the formal source path
and every CPU/GPU artifact bundle for structured fixed-function lookup
membership arguments, direct fixed-range arguments, committed rescale bindings,
GKR final-oracle opening bindings, combine proof binding, and GPU hot-path
contribution binding.
