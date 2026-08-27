# GAT-ZKML zkGPT comparison baseline

This directory is an independent zkGPT-native GAT implementation. Its proof
backend is the BN254 Hyrax-style MLE commitment layer with masked logarithmic
IPA openings, native Lasso-style packed-table/range arguments, dedicated
integer matrix-product Sumcheck, layered GKR relation claims, input-layer
combine, and independent replay verification.

The governing protocol is [`GAT-ZKML-zkGPT.md`](GAT-ZKML-zkGPT.md). The public
statement contains public graph/features/topology, registered `com_W`, static
tables and the full public logits. The input layer is canonically
`X_pub|W|Aux|L_qry`; `X_pub` is reconstructed by the verifier, while `Aux` and
`L_qry` commitments and all masking/opening randomness are fresh per proof.

## Layout

* `src/zkgpt_gat/` — exact integer GAT forward, BN254 group Hyrax/IPA, Lasso,
  MatMul Sumcheck, layered GKR and independent verifier.
* `backend/cuda/` — CUDA proof-primitive execution backend.
* `scripts/` — formal preflight, build, single backend/dataset runner, matrix
  runner, collection and validation.
* `configs/` — capability and generated run configurations.
* `results/` — the only formal result root.  Per-cell artifacts are stored
  under `results/runs/<backend>/<dataset>/`; summaries are written directly
  under `results/`.  The retired `results/<backend>/` layout is invalid.

Python requires NumPy. CUDA runs additionally require `nvcc`, a CUDA device and
the configured CUDA runtime. The GPU path fails if the native CUDA proof
backend is not selected; it never relabels CPU work as GPU work.
GPU proof artifacts use the strict `zkGPT_gpu_proof_messages_v3` contract and
`zkgpt_native_bn254_fr_batch_v3` kernel identity. Validation checks canonical
BN254-Fr operation cardinalities, exact Lasso/Sumcheck/Hyrax round counts,
coefficient-degree/output-slot contracts, device UUID, CUDA-event timing and
primitive counters; the formal verifier checks the masked proof messages and
their actual field values. A probe-only label or CPU result is not accepted.
The run-scoped execution envelope is
`zkGPT_gpu_proof_execution_v3` with semantic id
`zkgpt_cuda_authors_bn254_fr_proof_messages_v3`.

The callable backend API is `gpu_proof_message_batch(zk_root, operations,
label)`; operations are `lasso_product_tree`, `sumcheck_mle_fold`,
`sumcheck_product2/3` and `hyrax_linear_eval_fold`. For sequential
Fiat--Shamir rounds, `gpu_proof_round(...)` returns the current GPU-authored
round message before the next challenge is derived. The independent CPU
reference is `recompute_gpu_proof_batch(...)`.

## Formal commands

All commands below are run from this baseline root (`comparison/zkGPT`).  The complete formal
datasets, model checkpoints and public logits are owned and staged by the
project owner under the paths declared in `resources/resource_manifest.json`;
the preparation command never downloads, preprocesses, or executes them.

```bash
cd comparison/zkGPT
python3 scripts/prepare_resources.py
# Stage the five owner-managed resource bundles under data/cache/,
# artifacts/checkpoints/, and resources/public_y/ before continuing.
# This creates only empty baseline-local layout directories; model registration
# later keeps blinders under artifacts/private_setup/model_registry/.
python3 scripts/prepare_resources.py --check
python3 scripts/formal_preflight.py --source-only
python3 scripts/formal_preflight.py
./scripts/build.sh --cpu
./scripts/build.sh --gpu
for dataset in cora citeseer pubmed ppi ogbn-arxiv; do
  python3 scripts/register_model.py --dataset "$dataset"
done
# The following single-cell command is the owner-run proof + independent
# verify path; it writes a manifest only after fresh VERIFY_OK.
python3 scripts/gat_zkgpt_experiment.py --datasets cora --backends cpu
python3 scripts/collect_results.py
# After the owner has run all five datasets on both backends:
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

`verify_proof.py` replays only the serialized public statement and proof
messages from a fresh process; it never loads the witness arrays. A single
cell is successful only when its real proof and fresh independent verifier
return `VERIFY_OK`; the matrix validator is not a prerequisite for that cell.
The five-dataset CPU/GPU matrix and all performance measurements are left to
the project owner. No formal matrix result is claimed by this repository.

The strict privacy profile is `masked_full_multi_session_v1`, with native
BN254 security parameters and fresh per-proof masks. Constraint fusion is
disabled unless exact integer checkpoint equivalence is established; circuit
squeeze only changes GKR topology and not the relation registry.
