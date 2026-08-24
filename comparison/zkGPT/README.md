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
* `results/` — formal artifacts and summaries.

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

```bash
./scripts/build.sh --cpu
./scripts/build.sh --gpu
PUBLIC_Y_ROOT=/path/to/independent/public_y
python3 scripts/gat_zkgpt_experiment.py --datasets cora --backends cpu --public-y-root "$PUBLIC_Y_ROOT"
python3 scripts/gat_zkgpt_experiment.py --datasets cora --backends gpu --public-y-root "$PUBLIC_Y_ROOT"
REGISTRY="$(python3 scripts/register_model.py --dataset cora)"
PUBLIC_Y="$PUBLIC_Y_ROOT/cora.json"
python3 scripts/verify_proof.py \
  --project-root /home/pzh/project \
  --config configs/generated/cpu/cora_full.cfg \
  --trusted-registry-public "$REGISTRY" \
  --artifact-dir results/runs/cpu/cora \
  --public-y "$PUBLIC_Y"
./scripts/run_gat_zkgpt_all.sh --public-y-root "$PUBLIC_Y_ROOT"
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py --public-y-root "$PUBLIC_Y_ROOT"
```

`verify_proof.py` replays only the serialized public statement and proof
messages from a fresh process; it never loads the witness arrays. The commands
above are prepared for formal execution; full proof generation,
verification and performance measurements are intentionally left to the
project owner for the formal experiment stage. Historical artifacts are not
used as current verification evidence.

The strict privacy profile is `masked_full_multi_session_v1`, with native
BN254 security parameters and fresh per-proof masks. Constraint fusion is
disabled unless exact integer checkpoint equivalence is established; circuit
squeeze only changes GKR topology and not the relation registry.
