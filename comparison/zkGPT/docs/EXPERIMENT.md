# zkGPT formal experiment commands

This baseline implements the native four-segment `X_pub|W|Aux|L_qry` path.
Run the source preflight and one-time model registration before the formal
matrix:

```bash
python3 scripts/formal_preflight.py --source-only
REGISTRY="$(python3 scripts/register_model.py --dataset cora)"
```

Run CPU and GPU matrices separately, then collect and validate artifacts:

```bash
PUBLIC_Y_ROOT=/path/to/independent/public_y
python3 scripts/gat_zkgpt_experiment.py --datasets all --backends cpu --public-y-root "$PUBLIC_Y_ROOT" --force
python3 scripts/gat_zkgpt_experiment.py --datasets all --backends gpu --public-y-root "$PUBLIC_Y_ROOT" --force
PUBLIC_Y="$PUBLIC_Y_ROOT/cora.json"
python3 scripts/verify_proof.py \
  --project-root /home/pzh/project \
  --config configs/generated/cpu/cora_full.cfg \
  --trusted-registry-public "$REGISTRY" \
  --artifact-dir results/runs/cpu/cora \
  --public-y "$PUBLIC_Y"
python3 scripts/verify_proof.py \
  --project-root /home/pzh/project \
  --config configs/generated/gpu/cora_full.cfg \
  --trusted-registry-public "$REGISTRY" \
  --artifact-dir results/runs/gpu/cora \
  --public-y "$PUBLIC_Y"
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py --public-y-root "$PUBLIC_Y_ROOT"
```

The GPU command requires a CUDA device and the built
`build/zkgpt_cuda_native_prover`; no CPU substitution is performed.
GPU artifacts must carry schema `zkGPT_gpu_proof_messages_v3` and kernel
identity `zkgpt_native_bn254_fr_batch_v3`. Collection and validation reject
missing or incomplete operation cardinalities, incorrect coefficient-degree or
output-slot counts, and counter mismatches; the independent proof verifier
checks the actual masked field messages. The run-scoped execution envelope is
`zkGPT_gpu_proof_execution_v3`.
The artifact also records the CUDA device UUID and a
`cudaEventElapsedTime` measurement for the native invocation.
