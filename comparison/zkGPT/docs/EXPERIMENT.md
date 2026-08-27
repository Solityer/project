# zkGPT formal experiment commands

This baseline implements the native four-segment `X_pub|W|Aux|L_qry` path.
Run the commands from the baseline root `comparison/zkGPT`.  Resource
preparation creates only baseline-local directories and the canonical
registration manifest.  The project owner must stage each complete dataset,
checkpoint bundle and public-Y file under the declared baseline-local paths;
no command below imports resources from the main experiment or another
baseline.

```bash
cd comparison/zkGPT
python3 scripts/prepare_resources.py
# Owner stages data/cache/{cora,citeseer,pubmed,ppi,ogbn_arxiv}/,
# artifacts/checkpoints/*_gat/, and resources/public_y/*.json.
# The command creates only empty baseline-local layout directories, including
# results/model_registry/ and artifacts/private_setup/model_registry/.
python3 scripts/prepare_resources.py --check
python3 scripts/formal_preflight.py --source-only
python3 scripts/formal_preflight.py
./scripts/build.sh --cpu
./scripts/build.sh --gpu
for dataset in cora citeseer pubmed ppi ogbn-arxiv; do
  python3 scripts/register_model.py --dataset "$dataset"
done
```

Each registration is one-time setup: after the capability gate, staged-resource
check and identity/path checks, it reads that dataset's checkpoint and writes
the authenticated public `com_W` descriptor under `results/model_registry/`.
Hyrax blinders remain only in the baseline-local private setup root under
`artifacts/private_setup/model_registry/`; the independent verifier receives
only the public registry file.

The owner then runs each single cell (or the matrix runner).  Every single-cell
manifest is written only after a real proof and fresh independent verifier
return `VERIFY_OK`:

```bash
python3 scripts/gat_zkgpt_experiment.py --datasets cora --backends cpu
python3 scripts/gat_zkgpt_experiment.py --datasets cora --backends gpu
# Repeat the two commands for citeseer, pubmed, ppi, and ogbn-arxiv.
```

The runner accepts `--results-dir` only when it names this baseline's exact
`results/` root.  Each cell is written to the canonical
`results/runs/<backend>/<dataset>/` directory; sibling aliases and the
non-canonical `results/<backend>/` layout are rejected.

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

After all five CPU and five GPU cells are completed by the owner, collect and
run the strict matrix-level final validator:

```bash
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

The collector fresh-replays every present cell and rejects test fixtures,
synthetic caches, stopped/partial artifacts, `FAILED_DIAGNOSTIC` states, and
duplicate proof names.  The final validator requires exactly 5×2 real cells,
fresh replay evidence, and multi-session privacy checks.  Neither tool is a
substitute for a real single-cell proof/verify, and this preparation stage
does not claim any formal dataset result.
