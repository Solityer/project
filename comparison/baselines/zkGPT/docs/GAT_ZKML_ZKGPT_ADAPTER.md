# GAT-ZKML-zkGPT Adapter

This directory contains the zkGPT baseline entrypoint used for the GAT-ZKML comparison run.

Official commands:

```bash
bash build.sh Release
./scripts/run_gat_zkgpt_all.sh --force
```

Backend-specific commands:

```bash
./scripts/run_gat_zkgpt_backend.sh cpu all --force
./scripts/run_gat_zkgpt_backend.sh gpu all --force
```

The scripts generate local configs under `generated_configs/` and write all run outputs under
`results/`. Source datasets, checkpoint bundles, and the main repository configs are read-only
inputs from `/home/pzh/project`.

Required result tables:

- `results/cpu_results.csv`
- `results/gpu_results.csv`
- `results/summary.csv`
- `results/summary.json`

Each dataset/backend artifact directory contains:

- `run_manifest.json`
- `metadata.json`
- `challenges.json`
- `commitments.json`
- `proof_order.json`
- `proof_object.json`
- `verification.json`
- `benchmark_summary.json`

`proof_size_bytes` follows the verified proof payload size exported by the execution manifest.
Audit JSON files are excluded from this byte count.
