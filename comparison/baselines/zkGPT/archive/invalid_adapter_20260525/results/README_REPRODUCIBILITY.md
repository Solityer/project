# GAT-ZKML-zkGPT Reproducibility

All generated experiment artifacts live under this zkGPT directory.
The source datasets, checkpoint bundles, and config semantics are read-only inputs from /home/pzh/project.

Run all official rows:

```bash
./scripts/run_gat_zkgpt_all.sh
```

Result tables:

- results/cpu_results.csv
- results/gpu_results.csv
- results/summary.csv

proof_size_bytes is copied from the verified engine run_manifest proof payload size.
Audit JSON files such as metadata.json, challenges.json, and proof_object.json are excluded from proof_size_bytes.
