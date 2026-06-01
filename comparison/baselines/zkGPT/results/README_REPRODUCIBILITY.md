# GAT-ZKML-zkGPT Results

This directory contains the official rerun independent zkGPT-style baseline results.
The previous adapter and digest/replay artifacts live under `archive/` and are excluded from these tables.

Official tables:

- `summary.csv` / `summary.json`
- `cpu_results.csv`
- `gpu_results.csv`

Each dataset/backend directory includes metadata, full chunked segment commitments, transcript, lookup proof, matrix sumcheck proof, GKR proof, Hyrax/MLE openings, masked openings, final proof payload, verifier output, witness manifest, benchmark summary, and run manifest.

`proof_size_bytes` is the canonical serialized `proof_payload.json` payload size, including verifier-facing proof messages. Witness manifests, raw witness materialization, datasets, checkpoints, logs, and audit metadata are not counted.
