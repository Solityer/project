# GAT-ZKML-zkGPT Baseline

This directory is the independent zkGPT baseline project for the GAT-ZKML comparison.
It follows `/home/pzh/project/comparison/GAT-ZKML-zkGPT.md` for the GAT model semantics,
public metadata, transcript order, proof object layout, and verifier acceptance policy.

## Layout

- `src/zkgpt_gat/`: Python implementation modules.
  - `config.py`: config generation and parsing aligned with the main repository configs.
  - `data.py`: cached dataset loading, full-graph normalization, self-loop insertion, edge sorting, `node_ptr`/`edge_ptr`.
  - `model.py`: checkpoint bundle reader and multi-layer/multi-head GAT parameter adapter.
  - `forward.py`: sparse full-graph GAT witness generation.
  - `proof.py`: zkGPT-style commitments, Fiat-Shamir transcript, Lasso/MatMul/GKR proof blocks, masked opening records, verifier checks.
  - `backend.py`: CPU/GPU backend selection; GPU runs must execute the local CUDA helper.
- `backend/cuda/`: local CUDA helper used by GPU proof runs.
- `scripts/`: build and experiment entrypoints.
- `configs/generated/`: generated per-backend configs used by official runs.
- `results/`: official rerun artifacts and summaries.
- `archive/invalid_adapter_20260525/`: previous adapter-only artifacts, excluded from official results.
- `3rd/mcl/`: local copy of `/home/pzh/project/third_party/mcl`, not a symlink.

## Build

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target gat_zkgpt_check
cmake --build cmake-build-release --target zkgpt_cuda_backend
```

The original zkGPT LLM demo can be attempted with `-DZKGPT_BUILD_LEGACY_LLM_DEMO=ON`,
but it is not part of the GAT baseline path. The GAT baseline build does not call the
main repository `gatzk_run` binary.

## Run

```bash
./scripts/run_gat_zkgpt_all.sh --force
```

The official result table is `results/summary.csv`.
Each dataset/backend directory contains:

- `metadata.json`
- `commitments.json`
- `transcript.json` / `challenges.json`
- `proof_payload.json` / `proof_object.json`
- `verification.json` / `verifier_output.json`
- `witness_manifest.json`
- `benchmark_summary.json`
- `run_manifest.json`

`proof_size_bytes` is `len(canonical_json(proof_payload))`. Audit metadata and witness
materialization records are not counted.

