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
  - `proof.py`: invalid legacy digest/replay adapter retained for audit only; the official CLI refuses to run it by default.
  - `field.py` / `native_hyrax.py`: BN254 field encoding and wrapper for the native mcl Hyrax/MLE backend.
  - `backend.py`: CPU/GPU backend selection; GPU runs must execute the local CUDA helper.
- `backend/cuda/`: local CUDA helper used by GPU proof runs.
- `backend/native/`: native mcl BN254 Hyrax/MLE commitment and opening backend.
- `scripts/`: build and experiment entrypoints.
- `configs/generated/`: generated per-backend configs used by official runs.
- `results/`: reserved for future formal rerun artifacts and summaries; currently contains no official VERIFY_OK rows.
- `archive/invalid_adapter_20260525/`: previous adapter-only artifacts, excluded from official results.
- `archive/invalid_digest_replay_20260529/`: digest/replay artifacts removed from `results/`.
- `3rd/mcl/`: local copy of `/home/pzh/project/third_party/mcl`, not a symlink.

## Build

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release --target gat_zkgpt_check
cmake --build cmake-build-release --target zkgpt_hyrax_mle
cmake --build cmake-build-release --target zkgpt_cuda_backend
```

The original zkGPT LLM demo can be attempted with `-DZKGPT_BUILD_LEGACY_LLM_DEMO=ON`,
but it is not part of the GAT baseline path. The GAT baseline build does not call the
main repository `gatzk_run` binary.

## Run

```bash
./scripts/run_gat_zkgpt_all.sh --force
```

The legacy Python digest/replay runner is disabled for official runs. It can no
longer create `results/summary.csv` or `VERIFY_OK` rows unless the hidden
audit-only flag is passed. Do not use that flag for formal reporting.

When the formal zkGPT proof backend is complete, the official result table will be
`results/summary.csv`.
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
