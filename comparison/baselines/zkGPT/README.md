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
  - `proof.py`: official GAT-ZKML-zkGPT proof assembly, full witness segment chunking, Hyrax/MLE masked openings, fixed-table Lasso lookup proofs, strict matrix sumcheck proofs, ML-friendly GKR layered relation transcripts, combine payload, and independent verifier checks.
  - `field.py` / `native_hyrax.py`: BN254 field encoding and Hyrax-style row/column MLE OpenMasked backend.
  - `backend.py`: CPU/GPU backend selection; GPU runs execute the CUDA proof hot-path binding over proof-claim field words.
- `backend/cuda/`: local CUDA proof hot-path kernel used by GPU proof runs.
- `backend/native/`: native mcl BN254 Hyrax/MLE commitment and opening backend.
- `scripts/`: build and experiment entrypoints.
- `configs/generated/`: generated per-backend configs used by official runs.
- `results/`: current official rerun artifacts and summaries.
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

If runs were executed in batches, rebuild the unified CPU/GPU summary from the
existing artifact directories with:

```bash
./scripts/rebuild_summary_from_artifacts.py
```

The official result table is `results/summary.csv`. It contains five full CPU
runs and five full GPU runs: cora, citeseer, pubmed, ppi, and ogbn-arxiv.
The current rerun has 10 `VERIFY_OK` full-dataset proof bundles, with no
CPU fallback in GPU rows. The detailed CPU table, GPU table, CPU/GPU comparison,
proof-size accounting, and protocol alignment notes are in `docs/EXPERIMENT.md`.
Each dataset/backend directory contains:

- `metadata.json`
- `commitments.json`
- `transcript.json` / `challenges.json`
- `fixed_lookup_tables.json`
- `lookup_proof.json`
- `matrix_sumcheck_proof.json`
- `gkr_proof.json`
- `combine_proof.json`
- `hyrax_mle_openings.json`
- `masked_openings.json`
- `gpu_proof_hot_path.json` for GPU runs
- `proof_payload.json` / `proof_object.json`
- `verification.json` / `verifier_output.json`
- `witness_manifest.json`
- `benchmark_summary.json`
- `run_manifest.json`

`proof_size_bytes` is `len(canonical_json(proof_payload))`. Audit metadata and witness
materialization records are not counted. The payload includes the verifier-facing
proof messages, fixed lookup table commitments, and GPU proof contribution for
GPU runs, not just artifact references.

Run negative tests with:

```bash
./scripts/run_negative_tests.py
./scripts/audit_formal_proof_semantics.py
```

The negative suite writes `results/negative_tests/formal_negative_tests.json`.
The semantics audit writes `results/formal_proof_semantics_audit.json`.

Current validation status:

- `results/cpu_results.csv`, `results/gpu_results.csv`, `results/summary.csv`,
  and `results/summary.json` were regenerated from local artifacts.
- `results/negative_tests/formal_negative_tests.json`: `PASS`.
- `results/formal_proof_semantics_audit.json`: `PASS`.
- Lookup uses structured fixed-function VK commitments and materialized fixed
  range table commitments; matrix residuals are committed and range-bound; GKR
  final oracle claims are bound to committed masked openings; GPU proof runs
  include `GPU_proof_contribution` in the proof payload.
