# zkGAT

**Zero-Knowledge Verifiable Inference for Graph Attention Networks**

zkGAT is a research implementation of zero-knowledge verifiable inference for multi-layer, multi-head Graph Attention Networks (GATs). It combines sparse-graph execution with polynomial-commitment-based proofs and independent verification. The paper evaluation uses the **CUDA-accelerated proving path** on complete graph datasets. Model training is used to prepare checkpoints; the proved computation is GAT inference.

## Datasets

The canonical models contain three GAT layers: two eight-head hidden layers and a two-head output layer.

| Dataset | CLI name | Task |
| --- | --- | --- |
| Cora | `cora` | Transductive node classification |
| CiteSeer | `citeseer` | Transductive node classification |
| PubMed | `pubmed` | Transductive node classification |
| PPI | `ppi` | Inductive, multi-label classification over 295 graphs |
| OGBN-Arxiv | `ogbn_arxiv` | Transductive node classification |

All main evaluations use the complete corresponding dataset.

## Requirements and Build

A Linux system with an NVIDIA GPU, a compatible CUDA toolkit, CMake (3.24+ for native CUDA architecture detection), a C++20 compiler, GNU Make, Python 3, OpenSSL development files, and the LLVM toolchain required by the bundled MCL build. The CUDA build uses the pinned ICICLE Git submodule.

```bash
git clone --recurse-submodules https://github.com/Solityer/project.git
cd project
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda --parallel 2 --target gatzk_run gatzk_mechanism gatzk_srs_setup
```

## Data and Cryptographic Setup

The raw datasets are under `data/`. Prepare their canonical caches with Python (NumPy and SciPy are required):

```bash
python3 scripts/prepare_planetoid.py --data-root data --dataset cora --cache-root data/cache
python3 scripts/prepare_planetoid.py --data-root data --dataset citeseer --cache-root data/cache
python3 scripts/prepare_planetoid.py --data-root data --dataset pubmed --cache-root data/cache
python3 scripts/prepare_ppi.py --data-root data/ppi --output-root data/cache/ppi
python3 scripts/prepare_ogbn_arxiv.py --project-root . --data-root data
```

Each formal run requires a matching **trained checkpoint bundle** (`artifacts/checkpoints/<dataset>_gat/`) and **trusted SRS** (`artifacts/srs/<dataset>_formal_srs/`). These generated materials are not part of the source checkout. Use the training, checkpoint-export, and bundle-validation tools in `scripts/` to prepare model parameters. Use `build_cuda/gatzk_srs_setup` (`compile`, `ceremony`, `validate`) with the corresponding SRS envelope in `configs/` to prepare and validate the SRS. Preserve the configuration, checkpoint, and SRS bindings when reproducing results.

## Quick Start

Run a complete CUDA inference proof from the repository root after preparing its inputs:

```bash
build_cuda/gatzk_run cora
```

The CUDA-enabled executable selects `cuda_hotspots` by default. A successful run prints `status=VERIFY_OK`; its unique output directory contains `proof.bin` and `run_manifest.json` under `runs/<dataset>_secure/<tag>/single/`. CUDA accelerates selected proving operations; other protocol operations may execute on the host.

## Reproducing the Main Evaluation

Run all five datasets sequentially on the CUDA path:

```bash
bash scripts/run_paper_matrix.sh
```

To run just one dataset, supply its CLI name:

```bash
bash scripts/run_paper_matrix.sh ogbn_arxiv
```

For an SSH-disconnect-safe run, launch the same script in a detached session:

```bash
mkdir -p runs && nohup setsid bash scripts/run_paper_matrix.sh > runs/gpu_paper_launcher.log 2>&1 < /dev/null &
```

Alternatively, invoke individual datasets directly:

```bash
build_cuda/gatzk_run cora
build_cuda/gatzk_run citeseer
build_cuda/gatzk_run pubmed
build_cuda/gatzk_run ppi
build_cuda/gatzk_run ogbn_arxiv
```

The runner preserves a separate `gpu_<dataset>.log` for each execution, a `status.tsv` summary, resource logs (when GNU `time` is available), and a `LATEST_GPU` pointer under `runs/full_dataset_logs/`. The main evaluation reports four metrics: **commitment time**, **prove time**, **verify time**, and **proof size** (`commitment_time_ms`, `prove_time_ms`, `verify_time_ms`, `proof_size_bytes`). An accepted proof and a matching `VERIFY_OK` record are required; final security claims additionally depend on the corresponding artifact and tamper-test checks.

## Mechanism Ablation

Two real-data CUDA mechanism experiments study the proving-cost reductions:

- **Feature Folding:** folded feature relations versus coordinate-wise proofs.
- **Shared Neighborhood:** shared-neighborhood relations versus separate denominator and aggregation proofs.

The complete matrix contains 47 configurations: Feature Folding has five datasets at D=1..8 and G=1 (40 groups); Shared Neighborhood has five datasets at D=8 and G=1 (5 groups); the appendix adds both mechanisms on all 295 PPI graphs at D=8 (2 groups). With three repetitions and Original/Control, the complete matrix has 141 pairs / 282 cells. The commands below split it into three independent runs. Here `main` means the mechanism ablation main-table inputs, not the zkGAT main experiment. Omitting selectors still runs all 47 groups. These are mechanism-level proof comparisons, not end-to-end GAT speedup measurements.

Run the bounded metadata-only preflight for each selected group before launching it. Preflight checks only the selected datasets and SRS; it does not prepare real proof inputs, rehash key contents, or run Prove/Verify. Launch one command at a time; each run writes to its own new directory:

```bash
python3 scripts/run_mechanism_ablation.py preflight --device 0 --mechanism feature_folding --input-scope main
python3 scripts/run_mechanism_ablation.py launch --device 0 --repetitions 3 \
  --mechanism feature_folding --input-scope main \
  --output-root results/mechanism_ablation_final/ff

python3 scripts/run_mechanism_ablation.py preflight --device 0 --mechanism shared_neighborhood --input-scope main
python3 scripts/run_mechanism_ablation.py launch --device 0 --repetitions 3 \
  --mechanism shared_neighborhood --input-scope main \
  --output-root results/mechanism_ablation_final/sn

python3 scripts/run_mechanism_ablation.py preflight --device 0 --mechanism all --input-scope ppi_full
python3 scripts/run_mechanism_ablation.py launch --device 0 --repetitions 3 \
  --mechanism all --input-scope ppi_full \
  --output-root results/mechanism_ablation_final/ppi_full
```

The runner creates a new session directory and runs detached. Take `launcher_dir` from the launch output, wait for `MECHANISM_SESSION_STARTED ... output=...` in its `runner.log`, then use that output path as `<session>` for `cat <session>/status.json` and `tail -f <session>/session.log`. Each new session's two reader outputs are Chinese `summary.md` and UTF-8-BOM `results.csv`; raw logs, proofs, statements, receipts, and audit evidence remain alongside them. See [the mechanism ablation instructions](docs/mechanism_ablation.md). The older 11-configuration session is historical only and is not the current matrix.

## Repository Structure

| Path | Contents |
| --- | --- |
| `src/`, `include/` | GAT inference, commitments, prover, and verifier |
| `configs/` | Dataset, proof, and SRS configurations |
| `scripts/` | Data preparation, checkpoint tooling, and experiment runners |
| `comparison/` | Adapted zkGPT, ezkl, and zkDL baselines |
| `tests/` | Execution, proof, and mechanism regression tests |
| [`docs/mechanism_ablation.md`](docs/mechanism_ablation.md) | Mechanism experiments and their verification status |
| [`docs/security/`](docs/security/) | Threat model and security documentation |

## Citation and License

Citation information will be provided with the paper. No repository-wide license is currently declared; third-party components retain their respective licenses.
