# EZKL Full-Dataset Baseline

This directory contains the ezkl/Halo2/KZG comparison baseline for `GAT-ZKML-ezkl.md`.
The experiment replaces only the proving system. Dataset, checkpoint, model identity,
preprocessing, self-loop rule, edge sort rule, batching rule, chunking rule, full-graph
semantics, output semantics, and public metadata semantics stay aligned with the main
repository.

## Statement

The runnable route is:

`ezkl_timing_private_input_public_output_full`

For this route the full dataset graph, features, checkpoint parameters, edge order,
self-loops, preprocessing metadata, and full forward logits are fixed artifacts bound by
metadata hashes. The ezkl circuit proves that the public logits equal the private full
forward logits binding witness for that fixed full-dataset artifact.

This is a timing route. It is not reported as `ezkl_strict_public_input_full` success.

## Layout

- `configs/`: full dataset specs and allowed ezkl routes.
- `cpp/`: staging code that reproduces the full dataset preprocessing and reference forward.
- `scripts/run_ezkl_baseline.py`: only recommended runner.
- `scripts/export_full_dataset.py`: full dataset exporter used by the runner.
- `scripts/collect_results.py`: only result collector.
- `reference_runs/`: main-scheme warm run manifests.
- `runs/`: latest formal ezkl run artifacts.
- `results/`: paper-facing result tables.

## Run

```bash
python3 comparison/baselines/ezkl/scripts/run_ezkl_baseline.py --dataset cora_full --mode full --clean --run
```

Resume an interrupted run:

```bash
python3 comparison/baselines/ezkl/scripts/run_ezkl_baseline.py --dataset cora_full --mode full --resume --run
```

Collect tables:

```bash
python3 comparison/baselines/ezkl/scripts/collect_results.py
```

## Metrics

The comparison table reports:

- `commitment_time_ms`
- `prove_time_ms`
- `verify_time_ms`
- `proof_size_bytes`

ezkl does not expose the main scheme's internal commitment/MSM timer here, so
`ezkl_commitment_time_ms = null` and
`ezkl_commitment_time_semantics = included_in_ezkl_prove_ms_unless_instrumented`.
`setup_ms` is reported separately and is never used as commitment time.

## Full-Dataset Constraint

Only full formal datasets are supported:

- `cora_full`
- `citeseer_full`
- `pubmed_full`
- `ppi_full_formal`
- `ogbn_arxiv_full`

There are no smoke, toy, subset, subgraph, scaled, shard, or debug result routes.

## Current Status

The current status is read from `runs/<dataset>/run_manifest.json` by:

```bash
python3 comparison/baselines/ezkl/scripts/collect_results.py
```

The final tables are:

- `results/ezkl_full_results.json`
- `results/ezkl_full_results.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/main_vs_ezkl_full_comparison.csv`
