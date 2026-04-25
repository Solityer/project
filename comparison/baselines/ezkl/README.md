# EZKL Full-Dataset Baseline

This directory contains the ezkl/Halo2/KZG comparison baseline for `GAT-ZKML-ezkl.md`.
The experiment replaces only the proving system. Dataset, checkpoint, model identity,
preprocessing, self-loop rule, edge sort rule, batching rule, chunking rule, full-graph
semantics, output semantics, and public metadata semantics stay aligned with the main
repository.

## Statement

The runnable routes are:

`ezkl_timing_private_input_public_output_full`

`ezkl_timing_private_input_digest_output_full`

For both routes the full dataset graph, features, checkpoint parameters, edge order,
self-loops, preprocessing metadata, and full forward logits are fixed artifacts bound by
metadata hashes.

`identity_public_logits` publishes every logit as an ezkl public output. It is used for
Cora, Citeseer, and Pubmed where the public instance count is small enough.

`field_folding` is used for PPI and ogbn-arxiv to avoid million-scale Halo2 public
instances. The full logits file is still generated and saved, but the ezkl public output
contains four deterministic folded values plus shape, metadata, and output-artifact hash
limbs. The circuit reads the full padded private logits witness and computes the folds.
This is not a cryptographic commitment; it is a random linear folding used only for
public-instance compression in the timing baseline.

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

For the large full datasets:

```bash
python3 comparison/baselines/ezkl/scripts/run_ezkl_baseline.py --dataset ppi_full_formal --mode full --binding-backend auto --clean --run
python3 comparison/baselines/ezkl/scripts/run_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --binding-backend auto --clean --run
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

The current status is read from `runs/<dataset>/run_manifest.json`.

| dataset | route | backend | status |
| --- | --- | --- | --- |
| `cora_full` | public output | `identity_public_logits` | verified |
| `citeseer_full` | public output | `identity_public_logits` | verified |
| `pubmed_full` | public output | `identity_public_logits` | verified |
| `ppi_full_formal` | digest output | `field_folding` | blocked at ezkl setup resource limit |
| `ogbn_arxiv_full` | digest output | `field_folding` | blocked at ezkl setup resource limit |

Regenerate the final tables by:

```bash
python3 comparison/baselines/ezkl/scripts/collect_results.py
```

The final tables are:

- `results/ezkl_full_results.json`
- `results/ezkl_full_results.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/main_vs_ezkl_full_comparison.csv`
