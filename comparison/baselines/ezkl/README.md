# EZKL GAT Formal Baseline

This repository is the `ezkl`-based baseline for the GAT-ZKML comparison experiment. The active baseline path is:

1. read the repo-local formal dataset cache and checkpoint bundle copies;
2. run the C++ stage tool that reproduces the formal preprocessing semantics and reference forward from `GAT-ZKML-ezkl.md`;
3. build a fixed-shape ONNX graph from the staged graph tensors and staged checkpoint tensors;
4. run the `ezkl` pipeline on that ONNX graph inside this repository only.

## Active Proof Route

The active baseline uses `ezkl_timing_private_input_public_output_full`:

- `proof_route_id = ezkl_timing_private_input_public_output_full`
- `visibility_id = private_public_fixed`
- `input_visibility = private`, `output_visibility = public`, `param_visibility = fixed`

This route is the sole runnable path for all five full-scale datasets. The strict public-input route (`ezkl_strict_public_input_full`, `input_visibility=public`) was attempted first on all five datasets and was found infeasible at the full formal scale under Halo2/KZG constraints:

| Dataset | Strict Route Failure Reason |
|---|---|
| cora_full | setup OOM (signal 9) at k=25 after ~37 min; peak RSS 123 GB |
| citeseer_full | Halo2 domain bound exceeded at k=25: all row-fitting layouts have quotient_poly_degree > BN254 S=28 |
| pubmed_full | row capacity exceeded: num_inner_cols=32 requires row_min_logrows=27 at k=25 |
| ppi_full_formal | gen-settings did not complete after 17 min; high-CPU timeout before row-bound result |
| ogbn_arxiv_full | gen-settings did not complete after 10 min; high-CPU timeout before row-bound result |

The timing route keeps the same model, checkpoint, graph rules, and benchmark口径 as the main repo. Only the ezkl visibility interface is adapted: inputs become private witness rather than public instances. This does not change the GAT forward computation, quantization semantics, or output binding. Per `GAT-ZKML-ezkl.md` §0.9.1–§0.9.3, timing route results must not be interpreted as strict public-input route success.

## Scope

The baseline keeps only the formal full-dataset comparison line:

- `cora`
- `citeseer`
- `pubmed`
- `ppi`
- `ogbn_arxiv`

## Repository Layout

- `GAT-ZKML-ezkl.md`: formal replacement note and highest-priority design reference.
- `manifests/gat_formal_datasets.json`: dataset/checkpoint/reference manifest copied into the baseline repo.
- `cpp/gat_baseline/`: C++ stage tool for full-dataset preprocessing, checkpoint parsing, and reference forward computation.
- `scripts/gat_baseline/`: thin Python/Rust glue for staged-tensor ONNX construction and `ezkl` pipeline orchestration.
- `src/bin/gat_baseline.rs`: Rust CLI that drives export and the `ezkl` compile/witness/prove pipeline.
- `runs/<dataset>/`: generated artifacts for the formal baseline run of one dataset.

## Formal Semantics

The exporter follows the main comparison semantics:

- row-normalized cached features
- full-graph or full multi-graph batch preprocessing
- deterministic `dst`-stable edge ordering
- explicit self-loops
- hidden-layer multi-head concat
- output-layer per-head bias then arithmetic mean

For `ezkl` practicality, the ONNX graph uses precomputed incoming-neighbor tables per frozen formal graph. The public graph tensors `src`, `dst`, `node_ptr`, and `edge_ptr` are still inputs, and the graph emits a `graph_checks` output that must remain zero, binding those public tensors to the frozen formal graph layout used by the circuit.

## Export Only

Use the Python exporter directly when you only need the formal ONNX/input bundle:

```bash
PYTHONPATH=/home/pzh/project/comparison/baselines/ezkl/.pydeps:. \
python3 -m scripts.gat_baseline.cli \
  --dataset cora
```

This writes, under `runs/<dataset>_full.../`:

- `staged/` with the C++-materialized graph tensors and checkpoint tensors
- `network.onnx`
- `input.json`
- `model_info.json`
- `validation.json`

`validation.json` compares the C++ staged reference forward with onnxruntime on the exported graph.

## Full Pipeline

The Rust CLI wraps the formal baseline export and the downstream `ezkl` steps:

```bash
export RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup_local
export CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local
export PATH=/home/pzh/project/comparison/baselines/ezkl/.cargo_local/bin:$PATH

cargo run --bin gat_baseline -- pipeline \
  --dataset cora \
  --prove
```

With `--prove`, the CLI continues through:

- `gen-srs` when the exact local SRS is not already present
- `setup` with the compiled circuit, witness, and exact local SRS
- `prove`
- `verify`

Without `--prove`, it stops after:

- `gen-settings`
- `compile-circuit`
- `gen-witness`

Each step writes its own log into the dataset run directory, and `pipeline_summary.json` collects:

- exported model metadata
- the single formal proving layout selected by the wrapper
- onnxruntime validation numbers
- step durations
- reference timing fields read from the repo-local copied comparison manifest

The active formal proving rule is:

`prove_logrows = 25` when `public_output.minimum_logrows <= 25`. For the timing route, `minimum_public_logrows` is derived from public output count only (inputs are private witness). The wrapper first probes `num_inner_cols = 32` to fast-reject datasets that still exceed `k=25` at the largest allowed inner-column layout; otherwise it chooses the smallest `num_inner_cols` in `[2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]` whose row bound fits `prove_logrows` and whose Halo2 extended-k/domain bound is feasible.

This keeps the public output semantics unchanged while moving only the proving-layout adaptation into the `ezkl`-specific layer. If no candidate `num_inner_cols` fits `k=25`, the dataset is reported as blocked rather than silently falling back.

Blocked runs also write both `settings_selection.json` and `pipeline_summary.json`, so the formal run directory records the failure reason without opening a second route.

The wrapper also fixes:

- `EZKL_REPO_PATH=/home/pzh/project/comparison/baselines/ezkl/.ezkl_local`
- baseline-local `.ezkl_local` as the default cache root for the local `ezkl` binary
- baseline-local exact-k SRS generation/reuse
- `private/public/fixed` visibility (timing route)
- no `mock` step on the formal proving path

## Notes

- The baseline reads datasets, checkpoint bundles, configs, and reference run manifests from repo-local copies under this directory.
- All generated artifacts stay under this baseline repository.
- Large datasets produce large fixed-shape ONNX graphs. That is expected for the full-dataset comparison path.
- Timing route results (`proof_route_id=ezkl_timing_private_input_public_output_full`) must not be interpreted as strict public-input route success. See `GAT-ZKML-ezkl.md` §0.9.3.
