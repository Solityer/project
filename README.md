# GAT-ZKML Planetoid Mainline

`project/` is the only formal project root. `GAT-ZKML-单层多头.md` is immutable and remains the source of truth for protocol objects, transcript order, work domains, proof block order, and formal semantics.

## Current Status

- Full Cora has a complete real-checkpoint formal run and prints `VERIFY_OK`.
- Full Citeseer now also has a real project-local training checkpoint, exported bundle, `manifest.json`, exact full-graph parity, and a formal warm benchmark.
- Full Pubmed now has a real project-local training checkpoint, exported bundle, `manifest.json`, exact full-graph parity, and a formal warm benchmark.
- All formal checkpoint assets now live under:
  - `artifacts/checkpoints/cora_gat/`
  - `artifacts/checkpoints/citeseer_gat/`
  - `artifacts/checkpoints/pubmed_gat/`
- The formal mainline remains:
  - single-layer GAT
  - hidden heads = 8
  - output heads = 1
  - hidden ELU
  - output identity
  - no-bias output attention in the formal path
  - explicit self-loops
  - stable nondecreasing `dst` edge order

## Local Reference Sources Copied Into `project/`

These files were copied from sibling local repositories so that the final project no longer depends on code outside `project/`.

From local `GAT/`, copied into `reference/gat_official/`:

- `reference/gat_official/execute_cora.py`
- `reference/gat_official/models/gat.py`
- `reference/gat_official/utils/layers.py`

Use:

- local provenance for the original Cora-style forward semantics
- reference for tensor naming/layout compatibility

From local `GAT-for-PPI/`, copied into `reference/gat_for_ppi/`:

- `reference/gat_for_ppi/execute.py`
- `reference/gat_for_ppi/execute_sparse.py`
- `reference/gat_for_ppi/models/base_gattn.py`
- `reference/gat_for_ppi/models/gat.py`
- `reference/gat_for_ppi/models/sp_gat.py`
- `reference/gat_for_ppi/utils/layers.py`
- `reference/gat_for_ppi/utils/process.py`

Use:

- local provenance for sparse full-graph transductive training flow on Planetoid-style datasets
- reference for Citeseer / Pubmed data handling and sparse attention execution

The final runtime/training/proving path does **not** import from sibling `GAT/` or `GAT-for-PPI/` directories.

## Chosen Training Mainline

The formal training/export path used for Citeseer and Pubmed is fully project-local:

- training: `scripts/train_planetoid_gat.py`
- bundle export: `scripts/export_torch_gat_bundle.py`
- parity: `scripts/check_planetoid_parity.py`

Why this mainline is used:

- it stays inside `project/`
- it is sparse full-graph and scales better to Citeseer / Pubmed than the dense TensorFlow reference path
- it is aligned with the formal note semantics instead of reintroducing an output affine+bias classifier

Formal-consistency checks for this training mainline:

- hidden heads = 8
- output heads = 1
- hidden ELU
- output identity
- no extra output bias in the actual attention aggregation semantics
- self-loops inserted before sorted edge-index materialization
- full-graph inference only
- project-local sorted edge order and feature order reused by parity and proving

Note on exported bias tensors:

- `export_torch_gat_bundle.py` emits zero-valued `BiasAdd*` tensors for loader/bundle compatibility
- those tensors do **not** reintroduce a learned extra output bias into the formal mainline

## Code Map

- Forward path: `src/model/gat.cpp`
- Formal trace construction: `src/protocol/trace.cpp`
- Transcript labels and ordering: `src/protocol/challenges.cpp`
- Prover entry: `src/protocol/prover.cpp`
- Verifier entry: `src/protocol/verifier.cpp`
- Quotient helpers: `src/protocol/quotients.cpp`
- Proof / metadata / work domains: `include/gatzk/protocol/proof.hpp`
- Main regression tests: `tests/test_main.cpp`

## Formal Objects

- Hidden heads: `P_h{r}_H_prime`, `P_h{r}_E_src`, `P_h{r}_E_dst`, `P_h{r}_H_star`, `P_h{r}_S`, `P_h{r}_Z`, `P_h{r}_M`, `P_h{r}_Delta`, `P_h{r}_U`, `P_h{r}_Sum`, `P_h{r}_inv`, `P_h{r}_alpha`, `P_h{r}_H_agg_pre`, `P_h{r}_H_agg`, `P_h{r}_H_agg_star`, `P_h{r}_PSQ`
- Concat stage: `P_H_cat`, `P_H_cat_star`, `P_cat_a`, `P_cat_b`, `P_cat_Acc`
- Output stage: `P_out_Y_prime`, `P_out_E_src`, `P_out_E_dst`, `P_out_Y_prime_star`, `P_out_Y_prime_star_edge`, `P_out_widehat_y_star`, `P_out_PSQ`, `P_Y`, `P_out_Y_star`, `P_out_Y_star_edge`
- Fixed proof block order: `M_pub`, `Com_dyn`, `S_route`, `Eval_ext`, `Eval_dom`, `Com_quot`, `Open_dom`, `W_ext`, `Pi_bind`

## Work Domains

- `H_FH`
- `H_edge`
- `H_in`
- `H_d_h`
- `H_cat`
- `H_C`
- `H_N`

`H_C` is distinct and must not reuse `H_d_h`.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Data Preparation

Prepare cached full-dataset artifacts inside `project/data/cache/`:

```bash
python3 scripts/prepare_planetoid.py --data-root data --dataset cora --cache-root data/cache
python3 scripts/prepare_planetoid.py --data-root data --dataset citeseer --cache-root data/cache
python3 scripts/prepare_planetoid.py --data-root data --dataset pubmed --cache-root data/cache
```

## Checkpoint Training, Export, And Manifest Generation

### Cora

The existing real Cora checkpoint bundle is already under:

- `artifacts/checkpoints/cora_gat/manifest.json`
- `artifacts/checkpoints/cora_gat/tensors.npz`
- `artifacts/checkpoints/cora_gat/tensors.txt`

### Citeseer

Train:

```bash
python3 scripts/train_planetoid_gat.py --config configs/citeseer_train.cfg
```

Export bundle + manifest:

```bash
python3 scripts/export_torch_gat_bundle.py \
  --checkpoint artifacts/checkpoints/citeseer_gat/best_model.pt \
  --output-dir artifacts/checkpoints/citeseer_gat
```

### Pubmed

Train:

```bash
python3 scripts/train_planetoid_gat.py --config configs/pubmed_train.cfg
```

Export bundle + manifest:

```bash
python3 scripts/export_torch_gat_bundle.py \
  --checkpoint artifacts/checkpoints/pubmed_gat/best_model.pt \
  --output-dir artifacts/checkpoints/pubmed_gat
```

## Full-Graph Parity

Citeseer:

```bash
python3 scripts/check_planetoid_parity.py \
  --checkpoint artifacts/checkpoints/citeseer_gat/best_model.pt \
  --bundle-dir artifacts/checkpoints/citeseer_gat \
  --data-root data \
  --dataset citeseer \
  --output runs/citeseer_full/parity/summary.json
```

Pubmed:

```bash
python3 scripts/check_planetoid_parity.py \
  --checkpoint artifacts/checkpoints/pubmed_gat/best_model.pt \
  --bundle-dir artifacts/checkpoints/pubmed_gat \
  --data-root data \
  --dataset pubmed \
  --output runs/pubmed_full/parity/summary.json
```

Current parity results:

- Citeseer: passed, `107/107`, `max_abs_error = 0.0`
- Pubmed: passed, `107/107`, `max_abs_error = 0.0`

## Formal Benchmark Commands

Cora:

```bash
./build/gatzk_run --config configs/cora_full.cfg --benchmark-mode both --export-tag cora_paper
```

Citeseer:

```bash
./build/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode both --export-tag citeseer_paper
```

Pubmed:

```bash
./build/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm --export-tag pubmed_paper
```

For Pubmed the current paper number is taken from a formal warm run because the full dataset is much heavier; the command still goes through the formal entry path and prints `VERIFY_OK`.

## Paper-Style Results

### Cora

- proving time: `7229.627 ms`
- verification time: `1270.231 ms`
- proof size: `37203 bytes`

### Citeseer

- proving time: `18913.604 ms`
- verification time: `4066.093 ms`
- proof size: `37207 bytes`

### Pubmed

- proving time: `36049.812 ms`
- verification time: `3911.717 ms`
- proof size: `37207 bytes`

## Current Bottlenecks

Cora warm bottlenecks:

- `trace_generation_ms = 2405.358`
- `quotient_build_ms = 1499.974`
- `commit_dynamic_ms = 1262.450`

Citeseer warm bottlenecks:

- `trace_generation_ms = 6285.418`
- `quotient_build_ms = 5368.912`
- `commit_dynamic_ms = 2749.166`

Pubmed warm bottlenecks:

- `trace_generation_ms = 12353.890`
- `quotient_build_ms = 8228.090`
- `commit_dynamic_ms = 5827.294`

These are the same kinds of scale-sensitive costs that matter for larger Planetoid graphs:

- repeated per-edge and per-node trace materialization
- domain conversion in dynamic commitments
- edge-domain opening and quotient assembly

That is why the recent optimizations have focused on reusable caches, same-domain batching, and removal of repeated `O(N)`, `O(E)`, and `O(N * d_in)` helper construction rather than Cora-only shortcuts.

## Proof Size Status

- current proof size is `37203 bytes` on Cora and `37207 bytes` on Citeseer / Pubmed
- no protocol-safe size reduction has been merged yet
- remaining size work is structural bundle compaction, not dropping formal objects, openings, or checks

## Tests

Run the local regression set:

```bash
./build/gatzk_tests
```

Current status:

- `25/25` passing

Key checks include:

- full-Cora formal prove / verify round-trip
- tampered witness rejection
- metadata mismatch rejection
- proof block order rejection
- wrong `H_C` reuse rejection
- `H_cat_star` tamper rejection
- `Y_star` tamper rejection
- `PSQ_out` tamper rejection

## Next Step

The current project bottleneck is no longer missing Citeseer / Pubmed assets. The next honest optimization targets are:

- further reducing prove time on larger graphs, especially Pubmed
- shrinking `trace_generation_ms`, `quotient_build_ms`, and `commit_dynamic_ms`
- only after that, revisiting verifier-side fixed overheads and proof-size compaction
