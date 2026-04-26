# GAT-ZKML ezkl Baseline Implementation Plan

## Scope And Guardrails

- Work root: `/home/pzh/project/comparison/baselines/ezkl`.
- Main repository inputs are read-only references. No generated build, cache, run, or result file may be written outside this directory.
- Formal semantics are governed by `/home/pzh/project/GAT-ZKML-多层多头.md` and `/home/pzh/project/comparison/baselines/GAT-ZKML-ezkl.md`.
- Only full datasets are in scope: `cora_full`, `citeseer_full`, `pubmed_full`, `ppi_full_formal`, `ogbn_arxiv_full`.
- Forbidden substitutes: toy, smoke, mini, sample, subset, subgraph, scaled, shard, partial, debug, debug-small.
- Python is limited to orchestration and result collection. Dataset loading, graph normalization, checkpoint validation, GAT semantics, quantization metadata, hashes, and export decisions are implemented in C++.

## Required Semantic Alignment

- Full graph, no sampling or sharding.
- Explicit `node_ptr` and `edge_ptr` for every run, including single-graph datasets.
- Per-node self-loop insertion before proof input construction.
- Edge order: graph block first, then stable destination-node order inside each graph block.
- PPI must keep all configured graph blocks; silent `batch_graphs` clipping is forbidden.
- Hidden layers use multi-head attention, per-head ELU, and concat.
- Output layer uses attention heads, per-head output bias, and `K_out` arithmetic average.
- Public metadata binds dataset, graph preprocessing, model architecture, checkpoint parameter id, quantization id, shape, hashes, visibility, and route id.

## Route Policy

- `strict_gat_full_onnx_v1` is the only route that may set `strict_public_input_equivalent=true`.
- It requires the full GAT forward path to be represented by ezkl/Halo2 without replacing grouped graph attention, output logits, or visibility semantics.
- If strict export cannot be completed for a full dataset without changing semantics or exceeding the backend resource envelope, the run is recorded as `status=blocked` with `strict_public_input_equivalent=false`.
- Timing or output-binding routes are not allowed to masquerade as strict route success. This implementation does not use precomputed logits as an identity, reshape, digest, folding, or output-binding proof.

## Execution Plan

1. Build C++ adapter with:
   `cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release`
   `cmake --build build_gat_adapter -j`
2. Run one dataset first:
   `python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --clean --run`
3. Run the remaining official datasets in order.
4. Collect results with:
   `python3 scripts/collect_gat_ezkl_results.py`
5. Verify that every dirty path is under `comparison/baselines/ezkl/`.
