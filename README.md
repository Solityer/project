# GAT-ZKML Full-Cora Mainline

`project/` is the only formal project root. `GAT-ZKML-单层多头.md` is the immutable implementation note and remains the source of truth for protocol objects, transcript order, work domains, and proof block order.

## Current Status

- Full Cora is the only formal dataset and benchmark target.
- The real checkpoint bundle under `artifacts/checkpoints/cora_gat/` is used by the formal `cora_full` path.
- The official GAT forward semantics are aligned locally against `reference/gat_official/` and `runs/cora_full/reference/`.
- `gatzk_run --config configs/cora_full.cfg` currently completes and prints `VERIFY_OK`.
- The formal multi-head mainline now commits non-placeholder `t_FH / t_edge / t_in / t_d_h / t_cat / t_C / t_N`, opens seven work domains, and replays Fiat-Shamir directly from proof commitments instead of rebuilding an expected trace inside the verifier.
- The formal note-style forward path now removes extra output bias from the formal mainline: hidden heads use `ELU(H_agg_pre)` and the output head uses identity on the attention aggregation result.
- The repository root keeps only `README.md` plus the immutable note `GAT-ZKML-单层多头.md`.
- The current system is still not note-complete enough to call the implementation final: the quotient layer only constrains the currently materialized multi-head objects, and the full note-level lookup/state-machine families are not all materialized yet.

## Architecture

- Single-layer GAT
- Hidden heads: `8`
- Output heads: `1`
- Hidden activation: `ELU`
- Hidden concat width: `d_cat = 8 * d_h = 64`
- Output activation: identity
- Self-loops: explicit and included in attention aggregation
- Edge order: stable nondecreasing `dst`

## Code Map

- Forward path: `src/model/gat.cpp`
- Formal trace construction: `src/protocol/trace.cpp`
- Transcript labels and ordering: `src/protocol/challenges.cpp`
- Prover entry: `src/protocol/prover.cpp`
- Verifier entry: `src/protocol/verifier.cpp`
- Quotient helpers: `src/protocol/quotients.cpp`
- Proof / metadata / work domains: `include/gatzk/protocol/proof.hpp`
- Main regression tests: `tests/test_main.cpp`
- Local official reference copy: `reference/gat_official/`

## Formal Objects

- Hidden heads: `P_h{r}_H_prime`, `P_h{r}_E_src`, `P_h{r}_E_dst`, `P_h{r}_H_star`, `P_h{r}_S`, `P_h{r}_Z`, `P_h{r}_M`, `P_h{r}_Delta`, `P_h{r}_U`, `P_h{r}_Sum`, `P_h{r}_inv`, `P_h{r}_alpha`, `P_h{r}_H_agg_pre`, `P_h{r}_H_agg`, `P_h{r}_H_agg_star`, `P_h{r}_PSQ`
- Concat stage: `P_H_cat`, `P_H_cat_star`, `P_cat_a`, `P_cat_b`, `P_cat_Acc`
- Output stage: `P_out_Y_prime`, `P_out_E_src`, `P_out_E_dst`, `P_out_Y_prime_star`, `P_out_Y_prime_star_edge`, `P_out_widehat_y_star`, `P_out_PSQ`, `P_Y`, `P_out_Y_star`, `P_out_Y_star_edge`
- Public metadata: `protocol_id`, `model_arch_id`, `model_param_id`, `static_table_id`, `quant_cfg_id`, `domain_cfg`, `dim_cfg`, `encoding_id`, `padding_rule_id`, `degree_bound_id`
- Fixed proof block order: `M_pub`, `Com_dyn`, `S_route`, `Eval_ext`, `Eval_dom`, `Com_quot`, `Open_dom`, `W_ext`, `Pi_bind`

## Work Domains

- `H_FH`
- `H_edge`
- `H_in`
- `H_d_h`
- `H_cat`
- `H_C`
- `H_N`

`H_C` is a distinct work domain and must not reuse `H_d_h`.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Data And Reference

Prepare full Cora locally:

```bash
python3 scripts/prepare_planetoid.py --data-root data --dataset cora --cache-root data/cache
```

Export the real checkpoint bundle into `project/`:

```bash
python3 scripts/export_gat_checkpoint.py \
  --checkpoint-prefix ../GAT/pre_trained/cora/mod_cora.ckpt \
  --output-dir artifacts/checkpoints/cora_gat
```

Rebuild project-local reference artifacts:

```bash
python3 scripts/gat_reference.py \
  --checkpoint-dir artifacts/checkpoints/cora_gat \
  --data-root data \
  --dataset cora \
  --output-dir runs/cora_full/reference
```

## Formal Run

```bash
./build/gatzk_run --config configs/cora_full.cfg
```

Expected terminal tail:

```text
[gatzk] proving
[gatzk] verifying
VERIFY_OK
```

## Tests

Run the full local regression set:

```bash
./build/gatzk_tests
```

Key full-Cora checks:

- forward parity against local reference artifacts
- formal no-bias output-head path
- real checkpoint bundle loading
- `H_cat / H_C` formal context materialization
- full-Cora prove / verify round-trip
- tamper rejection for witness, `H_cat_star`, `Y_star`, `PSQ_out`
- metadata mismatch rejection
- proof block order rejection
- `H_C` wrong-domain reuse rejection

## Benchmark

Measured from the formal entry command on the local Linux server, using:

```bash
./build/gatzk_run --config configs/cora_full.cfg --benchmark-mode both
```

Current build flags:

- CPU-only formal path
- `enabled_fast_msm=true`
- `enabled_parallel_fft=true`
- `enabled_fft_backend_upgrade=true`
- `enabled_fft_kernel_upgrade=true`
- `enabled_trace_layout_upgrade=true`
- `enabled_fast_verify_pairing=true`

Cold path:

- `cold_prove_ms = 41271.571`
- `cold_verify_ms = 5255.226`
- `forward_ms = 891.912`
- `trace_generation_ms = 39329.586`
- `quotient_build_ms = 16756.853`
- `domain_opening_ms = 24511.542`
- `external_opening_ms = 2.603`
- `proof_size_bytes = 25247`

Warm path:

- `warm_prove_ms = 3115.876`
- `warm_verify_ms = 5249.247`
- `forward_ms = 872.323`
- `trace_generation_ms = 39244.068`
- `quotient_build_ms = 1585.404`
- `domain_opening_ms = 1527.269`
- `external_opening_ms = 2.602`
- `proof_size_bytes = 25247`

Hot-path breakdown from the latest warm run:

- `hidden_forward_projection_ms = 829.038`
- `hidden_forward_attention_ms = 26.375`
- `hidden_concat_ms = 0.825`
- `output_forward_projection_ms = 4.260`
- `output_forward_attention_ms = 2.907`
- `lookup_trace_ms = 12210.422`
- `zkmap_trace_ms = 10073.877`
- `witness_materialization_ms = 384.387`
- `route_trace_ms = 319.801`
- `psq_trace_ms = 235.000`
- `quotient_t_fh_ms = 837.330`
- `quotient_t_edge_ms = 543.309`
- `quotient_t_N_ms = 243.888`
- `domain_open_FH_ms = 1526.481`
- `domain_open_edge_ms = 616.197`
- `domain_open_N_ms = 323.798`
- `verify_quotient_ms = 5037.195`
- `verify_FH_ms = 4963.187`

Warm-path optimization that materially changed runtime:

- proof-scoped domain barycentric weights are now cached process-wide, so the warm prove core no longer rebuilds the `H_FH` weight vectors for repeated full-Cora runs inside the same process
- This reduced `warm_prove_ms` from the earlier `~42.0s` baseline to `~3.1s` on the current machine

Benchmark command with isolated artifacts:

```bash
./build/gatzk_run \
  --config configs/cora_full.cfg \
  --benchmark-mode both \
  --export-tag final
```

## Remaining Gaps

- The code no longer uses placeholder quotients or trace-reconstruction verification, but the quotient identities still cover only the currently materialized multi-head columns rather than every note-level lookup/query/multiplicity/accumulator/state-machine family.
- Warm prove/verify are still far from millisecond-scale hot paths. The dominant bottlenecks are now:
  - end-to-end trace generation on full Cora (`~39.2s`)
  - warm verifier quotient replay on `H_FH` (`verify_FH_ms ~ 4.96s`)
  - warm prove `H_FH` opening/evaluation (`domain_open_FH_ms ~ 1.53s`, `quotient_t_fh_ms ~ 0.84s`)
- Legacy synthetic/single-head debug paths still exist in the codebase for non-formal unit tests and have not been fully deleted yet.
- The reference checkpoint parser still preserves the original bias tensors for read-only parity/reference work, but the formal multi-head mainline no longer uses an extra output bias term.
- GPU acceleration is not enabled in the current formal build: `CMakeLists.txt` still compiles with `GATZK_ENABLE_CUDA_BACKEND=0`, so the latest speedups are CPU/batching/cache improvements only.
