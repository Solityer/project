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
- The quotient layer now covers all seven formal work domains, including `H_FH` feature retrieval lookup plus the note-level `C_feat,tbl` and `C_feat,qry` binding constraints; the remaining non-final work is cleanup/performance, not missing `FH`-domain quotient coverage.

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

Accounting definitions:

- `cold_*`: one formal run in a fresh process state, including first-use static context/FFT/SRS cache construction inside the same `gatzk_run` process.
- `warm_*`: measured after one in-process `warm_probe`; the exported `warm` sample is the second hot-path prove/verify in the same process and excludes one-time cache construction.
- `prove_time_ms`: wall-clock from `build_trace()` entry to `prove()` return.
- `forward_ms`: note-style clear forward only.
- `trace_generation_ms`: witness/table/query/route/state materialization after subtracting `forward_ms` and `commit_dynamic_ms`.
- `commit_dynamic_ms`: dynamic polynomial/commitment path only.
- `quotient_build_ms`: quotient evaluation and quotient commitment construction only.
- `domain_opening_ms`: seven-domain opening gather + witness/opening only.
- `external_opening_ms`: external folded opening only.
- `prove_finalize_ms`: residual prove wall-clock not covered by the prove subitems above.
- `verify_time_ms`: wall-clock from `verify()` entry to verifier return.
- `verify_metadata_ms`: metadata + proof block order checks.
- `verify_transcript_ms`: Fiat-Shamir replay only.
- `verify_domain_opening_ms`: seven-domain opening verification only.
- `verify_quotient_ms`: quotient identity evaluation/check only.
- `verify_external_fold_ms`: external fold verification only.
- `verify_misc_ms`: residual verifier wall-clock not covered by the verify subitems above.
- `prove_accounted_ms` / `verify_accounted_ms`: exact sums of their respective subitems.
- `prove_accounting_gap_ms` / `verify_accounting_gap_ms`: `total - accounted`; the target is `0` up to timer noise.

Current build flags:

- CPU-only formal path
- `enabled_fast_msm=true`
- `enabled_parallel_fft=true`
- `enabled_fft_backend_upgrade=true`
- `enabled_fft_kernel_upgrade=true`
- `enabled_trace_layout_upgrade=true`
- `enabled_fast_verify_pairing=true`

Current formal benchmark (`stage6_traceopt`):

Cold path:

- `cold_prove_ms = 74435.312`
- `cold_verify_ms = 10153.812`
- `forward_ms = 885.334`
- `trace_generation_ms = 11556.342`
- `commit_dynamic_ms = 9068.943`
- `quotient_build_ms = 20649.945`
- `domain_opening_ms = 31996.914`
- `external_opening_ms = 2.697`
- `prove_finalize_ms = 275.138`
- `prove_accounted_ms = 74435.312`
- `prove_accounting_gap_ms = 0.000`
- `verify_metadata_ms = 0.018`
- `verify_transcript_ms = 175.720`
- `verify_domain_opening_ms = 114.683`
- `verify_quotient_ms = 8809.731`
- `verify_external_fold_ms = 16.831`
- `verify_misc_ms = 1036.829`
- `verify_accounted_ms = 10153.812`
- `verify_accounting_gap_ms = 0.000`
- `proof_size_bytes = 37203`

Warm path:

- `warm_prove_ms = 18910.522`
- `warm_verify_ms = 10279.818`
- `forward_ms = 881.127`
- `trace_generation_ms = 9989.301`
- `commit_dynamic_ms = 2282.849`
- `quotient_build_ms = 3633.184`
- `domain_opening_ms = 1847.662`
- `external_opening_ms = 2.704`
- `prove_finalize_ms = 273.695`
- `prove_accounted_ms = 18910.522`
- `prove_accounting_gap_ms = 0.000`
- `verify_metadata_ms = 0.017`
- `verify_transcript_ms = 175.763`
- `verify_domain_opening_ms = 114.677`
- `verify_quotient_ms = 8904.573`
- `verify_external_fold_ms = 16.797`
- `verify_misc_ms = 1067.992`
- `verify_accounted_ms = 10279.818`
- `verify_accounting_gap_ms = 0.000`
- `proof_size_bytes = 37203`

Current trace-generation profile (`stage6_traceopt/warm`):

- `lookup_trace_ms = 4841.549`
- `zkmap_trace_ms = 1148.041`
- `route_trace_ms = 510.616`
- `state_machine_trace_ms = 573.787`
- `public_poly_trace_ms = 248.812`
- `padding_selector_trace_ms = 337.607`
- `hidden_head_trace_ms = 543.985`
- `output_head_trace_ms = 46.631`
- `trace_misc_ms = 1738.271`

Current `H_FH` prove-side profile (`stage6_traceopt/warm`):

- All numbers come from the formal entry command only; no test-binary timing is used.
- `fh_table_materialization_ms = 264.242`
- `fh_query_materialization_ms = 55.263`
- `fh_multiplicity_build_ms = 48.093`
- `fh_accumulator_build_ms = 75.337`
- `fh_interpolation_ms = 1841.856`
- `fh_eval_prep_ms = 0.002`
- `fh_public_eval_reuse_ms = 0.015`
- `fh_quotient_assembly_ms = 0.338`
- `fh_open_gather_ms = 0.214`
- `fh_open_witness_ms = 3.622`
- `fh_open_fold_prepare_ms = 0.034`

This round's trace delta (`stage5_hfh_final2/warm` -> `stage6_traceopt/warm`):

- `warm_prove_ms`: `27115.654 -> 18910.522` (`-30.3%`)
- `trace_generation_ms`: `18841.211 -> 9989.301` (`-47.0%`)
- `quotient_build_ms`: `3338.982 -> 3633.184` (`+8.8%`)
- `domain_opening_ms`: `1829.106 -> 1847.662` (`+1.0%`)
- `lookup_trace_ms`: now the largest warm trace bucket at `4841.549`
- `zkmap_trace_ms`: now the second largest warm trace bucket at `1148.041`

What materially changed in this round:

- trace-side pair lookups no longer hash decimal strings; they use packed fixed-size field keys instead.
- repeated `count_by_src / count_by_dst` work is now built once and reused across all hidden/output routing traces.
- repeated `dst` group walks for max-uniqueness state reuse one precomputed edge-group layout.
- `build_binding_trace()` now reuses process-wide native barycentric weights and folds columns directly from native row weights without copying a `FieldElement` weight slice.
- trace timing is split into mutually exclusive top-level buckets so the formal benchmark itself exposes the current top 2 trace hotspots.

Benchmark command with isolated artifacts:

```bash
  ./build/gatzk_run \
  --config configs/cora_full.cfg \
  --benchmark-mode both \
  --export-tag stage6_traceopt
```

## Remaining Gaps

- Quotients now cover the seven formal work domains used by the codebase, but the implementation still does not materialize every note-level lookup/query/multiplicity/accumulator/state-machine family as separate first-class runtime objects.
- Warm prove/verify are still far from millisecond-scale hot paths. The dominant bottlenecks are now:
  - warm end-to-end trace generation on full Cora (`trace_generation_ms ~ 9.99s`)
  - warm verifier quotient replay (`verify_quotient_ms ~ 8.90s`)
  - warm prove `H_FH` / `H_edge` quotient-plus-opening (`quotient_t_fh_ms ~ 1.72s`, `domain_open_FH_ms ~ 1.85s`, `quotient_t_edge_ms ~ 1.78s`)
- Legacy synthetic/single-head debug paths still exist in the codebase for non-formal unit tests and have not been fully deleted yet.
- The reference checkpoint parser still preserves the original bias tensors for read-only parity/reference work, but the formal multi-head mainline no longer uses an extra output bias term.
- GPU acceleration is still not in the formal mainline. The server does expose an NVIDIA GB10 GPU, but the current repository still compiles with `GATZK_ENABLE_CUDA_BACKEND=0` and no CUDA translation units (`src/**/*.cu`) remain, so there is no linkable formal GPU backend to benchmark honestly in this tree. The retained gains in this round are trace-side dense indexing, reused routing layouts, and cached native barycentric folding on the CPU formal mainline.
