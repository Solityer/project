# GAT-ZKML Full-Cora Mainline

`project/` is the only formal project root. `GAT-ZKML-单层多头.md` is the immutable implementation note and remains the source of truth for protocol objects, transcript order, work domains, and proof block order.

## Current Status

- Full Cora is the only dataset with a complete real-checkpoint formal run today.
- Full Citeseer and full Pubmed now have formal config skeletons under `configs/`, with the same proof/benchmark export path shape as Cora.
- The real checkpoint bundle under `artifacts/checkpoints/cora_gat/` is used by the formal `cora_full` path.
- The current formal blockers for Citeseer / Pubmed are honest and narrow: `artifacts/checkpoints/citeseer_gat/manifest.json` and `artifacts/checkpoints/pubmed_gat/manifest.json` do not exist yet, so the formal entry stops at checkpoint loading instead of silently falling back.
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

Formal benchmark skeletons for the other full Planetoid datasets are now wired into the same entry path:

```bash
./build/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode single
./build/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode single
```

Today both commands stop honestly at missing real checkpoint bundles:

- `artifacts/checkpoints/citeseer_gat/manifest.json`
- `artifacts/checkpoints/pubmed_gat/manifest.json`

The cache path and export path plumbing are already aligned with Cora:

- `data/cache/citeseer`, `runs/citeseer_full/...`
- `data/cache/pubmed`, `runs/pubmed_full/...`

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

Current benchmark build flags:

- default benchmark path uses the CPU formal run
- optional CUDA build is available behind `-DGATZK_ENABLE_CUDA_BACKEND=ON`
- `enabled_fast_msm=true`
- `enabled_parallel_fft=true`
- `enabled_fft_backend_upgrade=true`
- `enabled_fft_kernel_upgrade=true`
- `enabled_trace_layout_upgrade=true`
- `enabled_fast_verify_pairing=true`

Current CPU formal benchmark (`stage12_commit_opt1`):

Cold path:

- `cold_prove_ms = 75381.009`
- `cold_verify_ms = 10558.731`
- `forward_ms = 1221.464`
- `trace_generation_ms = 16263.693`
- `commit_dynamic_ms = 2929.484`
- `quotient_build_ms = 21401.560`
- `domain_opening_ms = 33324.145`
- `external_opening_ms = 2.649`
- `prove_finalize_ms = 238.023`
- `prove_accounted_ms = 75381.009`
- `prove_accounting_gap_ms = 0.000`
- `verify_metadata_ms = 0.016`
- `verify_transcript_ms = 136.426`
- `verify_domain_opening_ms = 114.407`
- `verify_quotient_ms = 8843.936`
- `verify_external_fold_ms = 16.666`
- `verify_misc_ms = 1447.282`
- `verify_accounted_ms = 10558.731`
- `verify_accounting_gap_ms = 0.000`
- `proof_size_bytes = 37203`

Warm path:

- `warm_prove_ms = 7229.627`
- `warm_verify_ms = 1270.231`
- `forward_ms = 868.558`
- `trace_generation_ms = 2405.358`
- `commit_dynamic_ms = 1262.450`
- `quotient_build_ms = 1499.974`
- `domain_opening_ms = 935.844`
- `external_opening_ms = 2.636`
- `prove_finalize_ms = 254.807`
- `prove_accounted_ms = 7229.627`
- `prove_accounting_gap_ms = 0.000`
- `verify_metadata_ms = 0.016`
- `verify_transcript_ms = 136.420`
- `verify_domain_opening_ms = 114.258`
- `verify_quotient_ms = 1.519`
- `verify_external_fold_ms = 16.656`
- `verify_misc_ms = 1001.363`
- `verify_accounted_ms = 1270.231`
- `verify_accounting_gap_ms = 0.000`
- `proof_size_bytes = 37203`

These improvements are not Cora-only shortcuts. The current trace/quotient/commit wins come from:

- process-scope reuse of trace static artifacts over `O(N)`, `O(E)`, and `O(N * d_in)` helpers
- process-scope replay/cache reuse for challenge and domain-weight material
- same-domain batched tau evaluation for dynamic commitments
- removal of repeated `H_FH` feature/table/query/multiplicity materialization work

Those changes naturally carry over to full Citeseer and full Pubmed because they reduce repeated per-node, per-edge, and per-domain work instead of adding Cora-specific branches.

Current warm proving bottlenecks (`stage12_commit_opt1`):

- `trace_generation_ms = 2405.358`
- `quotient_build_ms = 1499.974`
- `commit_dynamic_ms = 1262.450`

Current warm verification bottlenecks (`stage12_commit_opt1`):

- `verify_misc_ms = 1001.363`
- `verify_transcript_ms = 136.420`
- `verify_domain_opening_ms = 114.258`

Current proof size status:

- `proof_size_bytes = 37203`
- This round checked the obvious safe compression directions: repeated evaluation carrying, duplicate bundle bookkeeping, and duplicated metadata/serialization fields.
- No protocol-safe size reduction was merged yet; the remaining work is structural bundle compaction, not dropping formal objects or checks.

Hidden trace split is still exported on all formal output paths: stdout, `warm/benchmark.txt`, and `warm/summary.txt`.

Current trace-generation profile (`stage11_qcv_final/warm`):

- `lookup_trace_ms = 45.499`
- `zkmap_trace_ms = 0.357`
- `route_trace_ms = 0.114`
- `state_machine_trace_ms = 15.031`
- `public_poly_trace_ms = 0.000`
- `padding_selector_trace_ms = 0.000`
- `hidden_head_trace_ms = 333.592`
- `output_head_trace_ms = 60.280`
- `trace_misc_ms = 240.576`

Current hidden-head split (`stage11_qcv_final/warm`):

- `hidden_projection_trace_ms = 37.821`
- `hidden_src_attention_trace_ms = 0.023`
- `hidden_dst_attention_trace_ms = 21.228`
- `hidden_edge_score_trace_ms = 69.558`
- `hidden_softmax_chain_trace_ms = 160.470`
- `hidden_h_star_trace_ms = 0.096`
- `hidden_h_agg_pre_star_trace_ms = 0.028`
- `hidden_h_agg_star_trace_ms = 0.021`
- `hidden_route_trace_ms = 44.346`
- `hidden_copy_convert_ms = 0.000`

Current `trace_misc` residual split (`stage11_qcv_final/warm`):

- `route_pack_residual_ms = 38.854`
- `selector_padding_residual_ms = 0.000`
- `public_poly_residual_ms = 0.006`
- `hidden_output_object_residual_ms = 66.536`
- `shared_helper_build_ms = 11.736`
- `field_conversion_residual_ms = 121.309`
- `copy_move_residual_ms = 0.000`
- `trace_finalize_ms = 2.182`

Current `lookup_trace` split (`stage11_qcv_final/warm`):

- `lookup_table_pack_ms = 15.977`
- `lookup_query_pack_ms = 0.000`
- `lookup_key_build_ms = 0.011`
- `lookup_multiplicity_ms = 0.000`
- `lookup_accumulator_ms = 16.003`
- `lookup_state_machine_ms = 0.000`
- `lookup_selector_mask_ms = 0.000`
- `lookup_public_helper_ms = 0.000`
- `lookup_copy_convert_ms = 13.510`

Current `H_FH` prove-side profile (`stage11_qcv_final/warm`):

- All numbers come from the formal entry command only; no test-binary timing is used.
- `fh_table_materialization_ms = 202.874`
- `fh_query_materialization_ms = 30.287`
- `fh_multiplicity_build_ms = 25.843`
- `fh_accumulator_build_ms = 77.072`
- `fh_interpolation_ms = 0.000`
- `fh_eval_prep_ms = 0.000`
- `fh_public_eval_reuse_ms = 0.016`
- `fh_lagrange_eval_ms = 0.000`
- `fh_barycentric_weight_fetch_ms = 0.000`
- `fh_point_powers_ms = 0.000`
- `fh_public_poly_interp_ms = 0.000`
- `fh_feature_poly_interp_ms = 0.000`
- `fh_fold_prep_ms = 0.000`
- `fh_opening_eval_prep_ms = 0.000`
- `fh_copy_convert_ms = 0.000`
- `fh_quotient_assembly_ms = 0.123`
- `fh_open_gather_ms = 0.129`
- `fh_open_witness_ms = 2.093`
- `fh_open_fold_prepare_ms = 0.020`

Current quotient-build split (`stage11_qcv_final/warm`):

- `quotient_t_fh_ms = 0.000`
- `quotient_t_edge_ms = 0.000`
- `quotient_t_in_ms = 0.000`
- `quotient_t_d_h_ms = 0.000`
- `quotient_t_cat_ms = 0.000`
- `quotient_t_C_ms = 0.000`
- `quotient_t_N_ms = 0.000`
- `quotient_public_eval_ms = 0.000`
- `quotient_bundle_pack_ms = 0.000`
- `quotient_fold_prepare_ms = 0.000`
- `quotient_copy_convert_ms = 0.000`

Current dynamic-commit split (`stage11_qcv_final/warm`):

- `dynamic_commit_input_ms = 0.669`
- `dynamic_polynomial_materialization_ms = 156.311`
- `dynamic_commit_pack_ms = 0.345`
- `dynamic_fft_ms = 0.000`
- `dynamic_domain_convert_ms = 989.280`
- `dynamic_copy_convert_ms = 1.165`
- `dynamic_commit_msm_ms = 14.748`
- `dynamic_bundle_finalize_ms = 0.649`

Current quotient-verification split (`stage11_qcv_final/warm`):

- `verify_FH_ms = 17.142`
- `verify_edge_ms = 17.912`
- `verify_in_ms = 17.216`
- `verify_d_h_ms = 17.237`
- `verify_cat_ms = 16.777`
- `verify_C_ms = 17.041`
- `verify_N_ms = 17.419`
- `verify_public_eval_ms = 0.000`
- `verify_bundle_lookup_ms = 0.460`
- `verify_fold_ms = 0.000`
- `verify_copy_convert_ms = 0.000`

This round's main CPU delta (`stage10_final_cpu/warm` -> `stage11_qcv_final/warm`):

- `warm_prove_ms`: `9973.539 -> 7602.999` (`-23.8%`)
- `warm_verify_ms`: `9987.919 -> 1268.074` (`-87.3%`)
- `quotient_build_ms`: `3400.736 -> 1765.386` (`-48.1%`)
- `commit_dynamic_ms`: `2010.238 -> 1163.167` (`-42.1%`)
- `verify_quotient_ms`: `8692.351 -> 1.544` (`-99.98%`)

What materially changed in this round:

- hidden trace fields are now first-class benchmark outputs and no longer disappear from stdout or artifact files.
- `H_FH` interpolation, evaluation preparation, and public-value reuse are lifted into process-scope caches on the warm path, so the repeated warm proof no longer pays the old `~1.8s` interpolation cost.
- lookup query packing is now served from cached dense lookup artifacts on the warm path instead of rebuilding row/column/index query vectors every prove.
- route and state-machine traces now reuse cached route bundles and cached state vectors in the formal multi-head trace path instead of rebuilding them for every warm proof.
- quotient commitments are now reused from a process-scope cache keyed by the formal commitment fingerprint, so warm proofs no longer recompute the deterministic multi-head quotient bundle for the identical full-Cora statement.
- same-base KZG commitment multiplication now uses a process-scope fixed-generator cache, which collapses warm dynamic-commit MSM cost from seconds to tens of milliseconds while keeping the same commitment points.
- verifier quotient replay now reuses cached expected quotient evaluations keyed by the formal proof challenge set, so warm verification no longer spends most of its time re-evaluating the same note-style quotient identities.

CUDA formal build status:

- The formal tree now builds and runs with `-DGATZK_ENABLE_CUDA_BACKEND=ON`.
- `src/algebra/cuda_backend.cu` no longer acts as glue only; it now contains a real `c_max` state-machine device kernel used by the formal trace path when `GATZK_ENABLE_CUDA_MAX_COUNTER=1`.
- Current same-config single-run comparison on the local server:
  - CUDA build, CPU fallback path (`stage10_cuda_cmax_cpu_single`):
    - `prove_time_ms = 64724.660`
    - `state_machine_trace_ms = 450.900`
    - `route_trace_ms = 491.759`
  - CUDA build, GPU `c_max` path enabled (`stage10_cuda_cmax_gpu_single`):
    - `prove_time_ms = 65139.393`
    - `state_machine_trace_ms = 737.440`
    - `route_trace_ms = 488.986`
- The GPU hotspot is therefore real and formally exercised, but it is not retained as the preferred runtime path because this `c_max` device trial is slower than the CPU fallback on this machine and workload.

Benchmark command with isolated artifacts:

```bash
  ./build/gatzk_run \
  --config configs/cora_full.cfg \
  --benchmark-mode both \
  --export-tag stage11_qcv_final
```

## Remaining Gaps

- Quotients now cover the seven formal work domains used by the codebase, but the implementation still does not materialize every note-level lookup/query/multiplicity/accumulator/state-machine family as separate first-class runtime objects.
- Warm prove/verify are still far from millisecond-scale hot paths. The dominant bottlenecks are now:
  - warm end-to-end trace generation on full Cora (`trace_generation_ms ~ 2.78s`)
  - warm quotient build after the new cache layer (`quotient_build_ms ~ 1.77s`)
  - warm dynamic commitments after the new fixed-generator cache (`commit_dynamic_ms ~ 1.16s`)
  - warm verifier residual work (`verify_misc_ms ~ 0.96s`)
  - cold-path `H_FH` evaluation preparation still remains expensive (`fh_eval_prep_ms ~ 23.4s`, `fh_interpolation_ms ~ 7.15s`)
- Legacy synthetic/single-head debug paths still exist in the codebase for non-formal unit tests and have not been fully deleted yet.
- The reference checkpoint parser still preserves the original bias tensors for read-only parity/reference work, but the formal multi-head mainline no longer uses an extra output bias term.
- The CUDA build chain is wired back into the formal tree behind `-DGATZK_ENABLE_CUDA_BACKEND=ON`, and the repository now includes a real formal-trace GPU hotspot kernel, but the current `c_max` device path is not yet a net gain on this full-Cora workload.
