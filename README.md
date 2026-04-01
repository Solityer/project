# GAT-ZKML Full-Cora Mainline

`project/` is the only formal project root. `GAT-ZKML-单层多头.md` is the immutable implementation note and remains the source of truth for protocol objects, transcript order, work domains, and proof block order.

## Current Status

- Full Cora is the only formal dataset and benchmark target.
- The real checkpoint bundle under `artifacts/checkpoints/cora_gat/` is used by the formal `cora_full` path.
- The official GAT forward semantics are aligned locally against `reference/gat_official/` and `runs/cora_full/reference/`.
- `gatzk_run --config configs/cora_full.cfg` currently completes and prints `VERIFY_OK`.
- The formal multi-head mainline now commits non-placeholder `t_FH / t_edge / t_in / t_d_h / t_cat / t_C / t_N`, opens seven work domains, and replays Fiat-Shamir directly from proof commitments instead of rebuilding an expected trace inside the verifier.
- The repository root keeps only `README.md` plus the immutable note `GAT-ZKML-单层多头.md`.
- The current system is still not note-complete enough to call the implementation final: the quotient layer only constrains the currently materialized multi-head objects, and the forward path still carries the local checkpoint/reference output-bias route that conflicts with the note's stricter output-head wording.

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

Cold path:

- `cold_prove_ms = 41659.458`
- `cold_verify_ms = 5562.720`
- `forward_ms = 855.550`
- `trace_generation_ms = 38522.788`
- `commit_dynamic_ms = 7857.364`
- `quotient_build_ms = 16986.371`
- `domain_opening_ms = 24669.891`
- `external_opening_ms = 2.626`
- `proof_size_bytes = 25247`

Warm path:

- `warm_prove_ms = 41909.616`
- `warm_verify_ms = 5570.337`
- `forward_ms = 865.645`
- `trace_generation_ms = 38647.544`
- `commit_dynamic_ms = 7903.213`
- `quotient_build_ms = 17213.592`
- `domain_opening_ms = 24692.801`
- `external_opening_ms = 2.648`
- `proof_size_bytes = 25247`

Cold-only initialization fields:

- `load_static_ms = 7837.584`
- `fft_plan_ms = 2566.497`
- `srs_prepare_ms = 9.868`

Warm-only initialization fields:

- `load_static_ms = 232.246`
- `fft_plan_ms = 0.000`
- `srs_prepare_ms = 0.000`

## Remaining Gaps

- The code no longer uses placeholder quotients or trace-reconstruction verification, but the quotient identities still cover only the currently materialized multi-head columns rather than every note-level table/query/state object.
- Warm prove/verify are still far from millisecond-scale hot paths. The dominant costs are `trace_generation_ms`, `quotient_build_ms`, and especially `domain_opening_ms`; simple top-level async parallelization did not improve these numbers on this machine and was reverted.
- Legacy synthetic/single-head debug paths still exist in the codebase for non-formal unit tests and have not been fully deleted yet.
- The forward path still carries checkpoint/reference output-bias handling, and the repository still contains the full-Cora bias parity test. That is a real remaining conflict with the note's stricter output-layer wording.
