# GAT-ZKML Full-Cora Mainline

`project/` is the only formal project root. `GAT-ZKML-单层多头.md` is the immutable implementation note and remains the source of truth for protocol objects, transcript order, work domains, and proof block order.

## Current Status

- Full Cora is the only formal dataset and benchmark target.
- The real checkpoint bundle under `artifacts/checkpoints/cora_gat/` is used by the formal `cora_full` path.
- The official GAT forward semantics are aligned locally against `reference/gat_official/` and `runs/cora_full/reference/`.
- `gatzk_run --config configs/cora_full.cfg` currently completes and prints `VERIFY_OK`.
- The current multi-head formal mainline materializes hidden-head, concat, and output-stage objects and runs end-to-end, but the seven-domain quotient layer is still emitted as zero-evaluation placeholder polynomials. The verifier therefore relies on deterministic trace reconstruction plus KZG opening checks rather than a fully note-complete quotient identity system.

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

Measured from the formal entry command on the local Linux server:

- `prove_time_ms = 38006.325`
- `verify_time_ms = 45746.077`
- `forward_ms = 871.468`
- `trace_generation_ms = 38498.351`
- `commit_dynamic_ms = 7851.770`
- `proof_size_bytes = 25247`

Other measured fields from the same run:

- `load_static_ms = 7887.845`
- `fft_plan_ms = 2549.473`
- `srs_prepare_ms = 9.913`
- `quotient_build_ms = 0.000`
- `domain_opening_ms = 0.000`
- `external_opening_ms = 0.000`

## Remaining Gaps

- The multi-head formal mainline runs end-to-end on full Cora, but `t_FH / t_edge / t_in / t_d_h / t_cat / t_C / t_N` are still placeholder zero-evaluation quotient polynomials.
- The verifier currently reconstructs the expected multi-head trace and checks commitments, openings, metadata, and block order against that reconstruction.
- Legacy single-head debug code paths still exist for non-formal synthetic tests and have not been fully removed from the codebase.
