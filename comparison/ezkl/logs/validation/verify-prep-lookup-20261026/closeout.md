# Lookup-H cache and Verify preparation closeout

Date: 2026-10-26

This note records three focused CPU checks and the final official GPU release build for the lookup-H product cache and public-instance preparation changes. It does not include a full dataset experiment or a new performance benchmark.

## Changes covered

- [`evaluation.rs`](../../../vendor/halo2/halo2_proofs/src/plonk/evaluation.rs) reuses each lookup input's evaluated values to compute the product terms in the same row pass, avoiding a second graph evaluation. Input values occupy the buffer prefix and products the suffix; only the input prefix is batch-inverted. The chunk bound remains `rows_per_chunk × (input_count + 1) ≤ 4096 × input_count`, with the zero-input guard preserved.
- [`graph/mod.rs`](../../../src/graph/mod.rs) adds a borrowed public-instance builder that avoids cloning the complete witness while preserving instance ordering and the stored-witness behavior of the legacy API.
- [`execute.rs`](../../../src/execute.rs) converts large signed and `usize` public-instance vectors in parallel while preserving field values and element order; below the threshold it retains the serial legacy path. The four core metrics and historical proof artifacts were not changed. See the [historical size and timing audit](../metrics-scope-20261026/audit.md) for the distinction between core timings, command-phase timings, and serialized proof size.

## Focused CPU validation

Each selected test ran once and executed one test case. The vendor test used the offline CPU feature set `mv-lookup,circuit-params,parallel-poly-read,formal-permutation-spill`, with no `formal-gpu-msm`. The two root tests used `formal-gpu` because their production code is feature-gated, but ran CPU-only functions with `ENABLE_ICICLE_GPU`, `CUDA_VISIBLE_DEVICES`, and `ICICLE_DEVICE_ID` unset.

| Check | Result | Runtime |
| --- | ---: | ---: |
| `plonk::evaluation::tests::native_mv_lookup_h_product_cache_matches_legacy_rows` | 1 passed, 0 failed | 5.18 s |
| `graph::tests::borrowed_public_instances_match_legacy_order_and_stored_witness_behavior` | 1 passed, 0 failed | <0.01 s |
| `execute::formal_public_instance_conversion_tests::formal_instance_parallel_conversion_matches_legacy_values_and_order` | 1 passed, 0 failed | 0.10 s |

The lookup oracle exercises one to three inputs, both lookup cases, `y = 0` and `y = 1`, three parts, rotation wraps, zero-valued inputs, and 4,101 rows. It compares cached products and final H rows with the former repeated-evaluation path and checks graph-call counts. No real multi-dataset proving or verification was run.

## Official build and receipt

The official `scripts/build_ezkl.sh --gpu` release build completed in 3m41s. Existing native backend authentication passed, so the build did not rerun CUDA setup. The final CLI contract and native source identity checks passed.

All recomputed identities matched `artifacts/build/gpu/build_receipt.json`:

| Identity | SHA-256 |
| --- | --- |
| Source tree | `fbbd513ce386ce1e3c52709a568a8505c9bc706689279116fa19970f57187074` |
| GPU release binary | `f8e8ea0d247a1c993c59441a751949e43d607b7de9c1344b5c0245c55d1b5aa8` |
| Native backend manifest | `4b814782da97ae253c4a57fb577a20c2fae71817ea4ca331146ab8e7f17d4674` |
| Resource manifest | `fdf1c7729efed73266beff9111765549120ea0b32ecf58e212bb317ba5` |
| Adapter contract (`ezkl-gat-native-v1`) | `df6266c06496ee7a36586b94bd79d7955d92612667c16315015476bfcbddd2d2` |

The native libraries also matched the receipt. It binds the native backend to GPU UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71` and policy `halo2_icicle-bn254-gpu-hgraph-fused-ntt-16-25-msm-lagrange-resident-f4-v5`.

The post-build read-only process snapshot showed the GPU occupied by the user's zkDL `gat_runner` (PID 3809019, 671 MiB), with its `formal_runner.py` at PID 3774319. No EZKL formal runner was present. Neither process was changed or stopped, and no GPU kernel was launched.

## Evidence boundary

The tests establish parity for the selected lookup and instance-preparation paths; the official build establishes that the release binary and recorded source/native identities agree. These checks provide no PPI or OGBN-Arxiv wall-time or proof-size measurement and do not establish a workload-level speedup. No formal dataset run or independent formal Verify was performed.
