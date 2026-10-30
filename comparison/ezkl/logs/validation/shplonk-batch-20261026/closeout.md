# SHPLONK and coset optimization closeout

Date: 2026-10-26

This closeout records the bounded CPU checks and official GPU release build for the shared coset-factor buffer and SHPLONK accumulation batches. It does not report a new dataset-level performance measurement.

## Changes covered

- [`poly/domain.rs`](../../../vendor/halo2/halo2_proofs/src/poly/domain.rs) adds a mapped coefficient-factor buffer and an in-place conversion that turns that buffer into the deferred coset output.
- [`plonk/evaluation.rs`](../../../vendor/halo2/halo2_proofs/src/plonk/evaluation.rs) shares the factor buffer across missing cosets and leaves the last advice or instance coset deferred for in-place output. The added N-field factor/output workspace is one mapped allocation; the final transform reuses its address. The regression checks pointer reuse and equality with the legacy conversion.
- [`shplonk/prover.rs`](../../../vendor/halo2/halo2_proofs/src/poly/kzg/multiopen/shplonk/prover.rs) batches quotient and linearisation contributions into tiled accumulator passes, preserving the existing query order and scalar weights. The release compiler exposed that the existing `accumulate_scaled` helper was still behind `#[cfg(test)]` despite production callers. The annotation was removed; the helper's implementation and proof math were unchanged.
- [`src/execute.rs`](../../../src/execute.rs) adds per-stage Verify timing records. The measurement definitions remain unchanged; the old proof and timing artifacts were not edited. See the [historical size and timing audit](../metrics-scope-20261026/audit.md) for the distinction between core operation timings and full command-phase timings.

## Bounded validation

Each listed test ran once and executed one test case. All used the offline CPU Halo2 configuration `--no-default-features --features mv-lookup,circuit-params,parallel-poly-read,formal-permutation-spill`, with the pinned nightly and isolated temporary lockfile. No GPU kernel was invoked.

| Check | Result | Runtime |
| --- | ---: | ---: |
| `plonk::evaluation::tests::native_gate_group_fusion_and_coset_cache_match_legacy` | 1 passed, 0 failed | 0.07 s |
| `poly::kzg::multiopen::shplonk::prover::tests::fused_query_accumulation_matches_materialize_scale_and_add` | 1 passed, 0 failed | <0.01 s |
| `poly::multiopen_test::test::test_roundtrip_shplonk` (K=4 CPU create-proof plus `VerifierSHPLONK`) | 1 passed, 0 failed | 0.17 s |

An initial abbreviated `--exact` filter for the coset test matched zero cases; it was not counted as a pass. The same check then ran with its full module path and passed. The CPU checks preceded the visibility-only `#[cfg(test)]` fix; the final release build below compiled that fix and the production call sites.

## Official build and identity

The official `scripts/build_ezkl.sh --gpu` release build completed successfully in 3m42s. Native prerequisite authentication passed and reused the already authenticated native backend. No native CUDA setup, dataset command, proof, independent formal Verify, or GPU-kernel test was run.

The resulting receipt is `artifacts/build/gpu/build_receipt.json`. Recomputed source, binary, native-library, native-manifest, resource-manifest, and adapter-contract identities all matched the receipt:

| Identity | SHA-256 |
| --- | --- |
| Source tree | `d81f9c7faa06d4b8c915c846026c4675387826fa24b77b4d0e67b9b0edec5257` |
| GPU release binary | `4b474066038bd1505ea038ed855e8b2c23d8002fa65150b67d897cfb0d5d86a7` |
| Native backend manifest | `4b814782da97ae253c4a57fb577a20c2fae71817ea4ca331146ab8e7f17d4674` |
| Resource manifest | `fdf1c7729efed73266beff9111765549120ea0b32e99132ecf58e212bb317ba5` |
| Adapter contract (`ezkl-gat-native-v1`) | `df6266c06496ee7a36586b94bd79d7955d92612667c16315015476bfcbddd2d2` |

The receipt binds the native backend to GPU UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71` and policy `halo2_icicle-bn254-gpu-hgraph-fused-ntt-16-25-msm-lagrange-resident-f4-v5`.

At the post-build read-only process snapshot, the GPU was owned by the user's zkDL `gat_runner` (PID 3809019, 671 MiB); its `formal_runner.py` was PID 3774319. No EZKL formal runner was present. Neither process was changed or stopped.

## Evidence boundary

The CPU checks establish the tested algebra, coset equivalence, and a small K=4 SHPLONK proof/verifier round trip. The official build establishes that the GPU release binary and source/native receipts are consistent. No PPI or OGBN-Arxiv run, GPU-kernel execution, or new dataset-level latency/size measurement was performed, so this closeout makes no real-workload speedup claim.
