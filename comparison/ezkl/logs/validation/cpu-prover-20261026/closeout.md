# EZKL prover-path closeout — 2026-10-26

## Changes

This batch changes two source files:

- `vendor/halo2/halo2_proofs/src/icicle.rs`: bulk host packing and writeback for the BN254 raw Montgomery H-query path when the stride contract permits it. The checked legacy row path remains the fallback; output residues are validated before destination mutation.
- `vendor/halo2/halo2_proofs/src/poly/kzg/multiopen/shplonk/prover.rs`: borrow dense query polynomials after accumulator initialization and fuse low-degree correction, challenge scaling, and accumulation into the existing coefficient pass. Deferred fixed/sigma/lookup polynomials retain materialization. Query order, challenge ranks, cache skips, error propagation, and memory/task caps are unchanged.

The standalone compile exposed two uses of unsupported `Polynomial[RangeTo]`. They now use the existing `as_mut_slice()[..len]` and `as_slice()[..len]` APIs, preserving the same checked slice boundary and panic behavior. No field arithmetic, planner, proof transcript, or resource limit was changed.

## Bounded CPU validation

The vendor Halo2 crate was tested with the repository-pinned `nightly-2025-12-01`, `CARGO_BUILD_JOBS=2`, a temporary lockfile at `/tmp/ezkl-halo2-closeout-20261026/Cargo.lock`, and target directory `target/gpu`. The repository and vendor `Cargo.lock` files were not modified. `ENABLE_ICICLE_GPU` and `CUDA_VISIBLE_DEVICES` were unset during these tests.

1. `icicle::h_eval_gpu_fixture::raw_h_query_pack_and_writeback_match_legacy_word_path` — **passed, 1/1**. It compiled with the formal GPU feature so the tested code is present, but performs only CPU field and host-memory comparisons; it does not select a device or call a CUDA kernel.
2. `poly::kzg::multiopen::shplonk::prover::tests::fused_query_accumulation_matches_materialize_scale_and_add` — **passed, 1/1**, CPU features only.
3. `poly::multiopen_test::test::test_roundtrip_shplonk` — **passed, 1/1**, K=4 CPU SHPLONK proof creation followed by `VerifierSHPLONK` verification.

## Official GPU build

The official `scripts/build_ezkl.sh --gpu` path completed successfully in 3m44s. The log records prerequisite authentication, Rust release build, native CLI contract verification, source identity verification, and completion. It did not run native CUDA setup or a GPU kernel.

- Build log: `logs/validation/cpu-prover-20261026/official_gpu_build.log`
- Binary: `target/gpu/release/ezkl`
- Binary SHA256 (receipt and file): `a9d7803c21f91bf29a2338ec9d09fe6d42663e9e843286e273ed1f1785843092`
- Source tree SHA256: `840a166e968bb9b889567d8581e7ec50c90f3fef2d27eeae2a87fe25b0f315bd`
- Resource manifest SHA256: `fdf1c7729efed73266beff9111765549120ea0b32e99132ecf58e212bb317ba5`
- Adapter contract: `ezkl-gat-native-v1`, SHA256 `df6266c06496ee7a36586b94bd79d7955d92612667c16315015476bfcbddd2d2`
- Native backend manifest SHA256 (receipt and file): `4b814782da97ae253c4a57fb577a20c2fae71817ea4ca331146ab8e7f17d4674`
- Native CUDA source SHA256: `471916832b77262853bcdfcd8ff7ad4db0254ca1723ee2529e6a24e5e36b46be`
- Native execution policy: `halo2_icicle-bn254-gpu-hgraph-fused-ntt-16-25-msm-lagrange-resident-f4-v5`

## Runtime boundary and handoff

The post-build process snapshot found no EZKL binary or formal EZKL runner active. The only CUDA process was the user-owned `/home/pzh/project/comparison/zkdl/gat_runner` (PID `3809019`, 671 MiB) on GPU UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71`. It was left untouched. Native GPU kernel execution and formal PPI/OGBN proof timing therefore remain unverified; no speedup or end-to-end completion time is claimed.

Start a fresh official matrix session with:

```bash
cd /home/pzh/project/comparison/ezkl
python3 scripts/run_formal_matrix.py --execute-formal
```

Do not resume the prior layout-v1 session. This closeout did not run Setup, witness generation, formal Prove/Verify, or any dataset experiment.
