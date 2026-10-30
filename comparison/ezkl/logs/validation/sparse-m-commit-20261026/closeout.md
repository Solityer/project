# Sparse mv-lookup `m` commitment closeout

Date: 2026-10-26

## Bounded change under validation

The frozen implementation replaces construction of a dense `N`-row polynomial plus full-domain initial `m` commitment with an indexed CPU commitment when nonzero support is at most 32. Larger support retains the dense fallback. The KZG path continues to ignore `Blind`; the IPA path includes `r*w`, including empty support. Later lookup evaluation, H/SHPLONK, and proof semantics remain unchanged. This run contains no formal dataset timing, so it makes no measured end-to-end speedup claim.

The approved execution policy in source and the current GPU build receipt is `halo2_icicle-bn254-gpu-hgraph-fused-ntt-16-25-msm-lagrange-resident-f4-sparse-m-cpu32-v6`.

## Limited checks

| Check | Result | Evidence |
| --- | --- | --- |
| `plonk::mv_lookup::prover::native_table_map_tests::sparse_m_commit_matches_legacy_for_kzg_and_ipa` | 1 passed, 0 failed (0.64 s) | [`sparse_m_commit_matches_legacy_for_kzg_and_ipa.log`](sparse_m_commit_matches_legacy_for_kzg_and_ipa.log) |
| Existing `plonk_api` proof/verifier test | 1 passed, 0 failed (2.88 s) | [`plonk_api.log`](plonk_api.log) |
| `test_five_dataset_ntt_plans_and_integrated_memory_envelopes` | 1 passed | [`test_five_dataset_ntt_plans_and_integrated_memory_envelopes.log`](test_five_dataset_ntt_plans_and_integrated_memory_envelopes.log) |
| Sparse prepare/commit budget delta through the production `_native_capacity_requirements` using the existing admission fixture | Passed: prepare 3248 bytes, later commit 512 bytes | [`native-capacity-targeted-delta.py`](native-capacity-targeted-delta.py), [`native-capacity-targeted-delta.log`](native-capacity-targeted-delta.log) |

The broader `test_native_capacity_model_counts_setup_and_quotient_mappings` selector was attempted once and did not pass: it stops at the existing `setup_persisted_mapped_bytes` assertion (expected 9216, actual 45892), before reaching this change's prepare/commit assertions. The actual 45892 value follows the current fixed-source mapping model. Per scope, that unrelated stale expectation was not changed; the bounded delta check above calls the same production requirements function with that same fixture. See [`test_native_capacity_model_counts_setup_and_quotient_mappings.log`](test_native_capacity_model_counts_setup_and_quotient_mappings.log).

The initial Rust invocation also failed because Cargo selected the vendor directory's `nightly-2024-07-18` rustc shim against cached `nightly-2025-12-01` dependencies (E0514). That log is preserved; setting the pinned `RUSTUP_TOOLCHAIN`, `RUSTC`, and `RUSTDOC` explicitly made the exact oracle pass once. No source fix was made for this environment issue.

## Official GPU build and identity

The full official GPU release command (including `--python`, absolute toolchain/compiler paths, selected-device JSON, and canonical build-environment JSON) completed with exit code 0 in 3m48s. Its full command and output are in [`official-gpu-build-full.log`](official-gpu-build-full.log). The source-identity and native CLI-contract checks passed and a current receipt was written at [`build_receipt.json`](../../../artifacts/build/gpu/build_receipt.json).

Post-build verification passed all 9 identity checks in [`postbuild-identity-verification.log`](postbuild-identity-verification.log): source tree, binary, native manifest, both native libraries, resource manifest, adapter contract raw bytes, canonical build environment, and the embedded build-environment digest. It additionally confirmed the v6 policy. Source tree SHA-256 is `5fc56cae2cb07804acb587ce1a9d0f3b8c780704ba1292ea7fe910aec3ef4f3a`; GPU binary SHA-256 is `2d61ac9ff3a0d2fdb7f7cc32efef23a749c21d65afb2b3becdde41cade7c0705`; canonical build-environment SHA-256 is `2de96669cf8a04268cb8da23719c3eecd68d25cf12889ffe602352e827a1bd6b`.

The first post-build identity attempt lacked the official selected-device environment and stopped after source/binary checks. It is preserved as [`postbuild-identity-verification.initial-environment-gap.log`](postbuild-identity-verification.initial-environment-gap.log). The corrected read-only check used the official command's device identity; pre/post snapshots independently show the same physical GB10 UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71`. No receipt, manifest, native library, or source was altered during the check.

## Execution boundary and process snapshot

Read-only process and GPU snapshots are in [`prebuild-process-snapshot.txt`](prebuild-process-snapshot.txt), [`prebuild-gpu-snapshot.csv`](prebuild-gpu-snapshot.csv), [`postbuild-process-snapshot.txt`](postbuild-process-snapshot.txt), and [`postbuild-gpu-snapshot.csv`](postbuild-gpu-snapshot.csv). The user's zkDL `formal_runner.py` and `gat_runner verify` remained active; no user process was stopped or modified.

No formal dataset run, full dataset preprocessing, SRS/PK-hash operation, proving/verifying experiment, or CUDA kernel was launched. The Rust checks unset `ENABLE_ICICLE_GPU`, `CUDA_VISIBLE_DEVICES`, and `ICICLE_DEVICE_ID`; they used pinned `nightly-2025-12-01`, explicit rustc/rustdoc, the isolated lockfile, and at most two Cargo/Rayon workers. The official GPU command only built and identity-checked the release artifacts.

The source was frozen after the build receipt. Only validation records under this directory were written afterward.
