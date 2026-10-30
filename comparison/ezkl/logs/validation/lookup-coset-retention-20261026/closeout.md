# Lookup adjacent-union coset retention closeout

Date: 2026-10-26

## Change

[`evaluation.rs`](../../../vendor/halo2/halo2_proofs/src/plonk/evaluation.rs) now retains only fixed, advice, and instance cosets present in the next lookup argument's dependency union. It evicts inactive columns from all three source classes before loading missing columns, then uses the existing gate-group loader. Lookup order, graph evaluation, H accumulation, and input/product evaluation are unchanged. The cache is recreated at each part/proof-instance boundary and explicitly dropped before shuffle evaluation.

The production helper is `refresh_lookup_argument_cosets` (line 1525); both lookup branches use it. The CPU oracle `native_lookup_argument_coset_cache_matches_legacy` (line 4745) calls that production helper and compares cosets and H against the independent legacy loaders/evaluator.

## Focused CPU checks

Both checks ran once using `nightly-2025-12-01`, isolated lockfile `/tmp/ezkl-perm-stream-20261026/Cargo.lock`, Halo2 features `mv-lookup,circuit-params,parallel-poly-read,formal-permutation-spill`, `CARGO_BUILD_JOBS=2`, and `RAYON_NUM_THREADS=2`. `ENABLE_ICICLE_GPU`, `CUDA_VISIBLE_DEVICES`, and `ICICLE_DEVICE_ID` were unset.

| Check | Result | Runtime |
| --- | ---: | ---: |
| `plonk::evaluation::tests::native_lookup_argument_coset_cache_matches_legacy` | 1 passed, 0 failed | 0.01 s |
| `tests/plonk_api.rs::plonk_api` | 1 passed, 0 failed | 0.49 s |

The existing `plonk_api` test completed its three small proof/verifier round trips (GWC/KZG, SHPLONK/KZG, and IPA/Pasta). The exact invocations and outputs are in [`native_lookup_argument_coset_cache_matches_legacy.log`](native_lookup_argument_coset_cache_matches_legacy.log) and [`plonk_api.log`](plonk_api.log).

For the oracle fixture, each of two parts and two independently rebuilt proof-instance caches evaluates the same four-argument dependency sequence. Summing its actual source-column sets gives 18 legacy dense transforms per context (5 fixed nonconstant + 8 advice + 5 instance) and 12 with retention (3 + 5 + 4). Across the four fixture contexts this is 72 versus 48 transforms, a reduction of 24. Constant fixed-column fills are not FFTs: the fixture loads 12 through the legacy path and 8 through the retained path. These are small synthetic fixture counts, not formal dataset or PPI counts. Every step also asserts exact active-cache equality with the current dependency union, no stale entries, and equal legacy/cached H output.

## Official GPU release build and identities

The full official GPU release command, including required Python, compiler/toolchain, selected-device, and canonical build-environment arguments, is recorded as `OFFICIAL_GPU_FINAL_COMMAND` in [`official-gpu-build-full.log`](official-gpu-build-full.log). It completed successfully in 228.371 seconds (exit 0), passed the native CLI-contract and source-identity checks, and reused the authenticated native backend. It did not launch a CUDA kernel.

Post-build identity verification passed 9/9 checks. Adapter identity below is SHA-256 over the raw bytes of `artifacts/gat_adapter/formal_cli.json`; build-environment identity is the canonical JSON digest after removing its embedded digest field. Full evidence is in [`postbuild-identity-verification.log`](postbuild-identity-verification.log) and [`build_receipt.json`](../../../artifacts/build/gpu/build_receipt.json).

| Identity | SHA-256 |
| --- | --- |
| Source tree | `edc648509986e97711830af7ab9b125ee3d3febc7148926c1da14aea373e5c9c` |
| GPU release binary | `c2d496e6c1041fe37653e1223c0eb687af4808daabb7483cad04f9dff269d252` |
| Native backend manifest | `4b814782da97ae253c4a57fb577a20c2fae71817ea4ca331146ab8e7f17d4674` |
| `libingo_curve_bn254.a` | `f3aa2afea74e467e61e7283a2ba75c472054458d51f84f581d373bcda7757f60` |
| `libingo_field_bn254.a` | `99d3fc55944526354b95914c2b6711288f022e24d3f0b40dd5ea4b576fb22317` |
| Resource manifest | `fdf1c7729efed73266beff9111765549120ea0b32e99132ecf58e212bb317ba5` |
| Adapter contract (raw bytes) | `df6266c06496ee7a36586b94bd79d7955d92612667c16315015476bfcbddd2d2` |
| Build environment (canonical JSON) | `2de96669cf8a04268cb8da23719c3eecd68d25cf12889ffe602352e827a1bd6b` |

The selected device remained NVIDIA GB10, UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71`, driver `580.142`. Read-only pre/post build process snapshots show the user's zkDL runner still active; no user process was stopped or altered. See [`prebuild-process-snapshot.txt`](prebuild-process-snapshot.txt), [`postbuild-process-snapshot.txt`](postbuild-process-snapshot.txt), and the GPU snapshots.

## Validation boundary

No formal dataset run, full dataset preprocessing, SRS or PK-hash operation, Prove/Verify session, or GPU-kernel execution was performed. The checks establish the lookup cache's field-level equivalence, a small existing proof/verifier round trip across three protocols, and the current official GPU release build identity; they do not measure formal PPI performance or establish full-dataset completion.
