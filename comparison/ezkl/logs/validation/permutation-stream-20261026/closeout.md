# Permutation Z-stream closeout

Date: 2026-10-26

## Change

[`evaluation.rs`](../../../vendor/halo2/halo2_proofs/src/plonk/evaluation.rs) now streams the weighted permutation-H fold across product sets. The single-set branch keeps its previous evaluation order. For `m > 1`, the count of **permutation Z extended-domain transforms** changes from `2m+2` to `m`; this count does not describe all proof transforms or imply that total proof time is halved. The weighted terms, Horner order, split contributions, and challenge handling remain equivalent to the prior evaluation. The full mapped working set peaks at `4N` values, including source and sigma, matching the prior bound.

## Focused CPU checks

Both checks ran once with the pinned `nightly-2025-12-01`, isolated lockfile `/tmp/ezkl-perm-stream-20261026/Cargo.lock`, CPU-only Halo2 features `mv-lookup,circuit-params,parallel-poly-read,formal-permutation-spill`, and `CARGO_BUILD_JOBS=2` / `RAYON_NUM_THREADS=2`. `ENABLE_ICICLE_GPU`, `CUDA_VISIBLE_DEVICES`, and `ICICLE_DEVICE_ID` were unset.

| Check | Result | Runtime |
| --- | ---: | ---: |
| `plonk::evaluation::tests::native_permutation_weighted_fold_stream_matches_legacy_horner` | 1 passed, 0 failed | 0.37 s |
| `tests/plonk_api.rs::plonk_api` | 1 passed, 0 failed | 0.48 s |

The first oracle compares the streamed production row helper against the legacy Horner computation for 0, 1, 2, 3, 16, and 130 product sets and challenges `y = 0`, `1`, and `73`. The existing K=5 `plonk_api` integration test exercises 12 equality columns at degree 4, producing six permutation sets; its one test body completes and verifies three small round trips: GWC/KZG, SHPLONK/KZG, and IPA/Pasta.

The first oracle compile exposed three errors in the new test helpers only. The RNG iterator closures now reborrow the mutable RNG, and the zero predicate explicitly converts `Choice` to `bool`; the production arithmetic was not changed in response. The corrected oracle passed once.

## Official build and identities

The official GPU release build completed in 3m42s. Native prerequisite authentication passed and reused the authenticated native backend. The build completed CLI-contract and source-identity checks; it did not run native CUDA setup or a CUDA kernel. Build output is in [`official-gpu-build-full.log`](official-gpu-build-full.log). The initial invocation without the script's required `--python` and identity arguments exited at usage before building; the successful run reused the previously recorded complete official command and identity values.

Post-build checks matched the receipt at [`build_receipt.json`](../../../artifacts/build/gpu/build_receipt.json):

| Identity | SHA-256 |
| --- | --- |
| Source tree | `33ad996ad3f715c22a5539c32ccc3eb8f94c13a3b559800d9ca54a76c72f41e5` |
| GPU release binary | `59cd4fec3c050104c80b0bfa299d3433743b3ca400e02eeb20feee6a7b6d1976` |
| Native backend manifest | `4b814782da97ae253c4a57fb577a20c2fae71817ea4ca331146ab8e7f17d4674` |
| Resource manifest | `fdf1c7729efed73266beff9111765549120ea0b32ecf58e212bb317ba5` |
| Adapter CLI contract | `df6266c06496ee7a36586b94bd79d7955d92612667c16315015476bfcbddd2d2` |
| Build environment | `2de96669cf8a04268cb8da23719c3eecd68d25cf12889ffe602352e827a1bd6b` |

Both native library digests also matched the manifest: `libingo_curve_bn254.a` `f3aa2afea74e467e61e7283a2ba75c472054458d51f84f581d373bcda7757f60`; `libingo_field_bn254.a` `99d3fc55944526354b95914c2b6711288f022e24d3f0b40dd5ea4b576fb22317`. The adapter digest uses the official build's SHA-256 over the raw CLI-contract file bytes.

The post-build read-only GPU snapshot showed the user's zkDL `gat_runner` (PID 3809019, 671 MiB) on GB10 UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71`, with `formal_runner.py` PID 3774319. GPU utilization was 96%; memory totals were unavailable in the query. No EZKL formal runner was found. No process was started or stopped by this validation.

## Evidence boundary

No formal dataset cell, dataset-level runtime comparison, or GPU-kernel execution was run. These checks establish field-level equivalence, small proof/verifier compatibility, and the release build identity; they do not establish PPI or OGBN-Arxiv completion or a measured end-to-end speedup.
