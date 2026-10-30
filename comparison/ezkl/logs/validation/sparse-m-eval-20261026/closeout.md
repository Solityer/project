# Sparse `m(x)` evaluation closeout

This closeout covers the bounded sparse multiplicity evaluation path in EZKL's native Halo2 lookup prover. For Compact lookup sources with at most 32 actual nonzero rows, `Committed.evaluate` evaluates `m(x)` directly from indexed entries instead of materializing an N-element Lagrange polynomial and running the existing IFFT/Horner path. Dense sources and support above 32 retain the old materialize-and-Horner fallback. The change does not alter `phi`, transcript order, H, SHPLONK, or the later `m` consumers. This is an operation-count expectation, not a measured end-to-end speedup claim.

The sparse path validates every codebook winner row, including zero-count rows, before empty/support/point early returns, matching the old materializer's metadata contract. If a zero-count row overlaps a nonzero winner row, it returns to the dense path so the legacy assignment overwrite behavior is preserved. The final numerical oracle calls the real `materialize_coeff(Multiplicity)` implementation and then Horner evaluation; its separate commitment oracle retains the old dense scatter semantics.

## Bounded validation

- `sparse_m_commit_matches_legacy_for_kzg_and_ipa`: 1 passed, 0 failed. This is the final post-contract-repair run; it compares the production sparse evaluation and commitment helpers against the real legacy source and KZG/IPA outputs. See [`sparse_m_commit_matches_legacy_for_kzg_and_ipa.post-contract-fix.log`](sparse_m_commit_matches_legacy_for_kzg_and_ipa.post-contract-fix.log).
- `plonk_api`: 1 passed, 0 failed. See [`plonk_api.post-contract-fix.log`](plonk_api.post-contract-fix.log).
- The bounded Python capacity script called the existing `_capacity_admission` fixture and production `_native_capacity_requirements`. It confirmed sparse evaluation scratch of 3,912 bytes is a single-task bound inside the evaluation transient maximum; the fixture's existing H task plus source window remains the maximum at 295,328 bytes. The returned total evaluation peak exactly matched the producer formula. Sparse prepare and grand-sum commit task bounds remained 3,248 and 512 bytes. See [`native-capacity-proof-eval.py`](native-capacity-proof-eval.py) and [`native-capacity-proof-eval.log`](native-capacity-proof-eval.log).
- The broad legacy capacity unittest was not rerun; its unrelated Setup expectation was already known to be stale. This run checks only the changed evaluation envelope and the unchanged prepare/commit values.

## Official GPU release build and identities

The complete official GPU release build reused the archived command with `--python`, absolute Cargo/Rust toolchain and compiler paths, selected-device JSON, and canonical build-environment JSON. It completed with exit code 0 in 225.834 seconds and passed the official CLI-contract and source-identity checks. Full command and output are in [`official-gpu-build-full.log`](official-gpu-build-full.log).

All nine post-build identity checks passed in [`postbuild-identity-verification.log`](postbuild-identity-verification.log): source tree `71efc8f837f46df82dfe90d32dd3397c6d237824e788685d9798a3255701f16a`, binary `223a3f2bee45f2e1a2c985625561f8467629d4ec03425fd6f5891b5de9594154`, native manifest, both native libraries, resource manifest, adapter contract raw bytes, canonical build-environment JSON, and its embedded digest. The active native execution policy remains `halo2_icicle-bn254-gpu-hgraph-fused-ntt-16-25-msm-lagrange-resident-f4-sparse-m-cpu32-v6`.

The pre-build and post-build read-only snapshots both show physical GB10 UUID `GPU-3cde57ba-ae43-4452-bd59-7b156b36db71`; snapshots are in [`prebuild-gpu-snapshot.csv`](prebuild-gpu-snapshot.csv), [`prebuild-process-snapshot.txt`](prebuild-process-snapshot.txt), [`postbuild-gpu-snapshot.csv`](postbuild-gpu-snapshot.csv), and [`postbuild-process-snapshot.txt`](postbuild-process-snapshot.txt). The user's zkDL PPI proof remained active through the build (CPU proving with the Hyrax IPA GPU-offload stage, 206 MiB reported on the device). No user process was stopped or modified.

No formal dataset experiment, dataset preprocessing, CUDA kernel, SRS load, or PK hash was run. The official command built and identity-checked the release binary only.
