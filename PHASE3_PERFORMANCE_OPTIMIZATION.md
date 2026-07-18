# Phase 3 Performance Optimization

Last updated: 2026-07-18 (Asia/Shanghai)

## Status and evidence boundary

The current non-full-dataset Phase 3 scope is complete: every top-level formal
stage has been audited and every known major hotspot has a retained change, a
measured rollback, a low-share conclusion, or a conclusion that now requires a
user-authorized full-dataset run. No full-dataset proof or verification was run
during this work.

All measurements below are either small formal-call-chain evidence or bounded
stage-benchmark evidence. They are not paper results and are not substitutes
for full Cora, Citeseer, Pubmed, PPI, or ogbn-arxiv measurements. Historical
run directories were used only to form hypotheses.

## Workspace and environment

- Starting branch/commit: `main`,
  `fb5bb3b3cd27d572431c7267b72d0cbc0f16114b`.
- The worktree was clean at Phase 3 start. Current uncommitted changes are the
  Phase 3 implementation and documentation changes.
- CPU build: CMake Release, `-O3 -DNDEBUG`, tests enabled, vendored mcl,
  OpenSSL and pthreads.
- CUDA build: the same Release configuration with
  `GATZK_ENABLE_CUDA_BACKEND=ON`.
- Host: aarch64, 20 logical CPUs; NVIDIA GB10, driver 580.142, unified
  addressing available.
- CUDA remains a partial-hotspot backend. Curve MSM, pairing, NTT/IFFT,
  quotient materialization and opening witnesses are CPU work.

## Formal timing and backend reporting

The formal executable now records the required boundaries directly:

- `commitment_time_ms = commit_dynamic_ms + quotient_bundle_pack_ms`; dynamic
  trace commitments include binding/helper/shift objects, while the quotient
  bundle includes quotient and mask commitments.
- Trace/forward preparation is outside `prove_time_ms`. The measured span
  starts after dynamic witness construction and includes dynamic commitments,
  quotient/opening/proof assembly, and canonical serialization.
- `verify_time_ms` starts before strict parsing consumes the first byte and ends
  after the verifier returns.
- `proof_size_bytes` is the actual canonical serialized byte-vector length.
- Parse and serialization are recorded separately, and both prove and verify
  accounting gaps are checked.
- A requested CUDA run reports the actual kernel count. An input below every
  dispatch threshold is reported as `none_on_this_input`, not as GPU-executed.

The final CPU small-input manifest has zero prove and verify accounting gaps.
The final CUDA small-input manifest requested CUDA but executed zero kernels;
its status correctly records CPU fallback for that input.

## Latest small formal-call-chain decomposition

Evidence path:
`build/unit_security_regression/phase3_final_cpu/warm/`. The configuration is
`configs/unit_security_regression.cfg`; it is a three-node/six-edge security
fixture, not a full dataset.

| Stage | Time (ms) | Conclusion |
| --- | ---: | --- |
| commitment total | 18613.225 | Dynamic curve MSM dominates |
| dynamic commitments | 17893.278 | Retained MSM and batch scheduling are active |
| dynamic commitment MSM | 17818.578 | Remaining cost is primarily curve arithmetic |
| polynomial materialization | 68.365 | Low share |
| dynamic FFT | 0.000 | No current formal-path hotspot on this input |
| quotient commitment bundle | 719.947 | Low relative share |
| prove total | 46496.923 | `t_edge`, commitments and openings dominate |
| quotient build | 21336.997 | Mostly `t_edge`; includes quotient commitment |
| `t_edge` quotient | 20370.656 | Top quotient hotspot; multihead-specific work is stage-tested separately |
| `t_FH` | 4.546 | Low share |
| `t_N` | 17.429 | Low share |
| `t_in` / `t_d_h` / `t_cat` / `t_C` | 9.011 combined | Low share |
| domain opening | 5201.739 | Shared weights and bounded claim parallelism retained |
| external opening | 2029.492 | Bounded point-group fan-out retained |
| canonical serialization | 34.516 | Low share and no observed scaling concern |
| verify total, including parse | 4926.633 | Strict parse and domain pairings dominate |
| strict parse | 1798.218 | Required canonical mcl G1 decoding dominates |
| verifier metadata | 476.781 | Duplicate compiler-artifact validation removed |
| verifier transcript | 148.057 | Duplicate canonical metadata construction removed |
| verifier domain openings | 2338.299 | Independent two-point pairing concurrency retained |
| verifier external fold | 101.286 | Independent point groups verified concurrently |
| verifier quotient recomputation | 56.384 | Low share |
| verifier miscellaneous | 7.608 | Low share |

The proof was accepted, reference consistency and tamper rejection passed,
private plain-value count was zero, and the canonical proof was 242789 bytes.
Proof-size differences between randomized runs are not an optimization claim.

For orientation only, the first current-code small formal baseline after the
measurement repair was 31802.932 / 89906.867 / 9744.221 ms for
commit/prove/verify. The latest small-input result is respectively -41.5%,
-48.3%, and -49.4%. These deltas are local Phase 3 evidence, not full-dataset
performance claims.

The priority for user-run scale confirmation is now: (1) multihead `t_edge`
and dynamic curve MSM, (2) domain/external opening construction and domain
pairings, (3) strict parsing, and (4) whether eligible CUDA field/lookup work
has enough end-to-end weight to offset partial-backend CPU fallback. The other
quotients, materialization, serialization and verifier quotient recomputation
are currently low-share controls rather than active optimization targets.

## Retained optimizations and formal-path connections

### CPU MSM and commitment scheduling

`src/crypto/curve.cpp` uses four workers instead of two for MSM sizes 1024
through 8191. The 8192+ route is unchanged.

| MSM case | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| full-width 1024 | 194.923 | 114.428 | -41.3% |
| full-width 4096 | 492.376 | 277.423 | -43.7% |
| sparse small-scalar 6554 | about 68.4 | 51.383 | about -24.9% |

The generic KZG outer scheduler also uses finer bounded fan-out for batches of
many independent objects. Same-binary commit-only medians at domain 4096:

| Objects | Before (ms) | After (ms) | Change |
| ---: | ---: | ---: | ---: |
| 32 | 1740.625 | 1036.805 | -40.4% |
| 64 | 2012.147 | 1891.160 | -6.0% |
| 256 | 6301.909 | unchanged route | no regression |

Small batches up to 20 objects use the separately measured low-work threshold;
three 4096-scalar objects changed from about 811.565 to 374.2 ms and four from
1089.667 to 376.8 ms. These routes are used by dynamic, binding/helper/shift,
and quotient commitments through the shared KZG commit path.

### `t_edge` quotient

Fixed signed-division label plans are cached without caching proof-dependent
values. The eight-hidden-head stage median changed from about 1556.142 to
1499.612 ms (-3.6%). Direct committed/spilled coset vectors are prewarmed and
looked up without repeated cache locking; derived expressions continue to be
evaluated from their blinded source cosets.

The formal materializer uses one task per 8192 `t_edge` points rather than per
32768 points. A Cora-shaped 512-point stage changed from about 399.5 to 230.1
ms with the retained worker count, while the PPI-shaped one-head case changed
from about 1048.2 to 821.9 ms. On an earlier small formal checkpoint,
`t_edge` changed from 34959.045 to 20412.171 ms (-41.6%).

The quotient accumulator now optionally reuses the public powers
`[1, alpha_quot, alpha_quot^2, ...]`, eliminating one identical field
multiplication per constraint term. Same-binary, 20-worker results at 2048
sample points were:

| Shape | Incremental powers (ms) | Precomputed (ms) | Decision |
| --- | ---: | ---: | --- |
| 8 hidden heads | 964.192 median | 722.482 median | retain, -25.1% |
| 1 hidden head | 824.899 median | 832.227 median | do not dispatch, +0.9% |

The formal materializer therefore dispatches precomputed powers only for four
or more total heads. Both modes produced identical accumulators at every
representative point. The small security fixture has two total heads and
therefore exercises the original incremental route; the multihead benefit must
still be confirmed by the user's full multihead run.

### KZG opening construction

Domain claims at a common domain/point share barycentric weights. Every
individual claimed value is still checked against its exact polynomial, with
bounded parallel claim checking. A new regression rejects an incorrect
individual claim.

For 910 terms on the unit edge domain, `open_batch` changed from 16056.433 to
about 3075 ms (-80.8%), and its precompute component from 13146.758 to about
179 ms (-98.6%). The first retained small formal checkpoint changed domain
opening from 18074.337 to 5014.659 ms (-72.3%).

External groups use no more than half the logical CPU count as outer workers,
with inner MSM caps derived from the same budget. Eight-group median changed
777.767 to 710.408 ms (-8.7%); 17-group median changed 1427.663 to 1377.437 ms
(-3.5%). The small formal external-opening component changed 2498.374 to
2140.057 ms at the retained checkpoint.

### Verifier parsing, metadata, transcript and pairings

Expected dynamic/quotient commitment labels are built once before order
validation instead of being rebuilt for every proof entry. The verifier
validates the compiler artifact once, then passes that result into canonical
metadata construction. Transcript replay directly absorbs the already
validated proof metadata rather than constructing an unused canonical copy.

On the canonical 242784-byte stage proof, these metadata/transcript changes
reduced parser-excluded verification from 4264.926 to 3298.715 ms before the
later pairing changes. The corresponding small formal metadata and transcript
components reached 476.620 and 328.689 ms at that checkpoint; the final run
measured 476.781 and 148.057 ms.

The two canonical domain-opening points retain separate pairing equations and
acceptance bits but are verified concurrently with a two-worker cap. The
parser-excluded verifier median changed from 7469.614 to 5205.257 ms at the
first checkpoint. Model-binding domain-divisibility equations are scheduled as
independent bounded tasks, and their unchanged equation uses the existing
mixed-prepared pairing product when the prepared verifier key is available.
The final parser-excluded seven-run median is 3138.048 ms.

External-fold point groups retain separate equations and are checked
concurrently. The 17-group stage median changed from 379.444 to 38.997 ms
(-89.7%); the small formal component changed from 1066.035 to about 103 ms.

### CUDA partial-hotspot route

`src/algebra/eval_backend.cpp` sends existing full-domain evaluation columns
directly to the columnar CUDA kernel on unified-addressing devices. Short
columns keep the exact zero-padded fallback route. Lookup histogram dispatch
starts at 16384 indices because 4096 is slower on this device.

Final exact-match stage results:

| CUDA stage | CPU (ms) | CUDA (ms) | Result |
| --- | ---: | ---: | --- |
| barycentric, domain 65536 | 88.582 | 3.188 | exact |
| formal weighted route, 32 columns | 56.805 | 0.788 | exact |
| formal weighted padded route | 55.527 | 1.595 | exact |
| histogram 4096 / 1024 | 0.330 | 0.767 | retain CPU dispatch |
| histogram 16384 / 1024 | 1.341 | 0.780 | dispatch CUDA |
| histogram 65536 / 4096 | 5.400 | 1.295 | exact |
| histogram 1M / 4096 | 46.823 | 2.206 | exact |
| histogram 4M / 65536 | 190.065 | 35.307 | exact |

The direct columnar formal route was about 1.528 ms before the zero-copy
selection and is now 0.788 ms (-48.4%). The final CUDA security fixture was
`VERIFY_OK` with CPU/GPU reference consistency and tamper rejection, but its
zero kernel count means its timing is a CPU-fallback measurement, not evidence
of end-to-end GPU speedup.

## Reverted and cleaned experiments

- Lowering the Pippenger threshold to cover an 8192-point full-width MSM
  regressed 503.255 to about 721.4 ms. It was fully restored.
- A generic cached head-prefix label layer changed the `t_edge` stage from
  1499.612 to 1500.038 ms. It was removed.
- Generic per-point `t_edge` value memoization was rejected after an audit
  found 3325 dependency calls but 3192 unique label/rotation keys per point;
  the roughly 4% duplicate rate did not justify a proof-wide cache.
- A G1 decode cache changed strict parse from 2235.925 to 2257.244 ms. A
  pointer/zero-allocation decode variant measured 2257.351 ms. Both parser
  experiments and their APIs were removed; the final retained parser median is
  1775.351 ms on seven runs.
- Running all seven domain-bundle KZG verifications concurrently improved an
  isolated median from 4302.588 to 3882.475 ms, but the formal path regressed
  6527.966 to 6575.595 ms and domain-opening wall time did not improve because
  of pairing contention. The outer concurrency code was removed.
- Forced alpha-power precomputation on the one-head representative shape
  regressed its median by 0.9%; the formal dispatcher excludes that shape.
- Temporary A/B gates, diagnostic-only switches and failed implementation
  branches from these attempts were removed.
- The zero-byte failed `perf` profiler artifact
  `build/phase3_bench/proof_codec/perf_parse_baseline.data` was deleted.
- Preliminary alpha-power candidate/routing CSVs were deleted after the
  same-binary `off`/`on` evidence superseded them; only the baseline and final
  decision-grade A/B files remain in the ignored build evidence directory.
- The CUDA stage executable was initially added unconditionally; a CPU-only
  full build exposed the undefined CUDA symbol. CMake now creates that target
  only in CUDA-enabled builds, and both CPU full build and CUDA target build
  pass.
- The default CTest registration previously entered a large-MSM test even
  though the default unit SRS is intentionally smaller than its first case.
  That regression is now explicit opt-in via `GATZK_RUN_LARGE_MSM_TESTS=1`
  with a sufficiently large overridden test SRS; bounded default CTest remains
  runnable without generating large SRS material.

## Top-level audit conclusions

| Formal area | Current status |
| --- | --- |
| dynamic, binding/helper and shift commitments | MSM/outer scheduling optimized; full scale requires user run |
| quotient commitments and masks | audited, low share on bounded formal input |
| `t_edge` | worker, fixed-label, coset lookup and multihead alpha-power changes retained |
| `t_FH`, `t_N`, `t_in`, `t_d_h`, `t_cat`, `t_C` | measured low share; no current expansion signal |
| polynomial materialization/copies | 68.365 ms, low share |
| base/derived cosets and IFFT | included in quotient timings; `t_edge` prewarm remains dominant optimized route |
| FFT/NTT/IFFT outside quotient | no measured current hotspot; no new GPU dependency introduced |
| domain and external openings | shared work and bounded scheduling retained |
| CPU KZG/MSM | optimized; remaining dominant arithmetic requires full-scale evidence |
| GPU MSM/FFT | not implemented; adding a heavyweight dependency was outside authorization |
| verifier MSM/pairing | two-point, external-group and mixed-prepared changes retained |
| strict parser | two reasonable optimizations regressed and were removed |
| serialization | audited low share |
| CUDA allocation/H2D/kernel/sync/D2H | bounded exact-match stage audited; size-aware thresholds retained |
| large-object lifetime/cache/copies | no unbounded new cache; proof-local/shared-weight caches are bounded by the operation |
| thread scheduling/oversubscription | outer and inner worker caps applied at MSM/opening/pairing boundaries |

No known high-share bounded-input path remains without a conclusion. The
remaining uncertainty is scale validation on full datasets, especially
multihead `t_edge`, large dynamic MSM batches, SRS residency, and whether the
partial CUDA stages have enough weight to change end-to-end time.

## Verification completed

- CPU and CUDA Release builds succeeded; the CPU default all-target build also
  succeeded after the CUDA benchmark target guard.
- Default CPU and CUDA secure regression suites passed with large-SRS and full
  dataset tests deliberately skipped.
- The expanded unit coefficient-basis closure regression passed with
  `GATZK_RUN_UNIT_SECURITY_TESTS=1`.
- Small CPU and requested-CUDA formal call chains returned `VERIFY_OK`, passed
  reference consistency, canonical parse/round-trip and tamper rejection, and
  contained zero private plain values.
- CUDA hotspot stage: all CPU/GPU comparisons were exact.
- Proof codec: seven strict parses passed at 1775.351 ms median; seven
  parser-excluded verifier runs passed at 3138.048 ms median.
- `python3 tools/security/static_scan_no_tau.py` passed. Missing large-dataset
  SRS directories were reported as skipped, not validated.
- `git diff --check` passed.

## Remaining user-authorized full-dataset handoff

The user must first ensure every dataset and checkpoint is present. Cora and
ogbn-arxiv have automatic data preparation disabled. Only the Cora formal SRS
currently exists; the other formal SRS directories must be produced by the
user from the existing compiler artifacts. The ceremony tool refuses to
overwrite an existing output directory.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gatzk_srs_setup gatzk_run gatzk_tests -j 8
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release \
  -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda --target gatzk_run gatzk_tests -j 8

./build/gatzk_srs_setup ceremony \
  artifacts/srs/citeseer_formal_srs_v1 \
  artifacts/srs/citeseer_compiler_artifact_v1.txt citeseer_formal_srs_v1
./build/gatzk_srs_setup ceremony \
  artifacts/srs/pubmed_formal_srs_v1 \
  artifacts/srs/pubmed_compiler_artifact_v1.txt pubmed_formal_srs_v1
./build/gatzk_srs_setup ceremony \
  artifacts/srs/ppi_formal_srs_v1 \
  artifacts/srs/ppi_compiler_artifact_v1.txt ppi_formal_srs_v1
./build/gatzk_srs_setup ceremony \
  artifacts/srs/ogbn_arxiv_formal_srs_v1 \
  artifacts/srs/ogbn_arxiv_compiler_artifact_v1.txt ogbn_arxiv_formal_srs_v1

python3 tools/security/static_scan_no_tau.py --require-all-configured-srs
```

Recommended execution order is smallest/known-risk first, one row at a time:
Cora CPU, Cora GPU, Citeseer CPU/GPU, Pubmed CPU/GPU, PPI CPU/GPU, then
ogbn-arxiv CPU/GPU. The first command builds and runs bounded preflight tests;
`--skip-full-security-tests` avoids executing a duplicate complete Cora proof
inside the test preflight. Subsequent commands can use `--skip-build
--skip-tests` after the source tree and binaries are frozen.

```bash
python3 scripts/run_secure_main_experiments.py \
  --datasets cora --backends cpu --skip-full-security-tests \
  --skip-export --timeout-seconds 21600

python3 scripts/run_secure_main_experiments.py \
  --datasets cora --backends gpu --skip-build --skip-tests \
  --skip-export --timeout-seconds 21600

# Repeat the second form for citeseer, pubmed, ppi, and ogbn-arxiv,
# selecting one backend per invocation.

# After all ten rows are valid, this resumes strict-valid rows and performs
# the final strict export without recomputing them.
python3 scripts/run_secure_main_experiments.py \
  --skip-build --skip-tests --timeout-seconds 21600
```

Do not use `--force` when resuming unless a deliberate rerun is wanted. A
SIGINT/SIGTERM or timeout can leave a noncanonical partial run directory; the
runner archives stale/partial state and only resumes a row whose strict
manifest, hashes and gates are valid. Preserve runner logs and do not treat a
partial `benchmark.txt` as accepted evidence.

Resource cautions:

- A previously interrupted Cora CPU security run had already reached more than
  three hours and about 35 GiB RSS before proof construction returned; the
  final rows may exceed this.
- PPI and ogbn-arxiv can require substantially larger SRS storage, RAM and wall
  time. Generate and validate one SRS at a time and verify free disk first.
- The CUDA backend does not accelerate curve MSM, pairing, NTT/IFFT, quotient
  materialization or opening witnesses, so GPU rows can remain CPU- and
  memory-bound.
- Avoid running CPU and GPU rows concurrently on this host; they share CPU,
  memory bandwidth and SRS residency, invalidating timing comparability and
  increasing OOM risk.

**Full-dataset proof validation was not executed; it must be decided and run by
the user.**
