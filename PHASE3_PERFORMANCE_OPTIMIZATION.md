# Phase 3 Performance Optimization

Last updated: 2026-07-22 (Asia/Shanghai)

## Status and scope

The bounded representative-path objective for this round is met.  The median
of `commitment_time + prove_time + verify_time` fell by 92.029% against the
single frozen baseline, and the median wall time fell by 91.967%.  This is not
a five-dataset result and does not authorize or claim completion of the full
CPU/GPU experiment matrix.

No complete Cora, Citeseer, Pubmed, PPI, or ogbn-arxiv proof was run.  No large
SRS was generated.  All measurements used the existing
`unit_security_srs_v1`, the Release build, CPUs 0-19, and the production
`exclusive_commit_prove_verify_v2` timing contract.

The representative entry was
`gatzk_proof_codec_stage_bench generate configs/unit_security_regression.cfg`.
It executes the production context, trace, dynamic commitments, quotient,
point-group openings, proof assembly, in-memory canonical serialization,
strict parsing, and verifier.  Its input is bounded, but it uses the same
algorithms, proof objects, KZG keys, codec/parser, and verifier entry points as
the five formal configurations.  Every sample was a fresh process with the
same config, seed policy, backend, CPU affinity, and SRS.

## Frozen baseline

Exactly three runs were taken before the first production optimization in this
round.  The baseline was never rolled forward.

| Metric | Median | Min | Max |
| --- | ---: | ---: | ---: |
| `commitment_time` (ms) | 21184.461 | 21143.730 | 21302.674 |
| `prove_time` (ms) | 24829.257 | 24761.946 | 24991.354 |
| `verify_time` (ms) | 649.522 | 627.620 | 665.983 |
| Three-time sum (ms) | 46612.391 | 46600.606 | 46943.551 |
| Wall time (ms) | 48876.405 | 48817.595 | 49185.314 |
| Peak RSS (KiB) | 793004 | 792960 | 793140 |
| `proof_size_bytes` | 250449 | 250439 | 250451 |

The mutually exclusive baseline candidates covered 91.516% of the formal
three-time sum:

| Candidate | Median (ms) | Share of formal sum |
| --- | ---: | ---: |
| Dynamic commitments/MSM | 17046.960 | 36.57% |
| Quotient non-commitment materialization | 16764.043 | 35.96% |
| Domain plus external opening witnesses | 5402.863 | 11.59% |
| Derived commitment points | 3444.044 | 7.39% |

## Retained production changes

### Shared arithmetic backend

The vendored MCL archive was previously compiled without any optimization
level even though the main project was a Release build.  Its formal CMake rule
now supplies `-O3 -DNDEBUG`.  This is the largest retained change and directly
accelerates field arithmetic, curve multiplication/MSM, opening witnesses, and
pairing without changing their inputs, equations, or serialized results.

MCL LLVM, bint assembly, and MCL's native MSM remain disabled.  On this aarch64
host the bundled native MSM implementation is AVX-512-only, and the LLVM
material requires a clang toolchain that is not installed.  The retained build
therefore uses the existing portable backend and current dependencies.

### Dynamic commitments/MSM

- MSM fanout for 256-or-larger effective inputs is raised to at most 16 workers.
- Batch scheduling uses effective nonzero/folded work rather than padded vector
  length, and cached/small batches may use bounded outer parallelism.
- A per-proof exact semantic evaluation-commitment cache shares only
  `sum(value_i * L_i)` for byte-for-byte equal field vectors on the same domain
  and Lagrange SRS.  Hash matches are confirmed by full vector equality.
- Every polynomial still adds its own independent `Z_D * R` blinding
  commitment.  The cache is execution-only, never serialized, and does not
  cross proofs, challenges, owners, witnesses, or opening groups.

### Quotient materialization and `t_edge`

- The bounded `t_edge` point evaluator now exposes enough independent work to
  all 20 CPUs instead of producing only two oversized tasks.
- Derived coset material that is already computed by the production source of
  truth is admitted to the fast read-only lookup table.
- Exact-equal semantic evaluation columns share one base coset transform.  Each
  label then adds its own zero-domain blinder pointwise, preserving its exact
  private polynomial.  Base material is released after the group is consumed.
- Debug quotient self-checks compared the optimized materialization with the
  canonical evaluator before the non-debug final measurements.

### Measurement support

The production-path benchmark now reports dynamic duplicate diagnostics and
mutually exclusive quotient subcomponents.  The MSM stage benchmark accepts an
explicit SRS path/type/id, allowing the same test SRS as the formal-path canary.
These additions do not create a benchmark-only execution route.

## Reverted or rejected schemes

- Lowering the custom Pippenger threshold from 262144 to 256 scalars regressed
  the 1024-scalar case from about 109 ms to 147-152 ms and the 4096-scalar case
  from about 278 ms to 406-412 ms.  The threshold change and all associated
  experimental code were fully reverted.
- A final-commitment-point duplicate cache was rejected before implementation:
  only 9 duplicates totaling about 15 ms existed, so it could not materially
  move the formal sum.
- Earlier persistent worker-pool, forced all-core batch, folded cross-point
  opening, parser cache/direct-decode, and generic `t_edge` memoization
  candidates remain removed.  Their premises did not change in this round.

## Final three-run result

The final state was measured three times under the frozen conditions.

| Metric | Baseline median | Final median | Final min-max | Change |
| --- | ---: | ---: | ---: | ---: |
| `commitment_time` (ms) | 21184.461 | 1617.852 | 1595.454-1634.098 | -92.36% |
| `prove_time` (ms) | 24829.257 | 2005.447 | 1951.463-2106.865 | -91.92% |
| `verify_time` (ms) | 649.522 | 76.822 | 76.069-77.142 | -88.17% |
| Three-time sum (ms) | 46612.391 | 3715.614 | 3623.739-3801.860 | -92.029% |
| Wall time (ms) | 48876.405 | 3926.298 | 3837.015-4010.134 | -91.967% |
| Peak RSS (KiB) | 793004 | 803772 | 803580-804260 | +1.36% |
| `proof_size_bytes` | 250449 | 250435 | 250431-250453 | recorded only |

The formal speedup is 12.545x and the wall-time speedup is 12.448x.  The
formal target was at most 11653.098 ms; the final median is 3715.614 ms.

The baseline median formal/wall difference was 2264.015 ms.  The final median
difference is 210.684 ms.  Both are outside the three formal metrics and are
attributable to context/SRS loading, scheduling, logging, and benchmark
harness work.  No work was moved across a formal timing boundary.

The latest mutually exclusive candidates still cover 85.218% of the formal
sum: dynamic commitment MSM 1343.849 ms (36.16%), quotient non-commitment
materialization 1046.301 ms (28.16%, including `t_edge` 938.105 ms), domain
plus external opening witnesses 572.141 ms (15.40%), and derived commitments
204.075 ms (5.49%).  The 75% termination condition is already satisfied, so
the alternate requirement to exhaust three scheme classes for every remaining
10% hotspot does not apply.

## Verification

- CPU Release full build: passed.
- CUDA Release full build on NVIDIA GB10: passed.
- CPU and CUDA `ctest`: `gatzk_tests`, `static_secure_scan`, and
  `runner_semantics` all passed.
- `GATZK_RUN_UNIT_SECURITY_TESTS=1 gatzk_tests`: passed.
- Three final generation runs each completed canonical serialization, strict
  parse, and verification with `private_plain_value_count=0` and
  `verifier_reads_witness=false`.
- Independent strict parse and verifier-only runs: three repeats each, passed.
- CUDA bounded exact-match stages: all passed; 90 kernel invocations were
  observed.
- Failed Pippenger code is absent and `git diff --check` passes.

The proof schema, transcript, canonical codec, parser, SRS structure, and KZG
opening equations did not change.  The existing small SRS, compiler artifact,
and manifest remain valid.  Formal dataset v2 SRS files are still absent and
must later be generated only through the setup-only flow; this round neither
requires nor authorizes their generation.

## Remaining real boundaries

- These numbers are a bounded production-path canary, not a complete-dataset
  performance claim.  Full-scale cache lifetime, memory, and scheduling remain
  unmeasured until user-authorized formal rows exist.
- CUDA still does not implement curve MSM, pairing, NTT/IFFT, quotient
  materialization, or opening witnesses.  Current exact CUDA coverage remains
  limited to the existing field hotspots.
- Further portable aarch64 curve gains would require a materially different
  fixed-base/MSM implementation or an available LLVM/aarch64 backend.  The
  bundled AVX-512 MCL MSM cannot run on this hardware.
- The five-dataset CPU/GPU experiment matrix and formal v2 SRS generation
  remain explicitly user-authorized work.
