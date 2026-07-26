# Phase 3 Performance Optimization

Last updated: 2026-07-26 (Asia/Shanghai)

## Status and measurement contract

This record describes only bounded production-path work. No complete Cora,
Citeseer, Pubmed, PPI, or ogbn-arxiv proof and no five-dataset CPU/GPU matrix
was run. No SRS was generated or modified.

The representative entry is:

```text
taskset -c 0-19 build/gatzk_proof_codec_stage_bench generate \
  configs/unit_security_regression.cfg <proof.bin> 1
```

Each fresh process uses the Release CPU build and the existing
`unit_security_srs_v1`. The entry executes the production context, trace,
dynamic commitments, quotient materialization, point-group witnesses, proof
assembly, in-memory canonical serialization, strict parsing, and verifier.
All timing uses `exclusive_commit_prove_verify_v2`; the formal acceptance
metric is the sum of `commitment_time`, `prove_time`, and `verify_time`.

The prior 75% reduction target was not reached. The retained 2026-07-22 median
is 26.207% below that round's frozen baseline, with a corresponding 24.859%
wall-time reduction. The continuation at the end of this document freezes a
new current-source baseline and reports only newly measured changes; it does
not count the earlier 26.207% as a new result and does not claim that the
complete experiment matrix was run.

## Frozen baseline

Exactly three runs were taken before this round's production changes. The
baseline was never rolled forward.

| Metric | Median | Min | Max |
| --- | ---: | ---: | ---: |
| `commitment_time` (ms) | 1682.144 | 1677.759 | 1696.085 |
| `prove_time` (ms) | 2143.149 | 2101.188 | 2146.575 |
| `verify_time` (ms) | 76.633 | 75.903 | 77.822 |
| Three-time sum (ms) | 3897.541 | 3873.176 | 3906.541 |
| Program wall time (ms) | 4110.572 | 4085.059 | 4115.829 |
| Peak RSS (KiB) | 803868 | 803760 | 804364 |
| `proof_size_bytes` | 250447 | 250440 | 250448 |

Baseline diagnostic medians included dynamic commitment MSM 1411.456 ms,
`t_edge` 945.998 ms, and domain plus external opening 607.476 ms.

## Retained production changes

### Common MSM and commitment paths

- Replaced repeated per-call `std::async` creation in the common curve layer
  with bounded persistent inner-MSM and object-batch worker pools. The pools
  are separate so an object worker can wait for an inner MSM without consuming
  the same queue needed to complete it.
- Split large independent MSM work into twice as many chunks as active workers
  to reduce long-tail imbalance on this heterogeneous 20-core aarch64 CPU.
- Routed dynamic commitment batches and same-base multiplication through the
  shared bounded workers. Results are accumulated in original index order, so
  commitment and transcript order are unchanged.
- Added a proof-local pointer-identity fast lookup ahead of the existing exact
  semantic evaluation-commitment cache. It only reuses a point for the same
  retained Polynomial value object, domain, and Lagrange SRS; the existing
  hash plus full-vector equality path remains the semantic fallback. No scalar,
  blinder, randomness, or cache state crosses proofs.

These changes apply to the production KZG commitment helpers used by dynamic,
binding/helper/shift, quotient, proof-local reblinding, and witness MSM paths.
Every benchmark compares the resulting G1 point against the original exact
MCL result and rejects a mismatch.

### Point-group opening witnesses

- Replaced the folded-plan linear search by a proof-local pointer index while
  preserving first-occurrence order and point-local powers.
- Made folded-row fanout depend on both row count and term count, avoiding
  parallel overhead for small groups and exposing work for large groups.
- Added execution-only field/MSM caps to `open_point_group`; they do not enter
  the transcript, proof, or verifier equation.
- Scheduled larger groups first. Domain groups use bounded outer workers with
  adaptive inner caps; external groups use full outer concurrency and one
  inner worker per group on this host. Witnesses are stored back by canonical
  group index, so proof order is unchanged.
- Added mutually exclusive opening diagnostics for quotient assembly, witness
  MSM, and blinder witness work.

Each evaluation point still has its own challenge, group, owner, witness, and
KZG equation. No point, commitment, opening, or blinding contribution was
removed or merged.

## Local and real-work-unit evidence

The final common-MSM benchmark used production `g1_multi_scalar_mul` and three
repeats per shape:

| Shape | Production median (ms) | Raw MCL median (ms) |
| --- | ---: | ---: |
| 1024 nonzero scalars in edge SRS | 5.189 | 22.425 |
| 4096 dense edge scalars | 14.311 | 67.214 |
| 1,048,576 dense coefficient scalars | 152.479 | 1267.784 |

The one-million-scalar case is a complete single real commitment using the
existing read-only `cora_formal_srs_v2`; it is not a complete Cora run. The
retained 2x overdecomposition reduced its earlier production median from about
179.833 ms to about 152.5 ms.

For a 910-term materialized opening group, three production runs had opening
medians of 53.987 ms, including 25.057 ms plan/precompute and 27.040 ms witness
MSM. The pre-change opening/precompute measurements were 211.536/181.544 ms.
All runs produced two independent witnesses and verified successfully. A
17-point external-group shape had a 242.934 ms median in the serial stage
driver; the formal outer scheduler is measured by the canary below.

The existing production `t_edge` evaluator was also checked for 8192 complete
relation points with 20 workers. Three-run medians were 273.606 ms (Cora),
274.865 ms (Citeseer), 260.601 ms (Pubmed), and 65.602 ms (PPI). The former
ogbn-arxiv case used a stale 1,048,576-row edge shape and is no longer evidence.
The case is now corrected to the production 8,388,608-row edge domain; the
2026-07-26 three-run diagnostic median was 79.101 ms for the same 8192 sampled
points. An unrelated two-core workload was active, so this last value is a
shape/correctness check rather than a clean performance baseline. Repeated
accumulators were identical.

## Reverted or rejected schemes

All code for the following candidates was removed after measurement:

- A whole-window custom Pippenger route was exact but regressed 1024 scalars
  from about 1.94 to 7.88 ms and one million scalars from about 180 to 835 ms.
- MCL `mulVec` with OpenMP outer splitting regressed 512 scalars from about
  2.576 to 3.079 ms and 1024 scalars from about 4.438 to 5.994 ms.
- A public fixed-base window table was exact but regressed 512/1024 scalar
  cases from 2.576/4.438 to 4.028/7.345 ms and increased memory.
- Opening schedulers using uniform outer-5/inner-4, external outer-10/inner-2,
  and full-outer/inner-2 either regressed domain work or raised external
  opening from about 152 to 184 ms. Only the adaptive scheduler remains.
- `t_edge` prebound string views, proof-local compiled derived expressions, a
  fuller SignedRoundDiv/logup binding plan, and raw native alpha-power access
  were pointwise exact but did not lower the formal total. The final candidate
  had a 2878.393 ms total median versus 2874.400 ms for the retained state.
- Reusing half-domain coset NTT scratch was exact, but its formal median was
  2905.216 ms, `t_edge` was 958.992 ms, and RSS was about 808700 KiB; it was
  removed.
- Full-outer opening with inner-2 MSM had a roughly 2950.8 ms formal median and
  was removed.
- A 32-task `t_edge` overdecomposition candidate reduced its isolated sampled
  kernel, but three complete controlled production-path runs had formal sums
  3004.489, 2949.553, and 3113.691 ms, with unstable external-opening tails.
  It was removed; the final source retains the 20-worker policy.

## Final three-run result

| Metric | Baseline median | Final median | Final min-max | Change |
| --- | ---: | ---: | ---: | ---: |
| `commitment_time` (ms) | 1682.144 | 976.702 | 968.523-977.619 | -41.937% |
| `prove_time` (ms) | 2143.149 | 1823.338 | 1811.814-1859.302 | -14.922% |
| `verify_time` (ms) | 76.633 | 78.102 | 76.085-79.085 | +1.917% |
| Three-time sum (ms) | 3897.541 | 2876.125 | 2858.439-2916.007 | -26.207% |
| Program wall time (ms) | 4110.572 | 3088.730 | 3056.590-3127.245 | -24.859% |
| Peak RSS (KiB) | 803868 | 808164 | 808016-808312 | +0.534% |
| `proof_size_bytes` | 250447 | 250441 | 250436-250446 | recorded only |

The final formal/wall difference is 212.605 ms at the median, versus 213.030
ms at baseline. It remains setup, SRS/config loading, I/O, scheduling, and
harness overhead outside the three formal metrics. No timed work moved across
a boundary.

The post-optimization mutually exclusive candidates covering the formal sum
are dynamic commitment MSM 749.159 ms (26.048%), quotient non-commitment
materialization 1048.971 ms (36.472%, including `t_edge` 940.858 ms), domain
plus external openings 401.395 ms (13.956%), and non-commit trace construction
279.192 ms (9.707%). Their cumulative coverage is 86.183%.

## Validation

- CPU Release and CUDA Release full builds passed.
- CPU and CUDA `ctest` each passed `gatzk_tests` and `static_secure_scan`
  (2/2) after removal of the obsolete matrix runner and its runner-only test.
- `GATZK_RUN_UNIT_SECURITY_TESTS=1 build/gatzk_tests` passed, including KZG
  opening, external point grouping, schema/metadata gates, canonical digest,
  domain challenge exclusion, and tamper/security checks.
- All three final generation runs completed canonical serialization, strict
  parse, and verification with `private_plain_value_count=0` and
  `verifier_reads_witness=false`.
- Independent strict parse (three repeats), verifier-only (three repeats), and
  compiler artifact validation passed. Verifier-only median was 48.525 ms
  after its one-time parse.
- `git diff --check` passed after rejected code was removed.

The proof schema, transcript, canonical codec/parser, verifier checks, SRS and
KZG structures, and public/private boundary did not change. The existing Cora
SRS compiler artifact therefore remains an exact byte-for-byte match. Missing
dataset SRS artifacts still require independently trusted setup; this record
does not treat local compiler preflight as an SRS publication.

## Remaining measured boundaries and full-run risks

- Dynamic commitments remain 26.048% of the bounded formal total. Profiling
  found 290 sequential commitment batches whose results are absorbed before
  subsequent challenges or polynomial construction. Cross-barrier batching
  would change the transcript/protocol. Each existing batch already finishes
  close to its largest object; the tested Pippenger, OpenMP, and fixed-base
  alternatives were slower on this aarch64 backend.
- `t_edge` remains the largest single kernel (32.713% of the total). Its
  prewarm already deduplicates semantic columns and caches plans/shift powers;
  point evaluation is field-arithmetic bound. Five materially different
  layout/materialization/reuse candidates failed to give a stable formal gain.
- Domain plus external openings remain 13.956%. The retained scheduler is the
  best of the measured outer/inner combinations; further curve acceleration
  needs a faster aarch64 MSM backend or a production GPU curve implementation.
- The installed MCL GPU/CPU dependencies provide no production CUDA curve MSM,
  KZG witness, or pairing path. The project instruction forbids introducing a
  new GPU curve/KZG/pairing implementation in this phase.
- The 0.534% RSS increase and two dormant bounded worker pools should be
  watched in user-run full Cora/Pubmed/PPI rows, especially under simultaneous
  large quotient materialization. Formal-run commitment order, peak RSS,
  thread saturation, and external-opening long tails are the main follow-up
  observations.
- Full-dataset behavior and the CPU/GPU ten-row matrix remain unmeasured and
  must be executed by the project owner.

## 2026-07-26 source convergence addendum

- The Cora `vector::reserve` failure was not an allocation-capacity problem.
  A trailing MSM task could start past `count` when ceil-sized chunks did not
  divide a non-divisible effective scalar count; `end - begin` then underflowed.
  The common production curve layer now uses quotient/remainder balanced,
  non-empty chunks, with exact 257- and 1009-scalar regression comparisons
  against the single-thread MCL result.
- Cora production context construction was stopped under a debugger only after
  all public/static commitments, including every hidden/output weight column,
  had completed. No trace or proof was run.
- Three source-matched production-path diagnostics after the fix had formal
  sums 3517.402, 2898.343, and 2953.748 ms. The first run coincided with the
  unrelated host workload and is an interference outlier. These checks all
  serialized, strict-parsed, and verified; they do not replace the stable
  2026-07-22 baseline/final comparison above.
- The SRS context preflight now requires `4|FH|` coefficient powers, matching
  the quotient materializer instead of the old `2|FH|` under-check. The
  ogbn-arxiv maximum-envelope chunk count was corrected from two to three.
- `gatzk_run` now saves the exact canonical `proof.bin`; its byte length must
  equal `proof_size_bytes`, and `run_manifest.json` binds it with SHA-256.

## 2026-07-26 continuation: model commitments and CUDA domain evaluation

### Current-source baseline and host state

Before the continuation changes, five fresh Release CPU processes were run
with the same unit-security config, seed, SRS, `taskset -c 0-19`, and formal
timing schema used for the final check. The medians were:

| Metric | Current-source baseline median |
| --- | ---: |
| `commitment_time_ms` | 999.433 |
| `prove_time_ms` | 1845.469 |
| `verify_time_ms` | 76.099 |
| Three-time sum | 2924.897 |
| Wall time | 3.20 s |
| Peak RSS | 808000 KiB |
| `proof_size_bytes` | 250444 |

All five proofs strict-parsed and verified. This is the continuation baseline,
not the older 3897.541 ms baseline used to establish the retained 26.207%
improvement.

The locked implementation evidence is:

- Git HEAD `18d92aa1f5c532f57f68b7ec173db8c06cec3d70`;
- code/config tree SHA-256 (`CMakeLists.txt`, `include/`, `src/`, `tools/`,
  `tests/`, `configs/`, and `scripts/`)
  `475f913a32b209f8c82d67c6f5cd33fbeda1fcba53a9b892e8e13be5b29e1993`;
- CPU `gatzk_run` SHA-256
  `05d73247e704fc085646ade04559b1341cec66d8c6e9d1ed42f6bad8ebe651ed`;
- CUDA `gatzk_run` SHA-256
  `53580a10ab3f184b39f9729ab376e86cbea74ab63d3183be601d09e929eb7ddc`;
- Release, native-architecture builds; CUDA is disabled/enabled respectively.

The host is Linux 6.17 aarch64 with 10 Cortex-X925 plus 10 Cortex-A725 cores,
one hardware thread per core, 121 GiB RAM, and an NVIDIA GB10 with driver
580.142. CUDA zero-copy dispatch additionally requires both UVA and pageable
host-memory access; otherwise the same production API uses its explicit-copy
fallback.

The continuation evidence can be reproduced without a separate protocol
runner:

```text
for case in \
  model_reblind_d16_batch3_full model_reblind_d16_sequential3_full \
  model_reblind_cat128_batch8_full model_reblind_cat128_sequential8_full \
  model_reblind_in2048_batch8_full model_reblind_in2048_sequential8_full \
  model_reblind_edge4096_batch8_full model_reblind_edge4096_sequential8_full
do
  for repeat in 1 2 3
  do
    taskset -c 0-19 build/gatzk_kzg_opening_stage_bench "$case"
  done
done
build_cuda/gatzk_cuda_hotspots_stage_bench
```

Both executables call the production KZG/evaluation functions directly. The
CPU controlled proof command is the representative entry recorded at the top
of this file.

The clean comparable final CPU canary was neutral: commitment 1013.382 ms
(+1.396%), prove 1839.767 ms (-0.309%), verify 79.591 ms (+4.589%), and total
2921.514 ms (-0.116%). Wall remained 3.20 s and RSS changed from 808000 to
807824 KiB. The unit model contains fewer than four same-domain reblind
objects, so it deliberately remains on the original serial route and cannot
measure the scale-gated model batch below. Later CPU samples were excluded
because an unrelated `zkdl_cpu_cli` and two editor processes raised host load;
they are not used to claim either a gain or a regression.

### Retained CPU model-reblind batching

Proof-local model reblinding previously committed every independent weight
column serially even though all commitments in a same-domain model block exist
before the transcript absorbs any of them. `trace.cpp` now prepares the same
fresh blinders and canonical polynomials in original order and sends groups of
at least four to the production `KZG::commit_batch`. Groups of one to three
retain the old per-object order; this is a generic amortization threshold, not
a dataset branch.

The retained benchmark calls the same production `KZG::commit` and
`KZG::commit_batch` functions over evaluation-basis objects backed by the
installed SRS. Three-run medians were:

| Same-domain model shape | Serial (ms) | Batch (ms) | Change |
| --- | ---: | ---: | ---: |
| 8 columns, domain 128 | 34.500 | 28.583 | -17.151% |
| 8 columns, domain 2048 | 57.290 | 47.778 | -16.604% |
| 8 columns, domain 4096 | 120.680 | 96.732 | -19.846% |
| 3 columns, domain 16 | 1.679 | 1.766 | +5.182% |

The last row is why small groups remain serial. Cora/Citeseer hidden weight
blocks use the measured 8-column route; the size-based implementation is also
common to Pubmed, PPI, and ogbn-arxiv and to larger output-class blocks. No
commitment, blinder, polynomial, transcript order, or verifier equation was
changed.

### Retained CUDA single-point and multi-rotation routes

On the NVIDIA GB10 unified-addressing device, production evaluation
polynomials are already in the column layout consumed by the CUDA field
kernel. The old padded route first materialized a full zero-padded/interleaved
copy. The old rotated route additionally rebuilt the subset and performed a
separate allocation, rotation, launch, reduction, and synchronization for
each opening point.

The retained route:

- passes the canonical read-only columns and their true lengths directly;
- makes the kernel apply zero tails and weight rotations by index;
- evaluates all rotations in one 3-D launch and one deterministic per-output
  reduction, preserving `[point][polynomial]` output order;
- chunks only when partial-reduction storage would exceed 256 MiB;
- fails on a CUDA error instead of silently falling back to CPU.

Three real-device exact-match runs for domain 65536, 32 rows, and three
rotations changed the production API median as follows:

| Shape | Before (ms) | Final (ms) | Change |
| --- | ---: | ---: | ---: |
| Full columns, three rotations | 4.591 | 1.596 | -65.236% |
| One half-length padded column, three rotations | 4.948 | 1.567 | -68.331% |
| One half-length padded column, one point | 1.462 | 0.734 | -49.795% |

The comparable fixed suite drops from 175 to 105 real kernel invocations
because each three-point group now uses two kernels instead of eighteen; the
final expanded suite reports 129 invocations because it adds four shape
checks. Final padded three-point medians were 1.585 ms at `32768 x 64`,
2.631 ms at `262144 x 8`, 3.242 ms at `1048576 x 2`, and 6.825 ms at
`4194304 x 1`. Every CPU/CUDA result was exact. These cover dense-row,
large-domain, and short-column shapes without selecting by dataset name.

The controlled unit proof is below the formal CUDA dispatch threshold, while
complete Cora and the five-dataset matrix are prohibited in this round.
Therefore these are real-kernel production-function results, not a claimed
CUDA end-to-end `commitment/prove/verify` result.

### New candidates rejected in the continuation

The following code was removed after measurement:

- dynamic `t_edge` scheduling, temporary-buffer reuse, shifted-base/two-scalar
  commitment forms, disabled commitment profiling, shared opening inverse
  workspaces, LTO, `-mcpu=native`, and lower general commitment thresholds;
- capacity-weighted pinned `t_edge` ranges, which changed the five-run median
  from 962.378 to 963.139 ms and worsened total/long-tail behavior;
- a 12-worker medium MSM policy: isolated 256-1024 scalar cases improved, but
  five complete production-path runs raised the total median from 2924.897 to
  2949.407 ms;
- a 16-worker `t_edge` cap: five formal-shape local medians improved in
  aggregate, but seven paired production proofs changed `t_edge` only
  949.160 to 948.036 ms and worsened the total median from 2926.495 to
  2936.719 ms.

No candidate switch, failed implementation, or generated artifact remains.

### Continuation closure and validation boundary

The remaining CPU total is still dominated by `t_edge`, dynamic MSM, and
curve-bound opening witnesses. Multiple evaluator layouts/schedulers,
Pippenger/OpenMP/fixed-base routes, batch thresholds, FFT scratch reuse, and
opening worker allocations have now been measured at both local and complete
controlled-production levels; none gave a stable additional CPU total
reduction. Paired/split NTT remains strongly beneficial at larger domains, and
the tested code already uses it. Verifier pairings already use the retained
mixed precomputation; algebraically batching independent accept/reject
equations would permit cancellation and is not an allowed optimization.

CUDA field-domain evaluation now has a large stable improvement and scale
checks through a 4,194,304-point domain. Curve MSM, KZG witness MSM, and
pairing remain CPU-bound because the installed dependency has no production
GPU curve backend; that limits those specific stages, not the field-kernel
work completed here.
