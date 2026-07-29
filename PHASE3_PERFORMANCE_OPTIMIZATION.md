# Phase 3 Performance Optimization

Last updated: 2026-07-30 (Asia/Shanghai)

## Status and measurement contract

Phase 3 production implementation is converged. The only formal entry remains
`gatzk_run`; tests and stage benchmarks call the same Context, Trace, Prover,
Verifier, KZG, quotient, codec, and parser functions. There is no dataset
branch, alternate runner, reduced proof workload, hidden fast path, or second
protocol implementation.

All timings use `exclusive_commit_prove_verify_v2`. A stage result is evidence
for its production function, not a substitute for a complete dataset result.
The current-source Cora run completed `VERIFY_OK`, but unrelated CPU/GPU
training and CPU test workloads were active. Its proof is valid; its timing is
reported as host-contaminated rather than a controlled frozen baseline.
Citeseer, Pubmed, PPI, and ogbn-arxiv were not rerun after this optimization
batch.

## Current production changes

### File-backed polynomial lifetime and independent blinding

The formal FH trace can move private evaluation polynomials into
`TraceArtifacts::spilled_polynomials`. The old quotient lookup order checked a
public lazy full-identity alias first. That alias reconstructed the canonical
evaluation column but not the private polynomial's independently sampled
`Z_FH` blinding tail.

The production materializer now resolves a spilled private object before all
public/lazy aliases. The direct correctness fix loaded each complete spilled
column and reached exact quotient self-check, but t_FH prewarm took 53.904 s.
The retained implementation instead:

- constructs the feature-table semantic coset once from the real public row,
  column, and feature dependencies;
- derives Query from Table plus the exact difference between the two private
  blinder polynomials;
- reuses the canonical selector semantic coset for multiplicity and remainder
  while adding each registered independent blinder;
- applies the same blinder correction in proof point evaluation;
- lets debug FH relation checks load canonical spilled values.

The retained source produced 12,582,915 t_FH coefficients and passed exact
hidden-coefficient self-check. A broken alias run produced 12,582,912 and was
not retained.

### Quotient coset storage and reuse

Unlinked quotient coset files no longer execute `msync(MS_SYNC)` or discard
their complete page range before first consumption. They remain sequential
`MAP_SHARED` mappings and discard only already-consumed prefixes. This removes
a forced temporary-file writeback/read cycle without changing values.

Semantic reuse now shares a canonical coset even when formal polynomials have
independent zero-domain blinders; every object gets its own lazy
`Z_domain * r(X)` correction. Direct aliasing remains restricted to unblinded
objects.

Full-security t_edge storage changed from:

```text
files=2270 bytes=9521070080
lazy_blinder_views=619 lazy_blinder_bytes_avoided=2596274176
```

to:

```text
files=1690 bytes=7088373760
lazy_blinder_views=1227 lazy_blinder_bytes_avoided=5146411008
```

This is 580 fewer files and 25.551% less materialized t_edge storage.

### Batched quotient boundary inverses

The production t_edge and t_FH loops compute first/last Lagrange boundary
values in 8192-point blocks. They batch-invert denominator products and pass
the exact boundary values into the unchanged quotient evaluator. A new
regression compares both boundary values and the complete `evaluate_t_fh`
result with the canonical direct formula.

### Lazy public evaluation backend

`PackedEvaluationBackend` previously constructed a full global interleaved
copy of every registered public evaluation polynomial. It now validates and
indexes canonical columns only. An actually used small subset is interleaved
lazily when the existing subset policy requests it; CPU and CUDA fallbacks read
canonical values directly.

Because these backends retain non-owning polynomial pointers, Prover and
Verifier registries are proof/verification-local. The old process-static
registries and incomplete string cache key were removed.

### Existing retained Phase 3 work

The current production source also retains:

- shared KZG commitment batching, semantic commitment reuse, sparse/tail MSM,
  and bounded persistent workers;
- proof-local model reblinding batches for groups of at least four;
- point-group opening plans, folded single-point witnesses, denominator reuse,
  and mixed pairing;
- fixed-length Taylor shift with exact cyclic-tail correction;
- one batched LogUp inverse pass with reusable accumulators;
- CUDA canonical-column single/multi-rotation field-domain evaluation.

## Measurements

### Full-dataset artifacts

| Source state | Dataset | commitment | prove | verify | proof bytes | status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| pre-batch baseline | Cora CPU | 34.849 s | 159.022 s | 1.199 s | 1,462,631 | `VERIFY_OK` |
| pre-batch baseline | Citeseer CPU | 78.635 s | 429.439 s | 3.373 s | 1,493,238 | `VERIFY_OK` |
| pre-batch baseline | Pubmed CPU | 113.707 s | 1020.030 s | 12.653 s | 2,743,639 | `VERIFY_OK` |
| current source, host-contaminated | Cora CPU | 43.858 s | 152.137 s | 0.713 s | 1,462,616 | `VERIFY_OK` |

The first three rows remain valid canonical results but are
`stale-for-current-source`. PPI and ogbn-arxiv old-binary opening-material
failures are not results. Their current state is local validation passed,
frozen full run pending.

The current Cora formal change is +25.853% commitment, -4.330% prove,
-40.556% verify, and +0.840% for the three-time sum. The commitment regression
is not accepted as a code conclusion: the run had unrelated host workloads,
and internal counters show FH commitment increasing 3.600 s to 8.650 s,
dynamic finalize 0.007 s to 1.694 s, and multiple opening/trace long tails.

### Quotient and verifier evidence

| Metric | Before | Current | Change |
| --- | ---: | ---: | ---: |
| full-security t_edge prewarm + evaluate | 44.339 s | 29.075 s | -34.425% |
| full-security t_FH prewarm | 42.758 s | 38.049 s | -11.013% |
| Cora quotient build | 126.253 s | 102.171 s | -19.075% |
| Cora t_edge quotient | 55.326 s | 41.185 s | -25.559% |
| Cora t_FH quotient | 57.545 s | 47.497 s | -17.462% |

The final full-security t_FH stages were 38.049 s prewarm, 6.109 s evaluate,
and 5.493 s coefficient conversion. Exact quotient self-check passed.

Direct production verification of the existing Cora canonical proof completed
three times with `verify_ok=true`; median was 512.707 ms, 57.252% below its
original 1,199.368 ms run. The median profile was:

| Verifier component | Time |
| --- | ---: |
| metadata | 14.888 ms |
| transcript | 101.482 ms |
| domain opening | 8.452 ms |
| backend registry | 42.889 ms |
| quotient relations | 313.063 ms |
| external opening | 28.428 ms |

### Local production benchmarks

- Taylor shift, degree 32769: 40.753–41.592 ms warm.
- LogUp inverse/accumulator stage: 3.010–3.037 ms steady.
- t_edge production evaluator, Cora shape, 32768 points: approximately
  8.518–8.789 s in the isolated evaluator.
- quotient transform, 1,048,576 coefficients: serial 394.556 ms, split
  parallel 259.219 ms; forward pair serial 1497 ms, parallel 364 ms.
- opening, 96 terms/fanout 7: open 53.684 ms, verify 15.105 ms.
- opening, 17 external materialized points: open 237.373 ms, verify 19.087 ms.
- opening, 910 terms materialized: open 57.730 ms, verify 12.630 ms.

Every opening case verified. Stage drivers call production functions and do
not reproduce the algorithms.

## Resource results

| Resource | Before | Current | Change |
| --- | ---: | ---: | ---: |
| full-security peak RSS | 25,899,732 KiB | 17,250,228 KiB | -33.396% |
| t_edge temporary coset | 9,521,070,080 B | 7,088,373,760 B | -25.551% |
| lazy blinder materialization avoided | 2,596,274,176 B | 5,146,411,008 B | +2,550,136,832 B |
| t_FH temporary coset | 4,831,838,208 B | 4,831,838,208 B | unchanged |

Cleanup removed 402 stale top-level trace spills and three interrupted
per-proof spill directories: 73,488,179,904 bytes (68.44 GiB). No formal
proof, SRS, data, checkpoint, or accepted baseline was removed.

The CUDA Release build and unit/PPI security suite pass. The current installed
backend accelerates Fr/domain kernels; curve MSM, KZG witnesses, pairings,
NTT/IFFT, and quotient materialization remain CPU work. CUDA stage timings from
this round are excluded because another process occupied the GPU.

## Retained and rejected candidates

Retained:

- spilled-private-first materialization with exact lazy blinder correction;
- semantic coset sharing with independent blinding preserved;
- removal of synchronous temporary-coset writeback;
- batched t_edge/t_FH boundary inverses;
- lazy canonical-column evaluation backend;
- proof-local backend registries;
- the FH boundary equivalence regression and formal-proof verifier stage mode.

Rejected and removed:

- direct full spilled-column load for every lazy FH object: correct but
  t_FH prewarm was 53.904 s;
- generalized all-label t_FH prewarm grouping: no stable benefit;
- obsolete full-identity/fused-table/selector-boolean coset helpers;
- process-static backend registries.

Previously measured regressions remain absent: the 8 GiB opening scheduler,
32-task t_edge policy, whole-window custom Pippenger, OpenMP outer MSM split,
fixed-base tables, shared opening inverse workspace, shifted-base candidates,
coset scratch reuse, LTO/native-only candidates, and dataset-specific worker
caps.

## Validation

- CPU Release build: passed.
- CUDA Release build: passed.
- CPU and CUDA unit security plus PPI boundary suites: passed.
- CPU full security suite with exact quotient self-check: passed.
- Current-source Cora CPU complete proof: `VERIFY_OK`.
- Current verifier against prior Cora canonical proof: three passes.
- Taylor, LogUp, t_edge, quotient-transform, KZG-opening stage benchmarks:
  passed.
- Strict static security and all-configured-SRS scan: passed.
- Source whitespace check: passed.

The full-security process used 17,250,228 KiB peak RSS, did not swap, and
finished in 11:00.57. Its performance stages were also affected by shared-host
load, but its correctness evidence is valid.

## Closure

The remaining large work units are protocol-required dynamic commitments,
t_edge/t_FH relations, FH opening witnesses, and verifier quotient equations.
Their implemented candidates have been measured; skipping, merging across
challenge barriers, or reducing them changes the protocol or proof workload.
A new production GPU curve/KZG/pairing implementation is outside the current
audited dependency boundary and is not an unimplemented patch-sized candidate.

Therefore:

- no identified, feasible, protocol-equivalent candidate with likely
  significant stable end-to-end gain remains open;
- the main experiment has one formal entry and one implementation;
- PPI and ogbn-arxiv remain “local validation passed; frozen full validation
  pending” until the final commands in `MAIN_EXPERIMENT_FINAL_REPORT.md`
  complete;
- the ten-row CPU/CUDA acceptance matrix remains a post-freeze execution task,
  not a claimed result.
