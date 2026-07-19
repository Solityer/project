# Phase 3 Performance Optimization

Last updated: 2026-07-18 (Asia/Shanghai)

## Status

Phase 3 is open. The earlier statement that the non-full-dataset scope had
converged is withdrawn. This file is the only Phase 3 performance record.
Historical measurements from the former protocol are not current baselines and
must not be used as run-validity, correctness, security, resume, deletion, or
archival gates.

No complete end-to-end proof was run for Cora, Citeseer, Pubmed, PPI, or
ogbn-arxiv during this work. All numbers below are bounded hotspot or small
security-fixture evidence, not formal dataset `commitment_time`, `prove_time`,
or `verify_time`.

## Workspace and environment

- Branch/starting commit: `main`,
  `9aa1cc883e218a5cfade7e5d0b1fdbe2757ce146`.
- CPU build: CMake Release, vendored mcl, OpenSSL, pthreads, 20 logical CPUs.
- CUDA build: Release with `GATZK_ENABLE_CUDA_BACKEND=ON`.
- GPU: NVIDIA GB10; CUDA runtime and unified addressing available.
- CUDA is a partial-hotspot backend. KZG MSM, pairing, NTT/IFFT, quotient
  materialization, opening witnesses, and small-domain work remain on CPU.
- The two highest-priority specification files are unchanged.

## Formal measurement contract

`src/main.cpp` is the single source for the three formal top-level times:

- `commitment_time_ms = commit_dynamic_ms + quotient_bundle_pack_ms`. Dynamic
  commitments include binding/helper/shift objects; the quotient bundle
  includes quotient and quotient-mask commitments.
- `prove_time_ms` begins after dynamic witness construction. Dynamic
  commitments produced while building the trace are explicitly added to the
  contiguous quotient/opening/proof-assembly/canonical-serialization span.
- `verify_time_ms` begins before the strict parser reads the first byte and ends
  after the verifier returns.
- `proof_size_bytes` is the actual canonical serialized byte-vector length.

The earlier secondary write to `prove_time_ms` inside `prove()` was removed.
The old exporter fallback that reconstructed commitment time from subfields was
also removed; a manifest without the formal field now fails closed.

## Runner and obsolete performance-policy cleanup

These changes are measurement/cleanup prerequisites, not core speedups:

- User `--timeout-seconds` is the exact process timeout. The removed
  `row_hard_timeout_ms` path can no longer silently shorten it.
- Timeout, correctness/security acceptance, and performance evaluation are
  separate.
- Resume validates run identity, manifest integrity, completion, correctness,
  and security. Historical performance targets are not resume gates.
- Current rows report
  `performance_baseline_status=pending_rebuild_current_protocol` and
  `performance_evaluation_status=informational_not_run_validity_gate`.
- Correct/security-passing rows are not deleted or archived because of an old
  speed target.
- Hardcoded `2x`/`near_2_2x` budgets and old baseline reports were deleted.
- The paper exporter no longer generates a report containing hardcoded old
  Cora speed ratios or the old performance-archive policy.
- Manifest route notes now report only real configurable routes. Historical
  hardcoded `enabled_*=true` claims were removed.

`tests/test_runner_semantics.py` covers exact timeout forwarding, performance-
independent resume, and absence of legacy budget fields.

## Current hotspot ranking

The ranking follows the current production call chain and bounded current-code
measurements:

| Priority | Production stage | Current conclusion |
| ---: | --- | --- |
| 1 | `t_edge` pointwise quotient evaluation | Still dominant and not converged; 887/1007 eval lookups per point for PPI/ogbn 1+1 heads and 3485 for 8+1 heads |
| 2 | dynamic/binding/helper/shift KZG commitments | Shared MSM path is active; curve arithmetic remains a formal-scale risk |
| 3 | quotient coset materialization and final IFFT | Domain IFFT, paired 4D forward NTT, and final IFFT structurally improved; dependency count/cache lifetime remain material |
| 4 | domain and external openings | Shared domain materialization retained; external groups still build individual soundness equations |
| 5 | verifier parser/metadata and pairing | Large bounded improvements retained; formal-scale proof cardinality still needs user evidence |
| 6 | CUDA placement and CPU/GPU coordination | Exact partial-hotspot speedups; no CUDA MSM/NTT/quotient/pairing path |
| 7 | other quotients and serialization | Low on the small call chain; scaling remains monitored through domain size |

The corrected representative `t_edge` benchmark directly invokes the
production evaluator and includes the formal hidden layer, one output head,
aggregation/route metadata, coefficient-basis work, class count, 4x quotient
coset, and production content lookup behavior. Its projected full-coset values
are diagnostic extrapolations only and are not formal dataset times.

## Retained structural optimizations

### Dynamic commitments and shared MSM

- Shared KZG/MSM routes skip zero scalars, compact sparse inputs, fold long
  constant evaluation tails, and retain bounded outer/inner scheduling.
- The verifier-relevant MSM parallel threshold was lowered from 1024 to 256
  scalars after multi-size evidence:

| Scalars | Before (ms) | After (ms) | Change |
| ---: | ---: | ---: | ---: |
| 910 | 316.962 | 117.380 | -63.0% |
| 512 | 179.179 | 76.921 | -57.1% |
| 256 | 95.835 | 39.748 | -58.5% |

The 128-scalar route remains serial. A forced all-core scheduler for a typical
12-object formal batch had no stable gain and was removed.

### Domain opening and external opening

Folded domain openings share materialization and barycentric work while still
checking every individual claimed value. For a 4096-domain/910-term stage,
opening changed from 3034.432 to 1793.316 ms (-40.9%). A Cora-shaped bounded
case changed from 4619.858 to about 4134 ms (-10%).

Every witness, folding challenge, and verifier equation remains present. No
opening was removed merely because another equation looked similar.

### `t_edge`

- All quotient boundary values now share one `z^n` and one combined inverse;
  the formal materializer batch-inverts boundary denominators in 8192-point
  blocks and cycles the four coset zero-polynomial values.
- Coset-to-coefficient conversion consumes the quotient vector in place; the
  old out-of-place duplicate was deleted.
- The old benchmark omitted output heads and binding metadata. It was corrected
  and renamed to `*_formal_shape`; old hidden-core projections are invalid as
  five-dataset evidence.
- Corrected PPI/ogbn shapes are 1 hidden + 1 output head. Precomputing
  `alpha_quot` powers is therefore beneficial, contrary to the old hidden-only
  single-head result. Same-output medians at 2048 sampled points were:

| Shape | Incremental median (ms) | Precomputed median (ms) | Change |
| --- | ---: | ---: | ---: |
| PPI 1+1 heads, 14-bit range | 259.642 | 180.478 | -30.5% |
| ogbn-arxiv 1+1 heads, 12-bit range | 297.413 | 218.067 | -26.7% |

The production dispatcher now precomputes for at least two total heads; all
five formal configurations meet that generic condition. The 8+1-head route was
already precomputed. Accumulators were byte-for-byte identical, and the small
security proof/verifier regression passed after the production change.

Despite these improvements, the corrected evaluator still executes 887
(PPI), 1007 (ogbn-arxiv), or 3485 (8+1 heads) lookups per coset point and
remains the largest unresolved wall-time risk.

### Shared field arithmetic

`FieldElement` arithmetic outputs are now constructed through an internal
overwrite-only path. An operand already proves mcl initialization has occurred,
and `add`/`sub`/`mul`/`pow`/`inv` completely overwrite their output, so the old
per-operation clear and global `call_once` check were redundant. Public/default
construction and first-use initialization are unchanged.

Single-worker, 2048-point corrected `t_edge` medians changed as follows, with
identical accumulators:

| Shape | Before (ms) | After (ms) | Change |
| --- | ---: | ---: | ---: |
| PPI 1+1, 14-bit | 2022.387 | 1963.572 | -2.9% |
| ogbn-arxiv 1+1, 12-bit | 2303.133 | 2247.937 | -2.4% |
| Cora 8+1, 12-bit | 7952.581 | 7709.212 | -3.1% |

This is retained as a small cross-path primitive improvement, not counted as
the main structural result.

### Quotient and dependency coset transforms

The final quotient coset-to-coefficient inverse transform now splits even and
odd frequency points into two independent half-size INTTs, runs them in
parallel, then recombines with the exact inverse-root factors. Small transforms
retain the canonical full INTT.

The benchmark calls the production `algebra::intt_in_place` and compares every
field element with `ntt_plan(size)->intt`:

| Size | Full INTT (ms) | Production split INTT (ms) | Change |
| ---: | ---: | ---: | ---: |
| 32,768 | 91.455 | 60.722 | -33.6% |
| 131,072 | 405.905 | 279.184 | -31.2% |
| 1,048,576 | 4069.896 | 2424.592 | -40.4% |
| 4,194,304 | 17902.395 | 10397.862 | -41.9% |

The 4M result was measured with the equivalent candidate immediately before
it was moved into the shared production primitive. The production primitive
was rechecked through 1M. A 32K exact-equivalence regression is always run.
Peak temporary memory increases by one native field vector: about 128 MiB at
4M points. This is a user-run scale risk to monitor.

Every evaluation-basis quotient dependency previously still used the serial
domain IFFT. It now calls the same exact split-INTT primitive. The 4D forward
coset code had already decomposed its work into two independent 2D NTTs, but
executed those halves serially; production now runs the pair concurrently.
No new transform buffer is added by the paired-forward change because both
half-size arrays already existed.

The stage benchmark invokes the production paired-NTT primitive and compares
both output vectors against two canonical serial NTT calls:

| Half-transform size | Serial pair (ms) | Production pair (ms) | Change |
| ---: | ---: | ---: | ---: |
| 65,536 | 352.643 | 190.791 | -45.9% |
| 524,288 | 3391.669 | 1679.778 | -50.5% |

For a Cora/Citeseer-class dependency, the measured transform components
`D=32768` IFFT plus the two `2D=65536` forward NTTs change from about
444.098 to 251.513 ms (-43.4%). This is a component composition, not a formal
dataset wall-time measurement. The same production path is shared by all five
formal configurations.

### Strict parser and verifier

- Strict G1 encodings are read in wire order, then independent named groups are
  decoded in parallel. Every point still undergoes mcl deserialization/subgroup
  validation and exact canonical reserialization comparison.
- On the bounded 242,795-byte proof containing 1163 G1 points, strict parse
  changed from 1731.254 to 646.602 ms initially; the latest five-run median is
  629.423 ms.
- Invalid/non-subgroup encodings are covered by a rejection regression.
- The seven independent domain-bundle KZG equations retain separate acceptance
  bits but execute concurrently. Isolated fanout-7 verification changed from
  390.788 to 80.756 ms.
- Folded verifier commitment algebra changed the 4096/910 case from 1877.339
  to 342.075 ms before the shared MSM threshold; the latest KZG verify stage is
  about 134.700 ms.
- The latest bounded parser-excluded verifier median is 891.856 ms:
  metadata 433.625, transcript 148.181, domain opening 134.414, quotient
  52.271, external 104.552 ms.
- Exact-rational compiler ledger totals aggregate identical denominators before
  final exact addition. Individual ledger entries remain unchanged. Artifact
  build changed from 468.530 to about 428.501 ms.
- A verifier cache-key path that always forced a miss but serialized the full
  challenge/commitment set was deleted.

### CUDA partial hotspots

Latest sequential exact-match stage results:

| Stage | CPU (ms) | CUDA (ms) |
| --- | ---: | ---: |
| barycentric, domain 65536 | 87.502 | 3.294 |
| formal weighted route, 32 columns | 58.225 | 0.792 |
| formal weighted padded route | 74.010 | 1.595 |
| histogram 4096/1024 | 0.328 | 0.757 (CPU route retained) |
| histogram 16384/1024 | 1.325 | 0.749 |
| histogram 1M/4096 | 47.009 | 2.213 |
| histogram 4M/65536 | 190.304 | 34.619 |

The benchmark recorded 90 CUDA kernel invocations. CUDA timing is not an
end-to-end GPU claim.

## Reverted and cleaned experiments

- Parser direct-pointer/stack decode: 1731.254 -> 1760.180 ms; removed.
- Parser G1 content cache: almost no duplicate points (12/1163); removed.
- `t_edge` pointer-key lookup front cache: 8+1-head median regressed about
  394.2 -> 412.7 ms; its code and switch were removed. The 1+1 result was too
  small/inconsistent to justify shape-specific complexity.
- `t_edge` suffix map and generic per-point memoization: no stable net gain;
  removed.
- Precomputing selector/table complements inside signed division and logup
  constraints was output-equivalent but the PPI 1+1 median regressed from
  180.478 to 186.400 ms across the expanded sample; fully reverted without a
  range-bit or dataset-specific branch.
- Forced all-core dynamic commitment scheduling: no stable formal-shape gain;
  removed.
- A persistent MSM worker-pool candidate improved isolated 910-scalar calls,
  but the production `commit_batch` remained approximately 1086.4 -> 1085.3 ms
  on the Cora-shaped 12-object case and the 64-object case was unstable. The
  pool, its queue, and all related code were removed.
- Earlier forced alpha-power precomputation conclusion was based on an invalid
  hidden-only shape. The benchmark was corrected rather than preserving that
  obsolete route decision.
- `perf` sampling was blocked by the host kernel policy. A `-pg` artifact build
  became hundreds of times slower and produced no useful sample; the processes
  were terminated and all temporary profiling builds/logs were removed.
- Cross-proof private commitment fingerprint/cache code, stale exporter
  reports, hardcoded route-note claims, and the dead out-of-place quotient
  conversion were deleted.

## Verification completed at this checkpoint

- Full CPU Release build: passed.
- `ctest`: `gatzk_tests`, static security scan, and runner semantics all passed.
- Extended small `GATZK_RUN_UNIT_SECURITY_TESTS=1` regression: passed after the
  parser, verifier, KZG/MSM, schema, quotient-boundary, and alpha-power changes.
- Canonical serialization/parser round trip, invalid G1 rejection, verifier-
  only checks, tamper rejection, and private plain-value count: passed on the
  small security fixture.
- PPI representative boundary: two complete source graphs loaded and checked;
  no cross-graph edge and no node/edge cropping; no proof was constructed.
- CUDA barycentric/evaluation/histogram CPU-reference equality: passed at every
  tested size.
- Split INTT versus canonical full INTT: exact at all tested sizes.
- Paired forward NTT versus two canonical NTTs: exact at all tested sizes in
  both CPU and CUDA-enabled builds; the full CUDA test target also compiled and
  passed its bounded regression.

## Remaining work

Phase 3 is not complete. The following remain open:

1. Continue reducing `t_edge` pointwise algebra and the number/lifetime of
   base/derived coset materials; transform kernels are substantially lower but
   the dependency cardinality and projected wall time remain unacceptable.
2. Recheck dynamic commitment object profiles on current formal shapes without
   running a complete dataset proof; curve MSM remains a high-scale risk.
3. Continue auditing external-opening construction and verifier pairing
   cardinality against the current schema/transcript. No object or equation may
   be removed without a complete verifier-consumption proof.
4. Quantify split-IFFT memory pressure together with coset cache lifetime on a
   bounded large-shape stage; the paired forward NTT adds no buffer beyond the
   arrays already present.
5. Rebuild formal performance baselines only through user-authorized complete
   runs. Until then all five dataset rows remain `pending_rebuild_current_protocol`.

## User-authorized full-dataset handoff

Do not run rows concurrently: quotient cosets, SRS material, and MSM can consume
tens of GiB. Ensure the dataset cache, checkpoint bundle, dataset-specific SRS,
CUDA runtime (for GPU rows), and sufficient disk space are present first. Use a
timeout that reflects the desired real process limit; the runner no longer
shortens it.

Recommended order is one CPU row first, then its GPU row, and only then the
next dataset. Example commands (to be run only by the user):

```bash
python3 scripts/run_secure_main_experiments.py \
  --datasets cora --backends cpu --timeout-seconds 21600 --skip-export
python3 scripts/run_secure_main_experiments.py \
  --datasets cora --backends gpu --timeout-seconds 21600 --skip-export

# Repeat serially for citeseer, pubmed, ppi, and ogbn-arxiv.
# Regenerate the aggregate export only after every selected row is complete:
python3 scripts/run_secure_main_experiments.py \
  --datasets all --backends all --timeout-seconds 21600
```

Do not use `--force` when resuming unless rerunning a valid row is intentional.
An interruption preserves correctness/security-valid completed rows; partial or
identity-mismatched rows are not accepted as resumable results. Monitor RSS,
GPU memory, disk space, and progress logs, and stop before system-wide memory
pressure threatens other work.

**五个完整数据集的证明验证未执行，必须由用户本人决定并运行。**
