# Cache Policy Audit

Updated: 2026-06-28T09:57:00Z

## Scope

This audit covers proof-specific persistent caches left by earlier timeout canaries and whether the final secure main experiment path can read or write them.

## Discovered Persistent Caches

At startup, `runs/performance_cache` existed with:

```text
dynamic_commitment_v1 files: 455
quotient_v1 .poly files: 7
quotient_v1 .point files: 12
quotient_v1 .coset files: 2
quotient_v1 .coeff files: 1
quotient_v1 .qvals files: 0
```

These are proof-specific artifacts created by failed/timeout diagnostic canaries and are not valid final-run inputs.

## Archived Diagnostic-Only Cache

The cache directory was moved, not deleted:

```text
runs/performance_cache_diagnostic_archive/20260628T095654Z/performance_cache
```

`runs/performance_cache` is absent after the move.

## Disabled Or Removed Code Paths

Removed from final source path:

```text
src/protocol/trace.cpp: persistent dynamic commitment file cache
src/protocol/prover.cpp: persistent quotient field-vector cache
src/protocol/prover.cpp: persistent quotient polynomial cache
src/protocol/prover.cpp: persistent quotient point cache
src/protocol/prover.cpp: persistent t_FH coset/coeff/qvals cache
src/protocol/prover.cpp: cross-run chunked MSM point continuation cache
```

Retained as valid clean/cold-cache optimizations:

```text
src/protocol/trace.cpp: same-row dynamic commitment reuse and in-batch duplicate suppression
src/protocol/prover.cpp: same-row t_FH use of current trace commitment opening polynomial for P_Table_feat
src/protocol/prover.cpp: in-process proof/backend evaluation caches that do not read failed-run files
src/crypto/kzg.cpp: SRS selective/prefix loading and MSM scheduling changes
```

## Source Scan

After the correction patch, a source scan over `src`, `include`, `scripts`, and `tools` found no active `runs/performance_cache`, `GATZK_*CACHE`, `lookup_persistent_*`, `store_persistent_*`, `persistent_cache`, or cross-run `chunked_msm` final-path symbols. Remaining `runs/performance_cache` strings are historical entries in audit reports only.

## Final Path Policy

- final runner reads failed-run cache: no
- final runner writes failed-run cache: no
- clean/cold-cache canary depends on `runs/performance_cache`: no
- failed-run cache enters final path: no
- diagnostic cache retained only in archive: yes

