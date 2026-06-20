# zkGAT Performance Redesign Plan

Generated: 2026-06-17 14:45 Asia/Shanghai

This plan is for performance engineering only. It does not authorize editing final result artifacts, changing security parameters, reducing datasets, using fake KZG/GPU paths, or resuming the final 10-row matrix.

Superseded for route selection on 2026-06-19 by:

- `results/performance_engineering/performance_budget_report.md`, section `Amdahl Upper-Bound Analysis For Cora`;
- `results/performance_engineering/order_of_magnitude_redesign_plan.md`.

The sections below remain historical context. They must not be used as the current Cora readiness gate because newer Cora evidence is `total_wall_clock_ms=785,469.068`, not `925,920.760`.

## Confirmed Bottlenecks

Latest current-source Cora row:

- `total_wall_clock_ms=925,920.760`
- `commitment_time_ms=328,297.104`
- `quotient_build_ms=373,118.559`
- `quotient_commitment_ms=112,627.306`
- `domain_opening_ms=273,004.319`
- `domain_open_FH_ms=269,294.034`
- `fh_open_witness_ms=258,161.924`
- `external_opening_ms=35,691.081`

Common bottleneck classes:

| bottleneck | current Cora evidence | expected amplifier |
|---|---|---|
| edge-domain dynamic KZG MSM | edge-domain object MSM sum `511,276.320 ms`; top objects are `R_t/R_src/R_dst/PSQ/Table_ELU` | PubMed/PPI/OGBN |
| FH quotient materialization | `quotient_t_fh_ms=258,017.780`; Cora FH domain is 4,194,304 | Citeseer/PubMed/OGBN |
| edge quotient materialization | `quotient_t_edge_ms=228,138.517` | PubMed/PPI/OGBN |
| FH opening witness | `fh_open_witness_ms=258,161.924` | Citeseer/PubMed/OGBN |
| quotient commitment MSM | `quotient_commitment_ms=112,627.306` | all full-dataset rows |

The current CUDA backend does not accelerate KZG MSM/opening/pairing, so a GPU row cannot be considered a real performance PASS until this changes.

## Completed Engineering In This Loop

Implemented and verified:

- end-to-end row timing in `src/main.cpp`;
- manifest timing export in `src/protocol/prover.cpp`;
- runner wall-clock and overhead enrichment in `scripts/run_secure_main_experiments.py`;
- strict timing validation in `scripts/export_paper_secure.py`;
- `quotient_commitment_ms` instrumentation in the prover and exporter schema;
- dense and sparse MSM scalar zero-copy in `include/gatzk/algebra/field.hpp` and `src/crypto/curve.cpp`;
- repeated-table full-value fingerprint coverage for safe dynamic commitment reuse in `src/protocol/trace.cpp`;
- route notes documenting the enabled optimizations in `src/util/route2.cpp`.

Rejected:

- KZG opening pre-fold in `src/crypto/kzg.cpp`. It increased Cora `domain_opening_ms` from about 273s to 326s and `fh_open_witness_ms` from about 258s to 309s. It has been fully removed.

Validation after the current code state:

- `cmake --build build -j 12`: passed
- `cmake --build build_cuda -j 12`: passed
- `ctest --test-dir build --output-on-failure`: passed
- `ctest --test-dir build_cuda --output-on-failure`: passed
- `python3 tools/security/static_scan_no_tau.py`: passed
- `GATZK_RUN_FULL_SECURITY_TESTS=1 build/gatzk_tests`: passed
- current Cora full-dataset official row: passed verify/audit/tamper, but failed performance gate

## Optimization Direction

Small scheduler or instrumentation changes are not sufficient. Cora still needs more than a 15x speedup for the hard gate and more than a 90x speedup for the target. The next engineering work must reduce common proof work.

Allowed protocol-equivalent routes:

| route | target | expected effect | risk |
|---|---|---|---|
| avoid repeated dense domain materialization in quotient/opening | FH and edge quotient/opening | reduce memory bandwidth and allocation pressure | medium |
| stream quotient values directly into commitment/opening MSM when transcript order is unchanged | quotient commitment and opening witness | avoid large vectors and rereads | medium/high |
| improve CPU MSM implementation/scheduling for sparse and dense SRS MSM | dynamic commitments, quotient commitments, witnesses | moderate unless algorithmic | medium |
| add real GPU MSM for G1 KZG commitments and opening witnesses | dominant KZG work | potentially required for PASS | high |
| add GPU/fused field kernels for FH/edge quotient arithmetic | quotient build/opening numerator prep | high for large domains | high |
| public immutable cache accounting | warm rows only | helps warm-mode, must report cold/amortized cost | medium |

Not allowed silently:

- reducing hidden heads, features, rows, edges, domains, SRS degree, KZG security, opening checks, quotient equations, lookup/table semantics, or verifier checks;
- changing proof schema/transcript/order unless a separate protocol redesign is documented and revalidated;
- hand-patching manifests or result CSV/JSON/summary files.

## Non-Cora Validation Rule

The next non-Cora row must be selected by the bottleneck actually optimized:

- FH/table/context/feature-domain optimization: run Citeseer or OGBN-arxiv.
- edge-domain MSM/route/PSQ/opening optimization: run PubMed or PPI.
- optimization affects both: choose the dataset most likely to amplify it, normally PubMed for current evidence because it has both larger edge-domain pressure and large FH-domain work.

Do not run the full matrix until Cora and a selected non-Cora row demonstrate a general speedup.

## Required Before Final Matrix

Before resuming five-dataset CPU/GPU final rows, write in `performance_readiness_report.md`:

- the real Cora bottleneck;
- the common optimization implemented;
- secure main-path files affected;
- why the change is not Cora-specific;
- which non-Cora full-dataset row validated it;
- which old rows are stale and must be rerun.

Current answer: not ready; no final matrix.
