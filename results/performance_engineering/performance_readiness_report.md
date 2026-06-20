# zkGAT Performance Readiness Report

Generated: 2026-06-19 15:55 Asia/Shanghai

Readiness: `NOT_PERFORMANCE_READY`

## Decision

Do not resume the final five-dataset CPU/GPU matrix and do not regenerate `results/paper_secure/latest.csv`, `latest.json`, or `summary.md`.

The latest completed Cora full-dataset official row is correct but fails the performance gate:

- row path: `runs/cora_secure/paper_secure_final/cpu/warm/run_manifest.json`
- runner log: `runs/secure_main_runner/logs/run_cpu_cora_20260619T071854Z.log`
- `total_wall_clock_ms=785,469.068`
- required Cora CPU hard gate: `<60,000 ms`
- gap: `13.09x` over hard gate
- gap to `10s`: `78.55x`
- `official_runner_wall_clock_ms=867,633.034`
- `runner_overhead_ms=82,163.966`

Security/status checks for that row passed:

- `verified=true`
- `proof_payload_audit_status=passed`
- `reference_consistency_passed=true`
- `tamper_tests_passed=true`
- `verifier_reads_witness=false`
- `private_plain_value_count=0`

## Amdahl Decision

The Amdahl upper-bound section in `performance_budget_report.md` is now the gate before any new Cora official row.

Counterfactual upper bounds from the latest Cora row:

- If all `quotient_build_ms` became zero, Cora would still be `436,095.196 ms` (`1.80x` upper-bound speedup).
- If all `domain_opening_ms` became zero, Cora would still be `526,393.378 ms` (`1.49x` upper-bound speedup).
- If all `quotient_commitment_ms` became zero, Cora would still be `692,071.237 ms` (`1.13x` upper-bound speedup).
- If all `commit_dynamic_ms` became zero, Cora would still be `671,579.666 ms` (`1.17x` upper-bound speedup).

Therefore single-stage or local quotient materialization patches cannot make Cora `<60s` and cannot approach `10s`.

## Current Local Patch Status

The current working tree contains a local t_FH/t_edge coset-fusion and linear-coset-sum patch. It has passed correctness/security guards:

- `cmake --build build -j 12`: passed
- `python3 tools/security/static_scan_no_tau.py`: passed
- `ctest --test-dir build --output-on-failure`: passed
- `cmake --build build_cuda -j 12`: passed
- `ctest --test-dir build_cuda --output-on-failure`: passed
- `GATZK_RUN_FULL_SECURITY_TESTS=1 build/gatzk_tests`: passed, including reference consistency, verifier-only boundary, and tamper rejection suite

This does not make the patch a valid main performance route. Amdahl analysis shows that even deleting the entire quotient build would not reach `<60s`; the patch affects only a subset of quotient materialization and does not reduce proof object count, commitment count, opening count, or KZG witness equations.

Status of this patch:

- allowed only as an auxiliary optimization if later structural routes make it useful;
- not sufficient to trigger a new Cora official row;
- not sufficient to mark readiness conditional or passed;
- may be reverted if it complicates the structural route.

## Current Bottleneck

Cora is slow because of real secure proof computation across multiple common paths:

- dynamic KZG commitments: `commit_dynamic_ms=113,889.402`, `dynamic_commit_msm_ms=113,527.543`;
- quotient construction: `quotient_build_ms=349,373.872`, including `quotient_t_fh_ms=253,648.317` and `quotient_t_edge_ms=214,312.876`;
- quotient commitment: `quotient_commitment_ms=93,397.831`;
- domain opening: `domain_opening_ms=259,075.690`, mostly `domain_open_FH_ms=255,265.889` and `fh_open_witness_ms=245,564.321`;
- external opening: `external_opening_ms=34,977.816`;
- verify: `verify_time_ms=8,354.370`;
- runner overhead: `82,163.966 ms`.

The row is not slow because of logging, audit, reference consistency, tamper tests, or final export.

## Required Next Route

The structural route plan is now in `results/performance_engineering/order_of_magnitude_redesign_plan.md`.
The GPU code-audit gap report is `results/performance_engineering/gpu_backend_gap_report.md`.

Next main route priority:

1. true GPU backend for real KZG MSM, NTT/IFFT, opening witness, and quotient kernels;
2. domain-grouped multi-opening aggregation and proof-object/domain-level aggregation;
3. streaming quotient-to-MSM and fused execution;
4. cross-object reuse and public/static precomputation only as auxiliary routes with cold/warm/amortized accounting.

A new Cora official row is forbidden until a structural prototype or patch has stage benchmark evidence and Amdahl modeling predicts at least `2x` end-to-end speedup.

## Non-Cora Evidence

Latest available PubMed full-dataset profile remains stale for final use but supports route selection:

- `commit_dynamic_ms=517,208.607`
- `quotient_build_ms=1,476,653.437`
- `quotient_t_fh_ms=1,091,480.221`
- `quotient_t_edge_ms=449,073.490`
- `domain_opening_ms=948,546.527`
- `fh_open_witness_ms=916,202.968`

This confirms that FH-domain and edge-domain bottlenecks scale beyond Cora. It does not provide final readiness evidence because it is not from the current source/binary.

## Final Matrix Gate

Final `results/paper_secure` remains blocked until:

- `performance_readiness_report.md` explicitly says `PASS`;
- Cora/Citeseer CPU are in the target order of magnitude;
- PubMed/PPI CPU are in the target order of magnitude or have a strict scale-backed acceptance rationale;
- OGBN-arxiv is near the target order of magnitude;
- GPU/CUDA backend truly accelerates the main bottlenecks rather than CPU fallback;
- all 10 final rows are full-dataset `VERIFY_OK` with audit/reference/tamper pass from frozen source/binary;
- `comparison/**` remains ignored.
