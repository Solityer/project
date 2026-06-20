# zkGAT GPU Backend Gap Report

Generated: 2026-06-19 15:55 Asia/Shanghai

This report records the Route B code audit. It is not readiness evidence and does not authorize GPU final rows.

## Current Implementation

CUDA build support exists only as an optional algebra-hotspot layer:

- `CMakeLists.txt` includes `src/algebra/cuda_backend.cu` only when `GATZK_ENABLE_CUDA_BACKEND=ON`.
- `src/main.cpp` accepts `--compute-backend cuda_hotspots`, but still sets `GATZK_ALGEBRA_BACKEND=cpu`.
- `metrics.gpu_acceleration_status` explicitly records: `accelerated_stages=barycentric_weights,packed_weighted_evaluation,lookup_histogram_size_aware; cpu_fallback=KZG_MSM,pairing,small_domains`.
- `src/algebra/cuda_backend.cu` implements Fr barycentric weights, weighted polynomial evaluation, and lookup histogram kernels.

The current GPU path does not implement the dominant bottlenecks:

- no CUDA G1 MSM;
- no CUDA KZG commitment path;
- no CUDA opening witness path;
- no CUDA pairing path;
- no CUDA NTT/IFFT path for quotient materialization;
- no GPU SRS upload/cache for `powers_g1` or Lagrange SRS material.

## Crypto Path Evidence

KZG commitment and opening still call the CPU curve backend:

- `src/crypto/kzg.cpp` calls `g1_multi_scalar_mul_profiled()` through `commit_polynomial_point()`.
- `src/crypto/kzg.cpp` builds opening witnesses through `open_single_point_fold()` and again calls `commit_polynomial_point()`.
- `src/crypto/curve.cpp` implements `g1_multi_scalar_mul_profiled()` with `mcl::G1::mulVec`, optional CPU chunking, and sparse compaction.
- `src/crypto/curve.cpp` has no CUDA branch, device SRS representation, GPU point arithmetic, or GPU MSM dispatch.

The third-party `mcl` tree available in this repository does not expose a ready CUDA G1 MSM integration path for the current CMake build.

## Performance Implication

Latest Cora bottlenecks require true crypto/NTT acceleration:

- `dynamic_commit_msm_ms=113,527.543`
- `quotient_commitment_ms=93,397.831`
- `fh_open_witness_ms=245,564.321`
- `quotient_build_ms=349,373.872`

Existing CUDA hotspots do not target these stages. A GPU row produced by the current backend would be a partial algebra-hotspot row with CPU KZG fallback, not a performance PASS row.

## Minimum Prototype Definition

The next true GPU prototype must target large SRS-backed G1 MSM first.

Prototype scope:

1. Add a GPU-capable MSM dispatch behind `g1_multi_scalar_mul_profiled()`.
2. Support dense SRS-backed MSM sizes at least `2^22` to match `t_FH` quotient commitment and FH opening witness scale.
3. Include explicit SRS transfer/cache timing in metrics.
4. Return exactly the same `G1Point` as the CPU `mcl::G1::mulVec` path on deterministic test vectors.
5. Fall back to CPU with an explicit manifest flag if GPU runtime, memory, or kernel support is unavailable.

Prototype success gate before any Cora official row:

- unit/stage benchmark shows at least `8x` speedup on an 8M-scale G1 MSM including transfer/cache accounting;
- correctness test compares GPU and CPU commitments/witnesses;
- Amdahl model predicts at least `2x` Cora end-to-end speedup in combination with the existing stages;
- CPU and CUDA builds/tests/static scan/full security regression pass.

If this prototype cannot meet the stage gate, the next main route is domain-grouped multi-opening aggregation, not another worker/thread/local quotient patch.
