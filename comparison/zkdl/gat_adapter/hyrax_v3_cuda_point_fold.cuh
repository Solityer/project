#pragma once

#include "../proof.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#if !defined(ZKDL_USE_BLST_G1_CODEC)
#error "Hyrax v3 CUDA point fold requires the baseline-local BLST G1 codec"
#endif

namespace zkdl_cuda_fold {

// The packed coordinates are native BLS12-381 Fp Montgomery residues.  Each
// coordinate occupies twelve little-significance-first u32 limbs (x, y, z).
struct MontJacobianPoint {
    std::array<std::uint32_t, 36U> limb{};
};
static_assert(sizeof(MontJacobianPoint) == 144U,
              "packed Montgomery Jacobian points must be 144 bytes");

// These digests are supplied by the caller after authenticating/loading the
// basis and binding this session to its proof context.  This adapter retains
// the binding for the whole session; it does not authenticate either digest.
struct PointFoldSourceBinding {
    std::array<std::uint8_t, 32U> authenticated_basis_manifest_sha256{};
    std::array<std::uint8_t, 32U> proof_context_sha256{};
};

struct CudaDeviceIdentity {
    int ordinal = -1;
    std::array<std::uint8_t, 16U> uuid{};
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
};

struct PointFoldStats {
    CudaDeviceIdentity device{};
    PointFoldSourceBinding source_binding{};
    std::size_t reserved_current_count = 0U;
    std::size_t last_current_count = 0U;
    std::uint64_t host_pinned_buffer_bytes = 0U;
    std::uint64_t device_buffer_bytes = 0U;
    std::uint64_t peak_buffer_allocation_bytes = 0U;
    std::uint64_t successful_fold_calls = 0U;
    std::uint64_t kernel_launch_count = 0U;
    std::uint64_t h2d_copy_count = 0U;
    std::uint64_t d2h_copy_count = 0U;
    std::uint64_t bytes_h2d = 0U;
    std::uint64_t bytes_d2h = 0U;
    float last_h2d_event_ms = 0.0F;
    float last_kernel_event_ms = 0.0F;
    float last_d2h_event_ms = 0.0F;
    double cumulative_h2d_event_ms = 0.0;
    double cumulative_kernel_event_ms = 0.0;
    double cumulative_d2h_event_ms = 0.0;
    std::uint64_t free_device_bytes_at_construction = 0U;
    std::uint64_t total_device_bytes_at_construction = 0U;
};

CudaDeviceIdentity query_cuda_device_identity(int ordinal);

// Explicit limb split/merge.  The BLST Fp and local Fp_t have the same
// BLS12-381 Montgomery residue R=2^384; no field conversion is performed.
MontJacobianPoint pack_blst_p1(const zkdl_blst::blst_p1& point);
zkdl_blst::blst_p1 unpack_blst_p1(const MontJacobianPoint& point);

class HyraxV3CudaPointFoldSession {
public:
    HyraxV3CudaPointFoldSession(
        std::size_t max_current_count,
        const CudaDeviceIdentity& fixed_device,
        const PointFoldSourceBinding& source_binding);
    ~HyraxV3CudaPointFoldSession();

    HyraxV3CudaPointFoldSession(const HyraxV3CudaPointFoldSession&) = delete;
    HyraxV3CudaPointFoldSession& operator=(const HyraxV3CudaPointFoldSession&) = delete;
    HyraxV3CudaPointFoldSession(HyraxV3CudaPointFoldSession&&) = delete;
    HyraxV3CudaPointFoldSession& operator=(HyraxV3CudaPointFoldSession&&) = delete;

    // Applies the Hyrax v3 contiguous-half generator fold
    // out[i] = inverse * current[i] + challenge * current[M/2+i].
    // M must be a power of two >= 2 and no larger than the constructor bound.
    // Scalar wires are canonical, nonzero Fr residues and must be inverses.
    // Point authentication and SRS provenance are the caller's responsibility.
    std::vector<MontJacobianPoint> fold(
        const MontJacobianPoint* current, std::size_t current_count,
        const zkdl::FrWire& challenge, const zkdl::FrWire& inverse);

    const PointFoldStats& stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace zkdl_cuda_fold
