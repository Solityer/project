#include "gatzk/protocol/cuda_t_fh.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "gatzk/algebra/cuda_field.cuh"
#include "gatzk/algebra/cuda_ntt_device.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/effective_memory.hpp"

namespace gatzk::protocol
{
    namespace
    {
        using GpuFr = algebra::gpu::GpuFr;
        constexpr unsigned kBlockSize = 256;
        constexpr std::size_t kMinimumTileSize = std::size_t{1} << 18;
        constexpr std::size_t kDefaultTileSize = std::size_t{1} << 20;
        constexpr std::size_t kMaximumTileSize = std::size_t{1} << 22;
        constexpr std::size_t kMaximumTargetSize = std::size_t{1} << 27;
        constexpr std::size_t kPipelineSlotCount = 2;

        [[noreturn]] void throw_cuda_t_fh_error(const char *operation, cudaError_t status)
        {
            throw std::runtime_error(std::string("CUDA t_FH ") + operation + " failed: " + cudaGetErrorString(status));
        }

        GpuFr gpu_field(const algebra::FieldElement &value)
        {
            GpuFr out{};
            std::memcpy(&out, &value.native(), sizeof(out));
            return out;
        }

        __global__ void prepare_boundary_denominator_kernel(
            GpuFr *denominators,
            std::size_t logical_begin,
            std::size_t count,
            GpuFr shift,
            GpuFr last_point,
            const GpuFr *lane_root_powers,
            const GpuFr *block_root_powers)
        {
            const auto local = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (local >= count)
                return;

            const auto global = logical_begin + local;
            const auto lane = static_cast<std::size_t>(global & (kBlockSize - 1U));
            const auto block = global / kBlockSize;
            const auto z = algebra::gpu::fr_mul(shift, algebra::gpu::fr_mul(block_root_powers[block], lane_root_powers[lane]));
            const auto first_denominator = algebra::gpu::fr_sub(z, algebra::gpu::fr_one());
            const auto last_denominator = algebra::gpu::fr_sub(z, last_point);
            denominators[local] = algebra::gpu::fr_mul(first_denominator, last_denominator);
        }

        __global__ void evaluate_t_fh_kernel(
            GpuFr *quotient,
            std::size_t logical_begin,
            std::size_t count,
            std::size_t cycle_size,
            GpuFr shift,
            GpuFr last_point,
            GpuFr inv_domain_size,
            GpuFr alpha_quot,
            GpuFr beta_feat,
            GpuFr eta_feat,
            const GpuFr *lane_root_powers,
            const GpuFr *block_root_powers,
            const GpuFr *zero_cycle,
            const GpuFr *zero_inverse_cycle,
            const GpuFr *boundary_inverse_products,
            const GpuFr *q_table,
            const GpuFr *q_query,
            const GpuFr *table,
            const GpuFr *query,
            const GpuFr *multiplicity,
            const GpuFr *j_table,
            const GpuFr *j_query,
            const GpuFr *accumulator,
            const GpuFr *accumulator_next,
            const GpuFr *table_row,
            const GpuFr *table_column,
            const GpuFr *table_feature,
            const GpuFr *query_row,
            const GpuFr *query_column,
            const GpuFr *query_feature,
            bool feature_encoding_terms_enabled)
        {
            const auto local = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (local >= count)
                return;

            const auto global = logical_begin + local;
            const auto lane = static_cast<std::size_t>(global & (kBlockSize - 1U));
            const auto block = global / kBlockSize;
            const auto z = algebra::gpu::fr_mul(shift, algebra::gpu::fr_mul(block_root_powers[block], lane_root_powers[lane]));
            const auto zero = zero_cycle[global % cycle_size];
            const auto zero_inverse = zero_inverse_cycle[global % cycle_size];
            const auto zero_scaled = algebra::gpu::fr_mul(zero, inv_domain_size);
            const auto first_denominator = algebra::gpu::fr_sub(z, algebra::gpu::fr_one());
            const auto last_denominator = algebra::gpu::fr_sub(z, last_point);
            const auto inverse_product = boundary_inverse_products[local];
            const auto first = algebra::gpu::fr_mul(algebra::gpu::fr_mul(zero_scaled, last_denominator), inverse_product);
            const auto last = algebra::gpu::fr_mul(algebra::gpu::fr_mul(algebra::gpu::fr_mul(zero_scaled, last_point), first_denominator), inverse_product);
            const auto not_last = algebra::gpu::fr_sub(algebra::gpu::fr_one(), last);

            const auto term0 = algebra::gpu::fr_mul(first, accumulator[local]);
            auto recurrence = algebra::gpu::fr_sub(accumulator_next[local], accumulator[local]);
            recurrence = algebra::gpu::fr_sub(recurrence, algebra::gpu::fr_mul(algebra::gpu::fr_mul(q_table[local], multiplicity[local]), j_table[local]));
            recurrence = algebra::gpu::fr_add(recurrence, algebra::gpu::fr_mul(q_query[local], j_query[local]));
            const auto term1 = algebra::gpu::fr_mul(not_last, recurrence);
            const auto term2 = algebra::gpu::fr_mul(last, accumulator[local]);
            const auto term3 = algebra::gpu::fr_mul(q_table[local], algebra::gpu::fr_sub(algebra::gpu::fr_mul(algebra::gpu::fr_add(table[local], beta_feat), j_table[local]), algebra::gpu::fr_one()));
            const auto term4 = algebra::gpu::fr_mul(algebra::gpu::fr_sub(algebra::gpu::fr_one(), q_table[local]), j_table[local]);
            const auto term5 = algebra::gpu::fr_mul(q_query[local], algebra::gpu::fr_sub(algebra::gpu::fr_mul(algebra::gpu::fr_add(query[local], beta_feat), j_query[local]), algebra::gpu::fr_one()));
            const auto term6 = algebra::gpu::fr_mul(algebra::gpu::fr_sub(algebra::gpu::fr_one(), q_query[local]), j_query[local]);

            GpuFr numerator;
            if (!feature_encoding_terms_enabled)
            {
                numerator = term6;
                numerator = algebra::gpu::fr_add(term5, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term4, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term3, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term2, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term1, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term0, algebra::gpu::fr_mul(alpha_quot, numerator));
            }
            else
            {
                const auto eta_squared = algebra::gpu::fr_mul(eta_feat, eta_feat);
                auto encoded_table = algebra::gpu::fr_add(table_row[local], algebra::gpu::fr_mul(eta_feat, table_column[local]));
                encoded_table = algebra::gpu::fr_add(encoded_table, algebra::gpu::fr_mul(eta_squared, table_feature[local]));
                auto encoded_query = algebra::gpu::fr_add(query_row[local], algebra::gpu::fr_mul(eta_feat, query_column[local]));
                encoded_query = algebra::gpu::fr_add(encoded_query, algebra::gpu::fr_mul(eta_squared, query_feature[local]));
                const auto term7 = algebra::gpu::fr_sub(table[local], encoded_table);
                const auto term8 = algebra::gpu::fr_sub(query[local], encoded_query);

                numerator = term8;
                numerator = algebra::gpu::fr_add(term7, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term6, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term5, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term4, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term3, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term2, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term1, algebra::gpu::fr_mul(alpha_quot, numerator));
                numerator = algebra::gpu::fr_add(term0, algebra::gpu::fr_mul(alpha_quot, numerator));
            }

            quotient[global] = algebra::gpu::fr_mul(numerator, zero_inverse);
        }

        class CudaTFHExecutorImpl final : public CudaTFHExecutor
        {
        public:
            CudaTFHExecutorImpl(
                std::size_t domain_size,
                std::size_t target_size,
                const algebra::FieldElement &shift,
                const algebra::FieldElement &domain_omega,
                const CudaTFHParameters &parameters,
                std::size_t requested_tile_size)
                : domain_size_(domain_size),
                  target_size_(target_size),
                  shift_(shift),
                  parameters_(parameters),
                  tile_capacity_(std::clamp(requested_tile_size, kMinimumTileSize, kMaximumTileSize))
            {
                if (domain_size_ == 0 || target_size_ == 0 || target_size_ % domain_size_ != 0)
                    throw std::runtime_error("invalid CUDA t_FH domain shape");
                if ((domain_size_ & (domain_size_ - 1U)) != 0U || (target_size_ & (target_size_ - 1U)) != 0U)
                    throw std::runtime_error("CUDA t_FH requires radix-2 domains");
                if (target_size_ > kMaximumTargetSize)
                    throw std::runtime_error("CUDA t_FH target exceeds bounded implementation");
                if (tile_capacity_ > target_size_)
                    tile_capacity_ = target_size_;
                if ((tile_capacity_ & (tile_capacity_ - 1U)) != 0U)
                    throw std::runtime_error("CUDA t_FH tile size must be a power of two");

                const auto root = algebra::FieldElement::root_of_unity(target_size_);
                const auto cycle_size = target_size_ / domain_size_;
                const auto block_count = (target_size_ + kBlockSize - 1U) / kBlockSize;

                std::vector<algebra::FieldElement> lane_root_powers(kBlockSize);
                std::vector<algebra::FieldElement> block_root_powers(block_count);
                std::vector<algebra::FieldElement> zero_cycle(cycle_size);
                std::vector<algebra::FieldElement> zero_inverse_cycle(cycle_size);
                algebra::geometric_powers_native(algebra::FieldElement::mutable_native_array_data(lane_root_powers.data()), lane_root_powers.size(), root);
                algebra::geometric_powers_native(algebra::FieldElement::mutable_native_array_data(block_root_powers.data()), block_root_powers.size(), root.pow(kBlockSize));

                auto zero_power = shift_.pow(domain_size_);
                const auto zero_step = root.pow(domain_size_);
                for (std::size_t i = 0; i < cycle_size; ++i)
                {
                    zero_cycle[i] = zero_power - algebra::FieldElement::one();
                    if (zero_cycle[i].is_zero())
                        throw std::runtime_error("CUDA t_FH coset overlaps FH domain");
                    zero_inverse_cycle[i] = zero_cycle[i].inv();
                    zero_power *= zero_step;
                }

                last_point_ = domain_omega.pow(domain_size_ - 1);
                inv_domain_size_ = algebra::FieldElement(domain_size_).inv();

                allocate_and_upload(&device_lane_root_powers_, lane_root_powers);
                allocate_and_upload(&device_block_root_powers_, block_root_powers);
                allocate_and_upload(&device_zero_cycle_, zero_cycle);
                allocate_and_upload(&device_zero_inverse_cycle_, zero_inverse_cycle);

                const auto quotient_bytes = target_size_ * sizeof(GpuFr);
                auto status = cudaMalloc(reinterpret_cast<void **>(&device_quotient_), quotient_bytes);
                if (status != cudaSuccess)
                    throw_cuda_t_fh_error("quotient allocation", status);
                for (std::size_t slot = 0;
                     slot < kPipelineSlotCount;
                     ++slot)
                {
                    status =
                        cudaStreamCreateWithFlags(
                            &tile_streams_[slot],
                            cudaStreamNonBlocking);

                    if (status !=
                        cudaSuccess)
                    {
                        throw_cuda_t_fh_error(
                            "tile stream creation",
                            status);
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_boundary_products_[slot]),
                            tile_capacity_ *
                                sizeof(GpuFr));

                    if (status !=
                        cudaSuccess)
                    {
                        throw_cuda_t_fh_error(
                            "boundary product allocation",
                            status);
                    }

                    status =
                        cudaMalloc(
                            reinterpret_cast<
                                void **>(
                                &device_boundary_inverse_products_[slot]),
                            tile_capacity_ *
                                sizeof(GpuFr));

                    if (status !=
                        cudaSuccess)
                    {
                        throw_cuda_t_fh_error(
                            "boundary inverse allocation",
                            status);
                    }
                }
            }

            ~CudaTFHExecutorImpl() override
            {
                release_noexcept();
            }

            std::size_t tile_capacity() const override
            {
                return tile_capacity_;
            }

            std::size_t pipeline_slot_count() const override
            {
                return kPipelineSlotCount;
            }

            void wait_for_slot(
                std::size_t slot) override
            {
                if (slot >=
                    kPipelineSlotCount)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_FH pipeline slot");
                }

                if (!slot_in_flight_[slot])
                {
                    return;
                }

                const auto status =
                    cudaStreamSynchronize(
                        tile_streams_[slot]);

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_t_fh_error(
                        "tile completion synchronization",
                        status);
                }

                algebra::
                    record_cuda_stream_sync();

                slot_in_flight_[slot] =
                    false;
            }

            void evaluate_tile(
                const CudaTFHTile &tile,
                std::size_t slot) override
            {
                validate_tile(
                    tile);

                if (slot >=
                    kPipelineSlotCount)
                {
                    throw std::runtime_error(
                        "invalid CUDA t_FH pipeline slot");
                }

                if (slot_in_flight_[slot])
                {
                    throw std::runtime_error(
                        "CUDA t_FH pipeline slot reused before completion");
                }

                const auto blocks =
                    static_cast<unsigned>(
                        (tile.count +
                         kBlockSize - 1U) /
                        kBlockSize);

                const auto stream =
                    tile_streams_[slot];

                prepare_boundary_denominator_kernel<<<
                    blocks,
                    kBlockSize,
                    0,
                    stream>>>(
                    device_boundary_products_[slot],
                    tile.logical_begin,
                    tile.count,
                    gpu_field(shift_),
                    gpu_field(last_point_),
                    device_lane_root_powers_,
                    device_block_root_powers_);

                auto status =
                    cudaGetLastError();

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_t_fh_error(
                        "boundary denominator launch",
                        status);
                }

                status =
                    cudaStreamSynchronize(
                        stream);

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_t_fh_error(
                        "boundary denominator synchronization",
                        status);
                }

                algebra::
                    record_cuda_kernel_invocations(
                        1);

                algebra::
                    record_cuda_stream_sync();

                if (!algebra::
                        try_cuda_batch_inverse_native(
                            reinterpret_cast<
                                const mcl::Fr *>(
                                device_boundary_products_[slot]),
                            reinterpret_cast<
                                mcl::Fr *>(
                                device_boundary_inverse_products_[slot]),
                            tile.count))
                {
                    throw std::runtime_error(
                        "selected CUDA t_FH boundary inverse route was unavailable");
                }

                evaluate_t_fh_kernel<<<
                    blocks,
                    kBlockSize,
                    0,
                    stream>>>(
                    device_quotient_,
                    tile.logical_begin,
                    tile.count,
                    target_size_ /
                        domain_size_,
                    gpu_field(shift_),
                    gpu_field(last_point_),
                    gpu_field(inv_domain_size_),
                    gpu_field(
                        parameters_
                            .alpha_quot),
                    gpu_field(
                        parameters_
                            .beta_feat),
                    gpu_field(
                        parameters_
                            .eta_feat),
                    device_lane_root_powers_,
                    device_block_root_powers_,
                    device_zero_cycle_,
                    device_zero_inverse_cycle_,
                    device_boundary_inverse_products_[slot],
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.q_table),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.q_query),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.table),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.query),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.multiplicity),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.j_table),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.j_query),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.accumulator),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.accumulator_next),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.table_row),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.table_column),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.table_feature),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.query_row),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.query_column),
                    reinterpret_cast<
                        const GpuFr *>(
                        tile.query_feature),
                    parameters_
                        .feature_encoding_terms_enabled);

                status =
                    cudaGetLastError();

                if (status !=
                    cudaSuccess)
                {
                    throw_cuda_t_fh_error(
                        "quotient evaluation launch",
                        status);
                }

                algebra::
                    record_cuda_kernel_invocations(
                        1);

                slot_in_flight_[slot] =
                    true;
            }
            std::shared_ptr<const algebra::CudaResidentCoefficientBuffer> finalize_coefficients(
                algebra::FieldElement *host_coefficients,
                std::size_t host_capacity) override
            {
                if (host_coefficients == nullptr || host_capacity < target_size_)
                    throw std::runtime_error("CUDA t_FH coefficient output is too small");

                for (std::size_t slot = 0;
                     slot < kPipelineSlotCount;
                     ++slot)
                {
                    wait_for_slot(
                        slot);
                }

                release_boundary_buffers();

                mcl::Fr *device_coefficients = nullptr;
                auto status = cudaMalloc(reinterpret_cast<void **>(&device_coefficients), target_size_ * sizeof(mcl::Fr));
                if (status != cudaSuccess)
                    throw_cuda_t_fh_error("resident coefficient allocation", status);

                std::shared_ptr<const algebra::CudaHostTransferGuard> host_transfer;
                std::size_t logical_count = 0;
                std::size_t nonzero_count = 0;
                const bool executed = algebra::cuda_coset_device_to_coefficients_exact(
                    reinterpret_cast<const mcl::Fr *>(device_quotient_),
                    target_size_,
                    shift_,
                    algebra::FieldElement::mutable_native_array_data(host_coefficients),
                    true,
                    device_coefficients,
                    &host_transfer,
                    &logical_count,
                    &nonzero_count);

                if (!executed)
                {
                    cudaFree(device_coefficients);
                    throw std::runtime_error("CUDA t_FH device quotient INTT was not admitted");
                }

                int device = -1;
                status = cudaGetDevice(&device);
                if (status != cudaSuccess)
                {
                    cudaFree(device_coefficients);
                    throw_cuda_t_fh_error("resident coefficient device lookup", status);
                }

                std::shared_ptr<void> allocation(device_coefficients, [device](void *pointer) noexcept
                                                 {
                    if (pointer == nullptr) return;
                    int previous = -1;
                    if (cudaGetDevice(&previous) != cudaSuccess) return;
                    if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
                    (void)cudaFree(pointer);
                    if (previous != device) (void)cudaSetDevice(previous); });

                auto descriptor = std::make_shared<algebra::CudaResidentCoefficientBuffer>();
                descriptor->allocation = std::move(allocation);
                descriptor->device_montgomery_limbs = reinterpret_cast<const std::uint64_t *>(device_coefficients);
                descriptor->capacity = target_size_;
                descriptor->logical_count = logical_count;
                descriptor->nonzero_count = nonzero_count;
                descriptor->nonzero_count_known = true;
                descriptor->device = device;
                descriptor->host_transfer = std::move(host_transfer);
                return descriptor;
            }

        private:
            void allocate_and_upload(GpuFr **device_pointer, const std::vector<algebra::FieldElement> &values)
            {
                const auto bytes = values.size() * sizeof(GpuFr);
                auto status = cudaMalloc(reinterpret_cast<void **>(device_pointer), bytes);
                if (status != cudaSuccess)
                    throw_cuda_t_fh_error("table allocation", status);
                status = cudaMemcpy(*device_pointer, algebra::FieldElement::native_array_data(values.data()), bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess)
                    throw_cuda_t_fh_error("table upload", status);
                algebra::record_cuda_h2d(bytes);
            }

            void validate_tile(const CudaTFHTile &tile) const
            {
                if (tile.count == 0 || tile.count > tile_capacity_)
                    throw std::runtime_error("invalid CUDA t_FH tile count");
                if (tile.logical_begin > target_size_ || tile.count > target_size_ - tile.logical_begin)
                    throw std::runtime_error("CUDA t_FH tile exceeds target");
                if ((tile.logical_begin & (kBlockSize - 1U)) != 0U)
                    throw std::runtime_error("CUDA t_FH tile begin must be block aligned");
                if (tile.q_table == nullptr || tile.q_query == nullptr || tile.table == nullptr || tile.query == nullptr || tile.multiplicity == nullptr || tile.j_table == nullptr || tile.j_query == nullptr || tile.accumulator == nullptr || tile.accumulator_next == nullptr)
                    throw std::runtime_error("CUDA t_FH tile is missing a required source");
                if (parameters_.feature_encoding_terms_enabled && (tile.table_row == nullptr || tile.table_column == nullptr || tile.table_feature == nullptr || tile.query_row == nullptr || tile.query_column == nullptr || tile.query_feature == nullptr))
                    throw std::runtime_error("CUDA t_FH tile is missing feature-encoding sources");
            }

            void release_boundary_buffers() noexcept
            {
                for (std::size_t slot = 0;
                     slot < kPipelineSlotCount;
                     ++slot)
                {
                    if (tile_streams_[slot] !=
                        nullptr)
                    {
                        (void)cudaStreamSynchronize(
                            tile_streams_[slot]);
                    }

                    if (device_boundary_products_[slot] != nullptr)
                    {
                        (void)cudaFree(
                            device_boundary_products_[slot]);

                        device_boundary_products_[slot] = nullptr;
                    }

                    if (device_boundary_inverse_products_[slot] != nullptr)
                    {
                        (void)cudaFree(
                            device_boundary_inverse_products_[slot]);

                        device_boundary_inverse_products_[slot] = nullptr;
                    }

                    if (tile_streams_[slot] !=
                        nullptr)
                    {
                        (void)cudaStreamDestroy(
                            tile_streams_[slot]);

                        tile_streams_[slot] =
                            nullptr;
                    }

                    slot_in_flight_[slot] =
                        false;
                }
            }

            void release_noexcept() noexcept
            {
                release_boundary_buffers();
                if (device_quotient_ != nullptr)
                    (void)cudaFree(device_quotient_);
                if (device_lane_root_powers_ != nullptr)
                    (void)cudaFree(device_lane_root_powers_);
                if (device_block_root_powers_ != nullptr)
                    (void)cudaFree(device_block_root_powers_);
                if (device_zero_cycle_ != nullptr)
                    (void)cudaFree(device_zero_cycle_);
                if (device_zero_inverse_cycle_ != nullptr)
                    (void)cudaFree(device_zero_inverse_cycle_);
                device_quotient_ = nullptr;
                device_lane_root_powers_ = nullptr;
                device_block_root_powers_ = nullptr;
                device_zero_cycle_ = nullptr;
                device_zero_inverse_cycle_ = nullptr;
            }

            std::size_t domain_size_ = 0;
            std::size_t target_size_ = 0;
            algebra::FieldElement shift_;
            algebra::FieldElement last_point_;
            algebra::FieldElement inv_domain_size_;
            CudaTFHParameters parameters_;
            std::size_t tile_capacity_ = kDefaultTileSize;
            GpuFr *device_quotient_ = nullptr;

            std::array<
                GpuFr *,
                kPipelineSlotCount>
                device_boundary_products_{
                    nullptr,
                    nullptr};

            std::array<
                GpuFr *,
                kPipelineSlotCount>
                device_boundary_inverse_products_{
                    nullptr,
                    nullptr};

            std::array<
                cudaStream_t,
                kPipelineSlotCount>
                tile_streams_{
                    nullptr,
                    nullptr};

            std::array<
                bool,
                kPipelineSlotCount>
                slot_in_flight_{
                    false,
                    false};

            GpuFr *device_lane_root_powers_ = nullptr;
            GpuFr *device_block_root_powers_ = nullptr;
            GpuFr *device_zero_cycle_ = nullptr;
            GpuFr *device_zero_inverse_cycle_ = nullptr;
        };
    }

    bool cuda_t_fh_executor_available()
    {
        return algebra::cuda_backend_available() && algebra::cuda_backend_unified_addressing_available();
    }

    bool cuda_t_fh_executor_can_admit(std::size_t domain_size, std::size_t target_size, std::size_t requested_tile_size)
    {
        if (!cuda_t_fh_executor_available() || domain_size == 0 || target_size == 0 || target_size % domain_size != 0)
            return false;
        if ((domain_size & (domain_size - 1U)) != 0U || (target_size & (target_size - 1U)) != 0U || target_size > kMaximumTargetSize)
            return false;

        const auto tile_size = std::clamp(requested_tile_size, kMinimumTileSize, kMaximumTileSize);
        const auto host = util::cached_effective_memory_snapshot();
        if (!host.effective_available_known)
            return false;

        const auto quotient_bytes =
            target_size *
            sizeof(GpuFr);

        const auto tile_scratch_bytes =
            kPipelineSlotCount *
            2 *
            tile_size *
            sizeof(GpuFr);
        const auto reserve = std::max<std::size_t>(std::size_t{16} << 30, host.effective_total_known ? host.effective_total_bytes / 8 : std::size_t{16} << 30);
        if (host.effective_available_bytes <= reserve)
            return false;
        return quotient_bytes + tile_scratch_bytes <= host.effective_available_bytes - reserve;
    }

    std::unique_ptr<CudaTFHExecutor> create_cuda_t_fh_executor(
        std::size_t domain_size,
        std::size_t target_size,
        const algebra::FieldElement &shift,
        const algebra::FieldElement &domain_omega,
        const CudaTFHParameters &parameters,
        std::size_t requested_tile_size)
    {
        if (!cuda_t_fh_executor_can_admit(domain_size, target_size, requested_tile_size))
            throw std::runtime_error("CUDA t_FH executor admission failed");
        return std::make_unique<CudaTFHExecutorImpl>(domain_size, target_size, shift, domain_omega, parameters, requested_tile_size);
    }

    void release_cuda_t_fh_executor_cache()
    {
    }
}