#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/algebra/cuda_ntt_device.hpp"
#include "gatzk/algebra/cuda_weighted_session.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

#include "gatzk/util/effective_memory.hpp"
#include "gatzk/util/route2.hpp"

#if GATZK_ENABLE_CUDA_BACKEND
#include <cuda_runtime_api.h>
#endif

namespace gatzk::algebra
{
#if GATZK_ENABLE_CUDA_BACKEND
    bool cuda_backend_runtime_available();
    bool cuda_field_hotspots_available();
    void scaled_accumulate_cuda(uint64_t *target_limbs, const uint64_t *source_limbs, const uint64_t scale_limbs[4], std::size_t element_count);

    void scaled_accumulate_pair_cuda(uint64_t *first_target_limbs, uint64_t *second_target_limbs, const uint64_t *source_limbs,
                                     const uint64_t first_scale_limbs[4], const uint64_t second_scale_limbs[4], std::size_t element_count);

    bool batch_inverse_cuda(const mcl::Fr *input, mcl::Fr *output, std::size_t element_count);

    bool opening_inverse_denominators_cuda(mcl::Fr *output, std::size_t domain_size, const mcl::Fr &omega, const mcl::Fr &point);

    bool opening_quotient_assembly_cuda(const mcl::Fr *folded_values, mcl::Fr *quotient_values, const mcl::Fr &scaled_claim_sum, std::size_t element_count);

    bool opening_inverse_and_quotient_cuda(const mcl::Fr *folded_values, mcl::Fr *quotient_values, const mcl::Fr &scaled_claim_sum, std::size_t domain_size,
                                           const mcl::Fr &omega, const mcl::Fr &point);

    bool opening_inverse_and_quotient_resident_cuda(const mcl::Fr *folded_values, const mcl::Fr &scaled_claim_sum, std::size_t domain_size,
                                                    const mcl::Fr &omega, const mcl::Fr &point,
                                                    std::shared_ptr<const CudaResidentCoefficientBuffer> *resident_output);

    bool coefficient_opening_quotient_resident_cuda(const mcl::Fr *coefficients, std::size_t coefficient_count, const mcl::Fr &point,
                                                    const mcl::Fr &claimed_value, std::shared_ptr<const CudaResidentCoefficientBuffer> *resident_output);

    bool logup_inverse_and_accumulator_cuda(const mcl::Fr *table, const mcl::Fr *query, const mcl::Fr *q_table, const mcl::Fr *q_query, const mcl::Fr &beta,
                                            const mcl::Fr *multiplicity, mcl::Fr *table_inverse, mcl::Fr *query_inverse, mcl::Fr *accumulator,
                                            std::size_t active_count, std::size_t element_count, CudaLogupExecutionProfile *profile);

    bool logup_inverse_columns_cuda(const mcl::Fr *table, const mcl::Fr *query, const mcl::Fr *q_table, const mcl::Fr *q_query, const mcl::Fr &beta,
                                    mcl::Fr *table_inverse, mcl::Fr *query_inverse, std::size_t element_count);

    bool logup_accumulator_cuda(const mcl::Fr *table_inverse, const mcl::Fr *query_inverse, const mcl::Fr *multiplicity, const mcl::Fr *q_table,
                                const mcl::Fr *q_query, mcl::Fr *accumulator, std::size_t active_count, std::size_t element_count);

    bool scaled_prefix_accumulator_cuda(const mcl::Fr *values, const mcl::Fr &scale, mcl::Fr *accumulator, std::size_t active_count, std::size_t element_count);

    bool cuda_intt_in_place_exact(mcl::Fr *values, std::size_t size);

    bool cuda_ntt_in_place_exact(mcl::Fr *values, std::size_t size);

    bool cuda_ntt_pair_in_place_exact(mcl::Fr *first, mcl::Fr *second, std::size_t size);

    bool cuda_taylor_convolution_in_place_exact(mcl::Fr *values, const mcl::Fr *kernel_ntt, std::size_t size);
#endif

    namespace
    {
        std::atomic<std::size_t> &cuda_kernel_counter()
        {
            static std::atomic<std::size_t> counter{0};
            return counter;
        }

        struct CudaExecutionCounters
        {
            std::atomic<std::uint64_t> h2d_bytes{0};
            std::atomic<std::uint64_t> d2h_bytes{0};
            std::atomic<std::uint64_t> h2d_operations{0};
            std::atomic<std::uint64_t> d2h_operations{0};
            std::atomic<std::uint64_t> stream_synchronizations{0};
            std::atomic<std::uint64_t> ntt_calls{0};
            std::atomic<std::uint64_t> intt_calls{0};
            std::atomic<std::uint64_t> quotient_calls{0};
            std::atomic<std::uint64_t> t_edge_calls{0};
            std::atomic<std::uint64_t> t_fh_calls{0};
            std::atomic<std::uint64_t> opening_fold_calls{0};
            std::atomic<std::uint64_t> msm_single_calls{0};
            std::atomic<std::uint64_t> msm_batch_calls{0};
            std::atomic<std::uint64_t> resident_hits{0};
            std::atomic<std::uint64_t> resident_misses{0};
        };

        CudaExecutionCounters &cuda_execution_counters()
        {
            static CudaExecutionCounters counters;
            return counters;
        }

        bool cuda_field_route_enabled()
        {
            return util::route2_options().cuda_field_hotspots && cuda_field_hotspots_build_enabled() && cuda_field_hotspots_available();
        }

        bool cuda_trace_route_enabled() { return util::route2_options().cuda_trace_hotspots && cuda_backend_available(); }

        unsigned __int128 multiply_u128(std::size_t lhs, std::size_t rhs) { return static_cast<unsigned __int128>(lhs) * static_cast<unsigned __int128>(rhs); }

        std::size_t bounded_worker_count(std::size_t max_worker_count)
        {
            const auto hardware_workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());

            if (max_worker_count == 0)
            {
                return hardware_workers;
            }

            return std::max<std::size_t>(1, std::min<std::size_t>(hardware_workers, max_worker_count));
        }

        AlgebraBackend parse_backend_env()
        {
            const char *value = std::getenv("GATZK_ALGEBRA_BACKEND");
            if (value == nullptr)
            {
                return AlgebraBackend::Cpu;
            }
            const std::string backend(value);

            if (backend == "cpu")
            {
                return AlgebraBackend::Cpu;
            }
            if (backend == "cuda")
            {
                throw std::runtime_error("GATZK_ALGEBRA_BACKEND=cuda is unsupported; use --compute-backend cuda_hotspots on a CUDA-enabled build");
            }
            throw std::runtime_error("unsupported GATZK_ALGEBRA_BACKEND value: " + backend);
        }

        FieldElement dot_product_range(const std::vector<FieldElement> &lhs, const std::vector<FieldElement> &rhs, std::size_t begin, std::size_t end)
        {
            mcl::Fr sum;
            sum.clear();
            for (std::size_t i = begin; i < end; ++i)
            {
                mcl::Fr term;
                mcl::Fr::mul(term, lhs[i].native(), rhs[i].native());
                mcl::Fr::add(sum, sum, term);
            }
            return FieldElement::from_native(sum);
        }

        FieldElement dot_product_native_range(const std::vector<FieldElement> &lhs, const std::vector<mcl::Fr> &rhs, std::size_t begin, std::size_t end)
        {
            mcl::Fr sum;
            sum.clear();
            for (std::size_t i = begin; i < end; ++i)
            {
                mcl::Fr term;
                mcl::Fr::mul(term, lhs[i].native(), rhs[i]);
                mcl::Fr::add(sum, sum, term);
            }
            return FieldElement::from_native(sum);
        }

    }

    AlgebraBackend configured_algebra_backend()
    {
        static const auto backend = parse_backend_env();
        return backend;
    }

    std::string configured_algebra_backend_name()
    {
        (void)configured_algebra_backend();
        return "cpu";
    }

    std::shared_ptr<const mcl::Ntt<mcl::Fr>> ntt_plan(std::size_t size)
    {
        struct CachedPlan
        {
            std::shared_ptr<const mcl::Ntt<mcl::Fr>> plan;
            std::size_t bytes = 0;
            std::size_t last_use = 0;
        };

        static std::mutex mutex;
        static std::unordered_map<std::size_t, CachedPlan> cache;
        static std::size_t cached_bytes = 0;
        static std::size_t use_clock = 0;

        static constexpr std::size_t kMinimumCacheBudget = std::size_t{64} << 20;

        static constexpr std::size_t kDefaultCacheBudget = std::size_t{512} << 20;

        static constexpr std::size_t kMaximumCacheBudget = std::size_t{2} << 30;

        static constexpr std::size_t kPinnedSmallPlanMaximum = std::size_t{1} << 20;

        const auto cache_budget = []() -> std::size_t
        {
            const auto memory = util::cached_effective_memory_snapshot();

            std::size_t budget = kDefaultCacheBudget;

            if (memory.effective_total_known)
            {
                budget = std::clamp(memory.effective_total_bytes / std::size_t{16}, kMinimumCacheBudget, kMaximumCacheBudget);
            }

            if (memory.effective_available_known)
            {
                const auto available_budget = std::max(kMinimumCacheBudget, memory.effective_available_bytes / std::size_t{8});

                budget = std::min(budget, available_budget);
            }

            return budget;
        }();

        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto it = cache.find(size);

            if (it != cache.end())
            {
                it->second.last_use = ++use_clock;
                return it->second.plan;
            }
        }

        auto plan = std::make_shared<mcl::Ntt<mcl::Fr>>();
        const mcl::Fr generator = 5;
        const mcl::Vint root_exponent = (mcl::Fr::getOp().mp - 1) >> 32;

        mcl::Fr::pow(plan->root_, generator, root_exponent);

        if (!plan->init(size))
        {
            throw std::runtime_error("failed to initialize NTT plan for size " + std::to_string(size));
        }

        if (!(FieldElement::from_native(plan->getW()) == FieldElement::root_of_unity(size)))
        {
            throw std::runtime_error("NTT root does not match protocol domain root for size " + std::to_string(size));
        }

        const auto plan_bytes = mcl::Ntt<mcl::Fr>::requiredByteSize(size);

        std::lock_guard<std::mutex> lock(mutex);

        if (const auto existing = cache.find(size); existing != cache.end())
        {
            existing->second.last_use = ++use_clock;
            return existing->second.plan;
        }

        if (plan_bytes <= cache_budget)
        {
            while (cached_bytes > cache_budget - plan_bytes)
            {
                auto victim = cache.end();

                for (auto it = cache.begin(); it != cache.end(); ++it)
                {
                    if (it->first <= kPinnedSmallPlanMaximum)
                    {
                        continue;
                    }

                    if (victim == cache.end() || it->second.last_use < victim->second.last_use)
                    {
                        victim = it;
                    }
                }

                if (victim == cache.end())
                {
                    break;
                }

                cached_bytes -= victim->second.bytes;
                cache.erase(victim);
            }

            if (cached_bytes <= cache_budget - plan_bytes)
            {
                CachedPlan entry;
                entry.plan = plan;
                entry.bytes = plan_bytes;
                entry.last_use = ++use_clock;

                cached_bytes += plan_bytes;

                const auto [it, inserted] = cache.emplace(size, std::move(entry));

                if (inserted)
                {
                    return it->second.plan;
                }

                cached_bytes -= plan_bytes;
                it->second.last_use = ++use_clock;
                return it->second.plan;
            }
        }

        return plan;
    }
    void geometric_powers_native(mcl::Fr *output, std::size_t size, const FieldElement &base, std::size_t max_worker_count)
    {
        if (size == 0)
        {
            return;
        }
        if (output == nullptr)
        {
            throw std::runtime_error("geometric powers require a non-null output");
        }
        constexpr std::size_t kMinRowsPerTask = 1ULL << 12;
        const auto available_workers = bounded_worker_count(max_worker_count);
        const auto task_count =
            size < kMinRowsPerTask ? std::size_t{1} : std::min<std::size_t>(available_workers, std::max<std::size_t>(2, size / kMinRowsPerTask));
        const auto chunk_size = (size + task_count - 1) / task_count;
        const auto base_native = base.native();
        std::vector<std::future<void>> futures;
        futures.reserve(task_count);
        for (std::size_t task = 0; task < task_count; ++task)
        {
            const auto begin = task * chunk_size;
            const auto end = std::min(size, begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(std::launch::async,
                                         [=]
                                         {
                                             auto power = base.pow(static_cast<std::uint64_t>(begin)).native();
                                             for (std::size_t index = begin; index < end; ++index)
                                             {
                                                 output[index] = power;
                                                 mcl::Fr::mul(power, power, base_native);
                                             }
                                         }));
        }
        for (auto &future : futures)
        {
            future.get();
        }
    }

    void scale_by_geometric_powers_native(mcl::Fr *values, std::size_t size, const FieldElement &base, std::size_t max_worker_count)
    {
        if (size == 0)
        {
            return;
        }

        if (values == nullptr)
        {
            throw std::runtime_error("geometric in-place scaling requires a non-null buffer");
        }

        constexpr std::size_t kMinRowsPerWorker = std::size_t{1} << 18;

        const auto available_workers = bounded_worker_count(max_worker_count);

        const auto worker_count = std::min<std::size_t>({
            8,
            available_workers,
            std::max<std::size_t>(1, (size + kMinRowsPerWorker - 1) / kMinRowsPerWorker),
        });

        const auto base_native = base.native();

        const auto scale_range = [&](std::size_t begin, std::size_t end)
        {
            auto power = base.pow(static_cast<std::uint64_t>(begin)).native();

            for (std::size_t index = begin; index < end; ++index)
            {
                mcl::Fr::mul(values[index], values[index], power);

                mcl::Fr::mul(power, power, base_native);
            }
        };

        if (worker_count <= 1)
        {
            scale_range(0, size);
            return;
        }

        const auto chunk_size = (size + worker_count - 1) / worker_count;

        std::vector<std::future<void>> futures;

        futures.reserve(worker_count);

        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            const auto begin = worker * chunk_size;

            const auto end = std::min(size, begin + chunk_size);

            if (begin >= end)
            {
                break;
            }

            futures.push_back(std::async(std::launch::async, [&, begin, end]() { scale_range(begin, end); }));
        }

        for (auto &future : futures)
        {
            future.get();
        }
    }

    void copy_scaled_by_geometric_powers_native(const mcl::Fr *input, mcl::Fr *output, std::size_t size, const FieldElement &base, std::size_t max_worker_count)
    {
        if (size == 0)
        {
            return;
        }

        if (input == nullptr || output == nullptr)
        {
            throw std::runtime_error("geometric copy scaling requires non-null buffers");
        }

        constexpr std::size_t kMinRowsPerWorker = std::size_t{1} << 18;

        const auto available_workers = bounded_worker_count(max_worker_count);

        const auto worker_count = std::min<std::size_t>({
            8,
            available_workers,
            std::max<std::size_t>(1, (size + kMinRowsPerWorker - 1) / kMinRowsPerWorker),
        });

        const auto base_native = base.native();

        const auto copy_range = [&](std::size_t begin, std::size_t end)
        {
            auto power = base.pow(static_cast<std::uint64_t>(begin)).native();

            for (std::size_t index = begin; index < end; ++index)
            {
                mcl::Fr::mul(output[index], input[index], power);

                mcl::Fr::mul(power, power, base_native);
            }
        };

        if (worker_count <= 1)
        {
            copy_range(0, size);
            return;
        }

        const auto chunk_size = (size + worker_count - 1) / worker_count;

        std::vector<std::future<void>> futures;

        futures.reserve(worker_count);

        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            const auto begin = worker * chunk_size;

            const auto end = std::min(size, begin + chunk_size);

            if (begin >= end)
            {
                break;
            }

            futures.push_back(std::async(std::launch::async, [&, begin, end]() { copy_range(begin, end); }));
        }

        for (auto &future : futures)
        {
            future.get();
        }
    }

    void split_coset_coefficients_native(const mcl::Fr *coefficients, const mcl::Fr *shift_powers, std::size_t coefficient_count, const FieldElement &odd_root,
                                         mcl::Fr *even_output, mcl::Fr *odd_output, std::size_t max_worker_count)
    {
        if (coefficient_count == 0)
        {
            return;
        }

        if (coefficients == nullptr || shift_powers == nullptr || even_output == nullptr || odd_output == nullptr)
        {
            throw std::runtime_error("coset split scaling requires non-null buffers");
        }

        constexpr std::size_t kMinRowsPerTask = 1ULL << 18;

        const auto available_workers = bounded_worker_count(max_worker_count);

        const auto worker_count = std::min<std::size_t>({
            8,
            available_workers,
            std::max<std::size_t>(1, (coefficient_count + kMinRowsPerTask - 1) / kMinRowsPerTask),
        });

        const auto chunk_size = (coefficient_count + worker_count - 1) / worker_count;

        const auto root_native = odd_root.native();

        std::vector<std::future<void>> futures;
        futures.reserve(worker_count);

        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            const auto begin = worker * chunk_size;
            const auto end = std::min(coefficient_count, begin + chunk_size);

            if (begin >= end)
            {
                break;
            }

            futures.push_back(std::async(std::launch::async,
                                         [=]
                                         {
                                             auto root_power = odd_root.pow(begin).native();

                                             for (std::size_t index = begin; index < end; ++index)
                                             {
                                                 mcl::Fr::mul(even_output[index], coefficients[index], shift_powers[index]);

                                                 mcl::Fr odd_shift_power;
                                                 mcl::Fr::mul(odd_shift_power, shift_powers[index], root_power);

                                                 mcl::Fr::mul(odd_output[index], coefficients[index], odd_shift_power);

                                                 mcl::Fr::mul(root_power, root_power, root_native);
                                             }
                                         }));
        }

        for (auto &future : futures)
        {
            future.get();
        }
    }

    void interleave_native_lanes(const mcl::Fr *even_values, const mcl::Fr *odd_values, std::size_t lane_size, mcl::Fr *output, std::size_t max_worker_count)
    {
        if (lane_size == 0)
        {
            return;
        }

        if (even_values == nullptr || odd_values == nullptr || output == nullptr)
        {
            throw std::runtime_error("native lane interleave requires non-null buffers");
        }

        constexpr std::size_t kMinRowsPerTask = 1ULL << 18;

        const auto available_workers = bounded_worker_count(max_worker_count);

        const auto worker_count = std::min<std::size_t>({
            8,
            available_workers,
            std::max<std::size_t>(1, (lane_size + kMinRowsPerTask - 1) / kMinRowsPerTask),
        });

        const auto chunk_size = (lane_size + worker_count - 1) / worker_count;

        std::vector<std::future<void>> futures;
        futures.reserve(worker_count);

        for (std::size_t worker = 0; worker < worker_count; ++worker)
        {
            const auto begin = worker * chunk_size;
            const auto end = std::min(lane_size, begin + chunk_size);

            if (begin >= end)
            {
                break;
            }

            futures.push_back(std::async(std::launch::async,
                                         [=]
                                         {
                                             for (std::size_t index = begin; index < end; ++index)
                                             {
                                                 output[2 * index] = even_values[index];
                                                 output[2 * index + 1] = odd_values[index];
                                             }
                                         }));
        }

        for (auto &future : futures)
        {
            future.get();
        }
    }

    namespace
    {
        void radix2_ntt_in_place_parallel_impl(mcl::Fr *values, std::size_t size, std::size_t transform_workers, bool inverse)
        {
            const auto plan = ntt_plan(size);

            constexpr std::size_t kStageParallelThreshold = std::size_t{1} << 15;

            if (size < kStageParallelThreshold || transform_workers < 2)
            {
                if (inverse)
                    plan->intt(values);
                else
                    plan->ntt(values);
                return;
            }

            const auto worker_budget = std::max<std::size_t>(1, transform_workers);

            const auto parallel_ranges = [&](std::size_t item_count, std::size_t minimum_items_per_worker, const auto &range_fn)
            {
                if (item_count == 0)
                    return;

                const auto useful_workers = std::max<std::size_t>(1, 1 + (item_count - 1) / minimum_items_per_worker);

                const auto worker_count = std::min<std::size_t>({
                    worker_budget,
                    item_count,
                    useful_workers,
                });

                if (worker_count <= 1)
                {
                    range_fn(0, item_count);
                    return;
                }

                const auto chunk_size = (item_count + worker_count - 1) / worker_count;

                std::vector<std::future<void>> futures;
                futures.reserve(worker_count - 1);

                for (std::size_t worker = 1; worker < worker_count; ++worker)
                {
                    const auto begin = worker * chunk_size;
                    const auto end = std::min(item_count, begin + chunk_size);

                    if (begin >= end)
                        break;

                    futures.push_back(std::async(std::launch::async, [&, begin, end] { range_fn(begin, end); }));
                }

                range_fn(0, std::min(item_count, chunk_size));

                for (auto &future : futures)
                    future.get();
            };

            parallel_ranges(size, std::size_t{1} << 16,
                            [&](std::size_t begin, std::size_t end)
                            {
                                for (std::size_t index = begin; index < end; ++index)
                                {
                                    const auto reversed = static_cast<std::size_t>(plan->br_.rev(index, plan->bitN_));

                                    if (index < reversed)
                                    {
                                        std::swap(values[index], values[reversed]);
                                    }
                                }
                            });

            const auto *weights = plan->getWs();

            std::size_t twiddle_stride = size;

            for (std::size_t lane_size = 1; lane_size < size; lane_size <<= 1U)
            {
                twiddle_stride >>= 1U;

                const auto butterfly_count = size / 2;

                parallel_ranges(butterfly_count, std::size_t{1} << 14,
                                [&](std::size_t begin, std::size_t end)
                                {
                                    for (std::size_t flat = begin; flat < end; ++flat)
                                    {
                                        const auto block = flat / lane_size;
                                        const auto offset = flat - block * lane_size;
                                        const auto left = block * (lane_size * 2) + offset;
                                        const auto right = left + lane_size;
                                        const auto twiddle_index = offset * twiddle_stride;
                                        const auto weight_index = inverse ? size - twiddle_index : twiddle_index;

                                        mcl::Fr product;
                                        mcl::Fr::mul(product, values[right], weights[weight_index]);

                                        const auto left_value = values[left];

                                        mcl::Fr::sub(values[right], left_value, product);

                                        mcl::Fr::add(values[left], left_value, product);
                                    }
                                });
            }

            if (!inverse)
                return;

            const auto inverse_size = plan->invN_;

            parallel_ranges(size, std::size_t{1} << 16,
                            [&](std::size_t begin, std::size_t end)
                            {
                                for (std::size_t index = begin; index < end; ++index)
                                {
                                    mcl::Fr::mul(values[index], values[index], inverse_size);
                                }
                            });
        }

        void ntt_in_place_parallel_impl(mcl::Fr *values, std::size_t size, std::size_t transform_workers)
        {
            radix2_ntt_in_place_parallel_impl(values, size, transform_workers, false);
        }

        void intt_in_place_parallel_impl(mcl::Fr *values, std::size_t size, std::size_t transform_workers)
        {
            radix2_ntt_in_place_parallel_impl(values, size, transform_workers, true);
        }
    }

    void intt_in_place(mcl::Fr *values, std::size_t size, std::size_t max_worker_count)
    {
        if (values == nullptr || size == 0 || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error("INTT requires a non-null power-of-two field vector");
        }

        if (size == 1)
        {
            return;
        }

        if (should_use_cuda_intt(size))
        {
            if (!try_cuda_intt_in_place(values, size))
            {
                throw std::runtime_error("CUDA production INTT route was selected but not admitted: size=" + std::to_string(size) + " batch_size=1");
            }

            return;
        }

        const auto transform_workers = std::min<std::size_t>(32, bounded_worker_count(max_worker_count));

        intt_in_place_parallel_impl(values, size, transform_workers);
    }

    void ntt_in_place(mcl::Fr *values, std::size_t size, std::size_t max_worker_count)
    {
        if (values == nullptr || size == 0 || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error("NTT requires a non-null power-of-two field vector");
        }

        if (size == 1)
        {
            return;
        }

        if (should_use_cuda_ntt(size))
        {
            if (!try_cuda_ntt_in_place(values, size))
            {
                throw std::runtime_error("CUDA production NTT route was selected but not admitted: size=" + std::to_string(size) + " batch_size=1");
            }

            return;
        }

        const auto transform_workers = std::min<std::size_t>(32, bounded_worker_count(max_worker_count));

        ntt_in_place_parallel_impl(values, size, transform_workers);
    }
    void ntt_pair_in_place(mcl::Fr *first, mcl::Fr *second, std::size_t size, std::size_t max_worker_count)
    {
        if (first == nullptr || second == nullptr || first == second || size == 0 || (size & (size - 1U)) != 0U)
        {
            throw std::runtime_error("paired NTT requires two distinct non-null power-of-two vectors");
        }

        if (should_use_cuda_ntt_pair(size))
        {
            if (!try_cuda_ntt_pair_in_place(first, second, size))
            {
                throw std::runtime_error("CUDA production paired NTT route was selected but not admitted: size=" + std::to_string(size) + " batch_size=2");
            }

            return;
        }

        constexpr std::size_t kParallelPairThreshold = std::size_t{1} << 15;

        constexpr std::size_t kStageParallelPairThreshold = std::size_t{1} << 18;

        const auto available_workers = bounded_worker_count(max_worker_count);

        if (size < kParallelPairThreshold || available_workers < 2)
        {
            const auto plan = ntt_plan(size);

            plan->ntt(first);
            plan->ntt(second);

            return;
        }

        if (size >= kStageParallelPairThreshold && available_workers >= 4)
        {
            const auto per_transform_workers = std::max<std::size_t>(2, std::min<std::size_t>(16, available_workers / 2));

            auto first_task = std::async(std::launch::async, [=] { ntt_in_place_parallel_impl(first, size, per_transform_workers); });

            ntt_in_place_parallel_impl(second, size, per_transform_workers);

            first_task.get();

            return;
        }

        const auto plan = ntt_plan(size);

        auto first_task = std::async(std::launch::async, [=] { plan->ntt(first); });

        plan->ntt(second);
        first_task.get();
    }

    bool try_cuda_intt_in_place(mcl::Fr *values, std::size_t size)
    {
#if GATZK_ENABLE_CUDA_BACKEND
        if (!cuda_field_hotspots_available())
        {
            return false;
        }

        return cuda_intt_in_place_exact(values, size);
#else
        (void)values;
        (void)size;
        return false;
#endif
    }

    bool try_cuda_ntt_in_place(mcl::Fr *values, std::size_t size)
    {
#if GATZK_ENABLE_CUDA_BACKEND
        if (!cuda_field_hotspots_available())
        {
            return false;
        }

        return cuda_ntt_in_place_exact(values, size);
#else
        (void)values;
        (void)size;
        return false;
#endif
    }

    bool try_cuda_taylor_convolution(mcl::Fr *values, const mcl::Fr *kernel_ntt, std::size_t size)
    {
        if (!should_use_cuda_ntt(size))
            return false;
        if (values == nullptr || kernel_ntt == nullptr || values == kernel_ntt || size < 2 || (size & (size - 1U)) != 0U)
            throw std::runtime_error("invalid CUDA Taylor convolution input");
#if GATZK_ENABLE_CUDA_BACKEND
        if (!cuda_taylor_convolution_in_place_exact(values, kernel_ntt, size))
            throw std::runtime_error("selected CUDA Taylor convolution was not admitted: size=" + std::to_string(size));
        return true;
#else
        throw std::runtime_error("CUDA Taylor convolution selected in a CPU build");
#endif
    }

    bool try_cuda_ntt_pair_in_place(mcl::Fr *first, mcl::Fr *second, std::size_t size)
    {
#if GATZK_ENABLE_CUDA_BACKEND
        if (!cuda_field_hotspots_available())
        {
            return false;
        }

        return cuda_ntt_pair_in_place_exact(first, second, size);
#else
        (void)first;
        (void)second;
        (void)size;
        return false;
#endif
    }

    std::size_t cuda_kernel_invocation_count() { return cuda_kernel_counter().load(std::memory_order_relaxed); }

    void record_cuda_kernel_invocations(std::size_t count) { cuda_kernel_counter().fetch_add(count, std::memory_order_relaxed); }

    CudaExecutionStats cuda_execution_stats_snapshot()
    {
        const auto &counters = cuda_execution_counters();

        return CudaExecutionStats{
            .h2d_bytes = counters.h2d_bytes.load(std::memory_order_relaxed),
            .d2h_bytes = counters.d2h_bytes.load(std::memory_order_relaxed),
            .h2d_operations = counters.h2d_operations.load(std::memory_order_relaxed),
            .d2h_operations = counters.d2h_operations.load(std::memory_order_relaxed),
            .stream_synchronizations = counters.stream_synchronizations.load(std::memory_order_relaxed),
            .ntt_calls = counters.ntt_calls.load(std::memory_order_relaxed),
            .intt_calls = counters.intt_calls.load(std::memory_order_relaxed),
            .quotient_calls = counters.quotient_calls.load(std::memory_order_relaxed),
            .t_edge_calls = counters.t_edge_calls.load(std::memory_order_relaxed),
            .t_fh_calls = counters.t_fh_calls.load(std::memory_order_relaxed),
            .opening_fold_calls = counters.opening_fold_calls.load(std::memory_order_relaxed),
            .msm_single_calls = counters.msm_single_calls.load(std::memory_order_relaxed),
            .msm_batch_calls = counters.msm_batch_calls.load(std::memory_order_relaxed),
            .resident_hits = counters.resident_hits.load(std::memory_order_relaxed),
            .resident_misses = counters.resident_misses.load(std::memory_order_relaxed),
        };
    }

    void record_cuda_h2d(std::size_t bytes)
    {
        auto &counters = cuda_execution_counters();

        counters.h2d_bytes.fetch_add(bytes, std::memory_order_relaxed);

        counters.h2d_operations.fetch_add(1, std::memory_order_relaxed);
    }

    void record_cuda_d2h(std::size_t bytes)
    {
        auto &counters = cuda_execution_counters();

        counters.d2h_bytes.fetch_add(bytes, std::memory_order_relaxed);

        counters.d2h_operations.fetch_add(1, std::memory_order_relaxed);
    }

    void record_cuda_stream_sync() { cuda_execution_counters().stream_synchronizations.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_ntt_call() { cuda_execution_counters().ntt_calls.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_intt_call() { cuda_execution_counters().intt_calls.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_quotient_call() { cuda_execution_counters().quotient_calls.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_t_edge_call() { cuda_execution_counters().t_edge_calls.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_t_fh_call() { cuda_execution_counters().t_fh_calls.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_opening_fold_call() { cuda_execution_counters().opening_fold_calls.fetch_add(1, std::memory_order_relaxed); }
    void record_cuda_msm_single_call() { cuda_execution_counters().msm_single_calls.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_msm_batch_call() { cuda_execution_counters().msm_batch_calls.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_resident_hit() { cuda_execution_counters().resident_hits.fetch_add(1, std::memory_order_relaxed); }

    void record_cuda_resident_miss() { cuda_execution_counters().resident_misses.fetch_add(1, std::memory_order_relaxed); }
#if !GATZK_ENABLE_CUDA_BACKEND

    bool cuda_coset_evaluations_to_host_exact(const mcl::Fr *, std::size_t, std::size_t, const mcl::Fr *, std::size_t, const FieldElement &, mcl::Fr *,
                                              std::size_t, bool)
    {
        return false;
    }

    bool cuda_coset_coefficients_to_host_exact(const mcl::Fr *, std::size_t, const FieldElement &, mcl::Fr *, std::size_t, bool) { return false; }

    bool cuda_coset_host_to_coefficients_exact(const mcl::Fr *, std::size_t, const FieldElement &, mcl::Fr *) { return false; }

    bool cuda_coset_host_to_resident_coefficients_exact(const mcl::Fr *, std::size_t, const FieldElement &, mcl::Fr *,
                                                        std::shared_ptr<const CudaResidentCoefficientBuffer> *)
    {
        return false;
    }

    void release_cuda_ntt_workspace() {}

    void release_cuda_ntt_execution_workspace() {}

    void release_cuda_scaled_accumulate_workspace() {}

    void release_cuda_scan_workspace() {}

    std::unique_ptr<CudaScaledAccumulateSession> create_cuda_scaled_accumulate_session(mcl::Fr *, std::size_t, bool) { return nullptr; }

    std::shared_ptr<CudaScaledAccumulatePairSession> create_cuda_scaled_accumulate_pair_session(mcl::Fr *, mcl::Fr *, std::size_t, bool, bool)
    {
        return nullptr;
    }

    std::unique_ptr<CudaWeightedDotSession> create_cuda_weighted_dot_session(const std::vector<const mcl::Fr *> &, std::size_t, std::size_t) { return nullptr; }

    void release_cuda_weighted_dot_cache() {}

#endif

    bool cuda_backend_build_enabled()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        return true;
#else
        return false;
#endif
    }

    bool cuda_backend_available()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_backend_runtime_available();
#else
        return false;
#endif
    }

    bool cuda_backend_unified_addressing_available()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        static const bool available = []()
        {
            if (!cuda_backend_runtime_available())
            {
                return false;
            }
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess)
            {
                return false;
            }
            int unified = 0;
            if (cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, device) != cudaSuccess)
            {
                return false;
            }
            int pageable_access = 0;
            if (cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, device) != cudaSuccess)
            {
                return false;
            }
            int host_page_tables = 0;
            if (cudaDeviceGetAttribute(&host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device) != cudaSuccess)
            {
                return false;
            }
            return unified != 0 && pageable_access != 0 && host_page_tables != 0;
        }();
        return available;
#else
        return false;
#endif
    }

    std::string cuda_device_summary()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0)
        {
            return "cuda_build=true; cuda_runtime=false; cuda_device=unavailable";
        }
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess)
        {
            device = 0;
        }
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device) != cudaSuccess)
        {
            return "cuda_build=true; cuda_runtime=true; cuda_device=unknown; cuda_device_count=" + std::to_string(device_count);
        }
        int pageable_access = 0;
        int host_page_tables = 0;
        (void)cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, device);
        (void)cudaDeviceGetAttribute(&host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);
        return "cuda_build=true; cuda_runtime=true; cuda_device=" + std::string(props.name) + "; cuda_device_count=" + std::to_string(device_count) +
               "; cuda_integrated=" + std::string(props.integrated ? "true" : "false") +
               "; cuda_unified_addressing=" + std::string(props.unifiedAddressing ? "true" : "false") +
               "; cuda_pageable_memory_access=" + std::string(pageable_access != 0 ? "true" : "false") +
               "; cuda_pageable_host_page_tables=" + std::string(host_page_tables != 0 ? "true" : "false");
#else
        return "cuda_build=false; cuda_runtime=false; cuda_device=unavailable";
#endif
    }

    bool cuda_field_hotspots_build_enabled()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        return true;
#else
        return false;
#endif
    }

    bool cuda_field_hotspots_available()
    {
#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_backend_runtime_available();
#else
        return false;
#endif
    }

    bool should_use_cuda_barycentric_weights(std::size_t domain_size, bool points_precomputed)
    {
        if (!cuda_field_route_enabled() || points_precomputed || domain_size == 0)
        {
            return false;
        }

        const auto min_domain_size = cuda_backend_unified_addressing_available() ? (1ULL << 16) : (1ULL << 18);
        return domain_size >= min_domain_size;
    }

    bool should_use_cuda_weighted_evaluation(std::size_t domain_size, std::size_t vector_count, std::size_t point_count)
    {
        if (!cuda_field_route_enabled() || domain_size == 0 || vector_count == 0 || point_count == 0)
        {
            return false;
        }

        const bool unified = cuda_backend_unified_addressing_available();
        const auto min_domain_size = unified ? (1ULL << 15) : (1ULL << 16);
        const auto min_work_items = unified ? 1ULL : 4ULL;
        const auto work_items = multiply_u128(vector_count, point_count);
        const auto total_terms = multiply_u128(domain_size, vector_count) * static_cast<unsigned __int128>(point_count);
        const auto min_total_terms = unified ? (static_cast<unsigned __int128>(1) << 21) : (static_cast<unsigned __int128>(1) << 24);

        return domain_size >= min_domain_size && work_items >= static_cast<unsigned __int128>(min_work_items) && total_terms >= min_total_terms;
    }

    bool should_use_cuda_lookup_histogram(std::size_t index_count, std::size_t bucket_count)
    {
        if (!cuda_trace_route_enabled() || index_count == 0 || bucket_count == 0)
        {
            return false;
        }

        const bool direct_pageable_host_access = cuda_backend_unified_addressing_available();

        const auto min_index_count = direct_pageable_host_access ? (1ULL << 18) : (1ULL << 22);

        const auto min_bucket_count = direct_pageable_host_access ? 256ULL : 4096ULL;

        const auto min_indices_per_bucket = direct_pageable_host_access ? 64ULL : 1024ULL;

        return index_count >= min_index_count && bucket_count >= min_bucket_count && index_count / bucket_count >= min_indices_per_bucket;
    }

    bool should_use_cuda_intt(std::size_t size)
    {
        const std::size_t kCudaInttMinimumSize = cuda_backend_unified_addressing_available() ? (std::size_t{1} << 18) : (std::size_t{1} << 23);

        constexpr std::size_t kCudaInttMaximumSize = std::size_t{1} << 26;

        const auto &options = util::route2_options();
        if (options.cuda_force_ntt_for_mechanism)
        {
            if (!options.cuda_field_hotspots || size < 2 || size > kCudaInttMaximumSize || !cuda_field_hotspots_available())
                throw std::runtime_error("mechanism forced CUDA INTT is unavailable for the requested supported size");
            return true;
        }
        if (!options.cuda_field_hotspots || size < kCudaInttMinimumSize || size > kCudaInttMaximumSize)
        {
            return false;
        }

#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_field_hotspots_available();
#else
        return false;
#endif
    }

    bool should_use_cuda_ntt(std::size_t size)
    {
        const std::size_t kCudaNttMinimumSize = cuda_backend_unified_addressing_available() ? (std::size_t{1} << 18) : (std::size_t{1} << 23);

        constexpr std::size_t kCudaNttMaximumSize = std::size_t{1} << 26;

        const auto &options = util::route2_options();
        if (options.cuda_force_ntt_for_mechanism)
        {
            if (!options.cuda_field_hotspots || size < 2 || size > kCudaNttMaximumSize || !cuda_field_hotspots_available())
                throw std::runtime_error("mechanism forced CUDA NTT is unavailable for the requested supported size");
            return true;
        }
        if (!options.cuda_field_hotspots || size < kCudaNttMinimumSize || size > kCudaNttMaximumSize)
        {
            return false;
        }

#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_field_hotspots_available();
#else
        return false;
#endif
    }

    bool should_use_cuda_ntt_pair(std::size_t size)
    {
        const std::size_t kCudaPairMinimumSize = cuda_backend_unified_addressing_available() ? (std::size_t{1} << 18) : (std::size_t{1} << 23);

        constexpr std::size_t kCudaPairMaximumSize = std::size_t{1} << 26;

        if (!util::route2_options().cuda_field_hotspots || size < kCudaPairMinimumSize || size > kCudaPairMaximumSize)
        {
            return false;
        }

#if GATZK_ENABLE_CUDA_BACKEND
        return cuda_field_hotspots_available();
#else
        return false;
#endif
    }

    bool should_use_cuda_coset_transform(std::size_t target_size)
    {
        const std::size_t kMinimumTargetSize = cuda_backend_unified_addressing_available() ? (std::size_t{1} << 20) : (std::size_t{1} << 24);

        constexpr std::size_t kMaximumTargetSize = std::size_t{1} << 27;

        if (!cuda_field_route_enabled() || target_size < kMinimumTargetSize || target_size > kMaximumTargetSize || target_size < 2 ||
            (target_size & (target_size - 1U)) != 0U)
        {
            return false;
        }

        return true;
    }

    bool try_cuda_batch_inverse_native(const mcl::Fr *input, mcl::Fr *output, std::size_t element_count)
    {
        if (!cuda_field_route_enabled() || !cuda_backend_unified_addressing_available() || element_count < (std::size_t{1} << 18))
        {
            return false;
        }

        if (input == nullptr || output == nullptr)
        {
            throw std::runtime_error("CUDA batch inverse received a null native buffer");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        if (!batch_inverse_cuda(input, output, element_count))
        {
            throw std::runtime_error("selected CUDA batch inverse was unavailable");
        }

        return true;
#else
        return false;
#endif
    }

    bool try_cuda_opening_inverse_denominators_native(mcl::Fr *output, std::size_t domain_size, const FieldElement &omega, const FieldElement &point)
    {
        if (!cuda_field_route_enabled() || !cuda_backend_unified_addressing_available() || domain_size < (std::size_t{1} << 18))
        {
            return false;
        }

        if (output == nullptr)
        {
            throw std::runtime_error("CUDA opening denominator inverse received a null output");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        if (!opening_inverse_denominators_cuda(output, domain_size, omega.native(), point.native()))
        {
            throw std::runtime_error("selected CUDA opening denominator inverse was unavailable");
        }

        return true;
#else
        return false;
#endif
    }

    bool try_cuda_opening_quotient_assembly_native(const mcl::Fr *folded_values, mcl::Fr *quotient_values, const mcl::Fr &scaled_claim_sum,
                                                   std::size_t element_count)
    {
        if (!cuda_field_route_enabled() || !cuda_backend_unified_addressing_available() || element_count < (std::size_t{1} << 18))
            return false;
        if (folded_values == nullptr || quotient_values == nullptr)
            throw std::runtime_error("CUDA opening quotient assembly received a null buffer");
#if GATZK_ENABLE_CUDA_BACKEND
        if (!opening_quotient_assembly_cuda(folded_values, quotient_values, scaled_claim_sum, element_count))
            throw std::runtime_error("selected CUDA opening quotient assembly was unavailable");
        return true;
#else
        return false;
#endif
    }

    bool try_cuda_opening_inverse_and_quotient_native(const mcl::Fr *folded_values, mcl::Fr *quotient_values, const mcl::Fr &scaled_claim_sum,
                                                      std::size_t domain_size, const FieldElement &omega, const FieldElement &point)
    {
        if (!cuda_field_route_enabled() || !cuda_backend_unified_addressing_available() || domain_size < (std::size_t{1} << 18))
            return false;
        if (folded_values == nullptr || quotient_values == nullptr)
            throw std::runtime_error("CUDA fused opening received a null buffer");
#if GATZK_ENABLE_CUDA_BACKEND
        if (!opening_inverse_and_quotient_cuda(folded_values, quotient_values, scaled_claim_sum, domain_size, omega.native(), point.native()))
            throw std::runtime_error("selected CUDA fused opening route was unavailable");
        return true;
#else
        return false;
#endif
    }

    bool try_cuda_opening_inverse_and_quotient_resident_native(const mcl::Fr *folded_values, const mcl::Fr &scaled_claim_sum, std::size_t domain_size,
                                                               const FieldElement &omega, const FieldElement &point,
                                                               std::shared_ptr<const CudaResidentCoefficientBuffer> *resident_output)
    {
        if (!cuda_field_route_enabled() || !cuda_backend_unified_addressing_available() || domain_size < (std::size_t{1} << 18))
        {
            return false;
        }

        if (folded_values == nullptr || resident_output == nullptr)
        {
            throw std::runtime_error("CUDA resident fused opening received a null buffer");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        if (!opening_inverse_and_quotient_resident_cuda(folded_values, scaled_claim_sum, domain_size, omega.native(), point.native(), resident_output))
        {
            throw std::runtime_error("selected CUDA resident fused opening route was unavailable");
        }

        return true;
#else
        return false;
#endif
    }

    bool try_cuda_coefficient_opening_quotient_resident_native(const mcl::Fr *coefficients, std::size_t coefficient_count, const mcl::Fr &point,
                                                               const mcl::Fr &claimed_value,
                                                               std::shared_ptr<const CudaResidentCoefficientBuffer> *resident_output)
    {
        if (!cuda_field_route_enabled() || !cuda_backend_unified_addressing_available() || coefficient_count <= 1 ||
            coefficient_count - 1 < (std::size_t{1} << 18))
        {
            return false;
        }

        if (coefficients == nullptr || resident_output == nullptr)
        {
            throw std::runtime_error("CUDA resident coefficient opening received a null buffer");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        if (!coefficient_opening_quotient_resident_cuda(coefficients, coefficient_count, point, claimed_value, resident_output))
        {
            throw std::runtime_error("selected CUDA resident coefficient opening route was unavailable");
        }

        return true;
#else
        return false;
#endif
    }

    bool try_cuda_logup_inverse_and_accumulator_native(const mcl::Fr *table, const mcl::Fr *query, const mcl::Fr *q_table, const mcl::Fr *q_query,
                                                       const mcl::Fr &beta, const mcl::Fr *multiplicity, mcl::Fr *table_inverse, mcl::Fr *query_inverse,
                                                       mcl::Fr *accumulator, std::size_t active_count, std::size_t element_count,
                                                       CudaLogupExecutionProfile *profile)
    {
        if (!(cuda_field_route_enabled() || cuda_trace_route_enabled()) || !cuda_backend_unified_addressing_available() ||
            element_count < (std::size_t{1} << 18) || active_count < (std::size_t{1} << 18))
        {
            return false;
        }

        if (table == nullptr || query == nullptr || q_table == nullptr || q_query == nullptr || multiplicity == nullptr || table_inverse == nullptr ||
            query_inverse == nullptr || accumulator == nullptr)
        {
            throw std::runtime_error("CUDA fused LogUp received a null buffer");
        }

        if (active_count >= element_count)
        {
            throw std::runtime_error("CUDA fused LogUp active count exceeds transition domain");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        if (!logup_inverse_and_accumulator_cuda(table, query, q_table, q_query, beta, multiplicity, table_inverse, query_inverse, accumulator, active_count,
                                                element_count, profile))
        {
            throw std::runtime_error("selected CUDA fused LogUp route was unavailable");
        }

        return true;
#else
        (void)profile;
        return false;
#endif
    }

    bool try_cuda_logup_inverse_columns_native(const mcl::Fr *table, const mcl::Fr *query, const mcl::Fr *q_table, const mcl::Fr *q_query, const mcl::Fr &beta,
                                               mcl::Fr *table_inverse, mcl::Fr *query_inverse, std::size_t element_count)
    {
        if (!(cuda_field_route_enabled() || cuda_trace_route_enabled()) || !cuda_backend_unified_addressing_available() ||
            element_count < (std::size_t{1} << 18))
        {
            return false;
        }

        if (table == nullptr || query == nullptr || q_table == nullptr || q_query == nullptr || table_inverse == nullptr || query_inverse == nullptr)
        {
            throw std::runtime_error("CUDA LogUp inverse received a null buffer");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        if (!logup_inverse_columns_cuda(table, query, q_table, q_query, beta, table_inverse, query_inverse, element_count))
        {
            throw std::runtime_error("selected CUDA LogUp inverse was unavailable");
        }

        return true;
#else
        return false;
#endif
    }

    bool try_cuda_logup_accumulator_native(const mcl::Fr *table_inverse, const mcl::Fr *query_inverse, const mcl::Fr *multiplicity, const mcl::Fr *q_table,
                                           const mcl::Fr *q_query, mcl::Fr *accumulator, std::size_t active_count, std::size_t element_count)
    {
        if (!(cuda_field_route_enabled() || cuda_trace_route_enabled()) || !cuda_backend_unified_addressing_available() ||
            active_count < (std::size_t{1} << 18))
        {
            return false;
        }

        if (table_inverse == nullptr || query_inverse == nullptr || multiplicity == nullptr || q_table == nullptr || q_query == nullptr ||
            accumulator == nullptr)
        {
            throw std::runtime_error("CUDA LogUp accumulator received a null buffer");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        if (!logup_accumulator_cuda(table_inverse, query_inverse, multiplicity, q_table, q_query, accumulator, active_count, element_count))
        {
            throw std::runtime_error("selected CUDA LogUp accumulator was unavailable");
        }

        return true;
#else
        return false;
#endif
    }

    bool try_cuda_scaled_prefix_accumulator_native(const mcl::Fr *values, const mcl::Fr &scale, mcl::Fr *accumulator, std::size_t active_count,
                                                   std::size_t element_count)
    {
        if (!(cuda_field_route_enabled() || cuda_trace_route_enabled()) || !cuda_backend_unified_addressing_available() ||
            active_count < (std::size_t{1} << 18))
        {
            return false;
        }

        if (values == nullptr || accumulator == nullptr)
        {
            throw std::runtime_error("CUDA prefix accumulator received a null buffer");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        if (!scaled_prefix_accumulator_cuda(values, scale, accumulator, active_count, element_count))
        {
            throw std::runtime_error("selected CUDA prefix accumulator was unavailable");
        }

        return true;
#else
        return false;
#endif
    }

    bool try_cuda_scaled_accumulate_native(mcl::Fr *target, const mcl::Fr *source, const mcl::Fr &scale, std::size_t element_count)
    {
        const auto min_cuda_elements = cuda_backend_unified_addressing_available() ? (1ULL << 18) : (1ULL << 22);

        if (element_count < min_cuda_elements || !cuda_field_route_enabled())
        {
            return false;
        }
        if (target == nullptr || source == nullptr)
        {
            throw std::runtime_error("CUDA scaled accumulate received a null native buffer");
        }
#if GATZK_ENABLE_CUDA_BACKEND
        static_assert(sizeof(mcl::Fr) == 4 * sizeof(uint64_t), "CUDA scaled accumulate requires four-limb mcl::Fr storage");
        scaled_accumulate_cuda(reinterpret_cast<uint64_t *>(target), reinterpret_cast<const uint64_t *>(source), reinterpret_cast<const uint64_t *>(&scale),
                               element_count);
        return true;
#else
        (void)scale;
        return false;
#endif
    }

    bool try_cuda_scaled_accumulate_pair_native(mcl::Fr *first_target, mcl::Fr *second_target, const mcl::Fr *source, const mcl::Fr &first_scale,
                                                const mcl::Fr &second_scale, std::size_t element_count)
    {
        const auto min_cuda_elements = cuda_backend_unified_addressing_available() ? (1ULL << 18) : (1ULL << 22);

        if (element_count < min_cuda_elements || !cuda_field_route_enabled())
        {
            return false;
        }

        if (first_target == nullptr || second_target == nullptr || source == nullptr)
        {
            throw std::runtime_error("CUDA paired scaled accumulate received a null native buffer");
        }

        if (first_target == second_target || first_target == source || second_target == source)
        {
            throw std::runtime_error("CUDA paired scaled accumulate requires distinct source and targets");
        }

#if GATZK_ENABLE_CUDA_BACKEND
        static_assert(sizeof(mcl::Fr) == 4 * sizeof(uint64_t), "CUDA paired scaled accumulate requires four-limb mcl::Fr storage");

        scaled_accumulate_pair_cuda(reinterpret_cast<uint64_t *>(first_target), reinterpret_cast<uint64_t *>(second_target),
                                    reinterpret_cast<const uint64_t *>(source), reinterpret_cast<const uint64_t *>(&first_scale),
                                    reinterpret_cast<const uint64_t *>(&second_scale), element_count);

        return true;
#else
        (void)first_scale;
        (void)second_scale;
        return false;
#endif
    }

#if !GATZK_ENABLE_CUDA_BACKEND
    bool barycentric_weights_gpu(const uint64_t[4], const uint64_t[4], const uint64_t[4], const uint64_t[4], uint64_t *, std::size_t) { return false; }
    bool weighted_poly_eval_gpu(const uint64_t *, const uint64_t *, uint64_t *, std::size_t, std::size_t) { return false; }
    bool weighted_poly_eval_gpu_columnar(const uint64_t *, const uint64_t *const *, uint64_t *, std::size_t, std::size_t, const std::size_t *, std::size_t)
    {
        return false;
    }
    bool weighted_poly_eval_gpu_multi(const uint64_t *, const uint64_t *const *, uint64_t *, std::size_t, std::size_t, const std::size_t *, std::size_t)
    {
        return false;
    }
    bool weighted_poly_eval_gpu_multi_rotations(const uint64_t *, const uint64_t *const *, uint64_t *, std::size_t, std::size_t, const std::size_t *,
                                                std::size_t, const std::size_t *)
    {
        return false;
    }
    bool weighted_poly_eval_gpu_interleaved_multi_rotations(const uint64_t *, const uint64_t *, uint64_t *, std::size_t, std::size_t, const std::size_t *,
                                                            std::size_t)
    {
        return false;
    }
    bool cuda_interleaved_multi_rotation_admissible(std::size_t, std::size_t, std::size_t, std::size_t) { return false; }
#endif

    FieldElement dot_product(const std::vector<FieldElement> &lhs, const std::vector<FieldElement> &rhs)
    {
        if (lhs.size() != rhs.size())
            throw std::runtime_error("dot product size mismatch");
        if (lhs.empty())
            return FieldElement::zero();
        if (should_use_cuda_weighted_evaluation(lhs.size(), 1))
        {
            static_assert(sizeof(mcl::Fr) == 4 * sizeof(std::uint64_t));
            const std::uint64_t *columns[] = {reinterpret_cast<const std::uint64_t *>(FieldElement::native_array_data(lhs.data()))};
            mcl::Fr sum;
            if (!weighted_poly_eval_gpu_multi(reinterpret_cast<const std::uint64_t *>(FieldElement::native_array_data(rhs.data())), columns,
                                              reinterpret_cast<std::uint64_t *>(&sum), lhs.size(), 1))
                throw std::runtime_error("selected CUDA dot_product was not admitted");
            return FieldElement::from_native(sum);
        }

        const auto &route2 = util::route2_options();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        if (!route2.parallel_fft || lhs.size() < 1024 || cpu_count == 1)
        {
            return dot_product_range(lhs, rhs, 0, lhs.size());
        }

        const auto task_count = std::min<std::size_t>(cpu_count, lhs.size() / 512);
        if (task_count <= 1)
        {
            return dot_product_range(lhs, rhs, 0, lhs.size());
        }

        std::vector<std::future<FieldElement>> futures;
        futures.reserve(task_count);
        const auto chunk_size = (lhs.size() + task_count - 1) / task_count;
        for (std::size_t task = 0; task < task_count; ++task)
        {
            const auto begin = task * chunk_size;
            const auto end = std::min(lhs.size(), begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(std::launch::async, [&lhs, &rhs, begin, end]() { return dot_product_range(lhs, rhs, begin, end); }));
        }

        FieldElement sum = FieldElement::zero();
        for (auto &future : futures)
        {
            sum += future.get();
        }
        return sum;
    }

    FieldElement dot_product_native_weights(const std::vector<FieldElement> &lhs, const std::vector<mcl::Fr> &rhs)
    {
        if (lhs.size() != rhs.size())
            throw std::runtime_error("dot product size mismatch");
        if (lhs.empty())
            return FieldElement::zero();
        if (should_use_cuda_weighted_evaluation(lhs.size(), 1))
        {
            static_assert(sizeof(mcl::Fr) == 4 * sizeof(std::uint64_t));
            const std::uint64_t *columns[] = {reinterpret_cast<const std::uint64_t *>(FieldElement::native_array_data(lhs.data()))};
            mcl::Fr sum;
            if (!weighted_poly_eval_gpu_multi(reinterpret_cast<const std::uint64_t *>(rhs.data()), columns, reinterpret_cast<std::uint64_t *>(&sum), lhs.size(),
                                              1))
                throw std::runtime_error("selected CUDA dot_product_native_weights was not admitted");
            return FieldElement::from_native(sum);
        }

        const auto &route2 = util::route2_options();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        if (!route2.parallel_fft || lhs.size() < 1024 || cpu_count == 1)
        {
            return dot_product_native_range(lhs, rhs, 0, lhs.size());
        }

        const auto task_count = std::min<std::size_t>(cpu_count, lhs.size() / 512);
        if (task_count <= 1)
        {
            return dot_product_native_range(lhs, rhs, 0, lhs.size());
        }

        std::vector<std::future<FieldElement>> futures;
        futures.reserve(task_count);
        const auto chunk_size = (lhs.size() + task_count - 1) / task_count;
        for (std::size_t task = 0; task < task_count; ++task)
        {
            const auto begin = task * chunk_size;
            const auto end = std::min(lhs.size(), begin + chunk_size);
            if (begin >= end)
            {
                break;
            }
            futures.push_back(std::async(std::launch::async, [&lhs, &rhs, begin, end]() { return dot_product_native_range(lhs, rhs, begin, end); }));
        }

        FieldElement sum = FieldElement::zero();
        for (auto &future : futures)
        {
            sum += future.get();
        }
        return sum;
    }
}
