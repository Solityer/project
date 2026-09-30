
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/algebra/vector_ops.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

// lookup 选择器与 LogUp 累加器
// 特征、激活、range 与 exp 表查询约束的底层公共实现

namespace gatzk::protocol
{
    namespace
    {

        using algebra::FieldElement;

        constexpr std::size_t kMaxParallelWorkers = 32;

        void batch_inverse_native(const std::vector<mcl::Fr> &values, std::vector<mcl::Fr> &out, const std::string &label)
        {
            out.assign(values.size(), mcl::Fr{});
            if (values.empty())
            {
                return;
            }

            if (algebra::try_cuda_batch_inverse_native(values.data(), out.data(), values.size()))
            {
                return;
            }

            // 把同一批分母统一做前缀积和一次逆元，减少 LogUp 热路径里的逐项求逆。
            // This buffer is shared by the disjoint chunk workers below, so
            // it must be an ordinary function-local allocation. A
            // thread_local buffer would resolve to an empty, different vector
            // inside each async worker.
            std::vector<mcl::Fr> prefix(values.size());
            const auto hardware_workers = std::max<std::size_t>(1, std::min<std::size_t>(kMaxParallelWorkers, std::thread::hardware_concurrency()));
            constexpr std::size_t kParallelThreshold = 1ULL << 18;
            if (values.size() >= kParallelThreshold && hardware_workers > 1)
            {
                const auto chunk_count = std::min<std::size_t>(hardware_workers, std::max<std::size_t>(2, values.size() / (1ULL << 17)));
                const auto chunk_size = (values.size() + chunk_count - 1) / chunk_count;
                std::vector<mcl::Fr> chunk_products(chunk_count);
                const auto parallel_chunks = [&](const auto &fn)
                {
                    std::vector<std::future<void>> futures;
                    futures.reserve(chunk_count);
                    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk)
                    {
                        const auto begin = chunk * chunk_size;
                        const auto end = std::min(values.size(), begin + chunk_size);
                        if (begin >= end)
                        {
                            break;
                        }
                        futures.push_back(std::async(std::launch::async, [&, chunk, begin, end] { fn(chunk, begin, end); }));
                    }
                    for (auto &future : futures)
                    {
                        future.get();
                    }
                };
                parallel_chunks(
                    [&](std::size_t chunk, std::size_t begin, std::size_t end)
                    {
                        mcl::Fr product = 1;
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            if (values[i].isZero())
                            {
                                throw std::runtime_error("cannot invert zero in " + label);
                            }
                            prefix[i] = product;
                            mcl::Fr::mul(product, product, values[i]);
                        }
                        chunk_products[chunk] = product;
                    });

                // Only O(worker_count) values remain, so one serial batch
                // inversion is cheaper than another task launch and keeps the
                // same single-inversion algebra as the parallel path.
                std::vector<mcl::Fr> chunk_prefix(chunk_count);
                mcl::Fr chunk_product = 1;
                for (std::size_t chunk = 0; chunk < chunk_count; ++chunk)
                {
                    chunk_prefix[chunk] = chunk_product;
                    mcl::Fr::mul(chunk_product, chunk_product, chunk_products[chunk]);
                }
                mcl::Fr chunk_suffix_inverse;
                mcl::Fr::inv(chunk_suffix_inverse, chunk_product);
                for (std::size_t chunk = chunk_count; chunk-- > 0;)
                {
                    mcl::Fr inverse;
                    mcl::Fr::mul(inverse, chunk_suffix_inverse, chunk_prefix[chunk]);
                    mcl::Fr::mul(chunk_suffix_inverse, chunk_suffix_inverse, chunk_products[chunk]);
                    chunk_products[chunk] = inverse;
                }

                parallel_chunks(
                    [&](std::size_t chunk, std::size_t begin, std::size_t end)
                    {
                        auto suffix_inverse = chunk_products[chunk];
                        for (std::size_t i = end; i-- > begin;)
                        {
                            mcl::Fr::mul(out[i], suffix_inverse, prefix[i]);
                            mcl::Fr::mul(suffix_inverse, suffix_inverse, values[i]);
                        }
                    });
                return;
            }

            mcl::Fr product = 1;
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (values[i].isZero())
                {
                    throw std::runtime_error("cannot invert zero in " + label);
                }
                prefix[i] = product;
                mcl::Fr::mul(product, product, values[i]);
            }

            mcl::Fr suffix_inverse;
            mcl::Fr::inv(suffix_inverse, product);
            for (std::size_t i = values.size(); i-- > 0;)
            {
                mcl::Fr::mul(out[i], prefix[i], suffix_inverse);
                mcl::Fr::mul(suffix_inverse, suffix_inverse, values[i]);
            }
        }

    }

    std::vector<algebra::FieldElement> build_selector(std::size_t valid_length, std::size_t domain_size)
    {
        std::vector<algebra::FieldElement> out(domain_size, algebra::FieldElement::zero());
        // 选择器只标记有效前缀，剩余位置全部留给 padding 与零约束处理。
        for (std::size_t i = 0; i < valid_length && i < domain_size; ++i)
        {
            out[i] = algebra::FieldElement::one();
        }
        return out;
    }

    IndexedViewActivePrefixWitness build_indexed_view_active_prefix_witness(const std::vector<algebra::FieldElement> &tuples, std::size_t active_length,
                                                                            const algebra::FieldElement &active_multiplicity, const algebra::FieldElement &beta)
    {
        if (tuples.empty())
        {
            if (active_length != 0)
            {
                throw std::runtime_error("indexed-view active prefix exceeds empty tuple column");
            }
            return {};
        }
        if (active_length >= tuples.size())
        {
            throw std::runtime_error("indexed-view active prefix leaves no terminal row");
        }
        if (std::any_of(tuples.begin() + static_cast<std::ptrdiff_t>(active_length), tuples.end(), [](const FieldElement &value) { return !value.is_zero(); }))
        {
            throw std::runtime_error("indexed-view tuple column is nonzero outside active prefix");
        }

        IndexedViewActivePrefixWitness out;
        out.inverses = build_active_prefix_inverse_column(tuples, active_length, beta);
        out.accumulator.assign(tuples.size(), FieldElement::zero());

        if (algebra::try_cuda_scaled_prefix_accumulator_native(FieldElement::native_array_data(out.inverses.data()), active_multiplicity.native(),
                                                               FieldElement::mutable_native_array_data(out.accumulator.data()), active_length, tuples.size()))
        {
            return out;
        }

        mcl::Fr running;
        running.clear();

        for (std::size_t i = 0; i < active_length; ++i)
        {
            mcl::Fr term;

            mcl::Fr::mul(term, active_multiplicity.native(), out.inverses[i].native());

            mcl::Fr::add(running, running, term);

            out.accumulator[i + 1] = FieldElement::from_native(running);
        }
        if (active_length + 1 < tuples.size())
        {
            std::fill(out.accumulator.begin() + static_cast<std::ptrdiff_t>(active_length + 1), out.accumulator.end(), out.accumulator[active_length]);
        }
        return out;
    }

    namespace
    {
        LogUpInverseColumns build_logup_inverse_columns_impl(const std::vector<algebra::FieldElement> &table, const std::vector<algebra::FieldElement> &query,
                                                             const std::vector<algebra::FieldElement> &q_table,
                                                             const std::vector<algebra::FieldElement> &q_query, const algebra::FieldElement &beta)
        {
            const auto n = table.size();
            if (query.size() != n || q_table.size() != n || q_query.size() != n)
            {
                throw std::runtime_error("lookup inverse column input size mismatch");
            }
            LogUpInverseColumns out;
            out.table.assign(n, FieldElement::zero());

            out.query.assign(n, FieldElement::zero());

            if (n != 0 && algebra::try_cuda_logup_inverse_columns_native(
                              FieldElement::native_array_data(table.data()), FieldElement::native_array_data(query.data()),
                              FieldElement::native_array_data(q_table.data()), FieldElement::native_array_data(q_query.data()), beta.native(),
                              FieldElement::mutable_native_array_data(out.table.data()), FieldElement::mutable_native_array_data(out.query.data()), n))
            {
                return out;
            }

            thread_local std::vector<mcl::Fr> denominators;
            thread_local std::vector<mcl::Fr> inverses;
            denominators.clear();
            denominators.reserve(n * 2);
            for (std::size_t i = 0; i < n; ++i)
            {
                if (!q_table[i].is_zero())
                {
                    mcl::Fr denominator;
                    mcl::Fr::add(denominator, table[i].native(), beta.native());
                    denominators.push_back(denominator);
                }
                if (!q_query[i].is_zero())
                {
                    mcl::Fr denominator;
                    mcl::Fr::add(denominator, query[i].native(), beta.native());
                    denominators.push_back(denominator);
                }
            }
            batch_inverse_native(denominators, inverses, "lookup inverse columns");
            std::size_t inverse_index = 0;
            for (std::size_t i = 0; i < n; ++i)
            {
                if (!q_table[i].is_zero())
                {
                    out.table[i] = FieldElement::from_native(inverses[inverse_index++]);
                }
                if (!q_query[i].is_zero())
                {
                    out.query[i] = FieldElement::from_native(inverses[inverse_index++]);
                }
            }
            if (inverse_index != inverses.size())
            {
                throw std::runtime_error("lookup inverse column cursor mismatch");
            }
            return out;
        }
    }

    LogUpInverseColumns build_logup_inverse_columns(const std::vector<algebra::FieldElement> &table, const std::vector<algebra::FieldElement> &query,
                                                    const std::vector<algebra::FieldElement> &q_table, const std::vector<algebra::FieldElement> &q_query,
                                                    const algebra::FieldElement &beta)
    {
        return build_logup_inverse_columns_impl(table, query, q_table, q_query, beta);
    }

    LogUpRoundColumns build_logup_round_columns(const std::vector<algebra::FieldElement> &table, const std::vector<algebra::FieldElement> &query,
                                                const std::vector<algebra::FieldElement> &multiplicity, const std::vector<algebra::FieldElement> &q_table,
                                                const std::vector<algebra::FieldElement> &q_query, const algebra::FieldElement &beta, std::size_t active_count)
    {
        const auto n = table.size();

        if (query.size() != n || multiplicity.size() != n || q_table.size() != n || q_query.size() != n)
        {
            throw std::runtime_error("fused LogUp input size mismatch");
        }

        LogUpRoundColumns out;

        out.inverses.table.assign(n, FieldElement::zero());

        out.inverses.query.assign(n, FieldElement::zero());

        out.accumulator.assign(n, FieldElement::zero());

        const auto wall_start = std::chrono::steady_clock::now();

        if (n > 1)
        {
            const auto bounded_active = std::min(active_count, n - 1U);

            algebra::CudaLogupExecutionProfile cuda_profile;

            if (algebra::try_cuda_logup_inverse_and_accumulator_native(
                    FieldElement::native_array_data(table.data()), FieldElement::native_array_data(query.data()),
                    FieldElement::native_array_data(q_table.data()), FieldElement::native_array_data(q_query.data()), beta.native(),
                    FieldElement::native_array_data(multiplicity.data()), FieldElement::mutable_native_array_data(out.inverses.table.data()),
                    FieldElement::mutable_native_array_data(out.inverses.query.data()), FieldElement::mutable_native_array_data(out.accumulator.data()),
                    bounded_active, n, &cuda_profile))
            {
                const auto wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();

                const auto gpu_ms = cuda_profile.inverse_ms + cuda_profile.accumulator_ms;

                if (gpu_ms > 0.0 && gpu_ms <= wall_ms)
                {
                    out.inverse_ms = cuda_profile.inverse_ms;

                    out.accumulator_ms = cuda_profile.accumulator_ms + (wall_ms - gpu_ms);
                }
                else
                {
                    out.inverse_ms = wall_ms;

                    out.accumulator_ms = 0.0;
                }

                return out;
            }
        }

        auto phase_start = std::chrono::steady_clock::now();

        out.inverses = build_logup_inverse_columns_impl(table, query, q_table, q_query, beta);

        out.inverse_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phase_start).count();

        phase_start = std::chrono::steady_clock::now();

        out.accumulator = build_logup_accumulator_from_inverse_columns_with_active_count(out.inverses.table, out.inverses.query, multiplicity, q_table, q_query,
                                                                                         active_count);

        out.accumulator_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - phase_start).count();

        return out;
    }

    namespace
    {

        std::vector<algebra::FieldElement> build_logup_accumulator_from_inverse_columns_impl(const std::vector<algebra::FieldElement> &table_inverse,
                                                                                             const std::vector<algebra::FieldElement> &query_inverse,
                                                                                             const std::vector<algebra::FieldElement> &multiplicity,
                                                                                             const std::vector<algebra::FieldElement> &q_table,
                                                                                             const std::vector<algebra::FieldElement> &q_query,
                                                                                             std::size_t active_count)
        {
            const auto n = table_inverse.size();
            if (query_inverse.size() != n || multiplicity.size() != n || q_table.size() != n || q_query.size() != n)
            {
                throw std::runtime_error("lookup inverse accumulator input size mismatch");
            }
            std::vector<algebra::FieldElement> accumulator(n, FieldElement::zero());
            if (n <= 1)
            {
                return accumulator;
            }
            active_count = std::min(active_count, n - 1U);

            if (algebra::try_cuda_logup_accumulator_native(
                    FieldElement::native_array_data(table_inverse.data()), FieldElement::native_array_data(query_inverse.data()),
                    FieldElement::native_array_data(multiplicity.data()), FieldElement::native_array_data(q_table.data()),
                    FieldElement::native_array_data(q_query.data()), FieldElement::mutable_native_array_data(accumulator.data()), active_count, n))
            {
                return accumulator;
            }

            const auto one = FieldElement::one();

            const auto accumulate_range = [&](std::size_t begin, std::size_t end, const mcl::Fr &offset)
            {
                auto running = offset;
                for (std::size_t i = begin; i < end; ++i)
                {
                    if (!(q_table[i].is_zero() || q_table[i] == one) || !(q_query[i].is_zero() || q_query[i] == one))
                    {
                        throw std::runtime_error("lookup accumulator selector is not boolean");
                    }
                    if (!q_table[i].is_zero())
                    {
                        mcl::Fr table_term;
                        mcl::Fr::mul(table_term, multiplicity[i].native(), table_inverse[i].native());
                        mcl::Fr::add(running, running, table_term);
                    }
                    if (!q_query[i].is_zero())
                    {
                        mcl::Fr::sub(running, running, query_inverse[i].native());
                    }
                    accumulator[i + 1] = FieldElement::from_native(running);
                }
                return running;
            };

            const auto hardware_workers = std::max<std::size_t>(1, std::min<std::size_t>(kMaxParallelWorkers, std::thread::hardware_concurrency()));
            constexpr std::size_t kParallelThreshold = 1ULL << 18;
            if (active_count >= kParallelThreshold && hardware_workers > 1)
            {
                const auto chunk_count = std::min<std::size_t>(hardware_workers, std::max<std::size_t>(2, active_count / (1ULL << 16)));
                const auto chunk_size = (active_count + chunk_count - 1) / chunk_count;
                std::vector<mcl::Fr> chunk_totals(chunk_count);
                std::vector<std::future<void>> futures;
                futures.reserve(chunk_count);
                for (std::size_t chunk = 0; chunk < chunk_count; ++chunk)
                {
                    const auto begin = chunk * chunk_size;
                    const auto end = std::min(active_count, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(std::launch::async,
                                                 [&, chunk, begin, end]
                                                 {
                                                     mcl::Fr zero;
                                                     zero.clear();
                                                     chunk_totals[chunk] = accumulate_range(begin, end, zero);
                                                 }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }

                std::vector<mcl::Fr> chunk_offsets(chunk_count);
                mcl::Fr offset;
                offset.clear();
                for (std::size_t chunk = 0; chunk < chunk_count; ++chunk)
                {
                    chunk_offsets[chunk] = offset;
                    mcl::Fr::add(offset, offset, chunk_totals[chunk]);
                }
                futures.clear();
                for (std::size_t chunk = 1; chunk < chunk_count; ++chunk)
                {
                    const auto begin = chunk * chunk_size;
                    const auto end = std::min(active_count, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(std::launch::async,
                                                 [&, chunk, begin, end]
                                                 {
                                                     const auto chunk_offset = chunk_offsets[chunk];
                                                     for (std::size_t i = begin; i < end; ++i)
                                                     {
                                                         mcl::Fr value;
                                                         mcl::Fr::add(value, accumulator[i + 1].native(), chunk_offset);
                                                         accumulator[i + 1] = FieldElement::from_native(value);
                                                     }
                                                 }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }
            }
            else
            {
                mcl::Fr zero;
                zero.clear();
                (void)accumulate_range(0, active_count, zero);
            }
            if (active_count + 1 < n)
            {
                std::fill(accumulator.begin() + static_cast<std::ptrdiff_t>(active_count + 1), accumulator.end(), accumulator[active_count]);
            }
            return accumulator;
        }
    }

    std::vector<algebra::FieldElement> build_logup_accumulator_from_inverse_columns_with_active_count(const std::vector<algebra::FieldElement> &table_inverse,
                                                                                                      const std::vector<algebra::FieldElement> &query_inverse,
                                                                                                      const std::vector<algebra::FieldElement> &multiplicity,
                                                                                                      const std::vector<algebra::FieldElement> &q_table,
                                                                                                      const std::vector<algebra::FieldElement> &q_query,
                                                                                                      std::size_t active_count)
    {
        return build_logup_accumulator_from_inverse_columns_impl(table_inverse, query_inverse, multiplicity, q_table, q_query, active_count);
    }

    std::vector<algebra::FieldElement> build_active_prefix_inverse_column(const std::vector<algebra::FieldElement> &values, std::size_t active_length,
                                                                          const algebra::FieldElement &beta)
    {
        if (active_length > values.size())
        {
            throw std::runtime_error("active-prefix inverse length exceeds column size");
        }
        std::vector<FieldElement> out(values.size(), FieldElement::zero());
        std::vector<mcl::Fr> denominators(active_length);
        for (std::size_t i = 0; i < active_length; ++i)
        {
            mcl::Fr::add(denominators[i], values[i].native(), beta.native());
        }
        std::vector<mcl::Fr> inverses;
        batch_inverse_native(denominators, inverses, "active-prefix inverse column");
        for (std::size_t i = 0; i < active_length; ++i)
        {
            out[i] = FieldElement::from_native(inverses[i]);
        }
        return out;
    }
}
