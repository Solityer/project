#include "gatzk/protocol/lookup.hpp"

#include <algorithm>
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

        std::size_t active_transition_count(
            const std::vector<FieldElement> &q_table,
            const std::vector<FieldElement> &q_query)
        {
            // 只把最后一个仍有真实 table/query 活动的位置算进来，后面全视为填充区。
            const auto n = q_table.size();
            if (q_query.size() != n)
            {
                throw std::runtime_error("lookup selector size mismatch");
            }
            if (n <= 1)
            {
                return 0;
            }
            std::size_t active = 0;
            for (std::size_t i = 0; i + 1 < n; ++i)
            {
                if (!q_table[i].is_zero() || !q_query[i].is_zero())
                {
                    active = i + 1;
                }
            }
            return active;
        }

        enum class BatchInverseExecution
        {
            Serial,
            Parallel,
        };

        void batch_inverse_native(
            const std::vector<mcl::Fr> &values,
            std::vector<mcl::Fr> &out,
            const std::string &label,
            BatchInverseExecution execution =
                BatchInverseExecution::Parallel)
        {
            out.assign(values.size(), mcl::Fr{});
            if (values.empty())
            {
                return;
            }

            // 把同一批分母统一做前缀积和一次逆元，减少 LogUp 热路径里的逐项求逆。
            // This buffer is shared by the disjoint chunk workers below, so
            // it must be an ordinary function-local allocation. A
            // thread_local buffer would resolve to an empty, different vector
            // inside each async worker.
            std::vector<mcl::Fr> prefix(values.size());
            const auto hardware_workers = std::max<std::size_t>(
                1,
                std::min<std::size_t>(
                    kMaxParallelWorkers,
                    std::thread::hardware_concurrency()));
            constexpr std::size_t kParallelThreshold = 1ULL << 18;
            if (execution == BatchInverseExecution::Parallel
                && values.size() >= kParallelThreshold
                && hardware_workers > 1)
            {
                const auto chunk_count = std::min<std::size_t>(
                    hardware_workers,
                    std::max<std::size_t>(
                        2, values.size() / (1ULL << 17)));
                const auto chunk_size =
                    (values.size() + chunk_count - 1) / chunk_count;
                std::vector<mcl::Fr> chunk_products(chunk_count);
                const auto parallel_chunks = [&](const auto &fn)
                {
                    std::vector<std::future<void>> futures;
                    futures.reserve(chunk_count);
                    for (std::size_t chunk = 0;
                         chunk < chunk_count; ++chunk)
                    {
                        const auto begin = chunk * chunk_size;
                        const auto end = std::min(
                            values.size(), begin + chunk_size);
                        if (begin >= end)
                        {
                            break;
                        }
                        futures.push_back(std::async(
                            std::launch::async,
                            [&, chunk, begin, end]
                            {
                                fn(chunk, begin, end);
                            }));
                    }
                    for (auto &future : futures)
                    {
                        future.get();
                    }
                };
                parallel_chunks(
                    [&](std::size_t chunk,
                        std::size_t begin,
                        std::size_t end)
                    {
                        mcl::Fr product = 1;
                        for (std::size_t i = begin; i < end; ++i)
                        {
                            if (values[i].isZero())
                            {
                                throw std::runtime_error(
                                    "cannot invert zero in " + label);
                            }
                            prefix[i] = product;
                            mcl::Fr::mul(product, product, values[i]);
                        }
                        chunk_products[chunk] = product;
                    });

                // Only O(worker_count) values remain, so one serial batch
                // inversion is cheaper than another task launch and keeps the
                // same single-inversion algebra as the former implementation.
                std::vector<mcl::Fr> chunk_prefix(chunk_count);
                mcl::Fr chunk_product = 1;
                for (std::size_t chunk = 0;
                     chunk < chunk_count; ++chunk)
                {
                    chunk_prefix[chunk] = chunk_product;
                    mcl::Fr::mul(
                        chunk_product,
                        chunk_product,
                        chunk_products[chunk]);
                }
                mcl::Fr chunk_suffix_inverse;
                mcl::Fr::inv(chunk_suffix_inverse, chunk_product);
                for (std::size_t chunk = chunk_count;
                     chunk-- > 0;)
                {
                    mcl::Fr inverse;
                    mcl::Fr::mul(
                        inverse,
                        chunk_suffix_inverse,
                        chunk_prefix[chunk]);
                    mcl::Fr::mul(
                        chunk_suffix_inverse,
                        chunk_suffix_inverse,
                        chunk_products[chunk]);
                    chunk_products[chunk] = inverse;
                }

                parallel_chunks(
                    [&](std::size_t chunk,
                        std::size_t begin,
                        std::size_t end)
                    {
                        auto suffix_inverse = chunk_products[chunk];
                        for (std::size_t i = end; i-- > begin;)
                        {
                            mcl::Fr::mul(
                                out[i], suffix_inverse, prefix[i]);
                            mcl::Fr::mul(
                                suffix_inverse,
                                suffix_inverse,
                                values[i]);
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

    IndexedViewSideTrace build_indexed_view_side(
        const std::vector<algebra::FieldElement>& indices,
        const std::vector<algebra::FieldElement>& values,
        const std::vector<algebra::FieldElement>& selector,
        const std::vector<algebra::FieldElement>& multiplicity,
        const algebra::FieldElement& eta,
        const algebra::FieldElement& beta)
    {
        const auto n = indices.size();
        if (values.size() != n || selector.size() != n || multiplicity.size() != n)
            throw std::runtime_error("indexed-view side input size mismatch");
        IndexedViewSideTrace out;
        out.tuples.assign(n, FieldElement::zero());
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!(selector[i] == FieldElement::zero())
                && !(selector[i] == FieldElement::one()))
                throw std::runtime_error("indexed-view selector must be boolean");
            if (selector[i].is_zero() && !multiplicity[i].is_zero())
                throw std::runtime_error(
                    "indexed-view multiplicity must be zero outside the selected region");
            out.tuples[i] = selector[i] * (indices[i] + eta * values[i]);
        }
        out.inverses = build_selected_inverse_column(out.tuples, selector, beta);
        out.accumulator.assign(n, FieldElement::zero());
        if (n == 0) return out;
        auto running = FieldElement::zero();
        out.accumulator[0] = running;
        for (std::size_t i = 0; i + 1 < n; ++i)
        {
            running += selector[i] * multiplicity[i] * out.inverses[i];
            out.accumulator[i + 1] = running;
        }
        return out;
    }

    IndexedViewActivePrefixWitness
    build_indexed_view_active_prefix_witness(
        const std::vector<algebra::FieldElement> &tuples,
        std::size_t active_length,
        const algebra::FieldElement &active_multiplicity,
        const algebra::FieldElement &beta)
    {
        if (tuples.empty())
        {
            if (active_length != 0)
            {
                throw std::runtime_error(
                    "indexed-view active prefix exceeds empty tuple column");
            }
            return {};
        }
        if (active_length >= tuples.size())
        {
            throw std::runtime_error(
                "indexed-view active prefix leaves no terminal row");
        }
        if (std::any_of(
                tuples.begin()
                    + static_cast<std::ptrdiff_t>(active_length),
                tuples.end(),
                [](const FieldElement &value)
                { return !value.is_zero(); }))
        {
            throw std::runtime_error(
                "indexed-view tuple column is nonzero outside active prefix");
        }

        IndexedViewActivePrefixWitness out;
        out.inverses = build_active_prefix_inverse_column(
            tuples, active_length, beta);
        out.accumulator.assign(
            tuples.size(), FieldElement::zero());
        mcl::Fr running;
        running.clear();
        for (std::size_t i = 0; i < active_length; ++i)
        {
            mcl::Fr term;
            mcl::Fr::mul(
                term,
                active_multiplicity.native(),
                out.inverses[i].native());
            mcl::Fr::add(running, running, term);
            out.accumulator[i + 1] = FieldElement::from_native(running);
        }
        if (active_length + 1 < tuples.size())
        {
            std::fill(
                out.accumulator.begin()
                    + static_cast<std::ptrdiff_t>(active_length + 1),
                out.accumulator.end(),
                out.accumulator[active_length]);
        }
        return out;
    }

    // 标准入口：自动按选择器截断到最后一个有效转移。
    std::vector<algebra::FieldElement> build_logup_accumulator(
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta)
    {
        return build_logup_accumulator_with_active_count(
            table,
            query,
            multiplicity,
            q_table,
            q_query,
            beta,
            active_transition_count(q_table, q_query));
    }

    namespace
    {
        LogUpInverseColumns build_logup_inverse_columns_impl(
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta,
        BatchInverseExecution execution)
    {
        const auto n = table.size();
        if (query.size() != n || q_table.size() != n || q_query.size() != n)
        {
            throw std::runtime_error("lookup inverse column input size mismatch");
        }
        LogUpInverseColumns out;
        out.table.assign(n, FieldElement::zero());
        out.query.assign(n, FieldElement::zero());
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
        batch_inverse_native(
            denominators,
            inverses,
            "lookup inverse columns",
            execution);
        std::size_t inverse_index = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            if (!q_table[i].is_zero())
            {
                out.table[i] =
                    FieldElement::from_native(
                        inverses[inverse_index++]);
            }
            if (!q_query[i].is_zero())
            {
                out.query[i] =
                    FieldElement::from_native(
                        inverses[inverse_index++]);
            }
        }
        if (inverse_index != inverses.size())
        {
            throw std::runtime_error(
                "lookup inverse column cursor mismatch");
        }
        return out;
    }
    }

    LogUpInverseColumns build_logup_inverse_columns(
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta)
    {
        return build_logup_inverse_columns_impl(
            table, query, q_table, q_query, beta,
            BatchInverseExecution::Parallel);
    }

    LogUpInverseColumns
    build_logup_inverse_columns_serial_for_stage_benchmark(
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta)
    {
        return build_logup_inverse_columns_impl(
            table, query, q_table, q_query, beta,
            BatchInverseExecution::Serial);
    }

    namespace
    {
        enum class AccumulatorExecution
        {
            Serial,
            Parallel,
        };

        std::vector<algebra::FieldElement>
        build_logup_accumulator_from_inverse_columns_impl(
        const std::vector<algebra::FieldElement> &table_inverse,
        const std::vector<algebra::FieldElement> &query_inverse,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        std::size_t active_count,
        AccumulatorExecution execution)
        {
            const auto n = table_inverse.size();
            if (query_inverse.size() != n
                || multiplicity.size() != n
                || q_table.size() != n
                || q_query.size() != n)
            {
                throw std::runtime_error(
                    "lookup inverse accumulator input size mismatch");
            }
            std::vector<algebra::FieldElement> accumulator(
                n, FieldElement::zero());
            if (n <= 1)
            {
                return accumulator;
            }
            active_count = std::min(active_count, n - 1U);
            const auto one = FieldElement::one();
            const auto accumulate_range = [&]
            (
                std::size_t begin,
                std::size_t end,
                const mcl::Fr &offset)
            {
                auto running = offset;
                for (std::size_t i = begin; i < end; ++i)
                {
                    if (!(q_table[i].is_zero()
                            || q_table[i] == one)
                        || !(q_query[i].is_zero()
                            || q_query[i] == one))
                    {
                        throw std::runtime_error(
                            "lookup accumulator selector is not boolean");
                    }
                    if (!q_table[i].is_zero())
                    {
                        mcl::Fr table_term;
                        mcl::Fr::mul(
                            table_term,
                            multiplicity[i].native(),
                            table_inverse[i].native());
                        mcl::Fr::add(running, running, table_term);
                    }
                    if (!q_query[i].is_zero())
                    {
                        mcl::Fr::sub(
                            running,
                            running,
                            query_inverse[i].native());
                    }
                    accumulator[i + 1] =
                        FieldElement::from_native(running);
                }
                return running;
            };

            const auto hardware_workers = std::max<std::size_t>(
                1,
                std::min<std::size_t>(
                    kMaxParallelWorkers,
                    std::thread::hardware_concurrency()));
            constexpr std::size_t kParallelThreshold = 1ULL << 18;
            if (execution == AccumulatorExecution::Parallel
                && active_count >= kParallelThreshold
                && hardware_workers > 1)
            {
                const auto chunk_count = std::min<std::size_t>(
                    hardware_workers,
                    std::max<std::size_t>(
                        2, active_count / (1ULL << 16)));
                const auto chunk_size =
                    (active_count + chunk_count - 1) / chunk_count;
                std::vector<mcl::Fr> chunk_totals(chunk_count);
                std::vector<std::future<void>> futures;
                futures.reserve(chunk_count);
                for (std::size_t chunk = 0;
                     chunk < chunk_count; ++chunk)
                {
                    const auto begin = chunk * chunk_size;
                    const auto end = std::min(
                        active_count, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, chunk, begin, end]
                        {
                            mcl::Fr zero;
                            zero.clear();
                            chunk_totals[chunk] =
                                accumulate_range(begin, end, zero);
                        }));
                }
                for (auto &future : futures)
                {
                    future.get();
                }

                std::vector<mcl::Fr> chunk_offsets(chunk_count);
                mcl::Fr offset;
                offset.clear();
                for (std::size_t chunk = 0;
                     chunk < chunk_count; ++chunk)
                {
                    chunk_offsets[chunk] = offset;
                    mcl::Fr::add(
                        offset, offset, chunk_totals[chunk]);
                }
                futures.clear();
                for (std::size_t chunk = 1;
                     chunk < chunk_count; ++chunk)
                {
                    const auto begin = chunk * chunk_size;
                    const auto end = std::min(
                        active_count, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, chunk, begin, end]
                        {
                            const auto chunk_offset =
                                chunk_offsets[chunk];
                            for (std::size_t i = begin;
                                 i < end; ++i)
                            {
                                mcl::Fr value;
                                mcl::Fr::add(
                                    value,
                                    accumulator[i + 1].native(),
                                    chunk_offset);
                                accumulator[i + 1] =
                                    FieldElement::from_native(value);
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
                std::fill(
                    accumulator.begin()
                        + static_cast<std::ptrdiff_t>(active_count + 1),
                    accumulator.end(),
                    accumulator[active_count]);
            }
            return accumulator;
        }
    }

    std::vector<algebra::FieldElement>
    build_logup_accumulator_from_inverse_columns_with_active_count(
        const std::vector<algebra::FieldElement> &table_inverse,
        const std::vector<algebra::FieldElement> &query_inverse,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        std::size_t active_count)
    {
        return build_logup_accumulator_from_inverse_columns_impl(
            table_inverse,
            query_inverse,
            multiplicity,
            q_table,
            q_query,
            active_count,
            AccumulatorExecution::Parallel);
    }

    std::vector<algebra::FieldElement>
    build_logup_accumulator_from_inverse_columns_with_active_count_serial_for_stage_benchmark(
        const std::vector<algebra::FieldElement> &table_inverse,
        const std::vector<algebra::FieldElement> &query_inverse,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        std::size_t active_count)
    {
        return build_logup_accumulator_from_inverse_columns_impl(
            table_inverse,
            query_inverse,
            multiplicity,
            q_table,
            q_query,
            active_count,
            AccumulatorExecution::Serial);
    }

    std::vector<algebra::FieldElement> build_selected_inverse_column(
        const std::vector<algebra::FieldElement> &values,
        const std::vector<algebra::FieldElement> &selector,
        const algebra::FieldElement &beta)
    {
        if (values.size() != selector.size())
        {
            throw std::runtime_error("selected inverse column input size mismatch");
        }
        std::vector<FieldElement> out(values.size(), FieldElement::zero());
        std::vector<mcl::Fr> denominators;
        std::vector<std::size_t> locations;
        denominators.reserve(values.size());
        locations.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (!selector[i].is_zero())
            {
                mcl::Fr denominator;
                mcl::Fr::add(denominator, values[i].native(), beta.native());
                denominators.push_back(denominator);
                locations.push_back(i);
            }
        }
        std::vector<mcl::Fr> inverses;
        batch_inverse_native(denominators, inverses, "selected inverse column");
        for (std::size_t i = 0; i < locations.size(); ++i)
        {
            out[locations[i]] = FieldElement::from_native(inverses[i]);
        }
        return out;
    }

    std::vector<algebra::FieldElement> build_active_prefix_inverse_column(
        const std::vector<algebra::FieldElement> &values,
        std::size_t active_length,
        const algebra::FieldElement &beta)
    {
        if (active_length > values.size())
        {
            throw std::runtime_error(
                "active-prefix inverse length exceeds column size");
        }
        std::vector<FieldElement> out(
            values.size(), FieldElement::zero());
        std::vector<mcl::Fr> denominators(active_length);
        for (std::size_t i = 0; i < active_length; ++i)
        {
            mcl::Fr::add(
                denominators[i], values[i].native(), beta.native());
        }
        std::vector<mcl::Fr> inverses;
        batch_inverse_native(
            denominators, inverses, "active-prefix inverse column");
        for (std::size_t i = 0; i < active_length; ++i)
        {
            out[i] = FieldElement::from_native(inverses[i]);
        }
        return out;
    }

    // 真正的 LogUp 递推实现。`active_count` 只裁掉填充区，不改变 formal 递推本身。
    std::vector<algebra::FieldElement> build_logup_accumulator_with_active_count(
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta,
        std::size_t active_count)
    {
        const auto n = table.size();
        if (query.size() != n || multiplicity.size() != n || q_table.size() != n || q_query.size() != n)
        {
            throw std::runtime_error("lookup accumulator input size mismatch");
        }
        std::vector<algebra::FieldElement> accumulator;
        accumulator.resize(n);
        if (n <= 1)
        {
            return accumulator;
        }
        active_count = std::min(active_count, n - 1U);
        if (active_count == 0)
        {
            return accumulator;
        }

        thread_local std::vector<mcl::Fr> denominators;
        thread_local std::vector<mcl::Fr> inverses;
        denominators.resize(active_count * 2);
        for (std::size_t i = 0; i < active_count; ++i)
        {
            mcl::Fr::add(denominators[i * 2], table[i].native(), beta.native());
            mcl::Fr::add(denominators[i * 2 + 1], query[i].native(), beta.native());
        }
        batch_inverse_native(denominators, inverses, "lookup accumulator");

        // 这里仍然是 formal note 中相同的 LogUp 递推，只是把分母求逆合并成批处理。
        mcl::Fr running;
        running.clear();
        accumulator[0] = FieldElement::zero();
        for (std::size_t i = 0; i < active_count; ++i)
        {
            mcl::Fr table_term;
            mcl::Fr table_product;
            mcl::Fr::mul(table_product, q_table[i].native(), multiplicity[i].native());
            mcl::Fr::mul(table_term, table_product, inverses[i * 2]);

            mcl::Fr query_term;
            mcl::Fr::mul(query_term, q_query[i].native(), inverses[i * 2 + 1]);

            mcl::Fr::add(running, running, table_term);
            mcl::Fr::sub(running, running, query_term);
            accumulator[i + 1] = FieldElement::from_native(running);
        }
        if (active_count + 1 < n)
        {
            std::fill(
                accumulator.begin() + static_cast<std::ptrdiff_t>(active_count + 1),
                accumulator.end(),
                accumulator[active_count]);
        }
        return accumulator;
    }

    // 缓存版本只复用执行结果，不改变输入、输出或累加器定义。
    std::vector<algebra::FieldElement> build_logup_accumulator_cached(
        const std::string &cache_key,
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta)
    {
        return build_logup_accumulator_cached_with_active_count(
            cache_key,
            table,
            query,
            multiplicity,
            q_table,
            q_query,
            beta,
            active_transition_count(q_table, q_query));
    }

    // 大图热路径常用入口：同时限制有效前缀并允许缓存同形状递推结果。
    std::vector<algebra::FieldElement> build_logup_accumulator_cached_with_active_count(
        const std::string &cache_key,
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta,
        std::size_t active_count)
    {
        active_count = table.empty() ? 0 : std::min(active_count, table.size() - 1U);
        (void)cache_key;
        return build_logup_accumulator_with_active_count(
            table,
            query,
            multiplicity,
            q_table,
            q_query,
            beta,
            active_count);
    }

}
