#include "gatzk/protocol/lookup.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

// lookup 选择器与 LogUp 累加器
// 特征、激活、range 与 exp 表查询约束的底层公共实现

namespace gatzk::protocol
{
    namespace
    {

        using algebra::FieldElement;

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

        void batch_inverse_native(
            const std::vector<mcl::Fr> &values,
            std::vector<mcl::Fr> &out,
            const std::string &label)
        {
            out.assign(values.size(), mcl::Fr{});
            if (values.empty())
            {
                return;
            }

            // 把同一批分母统一做前缀积和一次逆元，减少 LogUp 热路径里的逐项求逆。
            thread_local std::vector<mcl::Fr> prefix;
            prefix.resize(values.size());
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

    LogUpInverseColumns build_logup_inverse_columns(
        const std::vector<algebra::FieldElement> &table,
        const std::vector<algebra::FieldElement> &query,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        const algebra::FieldElement &beta)
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
        batch_inverse_native(denominators, inverses, "lookup inverse columns");
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

    std::vector<algebra::FieldElement>
    build_logup_accumulator_from_inverse_columns_with_active_count(
        const std::vector<algebra::FieldElement> &table_inverse,
        const std::vector<algebra::FieldElement> &query_inverse,
        const std::vector<algebra::FieldElement> &multiplicity,
        const std::vector<algebra::FieldElement> &q_table,
        const std::vector<algebra::FieldElement> &q_query,
        std::size_t active_count)
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
        mcl::Fr running;
        running.clear();
        const auto one = FieldElement::one();
        for (std::size_t i = 0; i < active_count; ++i)
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
            accumulator[i + 1] = FieldElement::from_native(running);
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
