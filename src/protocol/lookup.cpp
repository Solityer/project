#include "gatzk/protocol/lookup.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace gatzk::protocol {
namespace {

using algebra::FieldElement;

std::size_t active_transition_count(
    const std::vector<FieldElement>& q_table,
    const std::vector<FieldElement>& q_query) {
    const auto n = q_table.size();
    if (q_query.size() != n) {
        throw std::runtime_error("lookup selector size mismatch");
    }
    if (n <= 1) {
        return 0;
    }
    std::size_t active = 0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        if (!q_table[i].is_zero() || !q_query[i].is_zero()) {
            active = i + 1;
        }
    }
    return active;
}

void batch_inverse_native(
    const std::vector<mcl::Fr>& values,
    std::vector<mcl::Fr>& out,
    const std::string& label) {
    out.assign(values.size(), mcl::Fr{});
    if (values.empty()) {
        return;
    }

    thread_local std::vector<mcl::Fr> prefix;
    prefix.resize(values.size());
    mcl::Fr product = 1;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i].isZero()) {
            throw std::runtime_error("cannot invert zero in " + label);
        }
        prefix[i] = product;
        mcl::Fr::mul(product, product, values[i]);
    }

    mcl::Fr suffix_inverse;
    mcl::Fr::inv(suffix_inverse, product);
    for (std::size_t i = values.size(); i-- > 0;) {
        mcl::Fr::mul(out[i], prefix[i], suffix_inverse);
        mcl::Fr::mul(suffix_inverse, suffix_inverse, values[i]);
    }
}

std::string logup_cache_key(
    const std::string& cache_key,
    std::size_t table_size,
    std::size_t query_size,
    std::size_t active_count,
    const algebra::FieldElement& beta) {
    return cache_key
        + ":" + std::to_string(table_size)
        + ":" + std::to_string(query_size)
        + ":" + std::to_string(active_count)
        + ":" + beta.to_string();
}

}

std::vector<algebra::FieldElement> build_selector(std::size_t valid_length, std::size_t domain_size) {
    std::vector<algebra::FieldElement> out(domain_size, algebra::FieldElement::zero());
    for (std::size_t i = 0; i < valid_length && i < domain_size; ++i) {
        out[i] = algebra::FieldElement::one();
    }
    return out;
}

std::vector<algebra::FieldElement> build_logup_accumulator(
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta) {
    return build_logup_accumulator_with_active_count(
        table,
        query,
        multiplicity,
        q_table,
        q_query,
        beta,
        active_transition_count(q_table, q_query));
}

std::vector<algebra::FieldElement> build_logup_accumulator_with_active_count(
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta,
    std::size_t active_count) {
    const auto n = table.size();
    if (query.size() != n || multiplicity.size() != n || q_table.size() != n || q_query.size() != n) {
        throw std::runtime_error("lookup accumulator input size mismatch");
    }
    std::vector<algebra::FieldElement> accumulator;
    accumulator.resize(n);
    if (n <= 1) {
        return accumulator;
    }
    active_count = std::min(active_count, n - 1U);
    if (active_count == 0) {
        return accumulator;
    }

    thread_local std::vector<mcl::Fr> denominators;
    thread_local std::vector<mcl::Fr> inverses;
    denominators.resize(active_count * 2);
    for (std::size_t i = 0; i < active_count; ++i) {
        mcl::Fr::add(denominators[i * 2], table[i].native(), beta.native());
        mcl::Fr::add(denominators[i * 2 + 1], query[i].native(), beta.native());
    }
    batch_inverse_native(denominators, inverses, "lookup accumulator");

    // This is still the same LogUp recurrence from the main spec. The only
    // change is engineering-level reuse of a single batch inversion across the
    // whole row block instead of re-inverting each denominator independently.
    mcl::Fr running;
    running.clear();
    accumulator[0] = FieldElement::zero();
    for (std::size_t i = 0; i < active_count; ++i) {
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
    if (active_count + 1 < n) {
        std::fill(
            accumulator.begin() + static_cast<std::ptrdiff_t>(active_count + 1),
            accumulator.end(),
            accumulator[active_count]);
    }
    return accumulator;
}

std::vector<algebra::FieldElement> build_logup_accumulator_cached(
    const std::string& cache_key,
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta) {
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

std::vector<algebra::FieldElement> build_logup_accumulator_cached_with_active_count(
    const std::string& cache_key,
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta,
    std::size_t active_count) {
    active_count = table.empty() ? 0 : std::min(active_count, table.size() - 1U);
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<std::vector<algebra::FieldElement>>> cache;
    const auto key = logup_cache_key(cache_key, table.size(), query.size(), active_count, beta);
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(key); it != cache.end()) {
            return *it->second;
        }
    }

    auto values = std::make_shared<std::vector<algebra::FieldElement>>(
        build_logup_accumulator_with_active_count(
            table,
            query,
            multiplicity,
            q_table,
            q_query,
            beta,
            active_count));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto [it, _] = cache.emplace(key, values);
        return *it->second;
    }
}

}
