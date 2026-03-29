#include "gatzk/protocol/lookup.hpp"

#include <stdexcept>

namespace gatzk::protocol {
namespace {

using algebra::FieldElement;

std::vector<FieldElement> batch_inverse(
    const std::vector<FieldElement>& values,
    const std::string& label) {
    std::vector<FieldElement> out(values.size(), FieldElement::zero());
    if (values.empty()) {
        return out;
    }

    std::vector<mcl::Fr> prefix(values.size());
    mcl::Fr product = 1;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i].is_zero()) {
            throw std::runtime_error("cannot invert zero in " + label);
        }
        prefix[i] = product;
        mcl::Fr::mul(product, product, values[i].native());
    }

    mcl::Fr suffix_inverse;
    mcl::Fr::inv(suffix_inverse, product);
    for (std::size_t i = values.size(); i-- > 0;) {
        mcl::Fr inverse;
        mcl::Fr::mul(inverse, prefix[i], suffix_inverse);
        out[i] = FieldElement::from_native(inverse);
        mcl::Fr::mul(suffix_inverse, suffix_inverse, values[i].native());
    }
    return out;
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
    const auto n = table.size();
    if (query.size() != n || multiplicity.size() != n || q_table.size() != n || q_query.size() != n) {
        throw std::runtime_error("lookup accumulator input size mismatch");
    }
    std::vector<algebra::FieldElement> accumulator(n, algebra::FieldElement::zero());
    if (n <= 1) {
        return accumulator;
    }

    std::vector<FieldElement> denominators;
    denominators.reserve((n - 1) * 2);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        denominators.push_back(table[i] + beta);
        denominators.push_back(query[i] + beta);
    }
    const auto inverses = batch_inverse(denominators, "lookup accumulator");

    // This is still the same LogUp recurrence from the main spec. The only
    // change is engineering-level reuse of a single batch inversion across the
    // whole row block instead of re-inverting each denominator independently.
    mcl::Fr running;
    running.clear();
    for (std::size_t i = 0; i + 1 < n; ++i) {
        mcl::Fr table_term;
        mcl::Fr table_product;
        mcl::Fr::mul(table_product, q_table[i].native(), multiplicity[i].native());
        mcl::Fr::mul(table_term, table_product, inverses[i * 2].native());

        mcl::Fr query_term;
        mcl::Fr::mul(query_term, q_query[i].native(), inverses[i * 2 + 1].native());

        mcl::Fr::add(running, running, table_term);
        mcl::Fr::sub(running, running, query_term);
        accumulator[i + 1] = FieldElement::from_native(running);
    }
    return accumulator;
}

}
