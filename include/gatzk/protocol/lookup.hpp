#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol {

struct LogUpTrace {
    std::vector<algebra::FieldElement> table;
    std::vector<algebra::FieldElement> query;
    std::vector<algebra::FieldElement> multiplicity;
    std::vector<algebra::FieldElement> q_table;
    std::vector<algebra::FieldElement> q_query;
    std::vector<algebra::FieldElement> accumulator;
};

std::vector<algebra::FieldElement> build_selector(std::size_t valid_length, std::size_t domain_size);
std::vector<algebra::FieldElement> build_logup_accumulator(
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta);
std::vector<algebra::FieldElement> build_logup_accumulator_with_active_count(
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta,
    std::size_t active_count);
std::vector<algebra::FieldElement> build_logup_accumulator_cached(
    const std::string& cache_key,
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta);
std::vector<algebra::FieldElement> build_logup_accumulator_cached_with_active_count(
    const std::string& cache_key,
    const std::vector<algebra::FieldElement>& table,
    const std::vector<algebra::FieldElement>& query,
    const std::vector<algebra::FieldElement>& multiplicity,
    const std::vector<algebra::FieldElement>& q_table,
    const std::vector<algebra::FieldElement>& q_query,
    const algebra::FieldElement& beta,
    std::size_t active_count);

}  // namespace gatzk::protocol
