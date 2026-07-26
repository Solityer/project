#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gatzk/protocol/lookup.hpp"

namespace {

using gatzk::algebra::FieldElement;

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto repeats = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : 20;
        if (argc > 2) {
            throw std::runtime_error(
                "usage: gatzk_logup_stage_bench [repeats]");
        }
        constexpr std::size_t domain_size = 32768;
        constexpr std::size_t table_count = 4096;
        constexpr std::size_t query_count = 21664;
        std::vector<FieldElement> table(
            domain_size, FieldElement::zero());
        std::vector<FieldElement> query(
            domain_size, FieldElement::zero());
        std::vector<FieldElement> multiplicity(
            domain_size, FieldElement::zero());
        for (std::size_t i = 0; i < table_count; ++i) {
            table[i] = FieldElement(i);
            multiplicity[i] =
                FieldElement(1 + (i * 17) % 11);
        }
        for (std::size_t i = 0; i < query_count; ++i) {
            query[i] = FieldElement((i * 29) % table_count);
        }
        const auto q_table =
            gatzk::protocol::build_selector(
                table_count, domain_size);
        const auto q_query =
            gatzk::protocol::build_selector(
                query_count, domain_size);

        std::cout << "repeat,inverse_ms,accumulator_ms,total_ms\n";
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            const auto beta = FieldElement(65537 + repeat);
            const auto total_start =
                std::chrono::steady_clock::now();
            const auto inverse_start = total_start;
            const auto inverses =
                gatzk::protocol::build_logup_inverse_columns(
                    table, query, q_table, q_query, beta);
            const auto inverse_duration_ms =
                elapsed_ms(inverse_start);
            const auto accumulator_start =
                std::chrono::steady_clock::now();
            const auto accumulator =
                gatzk::protocol::
                    build_logup_accumulator_from_inverse_columns_with_active_count(
                        inverses.table,
                        inverses.query,
                        multiplicity,
                        q_table,
                        q_query,
                        query_count);
            const auto accumulator_duration_ms =
                elapsed_ms(accumulator_start);
            if (inverses.table[table_count - 1]
                    * (table[table_count - 1] + beta)
                    != FieldElement::one()
                || inverses.query[query_count - 1]
                    * (query[query_count - 1] + beta)
                    != FieldElement::one()
                || accumulator.size() != domain_size) {
                throw std::runtime_error(
                    "LogUp stage benchmark failed its exact audit");
            }
            std::cout
                << repeat << ','
                << std::fixed << std::setprecision(3)
                << inverse_duration_ms << ','
                << accumulator_duration_ms << ','
                << elapsed_ms(total_start) << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
