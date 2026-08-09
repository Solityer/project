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
        const auto domain_size = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : std::size_t{32768};
        const auto table_count = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3]))
            : std::size_t{4096};
        const auto query_count = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4]))
            : std::size_t{21664};
        if (argc > 5) {
            throw std::runtime_error(
                "usage: gatzk_logup_stage_bench "
                "[repeats domain_size table_count query_count]");
        }
        if (domain_size == 0
            || (domain_size & (domain_size - 1U)) != 0U
            || table_count > domain_size
            || query_count > domain_size) {
            throw std::runtime_error("invalid LogUp benchmark shape");
        }
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

        std::cout
            << "repeat,serial_inverse_ms,parallel_inverse_ms,"
               "serial_accumulator_ms,parallel_accumulator_ms,"
               "total_parallel_ms,exact,inverse_speedup,"
               "accumulator_speedup,"
               "selected_prefix_ms,active_prefix_ms,prefix_exact,"
               "prefix_speedup\n";
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            const auto beta = FieldElement(65537 + repeat);
            const auto serial_start =
                std::chrono::steady_clock::now();
            const auto serial_inverses =
                gatzk::protocol::
                    build_logup_inverse_columns_serial_for_stage_benchmark(
                        table, query, q_table, q_query, beta);
            const auto serial_inverse_ms = elapsed_ms(serial_start);
            const auto inverse_start =
                std::chrono::steady_clock::now();
            const auto inverses =
                gatzk::protocol::build_logup_inverse_columns(
                    table, query, q_table, q_query, beta);
            const auto inverse_duration_ms =
                elapsed_ms(inverse_start);
            std::vector<FieldElement> serial_accumulator;
            std::vector<FieldElement> parallel_accumulator;
            double serial_accumulator_ms = 0.0;
            double parallel_accumulator_ms = 0.0;
            const auto run_serial_accumulator = [&]() {
                const auto start = std::chrono::steady_clock::now();
                serial_accumulator = gatzk::protocol::
                    build_logup_accumulator_from_inverse_columns_with_active_count_serial_for_stage_benchmark(
                        inverses.table,
                        inverses.query,
                        multiplicity,
                        q_table,
                        q_query,
                        query_count);
                serial_accumulator_ms = elapsed_ms(start);
            };
            const auto run_parallel_accumulator = [&]() {
                const auto start = std::chrono::steady_clock::now();
                parallel_accumulator = gatzk::protocol::
                    build_logup_accumulator_from_inverse_columns_with_active_count(
                        inverses.table,
                        inverses.query,
                        multiplicity,
                        q_table,
                        q_query,
                        query_count);
                parallel_accumulator_ms = elapsed_ms(start);
            };
            if ((repeat & 1U) == 0U) {
                run_serial_accumulator();
                run_parallel_accumulator();
            } else {
                run_parallel_accumulator();
                run_serial_accumulator();
            }
            const bool exact =
                serial_inverses.table == inverses.table
                && serial_inverses.query == inverses.query
                && serial_accumulator == parallel_accumulator;
            if (!exact
                || inverses.table[table_count - 1]
                    * (table[table_count - 1] + beta)
                    != FieldElement::one()
                || inverses.query[query_count - 1]
                    * (query[query_count - 1] + beta)
                    != FieldElement::one()
                || parallel_accumulator.size() != domain_size) {
                throw std::runtime_error(
                    "LogUp stage benchmark failed its exact audit");
            }
            std::vector<FieldElement> selected_prefix;
            std::vector<FieldElement> active_prefix;
            double selected_prefix_ms = 0.0;
            double active_prefix_ms = 0.0;
            const auto run_selected_prefix = [&]() {
                const auto start = std::chrono::steady_clock::now();
                const auto selector =
                    gatzk::protocol::build_selector(
                        query_count, domain_size);
                selected_prefix =
                    gatzk::protocol::build_selected_inverse_column(
                        query, selector, beta);
                selected_prefix_ms = elapsed_ms(start);
            };
            const auto run_active_prefix = [&]() {
                const auto start = std::chrono::steady_clock::now();
                active_prefix =
                    gatzk::protocol::build_active_prefix_inverse_column(
                        query, query_count, beta);
                active_prefix_ms = elapsed_ms(start);
            };
            if ((repeat & 1U) == 0U) {
                run_selected_prefix();
                run_active_prefix();
            } else {
                run_active_prefix();
                run_selected_prefix();
            }
            const bool prefix_exact = selected_prefix == active_prefix;
            if (!prefix_exact) {
                throw std::runtime_error(
                    "active-prefix inverse output mismatch");
            }
            std::cout
                << repeat << ','
                << std::fixed << std::setprecision(3)
                << serial_inverse_ms << ','
                << inverse_duration_ms << ','
                << serial_accumulator_ms << ','
                << parallel_accumulator_ms << ','
                << inverse_duration_ms + parallel_accumulator_ms << ','
                << (exact ? "true" : "false") << ','
                << serial_inverse_ms / inverse_duration_ms << ','
                << serial_accumulator_ms / parallel_accumulator_ms << ','
                << selected_prefix_ms << ','
                << active_prefix_ms << ','
                << (prefix_exact ? "true" : "false") << ','
                << selected_prefix_ms / active_prefix_ms << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
