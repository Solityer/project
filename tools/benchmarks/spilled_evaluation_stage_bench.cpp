#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>

#include "gatzk/protocol/prover.hpp"

namespace {

using gatzk::algebra::FieldElement;
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

FieldElement choose_non_domain_point(
    const gatzk::algebra::RootOfUnityDomain& domain) {
    for (std::uint64_t candidate = 17; candidate < 10000; ++candidate) {
        const FieldElement point(candidate);
        if (!domain.rotation_shift(FieldElement::one(), point).has_value()) {
            return point;
        }
    }
    throw std::runtime_error("failed to select a non-domain benchmark point");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto domain_size = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : (1U << 18U);
        const auto repeats = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : 3;
        if (argc > 3 || repeats == 0 || domain_size == 0
            || (domain_size & (domain_size - 1U)) != 0U) {
            throw std::runtime_error(
                "usage: gatzk_spilled_evaluation_stage_bench [power_of_two_size] [repeats]");
        }

        const auto domain =
            gatzk::algebra::RootOfUnityDomain::create(
                "spilled_evaluation_stage", domain_size);
        constexpr std::size_t job_count = 8;
        const auto spill_nonce =
            std::to_string(Clock::now().time_since_epoch().count());
        std::vector<std::filesystem::path> spill_paths;
        spill_paths.reserve(job_count);
        struct SpillCleanup {
            std::vector<std::filesystem::path>* paths;
            ~SpillCleanup() {
                for (const auto& path : *paths) {
                    std::error_code ignored;
                    std::filesystem::remove(path, ignored);
                }
            }
        } cleanup{&spill_paths};
        std::vector<
            gatzk::protocol::TraceArtifacts::SpilledEvaluationPolynomial>
            spills;
        spills.reserve(job_count);
        for (std::size_t job = 0; job < job_count; ++job) {
            spill_paths.push_back(
                std::filesystem::temp_directory_path()
                / ("gatzk_spilled_evaluation_stage_"
                    + spill_nonce + "_" + std::to_string(job) + ".bin"));
            std::ofstream output(
                spill_paths.back(), std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("failed to create stage spill file");
            }
            std::array<std::uint8_t, 32> bytes{};
            for (std::size_t index = 0; index < domain_size; ++index) {
                const FieldElement value(
                    static_cast<std::uint64_t>(
                        index * 17U + 29U + job * 101U));
                value.write_little_endian(bytes.data(), bytes.size());
                output.write(
                    reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
            }
            if (!output) {
                throw std::runtime_error("failed to write stage spill file");
            }
            spills.push_back(
                gatzk::protocol::TraceArtifacts::SpilledEvaluationPolynomial{
                    .path = spill_paths.back().string(),
                    .domain_name = domain->name,
                    .size = domain_size,
                    .zero_domain_blinder = {
                        FieldElement(31 + job),
                        FieldElement(37 + job),
                        FieldElement(41 + job)},
                });
        }
        const auto& spilled = spills.front();
        const auto point = choose_non_domain_point(*domain);
        const std::vector<FieldElement> points{
            point,
            point * domain->omega,
        };
        gatzk::algebra::DomainEvaluationWeightCache weight_cache;
        std::vector<const gatzk::algebra::DomainEvaluationWeights*> weights;
        weights.reserve(points.size());
        for (const auto& current_point : points) {
            weights.push_back(&weight_cache.get(domain, current_point));
        }

        double single_total_ms = 0.0;
        double grouped_total_ms = 0.0;
        gatzk::protocol::SpilledEvaluationProfile single_profile;
        gatzk::protocol::SpilledEvaluationProfile grouped_profile;
        FieldElement sink = FieldElement::zero();
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            std::vector<FieldElement> single_values;
            single_values.reserve(points.size());
            const auto single_start = Clock::now();
            for (std::size_t point_index = 0;
                 point_index < points.size();
                 ++point_index) {
                single_values.push_back(
                    gatzk::protocol::evaluate_spilled_evaluation_polynomial(
                        spilled,
                        domain,
                        points[point_index],
                        *weights[point_index],
                        &single_profile));
            }
            single_total_ms += elapsed_ms(single_start);

            const auto grouped_start = Clock::now();
            const auto grouped_values =
                gatzk::protocol::evaluate_spilled_evaluation_polynomial_points(
                    spilled,
                    domain,
                    points,
                    weights,
                    &grouped_profile);
            grouped_total_ms += elapsed_ms(grouped_start);
            if (grouped_values != single_values) {
                throw std::runtime_error(
                    "grouped spill evaluation differs from single-point path");
            }
            sink += grouped_values.front();
        }

        const auto expected_single_passes = repeats * points.size();
        if (single_profile.sequential_pass_count != expected_single_passes
            || grouped_profile.sequential_pass_count != repeats
            || grouped_profile.field_elements_streamed
                != repeats * domain_size) {
            throw std::runtime_error(
                "spill stage pass-count audit failed");
        }
        const auto single_avg_ms = single_total_ms / repeats;
        const auto grouped_avg_ms = grouped_total_ms / repeats;
        std::cout
            << "domain_size,points,repeats,single_passes,grouped_passes,"
               "single_avg_ms,grouped_avg_ms,speedup,sink\n"
            << domain_size << ','
            << points.size() << ','
            << repeats << ','
            << single_profile.sequential_pass_count << ','
            << grouped_profile.sequential_pass_count << ','
            << std::fixed << std::setprecision(3)
            << single_avg_ms << ','
            << grouped_avg_ms << ','
            << (single_avg_ms / grouped_avg_ms) << ','
            << sink.to_string() << '\n';

        std::vector<gatzk::protocol::SpilledEvaluationRequest> requests;
        requests.reserve(spills.size());
        for (const auto& job_spill : spills) {
            requests.push_back(gatzk::protocol::SpilledEvaluationRequest{
                .spilled = &job_spill,
                .domain = domain,
                .points = points,
                .weights = weights,
            });
        }
        double serial_batch_total_ms = 0.0;
        double bounded_batch_total_ms = 0.0;
        std::size_t bounded_worker_count = 0;
        auto run_batch = [&](
                             std::size_t workers,
                             gatzk::protocol::SpilledEvaluationBatchProfile* profile,
                             std::vector<std::vector<FieldElement>>* values) {
            const auto start = Clock::now();
            *values = gatzk::protocol::evaluate_spilled_evaluation_requests(
                requests, workers, profile);
            return elapsed_ms(start);
        };
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            gatzk::protocol::SpilledEvaluationBatchProfile serial_profile;
            gatzk::protocol::SpilledEvaluationBatchProfile bounded_profile;
            std::vector<std::vector<FieldElement>> serial_values;
            std::vector<std::vector<FieldElement>> bounded_values;
            if ((repeat & 1U) == 0U) {
                serial_batch_total_ms += run_batch(
                    1, &serial_profile, &serial_values);
                bounded_batch_total_ms += run_batch(
                    4, &bounded_profile, &bounded_values);
            } else {
                bounded_batch_total_ms += run_batch(
                    4, &bounded_profile, &bounded_values);
                serial_batch_total_ms += run_batch(
                    1, &serial_profile, &serial_values);
            }
            if (serial_values != bounded_values
                || serial_profile.worker_count != 1
                || bounded_profile.worker_count < 2
                || bounded_profile.worker_count > 4
                || serial_profile.jobs.size() != job_count
                || bounded_profile.jobs.size() != job_count) {
                throw std::runtime_error(
                    "bounded spill batch changed values or worker limits");
            }
            bounded_worker_count = bounded_profile.worker_count;
            for (std::size_t job = 0; job < job_count; ++job) {
                if (serial_profile.jobs[job].sequential_pass_count != 1
                    || bounded_profile.jobs[job].sequential_pass_count != 1
                    || serial_profile.jobs[job].field_elements_streamed
                        != domain_size
                    || bounded_profile.jobs[job].field_elements_streamed
                        != domain_size) {
                    throw std::runtime_error(
                        "bounded spill batch violated one-pass-per-file semantics");
                }
                sink += bounded_values[job].front();
            }
        }
        const auto serial_batch_avg_ms =
            serial_batch_total_ms / repeats;
        const auto bounded_batch_avg_ms =
            bounded_batch_total_ms / repeats;
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) {
            throw std::runtime_error("failed to read stage resource usage");
        }
        std::cout
            << "domain_size,jobs,points,repeats,input_mib,serial_workers,"
               "bounded_workers,serial_avg_ms,bounded_avg_ms,speedup,"
               "peak_rss_kib,sink\n"
            << domain_size << ','
            << job_count << ','
            << points.size() << ','
            << repeats << ','
            << std::fixed << std::setprecision(3)
            << (static_cast<double>(domain_size * job_count * 32U)
                / (1024.0 * 1024.0)) << ','
            << 1 << ','
            << bounded_worker_count << ','
            << serial_batch_avg_ms << ','
            << bounded_batch_avg_ms << ','
            << (serial_batch_avg_ms / bounded_batch_avg_ms) << ','
            << usage.ru_maxrss << ','
            << sink.to_string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
