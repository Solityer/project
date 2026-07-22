#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"

namespace {

using gatzk::algebra::FieldElement;
using gatzk::algebra::Polynomial;
using gatzk::algebra::RootOfUnityDomain;

struct CaseSpec {
    std::string label;
    std::string srs_dir;
    std::string srs_id;
    std::string domain_name;
    std::size_t domain_size = 0;
    std::size_t commitment_count = 0;
    std::size_t nonzero_stride = 1;
    bool full_width_values = false;
    bool external_unique_points = false;
    bool commit_only = false;
    std::size_t verify_bundle_fanout = 1;
    bool materialized_claims = false;
};

std::filesystem::path repo_root() {
    auto cwd = std::filesystem::current_path();
    if (std::filesystem::exists(cwd / "configs" / "cora_secure.cfg")) {
        return cwd;
    }
    if (std::filesystem::exists(cwd.parent_path() / "configs" / "cora_secure.cfg")) {
        return cwd.parent_path();
    }
    return cwd;
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::vector<FieldElement> make_values(
    std::size_t size,
    std::size_t case_seed,
    std::size_t commitment_index,
    std::size_t nonzero_stride,
    bool full_width_values) {
    std::vector<FieldElement> values(size, FieldElement::zero());
    std::uint64_t state = 0x6a09e667f3bcc909ULL
        ^ static_cast<std::uint64_t>(case_seed)
        ^ (static_cast<std::uint64_t>(commitment_index) << 32U);
    for (std::size_t i = 0; i < size; ++i) {
        if (nonzero_stride > 1 && ((i + commitment_index) % nonzero_stride) != 0) {
            continue;
        }
        if (!full_width_values) {
            const auto raw = 31 + ((case_seed + 17 * commitment_index + 131 * i) % 1000003ULL);
            values[i] = FieldElement(raw);
            continue;
        }
        std::array<std::uint8_t, 32> bytes{};
        for (std::size_t word = 0; word < 4; ++word) {
            state ^= state >> 12U;
            state ^= state << 25U;
            state ^= state >> 27U;
            const auto value = state * 0x2545f4914f6cdd1dULL;
            for (std::size_t byte = 0; byte < 8; ++byte) {
                bytes[word * 8 + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
            }
        }
        values[i] = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
        if (values[i].is_zero()) {
            values[i] = FieldElement::one();
        }
    }
    return values;
}

std::vector<std::vector<FieldElement>> claimed_values_for_points(
    const std::vector<Polynomial>& polynomials,
    const std::vector<FieldElement>& points) {
    std::vector<std::vector<FieldElement>> out;
    out.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
        std::vector<FieldElement> values;
        values.reserve(points.size());
        for (const auto& point : points) {
            values.push_back(polynomial.evaluate(point));
        }
        out.push_back(std::move(values));
    }
    return out;
}

void run_case(const CaseSpec& spec) {
    const auto key = gatzk::crypto::KZG::load_keys(
        (repo_root() / spec.srs_dir).string(),
        "trusted_srs",
        spec.srs_id);
    const auto domain = RootOfUnityDomain::create(spec.domain_name, spec.domain_size);
    std::vector<FieldElement> points;
    if (spec.external_unique_points) {
        points.reserve(spec.commitment_count);
        for (std::size_t i = 0; i < spec.commitment_count; ++i) {
            points.push_back(FieldElement(1009 + 37 * i));
        }
    } else {
        const auto point = FieldElement(7);
        points = {point, point * domain->omega};
    }
    std::vector<Polynomial> polynomials;
    polynomials.reserve(spec.commitment_count);
    for (std::size_t i = 0; i < spec.commitment_count; ++i) {
        auto values = make_values(
                spec.domain_size,
                spec.domain_size,
                i,
                spec.nonzero_stride,
                spec.full_width_values);
        const auto name = "bench_" + spec.label + "_" + std::to_string(i);
        if (spec.commit_only && values.size() >= 1024) {
            polynomials.push_back(Polynomial::from_shared_evaluations(
                name,
                std::make_shared<const std::vector<FieldElement>>(
                    std::move(values)),
                domain));
        } else {
            polynomials.push_back(Polynomial::from_evaluations(
                name, std::move(values), domain));
        }
    }

    std::vector<std::pair<std::string, const Polynomial*>> named_polynomials;
    named_polynomials.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
        named_polynomials.push_back({polynomial.name, &polynomial});
    }

    const auto commit_start = std::chrono::steady_clock::now();
    auto commitments = gatzk::crypto::KZG::commit_batch(named_polynomials, key);
    const auto commit_ms = elapsed_ms(commit_start);

    for (std::size_t i = 0; i < polynomials.size(); ++i) {
        if (polynomials[i].shared_data != nullptr
            && (commitments[i].opening_polynomial == nullptr
                || commitments[i].opening_polynomial->shared_data.get()
                    != polynomials[i].shared_data.get())) {
            throw std::runtime_error(
                "commitment opening material did not retain shared values");
        }
    }

    if (spec.commit_only) {
        std::cout
            << spec.label << ',' << spec.srs_id << ',' << spec.domain_name << ','
            << spec.domain_size << ',' << spec.commitment_count << ','
            << spec.nonzero_stride << ',' << std::fixed << std::setprecision(3)
            << commit_ms
            << ",0.000,0.000,0.000,0.000,0.000,0.000,0,0,0,0,0,0,"
               "0.000,0.000,1,true\n";
        return;
    }

    const auto claim_start = std::chrono::steady_clock::now();
    std::vector<std::vector<FieldElement>> claimed_values;
    if (spec.external_unique_points) {
        claimed_values.reserve(polynomials.size());
        for (std::size_t i = 0; i < polynomials.size(); ++i) {
            claimed_values.push_back({polynomials[i].evaluate(points[i])});
        }
    } else {
        claimed_values = claimed_values_for_points(polynomials, points);
    }
    const auto claim_ms = elapsed_ms(claim_start);

    gatzk::crypto::BatchOpeningProfile profile;
    const auto open_start = std::chrono::steady_clock::now();
    std::vector<std::vector<gatzk::crypto::Commitment>> group_commitments;
    std::vector<std::vector<FieldElement>> group_values;
    std::vector<FieldElement> group_points;
    if (spec.external_unique_points) {
        for (std::size_t i = 0; i < commitments.size(); ++i) {
            group_commitments.push_back({commitments[i]});
            group_values.push_back({claimed_values[i][0]});
            group_points.push_back(points[i]);
        }
    } else {
        for (std::size_t point_index = 0;
             point_index < points.size(); ++point_index) {
            group_commitments.push_back(commitments);
            std::vector<FieldElement> values;
            values.reserve(claimed_values.size());
            for (const auto& column : claimed_values) {
                values.push_back(column[point_index]);
            }
            group_values.push_back(std::move(values));
            group_points.push_back(points[point_index]);
        }
    }
    std::vector<gatzk::crypto::G1Point> witnesses;
    witnesses.reserve(group_points.size());
    for (std::size_t i = 0; i < group_points.size(); ++i) {
        gatzk::crypto::BatchOpeningProfile local_profile;
        witnesses.push_back(gatzk::crypto::KZG::open_point_group(
            group_commitments[i],
            group_points[i],
            group_values[i],
            FieldElement(19 + i),
            key,
            &local_profile,
            spec.materialized_claims
                ? gatzk::crypto::BatchOpeningClaimValidation::
                    AlreadyComputedFromOpeningPolynomials
                : gatzk::crypto::BatchOpeningClaimValidation::
                    VerifyAgainstOpeningPolynomials));
        profile.precompute_ms += local_profile.precompute_ms;
        profile.denominator_inverse_ms += local_profile.denominator_inverse_ms;
        profile.denominator_inverse_reuse_ms +=
            local_profile.denominator_inverse_reuse_ms;
        profile.denominator_inverse_cache_misses +=
            local_profile.denominator_inverse_cache_misses;
        profile.denominator_inverse_reuse_hits +=
            local_profile.denominator_inverse_reuse_hits;
    }
    const auto open_ms = elapsed_ms(open_start);

    const auto verify_all = [&] {
        for (std::size_t i = 0; i < group_points.size(); ++i) {
            if (!gatzk::crypto::KZG::verify_point_group(
                    group_commitments[i],
                    group_points[i],
                    group_values[i],
                    FieldElement(19 + i),
                    witnesses[i],
                    key.vk)) {
                return false;
            }
        }
        return true;
    };
    const auto verify_start = std::chrono::steady_clock::now();
    const bool verify_ok = verify_all();
    const auto verify_ms = elapsed_ms(verify_start);

    double serial_bundle_verify_ms = 0.0;
    double parallel_bundle_verify_ms = 0.0;
    bool bundle_verify_ok = true;
    if (!spec.external_unique_points && spec.verify_bundle_fanout > 1) {
        const auto serial_start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < spec.verify_bundle_fanout; ++i) {
            bundle_verify_ok = verify_all() && bundle_verify_ok;
        }
        serial_bundle_verify_ms = elapsed_ms(serial_start);

        const auto parallel_start = std::chrono::steady_clock::now();
        std::vector<std::future<bool>> futures;
        futures.reserve(spec.verify_bundle_fanout);
        for (std::size_t i = 0; i < spec.verify_bundle_fanout; ++i) {
            futures.push_back(std::async(std::launch::async, [&] {
                return verify_all();
            }));
        }
        for (auto& future : futures) {
            bundle_verify_ok = future.get() && bundle_verify_ok;
        }
        parallel_bundle_verify_ms = elapsed_ms(parallel_start);
    }

    std::cout
        << spec.label << ','
        << spec.srs_id << ','
        << spec.domain_name << ','
        << spec.domain_size << ','
        << spec.commitment_count << ','
        << spec.nonzero_stride << ','
        << std::fixed << std::setprecision(3)
        << commit_ms << ','
        << claim_ms << ','
        << open_ms << ','
        << verify_ms << ','
        << profile.precompute_ms << ','
        << profile.denominator_inverse_ms << ','
        << profile.denominator_inverse_reuse_ms << ','
        << profile.denominator_inverse_cache_misses << ','
        << profile.denominator_inverse_reuse_hits << ','
        << profile.point_parallel_batches << ','
        << profile.point_parallel_workers << ','
        << profile.point_msm_worker_cap << ','
        << witnesses.size() << ','
        << serial_bundle_verify_ms << ','
        << parallel_bundle_verify_ms << ','
        << spec.verify_bundle_fanout << ','
        << (verify_ok && bundle_verify_ok ? "true" : "false") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<CaseSpec> cases = {
            {
                "unit_edge_opening_terms910",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                910,
                7,
                false,
                false,
            },
            {
                "unit_edge_opening_terms910_materialized_claims",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                910,
                7,
                false,
                false,
                false,
                1,
                true,
            },
            {
                "unit_edge_commit_batch3_full",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                3,
                1,
                true,
                false,
            },
            {
                "unit_edge_commit_batch4_full",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                4,
                1,
                true,
                false,
            },
            {
                "unit_edge_commit_batch12_sparse4",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                12,
                4,
                true,
                false,
                true,
            },
            {
                "unit_edge_commit_batch32_sparse4",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                32,
                4,
                true,
                false,
                true,
            },
            {
                "unit_edge_commit_batch64_sparse4",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                64,
                4,
                true,
                false,
                true,
            },
            {
                "unit_edge_commit_batch256_sparse4",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                256,
                4,
                true,
                false,
                true,
            },
            {
                "unit_external_groups8_full",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                8,
                1,
                true,
                true,
            },
            {
                "unit_domain_bundle_fanout7_terms96",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                96,
                7,
                false,
                false,
                false,
                7,
            },
            {
                "unit_external_groups17_full",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                17,
                1,
                true,
                true,
            },
            {
                "unit_external_groups17_materialized_claims",
                "artifacts/srs/unit_security_srs_v1",
                "unit_security_srs_v1",
                "edge",
                4096,
                17,
                1,
                true,
                true,
                false,
                1,
                true,
            },
            {
                "cora_edge_opening_terms96",
                "artifacts/srs/cora_formal_srs_v1",
                "cora_formal_srs_v1",
                "edge",
                32768,
                96,
                7,
                false,
                false,
            },
            {
                "cora_edge_commit_batch12_sparse7",
                "artifacts/srs/cora_formal_srs_v1",
                "cora_formal_srs_v1",
                "edge",
                32768,
                12,
                7,
                true,
                false,
                true,
            },
        };
        if (argc != 2) {
            throw std::runtime_error(
                "usage: gatzk_kzg_opening_stage_bench <case|all>");
        }
        const std::string selected_case = argv[1];
        std::cout
            << "case,srs_id,domain_name,domain_size,commitment_count,nonzero_stride,"
            << "commit_setup_ms,claim_eval_ms,open_point_groups_ms,verify_ms,precompute_ms,"
            << "denominator_inverse_ms,denominator_inverse_reuse_ms,"
            << "denominator_inverse_cache_misses,denominator_inverse_reuse_hits,"
            << "point_parallel_batches,point_parallel_workers,point_msm_worker_cap,"
            << "proof_witness_count,serial_bundle_verify_ms,parallel_bundle_verify_ms,"
            << "verify_bundle_fanout,verify_ok\n";
        for (const auto& spec : cases) {
            if (selected_case != "all" && selected_case != spec.label) {
                continue;
            }
            std::cout.flush();
            run_case(spec);
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
