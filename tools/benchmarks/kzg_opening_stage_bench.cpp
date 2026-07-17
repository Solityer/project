#include <algorithm>
#include <chrono>
#include <cstddef>
#include <filesystem>
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
    std::string domain_name;
    std::size_t domain_size = 0;
    std::size_t commitment_count = 0;
    std::size_t nonzero_stride = 1;
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
    std::size_t nonzero_stride) {
    std::vector<FieldElement> values(size, FieldElement::zero());
    for (std::size_t i = 0; i < size; ++i) {
        if (nonzero_stride > 1 && ((i + commitment_index) % nonzero_stride) != 0) {
            continue;
        }
        const auto raw = 31 + ((case_seed + 17 * commitment_index + 131 * i) % 1000003ULL);
        values[i] = FieldElement(raw);
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

void run_case(const CaseSpec& spec, const gatzk::crypto::KZGKeyPair& key) {
    const auto domain = RootOfUnityDomain::create(spec.domain_name, spec.domain_size);
    const auto point = FieldElement(7);
    const std::vector<FieldElement> points = {point, point * domain->omega};
    std::vector<Polynomial> polynomials;
    polynomials.reserve(spec.commitment_count);
    for (std::size_t i = 0; i < spec.commitment_count; ++i) {
        polynomials.push_back(Polynomial::from_evaluations(
            "bench_" + spec.label + "_" + std::to_string(i),
            make_values(spec.domain_size, spec.domain_size, i, spec.nonzero_stride),
            domain));
    }

    std::vector<std::pair<std::string, const Polynomial*>> named_polynomials;
    named_polynomials.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
        named_polynomials.push_back({polynomial.name, &polynomial});
    }

    const auto commit_start = std::chrono::steady_clock::now();
    auto commitments = gatzk::crypto::KZG::commit_batch(named_polynomials, key);
    const auto commit_ms = elapsed_ms(commit_start);

    const auto claim_start = std::chrono::steady_clock::now();
    const auto claimed_values = claimed_values_for_points(polynomials, points);
    const auto claim_ms = elapsed_ms(claim_start);

    gatzk::crypto::BatchOpeningProfile profile;
    const auto open_start = std::chrono::steady_clock::now();
    const auto proof = gatzk::crypto::KZG::open_batch(
        commitments,
        points,
        claimed_values,
        FieldElement(19),
        key,
        &profile);
    const auto open_ms = elapsed_ms(open_start);

    const auto verify_start = std::chrono::steady_clock::now();
    const bool verify_ok = gatzk::crypto::KZG::verify_batch(
        commitments,
        points,
        claimed_values,
        FieldElement(19),
        proof,
        key.vk);
    const auto verify_ms = elapsed_ms(verify_start);

    std::cout
        << spec.label << ','
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
        << proof.witnesses.size() << ','
        << (verify_ok ? "true" : "false") << '\n';
}

}  // namespace

int main() {
    try {
        const auto key = gatzk::crypto::KZG::load_keys(
            (repo_root() / "artifacts/srs/cora_formal_srs_v1").string(),
            "trusted_srs",
            "cora_formal_srs_v1");
        const std::vector<CaseSpec> cases = {
            {"cora_edge_opening_terms96", "edge", 131072, 96, 7},
            {"pubmed_edge_opening_terms96", "edge", 262144, 96, 7},
            {"ppi_edge_opening_terms32", "edge", 1048576, 32, 11},
        };
        std::cout
            << "case,domain_name,domain_size,commitment_count,nonzero_stride,"
            << "commit_setup_ms,claim_eval_ms,open_batch_ms,verify_ms,precompute_ms,"
            << "denominator_inverse_ms,denominator_inverse_reuse_ms,"
            << "denominator_inverse_cache_misses,denominator_inverse_reuse_hits,"
            << "point_parallel_batches,point_parallel_workers,point_msm_worker_cap,"
            << "proof_witness_count,verify_ok\n";
        for (const auto& spec : cases) {
            run_case(spec, key);
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
