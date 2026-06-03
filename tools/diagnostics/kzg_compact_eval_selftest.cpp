#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/kzg.hpp"

namespace {

using gatzk::algebra::FieldElement;

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

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

FieldElement choose_non_domain_point(const gatzk::algebra::RootOfUnityDomain& domain) {
    for (std::uint64_t raw = 17; raw < 10000; ++raw) {
        const FieldElement candidate(raw);
        bool overlap = false;
        for (std::size_t i = 0; i < domain.size; ++i) {
            if (domain.point_at(i) == candidate) {
                overlap = true;
                break;
            }
        }
        if (!overlap) {
            return candidate;
        }
    }
    throw std::runtime_error("failed to find a non-domain opening point");
}

}  // namespace

int main() {
    try {
        const auto keys = gatzk::crypto::KZG::load_keys(
            (repo_root() / "artifacts/srs/zkgat_secure_main_srs_v1").string(),
            "trusted_srs",
            "zkgat_secure_main_srs_v1");

        const auto n_size = keys.manifest.domain_sizes.at("N");
        auto domain = gatzk::algebra::RootOfUnityDomain::create("N", n_size);
        auto polynomial = gatzk::algebra::Polynomial::from_evaluations(
            "P_compact_eval",
            {FieldElement(3), FieldElement(5), FieldElement(7), FieldElement(11), FieldElement(13)},
            domain);
        const auto point = choose_non_domain_point(*domain);
        const auto value = polynomial.evaluate(point);
        const auto commitment = gatzk::crypto::KZG::commit("P_compact_eval", polynomial, keys);
        const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> openings{
            {commitment, value},
        };
        const std::vector<FieldElement> points{point};
        const auto proof = gatzk::crypto::KZG::open_external_fold(openings, points, FieldElement(29), keys);

        require(
            gatzk::crypto::KZG::verify_external_fold(openings, points, FieldElement(29), proof, keys),
            "compact evaluation external opening must verify");

        auto tampered = openings;
        tampered[0].second += FieldElement::one();
        require(
            !gatzk::crypto::KZG::verify_external_fold(tampered, points, FieldElement(29), proof, keys),
            "compact evaluation external opening must reject a tampered value");

        std::cout << "compact evaluation KZG selftest passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
