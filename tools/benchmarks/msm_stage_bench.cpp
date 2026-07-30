#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/curve.hpp"
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

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

std::vector<FieldElement> make_scalars(std::size_t count, std::size_t nonzero_stride) {
    std::vector<FieldElement> scalars(count, FieldElement::zero());
    for (std::size_t i = 0; i < count; ++i) {
        if (nonzero_stride > 1 && (i % nonzero_stride) != 0) {
            continue;
        }
        scalars[i] = FieldElement(17 + (i * 131) % 65521);
    }
    return scalars;
}

std::vector<FieldElement> make_full_width_prefix_scalars(
    std::size_t count,
    std::size_t prefix_count) {
    std::vector<FieldElement> scalars(count, FieldElement::zero());
    std::uint64_t state = 0x6a09e667f3bcc909ULL;
    for (std::size_t i = 0; i < std::min(count, prefix_count); ++i) {
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
        scalars[i] = FieldElement::from_little_endian_mod(bytes.data(), bytes.size());
        if (scalars[i].is_zero()) {
            scalars[i] = FieldElement::one();
        }
    }
    return scalars;
}

std::size_t nonzero_count(const std::vector<FieldElement>& scalars) {
    std::size_t out = 0;
    for (const auto& scalar : scalars) {
        if (!scalar.is_zero()) {
            ++out;
        }
    }
    return out;
}

gatzk::crypto::G1Point raw_mcl_mulvec(
    const std::vector<gatzk::crypto::G1Point>& points,
    const std::vector<FieldElement>& scalars,
    std::size_t count) {
    gatzk::crypto::G1Point out;
    out.value.clear();
    auto* native_points = const_cast<mcl::G1*>(
        reinterpret_cast<const mcl::G1*>(points.data()));
    const auto* native_scalars = FieldElement::native_array_data(scalars.data());
    mcl::G1::mulVec(out.value, native_points, native_scalars, count);
    return out;
}

gatzk::crypto::G1Point compact_raw_mcl_mulvec(
    const std::vector<gatzk::crypto::G1Point>& points,
    const std::vector<FieldElement>& scalars,
    std::size_t count) {
    std::vector<gatzk::crypto::G1Point> compact_points;
    std::vector<FieldElement> compact_scalars;
    compact_points.reserve(count);
    compact_scalars.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!scalars[i].is_zero()) {
            compact_points.push_back(points[i]);
            compact_scalars.push_back(scalars[i]);
        }
    }
    if (compact_scalars.empty()) {
        return gatzk::crypto::g1_zero();
    }
    return raw_mcl_mulvec(compact_points, compact_scalars, compact_scalars.size());
}

void run_case(
    const std::string& label,
    const std::vector<gatzk::crypto::G1Point>& points,
    const std::vector<FieldElement>& scalars,
    std::size_t repeats,
    std::size_t max_worker_count,
    bool stable_srs_bases = false) {
    const auto count = scalars.size();
    if (points.size() < count) {
        throw std::runtime_error(label + " SRS has insufficient points");
    }
    const auto nz = nonzero_count(scalars);

    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        gatzk::crypto::G1MSMProfile profiled_profile;
        const gatzk::crypto::G1MSMExecutionOptions options{
            .max_worker_count = max_worker_count,
            .stable_srs_bases = stable_srs_bases,
        };
        const auto profiled_start = std::chrono::steady_clock::now();
        const auto profiled = gatzk::crypto::g1_multi_scalar_mul_profiled(
            points.data(),
            scalars.data(),
            count,
            &profiled_profile,
            max_worker_count == 0 && !stable_srs_bases
                ? nullptr
                : &options);
        const auto profiled_ms = elapsed_ms(profiled_start);

        const auto raw_start = std::chrono::steady_clock::now();
        const auto raw = raw_mcl_mulvec(points, scalars, count);
        const auto raw_ms = elapsed_ms(raw_start);
        if (!(profiled == raw)) {
            throw std::runtime_error(label + " profiled MSM differs from raw mulVec");
        }

        const auto compact_start = std::chrono::steady_clock::now();
        const auto compact = compact_raw_mcl_mulvec(points, scalars, count);
        const auto compact_ms = elapsed_ms(compact_start);
        if (!(compact == raw)) {
            throw std::runtime_error(label + " compact raw MSM differs from raw mulVec");
        }

        std::cout
            << label << ','
            << count << ','
            << nz << ','
            << max_worker_count << ','
            << repeat << ','
            << std::fixed << std::setprecision(3)
            << profiled_ms << ','
            << raw_ms << ','
            << compact_ms << ','
            << (profiled_profile.multithreaded ? "true" : "false") << ','
            << (profiled_profile.sparse_compaction ? "true" : "false") << '\n';
    }
}

void run_profiled_worker_sweep(
    const std::string& label,
    const std::vector<gatzk::crypto::G1Point>& points,
    const std::vector<FieldElement>& scalars,
    const std::vector<std::size_t>& worker_counts) {
    if (points.size() < scalars.size()) {
        throw std::runtime_error(label + " SRS has insufficient points");
    }
    const auto nz = nonzero_count(scalars);
    std::optional<gatzk::crypto::G1Point> expected;
    std::cout
        << "worker_sweep_case,input_count,nonzero_count,max_workers,profiled_ms,"
        << "profiled_multithreaded,profiled_sparse_compaction\n";
    for (const auto worker_count : worker_counts) {
        gatzk::crypto::G1MSMProfile profile;
        const gatzk::crypto::G1MSMExecutionOptions options{
            .max_worker_count = worker_count,
        };
        const auto start = std::chrono::steady_clock::now();
        const auto result = gatzk::crypto::g1_multi_scalar_mul_profiled(
            points.data(),
            scalars.data(),
            scalars.size(),
            &profile,
            &options);
        const auto profiled_ms = elapsed_ms(start);
        if (expected.has_value() && !(result == *expected)) {
            throw std::runtime_error(
                label + " worker-count sweep produced inconsistent MSM results");
        }
        expected = result;
        std::cout
            << label << ','
            << scalars.size() << ','
            << nz << ','
            << worker_count << ','
            << std::fixed << std::setprecision(3) << profiled_ms << ','
            << (profile.multithreaded ? "true" : "false") << ','
            << (profile.sparse_compaction ? "true" : "false") << '\n'
            << std::flush;
    }
}

void run_edge_logup_commit_batch(
    const gatzk::crypto::KZGKeyPair& keys,
    std::size_t edge_size,
    std::size_t repeats) {
    const auto domain =
        gatzk::algebra::RootOfUnityDomain::create("edge", edge_size);
    std::vector<gatzk::algebra::Polynomial> polynomials;
    polynomials.reserve(3);
    polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
        "J_tbl",
        make_full_width_prefix_scalars(edge_size, std::min<std::size_t>(4096, edge_size)),
        domain));
    polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
        "J_qry",
        make_full_width_prefix_scalars(edge_size, std::min<std::size_t>(21664, edge_size)),
        domain));
    polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
        "R",
        make_full_width_prefix_scalars(edge_size, std::min<std::size_t>(21663, edge_size)),
        domain));
    std::vector<std::pair<std::string, const gatzk::algebra::Polynomial*>> named;
    named.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
        named.push_back({polynomial.name, &polynomial});
    }

    std::vector<gatzk::crypto::G1Point> expected;
    expected.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
        expected.push_back(
            gatzk::crypto::KZG::commit(polynomial.name, polynomial, keys).point);
    }

    std::cout
        << "commit_batch_case,repeat,wall_ms,profile_msm_ms,"
        << "object_0_ms,object_1_ms,object_2_ms\n";
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        gatzk::crypto::CommitBatchProfile profile;
        const auto start = std::chrono::steady_clock::now();
        const auto commitments =
            gatzk::crypto::KZG::commit_batch(named, keys, &profile);
        const auto wall_ms = elapsed_ms(start);
        if (commitments.size() != expected.size()) {
            throw std::runtime_error(
                "edge LogUp batch commitment count mismatch");
        }
        for (std::size_t i = 0; i < commitments.size(); ++i) {
            if (!(commitments[i].point == expected[i])) {
                throw std::runtime_error(
                    "edge LogUp batch differs from standalone production commitment");
            }
        }
        std::cout
            << "edge_logup_batch_full,"
            << repeat << ','
            << std::fixed << std::setprecision(3)
            << wall_ms << ','
            << profile.msm_ms;
        for (std::size_t i = 0; i < 3; ++i) {
            std::cout << ','
                << (i < profile.objects.size() ? profile.objects[i].msm_ms : 0.0);
        }
        std::cout << '\n' << std::flush;
    }
}

void run_cora_mixed_route_commit_batch(
    const gatzk::crypto::KZGKeyPair& keys,
    std::size_t node_size,
    std::size_t edge_size,
    std::size_t repeats) {
    const auto node_domain =
        gatzk::algebra::RootOfUnityDomain::create("N", node_size);
    const auto edge_domain =
        gatzk::algebra::RootOfUnityDomain::create("edge", edge_size);
    std::vector<gatzk::algebra::Polynomial> polynomials;
    polynomials.reserve(4);
    polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
        "J_node",
        make_full_width_prefix_scalars(
            node_size, std::min<std::size_t>(2708, node_size)),
        node_domain));
    polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
        "J_edge",
        make_full_width_prefix_scalars(
            edge_size, std::min<std::size_t>(13264, edge_size)),
        edge_domain));
    polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
        "R_node",
        make_full_width_prefix_scalars(
            node_size, std::min<std::size_t>(4095, node_size)),
        node_domain));
    polynomials.push_back(gatzk::algebra::Polynomial::from_evaluations(
        "R_edge",
        make_full_width_prefix_scalars(
            edge_size, std::min<std::size_t>(13264, edge_size)),
        edge_domain));
    for (std::size_t index = 0; index < polynomials.size(); ++index) {
        polynomials[index].set_zero_domain_blinder(
            {FieldElement(101 + index), FieldElement(211 + index)});
    }

    std::vector<std::pair<std::string, const gatzk::algebra::Polynomial*>> named;
    named.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
        named.push_back({polynomial.name, &polynomial});
    }

    std::vector<gatzk::crypto::G1Point> expected;
    expected.reserve(polynomials.size());
    for (const auto& polynomial : polynomials) {
        expected.push_back(
            gatzk::crypto::KZG::commit(
                polynomial.name, polynomial, keys).point);
    }

    std::cout
        << "mixed_route_batch_case,repeat,wall_ms,profile_msm_ms,"
        << "node_j_ms,edge_j_ms,node_r_ms,edge_r_ms\n";
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        gatzk::crypto::CommitBatchProfile profile;
        const auto start = std::chrono::steady_clock::now();
        const auto commitments =
            gatzk::crypto::KZG::commit_batch(named, keys, &profile);
        const auto wall_ms = elapsed_ms(start);
        if (commitments.size() != expected.size()) {
            throw std::runtime_error(
                "mixed route batch commitment count mismatch");
        }
        for (std::size_t index = 0; index < commitments.size(); ++index) {
            if (!(commitments[index].point == expected[index])) {
                throw std::runtime_error(
                    "mixed route batch differs from standalone production commitment");
            }
        }
        std::cout
            << "cora_mixed_route_batch,"
            << repeat << ','
            << std::fixed << std::setprecision(3)
            << wall_ms << ','
            << profile.msm_ms;
        for (std::size_t index = 0; index < 4; ++index) {
            std::cout << ','
                << (index < profile.objects.size()
                        ? profile.objects[index].msm_ms
                        : 0.0);
        }
        std::cout << '\n' << std::flush;
    }
}

void run_edge_scaled_evaluation_cache(
    const gatzk::crypto::KZGKeyPair& keys,
    std::size_t edge_size,
    std::size_t repeats) {
    const auto domain =
        gatzk::algebra::RootOfUnityDomain::create("edge", edge_size);
    const auto active_count = std::min<std::size_t>(21664, edge_size - 1);
    std::vector<FieldElement> base(edge_size, FieldElement::zero());
    for (std::size_t i = 1; i < active_count; ++i) {
        base[i] = FieldElement(active_count - i);
    }
    gatzk::crypto::EvaluationCommitmentCache cache;
    std::cout
        << "scaled_cache_case,repeat,wall_ms,object_msm_ms,semantic_reuse\n";
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto scale = make_full_width_prefix_scalars(1, 1).front()
            + FieldElement(repeat + 1);
        auto values =
            std::make_shared<std::vector<FieldElement>>(edge_size);
        for (std::size_t i = 0; i < edge_size; ++i) {
            (*values)[i] = base[i] * scale;
        }
        auto polynomial = gatzk::algebra::Polynomial::from_shared_evaluations(
            "scaled_R",
            values,
            domain);
        polynomial.set_zero_domain_blinder(
            {FieldElement(101 + repeat), FieldElement(211 + repeat)});
        gatzk::crypto::CommitBatchProfile batch_profile;
        const auto start = std::chrono::steady_clock::now();
        const auto cached = gatzk::crypto::KZG::commit_batch(
            {{"scaled_R", &polynomial}},
            keys,
            &batch_profile,
            &cache);
        const auto wall_ms = elapsed_ms(start);
        const auto expected =
            gatzk::crypto::KZG::commit(
                "scaled_R",
                polynomial,
                keys);
        if (cached.size() != 1 || !(cached.front().point == expected.point)) {
            throw std::runtime_error(
                "scaled evaluation cache changed the blinded KZG commitment");
        }
        std::cout
            << "edge_scaled_cache_full,"
            << repeat << ','
            << std::fixed << std::setprecision(3)
            << wall_ms << ','
            << (batch_profile.objects.empty()
                    ? 0.0
                    : batch_profile.objects.front().msm_ms)
            << ','
            << (!batch_profile.objects.empty()
                    && batch_profile.objects.front().semantic_commitment_reuse
                    ? "true"
                    : "false")
            << '\n' << std::flush;
    }
}

void run_edge_logup_mixed_cache_batch(
    const gatzk::crypto::KZGKeyPair& keys,
    std::size_t edge_size,
    std::size_t repeats) {
    const auto domain =
        gatzk::algebra::RootOfUnityDomain::create("edge", edge_size);
    const auto query_count = std::min<std::size_t>(21664, edge_size - 1);
    const auto j_table_base =
        make_full_width_prefix_scalars(
            edge_size,
            std::min<std::size_t>(4096, edge_size));
    const auto query_base =
        make_full_width_prefix_scalars(edge_size, query_count);
    std::vector<FieldElement> accumulator_base(
        edge_size,
        FieldElement::zero());
    for (std::size_t i = 1; i < query_count; ++i) {
        accumulator_base[i] = FieldElement(query_count - i);
    }

    gatzk::crypto::EvaluationCommitmentCache cache;
    std::cout
        << "mixed_cache_batch_case,repeat,wall_ms,profile_msm_ms,"
        << "j_table_ms,j_query_ms,accumulator_ms,"
        << "j_table_reuse,j_query_reuse,accumulator_reuse\n";
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto scale = make_full_width_prefix_scalars(1, 1).front()
            + FieldElement(repeat + 1);
        auto j_table =
            std::make_shared<std::vector<FieldElement>>(j_table_base);
        auto j_query =
            std::make_shared<std::vector<FieldElement>>(edge_size);
        auto accumulator =
            std::make_shared<std::vector<FieldElement>>(edge_size);
        for (std::size_t i = 0; i < j_table->size(); ++i) {
            if (!(*j_table)[i].is_zero()) {
                (*j_table)[i] +=
                    FieldElement(repeat + 1) * FieldElement(i + 1);
            }
            (*j_query)[i] = query_base[i] * scale;
            (*accumulator)[i] = accumulator_base[i] * scale;
        }
        std::vector<gatzk::algebra::Polynomial> polynomials;
        polynomials.reserve(3);
        polynomials.push_back(
            gatzk::algebra::Polynomial::from_shared_evaluations(
                "J_tbl", j_table, domain));
        polynomials.push_back(
            gatzk::algebra::Polynomial::from_shared_evaluations(
                "J_qry", j_query, domain));
        polynomials.push_back(
            gatzk::algebra::Polynomial::from_shared_evaluations(
                "R", accumulator, domain));
        std::vector<std::pair<
            std::string,
            const gatzk::algebra::Polynomial*>> named;
        for (auto& polynomial : polynomials) {
            polynomial.set_zero_domain_blinder(
                {FieldElement(101 + repeat), FieldElement(211 + repeat)});
            named.push_back({polynomial.name, &polynomial});
        }

        gatzk::crypto::CommitBatchProfile profile;
        const auto start = std::chrono::steady_clock::now();
        const auto commitments =
            gatzk::crypto::KZG::commit_batch(
                named,
                keys,
                &profile,
                &cache);
        const auto wall_ms = elapsed_ms(start);
        for (std::size_t i = 0; i < polynomials.size(); ++i) {
            const auto expected = gatzk::crypto::KZG::commit(
                polynomials[i].name,
                polynomials[i],
                keys);
            if (!(commitments[i].point == expected.point)) {
                throw std::runtime_error(
                    "mixed-cache LogUp batch changed a blinded KZG commitment");
            }
        }
        std::cout
            << "edge_logup_mixed_cache_batch,"
            << repeat << ','
            << std::fixed << std::setprecision(3)
            << wall_ms << ','
            << profile.msm_ms;
        for (std::size_t i = 0; i < 3; ++i) {
            std::cout << ','
                << (i < profile.objects.size()
                        ? profile.objects[i].msm_ms
                        : 0.0);
        }
        for (std::size_t i = 0; i < 3; ++i) {
            std::cout << ','
                << (i < profile.objects.size()
                        && profile.objects[i].semantic_commitment_reuse
                        ? "true"
                        : "false");
        }
        std::cout << '\n' << std::flush;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::size_t repeats = 3;
        if (argc > 1) {
            repeats = static_cast<std::size_t>(std::stoull(argv[1]));
        }
        const std::string selected_case = argc > 2 ? argv[2] : "all";
        const std::size_t max_worker_count = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3]))
            : 0;
        const auto should_run = [&](const std::string& label) {
            return selected_case == "all" || selected_case == label;
        };
        if (argc > 8) {
            throw std::runtime_error(
                "usage: gatzk_msm_stage_bench [repeats] [case|all] [max_workers|0] "
                "[srs_path] [srs_type] [srs_id] [edge_domain_size]");
        }
        const auto srs_path = argc > 4
            ? std::filesystem::path(argv[4])
            : repo_root() / "artifacts/srs/unit_security_srs_v1";
        const std::string srs_type = argc > 5 ? argv[5] : "trusted_srs";
        const std::string srs_id = argc > 6 ? argv[6] : "unit_security_srs_v1";
        const std::size_t selected_edge_domain_size =
            argc > 7
                ? static_cast<std::size_t>(std::stoull(argv[7]))
                : 0;
        const bool fh_only =
            selected_case == "fh_cora_feature_prefix_full";
        const auto keys = fh_only
            ? gatzk::crypto::KZG::load_keys_for_domains(
                  srs_path.string(),
                  srs_type,
                  srs_id,
                  {{"FH", 4194304}},
                  4194305)
            : (selected_edge_domain_size == 0
                ? gatzk::crypto::KZG::load_keys(
                      srs_path.string(),
                      srs_type,
                      srs_id)
                : gatzk::crypto::KZG::load_keys_for_domains(
                      srs_path.string(),
                      srs_type,
                      srs_id,
                      {{"edge", selected_edge_domain_size}},
                      selected_edge_domain_size + 1));
        const auto edge_it = keys.pk.lagrange_srs_by_domain.find("edge");
        if (!fh_only
            && (edge_it == keys.pk.lagrange_srs_by_domain.end()
                || edge_it->second == nullptr)) {
            throw std::runtime_error("missing edge Lagrange SRS");
        }
        const auto node_it = keys.pk.lagrange_srs_by_domain.find("N");
        const auto fh_it = keys.pk.lagrange_srs_by_domain.find("FH");
        if (keys.pk.powers_g1 == nullptr) {
            throw std::runtime_error("missing coefficient SRS");
        }

        std::cout
            << "case,input_count,nonzero_count,max_workers,repeat,profiled_ms,raw_mulvec_ms,"
            << "compact_raw_mulvec_ms,profiled_multithreaded,profiled_sparse_compaction\n";
        bool ran_case = false;
        if (should_run("edge_sparse5")) {
            run_case(
                "edge_current_sparse5",
                *edge_it->second,
                make_scalars(edge_it->second->size(), 5),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_dense")) {
            run_case(
                "edge_current_dense",
                *edge_it->second,
                make_scalars(edge_it->second->size(), 1),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_prefix_quarter_full")) {
            run_case(
                "edge_current_prefix_quarter_full",
                *edge_it->second,
                make_full_width_prefix_scalars(
                    edge_it->second->size(), edge_it->second->size() / 4),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_prefix_21664_full")) {
            run_case(
                "edge_current_prefix_21664_full",
                *edge_it->second,
                make_full_width_prefix_scalars(
                    edge_it->second->size(),
                    std::min<std::size_t>(21664, edge_it->second->size())),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_prefix_21664_stable_full")) {
            run_case(
                "edge_current_prefix_21664_stable",
                *edge_it->second,
                make_full_width_prefix_scalars(
                    edge_it->second->size(),
                    std::min<std::size_t>(21664, edge_it->second->size())),
                repeats,
                max_worker_count,
                true);
            ran_case = true;
        }
        if (should_run("edge_prefix_21664_worker_sweep")) {
            run_profiled_worker_sweep(
                "edge_current_prefix_21664_full",
                *edge_it->second,
                make_full_width_prefix_scalars(
                    edge_it->second->size(),
                    std::min<std::size_t>(
                        21664, edge_it->second->size())),
                max_worker_count == 0
                    ? std::vector<std::size_t>{4, 5, 6, 8, 10, 12, 16, 20}
                    : std::vector<std::size_t>{max_worker_count});
            ran_case = true;
        }
        if (should_run("edge_prefix_4096_full")) {
            run_case(
                "edge_current_prefix_4096_full",
                *edge_it->second,
                make_full_width_prefix_scalars(edge_it->second->size(), 4096),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_prefix_4096_worker_sweep")) {
            run_profiled_worker_sweep(
                "edge_current_prefix_4096_full",
                *edge_it->second,
                make_full_width_prefix_scalars(edge_it->second->size(), 4096),
                max_worker_count == 0
                    ? std::vector<std::size_t>{1, 2, 4, 6, 8, 10, 12, 16, 20}
                    : std::vector<std::size_t>{max_worker_count});
            ran_case = true;
        }
        if (should_run("edge_prefix_1024_full")) {
            run_case(
                "edge_current_prefix_1024_full",
                *edge_it->second,
                make_full_width_prefix_scalars(edge_it->second->size(), 1024),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_prefix_910_full")) {
            run_case(
                "edge_current_prefix_910_full",
                *edge_it->second,
                make_full_width_prefix_scalars(edge_it->second->size(), 910),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_prefix_512_full")) {
            run_case(
                "edge_current_prefix_512_full",
                *edge_it->second,
                make_full_width_prefix_scalars(edge_it->second->size(), 512),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_prefix_256_full")) {
            run_case(
                "edge_current_prefix_256_full",
                *edge_it->second,
                make_full_width_prefix_scalars(edge_it->second->size(), 256),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_prefix_128_full")) {
            run_case(
                "edge_current_prefix_128_full",
                *edge_it->second,
                make_full_width_prefix_scalars(edge_it->second->size(), 128),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("edge_logup_batch_full")) {
            run_edge_logup_commit_batch(
                keys,
                edge_it->second->size(),
                repeats);
            ran_case = true;
        }
        if (should_run("cora_mixed_route_batch")) {
            if (node_it == keys.pk.lagrange_srs_by_domain.end()
                || node_it->second == nullptr) {
                throw std::runtime_error(
                    "cora_mixed_route_batch requires the N Lagrange SRS");
            }
            run_cora_mixed_route_commit_batch(
                keys,
                node_it->second->size(),
                edge_it->second->size(),
                repeats);
            ran_case = true;
        }
        if (should_run("edge_scaled_cache_full")) {
            run_edge_scaled_evaluation_cache(
                keys,
                edge_it->second->size(),
                repeats);
            ran_case = true;
        }
        if (should_run("edge_logup_mixed_cache_batch")) {
            run_edge_logup_mixed_cache_batch(
                keys,
                edge_it->second->size(),
                repeats);
            ran_case = true;
        }
        if (should_run("coef_1048576_sparse3")) {
            run_case(
                "coef_1048576_sparse3",
                *keys.pk.powers_g1,
                make_scalars(1048576, 3),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("coef_1048576_dense")) {
            run_case(
                "coef_1048576_dense",
                *keys.pk.powers_g1,
                make_scalars(1048576, 1),
                repeats,
                max_worker_count);
            ran_case = true;
        }
        if (should_run("coef_8388608_full")) {
            if (keys.pk.powers_g1->size() < 8388608) {
                throw std::runtime_error(
                    "coef_8388608_full requires at least 8388608 coefficient points");
            }
            run_profiled_worker_sweep(
                "coef_8388608_full",
                *keys.pk.powers_g1,
                make_full_width_prefix_scalars(8388608, 8388608),
                max_worker_count == 0
                    ? std::vector<std::size_t>{16, 15}
                    : std::vector<std::size_t>{max_worker_count});
            ran_case = true;
        }
        if (should_run("fh_cora_feature_prefix_full")) {
            if (fh_it == keys.pk.lagrange_srs_by_domain.end()
                || fh_it->second == nullptr
                || fh_it->second->size() < 4194304) {
                throw std::runtime_error(
                    "fh_cora_feature_prefix_full requires the Cora FH Lagrange SRS");
            }
            run_profiled_worker_sweep(
                "fh_cora_feature_prefix_full",
                *fh_it->second,
                make_full_width_prefix_scalars(4194304, 3880564),
                max_worker_count == 0
                    ? std::vector<std::size_t>{8, 12, 16, 20}
                    : std::vector<std::size_t>{max_worker_count});
            ran_case = true;
        }
        if (!ran_case) {
            throw std::runtime_error("unknown MSM benchmark case: " + selected_case);
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
