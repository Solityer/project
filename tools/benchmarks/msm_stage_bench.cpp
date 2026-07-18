#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mcl/bn.hpp>

#include "gatzk/algebra/field.hpp"
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
    std::size_t max_worker_count) {
    const auto count = scalars.size();
    if (points.size() < count) {
        throw std::runtime_error(label + " SRS has insufficient points");
    }
    const auto nz = nonzero_count(scalars);

    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        gatzk::crypto::G1MSMProfile profiled_profile;
        const gatzk::crypto::G1MSMExecutionOptions options{
            .max_worker_count = max_worker_count,
        };
        const auto profiled_start = std::chrono::steady_clock::now();
        const auto profiled = gatzk::crypto::g1_multi_scalar_mul_profiled(
            points.data(),
            scalars.data(),
            count,
            &profiled_profile,
            max_worker_count == 0 ? nullptr : &options);
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
        if (argc > 4) {
            throw std::runtime_error("usage: gatzk_msm_stage_bench [repeats] [case|all] [max_workers|0]");
        }
        const auto keys = gatzk::crypto::KZG::load_keys(
            (repo_root() / "artifacts/srs/cora_formal_srs_v1").string(),
            "trusted_srs",
            "cora_formal_srs_v1");
        const auto edge_it = keys.pk.lagrange_srs_by_domain.find("edge");
        if (edge_it == keys.pk.lagrange_srs_by_domain.end() || edge_it->second == nullptr) {
            throw std::runtime_error("missing edge Lagrange SRS");
        }
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
        if (should_run("edge_prefix_4096_full")) {
            run_case(
                "edge_current_prefix_4096_full",
                *edge_it->second,
                make_full_width_prefix_scalars(edge_it->second->size(), 4096),
                repeats,
                max_worker_count);
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
        if (!ran_case) {
            throw std::runtime_error("unknown MSM benchmark case: " + selected_case);
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
