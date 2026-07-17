#include <chrono>
#include <cstddef>
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
    std::size_t count,
    std::size_t nonzero_stride,
    std::size_t repeats) {
    if (points.size() < count) {
        throw std::runtime_error(label + " SRS has insufficient points");
    }
    const auto scalars = make_scalars(count, nonzero_stride);
    const auto nz = nonzero_count(scalars);

    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        gatzk::crypto::G1MSMProfile profiled_profile;
        const auto profiled_start = std::chrono::steady_clock::now();
        const auto profiled = gatzk::crypto::g1_multi_scalar_mul_profiled(
            points.data(),
            scalars.data(),
            count,
            &profiled_profile,
            nullptr);
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
            << "case,input_count,nonzero_count,repeat,profiled_ms,raw_mulvec_ms,"
            << "compact_raw_mulvec_ms,profiled_multithreaded,profiled_sparse_compaction\n";
        run_case("edge_131072_sparse5", *edge_it->second, 131072, 5, repeats);
        run_case("edge_131072_dense", *edge_it->second, 131072, 1, repeats);
        run_case("coef_1048576_sparse3", *keys.pk.powers_g1, 1048576, 3, repeats);
        run_case("coef_1048576_dense", *keys.pk.powers_g1, 1048576, 1, repeats);
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
