#include "../cpu-backend.cuh"

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

zkdl::FrWire scalar_u64(std::uint64_t value) {
    zkdl::FrWire out{};
    out[0] = static_cast<std::uint32_t>(value);
    out[1] = static_cast<std::uint32_t>(value >> 32U);
    return out;
}

bool require(bool condition, const char* message) {
    if (condition) return true;
    std::cerr << "cpu_g1_blst_check status=FAIL case=" << message << '\n';
    return false;
}

}  // namespace

int main() {
#if !defined(ZKDL_USE_BLST_G1_CODEC)
    std::cerr << "cpu_g1_blst_check status=FAIL reason=BLST backend is required\n";
    return 2;
#else
    using namespace zkdl;
    const CpuG1Point generator = cpu_g1_generator();
    if (!require(cpu_g1_point_valid(generator), "generator_valid")) return 1;

    const CpuG1Point doubled = cpu_g1_double(generator);
    if (!require(cpu_g1_equal_portable(doubled,
                                       cpu_g1_double_portable(generator)),
                 "double_matches_reference")) return 1;
    const CpuG1Point added = cpu_g1_add(generator, generator);
    if (!require(cpu_g1_equal_portable(added, cpu_g1_add_portable(generator, generator)),
                 "add_matches_reference")) return 1;
    if (!require(cpu_g1_equal(cpu_g1_add(generator, cpu_g1_neg(generator)),
                              CpuG1Point::zero()),
                 "negative_and_infinity")) return 1;

    std::mt19937_64 random(0x7a6b646cULL);
    std::vector<FrWire> scalars = {
        FrWire{}, scalar_u64(1U), scalar_u64(2U), scalar_u64(0x123456789ULL),
        fr_sub_raw(kScalarModulus, scalar_u64(1U))};
    for (unsigned sample = 0U; sample < 2U; ++sample) {
        FrWire candidate{};
        do {
            for (auto& limb : candidate) limb = static_cast<std::uint32_t>(random());
        } while (!fr_canonical(candidate));
        scalars.push_back(candidate);
    }
    scalars.push_back(fr_neg(scalar_u64(0x3141592653589793ULL)));

    for (std::size_t i = 0U; i < scalars.size(); ++i) {
        const CpuG1Point accelerated = cpu_g1_mul(generator, scalars[i]);
        const CpuG1Point reference = cpu_g1_mul_portable(generator, scalars[i]);
        if (!require(cpu_g1_equal_portable(accelerated, reference), "scalar_mul_matches_reference")) {
            std::cerr << "cpu_g1_blst_check scalar_case=" << i << '\n';
            return 1;
        }
    }

    const CpuG1Point order_product = cpu_g1_mul_integer_bits(generator, kScalarModulus);
    const CpuG1Point order_reference = cpu_g1_mul_portable(generator, kScalarModulus);
    if (!require(order_product.is_zero() && cpu_g1_equal_portable(order_product, order_reference),
                 "full_integer_order_boundary")) return 1;
    FrWire order_plus_one = kScalarModulus;
    order_plus_one[0] += 1U;
    const CpuG1Point order_plus_one_product = cpu_g1_mul_integer_bits(generator, order_plus_one);
    const CpuG1Point order_plus_one_reference = cpu_g1_mul_portable(generator, order_plus_one);
    if (!require(cpu_g1_equal(order_plus_one_product, generator) &&
                 cpu_g1_equal_portable(order_plus_one_product, order_plus_one_reference),
                 "full_integer_above_order")) return 1;
    if (!require(cpu_g1_mul(CpuG1Point::zero(), scalar_u64(1U)).is_zero(),
                 "infinity_scalar_multiplication")) return 1;

    bool rejected_scalar = false;
    try {
        (void)cpu_g1_mul_canonical(generator, kScalarModulus);
    } catch (const std::invalid_argument&) {
        rejected_scalar = true;
    }
    if (!require(rejected_scalar, "noncanonical_scalar_rejected")) return 1;

    CpuG1Point invalid_point;
    invalid_point.x = cpu_fp_one();
    invalid_point.y = cpu_fp_one();
    invalid_point.z = cpu_fp_one();
    if (!require(!cpu_g1_point_valid(invalid_point), "off_curve_point_rejected")) return 1;
    bool rejected_point = false;
    try {
        (void)cpu_g1_mul(invalid_point, scalar_u64(1U));
    } catch (const std::invalid_argument&) {
        rejected_point = true;
    }
    if (!require(rejected_point, "arithmetic_rejects_off_curve_point")) return 1;

    std::vector<CpuG1Point> points;
    std::vector<FrWire> weights;
    for (std::uint64_t i = 1U; i <= 8U; ++i) {
        points.push_back(cpu_g1_mul(generator, scalar_u64(i)));
        weights.push_back(scalar_u64(i * i + 3U));
    }
    const CpuG1Point msm = cpu_g1_sum(points, weights);
    const CpuG1Point msm_reference = cpu_g1_sum_portable(points, weights);
    if (!require(cpu_g1_equal_portable(msm, msm_reference), "pippenger_matches_reference")) return 1;

    std::cout << "cpu_g1_blst_check status=PASS scalar_cases=" << scalars.size()
              << " msm_points=" << points.size()
              << " workspace_bound_bytes=" << cpu_g1_msm_workspace_bound_bytes() << '\n';
    return 0;
#endif
}
