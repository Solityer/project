#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"

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
            : 5;
        const auto configured_degree = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : 32769;
        if (argc > 3) {
            throw std::runtime_error(
                "usage: gatzk_taylor_shift_stage_bench "
                "[repeats] [configured_degree]");
        }

        std::vector<FieldElement> coefficients(
            configured_degree + 1,
            FieldElement::zero());
        for (std::size_t i = 0; i < coefficients.size(); ++i) {
            coefficients[i] =
                FieldElement(17 + (i * 131) % 65521);
        }
        const FieldElement delta(37);
        const FieldElement audit_point(101);
        const auto expected = gatzk::algebra::horner(
            coefficients, audit_point + delta);

        std::cout << "repeat,configured_degree,shift_ms\n";
        for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
            const auto start = std::chrono::steady_clock::now();
            const auto shifted =
                gatzk::algebra::coefficient_taylor_shift(
                    coefficients, configured_degree, delta);
            const auto duration_ms = elapsed_ms(start);
            if (gatzk::algebra::horner(shifted, audit_point)
                != expected) {
                throw std::runtime_error(
                    "Taylor-shift benchmark changed polynomial semantics");
            }
            std::cout
                << repeat << ','
                << configured_degree << ','
                << std::fixed << std::setprecision(3)
                << duration_ms << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
