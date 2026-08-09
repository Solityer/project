#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gatzk/protocol/trace.hpp"

namespace {

double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto middle = samples.size() / 2;
    if (samples.size() % 2 != 0) {
        return samples[middle];
    }
    return (samples[middle - 1] + samples[middle]) / 2.0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto active_count = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : std::size_t{262144};
        const auto repeats = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : std::size_t{5};
        if (argc > 3) {
            throw std::runtime_error(
                "usage: gatzk_signed_round_div_inverse_stage_bench "
                "[active_count repeats]");
        }
        const auto profile =
            gatzk::protocol::benchmark_signed_round_div_inverse_stage(
                active_count, repeats);
        std::cout
            << "active_count=" << profile.active_count
            << " column_size=" << profile.column_size
            << " zero_magnitudes=" << profile.zero_magnitude_count
            << " positive_numerators=" << profile.positive_numerator_count
            << " negative_numerators=" << profile.negative_numerator_count
            << " prefix_scratch_bytes="
            << profile.active_count * sizeof(mcl::Fr) << '\n';
        std::cout << "repeat,scalar_inverse_ms,batch_inverse_ms\n";
        for (std::size_t i = 0; i < repeats; ++i) {
            std::cout
                << i << ',' << std::fixed << std::setprecision(3)
                << profile.scalar_inverse_ms[i] << ','
                << profile.batch_inverse_ms[i] << '\n';
        }
        const auto scalar_median = median(profile.scalar_inverse_ms);
        const auto batch_median = median(profile.batch_inverse_ms);
        std::cout
            << "scalar_median_ms=" << scalar_median
            << " batch_median_ms=" << batch_median
            << " speedup=" << scalar_median / batch_median << 'x' << '\n';
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
