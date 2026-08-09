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
    return samples[samples.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto polynomial_size = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : std::size_t{1} << 20;
        const auto repeats = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : std::size_t{5};
        if (argc > 3) {
            throw std::runtime_error(
                "usage: gatzk_taylor_shift_self_check_stage_bench "
                "[polynomial_size repeats]");
        }
        const auto profile =
            gatzk::protocol::benchmark_taylor_shift_self_check_stage(
                polynomial_size, repeats);
        std::cout
            << "polynomial_size=" << profile.polynomial_size
            << " retained_u_rel=" << profile.retained_u_rel.to_string()
            << '\n';
        std::cout << "repeat,full_self_check_ms,default_path_ms\n";
        for (std::size_t i = 0; i < repeats; ++i) {
            std::cout
                << i << ',' << std::fixed << std::setprecision(3)
                << profile.full_self_check_ms[i] << ','
                << profile.default_path_ms[i] << '\n';
        }
        const auto full_median = median(profile.full_self_check_ms);
        const auto default_median = median(profile.default_path_ms);
        std::cout
            << "full_self_check_median_ms=" << full_median
            << " default_path_median_ms=" << default_median
            << " speedup=" << full_median / default_median << 'x'
            << " saved_ms=" << full_median - default_median << '\n';
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
