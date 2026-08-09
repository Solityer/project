#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gatzk/protocol/trace.hpp"

namespace {

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto rows = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : std::size_t{8192};
        const auto columns = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : std::size_t{128};
        const auto repeats = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3]))
            : std::size_t{5};
        if (argc > 4) {
            throw std::runtime_error(
                "usage: gatzk_output_hash_stage_bench [rows columns repeats]");
        }
        const auto profile = gatzk::protocol::benchmark_output_hash_stage(
            rows, columns, repeats);
        const auto buffered = median(profile.buffered_three_pass_ms);
        const auto streamed = median(profile.streamed_reuse_ms);
        std::cout << std::fixed << std::setprecision(3)
                  << "elements=" << profile.element_count
                  << " repeats=" << repeats
                  << " buffered_three_pass_median_ms=" << buffered
                  << " streamed_reuse_median_ms=" << streamed
                  << " speedup=" << buffered / streamed
                  << " exact=1\n";
        for (std::size_t i = 0; i < repeats; ++i) {
            std::cout << "repeat=" << i
                      << " buffered_three_pass_ms="
                      << profile.buffered_three_pass_ms[i]
                      << " streamed_reuse_ms="
                      << profile.streamed_reuse_ms[i] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
