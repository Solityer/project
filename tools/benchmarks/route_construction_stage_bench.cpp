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
        const auto node_domain_size = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1]))
            : std::size_t{1} << 20;
        const auto node_active_count = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : std::size_t{900000};
        const auto edge_domain_size = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3]))
            : std::size_t{1} << 20;
        const auto edge_active_count = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4]))
            : std::size_t{900000};
        const auto repeats = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5]))
            : std::size_t{5};
        if (argc > 6) {
            throw std::runtime_error(
                "usage: gatzk_route_construction_stage_bench "
                "[node_domain node_active edge_domain edge_active repeats]");
        }
        const auto profile =
            gatzk::protocol::benchmark_route_construction_stage(
                node_domain_size,
                node_active_count,
                edge_domain_size,
                edge_active_count,
                repeats);
        std::cout
            << "node_domain_size=" << profile.node_domain_size
            << " node_active_count=" << profile.node_active_count
            << " edge_domain_size=" << profile.edge_domain_size
            << " edge_active_count=" << profile.edge_active_count << '\n';
        std::cout << "repeat,legacy_route_ms,from_padded_route_ms\n";
        for (std::size_t i = 0; i < repeats; ++i) {
            std::cout
                << i << ',' << std::fixed << std::setprecision(3)
                << profile.legacy_route_ms[i] << ','
                << profile.from_padded_route_ms[i] << '\n';
        }
        const auto legacy_median = median(profile.legacy_route_ms);
        const auto from_padded_median =
            median(profile.from_padded_route_ms);
        std::cout
            << "legacy_median_ms=" << legacy_median
            << " from_padded_median_ms=" << from_padded_median
            << " speedup=" << legacy_median / from_padded_median << 'x'
            << " saved_ms=" << legacy_median - from_padded_median << '\n';
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
