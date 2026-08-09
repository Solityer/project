#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/util/config.hpp"

namespace {

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            throw std::runtime_error(
                "usage: gatzk_t_edge_access_plan_stage_bench <small-config> [repeats]");
        }
        const auto repeats = argc == 3
            ? static_cast<std::size_t>(std::stoull(argv[2]))
            : std::size_t{3};
        if (repeats == 0) {
            throw std::runtime_error("repeats must be positive");
        }
        const auto config = gatzk::util::load_config(argv[1]);
        const auto context = gatzk::protocol::build_context(config);
        constexpr std::size_t kMaxNodes = 1024;
        constexpr std::size_t kMaxEdges = 16384;
        if (context.local.num_nodes > kMaxNodes
            || context.local.edges.size() > kMaxEdges) {
            throw std::runtime_error(
                "t_edge access-plan stage benchmark refuses non-small inputs");
        }
        const auto trace = gatzk::protocol::build_trace(context);
        const auto challenges = gatzk::protocol::replay_challenges(
            context, trace.commitments, {}, &trace.public_metadata);
        const auto profile =
            gatzk::protocol::benchmark_t_edge_access_plan_stage(
                context, trace, challenges, repeats);
        const auto legacy_median = median(profile.legacy_resolver_ms);
        const auto bound_median = median(profile.bound_access_plan_ms);
        std::cout << std::fixed << std::setprecision(3)
                  << "quotient_points=" << profile.quotient_points
                  << " access_plan_entries=" << profile.access_plan_entries
                  << " late_bound_dependencies="
                  << profile.late_bound_dependencies
                  << " repeats=" << repeats
                  << " legacy_median_ms=" << legacy_median
                  << " bound_median_ms=" << bound_median
                  << " speedup=" << (legacy_median / bound_median)
                  << " exact=1\n";
        for (std::size_t i = 0; i < repeats; ++i) {
            std::cout << "repeat=" << i
                      << " legacy_ms=" << profile.legacy_resolver_ms[i]
                      << " bound_ms=" << profile.bound_access_plan_ms[i]
                      << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
