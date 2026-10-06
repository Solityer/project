#include "gat_adapter/static_table_bounds.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        const std::vector<std::int64_t> monotone{-4, -2, 0, 1, 2, 3};
        std::int64_t output_min = 0;
        std::int64_t output_max = 0;
        require(zkdl_gat::static_table_bounds::symmetric_output_interval(
                    monotone, -2, 3, 2, output_min, output_max),
                "valid authenticated table interval was rejected");
        require(output_min == -4 && output_max == 2, "endpoint outputs do not match the selected input interval");
        const std::int64_t span = output_max - output_min;
        require(zkdl_gat::static_table_bounds::exp_domain_covers_delta(span, static_cast<std::uint64_t>(span)),
                "delta exactly at output span must be admitted");
        require(!zkdl_gat::static_table_bounds::exp_domain_covers_delta(span + 1, static_cast<std::uint64_t>(span)),
                "delta above output span must be rejected");
        require(!zkdl_gat::static_table_bounds::symmetric_output_interval(
                    monotone, -2, 3, 4, output_min, output_max),
                "input interval outside the authenticated table was accepted");
        const std::vector<std::int64_t> non_monotone{-4, -2, 0, -1, 2, 3};
        require(!zkdl_gat::static_table_bounds::symmetric_output_interval(
                    non_monotone, -2, 3, 2, output_min, output_max),
                "non-monotone table interval was accepted");
        const std::vector<std::int64_t> malformed{-4, -2, 0, 1, 2};
        require(!zkdl_gat::static_table_bounds::symmetric_output_interval(
                    malformed, -2, 3, 2, output_min, output_max),
                "table row count mismatch was accepted");

        const std::vector<std::int64_t> high_degree_weights(295U, 16);
        std::int64_t weighted_round_bound = 0;
        require(zkdl_gat::static_table_bounds::positive_weighted_round_abs_bound(
                    high_degree_weights, 1U, 20803, 16U, 64U, weighted_round_bound),
                "positive high-degree weighted average bound was rejected");
        require(weighted_round_bound == 5201,
                "weighted-average bound did not preserve 16/64 scaling and ties-away rounding");
        require(!zkdl_gat::static_table_bounds::positive_weighted_round_abs_bound(
                    high_degree_weights, 0U, 20803, 16U, 64U, weighted_round_bound),
                "empty destination group was accepted");
        require(!zkdl_gat::static_table_bounds::positive_weighted_round_abs_bound(
                    high_degree_weights, 1U, 20803, 16U, 0U, weighted_round_bound),
                "zero aggregate divisor scale was accepted");
        const std::vector<std::int64_t> invalid_weights{16, 0};
        require(!zkdl_gat::static_table_bounds::positive_weighted_round_abs_bound(
                    invalid_weights, 1U, 20803, 16U, 64U, weighted_round_bound),
                "non-positive authenticated ExpMap weight was accepted");
        require(!zkdl_gat::static_table_bounds::positive_weighted_round_abs_bound(
                    high_degree_weights, 1U, std::numeric_limits<std::int64_t>::max(),
                    std::numeric_limits<std::uint64_t>::max(), 64U, weighted_round_bound),
                "weighted quotient arithmetic overflow was accepted");
        std::cout << "STATIC_TABLE_BOUNDS_CHECK_OK\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
