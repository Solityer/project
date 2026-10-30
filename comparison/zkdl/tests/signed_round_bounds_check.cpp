#include "gat_adapter/signed_round_bounds.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

zkdl_gat::IntegerBoundFamily family(std::string id, std::uint64_t layer, std::uint64_t head,
                                    bool output, std::int64_t dividend, std::int64_t auxiliary,
                                    std::int64_t divisor) {
    zkdl_gat::IntegerBoundFamily result;
    result.family_id = std::move(id);
    result.layer = layer;
    result.head = head;
    result.output = output;
    result.max_dividend_abs = dividend;
    result.max_aux_abs = auxiliary;
    result.max_divisor = divisor;
    return result;
}

template <typename Function>
bool rejected(Function &&function) {
    try {
        function();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    try {
        zkdl_gat::Config config;
        config.architecture.hidden_dims = {8U};
        config.architecture.output_heads = 2U;
        config.range_bits = 1U;
        config.quantization.hidden_aggregate_pre = {16U};
        config.quantization.hidden_h_prime = {{64U}};
        config.quantization.output_y_prime = {8U, 16U};
        config.quantization.output_pre = 16U;
        config.quantization.output_head = 8U;

        zkdl_gat::IntegerBoundLedger ledger;
        ledger.authenticated = true;
        ledger.families = {
            family("hidden/layer0/head0", 0U, 0U, false, 1001, 4000, 7),
            family("output/head0", 1U, 0U, true, 8193, 20000, 11),
            family("output/head1", 1U, 1U, true, 7001, 20000, 13),
        };

        const auto hidden = zkdl_gat::signed_round_bounds::derive(config, ledger, "hidden/layer0/head0/signed_round_div");
        if (hidden.numerator_scale != 16 || !hidden.binds_raw_denominator || hidden.denominator_scale != 64 || hidden.a_bits != 10U ||
            hidden.q_max != 16 || hidden.q_bits != 5U || hidden.r_max != 13 || hidden.r_bits != 4U ||
            hidden.divisor_bits != 3U) {
            std::cerr << "hidden signed-round widths do not follow its authenticated family bounds\n";
            return 1;
        }

        const auto output = zkdl_gat::signed_round_bounds::derive(config, ledger, "output/head0/signed_round_div");
        if (output.numerator_scale != 16 || !output.binds_raw_denominator || output.denominator_scale != 8 || output.a_bits != 14U ||
            output.q_max != 1025 || output.q_bits != 11U || output.r_bits != 5U || output.divisor_bits != 4U) {
            std::cerr << "output signed-round widths do not follow its authenticated family bounds\n";
            return 1;
        }

        const auto alignment = zkdl_gat::signed_round_bounds::derive(config, ledger, "output/head1/output_scale_alignment");
        if (alignment.numerator_scale != 8 || alignment.binds_raw_denominator || alignment.denominator_scale != 1 || alignment.a_max != 7001 ||
            alignment.divisor_max != 16 || alignment.q_max != 438 || alignment.q_bits != 9U || alignment.r_max != 31) {
            std::cerr << "output alignment widths do not use the authenticated output-family bounds\n";
            return 1;
        }

        const auto average = zkdl_gat::signed_round_bounds::derive(config, ledger, "final_average");
        if (average.numerator_scale != 1 || average.binds_raw_denominator || average.divisor_max != 2 || average.a_max != 8193 ||
            average.q_max != 4097 || average.q_bits != 13U || average.r_max != 3) {
            std::cerr << "final-average widths do not use the authenticated output-family bounds\n";
            return 1;
        }

        auto unauthenticated = ledger;
        unauthenticated.authenticated = false;
        auto invalid_scale = config;
        invalid_scale.quantization.hidden_h_prime[0][0] = 0U;
        auto invalid_numerator_scale = config;
        invalid_numerator_scale.quantization.hidden_aggregate_pre[0] = 0U;
        auto too_small_aux = ledger;
        too_small_aux.families[0].max_aux_abs = 12;
        auto malformed_family = ledger;
        malformed_family.families[0].max_divisor = 0;
        if (!rejected([&] { (void)zkdl_gat::signed_round_bounds::derive(config, unauthenticated, "hidden/layer0/head0/signed_round_div"); }) ||
            !rejected([&] { (void)zkdl_gat::signed_round_bounds::derive(invalid_scale, ledger, "hidden/layer0/head0/signed_round_div"); }) ||
            !rejected([&] { (void)zkdl_gat::signed_round_bounds::derive(invalid_numerator_scale, ledger, "hidden/layer0/head0/signed_round_div"); }) ||
            !rejected([&] { (void)zkdl_gat::signed_round_bounds::derive(config, too_small_aux, "hidden/layer0/head0/signed_round_div"); }) ||
            !rejected([&] { (void)zkdl_gat::signed_round_bounds::derive(config, malformed_family, "hidden/layer0/head0/signed_round_div"); })) {
            std::cerr << "invalid signed-round ledger/scale was not rejected\n";
            return 1;
        }
        std::cout << "signed_round_bounds=PASS numerator_scale=hidden:16,output:16,alignment:8,average:1 malformed=rejected\n";
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
