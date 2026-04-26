#pragma once

#include <cstdint>
#include <string>

namespace gat_ezkl {

struct QuantConfig {
    std::string quant_cfg_id;
    double input_scale = 16.0;
    double weight_scale = 16.0;
    std::string rounding = "half_away_from_zero";
    std::string signed_encoding = "twos-complement-field-representative";
};

QuantConfig quant_config_from_id(const std::string& quant_cfg_id, std::size_t range_bits);
std::int64_t quantize_to_i64(double value, double scale);

}  // namespace gat_ezkl
