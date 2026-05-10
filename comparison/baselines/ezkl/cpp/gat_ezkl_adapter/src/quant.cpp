#include "gat_ezkl/quant.hpp"

#include <cmath>

namespace gat_ezkl 
{

    QuantConfig quant_config_from_id(const std::string& quant_cfg_id, std::size_t range_bits) 
    {
        QuantConfig out;
        out.quant_cfg_id = quant_cfg_id.empty() ? ("range_bits=" + std::to_string(range_bits)) : quant_cfg_id;
        return out;
    }

    // 浮点数量化为 int64：乘以 scale 后四舍五入
    std::int64_t quantize_to_i64(double value, double scale) 
    {
        const auto scaled = value * scale;
        return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
    }
}