#pragma once

#include <cstdint>
#include <string>

namespace gat_ezkl 
{
    // 量化参数配置：控制浮点转定点整数的规则
    struct QuantConfig 
    {
        std::string quant_cfg_id;
        double input_scale = 16.0;  // 激活值的量化缩放因子
        double weight_scale = 16.0;  // 权重的量化缩放因子
        std::string rounding = "half_away_from_zero";  // 舍入策略（四舍五入）
        std::string signed_encoding = "twos-complement-field-representative";  // 有符号数编码方式
    };

    // 根据配置 ID 和量化位宽，生成完整的量化配置
    QuantConfig quant_config_from_id(const std::string& quant_cfg_id, std::size_t range_bits);

    // 将 double 按规模缩放后转为 int64（嵌入舍入规则）
    std::int64_t quantize_to_i64(double value, double scale);
}