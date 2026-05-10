#pragma once

#include <cstdint>
#include <string>

#include "gat_ezkl/quant.hpp"

namespace gat_ezkl 
{
    // 分数边界配置：用于约束 GAT 中间激活值的量化范围，确保不会溢出
    struct ScoreBounds 
    {
        bool inferred = false;  // 从量化配置自动推导出边界
        std::size_t range_bits = 0;  // 分数范围（位宽）
        std::int64_t score_lower_bound = 0;  // 合法分数下限（量化域，整数）
        std::int64_t score_upper_bound = 0;   // 合法分数上限
        std::int64_t invalid_score_sentinel = 0;  // 哨兵值，标记无效分数（如 masked attention）
        bool sentinel_bound_verified = false;   // 哨兵值是否在安全范围内
        std::string blocker;  // 如果边界有问题，记录阻塞原因
        std::string explanation; // 边界推导说明
    };

    // 根据量化配置和指定位宽，推断分数的合理取值范围
    ScoreBounds infer_score_bounds_from_quant_cfg(const QuantConfig& quant, std::size_t range_bits);
    ScoreBounds choose_invalid_score_sentinel(const ScoreBounds& bounds);
    bool verify_sentinel_bound(const ScoreBounds& bounds, std::string* reason);
}