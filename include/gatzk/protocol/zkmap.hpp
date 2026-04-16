#pragma once
// 本文件定义 zkMaP 风格的乘积累加轨迹，
// 用于把线性绑定关系转成可承诺、可 opening 的辅助列。

#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol {

// 绑定轨迹使用的标准列布局。
struct ZkMapAccumulatorTrace {
    std::vector<algebra::FieldElement> a_values;
    std::vector<algebra::FieldElement> b_values;
    std::vector<algebra::FieldElement> accumulator;
    algebra::FieldElement mu = algebra::FieldElement::zero();
};

// 构造 `a · b` 的前缀累加轨迹，供 trace / verify 在外点绑定关系中复用。
ZkMapAccumulatorTrace build_zkmap_trace(
    const std::vector<algebra::FieldElement>& a_values,
    const std::vector<algebra::FieldElement>& b_values,
    std::size_t domain_size);

}  // namespace gatzk::protocol
