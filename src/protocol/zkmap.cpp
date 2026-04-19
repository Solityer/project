#include "gatzk/protocol/zkmap.hpp"

#include <stdexcept>

// 本文件实现 zkMaP 风格的绑定累加轨迹。
// 这些辅助列会进入外点绑定关系与相应的 KZG opening 检查。

namespace gatzk::protocol {

// 构造 `a_values` 与 `b_values` 的前缀乘积和轨迹。
ZkMapAccumulatorTrace build_zkmap_trace(
    const std::vector<algebra::FieldElement>& a_values,
    const std::vector<algebra::FieldElement>& b_values,
    std::size_t domain_size) {
    if (a_values.size() != b_values.size()) {
        throw std::runtime_error("zkMaP input size mismatch");
    }

    ZkMapAccumulatorTrace trace;
    trace.a_values.assign(domain_size, algebra::FieldElement::zero());
    trace.b_values.assign(domain_size, algebra::FieldElement::zero());
    trace.accumulator.assign(domain_size, algebra::FieldElement::zero());
    // 先把有效区的原始列拷进去，并顺手求出最终要绑定的整体内积 `mu`。
    for (std::size_t i = 0; i < a_values.size(); ++i) {
        trace.a_values[i] = a_values[i];
        trace.b_values[i] = b_values[i];
        trace.mu += a_values[i] * b_values[i];
    }
    // 前缀累加从第 1 格开始推进，位置 0 固定为 0，方便 quotient 写首项边界条件。
    for (std::size_t i = 0; i < a_values.size(); ++i) {
        trace.accumulator[i + 1] = trace.accumulator[i] + a_values[i] * b_values[i];
    }
    for (std::size_t i = a_values.size() + 1; i < domain_size; ++i) {
        trace.accumulator[i] = trace.accumulator[a_values.size()];
    }
    return trace;
}

}  // namespace gatzk::protocol
