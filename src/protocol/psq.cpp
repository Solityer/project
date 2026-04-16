#include "gatzk/protocol/psq.hpp"

#include <stdexcept>

// 本文件实现 PSQ / state-machine 前缀状态列。
// 单图快捷路径只是一种等价执行策略，不改变 formal 递推含义。

namespace gatzk::protocol {

namespace {

// 统一的前缀状态递推，`q_new` 为 1 时从新组开始，为 0 时延续上一组。
std::vector<algebra::FieldElement> build_selector_prefix_state(
    const std::vector<algebra::FieldElement>& values,
    const std::vector<algebra::FieldElement>& q_new,
    bool enable_single_graph_selector_shortcut) {
    if (values.size() != q_new.size()) {
        throw std::runtime_error("prefix state input size mismatch");
    }
    if (values.empty()) {
        return {};
    }
    std::vector<algebra::FieldElement> out(values.size(), algebra::FieldElement::zero());
    out[0] = values[0];
    const auto one = algebra::FieldElement::one();
    for (std::size_t i = 1; i < values.size(); ++i) {
        const auto& selector = q_new[i];
        const auto& value = values[i];
        if (enable_single_graph_selector_shortcut) {
            if (selector.is_zero()) {
                out[i] = out[i - 1] + value;
                continue;
            }
            if (selector == one) {
                out[i] = value;
                continue;
            }
        }
        out[i] = selector * value + (one - selector) * (out[i - 1] + value);
    }
    return out;
}

}  // namespace

// PSQ 链直接复用同一递推模板。
std::vector<algebra::FieldElement> build_group_prefix_state(
    const std::vector<algebra::FieldElement>& values,
    const std::vector<algebra::FieldElement>& q_new,
    bool enable_single_graph_selector_shortcut) {
    return build_selector_prefix_state(values, q_new, enable_single_graph_selector_shortcut);
}

// `c_max / s_max` 状态机也沿用相同的分组递推骨架。
std::vector<algebra::FieldElement> build_max_counter_state(
    const std::vector<algebra::FieldElement>& s_max,
    const std::vector<algebra::FieldElement>& q_new,
    bool enable_single_graph_selector_shortcut) {
    if (s_max.size() != q_new.size()) {
        throw std::runtime_error("max counter input size mismatch");
    }
    return build_selector_prefix_state(s_max, q_new, enable_single_graph_selector_shortcut);
}
}
