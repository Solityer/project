#pragma once
// 本文件定义 PSQ / state-machine 相关的前缀状态构造，
// 主要服务按目标节点分组的 softmax 链与最大值状态约束。

#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol {

// 构造按 `q_new` 断开的前缀和状态，是 PSQ 链的基础列。
std::vector<algebra::FieldElement> build_group_prefix_state(
    const std::vector<algebra::FieldElement>& values,
    const std::vector<algebra::FieldElement>& q_new,
    bool enable_single_graph_selector_shortcut = false);

// 构造 `c_max / s_max` 类状态列，单图时允许使用等价的快捷递推。
std::vector<algebra::FieldElement> build_max_counter_state(
    const std::vector<algebra::FieldElement>& s_max,
    const std::vector<algebra::FieldElement>& q_new,
    bool enable_single_graph_selector_shortcut = false);

}  // namespace gatzk::protocol
