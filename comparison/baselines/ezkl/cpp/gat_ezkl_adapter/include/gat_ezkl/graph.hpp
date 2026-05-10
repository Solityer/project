#pragma once

#include <string>

#include "gat_ezkl/dataset.hpp"

namespace gat_ezkl
{
    // 校验边的排序规则是否满足：先按 graph_id 升序，同一子图内按 dst 节点索引升序，排序不一致会导致证明失败
    // reason 输出失败原因，返回 true 表示校验通过
    bool edge_order_is_gid_then_dst_stable(const NormalizedGraph &graph, std::string *reason);

    // 校验 CSR 分块结构的合法性：node_ptr/edge_ptr 长度匹配 graph_count，
    // 块内区间不重叠、不越界，且 src/dst 索引在合法范围内，用于在进入 ZK 电路前确保图数据不会被错误解析
    bool graph_blocks_are_valid(const NormalizedGraph &graph, std::string *reason);
}