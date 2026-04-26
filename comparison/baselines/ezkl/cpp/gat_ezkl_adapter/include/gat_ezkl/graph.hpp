#pragma once

#include <string>

#include "gat_ezkl/dataset.hpp"

namespace gat_ezkl {

bool edge_order_is_gid_then_dst_stable(const NormalizedGraph& graph, std::string* reason);
bool graph_blocks_are_valid(const NormalizedGraph& graph, std::string* reason);

}  // namespace gat_ezkl
