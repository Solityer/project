#include "gat_ezkl/graph.hpp"

#include <string>

namespace gat_ezkl {

bool edge_order_is_gid_then_dst_stable(const NormalizedGraph& graph, std::string* reason) {
    for (std::size_t i = 1; i < graph.edges.size(); ++i) {
        const auto& prev = graph.edges[i - 1];
        const auto& curr = graph.edges[i];
        if (prev.graph_id > curr.graph_id) {
            if (reason) *reason = "edge graph_id order decreases at edge " + std::to_string(i);
            return false;
        }
        if (prev.graph_id == curr.graph_id && prev.dst > curr.dst) {
            if (reason) *reason = "edge dst order decreases inside graph block at edge " + std::to_string(i);
            return false;
        }
    }
    return true;
}

bool graph_blocks_are_valid(const NormalizedGraph& graph, std::string* reason) {
    if (graph.node_ptr.size() != graph.graph_count + 1) {
        if (reason) *reason = "node_ptr size does not equal graph_count+1";
        return false;
    }
    if (graph.edge_ptr.size() != graph.graph_count + 1) {
        if (reason) *reason = "edge_ptr size does not equal graph_count+1";
        return false;
    }
    if (graph.node_ptr.empty() || graph.node_ptr.front() != 0 || graph.node_ptr.back() != graph.node_count) {
        if (reason) *reason = "node_ptr does not span all nodes";
        return false;
    }
    if (graph.edge_ptr.empty() || graph.edge_ptr.front() != 0 || graph.edge_ptr.back() != graph.edge_count) {
        if (reason) *reason = "edge_ptr does not span all edges";
        return false;
    }
    for (const auto& edge : graph.edges) {
        if (edge.graph_id >= graph.graph_count) {
            if (reason) *reason = "edge graph_id out of range";
            return false;
        }
        if (edge.src >= graph.node_count || edge.dst >= graph.node_count) {
            if (reason) *reason = "edge endpoint out of range";
            return false;
        }
        const auto g = edge.graph_id;
        if (edge.src < graph.node_ptr[g] || edge.src >= graph.node_ptr[g + 1]
            || edge.dst < graph.node_ptr[g] || edge.dst >= graph.node_ptr[g + 1]) {
            if (reason) *reason = "edge endpoint crosses graph block";
            return false;
        }
    }
    return true;
}

}  // namespace gat_ezkl
