#pragma once

#include "gatzk/data/dataset.hpp"
#include "gatzk/util/config.hpp"

namespace gatzk::data {

GraphDataset load_dataset(const util::AppConfig& config);
LocalGraph normalize_graph_input(const GraphDataset& dataset, const util::AppConfig& config);
LocalGraph extract_local_subgraph(const GraphDataset& dataset, std::size_t center_node, std::size_t local_nodes);

}  // namespace gatzk::data
