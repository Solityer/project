#pragma once

#include "gatzk/data/dataset.hpp"
#include "gatzk/util/config.hpp"

namespace gatzk::data {

GraphDataset load_dataset(const util::AppConfig& config);
LocalGraph normalize_graph_input(const GraphDataset& dataset, const util::AppConfig& config);

}  // namespace gatzk::data
