#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::data {

struct Edge {
    std::size_t src = 0;
    std::size_t dst = 0;
};

struct GraphDataset {
    std::string name;
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::vector<std::vector<algebra::FieldElement>> features;
    std::vector<int> labels;
    std::vector<Edge> edges;
};

struct LocalGraph {
    std::string name;
    std::size_t num_nodes = 0;
    std::size_t num_features = 0;
    std::size_t num_classes = 0;
    std::vector<std::size_t> absolute_ids;
    std::vector<std::vector<algebra::FieldElement>> features;
    std::vector<int> labels;
    std::vector<Edge> edges;
};

}  // namespace gatzk::data
