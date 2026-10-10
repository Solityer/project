#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "gatzk/data/dataset.hpp"
#include "gatzk/util/config.hpp"

namespace gatzk::data
{
    std::int64_t quantize_raw_at_scale(double value, std::uint64_t scale);
    algebra::FieldElement quantize_field_at_scale(double value, std::uint64_t scale);

    std::vector<std::vector<double>> preprocess_input_features(const std::vector<std::vector<double>> &raw_features, std::string_view dataset,
                                                               std::string_view input_preprocess_id);

    GraphDataset load_dataset(const util::AppConfig &config);
    LocalGraph normalize_graph_input(const GraphDataset &dataset, const util::AppConfig &config);
    LocalGraph normalize_graph_input(const GraphDataset &dataset, const util::AppConfig &config,
                                    const std::vector<std::size_t> &selected_components);
    void validate_input_feature_views(const GraphDataset &dataset, const LocalGraph &local, std::uint64_t input_scale, std::string_view input_preprocess_id);
    void validate_normalized_graph_input(const LocalGraph &local);
}
