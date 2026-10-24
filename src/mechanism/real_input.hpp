#pragma once

#include <string_view>
#include <vector>

#include "gatzk/mechanism/internal.hpp"

namespace gatzk::mechanism::internal
{
    struct RealDatasetCatalog
    {
        std::string_view dataset;
        std::string_view input_preprocess_id;
        std::string_view srs_id;
        std::size_t node_domain_size;
        std::size_t edge_domain_size;
    };

    const RealDatasetCatalog *registered_real_dataset_catalog(std::string_view dataset) noexcept;
    bool valid_real_graph_count(std::string_view dataset, std::size_t graphs) noexcept;
    bool valid_real_component_selection(std::string_view dataset, const std::vector<std::size_t> &components) noexcept;

    RealMechanismInput read_real_mechanism_input(const std::filesystem::path &path);
    std::vector<std::uint8_t> encode_real_mechanism_input(const RealMechanismInput &input);
    std::string real_mechanism_relation_digest(const RealMechanismInput &input, Mechanism mechanism, std::size_t width);
    std::string real_mechanism_witness_digest(const RealMechanismInput &input);
}
