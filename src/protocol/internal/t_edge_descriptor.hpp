#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gatzk/algebra/field.hpp"

namespace gatzk::protocol::internal
{

    struct TEdgeDescriptorShape
    {
        std::vector<std::size_t> hidden_head_counts;
        std::size_t output_head_count = 0;
    };

    struct TEdgeDerivedLinearSource
    {
        std::string label;
        algebra::FieldElement scale = algebra::FieldElement::zero();
    };

    struct TEdgeDerivedLinearDescriptor
    {
        std::string target_label;

        std::vector<algebra::FieldElement> target_zero_domain_blinder;
        std::vector<TEdgeDerivedLinearSource> sources;
    };

    enum class TEdgeDerivedLinearFlattenStatus
    {
        Success,
        UnknownSource,
        Cycle,
        DepthExceeded,
    };

    struct TEdgeDerivedLinearFlattenResult
    {
        TEdgeDerivedLinearFlattenStatus status = TEdgeDerivedLinearFlattenStatus::Success;
        std::vector<TEdgeDerivedLinearSource> sources;
    };

    std::optional<TEdgeDerivedLinearDescriptor> make_t_edge_derived_linear_descriptor(std::string_view name, const TEdgeDescriptorShape &shape,
                                                                                      const std::map<std::string, algebra::FieldElement> &challenges,
                                                                                      const std::function<bool(std::string_view)> &collision_predicate = {});

    using TEdgeDescriptorLookup = std::function<std::optional<TEdgeDerivedLinearDescriptor>(std::string_view)>;
    using TEdgeRegisteredSourcePredicate = std::function<bool(std::string_view)>;

    TEdgeDerivedLinearFlattenResult flatten_t_edge_derived_linear_sources(const TEdgeDerivedLinearDescriptor &descriptor, const TEdgeDescriptorLookup &lookup,
                                                                          const TEdgeRegisteredSourcePredicate &is_registered_source);

}
