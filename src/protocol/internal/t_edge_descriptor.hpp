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

    // Execution-only shape information used to recognize the canonical
    // layer/head prefixes.  It deliberately does not expose a proving
    // context, proof schema, or any installed protocol API.
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

        // The target blinder is owned and filled by the production materializer
        // wrapper.  The pure mapper leaves it empty; keeping the field here
        // lets the wrapper carry the same descriptor through its blinder
        // correction path without adding a public protocol object.
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

    // The challenge map is the already replayed canonical challenge set.  The
    // mapper only derives the challenge namespace from the shape coordinates;
    // it never constructs a transcript or changes challenge values.
    std::optional<TEdgeDerivedLinearDescriptor> make_t_edge_derived_linear_descriptor(std::string_view name, const TEdgeDescriptorShape &shape,
                                                                                      const std::map<std::string, algebra::FieldElement> &challenges,
                                                                                      const std::function<bool(std::string_view)> &collision_predicate = {});

    using TEdgeDescriptorLookup = std::function<std::optional<TEdgeDerivedLinearDescriptor>(std::string_view)>;
    using TEdgeRegisteredSourcePredicate = std::function<bool(std::string_view)>;

    // Flatten nested derived views into registered leaves.  The callbacks are
    // deliberately supplied by the caller so trace/spill ownership remains in
    // the production wrapper and cannot leak into this internal pure helper.
    TEdgeDerivedLinearFlattenResult flatten_t_edge_derived_linear_sources(const TEdgeDerivedLinearDescriptor &descriptor, const TEdgeDescriptorLookup &lookup,
                                                                          const TEdgeRegisteredSourcePredicate &is_registered_source);

} // namespace gatzk::protocol::internal
