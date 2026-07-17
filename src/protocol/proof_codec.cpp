#include "gatzk/protocol/proof_codec.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"

namespace gatzk::protocol
{
    namespace
    {
        constexpr std::array<std::uint8_t, 8> kMagic{'Z', 'K', 'G', 'A', 'T', 'P', 'F', '1'};
        constexpr std::uint64_t kWireVersion = 2;
        constexpr std::uint64_t kMaxContainerItems = 1ULL << 24;
        constexpr std::uint64_t kMaxStringBytes = 1ULL << 20;

        class Writer
        {
        public:
            void raw(const std::uint8_t* data, std::size_t size)
            {
                bytes_.insert(bytes_.end(), data, data + static_cast<std::ptrdiff_t>(size));
            }

            void u64(std::uint64_t value)
            {
                for (std::size_t i = 0; i < 8; ++i)
                {
                    bytes_.push_back(static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU));
                }
            }

            void string(const std::string& value)
            {
                u64(value.size());
                raw(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
            }

            void scalar(const algebra::FieldElement& value)
            {
                std::array<std::uint8_t, 32> encoded{};
                value.write_little_endian(encoded.data(), encoded.size());
                raw(encoded.data(), encoded.size());
            }

            void point(const crypto::G1Point& point)
            {
                const auto encoded = crypto::serialize(point);
                u64(encoded.size());
                raw(encoded.data(), encoded.size());
            }

            void block(const std::string& tag, Writer payload)
            {
                string(tag);
                u64(payload.bytes_.size());
                raw(payload.bytes_.data(), payload.bytes_.size());
            }

            std::vector<std::uint8_t> take() { return std::move(bytes_); }

        private:
            std::vector<std::uint8_t> bytes_;
        };

        class Reader
        {
        public:
            Reader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

            std::uint64_t u64()
            {
                require(8);
                std::uint64_t value = 0;
                for (std::size_t i = 0; i < 8; ++i)
                {
                    value |= static_cast<std::uint64_t>(data_[offset_++]) << (8U * i);
                }
                return value;
            }

            std::size_t count(const char* label)
            {
                const auto value = u64();
                if (value > kMaxContainerItems || value > std::numeric_limits<std::size_t>::max())
                {
                    throw std::runtime_error(std::string("proof parser invalid count: ") + label);
                }
                return static_cast<std::size_t>(value);
            }

            std::string string()
            {
                const auto length = u64();
                if (length > kMaxStringBytes || length > remaining())
                {
                    throw std::runtime_error("proof parser invalid string length");
                }
                const auto* begin = reinterpret_cast<const char*>(data_ + offset_);
                offset_ += static_cast<std::size_t>(length);
                return std::string(begin, begin + static_cast<std::ptrdiff_t>(length));
            }

            algebra::FieldElement scalar()
            {
                constexpr std::size_t kScalarBytes = 32;
                require(kScalarBytes);
                const auto* begin = data_ + offset_;
                auto value = algebra::FieldElement::from_little_endian_mod(begin, kScalarBytes);
                std::array<std::uint8_t, kScalarBytes> canonical{};
                value.write_little_endian(canonical.data(), canonical.size());
                for (std::size_t i = 0; i < canonical.size(); ++i)
                {
                    if (canonical[i] != begin[i])
                    {
                        throw std::runtime_error("proof parser rejected non-canonical field encoding");
                    }
                }
                offset_ += kScalarBytes;
                return value;
            }

            crypto::G1Point point()
            {
                const auto length = u64();
                if (length == 0 || length > 256 || length > remaining())
                {
                    throw std::runtime_error("proof parser invalid G1 length");
                }
                std::vector<std::uint8_t> encoded(
                    data_ + offset_, data_ + offset_ + static_cast<std::ptrdiff_t>(length));
                offset_ += static_cast<std::size_t>(length);
                auto point = crypto::deserialize_g1(encoded);
                if (crypto::serialize(point) != encoded)
                {
                    throw std::runtime_error("proof parser rejected non-canonical G1 encoding");
                }
                return point;
            }

            Reader block(const std::string& expected_tag)
            {
                if (string() != expected_tag)
                {
                    throw std::runtime_error("proof parser block order/tag mismatch: " + expected_tag);
                }
                const auto length = u64();
                if (length > remaining())
                {
                    throw std::runtime_error("proof parser truncated block: " + expected_tag);
                }
                Reader out(data_ + offset_, static_cast<std::size_t>(length));
                offset_ += static_cast<std::size_t>(length);
                return out;
            }

            void finish(const char* label) const
            {
                if (offset_ != size_)
                {
                    throw std::runtime_error(std::string("proof parser trailing bytes in ") + label);
                }
            }

            std::size_t remaining() const { return size_ - offset_; }

        private:
            void require(std::size_t size)
            {
                if (size > remaining())
                {
                    throw std::runtime_error("proof parser truncated payload");
                }
            }

            const std::uint8_t* data_ = nullptr;
            std::size_t size_ = 0;
            std::size_t offset_ = 0;
        };

        template <typename Fn>
        void metadata_fields(PublicMetadata& metadata, Fn&& fn)
        {
            fn(metadata.protocol_name); fn(metadata.proof_nonce); fn(metadata.proof_task);
            fn(metadata.srs_id); fn(metadata.protocol_schema_id); fn(metadata.instance_schema_hash);
            fn(metadata.srs_type); fn(metadata.srs_degree_bound); fn(metadata.kzg_assumption_profile);
            fn(metadata.privacy_policy); fn(metadata.input_privacy); fn(metadata.output_mode);
            fn(metadata.opening_mode); fn(metadata.binding_mode); fn(metadata.model_arch);
            fn(metadata.model_params); fn(metadata.static_tables); fn(metadata.quant_config);
            fn(metadata.domain_cfg); fn(metadata.dim_cfg); fn(metadata.encoding);
            fn(metadata.padding_rule); fn(metadata.degree_bound); fn(metadata.security_budget);
            fn(metadata.public_output);
        }

        void write_opening(Writer& writer, const crypto::OpeningProof& proof)
        {
            writer.u64(proof.witnesses.size());
            for (const auto& witness : proof.witnesses) writer.point(witness);
        }

        crypto::OpeningProof read_opening(Reader& reader)
        {
            crypto::OpeningProof proof;
            const auto count = reader.count("opening witnesses");
            proof.witnesses.reserve(count);
            for (std::size_t i = 0; i < count; ++i) proof.witnesses.push_back(reader.point());
            return proof;
        }

        PayloadValueKind read_kind(Reader& reader)
        {
            const auto raw = reader.u64();
            if (raw > static_cast<std::uint64_t>(PayloadValueKind::MaskedPrivateValue))
            {
                throw std::runtime_error("proof parser invalid payload kind");
            }
            return static_cast<PayloadValueKind>(raw);
        }

        void write_named_commitments(
            Writer& writer,
            const std::vector<std::pair<std::string, crypto::Commitment>>& commitments)
        {
            writer.u64(commitments.size());
            for (const auto& [name, commitment] : commitments)
            {
                writer.string(name);
                writer.point(commitment.point);
            }
        }

        std::vector<std::pair<std::string, crypto::Commitment>> read_named_commitments(Reader& reader)
        {
            std::vector<std::pair<std::string, crypto::Commitment>> out;
            const auto count = reader.count("commitments");
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                auto name = reader.string();
                crypto::Commitment commitment;
                commitment.name = name;
                commitment.point = reader.point();
                out.push_back({std::move(name), std::move(commitment)});
            }
            return out;
        }

        void write_optional_index(Writer& writer, const std::optional<std::size_t>& value)
        {
            writer.u64(value.has_value() ? 1 : 0);
            if (value.has_value()) writer.u64(*value);
        }

        std::optional<std::size_t> read_optional_index(Reader& reader, const char* label)
        {
            const auto present = reader.u64();
            if (present > 1) throw std::runtime_error(std::string("proof parser invalid optional flag: ") + label);
            if (present == 0) return std::nullopt;
            const auto value = reader.u64();
            if (value > std::numeric_limits<std::size_t>::max())
                throw std::runtime_error(std::string("proof parser invalid optional index: ") + label);
            return static_cast<std::size_t>(value);
        }

        void write_string_vector(Writer& writer, const std::vector<std::string>& values)
        {
            writer.u64(values.size());
            for (const auto& value : values) writer.string(value);
        }

        std::vector<std::string> read_string_vector(Reader& reader, const char* label)
        {
            std::vector<std::string> out;
            const auto count = reader.count(label);
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i) out.push_back(reader.string());
            return out;
        }

        void write_size_vector(Writer& writer, const std::vector<std::size_t>& values)
        {
            writer.u64(values.size());
            for (const auto value : values) writer.u64(value);
        }

        std::vector<std::size_t> read_size_vector(Reader& reader, const char* label)
        {
            std::vector<std::size_t> out;
            const auto count = reader.count(label);
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                const auto value = reader.u64();
                if (value > std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error(std::string("proof parser invalid size: ") + label);
                out.push_back(static_cast<std::size_t>(value));
            }
            return out;
        }

        template <typename NamedRange>
        void require_unique_names(const NamedRange& values, const char* label)
        {
            std::unordered_set<std::string> names;
            names.reserve(values.size());
            for (const auto& item : values)
            {
                if (item.first.empty() || !names.insert(item.first).second)
                {
                    throw std::runtime_error(std::string("proof parser duplicate/empty name in ") + label);
                }
            }
        }

        void require_canonical_point_group_shape(
            const std::vector<algebra::FieldElement>& points,
            const crypto::OpeningProof& opening,
            const char* label)
        {
            std::unordered_set<std::string> unique_points;
            for (const auto& point : points) unique_points.insert(point.to_string());
            if (opening.witnesses.size() != unique_points.size())
            {
                throw std::runtime_error(std::string("proof parser non-canonical point-group witness count in ") + label);
            }
        }

        void require_exact_pi_bind_shape(
            const ProtocolContext& context,
            const Proof& proof)
        {
            const auto expected_meta = build_binding_relation_metadata(context);
            if (proof.pi_bind.subproofs.size() != expected_meta.size())
                throw std::runtime_error(
                    "proof parser Pi_bind instance subproof count mismatch");

            std::size_t global_hidden_head = 0;
            for (std::size_t index = 0; index < expected_meta.size(); ++index)
            {
                const auto& subproof = proof.pi_bind.subproofs[index];
                const auto& meta = expected_meta[index];
                if (!(subproof.relation_meta == meta))
                    throw std::runtime_error(
                        "proof parser Pi_bind instance RelationMeta mismatch");

                std::vector<std::pair<std::string, std::string>> expected_models;
                if (meta.object_kind == "hidden")
                {
                    const auto layer = meta.layer_id.value() - 1;
                    const auto width =
                        context.config.hidden_profile.at(layer).head_dim;
                    for (std::size_t column = 0; column < width; ++column)
                        expected_models.push_back({
                            hidden_weight_column_label(global_hidden_head, column),
                            context.domains.in->name});
                    expected_models.push_back({
                        hidden_src_label(global_hidden_head),
                        context.domains.d->name});
                    expected_models.push_back({
                        hidden_dst_label(global_hidden_head),
                        context.domains.d->name});
                    ++global_hidden_head;
                }
                else if (meta.object_kind == "output")
                {
                    const auto head = meta.head_id.value();
                    for (std::size_t column = 0;
                         column < context.local.num_classes; ++column)
                        expected_models.push_back({
                            output_weight_column_label(head, column),
                            context.domains.cat->name});
                    expected_models.push_back({
                        output_src_label(head), context.domains.c->name});
                    expected_models.push_back({
                        output_dst_label(head), context.domains.c->name});
                    expected_models.push_back({
                        output_bias_label(head), context.domains.c->name});
                }
                if (subproof.model_reblinds.size() != expected_models.size())
                    throw std::runtime_error(
                        "proof parser Pi_bind model-reblind count mismatch");
                for (std::size_t model = 0;
                     model < expected_models.size(); ++model)
                {
                    const auto& [anchor, domain] = expected_models[model];
                    const auto& actual = subproof.model_reblinds[model];
                    if (actual.object_id != anchor
                        || actual.domain != domain
                        || actual.vk_model_anchor.object_id != anchor
                        || actual.proof_local_commitment.name
                            != model_proof_local_label(anchor)
                        || actual.blinder_commitment.name
                            != model_reblind_witness_label(anchor)
                        || actual.blinder_commitment.point == crypto::g1_zero())
                    {
                        throw std::runtime_error(
                            "proof parser Pi_bind model-reblind shape mismatch");
                    }
                }

                std::vector<std::string> expected_canonical_references;
                std::vector<std::string> expected_canonical_commitments;
                std::vector<std::tuple<
                    std::string, std::string, std::size_t,
                    PayloadValueKind>> expected_canonical_evaluations;
                for (const auto& matrix : meta.canonical_matrices)
                {
                    expected_canonical_references.push_back(
                        matrix.matrix_object);
                    for (const auto& column : matrix.column_objects)
                    {
                        expected_canonical_commitments.push_back(column);
                        expected_canonical_evaluations.push_back({
                            column, "N", 2,
                            PayloadValueKind::MaskedPrivateValue});
                    }
                    expected_canonical_commitments.push_back(
                        matrix.compressed_object);
                    expected_canonical_evaluations.push_back({
                        matrix.compressed_object, "N", 2,
                        PayloadValueKind::MaskedPrivateValue});
                }
                const auto canonical_references_match = [&]
                {
                    return subproof.view_references.size()
                            >= expected_canonical_references.size()
                        && std::equal(
                            subproof.view_references.begin(),
                            subproof.view_references.begin()
                                + static_cast<std::ptrdiff_t>(
                                    expected_canonical_references.size()),
                            expected_canonical_references.begin(),
                            [](const MetadataReference& actual,
                               const std::string& expected)
                            { return actual.object_id == expected; });
                };
                const auto canonical_commitments_match = [&]
                {
                    return subproof.view_commitments.size()
                            == expected_canonical_commitments.size()
                        && std::equal(
                            subproof.view_commitments.begin(),
                            subproof.view_commitments.end(),
                            expected_canonical_commitments.begin(),
                            [](const auto& actual,
                               const std::string& expected)
                            { return actual.first == expected; });
                };
                std::vector<std::string> expected_aggregation_commitments;
                std::vector<std::tuple<
                    std::string, std::string, std::size_t,
                    PayloadValueKind>> expected_aggregation_relation_evaluations;
                std::vector<std::string> expected_route_commitments;
                for (const auto& route : meta.route_terminal_relations)
                {
                    expected_route_commitments.push_back(
                        route.shifted_accumulator_object);
                    expected_aggregation_relation_evaluations.push_back({
                        route.shifted_accumulator_object + ":u_rel",
                        "relation", 1,
                        PayloadValueKind::MaskedPrivateValue});
                }
                std::vector<std::tuple<
                    std::string, std::string, std::size_t,
                    PayloadValueKind>> expected_aggregation_domain_evaluations;
                std::vector<std::string> expected_aggregation_openings;
                const auto append_aggregation_opening = [&](std::string value)
                {
                    if (std::find(
                            expected_aggregation_openings.begin(),
                            expected_aggregation_openings.end(), value)
                        == expected_aggregation_openings.end())
                        expected_aggregation_openings.push_back(
                            std::move(value));
                };
                for (const auto& route : meta.route_terminal_relations)
                {
                    const auto challenge = proof.challenges.find(
                        route.challenge_label);
                    if (challenge == proof.challenges.end())
                        throw std::runtime_error(
                            "proof parser Pi_bind missing route terminal challenge");
                    append_aggregation_opening(
                        "W_ext:" + challenge->second.to_string());
                    append_aggregation_opening(
                        "W_ext:"
                        + (challenge->second + route.terminal_delta)
                              .to_string());
                    append_aggregation_opening(
                        "W_ext:" + route.node_terminal_point.to_string());
                }
                for (const auto& aggregation : meta.aggregation_relations)
                {
                    expected_aggregation_commitments.insert(
                        expected_aggregation_commitments.end(),
                        aggregation.commitment_order.begin(),
                        aggregation.commitment_order.end());
                    for (const auto* route : {
                             &aggregation.source_route,
                             &aggregation.target_route})
                    {
                        expected_aggregation_relation_evaluations.push_back({
                            route->shifted_accumulator_object + ":u_rel",
                            "relation", 1,
                            PayloadValueKind::MaskedPrivateValue});
                        const auto challenge = proof.challenges.find(
                            route->terminal_challenge_label);
                        if (challenge == proof.challenges.end())
                            throw std::runtime_error(
                                "proof parser Pi_bind missing aggregation terminal challenge");
                        append_aggregation_opening(
                            "W_ext:" + challenge->second.to_string());
                        append_aggregation_opening(
                            "W_ext:"
                            + (challenge->second + route->terminal_delta)
                                  .to_string());
                        append_aggregation_opening(
                            "W_ext:" + route->node_terminal_point.to_string());
                    }
                    for (const auto* object : {
                             &aggregation.source_node_object,
                             &aggregation.target_node_object})
                        expected_aggregation_domain_evaluations.push_back({
                            *object, aggregation.node_domain, 2,
                            PayloadValueKind::MaskedPrivateValue});
                    for (const auto* route : {
                             &aggregation.source_route,
                             &aggregation.target_route})
                        for (const auto* object : {
                                 &route->table_object,
                                 &route->node_inverse_object,
                                 &route->node_accumulator_object})
                            expected_aggregation_domain_evaluations.push_back({
                                *object, aggregation.node_domain, 2,
                                PayloadValueKind::MaskedPrivateValue});
                    for (const auto* object : {
                             &aggregation.source_edge_object,
                             &aggregation.target_edge_object,
                             &aggregation.product_object,
                             &aggregation.psq_object})
                        expected_aggregation_domain_evaluations.push_back({
                            *object, aggregation.edge_domain, 2,
                            PayloadValueKind::MaskedPrivateValue});
                    for (const auto* route : {
                             &aggregation.source_route,
                             &aggregation.target_route})
                        for (const auto* object : {
                                 &route->query_object,
                                 &route->edge_inverse_object,
                                 &route->edge_accumulator_object})
                            expected_aggregation_domain_evaluations.push_back({
                                *object, aggregation.edge_domain, 2,
                                PayloadValueKind::MaskedPrivateValue});
                }
                const bool aggregation_commitments_match =
                    subproof.aggregation_commitments.size()
                        == expected_aggregation_commitments.size()
                    && std::equal(
                        subproof.aggregation_commitments.begin(),
                        subproof.aggregation_commitments.end(),
                        expected_aggregation_commitments.begin(),
                        [](const auto& actual, const std::string& expected)
                        { return actual.first == expected; });
                if (meta.object_kind != "output")
                {
                    auto expected_evaluations =
                        expected_aggregation_relation_evaluations;
                    expected_evaluations.insert(
                        expected_evaluations.end(),
                        expected_canonical_evaluations.begin(),
                        expected_canonical_evaluations.end());
                    expected_evaluations.insert(
                        expected_evaluations.end(),
                        expected_aggregation_domain_evaluations.begin(),
                        expected_aggregation_domain_evaluations.end());
                    const bool evaluations_match =
                        subproof.evaluations.size()
                            == expected_evaluations.size()
                        && std::equal(
                            subproof.evaluations.begin(),
                            subproof.evaluations.end(),
                            expected_evaluations.begin(),
                            [](const BindingEvaluation& actual,
                               const auto& expected)
                            {
                                return actual.object_id == std::get<0>(expected)
                                    && actual.domain == std::get<1>(expected)
                                    && actual.values.size() == std::get<2>(expected)
                                    && actual.kind == std::get<3>(expected);
                            });
                    auto expected_openings = expected_aggregation_openings;
                    if (!meta.canonical_matrices.empty())
                    {
                        const auto point = proof.challenges.at("z_N");
                        expected_openings.push_back(
                            "Open_dom:N:" + point.to_string());
                        expected_openings.push_back(
                            "Open_dom:N:"
                            + (point * context.domains.n->omega).to_string());
                    }
                    if (!meta.aggregation_relations.empty())
                    {
                        const auto point = proof.challenges.at("z_edge");
                        expected_openings.push_back(
                            "Open_dom:edge:" + point.to_string());
                        expected_openings.push_back(
                            "Open_dom:edge:"
                            + (point * context.domains.edge->omega).to_string());
                    }
                    const bool opening_references_match =
                        subproof.opening_references.size()
                            == expected_openings.size()
                        && std::equal(
                            subproof.opening_references.begin(),
                            subproof.opening_references.end(),
                            expected_openings.begin(),
                            [](const MetadataReference& actual,
                               const std::string& expected)
                            { return actual.object_id == expected; });
                    if (!canonical_references_match()
                        || subproof.view_references.size()
                            != expected_canonical_references.size()
                        || !canonical_commitments_match()
                        || !subproof.view_logup_commitments.empty()
                        || !aggregation_commitments_match
                        || !subproof.basis_commitments.empty()
                        || !subproof.crpc_commitments.empty()
                        || subproof.relation_commitments.size()
                            != expected_route_commitments.size()
                        || !std::equal(
                            subproof.relation_commitments.begin(),
                            subproof.relation_commitments.end(),
                            expected_route_commitments.begin(),
                            [](const auto& actual,
                               const std::string& expected)
                            { return actual.first == expected; })
                        || !evaluations_match
                        || !opening_references_match)
                    {
                        throw std::runtime_error(
                            "proof parser Pi_bind unexpected non-output payload");
                    }
                    continue;
                }

                const auto head = meta.head_id.value();
                if (meta.indexed_views.size() != 1)
                    throw std::runtime_error(
                        "proof parser Pi_bind output relation shape mismatch");
                const auto& view = meta.indexed_views.front();
                const auto* relation_ptr = find_unique_shift_relation_by_kind(
                    meta, "output-bias-terminal");
                if (relation_ptr == nullptr)
                    throw std::runtime_error(
                        "proof parser Pi_bind output bias relation is not unique");
                const auto& relation = *relation_ptr;
                const bool same_point = relation.configured_degrees.empty();
                if (meta.coefficient_basis_relations.size() != 1)
                    throw std::runtime_error(
                        "proof parser Pi_bind coefficient basis relation is not unique");
                const auto& basis = meta.coefficient_basis_relations.front();
                auto expected_view_references = expected_canonical_references;
                expected_view_references.insert(
                    expected_view_references.end(),
                    {
                        view.canonical_object,
                        view.view_object,
                        "public:" + view.canonical_selector_id,
                        "public:" + view.query_selector_id,
                    });
                if (subproof.view_references.size()
                        != expected_view_references.size()
                    || !std::equal(
                        subproof.view_references.begin(),
                        subproof.view_references.end(),
                        expected_view_references.begin(),
                        [](const MetadataReference& actual,
                           const std::string& expected)
                        { return actual.object_id == expected; }))
                {
                    throw std::runtime_error(
                        "proof parser Pi_bind Com_view reference shape mismatch");
                }
                if (!canonical_commitments_match()
                    || !aggregation_commitments_match
                    || !subproof.crpc_commitments.empty()
                    || subproof.basis_commitments.size()
                        != basis.commitment_order.size()
                    || !std::equal(
                        subproof.basis_commitments.begin(),
                        subproof.basis_commitments.end(),
                        basis.commitment_order.begin(),
                        [](const auto& actual, const std::string& expected)
                        { return actual.first == expected; })
                    || subproof.view_logup_commitments.size()
                        != view.commitment_order.size()
                    || !std::equal(
                        subproof.view_logup_commitments.begin(),
                        subproof.view_logup_commitments.end(),
                        view.commitment_order.begin(),
                        [](const auto& actual, const std::string& expected)
                        { return actual.first == expected; })
                    || subproof.relation_commitments.size()
                        != expected_route_commitments.size()
                            + basis.chunks.size()
                            + (same_point ? 0 : 1)
                    || !std::equal(
                        subproof.relation_commitments.begin(),
                        subproof.relation_commitments.begin()
                            + static_cast<std::ptrdiff_t>(
                                expected_route_commitments.size()),
                        expected_route_commitments.begin(),
                        [](const auto& actual,
                           const std::string& expected)
                        { return actual.first == expected; })
                    || (!same_point
                        && subproof.relation_commitments[
                               expected_route_commitments.size()].first
                            != output_bias_view_label(head, "R_shifted"))
                    || !std::equal(
                        subproof.relation_commitments.end()
                            - static_cast<std::ptrdiff_t>(basis.chunks.size()),
                        subproof.relation_commitments.end(),
                        basis.chunks.begin(),
                        [](const auto& actual, const auto& chunk)
                        { return actual.first
                            == chunk.shifted_accumulator_object; }))
                {
                    throw std::runtime_error(
                        "proof parser Pi_bind commitment group shape mismatch");
                }

                std::vector<std::tuple<
                    std::string, std::string, std::size_t, PayloadValueKind>>
                    expected_evaluations = {
                        {output_bias_view_label(head, "m_can") + ":public_bind",
                         "relation", 1, PayloadValueKind::PublicValue},
                        {model_proof_local_label(output_bias_label(head)),
                         "C", 2, PayloadValueKind::MaskedPrivateValue},
                        {output_bias_view_label(head, "Table_can"),
                         "C", 2, PayloadValueKind::MaskedPrivateValue},
                        {output_bias_view_label(head, "m_can"),
                         "C", 2, PayloadValueKind::PublicValue},
                        {output_bias_view_label(head, "J_can"),
                         "C", 2, PayloadValueKind::MaskedPrivateValue},
                        {output_bias_view_label(head, "R_can"),
                         "C", 2, PayloadValueKind::MaskedPrivateValue},
                        {output_bias_view_label(head, "Query"),
                         "edge", 2, PayloadValueKind::MaskedPrivateValue},
                        {output_bias_view_label(head, "J_qry"),
                         "edge", 2, PayloadValueKind::MaskedPrivateValue},
                        {output_bias_view_label(head, "R_qry"),
                         "edge", 2, PayloadValueKind::MaskedPrivateValue},
                    };
                expected_evaluations.insert(
                    expected_evaluations.begin(),
                    expected_aggregation_relation_evaluations.begin(),
                    expected_aggregation_relation_evaluations.end());
                if (!same_point)
                    expected_evaluations.insert(
                        expected_evaluations.begin()
                            + static_cast<std::ptrdiff_t>(
                                expected_aggregation_relation_evaluations.size()),
                        {output_bias_view_label(head, "R_shifted") + ":u_rel",
                         "relation", 1,
                         PayloadValueKind::MaskedPrivateValue});
                auto basis_eval_position = expected_evaluations.begin()
                    + static_cast<std::ptrdiff_t>(
                        expected_aggregation_relation_evaluations.size()
                        + (same_point ? 1 : 2));
                for (const auto& chunk : basis.chunks)
                    basis_eval_position = expected_evaluations.insert(
                        basis_eval_position,
                        {chunk.shifted_accumulator_object + ":u_rel",
                         "relation", 1,
                         PayloadValueKind::MaskedPrivateValue}) + 1;
                for (const auto& chunk : basis.chunks)
                {
                    expected_evaluations.push_back({
                        chunk.object_id, "edge", 2,
                        PayloadValueKind::MaskedPrivateValue});
                    expected_evaluations.push_back({
                        chunk.accumulator_object, "edge", 2,
                        PayloadValueKind::MaskedPrivateValue});
                }
                expected_evaluations.insert(
                    expected_evaluations.end(),
                    expected_canonical_evaluations.begin(),
                    expected_canonical_evaluations.end());
                expected_evaluations.insert(
                    expected_evaluations.end(),
                    expected_aggregation_domain_evaluations.begin(),
                    expected_aggregation_domain_evaluations.end());
                if (subproof.evaluations.size() != expected_evaluations.size()
                    || !std::equal(
                        subproof.evaluations.begin(),
                        subproof.evaluations.end(),
                        expected_evaluations.begin(),
                        [](const BindingEvaluation& actual, const auto& expected)
                        {
                            return actual.object_id == std::get<0>(expected)
                                && actual.domain == std::get<1>(expected)
                                && actual.values.size() == std::get<2>(expected)
                                && actual.kind == std::get<3>(expected);
                        }))
                {
                    throw std::runtime_error(
                        "proof parser Pi_bind evaluation shape mismatch");
                }

                const auto challenge_it = proof.challenges.find(
                    relation.challenge_label);
                if (challenge_it == proof.challenges.end())
                    throw std::runtime_error(
                        "proof parser Pi_bind missing relation challenge");
                std::vector<std::string> expected_opening_references =
                    expected_aggregation_openings;
                const auto append_unique = [&](std::string reference)
                {
                    if (std::find(
                            expected_opening_references.begin(),
                            expected_opening_references.end(), reference)
                        == expected_opening_references.end())
                        expected_opening_references.push_back(
                            std::move(reference));
                };
                const auto y_rel = challenge_it->second;
                if (same_point)
                {
                    append_unique("W_ext:" + y_rel.to_string());
                    append_unique(
                        "W_ext:" + relation.canonical_points.front().to_string());
                }
                else
                {
                    append_unique("W_ext:" + y_rel.to_string());
                    append_unique(
                        "W_ext:"
                        + (y_rel + relation.deltas.at(1)).to_string());
                    append_unique(
                        "W_ext:" + relation.canonical_points.at(0).to_string());
                }
                const auto gamma_it = proof.challenges.find(
                    basis.gamma_challenge_label);
                const auto basis_y_it = proof.challenges.find(
                    basis.relation_challenge_label);
                if (gamma_it == proof.challenges.end()
                    || basis_y_it == proof.challenges.end())
                    throw std::runtime_error(
                        "proof parser Pi_bind missing coefficient basis challenge");
                append_unique(
                    "W_ext:" + basis_y_it->second.to_string());
                for (const auto& chunk : basis.chunks)
                    append_unique(
                        "W_ext:"
                        + (basis_y_it->second + chunk.terminal_point
                           - gamma_it->second).to_string());
                append_unique("W_ext:" + gamma_it->second.to_string());
                for (const auto& [domain, point, omega] :
                     std::vector<std::tuple<std::string,
                         algebra::FieldElement, algebra::FieldElement>>{
                         {"C", proof.challenges.at("z_C"),
                          context.domains.c->omega},
                         {"edge", proof.challenges.at("z_edge"),
                          context.domains.edge->omega}})
                {
                    append_unique(
                        "Open_dom:" + domain + ":" + point.to_string());
                    append_unique(
                        "Open_dom:" + domain + ":"
                        + (point * omega).to_string());
                }
                const auto n_point = proof.challenges.at("z_N");
                append_unique(
                    "Open_dom:N:" + n_point.to_string());
                append_unique(
                    "Open_dom:N:"
                    + (n_point * context.domains.n->omega).to_string());
                if (subproof.opening_references.size()
                        != expected_opening_references.size()
                    || !std::equal(
                        subproof.opening_references.begin(),
                        subproof.opening_references.end(),
                        expected_opening_references.begin(),
                        [](const MetadataReference& actual,
                           const std::string& expected)
                        { return actual.object_id == expected; }))
                {
                    throw std::runtime_error(
                        "proof parser Pi_bind opening-reference shape mismatch");
                }
            }
        }
    }

    std::vector<std::uint8_t> serialize_proof(const Proof& proof)
    {
        Writer wire;
        wire.raw(kMagic.data(), kMagic.size());
        wire.u64(kWireVersion);

        Writer m_pub;
        auto metadata = proof.public_metadata;
        metadata_fields(metadata, [&](const std::string& field) { m_pub.string(field); });
        m_pub.u64(proof.block_order.size());
        for (const auto& tag : proof.block_order) m_pub.string(tag);
        m_pub.u64(proof.public_logits.size());
        for (const auto& value : proof.public_logits) m_pub.scalar(value);
        wire.block("M_pub", std::move(m_pub));

        Writer com_dyn;
        write_named_commitments(com_dyn, proof.dynamic_commitments);
        wire.block("Com_dyn", std::move(com_dyn));

        Writer s_route;
        s_route.u64(proof.route_references.size());
        for (const auto& reference : proof.route_references)
            s_route.string(reference.object_id);
        wire.block("S_route", std::move(s_route));

        Writer eval_ext;
        eval_ext.u64(proof.external_evaluations.size());
        for (std::size_t i = 0; i < proof.external_evaluations.size(); ++i)
        {
            eval_ext.string(proof.external_evaluations[i].first);
            eval_ext.scalar(proof.external_evaluations[i].second);
            eval_ext.u64(static_cast<std::uint64_t>(proof.external_evaluation_kinds.at(i)));
        }
        wire.block("Eval_ext", std::move(eval_ext));

        std::unordered_set<std::string> pi_bind_domain_evaluations;
        for (const auto& subproof : proof.pi_bind.subproofs)
            for (const auto& evaluation : subproof.evaluations)
                if (evaluation.domain != "relation")
                    pi_bind_domain_evaluations.insert(evaluation.object_id);

        Writer eval_dom;
        eval_dom.u64(proof.domain_openings.size());
        for (const auto& [name, bundle] : proof.domain_openings)
        {
            eval_dom.string(name);
            eval_dom.u64(bundle.points.size());
            for (const auto& point : bundle.points) eval_dom.scalar(point);
            const auto transmitted_count = std::count_if(
                bundle.values.begin(), bundle.values.end(),
                [&](const auto& entry)
                { return !pi_bind_domain_evaluations.contains(entry.first); });
            eval_dom.u64(transmitted_count);
            for (std::size_t i = 0; i < bundle.values.size(); ++i)
            {
                if (pi_bind_domain_evaluations.contains(bundle.values[i].first))
                    continue;
                eval_dom.string(bundle.values[i].first);
                eval_dom.u64(bundle.values[i].second.size());
                for (const auto& value : bundle.values[i].second) eval_dom.scalar(value);
                eval_dom.u64(static_cast<std::uint64_t>(bundle.value_kinds.at(i)));
            }
        }
        wire.block("Eval_dom", std::move(eval_dom));

        Writer com_quot;
        write_named_commitments(com_quot, proof.quotient_commitments);
        wire.block("Com_quot", std::move(com_quot));

        Writer open_dom;
        open_dom.u64(proof.domain_openings.size());
        for (const auto& [name, bundle] : proof.domain_openings)
        {
            open_dom.string(name);
            write_opening(open_dom, bundle.proof);
        }
        wire.block("Open_dom", std::move(open_dom));

        Writer w_ext;
        write_opening(w_ext, proof.external_proof);
        wire.block("W_ext", std::move(w_ext));

        Writer pi_bind;
        pi_bind.u64(proof.pi_bind.subproofs.size());
        for (const auto& subproof : proof.pi_bind.subproofs)
        {
            const auto& meta = subproof.relation_meta;
            pi_bind.string(meta.relation_id);
            pi_bind.string(meta.object_kind);
            write_optional_index(pi_bind, meta.layer_id);
            write_optional_index(pi_bind, meta.head_id);
            write_string_vector(pi_bind, meta.source_objects);
            write_string_vector(pi_bind, meta.target_objects);
            write_string_vector(pi_bind, meta.work_domains);
            write_size_vector(pi_bind, meta.real_lengths);
            write_string_vector(pi_bind, meta.quotient_relations);
            pi_bind.u64(meta.canonical_matrices.size());
            for (const auto& matrix : meta.canonical_matrices)
            {
                pi_bind.string(matrix.relation_id);
                pi_bind.string(matrix.matrix_object);
                pi_bind.string(matrix.canonical_domain);
                pi_bind.u64(matrix.row_count);
                pi_bind.u64(matrix.column_count);
                write_string_vector(pi_bind, matrix.column_objects);
                pi_bind.string(matrix.compressed_object);
                pi_bind.string(matrix.compression_challenge_label);
            }
            const auto write_aggregation_route = [&](const auto& route)
            {
                for (const auto* value : {
                         &route.relation_id, &route.index_object,
                         &route.node_value_object, &route.edge_value_object,
                         &route.table_object, &route.query_object,
                         &route.multiplicity_object, &route.node_inverse_object,
                         &route.edge_inverse_object,
                         &route.node_accumulator_object,
                         &route.edge_accumulator_object,
                         &route.shifted_accumulator_object,
                         &route.relation_virtual_object,
                         &route.eta_challenge_label,
                         &route.beta_challenge_label,
                         &route.terminal_challenge_label})
                    pi_bind.string(*value);
                pi_bind.scalar(route.node_terminal_point);
                pi_bind.scalar(route.edge_terminal_point);
                pi_bind.scalar(route.terminal_delta);
                pi_bind.u64(route.configured_shift_degree);
                pi_bind.u64(route.shift_domain_size);
            };
            pi_bind.u64(meta.aggregation_relations.size());
            for (const auto& relation : meta.aggregation_relations)
            {
                for (const auto* value : {
                         &relation.relation_id,
                         &relation.source_matrix_object,
                         &relation.target_matrix_object,
                         &relation.alpha_object,
                         &relation.node_domain,
                         &relation.edge_domain,
                         &relation.compression_challenge_label,
                         &relation.source_node_object,
                         &relation.target_node_object,
                         &relation.source_edge_object,
                         &relation.target_edge_object,
                         &relation.product_object,
                         &relation.psq_object})
                    pi_bind.string(*value);
                pi_bind.u64(relation.row_count);
                pi_bind.u64(relation.column_count);
                write_aggregation_route(relation.source_route);
                write_aggregation_route(relation.target_route);
                write_string_vector(pi_bind, relation.commitment_order);
            }
            pi_bind.u64(meta.route_terminal_relations.size());
            for (const auto& relation : meta.route_terminal_relations)
            {
                for (const auto* value : {
                         &relation.relation_id,
                         &relation.node_accumulator_object,
                         &relation.edge_accumulator_object,
                         &relation.shifted_accumulator_object,
                         &relation.relation_virtual_object,
                         &relation.node_domain,
                         &relation.edge_domain,
                         &relation.challenge_label})
                    pi_bind.string(*value);
                pi_bind.scalar(relation.node_terminal_point);
                pi_bind.scalar(relation.edge_terminal_point);
                pi_bind.scalar(relation.terminal_delta);
                pi_bind.u64(relation.configured_shift_degree);
                pi_bind.u64(relation.shift_domain_size);
            }
            pi_bind.u64(meta.indexed_views.size());
            for (const auto& view : meta.indexed_views)
            {
                pi_bind.string(view.relation_id);
                pi_bind.string(view.canonical_object);
                pi_bind.string(view.view_object);
                pi_bind.string(view.canonical_domain);
                pi_bind.string(view.view_domain);
                pi_bind.u64(view.canonical_length);
                pi_bind.u64(view.view_length);
                pi_bind.string(view.index_map_id);
                pi_bind.string(view.canonical_selector_id);
                pi_bind.string(view.query_selector_id);
                pi_bind.string(view.multiplicity_id);
                pi_bind.string(view.eta_challenge_label);
                pi_bind.string(view.beta_challenge_label);
                write_string_vector(pi_bind, view.commitment_order);
            }
            pi_bind.u64(meta.coefficient_basis_relations.size());
            for (const auto& relation : meta.coefficient_basis_relations)
            {
                pi_bind.string(relation.relation_id);
                pi_bind.string(relation.canonical_object);
                pi_bind.string(relation.coefficient_object);
                pi_bind.string(relation.carrier_domain);
                pi_bind.u64(relation.low_degree_length);
                pi_bind.u64(relation.tail_length);
                pi_bind.u64(relation.total_length);
                write_size_vector(pi_bind, relation.semantic_positions);
                pi_bind.u64(relation.chunks.size());
                for (const auto& chunk : relation.chunks)
                {
                    pi_bind.string(chunk.object_id);
                    pi_bind.string(chunk.accumulator_object);
                    pi_bind.string(chunk.shifted_accumulator_object);
                    pi_bind.u64(chunk.chunk_index);
                    pi_bind.u64(chunk.offset);
                    pi_bind.u64(chunk.length);
                    pi_bind.scalar(chunk.terminal_point);
                    pi_bind.u64(chunk.configured_shift_degree);
                    pi_bind.u64(chunk.shift_domain_size);
                }
                pi_bind.string(relation.power_object);
                pi_bind.string(relation.gamma_challenge_label);
                pi_bind.string(relation.relation_challenge_label);
                write_string_vector(pi_bind, relation.commitment_order);
                write_string_vector(pi_bind, relation.opening_claim_order);
            }
            pi_bind.u64(meta.shift_relations.size());
            for (const auto& relation : meta.shift_relations)
            {
                pi_bind.string(relation.relation_id);
                write_string_vector(pi_bind, relation.source_linear_maps);
                write_string_vector(pi_bind, relation.source_domains);
                pi_bind.u64(relation.canonical_points.size());
                for (const auto& point : relation.canonical_points) pi_bind.scalar(point);
                pi_bind.scalar(relation.lambda0);
                pi_bind.u64(relation.deltas.size());
                for (const auto& delta : relation.deltas) pi_bind.scalar(delta);
                write_string_vector(pi_bind, relation.source_value_kinds);
                pi_bind.string(relation.challenge_label);
                pi_bind.u64(relation.exclusion_degree);
                pi_bind.u64(relation.max_difference_degree);
                write_size_vector(pi_bind, relation.configured_degrees);
                write_size_vector(pi_bind, relation.shift_domain_sizes);
                write_string_vector(pi_bind, relation.opening_claim_order);
            }

            pi_bind.u64(subproof.model_reblinds.size());
            for (const auto& reblind : subproof.model_reblinds)
            {
                pi_bind.string(reblind.object_id);
                pi_bind.string(reblind.domain);
                pi_bind.string(reblind.vk_model_anchor.object_id);
                pi_bind.string(reblind.proof_local_commitment.name);
                pi_bind.point(reblind.proof_local_commitment.point);
                pi_bind.string(reblind.blinder_commitment.name);
                pi_bind.point(reblind.blinder_commitment.point);
            }
            pi_bind.u64(subproof.view_references.size());
            for (const auto& reference : subproof.view_references)
                pi_bind.string(reference.object_id);
            for (const auto* group : {
                     &subproof.view_commitments,
                     &subproof.view_logup_commitments,
                     &subproof.aggregation_commitments,
                     &subproof.basis_commitments,
                     &subproof.crpc_commitments,
                     &subproof.relation_commitments})
            {
                write_named_commitments(pi_bind, *group);
            }
            pi_bind.u64(subproof.evaluations.size());
            for (const auto& evaluation : subproof.evaluations)
            {
                pi_bind.string(evaluation.domain);
                pi_bind.string(evaluation.object_id);
                pi_bind.u64(evaluation.values.size());
                for (const auto& value : evaluation.values) pi_bind.scalar(value);
                pi_bind.u64(static_cast<std::uint64_t>(evaluation.kind));
            }
            pi_bind.u64(subproof.opening_references.size());
            for (const auto& reference : subproof.opening_references)
                pi_bind.string(reference.object_id);
        }
        wire.block("Pi_bind", std::move(pi_bind));
        return wire.take();
    }

    Proof parse_proof(const ProtocolContext& context, const std::vector<std::uint8_t>& bytes)
    {
        if (context.compiler_artifact == nullptr)
        {
            throw std::runtime_error(
                "proof parser is missing the authenticated ProtocolCompilerArtifact");
        }
        const auto& compiler_artifact = *context.compiler_artifact;
        std::string artifact_reason;
        if (!validate_protocol_compiler_artifact(context, compiler_artifact, &artifact_reason))
        {
            throw std::runtime_error("proof parser compiler artifact mismatch: " + artifact_reason);
        }
        // Read the fixed magic without accepting a length-delimited alias.
        if (bytes.size() < kMagic.size() + 8 || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
        {
            throw std::runtime_error("proof parser magic mismatch");
        }
        Reader body(bytes.data() + kMagic.size(), bytes.size() - kMagic.size());
        if (body.u64() != kWireVersion) throw std::runtime_error("proof parser version mismatch");

        Proof proof;
        auto m_pub = body.block("M_pub");
        metadata_fields(proof.public_metadata, [&](std::string& field) { field = m_pub.string(); });
        const auto block_count = m_pub.count("block order");
        proof.block_order.reserve(block_count);
        for (std::size_t i = 0; i < block_count; ++i) proof.block_order.push_back(m_pub.string());
        if (proof.block_order != proof_block_order()) throw std::runtime_error("proof parser embedded block order mismatch");
        const auto logits_count = m_pub.count("public logits");
        proof.public_logits.reserve(logits_count);
        for (std::size_t i = 0; i < logits_count; ++i) proof.public_logits.push_back(m_pub.scalar());
        m_pub.finish("M_pub");

        auto com_dyn = body.block("Com_dyn");
        proof.dynamic_commitments = read_named_commitments(com_dyn);
        require_unique_names(proof.dynamic_commitments, "Com_dyn");
        com_dyn.finish("Com_dyn");

        auto s_route = body.block("S_route");
        const auto reference_count = s_route.count("route references");
        proof.route_references.reserve(reference_count);
        for (std::size_t i = 0; i < reference_count; ++i)
            proof.route_references.push_back({s_route.string()});
        s_route.finish("S_route");

        auto eval_ext = body.block("Eval_ext");
        const auto ext_count = eval_ext.count("external evaluations");
        for (std::size_t i = 0; i < ext_count; ++i)
        {
            auto name = eval_ext.string();
            auto value = eval_ext.scalar();
            proof.external_evaluations.push_back({std::move(name), value});
            proof.external_evaluation_kinds.push_back(read_kind(eval_ext));
        }
        eval_ext.finish("Eval_ext");
        require_unique_names(proof.external_evaluations, "Eval_ext");

        auto eval_dom = body.block("Eval_dom");
        const auto domain_count = eval_dom.count("domain bundles");
        for (std::size_t domain_index = 0; domain_index < domain_count; ++domain_index)
        {
            auto name = eval_dom.string();
            DomainOpeningBundle bundle;
            const auto point_count = eval_dom.count("domain points");
            for (std::size_t i = 0; i < point_count; ++i) bundle.points.push_back(eval_dom.scalar());
            const auto value_count = eval_dom.count("domain value columns");
            for (std::size_t i = 0; i < value_count; ++i)
            {
                auto label = eval_dom.string();
                const auto item_count = eval_dom.count("domain values");
                std::vector<algebra::FieldElement> values;
                values.reserve(item_count);
                for (std::size_t j = 0; j < item_count; ++j) values.push_back(eval_dom.scalar());
                bundle.values.push_back({std::move(label), std::move(values)});
                bundle.value_kinds.push_back(read_kind(eval_dom));
            }
            require_unique_names(bundle.values, "Eval_dom bundle");
            proof.domain_openings.push_back({std::move(name), std::move(bundle)});
        }
        eval_dom.finish("Eval_dom");
        require_unique_names(proof.domain_openings, "Eval_dom");

        auto com_quot = body.block("Com_quot");
        proof.quotient_commitments = read_named_commitments(com_quot);
        require_unique_names(proof.quotient_commitments, "Com_quot");
        com_quot.finish("Com_quot");

        auto open_dom = body.block("Open_dom");
        const auto opening_count = open_dom.count("domain opening proofs");
        if (opening_count != proof.domain_openings.size()) throw std::runtime_error("proof parser domain opening count mismatch");
        for (std::size_t i = 0; i < opening_count; ++i)
        {
            if (open_dom.string() != proof.domain_openings[i].first) throw std::runtime_error("proof parser domain opening name mismatch");
            proof.domain_openings[i].second.proof = read_opening(open_dom);
            require_canonical_point_group_shape(
                proof.domain_openings[i].second.points,
                proof.domain_openings[i].second.proof,
                proof.domain_openings[i].first.c_str());
        }
        open_dom.finish("Open_dom");

        auto w_ext = body.block("W_ext");
        proof.external_proof = read_opening(w_ext);
        w_ext.finish("W_ext");

        auto pi_bind = body.block("Pi_bind");
        const auto expected_binding_metadata =
            build_binding_relation_metadata(context);
        const auto subproof_count = pi_bind.count("Pi_bind subproofs");
        if (subproof_count != expected_binding_metadata.size())
            throw std::runtime_error(
                "proof parser Pi_bind subproof count does not match instance schema");
        proof.pi_bind.subproofs.reserve(subproof_count);
        for (std::size_t subproof_index = 0; subproof_index < subproof_count; ++subproof_index)
        {
            BindingSubproof subproof;
            auto& meta = subproof.relation_meta;
            meta.relation_id = pi_bind.string();
            meta.object_kind = pi_bind.string();
            meta.layer_id = read_optional_index(pi_bind, "Pi_bind layer_id");
            meta.head_id = read_optional_index(pi_bind, "Pi_bind head_id");
            meta.source_objects = read_string_vector(pi_bind, "Pi_bind sources");
            meta.target_objects = read_string_vector(pi_bind, "Pi_bind targets");
            meta.work_domains = read_string_vector(pi_bind, "Pi_bind domains");
            meta.real_lengths = read_size_vector(pi_bind, "Pi_bind real lengths");
            meta.quotient_relations = read_string_vector(pi_bind, "Pi_bind quotient relations");
            const auto canonical_matrix_count = pi_bind.count(
                "Pi_bind canonical matrices");
            meta.canonical_matrices.reserve(canonical_matrix_count);
            for (std::size_t i = 0; i < canonical_matrix_count; ++i)
            {
                CanonicalMatrixRelationMeta matrix;
                matrix.relation_id = pi_bind.string();
                matrix.matrix_object = pi_bind.string();
                matrix.canonical_domain = pi_bind.string();
                matrix.row_count = pi_bind.count(
                    "Pi_bind canonical matrix rows");
                matrix.column_count = pi_bind.count(
                    "Pi_bind canonical matrix columns");
                matrix.column_objects = read_string_vector(
                    pi_bind, "Pi_bind canonical column objects");
                matrix.compressed_object = pi_bind.string();
                matrix.compression_challenge_label = pi_bind.string();
                meta.canonical_matrices.push_back(std::move(matrix));
            }
            const auto read_aggregation_route = [&]
            {
                AggregationRouteRelationMeta route;
                route.relation_id = pi_bind.string();
                route.index_object = pi_bind.string();
                route.node_value_object = pi_bind.string();
                route.edge_value_object = pi_bind.string();
                route.table_object = pi_bind.string();
                route.query_object = pi_bind.string();
                route.multiplicity_object = pi_bind.string();
                route.node_inverse_object = pi_bind.string();
                route.edge_inverse_object = pi_bind.string();
                route.node_accumulator_object = pi_bind.string();
                route.edge_accumulator_object = pi_bind.string();
                route.shifted_accumulator_object = pi_bind.string();
                route.relation_virtual_object = pi_bind.string();
                route.eta_challenge_label = pi_bind.string();
                route.beta_challenge_label = pi_bind.string();
                route.terminal_challenge_label = pi_bind.string();
                route.node_terminal_point = pi_bind.scalar();
                route.edge_terminal_point = pi_bind.scalar();
                route.terminal_delta = pi_bind.scalar();
                route.configured_shift_degree = pi_bind.count(
                    "Pi_bind aggregation route shift degree");
                route.shift_domain_size = pi_bind.count(
                    "Pi_bind aggregation route shift domain");
                return route;
            };
            const auto aggregation_relation_count = pi_bind.count(
                "Pi_bind aggregation relations");
            meta.aggregation_relations.reserve(aggregation_relation_count);
            for (std::size_t i = 0; i < aggregation_relation_count; ++i)
            {
                AggregationRelationMeta relation;
                relation.relation_id = pi_bind.string();
                relation.source_matrix_object = pi_bind.string();
                relation.target_matrix_object = pi_bind.string();
                relation.alpha_object = pi_bind.string();
                relation.node_domain = pi_bind.string();
                relation.edge_domain = pi_bind.string();
                relation.compression_challenge_label = pi_bind.string();
                relation.source_node_object = pi_bind.string();
                relation.target_node_object = pi_bind.string();
                relation.source_edge_object = pi_bind.string();
                relation.target_edge_object = pi_bind.string();
                relation.product_object = pi_bind.string();
                relation.psq_object = pi_bind.string();
                relation.row_count = pi_bind.count(
                    "Pi_bind aggregation rows");
                relation.column_count = pi_bind.count(
                    "Pi_bind aggregation columns");
                relation.source_route = read_aggregation_route();
                relation.target_route = read_aggregation_route();
                relation.commitment_order = read_string_vector(
                    pi_bind, "Pi_bind aggregation commitment order");
                meta.aggregation_relations.push_back(std::move(relation));
            }
            const auto route_terminal_count = pi_bind.count(
                "Pi_bind route terminal relations");
            meta.route_terminal_relations.reserve(route_terminal_count);
            for (std::size_t i = 0; i < route_terminal_count; ++i)
            {
                RouteTerminalRelationMeta relation;
                relation.relation_id = pi_bind.string();
                relation.node_accumulator_object = pi_bind.string();
                relation.edge_accumulator_object = pi_bind.string();
                relation.shifted_accumulator_object = pi_bind.string();
                relation.relation_virtual_object = pi_bind.string();
                relation.node_domain = pi_bind.string();
                relation.edge_domain = pi_bind.string();
                relation.challenge_label = pi_bind.string();
                relation.node_terminal_point = pi_bind.scalar();
                relation.edge_terminal_point = pi_bind.scalar();
                relation.terminal_delta = pi_bind.scalar();
                relation.configured_shift_degree = pi_bind.count(
                    "Pi_bind route terminal shift degree");
                relation.shift_domain_size = pi_bind.count(
                    "Pi_bind route terminal shift domain");
                meta.route_terminal_relations.push_back(std::move(relation));
            }
            const auto indexed_view_count = pi_bind.count("Pi_bind indexed views");
            meta.indexed_views.reserve(indexed_view_count);
            for (std::size_t i = 0; i < indexed_view_count; ++i)
            {
                IndexedViewRelationMeta view;
                view.relation_id = pi_bind.string();
                view.canonical_object = pi_bind.string();
                view.view_object = pi_bind.string();
                view.canonical_domain = pi_bind.string();
                view.view_domain = pi_bind.string();
                view.canonical_length = pi_bind.count("Pi_bind indexed canonical length");
                view.view_length = pi_bind.count("Pi_bind indexed view length");
                view.index_map_id = pi_bind.string();
                view.canonical_selector_id = pi_bind.string();
                view.query_selector_id = pi_bind.string();
                view.multiplicity_id = pi_bind.string();
                view.eta_challenge_label = pi_bind.string();
                view.beta_challenge_label = pi_bind.string();
                view.commitment_order = read_string_vector(
                    pi_bind, "Pi_bind indexed commitment order");
                meta.indexed_views.push_back(std::move(view));
            }
            const auto basis_relation_count = pi_bind.count(
                "Pi_bind coefficient basis relations");
            meta.coefficient_basis_relations.reserve(basis_relation_count);
            for (std::size_t i = 0; i < basis_relation_count; ++i)
            {
                CoefficientBasisRelationMeta relation;
                relation.relation_id = pi_bind.string();
                relation.canonical_object = pi_bind.string();
                relation.coefficient_object = pi_bind.string();
                relation.carrier_domain = pi_bind.string();
                relation.low_degree_length = pi_bind.count(
                    "Pi_bind coefficient basis low degree length");
                relation.tail_length = pi_bind.count(
                    "Pi_bind coefficient basis tail length");
                relation.total_length = pi_bind.count(
                    "Pi_bind coefficient basis total length");
                relation.semantic_positions = read_size_vector(
                    pi_bind, "Pi_bind coefficient basis semantic positions");
                const auto chunk_count = pi_bind.count(
                    "Pi_bind coefficient basis chunks");
                relation.chunks.reserve(chunk_count);
                for (std::size_t j = 0; j < chunk_count; ++j)
                {
                    relation.chunks.push_back(CoefficientBasisChunkMeta{
                        .object_id = pi_bind.string(),
                        .accumulator_object = pi_bind.string(),
                        .shifted_accumulator_object = pi_bind.string(),
                        .chunk_index = pi_bind.count(
                            "Pi_bind coefficient basis chunk index"),
                        .offset = pi_bind.count(
                            "Pi_bind coefficient basis chunk offset"),
                        .length = pi_bind.count(
                            "Pi_bind coefficient basis chunk length"),
                        .terminal_point = pi_bind.scalar(),
                        .configured_shift_degree = pi_bind.count(
                            "Pi_bind coefficient basis chunk shift degree"),
                        .shift_domain_size = pi_bind.count(
                            "Pi_bind coefficient basis chunk shift domain size"),
                    });
                }
                relation.power_object = pi_bind.string();
                relation.gamma_challenge_label = pi_bind.string();
                relation.relation_challenge_label = pi_bind.string();
                relation.commitment_order = read_string_vector(
                    pi_bind, "Pi_bind coefficient basis commitment order");
                relation.opening_claim_order = read_string_vector(
                    pi_bind, "Pi_bind coefficient basis opening order");
                meta.coefficient_basis_relations.push_back(std::move(relation));
            }
            const auto shift_relation_count = pi_bind.count("Pi_bind shift relations");
            meta.shift_relations.reserve(shift_relation_count);
            for (std::size_t i = 0; i < shift_relation_count; ++i)
            {
                KZGShiftRelationMeta relation;
                relation.relation_id = pi_bind.string();
                relation.source_linear_maps = read_string_vector(
                    pi_bind, "Pi_bind shift source maps");
                relation.source_domains = read_string_vector(
                    pi_bind, "Pi_bind shift source domains");
                const auto point_count = pi_bind.count("Pi_bind shift points");
                relation.canonical_points.reserve(point_count);
                for (std::size_t j = 0; j < point_count; ++j)
                    relation.canonical_points.push_back(pi_bind.scalar());
                relation.lambda0 = pi_bind.scalar();
                const auto delta_count = pi_bind.count("Pi_bind shift deltas");
                relation.deltas.reserve(delta_count);
                for (std::size_t j = 0; j < delta_count; ++j)
                    relation.deltas.push_back(pi_bind.scalar());
                relation.source_value_kinds = read_string_vector(
                    pi_bind, "Pi_bind shift source kinds");
                relation.challenge_label = pi_bind.string();
                relation.exclusion_degree = pi_bind.count("Pi_bind shift exclusion degree");
                relation.max_difference_degree = pi_bind.count("Pi_bind shift max degree");
                relation.configured_degrees = read_size_vector(
                    pi_bind, "Pi_bind shift configured degrees");
                relation.shift_domain_sizes = read_size_vector(
                    pi_bind, "Pi_bind shift domain sizes");
                relation.opening_claim_order = read_string_vector(
                    pi_bind, "Pi_bind shift opening order");
                meta.shift_relations.push_back(std::move(relation));
            }
            if (!(meta == expected_binding_metadata.at(subproof_index)))
                throw std::runtime_error(
                    "proof parser Pi_bind RelationMeta does not match instance schema");

            const auto model_count = pi_bind.count("Pi_bind model reblinds");
            subproof.model_reblinds.reserve(model_count);
            for (std::size_t i = 0; i < model_count; ++i)
            {
                ModelReblindProof reblind;
                reblind.object_id = pi_bind.string();
                reblind.domain = pi_bind.string();
                reblind.vk_model_anchor.object_id = pi_bind.string();
                reblind.proof_local_commitment.name = pi_bind.string();
                reblind.proof_local_commitment.point = pi_bind.point();
                reblind.blinder_commitment.name = pi_bind.string();
                reblind.blinder_commitment.point = pi_bind.point();
                subproof.model_reblinds.push_back(std::move(reblind));
            }
            const auto view_reference_count = pi_bind.count("Pi_bind Com_view references");
            subproof.view_references.reserve(view_reference_count);
            for (std::size_t i = 0; i < view_reference_count; ++i)
                subproof.view_references.push_back({pi_bind.string()});
            {
                std::unordered_set<std::string> references;
                for (const auto& reference : subproof.view_references)
                    if (reference.object_id.empty()
                        || !references.insert(reference.object_id).second)
                        throw std::runtime_error(
                            "proof parser duplicate/empty Pi_bind Com_view reference");
            }
            auto read_group = [&](auto& group, const char* label)
            {
                group = read_named_commitments(pi_bind);
                require_unique_names(group, label);
            };
            read_group(subproof.view_commitments, "Pi_bind Com_view");
            read_group(subproof.view_logup_commitments, "Pi_bind Com_viewLogUp");
            read_group(subproof.aggregation_commitments, "Pi_bind Com_agg");
            read_group(subproof.basis_commitments, "Pi_bind Com_basis");
            read_group(subproof.crpc_commitments, "Pi_bind Com_CRPC");
            read_group(subproof.relation_commitments, "Pi_bind Com_rel");

            const auto evaluation_count = pi_bind.count("Pi_bind evaluations");
            subproof.evaluations.reserve(evaluation_count);
            for (std::size_t i = 0; i < evaluation_count; ++i)
            {
                BindingEvaluation evaluation;
                evaluation.domain = pi_bind.string();
                evaluation.object_id = pi_bind.string();
                const auto value_count = pi_bind.count("Pi_bind evaluation values");
                evaluation.values.reserve(value_count);
                for (std::size_t j = 0; j < value_count; ++j)
                    evaluation.values.push_back(pi_bind.scalar());
                evaluation.kind = read_kind(pi_bind);
                subproof.evaluations.push_back(std::move(evaluation));
            }
            {
                std::unordered_set<std::string> evaluations;
                for (const auto& evaluation : subproof.evaluations)
                {
                    const auto key = evaluation.domain + "\n" + evaluation.object_id;
                    if (evaluation.domain.empty() || evaluation.object_id.empty()
                        || evaluation.values.empty()
                        || !evaluations.insert(key).second)
                        throw std::runtime_error(
                            "proof parser duplicate/empty Pi_bind evaluation");
                }
            }
            const auto reference_count = pi_bind.count("Pi_bind opening references");
            subproof.opening_references.reserve(reference_count);
            for (std::size_t i = 0; i < reference_count; ++i)
                subproof.opening_references.push_back({pi_bind.string()});
            {
                std::unordered_set<std::string> references;
                for (const auto& reference : subproof.opening_references)
                    if (reference.object_id.empty()
                        || !references.insert(reference.object_id).second)
                        throw std::runtime_error(
                            "proof parser duplicate/empty Pi_bind opening reference");
            }
            proof.pi_bind.subproofs.push_back(std::move(subproof));
        }
        pi_bind.finish("Pi_bind");
        body.finish("proof");

        // Reconstruct the in-memory global point groups after all physical
        // blocks have been parsed.  Eval_bind owns these values on the wire;
        // Open_dom owns their single folded witness.
        for (const auto& subproof : proof.pi_bind.subproofs)
        {
            for (const auto& evaluation : subproof.evaluations)
            {
                if (evaluation.domain == "relation") continue;
                const auto bundle_it = std::find_if(
                    proof.domain_openings.begin(), proof.domain_openings.end(),
                    [&](const auto& entry) { return entry.first == evaluation.domain; });
                if (bundle_it == proof.domain_openings.end())
                    throw std::runtime_error(
                        "proof parser Pi_bind evaluation references unknown domain");
                if (std::any_of(
                        bundle_it->second.values.begin(), bundle_it->second.values.end(),
                        [&](const auto& entry) { return entry.first == evaluation.object_id; }))
                    throw std::runtime_error(
                        "proof parser duplicate Eval_dom/Eval_bind claim owner");
                bundle_it->second.values.push_back(
                    {evaluation.object_id, evaluation.values});
                bundle_it->second.value_kinds.push_back(evaluation.kind);
            }
        }
        // Eval_dom omits values whose unique wire owner is Eval_bind.  Restore
        // the canonical global point-group order after merging both blocks;
        // appending Eval_bind values would place them after t_D/R_t_D and make
        // the parser disagree with the shared prover/verifier schema.
        for (auto& [bundle_name, bundle] : proof.domain_openings)
        {
            std::string trace_domain;
            std::string quotient_name;
            if (bundle_name == "FH")
            {
                trace_domain = "FH";
                quotient_name = "t_FH";
            }
            else if (bundle_name == "edge")
            {
                trace_domain = "edge";
                quotient_name = "t_edge";
            }
            else if (bundle_name == "in")
            {
                trace_domain = "in";
                quotient_name = "t_in";
            }
            else if (bundle_name == "d_h")
            {
                trace_domain = "d";
                quotient_name = "t_d_h";
            }
            else if (bundle_name == "cat")
            {
                trace_domain = "cat";
                quotient_name = "t_cat";
            }
            else if (bundle_name == "C")
            {
                trace_domain = "C";
                quotient_name = "t_C";
            }
            else if (bundle_name == "N")
            {
                trace_domain = "N";
                quotient_name = "t_N";
            }
            else
            {
                throw std::runtime_error(
                    "proof parser encountered unknown domain bundle: "
                    + bundle_name);
            }

            auto expected_labels = domain_opening_labels(context, trace_domain);
            expected_labels.push_back(quotient_name);
            expected_labels.push_back(quotient_mask_label(quotient_name));
            if (bundle.values.size() != expected_labels.size()
                || bundle.value_kinds.size() != bundle.values.size())
                throw std::runtime_error(
                    "proof parser reconstructed domain bundle shape mismatch: "
                    + bundle_name);

            std::unordered_map<std::string, std::size_t> positions;
            positions.reserve(bundle.values.size());
            for (std::size_t i = 0; i < bundle.values.size(); ++i)
                if (!positions.emplace(bundle.values[i].first, i).second)
                    throw std::runtime_error(
                        "proof parser reconstructed duplicate domain value: "
                        + bundle.values[i].first);

            std::vector<std::pair<std::string, std::vector<algebra::FieldElement>>>
                ordered_values;
            std::vector<PayloadValueKind> ordered_kinds;
            ordered_values.reserve(expected_labels.size());
            ordered_kinds.reserve(expected_labels.size());
            for (const auto& label : expected_labels)
            {
                const auto position = positions.find(label);
                if (position == positions.end())
                    throw std::runtime_error(
                        "proof parser reconstructed domain bundle is missing: "
                        + label);
                ordered_values.push_back(
                    std::move(bundle.values[position->second]));
                ordered_kinds.push_back(bundle.value_kinds[position->second]);
            }
            bundle.values = std::move(ordered_values);
            bundle.value_kinds = std::move(ordered_kinds);
        }

        std::unordered_map<std::string, crypto::Commitment> dynamic;
        std::unordered_map<std::string, crypto::Commitment> quotient;
        for (const auto& [name, commitment] : proof.dynamic_commitments) dynamic.emplace(name, commitment);
        for (const auto& subproof : proof.pi_bind.subproofs)
        {
            for (const auto& reblind : subproof.model_reblinds)
            {
                if (!dynamic.emplace(
                        reblind.proof_local_commitment.name,
                        reblind.proof_local_commitment).second
                    || !dynamic.emplace(
                        reblind.blinder_commitment.name,
                        reblind.blinder_commitment).second)
                {
                    throw std::runtime_error(
                        "proof parser duplicate Pi_bind model commitment reference");
                }
            }
            for (const auto* group : {
                     &subproof.view_commitments,
                     &subproof.view_logup_commitments,
                     &subproof.aggregation_commitments,
                     &subproof.basis_commitments,
                     &subproof.crpc_commitments,
                     &subproof.relation_commitments})
            {
                for (const auto& [name, commitment] : *group)
                {
                    if (name != commitment.name
                        || !dynamic.emplace(name, commitment).second)
                    {
                        throw std::runtime_error(
                            "proof parser duplicate or inconsistent Pi_bind commitment owner: "
                            + name);
                    }
                }
            }
        }
        for (const auto& [name, commitment] : proof.quotient_commitments) quotient.emplace(name, commitment);
        proof.challenges = replay_challenges(context, dynamic, quotient, &proof.public_metadata);
        const auto expected_route_references =
            route_opening_references(context, proof.challenges);
        if (proof.route_references != expected_route_references)
            throw std::runtime_error(
                "proof parser S_route reference registry mismatch");
        require_exact_pi_bind_shape(context, proof);
        return proof;
    }
}
