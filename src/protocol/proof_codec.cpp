#include "gatzk/protocol/proof_codec.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cstring>

#include "gatzk/crypto/curve.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/opening_registry.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/verifier_context.hpp"

namespace gatzk::protocol
{
    namespace
    {
        constexpr std::array<std::uint8_t, 7> kMagic{'Z', 'K', 'G', 'A', 'T', 'P', 'F'};
        constexpr std::uint64_t kMaxContainerItems = 1ULL << 24;
        constexpr std::uint64_t kMaxStringBytes = 1ULL << 20;

        struct EncodedG1View
        {
            const std::uint8_t *data = nullptr;
            std::size_t size = 0;
        };

        crypto::G1Point decode_canonical_g1(const EncodedG1View encoded)
        {
            if (encoded.data == nullptr || encoded.size == 0 || encoded.size > 256)
            {
                throw std::runtime_error("proof parser invalid G1 length");
            }

            auto point = crypto::deserialize_g1(encoded.data, encoded.size);

            std::array<std::uint8_t, 256> canonical{};

            const auto written = crypto::serialize(point, canonical.data(), canonical.size());

            if (written != encoded.size || std::memcmp(canonical.data(), encoded.data, encoded.size) != 0)
            {
                throw std::runtime_error("proof parser rejected non-canonical G1 encoding");
            }

            return point;
        }

        class Writer
        {
        public:
            void raw(const std::uint8_t *data, std::size_t size) { bytes_.insert(bytes_.end(), data, data + static_cast<std::ptrdiff_t>(size)); }

            void u64(std::uint64_t value)
            {
                for (std::size_t i = 0; i < 8; ++i)
                {
                    bytes_.push_back(static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU));
                }
            }

            void string(const std::string &value)
            {
                u64(value.size());
                raw(reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
            }

            void scalar(const algebra::FieldElement &value)
            {
                std::array<std::uint8_t, 32> encoded{};
                value.write_little_endian(encoded.data(), encoded.size());
                raw(encoded.data(), encoded.size());
            }

            void point(const crypto::G1Point &point)
            {
                std::array<std::uint8_t, 256> encoded{};

                const auto written = crypto::serialize(point, encoded.data(), encoded.size());

                u64(written);
                raw(encoded.data(), written);
            }

            void block(const std::string &tag, Writer payload)
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
            Reader(const std::uint8_t *data, std::size_t size) : data_(data), size_(size) {}

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

            std::size_t count(const char *label)
            {
                const auto value = u64();
                if (value > kMaxContainerItems || value > std::numeric_limits<std::size_t>::max())
                {
                    throw std::runtime_error(std::string("proof parser invalid count: ") + label);
                }
                return static_cast<std::size_t>(value);
            }

            std::size_t size(const char *label)
            {
                const auto value = u64();
                if (value > std::numeric_limits<std::size_t>::max())
                {
                    throw std::runtime_error(std::string("proof parser invalid size: ") + label);
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
                const auto *begin = reinterpret_cast<const char *>(data_ + offset_);
                offset_ += static_cast<std::size_t>(length);
                return std::string(begin, begin + static_cast<std::ptrdiff_t>(length));
            }

            algebra::FieldElement scalar()
            {
                constexpr std::size_t kScalarBytes = 32;
                require(kScalarBytes);

                const auto *begin = data_ + offset_;

                algebra::FieldElement value;

                try
                {
                    value = algebra::FieldElement::from_canonical_little_endian(begin, kScalarBytes);
                }
                catch (...)
                {
                    throw std::runtime_error("proof parser rejected non-canonical field encoding");
                }

                offset_ += kScalarBytes;
                return value;
            }

            EncodedG1View point_encoding()
            {
                const auto length = u64();

                if (length == 0 || length > 256 || length > remaining())
                {
                    throw std::runtime_error("proof parser invalid G1 length");
                }

                const auto encoded_size = static_cast<std::size_t>(length);

                const EncodedG1View encoded{
                    .data = data_ + offset_,
                    .size = encoded_size,
                };

                offset_ += encoded_size;
                return encoded;
            }

            crypto::G1Point point() { return decode_canonical_g1(point_encoding()); }

            Reader block(const std::string &expected_tag)
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

            void finish(const char *label) const
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

            const std::uint8_t *data_ = nullptr;
            std::size_t size_ = 0;
            std::size_t offset_ = 0;
        };

        std::vector<crypto::G1Point> decode_canonical_g1_batch(const std::vector<EncodedG1View> &encodings)
        {
            std::vector<crypto::G1Point> points(encodings.size());

            if (encodings.empty())
            {
                return points;
            }

            constexpr std::size_t kMinimumPointsPerWorker = 16;

            constexpr std::size_t kMaximumDecodeWorkers = 32;

            const auto hardware_workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());

            const auto size_workers = std::max<std::size_t>(1, (encodings.size() + kMinimumPointsPerWorker - 1) / kMinimumPointsPerWorker);

            const auto worker_count = std::min<std::size_t>({
                hardware_workers,
                kMaximumDecodeWorkers,
                size_workers,
            });

            const auto decode_range = [&](std::size_t begin, std::size_t end)
            {
                for (std::size_t index = begin; index < end; ++index)
                {
                    points[index] = decode_canonical_g1(encodings[index]);
                }
            };

            if (worker_count <= 1)
            {
                decode_range(0, encodings.size());

                return points;
            }

            const auto chunk_size = (encodings.size() + worker_count - 1) / worker_count;

            std::vector<std::future<void>> futures;

            futures.reserve(worker_count);

            for (std::size_t worker = 0; worker < worker_count; ++worker)
            {
                const auto begin = worker * chunk_size;

                const auto end = std::min(encodings.size(), begin + chunk_size);

                if (begin >= end)
                {
                    break;
                }

                futures.push_back(std::async(std::launch::async, decode_range, begin, end));
            }

            for (auto &future : futures)
            {
                future.get();
            }

            return points;
        }

        template <typename Fn> void metadata_fields(PublicMetadata &metadata, Fn &&fn)
        {
            fn(metadata.protocol_name);
            fn(metadata.proof_nonce);
            fn(metadata.proof_task);
            fn(metadata.srs_id);
            fn(metadata.protocol_schema_id);
            fn(metadata.instance_schema_hash);
            fn(metadata.srs_type);
            fn(metadata.srs_degree_bound);
            fn(metadata.kzg_assumption_profile);
            fn(metadata.privacy_policy);
            fn(metadata.input_privacy);
            fn(metadata.output_mode);
            fn(metadata.opening_mode);
            fn(metadata.binding_mode);
            fn(metadata.model_arch);
            fn(metadata.model_params);
            fn(metadata.input_preprocess_id);
            fn(metadata.static_tables);
            fn(metadata.quant_config);
            fn(metadata.domain_cfg);
            fn(metadata.dim_cfg);
            fn(metadata.encoding);
            fn(metadata.padding_rule);
            fn(metadata.degree_bound);
            fn(metadata.security_budget);
            fn(metadata.public_output);
        }

        void write_point_group_opening(Writer &writer, const PointGroupOpening &opening)
        {
            writer.string(opening.group_id);
            writer.string(opening.owner_block);
            writer.scalar(opening.point);
            writer.point(opening.witness);
        }

        struct EncodedPointGroupOpening
        {
            std::string group_id;
            std::string owner_block;
            algebra::FieldElement point = algebra::FieldElement::zero();
            EncodedG1View witness_encoding;
        };

        EncodedPointGroupOpening read_encoded_point_group_opening(Reader &reader)
        {
            return EncodedPointGroupOpening{
                .group_id = reader.string(),
                .owner_block = reader.string(),
                .point = reader.scalar(),
                .witness_encoding = reader.point_encoding(),
            };
        }

        std::vector<PointGroupOpening> decode_point_group_openings(std::vector<EncodedPointGroupOpening> encoded)
        {
            std::vector<EncodedG1View> encodings;
            encodings.reserve(encoded.size());
            for (const auto &opening : encoded)
                encodings.push_back(opening.witness_encoding);
            auto witnesses = decode_canonical_g1_batch(encodings);
            std::vector<PointGroupOpening> openings;
            openings.reserve(encoded.size());
            for (std::size_t i = 0; i < encoded.size(); ++i)
            {
                openings.push_back(PointGroupOpening{
                    .group_id = std::move(encoded[i].group_id),
                    .owner_block = std::move(encoded[i].owner_block),
                    .point = encoded[i].point,
                    .witness = std::move(witnesses[i]),
                });
            }
            return openings;
        }

        PayloadValueKind read_kind(Reader &reader)
        {
            const auto raw = reader.u64();
            if (raw > static_cast<std::uint64_t>(PayloadValueKind::MaskedPrivateValue))
            {
                throw std::runtime_error("proof parser invalid payload kind");
            }
            return static_cast<PayloadValueKind>(raw);
        }

        void write_named_commitments(Writer &writer, const std::vector<std::pair<std::string, crypto::Commitment>> &commitments)
        {
            writer.u64(commitments.size());
            for (const auto &[name, commitment] : commitments)
            {
                writer.string(name);
                writer.point(commitment.point);
            }
        }

        struct EncodedNamedCommitments
        {
            std::vector<std::string> names;
            std::vector<EncodedG1View> encodings;
        };

        struct EncodedBindingCommitments
        {
            std::vector<EncodedG1View> model_reblind_encodings;

            EncodedNamedCommitments view_commitments;

            EncodedNamedCommitments view_logup_commitments;

            EncodedNamedCommitments aggregation_commitments;

            EncodedNamedCommitments basis_commitments;

            EncodedNamedCommitments crpc_commitments;

            EncodedNamedCommitments relation_commitments;
        };

        EncodedNamedCommitments read_encoded_named_commitments(Reader &reader)
        {
            const auto count = reader.count("commitments");

            EncodedNamedCommitments encoded;

            encoded.names.reserve(count);

            encoded.encodings.reserve(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                encoded.names.push_back(reader.string());

                encoded.encodings.push_back(reader.point_encoding());
            }

            return encoded;
        }

        std::vector<std::pair<std::string, crypto::Commitment>> decode_named_commitments(EncodedNamedCommitments encoded)
        {
            if (encoded.names.size() != encoded.encodings.size())
            {
                throw std::runtime_error("proof parser encoded commitment shape mismatch");
            }

            auto points = decode_canonical_g1_batch(encoded.encodings);

            std::vector<std::pair<std::string, crypto::Commitment>> out;

            out.reserve(encoded.names.size());

            for (std::size_t i = 0; i < encoded.names.size(); ++i)
            {
                crypto::Commitment commitment;

                commitment.name = encoded.names[i];

                commitment.point = std::move(points[i]);

                out.push_back({
                    std::move(encoded.names[i]),
                    std::move(commitment),
                });
            }

            return out;
        }

        std::vector<std::pair<std::string, crypto::Commitment>> read_named_commitments(Reader &reader)
        {
            return decode_named_commitments(read_encoded_named_commitments(reader));
        }

        void write_optional_index(Writer &writer, const std::optional<std::size_t> &value)
        {
            writer.u64(value.has_value() ? 1 : 0);
            if (value.has_value())
                writer.u64(*value);
        }

        std::optional<std::size_t> read_optional_index(Reader &reader, const char *label)
        {
            const auto present = reader.u64();
            if (present > 1)
                throw std::runtime_error(std::string("proof parser invalid optional flag: ") + label);
            if (present == 0)
                return std::nullopt;
            const auto value = reader.u64();
            if (value > std::numeric_limits<std::size_t>::max())
                throw std::runtime_error(std::string("proof parser invalid optional index: ") + label);
            return static_cast<std::size_t>(value);
        }

        void write_string_vector(Writer &writer, const std::vector<std::string> &values)
        {
            writer.u64(values.size());
            for (const auto &value : values)
                writer.string(value);
        }

        std::vector<std::string> read_string_vector(Reader &reader, const char *label)
        {
            std::vector<std::string> out;
            const auto count = reader.count(label);
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
                out.push_back(reader.string());
            return out;
        }

        void write_size_vector(Writer &writer, const std::vector<std::size_t> &values)
        {
            writer.u64(values.size());
            for (const auto value : values)
                writer.u64(value);
        }

        std::vector<std::size_t> read_size_vector(Reader &reader, const char *label)
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

        template <typename NamedRange> void require_unique_names(const NamedRange &values, const char *label)
        {
            std::unordered_set<std::string> names;
            names.reserve(values.size());
            for (const auto &item : values)
            {
                if (item.first.empty() || !names.insert(item.first).second)
                {
                    throw std::runtime_error(std::string("proof parser duplicate/empty name in ") + label);
                }
            }
        }

        void require_structural_point_group_layout(const Proof &proof)
        {
            std::unordered_set<std::string> unique_ids;
            std::unordered_set<std::string> unique_points;
            std::unordered_set<std::string> expected_domain_ids;
            for (const auto &[domain, bundle] : proof.domain_openings)
            {
                for (const auto &point : bundle.points)
                {
                    expected_domain_ids.insert(point_group_id("Open_dom:" + domain, point));
                }
            }
            if (!std::is_sorted(proof.opening_groups.begin(), proof.opening_groups.end(),
                                [](const PointGroupOpening &lhs, const PointGroupOpening &rhs) { return lhs.point.to_string() < rhs.point.to_string(); }))
            {
                throw std::runtime_error("proof parser point groups are not in canonical point order");
            }
            for (const auto &opening : proof.opening_groups)
            {
                if (opening.group_id != point_group_id(opening.owner_block, opening.point) || !unique_ids.insert(opening.group_id).second ||
                    !unique_points.insert(opening.point.to_string()).second)
                {
                    throw std::runtime_error("proof parser non-canonical or duplicate point group");
                }
                if (opening.owner_block.rfind("Open_dom:", 0) == 0)
                {
                    if (!expected_domain_ids.contains(opening.group_id))
                    {
                        throw std::runtime_error("proof parser unregistered domain point-group owner");
                    }
                }
                else if (opening.owner_block != "W_ext")
                {
                    throw std::runtime_error("proof parser invalid point-group owner block");
                }
            }
            for (const auto &[domain, bundle] : proof.domain_openings)
            {
                for (const auto &point : bundle.points)
                {
                    const auto expected_id = point_group_id("Open_dom:" + domain, point);
                    const auto count = std::count_if(proof.opening_groups.begin(), proof.opening_groups.end(),
                                                     [&](const PointGroupOpening &opening) { return opening.group_id == expected_id; });
                    if (count != 1)
                    {
                        throw std::runtime_error("proof parser domain point lacks its unique owner/witness");
                    }
                }
            }
            const auto actual_domain_count =
                static_cast<std::size_t>(std::count_if(proof.opening_groups.begin(), proof.opening_groups.end(),
                                                       [](const PointGroupOpening &opening) { return opening.owner_block.rfind("Open_dom:", 0) == 0; }));
            if (actual_domain_count != expected_domain_ids.size())
            {
                throw std::runtime_error("proof parser domain point-group owner count mismatch");
            }
        }

        bool proof_contains_prover_material(const Proof &proof)
        {
            const auto has_material = [](const crypto::Commitment &commitment) { return commitment.opening_polynomial != nullptr; };
            for (const auto &[name, commitment] : proof.dynamic_commitments)
            {
                (void)name;
                if (has_material(commitment))
                    return true;
            }
            for (const auto &[name, commitment] : proof.quotient_commitments)
            {
                (void)name;
                if (has_material(commitment))
                    return true;
            }
            for (const auto &subproof : proof.pi_bind.subproofs)
            {
                for (const auto &reblind : subproof.model_reblinds)
                    if (has_material(reblind.proof_local_commitment) || has_material(reblind.blinder_commitment))
                        return true;
                for (const auto *group : {&subproof.view_commitments, &subproof.view_logup_commitments, &subproof.aggregation_commitments,
                                          &subproof.basis_commitments, &subproof.crpc_commitments, &subproof.relation_commitments})
                    for (const auto &[name, commitment] : *group)
                    {
                        (void)name;
                        if (has_material(commitment))
                            return true;
                    }
            }
            return false;
        }

        void require_exact_pi_bind_shape(const ProtocolContext &context, const Proof &proof, const std::vector<BindingRelationMeta> &expected_meta)
        {
            if (proof.pi_bind.subproofs.size() != expected_meta.size())
                throw std::runtime_error("proof parser Pi_bind instance subproof count mismatch");

            for (std::size_t index = 0; index < expected_meta.size(); ++index)
            {
                const auto &subproof = proof.pi_bind.subproofs[index];
                const auto &meta = expected_meta[index];
                if (!(subproof.relation_meta == meta))
                    throw std::runtime_error("proof parser Pi_bind instance RelationMeta mismatch");

                std::vector<std::pair<std::string, std::string>> expected_models;
                if (meta.object_kind == "hidden")
                {
                    const auto layer = meta.layer_id.value() - 1;
                    const auto head = meta.head_id.value();
                    const auto width = context.config.hidden_profile.at(layer).head_dim;
                    for (std::size_t column = 0; column < width; ++column)
                        expected_models.push_back({hidden_weight_column_label(layer, head, column), context.domains.in->name});
                    expected_models.push_back({hidden_src_label(layer, head), context.domains.d->name});
                    expected_models.push_back({hidden_dst_label(layer, head), context.domains.d->name});
                }
                else if (meta.object_kind == "output")
                {
                    const auto head = meta.head_id.value();
                    for (std::size_t column = 0; column < context.local.num_classes; ++column)
                        expected_models.push_back({output_weight_column_label(head, column), context.domains.cat->name});
                    expected_models.push_back({output_src_label(head), context.domains.c->name});
                    expected_models.push_back({output_dst_label(head), context.domains.c->name});
                    expected_models.push_back({output_bias_label(head), context.domains.c->name});
                }
                if (subproof.model_reblinds.size() != expected_models.size())
                    throw std::runtime_error("proof parser Pi_bind model-reblind count mismatch");
                for (std::size_t model = 0; model < expected_models.size(); ++model)
                {
                    const auto &[anchor, domain] = expected_models[model];
                    const auto &actual = subproof.model_reblinds[model];
                    if (actual.object_id != anchor || actual.domain != domain || actual.vk_model_anchor.object_id != anchor ||
                        actual.proof_local_commitment.name != model_proof_local_label(anchor) ||
                        actual.blinder_commitment.name != model_reblind_witness_label(anchor) || actual.blinder_commitment.point == crypto::g1_zero())
                    {
                        throw std::runtime_error("proof parser Pi_bind model-reblind shape mismatch");
                    }
                }

                std::vector<std::string> expected_canonical_references;
                std::vector<std::string> expected_canonical_commitments;
                std::vector<std::tuple<std::string, std::string, std::size_t, PayloadValueKind>> expected_canonical_evaluations;
                for (const auto &matrix : meta.canonical_matrices)
                {
                    expected_canonical_references.push_back(matrix.matrix_object);
                    for (const auto &column : matrix.column_objects)
                    {
                        expected_canonical_commitments.push_back(column);
                        expected_canonical_evaluations.push_back({column, "N", 2, PayloadValueKind::MaskedPrivateValue});
                    }
                    expected_canonical_commitments.push_back(matrix.compressed_object);
                    expected_canonical_evaluations.push_back({matrix.compressed_object, "N", 2, PayloadValueKind::MaskedPrivateValue});
                }
                const auto canonical_references_match = [&]
                {
                    return subproof.view_references.size() >= expected_canonical_references.size() &&
                           std::equal(subproof.view_references.begin(),
                                      subproof.view_references.begin() + static_cast<std::ptrdiff_t>(expected_canonical_references.size()),
                                      expected_canonical_references.begin(),
                                      [](const MetadataReference &actual, const std::string &expected) { return actual.object_id == expected; });
                };
                const auto canonical_commitments_match = [&]
                {
                    return subproof.view_commitments.size() == expected_canonical_commitments.size() &&
                           std::equal(subproof.view_commitments.begin(), subproof.view_commitments.end(), expected_canonical_commitments.begin(),
                                      [](const auto &actual, const std::string &expected) { return actual.first == expected; });
                };
                std::vector<std::string> expected_aggregation_commitments;
                std::vector<std::tuple<std::string, std::string, std::size_t, PayloadValueKind>> expected_aggregation_relation_evaluations;
                std::vector<std::string> expected_route_commitments;
                for (const auto &route : meta.route_terminal_relations)
                {
                    expected_route_commitments.push_back(route.shifted_accumulator_object);
                    expected_aggregation_relation_evaluations.push_back(
                        {route.shifted_accumulator_object + ":u_rel", "relation", 1, PayloadValueKind::MaskedPrivateValue});
                }
                std::vector<std::tuple<std::string, std::string, std::size_t, PayloadValueKind>> expected_aggregation_domain_evaluations;
                std::vector<std::string> expected_aggregation_openings;
                std::unordered_set<std::string> expected_aggregation_opening_ids;
                const auto append_aggregation_opening = [&](std::string value)
                {
                    if (expected_aggregation_opening_ids.insert(value).second)
                    {
                        expected_aggregation_openings.push_back(std::move(value));
                    }
                };
                for (const auto &route : meta.route_terminal_relations)
                {
                    const auto challenge = proof.challenges.find(route.challenge_label);
                    if (challenge == proof.challenges.end())
                        throw std::runtime_error("proof parser Pi_bind missing route terminal challenge");
                    append_aggregation_opening(opening_group_reference(context, proof.challenges, challenge->second));
                    append_aggregation_opening(opening_group_reference(context, proof.challenges, challenge->second + route.terminal_delta));
                    append_aggregation_opening(opening_group_reference(context, proof.challenges, route.node_terminal_point));
                }
                for (const auto &aggregation : meta.aggregation_relations)
                {
                    expected_aggregation_commitments.insert(expected_aggregation_commitments.end(), aggregation.commitment_order.begin(),
                                                            aggregation.commitment_order.end());
                    for (const auto *route : {&aggregation.source_route, &aggregation.target_route})
                    {
                        expected_aggregation_relation_evaluations.push_back(
                            {route->shifted_accumulator_object + ":u_rel", "relation", 1, PayloadValueKind::MaskedPrivateValue});
                        const auto challenge = proof.challenges.find(route->terminal_challenge_label);
                        if (challenge == proof.challenges.end())
                            throw std::runtime_error("proof parser Pi_bind missing aggregation terminal challenge");
                        append_aggregation_opening(opening_group_reference(context, proof.challenges, challenge->second));
                        append_aggregation_opening(opening_group_reference(context, proof.challenges, challenge->second + route->terminal_delta));
                        append_aggregation_opening(opening_group_reference(context, proof.challenges, route->node_terminal_point));
                    }
                    for (const auto *object : {&aggregation.source_node_object, &aggregation.target_node_object})
                        expected_aggregation_domain_evaluations.push_back({*object, aggregation.node_domain, 2, PayloadValueKind::MaskedPrivateValue});
                    for (const auto *route : {&aggregation.source_route, &aggregation.target_route})
                        for (const auto *object : {&route->table_object, &route->node_inverse_object, &route->node_accumulator_object})
                            expected_aggregation_domain_evaluations.push_back({*object, aggregation.node_domain, 2, PayloadValueKind::MaskedPrivateValue});
                    for (const auto *object :
                         {&aggregation.source_edge_object, &aggregation.target_edge_object, &aggregation.product_object, &aggregation.psq_object})
                        expected_aggregation_domain_evaluations.push_back({*object, aggregation.edge_domain, 2, PayloadValueKind::MaskedPrivateValue});
                    for (const auto *route : {&aggregation.source_route, &aggregation.target_route})
                        for (const auto *object : {&route->query_object, &route->edge_inverse_object, &route->edge_accumulator_object})
                            expected_aggregation_domain_evaluations.push_back({*object, aggregation.edge_domain, 2, PayloadValueKind::MaskedPrivateValue});
                }
                const bool aggregation_commitments_match =
                    subproof.aggregation_commitments.size() == expected_aggregation_commitments.size() &&
                    std::equal(subproof.aggregation_commitments.begin(), subproof.aggregation_commitments.end(), expected_aggregation_commitments.begin(),
                               [](const auto &actual, const std::string &expected) { return actual.first == expected; });
                if (meta.object_kind != "output")
                {
                    auto expected_evaluations = expected_aggregation_relation_evaluations;
                    expected_evaluations.insert(expected_evaluations.end(), expected_canonical_evaluations.begin(), expected_canonical_evaluations.end());
                    expected_evaluations.insert(expected_evaluations.end(), expected_aggregation_domain_evaluations.begin(),
                                                expected_aggregation_domain_evaluations.end());
                    const bool evaluations_match = subproof.evaluations.size() == expected_evaluations.size() &&
                                                   std::equal(subproof.evaluations.begin(), subproof.evaluations.end(), expected_evaluations.begin(),
                                                              [](const BindingEvaluation &actual, const auto &expected)
                                                              {
                                                                  return actual.object_id == std::get<0>(expected) && actual.domain == std::get<1>(expected) &&
                                                                         actual.values.size() == std::get<2>(expected) && actual.kind == std::get<3>(expected);
                                                              });
                    auto expected_openings = expected_aggregation_openings;
                    if (!meta.canonical_matrices.empty())
                    {
                        const auto point = proof.challenges.at("z_N");
                        expected_openings.push_back("Open_dom:N:" + point.to_string());
                        expected_openings.push_back("Open_dom:N:" + (point * context.domains.n->omega).to_string());
                    }
                    if (!meta.aggregation_relations.empty())
                    {
                        const auto point = proof.challenges.at("z_edge");
                        expected_openings.push_back("Open_dom:edge:" + point.to_string());
                        expected_openings.push_back("Open_dom:edge:" + (point * context.domains.edge->omega).to_string());
                    }
                    const bool opening_references_match =
                        subproof.opening_references.size() == expected_openings.size() &&
                        std::equal(subproof.opening_references.begin(), subproof.opening_references.end(), expected_openings.begin(),
                                   [](const MetadataReference &actual, const std::string &expected) { return actual.object_id == expected; });
                    if (!canonical_references_match() || subproof.view_references.size() != expected_canonical_references.size() ||
                        !canonical_commitments_match() || !subproof.view_logup_commitments.empty() || !aggregation_commitments_match ||
                        !subproof.basis_commitments.empty() || !subproof.crpc_commitments.empty() ||
                        subproof.relation_commitments.size() != expected_route_commitments.size() ||
                        !std::equal(subproof.relation_commitments.begin(), subproof.relation_commitments.end(), expected_route_commitments.begin(),
                                    [](const auto &actual, const std::string &expected) { return actual.first == expected; }) ||
                        !evaluations_match || !opening_references_match)
                    {
                        throw std::runtime_error("proof parser Pi_bind unexpected non-output payload");
                    }
                    continue;
                }

                const auto head = meta.head_id.value();
                if (meta.indexed_views.size() != 1)
                    throw std::runtime_error("proof parser Pi_bind output relation shape mismatch");
                const auto &view = meta.indexed_views.front();
                const auto *relation_ptr = find_unique_shift_relation_by_kind(meta, "output-bias-terminal");
                if (relation_ptr == nullptr)
                    throw std::runtime_error("proof parser Pi_bind output bias relation is not unique");
                const auto &relation = *relation_ptr;
                const bool same_point = relation.configured_degrees.empty();
                if (meta.coefficient_basis_relations.size() != 1)
                    throw std::runtime_error("proof parser Pi_bind coefficient basis relation is not unique");
                const auto &basis = meta.coefficient_basis_relations.front();
                auto expected_view_references = expected_canonical_references;
                expected_view_references.insert(expected_view_references.end(), {
                                                                                    view.canonical_object,
                                                                                    view.view_object,
                                                                                    "public:" + view.canonical_selector_id,
                                                                                    "public:" + view.query_selector_id,
                                                                                });
                if (subproof.view_references.size() != expected_view_references.size() ||
                    !std::equal(subproof.view_references.begin(), subproof.view_references.end(), expected_view_references.begin(),
                                [](const MetadataReference &actual, const std::string &expected) { return actual.object_id == expected; }))
                {
                    throw std::runtime_error("proof parser Pi_bind Com_view reference shape mismatch");
                }
                if (!canonical_commitments_match() || !aggregation_commitments_match || !subproof.crpc_commitments.empty() ||
                    subproof.basis_commitments.size() != basis.commitment_order.size() ||
                    !std::equal(subproof.basis_commitments.begin(), subproof.basis_commitments.end(), basis.commitment_order.begin(),
                                [](const auto &actual, const std::string &expected) { return actual.first == expected; }) ||
                    subproof.view_logup_commitments.size() != view.commitment_order.size() ||
                    !std::equal(subproof.view_logup_commitments.begin(), subproof.view_logup_commitments.end(), view.commitment_order.begin(),
                                [](const auto &actual, const std::string &expected) { return actual.first == expected; }) ||
                    subproof.relation_commitments.size() != expected_route_commitments.size() + basis.chunks.size() + (same_point ? 0 : 1) ||
                    !std::equal(subproof.relation_commitments.begin(),
                                subproof.relation_commitments.begin() + static_cast<std::ptrdiff_t>(expected_route_commitments.size()),
                                expected_route_commitments.begin(), [](const auto &actual, const std::string &expected) { return actual.first == expected; }) ||
                    (!same_point && subproof.relation_commitments[expected_route_commitments.size()].first != output_bias_view_label(head, "R_shifted")) ||
                    !std::equal(subproof.relation_commitments.end() - static_cast<std::ptrdiff_t>(basis.chunks.size()), subproof.relation_commitments.end(),
                                basis.chunks.begin(), [](const auto &actual, const auto &chunk) { return actual.first == chunk.shifted_accumulator_object; }))
                {
                    throw std::runtime_error("proof parser Pi_bind commitment group shape mismatch");
                }

                std::vector<std::tuple<std::string, std::string, std::size_t, PayloadValueKind>> expected_evaluations = {
                    {output_bias_view_label(head, "m_can") + ":public_bind", "relation", 1, PayloadValueKind::PublicValue},
                    {model_proof_local_label(output_bias_label(head)), "C", 2, PayloadValueKind::MaskedPrivateValue},
                    {output_bias_view_label(head, "Table_can"), "C", 2, PayloadValueKind::MaskedPrivateValue},
                    {output_bias_view_label(head, "m_can"), "C", 2, PayloadValueKind::PublicValue},
                    {output_bias_view_label(head, "J_can"), "C", 2, PayloadValueKind::MaskedPrivateValue},
                    {output_bias_view_label(head, "R_can"), "C", 2, PayloadValueKind::MaskedPrivateValue},
                    {output_bias_view_label(head, "Query"), "edge", 2, PayloadValueKind::MaskedPrivateValue},
                    {output_bias_view_label(head, "J_qry"), "edge", 2, PayloadValueKind::MaskedPrivateValue},
                    {output_bias_view_label(head, "R_qry"), "edge", 2, PayloadValueKind::MaskedPrivateValue},
                };
                expected_evaluations.insert(expected_evaluations.begin(), expected_aggregation_relation_evaluations.begin(),
                                            expected_aggregation_relation_evaluations.end());
                if (!same_point)
                    expected_evaluations.insert(expected_evaluations.begin() + static_cast<std::ptrdiff_t>(expected_aggregation_relation_evaluations.size()),
                                                {output_bias_view_label(head, "R_shifted") + ":u_rel", "relation", 1, PayloadValueKind::MaskedPrivateValue});
                auto basis_eval_position =
                    expected_evaluations.begin() + static_cast<std::ptrdiff_t>(expected_aggregation_relation_evaluations.size() + (same_point ? 1 : 2));
                for (const auto &chunk : basis.chunks)
                    basis_eval_position = expected_evaluations.insert(basis_eval_position, {chunk.shifted_accumulator_object + ":u_rel", "relation", 1,
                                                                                            PayloadValueKind::MaskedPrivateValue}) +
                                          1;
                for (const auto &chunk : basis.chunks)
                {
                    expected_evaluations.push_back({chunk.object_id, "edge", 2, PayloadValueKind::MaskedPrivateValue});
                    expected_evaluations.push_back({chunk.accumulator_object, "edge", 2, PayloadValueKind::MaskedPrivateValue});
                }
                expected_evaluations.insert(expected_evaluations.end(), expected_canonical_evaluations.begin(), expected_canonical_evaluations.end());
                expected_evaluations.insert(expected_evaluations.end(), expected_aggregation_domain_evaluations.begin(),
                                            expected_aggregation_domain_evaluations.end());
                if (subproof.evaluations.size() != expected_evaluations.size() ||
                    !std::equal(subproof.evaluations.begin(), subproof.evaluations.end(), expected_evaluations.begin(),
                                [](const BindingEvaluation &actual, const auto &expected)
                                {
                                    return actual.object_id == std::get<0>(expected) && actual.domain == std::get<1>(expected) &&
                                           actual.values.size() == std::get<2>(expected) && actual.kind == std::get<3>(expected);
                                }))
                {
                    throw std::runtime_error("proof parser Pi_bind evaluation shape mismatch");
                }

                const auto challenge_it = proof.challenges.find(relation.challenge_label);
                if (challenge_it == proof.challenges.end())
                    throw std::runtime_error("proof parser Pi_bind missing relation challenge");
                std::vector<std::string> expected_opening_references = expected_aggregation_openings;
                const auto append_unique = [&](std::string reference)
                {
                    if (std::find(expected_opening_references.begin(), expected_opening_references.end(), reference) == expected_opening_references.end())
                        expected_opening_references.push_back(std::move(reference));
                };
                const auto y_rel = challenge_it->second;
                if (same_point)
                {
                    append_unique(opening_group_reference(context, proof.challenges, y_rel));
                    append_unique(opening_group_reference(context, proof.challenges, relation.canonical_points.front()));
                }
                else
                {
                    append_unique(opening_group_reference(context, proof.challenges, y_rel));
                    append_unique(opening_group_reference(context, proof.challenges, y_rel + relation.deltas.at(1)));
                    append_unique(opening_group_reference(context, proof.challenges, relation.canonical_points.at(0)));
                }
                const auto gamma_it = proof.challenges.find(basis.gamma_challenge_label);
                const auto basis_y_it = proof.challenges.find(basis.relation_challenge_label);
                if (gamma_it == proof.challenges.end() || basis_y_it == proof.challenges.end())
                    throw std::runtime_error("proof parser Pi_bind missing coefficient basis challenge");
                append_unique(opening_group_reference(context, proof.challenges, basis_y_it->second));
                for (const auto &chunk : basis.chunks)
                    append_unique(opening_group_reference(context, proof.challenges, basis_y_it->second + chunk.terminal_point - gamma_it->second));
                append_unique(opening_group_reference(context, proof.challenges, gamma_it->second));
                for (const auto &[domain, point, omega] : std::vector<std::tuple<std::string, algebra::FieldElement, algebra::FieldElement>>{
                         {"C", proof.challenges.at("z_C"), context.domains.c->omega}, {"edge", proof.challenges.at("z_edge"), context.domains.edge->omega}})
                {
                    append_unique("Open_dom:" + domain + ":" + point.to_string());
                    append_unique("Open_dom:" + domain + ":" + (point * omega).to_string());
                }
                const auto n_point = proof.challenges.at("z_N");
                append_unique("Open_dom:N:" + n_point.to_string());
                append_unique("Open_dom:N:" + (n_point * context.domains.n->omega).to_string());
                if (subproof.opening_references.size() != expected_opening_references.size() ||
                    !std::equal(subproof.opening_references.begin(), subproof.opening_references.end(), expected_opening_references.begin(),
                                [](const MetadataReference &actual, const std::string &expected) { return actual.object_id == expected; }))
                {
                    throw std::runtime_error("proof parser Pi_bind opening-reference shape mismatch");
                }
            }
        }
    }

    std::vector<std::uint8_t> serialize_proof(const Proof &proof)
    {
        if (proof_contains_prover_material(proof))
            throw std::runtime_error("proof serializer refuses prover-only opening polynomial material");
        require_structural_point_group_layout(proof);
        Writer wire;
        wire.raw(kMagic.data(), kMagic.size());
        Writer m_pub;
        auto metadata = proof.public_metadata;
        metadata_fields(metadata, [&](const std::string &field) { m_pub.string(field); });
        m_pub.u64(proof.block_order.size());
        for (const auto &tag : proof.block_order)
            m_pub.string(tag);
        m_pub.u64(proof.public_logits.size());
        for (const auto &value : proof.public_logits)
            m_pub.scalar(value);
        wire.block("M_pub", std::move(m_pub));

        Writer com_dyn;
        write_named_commitments(com_dyn, proof.dynamic_commitments);
        wire.block("Com_dyn", std::move(com_dyn));

        Writer s_route;
        s_route.u64(proof.route_references.size());
        for (const auto &reference : proof.route_references)
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
        for (const auto &subproof : proof.pi_bind.subproofs)
            for (const auto &evaluation : subproof.evaluations)
                if (evaluation.domain != "relation")
                    pi_bind_domain_evaluations.insert(evaluation.object_id);

        Writer eval_dom;
        eval_dom.u64(proof.domain_openings.size());
        for (const auto &[name, bundle] : proof.domain_openings)
        {
            eval_dom.string(name);
            eval_dom.u64(bundle.points.size());
            for (const auto &point : bundle.points)
                eval_dom.scalar(point);
            const auto transmitted_count =
                std::count_if(bundle.values.begin(), bundle.values.end(), [&](const auto &entry) { return !pi_bind_domain_evaluations.contains(entry.first); });
            eval_dom.u64(transmitted_count);
            for (std::size_t i = 0; i < bundle.values.size(); ++i)
            {
                if (pi_bind_domain_evaluations.contains(bundle.values[i].first))
                    continue;
                eval_dom.string(bundle.values[i].first);
                eval_dom.u64(bundle.values[i].second.size());
                for (const auto &value : bundle.values[i].second)
                    eval_dom.scalar(value);
                eval_dom.u64(static_cast<std::uint64_t>(bundle.value_kinds.at(i)));
            }
        }
        wire.block("Eval_dom", std::move(eval_dom));

        Writer com_quot;
        write_named_commitments(com_quot, proof.quotient_commitments);
        wire.block("Com_quot", std::move(com_quot));

        Writer open_dom;
        open_dom.u64(proof.domain_openings.size());
        for (const auto &[name, bundle] : proof.domain_openings)
        {
            open_dom.string(name);
            open_dom.u64(bundle.points.size());
            for (const auto &point : bundle.points)
            {
                const auto group_id = point_group_id("Open_dom:" + name, point);
                const auto opening = std::find_if(proof.opening_groups.begin(), proof.opening_groups.end(),
                                                  [&](const PointGroupOpening &candidate) { return candidate.group_id == group_id; });
                if (opening == proof.opening_groups.end())
                {
                    throw std::runtime_error("proof serializer missing domain point-group witness");
                }
                write_point_group_opening(open_dom, *opening);
            }
        }
        wire.block("Open_dom", std::move(open_dom));

        Writer w_ext;
        const auto external_count = std::count_if(proof.opening_groups.begin(), proof.opening_groups.end(),
                                                  [](const PointGroupOpening &opening) { return opening.owner_block == "W_ext"; });
        w_ext.u64(external_count);
        for (const auto &opening : proof.opening_groups)
        {
            if (opening.owner_block == "W_ext")
                write_point_group_opening(w_ext, opening);
        }
        wire.block("W_ext", std::move(w_ext));

        Writer pi_bind;
        pi_bind.u64(proof.pi_bind.subproofs.size());
        for (const auto &subproof : proof.pi_bind.subproofs)
        {
            const auto &meta = subproof.relation_meta;
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
            for (const auto &matrix : meta.canonical_matrices)
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
            const auto write_aggregation_route = [&](const auto &route)
            {
                for (const auto *value :
                     {&route.relation_id, &route.index_object, &route.node_value_object, &route.edge_value_object, &route.table_object, &route.query_object,
                      &route.multiplicity_object, &route.node_inverse_object, &route.edge_inverse_object, &route.node_accumulator_object,
                      &route.edge_accumulator_object, &route.shifted_accumulator_object, &route.relation_virtual_object, &route.eta_challenge_label,
                      &route.beta_challenge_label, &route.terminal_challenge_label})
                    pi_bind.string(*value);
                pi_bind.scalar(route.node_terminal_point);
                pi_bind.scalar(route.edge_terminal_point);
                pi_bind.scalar(route.terminal_delta);
                pi_bind.u64(route.configured_shift_degree);
                pi_bind.u64(route.shift_domain_size);
            };
            pi_bind.u64(meta.aggregation_relations.size());
            for (const auto &relation : meta.aggregation_relations)
            {
                for (const auto *value :
                     {&relation.relation_id, &relation.source_matrix_object, &relation.target_matrix_object, &relation.alpha_object, &relation.node_domain,
                      &relation.edge_domain, &relation.compression_challenge_label, &relation.source_node_object, &relation.target_node_object,
                      &relation.source_edge_object, &relation.target_edge_object, &relation.product_object, &relation.psq_object})
                    pi_bind.string(*value);
                pi_bind.u64(relation.row_count);
                pi_bind.u64(relation.column_count);
                write_aggregation_route(relation.source_route);
                write_aggregation_route(relation.target_route);
                write_string_vector(pi_bind, relation.commitment_order);
            }
            pi_bind.u64(meta.route_terminal_relations.size());
            for (const auto &relation : meta.route_terminal_relations)
            {
                for (const auto *value :
                     {&relation.relation_id, &relation.node_accumulator_object, &relation.edge_accumulator_object, &relation.shifted_accumulator_object,
                      &relation.relation_virtual_object, &relation.node_domain, &relation.edge_domain, &relation.challenge_label})
                    pi_bind.string(*value);
                pi_bind.scalar(relation.node_terminal_point);
                pi_bind.scalar(relation.edge_terminal_point);
                pi_bind.scalar(relation.terminal_delta);
                pi_bind.u64(relation.configured_shift_degree);
                pi_bind.u64(relation.shift_domain_size);
            }
            pi_bind.u64(meta.indexed_views.size());
            for (const auto &view : meta.indexed_views)
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
            for (const auto &relation : meta.coefficient_basis_relations)
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
                for (const auto &chunk : relation.chunks)
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
            for (const auto &relation : meta.shift_relations)
            {
                pi_bind.string(relation.relation_id);
                write_string_vector(pi_bind, relation.source_linear_maps);
                write_string_vector(pi_bind, relation.source_domains);
                pi_bind.u64(relation.canonical_points.size());
                for (const auto &point : relation.canonical_points)
                    pi_bind.scalar(point);
                pi_bind.scalar(relation.lambda0);
                pi_bind.u64(relation.deltas.size());
                for (const auto &delta : relation.deltas)
                    pi_bind.scalar(delta);
                write_string_vector(pi_bind, relation.source_value_kinds);
                pi_bind.string(relation.challenge_label);
                pi_bind.u64(relation.exclusion_degree);
                pi_bind.u64(relation.max_difference_degree);
                write_size_vector(pi_bind, relation.configured_degrees);
                write_size_vector(pi_bind, relation.shift_domain_sizes);
                write_string_vector(pi_bind, relation.opening_claim_order);
            }

            pi_bind.u64(subproof.model_reblinds.size());
            for (const auto &reblind : subproof.model_reblinds)
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
            for (const auto &reference : subproof.view_references)
                pi_bind.string(reference.object_id);
            for (const auto *group : {&subproof.view_commitments, &subproof.view_logup_commitments, &subproof.aggregation_commitments,
                                      &subproof.basis_commitments, &subproof.crpc_commitments, &subproof.relation_commitments})
            {
                write_named_commitments(pi_bind, *group);
            }
            pi_bind.u64(subproof.evaluations.size());
            for (const auto &evaluation : subproof.evaluations)
            {
                pi_bind.string(evaluation.domain);
                pi_bind.string(evaluation.object_id);
                pi_bind.u64(evaluation.values.size());
                for (const auto &value : evaluation.values)
                    pi_bind.scalar(value);
                pi_bind.u64(static_cast<std::uint64_t>(evaluation.kind));
            }
            pi_bind.u64(subproof.opening_references.size());
            for (const auto &reference : subproof.opening_references)
                pi_bind.string(reference.object_id);
        }
        wire.block("Pi_bind", std::move(pi_bind));
        return wire.take();
    }

    Proof parse_proof_internal(const ProtocolContext &context, const std::vector<std::uint8_t> &bytes)
    {
        if (context.compiler_artifact == nullptr || context.validated_compiler_artifact_identity != context.compiler_artifact.get())
        {
            throw std::runtime_error("proof parser is missing the authenticated ProtocolCompilerArtifact identity");
        }
        if (bytes.size() < kMagic.size() || !std::equal(kMagic.begin(), kMagic.end(), bytes.begin()))
        {
            throw std::runtime_error("proof parser magic mismatch");
        }
        Reader body(bytes.data() + kMagic.size(), bytes.size() - kMagic.size());
        Proof proof;
        auto m_pub = body.block("M_pub");
        metadata_fields(proof.public_metadata, [&](std::string &field) { field = m_pub.string(); });
        const auto block_count = m_pub.count("block order");
        proof.block_order.reserve(block_count);
        for (std::size_t i = 0; i < block_count; ++i)
            proof.block_order.push_back(m_pub.string());
        if (proof.block_order != proof_block_order())
            throw std::runtime_error("proof parser embedded block order mismatch");
        const auto logits_count = m_pub.count("public logits");
        proof.public_logits.reserve(logits_count);
        for (std::size_t i = 0; i < logits_count; ++i)
            proof.public_logits.push_back(m_pub.scalar());
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
            for (std::size_t i = 0; i < point_count; ++i)
                bundle.points.push_back(eval_dom.scalar());
            const auto value_count = eval_dom.count("domain value columns");
            for (std::size_t i = 0; i < value_count; ++i)
            {
                auto label = eval_dom.string();
                const auto item_count = eval_dom.count("domain values");
                std::vector<algebra::FieldElement> values;
                values.reserve(item_count);
                for (std::size_t j = 0; j < item_count; ++j)
                    values.push_back(eval_dom.scalar());
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
        if (opening_count != proof.domain_openings.size())
            throw std::runtime_error("proof parser domain opening count mismatch");
        std::vector<EncodedPointGroupOpening> encoded_openings;
        for (std::size_t i = 0; i < opening_count; ++i)
        {
            if (open_dom.string() != proof.domain_openings[i].first)
                throw std::runtime_error("proof parser domain opening name mismatch");
            const auto group_count = open_dom.count("domain point-group openings");
            if (group_count != proof.domain_openings[i].second.points.size())
                throw std::runtime_error("proof parser domain point-group count mismatch");
            for (std::size_t j = 0; j < group_count; ++j)
            {
                auto encoded = read_encoded_point_group_opening(open_dom);
                const auto expected_owner = "Open_dom:" + proof.domain_openings[i].first;
                if (encoded.owner_block != expected_owner || encoded.point != proof.domain_openings[i].second.points[j] ||
                    encoded.group_id != point_group_id(expected_owner, encoded.point))
                {
                    throw std::runtime_error("proof parser domain point-group owner/point mismatch");
                }
                encoded_openings.push_back(std::move(encoded));
            }
        }
        open_dom.finish("Open_dom");

        auto w_ext = body.block("W_ext");
        const auto external_group_count = w_ext.count("external point-group openings");
        std::string previous_external_point;
        for (std::size_t i = 0; i < external_group_count; ++i)
        {
            auto encoded = read_encoded_point_group_opening(w_ext);
            const auto point_text = encoded.point.to_string();
            if (encoded.owner_block != "W_ext" || encoded.group_id != point_group_id("W_ext", encoded.point) ||
                (i != 0 && !(previous_external_point < point_text)))
            {
                throw std::runtime_error("proof parser non-canonical external point-group order/owner");
            }
            previous_external_point = point_text;
            encoded_openings.push_back(std::move(encoded));
        }
        w_ext.finish("W_ext");
        proof.opening_groups = decode_point_group_openings(std::move(encoded_openings));
        std::sort(proof.opening_groups.begin(), proof.opening_groups.end(),
                  [](const PointGroupOpening &lhs, const PointGroupOpening &rhs) { return lhs.point.to_string() < rhs.point.to_string(); });
        require_structural_point_group_layout(proof);

        auto pi_bind = body.block("Pi_bind");
        const auto expected_binding_metadata = build_binding_relation_metadata(context);
        const auto subproof_count = pi_bind.count("Pi_bind subproofs");
        if (subproof_count != expected_binding_metadata.size())
            throw std::runtime_error("proof parser Pi_bind subproof count does not match instance schema");
        proof.pi_bind.subproofs.reserve(subproof_count);

        std::vector<EncodedBindingCommitments> encoded_binding_commitments;

        encoded_binding_commitments.reserve(subproof_count);

        for (std::size_t subproof_index = 0; subproof_index < subproof_count; ++subproof_index)
        {
            BindingSubproof subproof;
            auto &meta = subproof.relation_meta;
            meta.relation_id = pi_bind.string();
            meta.object_kind = pi_bind.string();
            meta.layer_id = read_optional_index(pi_bind, "Pi_bind layer_id");
            meta.head_id = read_optional_index(pi_bind, "Pi_bind head_id");
            meta.source_objects = read_string_vector(pi_bind, "Pi_bind sources");
            meta.target_objects = read_string_vector(pi_bind, "Pi_bind targets");
            meta.work_domains = read_string_vector(pi_bind, "Pi_bind domains");
            meta.real_lengths = read_size_vector(pi_bind, "Pi_bind real lengths");
            meta.quotient_relations = read_string_vector(pi_bind, "Pi_bind quotient relations");
            const auto canonical_matrix_count = pi_bind.count("Pi_bind canonical matrices");
            meta.canonical_matrices.reserve(canonical_matrix_count);
            for (std::size_t i = 0; i < canonical_matrix_count; ++i)
            {
                CanonicalMatrixRelationMeta matrix;
                matrix.relation_id = pi_bind.string();
                matrix.matrix_object = pi_bind.string();
                matrix.canonical_domain = pi_bind.string();
                matrix.row_count = pi_bind.size("Pi_bind canonical matrix rows");
                matrix.column_count = pi_bind.size("Pi_bind canonical matrix columns");
                matrix.column_objects = read_string_vector(pi_bind, "Pi_bind canonical column objects");
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
                route.configured_shift_degree = pi_bind.size("Pi_bind aggregation route shift degree");
                route.shift_domain_size = pi_bind.size("Pi_bind aggregation route shift domain");
                return route;
            };
            const auto aggregation_relation_count = pi_bind.count("Pi_bind aggregation relations");
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
                relation.row_count = pi_bind.size("Pi_bind aggregation rows");
                relation.column_count = pi_bind.size("Pi_bind aggregation columns");
                relation.source_route = read_aggregation_route();
                relation.target_route = read_aggregation_route();
                relation.commitment_order = read_string_vector(pi_bind, "Pi_bind aggregation commitment order");
                meta.aggregation_relations.push_back(std::move(relation));
            }
            const auto route_terminal_count = pi_bind.count("Pi_bind route terminal relations");
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
                relation.configured_shift_degree = pi_bind.size("Pi_bind route terminal shift degree");
                relation.shift_domain_size = pi_bind.size("Pi_bind route terminal shift domain");
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
                view.canonical_length = pi_bind.size("Pi_bind indexed canonical length");
                view.view_length = pi_bind.size("Pi_bind indexed view length");
                view.index_map_id = pi_bind.string();
                view.canonical_selector_id = pi_bind.string();
                view.query_selector_id = pi_bind.string();
                view.multiplicity_id = pi_bind.string();
                view.eta_challenge_label = pi_bind.string();
                view.beta_challenge_label = pi_bind.string();
                view.commitment_order = read_string_vector(pi_bind, "Pi_bind indexed commitment order");
                meta.indexed_views.push_back(std::move(view));
            }
            const auto basis_relation_count = pi_bind.count("Pi_bind coefficient basis relations");
            meta.coefficient_basis_relations.reserve(basis_relation_count);
            for (std::size_t i = 0; i < basis_relation_count; ++i)
            {
                CoefficientBasisRelationMeta relation;
                relation.relation_id = pi_bind.string();
                relation.canonical_object = pi_bind.string();
                relation.coefficient_object = pi_bind.string();
                relation.carrier_domain = pi_bind.string();
                relation.low_degree_length = pi_bind.size("Pi_bind coefficient basis low degree length");
                relation.tail_length = pi_bind.size("Pi_bind coefficient basis tail length");
                relation.total_length = pi_bind.size("Pi_bind coefficient basis total length");
                relation.semantic_positions = read_size_vector(pi_bind, "Pi_bind coefficient basis semantic positions");
                const auto chunk_count = pi_bind.count("Pi_bind coefficient basis chunks");
                relation.chunks.reserve(chunk_count);
                for (std::size_t j = 0; j < chunk_count; ++j)
                {
                    relation.chunks.push_back(CoefficientBasisChunkMeta{
                        .object_id = pi_bind.string(),
                        .accumulator_object = pi_bind.string(),
                        .shifted_accumulator_object = pi_bind.string(),
                        .chunk_index = pi_bind.size("Pi_bind coefficient basis chunk index"),
                        .offset = pi_bind.size("Pi_bind coefficient basis chunk offset"),
                        .length = pi_bind.size("Pi_bind coefficient basis chunk length"),
                        .terminal_point = pi_bind.scalar(),
                        .configured_shift_degree = pi_bind.size("Pi_bind coefficient basis chunk shift degree"),
                        .shift_domain_size = pi_bind.size("Pi_bind coefficient basis chunk shift domain size"),
                    });
                }
                relation.power_object = pi_bind.string();
                relation.gamma_challenge_label = pi_bind.string();
                relation.relation_challenge_label = pi_bind.string();
                relation.commitment_order = read_string_vector(pi_bind, "Pi_bind coefficient basis commitment order");
                relation.opening_claim_order = read_string_vector(pi_bind, "Pi_bind coefficient basis opening order");
                meta.coefficient_basis_relations.push_back(std::move(relation));
            }
            const auto shift_relation_count = pi_bind.count("Pi_bind shift relations");
            meta.shift_relations.reserve(shift_relation_count);
            for (std::size_t i = 0; i < shift_relation_count; ++i)
            {
                KZGShiftRelationMeta relation;
                relation.relation_id = pi_bind.string();
                relation.source_linear_maps = read_string_vector(pi_bind, "Pi_bind shift source maps");
                relation.source_domains = read_string_vector(pi_bind, "Pi_bind shift source domains");
                const auto point_count = pi_bind.count("Pi_bind shift points");
                relation.canonical_points.reserve(point_count);
                for (std::size_t j = 0; j < point_count; ++j)
                    relation.canonical_points.push_back(pi_bind.scalar());
                relation.lambda0 = pi_bind.scalar();
                const auto delta_count = pi_bind.count("Pi_bind shift deltas");
                relation.deltas.reserve(delta_count);
                for (std::size_t j = 0; j < delta_count; ++j)
                    relation.deltas.push_back(pi_bind.scalar());
                relation.source_value_kinds = read_string_vector(pi_bind, "Pi_bind shift source kinds");
                relation.challenge_label = pi_bind.string();
                relation.exclusion_degree = pi_bind.size("Pi_bind shift exclusion degree");
                relation.max_difference_degree = pi_bind.size("Pi_bind shift max degree");
                relation.configured_degrees = read_size_vector(pi_bind, "Pi_bind shift configured degrees");
                relation.shift_domain_sizes = read_size_vector(pi_bind, "Pi_bind shift domain sizes");
                relation.opening_claim_order = read_string_vector(pi_bind, "Pi_bind shift opening order");
                meta.shift_relations.push_back(std::move(relation));
            }
            if (!(meta == expected_binding_metadata.at(subproof_index)))
                throw std::runtime_error("proof parser Pi_bind RelationMeta does not match instance schema");

            EncodedBindingCommitments encoded_binding;

            const auto model_count = pi_bind.count("Pi_bind model reblinds");

            subproof.model_reblinds.resize(model_count);

            encoded_binding.model_reblind_encodings.reserve(model_count * 2);

            for (std::size_t i = 0; i < model_count; ++i)
            {
                auto &reblind = subproof.model_reblinds[i];

                reblind.object_id = pi_bind.string();

                reblind.domain = pi_bind.string();

                reblind.vk_model_anchor.object_id = pi_bind.string();

                reblind.proof_local_commitment.name = pi_bind.string();

                encoded_binding.model_reblind_encodings.push_back(pi_bind.point_encoding());

                reblind.blinder_commitment.name = pi_bind.string();

                encoded_binding.model_reblind_encodings.push_back(pi_bind.point_encoding());
            }

            const auto view_reference_count = pi_bind.count("Pi_bind Com_view references");

            subproof.view_references.reserve(view_reference_count);

            for (std::size_t i = 0; i < view_reference_count; ++i)
            {
                subproof.view_references.push_back({pi_bind.string()});
            }

            {
                std::unordered_set<std::string> references;

                for (const auto &reference : subproof.view_references)
                {
                    if (reference.object_id.empty() || !references.insert(reference.object_id).second)
                    {
                        throw std::runtime_error("proof parser duplicate/empty Pi_bind Com_view reference");
                    }
                }
            }

            encoded_binding.view_commitments = read_encoded_named_commitments(pi_bind);

            encoded_binding.view_logup_commitments = read_encoded_named_commitments(pi_bind);

            encoded_binding.aggregation_commitments = read_encoded_named_commitments(pi_bind);

            encoded_binding.basis_commitments = read_encoded_named_commitments(pi_bind);

            encoded_binding.crpc_commitments = read_encoded_named_commitments(pi_bind);

            encoded_binding.relation_commitments = read_encoded_named_commitments(pi_bind);

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
                for (const auto &evaluation : subproof.evaluations)
                {
                    const auto key = evaluation.domain + "\n" + evaluation.object_id;
                    if (evaluation.domain.empty() || evaluation.object_id.empty() || evaluation.values.empty() || !evaluations.insert(key).second)
                        throw std::runtime_error("proof parser duplicate/empty Pi_bind evaluation");
                }
            }
            const auto reference_count = pi_bind.count("Pi_bind opening references");
            subproof.opening_references.reserve(reference_count);
            for (std::size_t i = 0; i < reference_count; ++i)
                subproof.opening_references.push_back({pi_bind.string()});
            {
                std::unordered_set<std::string> references;
                for (const auto &reference : subproof.opening_references)
                    if (reference.object_id.empty() || !references.insert(reference.object_id).second)
                        throw std::runtime_error("proof parser duplicate/empty Pi_bind opening reference");
            }
            encoded_binding_commitments.push_back(std::move(encoded_binding));

            proof.pi_bind.subproofs.push_back(std::move(subproof));
        }

        if (encoded_binding_commitments.size() != proof.pi_bind.subproofs.size())
        {
            throw std::runtime_error("proof parser Pi_bind deferred G1 shape mismatch");
        }

        std::size_t total_binding_g1 = 0;

        const auto add_binding_g1_count = [&](std::size_t count)
        {
            if (count > std::numeric_limits<std::size_t>::max() - total_binding_g1)
            {
                throw std::runtime_error("proof parser Pi_bind G1 count overflow");
            }

            total_binding_g1 += count;
        };

        for (const auto &encoded : encoded_binding_commitments)
        {
            add_binding_g1_count(encoded.model_reblind_encodings.size());

            for (const auto *group : {&encoded.view_commitments, &encoded.view_logup_commitments, &encoded.aggregation_commitments, &encoded.basis_commitments,
                                      &encoded.crpc_commitments, &encoded.relation_commitments})
            {
                add_binding_g1_count(group->encodings.size());
            }
        }

        std::vector<EncodedG1View> all_binding_g1_encodings;

        all_binding_g1_encodings.reserve(total_binding_g1);

        for (const auto &encoded : encoded_binding_commitments)
        {
            all_binding_g1_encodings.insert(all_binding_g1_encodings.end(), encoded.model_reblind_encodings.begin(), encoded.model_reblind_encodings.end());

            for (const auto *group : {&encoded.view_commitments, &encoded.view_logup_commitments, &encoded.aggregation_commitments, &encoded.basis_commitments,
                                      &encoded.crpc_commitments, &encoded.relation_commitments})
            {
                all_binding_g1_encodings.insert(all_binding_g1_encodings.end(), group->encodings.begin(), group->encodings.end());
            }
        }

        auto decoded_binding_points = decode_canonical_g1_batch(all_binding_g1_encodings);

        std::size_t decoded_position = 0;

        for (std::size_t subproof_index = 0; subproof_index < proof.pi_bind.subproofs.size(); ++subproof_index)
        {
            auto &subproof = proof.pi_bind.subproofs[subproof_index];

            auto &encoded = encoded_binding_commitments[subproof_index];

            if (encoded.model_reblind_encodings.size() != subproof.model_reblinds.size() * 2)
            {
                throw std::runtime_error("proof parser Pi_bind model-reblind deferred G1 shape mismatch");
            }

            for (std::size_t model = 0; model < subproof.model_reblinds.size(); ++model)
            {
                if (decoded_position + 2 > decoded_binding_points.size())
                {
                    throw std::runtime_error("proof parser Pi_bind deferred G1 decode underflow");
                }

                subproof.model_reblinds[model].proof_local_commitment.point = std::move(decoded_binding_points[decoded_position++]);

                subproof.model_reblinds[model].blinder_commitment.point = std::move(decoded_binding_points[decoded_position++]);
            }

            const auto decode_group = [&](auto &group, EncodedNamedCommitments &encoded_group, const char *label)
            {
                if (encoded_group.names.size() != encoded_group.encodings.size() ||
                    decoded_position + encoded_group.names.size() > decoded_binding_points.size())
                {
                    throw std::runtime_error(std::string("proof parser Pi_bind deferred G1 decode shape mismatch: ") + label);
                }

                group.reserve(encoded_group.names.size());

                for (std::size_t i = 0; i < encoded_group.names.size(); ++i)
                {
                    crypto::Commitment commitment;

                    commitment.name = encoded_group.names[i];

                    commitment.point = std::move(decoded_binding_points[decoded_position++]);

                    group.push_back({
                        std::move(encoded_group.names[i]),
                        std::move(commitment),
                    });
                }

                require_unique_names(group, label);
            };

            decode_group(subproof.view_commitments, encoded.view_commitments, "Pi_bind Com_view");

            decode_group(subproof.view_logup_commitments, encoded.view_logup_commitments, "Pi_bind Com_viewLogUp");

            decode_group(subproof.aggregation_commitments, encoded.aggregation_commitments, "Pi_bind Com_agg");

            decode_group(subproof.basis_commitments, encoded.basis_commitments, "Pi_bind Com_basis");

            decode_group(subproof.crpc_commitments, encoded.crpc_commitments, "Pi_bind Com_CRPC");

            decode_group(subproof.relation_commitments, encoded.relation_commitments, "Pi_bind Com_rel");
        }

        if (decoded_position != decoded_binding_points.size())
        {
            throw std::runtime_error("proof parser Pi_bind deferred G1 decode left unconsumed points");
        }

        pi_bind.finish("Pi_bind");
        body.finish("proof");

        std::unordered_map<std::string, std::size_t> domain_opening_index;

        domain_opening_index.reserve(proof.domain_openings.size());

        std::vector<std::unordered_set<std::string>> domain_opening_value_ids(proof.domain_openings.size());

        for (std::size_t bundle_index = 0; bundle_index < proof.domain_openings.size(); ++bundle_index)
        {
            const auto &[bundle_name, bundle] = proof.domain_openings[bundle_index];

            domain_opening_index.try_emplace(bundle_name, bundle_index);

            auto &value_ids = domain_opening_value_ids[bundle_index];

            value_ids.reserve(bundle.values.size());

            for (const auto &[name, values] : bundle.values)
            {
                (void)values;

                value_ids.insert(name);
            }
        }

        for (const auto &subproof : proof.pi_bind.subproofs)
        {
            for (const auto &evaluation : subproof.evaluations)
            {
                if (evaluation.domain == "relation")
                    continue;

                const auto bundle_position = domain_opening_index.find(evaluation.domain);

                if (bundle_position == domain_opening_index.end())
                {
                    throw std::runtime_error("proof parser Pi_bind evaluation references unknown domain");
                }

                const auto bundle_index = bundle_position->second;

                auto &bundle = proof.domain_openings[bundle_index].second;

                auto &value_ids = domain_opening_value_ids[bundle_index];

                if (!value_ids.insert(evaluation.object_id).second)
                {
                    throw std::runtime_error("proof parser duplicate Eval_dom/Eval_bind claim owner");
                }

                bundle.values.push_back({evaluation.object_id, evaluation.values});

                bundle.value_kinds.push_back(evaluation.kind);
            }
        }
        const auto opening_label_registry = build_domain_opening_label_registry(context);

        for (auto &[bundle_name, bundle] : proof.domain_openings)
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
                throw std::runtime_error("proof parser encountered unknown domain bundle: " + bundle_name);
            }

            const auto registered_labels = opening_label_registry.find(trace_domain);

            if (registered_labels == opening_label_registry.end())
            {
                throw std::runtime_error("proof parser has no opening-label registry for domain: " + trace_domain);
            }

            auto expected_labels = registered_labels->second;

            expected_labels.push_back(quotient_name);

            expected_labels.push_back(quotient_mask_label(quotient_name));

            if (bundle.values.size() != expected_labels.size() || bundle.value_kinds.size() != bundle.values.size())
                throw std::runtime_error("proof parser reconstructed domain bundle shape mismatch: " + bundle_name);

            std::unordered_map<std::string, std::size_t> positions;
            positions.reserve(bundle.values.size());
            for (std::size_t i = 0; i < bundle.values.size(); ++i)
                if (!positions.emplace(bundle.values[i].first, i).second)
                    throw std::runtime_error("proof parser reconstructed duplicate domain value: " + bundle.values[i].first);

            std::vector<std::pair<std::string, std::vector<algebra::FieldElement>>> ordered_values;
            std::vector<PayloadValueKind> ordered_kinds;
            ordered_values.reserve(expected_labels.size());
            ordered_kinds.reserve(expected_labels.size());
            for (const auto &label : expected_labels)
            {
                const auto position = positions.find(label);
                if (position == positions.end())
                    throw std::runtime_error("proof parser reconstructed domain bundle is missing: " + label);
                ordered_values.push_back(std::move(bundle.values[position->second]));
                ordered_kinds.push_back(bundle.value_kinds[position->second]);
            }
            bundle.values = std::move(ordered_values);
            bundle.value_kinds = std::move(ordered_kinds);
        }

        std::unordered_map<std::string, crypto::Commitment> dynamic;
        std::unordered_map<std::string, crypto::Commitment> quotient;

        dynamic.reserve(proof.dynamic_commitments.size() + total_binding_g1);

        quotient.reserve(proof.quotient_commitments.size());

        for (const auto &[name, commitment] : proof.dynamic_commitments)
            dynamic.emplace(name, commitment);
        for (const auto &subproof : proof.pi_bind.subproofs)
        {
            for (const auto &reblind : subproof.model_reblinds)
            {
                if (!dynamic.emplace(reblind.proof_local_commitment.name, reblind.proof_local_commitment).second ||
                    !dynamic.emplace(reblind.blinder_commitment.name, reblind.blinder_commitment).second)
                {
                    throw std::runtime_error("proof parser duplicate Pi_bind model commitment reference");
                }
            }
            for (const auto *group : {&subproof.view_commitments, &subproof.view_logup_commitments, &subproof.aggregation_commitments,
                                      &subproof.basis_commitments, &subproof.crpc_commitments, &subproof.relation_commitments})
            {
                for (const auto &[name, commitment] : *group)
                {
                    if (name != commitment.name || !dynamic.emplace(name, commitment).second)
                    {
                        throw std::runtime_error("proof parser duplicate or inconsistent Pi_bind commitment owner: " + name);
                    }
                }
            }
        }
        for (const auto &[name, commitment] : proof.quotient_commitments)
            quotient.emplace(name, commitment);
        proof.challenges = replay_challenges(context, dynamic, quotient, &proof.public_metadata);
        const auto expected_route_references = route_opening_references(context, proof.challenges);
        if (proof.route_references != expected_route_references)
            throw std::runtime_error("proof parser S_route reference registry mismatch");
        require_exact_pi_bind_shape(context, proof, expected_binding_metadata);
        return proof;
    }

    Proof parse_proof(const VerifierContext &context, const std::vector<std::uint8_t> &bytes)
    {
        if (const auto *execution_context = detail::validated_verifier_execution_context(context); execution_context != nullptr)
        {
            auto proof = parse_proof_internal(*execution_context, bytes);

            if (proof_contains_prover_material(proof))
            {
                throw std::runtime_error("proof parser rejected non-null prover-only opening polynomial");
            }

            return proof;
        }

        if (context.compiler_artifact == nullptr || !validate_verifier_context(context))
        {
            throw std::runtime_error("proof parser requires an authenticated point-only verifier registry");
        }

        auto execution_context = detail::make_verifier_execution_context(context);

        auto proof = parse_proof_internal(execution_context, bytes);

        if (proof_contains_prover_material(proof))
        {
            throw std::runtime_error("proof parser rejected non-null prover-only opening polynomial");
        }

        return proof;
    }
}
