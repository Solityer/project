#include "gatzk/protocol/opening_registry.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_set>

#include "gatzk/crypto/curve.hpp"
#include "gatzk/crypto/transcript.hpp"
#include "gatzk/protocol/challenges.hpp"

namespace gatzk::protocol
{
    namespace
    {
        std::string commitment_point_key(
            const crypto::G1Point &point)
        {
            std::array<std::uint8_t, 256> encoded{};
            const auto written = crypto::serialize(
                point,
                encoded.data(),
                encoded.size());

            return std::string(
                reinterpret_cast<const char *>(encoded.data()),
                written);
        }

        std::string point_lookup_key(
            const algebra::FieldElement &point)
        {
            std::array<std::uint8_t, 32> encoded{};
            point.write_little_endian(
                encoded.data(),
                encoded.size());

            return std::string(
                reinterpret_cast<const char *>(encoded.data()),
                encoded.size());
        }

        std::string decimal_length_prefix(
            std::size_t value)
        {
            std::array<char, 32>
                bytes{};

            const auto result =
                std::to_chars(
                    bytes.data(),
                    bytes.data() +
                        bytes.size(),
                    value);

            if (result.ec !=
                std::errc{})
            {
                throw std::runtime_error(
                    "standard opening claim length encoding failed");
            }

            return std::string(
                bytes.data(),
                result.ptr);
        }
        bool canonical_claim_less(
            const StandardOpeningClaim &lhs,
            const StandardOpeningClaim &rhs)
        {
            const auto compare_component =
                [&](std::size_t index,
                    const std::string &left,
                    const std::string &right)
            {
                const auto prefix_result =
                    lhs
                        .canonical_text_length_prefixes[index]
                        .compare(
                            rhs
                                .canonical_text_length_prefixes[index]);

                if (prefix_result != 0)
                {
                    return prefix_result;
                }

                return left.compare(
                    right);
            };

            int result =
                compare_component(
                    0,
                    lhs.claim_id,
                    rhs.claim_id);

            if (result != 0)
            {
                return result < 0;
            }

            result =
                compare_component(
                    1,
                    lhs.object_id,
                    rhs.object_id);

            if (result != 0)
            {
                return result < 0;
            }

            result =
                compare_component(
                    2,
                    lhs.commitment_reference,
                    rhs.commitment_reference);

            if (result != 0)
            {
                return result < 0;
            }

            result =
                compare_component(
                    3,
                    lhs.evaluation_kind,
                    rhs.evaluation_kind);

            if (result != 0)
            {
                return result < 0;
            }

            result =
                compare_component(
                    4,
                    lhs.claim_role,
                    rhs.claim_role);

            if (result != 0)
            {
                return result < 0;
            }

            result =
                compare_component(
                    5,
                    lhs.visibility_tag,
                    rhs.visibility_tag);

            if (result != 0)
            {
                return result < 0;
            }

            result =
                compare_component(
                    6,
                    lhs.value_type,
                    rhs.value_type);

            if (result != 0)
            {
                return result < 0;
            }

            result =
                compare_component(
                    7,
                    lhs.value_reference_id,
                    rhs.value_reference_id);

            return result < 0;
        }
        void require_claim_shape(const StandardOpeningClaim &claim)
        {
            for (const auto *field : {
                     &claim.claim_id,
                     &claim.object_id,
                     &claim.commitment_reference,
                     &claim.evaluation_kind,
                     &claim.claim_role,
                     &claim.visibility_tag,
                     &claim.value_type,
                     &claim.value_reference_id})
            {
                if (field->empty())
                {
                    throw std::runtime_error(
                        "standard opening claim contains an empty canonical tuple field");
                }
            }
            if (claim.execution_commitment ==
                    nullptr ||
                claim.commitment_reference !=
                    claim.execution_commitment
                        ->name)
            {
                throw std::runtime_error(
                    "standard opening claim commitment reference/name mismatch");
            }
            if (claim.visibility_tag != "public" && claim.visibility_tag != "masked_private")
            {
                throw std::runtime_error(
                    "standard opening claim has invalid visibility tag");
            }
            if (claim.value_type != "field_element")
            {
                throw std::runtime_error(
                    "standard opening claim has invalid value type");
            }
        }

        bool canonical_group_less(
            const StandardPointGroup &lhs,
            const StandardPointGroup &rhs)
        {
            return lhs.point.to_string() < rhs.point.to_string();
        }
    }

    std::string point_group_id(
        const std::string &owner_block,
        const algebra::FieldElement &point)
    {
        if (owner_block == "W_ext")
        {
            return "W_ext:" + point.to_string();
        }
        if (owner_block.rfind("Open_dom:", 0) != 0)
        {
            throw std::runtime_error("invalid standard point-group owner: " + owner_block);
        }
        return owner_block + ":" + point.to_string();
    }

    StandardOpeningClaim make_standard_opening_claim(
        std::string claim_id,
        std::string object_id,
        std::string evaluation_kind,
        std::string claim_role,
        PayloadValueKind visibility,
        std::string value_reference_id,
        const crypto::Commitment &commitment,
        const algebra::FieldElement &value)
    {
        const auto commitment_reference =
            commitment.name;

        std::string visibility_tag =
            visibility ==
                    PayloadValueKind::
                        PublicValue
                ? "public"
                : "masked_private";

        std::string value_type =
            "field_element";

        std::array<std::string, 8>
            length_prefixes{
                decimal_length_prefix(
                    claim_id.size()),
                decimal_length_prefix(
                    object_id.size()),
                decimal_length_prefix(
                    commitment_reference
                        .size()),
                decimal_length_prefix(
                    evaluation_kind.size()),
                decimal_length_prefix(
                    claim_role.size()),
                decimal_length_prefix(
                    visibility_tag.size()),
                decimal_length_prefix(
                    value_type.size()),
                decimal_length_prefix(
                    value_reference_id
                        .size()),
        };

        return StandardOpeningClaim{
            .claim_id =
                std::move(
                    claim_id),
            .object_id =
                std::move(
                    object_id),
            .commitment_reference =
                commitment_reference,
            .evaluation_kind =
                std::move(
                    evaluation_kind),
            .claim_role =
                std::move(
                    claim_role),
            .visibility_tag =
                std::move(
                    visibility_tag),
            .value_type =
                std::move(
                    value_type),
            .value_reference_id =
                std::move(
                    value_reference_id),
            .execution_commitment =
                &commitment,
            .value =
                value,
            .canonical_text_length_prefixes =
                std::move(
                    length_prefixes),
            .execution_commitment_identity =
                std::numeric_limits<
                    std::size_t>::max(),
        };
    }
    void StandardPointGroupRegistry::
        add_domain_bundle(
            const std::string &domain,
            const DomainOpeningBundle &bundle,
            const std::vector<
                crypto::Commitment>
                &commitments)
    {
        if (finalized_)
        {
            throw std::runtime_error(
                "standard point-group registry cannot change after finalization");
        }

        if (bundle.points.empty() ||
            bundle.values.size() !=
                commitments.size() ||
            bundle.value_kinds.size() !=
                commitments.size())
        {
            throw std::runtime_error(
                "invalid domain bundle for standard point-group registry");
        }

        for (std::size_t object_index = 0;
             object_index <
             commitments.size();
             ++object_index)
        {
            if (bundle.values[object_index]
                        .first !=
                    commitments[object_index]
                        .name ||
                bundle.values[object_index]
                        .second
                        .size() !=
                    bundle.points.size())
            {
                throw std::runtime_error(
                    "domain bundle commitment/value order mismatch");
            }
        }

        prepare_commitment_execution_identities(
            commitments);

        std::vector<
            const CommitmentExecutionIdentity *>
            prepared_commitments;

        prepared_commitments.reserve(
            commitments.size());

        for (const auto &commitment :
             commitments)
        {
            const auto prepared =
                commitment_execution_identity_by_reference_
                    .find(
                        commitment.name);

            if (prepared ==
                    commitment_execution_identity_by_reference_
                        .end() ||
                !(
                    prepared
                        ->second
                        .commitment
                        .point ==
                    commitment.point))
            {
                throw std::runtime_error(
                    "standard opening prepared commitment identity mismatch");
            }

            prepared_commitments.push_back(
                &prepared->second);
        }

        const auto requested_owner =
            "Open_dom:" +
            domain;

        for (std::size_t point_index = 0;
             point_index <
             bundle.points.size();
             ++point_index)
        {
            const auto group_index =
                ensure_group(
                    requested_owner,
                    bundle.points[point_index]);

            auto &group =
                groups_.at(
                    group_index);

            auto &claim_positions =
                claim_index_by_commitment_
                    .at(
                        group_index);

            group.claims.reserve(
                group.claims.size() +
                commitments.size());

            claim_positions.reserve(
                claim_positions.size() +
                commitments.size());

            for (std::size_t object_index = 0;
                 object_index <
                 commitments.size();
                 ++object_index)
            {
                const auto &prepared =
                    *prepared_commitments[object_index];

                const auto &object_id =
                    bundle.values[object_index]
                        .first;

                const auto claim_role =
                    object_id.rfind(
                        "t_",
                        0) == 0
                        ? "quotient"
                    : object_id.rfind(
                          "R_t_",
                          0) == 0
                        ? "quotient_blinder"
                        : "domain_polynomial";

                const auto suffix =
                    domain +
                    ":" +
                    object_id +
                    ":" +
                    std::to_string(
                        point_index);

                auto claim =
                    make_standard_opening_claim(
                        "claim:domain:" +
                            suffix,
                        object_id,
                        "domain_evaluation",
                        claim_role,
                        bundle.value_kinds[object_index],
                        "Eval_dom:" +
                            suffix,
                        prepared
                            .commitment,
                        bundle.values[object_index]
                            .second[point_index]);

                claim.execution_commitment =
                    &prepared
                         .commitment;

                claim.execution_commitment_identity =
                    prepared
                        .identity;

                require_claim_shape(
                    claim);

                insert_prepared_claim(
                    group_index,
                    std::move(
                        claim));
            }
        }
    }

    void add_domain_bundle_claims(
        StandardPointGroupRegistry &registry,
        const std::string &domain,
        const DomainOpeningBundle &bundle,
        const std::vector<
            crypto::Commitment>
            &commitments)
    {
        registry.add_domain_bundle(
            domain,
            bundle,
            commitments);
    }

    void add_external_claims(
        StandardPointGroupRegistry &registry,
        const std::vector<std::pair<crypto::Commitment,
                                    algebra::FieldElement>> &commitments_and_values,
        const std::vector<algebra::FieldElement> &points,
        const std::vector<PayloadValueKind> &eval_ext_kinds)
    {
        if (commitments_and_values.size() != points.size() ||
            eval_ext_kinds.size() > commitments_and_values.size())
        {
            throw std::runtime_error(
                "invalid external claims for standard point-group registry");
        }

        std::unordered_map<
            std::string,
            std::string>
            point_text_cache;

        point_text_cache.reserve(
            points.size());

        for (std::size_t i = 0;
             i < commitments_and_values.size();
             ++i)
        {
            const auto &[commitment, value] =
                commitments_and_values[i];

            const bool is_eval_ext =
                i < eval_ext_kinds.size();

            const bool is_public_output =
                commitment.name == "P_Y";

            const bool is_public_relation =
                commitment.name.find(
                    "relation_virtual") !=
                    std::string::npos ||
                commitment.name.ends_with(
                    "_m_can");

            const auto kind =
                is_eval_ext
                    ? eval_ext_kinds[i]
                    : (is_public_output ||
                               is_public_relation
                           ? PayloadValueKind::
                                 PublicValue
                           : PayloadValueKind::
                                 MaskedPrivateValue);

            const auto evaluation_kind =
                is_eval_ext
                    ? "external_evaluation"
                : is_public_output
                    ? "verifier_reconstructed_public_output"
                    : "binding_evaluation";

            const auto claim_role =
                is_public_output
                    ? "public_output"
                : is_public_relation
                    ? "binding_relation"
                : is_eval_ext
                    ? "ordinary_external"
                    : "binding_shift";

            const auto point_key =
                point_lookup_key(
                    points[i]);

            auto [point_it, inserted] =
                point_text_cache.try_emplace(
                    point_key);

            if (inserted)
            {
                point_it->second =
                    points[i].to_string();
            }

            const auto &point_text =
                point_it->second;

            const auto identity =
                commitment.name +
                ":" +
                point_text;

            registry.add_external_claim(
                points[i],
                make_standard_opening_claim(
                    "claim:external:" +
                        identity,
                    commitment.name,
                    evaluation_kind,
                    claim_role,
                    kind,
                    (is_eval_ext
                         ? "Eval_ext:"
                     : is_public_output
                         ? "M_pub:public_logits:"
                         : "Eval_bind:") +
                        identity,
                    commitment,
                    value));
        }
    }
    void StandardPointGroupRegistry::
        prepare_commitment_execution_identities(
            const std::vector<
                crypto::Commitment>
                &commitments)
    {
        if (finalized_)
        {
            throw std::runtime_error(
                "standard point-group registry cannot prepare identities after finalization");
        }

        std::vector<std::size_t>
            missing_indices;

        missing_indices.reserve(
            commitments.size());

        for (std::size_t index = 0;
             index < commitments.size();
             ++index)
        {
            const auto &commitment =
                commitments[index];

            const auto existing =
                commitment_execution_identity_by_reference_
                    .find(
                        commitment.name);

            if (existing ==
                commitment_execution_identity_by_reference_
                    .end())
            {
                missing_indices.push_back(
                    index);

                continue;
            }

            if (!(existing
                      ->second
                      .commitment
                      .point ==
                  commitment.point))
            {
                throw std::runtime_error(
                    "standard opening commitment reference resolves to multiple G1 points");
            }
        }

        if (missing_indices.empty())
        {
            return;
        }

        std::vector<std::string>
            serialized_keys(
                missing_indices.size());

        constexpr std::size_t
            kMinimumIdentitiesPerWorker =
                32;

        constexpr std::size_t
            kMaximumIdentityWorkers =
                16;

        const auto hardware_workers =
            std::max<std::size_t>(
                1,
                std::thread::
                    hardware_concurrency());

        const auto size_workers =
            std::max<std::size_t>(
                1,
                (missing_indices.size() +
                 kMinimumIdentitiesPerWorker -
                 1) /
                    kMinimumIdentitiesPerWorker);

        const auto worker_count =
            std::min<std::size_t>(
                {
                    hardware_workers,
                    kMaximumIdentityWorkers,
                    size_workers,
                });

        const auto serialize_range =
            [&](std::size_t begin,
                std::size_t end)
        {
            for (std::size_t local = begin;
                 local < end;
                 ++local)
            {
                serialized_keys[local] =
                    commitment_point_key(
                        commitments[missing_indices[local]]
                            .point);
            }
        };

        if (worker_count <= 1)
        {
            serialize_range(
                0,
                missing_indices.size());
        }
        else
        {
            const auto chunk_size =
                (missing_indices.size() +
                 worker_count - 1) /
                worker_count;

            std::vector<
                std::future<void>>
                futures;

            futures.reserve(
                worker_count);

            for (std::size_t worker = 0;
                 worker < worker_count;
                 ++worker)
            {
                const auto begin =
                    worker *
                    chunk_size;

                const auto end =
                    std::min(
                        missing_indices.size(),
                        begin +
                            chunk_size);

                if (begin >= end)
                {
                    break;
                }

                futures.push_back(
                    std::async(
                        std::launch::async,
                        serialize_range,
                        begin,
                        end));
            }

            for (auto &future :
                 futures)
            {
                future.get();
            }
        }

        for (std::size_t local = 0;
             local < missing_indices.size();
             ++local)
        {
            const auto &commitment =
                commitments[missing_indices[local]];

            const auto existing =
                commitment_execution_identity_by_reference_
                    .find(
                        commitment.name);

            if (existing !=
                commitment_execution_identity_by_reference_
                    .end())
            {
                if (!(existing
                          ->second
                          .commitment
                          .point ==
                      commitment.point))
                {
                    throw std::runtime_error(
                        "standard opening commitment reference resolves to multiple G1 points");
                }

                continue;
            }

            auto point_identity =
                commitment_execution_identity_by_point_
                    .find(
                        serialized_keys[local]);

            std::size_t identity = 0;

            if (point_identity ==
                commitment_execution_identity_by_point_
                    .end())
            {
                if (next_commitment_execution_identity_ ==
                    std::numeric_limits<
                        std::size_t>::max())
                {
                    throw std::runtime_error(
                        "standard opening commitment execution identity exhausted");
                }

                identity =
                    next_commitment_execution_identity_++;

                point_identity =
                    commitment_execution_identity_by_point_
                        .emplace(
                            std::move(
                                serialized_keys[local]),
                            identity)
                        .first;
            }
            else
            {
                identity =
                    point_identity->second;
            }

            commitment_execution_identity_by_reference_
                .emplace(
                    commitment.name,
                    CommitmentExecutionIdentity{
                        .commitment =
                            commitment,
                        .identity =
                            identity,
                    });
        }
    }

    void StandardPointGroupRegistry::
        refresh_claim_execution_keys(
            StandardOpeningClaim *claim)
    {
        if (claim == nullptr ||
            claim->execution_commitment ==
                nullptr)
        {
            throw std::runtime_error(
                "standard opening claim execution cache is null");
        }

        const auto *source_commitment =
            claim->execution_commitment;

        if (source_commitment->name !=
            claim->commitment_reference)
        {
            throw std::runtime_error(
                "standard opening claim commitment reference/name mismatch");
        }

        auto identity_it =
            commitment_execution_identity_by_reference_
                .find(
                    claim
                        ->commitment_reference);

        if (identity_it ==
            commitment_execution_identity_by_reference_
                .end())
        {
            auto serialized_key =
                commitment_point_key(
                    source_commitment
                        ->point);

            auto point_identity =
                commitment_execution_identity_by_point_
                    .find(
                        serialized_key);

            std::size_t identity =
                0;

            if (point_identity ==
                commitment_execution_identity_by_point_
                    .end())
            {
                if (next_commitment_execution_identity_ ==
                    std::numeric_limits<
                        std::size_t>::max())
                {
                    throw std::runtime_error(
                        "standard opening commitment execution identity exhausted");
                }

                identity =
                    next_commitment_execution_identity_++;

                point_identity =
                    commitment_execution_identity_by_point_
                        .emplace(
                            std::move(
                                serialized_key),
                            identity)
                        .first;
            }
            else
            {
                identity =
                    point_identity
                        ->second;
            }

            identity_it =
                commitment_execution_identity_by_reference_
                    .emplace(
                        claim
                            ->commitment_reference,
                        CommitmentExecutionIdentity{
                            .commitment =
                                *source_commitment,
                            .identity =
                                identity,
                        })
                    .first;
        }
        else if (!(
                     identity_it
                         ->second
                         .commitment
                         .point ==
                     source_commitment
                         ->point))
        {
            throw std::runtime_error(
                "standard opening commitment reference resolves to multiple G1 points");
        }

        claim->execution_commitment =
            &identity_it
                 ->second
                 .commitment;

        claim->execution_commitment_identity =
            identity_it
                ->second
                .identity;
    }

    std::size_t
    StandardPointGroupRegistry::
        ensure_group(
            const std::string &requested_owner,
            const algebra::FieldElement &point)
    {
        std::size_t group_index =
            0;

        if (last_group_valid_ &&
            last_group_point_ == point)
        {
            group_index =
                last_group_index_;
        }
        else
        {
            const auto point_key =
                point_lookup_key(
                    point);

            auto group_position =
                group_index_by_point_
                    .find(
                        point_key);

            if (group_position ==
                group_index_by_point_
                    .end())
            {
                group_index =
                    groups_.size();

                groups_.push_back(
                    StandardPointGroup{
                        .group_id =
                            point_group_id(
                                requested_owner,
                                point),
                        .owner_block =
                            requested_owner,
                        .point =
                            point,
                        .claims =
                            {},
                    });

                group_index_by_point_
                    .emplace(
                        point_key,
                        group_index);

                claim_index_by_commitment_
                    .emplace_back();
            }
            else
            {
                group_index =
                    group_position
                        ->second;
            }

            last_group_valid_ =
                true;

            last_group_point_ =
                point;

            last_group_index_ =
                group_index;
        }

        auto &group =
            groups_.at(
                group_index);

        if (requested_owner.rfind(
                "Open_dom:",
                0) == 0)
        {
            if (group.owner_block.rfind(
                    "Open_dom:",
                    0) == 0 &&
                group.owner_block !=
                    requested_owner)
            {
                throw std::runtime_error(
                    "distinct work domains resolved to the same opening point");
            }

            if (group.owner_block !=
                requested_owner)
            {
                group.owner_block =
                    requested_owner;

                group.group_id =
                    point_group_id(
                        requested_owner,
                        point);
            }
        }

        return group_index;
    }

    void StandardPointGroupRegistry::
        insert_prepared_claim(
            std::size_t group_index,
            StandardOpeningClaim claim)
    {
        auto &group =
            groups_.at(
                group_index);

        auto &claim_positions =
            claim_index_by_commitment_
                .at(
                    group_index);

        const auto commitment_identity =
            claim
                .execution_commitment_identity;

        if (commitment_identity ==
            std::numeric_limits<
                std::size_t>::max())
        {
            throw std::runtime_error(
                "standard opening claim execution identity is missing");
        }

        const auto duplicate_position =
            claim_positions.find(
                commitment_identity);

        if (duplicate_position ==
            claim_positions.end())
        {
            claim_positions.emplace(
                commitment_identity,
                group.claims.size());

            group.claims.push_back(
                std::move(
                    claim));

            return;
        }

        auto &duplicate =
            group.claims.at(
                duplicate_position
                    ->second);

        if (duplicate.value !=
            claim.value)
        {
            throw std::runtime_error(
                "conflicting values for the same commitment-point opening claim");
        }

        if (canonical_claim_less(
                claim,
                duplicate))
        {
            duplicate =
                std::move(
                    claim);
        }
    }
    void StandardPointGroupRegistry::add_domain_claim(
        const std::string &domain,
        const algebra::FieldElement &point,
        StandardOpeningClaim claim)
    {
        if (domain.empty())
        {
            throw std::runtime_error("standard domain point group has empty domain");
        }
        add_claim("Open_dom:" + domain, point, std::move(claim));
    }

    void StandardPointGroupRegistry::add_external_claim(
        const algebra::FieldElement &point,
        StandardOpeningClaim claim)
    {
        add_claim("W_ext", point, std::move(claim));
    }

    void StandardPointGroupRegistry::add_claim(
        const std::string &requested_owner,
        const algebra::FieldElement &point,
        StandardOpeningClaim claim)
    {
        if (finalized_)
        {
            throw std::runtime_error(
                "standard point-group registry cannot change after finalization");
        }

        require_claim_shape(
            claim);

        refresh_claim_execution_keys(
            &claim);

        const auto group_index =
            ensure_group(
                requested_owner,
                point);

        insert_prepared_claim(
            group_index,
            std::move(
                claim));
    }
    void StandardPointGroupRegistry::
        reserve_claim_capacity(
            const algebra::FieldElement &point,
            std::size_t expected_claim_count)
    {
        if (finalized_)
        {
            throw std::runtime_error(
                "standard point-group registry cannot reserve after finalization");
        }

        std::size_t group_index = 0;

        if (last_group_valid_ &&
            last_group_point_ == point)
        {
            group_index =
                last_group_index_;
        }
        else
        {
            const auto point_key =
                point_lookup_key(point);

            const auto it =
                group_index_by_point_.find(
                    point_key);

            if (it ==
                group_index_by_point_.end())
            {
                throw std::runtime_error(
                    "standard point-group capacity reservation requires an existing group");
            }

            group_index = it->second;

            last_group_valid_ = true;
            last_group_point_ = point;
            last_group_index_ = group_index;
        }

        groups_.at(group_index)
            .claims.reserve(
                expected_claim_count);

        claim_index_by_commitment_
            .at(group_index)
            .reserve(
                expected_claim_count);
    }

    void StandardPointGroupRegistry::finalize()
    {
        if (finalized_)
        {
            return;
        }
        if (groups_.empty())
        {
            throw std::runtime_error("standard point-group registry is empty");
        }
        for (auto &group : groups_)
        {
            if (group.claims.empty())
            {
                throw std::runtime_error("standard point group has no claims");
            }
            if (std::any_of(
                    group.claims.begin(),
                    group.claims.end(),
                    [](const StandardOpeningClaim &claim)
                    {
                        return claim.execution_commitment ==
                                   nullptr ||
                               claim.execution_commitment_identity ==
                                   std::numeric_limits<
                                       std::size_t>::max() ||
                               std::any_of(
                                   claim
                                       .canonical_text_length_prefixes
                                       .begin(),
                                   claim
                                       .canonical_text_length_prefixes
                                       .end(),
                                   [](const std::string &prefix)
                                   {
                                       return prefix.empty();
                                   });
                    }))
            {
                throw std::runtime_error(
                    "standard opening claim is missing execution cache");
            }
            std::sort(
                group.claims.begin(),
                group.claims.end(),
                canonical_claim_less);
            group.group_id = point_group_id(group.owner_block, group.point);
        }
        std::sort(
            groups_.begin(),
            groups_.end(),
            canonical_group_less);

        group_index_by_point_.clear();

        claim_index_by_commitment_.clear();

        // Claims keep stable non-owning pointers into
        // commitment_execution_identity_by_reference_.  Keep the interned
        // commitment table alive for the complete registry lifetime.
        commitment_execution_identity_by_point_
            .clear();

        last_group_valid_ = false;
        last_group_index_ = 0;

        finalized_ = true;
    }

    const std::vector<StandardPointGroup> &StandardPointGroupRegistry::groups() const
    {
        if (!finalized_)
        {
            throw std::runtime_error(
                "standard point-group registry must be finalized before use");
        }
        return groups_;
    }

    const StandardPointGroup &StandardPointGroupRegistry::group_at(
        const algebra::FieldElement &point) const
    {
        const auto &canonical_groups = groups();
        const auto it = std::find_if(
            canonical_groups.begin(), canonical_groups.end(),
            [&](const StandardPointGroup &group)
            { return group.point == point; });
        if (it == canonical_groups.end())
        {
            throw std::runtime_error("standard point-group lookup failed");
        }
        return *it;
    }

    algebra::FieldElement derive_point_group_challenge(
        const PublicMetadata &metadata,
        const StandardPointGroup &group)
    {
        if (group.claims.empty() || group.group_id != point_group_id(group.owner_block, group.point))
        {
            throw std::runtime_error(
                "invalid canonical point group for challenge derivation");
        }
        if (std::any_of(
                group.claims.begin(),
                group.claims.end(),
                [](const StandardOpeningClaim &claim)
                {
                    return claim.execution_commitment ==
                               nullptr ||
                           claim.execution_commitment_identity ==
                               std::numeric_limits<
                                   std::size_t>::max() ||
                           std::any_of(
                               claim
                                   .canonical_text_length_prefixes
                                   .begin(),
                               claim
                                   .canonical_text_length_prefixes
                                   .end(),
                               [](const std::string &prefix)
                               {
                                   return prefix.empty();
                               });
                }))
        {
            throw std::runtime_error(
                "point-group claim execution cache is missing");
        }

        if (!std::is_sorted(
                group.claims.begin(),
                group.claims.end(),
                canonical_claim_less))
        {
            throw std::runtime_error(
                "point-group claims are not in canonical tuple order");
        }

        crypto::Transcript transcript("zkgat-standard-point-group");
        absorb_public_metadata(transcript, metadata);
        transcript.absorb_text("point_group.group_id", group.group_id);
        transcript.absorb_text("point_group.owner_block", group.owner_block);
        transcript.absorb_scalar("point_group.point", group.point);
        transcript.absorb_scalar(
            "point_group.claim_count",
            algebra::FieldElement(group.claims.size()));
        for (std::size_t i = 0; i < group.claims.size(); ++i)
        {
            const auto prefix = "point_group.claim." + std::to_string(i) + ".";
            const auto &claim = group.claims[i];
            require_claim_shape(claim);
            transcript.absorb_text(prefix + "claim_id", claim.claim_id);
            transcript.absorb_text(prefix + "object_id", claim.object_id);
            transcript.absorb_text(
                prefix + "commitment_reference",
                claim.commitment_reference);
            transcript.absorb_commitment(
                prefix +
                    "commitment_point",
                claim
                    .execution_commitment
                    ->point);
            transcript.absorb_text(
                prefix + "evaluation_kind", claim.evaluation_kind);
            transcript.absorb_text(prefix + "claim_role", claim.claim_role);
            transcript.absorb_text(
                prefix + "visibility_tag", claim.visibility_tag);
            transcript.absorb_text(prefix + "value_type", claim.value_type);
            transcript.absorb_text(
                prefix + "value_reference_id", claim.value_reference_id);
            transcript.absorb_scalar(prefix + "resolved_value", claim.value);
        }
        return transcript.challenge_excluding(
            "point_group.rho",
            [](const algebra::FieldElement &candidate)
            { return candidate.is_zero(); });
    }

    void require_opening_layout(
        const std::vector<PointGroupOpening> &openings,
        const std::vector<StandardPointGroup> &expected_groups)
    {
        if (openings.size() != expected_groups.size())
        {
            throw std::runtime_error(
                "canonical point-group opening count mismatch");
        }
        std::unordered_set<std::string> group_ids;
        std::unordered_set<std::string> points;
        for (std::size_t i = 0; i < openings.size(); ++i)
        {
            const auto &opening = openings[i];
            const auto &expected = expected_groups[i];
            if (opening.group_id.empty() || opening.owner_block.empty() || opening.group_id != point_group_id(opening.owner_block, opening.point) || !group_ids.insert(opening.group_id).second || !points.insert(opening.point.to_string()).second)
            {
                throw std::runtime_error(
                    "non-canonical or duplicate point-group opening");
            }
            if (opening.group_id != expected.group_id || opening.owner_block != expected.owner_block || opening.point != expected.point)
            {
                throw std::runtime_error(
                    "point-group owner/reference/point layout mismatch");
            }
        }
    }
}
