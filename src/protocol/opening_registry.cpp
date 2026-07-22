#include "gatzk/protocol/opening_registry.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

#include "gatzk/crypto/curve.hpp"
#include "gatzk/crypto/transcript.hpp"
#include "gatzk/protocol/challenges.hpp"

namespace gatzk::protocol
{
    namespace
    {
        std::string claim_order_key(const StandardOpeningClaim& claim)
        {
            const auto commitment_bytes = crypto::serialize(
                claim.commitment.point);
            std::string key;
            const auto append = [&](const std::string& value)
            {
                key.append(std::to_string(value.size()));
                key.push_back(':');
                key.append(value);
                key.push_back('|');
            };
            append(claim.claim_id);
            append(claim.object_id);
            append(claim.commitment_reference);
            key.append(
                reinterpret_cast<const char*>(commitment_bytes.data()),
                commitment_bytes.size());
            key.push_back('|');
            append(claim.evaluation_kind);
            append(claim.claim_role);
            append(claim.visibility_tag);
            append(claim.value_type);
            append(claim.value_reference_id);
            return key;
        }

        std::string commitment_point_key(const crypto::G1Point& point)
        {
            const auto bytes = crypto::serialize(point);
            return std::string(
                reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }

        void require_claim_shape(const StandardOpeningClaim& claim)
        {
            for (const auto* field : {
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
            if (claim.commitment_reference != claim.commitment.name)
            {
                throw std::runtime_error(
                    "standard opening claim commitment reference/name mismatch");
            }
            if (claim.visibility_tag != "public"
                && claim.visibility_tag != "masked_private")
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
            const StandardPointGroup& lhs,
            const StandardPointGroup& rhs)
        {
            return lhs.point.to_string() < rhs.point.to_string();
        }
    }

    std::string point_group_id(
        const std::string& owner_block,
        const algebra::FieldElement& point)
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
        crypto::Commitment commitment,
        const algebra::FieldElement& value)
    {
        const auto commitment_reference = commitment.name;
        return StandardOpeningClaim{
            .claim_id = std::move(claim_id),
            .object_id = std::move(object_id),
            .commitment_reference = commitment_reference,
            .evaluation_kind = std::move(evaluation_kind),
            .claim_role = std::move(claim_role),
            .visibility_tag = visibility == PayloadValueKind::PublicValue
                ? "public"
                : "masked_private",
            .value_type = "field_element",
            .value_reference_id = std::move(value_reference_id),
            .commitment = std::move(commitment),
            .value = value,
        };
    }

    void add_domain_bundle_claims(
        StandardPointGroupRegistry& registry,
        const std::string& domain,
        const DomainOpeningBundle& bundle,
        const std::vector<crypto::Commitment>& commitments)
    {
        if (bundle.points.empty()
            || bundle.values.size() != commitments.size()
            || bundle.value_kinds.size() != commitments.size())
        {
            throw std::runtime_error(
                "invalid domain bundle for standard point-group registry");
        }
        for (std::size_t object_index = 0;
             object_index < commitments.size(); ++object_index)
        {
            if (bundle.values[object_index].first
                    != commitments[object_index].name
                || bundle.values[object_index].second.size()
                    != bundle.points.size())
            {
                throw std::runtime_error(
                    "domain bundle commitment/value order mismatch");
            }
            const auto& object_id = bundle.values[object_index].first;
            const auto claim_role = object_id.rfind("t_", 0) == 0
                ? "quotient"
                : object_id.rfind("R_t_", 0) == 0
                    ? "quotient_blinder"
                    : "domain_polynomial";
            for (std::size_t point_index = 0;
                 point_index < bundle.points.size(); ++point_index)
            {
                const auto suffix = domain + ":" + object_id + ":"
                    + std::to_string(point_index);
                registry.add_domain_claim(
                    domain,
                    bundle.points[point_index],
                    make_standard_opening_claim(
                        "claim:domain:" + suffix,
                        object_id,
                        "domain_evaluation",
                        claim_role,
                        bundle.value_kinds[object_index],
                        "Eval_dom:" + suffix,
                        commitments[object_index],
                        bundle.values[object_index].second[point_index]));
            }
        }
    }

    void add_external_claims(
        StandardPointGroupRegistry& registry,
        const std::vector<std::pair<crypto::Commitment,
            algebra::FieldElement>>& commitments_and_values,
        const std::vector<algebra::FieldElement>& points,
        const std::vector<PayloadValueKind>& eval_ext_kinds)
    {
        if (commitments_and_values.size() != points.size()
            || eval_ext_kinds.size() > commitments_and_values.size())
        {
            throw std::runtime_error(
                "invalid external claims for standard point-group registry");
        }
        for (std::size_t i = 0; i < commitments_and_values.size(); ++i)
        {
            const auto& [commitment, value] = commitments_and_values[i];
            const bool is_eval_ext = i < eval_ext_kinds.size();
            const bool is_public_output = commitment.name == "P_Y";
            const bool is_public_relation =
                commitment.name.find("relation_virtual")
                    != std::string::npos
                || commitment.name.ends_with("_m_can");
            const auto kind = is_eval_ext
                ? eval_ext_kinds[i]
                : (is_public_output || is_public_relation
                    ? PayloadValueKind::PublicValue
                    : PayloadValueKind::MaskedPrivateValue);
            const auto evaluation_kind = is_eval_ext
                ? "external_evaluation"
                : is_public_output
                    ? "verifier_reconstructed_public_output"
                    : "binding_evaluation";
            const auto claim_role = is_public_output
                ? "public_output"
                : is_public_relation
                    ? "binding_relation"
                    : is_eval_ext
                        ? "ordinary_external"
                        : "binding_shift";
            const auto point_text = points[i].to_string();
            const auto identity = commitment.name + ":" + point_text;
            registry.add_external_claim(
                points[i],
                make_standard_opening_claim(
                    "claim:external:" + identity,
                    commitment.name,
                    evaluation_kind,
                    claim_role,
                    kind,
                    (is_eval_ext ? "Eval_ext:" : is_public_output
                        ? "M_pub:public_logits:"
                        : "Eval_bind:") + identity,
                    commitment,
                    value));
        }
    }

    void StandardPointGroupRegistry::add_domain_claim(
        const std::string& domain,
        const algebra::FieldElement& point,
        StandardOpeningClaim claim)
    {
        if (domain.empty())
        {
            throw std::runtime_error("standard domain point group has empty domain");
        }
        add_claim("Open_dom:" + domain, point, std::move(claim));
    }

    void StandardPointGroupRegistry::add_external_claim(
        const algebra::FieldElement& point,
        StandardOpeningClaim claim)
    {
        add_claim("W_ext", point, std::move(claim));
    }

    void StandardPointGroupRegistry::add_claim(
        const std::string& requested_owner,
        const algebra::FieldElement& point,
        StandardOpeningClaim claim)
    {
        if (finalized_)
        {
            throw std::runtime_error(
                "standard point-group registry cannot change after finalization");
        }
        require_claim_shape(claim);
        const auto point_key = point.to_string();
        auto group_position = group_index_by_point_.find(point_key);
        std::size_t group_index = 0;
        if (group_position == group_index_by_point_.end())
        {
            group_index = groups_.size();
            groups_.push_back(StandardPointGroup{
                .group_id = point_group_id(requested_owner, point),
                .owner_block = requested_owner,
                .point = point,
                .claims = {},
            });
            group_index_by_point_.emplace(point_key, group_index);
            claim_index_by_commitment_.emplace_back();
        }
        else
        {
            group_index = group_position->second;
        }
        auto& group = groups_.at(group_index);
        if (requested_owner.rfind("Open_dom:", 0) == 0)
        {
            if (group.owner_block.rfind("Open_dom:", 0) == 0
                && group.owner_block != requested_owner)
            {
                throw std::runtime_error(
                    "distinct work domains resolved to the same opening point");
            }
            group.owner_block = requested_owner;
            group.group_id = point_group_id(requested_owner, point);
        }

        auto& claim_positions = claim_index_by_commitment_.at(group_index);
        const auto commitment_key = commitment_point_key(
            claim.commitment.point);
        const auto duplicate_position = claim_positions.find(commitment_key);
        if (duplicate_position == claim_positions.end())
        {
            claim_positions.emplace(commitment_key, group.claims.size());
            group.claims.push_back(std::move(claim));
            return;
        }
        auto& duplicate = group.claims.at(duplicate_position->second);
        if (duplicate.value != claim.value)
        {
            // This check deliberately happens while building the registry,
            // before any point-group challenge can be requested.
            throw std::runtime_error(
                "conflicting values for the same commitment-point opening claim");
        }
        if (claim_order_key(claim) < claim_order_key(duplicate))
        {
            duplicate = std::move(claim);
        }
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
        for (auto& group : groups_)
        {
            if (group.claims.empty())
            {
                throw std::runtime_error("standard point group has no claims");
            }
            std::vector<std::pair<std::string, StandardOpeningClaim>> keyed_claims;
            keyed_claims.reserve(group.claims.size());
            for (auto& claim : group.claims)
            {
                keyed_claims.push_back({
                    claim_order_key(claim), std::move(claim)});
            }
            std::sort(
                keyed_claims.begin(), keyed_claims.end(),
                [](const auto& lhs, const auto& rhs)
                { return lhs.first < rhs.first; });
            group.claims.clear();
            group.claims.reserve(keyed_claims.size());
            for (auto& keyed_claim : keyed_claims)
            {
                group.claims.push_back(std::move(keyed_claim.second));
            }
            group.group_id = point_group_id(group.owner_block, group.point);
        }
        std::sort(groups_.begin(), groups_.end(), canonical_group_less);
        group_index_by_point_.clear();
        claim_index_by_commitment_.clear();
        finalized_ = true;
    }

    const std::vector<StandardPointGroup>& StandardPointGroupRegistry::groups() const
    {
        if (!finalized_)
        {
            throw std::runtime_error(
                "standard point-group registry must be finalized before use");
        }
        return groups_;
    }

    const StandardPointGroup& StandardPointGroupRegistry::group_at(
        const algebra::FieldElement& point) const
    {
        const auto& canonical_groups = groups();
        const auto it = std::find_if(
            canonical_groups.begin(), canonical_groups.end(),
            [&](const StandardPointGroup& group)
            { return group.point == point; });
        if (it == canonical_groups.end())
        {
            throw std::runtime_error("standard point-group lookup failed");
        }
        return *it;
    }

    algebra::FieldElement derive_point_group_challenge(
        const PublicMetadata& metadata,
        const StandardPointGroup& group)
    {
        if (group.claims.empty()
            || group.group_id != point_group_id(group.owner_block, group.point))
        {
            throw std::runtime_error(
                "invalid canonical point group for challenge derivation");
        }
        if (!std::is_sorted(
                group.claims.begin(), group.claims.end(),
                [](const StandardOpeningClaim& lhs,
                   const StandardOpeningClaim& rhs)
                { return claim_order_key(lhs) < claim_order_key(rhs); }))
        {
            throw std::runtime_error(
                "point-group claims are not in canonical tuple order");
        }

        crypto::Transcript transcript("zkgat-standard-point-group-v2");
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
            const auto& claim = group.claims[i];
            require_claim_shape(claim);
            transcript.absorb_text(prefix + "claim_id", claim.claim_id);
            transcript.absorb_text(prefix + "object_id", claim.object_id);
            transcript.absorb_text(
                prefix + "commitment_reference",
                claim.commitment_reference);
            transcript.absorb_commitment(
                prefix + "commitment_point", claim.commitment.point);
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
            [](const algebra::FieldElement& candidate)
            { return candidate.is_zero(); });
    }

    void require_opening_layout(
        const std::vector<PointGroupOpening>& openings,
        const std::vector<StandardPointGroup>& expected_groups)
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
            const auto& opening = openings[i];
            const auto& expected = expected_groups[i];
            if (opening.group_id.empty()
                || opening.owner_block.empty()
                || opening.group_id
                    != point_group_id(opening.owner_block, opening.point)
                || !group_ids.insert(opening.group_id).second
                || !points.insert(opening.point.to_string()).second)
            {
                throw std::runtime_error(
                    "non-canonical or duplicate point-group opening");
            }
            if (opening.group_id != expected.group_id
                || opening.owner_block != expected.owner_block
                || opening.point != expected.point)
            {
                throw std::runtime_error(
                    "point-group owner/reference/point layout mismatch");
            }
        }
    }
}
