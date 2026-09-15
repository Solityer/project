#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{
    // Complete canonical tuple absorbed before deriving rho_z.  `commitment`
    // and `value` are resolved runtime data; the remaining fields are the
    // schema-bound identities of that data.
    struct StandardOpeningClaim
    {
        std::string claim_id;
        std::string object_id;
        std::string commitment_reference;
        std::string evaluation_kind;
        std::string claim_role;
        std::string visibility_tag;
        std::string value_type;
        std::string value_reference_id;
        crypto::Commitment commitment;
        algebra::FieldElement value = algebra::FieldElement::zero();

        // Execution-only caches. These bytes are never absorbed into the
        // transcript, serialized into the proof, or used as protocol identity.
        std::string execution_commitment_key;
        std::string execution_order_key;
    };

    struct StandardPointGroup
    {
        std::string group_id;
        std::string owner_block;
        algebra::FieldElement point = algebra::FieldElement::zero();
        std::vector<StandardOpeningClaim> claims;
    };

    class StandardPointGroupRegistry
    {
    public:
        // Domain owners are registered before any challenge is derived.
        void add_domain_claim(
            const std::string &domain,
            const algebra::FieldElement &point,
            StandardOpeningClaim claim);
        // External/route/binding claims merge into an existing domain owner
        // whenever the canonical field point is equal.
        void add_external_claim(
            const algebra::FieldElement &point,
            StandardOpeningClaim claim);
        // Execution-only capacity hint. It changes neither claim identity nor
        // canonical ordering.
        void reserve_claim_capacity(
            const algebra::FieldElement &point,
            std::size_t expected_claim_count);
        void finalize();

        const std::vector<StandardPointGroup> &groups() const;
        const StandardPointGroup &group_at(
            const algebra::FieldElement &point) const;

    private:
        struct CommitmentExecutionIdentity
        {
            crypto::G1Point point;
            std::string serialized_key;
        };

        void refresh_claim_execution_keys(
            StandardOpeningClaim *claim);

    private:
        void add_claim(
            const std::string &requested_owner,
            const algebra::FieldElement &point,
            StandardOpeningClaim claim);

        std::vector<StandardPointGroup> groups_;
        std::unordered_map<std::string, std::size_t> group_index_by_point_;
        std::unordered_map<
            std::string,
            CommitmentExecutionIdentity>
            commitment_execution_identity_by_reference_;

        bool last_group_valid_ = false;
        algebra::FieldElement last_group_point_ =
            algebra::FieldElement::zero();
        std::size_t last_group_index_ = 0;
        std::vector<std::unordered_map<std::string, std::size_t>>
            claim_index_by_commitment_;
        bool finalized_ = false;
    };

    std::string point_group_id(
        const std::string &owner_block,
        const algebra::FieldElement &point);

    StandardOpeningClaim make_standard_opening_claim(
        std::string claim_id,
        std::string object_id,
        std::string evaluation_kind,
        std::string claim_role,
        PayloadValueKind visibility,
        std::string value_reference_id,
        crypto::Commitment commitment,
        const algebra::FieldElement &value);

    void add_domain_bundle_claims(
        StandardPointGroupRegistry &registry,
        const std::string &domain,
        const DomainOpeningBundle &bundle,
        const std::vector<crypto::Commitment> &commitments);

    void add_external_claims(
        StandardPointGroupRegistry &registry,
        const std::vector<std::pair<crypto::Commitment,
                                    algebra::FieldElement>> &commitments_and_values,
        const std::vector<algebra::FieldElement> &points,
        const std::vector<PayloadValueKind> &eval_ext_kinds);

    algebra::FieldElement derive_point_group_challenge(
        const PublicMetadata &metadata,
        const StandardPointGroup &group);

    void require_opening_layout(
        const std::vector<PointGroupOpening> &openings,
        const std::vector<StandardPointGroup> &expected_groups);
}
