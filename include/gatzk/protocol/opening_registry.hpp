#pragma once

#include <array>
#include <limits>
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

        // Execution-only non-owning reference into the registry's interned
        // commitment table.  It never enters proof or transcript bytes.
        const crypto::Commitment
            *execution_commitment =
                nullptr;

        algebra::FieldElement value =
            algebra::FieldElement::zero();

        // Canonical tuple ordering compares decimal length prefixes before
        // every corresponding text component.  Precompute all eight prefixes
        // once instead of running to_chars repeatedly inside std::sort.
        std::array<std::string, 8>
            canonical_text_length_prefixes;

        std::size_t
            execution_commitment_identity =
                std::numeric_limits<
                    std::size_t>::max();
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

        // Precompute execution-only commitment identities in one bounded
        // parallel wave.  Protocol-visible commitment references and points
        // remain unchanged.
        void prepare_commitment_execution_identities(
            const std::vector<crypto::Commitment>
                &commitments);

        void finalize();

        const std::vector<StandardPointGroup> &groups() const;
        const StandardPointGroup &group_at(
            const algebra::FieldElement &point) const;

    private:
        struct CommitmentExecutionIdentity
        {
            crypto::Commitment commitment;

            std::size_t identity =
                std::numeric_limits<
                    std::size_t>::max();
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

        std::unordered_map<
            std::string,
            std::size_t>
            commitment_execution_identity_by_point_;

        std::size_t
            next_commitment_execution_identity_ =
                0;

        bool last_group_valid_ = false;
        algebra::FieldElement last_group_point_ =
            algebra::FieldElement::zero();
        std::size_t last_group_index_ = 0;

        std::vector<
            std::unordered_map<
                std::size_t,
                std::size_t>>
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
        const crypto::Commitment &commitment,
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
