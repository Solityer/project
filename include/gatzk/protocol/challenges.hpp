#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/crypto/transcript.hpp"
#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol
{
    PublicMetadata canonical_public_metadata(const ProtocolContext &context);
    void absorb_public_metadata(crypto::Transcript &transcript, const PublicMetadata &metadata);
    void absorb_binding_relation_metadata(crypto::Transcript &transcript, const BindingRelationMeta &metadata);
    std::vector<std::string> dynamic_commitment_labels(const ProtocolContext &context);
    std::vector<std::string> quotient_commitment_labels(const ProtocolContext &context);
    std::string quotient_mask_label(const std::string &quotient_label);

    struct PublicOutputChallenges
    {
        algebra::FieldElement point = algebra::FieldElement::zero();
    };

    PublicOutputChallenges derive_public_output_challenges(const PublicMetadata &metadata, const crypto::Commitment &output_commitment,
                                                           const std::vector<algebra::FieldElement> &public_logits, std::size_t node_count,
                                                           std::size_t class_count);

    std::string canonical_output_digest(const std::vector<algebra::FieldElement> &public_logits, std::size_t node_count, std::size_t class_count);
    std::string canonical_public_output_descriptor(const std::vector<algebra::FieldElement> &public_logits, std::size_t node_count, std::size_t class_count);
    std::string fresh_proof_nonce();

    std::map<std::string, algebra::FieldElement> derive_domain_opening_challenges(const ProtocolContext &context, crypto::Transcript &transcript);
    std::map<std::string, algebra::FieldElement> replay_challenges(const ProtocolContext &context,
                                                                   const std::unordered_map<std::string, crypto::Commitment> &dynamic_commitments,
                                                                   const std::unordered_map<std::string, crypto::Commitment> &quotient_commitments,
                                                                   const PublicMetadata *proof_metadata = nullptr);

}
