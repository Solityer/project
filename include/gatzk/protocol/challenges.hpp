#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/protocol/proof.hpp"

namespace gatzk::protocol {

std::vector<std::string> dynamic_commitment_labels(const ProtocolContext& context);
std::vector<std::string> quotient_commitment_labels(const ProtocolContext& context);

std::map<std::string, algebra::FieldElement> replay_challenges(
    const ProtocolContext& context,
    const std::unordered_map<std::string, crypto::Commitment>& dynamic_commitments,
    const std::unordered_map<std::string, crypto::Commitment>& quotient_commitments);

}  // namespace gatzk::protocol
