#include "gatzk/crypto/transcript.hpp"

#include <cstdint>

namespace gatzk::crypto {
namespace {

std::uint64_t fnv1a(const std::string& data) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : data) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

Transcript::Transcript(std::string label) : state_(std::move(label)) {}

void Transcript::absorb_scalar(const std::string& label, const algebra::FieldElement& value) {
    state_.append("|");
    state_.append(label);
    state_.append("=");
    state_.append(value.to_string());
}

void Transcript::absorb_commitment(const std::string& label, const G1Point& point) {
    state_.append("|");
    state_.append(label);
    state_.append("=");
    for (const auto byte : serialize(point)) {
        constexpr char kHex[] = "0123456789abcdef";
        state_.push_back(kHex[(byte >> 4U) & 0x0FU]);
        state_.push_back(kHex[byte & 0x0FU]);
    }
}

algebra::FieldElement Transcript::challenge(const std::string& label) {
    const auto hash = fnv1a(state_ + "|" + label + "|" + std::to_string(challenges_.size()));
    const auto challenge = algebra::FieldElement(hash);
    challenges_[label] = challenge;
    absorb_scalar(label, challenge);
    return challenge;
}

const std::map<std::string, algebra::FieldElement>& Transcript::issued_challenges() const {
    return challenges_;
}

}  // namespace gatzk::crypto
