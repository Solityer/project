#include "gatzk/crypto/transcript.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <openssl/sha.h>

namespace gatzk::crypto
{
    namespace
    {
        constexpr std::uint8_t kDomainRecord = 1;
        constexpr std::uint8_t kTextRecord = 2;
        constexpr std::uint8_t kScalarRecord = 3;
        constexpr std::uint8_t kCommitmentRecord = 4;
        constexpr std::uint8_t kChallengeRecord = 5;
        constexpr std::uint8_t kChallengeCandidateRecord = 6;

        void append_u64(std::vector<std::uint8_t> &out, std::uint64_t value)
        {
            for (std::size_t i = 0; i < 8; ++i)
            {
                out.push_back(static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU));
            }
        }

        void append_u32(std::vector<std::uint8_t> &out, std::uint32_t value)
        {
            for (std::size_t i = 0; i < 4; ++i)
            {
                out.push_back(static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU));
            }
        }

        std::vector<std::uint8_t> canonical_record(std::uint8_t type, const std::string &label, const std::uint8_t *payload, std::size_t payload_size)
        {
            if (label.size() > std::numeric_limits<std::uint64_t>::max() || payload_size > std::numeric_limits<std::uint64_t>::max())
            {
                throw std::runtime_error("transcript record exceeds canonical u64 length");
            }
            std::vector<std::uint8_t> out;
            out.reserve(1 + 8 + label.size() + 8 + payload_size);
            out.push_back(type);
            append_u64(out, static_cast<std::uint64_t>(label.size()));
            out.insert(out.end(), label.begin(), label.end());
            append_u64(out, static_cast<std::uint64_t>(payload_size));
            if (payload_size != 0)
            {
                if (payload == nullptr)
                {
                    throw std::runtime_error("transcript record has null payload");
                }
                out.insert(out.end(), payload, payload + payload_size);
            }
            return out;
        }

        std::array<std::uint8_t, SHA256_DIGEST_LENGTH> sha256(const std::uint8_t *first, std::size_t first_size, const std::uint8_t *second,
                                                              std::size_t second_size)
        {
            SHA256_CTX hash;
            SHA256_Init(&hash);
            if (first_size != 0)
                SHA256_Update(&hash, first, first_size);
            if (second_size != 0)
                SHA256_Update(&hash, second, second_size);
            std::array<std::uint8_t, SHA256_DIGEST_LENGTH> digest{};
            SHA256_Final(digest.data(), &hash);
            return digest;
        }
    }

    Transcript::Transcript(std::string label)
    {
        const auto record = canonical_record(kDomainRecord, "domain", reinterpret_cast<const std::uint8_t *>(label.data()), label.size());
        state_ = sha256(nullptr, 0, record.data(), record.size());
    }

    void Transcript::absorb_record(std::uint8_t type, const std::string &label, const std::uint8_t *payload, std::size_t payload_size)
    {
        const auto record = canonical_record(type, label, payload, payload_size);
        state_ = sha256(state_.data(), state_.size(), record.data(), record.size());
    }

    void Transcript::absorb_text(const std::string &label, const std::string &value)
    {
        absorb_record(kTextRecord, label, reinterpret_cast<const std::uint8_t *>(value.data()), value.size());
    }

    void Transcript::absorb_scalar(const std::string &label, const algebra::FieldElement &value)
    {
        std::array<std::uint8_t, 32> bytes{};
        value.write_little_endian(bytes.data(), bytes.size());
        absorb_record(kScalarRecord, label, bytes.data(), bytes.size());
    }

    void Transcript::absorb_commitment(const std::string &label, const G1Point &point)
    {
        const auto bytes = serialize(point);
        absorb_record(kCommitmentRecord, label, bytes.data(), bytes.size());
    }

    algebra::FieldElement Transcript::challenge(const std::string &label)
    {
        return challenge_excluding(label, [](const algebra::FieldElement &) { return false; });
    }

    algebra::FieldElement Transcript::challenge_excluding(const std::string &label, const std::function<bool(const algebra::FieldElement &)> &reject)
    {
        if (challenges_.contains(label))
        {
            throw std::runtime_error("duplicate transcript challenge label: " + label);
        }
        for (std::uint64_t counter = 0; counter <= std::numeric_limits<std::uint32_t>::max(); ++counter)
        {
            std::vector<std::uint8_t> counter_bytes;
            counter_bytes.reserve(4);
            append_u32(counter_bytes, static_cast<std::uint32_t>(counter));
            const auto candidate_record = canonical_record(kChallengeCandidateRecord, label, counter_bytes.data(), counter_bytes.size());
            const auto digest = sha256(state_.data(), state_.size(), candidate_record.data(), candidate_record.size());

            mcl::Vint integer;
            integer.setArray(digest.data(), digest.size());
            if (integer >= mcl::Fr::getOp().mp)
            {
                continue;
            }
            const auto value = algebra::FieldElement::from_little_endian_mod(digest.data(), digest.size());
            if (reject(value))
            {
                continue;
            }
            challenges_.emplace(label, value);
            std::array<std::uint8_t, 32> canonical{};
            value.write_little_endian(canonical.data(), canonical.size());
            absorb_record(kChallengeRecord, label, canonical.data(), canonical.size());
            return value;
        }
        throw std::runtime_error("transcript hash-to-field counter exhausted");
    }

    const std::map<std::string, algebra::FieldElement> &Transcript::issued_challenges() const { return challenges_; }
}
