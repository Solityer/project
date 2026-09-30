#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/crypto/curve.hpp"

namespace gatzk::crypto
{
    class Transcript
    {
    public:
        explicit Transcript(std::string label);

        void absorb_text(const std::string &label, const std::string &value);
        void absorb_scalar(const std::string &label, const algebra::FieldElement &value);
        void absorb_commitment(const std::string &label, const G1Point &point);
        algebra::FieldElement challenge(const std::string &label);
        algebra::FieldElement challenge_excluding(const std::string &label, const std::function<bool(const algebra::FieldElement &)> &reject);

        const std::map<std::string, algebra::FieldElement> &issued_challenges() const;

    private:
        void absorb_record(std::uint8_t type, const std::string &label, const std::uint8_t *payload, std::size_t payload_size);

        std::array<std::uint8_t, 32> state_{};
        std::map<std::string, algebra::FieldElement> challenges_;
    };
}
