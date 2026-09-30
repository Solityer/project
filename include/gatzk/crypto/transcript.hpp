#pragma once
// 本文件提供正式主线的 transcript 包装，
// 所有挑战都必须经由这里按固定顺序吸收公开量与承诺后生成。

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
    // 简化版 transcript 状态机。它不定义协议顺序，只执行 protocol 层给出的吸收顺序。
    class Transcript
    {
    public:
        explicit Transcript(std::string label);

        // Absorb canonical UTF-8 bytes using the transcript's type-tagged,
        // u64-length-prefixed record encoding.
        void absorb_text(const std::string &label, const std::string &value);
        // 将公开标量或中间外点值吸收到 transcript。
        void absorb_scalar(const std::string &label, const algebra::FieldElement &value);
        // 将承诺点按标签吸收到 transcript，供后续挑战重放。
        void absorb_commitment(const std::string &label, const G1Point &point);
        // 基于当前状态派生一个挑战，并记录到 proof 中供 verifier 重放。
        algebra::FieldElement challenge(const std::string &label);
        // Canonical public rejection sampling. The predicate must depend only
        // on already-absorbed public RelationMeta and the candidate value; true
        // rejects the candidate and increments the public u32 counter.
        algebra::FieldElement challenge_excluding(const std::string &label, const std::function<bool(const algebra::FieldElement &)> &reject);

        const std::map<std::string, algebra::FieldElement> &issued_challenges() const;

    private:
        void absorb_record(std::uint8_t type, const std::string &label, const std::uint8_t *payload, std::size_t payload_size);

        std::array<std::uint8_t, 32> state_{};
        std::map<std::string, algebra::FieldElement> challenges_;
    };
}
