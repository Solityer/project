#include "gatzk/crypto/transcript.hpp"

#include <cstdint>

namespace gatzk::crypto
{
    namespace
    {
        // FNV-1a哈希函数：将字符串数据计算为64位哈希值
        std::uint64_t fnv1a(const std::string& data)
        {
            std::uint64_t hash = 1469598103934665603ULL;
            for (const unsigned char ch : data)
            {
                hash ^= ch;
                hash *= 1099511628211ULL; // FNV 素数
            }
            return hash;
        }
    }

    Transcript::Transcript(std::string label) : state_(std::move(label)) {}

    // 吸收一个标量（域元素）到Transcript中，附带标签
    void Transcript::absorb_scalar(const std::string& label, const algebra::FieldElement& value)
    {
        state_.append("|");
        state_.append(label);
        state_.append("=");
        state_.append(value.to_string());
    }

    // 吸收一个G1点（承诺值）到Transcript中，附带标签
    void Transcript::absorb_commitment(const std::string& label, const G1Point& point)
    {
        state_.append("|");
        state_.append(label);
        state_.append("=");
        for (const auto byte : serialize(point))
        {
            constexpr char kHex[] = "0123456789abcdef";
            state_.push_back(kHex[(byte >> 4U) & 0x0FU]);
            state_.push_back(kHex[byte & 0x0FU]);
        }
    }

    // 生成一个挑战值（随机数）并吸收它，同时缓存该挑战值
    algebra::FieldElement Transcript::challenge(const std::string& label)
    {
        const auto hash = fnv1a(state_ + "|" + label + "|" + std::to_string(challenges_.size()));
        const auto challenge = algebra::FieldElement(hash);
        challenges_[label] = challenge;
        absorb_scalar(label, challenge);
        return challenge;
    }

    // 返回所有已发出的挑战值（标签到挑战值的映射）
    const std::map<std::string, algebra::FieldElement>& Transcript::issued_challenges() const
    {
        return challenges_;
    }
}