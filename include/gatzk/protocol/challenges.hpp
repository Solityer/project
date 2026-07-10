#pragma once
// 本文件集中定义 transcript 重放与挑战标签管理，
// 是 prove / verify 共享的 challenge 真源接口。

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
    // 生成对外公开的元数据视图，README / manifest / verifier 都以它为准。
    PublicMetadata canonical_public_metadata(const ProtocolContext& context);
    // 按固定字段顺序把公开元数据吸收到 transcript。
    void absorb_public_metadata(crypto::Transcript& transcript, const PublicMetadata& metadata);
    // 返回动态承诺的正式顺序，直接约束 proof schema 与 artifact contract。
    std::vector<std::string> dynamic_commitment_labels(const ProtocolContext& context);
    // 返回 quotient 承诺的正式顺序。
    std::vector<std::string> quotient_commitment_labels(const ProtocolContext& context);

    struct PublicOutputChallenges {
        algebra::FieldElement point = algebra::FieldElement::zero();
        algebra::FieldElement folding = algebra::FieldElement::zero();
    };

    // 公开 logits 必须在独立 transcript 中先吸收完整声明，再生成 KZG
    // 绑定点，防止看到旧挑战后替换为同一单点求值的另一组 logits。
    PublicOutputChallenges derive_public_output_challenges(
        const PublicMetadata& metadata,
        const crypto::Commitment& output_commitment,
        const std::vector<algebra::FieldElement>& public_logits,
        std::size_t node_count,
        std::size_t class_count);

    // 从公开元数据、动态承诺与 quotient 承诺重放全部挑战，是 verifier 的真值检查入口。
    std::map<std::string, algebra::FieldElement> replay_challenges(
        const ProtocolContext& context,
        const std::unordered_map<std::string, crypto::Commitment>& dynamic_commitments,
        const std::unordered_map<std::string, crypto::Commitment>& quotient_commitments);

}  // namespace gatzk::protocol
