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
    void absorb_binding_relation_metadata(
        crypto::Transcript& transcript,
        const BindingRelationMeta& metadata);
    // 返回动态承诺的正式顺序，直接约束 proof schema 与 artifact contract。
    std::vector<std::string> dynamic_commitment_labels(const ProtocolContext& context);
    // 返回 quotient 承诺的正式顺序。
    std::vector<std::string> quotient_commitment_labels(const ProtocolContext& context);
    std::string quotient_mask_label(const std::string& quotient_label);

    struct PublicOutputChallenges {
        algebra::FieldElement point = algebra::FieldElement::zero();
    };

    // 公开 logits 必须在独立 transcript 中先吸收完整声明，再生成 KZG
    // 绑定点，防止看到旧挑战后替换为同一单点求值的另一组 logits。
    PublicOutputChallenges derive_public_output_challenges(
        const PublicMetadata& metadata,
        const crypto::Commitment& output_commitment,
        const std::vector<algebra::FieldElement>& public_logits,
        std::size_t node_count,
        std::size_t class_count);

    std::string canonical_output_digest(
        const std::vector<algebra::FieldElement>& public_logits,
        std::size_t node_count,
        std::size_t class_count);
    std::string canonical_public_output_descriptor(
        const std::vector<algebra::FieldElement>& public_logits,
        std::size_t node_count,
        std::size_t class_count);
    std::string fresh_proof_nonce();

    algebra::FieldElement derive_batch_opening_folding_challenge(
        const PublicMetadata& metadata,
        const std::string& group_id,
        const std::vector<crypto::Commitment>& commitments,
        const std::vector<algebra::FieldElement>& points,
        const std::vector<std::vector<algebra::FieldElement>>& values);
    algebra::FieldElement derive_external_opening_folding_challenge(
        const PublicMetadata& metadata,
        const std::vector<std::pair<crypto::Commitment, algebra::FieldElement>>& claims,
        const std::vector<algebra::FieldElement>& points);
    // Derive the seven work-domain opening points with the canonical public
    // rejection rules: every z is outside its domain and all z,z*omega pairs
    // are mutually distinct across FH,edge,in,d_h,cat,C,N.
    std::map<std::string, algebra::FieldElement> derive_domain_opening_challenges(
        const ProtocolContext& context,
        crypto::Transcript& transcript);
    // 从公开元数据、动态承诺与 quotient 承诺重放全部挑战，是 verifier 的真值检查入口。
    std::map<std::string, algebra::FieldElement> replay_challenges(
        const ProtocolContext& context,
        const std::unordered_map<std::string, crypto::Commitment>& dynamic_commitments,
        const std::unordered_map<std::string, crypto::Commitment>& quotient_commitments,
        const PublicMetadata* proof_metadata = nullptr);

}  // namespace gatzk::protocol
