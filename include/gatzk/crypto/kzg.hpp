#pragma once
// 本文件定义 formal 主线使用的 KZG 承诺接口，
// 负责承诺、批量 opening 与 verifier 侧 batch checking 的统一契约。

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/curve.hpp"

namespace gatzk::crypto
{
    // SRS / 验证键的最小实现集合。formal note 里的 `tau` 语义在此处落到具体对象上。
    struct KZGKeyPair
    {
        algebra::FieldElement tau = algebra::FieldElement::zero();
        G1Point g1_generator;
        G2Point g2_one;
        G2Point g2_tau;
        std::shared_ptr<PreparedG2> g2_one_prepared;
        std::shared_ptr<PreparedG2> g2_tau_prepared;
    };

    // 承诺对象。`tau_evaluation` 会被 prove / verify 重复消费，用来避免重复求值。
    struct Commitment
    {
        std::string name;
        G1Point point;
        algebra::FieldElement tau_evaluation = algebra::FieldElement::zero();
    };

    // 批量 opening 的执行侧剖面，供正式 benchmark 细分 `domain_opening_ms`。
    struct BatchOpeningProfile
    {
        double precompute_ms = 0.0;
        double fold_commitment_ms = 0.0;
        double finalize_ms = 0.0;
    };

    // 批量承诺的执行侧剖面，供 `commit_dynamic_ms` 拆分热点。
    struct CommitBatchProfile
    {
        double tau_eval_ms = 0.0;
        double msm_ms = 0.0;
    };

    // 序列化大小直接参与 proof_size_bytes 统计。
    std::size_t serialized_size(const Commitment& commitment);

    class KZG
    {
    public:
        // 生成一次 formal run 使用的 KZG 参数。这里只影响执行实例，不改协议结构。
        static KZGKeyPair setup(std::uint64_t seed);

        // 直接基于 `tau` 点值构造承诺，适用于已缓存 `tau_evaluation` 的公共列或动态列。
        static Commitment commit_tau_evaluation(
            const std::string& name,
            const algebra::FieldElement& tau_evaluation,
            const KZGKeyPair& key);
        // 对单个多项式做正式承诺。
        static Commitment commit(const std::string& name, const algebra::Polynomial& polynomial, const KZGKeyPair& key);
        // 对已知 `tau_evaluation` 的一批列做承诺，减少 proving 热路径中的重复 MSM 准备。
        static std::vector<Commitment> commit_tau_evaluation_batch(
            const std::vector<std::pair<std::string, algebra::FieldElement>>& named_tau_evaluations,
            const KZGKeyPair& key,
            CommitBatchProfile* profile = nullptr);
        // 对一批多项式统一承诺，是 dynamic commitment 主链的核心入口。
        static std::vector<Commitment> commit_batch(
            const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials,
            const KZGKeyPair& key,
            CommitBatchProfile* profile = nullptr);

        // 对同一域 opening bundle 中的多列做折叠 opening。
        static G1Point open_batch(
            const std::vector<Commitment>& commitments,
            const std::vector<algebra::FieldElement>& points,
            const std::vector<std::vector<algebra::FieldElement>>& claimed_values,
            const algebra::FieldElement& folding_challenge,
            const KZGKeyPair& key,
            BatchOpeningProfile* profile = nullptr);

        // verifier 侧复验 `open_batch` 生成的见证。
        static bool verify_batch(
            const std::vector<Commitment>& commitments,
            const std::vector<algebra::FieldElement>& points,
            const std::vector<std::vector<algebra::FieldElement>>& claimed_values,
            const algebra::FieldElement& folding_challenge,
            const G1Point& witness,
            const KZGKeyPair& key);

        // 对 external evaluation 列做折叠 opening，服务 `rho_ext` 外点检查。
        static G1Point open_external_fold(
            const std::vector<std::pair<Commitment, algebra::FieldElement>>& commitments_and_values,
            const std::vector<algebra::FieldElement>& points,
            
            const algebra::FieldElement& folding_challenge,
            const KZGKeyPair& key);

        // verifier 侧复验 external fold opening。
        static bool verify_external_fold(
            const std::vector<std::pair<Commitment, algebra::FieldElement>>& commitments_and_values,
            const std::vector<algebra::FieldElement>& points,
            const algebra::FieldElement& folding_challenge,
            const G1Point& witness,
            const KZGKeyPair& key);
    };
}
