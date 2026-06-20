#pragma once
// 本文件定义 formal 主线使用的 KZG 承诺接口，
// 负责承诺、批量 opening 与 verifier 侧 batch checking 的统一契约。

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/crypto/curve.hpp"

namespace gatzk::crypto
{
    struct SRSManifest
    {
        std::string srs_id;
        std::string curve_id;
        std::string field_modulus_id;
        std::string srs_type;
        std::string ceremony_transcript_hash;
        std::size_t degree_bound = 0;
        std::unordered_map<std::string, std::size_t> domain_sizes;
        std::string pk_coef_path;
        std::string pk_coef_hash;
        std::unordered_map<std::string, std::string> pk_lagrange_paths;
        std::unordered_map<std::string, std::string> pk_lagrange_hashes;
        std::string vk_path;
        std::string vk_hash;
        std::string encoding;
        std::string generator_version;
        std::string security_level;
        std::string manifest_hash;
    };

    struct KZGProverKey
    {
        std::string srs_id;
        std::string srs_type;
        std::size_t degree_bound = 0;
        G1Point g1_generator;
        std::shared_ptr<const std::vector<G1Point>> powers_g1;
        std::unordered_map<std::string, std::shared_ptr<const std::vector<G1Point>>> lagrange_srs_by_domain;
        std::string manifest_hash;
    };

    struct KZGVerifierKey
    {
        std::string srs_id;
        std::string srs_type;
        std::size_t degree_bound = 0;
        G1Point g1_generator;
        G2Point g2_one;
        G2Point g2_tau;
        std::shared_ptr<PreparedG2> g2_one_prepared;
        std::shared_ptr<PreparedG2> g2_tau_prepared;
        std::string manifest_hash;
    };

    // SRS / 验证键的最小实现集合。主程序只从 manifest/key files 加载，不从实验 seed 生成。
    struct KZGKeyPair
    {
        SRSManifest manifest;
        KZGProverKey pk;
        KZGVerifierKey vk;
        G1Point g1_generator;
        G2Point g2_one;
        G2Point g2_tau;
        std::shared_ptr<PreparedG2> g2_one_prepared;
        std::shared_ptr<PreparedG2> g2_tau_prepared;
    };

    // 承诺对象只携带 proof 可见的群元素；prover 侧可保留多项式材料用于生成 opening witness。
    struct Commitment
    {
        std::string name;
        G1Point point;
        std::shared_ptr<const algebra::Polynomial> opening_polynomial;
    };

    // 批量 opening 的执行侧剖面，供正式 benchmark 细分 `domain_opening_ms`。
    struct BatchOpeningProfile
    {
        double precompute_ms = 0.0;
        double fold_commitment_ms = 0.0;
        double finalize_ms = 0.0;
        double denominator_inverse_ms = 0.0;
        double denominator_inverse_reuse_ms = 0.0;
        std::size_t denominator_inverse_reuse_hits = 0;
        std::size_t denominator_inverse_cache_misses = 0;
        std::size_t point_parallel_batches = 0;
        std::size_t point_parallel_workers = 0;
        std::size_t point_msm_worker_cap = 0;
    };

    // 批量承诺的执行侧剖面，供 `commit_dynamic_ms` 拆分热点。
    struct CommitObjectProfile
    {
        std::string name;
        std::string basis;
        std::string domain;
        std::size_t vector_length = 0;
        std::size_t nonzero_scalar_count = 0;
        std::size_t effective_scalar_count = 0;
        bool zero_scalar_skipping = false;
        bool sparse_compaction = false;
        bool constant_tail_folding = false;
        bool multithreaded_msm = false;
        double msm_ms = 0.0;
    };

    struct CommitBatchProfile
    {
        double evaluation_ms = 0.0;
        double msm_ms = 0.0;
        std::vector<CommitObjectProfile> objects;
    };

    // 序列化大小直接参与 proof_size_bytes 统计。
    std::size_t serialized_size(const Commitment& commitment);

    struct OpeningProof
    {
        std::vector<G1Point> witnesses;
    };

    std::size_t serialized_size(const OpeningProof& proof);

    class KZG
    {
    public:
        // 加载一次 secure formal run 使用的 SRS 参数。
        static KZGKeyPair load_keys(
            const std::string& srs_path,
            const std::string& expected_srs_type,
            const std::string& expected_srs_id);

        // 对常数多项式做正式 SRS 承诺；用于公开常数或已折叠的标量关系。
        static Commitment commit_constant(
            const std::string& name,
            const algebra::FieldElement& value,
            const KZGKeyPair& key);
        // 对单个多项式做正式承诺。
        static Commitment commit(const std::string& name, const algebra::Polynomial& polynomial, const KZGKeyPair& key);
        // 对一批常数多项式做正式 SRS 承诺。
        static std::vector<Commitment> commit_constant_batch(
            const std::vector<std::pair<std::string, algebra::FieldElement>>& named_values,
            const KZGKeyPair& key,
            CommitBatchProfile* profile = nullptr);
        // 对一批多项式统一承诺，是 dynamic commitment 主链的核心入口。
        static std::vector<Commitment> commit_batch(
            const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials,
            const KZGKeyPair& key,
            CommitBatchProfile* profile = nullptr);
        // 对刚物化出的多项式批量承诺并接管 opening 材料，避免为 quotient 等一次性对象再复制整列标量。
        static std::vector<Commitment> commit_owned_batch(
            std::vector<std::pair<std::string, algebra::Polynomial>>&& named_polynomials,
            const KZGKeyPair& key,
            CommitBatchProfile* profile = nullptr,
            const std::string& progress_prefix = "");

        // 对同一域 opening bundle 中的多列做折叠 opening。
        static OpeningProof open_batch(
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
            const OpeningProof& proof,
            const KZGVerifierKey& vk);

        // 对 external evaluation 列做折叠 opening，服务 `rho_ext` 外点检查。
        static OpeningProof open_external_fold(
            const std::vector<std::pair<Commitment, algebra::FieldElement>>& commitments_and_values,
            const std::vector<algebra::FieldElement>& points,
            
            const algebra::FieldElement& folding_challenge,
            const KZGKeyPair& key);

        // verifier 侧复验 external fold opening。
        static bool verify_external_fold(
            const std::vector<std::pair<Commitment, algebra::FieldElement>>& commitments_and_values,
            const std::vector<algebra::FieldElement>& points,
            const algebra::FieldElement& folding_challenge,
            const OpeningProof& proof,
            const KZGVerifierKey& vk);
    };
}
