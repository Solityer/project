#pragma once
// KZG承诺实现，负责承诺、批量opening与验证侧的批量检查

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
        std::string manifest_format;
        std::string srs_id;
        std::string curve_id;
        std::string field_modulus_id;
        std::string srs_type;
        std::string ceremony_transcript_hash;
        std::size_t degree_bound = 0;
        std::unordered_map<std::string, std::size_t> domain_sizes;
        std::unordered_map<std::string, std::string> domain_generators;
        std::string root_derivation_id;
        std::string row_order;
        std::string protocol_compiler_artifact_path;
        std::string protocol_compiler_artifact_hash;
        std::string protocol_schema_id;
        std::string max_instance_envelope_id;
        std::string max_instance_envelope_hash;
        std::unordered_map<std::string, std::size_t> max_instance_envelope;
        std::string proof_schema_template_hash;
        std::string observation_ledger_hash;
        std::string random_identity_ledger_hash;
        std::string security_budget_ledger_hash;
        std::string pk_coef_path;
        std::string pk_coef_hash;
        std::unordered_map<std::string, std::string> pk_lagrange_paths;
        std::unordered_map<std::string, std::string> pk_lagrange_hashes;
        std::string vk_path;
        std::string vk_hash;
        std::string encoding;
        std::string generator_version;
        std::string security_level;
        std::string kzg_assumption_profile_id;
        std::string kzg_assumption_domain_catalog_hash;
        std::string domain_generator_catalog_hash;
        std::size_t kzg_assumption_degree_bound = 0;
        std::size_t kzg_assumption_security_bits = 0;
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
        std::unordered_map<std::string, G2Point> domain_zero_g2;
        std::shared_ptr<PreparedG2> g2_one_prepared;
        std::shared_ptr<PreparedG2> g2_tau_prepared;
        std::string manifest_hash;
    };

    // SRS/验证键的最小实现集合，只从manifest/key文件中加载
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

    // 承诺对象只携带proof可见的群元素；prover侧可保留多项式材料用于生成opening见证
    struct Commitment
    {
        std::string name;
        G1Point point;
        std::shared_ptr<const algebra::Polynomial> opening_polynomial;
    };

    // 批量opening的执行，供正式 benchmark 细分 `domain_opening_ms`。
    struct BatchOpeningProfile
    {
        double precompute_ms = 0.0;
        double fold_commitment_ms = 0.0;
        double finalize_ms = 0.0;
        double denominator_inverse_ms = 0.0;
        double denominator_inverse_reuse_ms = 0.0;
        double quotient_assembly_ms = 0.0;
        double witness_msm_ms = 0.0;
        double blinder_witness_ms = 0.0;
        std::size_t denominator_inverse_reuse_hits = 0;
        std::size_t denominator_inverse_cache_misses = 0;
        std::size_t point_parallel_batches = 0;
        std::size_t point_parallel_workers = 0;
        std::size_t point_msm_worker_cap = 0;
    };

    enum class BatchOpeningClaimValidation
    {
        VerifyAgainstOpeningPolynomials,
        AlreadyComputedFromOpeningPolynomials,
    };

    struct PointGroupOpeningExecutionOptions
    {
        // These caps only govern execution resources. They never alter the
        // point-local folding challenge, witness polynomial, or KZG equation.
        std::size_t max_field_worker_count = 0;
        G1MSMExecutionOptions msm;
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
        bool semantic_commitment_reuse = false;
        double msm_ms = 0.0;
    };

    class EvaluationCommitmentCache
    {
    public:
        EvaluationCommitmentCache();
        ~EvaluationCommitmentCache();
        EvaluationCommitmentCache(const EvaluationCommitmentCache&) = delete;
        EvaluationCommitmentCache& operator=(const EvaluationCommitmentCache&) = delete;

        bool lookup_semantic(
            const algebra::RootOfUnityDomain* domain,
            const std::vector<G1Point>* lagrange_srs,
            const std::shared_ptr<const std::vector<algebra::FieldElement>>& values,
            G1Point* point);
        void store_semantic(
            const algebra::RootOfUnityDomain* domain,
            const std::vector<G1Point>* lagrange_srs,
            std::shared_ptr<const std::vector<algebra::FieldElement>> values,
            const G1Point& point);
    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
        friend class KZG;
    };

    struct CommitBatchProfile
    {
        double evaluation_ms = 0.0;
        double msm_ms = 0.0;
        std::vector<CommitObjectProfile> objects;
    };

    // 序列化大小直接参与proof_size_bytes统计
    std::size_t serialized_size(const Commitment& commitment);

    class KZG
    {
    public:
        // 加载一次安全正式运行使用的 SRS 参数。
        static KZGKeyPair load_keys(
            const std::string& srs_path,
            const std::string& expected_srs_type,
            const std::string& expected_srs_id);
        static KZGKeyPair load_keys_for_domains(
            const std::string& srs_path,
            const std::string& expected_srs_type,
            const std::string& expected_srs_id,
            const std::vector<std::pair<std::string, std::size_t>>& required_lagrange_domains,
            std::size_t required_coefficient_count = 0);

        // 对常数多项式做正式 SRS 承诺；用于公开常数或已折叠的标量关系。
        static Commitment commit_constant(
            const std::string& name,
            const algebra::FieldElement& value,
            const KZGKeyPair& key);
        // 对单个多项式做正式承诺。
        static Commitment commit(
            const std::string& name,
            const algebra::Polynomial& polynomial,
            const KZGKeyPair& key,
            CommitObjectProfile* profile = nullptr);
        // 对一批常数多项式做正式 SRS 承诺。
        static std::vector<Commitment> commit_constant_batch(
            const std::vector<std::pair<std::string, algebra::FieldElement>>& named_values,
            const KZGKeyPair& key,
            CommitBatchProfile* profile = nullptr);
        // 对一批多项式统一承诺，是 dynamic commitment 主链的核心入口。
        static std::vector<Commitment> commit_batch(
            const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials,
            const KZGKeyPair& key,
            CommitBatchProfile* profile = nullptr,
            EvaluationCommitmentCache* reuse_cache = nullptr);
        // Execution-only helpers for exact linear relations between evaluation
        // polynomials.  The component excludes a polynomial's independent
        // zero-domain blinder; rebuilding a commitment always attaches the
        // target polynomial's own blinder.
        static G1Point evaluation_commitment_component(
            const Commitment& commitment,
            const KZGKeyPair& key);
        static Commitment commit_evaluation_from_component(
            const std::string& name,
            const algebra::Polynomial& polynomial,
            const G1Point& evaluation_component,
            const KZGKeyPair& key);
        // 对刚物化出的多项式批量承诺并接管 opening 材料，避免为 quotient 等一次性对象再复制整列标量。
        static std::vector<Commitment> commit_owned_batch(
            std::vector<std::pair<std::string, algebra::Polynomial>>&& named_polynomials,
            const KZGKeyPair& key,
            CommitBatchProfile* profile = nullptr,
            const std::string& progress_prefix = "");

        // Prover-side construction of P_proof = P_anchor + (X^n - 1)S.
        // `blinder_coefficient_count` is fixed by the observation ledger and
        // must not depend on the polynomial's private effective degree.
        static std::pair<Commitment, Commitment> reblind_domain_polynomial(
            const Commitment& long_term_anchor,
            const std::string& proof_local_name,
            const std::string& blinder_name,
            const std::string& domain_name,
            std::size_t domain_size,
            std::size_t blinder_coefficient_count,
            const KZGKeyPair& key);

        // Build the unique compiler-sized commitment to F(X + delta) using the
        // protocol's coefficient_taylor_shift implementation.  The source
        // opening material includes its registered domain/counter blinder.
        static Commitment commit_taylor_shifted(
            const Commitment& source,
            const std::string& shifted_name,
            std::size_t configured_degree,
            const algebra::FieldElement& delta,
            const KZGKeyPair& key,
            CommitObjectProfile* profile = nullptr);

        // Open exactly one canonical point group.  Every distinct evaluation
        // point has an independently derived, non-zero folding challenge.
        static G1Point open_point_group(
            const std::vector<Commitment>& commitments,
            const algebra::FieldElement& point,
            const std::vector<algebra::FieldElement>& claimed_values,
            const algebra::FieldElement& folding_challenge,
            const KZGKeyPair& key,
            BatchOpeningProfile* profile = nullptr,
            BatchOpeningClaimValidation claim_validation =
                BatchOpeningClaimValidation::VerifyAgainstOpeningPolynomials,
            const PointGroupOpeningExecutionOptions* execution_options =
                nullptr);

        // Verify exactly one canonical point group and its single witness.
        static bool verify_point_group(
            const std::vector<Commitment>& commitments,
            const algebra::FieldElement& point,
            const std::vector<algebra::FieldElement>& claimed_values,
            const algebra::FieldElement& folding_challenge,
            const G1Point& witness,
            const KZGVerifierKey& vk);

        // Verify that a proof-local model polynomial differs from its long-term
        // anchor by Z_domain(X) * blinder(X), using the shifted-KZG equation
        // fixed by the verifier-key domain catalogue.  The blinder is committed
        // but is never opened.
        static bool verify_domain_divisibility(
            const Commitment& proof_local,
            const Commitment& long_term_anchor,
            const Commitment& blinder,
            const std::string& domain_name,
            const KZGVerifierKey& vk);
    };
}
