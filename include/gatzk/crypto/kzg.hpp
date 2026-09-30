#pragma once
#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/util/trace_spill.hpp"
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
        std::string setup_transcript_hash;
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
        std::string vanishing_g2_hash;
        std::string srs_audit_report_path;
        std::string srs_audit_report_hash;
        std::string srs_consistency_hash;
        std::string encoding;
        std::string generator_id;
        std::string security_level;
        std::string kzg_assumption_profile_id;
        std::string kzg_assumption_domain_catalog_hash;
        std::string domain_generator_catalog_hash;
        std::size_t kzg_assumption_degree_bound = 0;
        std::size_t kzg_assumption_security_bits = 0;
        std::string manifest_hash;
    };

    std::string canonical_kzg_key_resource_identity(const std::filesystem::path &srs_root, const SRSManifest &manifest,
                                                    const std::vector<std::string> &selected_lagrange_domains);

    struct KZGProverKey
    {
        std::string srs_id;
        std::string srs_type;
        std::size_t degree_bound = 0;
        G1Point g1_generator;
        std::shared_ptr<const std::vector<G1Point>> powers_g1;
        std::unordered_map<std::string, std::shared_ptr<const std::vector<G1Point>>> lagrange_srs_by_domain;
        std::string manifest_hash;
        std::string resource_identity;
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
        std::string resource_identity;
    };

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
        std::string resource_identity;
    };

    struct Commitment
    {
        std::string name;
        G1Point point;
        std::shared_ptr<const algebra::Polynomial> opening_polynomial;
    };

    using OpeningPolynomialResolver = std::function<std::shared_ptr<const algebra::Polynomial>(const Commitment &)>;

    struct OpeningSemanticSourceInfo
    {
        std::string key;

        std::shared_ptr<algebra::RootOfUnityDomain> domain;

        std::size_t value_count = 0;

        std::size_t stored_value_count = 0;
        bool zero_tail_compacted = false;

        std::vector<algebra::FieldElement> zero_domain_blinder;

        std::string file_path;
        bool file_backed = false;
        std::shared_ptr<const util::TraceSpillLease> lease;
    };

    using OpeningSemanticSourceResolver = std::function<std::optional<OpeningSemanticSourceInfo>(const Commitment &)>;

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
        std::size_t streamed_semantic_source_scan_count = 0;
        std::size_t streamed_semantic_source_scan_bytes = 0;
    };

    struct PointGroupOpeningExecutionOptions
    {
        std::size_t max_field_worker_count = 0;
        G1MSMExecutionOptions msm;
    };

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
        EvaluationCommitmentCache(const EvaluationCommitmentCache &) = delete;
        EvaluationCommitmentCache &operator=(const EvaluationCommitmentCache &) = delete;

        bool lookup_semantic(const algebra::RootOfUnityDomain *domain, const std::vector<G1Point> *lagrange_srs,
                             const std::shared_ptr<const std::vector<algebra::FieldElement>> &values, G1Point *point);
        void store_semantic(const algebra::RootOfUnityDomain *domain, const std::vector<G1Point> *lagrange_srs,
                            std::shared_ptr<const std::vector<algebra::FieldElement>> values, const G1Point &point);

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
        bool used_same_srs_mixed_prefix = false;
        bool used_same_srs_exact_count = false;
    };

    std::size_t serialized_size(const Commitment &commitment);

    class KZG
    {
    public:
        static KZGKeyPair load_keys(const std::string &srs_path, const std::string &expected_srs_type, const std::string &expected_srs_id);
        static KZGKeyPair load_keys_for_domains(const std::string &srs_path, const std::string &expected_srs_type, const std::string &expected_srs_id,
                                                const std::vector<std::pair<std::string, std::size_t>> &required_lagrange_domains,
                                                std::size_t required_coefficient_count = 0);

        static Commitment commit_constant(const std::string &name, const algebra::FieldElement &value, const KZGKeyPair &key);
        static Commitment commit(const std::string &name, const algebra::Polynomial &polynomial, const KZGKeyPair &key, CommitObjectProfile *profile = nullptr);
        static std::vector<Commitment> commit_constant_batch(const std::vector<std::pair<std::string, algebra::FieldElement>> &named_values,
                                                             const KZGKeyPair &key, CommitBatchProfile *profile = nullptr);
        static std::vector<Commitment> commit_batch(const std::vector<std::pair<std::string, const algebra::Polynomial *>> &named_polynomials,
                                                    const KZGKeyPair &key, CommitBatchProfile *profile = nullptr,
                                                    EvaluationCommitmentCache *reuse_cache = nullptr);
        static G1Point evaluation_commitment_component(const Commitment &commitment, const KZGKeyPair &key);
        static Commitment commit_evaluation_from_component(const std::string &name, const algebra::Polynomial &polynomial, const G1Point &evaluation_component,
                                                           const KZGKeyPair &key);
        static std::vector<Commitment> commit_owned_batch(std::vector<std::pair<std::string, algebra::Polynomial>> &&named_polynomials, const KZGKeyPair &key,
                                                          CommitBatchProfile *profile = nullptr, const std::string &progress_prefix = "");

        static std::pair<Commitment, Commitment> reblind_domain_polynomial(const Commitment &long_term_anchor, const std::string &proof_local_name,
                                                                           const std::string &blinder_name, const std::string &domain_name,
                                                                           std::size_t domain_size, std::size_t blinder_coefficient_count,
                                                                           const KZGKeyPair &key);

        static algebra::Polynomial materialize_taylor_shifted_polynomial(const Commitment &source, const std::string &shifted_name,
                                                                         std::size_t configured_degree, const algebra::FieldElement &delta,
                                                                         const KZGKeyPair &key);
        static Commitment commit_taylor_shifted(const Commitment &source, const std::string &shifted_name, std::size_t configured_degree,
                                                const algebra::FieldElement &delta, const KZGKeyPair &key, CommitObjectProfile *profile = nullptr);

        static G1Point open_point_group(const std::vector<Commitment> &commitments, const algebra::FieldElement &point,
                                        const std::vector<algebra::FieldElement> &claimed_values, const algebra::FieldElement &folding_challenge,
                                        const KZGKeyPair &key, BatchOpeningProfile *profile = nullptr,
                                        const PointGroupOpeningExecutionOptions *execution_options = nullptr,
                                        const OpeningPolynomialResolver *opening_resolver = nullptr,
                                        const OpeningSemanticSourceResolver *semantic_source_resolver = nullptr);

        static std::array<G1Point, 2> open_point_group_pair_shared_stream(const std::array<std::vector<Commitment>, 2> &commitments,
                                                                          const std::array<algebra::FieldElement, 2> &points,
                                                                          const std::array<std::vector<algebra::FieldElement>, 2> &claimed_values,
                                                                          const std::array<algebra::FieldElement, 2> &folding_challenges, const KZGKeyPair &key,
                                                                          std::array<BatchOpeningProfile, 2> *profiles = nullptr,
                                                                          const std::array<PointGroupOpeningExecutionOptions, 2> *execution_options = nullptr,
                                                                          const OpeningPolynomialResolver *opening_resolver = nullptr,
                                                                          const OpeningSemanticSourceResolver *semantic_source_resolver = nullptr);

        static bool verify_point_group(const std::vector<Commitment> &commitments, const algebra::FieldElement &point,
                                       const std::vector<algebra::FieldElement> &claimed_values, const algebra::FieldElement &folding_challenge,
                                       const G1Point &witness, const KZGVerifierKey &vk);

        static bool verify_domain_divisibility(const Commitment &proof_local, const Commitment &long_term_anchor, const Commitment &blinder,
                                               const std::string &domain_name, const KZGVerifierKey &vk);
    };
}
