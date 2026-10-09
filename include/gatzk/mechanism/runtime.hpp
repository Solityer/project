#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/crypto/kzg.hpp"

namespace gatzk::mechanism
{
    enum class Mechanism
    {
        FeatureFolding,
        SharedNeighborhood,
    };

    enum class Variant
    {
        Original,
        Control,
    };

    enum class Backend
    {
        Cpu,
        Cuda,
    };

    struct Edge
    {
        std::uint64_t graph = 0;
        std::uint64_t src = 0;
        std::uint64_t dst = 0;
    };

    struct PublicStatement
    {
        Mechanism mechanism = Mechanism::FeatureFolding;
        Variant variant = Variant::Original;
        std::size_t nodes = 0;
        std::size_t edges = 0;
        std::size_t width = 0;
        std::size_t graphs = 0;
        std::uint64_t seed = 0;
        std::vector<std::uint64_t> node_ptr;
        std::vector<std::uint64_t> edge_ptr;
        std::vector<Edge> topology;
        std::string srs_id;
        std::string srs_manifest_sha256;
        std::string parent_protocol_schema_id;
        std::size_t parent_srs_degree_bound = 0;
        std::string node_domain = "N";
        std::size_t node_domain_size = 0;
        std::string edge_domain = "edge";
        std::size_t edge_domain_size = 0;
        std::string mechanism_schema_id;
        std::string security_budget_ledger_sha256;
        std::string pair_input_commitment_sha256;

        std::string canonical_json() const;
        static PublicStatement parse_canonical_json(const std::string &bytes);
    };

    struct CommitmentRecord
    {
        std::string label;
        crypto::G1Point point;
    };

    struct OpeningGroup
    {
        std::string label;
        std::vector<std::string> commitment_labels;
        std::vector<algebra::FieldElement> values;
        crypto::G1Point witness;
    };

    struct MechanismProof
    {
        Mechanism mechanism = Mechanism::FeatureFolding;
        Variant variant = Variant::Original;
        std::string statement_sha256;
        std::string srs_id;
        std::string srs_manifest_sha256;
        std::string parent_protocol_schema_id;
        std::string mechanism_schema_id;
        std::string security_budget_ledger_sha256;
        std::vector<CommitmentRecord> commitments;
        std::vector<OpeningGroup> opening_groups;
    };

    struct RunOptions
    {
        Mechanism mechanism = Mechanism::FeatureFolding;
        Variant variant = Variant::Original;
        Backend backend = Backend::Cpu;
        int device = 0;
        std::size_t nodes = 0;
        std::size_t edges = 0;
        std::size_t width = 0;
        std::size_t graphs = 0;
        std::uint64_t seed = 0;
        std::filesystem::path witness_seed_file;
        std::filesystem::path srs_path;
        std::filesystem::path proof_out;
        std::filesystem::path statement_out;
        std::filesystem::path result_json;
    };

    struct VerifyOptions
    {
        std::filesystem::path proof_path;
        std::filesystem::path statement_path;
        std::filesystem::path srs_path;
        std::filesystem::path result_json;
    };

    struct RunReceipt
    {
        double commitment_time_ms = 0.0;
        double prove_time_ms = 0.0;
        std::size_t proof_size_bytes = 0;
        std::string proof_sha256;
        std::string statement_sha256;
        std::string pair_input_commitment_sha256;
        std::string srs_id;
        std::string srs_manifest_sha256;
        std::string parent_protocol_schema_id;
        bool actual_cuda = false;
        int device = -1;
        std::string device_name;
        std::uint64_t cuda_kernel_invocations_delta = 0;
        std::uint64_t cuda_h2d_bytes_delta = 0;
        std::uint64_t cuda_d2h_bytes_delta = 0;
        std::uint64_t cuda_h2d_operations_delta = 0;
        std::uint64_t cuda_d2h_operations_delta = 0;
        std::uint64_t cuda_stream_synchronizations_delta = 0;
        std::uint64_t cuda_ntt_calls_delta = 0;
        std::uint64_t cuda_intt_calls_delta = 0;
        std::uint64_t cuda_quotient_calls_delta = 0;
        std::uint64_t cuda_t_edge_calls_delta = 0;
        std::uint64_t cuda_t_fh_calls_delta = 0;
        std::uint64_t cuda_opening_fold_calls_delta = 0;
        std::uint64_t cuda_msm_single_calls_delta = 0;
        std::uint64_t cuda_msm_batch_calls_delta = 0;
        std::uint64_t cuda_resident_hits_delta = 0;
        std::uint64_t cuda_resident_misses_delta = 0;
        std::string security_budget_ledger_sha256;
        std::string security_budget_ledger_json;
    };

    struct VerifyReceipt
    {
        double verify_time_ms = 0.0;
        std::string proof_sha256;
        std::string statement_sha256;
        std::string srs_id;
        std::string srs_manifest_sha256;
        std::string parent_protocol_schema_id;
        std::string security_budget_ledger_sha256;
        std::string security_budget_ledger_json;
        std::uint64_t verifier_cuda_kernel_invocations_delta = 0;
    };

    Mechanism parse_mechanism(const std::string &value);
    Variant parse_variant(const std::string &value);
    Backend parse_backend(const std::string &value);
    std::string to_string(Mechanism value);
    std::string to_string(Variant value);
    std::string to_string(Backend value);

    RunReceipt run(const RunOptions &options);
    VerifyReceipt verify(const VerifyOptions &options);
}
