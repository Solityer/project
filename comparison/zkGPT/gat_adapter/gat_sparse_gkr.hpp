#pragma once

#include "gat_types.hpp"
#include "native_masked_sumcheck.hpp"
#include "proof_session.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace zkgpt {
namespace gat {

static constexpr const char* kSparseGkrRelationSchema = "zkgpt-gat-sparse-gkr-v1";
static constexpr std::uint32_t kSparseGkrRelationVersion = 1;
static constexpr std::uint64_t kSparseGkrChunkEntries = 1ULL << 12;
static constexpr const char* kSparseGkrArithmeticRule = "BN254:u64_checked:SignedRoundDiv-nearest-ties-away-v1";
static constexpr const char* kSparseGkrOverflowRule = "checked-u64-fail-closed-v1";

enum class SparseGkrRelationKind : std::uint8_t {
    AttentionScore = 1,
    MaskedSoftmax = 2,
    Aggregation = 3,
    SignedRoundDiv = 4,
    Residual = 5,
    Fconn = 6,
    InputBinding = 7,
    OutputBinding = 8,
};

const char* sparse_gkr_relation_kind_name(SparseGkrRelationKind kind);
bool sparse_gkr_relation_kind_valid(std::uint8_t kind);

bool sparse_gkr_checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t* out);
bool sparse_gkr_checked_mul(std::uint64_t left, std::uint64_t right, std::uint64_t* out);
bool sparse_gkr_next_power_of_two(std::uint64_t value, std::uint64_t* out);

struct SparseGkrRelationSpec {
    std::string relation_id;
    std::string family_id;
    SparseGkrRelationKind kind = SparseGkrRelationKind::Residual;
    std::uint64_t item_count = 0;
    std::uint64_t value_width = 0;
    std::uint64_t input_offset = 0;
    std::uint64_t source_offset = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t output_offset = 0;
    std::uint64_t chunk_size = kSparseGkrChunkEntries;
    std::string coordinate_rule;
    std::string group_rule;
    std::string arithmetic_rule = kSparseGkrArithmeticRule;
    std::string overflow_rule = kSparseGkrOverflowRule;
    std::string terminal_rule = "terminal-claim-chain/input-output-binding-v1";
    std::vector<std::string> equations;
};

struct SparseGkrClaimLink {
    std::string from_relation;
    std::string to_relation;
    std::string claim_name;
    std::uint32_t from_terminal = 0;
    std::uint32_t to_initial = 0;
};

struct SparseGkrPlan {
    std::string schema = kSparseGkrRelationSchema;
    std::uint32_t version = kSparseGkrRelationVersion;
    std::uint64_t graph_nodes = 0;
    std::uint64_t graph_edges = 0;
    std::uint64_t graph_count = 0;
    std::uint64_t chunk_size = kSparseGkrChunkEntries;
    std::string graph_topology_digest;
    std::string static_tables_id;
    std::vector<SparseGkrRelationSpec> relations;
    std::vector<SparseGkrClaimLink> claim_links;

    std::vector<SparseGkrClaimLink> binding_links;
    std::string encoding() const;
};

SparseGkrPlan build_sparse_gkr_plan(const Graph& graph, const std::string& graph_topology_digest, const std::string& static_tables_id,
                                    const std::vector<SparseGkrRelationSpec>& relations);

struct SparseGkrChunk {
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::uint64_t first_edge = 0;
    std::uint64_t first_node = 0;
};

using SparseGkrChunkCallback = std::function<bool(const SparseGkrRelationSpec&, const SparseGkrChunk&, std::string* reason)>;

struct SparseGkrRelationResult;

bool for_each_sparse_gkr_chunk(const Graph& graph, const SparseGkrRelationSpec& relation, const SparseGkrChunkCallback& callback,
                               std::string* reason = nullptr);

bool absorb_sparse_gkr_schedule(NativeProofSession& session, const SparseGkrPlan& plan, std::string* reason = nullptr);

struct SparseGkrProverCallbacks {
    std::function<bool(const SparseGkrRelationSpec&, const SparseGkrChunk&, std::size_t round, const std::vector<F>& prefix, std::vector<F>& polynomial,
                       std::string* reason)>
        round_polynomial;

    std::function<bool(const SparseGkrRelationSpec&, std::vector<F>& terminal_values, std::string* reason)> terminal_values;
    std::function<bool(const SparseGkrRelationSpec&, NativeMaskedTerminalSpec& terminal_spec, std::string* reason)> terminal_spec;
    std::function<bool(const SparseGkrRelationSpec&, F& initial_claim, std::string* reason)> initial_claim;

    std::function<void(const SparseGkrRelationResult&)> relation_finished;
};

struct SparseGkrRelationResult {
    std::string relation_id;
    NativeMaskedSumcheckResult sumcheck;
};

bool prove_sparse_gkr_plan(NativeProofSession& session, const Graph& graph, const SparseGkrPlan& plan, const SparseGkrProverCallbacks& callbacks,
                           std::vector<SparseGkrRelationResult>* results = nullptr, std::string* reason = nullptr);

struct SparseGkrVerifierCallbacks {
    std::function<bool(const SparseGkrRelationSpec&, NativeMaskedTerminalSpec& terminal_spec, std::string* reason)> terminal_spec;
    std::function<bool(const SparseGkrRelationSpec&, F& initial_claim, std::string* reason)> initial_claim;
    std::function<void(const SparseGkrRelationResult&)> relation_finished;
};

bool verify_sparse_gkr_plan(const NativeProofSession& session, const Graph& graph, const SparseGkrPlan& plan, const SparseGkrVerifierCallbacks& callbacks,
                            std::vector<SparseGkrRelationResult>* results = nullptr, std::string* reason = nullptr);

}
}
