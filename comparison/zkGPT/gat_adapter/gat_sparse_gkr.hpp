#pragma once

// Canonical sparse relation boundary for the zkGPT GAT adapter.  This file
// deliberately contains no witness representation.  It describes the public
// graph schedule and the streaming interface used by the native masked
// Sumcheck implementation.

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

static constexpr const char *kSparseGkrRelationSchema =
    "zkgpt-gat-sparse-gkr-v1";
static constexpr std::uint32_t kSparseGkrRelationVersion = 1;
static constexpr std::uint64_t kSparseGkrChunkEntries = 1ULL << 12;
static constexpr const char *kSparseGkrArithmeticRule =
    "BN254:u64_checked:SignedRoundDiv-nearest-ties-away-v1";
static constexpr const char *kSparseGkrOverflowRule =
    "checked-u64-fail-closed-v1";

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

const char *sparse_gkr_relation_kind_name(SparseGkrRelationKind kind);
bool sparse_gkr_relation_kind_valid(std::uint8_t kind);

// Checked arithmetic is part of the wire contract.  The implementation
// rejects a size_t truncation rather than relying on the host word size.
bool sparse_gkr_checked_add(std::uint64_t left, std::uint64_t right,
                            std::uint64_t *out);
bool sparse_gkr_checked_mul(std::uint64_t left, std::uint64_t right,
                            std::uint64_t *out);
bool sparse_gkr_next_power_of_two(std::uint64_t value,
                                  std::uint64_t *out);

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
    std::string terminal_rule =
        "terminal-claim-chain/input-output-binding-v1";
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
    // Public FCONN/input/output handoff claims.  These are separate from the
    // adjacent relation terminal chain and are consumed by the transcript
    // binder before any relation proof is verified.
    std::vector<SparseGkrClaimLink> binding_links;
    std::string encoding() const;
};

// Build the fixed-order public relation schedule from graph topology and the
// descriptor order.  No edge/coordinate gate array is created.  Descriptor
// offsets are copied as public coordinates and checked in u64 arithmetic.
SparseGkrPlan build_sparse_gkr_plan(
    const Graph &graph, const std::string &graph_topology_digest,
    const std::string &static_tables_id,
    const std::vector<SparseGkrRelationSpec> &relations);

// Enumerate canonical edge/node/value chunks.  The callback receives a
// relation-local logical offset; graph.src/dst/edge_gid remain public inputs
// and are never copied into a dense relation array.
struct SparseGkrChunk {
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
    std::uint64_t first_edge = 0;
    std::uint64_t first_node = 0;
};

using SparseGkrChunkCallback =
    std::function<bool(const SparseGkrRelationSpec &, const SparseGkrChunk &,
                       std::string *reason)>;

struct SparseGkrRelationResult;

bool for_each_sparse_gkr_chunk(const Graph &graph,
                               const SparseGkrRelationSpec &relation,
                               const SparseGkrChunkCallback &callback,
                               std::string *reason = nullptr);

// Bind the public schedule into the native transcript before any relation
// round is emitted.  This is intentionally independent of witness values.
bool absorb_sparse_gkr_schedule(NativeProofSession &session,
                                const SparseGkrPlan &plan,
                                std::string *reason = nullptr);

struct SparseGkrProverCallbacks {
    // Return the evaluations at x=0..degree for one chunk and one Sumcheck
    // round.  `prefix` is the already sampled low-bit-first challenge prefix.
    // The callback must not retain the chunk buffer after returning.
    std::function<bool(const SparseGkrRelationSpec &, const SparseGkrChunk &,
                       std::size_t round, const std::vector<F> &prefix,
                       std::vector<F> &polynomial, std::string *reason)>
        round_polynomial;
    // Return the private terminal values for the relation.  They are consumed
    // by NativeMaskedSumcheckProver and never enter the transcript in the
    // clear.
    std::function<bool(const SparseGkrRelationSpec &,
                       std::vector<F> &terminal_values, std::string *reason)>
        terminal_values;
    std::function<bool(const SparseGkrRelationSpec &,
                       NativeMaskedTerminalSpec &terminal_spec,
                       std::string *reason)>
        terminal_spec;
    std::function<bool(const SparseGkrRelationSpec &, F &initial_claim,
                       std::string *reason)>
        initial_claim;
    // Called after a relation has emitted its masked terminal claim.  The
    // next relation's initial claim is allowed to bind to this result; the
    // callback keeps that transition explicit instead of relying on a
    // caller-side transcript scan.
    std::function<void(const SparseGkrRelationResult &)> relation_finished;
};

struct SparseGkrRelationResult {
    std::string relation_id;
    NativeMaskedSumcheckResult sumcheck;
};

// Stream every relation in fixed plan order.  Each round polynomial is
// accumulated chunk-by-chunk; no full sparse relation vector is materialised.
bool prove_sparse_gkr_plan(
    NativeProofSession &session, const Graph &graph, const SparseGkrPlan &plan,
    const SparseGkrProverCallbacks &callbacks,
    std::vector<SparseGkrRelationResult> *results = nullptr,
    std::string *reason = nullptr);

// Independent verifier counterpart.  It reconstructs only the public
// schedule and asks native_masked_sumcheck_verify to consume the typed
// transcript.  No callback can provide witness values on this path.
struct SparseGkrVerifierCallbacks {
    std::function<bool(const SparseGkrRelationSpec &,
                       NativeMaskedTerminalSpec &terminal_spec,
                       std::string *reason)>
        terminal_spec;
    std::function<bool(const SparseGkrRelationSpec &, F &initial_claim,
                       std::string *reason)>
        initial_claim;
    std::function<void(const SparseGkrRelationResult &)> relation_finished;
};

bool verify_sparse_gkr_plan(
    const NativeProofSession &session, const Graph &graph,
    const SparseGkrPlan &plan, const SparseGkrVerifierCallbacks &callbacks,
    std::vector<SparseGkrRelationResult> *results = nullptr,
    std::string *reason = nullptr);

} // namespace gat
} // namespace zkgpt
