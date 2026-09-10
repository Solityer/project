// Baseline-local bridge to the official zkDL CUDA core.
//
// This file intentionally has a conservative fail-closed boundary.  The
// restored upstream tree exposes native producers for the old MatMul/Product
// demos and native Hyrax folding, but it does not expose a verifier-facing
// API for the indexed table, sparse-route, max/range, SignedRoundDiv, or
// multi-view FAC4DNN relations required by GAT-ZKML-zkDL.  Until those core
// entry points (and an authenticated SRS reader) are available, this bridge
// must not turn a partial transcript into a ProofEnvelope or VERIFY_OK.
//
// Keeping the symbol in the production link is important: `gat_runner` now
// reaches an explicit native-core capability decision, rather than silently
// taking the weak-hook/unlinked path.  No alternate field, group, transcript,
// or proof implementation is introduced here.

#include "gat_native_bridge.hpp"

#include "commitment.cuh"
#include "proof.cuh"
#include "zkmatmul.cuh"
#include "zkprod.cuh"

#include <cuda_runtime.h>

#include <array>
#include <filesystem>
#include <set>
#include <sstream>
#include <string>

namespace zkdl_gat {
namespace {

constexpr const char* kBridgeApi = "zkdl_native_gat_bridge_v1";

const std::set<std::string>& mandatory_operation_ids() {
    static const std::set<std::string> ids = {
        "projection", "attention_inner_product", "sparse_route", "score_add",
        "leaky_relu", "max_range", "exp_map", "group_sum", "inverse_alpha",
        "alpha_h_prime", "signed_round_div", "elu", "concat", "output_projection",
        "output_aggregation", "output_bias", "head_sum", "final_average",
        "terminal_opening"
    };
    return ids;
}

bool bridge_relation_plan_complete(const RelationPlan& relation, std::string* error) {
    std::set<std::string> present;
    for (const auto& operation : relation.operation_families) present.insert(operation.id);
    for (const auto& id : mandatory_operation_ids()) {
        if (!present.count(id)) {
            if (error) *error = "relation plan is missing mandatory operation family: " + id;
            return false;
        }
    }
    return true;
}

bool official_producer_surface_present() {
    // Take addresses so the bridge is linked against the actual official
    // native routines.  The routines are deliberately not invoked before
    // the capability/SRS checks: invoking one of the old demo protocols would
    // be an incomplete GAT proof.
    using InnerProductFn = std::vector<Fr_t> (*)(const FrTensor&, const FrTensor&, std::vector<Fr_t>);
    using HadamardFn = std::vector<Fr_t> (*)(const FrTensor&, const FrTensor&, std::vector<Fr_t>, std::vector<Fr_t>);
    InnerProductFn inner = &inner_product_sumcheck;
    HadamardFn hadamard = &hadamard_product_sumcheck;
    return inner != nullptr && hadamard != nullptr;
}

std::string missing_core_capabilities(const Config& config) {
    std::ostringstream message;
    message << kBridgeApi
            << ": full GAT proof is unavailable: official core has no "
            << "verifier-facing indexed-table/range, sparse-route, "
            << "FAC4DNN multi-view, or SignedRoundDiv relation verifier; "
            << "no authenticated SRS reader is present for "
            << config.srs_id << " (expected " << config.srs_path << ")";
    return message.str();
}

bool reject_without_mutation(NativeProofResult* result, std::string* error,
                             const std::string& message) {
    if (result) {
        *result = NativeProofResult{};
        result->envelope.native_core_api = kBridgeApi;
        result->envelope.status = "BLOCKED";
        result->envelope.silent_fallback = false;
        result->envelope.verifier_reads_witness = false;
        result->envelope.output_bound = false;
        result->envelope.error = message;
    }
    if (error) *error = message;
    return false;
}

}  // namespace

bool zkdl_gat_native_prove(const NativeProveRequest& request,
                           NativeProofResult* result,
                           std::string* error) {
    if (!result) {
        if (error) *error = "native prove result pointer is null";
        return false;
    }
    if (!request.config || !request.graph || !request.model || !request.tables ||
        !request.relation || !request.trace || !request.statement) {
        return reject_without_mutation(result, error,
                                       "native prove request is missing a canonical witness component");
    }
    if (!bridge_relation_plan_complete(*request.relation, error)) {
        return reject_without_mutation(result, error, error ? *error : "incomplete GAT relation plan");
    }
    if (!official_producer_surface_present()) {
        return reject_without_mutation(result, error,
                                       "official native Sumcheck producer surface is not linked");
    }
    if (!std::filesystem::is_regular_file(request.config->srs_path)) {
        return reject_without_mutation(result, error,
                                       "authenticated SRS artifact is missing: " +
                                       request.config->srs_path);
    }

    // The current official core cannot verify the complete relation.  Keep
    // this final guard after all structural checks so a future core exposing
    // the required APIs has a single, auditable integration point.
    return reject_without_mutation(result, error, missing_core_capabilities(*request.config));
}

bool zkdl_gat_native_verify(const NativeVerifyRequest& request,
                            std::string* error) {
    if (!request.config || !request.statement || !request.envelope ||
        !request.proof_bytes) {
        if (error) *error = "native verify request is missing public statement/envelope/proof";
        return false;
    }
    if (request.envelope->status != "VERIFIED" || request.envelope->silent_fallback ||
        request.envelope->verifier_reads_witness || !request.envelope->output_bound) {
        if (error) *error = "proof envelope is not an accepted witness-independent native proof";
        return false;
    }
    if (!std::filesystem::is_regular_file(request.config->srs_path)) {
        if (error) *error = "authenticated SRS artifact is missing: " + request.config->srs_path;
        return false;
    }

    // Parse only the official envelope format if a future prover supplies it,
    // but do not accept it here: the restored core still lacks the complete
    // GAT verifier and authenticated SRS path.
    zkdl::ProofEnvelope parsed;
    std::string parse_error;
    if (!zkdl::ProofEnvelope::deserialize(*request.proof_bytes, parsed, &parse_error)) {
        if (error) *error = "native proof envelope parse failed: " + parse_error;
        return false;
    }
    (void)parsed;
    if (error) *error = missing_core_capabilities(*request.config);
    return false;
}

}  // namespace zkdl_gat
