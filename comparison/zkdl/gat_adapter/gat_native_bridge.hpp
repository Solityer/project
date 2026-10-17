#pragma once

#include "gat_adapter.hpp"
#include "gat_native_abi.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace zkdl_gat {

struct NativeProveRequest {
    const Config *config = nullptr;
    const Graph *graph = nullptr;
    const ModelBundle *model = nullptr;
    const StaticTables *tables = nullptr;
    const RelationPlan *relation = nullptr;
    // Non-null run-lifetime token retained for ABI compatibility. Native
    // proving reads private values only through the authenticated storage
    // readers in witness; it does not dereference this trace pointer.
    const GatTrace *trace = nullptr;
    const PublicStatement *statement = nullptr;
    Backend backend = Backend::Cpu;
    bool gpu_forward_kernel_executed = false;
    // Only "hyrax_ipa_fold" is currently supported.  This is an explicit
    // compute offload under the CPU relation backend, not Backend::Gpu.
    std::string gpu_offload_stage = "none";

    const NativeRelationWitnessInput *witness = nullptr;
};

struct NativeProofResult {
    ProofEnvelope envelope;
    std::vector<std::uint8_t> proof_bytes;
};

struct NativeVerifyRequest {
    const Config *config = nullptr;
    const PublicStatement *statement = nullptr;
    ProofEnvelope *envelope = nullptr;
    const std::vector<std::uint8_t> *proof_bytes = nullptr;
    Backend backend = Backend::Cpu;

    const Graph *graph = nullptr;

    const NativeRelationWitnessInput *public_context = nullptr;

    const RelationPlan *relation = nullptr;
    const StaticTables *tables = nullptr;
};

struct NativeRelationPlanMetrics {
    std::uint64_t operation_count = 0U;
    std::uint64_t primitive_count = 0U;
    std::uint64_t constraint_count = 0U;
    std::uint64_t route_count = 0U;
    std::uint64_t fac_binding_count = 0U;
    std::uint64_t padding_family_count = 0U;
    std::uint64_t family_segment_count = 0U;
    std::uint64_t primitive_opening_segment_count = 0U;
    std::uint64_t constraint_opening_segment_count = 0U;
    std::uint64_t padding_opening_segment_count = 0U;
    std::uint64_t sorted_lookup_opening_segment_count = 0U;
    std::uint64_t sparse_route_opening_count = 0U;
    std::uint64_t fac_selector_opening_count = 0U;
    std::uint64_t route_aggregate_input_segment_count = 0U;
    std::uint64_t public_output_opening_segment_count = 0U;
    std::uint64_t computed_hyrax_opening_count = 0U;
    std::uint64_t retained_hyrax_opening_object_count = 0U;
    std::uint64_t retained_hyrax_terminal_mirror_copy_count = 0U;
    std::uint64_t total_hyrax_opening_segment_count = 0U;
    struct HyraxOpeningSegmentBucket {
        std::uint64_t logical_domain = 0U;
        std::uint64_t segment_size = 0U;
        std::uint64_t padded_generator_size = 0U;
        std::uint64_t computed_opening_count = 0U;
        std::uint64_t retained_object_count = 0U;
        std::uint64_t terminal_mirror_copy_count = 0U;
        std::uint64_t retained_round_vector_bytes_upper_bound = 0U;
        std::uint64_t retained_point_vector_bytes_upper_bound = 0U;
    };
    std::vector<HyraxOpeningSegmentBucket> hyrax_opening_segment_domain_histogram;
    std::uint64_t hyrax_round_vector_capacity_bytes_upper_bound = 0U;
    std::uint64_t hyrax_point_vector_capacity_bytes_upper_bound = 0U;
    std::uint64_t hyrax_opening_object_header_bytes_upper_bound = 0U;
    std::uint64_t hyrax_maximum_single_opening_vector_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_folded_state_budget_bytes = 0U;
    std::uint64_t sumcheck_cached_call_count = 0U;
    std::uint64_t sumcheck_fallback_call_count = 0U;
    std::uint64_t sumcheck_forced_cached_call_count = 0U;
    std::uint64_t sumcheck_cached_provider_value_slots_upper_bound = 0U;
    std::uint64_t sumcheck_fallback_provider_value_slots_upper_bound = 0U;
    std::uint64_t sumcheck_spill_call_count = 0U;
    std::uint64_t sumcheck_spill_provider_value_slots_upper_bound = 0U;
    std::uint64_t sumcheck_spill_provider_scan_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_spill_provider_source_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_spill_storage_io_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_max_spill_disk_peak_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_max_spill_ram_peak_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_cached_provider_scan_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_fallback_provider_scan_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_max_folded_state_peak_bytes_upper_bound = 0U;
    std::uint64_t sumcheck_max_streaming_peak_bytes_upper_bound = 0U;
    struct SumcheckShapeBucket {
        std::uint64_t logical_domain = 0U;
        std::uint64_t segment_size = 0U;
        std::uint64_t provider_count = 0U;
        std::uint64_t cached_call_count = 0U;
        std::uint64_t fallback_call_count = 0U;
        std::uint64_t spill_call_count = 0U;
        std::uint64_t forced_cached_call_count = 0U;
        std::uint64_t cached_provider_value_slots_upper_bound = 0U;
        std::uint64_t fallback_provider_value_slots_upper_bound = 0U;
        std::uint64_t spill_provider_value_slots_upper_bound = 0U;
    };
    std::vector<SumcheckShapeBucket> sumcheck_shape_histogram;
    std::uint64_t operation_descriptor_bytes = 0U;
    std::uint64_t operation_descriptor_max_bytes = 0U;
    std::uint64_t sumcheck_body_max_bytes = 0U;
    std::uint64_t terminal_relation_body_max_bytes = 0U;
    std::uint64_t constraint_body_max_bytes = 0U;
    std::uint64_t padding_record_body_max_bytes = 0U;
    std::uint64_t hidden_matmul_body_max_bytes = 0U;
    std::uint64_t sparse_route_body_max_bytes = 0U;
    std::uint64_t fac_line8_body_max_bytes = 0U;
    std::uint64_t sorted_lookup_relation_body_max_bytes = 0U;
    std::uint64_t sorted_lookup_proof_body_max_bytes = 0U;
    std::uint64_t sorted_lookup_outer_commitment_count = 0U;
    std::uint64_t relation_proof_sections_wire_bytes_upper_bound = 0U;
    std::uint64_t proof_wire_bytes_upper_bound = 0U;
    std::uint64_t commitment_section_wire_bytes_upper_bound = 0U;
    std::uint64_t operation_descriptor_resident_bytes_upper_bound = 0U;
    std::uint64_t trace_i64_payload_bytes = 0U;
    std::uint64_t trace_field_payload_bytes = 0U;
    std::uint64_t trace_u8_payload_bytes = 0U;
    std::uint64_t trace_object_metadata_bytes = 0U;
    std::uint64_t trace_forward_live_peak_bytes_upper_bound = 0U;
    std::uint64_t maximum_route_source_domain = 0U;
    std::uint64_t maximum_route_edge_domain = 0U;
    std::uint64_t maximum_route_selector_factor_elements = 0U;
    std::uint64_t route_sumcheck_live_peak_bytes_upper_bound = 0U;
    std::uint64_t segmented_sumcheck_live_peak_bytes_upper_bound = 0U;
};

struct NativeDescriptorShapeFootprint {
    std::uint64_t operation_count = 0U;
    std::uint64_t fac_binding_count = 0U;
    std::uint64_t route_count = 0U;
    std::uint64_t expected_unique_outer_map_bytes = 0U;
    std::uint64_t expected_unique_active_mask_bytes = 0U;
    std::uint64_t expected_feature_map_bytes = 0U;
    std::uint64_t received_selector_array_bytes = 0U;
    std::uint64_t ordinary_descriptor_capacity_bytes = 0U;
    std::uint64_t descriptor_object_header_bytes = 0U;
    std::uint64_t interner_metadata_bytes = 0U;
    std::uint64_t selector_allocator_metadata_bytes = 0U;
    std::uint64_t descriptor_compare_buffer_bytes = 0U;
    std::uint64_t operation_descriptor_wire_bytes = 0U;
    std::uint64_t operation_descriptor_max_bytes = 0U;
    std::uint64_t operation_descriptor_resident_bytes_upper_bound = 0U;
};

bool zkdl_gat_native_descriptor_shape_footprint(const Config &config, const Graph &graph,
                                                 const StaticTables &tables, const RelationPlan &plan,
                                                 NativeDescriptorShapeFootprint *footprint,
                                                 std::string *error);

bool zkdl_gat_native_relation_plan_metrics(const Config &config, const Graph &graph,
                                           const StaticTables &tables, const RelationPlan &plan,
                                           NativeRelationPlanMetrics *metrics, std::string *error,
                                           bool analysis_only_shape = false);

#if defined(__GNUC__) || defined(__clang__)
#define ZKDL_GAT_WEAK __attribute__((weak))
#else
#define ZKDL_GAT_WEAK
#endif

bool zkdl_gat_native_prove(const NativeProveRequest &request, NativeProofResult *result, std::string *error) ZKDL_GAT_WEAK;
bool zkdl_gat_native_verify(const NativeVerifyRequest &request, std::string *error) ZKDL_GAT_WEAK;

inline bool native_bridge_available() noexcept { return zkdl_gat_native_prove != nullptr && zkdl_gat_native_verify != nullptr; }

#undef ZKDL_GAT_WEAK

}
