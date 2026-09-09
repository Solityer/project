# Test-only runtime regression for the layered formal schema. It exercises
# the production label, selector, and checkpoint-shape APIs without touching
# canonical checkpoints, SRS files, or proof runs.
add_executable(gatzk_layered_schema_test layered_schema_runtime.cpp)
target_link_libraries(gatzk_layered_schema_test PRIVATE gatzkml)
add_test(NAME layered_schema_runtime COMMAND gatzk_layered_schema_test)

add_executable(gatzk_verifier_privacy_test verifier_privacy_runtime.cpp)
target_link_libraries(gatzk_verifier_privacy_test PRIVATE gatzkml)
add_test(NAME verifier_privacy_runtime COMMAND gatzk_verifier_privacy_test)

add_executable(gatzk_static_table_streaming_test static_table_streaming_runtime.cpp)
target_link_libraries(gatzk_static_table_streaming_test PRIVATE gatzkml)
add_test(NAME static_table_streaming_runtime COMMAND gatzk_static_table_streaming_test)

# Focused public AggBind reconstruction regression. The internal header is
# deliberately outside the installed include tree and is used only to share
# the production verifier helper with this small synthetic test.
add_executable(gatzk_agg_bind_multiplicity_test agg_bind_multiplicity_runtime.cpp)
target_include_directories(gatzk_agg_bind_multiplicity_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(gatzk_agg_bind_multiplicity_test PRIVATE gatzkml)
add_test(NAME agg_bind_multiplicity_runtime COMMAND gatzk_agg_bind_multiplicity_test)

# Short-lived cache identity/TOCTOU regression. It hashes only small
# temporary resource fixtures and never loads a real SRS or starts a proof.
add_executable(gatzk_kzg_resource_identity_test kzg_resource_identity_runtime.cpp)
target_link_libraries(gatzk_kzg_resource_identity_test PRIVATE gatzkml)
add_test(NAME kzg_resource_identity_runtime COMMAND gatzk_kzg_resource_identity_test)

add_executable(gatzk_compiler_artifact_test compiler_artifact_runtime.cpp)
target_link_libraries(gatzk_compiler_artifact_test PRIVATE gatzkml)
add_test(NAME compiler_artifact_runtime COMMAND gatzk_compiler_artifact_test)

add_executable(gatzk_input_quantization_test input_quantization_runtime.cpp)
target_link_libraries(gatzk_input_quantization_test PRIVATE gatzkml)
add_test(NAME input_quantization_runtime COMMAND gatzk_input_quantization_test)

add_executable(gatzk_spill_compaction_test spill_compaction_runtime.cpp)
target_link_libraries(gatzk_spill_compaction_test PRIVATE gatzkml)
add_test(NAME spill_compaction_runtime COMMAND gatzk_spill_compaction_test)

add_executable(gatzk_trace_spill_lease_test trace_spill_lease_runtime.cpp)
target_link_libraries(gatzk_trace_spill_lease_test PRIVATE gatzkml)
add_test(NAME trace_spill_lease_runtime COMMAND gatzk_trace_spill_lease_test)

# Exact shared-view identity regression for the production spill materializer.
# It uses only a small synthetic spill and cannot alter formal artifacts.
add_executable(gatzk_ppi_spill_identity_test ppi_spill_identity_runtime.cpp)
target_link_libraries(gatzk_ppi_spill_identity_test PRIVATE gatzkml)
target_compile_definitions(gatzk_ppi_spill_identity_test PRIVATE GATZK_BUILD_TESTS=1)
add_test(NAME ppi_spill_identity_runtime COMMAND gatzk_ppi_spill_identity_test)

# Authenticated PPI FH production-path spill/opening benchmark.  It enters
# dynamic materialization, the real spill writer, and paired streamed KZG
# opening; warmup/measured runs remain manual because FH is 2^22.
add_executable(gatzk_ppi_spill_targeted_bench ppi_spill_targeted_bench.cpp)
target_link_libraries(gatzk_ppi_spill_targeted_bench PRIVATE gatzkml)
target_compile_definitions(gatzk_ppi_spill_targeted_bench PRIVATE GATZK_BUILD_TESTS=1)

add_executable(gatzk_effective_memory_admission_test effective_memory_admission_runtime.cpp)
target_link_libraries(gatzk_effective_memory_admission_test PRIVATE gatzkml)
add_test(NAME effective_memory_admission_runtime COMMAND gatzk_effective_memory_admission_test)

# Production-path quotient materialization regression/benchmark.  The default
# CTest invocation is intentionally small; large direct/staged coset cases are
# manual commands so an OOM/long-running resource probe cannot block the test
# suite.  The executable itself calls materialize_quotient_polynomials through
# the production test seam and also exposes exact large-transform probes.
add_executable(
    gatzk_quotient_materializer_targeted_bench
    quotient_materializer_targeted_bench.cpp)
target_link_libraries(
    gatzk_quotient_materializer_targeted_bench
    PRIVATE gatzkml)
target_compile_definitions(
    gatzk_quotient_materializer_targeted_bench
    PRIVATE GATZK_BUILD_TESTS=1)
add_test(
    NAME quotient_materializer_targeted_smoke
    COMMAND gatzk_quotient_materializer_targeted_bench
            --mode materializer
            --route cpu
            --domain-log2 12
            --repeats 1)

# Manual cached-subset multi-rotation benchmark.  It enters the production
# PackedEvaluationBackend path and is intentionally not registered with
# CTest because the 2^18 x 64 synthetic cache is a high-memory probe.
add_executable(
    gatzk_cached_subset_rotation_transpose_targeted_bench
    cached_subset_rotation_transpose_targeted_bench.cpp)
target_link_libraries(
    gatzk_cached_subset_rotation_transpose_targeted_bench
    PRIVATE gatzkml)
target_compile_definitions(
    gatzk_cached_subset_rotation_transpose_targeted_bench
    PRIVATE GATZK_BUILD_TESTS=1)

# The tamper suite is an explicit test-only consumer of an already completed
# proof/run. It never constructs a proving context or starts a formal proof;
# production gatzk_run remains independent of this target.
add_executable(gatzk_protocol_tamper_tests protocol_tamper_tests.cpp)
target_link_libraries(gatzk_protocol_tamper_tests PRIVATE gatzkml)

add_executable(gatzk_t_edge_descriptor_test t_edge_descriptor_runtime.cpp)
target_include_directories(gatzk_t_edge_descriptor_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(gatzk_t_edge_descriptor_test PRIVATE gatzkml)
add_test(NAME t_edge_descriptor_runtime COMMAND gatzk_t_edge_descriptor_test)

# Manual partial multi-graph diagnostic for the real protocol chain. It is
# intentionally not registered with CTest because it requires canonical
# dataset/checkpoint/SRS resources. The executable rejects full-dataset
# contexts.
add_executable(gatzk_protocol_replay_diag protocol_replay_diagnostic.cpp)
target_link_libraries(gatzk_protocol_replay_diag PRIVATE gatzkml)

add_executable(gatzk_forward_trace_regression forward_trace_regression.cpp)
target_link_libraries(gatzk_forward_trace_regression PRIVATE gatzkml)
target_compile_definitions(
    gatzk_forward_trace_regression
    PRIVATE GATZK_SOURCE_ROOT="${PROJECT_SOURCE_DIR}")
add_test(NAME forward_trace_regression COMMAND gatzk_forward_trace_regression)

# Manual production-path KZG commitment benchmark.  It loads the authenticated
# PPI coefficient SRS through KZG's real loader, exercises commit_owned_batch,
# and compares mixed-prefix, exact-length legacy, and equal-length controls.
# The benchmark is intentionally not registered with CTest because even a
# prefix-authenticated SRS load hashes a large formal artifact.
add_executable(
    gatzk_kzg_msm_mixed_prefix_targeted_bench
    kzg_msm_mixed_prefix_targeted_bench.cpp)
target_link_libraries(
    gatzk_kzg_msm_mixed_prefix_targeted_bench
    PRIVATE gatzkml)
target_compile_definitions(
    gatzk_kzg_msm_mixed_prefix_targeted_bench
    PRIVATE GATZK_BUILD_TESTS=1)

# Manual authenticated-registry production seam for dynamic commitment
# batches.  It enters trace::add_dynamic_commitment_batch and KZG::commit_batch
# with dense d:16 evaluation columns, while retaining the production
# compiler-artifact/blinder path.  It is intentionally not a CTest case: the
# benchmark hashes and loads the authenticated PPI SRS/registry.
add_executable(
    gatzk_kzg_dynamic_trace_msm_targeted_bench
    kzg_dynamic_trace_msm_targeted_bench.cpp)
target_link_libraries(
    gatzk_kzg_dynamic_trace_msm_targeted_bench
    PRIVATE gatzkml)
target_compile_definitions(
    gatzk_kzg_dynamic_trace_msm_targeted_bench
    PRIVATE GATZK_BUILD_TESTS=1)

# Manual authenticated-registry production-shape coefficient MSM benchmark.
# It selects the real private coef matrix objects for hidden H_prime, hidden
# concat, and output Y_lin, then enters trace::add_dynamic_commitment_batch ->
# KZG::commit_batch with one mixed-length batch.  The benchmark is deliberately
# not a CTest case because it authenticates and loads a multi-million-point PPI
# coefficient SRS and is intended for an idle GPU only.
add_executable(
    gatzk_kzg_dynamic_trace_matrix_msm_targeted_bench
    kzg_dynamic_trace_matrix_msm_targeted_bench.cpp)
target_link_libraries(
    gatzk_kzg_dynamic_trace_matrix_msm_targeted_bench
    PRIVATE gatzkml)
target_compile_definitions(
    gatzk_kzg_dynamic_trace_matrix_msm_targeted_bench
    PRIVATE GATZK_BUILD_TESTS=1)

add_test(
    NAME cuda_weighted_eval_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_weighted_eval_contract.py")

add_test(
    NAME cuda_t_edge_descriptor_contract
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/test_t_edge_descriptor_contract.py")

if(GATZK_ENABLE_CUDA_BACKEND)
    add_executable(gatzk_cuda_t_edge_test cuda_t_edge_runtime.cpp)
    target_link_libraries(gatzk_cuda_t_edge_test PRIVATE gatzkml)
    add_test(NAME cuda_t_edge_runtime COMMAND gatzk_cuda_t_edge_test)
    set_tests_properties(cuda_t_edge_runtime PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gatzk_cuda_msm_test cuda_msm_runtime.cpp)
    target_link_libraries(gatzk_cuda_msm_test PRIVATE gatzkml)
    target_compile_definitions(gatzk_cuda_msm_test PRIVATE GATZK_BUILD_TESTS=1)
    add_test(NAME cuda_msm_runtime COMMAND gatzk_cuda_msm_test)
    set_tests_properties(cuda_msm_runtime PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gatzk_cuda_msm_big_triangle_bench cuda_msm_big_triangle_bench.cu)
    target_include_directories(
        gatzk_cuda_msm_big_triangle_bench
        PRIVATE "${PROJECT_SOURCE_DIR}/third_party/icicle/icicle/include")
    target_compile_definitions(
        gatzk_cuda_msm_big_triangle_bench
        PRIVATE CURVE=bls12_381 CURVE_ID=2 FIELD_ID=2)
    target_compile_options(
        gatzk_cuda_msm_big_triangle_bench
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-include=cstdint>)
    target_link_libraries(gatzk_cuda_msm_big_triangle_bench PRIVATE gatzkml)
    add_test(
        NAME cuda_msm_big_triangle_bench
        COMMAND gatzk_cuda_msm_big_triangle_bench
                --case 8193 1
                --repeats 1
                --warmups 1)
    set_tests_properties(
        cuda_msm_big_triangle_bench
        PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(gatzk_cuda_weighted_eval_test cuda_weighted_eval_runtime.cpp)
    target_link_libraries(gatzk_cuda_weighted_eval_test PRIVATE gatzkml)
    add_test(NAME cuda_weighted_eval_runtime COMMAND gatzk_cuda_weighted_eval_test)
    set_tests_properties(cuda_weighted_eval_runtime PROPERTIES SKIP_RETURN_CODE 77)

    # Manual natural-order generic NTT/INTT regression.  The default 2^25
    # case exercises the bounded staged production entry points directly;
    # it is intentionally not registered with CTest because its workspace is
    # multi-GiB and must remain under the caller's explicit resource gate.
    add_executable(gatzk_cuda_generic_ntt_runtime cuda_generic_ntt_runtime.cpp)
    target_link_libraries(gatzk_cuda_generic_ntt_runtime PRIVATE gatzkml)

    add_executable(
        gatzk_cuda_t_edge_linear_stage_bench
        cuda_t_edge_linear_stage_bench.cpp)
    target_link_libraries(gatzk_cuda_t_edge_linear_stage_bench PRIVATE gatzkml)
    add_test(
        NAME cuda_t_edge_linear_stage_bench
        COMMAND gatzk_cuda_t_edge_linear_stage_bench)
    set_tests_properties(
        cuda_t_edge_linear_stage_bench
        PROPERTIES SKIP_RETURN_CODE 77)
endif()
