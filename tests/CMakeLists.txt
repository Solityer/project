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

add_executable(gatzk_effective_memory_admission_test effective_memory_admission_runtime.cpp)
target_link_libraries(gatzk_effective_memory_admission_test PRIVATE gatzkml)
add_test(NAME effective_memory_admission_runtime COMMAND gatzk_effective_memory_admission_test)

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
