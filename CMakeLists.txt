cmake_minimum_required(VERSION 3.16)

project(gatzkml LANGUAGES CXX)

enable_testing()

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GATZK_ENABLE_CUDA_BACKEND "Enable optional CUDA algebra backend build" OFF)
option(GATZK_ENABLE_FAST_MSM_DEFAULT "Enable fast MSM / fixed-base batch route by default" ON)
option(GATZK_ENABLE_PARALLEL_FFT_DEFAULT "Enable parallel domain/FFT route by default" ON)
option(GATZK_ENABLE_PACKED_FFT_DEFAULT "Enable packed domain evaluation backend by default" ON)
option(GATZK_ENABLE_ROTATED_FFT_KERNEL_DEFAULT "Enable fused rotated-point FFT/NTT kernel backend by default" ON)
option(GATZK_ENABLE_COMPACT_TRACE_LAYOUT_DEFAULT "Enable compact trace/materialization layout by default" ON)
option(GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT "Enable fast verifier pairing route by default" ON)
option(GATZK_ENABLE_NATIVE_ARCH "Enable -march=native for local performance runs; keep OFF for reproducible portable builds" OFF)
set(GATZK_CRYPTO_BACKEND "mcl" CACHE STRING "Crypto backend preference; only mcl is supported")
set_property(CACHE GATZK_CRYPTO_BACKEND PROPERTY STRINGS mcl)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Embed a canonical source inventory and compiler receipt in every binary.
# Runtime artifact export additionally hashes the actual executable bytes, so
# arbitrary provenance environment variables cannot promote a formal run.
set(GATZK_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated")
set(GATZK_BUILD_PROVENANCE_HEADER
    "${GATZK_GENERATED_INCLUDE_DIR}/gatzk/build_provenance.hpp")
set(GATZK_SOURCE_MANIFEST
    "${GATZK_GENERATED_INCLUDE_DIR}/gatzk/source_manifest.json")
file(MAKE_DIRECTORY "${GATZK_GENERATED_INCLUDE_DIR}/gatzk")
set(GATZK_BUILD_FLAGS
    "CXX=${CMAKE_CXX_FLAGS};BUILD=${CMAKE_BUILD_TYPE};NATIVE=${GATZK_ENABLE_NATIVE_ARCH};CUDA=${GATZK_ENABLE_CUDA_BACKEND};FAST_MSM=${GATZK_ENABLE_FAST_MSM_DEFAULT};PARALLEL_FFT=${GATZK_ENABLE_PARALLEL_FFT_DEFAULT};PACKED_FFT=${GATZK_ENABLE_PACKED_FFT_DEFAULT};ROTATED_FFT_KERNEL=${GATZK_ENABLE_ROTATED_FFT_KERNEL_DEFAULT};COMPACT_TRACE_LAYOUT=${GATZK_ENABLE_COMPACT_TRACE_LAYOUT_DEFAULT};FAST_VERIFY=${GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT}")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/build_provenance.py"
            --root "${PROJECT_SOURCE_DIR}"
            --output "${GATZK_BUILD_PROVENANCE_HEADER}"
            --source-manifest-output "${GATZK_SOURCE_MANIFEST}"
            --compiler-id "${CMAKE_CXX_COMPILER_ID}"
            --compiler-version "${CMAKE_CXX_COMPILER_VERSION}"
            --build-type "${CMAKE_BUILD_TYPE}"
            --flags "${GATZK_BUILD_FLAGS}"
    RESULT_VARIABLE GATZK_PROVENANCE_RESULT)
if(NOT GATZK_PROVENANCE_RESULT EQUAL 0)
    message(FATAL_ERROR "failed to generate source-locked build provenance")
endif()
if(GATZK_ENABLE_NATIVE_ARCH)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" GATZK_CXX_SUPPORTS_MARCH_NATIVE)
endif()

include(cmake/ConfigureMCL.cmake)

if(GATZK_ENABLE_CUDA_BACKEND)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_LESS 3.24)
            message(FATAL_ERROR
                "GAT-ZKML: CUDA builds require an explicit "
                "-DCMAKE_CUDA_ARCHITECTURES=<architecture> with CMake < 3.24")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
            "CUDA architectures for the canonical CUDA backend")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(GATZK_SOURCES
    src/algebra/eval_backend.cpp
    src/algebra/field_io.cpp
    src/algebra/polynomial.cpp
    src/algebra/vector_ops.cpp
    src/crypto/curve.cpp
    src/crypto/cuda_msm_stub.cpp
    src/crypto/kzg.cpp
    src/crypto/transcript.cpp
    src/data/loader.cpp
    src/model/gat.cpp
    src/protocol/lookup.cpp
    src/protocol/challenges.cpp
    src/protocol/opening_registry.cpp
    src/protocol/artifact_export.cpp
    src/protocol/prover.cpp
    src/protocol/proof_codec.cpp
    src/protocol/psq.cpp
    src/protocol/quotients.cpp
    src/protocol/schema.cpp
    src/protocol/schema_labels.cpp
    src/protocol/trace.cpp
    src/protocol/verifier.cpp
    src/protocol/verifier_context.cpp
    src/protocol/verifier_registration_artifact.cpp
    src/protocol/zkmap.cpp
    src/util/config.cpp
    src/util/logging.cpp
    src/util/route2.cpp
    src/protocol/cuda_t_edge_stub.cpp
)

if(GATZK_ENABLE_CUDA_BACKEND)
    list(APPEND GATZK_SOURCES
        src/algebra/cuda_backend.cu
        src/algebra/cuda_ntt.cu
        src/crypto/cuda_msm.cu
        src/protocol/cuda_t_edge.cu
    )

    list(REMOVE_ITEM GATZK_SOURCES
        src/crypto/cuda_msm_stub.cpp
        src/protocol/cuda_t_edge_stub.cpp)

    set(GATZK_ICICLE_ROOT "${PROJECT_SOURCE_DIR}/third_party/icicle")
    if(NOT EXISTS "${GATZK_ICICLE_ROOT}/LICENSE"
        OR NOT EXISTS "${GATZK_ICICLE_ROOT}/icicle/src/msm/extern.cu"
        OR NOT EXISTS "${GATZK_ICICLE_ROOT}/icicle/src/ntt/extern.cu"
        OR NOT EXISTS "${GATZK_ICICLE_ROOT}/icicle/src/ntt/kernel_ntt.cu")
        message(FATAL_ERROR
            "GAT-ZKML: CUDA MSM/NTT requires ICICLE v2.8.0 at the pinned "
            "aacec3f72f14f53dd96d835698d2c87cde3a68b8 gitlink; "
            "run 'git submodule update --init third_party/icicle'")
    endif()
    add_library(
        gatzk_icicle_bls12_381_msm STATIC
        "${GATZK_ICICLE_ROOT}/icicle/src/msm/extern.cu")
    target_include_directories(
        gatzk_icicle_bls12_381_msm
        PUBLIC "${GATZK_ICICLE_ROOT}/icicle/include")
    target_compile_definitions(
        gatzk_icicle_bls12_381_msm
        PRIVATE CURVE=bls12_381 CURVE_ID=2 FIELD_ID=2)
    target_compile_options(
        gatzk_icicle_bls12_381_msm
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-include=cstdint>)
    target_link_libraries(
        gatzk_icicle_bls12_381_msm PRIVATE CUDA::cudart)
    set_target_properties(
        gatzk_icicle_bls12_381_msm
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)
    add_library(
        gatzk_icicle_bls12_381_ntt STATIC
        "${GATZK_ICICLE_ROOT}/icicle/src/ntt/extern.cu"
        "${GATZK_ICICLE_ROOT}/icicle/src/ntt/kernel_ntt.cu")

    target_include_directories(
        gatzk_icicle_bls12_381_ntt
        PUBLIC "${GATZK_ICICLE_ROOT}/icicle/include")

    target_compile_definitions(
        gatzk_icicle_bls12_381_ntt
        PRIVATE
            FIELD=bls12_381
            FIELD_ID=2)

    target_compile_options(
        gatzk_icicle_bls12_381_ntt
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
            $<$<COMPILE_LANGUAGE:CUDA>:-include=cstdint>)

    target_link_libraries(
        gatzk_icicle_bls12_381_ntt
        PRIVATE CUDA::cudart)

    set_target_properties(
        gatzk_icicle_bls12_381_ntt
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON)
endif()

add_library(gatzkml STATIC ${GATZK_SOURCES})
target_include_directories(gatzkml PUBLIC
    "${GATZK_GENERATED_INCLUDE_DIR}"
    ${PROJECT_SOURCE_DIR}/include)
target_include_directories(gatzkml PRIVATE
    ${PROJECT_SOURCE_DIR}/src)
target_compile_options(gatzkml PRIVATE -Wall -Wextra -Wpedantic)
if(GATZK_ENABLE_NATIVE_ARCH AND GATZK_CXX_SUPPORTS_MARCH_NATIVE)
    target_compile_options(
        gatzkml
        PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-march=native>)
endif()
target_compile_definitions(
    gatzkml
    PUBLIC
        GATZK_CRYPTO_BACKEND_NAME="${GATZK_SELECTED_BACKEND}"
        GATZK_ENABLE_FAST_MSM_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_FAST_MSM_DEFAULT}>,1,0>
        GATZK_ENABLE_PARALLEL_FFT_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_PARALLEL_FFT_DEFAULT}>,1,0>
        GATZK_ENABLE_PACKED_FFT_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_PACKED_FFT_DEFAULT}>,1,0>
        GATZK_ENABLE_ROTATED_FFT_KERNEL_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_ROTATED_FFT_KERNEL_DEFAULT}>,1,0>
        GATZK_ENABLE_COMPACT_TRACE_LAYOUT_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_COMPACT_TRACE_LAYOUT_DEFAULT}>,1,0>
        GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT}>,1,0>
        GATZK_ENABLE_CUDA_BACKEND=$<IF:$<BOOL:${GATZK_ENABLE_CUDA_BACKEND}>,1,0>
        GATZK_CANONICAL_SCALE_CHAIN_PATH="${PROJECT_SOURCE_DIR}/configs/quantization_scale_chain.cfg"
)

if(GATZK_USE_MCL)
    target_compile_definitions(gatzkml PUBLIC GATZK_USE_MCL=1)
    target_include_directories(gatzkml PUBLIC ${GATZK_MCL_INCLUDE_DIRS})
    target_link_libraries(gatzkml PUBLIC ${GATZK_MCL_LIBRARIES})
endif()

target_link_libraries(gatzkml PUBLIC Threads::Threads OpenSSL::Crypto)

if(GATZK_ENABLE_CUDA_BACKEND)
    target_link_libraries(
        gatzkml
        PUBLIC CUDA::cudart
        PRIVATE
            gatzk_icicle_bls12_381_msm
            gatzk_icicle_bls12_381_ntt)
    set_target_properties(gatzkml PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(gatzk_run src/main.cpp)
target_link_libraries(gatzk_run PRIVATE gatzkml)

# Explicit setup utility for the authenticated point-only VK_model/VK_static
# registry.  Formal runs consume an existing registry and fail closed when it
# is absent; this target is never invoked implicitly by gatzk_run.
add_executable(gatzk_registry_register tools/register_verifier_registry.cpp)
target_link_libraries(gatzk_registry_register PRIVATE gatzkml)

# Short-lived independent inspection utility used by the fail-closed result
# aggregator.  It consumes only the canonical point-only registration artifact
# and emits authenticated registry/ledger metadata; it never constructs a
# proving context or starts a proof run.
add_executable(gatzk_registry_inspect tools/inspect_verifier_registry.cpp)
target_link_libraries(gatzk_registry_inspect PRIVATE gatzkml)

add_executable(gatzk_srs_setup tools/srs/generate_srs.cpp)
target_include_directories(gatzk_srs_setup PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_srs_setup PRIVATE gatzkml)

# Test-only runtime regression for the layered formal schema.  It exercises
# the production label, selector, and checkpoint-shape APIs without touching
# canonical checkpoints, SRS files, or proof runs.
add_executable(gatzk_layered_schema_test tests/layered_schema_runtime.cpp)
target_link_libraries(gatzk_layered_schema_test PRIVATE gatzkml)
add_test(NAME layered_schema_runtime COMMAND gatzk_layered_schema_test)

add_executable(gatzk_verifier_privacy_test tests/verifier_privacy_runtime.cpp)
target_link_libraries(gatzk_verifier_privacy_test PRIVATE gatzkml)
add_test(NAME verifier_privacy_runtime COMMAND gatzk_verifier_privacy_test)

# Focused public AggBind reconstruction regression.  The internal header is
# deliberately outside the installed include tree and is used only to share
# the production verifier helper with this small synthetic test.
add_executable(gatzk_agg_bind_multiplicity_test tests/agg_bind_multiplicity_runtime.cpp)
target_include_directories(gatzk_agg_bind_multiplicity_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(gatzk_agg_bind_multiplicity_test PRIVATE gatzkml)
add_test(NAME agg_bind_multiplicity_runtime COMMAND gatzk_agg_bind_multiplicity_test)

# Short-lived cache identity/TOCTOU regression.  It hashes only small
# temporary resource fixtures and never loads a real SRS or starts a proof.
add_executable(gatzk_kzg_resource_identity_test tests/kzg_resource_identity_runtime.cpp)
target_link_libraries(gatzk_kzg_resource_identity_test PRIVATE gatzkml)
add_test(NAME kzg_resource_identity_runtime COMMAND gatzk_kzg_resource_identity_test)

add_executable(gatzk_compiler_artifact_test tests/compiler_artifact_runtime.cpp)
target_link_libraries(gatzk_compiler_artifact_test PRIVATE gatzkml)
add_test(NAME compiler_artifact_runtime COMMAND gatzk_compiler_artifact_test)

add_executable(gatzk_input_quantization_test tests/input_quantization_runtime.cpp)
target_link_libraries(gatzk_input_quantization_test PRIVATE gatzkml)
add_test(NAME input_quantization_runtime COMMAND gatzk_input_quantization_test)

if(GATZK_ENABLE_CUDA_BACKEND)
    add_executable(gatzk_cuda_t_edge_test tests/cuda_t_edge_runtime.cpp)
    target_link_libraries(gatzk_cuda_t_edge_test PRIVATE gatzkml)
    add_test(NAME cuda_t_edge_runtime COMMAND gatzk_cuda_t_edge_test)
    set_tests_properties(cuda_t_edge_runtime PROPERTIES SKIP_RETURN_CODE 77)
endif()
