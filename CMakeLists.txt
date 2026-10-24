cmake_minimum_required(VERSION 3.16)

project(gatzkml LANGUAGES CXX)

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
option(GATZK_ENABLE_NATIVE_ARCH "Enable -march=native for local performance runs; keep OFF for reproducible portable builds" ON)
set(GATZK_CRYPTO_BACKEND "mcl" CACHE STRING "Crypto backend preference; only mcl is supported")
set_property(CACHE GATZK_CRYPTO_BACKEND PROPERTY STRINGS mcl)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(GATZK_GENERATED_INCLUDE_DIR "${CMAKE_BINARY_DIR}/generated")
set(GATZK_BUILD_PROVENANCE_HEADER "${GATZK_GENERATED_INCLUDE_DIR}/gatzk/build_provenance.hpp")
set(GATZK_SOURCE_MANIFEST "${GATZK_GENERATED_INCLUDE_DIR}/gatzk/source_manifest.json")
file(MAKE_DIRECTORY "${GATZK_GENERATED_INCLUDE_DIR}/gatzk")
set(
    GATZK_BUILD_FLAGS
    "CXX=${CMAKE_CXX_FLAGS};BUILD=${CMAKE_BUILD_TYPE};NATIVE=${GATZK_ENABLE_NATIVE_ARCH};CUDA=${GATZK_ENABLE_CUDA_BACKEND};FAST_MSM=${GATZK_ENABLE_FAST_MSM_DEFAULT};PARALLEL_FFT=${GATZK_ENABLE_PARALLEL_FFT_DEFAULT};PACKED_FFT=${GATZK_ENABLE_PACKED_FFT_DEFAULT};ROTATED_FFT_KERNEL=${GATZK_ENABLE_ROTATED_FFT_KERNEL_DEFAULT};COMPACT_TRACE_LAYOUT=${GATZK_ENABLE_COMPACT_TRACE_LAYOUT_DEFAULT};FAST_VERIFY=${GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT}"
)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/build_provenance.py" --root "${PROJECT_SOURCE_DIR}" --output
                        "${GATZK_BUILD_PROVENANCE_HEADER}" --source-manifest-output "${GATZK_SOURCE_MANIFEST}" --compiler-id "${CMAKE_CXX_COMPILER_ID}"
                        --compiler-version "${CMAKE_CXX_COMPILER_VERSION}" --build-type "${CMAKE_BUILD_TYPE}" --flags "${GATZK_BUILD_FLAGS}"
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
            message(FATAL_ERROR "GAT-ZKML: CUDA builds require an explicit " "-DCMAKE_CUDA_ARCHITECTURES=<architecture> with CMake < 3.24")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures for the canonical CUDA backend")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(GATZK_SOURCES
    src/algebra/eval_backend.cpp src/algebra/field_io.cpp src/util/effective_memory.cpp src/util/spill_arena_accounting.cpp src/util/storage_reservation.cpp
    src/util/trace_spill.cpp src/algebra/polynomial.cpp src/algebra/vector_ops.cpp src/crypto/curve.cpp src/crypto/cuda_msm_stub.cpp src/crypto/kzg.cpp
    src/crypto/transcript.cpp src/data/loader.cpp src/model/gat.cpp src/protocol/lookup.cpp src/protocol/challenges.cpp src/protocol/opening_registry.cpp
    src/protocol/artifact_export.cpp src/protocol/internal/t_edge_descriptor.cpp src/protocol/prover.cpp src/protocol/proof_codec.cpp src/protocol/psq.cpp
    src/protocol/quotients.cpp src/protocol/schema.cpp src/protocol/schema_labels.cpp src/protocol/trace.cpp src/protocol/verifier.cpp
    src/protocol/verifier_context.cpp src/protocol/verifier_registration_artifact.cpp src/protocol/zkmap.cpp src/util/config.cpp src/util/logging.cpp
    src/util/route2.cpp src/protocol/cuda_t_edge_stub.cpp src/protocol/cuda_t_fh_stub.cpp src/mechanism/runtime.cpp src/mechanism/route.cpp src/mechanism/codec.cpp
    src/mechanism/feature_folding.cpp src/mechanism/shared_neighborhood.cpp src/mechanism/real_input.cpp)

if(GATZK_ENABLE_CUDA_BACKEND)
    list(APPEND GATZK_SOURCES src/algebra/cuda_backend.cu src/algebra/cuda_ntt.cu src/crypto/cuda_msm.cu src/protocol/cuda_t_edge.cu src/protocol/cuda_t_fh.cu)

    list(REMOVE_ITEM GATZK_SOURCES src/crypto/cuda_msm_stub.cpp src/protocol/cuda_t_edge_stub.cpp src/protocol/cuda_t_fh_stub.cpp)

    set(GATZK_ICICLE_ROOT "${PROJECT_SOURCE_DIR}/third_party/icicle")
    if(NOT EXISTS "${GATZK_ICICLE_ROOT}/LICENSE" OR NOT EXISTS "${GATZK_ICICLE_ROOT}/icicle/src/msm/extern.cu"
       OR NOT EXISTS "${GATZK_ICICLE_ROOT}/icicle/src/ntt/extern.cu" OR NOT EXISTS "${GATZK_ICICLE_ROOT}/icicle/src/ntt/kernel_ntt.cu")
        message(FATAL_ERROR "GAT-ZKML: CUDA MSM/NTT requires ICICLE v2.8.0 at the pinned " "aacec3f72f14f53dd96d835698d2c87cde3a68b8 gitlink; "
                            "run 'git submodule update --init third_party/icicle'")
    endif()
    set(GATZK_ICICLE_MSM_PATCHED_DIR "${CMAKE_BINARY_DIR}/generated/icicle_v28_msm")

    set(GATZK_ICICLE_MSM_PATCHED_EXTERN "${GATZK_ICICLE_MSM_PATCHED_DIR}/extern.cu")

    set(GATZK_ICICLE_MSM_PATCHED_IMPL "${GATZK_ICICLE_MSM_PATCHED_DIR}/msm.cu")

    add_custom_command(OUTPUT "${GATZK_ICICLE_MSM_PATCHED_EXTERN}" "${GATZK_ICICLE_MSM_PATCHED_IMPL}" COMMAND "${CMAKE_COMMAND}" -E make_directory
                                                                                                              "${GATZK_ICICLE_MSM_PATCHED_DIR}"
                       COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${GATZK_ICICLE_ROOT}/icicle/src/msm/extern.cu" "${GATZK_ICICLE_MSM_PATCHED_EXTERN}"
                       COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/patch_icicle_v28_msm.py" --input
                               "${GATZK_ICICLE_ROOT}/icicle/src/msm/msm.cu" --output "${GATZK_ICICLE_MSM_PATCHED_IMPL}"
                       DEPENDS "${PROJECT_SOURCE_DIR}/tools/patch_icicle_v28_msm.py" "${GATZK_ICICLE_ROOT}/icicle/src/msm/extern.cu"
                               "${GATZK_ICICLE_ROOT}/icicle/src/msm/msm.cu" VERBATIM)

    add_library(gatzk_icicle_bls12_381_msm STATIC "${GATZK_ICICLE_MSM_PATCHED_EXTERN}")
    target_include_directories(gatzk_icicle_bls12_381_msm PUBLIC "${GATZK_ICICLE_ROOT}/icicle/include")
    target_compile_definitions(gatzk_icicle_bls12_381_msm PRIVATE CURVE=bls12_381 CURVE_ID=2 FIELD_ID=2)
    target_compile_options(gatzk_icicle_bls12_381_msm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-include=cstdint>)
    target_link_libraries(gatzk_icicle_bls12_381_msm PRIVATE CUDA::cudart)
    set_target_properties(gatzk_icicle_bls12_381_msm PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
    add_library(gatzk_icicle_bls12_381_ntt STATIC "${GATZK_ICICLE_ROOT}/icicle/src/ntt/extern.cu" "${GATZK_ICICLE_ROOT}/icicle/src/ntt/kernel_ntt.cu")

    target_include_directories(gatzk_icicle_bls12_381_ntt PUBLIC "${GATZK_ICICLE_ROOT}/icicle/include")

    target_compile_definitions(gatzk_icicle_bls12_381_ntt PRIVATE FIELD=bls12_381 FIELD_ID=2)

    target_compile_options(gatzk_icicle_bls12_381_ntt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr> $<$<COMPILE_LANGUAGE:CUDA>:-include=cstdint>)

    target_link_libraries(gatzk_icicle_bls12_381_ntt PRIVATE CUDA::cudart)

    set_target_properties(gatzk_icicle_bls12_381_ntt PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
endif()

add_library(gatzkml STATIC ${GATZK_SOURCES})
target_include_directories(gatzkml PUBLIC "${GATZK_GENERATED_INCLUDE_DIR}" ${PROJECT_SOURCE_DIR}/include)
target_include_directories(gatzkml PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(gatzkml PRIVATE -Wall -Wextra -Wpedantic)
if(GATZK_ENABLE_NATIVE_ARCH AND GATZK_CXX_SUPPORTS_MARCH_NATIVE)
    target_compile_options(gatzkml PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-march=native>)
endif()
target_compile_definitions(gatzkml
                           PUBLIC GATZK_CRYPTO_BACKEND_NAME="${GATZK_SELECTED_BACKEND}"
                                  GATZK_ENABLE_FAST_MSM_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_FAST_MSM_DEFAULT}>,1,0>
                                  GATZK_ENABLE_PARALLEL_FFT_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_PARALLEL_FFT_DEFAULT}>,1,0>
                                  GATZK_ENABLE_PACKED_FFT_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_PACKED_FFT_DEFAULT}>,1,0>
                                  GATZK_ENABLE_ROTATED_FFT_KERNEL_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_ROTATED_FFT_KERNEL_DEFAULT}>,1,0>
                                  GATZK_ENABLE_COMPACT_TRACE_LAYOUT_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_COMPACT_TRACE_LAYOUT_DEFAULT}>,1,0>
                                  GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT}>,1,0>
                                  GATZK_ENABLE_CUDA_BACKEND=$<IF:$<BOOL:${GATZK_ENABLE_CUDA_BACKEND}>,1,0>
                                  GATZK_CANONICAL_SCALE_CHAIN_PATH="${PROJECT_SOURCE_DIR}/configs/quantization_scale_chain.cfg")

if(GATZK_USE_MCL)
    target_compile_definitions(gatzkml PUBLIC GATZK_USE_MCL=1)
    target_include_directories(gatzkml PUBLIC ${GATZK_MCL_INCLUDE_DIRS})
    target_link_libraries(gatzkml PUBLIC ${GATZK_MCL_LIBRARIES})
endif()

target_link_libraries(gatzkml PUBLIC Threads::Threads OpenSSL::Crypto)

if(GATZK_ENABLE_CUDA_BACKEND)
    target_link_libraries(gatzkml PUBLIC CUDA::cudart PRIVATE gatzk_icicle_bls12_381_msm gatzk_icicle_bls12_381_ntt)
    set_target_properties(gatzkml PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(gatzk_run src/main.cpp)
target_link_libraries(gatzk_run PRIVATE gatzkml)

add_executable(gatzk_mechanism src/mechanism/main.cpp)
target_link_libraries(gatzk_mechanism PRIVATE gatzkml)

add_executable(gatzk_registry_register tools/register_verifier_registry.cpp)
target_link_libraries(gatzk_registry_register PRIVATE gatzkml)

add_executable(gatzk_registry_inspect tools/inspect_verifier_registry.cpp)
target_link_libraries(gatzk_registry_inspect PRIVATE gatzkml)

add_executable(gatzk_srs_setup tools/srs/generate_srs.cpp)
target_include_directories(gatzk_srs_setup PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_srs_setup PRIVATE gatzkml)

option(GATZK_BUILD_CORE_EXECUTION_REGRESSION "Build isolated core execution regression" OFF)
if(GATZK_BUILD_CORE_EXECUTION_REGRESSION)
    add_executable(gatzk_core_execution_regression EXCLUDE_FROM_ALL tests/core_execution_regression.cpp)
    target_link_libraries(gatzk_core_execution_regression PRIVATE gatzkml)
    target_compile_definitions(gatzk_core_execution_regression PRIVATE GATZK_EXECUTION_TEST_ROOT="${PROJECT_SOURCE_DIR}/tests/.core_execution")
endif()

option(GATZK_BUILD_DEFAULT_PROOF_REGRESSION "Build default-only independent proof regression" OFF)
if(GATZK_BUILD_DEFAULT_PROOF_REGRESSION)
    add_executable(gatzk_default_proof_regression EXCLUDE_FROM_ALL tests/default_proof_regression.cpp)
    target_link_libraries(gatzk_default_proof_regression PRIVATE gatzkml)
    target_compile_definitions(
        gatzk_default_proof_regression
        PRIVATE GATZK_DEFAULT_PROOF_TEST_ROOT="${PROJECT_SOURCE_DIR}"
                GATZK_DEFAULT_PROOF_TEST_SRS_SETUP="$<TARGET_FILE:gatzk_srs_setup>")
    add_dependencies(gatzk_default_proof_regression gatzk_srs_setup)
endif()

option(GATZK_BUILD_MECHANISM_SECURITY_REGRESSION "Build mechanism transcript security regression" OFF)
if(GATZK_BUILD_MECHANISM_SECURITY_REGRESSION)
    add_executable(gatzk_mechanism_security_regression EXCLUDE_FROM_ALL tests/mechanism_security_regression.cpp)
    target_link_libraries(gatzk_mechanism_security_regression PRIVATE gatzkml)
endif()

option(GATZK_BUILD_MECHANISM_ABLATION_PROOF_REGRESSION "Build test-only complete mechanism proof/verify regression" OFF)
if(GATZK_BUILD_MECHANISM_ABLATION_PROOF_REGRESSION)
    add_executable(gatzk_mechanism_ablation_proof_regression EXCLUDE_FROM_ALL tests/mechanism_ablation_proof_regression.cpp)
    target_link_libraries(gatzk_mechanism_ablation_proof_regression PRIVATE gatzkml)
endif()

option(GATZK_BUILD_REAL_MECHANISM_INPUT_REGRESSION "Build real mechanism input relation and codec regression" OFF)
if(GATZK_BUILD_REAL_MECHANISM_INPUT_REGRESSION)
    add_executable(gatzk_real_mechanism_input_regression EXCLUDE_FROM_ALL tests/real_mechanism_input_regression.cpp)
    target_link_libraries(gatzk_real_mechanism_input_regression PRIVATE gatzkml)
endif()
