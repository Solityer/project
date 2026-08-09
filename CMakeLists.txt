cmake_minimum_required(VERSION 3.16)

project(gatzkml LANGUAGES CXX)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GATZK_BUILD_TESTS "Build unit and regression tests" ON)
option(GATZK_ENABLE_CUDA_BACKEND "Enable optional CUDA algebra backend build" OFF)
option(GATZK_ENABLE_FAST_MSM_DEFAULT "Enable fast MSM / fixed-base batch route by default" ON)
option(GATZK_ENABLE_PARALLEL_FFT_DEFAULT "Enable parallel domain/FFT route by default" ON)
option(GATZK_ENABLE_FFT_BACKEND_UPGRADE_DEFAULT "Enable packed domain evaluation backend by default" ON)
option(GATZK_ENABLE_FFT_KERNEL_UPGRADE_DEFAULT "Enable fused rotated-point FFT/NTT kernel backend by default" ON)
option(GATZK_ENABLE_TRACE_LAYOUT_UPGRADE_DEFAULT "Enable trace/materialization layout upgrade by default" ON)
option(GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT "Enable fast verifier pairing route by default" ON)
option(GATZK_ENABLE_NATIVE_ARCH "Enable -march=native for local performance runs; keep OFF for reproducible portable builds" OFF)
set(GATZK_CRYPTO_BACKEND "mcl" CACHE STRING "Crypto backend preference; only mcl is supported")
set_property(CACHE GATZK_CRYPTO_BACKEND PROPERTY STRINGS mcl)

find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
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
    src/protocol/zkmap.cpp
    src/util/config.cpp
    src/util/logging.cpp
    src/util/route2.cpp
)

if(GATZK_ENABLE_CUDA_BACKEND)
    list(APPEND GATZK_SOURCES
        src/algebra/cuda_backend.cu
    )
endif()

add_library(gatzkml STATIC ${GATZK_SOURCES})
target_include_directories(gatzkml PUBLIC ${PROJECT_SOURCE_DIR}/include)
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
        GATZK_ENABLE_FFT_BACKEND_UPGRADE_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_FFT_BACKEND_UPGRADE_DEFAULT}>,1,0>
        GATZK_ENABLE_FFT_KERNEL_UPGRADE_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_FFT_KERNEL_UPGRADE_DEFAULT}>,1,0>
        GATZK_ENABLE_TRACE_LAYOUT_UPGRADE_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_TRACE_LAYOUT_UPGRADE_DEFAULT}>,1,0>
        GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT=$<IF:$<BOOL:${GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT}>,1,0>
        GATZK_ENABLE_CUDA_BACKEND=$<IF:$<BOOL:${GATZK_ENABLE_CUDA_BACKEND}>,1,0>
)

if(GATZK_USE_MCL)
    target_compile_definitions(gatzkml PUBLIC GATZK_USE_MCL=1)
    target_include_directories(gatzkml PUBLIC ${GATZK_MCL_INCLUDE_DIRS})
    target_link_libraries(gatzkml PUBLIC ${GATZK_MCL_LIBRARIES})
endif()

target_link_libraries(gatzkml PUBLIC Threads::Threads OpenSSL::Crypto)

if(GATZK_ENABLE_CUDA_BACKEND)
    target_link_libraries(gatzkml PUBLIC CUDA::cudart)
    set_target_properties(gatzkml PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(gatzk_run src/main.cpp)
target_link_libraries(gatzk_run PRIVATE gatzkml)

add_executable(gatzk_srs_setup tools/srs/generate_srs.cpp)
target_include_directories(gatzk_srs_setup PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_srs_setup PRIVATE gatzkml)

add_executable(gatzk_msm_stage_bench tools/benchmarks/msm_stage_bench.cpp)
target_include_directories(gatzk_msm_stage_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_msm_stage_bench PRIVATE gatzkml)

add_executable(
    gatzk_taylor_shift_stage_bench
    tools/benchmarks/taylor_shift_stage_bench.cpp)
target_include_directories(
    gatzk_taylor_shift_stage_bench
    PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_taylor_shift_stage_bench PRIVATE gatzkml)

add_executable(
    gatzk_logup_stage_bench
    tools/benchmarks/logup_stage_bench.cpp)
target_include_directories(
    gatzk_logup_stage_bench
    PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_logup_stage_bench PRIVATE gatzkml)

add_executable(gatzk_t_edge_eval_stage_bench tools/benchmarks/t_edge_eval_stage_bench.cpp)
target_include_directories(gatzk_t_edge_eval_stage_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_t_edge_eval_stage_bench PRIVATE gatzkml)

add_executable(gatzk_t_edge_access_plan_stage_bench tools/benchmarks/t_edge_access_plan_stage_bench.cpp)
target_include_directories(gatzk_t_edge_access_plan_stage_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_t_edge_access_plan_stage_bench PRIVATE gatzkml)

add_executable(gatzk_kzg_opening_stage_bench tools/benchmarks/kzg_opening_stage_bench.cpp)
target_include_directories(gatzk_kzg_opening_stage_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_kzg_opening_stage_bench PRIVATE gatzkml)

add_executable(gatzk_proof_codec_stage_bench tools/benchmarks/proof_codec_stage_bench.cpp)
target_include_directories(gatzk_proof_codec_stage_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_proof_codec_stage_bench PRIVATE gatzkml)

add_executable(gatzk_quotient_transform_stage_bench tools/benchmarks/quotient_transform_stage_bench.cpp)
target_include_directories(gatzk_quotient_transform_stage_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_quotient_transform_stage_bench PRIVATE gatzkml)

add_executable(
    gatzk_spilled_evaluation_stage_bench
    tools/benchmarks/spilled_evaluation_stage_bench.cpp)
target_include_directories(
    gatzk_spilled_evaluation_stage_bench
    PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(gatzk_spilled_evaluation_stage_bench PRIVATE gatzkml)

if(GATZK_ENABLE_CUDA_BACKEND)
    add_executable(gatzk_cuda_hotspots_stage_bench tools/benchmarks/cuda_hotspots_stage_bench.cpp)
    target_include_directories(gatzk_cuda_hotspots_stage_bench PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(gatzk_cuda_hotspots_stage_bench PRIVATE gatzkml)
endif()

if(GATZK_BUILD_TESTS)
    enable_testing()
    add_executable(gatzk_tests tests/test_main.cpp)
    target_link_libraries(gatzk_tests PRIVATE gatzkml)
    target_compile_definitions(
        gatzk_tests
        PRIVATE GATZK_TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME gatzk_tests COMMAND gatzk_tests)
    add_executable(gatzk_family_scaling_tests tests/test_family_scaling.cpp)
    target_link_libraries(gatzk_family_scaling_tests PRIVATE gatzkml)
    target_compile_definitions(
        gatzk_family_scaling_tests
        PRIVATE
            GATZK_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
            GATZK_TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME gatzk_family_scaling_tests COMMAND gatzk_family_scaling_tests)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME static_secure_scan
            COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/security/static_scan_no_tau.py)
    endif()
endif()
