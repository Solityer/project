cmake_minimum_required(VERSION 3.16)

project(gatzkml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(GATZK_BUILD_TESTS "Build unit and smoke tests" ON)
option(GATZK_ENABLE_CUDA_BACKEND "Enable optional CUDA algebra backend build" OFF)
option(GATZK_ENABLE_FAST_MSM_DEFAULT "Enable fast MSM / fixed-base batch route by default" ON)
option(GATZK_ENABLE_PARALLEL_FFT_DEFAULT "Enable parallel domain/FFT route by default" ON)
option(GATZK_ENABLE_FFT_BACKEND_UPGRADE_DEFAULT "Enable packed domain evaluation backend by default" ON)
option(GATZK_ENABLE_FFT_KERNEL_UPGRADE_DEFAULT "Enable fused rotated-point FFT/NTT kernel backend by default" ON)
option(GATZK_ENABLE_TRACE_LAYOUT_UPGRADE_DEFAULT "Enable trace/materialization layout upgrade by default" ON)
option(GATZK_ENABLE_FAST_VERIFY_PAIRING_DEFAULT "Enable fast verifier pairing route by default" ON)
set(GATZK_CRYPTO_BACKEND "mcl" CACHE STRING "Crypto backend preference; only mcl is supported")
set_property(CACHE GATZK_CRYPTO_BACKEND PROPERTY STRINGS mcl)

find_package(Threads REQUIRED)

include(cmake/ConfigureMCL.cmake)

if(GATZK_ENABLE_CUDA_BACKEND)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(GATZK_SOURCES
    src/algebra/eval_backend.cpp
    src/algebra/packed_field.cpp
    src/algebra/polynomial.cpp
    src/algebra/vector_ops.cpp
    src/crypto/curve.cpp
    src/crypto/kzg.cpp
    src/crypto/transcript.cpp
    src/data/loader.cpp
    src/model/gat.cpp
    src/protocol/lookup.cpp
    src/protocol/challenges.cpp
    src/protocol/prover.cpp
    src/protocol/psq.cpp
    src/protocol/quotients.cpp
    src/protocol/schema.cpp
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

target_link_libraries(gatzkml PUBLIC Threads::Threads)

if(GATZK_ENABLE_CUDA_BACKEND)
    target_link_libraries(gatzkml PUBLIC CUDA::cudart)
    set_target_properties(gatzkml PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_executable(gatzk_run src/main.cpp)
target_link_libraries(gatzk_run PRIVATE gatzkml)

if(GATZK_BUILD_TESTS)
    enable_testing()
    add_executable(gatzk_tests tests/test_main.cpp)
    target_link_libraries(gatzk_tests PRIVATE gatzkml)
    add_test(NAME gatzk_tests COMMAND gatzk_tests)
endif()
