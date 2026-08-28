cmake_minimum_required(VERSION 3.18)

# The formal baseline vendors the exact Taskflow headers used by the ICICLE
# CPU primitives use only the baseline-local Taskflow headers.  Network
# dependency acquisition is forbidden in a formal offline build.
set(TASKFLOW_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../")
if(NOT EXISTS "${TASKFLOW_INCLUDE_DIR}/taskflow/taskflow.hpp")
  message(FATAL_ERROR
    "baseline-local Taskflow v3.8.0 headers are missing: ${TASKFLOW_INCLUDE_DIR}/taskflow/taskflow.hpp")
endif()
message(STATUS "Using baseline-local Taskflow v3.8.0 headers")
target_include_directories(icicle_device INTERFACE "${TASKFLOW_INCLUDE_DIR}")

# CPU backend is built directly into icicle library

target_sources(icicle_device PRIVATE src/cpu_device_api.cpp)
# field API library
if (FIELD)
    target_sources(icicle_field PRIVATE
        src/field/cpu_vec_ops.cpp
    )
    if (NTT)
        target_sources(icicle_field PRIVATE src/field/cpu_ntt.cpp src/polynomials/cpu_polynomial_backend.cpp)
    endif()
    if (POSEIDON)
        target_sources(icicle_field PRIVATE src/hash/cpu_poseidon.cpp)
    endif()
    if (POSEIDON2)
        target_sources(icicle_field PRIVATE src/hash/cpu_poseidon2.cpp)
    endif()
    if(SUMCHECK)
        target_sources(icicle_field PRIVATE src/field/cpu_sumcheck.cpp)
    endif()
    if(FRI)
        target_sources(icicle_field PRIVATE src/field/cpu_fri.cpp)
    endif()
    if(GATEOPS)
        target_sources(icicle_field PRIVATE src/field/cpu_gate_ops.cpp)
    endif()
    target_include_directories(icicle_field PRIVATE include)
endif() # FIELD

# curve API library
if (CURVE)
    target_sources(icicle_curve PRIVATE
        src/curve/cpu_mont_conversion.cpp
    )
    if (MSM)
        target_sources(icicle_curve PRIVATE src/curve/cpu_msm.cpp)
    endif()
    if (ECNTT)
        target_sources(icicle_curve PRIVATE src/curve/cpu_ecntt.cpp)
    endif()
    # incdir is PUBLIC because config-extension headers are exposed to app
    target_include_directories(icicle_curve PUBLIC include)
endif()

if (HASH)
    target_sources(icicle_hash PRIVATE
        src/hash/cpu_keccak.cpp
        src/hash/cpu_blake2s.cpp
        src/hash/cpu_blake3.cpp
        src/hash/blake3.c
        src/hash/blake3_dispatch.c
        src/hash/blake3_portable.c
        src/hash/cpu_merkle_tree.cpp
        src/hash/cpu_pow.cpp
    )
    target_include_directories(icicle_hash PUBLIC include)
endif()

if (RING)
    target_sources(icicle_ring PRIVATE 
        src/field/cpu_vec_ops.cpp 
        src/ring/cpu_balanced_decomposition.cpp
    )
    if (NTT)
        target_sources(icicle_ring PRIVATE src/field/cpu_ntt.cpp)
    endif()
    target_include_directories(icicle_ring PRIVATE include)
endif()
