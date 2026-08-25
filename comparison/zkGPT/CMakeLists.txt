cmake_minimum_required(VERSION 3.10)
project(zkGPT NONE)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(
    gat_zkgpt_formal_preflight
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/formal_preflight.py --source-only
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Source/configuration check for the native formal zkGPT route"
)

add_custom_target(
    zkgpt_cuda_native_prover
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cuda_backend.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building the CUDA proof-primitives backend"
)
