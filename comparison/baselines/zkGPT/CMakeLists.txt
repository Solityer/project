cmake_minimum_required(VERSION 3.10)
project(zkGPT)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-mcmodel=large -O3 -lpthread -pthread")

include_directories(src)
include_directories(3rd)
include_directories(3rd/mcl/include)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(
    gat_zkgpt_check
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gat_zkgpt_experiment.py --help
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking GAT-ZKML-zkGPT experiment entrypoint"
)

add_custom_target(
    zkgpt_cuda_backend
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cuda_backend.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building zkGPT CUDA backend helper"
)

option(ZKGPT_BUILD_LEGACY_LLM_DEMO "Build the original zkGPT LLM demo against 3rd/mcl" OFF)
if(ZKGPT_BUILD_LEGACY_LLM_DEMO)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rd/mcl/CMakeLists.txt")
        link_directories(3rd/mcl)
        add_subdirectory(3rd/mcl)
        add_subdirectory(src)
    else()
        message(FATAL_ERROR "ZKGPT_BUILD_LEGACY_LLM_DEMO=ON requires populated 3rd/mcl")
    endif()
endif()
