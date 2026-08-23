cmake_minimum_required(VERSION 3.10)
project(zkGPT)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-mcmodel=large -O3 -lpthread -pthread")

include_directories(src)
include_directories(3rd)
include_directories(3rd/mcl/include)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_target(
    gat_zkgpt_formal_preflight
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/formal_preflight.py --source-only
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Fail-closed check for the currently unsupported formal zkGPT route"
)

add_custom_target(
    zkgpt_cuda_reference_reducer
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cuda_backend.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building the non-formal retained CUDA u64 reference reducer"
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rd/mcl/include/mcl/bn.hpp" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rd/mcl/lib/libmcl.a")
    add_executable(zkgpt_hyrax_reference_cli backend/native/hyrax_mle_cli.cpp)
    target_include_directories(zkgpt_hyrax_reference_cli PRIVATE 3rd/mcl/include)
    target_link_libraries(zkgpt_hyrax_reference_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/3rd/mcl/lib/libmcl.a)
    target_compile_options(zkgpt_hyrax_reference_cli PRIVATE -fno-pic -fno-pie)
    target_link_options(zkgpt_hyrax_reference_cli PRIVATE -no-pie)
endif()

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
