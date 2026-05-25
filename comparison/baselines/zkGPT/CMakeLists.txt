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

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rd/mcl/CMakeLists.txt")
    link_directories(3rd/mcl)
    add_subdirectory(src)
    add_subdirectory(3rd/mcl)
else()
    message(WARNING "3rd/mcl is not populated; skipping legacy LLM demo target. GAT-ZKML-zkGPT experiment scripts remain available.")
endif()
