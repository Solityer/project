cmake_minimum_required(VERSION 3.10)
project(zkCNN)
set(CMAKE_CXX_STANDARD 14)
# The upstream demo was developed on x86_64, where the large code model is
# useful for its very large GPT tensors.  It is not accepted by Clang's
# default PIC mode on the aarch64 build host, and is unnecessary for the
# compact baseline adapter; keep it for the original x86_64 path only.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "-mcmodel=large -O3 -lpthread -pthread")
else()
    set(CMAKE_CXX_FLAGS "-O3 -lpthread -pthread")
endif()
link_directories(3rd/mcl)

include_directories(src)
include_directories(3rd)
include_directories(3rd/mcl/include)

option(ZKGPT_BUILD_TESTS "Build short zkGPT GAT adapter tests" OFF)
option(ZKGPT_ENABLE_CUDA "Build the real CUDA GAT relation bridge" OFF)
if(ZKGPT_BUILD_TESTS)
    enable_testing()
endif()

if(ZKGPT_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(src)
add_subdirectory(3rd/mcl)
add_subdirectory(gat_adapter)
if(ZKGPT_BUILD_TESTS)
    add_subdirectory(tests)
endif()
