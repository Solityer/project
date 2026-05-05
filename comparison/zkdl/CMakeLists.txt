cmake_minimum_required(VERSION 3.18)

project(zkdl_gat_full_forward_baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(zkdl_gat_full_forward
    src/zkdl_gat/experiment.cpp
    src/zkdl_gat/main.cpp
)

target_include_directories(zkdl_gat_full_forward PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

option(ZKDL_ENABLE_CUDA_BACKEND "Build the CUDA zkDL primitive probe" OFF)

if(ZKDL_ENABLE_CUDA_BACKEND)
    enable_language(CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
    add_executable(zkdl_gat_cuda_probe
        src/zkdl_gat/zkdl_backend_probe.cu
    )
    target_include_directories(zkdl_gat_cuda_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/zkdl
    )
    target_link_libraries(zkdl_gat_cuda_probe PRIVATE cudart)
    set_target_properties(zkdl_gat_cuda_probe PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()
