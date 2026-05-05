cmake_minimum_required(VERSION 3.18)

project(zkdl_gat_full_forward_baseline LANGUAGES CXX)

option(ZKDL_GAT_ENABLE_CUDA "Build the zkDL CUDA primitive probe" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(zkdl_gat_core
  src/zkdl_gat/baseline.cpp
)
target_include_directories(zkdl_gat_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(zkdl_gat_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(zkdl_gat_runner src/main.cpp)
target_link_libraries(zkdl_gat_runner PRIVATE zkdl_gat_core)

if(ZKDL_GAT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  add_executable(zkdl_gat_gpu_probe src/zkdl_gat_gpu_probe.cu)
  target_include_directories(zkdl_gat_gpu_probe
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries(zkdl_gat_gpu_probe PRIVATE CUDA::cudart CUDA::curand)
  set_target_properties(zkdl_gat_gpu_probe PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()
