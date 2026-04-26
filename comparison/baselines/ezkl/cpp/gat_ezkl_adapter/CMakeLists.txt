cmake_minimum_required(VERSION 3.18)
project(gat_ezkl_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gat_ezkl_export
  src/bounds.cpp
  src/checkpoint.cpp
  src/config.cpp
  src/dataset.cpp
  src/gat.cpp
  src/graph.cpp
  src/hash.cpp
  src/main.cpp
  src/metadata.cpp
  src/onnx_export.cpp
  src/quant.cpp
  src/segment_softmax.cpp
)

target_include_directories(gat_ezkl_export PRIVATE include)
target_compile_options(gat_ezkl_export PRIVATE -Wall -Wextra -Wpedantic)
