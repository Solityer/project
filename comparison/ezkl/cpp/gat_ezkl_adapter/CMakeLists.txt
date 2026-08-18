cmake_minimum_required(VERSION 3.18)
project(gat_ezkl_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(gat_ezkl_export
  src/checkpoint.cpp
  src/config.cpp
  src/dataset.cpp
  src/formal_csr.cpp
  src/formal_gat.cpp
  src/formal_hash.cpp
  src/formal_io.cpp
  src/graph.cpp
  src/main.cpp
)

target_include_directories(gat_ezkl_export PRIVATE include)
target_compile_options(gat_ezkl_export PRIVATE -Wall -Wextra -Wpedantic)
