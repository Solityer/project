cmake_minimum_required(VERSION 3.16)

project(gat_baseline_stage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(
  gat_baseline_stage
  src/main.cpp
  src/formal_config.cpp
)
target_include_directories(gat_baseline_stage PRIVATE include)
target_compile_options(gat_baseline_stage PRIVATE -Wall -Wextra -Wpedantic)
