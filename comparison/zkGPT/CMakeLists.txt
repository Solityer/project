cmake_minimum_required(VERSION 3.10)
project(zkGPT NONE)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# The formal entry point and CMake targets must use the same baseline-local
# interpreter.  A system interpreter may configure the project, but it may not
# execute the source/preflight target or silently select another baseline's
# environment.
set(ZKGPT_LOCAL_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/ensure_environment.py --print-python
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE ZKGPT_ENV_RESULT
    OUTPUT_VARIABLE ZKGPT_ENV_OUTPUT
    ERROR_VARIABLE ZKGPT_ENV_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT ZKGPT_ENV_RESULT EQUAL 0)
    message(FATAL_ERROR "zkGPT baseline-local environment is unavailable: ${ZKGPT_ENV_ERROR}")
endif()
if(NOT ZKGPT_ENV_OUTPUT STREQUAL ZKGPT_LOCAL_PYTHON)
    message(FATAL_ERROR "CMake must use baseline-local Python ${ZKGPT_LOCAL_PYTHON}, got ${ZKGPT_ENV_OUTPUT}")
endif()
set(Python3_EXECUTABLE "${ZKGPT_LOCAL_PYTHON}" CACHE FILEPATH "zkGPT baseline-local Python" FORCE)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(NOT Python3_EXECUTABLE STREQUAL ZKGPT_LOCAL_PYTHON)
    message(FATAL_ERROR "CMake selected a non-local Python interpreter: ${Python3_EXECUTABLE}")
endif()

add_custom_target(
    gat_zkgpt_formal_preflight
    COMMAND ${ZKGPT_LOCAL_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/formal_preflight.py --source-only
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Source/configuration check for the native formal zkGPT route"
)

add_custom_target(
    zkgpt_cuda_native_prover
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cuda_backend.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building the CUDA proof-primitives backend"
)
