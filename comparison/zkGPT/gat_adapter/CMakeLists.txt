add_library(gat_adapter STATIC
    gat_parser.cpp
    gat_forward.cpp
    gat_native.cpp
    gat_input_oracle.cpp
    gat_sparse_gkr.cpp
    gat_proof.cpp
    gat_cuda.hpp
)

if(ZKGPT_ENABLE_CUDA)
    add_library(gat_cuda STATIC gat_cuda.cu)
    set_target_properties(gat_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(gat_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/3rd/mcl/include
    )
    target_link_libraries(gat_cuda PUBLIC mcl mclbn256)
    target_compile_definitions(gat_adapter PUBLIC ZKGPT_HAS_CUDA=1)
    target_link_libraries(gat_adapter PUBLIC gat_cuda)
else()
    target_sources(gat_adapter PRIVATE gat_cuda_stub.cpp)
    target_compile_definitions(gat_adapter PUBLIC ZKGPT_HAS_CUDA=0)
endif()

target_include_directories(gat_adapter PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/3rd/mcl/include
)
target_link_libraries(gat_adapter PUBLIC gpt_lib mcl mclbn256)

# Single production entry point for one formal cell.  The executable is
# linked against the official zkGPT library and this adapter; all orchestral
# matrix/collection logic lives in scripts/ and never replaces the native
# prover/verifier path.
add_executable(zkgpt_gat_prover gat_cli.cpp)
target_link_libraries(zkgpt_gat_prover PRIVATE gat_adapter)

# Independent verifier process.  It has a deliberately smaller input surface
# than the prover and never links a checkpoint/model loader into its CLI
# entrypoint; the proof library remains the sole native verification boundary.
add_executable(zkgpt_gat_verifier gat_verifier_cli.cpp)
target_link_libraries(zkgpt_gat_verifier PRIVATE gat_adapter)
