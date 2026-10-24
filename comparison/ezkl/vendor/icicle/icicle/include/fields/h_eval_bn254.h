#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

extern "C" cudaError_t bn254_h_eval_program(const uint32_t* input, uint32_t* output, size_t rows, size_t instruction_count, size_t constant_count, size_t query_count, size_t scratch_slots);
extern "C" cudaError_t bn254_h_eval_kernel_attributes(size_t* local_bytes_per_thread, size_t* shared_bytes_per_block, size_t* stack_bytes_per_thread, int* registers_per_thread, int* max_threads_per_block);
