#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

extern "C" cudaError_t bn254_h_eval_program(const uint32_t* input, uint32_t* output, size_t rows, size_t instruction_count, size_t constant_count, size_t query_count, size_t scratch_slots);
extern "C" cudaError_t bn254_h_eval_mv_lookup_program(const uint32_t* input, uint32_t* h, uint32_t* scratch, size_t rows, size_t instruction_count, size_t constant_count, size_t query_count, size_t scratch_slots, size_t graph_query_count, size_t lookup_count);
extern "C" cudaError_t bn254_h_permutation_scale(const uint32_t* scale_words, uint32_t* h, size_t rows);
extern "C" cudaError_t bn254_h_permutation_twiddles(uint32_t* twiddles, size_t rows, size_t row_offset, const uint32_t* omega_words, size_t part_stride, size_t part_index);
extern "C" cudaError_t bn254_h_permutation_begin_set(const uint32_t* z, const uint32_t* l0, uint32_t* h, uint32_t* left, uint32_t* right, size_t rows, size_t row_offset, size_t full_size, size_t part_index, size_t part_stride, int rot_scale, int isize, int last_rotation, size_t set_index, size_t set_count, const uint32_t* scales);
extern "C" cudaError_t bn254_h_permutation_last_boundary(const uint32_t* l_last, uint32_t* h, const uint32_t* right, size_t rows, const uint32_t* scale_words);
extern "C" cudaError_t bn254_h_permutation_column(const uint32_t* source, const uint32_t* sigma, const uint32_t* row_twiddles, uint32_t* left, uint32_t* right, size_t rows, const uint32_t* scalars);
extern "C" cudaError_t bn254_h_permutation_copy_term(const uint32_t* l_active, const uint32_t* left, const uint32_t* right, uint32_t* h, size_t rows, const uint32_t* scale_words);
extern "C" cudaError_t bn254_h_coset_scale(uint32_t* input_words, uint32_t* powers_words, size_t row_offset, size_t rows, const uint32_t* factor_words, bool generate_powers);
extern "C" cudaError_t bn254_h_eval_kernel_attributes(size_t* local_bytes_per_thread, size_t* shared_bytes_per_block, size_t* stack_bytes_per_thread, int* registers_per_thread, int* max_threads_per_block);
