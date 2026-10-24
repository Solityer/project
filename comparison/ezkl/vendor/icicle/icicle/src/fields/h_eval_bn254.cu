#include "fields/field_config.cuh"
#include "fields/h_eval_bn254.h"

#ifndef CURVE_ID
#error "BN254 H evaluator requires CURVE_ID"
#endif
#if CURVE_ID != 1
#error "BN254 H evaluator may only be linked into the BN254 curve library"
#endif

using field_t = field_config::scalar_t;

static_assert(sizeof(field_t) == 32, "BN254 H evaluator expects 32-byte field elements");
static_assert(32 % alignof(field_t) == 0, "BN254 H evaluator arena offsets must preserve field alignment");
static_assert(field_t::TLC == 8 && field_t::NBITS == 254, "BN254 H evaluator requires the BN254 scalar field");
static_assert(field_config::fp_config::modulus.limbs[0] == 0xf0000001 && field_config::fp_config::modulus.limbs[1] == 0x43e1f593 && field_config::fp_config::modulus.limbs[2] == 0x79b97091 && field_config::fp_config::modulus.limbs[3] == 0x2833e848 && field_config::fp_config::modulus.limbs[4] == 0x8181585d && field_config::fp_config::modulus.limbs[5] == 0xb85045b6 && field_config::fp_config::modulus.limbs[6] == 0xe131a029 && field_config::fp_config::modulus.limbs[7] == 0x30644e72, "BN254 modulus mismatch");
static_assert(field_config::fp_config::montgomery_r.limbs[0] == 0x4ffffffb && field_config::fp_config::montgomery_r.limbs[1] == 0xac96341c && field_config::fp_config::montgomery_r.limbs[2] == 0x9f60cd29 && field_config::fp_config::montgomery_r.limbs[3] == 0x36fc7695 && field_config::fp_config::montgomery_r.limbs[4] == 0x7879462e && field_config::fp_config::montgomery_r.limbs[5] == 0x666ea36f && field_config::fp_config::montgomery_r.limbs[6] == 0x9a07df2f && field_config::fp_config::montgomery_r.limbs[7] == 0x0e0a77c1, "BN254 Montgomery R mismatch");

__global__ void bn254_h_eval_rows(const uint32_t* input, uint32_t* output, size_t rows, size_t instruction_count, size_t constant_count, size_t query_count, size_t scratch_slots)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const uint32_t* instructions = input;
  const field_t* constants = reinterpret_cast<const field_t*>(input + instruction_count * 8);
  const field_t* y = constants + constant_count;
  const field_t* queries = y + 1;
  field_t* h = reinterpret_cast<field_t*>(output);
  field_t* scratch = h + rows;
  field_t accumulator = field_t::from_montgomery(h[row]);
  for (size_t instruction_index = 0; instruction_index < instruction_count; ++instruction_index) {
    const uint32_t* instruction = instructions + instruction_index * 8;
    const uint32_t opcode = instruction[0];
    if (opcode == 0) {
      const size_t destination = instruction[1];
      const size_t query_index = instruction[2];
      scratch[destination * rows + row] = field_t::from_montgomery(queries[query_index * rows + row]);
    } else if (opcode == 1) {
      scratch[static_cast<size_t>(instruction[1]) * rows + row] = constants[instruction[2]];
    } else if (opcode == 2) {
      scratch[static_cast<size_t>(instruction[1]) * rows + row] = scratch[static_cast<size_t>(instruction[2]) * rows + row] + scratch[static_cast<size_t>(instruction[3]) * rows + row];
    } else if (opcode == 3) {
      scratch[static_cast<size_t>(instruction[1]) * rows + row] = scratch[static_cast<size_t>(instruction[2]) * rows + row] - scratch[static_cast<size_t>(instruction[3]) * rows + row];
    } else if (opcode == 4) {
      scratch[static_cast<size_t>(instruction[1]) * rows + row] = scratch[static_cast<size_t>(instruction[2]) * rows + row] * scratch[static_cast<size_t>(instruction[3]) * rows + row];
    } else if (opcode == 5) {
      const field_t value = scratch[static_cast<size_t>(instruction[2]) * rows + row];
      scratch[static_cast<size_t>(instruction[1]) * rows + row] = value * value;
    } else if (opcode == 6) {
      const field_t value = scratch[static_cast<size_t>(instruction[2]) * rows + row];
      scratch[static_cast<size_t>(instruction[1]) * rows + row] = value + value;
    } else if (opcode == 7) {
      scratch[static_cast<size_t>(instruction[1]) * rows + row] = field_t::neg(scratch[static_cast<size_t>(instruction[2]) * rows + row]);
    } else if (opcode == 8) {
      accumulator = accumulator * *y + scratch[static_cast<size_t>(instruction[1]) * rows + row];
    }
  }
  h[row] = field_t::to_montgomery(accumulator);
}

extern "C" cudaError_t bn254_h_eval_program(const uint32_t* input, uint32_t* output, size_t rows, size_t instruction_count, size_t constant_count, size_t query_count, size_t scratch_slots)
{
  if (input == nullptr || output == nullptr || rows == 0 || scratch_slots == 0) return cudaErrorInvalidValue;
  if (instruction_count > SIZE_MAX / 8 || constant_count > SIZE_MAX / 8 || query_count > SIZE_MAX / rows || scratch_slots > SIZE_MAX / rows) return cudaErrorInvalidValue;
  bn254_h_eval_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(input, output, rows, instruction_count, constant_count, query_count, scratch_slots);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_eval_kernel_attributes(size_t* local_bytes_per_thread, size_t* shared_bytes_per_block, size_t* stack_bytes_per_thread, int* registers_per_thread, int* max_threads_per_block)
{
  if (local_bytes_per_thread == nullptr || shared_bytes_per_block == nullptr || stack_bytes_per_thread == nullptr || registers_per_thread == nullptr || max_threads_per_block == nullptr) return cudaErrorInvalidValue;
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, bn254_h_eval_rows);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetLimit(stack_bytes_per_thread, cudaLimitStackSize);
  if (status != cudaSuccess) return status;
  *local_bytes_per_thread = attributes.localSizeBytes;
  *shared_bytes_per_block = attributes.sharedSizeBytes;
  *registers_per_thread = attributes.numRegs;
  *max_threads_per_block = attributes.maxThreadsPerBlock;
  return cudaSuccess;
}
