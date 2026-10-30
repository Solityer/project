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
  field_t block_accumulator = field_t::zero();
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
    } else if (opcode == 9) {
      accumulator = accumulator * constants[instruction[1]];
    } else if (opcode == 10) {
      block_accumulator = block_accumulator * *y + scratch[static_cast<size_t>(instruction[1]) * rows + row];
    } else if (opcode == 11) {
      accumulator = accumulator + block_accumulator * constants[instruction[1]];
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

__global__ void bn254_h_eval_mv_lookup_rows(const uint32_t* input, uint32_t* h_words, uint32_t* scratch_words, size_t rows, size_t instruction_count, size_t constant_count, size_t query_count, size_t scratch_slots, size_t graph_query_count, size_t lookup_count)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const uint32_t* instructions = input;
  const field_t* constants = reinterpret_cast<const field_t*>(input + instruction_count * 8);
  const field_t* y = constants + constant_count;
  const field_t* queries = y + 1;
  field_t* h = reinterpret_cast<field_t*>(h_words);
  field_t* scratch = reinterpret_cast<field_t*>(scratch_words);
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
    } else if (opcode == 12) {
      scratch[static_cast<size_t>(instruction[1]) * rows + row] = scratch[static_cast<size_t>(instruction[2]) * rows + row];
    } else if (opcode == 13) {
      const size_t output_start = instruction[1];
      const size_t input_count = instruction[2];
      const size_t lookup_index = instruction[3];
      field_t product = field_t::one();
      field_t products_except_sum = field_t::zero();
      for (size_t input_index = 0; input_index < input_count; ++input_index) {
        const field_t value = scratch[(output_start + input_index) * rows + row];
        products_except_sum = products_except_sum * value + product;
        product = product * value;
      }
      if (product == field_t::zero()) products_except_sum = field_t::zero();
      const field_t table_value = scratch[(output_start + input_count) * rows + row];
      const size_t auxiliary_query_base = graph_query_count + lookup_index * 3;
      const size_t shared_query_base = graph_query_count + lookup_count * 3;
      const field_t phi = field_t::from_montgomery(queries[auxiliary_query_base * rows + row]);
      const field_t phi_next = field_t::from_montgomery(queries[(auxiliary_query_base + 1) * rows + row]);
      const field_t multiplicity = field_t::from_montgomery(queries[(auxiliary_query_base + 2) * rows + row]);
      const field_t l0 = field_t::from_montgomery(queries[shared_query_base * rows + row]);
      const field_t l_last = field_t::from_montgomery(queries[(shared_query_base + 1) * rows + row]);
      const field_t l_active = field_t::from_montgomery(queries[(shared_query_base + 2) * rows + row]);
      const field_t lhs = table_value * product * (phi_next - phi);
      const field_t rhs = table_value * products_except_sum - product * multiplicity;
      accumulator = accumulator * *y + l0 * phi;
      accumulator = accumulator * *y + l_last * phi;
      accumulator = accumulator * *y + (lhs - rhs) * l_active;
    }
  }
  h[row] = field_t::to_montgomery(accumulator);
}

__global__ void bn254_h_permutation_scale_rows(uint32_t* h_words, size_t rows, field_t scale)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  field_t* h = reinterpret_cast<field_t*>(h_words);
  h[row] = field_t::to_montgomery(field_t::from_montgomery(h[row]) * scale);
}

__global__ void bn254_h_permutation_twiddle_rows(uint32_t* twiddle_words, size_t rows, size_t row_offset, field_t omega, size_t part_stride, size_t part_index)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const size_t exponent = (row + row_offset) * part_stride + part_index;
  field_t* twiddles = reinterpret_cast<field_t*>(twiddle_words);
  twiddles[row] = field_t::to_montgomery(field_t::pow(omega, static_cast<int>(exponent)));
}

__global__ void bn254_h_permutation_begin_set_rows(
    const uint32_t* z_words,
    const uint32_t* l0_words,
    uint32_t* h_words,
    uint32_t* left_words,
    uint32_t* right_words,
    size_t rows,
    size_t row_offset,
    size_t full_size,
    size_t part_index,
    size_t part_stride,
    int rot_scale,
    int isize,
    int last_rotation,
    size_t set_index,
    size_t set_count,
    field_t first_scale,
    field_t incoming_scale,
    field_t outgoing_scale)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const field_t* z = reinterpret_cast<const field_t*>(z_words);
  const field_t* l0 = reinterpret_cast<const field_t*>(l0_words);
  field_t* h = reinterpret_cast<field_t*>(h_words);
  field_t* left = reinterpret_cast<field_t*>(left_words);
  field_t* right = reinterpret_cast<field_t*>(right_words);
  const size_t full_row = row + row_offset;
  const size_t index = full_row * part_stride + part_index;
  int64_t next = static_cast<int64_t>(index) + static_cast<int64_t>(rot_scale);
  next %= static_cast<int64_t>(isize);
  if (next < 0) next += isize;
  int64_t last = static_cast<int64_t>(index) + static_cast<int64_t>(last_rotation) * rot_scale;
  last %= static_cast<int64_t>(isize);
  if (last < 0) last += isize;
  const field_t z_value = field_t::from_montgomery(z[full_row]);
  const field_t z_next = field_t::from_montgomery(z[static_cast<size_t>(next) / part_stride]);
  const field_t l0_value = field_t::from_montgomery(l0[row]);
  field_t contribution = field_t::zero();
  if (set_index == 0) contribution = contribution + (field_t::one() - z_value) * l0_value * first_scale;
  if (set_index != 0) contribution = contribution + z_value * l0_value * incoming_scale;
  if (set_index + 1 < set_count) {
    contribution = contribution - field_t::from_montgomery(z[static_cast<size_t>(last) / part_stride]) * l0_value * outgoing_scale;
  }
  h[row] = field_t::to_montgomery(field_t::from_montgomery(h[row]) + contribution);
  left[row] = field_t::to_montgomery(z_next);
  right[row] = field_t::to_montgomery(z_value);
  (void)full_size;
}

__global__ void bn254_h_permutation_last_boundary_rows(
    const uint32_t* l_last_words,
    uint32_t* h_words,
    const uint32_t* right_words,
    size_t rows,
    field_t last_scale)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const field_t* l_last = reinterpret_cast<const field_t*>(l_last_words);
  const field_t* right = reinterpret_cast<const field_t*>(right_words);
  field_t* h = reinterpret_cast<field_t*>(h_words);
  const field_t z = field_t::from_montgomery(right[row]);
  const field_t contribution = (z * z - z) * field_t::from_montgomery(l_last[row]) * last_scale;
  h[row] = field_t::to_montgomery(field_t::from_montgomery(h[row]) + contribution);
}

__global__ void bn254_h_permutation_column_rows(
    const uint32_t* source_words,
    const uint32_t* sigma_words,
    const uint32_t* row_twiddle_words,
    uint32_t* left_words,
    uint32_t* right_words,
    size_t rows,
    field_t beta,
    field_t gamma,
    field_t delta_start)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const field_t* source = reinterpret_cast<const field_t*>(source_words);
  const field_t* sigma = reinterpret_cast<const field_t*>(sigma_words);
  const field_t* row_twiddle = reinterpret_cast<const field_t*>(row_twiddle_words);
  field_t* left = reinterpret_cast<field_t*>(left_words);
  field_t* right = reinterpret_cast<field_t*>(right_words);
  const field_t value = field_t::from_montgomery(source[row]);
  const field_t sigma_value = field_t::from_montgomery(sigma[row]);
  const field_t beta_term = delta_start * field_t::from_montgomery(row_twiddle[row]);
  left[row] = field_t::to_montgomery(field_t::from_montgomery(left[row]) * (value + beta * sigma_value + gamma));
  right[row] = field_t::to_montgomery(field_t::from_montgomery(right[row]) * (value + beta_term + gamma));
}

__global__ void bn254_h_permutation_copy_rows(
    const uint32_t* l_active_words,
    const uint32_t* left_words,
    const uint32_t* right_words,
    uint32_t* h_words,
    size_t rows,
    field_t scale)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const field_t* l_active = reinterpret_cast<const field_t*>(l_active_words);
  const field_t* left = reinterpret_cast<const field_t*>(left_words);
  const field_t* right = reinterpret_cast<const field_t*>(right_words);
  field_t* h = reinterpret_cast<field_t*>(h_words);
  const field_t contribution = (field_t::from_montgomery(left[row]) - field_t::from_montgomery(right[row])) * field_t::from_montgomery(l_active[row]) * scale;
  h[row] = field_t::to_montgomery(field_t::from_montgomery(h[row]) + contribution);
}

__global__ void bn254_h_coset_scale_rows(uint32_t* input_words, uint32_t* powers_words, size_t row_offset, size_t rows, field_t factor, bool generate_powers)
{
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const size_t index = row_offset + row;
  field_t* input = reinterpret_cast<field_t*>(input_words);
  field_t* powers = reinterpret_cast<field_t*>(powers_words);
  const field_t power = generate_powers
    ? field_t::pow(field_t::from_montgomery(factor), static_cast<int>(index))
    : field_t::from_montgomery(powers[index]);
  if (generate_powers) powers[index] = field_t::to_montgomery(power);
  input[index] = field_t::to_montgomery(field_t::from_montgomery(input[index]) * power);
}

extern "C" cudaError_t bn254_h_eval_mv_lookup_program(const uint32_t* input, uint32_t* h, uint32_t* scratch, size_t rows, size_t instruction_count, size_t constant_count, size_t query_count, size_t scratch_slots, size_t graph_query_count, size_t lookup_count)
{
  if (input == nullptr || h == nullptr || scratch == nullptr || rows == 0 || scratch_slots == 0 || lookup_count == 0) return cudaErrorInvalidValue;
  if (instruction_count > SIZE_MAX / 8 || constant_count > SIZE_MAX / 8 || query_count > SIZE_MAX / rows || scratch_slots > SIZE_MAX / rows) return cudaErrorInvalidValue;
  if (graph_query_count > query_count || lookup_count > (query_count - graph_query_count) / 3 || query_count != graph_query_count + lookup_count * 3 + 3) return cudaErrorInvalidValue;
  bn254_h_eval_mv_lookup_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(input, h, scratch, rows, instruction_count, constant_count, query_count, scratch_slots, graph_query_count, lookup_count);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_permutation_scale(const uint32_t* scale_words, uint32_t* h, size_t rows)
{
  if (scale_words == nullptr || h == nullptr || rows == 0) return cudaErrorInvalidValue;
  const field_t scale = *reinterpret_cast<const field_t*>(scale_words);
  bn254_h_permutation_scale_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(h, rows, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_permutation_twiddles(uint32_t* twiddles, size_t rows, size_t row_offset, const uint32_t* omega_words, size_t part_stride, size_t part_index)
{
  if (twiddles == nullptr || omega_words == nullptr || rows == 0 || part_stride == 0 || part_index >= part_stride || row_offset > SIZE_MAX / part_stride || rows > SIZE_MAX / part_stride - row_offset) return cudaErrorInvalidValue;
  const field_t omega = *reinterpret_cast<const field_t*>(omega_words);
  bn254_h_permutation_twiddle_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(twiddles, rows, row_offset, omega, part_stride, part_index);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_permutation_begin_set(const uint32_t* z, const uint32_t* l0, uint32_t* h, uint32_t* left, uint32_t* right, size_t rows, size_t row_offset, size_t full_size, size_t part_index, size_t part_stride, int rot_scale, int isize, int last_rotation, size_t set_index, size_t set_count, const uint32_t* scales)
{
  if (z == nullptr || l0 == nullptr || h == nullptr || left == nullptr || right == nullptr || scales == nullptr || rows == 0 || full_size == 0 || part_stride == 0 || full_size % part_stride != 0 || part_index >= part_stride || set_count == 0 || set_index >= set_count || rot_scale <= 0 || isize <= 0 || static_cast<size_t>(isize) != full_size || row_offset > full_size / part_stride || rows > full_size / part_stride - row_offset) return cudaErrorInvalidValue;
  const field_t* values = reinterpret_cast<const field_t*>(scales);
  bn254_h_permutation_begin_set_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(z, l0, h, left, right, rows, row_offset, full_size, part_index, part_stride, rot_scale, isize, last_rotation, set_index, set_count, values[0], values[1], values[2]);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_permutation_last_boundary(const uint32_t* l_last, uint32_t* h, const uint32_t* right, size_t rows, const uint32_t* scale_words)
{
  if (l_last == nullptr || h == nullptr || right == nullptr || scale_words == nullptr || rows == 0) return cudaErrorInvalidValue;
  const field_t scale = *reinterpret_cast<const field_t*>(scale_words);
  bn254_h_permutation_last_boundary_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(l_last, h, right, rows, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_permutation_column(const uint32_t* source, const uint32_t* sigma, const uint32_t* row_twiddles, uint32_t* left, uint32_t* right, size_t rows, const uint32_t* scalars)
{
  if (source == nullptr || sigma == nullptr || row_twiddles == nullptr || left == nullptr || right == nullptr || scalars == nullptr || rows == 0) return cudaErrorInvalidValue;
  const field_t* values = reinterpret_cast<const field_t*>(scalars);
  bn254_h_permutation_column_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(source, sigma, row_twiddles, left, right, rows, values[0], values[1], values[2]);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_permutation_copy_term(const uint32_t* l_active, const uint32_t* left, const uint32_t* right, uint32_t* h, size_t rows, const uint32_t* scale_words)
{
  if (l_active == nullptr || left == nullptr || right == nullptr || h == nullptr || scale_words == nullptr || rows == 0) return cudaErrorInvalidValue;
  const field_t scale = *reinterpret_cast<const field_t*>(scale_words);
  bn254_h_permutation_copy_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(l_active, left, right, h, rows, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_coset_scale(uint32_t* input_words, uint32_t* powers_words, size_t row_offset, size_t rows, const uint32_t* factor_words, bool generate_powers)
{
  constexpr size_t max_coset_rows = size_t{1} << 25;
  if (input_words == nullptr || powers_words == nullptr || factor_words == nullptr || rows == 0 || rows > max_coset_rows || row_offset > max_coset_rows || rows > max_coset_rows - row_offset) return cudaErrorInvalidValue;
  const field_t factor = *reinterpret_cast<const field_t*>(factor_words);
  bn254_h_coset_scale_rows<<<static_cast<unsigned>((rows + 127) / 128), 128, 0, nullptr>>>(input_words, powers_words, row_offset, rows, factor, generate_powers);
  return cudaGetLastError();
}

extern "C" cudaError_t bn254_h_eval_kernel_attributes(size_t* local_bytes_per_thread, size_t* shared_bytes_per_block, size_t* stack_bytes_per_thread, int* registers_per_thread, int* max_threads_per_block)
{
  if (local_bytes_per_thread == nullptr || shared_bytes_per_block == nullptr || stack_bytes_per_thread == nullptr || registers_per_thread == nullptr || max_threads_per_block == nullptr) return cudaErrorInvalidValue;
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, bn254_h_eval_rows);
  if (status != cudaSuccess) return status;
  cudaFuncAttributes lookup_attributes{};
  status = cudaFuncGetAttributes(&lookup_attributes, bn254_h_eval_mv_lookup_rows);
  if (status != cudaSuccess) return status;
  cudaFuncAttributes permutation_attributes[6]{};
  status = cudaFuncGetAttributes(&permutation_attributes[0], bn254_h_permutation_scale_rows);
  if (status != cudaSuccess) return status;
  status = cudaFuncGetAttributes(&permutation_attributes[1], bn254_h_permutation_twiddle_rows);
  if (status != cudaSuccess) return status;
  status = cudaFuncGetAttributes(&permutation_attributes[2], bn254_h_permutation_begin_set_rows);
  if (status != cudaSuccess) return status;
  status = cudaFuncGetAttributes(&permutation_attributes[3], bn254_h_permutation_last_boundary_rows);
  if (status != cudaSuccess) return status;
  status = cudaFuncGetAttributes(&permutation_attributes[4], bn254_h_permutation_column_rows);
  if (status != cudaSuccess) return status;
  status = cudaFuncGetAttributes(&permutation_attributes[5], bn254_h_permutation_copy_rows);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetLimit(stack_bytes_per_thread, cudaLimitStackSize);
  if (status != cudaSuccess) return status;
  *local_bytes_per_thread = attributes.localSizeBytes > lookup_attributes.localSizeBytes ? attributes.localSizeBytes : lookup_attributes.localSizeBytes;
  *shared_bytes_per_block = attributes.sharedSizeBytes > lookup_attributes.sharedSizeBytes ? attributes.sharedSizeBytes : lookup_attributes.sharedSizeBytes;
  *registers_per_thread = attributes.numRegs > lookup_attributes.numRegs ? attributes.numRegs : lookup_attributes.numRegs;
  *max_threads_per_block = attributes.maxThreadsPerBlock < lookup_attributes.maxThreadsPerBlock ? attributes.maxThreadsPerBlock : lookup_attributes.maxThreadsPerBlock;
  for (const auto& permutation_attribute : permutation_attributes) {
    if (permutation_attribute.localSizeBytes > *local_bytes_per_thread) *local_bytes_per_thread = permutation_attribute.localSizeBytes;
    if (permutation_attribute.sharedSizeBytes > *shared_bytes_per_block) *shared_bytes_per_block = permutation_attribute.sharedSizeBytes;
    if (permutation_attribute.numRegs > *registers_per_thread) *registers_per_thread = permutation_attribute.numRegs;
    if (permutation_attribute.maxThreadsPerBlock < *max_threads_per_block) *max_threads_per_block = permutation_attribute.maxThreadsPerBlock;
  }
  cudaFuncAttributes coset_attributes{};
  status = cudaFuncGetAttributes(&coset_attributes, bn254_h_coset_scale_rows);
  if (status != cudaSuccess) return status;
  if (coset_attributes.localSizeBytes > *local_bytes_per_thread) *local_bytes_per_thread = coset_attributes.localSizeBytes;
  if (coset_attributes.sharedSizeBytes > *shared_bytes_per_block) *shared_bytes_per_block = coset_attributes.sharedSizeBytes;
  if (coset_attributes.numRegs > *registers_per_thread) *registers_per_thread = coset_attributes.numRegs;
  if (coset_attributes.maxThreadsPerBlock < *max_threads_per_block) *max_threads_per_block = coset_attributes.maxThreadsPerBlock;
  return cudaSuccess;
}
