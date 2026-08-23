#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../core/bls12-381.cuh"

// Native CUDA provider for the Rust formal prover.  The C ABI deliberately
// consumes the canonical Rust proof backend's BLS12-381 scalar encoding and
// blst_p1 Jacobian layout: it accelerates the same commitments and proof, not
// a second GPU-only proof format.

namespace {

using Fr = blstrs__scalar__Scalar;
using G1 = blstrs__g1__G1Affine_jacobian;

static_assert(sizeof(Fr) == 32, "unexpected BLS12-381 scalar layout");
static_assert(sizeof(G1) == 144, "CUDA G1 layout must match blst_p1");

thread_local std::string last_error;

int fail(const char* where, cudaError_t error) {
    last_error = std::string(where) + ": " + cudaGetErrorString(error);
    return -1;
}

int check_launch(const char* where) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return fail(where, error);
    error = cudaDeviceSynchronize();
    return error == cudaSuccess ? 0 : fail(where, error);
}

Fr raw_from_be(const std::uint8_t* bytes) {
    Fr value{};
    for (unsigned limb = 0; limb < 8; ++limb) {
        const unsigned offset = 32U - 4U * (limb + 1U);
        value.val[limb] = (std::uint32_t(bytes[offset]) << 24U)
            | (std::uint32_t(bytes[offset + 1U]) << 16U)
            | (std::uint32_t(bytes[offset + 2U]) << 8U)
            | std::uint32_t(bytes[offset + 3U]);
    }
    return value;
}

void raw_to_be(const Fr& value, std::uint8_t* bytes) {
    for (unsigned limb = 0; limb < 8; ++limb) {
        const unsigned offset = 32U - 4U * (limb + 1U);
        const std::uint32_t word = value.val[limb];
        bytes[offset] = std::uint8_t(word >> 24U);
        bytes[offset + 1U] = std::uint8_t(word >> 16U);
        bytes[offset + 2U] = std::uint8_t(word >> 8U);
        bytes[offset + 3U] = std::uint8_t(word);
    }
}

std::vector<Fr> decode_raw(const std::uint8_t* input, std::size_t count) {
    std::vector<Fr> out(count);
    for (std::size_t i = 0; i < count; ++i) out[i] = raw_from_be(input + 32U * i);
    return out;
}

int encode_montgomery(Fr* device, std::size_t count, std::uint8_t* output);

__device__ G1 g1_mul_raw(G1 base, Fr scalar) {
    G1 out = blstrs__g1__G1Affine_ZERO;
    #pragma unroll
    for (unsigned bit = 0; bit < 256; ++bit) {
        if ((scalar.val[bit / 32U] >> (bit % 32U)) & 1U) {
            out = blstrs__g1__G1Affine_add(out, base);
        }
        base = blstrs__g1__G1Affine_double(base);
    }
    return out;
}

__global__ void to_montgomery_kernel(Fr* values, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) values[i] = blstrs__scalar__Scalar_mont(values[i]);
}

__global__ void from_montgomery_kernel(Fr* values, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) values[i] = blstrs__scalar__Scalar_unmont(values[i]);
}

__global__ void inner_product_terms_kernel(
    const Fr* lhs, const Fr* rhs, Fr* terms, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) terms[i] = blstrs__scalar__Scalar_mul(lhs[i], rhs[i]);
}

__global__ void fr_reduce_kernel(const Fr* input, std::size_t count, Fr* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Fr acc = blstrs__scalar__Scalar_ZERO;
    for (std::size_t i = 0; i < count; ++i) acc = blstrs__scalar__Scalar_add(acc, input[i]);
    output[0] = acc;
}

__global__ void scalar_fold_split_kernel(
    const Fr* input, Fr* output, std::size_t half, Fr left_coefficient, Fr right_coefficient) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < half) {
        output[i] = blstrs__scalar__Scalar_add(
            blstrs__scalar__Scalar_mul(input[i], left_coefficient),
            blstrs__scalar__Scalar_mul(input[i + half], right_coefficient));
    }
}

__global__ void point_fold_split_kernel(
    const G1* input, G1* output, std::size_t half, Fr left_raw, Fr right_raw) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < half) {
        output[i] = blstrs__g1__G1Affine_add(
            g1_mul_raw(input[i], left_raw), g1_mul_raw(input[i + half], right_raw));
    }
}

__global__ void msm_terms_kernel(const G1* points, const Fr* scalars_raw, G1* terms, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) terms[i] = g1_mul_raw(points[i], scalars_raw[i]);
}

__global__ void g1_reduce_kernel(const G1* input, std::size_t count, G1* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    G1 acc = blstrs__g1__G1Affine_ZERO;
    for (std::size_t i = 0; i < count; ++i) acc = blstrs__g1__G1Affine_add(acc, input[i]);
    output[0] = acc;
}

enum : std::uint32_t { EXPR_CONST = 0, EXPR_COLUMN = 1, EXPR_ADD = 2, EXPR_SUB = 3, EXPR_MUL = 4, EXPR_PUBLIC_TABLE = 5 };

struct DeviceExprToken {
    std::uint32_t opcode;
    std::uint32_t argument;
    Fr constant;
};

__global__ void sumcheck_contributions_kernel(
    const Fr* columns,
    std::size_t column_count,
    std::size_t row_count,
    const Fr* beta,
    const Fr* mask,
    bool first_round,
    Fr factor,
    const Fr* lambdas,
    std::size_t residual_count,
    const DeviceExprToken* tokens,
    const std::uint32_t* residual_offsets,
    const Fr* lookup_values,
    const std::uint32_t* lookup_value_offsets,
    const std::uint32_t* lookup_bit_columns,
    const std::uint32_t* lookup_bit_offsets,
    std::size_t lookup_count,
    std::size_t half,
    std::size_t evaluation_count,
    Fr* contributions,
    std::uint32_t* device_error) {
    const std::size_t linear = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = half * evaluation_count;
    if (linear >= total) return;
    const std::size_t evaluation = linear / half;
    const std::size_t row = linear % half;
    const Fr x = blstrs__scalar__Scalar_mont((Fr){{std::uint32_t(evaluation), 0, 0, 0, 0, 0, 0, 0}});
    const Fr one_minus_x = blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ONE, x);
    Fr compressed = blstrs__scalar__Scalar_ZERO;
    for (std::size_t residual = 0; residual < residual_count; ++residual) {
        Fr stack[64];
        unsigned stack_size = 0;
        for (std::uint32_t pc = residual_offsets[residual]; pc < residual_offsets[residual + 1U]; ++pc) {
            const DeviceExprToken token = tokens[pc];
            if (token.opcode == EXPR_CONST) {
                if (stack_size >= 64U) { atomicExch(device_error, 1U); return; }
                stack[stack_size++] = token.constant;
            } else if (token.opcode == EXPR_COLUMN) {
                if (token.argument >= column_count || stack_size >= 64U) { atomicExch(device_error, 2U); return; }
                const Fr low = columns[std::size_t(token.argument) * row_count + row];
                const Fr high = columns[std::size_t(token.argument) * row_count + row + half];
                stack[stack_size++] = blstrs__scalar__Scalar_add(
                    blstrs__scalar__Scalar_mul(low, one_minus_x),
                    blstrs__scalar__Scalar_mul(high, x));
            } else if (token.opcode == EXPR_PUBLIC_TABLE) {
                if (token.argument >= lookup_count || stack_size >= 64U) { atomicExch(device_error, 6U); return; }
                const std::uint32_t value_begin=lookup_value_offsets[token.argument],value_end=lookup_value_offsets[token.argument+1U];
                const std::uint32_t bit_begin=lookup_bit_offsets[token.argument],bit_end=lookup_bit_offsets[token.argument+1U];
                const std::uint32_t bit_count=bit_end-bit_begin;
                if (value_end<=value_begin || (std::size_t(1)<<bit_count)!=(value_end-value_begin)) { atomicExch(device_error,7U);return; }
                Fr table_value=blstrs__scalar__Scalar_ZERO;
                for(std::uint32_t table_row=0;table_row<value_end-value_begin;++table_row){
                    Fr weight=blstrs__scalar__Scalar_ONE;
                    for(std::uint32_t bit_position=0;bit_position<bit_count;++bit_position){
                        const std::uint32_t column=lookup_bit_columns[bit_begin+bit_position];
                        if(column>=column_count){atomicExch(device_error,8U);return;}
                        const Fr low=columns[std::size_t(column)*row_count+row];const Fr high=columns[std::size_t(column)*row_count+row+half];
                        const Fr bit=blstrs__scalar__Scalar_add(blstrs__scalar__Scalar_mul(low,one_minus_x),blstrs__scalar__Scalar_mul(high,x));
                        // The canonical adapter emits bit_0 first.
                        const bool selected=((table_row>>bit_position)&1U)!=0;
                        weight=blstrs__scalar__Scalar_mul(weight,selected?bit:blstrs__scalar__Scalar_sub(blstrs__scalar__Scalar_ONE,bit));
                    }
                    table_value=blstrs__scalar__Scalar_add(table_value,blstrs__scalar__Scalar_mul(lookup_values[value_begin+table_row],weight));
                }
                stack[stack_size++]=table_value;
            } else {
                if (stack_size < 2U) { atomicExch(device_error, 3U); return; }
                const Fr rhs = stack[--stack_size];
                const Fr lhs = stack[--stack_size];
                Fr value;
                if (token.opcode == EXPR_ADD) value = blstrs__scalar__Scalar_add(lhs, rhs);
                else if (token.opcode == EXPR_SUB) value = blstrs__scalar__Scalar_sub(lhs, rhs);
                else if (token.opcode == EXPR_MUL) value = blstrs__scalar__Scalar_mul(lhs, rhs);
                else { atomicExch(device_error, 4U); return; }
                stack[stack_size++] = value;
            }
        }
        if (stack_size != 1U) { atomicExch(device_error, 5U); return; }
        compressed = blstrs__scalar__Scalar_add(
            compressed, blstrs__scalar__Scalar_mul(stack[0], lambdas[residual]));
    }
    const Fr eb = blstrs__scalar__Scalar_add(
        blstrs__scalar__Scalar_mul(beta[row], one_minus_x),
        blstrs__scalar__Scalar_mul(beta[row + half], x));
    Fr value = blstrs__scalar__Scalar_mul(eb, compressed);
    Fr mask_term;
    if (first_round) {
        mask_term = blstrs__scalar__Scalar_mul(
            blstrs__scalar__Scalar_mul(mask[row], x), one_minus_x);
    } else {
        const Fr folded_mask = blstrs__scalar__Scalar_add(
            blstrs__scalar__Scalar_mul(mask[row], one_minus_x),
            blstrs__scalar__Scalar_mul(mask[row + half], x));
        mask_term = blstrs__scalar__Scalar_mul(folded_mask, factor);
    }
    contributions[linear] = blstrs__scalar__Scalar_add(value, mask_term);
}

__global__ void sumcheck_reduce_kernel(
    const Fr* contributions, std::size_t half, std::size_t evaluation_count, Fr* output) {
    const std::size_t evaluation = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (evaluation >= evaluation_count) return;
    Fr acc = blstrs__scalar__Scalar_ZERO;
    const Fr* row = contributions + evaluation * half;
    for (std::size_t i = 0; i < half; ++i) acc = blstrs__scalar__Scalar_add(acc, row[i]);
    output[evaluation] = acc;
}

template<class T>
int allocate(T** pointer, std::size_t count, const char* label) {
    const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(pointer), sizeof(T) * count);
    return error == cudaSuccess ? 0 : fail(label, error);
}

template<class T>
int copy_to_device(T* device, const T* host, std::size_t count, const char* label) {
    const cudaError_t error = cudaMemcpy(device, host, sizeof(T) * count, cudaMemcpyHostToDevice);
    return error == cudaSuccess ? 0 : fail(label, error);
}

template<class T>
int copy_to_host(T* host, const T* device, std::size_t count, const char* label) {
    const cudaError_t error = cudaMemcpy(host, device, sizeof(T) * count, cudaMemcpyDeviceToHost);
    return error == cudaSuccess ? 0 : fail(label, error);
}

int upload_montgomery(const std::uint8_t* input, std::size_t count, Fr** device) {
    std::vector<Fr> raw = decode_raw(input, count);
    if (allocate(device, count, "cudaMalloc scalar") != 0) return -1;
    if (copy_to_device(*device, raw.data(), count, "upload scalar") != 0) return -1;
    constexpr unsigned threads = 256;
    to_montgomery_kernel<<<(count + threads - 1U) / threads, threads>>>(*device, count);
    return check_launch("scalar to Montgomery kernel");
}

int encode_montgomery(Fr* device, std::size_t count, std::uint8_t* output) {
    constexpr unsigned threads = 256;
    from_montgomery_kernel<<<(count + threads - 1U) / threads, threads>>>(device, count);
    if (check_launch("scalar from Montgomery kernel") != 0) return -1;
    std::vector<Fr> raw(count);
    if (copy_to_host(raw.data(), device, count, "download scalar") != 0) return -1;
    for (std::size_t i = 0; i < count; ++i) raw_to_be(raw[i], output + 32U * i);
    return 0;
}

} // namespace

extern "C" const char* zkdl_gpu_last_error() { return last_error.c_str(); }

extern "C" int zkdl_gpu_probe(char* name, std::size_t name_capacity, int* major, int* minor) {
    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return fail("cudaGetDevice", error);
    cudaDeviceProp properties{};
    error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return fail("cudaGetDeviceProperties", error);
    if (name != nullptr && name_capacity != 0) std::snprintf(name, name_capacity, "%s", properties.name);
    if (major != nullptr) *major = properties.major;
    if (minor != nullptr) *minor = properties.minor;
    // A real allocation/kernel/synchronization is part of the probe.  Merely
    // finding libcuda is not enough to enable the formal GPU route.
    Fr* value = nullptr;
    if (allocate(&value, 1, "probe cudaMalloc") != 0) return -1;
    to_montgomery_kernel<<<1, 1>>>(value, 0);
    const int status = check_launch("probe kernel");
    cudaFree(value);
    return status;
}

extern "C" int zkdl_gpu_inner_product(
    const std::uint8_t* lhs, const std::uint8_t* rhs, std::size_t count, std::uint8_t* output) {
    if (lhs == nullptr || rhs == nullptr || output == nullptr || count == 0) { last_error = "invalid inner-product arguments"; return -1; }
    Fr *d_lhs = nullptr, *d_rhs = nullptr, *d_terms = nullptr, *d_result = nullptr;
    int status = upload_montgomery(lhs, count, &d_lhs);
    if (status == 0) status = upload_montgomery(rhs, count, &d_rhs);
    if (status == 0) status = allocate(&d_terms, count, "inner product terms allocation");
    if (status == 0) status = allocate(&d_result, 1, "inner product result allocation");
    if (status == 0) {
        constexpr unsigned threads = 256;
        inner_product_terms_kernel<<<(count + threads - 1U) / threads, threads>>>(d_lhs, d_rhs, d_terms, count);
        status = check_launch("inner product products kernel");
    }
    if (status == 0) { fr_reduce_kernel<<<1, 1>>>(d_terms, count, d_result); status = check_launch("inner product reduction kernel"); }
    if (status == 0) status = encode_montgomery(d_result, 1, output);
    cudaFree(d_lhs); cudaFree(d_rhs); cudaFree(d_terms); cudaFree(d_result);
    return status;
}

extern "C" int zkdl_gpu_fold_scalars_split(
    const std::uint8_t* input, std::size_t count,
    const std::uint8_t* left_coefficient, const std::uint8_t* right_coefficient,
    std::uint8_t* output) {
    if (input == nullptr || left_coefficient == nullptr || right_coefficient == nullptr || output == nullptr || count < 2 || (count & 1U)) { last_error = "invalid scalar fold arguments"; return -1; }
    const std::size_t half = count / 2U;
    Fr *d_input = nullptr, *d_output = nullptr, *d_coefficients = nullptr;
    std::uint8_t coefficient_bytes[64]; std::memcpy(coefficient_bytes, left_coefficient, 32); std::memcpy(coefficient_bytes + 32, right_coefficient, 32);
    int status = upload_montgomery(input, count, &d_input);
    if (status == 0) status = upload_montgomery(coefficient_bytes, 2, &d_coefficients);
    if (status == 0) status = allocate(&d_output, half, "scalar fold output allocation");
    if (status == 0) {
        constexpr unsigned threads = 256;
        scalar_fold_split_kernel<<<(half + threads - 1U) / threads, threads>>>(d_input, d_output, half, d_coefficients[0], d_coefficients[1]);
        status = check_launch("scalar fold kernel");
    }
    if (status == 0) status = encode_montgomery(d_output, half, output);
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_coefficients);
    return status;
}

extern "C" int zkdl_gpu_fold_points_split(
    const std::uint8_t* points_raw, std::size_t count,
    const std::uint8_t* left_coefficient, const std::uint8_t* right_coefficient,
    std::uint8_t* output_raw) {
    if (points_raw == nullptr || left_coefficient == nullptr || right_coefficient == nullptr || output_raw == nullptr || count < 2 || (count & 1U)) { last_error = "invalid point fold arguments"; return -1; }
    const std::size_t half = count / 2U;
    G1 *d_input = nullptr, *d_output = nullptr;
    Fr* d_coefficients = nullptr;
    std::uint8_t coefficient_bytes[64]; std::memcpy(coefficient_bytes, left_coefficient, 32); std::memcpy(coefficient_bytes + 32, right_coefficient, 32);
    int status = allocate(&d_input, count, "point fold input allocation");
    if (status == 0) status = allocate(&d_output, half, "point fold output allocation");
    if (status == 0) status = copy_to_device(d_input, reinterpret_cast<const G1*>(points_raw), count, "upload point fold input");
    // Point multiplication consumes canonical/raw scalar limbs, not Montgomery.
    std::vector<Fr> raw_coefficients = decode_raw(coefficient_bytes, 2);
    if (status == 0) status = allocate(&d_coefficients, 2, "point fold coefficient allocation");
    if (status == 0) status = copy_to_device(d_coefficients, raw_coefficients.data(), 2, "upload point fold coefficients");
    if (status == 0) {
        constexpr unsigned threads = 128;
        point_fold_split_kernel<<<(half + threads - 1U) / threads, threads>>>(d_input, d_output, half, d_coefficients[0], d_coefficients[1]);
        status = check_launch("point fold kernel");
    }
    if (status == 0) status = copy_to_host(reinterpret_cast<G1*>(output_raw), d_output, half, "download point fold output");
    cudaFree(d_input); cudaFree(d_output); cudaFree(d_coefficients);
    return status;
}

extern "C" int zkdl_gpu_msm(
    const std::uint8_t* points_raw, const std::uint8_t* scalars_be,
    std::size_t count, std::uint8_t* output_raw) {
    if (points_raw == nullptr || scalars_be == nullptr || output_raw == nullptr || count == 0) { last_error = "invalid MSM arguments"; return -1; }
    G1 *d_points = nullptr, *d_terms = nullptr, *d_result = nullptr;
    Fr* d_scalars = nullptr;
    std::vector<Fr> raw_scalars = decode_raw(scalars_be, count);
    int status = allocate(&d_points, count, "MSM point allocation");
    if (status == 0) status = allocate(&d_scalars, count, "MSM scalar allocation");
    if (status == 0) status = allocate(&d_terms, count, "MSM term allocation");
    if (status == 0) status = allocate(&d_result, 1, "MSM result allocation");
    if (status == 0) status = copy_to_device(d_points, reinterpret_cast<const G1*>(points_raw), count, "upload MSM points");
    if (status == 0) status = copy_to_device(d_scalars, raw_scalars.data(), count, "upload MSM scalars");
    if (status == 0) {
        constexpr unsigned threads = 128;
        msm_terms_kernel<<<(count + threads - 1U) / threads, threads>>>(d_points, d_scalars, d_terms, count);
        status = check_launch("MSM multiplication kernel");
    }
    if (status == 0) { g1_reduce_kernel<<<1, 1>>>(d_terms, count, d_result); status = check_launch("MSM reduction kernel"); }
    if (status == 0) status = copy_to_host(reinterpret_cast<G1*>(output_raw), d_result, 1, "download MSM result");
    cudaFree(d_points); cudaFree(d_scalars); cudaFree(d_terms); cudaFree(d_result);
    return status;
}

extern "C" int zkdl_gpu_sumcheck_evaluations(
    const std::uint8_t* columns, std::size_t column_count, std::size_t row_count,
    const std::uint8_t* beta, const std::uint8_t* mask, std::size_t mask_count,
    int first_round, const std::uint8_t* factor,
    const std::uint8_t* lambdas, std::size_t residual_count,
    const std::uint32_t* opcodes, const std::uint32_t* arguments,
    const std::uint8_t* constants, std::size_t token_count,
    const std::uint32_t* residual_offsets, std::size_t degree,
    const std::uint8_t* lookup_values, std::size_t lookup_value_count,
    const std::uint32_t* lookup_value_offsets,
    const std::uint32_t* lookup_bit_columns, std::size_t lookup_bit_count,
    const std::uint32_t* lookup_bit_offsets, std::size_t lookup_count,
    std::uint8_t* output) {
    if (columns == nullptr || beta == nullptr || mask == nullptr || factor == nullptr || lambdas == nullptr || opcodes == nullptr || arguments == nullptr || constants == nullptr || residual_offsets == nullptr || output == nullptr || column_count == 0 || residual_count == 0 || row_count < 2 || (row_count & 1U)) { last_error = "invalid Sumcheck GPU arguments"; return -1; }
    const std::size_t half = row_count / 2U;
    if (mask_count != (first_round ? half : row_count)) { last_error = "invalid Sumcheck mask shape"; return -1; }
    Fr *d_columns = nullptr, *d_beta = nullptr, *d_mask = nullptr, *d_factor = nullptr, *d_lambdas = nullptr, *d_contributions = nullptr, *d_output = nullptr;
    DeviceExprToken* d_tokens = nullptr;Fr* d_lookup_values=nullptr;
    std::uint32_t *d_offsets = nullptr, *d_error = nullptr,*d_lookup_value_offsets=nullptr,*d_lookup_bits=nullptr,*d_lookup_bit_offsets=nullptr;
    int status = upload_montgomery(columns, column_count * row_count, &d_columns);
    if (status == 0) status = upload_montgomery(beta, row_count, &d_beta);
    if (status == 0) status = upload_montgomery(mask, mask_count, &d_mask);
    if (status == 0) status = upload_montgomery(factor, 1, &d_factor);
    if (status == 0) status = upload_montgomery(lambdas, residual_count, &d_lambdas);
    std::vector<DeviceExprToken> host_tokens(token_count);
    for (std::size_t i = 0; i < token_count; ++i) {
        host_tokens[i].opcode = opcodes[i]; host_tokens[i].argument = arguments[i];
        host_tokens[i].constant = raw_from_be(constants + 32U * i);
    }
    if (status == 0) status = allocate(&d_tokens, token_count, "Sumcheck token allocation");
    if (status == 0) status = copy_to_device(d_tokens, host_tokens.data(), token_count, "upload Sumcheck tokens");
    // Constants are non-contiguous in DeviceExprToken, so convert them through
    // a temporary contiguous device buffer before uploading the final tokens.
    if (status == 0) status = allocate(&d_offsets, residual_count + 1U, "Sumcheck offset allocation");
    if (status == 0) status = copy_to_device(d_offsets, residual_offsets, residual_count + 1U, "upload Sumcheck offsets");
    if(status==0&&lookup_count>0)status=upload_montgomery(lookup_values,lookup_value_count,&d_lookup_values);
    if(status==0&&lookup_count>0)status=allocate(&d_lookup_value_offsets,lookup_count+1U,"lookup value offset allocation");
    if(status==0&&lookup_count>0)status=copy_to_device(d_lookup_value_offsets,lookup_value_offsets,lookup_count+1U,"upload lookup value offsets");
    if(status==0&&lookup_count>0)status=allocate(&d_lookup_bits,lookup_bit_count,"lookup bit allocation");
    if(status==0&&lookup_count>0)status=copy_to_device(d_lookup_bits,lookup_bit_columns,lookup_bit_count,"upload lookup bits");
    if(status==0&&lookup_count>0)status=allocate(&d_lookup_bit_offsets,lookup_count+1U,"lookup bit offset allocation");
    if(status==0&&lookup_count>0)status=copy_to_device(d_lookup_bit_offsets,lookup_bit_offsets,lookup_count+1U,"upload lookup bit offsets");
    const std::size_t evaluation_count = degree + 1U;
    if (status == 0) status = allocate(&d_contributions, half * evaluation_count, "Sumcheck contribution allocation");
    if (status == 0) status = allocate(&d_output, evaluation_count, "Sumcheck output allocation");
    if (status == 0) status = allocate(&d_error, 1, "Sumcheck error allocation");
    if (status == 0) { std::uint32_t zero = 0; status = copy_to_device(d_error, &zero, 1, "clear Sumcheck device error"); }
    if (status == 0) {
        // Convert token constants one-by-one on the host before upload.  The
        // Montgomery conversion is field arithmetic, so use the canonical R2
        // constant through a device helper kernel with one thread per token.
        // Re-upload below after conversion in a temporary contiguous buffer.
        std::vector<Fr> token_constants(token_count);
        for (std::size_t i = 0; i < token_count; ++i) token_constants[i] = host_tokens[i].constant;
        Fr* d_token_constants = nullptr;
        status = allocate(&d_token_constants, token_count, "Sumcheck constant allocation");
        if (status == 0) status = copy_to_device(d_token_constants, token_constants.data(), token_count, "upload Sumcheck constants");
        if (status == 0) { to_montgomery_kernel<<<(token_count + 255U) / 256U, 256>>>(d_token_constants, token_count); status = check_launch("Sumcheck constants Montgomery kernel"); }
        if (status == 0) status = copy_to_host(token_constants.data(), d_token_constants, token_count, "download converted Sumcheck constants");
        cudaFree(d_token_constants);
        if (status == 0) { for (std::size_t i = 0; i < token_count; ++i) host_tokens[i].constant = token_constants[i]; status = copy_to_device(d_tokens, host_tokens.data(), token_count, "upload converted Sumcheck tokens"); }
    }
    if (status == 0) {
        constexpr unsigned threads = 128;
        const std::size_t total = half * evaluation_count;
        sumcheck_contributions_kernel<<<(total + threads - 1U) / threads, threads>>>(
            d_columns, column_count, row_count, d_beta, d_mask, first_round != 0, d_factor[0],
            d_lambdas, residual_count, d_tokens, d_offsets,d_lookup_values,d_lookup_value_offsets,d_lookup_bits,d_lookup_bit_offsets,lookup_count,half, evaluation_count,
            d_contributions, d_error);
        status = check_launch("Sumcheck contribution kernel");
    }
    if (status == 0) {
        std::uint32_t code = 0; status = copy_to_host(&code, d_error, 1, "download Sumcheck device error");
        if (status == 0 && code != 0) { last_error = "Sumcheck expression bytecode rejected on GPU, code=" + std::to_string(code); status = -1; }
    }
    if (status == 0) {
        sumcheck_reduce_kernel<<<(degree + 256U) / 256U, 256>>>(d_contributions, half, evaluation_count, d_output);
        status = check_launch("Sumcheck reduction kernel");
    }
    if (status == 0) status = encode_montgomery(d_output, evaluation_count, output);
    cudaFree(d_columns); cudaFree(d_beta); cudaFree(d_mask); cudaFree(d_factor); cudaFree(d_lambdas);
    cudaFree(d_tokens); cudaFree(d_offsets); cudaFree(d_contributions); cudaFree(d_output); cudaFree(d_error);
    cudaFree(d_lookup_values);cudaFree(d_lookup_value_offsets);cudaFree(d_lookup_bits);cudaFree(d_lookup_bit_offsets);
    return status;
}
