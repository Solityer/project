#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "gatzk/algebra/cuda_field.cuh"
#include "gatzk/algebra/vector_ops.hpp"

namespace gatzk::algebra {

bool cuda_backend_runtime_available() {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

// =========================================================================
// GPU Barycentric Weight Computation
// =========================================================================
// Computes weights[i] = Z(x) * omega^i / n / (x - omega^i) for i in [0, domain_size)
// Uses Montgomery batch inversion: single inversion + O(n) multiplications

namespace {

// Phase 1: Compute omega powers and denominators in parallel  
__global__ void barycentric_denominators_kernel(
    const gpu::GpuFr omega,
    const gpu::GpuFr x,
    gpu::GpuFr* __restrict__ omega_powers,    // output: omega^i
    gpu::GpuFr* __restrict__ denominators,     // output: x - omega^i
    std::size_t domain_size) {
    const auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= domain_size) return;
    // Compute omega^tid by repeated squaring
    // For power-of-two domains, we can compute omega^tid = omega_base^tid
    // We precompute omega^(2^k) in a small table and multiply selected entries
    gpu::GpuFr result = gpu::fr_one();
    gpu::GpuFr base = omega;
    std::size_t exp = tid;
    while (exp > 0) {
        if (exp & 1) result = gpu::fr_mul(result, base);
        base = gpu::fr_sqr(base);
        exp >>= 1;
    }
    omega_powers[tid] = result;
    denominators[tid] = gpu::fr_sub(x, result);
}

// Phase 2: Parallel prefix product (scan) on denominators
// We use a simple chunked approach: each chunk computes its product,
// then we do a serial cross-chunk reduction, then expand back
__global__ void prefix_product_chunk_kernel(
    const gpu::GpuFr* __restrict__ denominators,
    gpu::GpuFr* __restrict__ prefix_within,
    gpu::GpuFr* __restrict__ chunk_products,
    std::size_t domain_size,
    std::size_t chunk_size) {
    const auto chunk_id = static_cast<std::size_t>(blockIdx.x);
    const auto begin = chunk_id * chunk_size;
    const auto end = min(begin + chunk_size, domain_size);
    if (begin >= end) return;
    // Only thread 0 per block does the sequential scan
    if (threadIdx.x == 0) {
        gpu::GpuFr acc = gpu::fr_one();
        for (std::size_t i = begin; i < end; ++i) {
            prefix_within[i] = acc;
            acc = gpu::fr_mul(acc, denominators[i]);
        }
        chunk_products[chunk_id] = acc;
    }
}

// Phase 3: Final weight computation with inverse
__global__ void barycentric_weights_finalize_kernel(
    const gpu::GpuFr* __restrict__ omega_powers,
    const gpu::GpuFr* __restrict__ denominators,
    const gpu::GpuFr* __restrict__ prefix_within,
    const gpu::GpuFr* __restrict__ chunk_outer_inv,
    const gpu::GpuFr zero_eval,       // Z_N(x)
    const gpu::GpuFr inv_n,           // 1/N in Montgomery form
    gpu::GpuFr* __restrict__ weights,
    std::size_t domain_size,
    std::size_t chunk_size) {
    const auto chunk_id = static_cast<std::size_t>(blockIdx.x);
    const auto begin = chunk_id * chunk_size;
    const auto end = min(begin + chunk_size, domain_size);
    if (begin >= end) return;
    if (threadIdx.x != 0) return;
    gpu::GpuFr running_suffix = gpu::fr_one();
    for (std::size_t i = end; i-- > begin;) {
        // inv_d[i] = chunk_outer_inv[chunk_id] * prefix_within[i] * running_suffix
        gpu::GpuFr inv_d = gpu::fr_mul(chunk_outer_inv[chunk_id], prefix_within[i]);
        inv_d = gpu::fr_mul(inv_d, running_suffix);
        // weight[i] = Z(x) * omega^i * inv_n * inv_d[i]
        gpu::GpuFr scaled = gpu::fr_mul(omega_powers[i], inv_n);
        gpu::GpuFr w = gpu::fr_mul(zero_eval, scaled);
        w = gpu::fr_mul(w, inv_d);
        weights[i] = w;
        running_suffix = gpu::fr_mul(running_suffix, denominators[i]);
    }
}

} // anonymous namespace

// Host-callable function: compute barycentric weights on GPU
// Returns weights as raw 4×uint64 limbs matching mcl::Fr Montgomery form
bool barycentric_weights_gpu(
    const uint64_t omega_limbs[4],
    const uint64_t x_limbs[4],
    const uint64_t zero_eval_limbs[4],
    const uint64_t inv_n_limbs[4],
    uint64_t* output_weight_limbs,   // output: domain_size × 4 uint64
    std::size_t domain_size) {
    
    if (domain_size == 0) return true;
    
    gpu::GpuFr omega, x, zero_eval, inv_n;
    std::memcpy(omega.d, omega_limbs, 32);
    std::memcpy(x.d, x_limbs, 32);
    std::memcpy(zero_eval.d, zero_eval_limbs, 32);
    std::memcpy(inv_n.d, inv_n_limbs, 32);
    
    // Allocate GPU buffers
    gpu::GpuFr *d_omega_powers = nullptr, *d_denominators = nullptr;
    gpu::GpuFr *d_prefix = nullptr, *d_chunk_products = nullptr;
    gpu::GpuFr *d_chunk_outer_inv = nullptr, *d_weights = nullptr;
    
    const std::size_t num_chunks = std::min(domain_size, static_cast<std::size_t>(256));
    const std::size_t chunk_size = (domain_size + num_chunks - 1) / num_chunks;
    
    const auto elem_bytes = domain_size * sizeof(gpu::GpuFr);
    const auto chunk_bytes = num_chunks * sizeof(gpu::GpuFr);
    
    auto cleanup = [&]() {
        cudaFree(d_omega_powers); cudaFree(d_denominators);
        cudaFree(d_prefix); cudaFree(d_chunk_products);
        cudaFree(d_chunk_outer_inv); cudaFree(d_weights);
    };
    
    if (cudaMalloc(&d_omega_powers, elem_bytes) != cudaSuccess ||
        cudaMalloc(&d_denominators, elem_bytes) != cudaSuccess ||
        cudaMalloc(&d_prefix, elem_bytes) != cudaSuccess ||
        cudaMalloc(&d_chunk_products, chunk_bytes) != cudaSuccess ||
        cudaMalloc(&d_chunk_outer_inv, chunk_bytes) != cudaSuccess ||
        cudaMalloc(&d_weights, elem_bytes) != cudaSuccess) {
        cleanup();
        return false;
    }
    
    // Phase 1: Compute omega powers and denominators
    {
        constexpr unsigned block = 256;
        const auto grid = static_cast<unsigned>((domain_size + block - 1) / block);
        barycentric_denominators_kernel<<<grid, block>>>(omega, x, d_omega_powers, d_denominators, domain_size);
    }
    
    // Phase 2: Prefix products within chunks
    {
        prefix_product_chunk_kernel<<<static_cast<unsigned>(num_chunks), 1>>>(
            d_denominators, d_prefix, d_chunk_products, domain_size, chunk_size);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) { cleanup(); return false; }
    
    // Phase 2b: Cross-chunk reduction on CPU (num_chunks is small, ≤256)
    std::vector<gpu::GpuFr> h_chunk_products(num_chunks);
    std::vector<gpu::GpuFr> h_chunk_outer_inv(num_chunks);
    cudaMemcpy(h_chunk_products.data(), d_chunk_products, chunk_bytes, cudaMemcpyDeviceToHost);
    
    // Cross-chunk reduction using mcl on host (num_chunks ≤ 256, negligible cost)
    {
        // Helper: copy 4 limbs from GpuFr into mcl::Fr (both in Montgomery form)
        auto gpufr_to_mclfr = [](const gpu::GpuFr& src) -> mcl::Fr {
            mcl::Fr dst;
            dst.clear();
            std::memcpy(&dst, src.d, 32);
            return dst;
        };
        auto mclfr_to_gpufr = [](const mcl::Fr& src) -> gpu::GpuFr {
            gpu::GpuFr dst;
            std::memcpy(dst.d, &src, 32);
            return dst;
        };
        
        std::vector<mcl::Fr> mcl_chunk_prods(num_chunks);
        for (std::size_t c = 0; c < num_chunks; ++c) {
            mcl_chunk_prods[c] = gpufr_to_mclfr(h_chunk_products[c]);
        }
        
        std::vector<mcl::Fr> mcl_cross_prefix(num_chunks + 1);
        mcl_cross_prefix[0] = mcl::Fr(1);
        for (std::size_t c = 0; c < num_chunks; ++c) {
            mcl::Fr::mul(mcl_cross_prefix[c + 1], mcl_cross_prefix[c], mcl_chunk_prods[c]);
        }
        
        mcl::Fr global_inv;
        mcl::Fr::inv(global_inv, mcl_cross_prefix[num_chunks]);
        
        mcl::Fr cross_suffix;
        cross_suffix = mcl::Fr(1);
        for (std::size_t c = num_chunks; c-- > 0;) {
            mcl::Fr tmp;
            mcl::Fr::mul(tmp, global_inv, mcl_cross_prefix[c]);
            mcl::Fr outer;
            mcl::Fr::mul(outer, tmp, cross_suffix);
            h_chunk_outer_inv[c] = mclfr_to_gpufr(outer);
            mcl::Fr::mul(cross_suffix, cross_suffix, mcl_chunk_prods[c]);
        }
    }
    cudaMemcpy(d_chunk_outer_inv, h_chunk_outer_inv.data(), chunk_bytes, cudaMemcpyHostToDevice);
    
    // Phase 3: Finalize weights
    {
        barycentric_weights_finalize_kernel<<<static_cast<unsigned>(num_chunks), 1>>>(
            d_omega_powers, d_denominators, d_prefix, d_chunk_outer_inv,
            zero_eval, inv_n, d_weights, domain_size, chunk_size);
    }
    
    if (cudaDeviceSynchronize() != cudaSuccess) { cleanup(); return false; }
    
    // Download weights
    cudaMemcpy(output_weight_limbs, d_weights, elem_bytes, cudaMemcpyDeviceToHost);
    cleanup();
    return true;
}

// =========================================================================
// GPU Weighted Polynomial Evaluation (the main hot-path kernel)
// =========================================================================
// Computes result[j] = sum_{i=0}^{domain_size-1} weights[i] * values[i * row_count + j]
// for j in [0, row_count)
// This is the inner loop of polynomial evaluation at a point using barycentric weights.

namespace {

// ---- Columnar kernel: reads polynomial data via array of column pointers ----
// Each thread handles one domain element, accumulates w[col] * values[row][col] per polynomial.
// Grid: (num_domain_blocks, row_count)
// Block: (BLOCK_DIM)
// Shared memory reduction within each block.
__global__ void weighted_eval_columnar_kernel(
    const gpu::GpuFr* __restrict__ weights,           // [domain_size] in host/managed memory
    const gpu::GpuFr* const* __restrict__ columns,    // [row_count] array of pointers, each -> [domain_size]
    gpu::GpuFr* __restrict__ partial_results,          // [num_blocks * row_count]
    std::size_t domain_size,
    std::size_t row_count) {

    const auto tid = threadIdx.x;
    const auto col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const auto row = blockIdx.y;

    if (row >= row_count) return;

    gpu::GpuFr acc = gpu::fr_zero();
    if (col < domain_size) {
        acc = gpu::fr_mul(weights[col], columns[row][col]);
    }

    extern __shared__ gpu::GpuFr smem[];
    smem[tid] = acc;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = gpu::fr_add(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_results[static_cast<std::size_t>(blockIdx.x) * row_count + row] = smem[0];
    }
}

// Block-level parallel reduction for weighted poly eval
// Each block handles a contiguous chunk of the domain dimension,
// computing partial sums for ALL row_count polynomials.
// Grid: (num_domain_chunks, 1, 1)
// Block: (BLOCK_DIM, 1, 1)
// Each thread accumulates over (chunk_size / BLOCK_DIM) domain elements,
// then we reduce within the block using shared memory.
__global__ void weighted_eval_kernel(
    const gpu::GpuFr* __restrict__ weights,          // [domain_size]
    const gpu::GpuFr* __restrict__ values,            // [domain_size * row_count] interleaved
    gpu::GpuFr* __restrict__ partial_results,          // [num_chunks * row_count]
    std::size_t domain_size,
    std::size_t row_count,
    std::size_t chunk_start,
    std::size_t chunk_end) {
    
    const auto tid = threadIdx.x;
    const auto chunk_id = blockIdx.x;
    const auto row = blockIdx.y;
    
    if (row >= row_count) return;
    
    const auto real_chunk_start = chunk_start + chunk_id * blockDim.x;
    const auto col = real_chunk_start + tid;
    
    gpu::GpuFr acc = gpu::fr_zero();
    if (col < chunk_end && col < domain_size) {
        // values is uploaded starting at chunk_start, so index with (col - chunk_start)
        const auto local_col = col - chunk_start;
        acc = gpu::fr_mul(weights[col], values[local_col * row_count + row]);
    }
    
    // Block-level reduction using shared memory
    // We store partial Fr values in shared memory
    extern __shared__ gpu::GpuFr smem[];
    smem[tid] = acc;
    __syncthreads();
    
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = gpu::fr_add(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_results[chunk_id * row_count + row] = smem[0];
    }
}

// Final reduction: sum partial results across chunks for each row
__global__ void reduce_partials_kernel(
    const gpu::GpuFr* __restrict__ partial_results,   // [num_chunks * row_count]
    gpu::GpuFr* __restrict__ final_results,            // [row_count]
    std::size_t num_chunks,
    std::size_t row_count) {
    const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    gpu::GpuFr acc = gpu::fr_zero();
    for (std::size_t c = 0; c < num_chunks; ++c) {
        acc = gpu::fr_add(acc, partial_results[c * row_count + row]);
    }
    final_results[row] = acc;
}

__global__ void accumulate_results_kernel(
    const gpu::GpuFr* __restrict__ chunk_results,
    gpu::GpuFr* __restrict__ final_results,
    std::size_t row_count) {
    const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    final_results[row] = gpu::fr_add(final_results[row], chunk_results[row]);
}

} // anonymous namespace

// Host-callable: GPU weighted polynomial evaluation
// weights: domain_size × 4 uint64 limbs (Fr in Montgomery form)
// interleaved_values: domain_size × row_count × 4 uint64 limbs
// output: row_count × 4 uint64 limbs
bool weighted_poly_eval_gpu(
    const uint64_t* weight_limbs,
    const uint64_t* interleaved_value_limbs,
    uint64_t* output_limbs,
    std::size_t domain_size,
    std::size_t row_count) {
    
    if (domain_size == 0 || row_count == 0) return true;
    
    const auto weight_bytes = domain_size * sizeof(gpu::GpuFr);
    const auto result_bytes = row_count * sizeof(gpu::GpuFr);
    
    // For very large value arrays, process in streaming chunks
    // Each chunk covers a portion of the domain dimension
    constexpr std::size_t kMaxGpuValueBytes = 2ULL * 1024 * 1024 * 1024; // 2 GB per chunk
    const std::size_t max_cols_per_chunk = 
        row_count > 0 ? kMaxGpuValueBytes / (row_count * sizeof(gpu::GpuFr)) : domain_size;
    const std::size_t cols_per_chunk = std::min(domain_size, std::max(max_cols_per_chunk, static_cast<std::size_t>(1024)));
    
    const bool zero_copy = cuda_backend_unified_addressing_available();

    gpu::GpuFr* d_weights = nullptr;
    gpu::GpuFr* d_values = nullptr;
    gpu::GpuFr* d_final = nullptr;

    const auto cleanup = [&]() {
        if (d_values != nullptr) cudaFree(d_values);
        if (d_weights != nullptr) cudaFree(d_weights);
        if (d_final != nullptr) cudaFree(d_final);
    };

    if (!zero_copy && cudaMalloc(&d_weights, weight_bytes) != cudaSuccess) return false;
    if (cudaMalloc(&d_final, result_bytes) != cudaSuccess) {
        cleanup();
        return false;
    }

    const auto* weight_ptr =
        zero_copy ? reinterpret_cast<const gpu::GpuFr*>(weight_limbs) : nullptr;
    if (!zero_copy) {
        cudaMemcpy(d_weights, weight_limbs, weight_bytes, cudaMemcpyHostToDevice);
        weight_ptr = d_weights;
    }
    cudaMemset(d_final, 0, result_bytes);
    
    // Process domain in chunks to manage GPU memory
    for (std::size_t chunk_start = 0; chunk_start < domain_size; chunk_start += cols_per_chunk) {
        const auto chunk_end = std::min(domain_size, chunk_start + cols_per_chunk);
        const auto chunk_cols = chunk_end - chunk_start;
        const auto chunk_value_bytes = chunk_cols * row_count * sizeof(gpu::GpuFr);

        const auto* chunk_values = reinterpret_cast<const gpu::GpuFr*>(interleaved_value_limbs) + chunk_start * row_count;
        if (!zero_copy) {
            if (d_values != nullptr) {
                cudaFree(d_values);
                d_values = nullptr;
            }
            if (cudaMalloc(&d_values, chunk_value_bytes) != cudaSuccess) {
                cleanup();
                return false;
            }
            cudaMemcpy(d_values, chunk_values, chunk_value_bytes, cudaMemcpyHostToDevice);
            chunk_values = d_values;
        }
        
        // Launch evaluation kernel for this chunk
        constexpr unsigned kBlockDim = 128;
        const auto num_domain_blocks = static_cast<unsigned>((chunk_cols + kBlockDim - 1) / kBlockDim);
        
        // Allocate partial results for this chunk
        gpu::GpuFr* d_partials = nullptr;
        const auto partial_bytes = num_domain_blocks * row_count * sizeof(gpu::GpuFr);
        if (cudaMalloc(&d_partials, partial_bytes) != cudaSuccess) {
            cleanup();
            return false;
        }
        cudaMemset(d_partials, 0, partial_bytes);
        
        // Grid: (num_domain_blocks, row_count)
        // We limit row_count in y-dimension. If row_count > 65535, we loop.
        const auto grid_y = static_cast<unsigned>(std::min(row_count, static_cast<std::size_t>(65535)));
        dim3 grid(num_domain_blocks, grid_y);
        const auto smem_bytes = kBlockDim * sizeof(gpu::GpuFr);
        
        // Adjust kernel: d_values is offset within the chunk
        // We need the kernel to use the chunk-relative indexing
        // Actually, we uploaded chunk data starting at d_values[0], so column 0 in the kernel
        // corresponds to chunk_start in the global domain
        weighted_eval_kernel<<<grid, kBlockDim, smem_bytes>>>(
            weight_ptr, chunk_values, d_partials,
            domain_size, row_count, chunk_start, chunk_end);
        
        // Reduce partials into final results
        // We accumulate into d_final across chunks
        {
            constexpr unsigned rBlock = 256;
            const auto rGrid = static_cast<unsigned>((row_count + rBlock - 1) / rBlock);
            // First reduce partials for this chunk
            gpu::GpuFr* d_chunk_result = nullptr;
            if (cudaMalloc(&d_chunk_result, result_bytes) != cudaSuccess) {
                cudaFree(d_partials);
                cleanup();
                return false;
            }
            reduce_partials_kernel<<<rGrid, rBlock>>>(d_partials, d_chunk_result, num_domain_blocks, row_count);
            accumulate_results_kernel<<<rGrid, rBlock>>>(d_chunk_result, d_final, row_count);
            if (cudaDeviceSynchronize() != cudaSuccess) {
                cudaFree(d_chunk_result);
                cudaFree(d_partials);
                cleanup();
                return false;
            }
            cudaFree(d_chunk_result);
        }
        cudaFree(d_partials);
    }
    
    // Download final results
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        return false;
    }
    cudaMemcpy(output_limbs, d_final, result_bytes, cudaMemcpyDeviceToHost);

    cleanup();
    return true;
}

// ---- Zero-copy columnar GPU evaluation (for unified memory systems) ----
// Reads polynomial data directly from host pointers via column pointer array.
// No interleaving needed. No explicit cudaMemcpy for weights/values.
bool weighted_poly_eval_gpu_columnar(
    const uint64_t* weight_limbs,             // host ptr: domain_size × 4 uint64
    const uint64_t* const* column_ptrs,       // host array of poly_count pointers, each -> domain_size × 4 uint64
    uint64_t* output_limbs,                   // host ptr: poly_count × 4 uint64
    std::size_t domain_size,
    std::size_t row_count) {

    if (domain_size == 0 || row_count == 0) return true;

    // On unified memory (e.g. GB10), host pointers are directly accessible from GPU.
    // We only need to allocate device memory for partial results and column pointer array.
    constexpr unsigned kBlockDim = 256;
    const auto num_blocks = static_cast<unsigned>((domain_size + kBlockDim - 1) / kBlockDim);
    const auto grid_y = static_cast<unsigned>(std::min(row_count, static_cast<std::size_t>(65535)));

    // Allocate column pointer array on device
    const gpu::GpuFr** d_columns = nullptr;
    const auto ptr_bytes = row_count * sizeof(const gpu::GpuFr*);
    if (cudaMalloc(&d_columns, ptr_bytes) != cudaSuccess) return false;

    // Build host-side array of GpuFr pointers (just reinterpret_cast the uint64_t pointers)
    std::vector<const gpu::GpuFr*> h_columns(row_count);
    for (std::size_t i = 0; i < row_count; ++i) {
        h_columns[i] = reinterpret_cast<const gpu::GpuFr*>(column_ptrs[i]);
    }
    cudaMemcpy(d_columns, h_columns.data(), ptr_bytes, cudaMemcpyHostToDevice);

    // Allocate partial results on device
    const auto partial_bytes = static_cast<std::size_t>(num_blocks) * row_count * sizeof(gpu::GpuFr);
    gpu::GpuFr* d_partials = nullptr;
    if (cudaMalloc(&d_partials, partial_bytes) != cudaSuccess) {
        cudaFree(d_columns);
        return false;
    }
    cudaMemset(d_partials, 0, partial_bytes);

    // Launch columnar kernel: GPU reads weights and values directly from host memory
    dim3 grid(num_blocks, grid_y);
    const auto smem_bytes = kBlockDim * sizeof(gpu::GpuFr);
    weighted_eval_columnar_kernel<<<grid, kBlockDim, smem_bytes>>>(
        reinterpret_cast<const gpu::GpuFr*>(weight_limbs),
        d_columns,
        d_partials,
        domain_size,
        row_count);

    // Reduce partials → final results
    const auto result_bytes = row_count * sizeof(gpu::GpuFr);
    gpu::GpuFr* d_final = nullptr;
    if (cudaMalloc(&d_final, result_bytes) != cudaSuccess) {
        cudaFree(d_partials); cudaFree(d_columns);
        return false;
    }

    {
        constexpr unsigned rBlock = 256;
        const auto rGrid = static_cast<unsigned>((row_count + rBlock - 1) / rBlock);
        reduce_partials_kernel<<<rGrid, rBlock>>>(d_partials, d_final, num_blocks, row_count);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_final); cudaFree(d_partials); cudaFree(d_columns);
        return false;
    }

    // Download final results
    cudaMemcpy(output_limbs, d_final, result_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_final);
    cudaFree(d_partials);
    cudaFree(d_columns);
    return true;
}

// ---- Multi-polynomial evaluation with shared weights ----
// Processes poly_count polynomials against the same weight vector.
// On unified-memory devices (e.g. GB10), delegates to the columnar
// kernel which evaluates ALL polynomials in a single kernel launch
// using zero-copy host pointers.  On discrete GPUs, falls back to
// sequential per-polynomial processing with weight reuse.
bool weighted_poly_eval_gpu_multi(
    const uint64_t* weight_limbs,
    const uint64_t* const* column_value_ptrs,
    uint64_t* output_limbs,       // poly_count × 4 uint64
    std::size_t domain_size,
    std::size_t poly_count) {

    if (domain_size == 0 || poly_count == 0) return true;

    // Detect unified memory: if available, use zero-copy columnar kernel
    // which processes all poly_count polynomials in a SINGLE kernel launch
    // instead of poly_count sequential launches.
    if (cuda_backend_unified_addressing_available()) {
        return weighted_poly_eval_gpu_columnar(
            weight_limbs, column_value_ptrs, output_limbs,
            domain_size, poly_count);
    }

    // ---- Discrete GPU fallback: sequential per-polynomial with weight reuse ----
    const auto weight_bytes = domain_size * sizeof(gpu::GpuFr);
    const auto single_val_bytes = domain_size * sizeof(gpu::GpuFr);
    const auto result_bytes_single = sizeof(gpu::GpuFr);

    gpu::GpuFr* d_weights = nullptr;
    if (cudaMalloc(&d_weights, weight_bytes) != cudaSuccess) return false;
    cudaMemcpy(d_weights, weight_limbs, weight_bytes, cudaMemcpyHostToDevice);

    gpu::GpuFr* d_values = nullptr;
    if (cudaMalloc(&d_values, single_val_bytes) != cudaSuccess) {
        cudaFree(d_weights);
        return false;
    }

    constexpr unsigned kBlockDim = 128;
    const auto num_blocks = static_cast<unsigned>((domain_size + kBlockDim - 1) / kBlockDim);
    const auto partial_bytes = static_cast<std::size_t>(num_blocks) * sizeof(gpu::GpuFr);

    gpu::GpuFr* d_partials = nullptr;
    if (cudaMalloc(&d_partials, partial_bytes) != cudaSuccess) {
        cudaFree(d_values); cudaFree(d_weights);
        return false;
    }

    gpu::GpuFr* d_result = nullptr;
    if (cudaMalloc(&d_result, result_bytes_single) != cudaSuccess) {
        cudaFree(d_partials); cudaFree(d_values); cudaFree(d_weights);
        return false;
    }

    const auto smem_bytes = kBlockDim * sizeof(gpu::GpuFr);

    for (std::size_t p = 0; p < poly_count; ++p) {
        cudaMemcpy(d_values, column_value_ptrs[p], single_val_bytes, cudaMemcpyHostToDevice);
        cudaMemset(d_partials, 0, partial_bytes);

        dim3 grid(num_blocks, 1);
        weighted_eval_kernel<<<grid, kBlockDim, smem_bytes>>>(
            d_weights, d_values, d_partials,
            domain_size, 1, 0, domain_size);

        constexpr unsigned rBlock = 256;
        reduce_partials_kernel<<<1, rBlock>>>(d_partials, d_result, num_blocks, 1);

        auto* output_ptr = reinterpret_cast<gpu::GpuFr*>(output_limbs) + p;
        cudaMemcpy(output_ptr, d_result, result_bytes_single, cudaMemcpyDeviceToHost);
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_result); cudaFree(d_partials); cudaFree(d_values); cudaFree(d_weights);
        return false;
    }

    cudaFree(d_result);
    cudaFree(d_partials);
    cudaFree(d_values);
    cudaFree(d_weights);
    return true;
}

}  // namespace gatzk::algebra

namespace gatzk::protocol {
namespace {

__global__ void histogram_indices_kernel(
    const unsigned long long* indices,
    std::size_t count,
    unsigned long long* buckets,
    std::size_t bucket_count) {
    const auto tid =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (tid >= count) {
        return;
    }
    const auto bucket = static_cast<std::size_t>(indices[tid]);
    if (bucket < bucket_count) {
        atomicAdd(&buckets[bucket], 1ULL);
    }
}

}  // namespace

std::vector<std::size_t> lookup_histogram_indices_cuda(
    const std::vector<std::size_t>& indices,
    std::size_t domain_size) {
    std::vector<std::size_t> out(domain_size, 0);
    if (indices.empty() || domain_size == 0) {
        return out;
    }

    std::vector<unsigned long long> host_indices(indices.size(), 0ULL);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        host_indices[i] = static_cast<unsigned long long>(indices[i]);
    }
    std::vector<unsigned long long> host_counts(domain_size, 0ULL);

    unsigned long long* device_indices = nullptr;
    unsigned long long* device_counts = nullptr;
    const auto indices_bytes = sizeof(unsigned long long) * host_indices.size();
    const auto counts_bytes = sizeof(unsigned long long) * host_counts.size();
    if (cudaMalloc(&device_indices, indices_bytes) != cudaSuccess) {
        return {};  // signal GPU failure; caller falls back to CPU
    }
    if (cudaMalloc(&device_counts, counts_bytes) != cudaSuccess) {
        cudaFree(device_indices);
        return {};
    }

    const auto cleanup = [&]() {
        cudaFree(device_indices);
        cudaFree(device_counts);
    };

    if (cudaMemcpy(device_indices, host_indices.data(), indices_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda histogram failed to upload indices");
    }
    if (cudaMemset(device_counts, 0, counts_bytes) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda histogram failed to reset bucket buffer");
    }

    constexpr unsigned int block_size = 256;
    const auto grid_size = static_cast<unsigned int>((host_indices.size() + block_size - 1U) / block_size);
    histogram_indices_kernel<<<grid_size, block_size>>>(
        device_indices,
        host_indices.size(),
        device_counts,
        host_counts.size());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda histogram kernel launch failed");
    }

    if (cudaMemcpy(host_counts.data(), device_counts, counts_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda histogram failed to download counts");
    }
    cleanup();

    for (std::size_t i = 0; i < domain_size; ++i) {
        out[i] = static_cast<std::size_t>(host_counts[i]);
    }
    return out;
}

}  // namespace gatzk::protocol
