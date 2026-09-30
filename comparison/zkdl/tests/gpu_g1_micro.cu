#include <cuda_runtime.h>

#include <iostream>

#include "../commitment.cuh"
#include "../cpu-backend.cuh"

namespace {

bool cuda_ok(const char *stage) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cerr << stage << ": cudaGetLastError: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        std::cerr << stage << ": cudaDeviceSynchronize: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    return true;
}

bool cuda_alloc(void **ptr, size_t bytes, const char *stage) {
    const cudaError_t status = cudaMalloc(ptr, bytes);
    if (status != cudaSuccess) {
        std::cerr << stage << ": cudaMalloc: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    return true;
}

template <typename T> bool cuda_copy_to_device(T *dst, const T *src, size_t count, const char *stage) {
    const cudaError_t status = cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << stage << ": cudaMemcpy H2D: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    return true;
}

template <typename T> bool cuda_copy_to_host(T *dst, const T *src, size_t count, const char *stage) {
    const cudaError_t status = cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        std::cerr << stage << ": cudaMemcpy D2H: " << cudaGetErrorString(status) << std::endl;
        return false;
    }
    return true;
}

}

KERNEL void gpu_g1_micro_mul(GLOBAL G1Jacobian_t *points, GLOBAL Fr_t *scalars, GLOBAL G1Jacobian_t *outputs, uint n) {
    const uint gid = GET_GLOBAL_ID();
    if (gid < n) outputs[gid] = G1Jacobian_mul(points[gid], scalars[gid]);
}

int main() {
    using namespace zkdl;

    const CpuG1Point cpu_generator = cpu_g1_generator();
    const G1Jacobian_t native_generator = G1Jacobian_generator;
    const Fr_t scalar_values[4] = {fr_value(fr_from_u32(1U)), fr_value(fr_from_u32(2U)), fr_value(fr_from_u32(3U)), fr_value(fr_from_u32(4U))};
    const G1Jacobian_t point_values[4] = {native_generator, native_generator, native_generator, native_generator};

    G1Jacobian_t *d_points = nullptr;
    Fr_t *d_scalars = nullptr;
    G1Jacobian_t *d_products = nullptr;
    if (!cuda_alloc(reinterpret_cast<void **>(&d_points), sizeof(point_values), "mul points") ||
        !cuda_alloc(reinterpret_cast<void **>(&d_scalars), sizeof(scalar_values), "mul scalars") ||
        !cuda_alloc(reinterpret_cast<void **>(&d_products), sizeof(point_values), "mul outputs") ||
        !cuda_copy_to_device(d_points, point_values, 4U, "mul points") || !cuda_copy_to_device(d_scalars, scalar_values, 4U, "mul scalars")) {
        cudaFree(d_points);
        cudaFree(d_scalars);
        cudaFree(d_products);
        return 1;
    }
    gpu_g1_micro_mul<<<1, 4>>>(d_points, d_scalars, d_products, 4U);
    if (!cuda_ok("gpu scalar multiplication")) {
        cudaFree(d_points);
        cudaFree(d_scalars);
        cudaFree(d_products);
        return 2;
    }
    G1Jacobian_t products[4]{};
    if (!cuda_copy_to_host(products, d_products, 4U, "mul outputs")) {
        cudaFree(d_points);
        cudaFree(d_scalars);
        cudaFree(d_products);
        return 3;
    }
    for (uint i = 0; i < 4U; ++i) {
        const CpuG1Point expected = cpu_g1_mul(cpu_generator, fr_wire(scalar_values[i]));
        if (!cpu_g1_equal(cpu_g1_from_native(products[i]), expected)) {
            std::cerr << "scalar multiplication mismatch at " << i << std::endl;
            cudaFree(d_points);
            cudaFree(d_scalars);
            cudaFree(d_products);
            return 4;
        }
    }
    cudaFree(d_points);
    cudaFree(d_scalars);
    cudaFree(d_products);

    G1TensorJacobian two_points(2U, point_values);
    const G1Jacobian_t sum_two = two_points.sum();
    if (!cuda_ok("two-point sum reduction")) return 5;
    if (!cpu_g1_equal(cpu_g1_from_native(sum_two), cpu_g1_mul(cpu_generator, fr_from_u32(2U)))) {
        std::cerr << "two-point sum mismatch" << std::endl;
        return 6;
    }
    G1TensorJacobian four_points(4U, point_values);
    const G1Jacobian_t sum_four = four_points.sum();
    if (!cuda_ok("four-point sum reduction")) return 7;
    if (!cpu_g1_equal(cpu_g1_from_native(sum_four), cpu_g1_mul(cpu_generator, fr_from_u32(4U)))) {
        std::cerr << "four-point sum mismatch" << std::endl;
        return 8;
    }

    const Fr_t fold_scalars[2] = {fr_value(fr_from_u32(1U)), fr_value(fr_from_u32(2U))};
    const Fr_t fold_u = fr_value(fr_from_u32(7U));
    Fr_t *d_fold_scalars = nullptr;
    G1Jacobian_t *d_fold_generators = nullptr;
    Fr_t *d_new_scalars = nullptr;
    G1Jacobian_t *d_new_generators = nullptr;
    G1Jacobian_t *d_fold_p = nullptr;
    G1Jacobian_t *d_fold_a = nullptr;
    G1Jacobian_t *d_fold_b = nullptr;
    if (!cuda_alloc(reinterpret_cast<void **>(&d_fold_scalars), sizeof(fold_scalars), "fold scalars") ||
        !cuda_alloc(reinterpret_cast<void **>(&d_fold_generators), sizeof(point_values[0]) * 2U, "fold generators") ||
        !cuda_alloc(reinterpret_cast<void **>(&d_new_scalars), sizeof(fold_scalars[0]), "fold new scalar") ||
        !cuda_alloc(reinterpret_cast<void **>(&d_new_generators), sizeof(point_values[0]), "fold new generator") ||
        !cuda_alloc(reinterpret_cast<void **>(&d_fold_p), sizeof(point_values[0]), "fold P") ||
        !cuda_alloc(reinterpret_cast<void **>(&d_fold_a), sizeof(point_values[0]), "fold A") ||
        !cuda_alloc(reinterpret_cast<void **>(&d_fold_b), sizeof(point_values[0]), "fold B") ||
        !cuda_copy_to_device(d_fold_scalars, fold_scalars, 2U, "fold scalars") ||
        !cuda_copy_to_device(d_fold_generators, point_values, 2U, "fold generators")) {
        cudaFree(d_fold_scalars);
        cudaFree(d_fold_generators);
        cudaFree(d_new_scalars);
        cudaFree(d_new_generators);
        cudaFree(d_fold_p);
        cudaFree(d_fold_a);
        cudaFree(d_fold_b);
        return 9;
    }
    me_open_step<<<1, G1NumThread>>>(d_fold_scalars, d_fold_generators, fold_u, d_new_scalars, d_new_generators, d_fold_p, d_fold_a, d_fold_b, 1U);
    if (!cuda_ok("single-round opening fold")) {
        cudaFree(d_fold_scalars);
        cudaFree(d_fold_generators);
        cudaFree(d_new_scalars);
        cudaFree(d_new_generators);
        cudaFree(d_fold_p);
        cudaFree(d_fold_a);
        cudaFree(d_fold_b);
        return 10;
    }
    Fr_t new_scalar{};
    G1Jacobian_t new_generator{}, fold_p{}, fold_a{}, fold_b{};
    if (!cuda_copy_to_host(&new_scalar, d_new_scalars, 1U, "fold new scalar") ||
        !cuda_copy_to_host(&new_generator, d_new_generators, 1U, "fold new generator") || !cuda_copy_to_host(&fold_p, d_fold_p, 1U, "fold P") ||
        !cuda_copy_to_host(&fold_a, d_fold_a, 1U, "fold A") || !cuda_copy_to_host(&fold_b, d_fold_b, 1U, "fold B")) {
        cudaFree(d_fold_scalars);
        cudaFree(d_fold_generators);
        cudaFree(d_new_scalars);
        cudaFree(d_new_generators);
        cudaFree(d_fold_p);
        cudaFree(d_fold_a);
        cudaFree(d_fold_b);
        return 11;
    }
    if (fr_compare(fr_wire(new_scalar), fr_from_u32(8U)) != 0 || !cpu_g1_equal(cpu_g1_from_native(new_generator), cpu_generator) ||
        !cpu_g1_equal(cpu_g1_from_native(fold_p), cpu_g1_mul(cpu_generator, fr_from_u32(3U))) || !cpu_g1_equal(cpu_g1_from_native(fold_a), cpu_generator) ||
        !cpu_g1_equal(cpu_g1_from_native(fold_b), cpu_g1_mul(cpu_generator, fr_from_u32(2U)))) {
        std::cerr << "single-round opening fold mismatch" << std::endl;
        cudaFree(d_fold_scalars);
        cudaFree(d_fold_generators);
        cudaFree(d_new_scalars);
        cudaFree(d_new_generators);
        cudaFree(d_fold_p);
        cudaFree(d_fold_a);
        cudaFree(d_fold_b);
        return 12;
    }
    cudaFree(d_fold_scalars);
    cudaFree(d_fold_generators);
    cudaFree(d_new_scalars);
    cudaFree(d_new_generators);
    cudaFree(d_fold_p);
    cudaFree(d_fold_a);
    cudaFree(d_fold_b);

    std::cout << "zkdl GPU G1 micro ok" << std::endl;
    return 0;
}
