#pragma once
// 本文件实现 CUDA 侧 BLS12-381 Fr Montgomery 算术原语，
// 只服务 full_gpu 执行层热点，不改变 proving / verifying 的 formal 语义。
// 模数 p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001，
// Montgomery 基底 R = 2^256，mu = -p^{-1} mod 2^64 = 0xfffffffeffffffff。

#include <cstdint>
#include <cuda_runtime.h>

namespace gatzk { namespace algebra { namespace gpu {

struct alignas(32) GpuFr {
    uint64_t d[4]; // 小端 limb 布局，数值保持在 Montgomery 形式
};

// ===== 模数与常量 =====
// BLS12-381 Fr 模数，按小端 4×64-bit limb 存放。
__device__ __constant__ static const uint64_t FR_P[4] = {
    0xffffffff00000001ULL,
    0x53bda402fffe5bfeULL,
    0x3339d80809a1d805ULL,
    0x73eda753299d7d48ULL
};

// Montgomery 约化常量 mu = -p^{-1} mod 2^64。
__device__ __constant__ static const uint64_t FR_INV = 0xfffffffeffffffffULL;

// R mod p，即 Montgomery 形式下的 1。
__device__ __constant__ static const uint64_t FR_ONE[4] = {
    0x00000001fffffffeULL,
    0x5884b7fa00034802ULL,
    0x998c4fefecbc4ff5ULL,
    0x1824b159acc5056fULL
};

// R^2 mod p，用于把普通表示转换到 Montgomery 表示。
__device__ __constant__ static const uint64_t FR_R2[4] = {
    0xc999e990f3f29c6dULL,
    0x2b6cedcb87925c23ULL,
    0x05d314967254398fULL,
    0x0748d9d99f59ff11ULL
};

// ===== 基础辅助函数 =====
// 乘加器：计算 `(hi, lo) = a * b + c + carry_in`。
__device__ __forceinline__
void mac(uint64_t a, uint64_t b, uint64_t c, uint64_t carry_in,
         uint64_t &lo_out, uint64_t &hi_out) {
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    // 累加 c。
    lo += c;
    hi += (lo < c) ? 1ULL : 0ULL;
    // 累加来自上一 limb 的进位。
    lo += carry_in;
    hi += (lo < carry_in) ? 1ULL : 0ULL;
    lo_out = lo;
    hi_out = hi;
}

// 带进位加法：返回和，`carry_out` 带回额外进位。
__device__ __forceinline__
uint64_t adc(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t &carry_out) {
    uint64_t r = a + b;
    carry_out = (r < a) ? 1ULL : 0ULL;
    r += carry_in;
    carry_out += (r < carry_in) ? 1ULL : 0ULL;
    return r;
}

// 带借位减法：返回差，`borrow_out` 带回额外借位。
__device__ __forceinline__
uint64_t sbb(uint64_t a, uint64_t b, uint64_t borrow_in, uint64_t &borrow_out) {
    uint64_t r = a - b;
    borrow_out = (a < b) ? 1ULL : 0ULL;
    uint64_t prev = r;
    r -= borrow_in;
    borrow_out += (prev < borrow_in) ? 1ULL : 0ULL;
    return r;
}

// ===== 比较 =====
__device__ __forceinline__
bool fr_gte_p(const GpuFr &a) {
    for (int i = 3; i >= 0; --i) {
        if (a.d[i] > FR_P[i]) return true;
        if (a.d[i] < FR_P[i]) return false;
    }
    return true; // 全部 limb 相等时也视为 `>= p`
}

// ===== 加法：r = (a + b) mod p =====
__device__ __forceinline__
GpuFr fr_add(const GpuFr &a, const GpuFr &b) {
    GpuFr r;
    uint64_t carry = 0;
    r.d[0] = adc(a.d[0], b.d[0], 0, carry);
    r.d[1] = adc(a.d[1], b.d[1], carry, carry);
    r.d[2] = adc(a.d[2], b.d[2], carry, carry);
    r.d[3] = adc(a.d[3], b.d[3], carry, carry);
    // 若结果越过模数，则回退一个 p。
    if (carry || fr_gte_p(r)) {
        uint64_t borrow = 0;
        r.d[0] = sbb(r.d[0], FR_P[0], 0, borrow);
        r.d[1] = sbb(r.d[1], FR_P[1], borrow, borrow);
        r.d[2] = sbb(r.d[2], FR_P[2], borrow, borrow);
        r.d[3] = sbb(r.d[3], FR_P[3], borrow, borrow);
    }
    return r;
}

// ===== 减法：r = (a - b) mod p =====
__device__ __forceinline__
GpuFr fr_sub(const GpuFr &a, const GpuFr &b) {
    GpuFr r;
    uint64_t borrow = 0;
    r.d[0] = sbb(a.d[0], b.d[0], 0, borrow);
    r.d[1] = sbb(a.d[1], b.d[1], borrow, borrow);
    r.d[2] = sbb(a.d[2], b.d[2], borrow, borrow);
    r.d[3] = sbb(a.d[3], b.d[3], borrow, borrow);
    // 若发生下溢，则补回一个 p。
    if (borrow) {
        uint64_t carry = 0;
        r.d[0] = adc(r.d[0], FR_P[0], 0, carry);
        r.d[1] = adc(r.d[1], FR_P[1], carry, carry);
        r.d[2] = adc(r.d[2], FR_P[2], carry, carry);
        r.d[3] = adc(r.d[3], FR_P[3], carry, carry);
    }
    return r;
}

// ===== Montgomery 乘法：r = (a * b * R^{-1}) mod p =====
// 使用 CIOS 约化算法，与 CPU 侧 Montgomery 语义保持一致。
__device__ __forceinline__
GpuFr fr_mul(const GpuFr &a, const GpuFr &b) {
    uint64_t t[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        // 先把 `a * b[i]` 累加到临时寄存器。
        uint64_t carry = 0;
        uint64_t hi;
        mac(a.d[0], b.d[i], t[0], carry, t[0], hi); carry = hi;
        mac(a.d[1], b.d[i], t[1], carry, t[1], hi); carry = hi;
        mac(a.d[2], b.d[i], t[2], carry, t[2], hi); carry = hi;
        mac(a.d[3], b.d[i], t[3], carry, t[3], hi);
        t[4] += hi;

        // 再执行一次 Montgomery 约化，把最低 limb 消掉。
        uint64_t m = t[0] * FR_INV;
        mac(m, FR_P[0], t[0], 0, t[0], hi); carry = hi;
        mac(m, FR_P[1], t[1], carry, t[1], hi); carry = hi;
        mac(m, FR_P[2], t[2], carry, t[2], hi); carry = hi;
        mac(m, FR_P[3], t[3], carry, t[3], hi);
        uint64_t new_t4 = t[4] + hi;
        uint64_t t4_carry = (new_t4 < t[4]) ? 1ULL : 0ULL;

        // 整体右移一 limb，进入下一轮。
        t[0] = t[1]; t[1] = t[2]; t[2] = t[3]; t[3] = new_t4; t[4] = t4_carry;
    }

    GpuFr r = {{t[0], t[1], t[2], t[3]}};
    if (t[4] || fr_gte_p(r)) {
        uint64_t borrow = 0;
        r.d[0] = sbb(r.d[0], FR_P[0], 0, borrow);
        r.d[1] = sbb(r.d[1], FR_P[1], borrow, borrow);
        r.d[2] = sbb(r.d[2], FR_P[2], borrow, borrow);
        r.d[3] = sbb(r.d[3], FR_P[3], borrow, borrow);
    }
    return r;
}

// ===== 常用常量 =====
__device__ __forceinline__
GpuFr fr_zero() {
    GpuFr r = {{0, 0, 0, 0}};
    return r;
}

__device__ __forceinline__
GpuFr fr_one() {
    GpuFr r = {{FR_ONE[0], FR_ONE[1], FR_ONE[2], FR_ONE[3]}};
    return r;
}

// 平方直接复用乘法，保持实现简单且与语义一致。
__device__ __forceinline__
GpuFr fr_sqr(const GpuFr &a) {
    return fr_mul(a, a);
}

// 幂运算采用平方乘，指数以 4 个小端 64-bit limb 给出。
__device__ __forceinline__
GpuFr fr_pow(const GpuFr &base, const uint64_t exp[4]) {
    GpuFr result = fr_one();
    GpuFr cur = base;
    for (int limb = 0; limb < 4; ++limb) {
        uint64_t e = exp[limb];
        for (int bit = 0; bit < 64; ++bit) {
            if (e & 1ULL) {
                result = fr_mul(result, cur);
            }
            cur = fr_sqr(cur);
            e >>= 1;
        }
    }
    return result;
}

// 逆元通过 `a^(p-2)` 计算，等价于费马小定理上的域求逆。
__device__ __forceinline__
GpuFr fr_inv(const GpuFr &a) {
    const uint64_t pm2[4] = {
        0xfffffffeffffffffULL, // p[0] - 2
        0x53bda402fffe5bfeULL, // p[1]
        0x3339d80809a1d805ULL, // p[2]
        0x73eda753299d7d48ULL  // p[3]
    };
    return fr_pow(a, pm2);
}

// 判零只用于执行层分支，不参与协议判定。
__device__ __forceinline__
bool fr_is_zero(const GpuFr &a) {
    return a.d[0] == 0 && a.d[1] == 0 && a.d[2] == 0 && a.d[3] == 0;
}

}}} // namespace gatzk::algebra::gpu
