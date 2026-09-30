#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gatzk
{
    namespace algebra
    {
        namespace gpu
        {

            struct GpuFr
            {
                uint64_t d[4];
            };
            static_assert(sizeof(GpuFr) == 4 * sizeof(uint64_t));
            static_assert(alignof(GpuFr) == alignof(uint64_t));

            __device__ __constant__ static const uint64_t FR_P[4] = {0xffffffff00000001ULL, 0x53bda402fffe5bfeULL, 0x3339d80809a1d805ULL,
                                                                     0x73eda753299d7d48ULL};

            __device__ __constant__ static const uint64_t FR_INV = 0xfffffffeffffffffULL;

            __device__ __constant__ static const uint64_t FR_ONE[4] = {0x00000001fffffffeULL, 0x5884b7fa00034802ULL, 0x998c4fefecbc4ff5ULL,
                                                                       0x1824b159acc5056fULL};

            __device__ __constant__ static const uint64_t FR_R2[4] = {0xc999e990f3f29c6dULL, 0x2b6cedcb87925c23ULL, 0x05d314967254398fULL,
                                                                      0x0748d9d99f59ff11ULL};

            __device__ __forceinline__ void mac(uint64_t a, uint64_t b, uint64_t c, uint64_t carry_in, uint64_t &lo_out, uint64_t &hi_out)
            {
                uint64_t lo = a * b;
                uint64_t hi = __umul64hi(a, b);
                lo += c;
                hi += (lo < c) ? 1ULL : 0ULL;
                lo += carry_in;
                hi += (lo < carry_in) ? 1ULL : 0ULL;
                lo_out = lo;
                hi_out = hi;
            }

            __device__ __forceinline__ uint64_t adc(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t &carry_out)
            {
                uint64_t r = a + b;
                carry_out = (r < a) ? 1ULL : 0ULL;
                r += carry_in;
                carry_out += (r < carry_in) ? 1ULL : 0ULL;
                return r;
            }

            __device__ __forceinline__ uint64_t sbb(uint64_t a, uint64_t b, uint64_t borrow_in, uint64_t &borrow_out)
            {
                uint64_t r = a - b;
                borrow_out = (a < b) ? 1ULL : 0ULL;
                uint64_t prev = r;
                r -= borrow_in;
                borrow_out += (prev < borrow_in) ? 1ULL : 0ULL;
                return r;
            }

            __device__ __forceinline__ bool fr_gte_p(const GpuFr &a)
            {
                for (int i = 3; i >= 0; --i)
                {
                    if (a.d[i] > FR_P[i])
                        return true;
                    if (a.d[i] < FR_P[i])
                        return false;
                }
                return true;
            }

            __device__ __forceinline__ GpuFr fr_add(const GpuFr &a, const GpuFr &b)
            {
                GpuFr r;
                uint64_t carry = 0;
                r.d[0] = adc(a.d[0], b.d[0], 0, carry);
                r.d[1] = adc(a.d[1], b.d[1], carry, carry);
                r.d[2] = adc(a.d[2], b.d[2], carry, carry);
                r.d[3] = adc(a.d[3], b.d[3], carry, carry);
                if (carry || fr_gte_p(r))
                {
                    uint64_t borrow = 0;
                    r.d[0] = sbb(r.d[0], FR_P[0], 0, borrow);
                    r.d[1] = sbb(r.d[1], FR_P[1], borrow, borrow);
                    r.d[2] = sbb(r.d[2], FR_P[2], borrow, borrow);
                    r.d[3] = sbb(r.d[3], FR_P[3], borrow, borrow);
                }
                return r;
            }

            __device__ __forceinline__ GpuFr fr_sub(const GpuFr &a, const GpuFr &b)
            {
                GpuFr r;
                uint64_t borrow = 0;
                r.d[0] = sbb(a.d[0], b.d[0], 0, borrow);
                r.d[1] = sbb(a.d[1], b.d[1], borrow, borrow);
                r.d[2] = sbb(a.d[2], b.d[2], borrow, borrow);
                r.d[3] = sbb(a.d[3], b.d[3], borrow, borrow);
                if (borrow)
                {
                    uint64_t carry = 0;
                    r.d[0] = adc(r.d[0], FR_P[0], 0, carry);
                    r.d[1] = adc(r.d[1], FR_P[1], carry, carry);
                    r.d[2] = adc(r.d[2], FR_P[2], carry, carry);
                    r.d[3] = adc(r.d[3], FR_P[3], carry, carry);
                }
                return r;
            }

            __device__ __forceinline__ GpuFr fr_mul(const GpuFr &a, const GpuFr &b)
            {
                uint64_t t[5] = {0, 0, 0, 0, 0};
                for (int i = 0; i < 4; ++i)
                {
                    uint64_t carry = 0;
                    uint64_t hi;
                    mac(a.d[0], b.d[i], t[0], carry, t[0], hi);
                    carry = hi;
                    mac(a.d[1], b.d[i], t[1], carry, t[1], hi);
                    carry = hi;
                    mac(a.d[2], b.d[i], t[2], carry, t[2], hi);
                    carry = hi;
                    mac(a.d[3], b.d[i], t[3], carry, t[3], hi);
                    t[4] += hi;

                    uint64_t m = t[0] * FR_INV;
                    mac(m, FR_P[0], t[0], 0, t[0], hi);
                    carry = hi;
                    mac(m, FR_P[1], t[1], carry, t[1], hi);
                    carry = hi;
                    mac(m, FR_P[2], t[2], carry, t[2], hi);
                    carry = hi;
                    mac(m, FR_P[3], t[3], carry, t[3], hi);
                    uint64_t new_t4 = t[4] + hi;
                    uint64_t t4_carry = (new_t4 < t[4]) ? 1ULL : 0ULL;

                    t[0] = t[1];
                    t[1] = t[2];
                    t[2] = t[3];
                    t[3] = new_t4;
                    t[4] = t4_carry;
                }

                GpuFr r = {{t[0], t[1], t[2], t[3]}};
                if (t[4] || fr_gte_p(r))
                {
                    uint64_t borrow = 0;
                    r.d[0] = sbb(r.d[0], FR_P[0], 0, borrow);
                    r.d[1] = sbb(r.d[1], FR_P[1], borrow, borrow);
                    r.d[2] = sbb(r.d[2], FR_P[2], borrow, borrow);
                    r.d[3] = sbb(r.d[3], FR_P[3], borrow, borrow);
                }
                return r;
            }

            __device__ __forceinline__ GpuFr fr_zero()
            {
                GpuFr r = {{0, 0, 0, 0}};
                return r;
            }

            __device__ __forceinline__ GpuFr fr_one()
            {
                GpuFr r = {{FR_ONE[0], FR_ONE[1], FR_ONE[2], FR_ONE[3]}};
                return r;
            }

            __device__ __forceinline__ GpuFr fr_sqr(const GpuFr &a) { return fr_mul(a, a); }

            __device__ __forceinline__ GpuFr fr_pow(const GpuFr &base, const uint64_t exp[4])
            {
                GpuFr result = fr_one();
                GpuFr cur = base;
                for (int limb = 0; limb < 4; ++limb)
                {
                    uint64_t e = exp[limb];
                    for (int bit = 0; bit < 64; ++bit)
                    {
                        if (e & 1ULL)
                        {
                            result = fr_mul(result, cur);
                        }
                        cur = fr_sqr(cur);
                        e >>= 1;
                    }
                }
                return result;
            }

            __device__ __forceinline__ GpuFr fr_inv(const GpuFr &a)
            {
                const uint64_t pm2[4] = {0xfffffffeffffffffULL, 0x53bda402fffe5bfeULL, 0x3339d80809a1d805ULL, 0x73eda753299d7d48ULL};
                return fr_pow(a, pm2);
            }

            __device__ __forceinline__ bool fr_is_zero(const GpuFr &a) { return a.d[0] == 0 && a.d[1] == 0 && a.d[2] == 0 && a.d[3] == 0; }

        }
    }
}
