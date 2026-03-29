#include "gatzk/protocol/quotients.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "../algebra/cuda_field.cuh"

namespace gatzk::protocol {
#if GATZK_ENABLE_CUDA_BACKEND
namespace {

using algebra::FieldElement;
using algebra::PackedEvaluationDeviceResult;
using algebra::PackedFieldElement;
namespace cuda_detail = algebra::cuda_detail;

cuda_detail::DevicePackedFieldArena& scalar_output_arena() {
    static cuda_detail::DevicePackedFieldArena arena;
    return arena;
}

PackedFieldElement pack_scalar(const FieldElement& value) {
    const auto packed = algebra::pack_field_elements(std::vector<FieldElement>{value});
    return packed[0];
}

FieldElement copy_back_scalar(const std::shared_ptr<void>& storage, const std::string& trace_label) {
    const auto packed = cuda_detail::copy_back_decoded(storage, 1, trace_label, &scalar_output_arena());
    return algebra::unpack_field_element(packed[0]);
}

template <typename LaunchFn>
FieldElement run_scalar_kernel(const std::string& trace_label, LaunchFn&& launch) {
    auto storage = scalar_output_arena().acquire(1);
    auto* out = static_cast<PackedFieldElement*>(storage.get());

    cudaEvent_t start;
    cudaEvent_t stop;
    const bool trace = cuda_detail::cuda_trace_enabled();
    if (trace) {
        cuda_detail::cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
        cuda_detail::cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
        cuda_detail::cuda_check(cudaEventRecord(start), "cudaEventRecord(start)");
    }

    launch(out);
    cuda_detail::cuda_check(cudaGetLastError(), "quotient kernel launch");
    cuda_detail::cuda_check(cudaDeviceSynchronize(), "quotient kernel sync");

    if (trace) {
        cuda_detail::cuda_check(cudaEventRecord(stop), "cudaEventRecord(stop)");
        cuda_detail::cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
        float milliseconds = 0.0f;
        cuda_detail::cuda_check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        cuda_detail::cuda_trace(trace_label + "_kernel", milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return copy_back_scalar(storage, trace_label);
}

__device__ __forceinline__ PackedFieldElement encode_scalar(const PackedFieldElement& raw) {
    return cuda_detail::montgomery_mul(raw, cuda_detail::field_r2());
}

__device__ __forceinline__ PackedFieldElement eval_at(
    const PackedFieldElement* evaluations,
    std::size_t row_count,
    std::size_t point_index,
    std::size_t row_index) {
    return evaluations[point_index * row_count + row_index];
}

__device__ __forceinline__ PackedFieldElement add(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    return cuda_detail::field_add_mod(lhs, rhs);
}

__device__ __forceinline__ PackedFieldElement sub(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    return cuda_detail::field_sub_mod(lhs, rhs);
}

__device__ __forceinline__ PackedFieldElement mul(
    const PackedFieldElement& lhs,
    const PackedFieldElement& rhs) {
    return cuda_detail::montgomery_mul(lhs, rhs);
}

__device__ PackedFieldElement pow_small(PackedFieldElement base, unsigned exponent) {
    auto out = encode_scalar(cuda_detail::field_one());
    for (unsigned i = 0; i < exponent; ++i) {
        out = mul(out, base);
    }
    return out;
}

__device__ PackedFieldElement c_lookup_1_device(
    const PackedFieldElement* evaluations,
    std::size_t row_count,
    std::size_t row_r,
    std::size_t row_table,
    std::size_t row_query,
    std::size_t row_q_tbl,
    std::size_t row_m,
    std::size_t row_q_qry,
    const PackedFieldElement& beta) {
    const auto r_z = eval_at(evaluations, row_count, 0, row_r);
    const auto r_omega = eval_at(evaluations, row_count, 1, row_r);
    const auto table_z = eval_at(evaluations, row_count, 0, row_table);
    const auto query_z = eval_at(evaluations, row_count, 0, row_query);
    const auto q_tbl_z = eval_at(evaluations, row_count, 0, row_q_tbl);
    const auto m_z = eval_at(evaluations, row_count, 0, row_m);
    const auto q_qry_z = eval_at(evaluations, row_count, 0, row_q_qry);

    return add(
        sub(
            mul(
                mul(sub(r_omega, r_z), add(table_z, beta)),
                add(query_z, beta)),
            mul(mul(q_tbl_z, m_z), add(query_z, beta))),
        mul(q_qry_z, add(table_z, beta)));
}

__global__ void quotient_fh_kernel(
    const PackedFieldElement* evaluations,
    std::size_t row_count,
    PackedFieldElement alpha,
    PackedFieldElement beta_feat,
    PackedFieldElement l0,
    PackedFieldElement l_last,
    PackedFieldElement zero_inv,
    PackedFieldElement* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    alpha = encode_scalar(alpha);
    beta_feat = encode_scalar(beta_feat);
    l0 = encode_scalar(l0);
    l_last = encode_scalar(l_last);
    zero_inv = encode_scalar(zero_inv);
    const auto r_feat_z = eval_at(evaluations, row_count, 0, 0);
    const auto lookup = c_lookup_1_device(evaluations, row_count, 0, 1, 2, 3, 4, 5, beta_feat);
    auto numerator = mul(l0, r_feat_z);
    numerator = add(numerator, mul(alpha, lookup));
    numerator = add(numerator, mul(mul(alpha, alpha), mul(l_last, r_feat_z)));
    out[0] = mul(numerator, zero_inv);
}

__global__ void quotient_in_kernel(
    const PackedFieldElement* evaluations,
    std::size_t row_count,
    PackedFieldElement alpha,
    PackedFieldElement mu_proj,
    PackedFieldElement l0,
    PackedFieldElement ld,
    PackedFieldElement zero_inv,
    PackedFieldElement* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    alpha = encode_scalar(alpha);
    mu_proj = encode_scalar(mu_proj);
    l0 = encode_scalar(l0);
    ld = encode_scalar(ld);
    zero_inv = encode_scalar(zero_inv);
    const auto one = encode_scalar(cuda_detail::field_one());

    const auto q_valid = eval_at(evaluations, row_count, 0, 0);
    const auto a_proj = eval_at(evaluations, row_count, 0, 1);
    const auto b_proj = eval_at(evaluations, row_count, 0, 2);
    const auto acc_z = eval_at(evaluations, row_count, 0, 3);
    const auto acc_omega = eval_at(evaluations, row_count, 1, 3);

    const auto c0 = mul(l0, acc_z);
    const auto c1 = mul(q_valid, sub(sub(acc_omega, acc_z), mul(a_proj, b_proj)));
    const auto c2 = mul(sub(one, q_valid), sub(acc_omega, acc_z));
    const auto c3 = mul(ld, sub(acc_z, mu_proj));

    auto numerator = cuda_detail::field_zero();
    auto add_power = [&](unsigned exponent, const PackedFieldElement& term) {
        numerator = add(numerator, mul(pow_small(alpha, exponent), term));
    };
    add_power(30, c0);
    add_power(31, c1);
    add_power(32, c2);
    add_power(33, c3);
    out[0] = mul(numerator, zero_inv);
}

__global__ void quotient_d_kernel(
    const PackedFieldElement* evaluations,
    std::size_t row_count,
    PackedFieldElement alpha,
    PackedFieldElement mu_src,
    PackedFieldElement mu_dst,
    PackedFieldElement mu_star,
    PackedFieldElement mu_agg,
    PackedFieldElement mu_out,
    PackedFieldElement l0,
    PackedFieldElement ld,
    PackedFieldElement zero_inv,
    PackedFieldElement* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    alpha = encode_scalar(alpha);
    mu_src = encode_scalar(mu_src);
    mu_dst = encode_scalar(mu_dst);
    mu_star = encode_scalar(mu_star);
    mu_agg = encode_scalar(mu_agg);
    mu_out = encode_scalar(mu_out);
    l0 = encode_scalar(l0);
    ld = encode_scalar(ld);
    zero_inv = encode_scalar(zero_inv);
    const auto one = encode_scalar(cuda_detail::field_one());

    const auto q_valid = eval_at(evaluations, row_count, 0, 0);
    const auto q_invalid = sub(one, q_valid);
    auto c0 = [&](std::size_t acc) { return mul(l0, eval_at(evaluations, row_count, 0, acc)); };
    auto c1 = [&](std::size_t a, std::size_t b, std::size_t acc) {
        return mul(
            q_valid,
            sub(
                sub(eval_at(evaluations, row_count, 1, acc), eval_at(evaluations, row_count, 0, acc)),
                mul(eval_at(evaluations, row_count, 0, a), eval_at(evaluations, row_count, 0, b))));
    };
    auto c2 = [&](std::size_t acc) {
        return mul(
            q_invalid,
            sub(eval_at(evaluations, row_count, 1, acc), eval_at(evaluations, row_count, 0, acc)));
    };
    auto c3 = [&](std::size_t acc, const PackedFieldElement& mu) {
        return mul(ld, sub(eval_at(evaluations, row_count, 0, acc), mu));
    };

    PackedFieldElement numerator = cuda_detail::field_zero();
    auto alpha_power = pow_small(alpha, 34);
    auto add_term = [&](const PackedFieldElement& term) {
        numerator = add(numerator, mul(alpha_power, term));
        alpha_power = mul(alpha_power, alpha);
    };

    add_term(c0(3));
    add_term(c1(1, 2, 3));
    add_term(c2(3));
    add_term(c3(3, mu_src));
    add_term(c0(6));
    add_term(c1(4, 5, 6));
    add_term(c2(6));
    add_term(c3(6, mu_dst));
    add_term(c0(9));
    add_term(c1(7, 8, 9));
    add_term(c2(9));
    add_term(c3(9, mu_star));
    add_term(c0(12));
    add_term(c1(10, 11, 12));
    add_term(c2(12));
    add_term(c3(12, mu_agg));
    add_term(c0(15));
    add_term(c1(13, 14, 15));
    add_term(c2(15));
    add_term(c3(15, mu_out));
    out[0] = mul(numerator, zero_inv);
}

__global__ void quotient_n_kernel(
    const PackedFieldElement* evaluations,
    std::size_t row_count,
    PackedFieldElement alpha,
    PackedFieldElement beta_src,
    PackedFieldElement eta_src,
    PackedFieldElement beta_dst,
    PackedFieldElement eta_dst,
    PackedFieldElement s_src,
    PackedFieldElement s_dst,
    PackedFieldElement l0,
    PackedFieldElement l_last,
    PackedFieldElement zero_inv,
    PackedFieldElement* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    alpha = encode_scalar(alpha);
    beta_src = encode_scalar(beta_src);
    eta_src = encode_scalar(eta_src);
    beta_dst = encode_scalar(beta_dst);
    eta_dst = encode_scalar(eta_dst);
    s_src = encode_scalar(s_src);
    s_dst = encode_scalar(s_dst);
    l0 = encode_scalar(l0);
    l_last = encode_scalar(l_last);
    zero_inv = encode_scalar(zero_inv);
    const auto one = encode_scalar(cuda_detail::field_one());

    const auto q_n = eval_at(evaluations, row_count, 0, 1);
    const auto q_invalid = sub(one, q_n);
    const auto eta_src_2 = mul(eta_src, eta_src);
    const auto eta_dst_2 = mul(eta_dst, eta_dst);
    const auto eta_dst_3 = mul(eta_dst_2, eta_dst);
    const auto eta_dst_4 = mul(eta_dst_3, eta_dst);
    const auto eta_dst_5 = mul(eta_dst_4, eta_dst);

    const auto c_inv_0 = mul(q_n, sub(mul(eval_at(evaluations, row_count, 0, 9), eval_at(evaluations, row_count, 0, 10)), one));
    const auto c_src_node_0 = mul(l0, eval_at(evaluations, row_count, 0, 6));
    const auto c_src_node_1 = mul(
        q_n,
        sub(
            mul(
                sub(eval_at(evaluations, row_count, 1, 6), eval_at(evaluations, row_count, 0, 6)),
                add(eval_at(evaluations, row_count, 0, 4), beta_src)),
            eval_at(evaluations, row_count, 0, 5)));
    const auto c_src_node_2 = mul(q_invalid, sub(eval_at(evaluations, row_count, 1, 6), eval_at(evaluations, row_count, 0, 6)));
    const auto c_src_node_3 = mul(l_last, sub(eval_at(evaluations, row_count, 0, 6), s_src));
    const auto c_src_node_bind = sub(
        eval_at(evaluations, row_count, 0, 4),
        add(
            add(eval_at(evaluations, row_count, 0, 0), mul(eta_src, eval_at(evaluations, row_count, 0, 2))),
            mul(eta_src_2, eval_at(evaluations, row_count, 0, 3))));

    const auto c_dst_node_0 = mul(l0, eval_at(evaluations, row_count, 0, 14));
    const auto c_dst_node_1 = mul(
        q_n,
        sub(
            mul(
                sub(eval_at(evaluations, row_count, 1, 14), eval_at(evaluations, row_count, 0, 14)),
                add(eval_at(evaluations, row_count, 0, 12), beta_dst)),
            eval_at(evaluations, row_count, 0, 13)));
    const auto c_dst_node_2 = mul(q_invalid, sub(eval_at(evaluations, row_count, 1, 14), eval_at(evaluations, row_count, 0, 14)));
    const auto c_dst_node_3 = mul(l_last, sub(eval_at(evaluations, row_count, 0, 14), s_dst));
    const auto c_dst_node_bind = sub(
        eval_at(evaluations, row_count, 0, 12),
        add(
            add(
                add(
                    add(
                        add(eval_at(evaluations, row_count, 0, 0), mul(eta_dst, eval_at(evaluations, row_count, 0, 7))),
                        mul(eta_dst_2, eval_at(evaluations, row_count, 0, 8))),
                    mul(eta_dst_3, eval_at(evaluations, row_count, 0, 9))),
                mul(eta_dst_4, eval_at(evaluations, row_count, 0, 10))),
            mul(eta_dst_5, eval_at(evaluations, row_count, 0, 11))));

    PackedFieldElement numerator = cuda_detail::field_zero();
    auto add_power = [&](unsigned exponent, const PackedFieldElement& term) {
        numerator = add(numerator, mul(pow_small(alpha, exponent), term));
    };
    add_power(29, c_inv_0);
    add_power(60, c_src_node_0);
    add_power(61, c_src_node_1);
    add_power(62, c_src_node_2);
    add_power(63, c_src_node_3);
    add_power(64, c_src_node_bind);
    add_power(65, c_dst_node_0);
    add_power(66, c_dst_node_1);
    add_power(67, c_dst_node_2);
    add_power(68, c_dst_node_3);
    add_power(69, c_dst_node_bind);
    out[0] = mul(numerator, zero_inv);
}

__global__ void quotient_edge_kernel(
    const PackedFieldElement* evaluations,
    std::size_t row_count,
    PackedFieldElement alpha,
    PackedFieldElement beta_src,
    PackedFieldElement beta_dst,
    PackedFieldElement eta_src,
    PackedFieldElement eta_dst,
    PackedFieldElement beta_l,
    PackedFieldElement eta_l,
    PackedFieldElement beta_r,
    PackedFieldElement beta_exp,
    PackedFieldElement eta_exp,
    PackedFieldElement s_src,
    PackedFieldElement s_dst,
    PackedFieldElement l0,
    PackedFieldElement l_last,
    PackedFieldElement zero_inv,
    PackedFieldElement* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    alpha = encode_scalar(alpha);
    beta_src = encode_scalar(beta_src);
    beta_dst = encode_scalar(beta_dst);
    eta_src = encode_scalar(eta_src);
    eta_dst = encode_scalar(eta_dst);
    beta_l = encode_scalar(beta_l);
    eta_l = encode_scalar(eta_l);
    beta_r = encode_scalar(beta_r);
    beta_exp = encode_scalar(beta_exp);
    eta_exp = encode_scalar(eta_exp);
    s_src = encode_scalar(s_src);
    s_dst = encode_scalar(s_dst);
    l0 = encode_scalar(l0);
    l_last = encode_scalar(l_last);
    zero_inv = encode_scalar(zero_inv);
    const auto one = encode_scalar(cuda_detail::field_one());

    const auto eta_src_2 = mul(eta_src, eta_src);
    const auto eta_dst_2 = mul(eta_dst, eta_dst);
    const auto eta_dst_3 = mul(eta_dst_2, eta_dst);
    const auto eta_dst_4 = mul(eta_dst_3, eta_dst);
    const auto eta_dst_5 = mul(eta_dst_4, eta_dst);
    const auto eta_l_2 = mul(eta_l, eta_l);

    const auto c_src_0 = mul(l0, eval_at(evaluations, row_count, 0, 0));
    const auto c_src_1 = mul(
        eval_at(evaluations, row_count, 0, 2),
        sub(
            mul(
                sub(eval_at(evaluations, row_count, 1, 0), eval_at(evaluations, row_count, 0, 0)),
                add(eval_at(evaluations, row_count, 0, 1), beta_src)),
            one));
    const auto c_src_2 = mul(
        sub(one, eval_at(evaluations, row_count, 0, 2)),
        sub(eval_at(evaluations, row_count, 1, 0), eval_at(evaluations, row_count, 0, 0)));
    const auto c_src_3 = mul(l_last, sub(eval_at(evaluations, row_count, 0, 0), s_src));
    const auto c_src_bind = sub(
        eval_at(evaluations, row_count, 0, 1),
        add(
            add(eval_at(evaluations, row_count, 0, 51), mul(eta_src, eval_at(evaluations, row_count, 0, 3))),
            mul(eta_src_2, eval_at(evaluations, row_count, 0, 4))));

    const auto c_dst_0 = mul(l0, eval_at(evaluations, row_count, 0, 5));
    const auto c_dst_1 = mul(
        eval_at(evaluations, row_count, 0, 7),
        sub(
            mul(
                sub(eval_at(evaluations, row_count, 1, 5), eval_at(evaluations, row_count, 0, 5)),
                add(eval_at(evaluations, row_count, 0, 6), beta_dst)),
            one));
    const auto c_dst_2 = mul(
        sub(one, eval_at(evaluations, row_count, 0, 7)),
        sub(eval_at(evaluations, row_count, 1, 5), eval_at(evaluations, row_count, 0, 5)));
    const auto c_dst_3 = mul(l_last, sub(eval_at(evaluations, row_count, 0, 5), s_dst));
    const auto c_dst_bind = sub(
        eval_at(evaluations, row_count, 0, 6),
        add(
            add(
                add(
                    add(
                        add(eval_at(evaluations, row_count, 0, 52), mul(eta_dst, eval_at(evaluations, row_count, 0, 8))),
                        mul(eta_dst_2, eval_at(evaluations, row_count, 0, 9))),
                    mul(eta_dst_3, eval_at(evaluations, row_count, 0, 10))),
                mul(eta_dst_4, eval_at(evaluations, row_count, 0, 11))),
            mul(eta_dst_5, eval_at(evaluations, row_count, 0, 12))));

    const auto c_l_0 = mul(l0, eval_at(evaluations, row_count, 0, 13));
    const auto c_l_1 = c_lookup_1_device(evaluations, row_count, 13, 14, 15, 16, 17, 18, beta_l);
    const auto c_l_2 = mul(l_last, eval_at(evaluations, row_count, 0, 13));
    const auto c_l_bind_tbl = sub(eval_at(evaluations, row_count, 0, 14), add(eval_at(evaluations, row_count, 0, 46), mul(eta_l, eval_at(evaluations, row_count, 0, 47))));
    const auto c_l_bind_qry =
        sub(eval_at(evaluations, row_count, 0, 15), add(eval_at(evaluations, row_count, 0, 53), mul(eta_l, eval_at(evaluations, row_count, 0, 33))));

    const auto c_r_0 = mul(l0, eval_at(evaluations, row_count, 0, 19));
    const auto c_r_1 = c_lookup_1_device(evaluations, row_count, 19, 20, 21, 22, 23, 24, beta_r);
    const auto c_r_2 = mul(l_last, eval_at(evaluations, row_count, 0, 19));
    const auto c_r_bind_tbl = sub(eval_at(evaluations, row_count, 0, 20), eval_at(evaluations, row_count, 0, 48));
    const auto c_r_bind_qry = sub(eval_at(evaluations, row_count, 0, 21), eval_at(evaluations, row_count, 0, 32));

    const auto c_exp_0 = mul(l0, eval_at(evaluations, row_count, 0, 25));
    const auto c_exp_1 = c_lookup_1_device(evaluations, row_count, 25, 26, 27, 28, 29, 30, beta_exp);
    const auto c_exp_2 = mul(l_last, eval_at(evaluations, row_count, 0, 25));
    const auto c_exp_bind_tbl = sub(eval_at(evaluations, row_count, 0, 26), add(eval_at(evaluations, row_count, 0, 49), mul(eta_exp, eval_at(evaluations, row_count, 0, 50))));
    const auto c_exp_bind_qry = sub(eval_at(evaluations, row_count, 0, 27), add(eval_at(evaluations, row_count, 0, 32), mul(eta_exp, eval_at(evaluations, row_count, 0, 40))));

    const auto c_max_0 = add(sub(eval_at(evaluations, row_count, 0, 32), eval_at(evaluations, row_count, 0, 9)), eval_at(evaluations, row_count, 0, 33));
    const auto c_max_1 = mul(eval_at(evaluations, row_count, 0, 34), sub(eval_at(evaluations, row_count, 0, 34), one));
    const auto c_max_2 = mul(eval_at(evaluations, row_count, 0, 34), eval_at(evaluations, row_count, 0, 32));
    const auto c_max_3 = mul(l0, sub(eval_at(evaluations, row_count, 0, 35), eval_at(evaluations, row_count, 0, 34)));
    const auto c_max_4 = add(
        mul(
            eval_at(evaluations, row_count, 1, 36),
            sub(
                eval_at(evaluations, row_count, 1, 35),
                add(
                    mul(eval_at(evaluations, row_count, 1, 37), eval_at(evaluations, row_count, 1, 34)),
                    mul(
                        sub(one, eval_at(evaluations, row_count, 1, 37)),
                        add(eval_at(evaluations, row_count, 0, 35), eval_at(evaluations, row_count, 1, 34)))))),
        mul(
            sub(one, eval_at(evaluations, row_count, 1, 36)),
            sub(eval_at(evaluations, row_count, 1, 35), eval_at(evaluations, row_count, 0, 35))));
    const auto c_max_5 = mul(eval_at(evaluations, row_count, 0, 38), sub(eval_at(evaluations, row_count, 0, 35), one));

    const auto c_inv_1 = sub(eval_at(evaluations, row_count, 0, 39), mul(eval_at(evaluations, row_count, 0, 40), eval_at(evaluations, row_count, 0, 11)));
    const auto c_vhat_0 = sub(eval_at(evaluations, row_count, 0, 42), mul(eval_at(evaluations, row_count, 0, 39), eval_at(evaluations, row_count, 0, 4)));
    const auto c_psq_0 = mul(l0, sub(eval_at(evaluations, row_count, 0, 43), eval_at(evaluations, row_count, 0, 44)));
    const auto c_psq_1 = sub(
        eval_at(evaluations, row_count, 1, 43),
        add(
            mul(eval_at(evaluations, row_count, 1, 37), eval_at(evaluations, row_count, 1, 44)),
            mul(
                sub(one, eval_at(evaluations, row_count, 1, 37)),
                add(eval_at(evaluations, row_count, 0, 43), eval_at(evaluations, row_count, 1, 44)))));
    const auto c_psq_2 = mul(eval_at(evaluations, row_count, 0, 38), sub(eval_at(evaluations, row_count, 0, 43), eval_at(evaluations, row_count, 0, 45)));

    PackedFieldElement numerator = cuda_detail::field_zero();
    auto add_power = [&](unsigned exponent, const PackedFieldElement& term) {
        auto coeff = pow_small(alpha, exponent);
        numerator = add(numerator, mul(coeff, term));
    };
    add_power(3, c_src_0);
    add_power(4, c_src_1);
    add_power(5, c_src_2);
    add_power(6, c_dst_0);
    add_power(7, c_dst_1);
    add_power(8, c_dst_2);
    add_power(9, c_l_0);
    add_power(10, c_l_1);
    add_power(11, c_l_2);
    add_power(12, c_r_0);
    add_power(13, c_r_1);
    add_power(14, c_r_2);
    add_power(15, c_exp_0);
    add_power(16, c_exp_1);
    add_power(17, c_exp_2);
    add_power(18, c_max_0);
    add_power(19, c_max_1);
    add_power(20, c_max_2);
    add_power(21, c_max_3);
    add_power(22, c_max_4);
    add_power(23, c_max_5);
    add_power(24, c_inv_1);
    add_power(25, c_vhat_0);
    add_power(26, c_psq_0);
    add_power(27, c_psq_1);
    add_power(28, c_psq_2);
    add_power(54, c_l_bind_tbl);
    add_power(55, c_l_bind_qry);
    add_power(56, c_r_bind_tbl);
    add_power(57, c_r_bind_qry);
    add_power(58, c_exp_bind_tbl);
    add_power(59, c_exp_bind_qry);
    add_power(70, c_src_3);
    add_power(71, c_src_bind);
    add_power(72, c_dst_3);
    add_power(73, c_dst_bind);
    out[0] = mul(numerator, zero_inv);
}

FieldElement run_quotient_fh(
    const PackedEvaluationDeviceResult& evaluations,
    const std::map<std::string, FieldElement>& challenges,
    const FieldElement& l0,
    const FieldElement& l_last,
    const FieldElement& zero_inv) {
    return run_scalar_kernel(
        "quotient_fh",
        [&](PackedFieldElement* device_out) {
            quotient_fh_kernel<<<1, 1>>>(
                static_cast<const PackedFieldElement*>(evaluations.buffer.storage.get()),
                evaluations.row_count,
                pack_scalar(challenges.at("alpha_quot")),
                pack_scalar(challenges.at("beta_feat")),
                pack_scalar(l0),
                pack_scalar(l_last),
                pack_scalar(zero_inv),
                device_out);
        });
}

FieldElement run_quotient_in(
    const PackedEvaluationDeviceResult& evaluations,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& external_evaluations,
    const FieldElement& l0,
    const FieldElement& ld,
    const FieldElement& zero_inv) {
    return run_scalar_kernel(
        "quotient_in",
        [&](PackedFieldElement* device_out) {
            quotient_in_kernel<<<1, 1>>>(
                static_cast<const PackedFieldElement*>(evaluations.buffer.storage.get()),
                evaluations.row_count,
                pack_scalar(challenges.at("alpha_quot")),
                pack_scalar(external_evaluations.at("mu_proj")),
                pack_scalar(l0),
                pack_scalar(ld),
                pack_scalar(zero_inv),
                device_out);
        });
}

FieldElement run_quotient_d(
    const PackedEvaluationDeviceResult& evaluations,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& external_evaluations,
    const FieldElement& l0,
    const FieldElement& ld,
    const FieldElement& zero_inv) {
    return run_scalar_kernel(
        "quotient_d",
        [&](PackedFieldElement* device_out) {
            quotient_d_kernel<<<1, 1>>>(
                static_cast<const PackedFieldElement*>(evaluations.buffer.storage.get()),
                evaluations.row_count,
                pack_scalar(challenges.at("alpha_quot")),
                pack_scalar(external_evaluations.at("mu_src")),
                pack_scalar(external_evaluations.at("mu_dst")),
                pack_scalar(external_evaluations.at("mu_star")),
                pack_scalar(external_evaluations.at("mu_agg")),
                pack_scalar(external_evaluations.at("mu_Y_lin")),
                pack_scalar(l0),
                pack_scalar(ld),
                pack_scalar(zero_inv),
                device_out);
        });
}

FieldElement run_quotient_n(
    const PackedEvaluationDeviceResult& evaluations,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& witness_scalars,
    const FieldElement& l0,
    const FieldElement& l_last,
    const FieldElement& zero_inv) {
    return run_scalar_kernel(
        "quotient_n",
        [&](PackedFieldElement* device_out) {
            quotient_n_kernel<<<1, 1>>>(
                static_cast<const PackedFieldElement*>(evaluations.buffer.storage.get()),
                evaluations.row_count,
                pack_scalar(challenges.at("alpha_quot")),
                pack_scalar(challenges.at("beta_src")),
                pack_scalar(challenges.at("eta_src")),
                pack_scalar(challenges.at("beta_dst")),
                pack_scalar(challenges.at("eta_dst")),
                pack_scalar(witness_scalars.at("S_src")),
                pack_scalar(witness_scalars.at("S_dst")),
                pack_scalar(l0),
                pack_scalar(l_last),
                pack_scalar(zero_inv),
                device_out);
        });
}

FieldElement run_quotient_edge(
    const PackedEvaluationDeviceResult& evaluations,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& witness_scalars,
    const FieldElement& l0,
    const FieldElement& l_last,
    const FieldElement& zero_inv) {
    return run_scalar_kernel(
        "quotient_edge",
        [&](PackedFieldElement* device_out) {
            quotient_edge_kernel<<<1, 1>>>(
                static_cast<const PackedFieldElement*>(evaluations.buffer.storage.get()),
                evaluations.row_count,
                pack_scalar(challenges.at("alpha_quot")),
                pack_scalar(challenges.at("beta_src")),
                pack_scalar(challenges.at("beta_dst")),
                pack_scalar(challenges.at("eta_src")),
                pack_scalar(challenges.at("eta_dst")),
                pack_scalar(challenges.at("beta_L")),
                pack_scalar(challenges.at("eta_L")),
                pack_scalar(challenges.at("beta_R")),
                pack_scalar(challenges.at("beta_exp")),
                pack_scalar(challenges.at("eta_exp")),
                pack_scalar(witness_scalars.at("S_src")),
                pack_scalar(witness_scalars.at("S_dst")),
                pack_scalar(l0),
                pack_scalar(l_last),
                pack_scalar(zero_inv),
                device_out);
        });
}

}  // namespace

FieldElement evaluate_t_fh_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const PackedEvaluationDeviceResult& evaluations,
    const FieldElement& z) {
    return run_quotient_fh(
        evaluations,
        challenges,
        context.domains.fh->lagrange_basis_eval(0, z),
        context.domains.fh->lagrange_basis_eval(context.domains.fh->size - 1, z),
        context.domains.fh->zero_polynomial_eval(z).inv());
}

FieldElement evaluate_t_edge_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& witness_scalars,
    const PackedEvaluationDeviceResult& evaluations,
    const FieldElement& z) {
    return run_quotient_edge(
        evaluations,
        challenges,
        witness_scalars,
        context.domains.edge->lagrange_basis_eval(0, z),
        context.domains.edge->lagrange_basis_eval(context.domains.edge->size - 1, z),
        context.domains.edge->zero_polynomial_eval(z).inv());
}

FieldElement evaluate_t_n_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& witness_scalars,
    const PackedEvaluationDeviceResult& evaluations,
    const FieldElement& z) {
    return run_quotient_n(
        evaluations,
        challenges,
        witness_scalars,
        context.domains.n->lagrange_basis_eval(0, z),
        context.domains.n->lagrange_basis_eval(context.domains.n->size - 1, z),
        context.domains.n->zero_polynomial_eval(z).inv());
}

FieldElement evaluate_t_in_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& external_evaluations,
    const PackedEvaluationDeviceResult& evaluations,
    const FieldElement& z) {
    return run_quotient_in(
        evaluations,
        challenges,
        external_evaluations,
        context.domains.in->lagrange_basis_eval(0, z),
        context.domains.in->lagrange_basis_eval(context.local.num_features, z),
        context.domains.in->zero_polynomial_eval(z).inv());
}

FieldElement evaluate_t_d_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, FieldElement>& challenges,
    const std::map<std::string, FieldElement>& external_evaluations,
    const PackedEvaluationDeviceResult& evaluations,
    const FieldElement& z) {
    return run_quotient_d(
        evaluations,
        challenges,
        external_evaluations,
        context.domains.d->lagrange_basis_eval(0, z),
        context.domains.d->lagrange_basis_eval(context.model.a_src.size(), z),
        context.domains.d->zero_polynomial_eval(z).inv());
}

#endif
}  // namespace gatzk::protocol
