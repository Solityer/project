#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/packed_field.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/protocol/quotients.hpp"

namespace gatzk::algebra {
namespace {

FieldElement dot_product_cpu(
    const std::vector<FieldElement>& lhs,
    const std::vector<FieldElement>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("cuda shim dot product size mismatch");
    }
    mcl::Fr sum;
    sum.clear();
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        mcl::Fr term;
        mcl::Fr::mul(term, lhs[i].native(), rhs[i].native());
        mcl::Fr::add(sum, sum, term);
    }
    return FieldElement::from_native(sum);
}

FieldElement dot_product_native_cpu(
    const std::vector<FieldElement>& lhs,
    const std::vector<mcl::Fr>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("cuda shim native dot product size mismatch");
    }
    mcl::Fr sum;
    sum.clear();
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        mcl::Fr term;
        mcl::Fr::mul(term, lhs[i].native(), rhs[i]);
        mcl::Fr::add(sum, sum, term);
    }
    return FieldElement::from_native(sum);
}

std::vector<mcl::Fr> unpack_native_field_elements_cpu(const PackedFieldBuffer& packed) {
    std::vector<mcl::Fr> out(packed.size());
    for (std::size_t i = 0; i < packed.size(); ++i) {
        out[i] = unpack_native_field_element(packed[i]);
    }
    return out;
}

std::vector<FieldElement> evaluate_with_weights_cpu(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const std::vector<mcl::Fr>& weights) {
    if (weights.size() != backend.domain()->size) {
        throw std::runtime_error("cuda shim evaluation weight size mismatch");
    }
    const auto& row_indices = backend.subset_row_indices(labels);
    std::vector<FieldElement> out(row_indices.size(), FieldElement::zero());
    for (std::size_t row = 0; row < row_indices.size(); ++row) {
        const auto* polynomial = backend.polynomials()[row_indices[row]].second;
        if (polynomial == nullptr) {
            throw std::runtime_error("cuda shim evaluation missing polynomial");
        }
        mcl::Fr sum;
        sum.clear();
        for (std::size_t column = 0; column < weights.size(); ++column) {
            mcl::Fr term;
            mcl::Fr::mul(term, polynomial->data[column].native(), weights[column]);
            mcl::Fr::add(sum, sum, term);
        }
        out[row] = FieldElement::from_native(sum);
    }
    return out;
}

std::vector<std::vector<FieldElement>> evaluate_with_rotations_cpu(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const std::vector<mcl::Fr>& representative_weights,
    const std::vector<std::size_t>& rotations) {
    if (representative_weights.size() != backend.domain()->size) {
        throw std::runtime_error("cuda shim rotated evaluation weight size mismatch");
    }
    const auto& row_indices = backend.subset_row_indices(labels);
    const auto domain_mask = backend.domain()->size - 1U;
    std::vector<std::vector<FieldElement>> out(
        rotations.size(),
        std::vector<FieldElement>(row_indices.size(), FieldElement::zero()));
    for (std::size_t point_index = 0; point_index < rotations.size(); ++point_index) {
        for (std::size_t row = 0; row < row_indices.size(); ++row) {
            const auto* polynomial = backend.polynomials()[row_indices[row]].second;
            if (polynomial == nullptr) {
                throw std::runtime_error("cuda shim rotated evaluation missing polynomial");
            }
            mcl::Fr sum;
            sum.clear();
            for (std::size_t column = 0; column < representative_weights.size(); ++column) {
                const auto rotated_column = (column + rotations[point_index]) & domain_mask;
                mcl::Fr term;
                mcl::Fr::mul(term, polynomial->data[rotated_column].native(), representative_weights[column]);
                mcl::Fr::add(sum, sum, term);
            }
            out[point_index][row] = FieldElement::from_native(sum);
        }
    }
    return out;
}

struct HostPackedEvaluationStorage {
    PackedFieldBuffer packed;
};

PackedEvaluationDeviceResult make_device_result_from_points(
    const std::vector<std::vector<FieldElement>>& values) {
    PackedEvaluationDeviceResult result;
    if (values.empty()) {
        return result;
    }
    const auto point_count = values.size();
    const auto row_count = values.front().size();
    std::vector<FieldElement> flat;
    flat.reserve(point_count * row_count);
    for (const auto& point_values : values) {
        if (point_values.size() != row_count) {
            throw std::runtime_error("cuda shim device result row size mismatch");
        }
        flat.insert(flat.end(), point_values.begin(), point_values.end());
    }

    auto storage = std::make_shared<HostPackedEvaluationStorage>();
    pack_field_elements_into(flat, &storage->packed);
    result.buffer.storage = storage;
    result.buffer.count = storage->packed.size();
    result.row_count = row_count;
    result.point_count = point_count;
    return result;
}

const HostPackedEvaluationStorage& require_host_storage(const PackedEvaluationDeviceResult& result) {
    const auto storage = std::static_pointer_cast<HostPackedEvaluationStorage>(result.buffer.storage);
    if (storage == nullptr) {
        throw std::runtime_error("cuda shim device buffer storage is missing");
    }
    return *storage;
}

std::vector<std::vector<FieldElement>> materialize_points_cpu(const PackedEvaluationDeviceResult& result) {
    if (result.empty()) {
        return {};
    }
    const auto& storage = require_host_storage(result);
    if (storage.packed.size() != result.row_count * result.point_count) {
        throw std::runtime_error("cuda shim device materialization size mismatch");
    }
    std::vector<std::vector<FieldElement>> out(
        result.point_count,
        std::vector<FieldElement>(result.row_count, FieldElement::zero()));
    for (std::size_t point_index = 0; point_index < result.point_count; ++point_index) {
        for (std::size_t row = 0; row < result.row_count; ++row) {
            out[point_index][row] = unpack_field_element(storage.packed[point_index * result.row_count + row]);
        }
    }
    return out;
}

}  // namespace

bool cuda_backend_runtime_available() {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

FieldElement dot_product_cuda(
    const std::vector<FieldElement>& lhs,
    const std::vector<FieldElement>& rhs) {
    return dot_product_cpu(lhs, rhs);
}

FieldElement dot_product_native_weights_cuda(
    const std::vector<FieldElement>& lhs,
    const PackedFieldBuffer& packed_rhs) {
    return dot_product_native_cpu(lhs, unpack_native_field_elements_cpu(packed_rhs));
}

std::vector<FieldElement> evaluate_with_packed_native_weights_cuda(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& weights) {
    return evaluate_with_weights_cpu(backend, labels, unpack_native_field_elements_cpu(weights));
}

PackedEvaluationDeviceResult evaluate_device_with_packed_native_weights_cuda(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& weights) {
    return make_device_result_from_points({evaluate_with_packed_native_weights_cuda(backend, labels, weights)});
}

std::vector<std::vector<FieldElement>> evaluate_with_packed_native_weight_rotations_cuda(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& representative_weights,
    const std::vector<std::size_t>& rotations) {
    return evaluate_with_rotations_cpu(
        backend,
        labels,
        unpack_native_field_elements_cpu(representative_weights),
        rotations);
}

PackedEvaluationDeviceResult evaluate_device_with_packed_native_weight_rotations_cuda(
    const PackedEvaluationBackend& backend,
    const std::vector<std::string>& labels,
    const PackedFieldBuffer& representative_weights,
    const std::vector<std::size_t>& rotations) {
    return make_device_result_from_points(
        evaluate_with_packed_native_weight_rotations_cuda(backend, labels, representative_weights, rotations));
}

std::vector<FieldElement> materialize_device_result_cuda(const PackedEvaluationDeviceResult& result) {
    const auto points = materialize_points_cpu(result);
    return points.empty() ? std::vector<FieldElement>{} : points.front();
}

std::vector<std::vector<FieldElement>> materialize_device_rotation_result_cuda(
    const PackedEvaluationDeviceResult& result) {
    return materialize_points_cpu(result);
}

}  // namespace gatzk::algebra

namespace gatzk::protocol {
namespace {

const std::vector<std::string>& cuda_fh_labels() {
    static const std::vector<std::string> labels = {
        "P_Q_tbl_feat",
        "P_Q_qry_feat",
        "P_Table_feat",
        "P_Query_feat",
        "P_m_feat",
        "P_R_feat",
        "P_H",
    };
    return labels;
}

const std::vector<std::string>& cuda_edge_labels() {
    static const std::vector<std::string> labels = {
        "P_Q_new_edge",
        "P_Q_edge_valid",
        "P_S",
        "P_Z",
        "P_M_edge",
        "P_Delta",
        "P_U",
        "P_alpha",
        "P_H_src_star_edge",
        "P_H_agg_star_edge",
        "P_w_psq",
        "P_T_psq_edge",
        "P_PSQ",
    };
    return labels;
}

const std::vector<std::string>& cuda_n_labels() {
    static const std::vector<std::string> labels = {
        "P_I",
        "P_Q_N",
        "P_E_src",
        "P_H_star",
        "P_Table_src",
        "P_m_src",
        "P_R_src_node",
        "P_E_dst",
        "P_M",
        "P_Sum",
        "P_inv",
        "P_H_agg_star",
        "P_Table_dst",
        "P_m_dst",
        "P_R_dst_node",
    };
    return labels;
}

const std::vector<std::string>& cuda_in_labels() {
    static const std::vector<std::string> labels = {
        "P_Q_proj_valid",
        "P_a_proj",
        "P_b_proj",
        "P_Acc_proj",
        "P_a_out",
        "P_b_out",
        "P_Acc_out",
    };
    return labels;
}

const std::vector<std::string>& cuda_d_labels() {
    static const std::vector<std::string> labels = {
        "P_Q_d_valid",
        "P_a_src",
        "P_b_src",
        "P_Acc_src",
        "P_a_dst",
        "P_b_dst",
        "P_Acc_dst",
        "P_a_star",
        "P_b_star",
        "P_Acc_star",
        "P_a_agg",
        "P_b_agg",
        "P_Acc_agg",
        "P_a_out",
        "P_b_out",
        "P_Acc_out",
    };
    return labels;
}

template <typename Reducer>
algebra::FieldElement evaluate_from_device_points(
    const std::vector<std::string>& labels,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const std::shared_ptr<algebra::RootOfUnityDomain>& domain,
    const algebra::FieldElement& z,
    Reducer&& reduce) {
    const auto materialized = algebra::materialize_device_rotation_result_cuda(evaluations);
    if (materialized.empty()) {
        throw std::runtime_error("cuda shim device evaluation is empty");
    }
    std::unordered_map<std::string, std::size_t> row_by_label;
    row_by_label.reserve(labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        row_by_label.emplace(labels[i], i);
    }
    const auto z_omega = z * domain->omega;
    return reduce(
        [&](const std::string& name, const algebra::FieldElement& point) -> algebra::FieldElement {
            const auto it = row_by_label.find(name);
            if (it == row_by_label.end()) {
                throw std::runtime_error("cuda shim quotient is missing label: " + name);
            }
            std::size_t point_index = 0;
            if (materialized.size() > 1) {
                if (point == z_omega) {
                    point_index = 1;
                } else if (!(point == z)) {
                    throw std::runtime_error("cuda shim quotient received unsupported evaluation point");
                }
            }
            return materialized.at(point_index).at(it->second);
        });
}

}  // namespace

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

__global__ void max_counter_state_kernel(
    const unsigned long long* s_max,
    const unsigned long long* q_new,
    unsigned long long* out,
    std::size_t count) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || count == 0) {
        return;
    }
    out[0] = s_max[0];
    for (std::size_t i = 1; i < count; ++i) {
        out[i] = q_new[i] != 0ULL ? s_max[i] : (out[i - 1] + s_max[i]);
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
        throw std::runtime_error("cuda histogram failed to allocate index buffer");
    }
    if (cudaMalloc(&device_counts, counts_bytes) != cudaSuccess) {
        cudaFree(device_indices);
        throw std::runtime_error("cuda histogram failed to allocate count buffer");
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

std::vector<algebra::FieldElement> build_max_counter_state_cuda(
    const std::vector<algebra::FieldElement>& s_max,
    const std::vector<algebra::FieldElement>& q_new) {
    if (s_max.size() != q_new.size()) {
        throw std::runtime_error("cuda max counter input size mismatch");
    }
    if (s_max.empty()) {
        return {};
    }

    std::vector<unsigned long long> host_s_max(s_max.size(), 0ULL);
    std::vector<unsigned long long> host_q_new(q_new.size(), 0ULL);
    for (std::size_t i = 0; i < s_max.size(); ++i) {
        host_s_max[i] = static_cast<unsigned long long>(s_max[i].value());
        host_q_new[i] = static_cast<unsigned long long>(q_new[i].value());
    }
    std::vector<unsigned long long> host_out(s_max.size(), 0ULL);

    unsigned long long* device_s_max = nullptr;
    unsigned long long* device_q_new = nullptr;
    unsigned long long* device_out = nullptr;
    const auto bytes = sizeof(unsigned long long) * host_s_max.size();
    if (cudaMalloc(&device_s_max, bytes) != cudaSuccess) {
        throw std::runtime_error("cuda max counter failed to allocate s_max buffer");
    }
    if (cudaMalloc(&device_q_new, bytes) != cudaSuccess) {
        cudaFree(device_s_max);
        throw std::runtime_error("cuda max counter failed to allocate q_new buffer");
    }
    if (cudaMalloc(&device_out, bytes) != cudaSuccess) {
        cudaFree(device_s_max);
        cudaFree(device_q_new);
        throw std::runtime_error("cuda max counter failed to allocate output buffer");
    }

    const auto cleanup = [&]() {
        cudaFree(device_s_max);
        cudaFree(device_q_new);
        cudaFree(device_out);
    };

    if (cudaMemcpy(device_s_max, host_s_max.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess
        || cudaMemcpy(device_q_new, host_q_new.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda max counter failed to upload inputs");
    }

    max_counter_state_kernel<<<1, 1>>>(device_s_max, device_q_new, device_out, host_s_max.size());
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda max counter kernel launch failed");
    }

    if (cudaMemcpy(host_out.data(), device_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cleanup();
        throw std::runtime_error("cuda max counter failed to download output");
    }
    cleanup();

    std::vector<algebra::FieldElement> out(host_out.size(), algebra::FieldElement::zero());
    for (std::size_t i = 0; i < host_out.size(); ++i) {
        out[i] = algebra::FieldElement(host_out[i]);
    }
    return out;
}

algebra::FieldElement evaluate_t_fh_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z) {
    return evaluate_from_device_points(
        cuda_fh_labels(),
        evaluations,
        context.domains.fh,
        z,
        [&](const EvalFn& eval) { return evaluate_t_fh(context, challenges, eval, z, nullptr); });
}

algebra::FieldElement evaluate_t_edge_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& witness_scalars,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z) {
    return evaluate_from_device_points(
        cuda_edge_labels(),
        evaluations,
        context.domains.edge,
        z,
        [&](const EvalFn& eval) { return evaluate_t_edge(context, challenges, witness_scalars, eval, z); });
}

algebra::FieldElement evaluate_t_n_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& witness_scalars,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z) {
    return evaluate_from_device_points(
        cuda_n_labels(),
        evaluations,
        context.domains.n,
        z,
        [&](const EvalFn& eval) { return evaluate_t_n(context, challenges, witness_scalars, eval, z); });
}

algebra::FieldElement evaluate_t_in_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& external_evaluations,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z) {
    return evaluate_from_device_points(
        cuda_in_labels(),
        evaluations,
        context.domains.in,
        z,
        [&](const EvalFn& eval) { return evaluate_t_in(context, challenges, external_evaluations, eval, z); });
}

algebra::FieldElement evaluate_t_d_device_cuda(
    const ProtocolContext& context,
    const std::map<std::string, algebra::FieldElement>& challenges,
    const std::map<std::string, algebra::FieldElement>& external_evaluations,
    const algebra::PackedEvaluationDeviceResult& evaluations,
    const algebra::FieldElement& z) {
    return evaluate_from_device_points(
        cuda_d_labels(),
        evaluations,
        context.domains.d,
        z,
        [&](const EvalFn& eval) { return evaluate_t_d(context, challenges, external_evaluations, eval, z); });
}

}  // namespace gatzk::protocol
