#include "gatzk/algebra/eval_backend.hpp"

#include <algorithm>
#include <cstdint>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::algebra
{

#if GATZK_ENABLE_CUDA_BACKEND
    // 使用 CUDA 计算带权重的打包求值（返回主机端结果）
    std::vector<FieldElement> evaluate_with_packed_native_weights_cuda(
        const PackedEvaluationBackend& backend,
        const std::vector<std::string>& labels,
        const PackedFieldBuffer& weights);
    PackedEvaluationDeviceResult evaluate_device_with_packed_native_weights_cuda(
        const PackedEvaluationBackend& backend,
        const std::vector<std::string>& labels,
        const PackedFieldBuffer& weights);

    // 使用 CUDA 计算带权重旋转的打包求值（返回主机端结果）
    std::vector<std::vector<FieldElement>> evaluate_with_packed_native_weight_rotations_cuda(
        const PackedEvaluationBackend& backend,
        const std::vector<std::string>& labels,
        const PackedFieldBuffer& representative_weights,
        const std::vector<std::size_t>& rotations);
    PackedEvaluationDeviceResult evaluate_device_with_packed_native_weight_rotations_cuda(
        const PackedEvaluationBackend& backend,
        const std::vector<std::string>& labels,
        const PackedFieldBuffer& representative_weights,
        const std::vector<std::size_t>& rotations);
    std::vector<FieldElement> materialize_device_result_cuda(const PackedEvaluationDeviceResult& result);
    std::vector<std::vector<FieldElement>> materialize_device_rotation_result_cuda(
        const PackedEvaluationDeviceResult& result);
#endif

    namespace
    {
        // 根据标签列表生成唯一的缓存键（字符串）
        // 将每个标签后加换行符拼接，确保不同顺序的标签列表产生不同键
        std::string subset_cache_key(const std::vector<std::string>& labels)
        {
            std::string key;
            for (const auto& label : labels)
            {
                key += label;
                key.push_back('\n');
            }
            return key;
        }

        // 返回针对特定域和行数应该使用的打包子集阈值
        std::size_t packed_subset_threshold_for(
            const std::shared_ptr<RootOfUnityDomain>& domain,
            std::size_t row_count)
        {
            constexpr std::size_t kDefaultThreshold = 1U << 21U;
            if (domain != nullptr && domain->name == "edge" && domain->size <= (1U << 21U) && row_count >= 24)
            {
                return 1U << 27U;
            }
            return kDefaultThreshold;
        }

    }

    struct PackedEvaluationBackend::SubsetCacheState
    {
        std::mutex mutex;
        std::unordered_map<std::string, SubsetView> subset_cache;
    };

    struct PackedEvaluationBackend::SubsetView
    {
        std::vector<std::size_t> row_indices;
        std::vector<std::uint32_t> row_indices_u32;
        std::vector<mcl::Fr> packed_values;
        mutable PackedFieldBuffer packed_values_packed;
    };

    // 预计算所有多项式在单位根域上的值，并建立标签到索引的映射
    PackedEvaluationBackend::PackedEvaluationBackend(
        std::shared_ptr<RootOfUnityDomain> domain,
        std::vector<std::pair<std::string, const Polynomial*>> polynomials)
        : domain_(std::move(domain)), polynomials_(std::move(polynomials)), subset_cache_state_(std::make_shared<SubsetCacheState>())
    {
        if (domain_ == nullptr)
        {
            throw std::runtime_error("packed evaluation backend requires a domain");
        }
        const auto row_count = polynomials_.size();
        packed_values_.resize(domain_->size * row_count);
        for (std::size_t row = 0; row < row_count; ++row)
        {
            label_to_index_.emplace(polynomials_[row].first, row);
            const auto* polynomial = polynomials_[row].second;
            if (polynomial == nullptr || polynomial->basis != PolynomialBasis::Evaluation || polynomial->domain != domain_)
            {
                throw std::runtime_error("packed evaluation backend requires same-domain evaluation polynomials");
            }
            for (std::size_t column = 0; column < domain_->size; ++column)
            {
                packed_values_[column * row_count + row] = polynomial->data[column].native();
            }
        }
    }

    const PackedEvaluationBackend::SubsetView& PackedEvaluationBackend::subset_for(const std::vector<std::string>& labels) const
    {
        const auto key = subset_cache_key(labels);
        {
            std::lock_guard<std::mutex> lock(subset_cache_state_->mutex);
            if (const auto it = subset_cache_state_->subset_cache.find(key);
                it != subset_cache_state_->subset_cache.end())
            {
                return it->second;
            }
        }

        SubsetView subset;
        subset.row_indices.reserve(labels.size());
        subset.row_indices_u32.reserve(labels.size());
        for (const auto& label : labels)
        {
            const auto it = label_to_index_.find(label);
            if (it == label_to_index_.end())
            {
                throw std::runtime_error("packed evaluation backend is missing label: " + label);
            }
            if (it->second > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
            {
                throw std::runtime_error("packed evaluation backend row index exceeds uint32_t");
            }
            subset.row_indices.push_back(it->second);
            subset.row_indices_u32.push_back(static_cast<std::uint32_t>(it->second));
        }

        const auto row_count = subset.row_indices.size();
        const auto packed_term_count = domain_->size * row_count;
        const auto packed_subset_threshold = packed_subset_threshold_for(domain_, row_count);
        if (packed_term_count <= packed_subset_threshold && row_count > 1)
        {
            subset.packed_values.resize(packed_term_count);
            for (std::size_t column = 0; column < domain_->size; ++column)
            {
                const auto* packed_row = &packed_values_[column * polynomials_.size()];
                auto* subset_row = &subset.packed_values[column * row_count];
                for (std::size_t row = 0; row < row_count; ++row)
                {
                    subset_row[row] = packed_row[subset.row_indices[row]];
                }
            }
        }

        std::lock_guard<std::mutex> lock(subset_cache_state_->mutex);
        auto [it, inserted] = subset_cache_state_->subset_cache.emplace(key, std::move(subset));
        (void)inserted;
        return it->second;
    }

    // 根据标签列表解析出对应的行索引
    std::vector<std::size_t> PackedEvaluationBackend::resolve_row_indices(const std::vector<std::string>& labels) const
    {
        return subset_for(labels).row_indices;
    }

    // 返回标签子集的行索引
    const std::vector<std::size_t>& PackedEvaluationBackend::subset_row_indices(
        const std::vector<std::string>& labels) const
    {
        return subset_for(labels).row_indices;
    }

    // 返回标签子集的行索引
    const std::vector<std::uint32_t>& PackedEvaluationBackend::subset_row_indices_u32(
        const std::vector<std::string>& labels) const
    {
        return subset_for(labels).row_indices_u32;
    }

    // 返回所有打包值的设备端就绪缓冲区
    const PackedFieldBuffer& PackedEvaluationBackend::packed_values_device_ready() const
    {
        if (packed_values_packed_.empty() && !packed_values_.empty())
        {
            pack_native_field_elements_into(packed_values_, &packed_values_packed_);
        }
        return packed_values_packed_;
    }

    // 返回指定标签子集的设备端就绪打包值缓冲区
    const PackedFieldBuffer* PackedEvaluationBackend::subset_packed_values_device_ready(
        const std::vector<std::string>& labels) const
    {
        const auto& subset = subset_for(labels);
        if (subset.packed_values.empty())
        {
            return nullptr;
        }
        if (subset.packed_values_packed.empty())
        {
            pack_native_field_elements_into(subset.packed_values, &subset.packed_values_packed);
        }
        return &subset.packed_values_packed;
    }

    // 直接返回指定标签在给定列索引上的多项式值（每个多项式在该点的求值）
    std::vector<FieldElement> PackedEvaluationBackend::values_at_direct_index(
        const std::vector<std::string>& labels,
        std::size_t index) const
    {
        std::vector<FieldElement> out;
        out.reserve(labels.size());
        for (const auto row_index : subset_for(labels).row_indices)
        {
            out.push_back(polynomials_[row_index].second->data.at(index));
        }
        return out;
    }

    // 使用普通权重（域元素）计算加权和：先转换为原生权重，再调用原生版本
    std::vector<FieldElement> PackedEvaluationBackend::evaluate_with_weights(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& weights) const
    {
        std::vector<mcl::Fr> native_weights(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i)
        {
            native_weights[i] = weights[i].native();
        }
        return evaluate_with_native_weights(labels, native_weights);
    }

    // 使用原生权重（mcl::Fr）计算加权和：根据后端选择CUDA或CPU
    std::vector<FieldElement> PackedEvaluationBackend::evaluate_with_native_weights(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& weights) const
    {
        if (configured_algebra_backend() == AlgebraBackend::Cuda)
        {
#if GATZK_ENABLE_CUDA_BACKEND
            const auto packed_weights = pack_native_field_elements(weights);
            return evaluate_with_packed_native_weights_cuda(*this, labels, packed_weights);
#else
            throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
        }
        return evaluate_with_packed_native_weights(labels, weights, PackedFieldBuffer());
    }

    std::vector<FieldElement> PackedEvaluationBackend::evaluate_with_packed_native_weights(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& weights,
        const PackedFieldBuffer& packed_weights) const
    {
        if (weights.size() != domain_->size)
        {
            throw std::runtime_error("packed evaluation backend weight size mismatch");
        }

        if (configured_algebra_backend() == AlgebraBackend::Cuda)
        {
#if GATZK_ENABLE_CUDA_BACKEND
            if (packed_weights.size() != weights.size())
            {
                throw std::runtime_error("packed evaluation backend packed weight size mismatch");
            }
            return evaluate_with_packed_native_weights_cuda(*this, labels, packed_weights);
#else
            throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
        }

        const auto& subset = subset_for(labels);
        const auto row_count = subset.row_indices.size();
        const bool use_packed_subset = !subset.packed_values.empty();
        std::vector<mcl::Fr> native_out(row_count);
        for (auto& value : native_out)
        {
            value.clear();
        }

        const auto& route2 = util::route2_options();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const auto total_terms = domain_->size * row_count;
        if (route2.parallel_fft && cpu_count > 1 && domain_->size >= 1024 && total_terms >= 16384)
        {
            const auto task_count = std::min<std::size_t>(cpu_count, domain_->size / 512);
            if (task_count > 1)
            {
                const auto chunk_size = (domain_->size + task_count - 1) / task_count;
                std::vector<std::future<std::vector<mcl::Fr>>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min(domain_->size, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end, row_count]() {
                            std::vector<mcl::Fr> partial(row_count);
                            for (auto& value : partial)
                            {
                                value.clear();
                            }
                            for (std::size_t column = begin; column < end; ++column)
                            {
                                const auto& weight = weights[column];
                                for (std::size_t row = 0; row < row_count; ++row)
                                {
                                    mcl::Fr term;
                                    const auto& value = use_packed_subset
                                                            ? subset.packed_values[column * row_count + row]
                                                            : packed_values_[column * polynomials_.size() + subset.row_indices[row]];
                                    mcl::Fr::mul(term, value, weight);
                                    mcl::Fr::add(partial[row], partial[row], term);
                                }
                            }
                            return partial;
                        }));
                }
                for (auto& future : futures)
                {
                    const auto partial = future.get();
                    for (std::size_t row = 0; row < row_count; ++row)
                    {
                        mcl::Fr::add(native_out[row], native_out[row], partial[row]);
                    }
                }

                std::vector<FieldElement> out;
                out.reserve(row_count);
                for (const auto& value : native_out)
                {
                    out.push_back(FieldElement::from_native(value));
                }
                return out;
            }
        }

        for (std::size_t column = 0; column < domain_->size; ++column)
        {
            const auto& weight = weights[column];
            for (std::size_t row = 0; row < row_count; ++row)
            {
                mcl::Fr term;
                const auto& value = use_packed_subset
                                        ? subset.packed_values[column * row_count + row]
                                        : packed_values_[column * polynomials_.size() + subset.row_indices[row]];
                mcl::Fr::mul(term, value, weight);
                mcl::Fr::add(native_out[row], native_out[row], term);
            }
        }

        std::vector<FieldElement> out;
        out.reserve(row_count);
        for (const auto& value : native_out)
        {
            out.push_back(FieldElement::from_native(value));
        }
        return out;
    }

    // 设备端求值（仅CUDA）：返回设备结果句柄，结果留在GPU内存中
    PackedEvaluationDeviceResult PackedEvaluationBackend::evaluate_device_with_packed_native_weights(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& weights,
        const PackedFieldBuffer& packed_weights) const
    {
        if (weights.size() != domain_->size)
        {
            throw std::runtime_error("packed evaluation backend weight size mismatch");
        }
        if (configured_algebra_backend() != AlgebraBackend::Cuda)
        {
            throw std::runtime_error("device-side packed evaluation requires CUDA algebra backend");
        }
#if GATZK_ENABLE_CUDA_BACKEND
        if (packed_weights.size() != weights.size())
        {
            throw std::runtime_error("packed evaluation backend packed weight size mismatch");
        }
        return evaluate_device_with_packed_native_weights_cuda(*this, labels, packed_weights);
#else
        (void)labels;
        (void)packed_weights;
        throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
    }

    // 使用原生权重的旋转求值
    std::vector<std::vector<FieldElement>> PackedEvaluationBackend::evaluate_with_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<FieldElement>& representative_weights,
        const std::vector<std::size_t>& rotations) const
    {
        std::vector<mcl::Fr> native_weights(representative_weights.size());
        for (std::size_t i = 0; i < representative_weights.size(); ++i)
        {
            native_weights[i] = representative_weights[i].native();
        }
        return evaluate_with_native_weight_rotations(labels, native_weights, rotations);
    }

      // CPU 实现的带权重旋转求值（核心算法）
    std::vector<std::vector<FieldElement>> PackedEvaluationBackend::evaluate_with_native_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& representative_weights,
        const std::vector<std::size_t>& rotations) const
    {
        if (configured_algebra_backend() == AlgebraBackend::Cuda)
        {
#if GATZK_ENABLE_CUDA_BACKEND
            const auto packed_weights = pack_native_field_elements(representative_weights);
            return evaluate_with_packed_native_weight_rotations_cuda(*this, labels, packed_weights, rotations);
#else
            throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
        }
        return evaluate_with_packed_native_weight_rotations(
            labels,
            representative_weights,
            PackedFieldBuffer(),
            rotations);
    }

    // CPU 实现的带权重旋转求值（核心算法）
    std::vector<std::vector<FieldElement>> PackedEvaluationBackend::evaluate_with_packed_native_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& representative_weights,
        const PackedFieldBuffer& representative_weights_packed,
        const std::vector<std::size_t>& rotations) const
    {
        if (representative_weights.size() != domain_->size)
        {
            throw std::runtime_error("packed rotated evaluation backend weight size mismatch");
        }
        if (rotations.empty())
        {
            return {};
        }

        if (configured_algebra_backend() == AlgebraBackend::Cuda)
        {
#if GATZK_ENABLE_CUDA_BACKEND
            if (representative_weights_packed.size() != representative_weights.size())
            {
                throw std::runtime_error("packed rotated evaluation backend packed weight size mismatch");
            }
            return evaluate_with_packed_native_weight_rotations_cuda(
                *this,
                labels,
                representative_weights_packed,
                rotations);
#else
            throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
        }

        const auto& subset = subset_for(labels);
        const auto row_count = subset.row_indices.size();
        const auto point_count = rotations.size();
        const auto domain_mask = domain_->size - 1U;
        const bool use_packed_subset = !subset.packed_values.empty();
        std::vector<mcl::Fr> native_out(point_count * row_count);
        for (auto& value : native_out)
        {
            value.clear();
        }

        const auto& route2 = util::route2_options();
        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const auto total_terms = domain_->size * row_count * point_count;
        if (route2.parallel_fft && cpu_count > 1 && domain_->size >= 1024 && total_terms >= 32768)
        {
            const auto task_count = std::min<std::size_t>(cpu_count, domain_->size / 512);
            if (task_count > 1)
            {
                const auto chunk_size = (domain_->size + task_count - 1) / task_count;
                std::vector<std::future<std::vector<mcl::Fr>>> futures;
                futures.reserve(task_count);
                for (std::size_t task = 0; task < task_count; ++task)
                {
                    const auto begin = task * chunk_size;
                    const auto end = std::min(domain_->size, begin + chunk_size);
                    if (begin >= end)
                    {
                        break;
                    }
                    futures.push_back(std::async(
                        std::launch::async,
                        [&, begin, end, row_count, point_count, domain_mask]() {
                            std::vector<mcl::Fr> partial(point_count * row_count);
                            for (auto& value : partial)
                            {
                                value.clear();
                            }
                            for (std::size_t column = begin; column < end; ++column)
                            {
                                const auto& weight = representative_weights[column];
                                for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                                {
                                    const auto rotated_column = (column + rotations[point_index]) & domain_mask;
                                    auto* partial_row = &partial[point_index * row_count];
                                    for (std::size_t row = 0; row < row_count; ++row)
                                    {
                                        mcl::Fr term;
                                        const auto& value = use_packed_subset
                                                                ? subset.packed_values[rotated_column * row_count + row]
                                                                : packed_values_[rotated_column * polynomials_.size() + subset.row_indices[row]];
                                        mcl::Fr::mul(term, value, weight);
                                        mcl::Fr::add(partial_row[row], partial_row[row], term);
                                    }
                                }
                            }
                            return partial;
                        }));
                }
                for (auto& future : futures)
                {
                    const auto partial = future.get();
                    for (std::size_t i = 0; i < partial.size(); ++i)
                    {
                        mcl::Fr::add(native_out[i], native_out[i], partial[i]);
                    }
                }
            }
            else
            {
                for (std::size_t column = 0; column < domain_->size; ++column)
                {
                    const auto& weight = representative_weights[column];
                    for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                    {
                        const auto rotated_column = (column + rotations[point_index]) & domain_mask;
                        auto* native_row = &native_out[point_index * row_count];
                        for (std::size_t row = 0; row < row_count; ++row)
                        {
                            mcl::Fr term;
                            const auto& value = use_packed_subset
                                                    ? subset.packed_values[rotated_column * row_count + row]
                                                    : packed_values_[rotated_column * polynomials_.size() + subset.row_indices[row]];
                            mcl::Fr::mul(term, value, weight);
                            mcl::Fr::add(native_row[row], native_row[row], term);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::size_t column = 0; column < domain_->size; ++column)
            {
                const auto& weight = representative_weights[column];
                for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                {
                    const auto rotated_column = (column + rotations[point_index]) & domain_mask;
                    auto* native_row = &native_out[point_index * row_count];
                    for (std::size_t row = 0; row < row_count; ++row)
                    {
                        mcl::Fr term;
                        const auto& value = use_packed_subset
                                                ? subset.packed_values[rotated_column * row_count + row]
                                                : packed_values_[rotated_column * polynomials_.size() + subset.row_indices[row]];
                        mcl::Fr::mul(term, value, weight);
                        mcl::Fr::add(native_row[row], native_row[row], term);
                    }
                }
            }
        }

        std::vector<std::vector<FieldElement>> out(point_count, std::vector<FieldElement>(row_count, FieldElement::zero()));
        for (std::size_t point_index = 0; point_index < point_count; ++point_index)
        {
            for (std::size_t row = 0; row < row_count; ++row)
            {
                out[point_index][row] = FieldElement::from_native(native_out[point_index * row_count + row]);
            }
        }
        return out;
    }

    PackedEvaluationDeviceResult PackedEvaluationBackend::evaluate_device_with_packed_native_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& representative_weights,
        const PackedFieldBuffer& representative_weights_packed,
        const std::vector<std::size_t>& rotations) const
    {
        if (representative_weights.size() != domain_->size)
        {
            throw std::runtime_error("packed rotated evaluation backend weight size mismatch");
        }
        if (rotations.empty())
        {
            return {};
        }
        if (configured_algebra_backend() != AlgebraBackend::Cuda)
        {
            throw std::runtime_error("device-side packed rotated evaluation requires CUDA algebra backend");
        }
#if GATZK_ENABLE_CUDA_BACKEND
        if (representative_weights_packed.size() != representative_weights.size())
        {
            throw std::runtime_error("packed rotated evaluation backend packed weight size mismatch");
        }
        return evaluate_device_with_packed_native_weight_rotations_cuda(
            *this,
            labels,
            representative_weights_packed,
            rotations);
#else
        (void)labels;
        (void)representative_weights_packed;
        throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
    }

    // 将设备端结果拷贝回主机并展开为一维域元素向量
    std::vector<FieldElement> PackedEvaluationBackend::materialize_device_result(
        const PackedEvaluationDeviceResult& result) const
    {
        if (result.empty())
        {
            return {};
        }
        if (configured_algebra_backend() != AlgebraBackend::Cuda)
        {
            throw std::runtime_error("device-side packed evaluation materialization requires CUDA algebra backend");
        }
#if GATZK_ENABLE_CUDA_BACKEND
        return materialize_device_result_cuda(result);
#else
        throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
    }

    // 将设备端旋转结果拷贝回主机并展开为二维向量（每个旋转一行）
    std::vector<std::vector<FieldElement>> PackedEvaluationBackend::materialize_device_rotation_result(
        const PackedEvaluationDeviceResult& result) const
    {
        if (result.empty())
        {
            return {};
        }
        if (configured_algebra_backend() != AlgebraBackend::Cuda)
        {
            throw std::runtime_error("device-side packed evaluation materialization requires CUDA algebra backend");
        }
#if GATZK_ENABLE_CUDA_BACKEND
        return materialize_device_rotation_result_cuda(result);
#else
        throw std::runtime_error("CUDA algebra backend requested but this build was compiled without CUDA support");
#endif
    }
}