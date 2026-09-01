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
    // 供 quotient、domain opening 与 verifier 复算阶段批量求值复用。
    // 它属于 algebra 执行层加速，不改变 formal 多项式、challenge 或 opening 契约。

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

        // 返回针对特定域和行数应该使用的子集物化阈值
        std::size_t subset_cache_threshold_for(
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
        std::vector<mcl::Fr> interleaved_values;
    };

    // 校验同域多项式并建立标签到索引的映射；数值从 canonical 列直接
    // 读取，只有命中的小标签子集才会按阈值惰性物化交织布局。
    PackedEvaluationBackend::PackedEvaluationBackend(
        std::shared_ptr<RootOfUnityDomain> domain,
        std::vector<std::pair<std::string, const Polynomial*>> polynomials)
        : domain_(std::move(domain)), polynomials_(std::move(polynomials)), subset_cache_state_(std::make_shared<SubsetCacheState>())
    {
        if (domain_ == nullptr)
        {
            throw std::runtime_error("domain evaluation backend requires a domain");
        }
        const auto row_count = polynomials_.size();
        for (std::size_t row = 0; row < row_count; ++row)
        {
            label_to_index_.emplace(polynomials_[row].first, row);
            const auto* polynomial = polynomials_[row].second;
            if (polynomial == nullptr || polynomial->basis != PolynomialBasis::Evaluation || polynomial->domain != domain_)
            {
                throw std::runtime_error("domain evaluation backend requires same-domain evaluation polynomials");
            }
        }
    }

    const mcl::Fr& PackedEvaluationBackend::native_value_at(
        std::size_t polynomial_index,
        std::size_t column) const
    {
        static const FieldElement zero = FieldElement::zero();
        const auto& values =
            polynomials_.at(polynomial_index).second->values();
        return column < values.size()
            ? values[column].native()
            : zero.native();
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
                throw std::runtime_error("domain evaluation backend is missing label: " + label);
            }
            if (it->second > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
            {
                throw std::runtime_error("domain evaluation backend row index exceeds uint32_t");
            }
            subset.row_indices.push_back(it->second);
            subset.row_indices_u32.push_back(static_cast<std::uint32_t>(it->second));
        }

        const auto row_count = subset.row_indices.size();
        const auto interleaved_term_count = domain_->size * row_count;
        const auto subset_cache_threshold = subset_cache_threshold_for(domain_, row_count);
        if (interleaved_term_count <= subset_cache_threshold && row_count > 1)
        {
            subset.interleaved_values.resize(interleaved_term_count);
            for (std::size_t column = 0; column < domain_->size; ++column)
            {
                auto* subset_row = &subset.interleaved_values[column * row_count];
                for (std::size_t row = 0; row < row_count; ++row)
                {
                    subset_row[row] = native_value_at(
                        subset.row_indices[row], column);
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

    // 直接返回指定标签在给定列索引上的多项式值（每个多项式在该点的求值）
    std::vector<FieldElement> PackedEvaluationBackend::values_at_direct_index(
        const std::vector<std::string>& labels,
        std::size_t index) const
    {
        std::vector<FieldElement> out;
        out.reserve(labels.size());
        for (const auto row_index : subset_for(labels).row_indices)
        {
            const auto& values = polynomials_[row_index].second->values();
            out.push_back(index < values.size() ? values[index] : FieldElement::zero());
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

    // 使用原生权重（mcl::Fr）计算加权和（--parallel-fft on）
    std::vector<FieldElement> PackedEvaluationBackend::evaluate_with_native_weights(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& weights) const
    {
        if (weights.size() != domain_->size)
        {
            throw std::runtime_error("domain evaluation backend weight size mismatch");
        }

        const auto& route2 = util::route2_options();
        const auto row_count = labels.size();
        const bool try_gpu = should_use_cuda_weighted_evaluation(domain_->size, row_count);

        // On a unified-memory device, full-domain evaluation polynomials are
        // already stored in the columnar layout consumed by the CUDA kernel.
        // Use those read-only columns directly instead of materializing or
        // scanning an interleaved label subset first. Per-column lengths let
        // the same kernel preserve the canonical zero-padded tail semantics.
        if (try_gpu && cuda_backend_unified_addressing_available())
        {
            std::vector<const uint64_t*> direct_column_ptrs;
            std::vector<std::size_t> direct_column_lengths;
            direct_column_ptrs.reserve(row_count);
            direct_column_lengths.reserve(row_count);
            for (const auto& label : labels)
            {
                const auto label_it = label_to_index_.find(label);
                if (label_it == label_to_index_.end())
                {
                    throw std::runtime_error(
                        "domain evaluation backend is missing label: " + label);
                }
                const auto& values = polynomials_[label_it->second].second->values();
                direct_column_ptrs.push_back(reinterpret_cast<const uint64_t*>(
                    FieldElement::native_array_data(values.data())));
                direct_column_lengths.push_back(std::min(values.size(), domain_->size));
            }
            std::vector<mcl::Fr> gpu_results(row_count);
            for (auto& value : gpu_results) value.clear();
            const bool gpu_ok = weighted_poly_eval_gpu_multi(
                reinterpret_cast<const uint64_t*>(weights.data()),
                direct_column_ptrs.data(),
                reinterpret_cast<uint64_t*>(gpu_results.data()),
                domain_->size,
                row_count,
                direct_column_lengths.data());
            if (!gpu_ok)
            {
                throw std::runtime_error(
                    "CUDA direct columnar evaluation kernel failed; refusing silent CPU fallback");
            }
            std::vector<FieldElement> out;
            out.reserve(row_count);
            for (const auto& value : gpu_results)
            {
                out.push_back(FieldElement::from_native(value));
            }
            return out;
        }

        const auto& subset = subset_for(labels);
        const bool use_cached_subset = !subset.interleaved_values.empty();

        if (try_gpu)
        {
            if (use_cached_subset)
            {
                std::vector<mcl::Fr> gpu_results(row_count);
                for (auto& value : gpu_results)
                {
                    value.clear();
                }
                const bool gpu_ok = weighted_poly_eval_gpu(
                    reinterpret_cast<const uint64_t*>(weights.data()),
                    reinterpret_cast<const uint64_t*>(subset.interleaved_values.data()),
                    reinterpret_cast<uint64_t*>(gpu_results.data()),
                    domain_->size,
                    row_count);
                if (gpu_ok)
                {
                    std::vector<FieldElement> out;
                    out.reserve(row_count);
                    for (const auto& value : gpu_results)
                    {
                        out.push_back(FieldElement::from_native(value));
                    }
                    return out;
                }
            }

            // 先把交织布局拆成按列连续缓冲区，再统一交给 GPU。
            // 这样权重只上传一次，多个多项式共享同一组 barycentric 权重。
            std::vector<std::vector<mcl::Fr>> col_buffers(row_count);
            std::vector<const uint64_t*> col_ptrs(row_count);
            for (std::size_t r = 0; r < row_count; ++r)
            {
                col_buffers[r].resize(domain_->size);
                if (use_cached_subset)
                {
                    for (std::size_t col = 0; col < domain_->size; ++col)
                    {
                        col_buffers[r][col] = subset.interleaved_values[col * row_count + r];
                    }
                }
                else
                {
                    const auto global_row = subset.row_indices[r];
                    for (std::size_t col = 0; col < domain_->size; ++col)
                    {
                        col_buffers[r][col] =
                            native_value_at(global_row, col);
                    }
                }
                col_ptrs[r] = reinterpret_cast<const uint64_t*>(col_buffers[r].data());
            }

            std::vector<mcl::Fr> gpu_results(row_count);
            for (auto& v : gpu_results) v.clear();

            const bool gpu_ok = weighted_poly_eval_gpu_multi(
                reinterpret_cast<const uint64_t*>(weights.data()),
                col_ptrs.data(),
                reinterpret_cast<uint64_t*>(gpu_results.data()),
                domain_->size,
                row_count);

            if (gpu_ok)
            {
                std::vector<FieldElement> out;
                out.reserve(row_count);
                for (const auto& v : gpu_results)
                {
                    out.push_back(FieldElement::from_native(v));
                }
                return out;
            }
            throw std::runtime_error("CUDA weighted evaluation kernel failed; refusing silent CPU fallback");
        }

        // ---- CPU path ----
        std::vector<mcl::Fr> native_out(row_count);
        for (auto& value : native_out)
        {
            value.clear();
        }

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
                                    const auto& value = use_cached_subset
                                                            ? subset.interleaved_values[column * row_count + row]
                                                            : native_value_at(
                                                                subset.row_indices[row],
                                                                column);
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
                const auto& value = use_cached_subset
                                        ? subset.interleaved_values[column * row_count + row]
                                        : native_value_at(
                                            subset.row_indices[row],
                                            column);
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

      // CPU 实现的带权重旋转求值（核心算法）（--parallel-fft on）（--rotated-fft-kernel on）
    std::vector<std::vector<FieldElement>> PackedEvaluationBackend::evaluate_with_native_weight_rotations(
        const std::vector<std::string>& labels,
        const std::vector<mcl::Fr>& representative_weights,
        const std::vector<std::size_t>& rotations,
        EvaluationExecutionPolicy execution_policy) const
    {
        if (representative_weights.size() != domain_->size)
        {
            throw std::runtime_error("domain evaluation backend rotation weight size mismatch");
        }
        if (rotations.empty())
        {
            return {};
        }

        const auto row_count = labels.size();
        const auto point_count = rotations.size();
        const auto domain_mask = domain_->size - 1U;
        std::vector<mcl::Fr> native_out(point_count * row_count);
        for (auto& value : native_out)
        {
            value.clear();
        }

        const auto& route2 = util::route2_options();
        const bool try_gpu =
            execution_policy == EvaluationExecutionPolicy::Automatic
            && should_use_cuda_weighted_evaluation(
                domain_->size, row_count, point_count);

        // Unified-memory CUDA uses the canonical evaluation columns directly.
        // This avoids materializing an interleaved subset (or copying it back
        // into column buffers) before every group of rotated opening points.
        if (try_gpu && cuda_backend_unified_addressing_available())
        {
            std::vector<const uint64_t*> direct_column_ptrs;
            std::vector<std::size_t> direct_column_lengths;
            direct_column_ptrs.reserve(row_count);
            direct_column_lengths.reserve(row_count);
            for (const auto& label : labels)
            {
                const auto label_it = label_to_index_.find(label);
                if (label_it == label_to_index_.end())
                {
                    throw std::runtime_error(
                        "domain evaluation backend is missing label: " + label);
                }
                const auto& values = polynomials_[label_it->second].second->values();
                direct_column_ptrs.push_back(reinterpret_cast<const uint64_t*>(
                    FieldElement::native_array_data(values.data())));
                direct_column_lengths.push_back(
                    std::min(values.size(), domain_->size));
            }

            constexpr std::size_t kMaxRotationPartialBytes =
                256ULL * 1024ULL * 1024ULL;
            constexpr std::size_t kCudaReductionBlockSize = 256;
            const auto partial_blocks =
                (domain_->size + kCudaReductionBlockSize - 1)
                / kCudaReductionBlockSize;
            const auto partial_bytes_per_point =
                static_cast<unsigned __int128>(partial_blocks)
                * static_cast<unsigned __int128>(row_count)
                * sizeof(mcl::Fr);
            const auto rotations_per_launch =
                partial_bytes_per_point
                        > static_cast<unsigned __int128>(
                            kMaxRotationPartialBytes)
                    ? std::size_t{1}
                    : std::max<std::size_t>(
                        1,
                        kMaxRotationPartialBytes
                            / static_cast<std::size_t>(
                                partial_bytes_per_point));
            for (std::size_t rotation_begin = 0;
                 rotation_begin < point_count;
                 rotation_begin += rotations_per_launch)
            {
                const auto launch_count = std::min(
                    rotations_per_launch,
                    point_count - rotation_begin);
                const bool gpu_ok =
                    weighted_poly_eval_gpu_multi_rotations(
                        reinterpret_cast<const uint64_t*>(
                            representative_weights.data()),
                        direct_column_ptrs.data(),
                        reinterpret_cast<uint64_t*>(
                            native_out.data()
                            + rotation_begin * row_count),
                        domain_->size,
                        row_count,
                        rotations.data() + rotation_begin,
                        launch_count,
                        direct_column_lengths.data());
                if (!gpu_ok)
                {
                    throw std::runtime_error(
                        "CUDA direct batched rotated columnar evaluation kernel failed; refusing silent CPU fallback");
                }
            }

            std::vector<std::vector<FieldElement>> out(
                point_count,
                std::vector<FieldElement>(row_count, FieldElement::zero()));
            for (std::size_t point_index = 0;
                 point_index < point_count;
                 ++point_index)
            {
                for (std::size_t row = 0; row < row_count; ++row)
                {
                    out[point_index][row] = FieldElement::from_native(
                        native_out[point_index * row_count + row]);
                }
            }
            return out;
        }

        const auto& subset = subset_for(labels);
        const bool use_cached_subset = !subset.interleaved_values.empty();

        // ---- GPU path for rotated evaluations ----
        if (try_gpu)
        {
            if (use_cached_subset)
            {
                bool all_ok = true;
                for (std::size_t point_index = 0; point_index < point_count; ++point_index)
                {
                    const auto rotation = rotations[point_index];
                    std::vector<mcl::Fr> rotated_weights(domain_->size);
                    for (std::size_t col = 0; col < domain_->size; ++col)
                    {
                        rotated_weights[(col + rotation) & domain_mask] = representative_weights[col];
                    }

                    std::vector<mcl::Fr> gpu_results(row_count);
                    for (auto& value : gpu_results)
                    {
                        value.clear();
                    }

                    const bool gpu_ok = weighted_poly_eval_gpu(
                        reinterpret_cast<const uint64_t*>(rotated_weights.data()),
                        reinterpret_cast<const uint64_t*>(subset.interleaved_values.data()),
                        reinterpret_cast<uint64_t*>(gpu_results.data()),
                        domain_->size,
                        row_count);
                    if (!gpu_ok)
                    {
                        all_ok = false;
                        break;
                    }
                    for (std::size_t row = 0; row < row_count; ++row)
                    {
                        native_out[point_index * row_count + row] = gpu_results[row];
                    }
                }

                if (all_ok)
                {
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
                throw std::runtime_error("CUDA rotated evaluation kernel failed; refusing silent CPU fallback");
            }

            // 旋转点评估共享同一代表权重，只是在域上做循环平移。
            // 列缓冲区只构造一次，逐个旋转点复用。
            std::vector<std::vector<mcl::Fr>> col_buffers(row_count);
            std::vector<const uint64_t*> col_ptrs(row_count);
            for (std::size_t r = 0; r < row_count; ++r)
            {
                col_buffers[r].resize(domain_->size);
                if (use_cached_subset)
                {
                    for (std::size_t col = 0; col < domain_->size; ++col)
                    {
                        col_buffers[r][col] = subset.interleaved_values[col * row_count + r];
                    }
                }
                else
                {
                    const auto global_row = subset.row_indices[r];
                    for (std::size_t col = 0; col < domain_->size; ++col)
                    {
                        col_buffers[r][col] =
                            native_value_at(global_row, col);
                    }
                }
                col_ptrs[r] = reinterpret_cast<const uint64_t*>(col_buffers[r].data());
            }

            bool all_ok = true;
            for (std::size_t point_index = 0; point_index < point_count; ++point_index)
            {
                // `omega` 旋转只会导致权重向量循环位移，不改变 formal 求值点集合。
                const auto rotation = rotations[point_index];
                std::vector<mcl::Fr> rotated_weights(domain_->size);
                for (std::size_t col = 0; col < domain_->size; ++col)
                {
                    rotated_weights[(col + rotation) & domain_mask] = representative_weights[col];
                }

                std::vector<mcl::Fr> gpu_results(row_count);
                for (auto& v : gpu_results) v.clear();

                const bool gpu_ok = weighted_poly_eval_gpu_multi(
                    reinterpret_cast<const uint64_t*>(rotated_weights.data()),
                    col_ptrs.data(),
                    reinterpret_cast<uint64_t*>(gpu_results.data()),
                    domain_->size,
                    row_count);

                if (!gpu_ok) { all_ok = false; break; }
                for (std::size_t r = 0; r < row_count; ++r)
                {
                    native_out[point_index * row_count + r] = gpu_results[r];
                }
            }

            if (all_ok)
            {
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
            throw std::runtime_error("CUDA rotated multi-evaluation kernel failed; refusing silent CPU fallback");
        }

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
                                        const auto& value = use_cached_subset
                                                                ? subset.interleaved_values[rotated_column * row_count + row]
                                                                : native_value_at(
                                                                    subset.row_indices[row],
                                                                    rotated_column);
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
                            const auto& value = use_cached_subset
                                                    ? subset.interleaved_values[rotated_column * row_count + row]
                                                    : native_value_at(
                                                        subset.row_indices[row],
                                                        rotated_column);
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
                        const auto& value = use_cached_subset
                                                ? subset.interleaved_values[rotated_column * row_count + row]
                                                : native_value_at(
                                                    subset.row_indices[row],
                                                    rotated_column);
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
}
