#pragma once
// 本文件提供域上求值权重的构造与缓存，是 prove / verify 在
// domain opening、quotient 复算和批量求值阶段共用的执行层工具。

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"

namespace gatzk::algebra
{
    // 将挑战点编码成缓存键，供不同域上的求值权重复用。
    inline std::string point_cache_key(const FieldElement& point)
    {
        return point.to_string();
    }

    inline std::string domain_point_cache_key(
        const std::shared_ptr<RootOfUnityDomain>& domain,
        const FieldElement& point)
    {
        if (domain == nullptr)
        {
            throw std::runtime_error("domain_point_cache_key requires a domain");
        }
        return domain->name + ":" + std::to_string(domain->size) + ":" + point_cache_key(point);
    }

    struct DomainEvaluationWeights
    {
        // 挑战点恰好落在域上时可直接索引，否则走重心权重求值。
        std::optional<std::size_t> direct_index;
        std::vector<mcl::Fr> native_weights;
    };

    inline DomainEvaluationWeights build_domain_evaluation_weights(
        const std::shared_ptr<RootOfUnityDomain>& domain,
        const FieldElement& point)
    {
        if (domain == nullptr)
        {
            throw std::runtime_error("build_domain_evaluation_weights requires a domain");
        }

        DomainEvaluationWeights entry;
        if (const auto shift = domain->rotation_shift(
                FieldElement::one(), point);
            shift.has_value())
        {
            entry.direct_index = *shift;
        }

        if (!entry.direct_index.has_value())
        {
            entry.native_weights = domain->barycentric_weights_native(point);
        }
        return entry;
    }

class DomainEvaluationWeightCache
    {
    public:
        const DomainEvaluationWeights& get(
            const std::shared_ptr<RootOfUnityDomain>& domain,
            const FieldElement& point)
        {
            const auto cache_key =
                domain_point_cache_key(
                    domain,
                    point);

            return get_or_build(
                cache_key,
                [&]()
                {
                    return build_domain_evaluation_weights(
                        domain,
                        point);
                });
        }

        // For H = {omega^i},
        //
        //     L_i(z * omega) = L_{i-1 mod n}(z).
        //
        // Therefore a z/zomega pair needs one barycentric construction only.
        // The second point is produced by a cyclic rotation of the first
        // weight vector without a second scan/inversion.
        const DomainEvaluationWeights&
        get_forward_rotation(
            const std::shared_ptr<
                RootOfUnityDomain>& domain,
            const FieldElement& point)
        {
            if (domain == nullptr)
            {
                throw std::runtime_error(
                    "forward domain-weight rotation requires a domain");
            }

            const auto rotated_point =
                point * domain->omega;

            const auto base_key =
                domain_point_cache_key(
                    domain,
                    point);

            const auto rotated_key =
                domain_point_cache_key(
                    domain,
                    rotated_point);

            if (base_key == rotated_key)
            {
                return get(
                    domain,
                    point);
            }

            return get_or_build(
                rotated_key,
                [&]()
                {
                    const auto &base =
                        get(
                            domain,
                            point);

                    DomainEvaluationWeights
                        rotated;

                    if (base.direct_index.has_value())
                    {
                        rotated.direct_index =
                            (*base.direct_index + 1U) %
                            domain->size;

                        return rotated;
                    }

                    if (base.native_weights.size() !=
                        domain->size)
                    {
                        throw std::runtime_error(
                            "cached barycentric weight size mismatch");
                    }

                    rotated.native_weights.resize(
                        domain->size);

                    for (std::size_t index = 0;
                         index < domain->size;
                         ++index)
                    {
                        const auto source_index =
                            index == 0
                                ? domain->size - 1U
                                : index - 1U;

                        rotated.native_weights[index] =
                            base.native_weights[
                                source_index];
                    }

                    return rotated;
                });
        }

    private:
        template <typename Builder>
        const DomainEvaluationWeights&
        get_or_build(
            const std::string& cache_key,
            Builder&& builder)
        {
            {
                std::unique_lock<std::mutex>
                    lock(mutex_);

                for (;;)
                {
                    if (const auto it =
                            entries_.find(
                                cache_key);
                        it != entries_.end())
                    {
                        return it->second;
                    }

                    if (!in_flight_.count(
                            cache_key))
                    {
                        in_flight_.insert(
                            cache_key);

                        break;
                    }

                    cv_.wait(lock);
                }
            }

            try
            {
                auto entry =
                    builder();

                std::lock_guard<std::mutex>
                    lock(mutex_);

                in_flight_.erase(
                    cache_key);

                const auto [it, inserted] =
                    entries_.emplace(
                        cache_key,
                        std::move(entry));

                (void)inserted;

                cv_.notify_all();

                return it->second;
            }
            catch (...)
            {
                {
                    std::lock_guard<std::mutex>
                        lock(mutex_);

                    in_flight_.erase(
                        cache_key);
                }

                cv_.notify_all();

                throw;
            }
        }

        std::mutex mutex_;
        std::condition_variable cv_;
        std::unordered_set<std::string>
            in_flight_;
        std::unordered_map<
            std::string,
            DomainEvaluationWeights>
            entries_;
    };
}
