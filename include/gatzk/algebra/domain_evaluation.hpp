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
        // 线程安全地缓存“域 + 挑战点”对应的求值权重，避免 prove / verify
        // 在同一 formal run 中重复构造重心权重。
        const DomainEvaluationWeights& get(
            const std::shared_ptr<RootOfUnityDomain>& domain,
            const FieldElement& point)
        {
            const auto cache_key = domain_point_cache_key(domain, point);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                // 快路径：同一域点权重已经构造完成时直接复用。
                if (const auto it = entries_.find(cache_key); it != entries_.end())
                {
                    return it->second;
                }
                // 避免同一 key 被多线程重复构造，其他线程在这里等待首个构造完成。
                while (in_flight_.count(cache_key))
                {
                    cv_.wait(lock);
                    if (const auto it = entries_.find(cache_key); it != entries_.end())
                    {
                        return it->second;
                    }
                }
                in_flight_.insert(cache_key);
            }

            auto entry = build_domain_evaluation_weights(domain, point);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                in_flight_.erase(cache_key);
                const auto [it, inserted] = entries_.emplace(cache_key, std::move(entry));
                (void)inserted;
                cv_.notify_all();
                return it->second;
            }
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        std::unordered_set<std::string> in_flight_;
        std::unordered_map<std::string, DomainEvaluationWeights> entries_;
    };
}
