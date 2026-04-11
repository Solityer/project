#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/algebra/polynomial.hpp"

namespace gatzk::algebra
{
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
        if (domain->points_precomputed)
        {
            for (std::size_t i = 0; i < domain->points.size(); ++i)
            {
                if (domain->points[i] == point)
                {
                    entry.direct_index = i;
                    break;
                }
            }
        }
        else if (const auto shift = domain->rotation_shift(FieldElement::one(), point); shift.has_value())
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
            const auto cache_key = domain_point_cache_key(domain, point);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (const auto it = entries_.find(cache_key); it != entries_.end())
                {
                    return it->second;
                }
            }

            auto entry = build_domain_evaluation_weights(domain, point);

            std::lock_guard<std::mutex> lock(mutex_);
            const auto [it, inserted] = entries_.emplace(cache_key, std::move(entry));
            (void)inserted;
            return it->second;
        }

    private:
        std::mutex mutex_;
        std::unordered_map<std::string, DomainEvaluationWeights> entries_;
    };
}
