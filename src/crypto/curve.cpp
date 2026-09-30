#include "gatzk/crypto/curve.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/cuda_msm.hpp"
#include "gatzk/util/route2.hpp"
#define GLV2 GATZK_UNUSED_GLV2
#include "../../third_party/mcl/src/glv.hpp"
#undef GLV2

namespace gatzk::crypto
{
    // 实现 KZG 依赖的群运算、批量 MSM 与序列化，是 protocol 层访问曲线后端的唯一实现落点。
    // 属于 crypto 执行层，不允许改写承诺含义或 配对方程。

    namespace
    {
        struct ScalarKey
        {
            std::array<std::uint8_t, 32> bytes{};

            bool operator==(const ScalarKey &other) const
            {
                return bytes == other.bytes;
            }
        };

        class MsmWorkerPool
        {
        public:
            explicit MsmWorkerPool(std::size_t background_worker_count)
            {
                workers_.reserve(background_worker_count);
                for (std::size_t worker = 0; worker < background_worker_count; ++worker)
                {
                    workers_.emplace_back([this]() { worker_loop(); });
                }
            }

            ~MsmWorkerPool()
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping_ = true;
                }
                ready_.notify_all();
                for (auto &worker : workers_)
                {
                    worker.join();
                }
            }

            MsmWorkerPool(const MsmWorkerPool &) = delete;
            MsmWorkerPool &operator=(const MsmWorkerPool &) = delete;

            template <class Fn> void parallel_for(std::size_t task_count, Fn &&fn)
            {
                if (task_count <= 1 || workers_.empty())
                {
                    fn(0);
                    return;
                }

                struct TaskGroup
                {
                    explicit TaskGroup(std::size_t queued) : remaining(queued) {}

                    void record_failure(std::exception_ptr error)
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (failure == nullptr)
                        {
                            failure = std::move(error);
                        }
                    }

                    void finish_one()
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (--remaining == 0)
                        {
                            finished.notify_one();
                        }
                    }

                    void wait()
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        finished.wait(lock, [&]() { return remaining == 0; });
                    }

                    std::mutex mutex;
                    std::condition_variable finished;
                    std::size_t remaining;
                    std::exception_ptr failure;
                };

                const auto queued_task_count = task_count - 1;
                auto group = std::make_shared<TaskGroup>(queued_task_count);
                auto operation = std::make_shared<std::decay_t<Fn>>(std::forward<Fn>(fn));
                for (std::size_t task = 0; task < queued_task_count; ++task)
                {
                    enqueue(
                        [group, operation, task]()
                        {
                            try
                            {
                                (*operation)(task);
                            }
                            catch (...)
                            {
                                group->record_failure(std::current_exception());
                            }
                            group->finish_one();
                        });
                }

                try
                {
                    (*operation)(task_count - 1);
                }
                catch (...)
                {
                    group->record_failure(std::current_exception());
                }
                group->wait();
                if (group->failure != nullptr)
                {
                    std::rethrow_exception(group->failure);
                }
            }

        private:
            void enqueue(std::function<void()> task)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    tasks_.push_back(std::move(task));
                }
                ready_.notify_one();
            }

            void worker_loop()
            {
                while (true)
                {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        ready_.wait(lock, [&]() { return stopping_ || !tasks_.empty(); });
                        if (stopping_ && tasks_.empty())
                        {
                            return;
                        }
                        task = std::move(tasks_.front());
                        tasks_.pop_front();
                    }
                    task();
                }
            }

            std::mutex mutex_;
            std::condition_variable ready_;
            std::deque<std::function<void()>> tasks_;
            std::vector<std::thread> workers_;
            bool stopping_ = false;
        };

        MsmWorkerPool &msm_worker_pool()
        {
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            static MsmWorkerPool pool(cpu_count > 1 ? cpu_count - 1 : 0);
            return pool;
        }

        MsmWorkerPool &msm_batch_worker_pool()
        {
            const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
            static MsmWorkerPool pool(cpu_count > 1 ? cpu_count - 1 : 0);
            return pool;
        }

        struct StableBaseSliceKey
        {
            const G1Point *points = nullptr;
            std::size_t count = 0;

            bool operator==(const StableBaseSliceKey &other) const
            {
                return points == other.points && count == other.count;
            }
        };

        struct StableBaseSliceKeyHash
        {
            std::size_t operator()(const StableBaseSliceKey &key) const
            {
                auto hash = static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(key.points));
                hash ^= key.count + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
                return hash;
            }
        };

        class StableBaseGLVCache
        {
        public:
            std::shared_ptr<const std::vector<mcl::G1>> get_or_prepare(const G1Point *points, std::size_t count)
            {
                constexpr std::size_t kMinPreparedSlice = 128;
                constexpr std::size_t kMaxPreparedSlice = 4096;
                constexpr std::size_t kMaxExpandedPointCount = 1ULL << 20;
                if (points == nullptr || count < kMinPreparedSlice || count > kMaxPreparedSlice)
                {
                    return nullptr;
                }

                const StableBaseSliceKey key{
                    .points = points,
                    .count = count,
                };
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto [it, inserted] = entries_.try_emplace(key);
                    if (inserted)
                    {
                        it->second.seen = true;
                        return nullptr;
                    }
                    if (it->second.prepared != nullptr)
                    {
                        return it->second.prepared;
                    }
                    if (it->second.preparing || expanded_point_count_ + count * 2 > kMaxExpandedPointCount)
                    {
                        return nullptr;
                    }
                    it->second.preparing = true;
                }

                auto prepared = std::make_shared<std::vector<mcl::G1>>(count * 2);
                auto *native_points = const_cast<mcl::G1 *>(reinterpret_cast<const mcl::G1 *>(points));
                mcl::G1::normalizeVec(prepared->data(), native_points, count);
                for (std::size_t i = 0; i < count; ++i)
                {
                    mcl::GLV1::mulLambda((*prepared)[count + i], (*prepared)[i]);
                }

                std::lock_guard<std::mutex> lock(mutex_);
                auto &entry = entries_.at(key);
                if (entry.prepared == nullptr)
                {
                    expanded_point_count_ += prepared->size();
                    entry.prepared = std::move(prepared);
                }
                entry.preparing = false;
                return entry.prepared;
            }

        private:
            struct Entry
            {
                bool seen = false;
                bool preparing = false;
                std::shared_ptr<const std::vector<mcl::G1>> prepared;
            };

            std::mutex mutex_;
            std::unordered_map<StableBaseSliceKey, Entry, StableBaseSliceKeyHash> entries_;
            std::size_t expanded_point_count_ = 0;
        };

        StableBaseGLVCache &stable_base_glv_cache()
        {
            static StableBaseGLVCache cache;
            return cache;
        }

        struct ScalarKeyHash
        {
            std::size_t operator()(const ScalarKey &key) const
            {
                std::uint64_t hash = 1469598103934665603ULL;
                for (const auto byte : key.bytes)
                {
                    hash ^= byte;
                    hash *= 1099511628211ULL;
                }
                return static_cast<std::size_t>(hash);
            }
        };

        ScalarKey scalar_key(const algebra::FieldElement &scalar)
        {
            ScalarKey key;
            scalar.write_little_endian(key.bytes.data(), key.bytes.size());
            return key;
        }

        // 同基点批量乘法会反复命中相同 `(base, scalar)`，这里用字符串键复用结果。
        std::string same_base_cache_key(const G1Point &point, const algebra::FieldElement &scalar)
        {
            return point.to_string() + "|" + scalar.to_string();
        }

        // 仅缓存执行层群乘结果，不跨越协议对象边界。
        class SameBaseMulCache
        {
        public:
            bool lookup(const G1Point &point, const algebra::FieldElement &scalar, G1Point *out)
            {
                const auto key = same_base_cache_key(point, scalar);
                std::lock_guard<std::mutex> lock(mutex_);
                if (const auto it = entries_.find(key); it != entries_.end())
                {
                    if (out != nullptr)
                    {
                        *out = it->second;
                    }
                    return true;
                }
                return false;
            }

            void store(const G1Point &point, const algebra::FieldElement &scalar, const G1Point &value)
            {
                const auto key = same_base_cache_key(point, scalar);
                std::lock_guard<std::mutex> lock(mutex_);
                entries_.emplace(key, value);
            }

        private:
            std::mutex mutex_;
            std::unordered_map<std::string, G1Point> entries_;
        };

        // 进程级单例缓存，供 MSM / 承诺等热点复用。
        SameBaseMulCache &same_base_mul_cache()
        {
            static SameBaseMulCache cache;
            return cache;
        }

        // 序列化接口同时服务 transcript 吸收与 artifact 导出。
        template <class Point> std::vector<std::uint8_t> serialize_impl(const Point &point)
        {
            algebra::ensure_mcl_field_ready();
            std::array<std::uint8_t, 256> buffer{};
            const auto written = point.serialize(buffer.data(), buffer.size());
            if (written == 0)
            {
                throw std::runtime_error("failed to serialize mcl point");
            }
            return std::vector<std::uint8_t>(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(written));
        }

    }

    // 文本化输出主要服务日志、manifest 与调试，不参与协议运算。
    std::string G1Point::to_string() const
    {
        algebra::ensure_mcl_field_ready();
        return value.serializeToHexStr();
    }

    std::string G2Point::to_string() const
    {
        algebra::ensure_mcl_field_ready();
        return value.serializeToHexStr();
    }

    // backend 名称会进入 artifact，帮助 README 与 manifest 对齐当前实现。
    std::string backend_name()
    {
        return GATZK_CRYPTO_BACKEND_NAME;
    }

    G1Point g1_zero()
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        out.value.clear();
        return out;
    }

    G2Point g2_zero()
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        out.value.clear();
        return out;
    }

    G1Point g1_add(const G1Point &lhs, const G1Point &rhs)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G1::add(out.value, lhs.value, rhs.value);
        return out;
    }

    G1Point g1_sub(const G1Point &lhs, const G1Point &rhs)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G1::sub(out.value, lhs.value, rhs.value);
        return out;
    }

    G1Point g1_neg(const G1Point &point)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G1::neg(out.value, point.value);
        return out;
    }

    G1Point g1_mul(const G1Point &point, const algebra::FieldElement &scalar)
    {
        G1Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G1::mul(out.value, point.value, scalar.native());
        return out;
    }

    std::size_t msm_worker_count(std::size_t scalar_count, std::size_t cpu_count)
    {
        if (scalar_count < 256 || cpu_count <= 1)
        {
            return 1;
        }
        std::size_t by_size = 1;
        if (scalar_count >= (1ULL << 18))
        {
            by_size = 16;
        }
        else if (scalar_count >= (1ULL << 16))
        {
            by_size = 8;
        }
        else if (scalar_count >= (1ULL << 13))
        {
            by_size = 16;
        }
        else if (scalar_count >= 256)
        {
            by_size = 16;
        }
        else
        {
            by_size = 1;
        }
        return std::max<std::size_t>(1, std::min(cpu_count, by_size));
    }

    void mcl_mul_vec(mcl::G1 &out, const G1Point *points, const algebra::FieldElement *scalars, std::size_t count, bool stable_srs_bases)
    {
        static_assert(sizeof(G1Point) == sizeof(mcl::G1), "G1Point must wrap a single mcl::G1");
        if (stable_srs_bases)
        {
            const auto prepared = stable_base_glv_cache().get_or_prepare(points, count);
            if (prepared != nullptr)
            {
                const auto split_count = count * mcl::GLV1::splitN;
                const auto unit_count = mcl::Fr::getUnitSize();
                thread_local std::vector<mcl::G1> signed_points;
                thread_local std::vector<mcl::Unit> split_scalars;
                signed_points.assign(prepared->begin(), prepared->end());
                split_scalars.resize(split_count * unit_count);
                mpz_class scalar;
                mpz_class split[mcl::GLV1::splitN];
                for (std::size_t i = 0; i < count; ++i)
                {
                    bool ok = false;
                    scalars[i].native().getMpz(&ok, scalar);
                    if (!ok)
                    {
                        throw std::runtime_error("failed to export scalar for stable-base GLV MSM");
                    }
                    mcl::GLV1::split(split, scalar);
                    for (std::size_t part = 0; part < mcl::GLV1::splitN; ++part)
                    {
                        const auto index = part * count + i;
                        if (split[part] < 0)
                        {
                            split[part] = -split[part];
                            mcl::G1::neg(signed_points[index], signed_points[index]);
                        }
                        mcl::gmp::getArray(&ok, split_scalars.data() + index * unit_count, unit_count, split[part]);
                        if (!ok)
                        {
                            throw std::runtime_error("failed to split scalar for stable-base GLV MSM");
                        }
                    }
                }
                mcl::ec::mulVecLong(out, signed_points.data(), split_scalars.data(), unit_count, unit_count, split_count, 0, false);
                return;
            }
        }
        auto *native_points = const_cast<mcl::G1 *>(reinterpret_cast<const mcl::G1 *>(points));
        const auto *native_scalars = algebra::FieldElement::native_array_data(scalars);
        mcl::G1::mulVec(out, native_points, native_scalars, count);
    }

    void msm_execute_single(mcl::G1 &out, const G1Point *points, const algebra::FieldElement *scalars, std::size_t count, bool stable_srs_bases)
    {
        mcl_mul_vec(out, points, scalars, count, stable_srs_bases);
    }

    void msm_execute_chunked(mcl::G1 &out, const G1Point *points, const algebra::FieldElement *scalars, std::size_t count, std::size_t worker_count,
                             bool stable_srs_bases)
    {
        if (worker_count <= 1 || count < 256)
        {
            msm_execute_single(out, points, scalars, count, stable_srs_bases);
            return;
        }
        const auto task_count = std::min<std::size_t>(worker_count * 3, count);
        const auto base_chunk_size = count / task_count;
        const auto larger_chunk_count = count % task_count;
        std::vector<mcl::G1> partials(task_count);
        msm_worker_pool().parallel_for(task_count,
                                       [&](std::size_t task)
                                       {
                                           const auto begin = task * base_chunk_size + std::min(task, larger_chunk_count);
                                           const auto end = begin + base_chunk_size + (task < larger_chunk_count ? 1 : 0);
                                           msm_execute_single(partials[task], points + begin, scalars + begin, end - begin, stable_srs_bases);
                                       });
        for (const auto &partial : partials)
        {
            mcl::G1::add(out, out, partial);
        }
    }

    bool should_try_grouped_scalar_msm(const algebra::FieldElement *scalars, std::size_t count, const algebra::FieldElement &zero_scalar,
                                       std::size_t nonzero_count)
    {
        if (!util::route2_options().fast_msm || count < 4096 || nonzero_count < 512)
        {
            return false;
        }
        constexpr std::size_t kTargetSamples = 2048;
        const auto stride = std::max<std::size_t>(1, count / kTargetSamples);
        std::unordered_map<ScalarKey, std::size_t, ScalarKeyHash> sampled;
        sampled.reserve(std::min<std::size_t>(kTargetSamples, nonzero_count));
        std::size_t sampled_nonzero = 0;
        for (std::size_t i = 0; i < count && sampled_nonzero < kTargetSamples; i += stride)
        {
            if (scalars[i] == zero_scalar)
            {
                continue;
            }
            sampled.emplace(scalar_key(scalars[i]), sampled_nonzero);
            ++sampled_nonzero;
        }
        if (sampled_nonzero < 128)
        {
            return false;
        }
        return sampled.size() * 2 <= sampled_nonzero;
    }

    bool sample_single_nonzero_scalar(const algebra::FieldElement *scalars, std::size_t count, const algebra::FieldElement &zero_scalar,
                                      std::size_t nonzero_count, algebra::FieldElement *representative)
    {
        if (!util::route2_options().fast_msm || count < 4096 || nonzero_count < 512 || representative == nullptr)
        {
            return false;
        }
        constexpr std::size_t kTargetSamples = 2048;
        const auto stride = std::max<std::size_t>(1, count / kTargetSamples);
        bool found = false;
        std::size_t sampled_nonzero = 0;
        algebra::FieldElement candidate = zero_scalar;
        for (std::size_t i = 0; i < count && sampled_nonzero < kTargetSamples; i += stride)
        {
            if (scalars[i] == zero_scalar)
            {
                continue;
            }
            if (!found)
            {
                candidate = scalars[i];
                found = true;
            }
            else if (!(scalars[i] == candidate))
            {
                return false;
            }
            ++sampled_nonzero;
        }
        const auto min_samples = std::min<std::size_t>(128, nonzero_count);
        if (!found || sampled_nonzero < min_samples)
        {
            return false;
        }
        *representative = candidate;
        return true;
    }

    bool try_single_nonzero_scalar_msm(G1Point &out, const G1Point *points, const algebra::FieldElement *scalars, std::size_t count,
                                       const algebra::FieldElement &zero_scalar, std::size_t nonzero_count, G1MSMProfile *profile,
                                       const G1MSMExecutionOptions *options)
    {
        algebra::FieldElement representative = zero_scalar;
        if (!sample_single_nonzero_scalar(scalars, count, zero_scalar, nonzero_count, &representative))
        {
            return false;
        }

        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        auto worker_count = msm_worker_count(nonzero_count, cpu_count);
        if (options != nullptr && options->max_worker_count != 0)
        {
            worker_count = std::min(worker_count, options->max_worker_count);
        }
        worker_count = std::max<std::size_t>(1, worker_count);

        struct PartialSum
        {
            bool ok = true;
            mcl::G1 sum;
        };

        auto accumulate_range = [&](std::size_t begin, std::size_t end)
        {
            PartialSum partial;
            partial.sum.clear();
            for (std::size_t i = begin; i < end; ++i)
            {
                if (scalars[i] == zero_scalar)
                {
                    continue;
                }
                if (!(scalars[i] == representative))
                {
                    partial.ok = false;
                    return partial;
                }
                mcl::G1::add(partial.sum, partial.sum, points[i].value);
            }
            return partial;
        };

        mcl::G1 point_sum;
        point_sum.clear();
        bool ok = true;
        if (worker_count <= 1 || count < 4096)
        {
            auto partial = accumulate_range(0, count);
            ok = partial.ok;
            point_sum = partial.sum;
        }
        else
        {
            const auto task_count = std::min<std::size_t>(worker_count, count);
            const auto chunk_size = (count + task_count - 1) / task_count;
            std::vector<PartialSum> partials(task_count);
            msm_worker_pool().parallel_for(task_count,
                                           [&](std::size_t task)
                                           {
                                               const auto begin = task * chunk_size;
                                               const auto end = std::min(count, begin + chunk_size);
                                               partials[task] = accumulate_range(begin, end);
                                           });
            for (const auto &partial : partials)
            {
                if (!partial.ok)
                {
                    ok = false;
                }
                mcl::G1::add(point_sum, point_sum, partial.sum);
            }
        }
        if (!ok)
        {
            return false;
        }
        mcl::G1::mul(out.value, point_sum, representative.native());
        if (profile != nullptr)
        {
            profile->zero_scalar_skipping = nonzero_count != count;
            profile->sparse_compaction = true;
            profile->effective_scalar_count = 1;
            profile->multithreaded = worker_count > 1;
        }
        return true;
    }

    struct PendingCudaMsmRequest
    {
        G1Point *output = nullptr;
        const G1Point *points = nullptr;
        const algebra::FieldElement *scalars = nullptr;

        const std::uint64_t *device_montgomery_scalars = nullptr;

        std::size_t device_montgomery_scalar_capacity = 0;

        int device_montgomery_scalar_device = -1;

        std::size_t count = 0;
        std::size_t stable_srs_capacity = 0;
        bool executed_on_cuda = false;
        bool done = false;
        std::string error;
    };

    class CudaMsmBatchWindow
    {
    public:
        explicit CudaMsmBatchWindow(std::size_t active_workers) : active_workers_(active_workers) {}

        bool submit(G1Point *output, const G1Point *points, const algebra::FieldElement *scalars, std::size_t count, std::size_t stable_srs_capacity,
                    const std::uint64_t *device_montgomery_scalars, std::size_t device_montgomery_scalar_capacity, int device_montgomery_scalar_device)
        {
            PendingCudaMsmRequest request{
                .output = output,
                .points = points,
                .scalars = scalars,
                .device_montgomery_scalars = device_montgomery_scalars,
                .device_montgomery_scalar_capacity = device_montgomery_scalar_capacity,
                .device_montgomery_scalar_device = device_montgomery_scalar_device,
                .count = count,
                .stable_srs_capacity = stable_srs_capacity,
                .executed_on_cuda = false,
                .done = false,
                .error = {},
            };

            std::unique_lock<std::mutex> lock(mutex_);

            pending_.push_back(&request);

            ready_.notify_all();

            while (!request.done)
            {
                drain_if_ready(lock);

                if (!request.done)
                {
                    ready_.wait(lock);
                }
            }

            if (!request.error.empty())
            {
                throw std::runtime_error(request.error);
            }

            return request.executed_on_cuda;
        }

        void worker_finished()
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (active_workers_ == 0)
            {
                return;
            }

            --active_workers_;

            drain_if_ready(lock);

            ready_.notify_all();
        }

    private:
        struct CompatibleGroup
        {
            const G1Point *points = nullptr;
            std::size_t stable_srs_capacity = 0;

            bool resident_scalars = false;
            int resident_scalar_device = -1;

            std::vector<PendingCudaMsmRequest *> requests;
        };

        static void set_request_error(PendingCudaMsmRequest *request, const std::string &message)
        {
            if (request != nullptr && request->error.empty())
            {
                request->error = message;
            }
        }

        void execute_requests(std::vector<PendingCudaMsmRequest *> &requests)
        {
            std::vector<CompatibleGroup> groups;

            groups.reserve(requests.size());

            for (auto *request : requests)
            {
                auto group = std::find_if(groups.begin(), groups.end(),
                                          [&](const CompatibleGroup &candidate)
                                          {
                                              const bool request_resident = request->device_montgomery_scalars != nullptr;

                                              return candidate.points == request->points && candidate.stable_srs_capacity == request->stable_srs_capacity &&
                                                     candidate.resident_scalars == request_resident &&
                                                     (!request_resident || candidate.resident_scalar_device == request->device_montgomery_scalar_device);
                                          });

                if (group == groups.end())
                {
                    CompatibleGroup created;
                    created.points = request->points;

                    created.stable_srs_capacity = request->stable_srs_capacity;

                    created.resident_scalars = request->device_montgomery_scalars != nullptr;

                    created.resident_scalar_device = created.resident_scalars ? request->device_montgomery_scalar_device : -1;
                    created.requests.push_back(request);

                    groups.push_back(std::move(created));
                }
                else
                {
                    group->requests.push_back(request);
                }
            }

            const std::size_t kMaximumBatchSize = cuda_g1_msm_integrated_memory_device() ? 16 : 8;

            for (auto &group : groups)
            {
                for (std::size_t offset = 0; offset < group.requests.size(); offset += kMaximumBatchSize)
                {
                    const auto current_batch = std::min<std::size_t>(kMaximumBatchSize, group.requests.size() - offset);

                    try
                    {
                        if (current_batch == 1)
                        {
                            auto *request = group.requests[offset];

                            // The first stable-SRS CUDA use is allowed to seed
                            // the resident base cache.  Returning a cold
                            // singleton to CPU prevents later commitments and
                            // openings from amortizing the immutable SRS
                            // preparation cost.
                            if (!try_cuda_g1_msm(request->output, request->points, request->scalars, request->count, true, request->stable_srs_capacity,
                                                 request->device_montgomery_scalars, request->device_montgomery_scalar_capacity,
                                                 request->device_montgomery_scalar_device))
                            {
                                throw std::runtime_error("CUDA stable-SRS singleton MSM "
                                                         "attempted an unexpected fallback");
                            }

                            algebra::record_cuda_msm_single_call();

                            request->executed_on_cuda = true;

                            continue;
                        }

                        std::vector<G1Point *> outputs(current_batch);

                        std::vector<const algebra::FieldElement *> scalar_vectors(current_batch);

                        std::vector<std::size_t> counts(current_batch);

                        std::vector<const std::uint64_t *> device_scalar_vectors;

                        std::vector<std::size_t> device_scalar_capacities;

                        std::vector<int> device_scalar_devices;

                        if (group.resident_scalars)
                        {
                            device_scalar_vectors.resize(current_batch);

                            device_scalar_capacities.resize(current_batch);

                            device_scalar_devices.resize(current_batch);
                        }

                        std::size_t max_count = 0;

                        for (std::size_t batch = 0; batch < current_batch; ++batch)
                        {
                            auto *request = group.requests[offset + batch];

                            outputs[batch] = request->output;

                            scalar_vectors[batch] = request->scalars;

                            counts[batch] = request->count;

                            if (group.resident_scalars)
                            {
                                device_scalar_vectors[batch] = request->device_montgomery_scalars;

                                device_scalar_capacities[batch] = request->device_montgomery_scalar_capacity;

                                device_scalar_devices[batch] = request->device_montgomery_scalar_device;
                            }

                            max_count = std::max(max_count, counts[batch]);
                        }

                        if (!try_cuda_g1_msm_same_bases_batch(outputs.data(), group.points, scalar_vectors.data(), counts.data(), max_count, current_batch,
                                                              group.stable_srs_capacity, group.resident_scalars ? device_scalar_vectors.data() : nullptr,
                                                              group.resident_scalars ? device_scalar_capacities.data() : nullptr,
                                                              group.resident_scalars ? device_scalar_devices.data() : nullptr))
                        {
                            throw std::runtime_error("CUDA same-SRS batch "
                                                     "attempted a CPU fallback");
                        }

                        algebra::record_cuda_msm_batch_call();

                        for (std::size_t batch = 0; batch < current_batch; ++batch)
                        {
                            group.requests[offset + batch]->executed_on_cuda = true;
                        }
                    }
                    catch (const std::exception &error)
                    {
                        for (std::size_t batch = 0; batch < current_batch; ++batch)
                        {
                            set_request_error(group.requests[offset + batch], error.what());
                        }
                    }
                    catch (...)
                    {
                        for (std::size_t batch = 0; batch < current_batch; ++batch)
                        {
                            set_request_error(group.requests[offset + batch], "unknown CUDA same-SRS "
                                                                              "batch failure");
                        }
                    }
                }
            }
        }

        void drain_if_ready(std::unique_lock<std::mutex> &lock)
        {
            if (draining_ || pending_.empty() || pending_.size() < active_workers_)
            {
                return;
            }

            auto requests = std::move(pending_);

            pending_.clear();
            draining_ = true;

            lock.unlock();

            try
            {
                execute_requests(requests);
            }
            catch (const std::exception &error)
            {
                for (auto *request : requests)
                {
                    set_request_error(request, error.what());
                }
            }
            catch (...)
            {
                for (auto *request : requests)
                {
                    set_request_error(request, "unknown CUDA MSM batch "
                                               "coordinator failure");
                }
            }

            lock.lock();

            for (auto *request : requests)
            {
                request->done = true;
            }

            draining_ = false;

            ready_.notify_all();
        }

        std::mutex mutex_;
        std::condition_variable ready_;

        std::size_t active_workers_ = 0;

        bool draining_ = false;

        std::vector<PendingCudaMsmRequest *> pending_;
    };

    CudaMsmBatchWindow *&current_cuda_msm_batch_window()
    {
        thread_local CudaMsmBatchWindow *window = nullptr;

        return window;
    }

    class ScopedCudaMsmBatchWindow
    {
    public:
        explicit ScopedCudaMsmBatchWindow(CudaMsmBatchWindow *window) : previous_(current_cuda_msm_batch_window())
        {
            current_cuda_msm_batch_window() = window;
        }

        ~ScopedCudaMsmBatchWindow()
        {
            current_cuda_msm_batch_window() = previous_;
        }

    private:
        CudaMsmBatchWindow *previous_ = nullptr;
    };

    bool should_use_cuda_msm_shape(const G1Point *points, std::size_t count, bool stable_srs_bases, std::size_t stable_srs_capacity)
    {
        if (count == 0)
        {
            return false;
        }

        const bool resident = stable_srs_bases && cuda_g1_msm_resident_prefix_available(points, count, stable_srs_capacity);

        if (stable_srs_bases)
        {
            if (resident)
            {
                algebra::record_cuda_resident_hit();
            }
            else
            {
                algebra::record_cuda_resident_miss();
            }
        }

        // A same-SRS batch window is an explicit request to coalesce several
        // immutable-base MSMs.  Admit the 2^12+ shape even when the resident
        // base cache is still cold: the batch coordinator owns the single
        // base upload and seeds residency for the whole wave.
        if (stable_srs_bases && current_cuda_msm_batch_window() != nullptr && count >= (std::size_t{1} << 12))
        {
            return true;
        }

        if (resident)
        {
            return count >= (std::size_t{1} << 12);
        }

        return stable_srs_bases ? count >= (std::size_t{1} << 14) : count >= (std::size_t{1} << 16);
    }

    bool execute_cuda_msm_when_requested(G1Point *output, const G1Point *points, const algebra::FieldElement *scalars, std::size_t count, bool stable_srs_bases,
                                         std::size_t stable_srs_capacity, const std::uint64_t *device_montgomery_scalars = nullptr,
                                         std::size_t device_montgomery_scalar_capacity = 0, int device_montgomery_scalar_device = -1)
    {
        if (!util::route2_options().cuda_field_hotspots)
        {
            return false;
        }
        if (device_montgomery_scalars == nullptr && !should_use_cuda_msm_shape(points, count, stable_srs_bases, stable_srs_capacity))
        {
            return false;
        }

        constexpr std::size_t kMinimumCoalescedMsmCount = 1ULL << 12;

        if (stable_srs_bases && count >= kMinimumCoalescedMsmCount)
        {
            if (auto *batch_window = current_cuda_msm_batch_window(); batch_window != nullptr)
            {
                return batch_window->submit(output, points, scalars, count, stable_srs_capacity, device_montgomery_scalars, device_montgomery_scalar_capacity,
                                            device_montgomery_scalar_device);
            }
        }

        if (!try_cuda_g1_msm(output, points, scalars, count, stable_srs_bases, stable_srs_capacity, device_montgomery_scalars,
                             device_montgomery_scalar_capacity, device_montgomery_scalar_device))
        {
            throw std::runtime_error("cuda_hotspots mode attempted a G1 MSM "
                                     "CPU fallback");
        }

        algebra::record_cuda_msm_single_call();

        return true;
    }

    bool try_grouped_scalar_msm(G1Point &out, const G1Point *points, const algebra::FieldElement *scalars, std::size_t count,
                                const algebra::FieldElement &zero_scalar, std::size_t nonzero_count, G1MSMProfile *profile,
                                const G1MSMExecutionOptions *options)
    {
        struct Group
        {
            algebra::FieldElement scalar;
            G1Point point_sum;
        };

        const auto max_group_count = std::max<std::size_t>(64, nonzero_count / 2);
        std::unordered_map<ScalarKey, std::size_t, ScalarKeyHash> group_index;
        group_index.reserve(std::min<std::size_t>(max_group_count + 1, nonzero_count));
        std::vector<Group> groups;
        groups.reserve(std::min<std::size_t>(max_group_count, nonzero_count));

        for (std::size_t i = 0; i < count; ++i)
        {
            if (scalars[i] == zero_scalar)
            {
                continue;
            }
            const auto key = scalar_key(scalars[i]);
            const auto [it, inserted] = group_index.emplace(key, groups.size());
            if (inserted)
            {
                if (groups.size() >= max_group_count)
                {
                    return false;
                }
                groups.push_back(Group{
                    .scalar = scalars[i],
                    .point_sum = points[i],
                });
            }
            else
            {
                auto &point_sum = groups[it->second].point_sum.value;
                mcl::G1::add(point_sum, point_sum, points[i].value);
            }
        }

        if (groups.empty())
        {
            return true;
        }
        if (groups.size() * 2 > nonzero_count)
        {
            return false;
        }

        std::vector<G1Point> grouped_points;
        std::vector<algebra::FieldElement> grouped_scalars;
        grouped_points.reserve(groups.size());
        grouped_scalars.reserve(groups.size());
        for (auto &group : groups)
        {
            grouped_points.push_back(group.point_sum);
            grouped_scalars.push_back(group.scalar);
        }

        if (execute_cuda_msm_when_requested(&out, grouped_points.data(), grouped_scalars.data(), grouped_scalars.size(), false, 0))
        {
            if (profile != nullptr)
            {
                profile->zero_scalar_skipping = nonzero_count != count;

                profile->sparse_compaction = true;

                profile->effective_scalar_count = grouped_scalars.size();
            }

            return true;
        }

        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        auto worker_count = msm_worker_count(grouped_scalars.size(), cpu_count);
        if (options != nullptr && options->max_worker_count != 0)
        {
            worker_count = std::min(worker_count, options->max_worker_count);
        }
        if (worker_count > 1)
        {
            if (profile != nullptr)
            {
                profile->multithreaded = true;
            }
            msm_execute_chunked(out.value, grouped_points.data(), grouped_scalars.data(), grouped_scalars.size(), worker_count, false);
        }
        else
        {
            msm_execute_single(out.value, grouped_points.data(), grouped_scalars.data(), grouped_scalars.size(), false);
        }
        if (profile != nullptr)
        {
            profile->zero_scalar_skipping = nonzero_count != count;
            profile->sparse_compaction = true;
            profile->effective_scalar_count = grouped_scalars.size();
        }
        return true;
    }

    G1Point g1_multi_scalar_mul_profiled(const G1Point *points, const algebra::FieldElement *scalars, std::size_t count, G1MSMProfile *profile,
                                         const G1MSMExecutionOptions *options)
    {
        const auto *early_device_montgomery_scalars = options != nullptr ? options->device_montgomery_scalars : nullptr;

        if ((points == nullptr || (scalars == nullptr && early_device_montgomery_scalars == nullptr)) && count != 0)
        {
            throw std::runtime_error("G1 MSM received null input");
        }
        if (profile != nullptr)
        {
            profile->input_count = count;
            profile->effective_scalar_count = count;
            profile->nonzero_scalar_count = count;
            profile->zero_scalar_skipping = false;
            profile->sparse_compaction = false;
            profile->multithreaded = false;
        }
        G1Point out = g1_zero();
        if (count == 0)
        {
            if (profile != nullptr)
            {
                profile->nonzero_scalar_count = 0;
                profile->effective_scalar_count = 0;
            }
            return out;
        }
        algebra::ensure_mcl_field_ready();
        const bool stable_srs_bases = options != nullptr && options->stable_srs_bases;

        const auto stable_srs_capacity = options != nullptr ? options->stable_srs_capacity : 0;

        const auto *device_montgomery_scalars = options != nullptr ? options->device_montgomery_scalars : nullptr;

        const auto device_montgomery_scalar_capacity = options != nullptr ? options->device_montgomery_scalar_capacity : 0;

        const auto device_montgomery_scalar_device = options != nullptr ? options->device_montgomery_scalar_device : -1;

        if (device_montgomery_scalars != nullptr && device_montgomery_scalar_capacity < count)
        {
            throw std::runtime_error("G1 MSM resident CUDA scalar capacity is smaller than input count");
        }

        const bool resident_nonzero_count_known =
            device_montgomery_scalars != nullptr && options != nullptr && options->device_montgomery_scalar_nonzero_count_known;

        if (resident_nonzero_count_known)
        {
            const auto known_nonzero_count = options->device_montgomery_scalar_nonzero_count;

            if (known_nonzero_count > count)
            {
                throw std::runtime_error("G1 MSM resident CUDA scalar nonzero count exceeds input count");
            }

            if (profile != nullptr)
            {
                profile->nonzero_scalar_count = known_nonzero_count;

                profile->effective_scalar_count = known_nonzero_count == 0 ? 0 : count;

                profile->zero_scalar_skipping = known_nonzero_count != count;

                profile->sparse_compaction = known_nonzero_count == 0;
            }

            if (known_nonzero_count == 0)
            {
                return out;
            }

            if (!execute_cuda_msm_when_requested(&out, points, scalars, count, stable_srs_bases, stable_srs_capacity, device_montgomery_scalars,
                                                 device_montgomery_scalar_capacity, device_montgomery_scalar_device))
            {
                throw std::runtime_error("resident CUDA scalar MSM attempted a CPU fallback");
            }

            return out;
        }

        if (device_montgomery_scalars != nullptr && scalars == nullptr)
        {
            if (profile != nullptr)
            {
                profile->nonzero_scalar_count = count;

                profile->effective_scalar_count = count;

                profile->zero_scalar_skipping = false;

                profile->sparse_compaction = false;
            }

            if (!execute_cuda_msm_when_requested(&out, points, nullptr, count, stable_srs_bases, stable_srs_capacity, device_montgomery_scalars,
                                                 device_montgomery_scalar_capacity, device_montgomery_scalar_device))
            {
                throw std::runtime_error("device-only resident CUDA scalar MSM attempted a CPU fallback");
            }

            return out;
        }

        const auto zero_scalar = algebra::FieldElement::zero();
        std::size_t nonzero_count = 0;
        std::size_t last_nonzero_exclusive = 0;
        if (count >= 1024 || profile != nullptr || device_montgomery_scalars != nullptr)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                if (!(scalars[i] == zero_scalar))
                {
                    ++nonzero_count;
                    last_nonzero_exclusive = i + 1;
                }
            }
            if (profile != nullptr)
            {
                profile->nonzero_scalar_count = nonzero_count;
            }
            if (nonzero_count == 0)
            {
                if (profile != nullptr)
                {
                    profile->zero_scalar_skipping = true;
                    profile->sparse_compaction = true;
                    profile->effective_scalar_count = 0;
                }

                return out;
            }

            if (device_montgomery_scalars != nullptr)
            {
                if (profile != nullptr)
                {
                    profile->effective_scalar_count = count;

                    profile->zero_scalar_skipping = nonzero_count != count;

                    profile->sparse_compaction = false;
                }

                if (!execute_cuda_msm_when_requested(&out, points, scalars, count, stable_srs_bases, stable_srs_capacity, device_montgomery_scalars,
                                                     device_montgomery_scalar_capacity, device_montgomery_scalar_device))
                {
                    throw std::runtime_error("resident CUDA scalar MSM attempted a CPU fallback");
                }

                return out;
            }

            // Preserve immutable SRS bases before any sparse/grouped path
            // constructs temporary base vectors.  Large single MSMs keep the
            // existing threshold.  Smaller 2^12+ objects take this route only
            // while an explicit same-SRS batch window is active, so isolated
            // small MSMs retain the existing CPU heuristics.
            const bool coalesced_same_srs_request = current_cuda_msm_batch_window() != nullptr && count >= (std::size_t{1} << 12);

            if (stable_srs_bases && util::route2_options().cuda_field_hotspots && (count >= (std::size_t{1} << 14) || coalesced_same_srs_request))
            {
                if (profile != nullptr)
                {
                    profile->zero_scalar_skipping = nonzero_count != count;

                    profile->sparse_compaction = false;

                    profile->effective_scalar_count = count;
                }

                if (execute_cuda_msm_when_requested(&out, points, scalars, count, true, stable_srs_capacity))
                {
                    return out;
                }
            }

            if (try_single_nonzero_scalar_msm(out, points, scalars, count, zero_scalar, nonzero_count, profile, options))
            {
                return out;
            }
            if (should_try_grouped_scalar_msm(scalars, count, zero_scalar, nonzero_count) &&
                try_grouped_scalar_msm(out, points, scalars, count, zero_scalar, nonzero_count, profile, options))
            {
                return out;
            }
            if (last_nonzero_exclusive < count && last_nonzero_exclusive == nonzero_count)
            {
                if (profile != nullptr)
                {
                    profile->zero_scalar_skipping = true;
                    profile->sparse_compaction = false;
                    profile->effective_scalar_count = nonzero_count;
                }
                if (execute_cuda_msm_when_requested(&out, points, scalars, nonzero_count, stable_srs_bases, stable_srs_capacity))
                {
                    return out;
                }
                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                auto worker_count = msm_worker_count(nonzero_count, cpu_count);
                if (options != nullptr && options->max_worker_count != 0)
                {
                    worker_count = std::min(worker_count, options->max_worker_count);
                }
                if (worker_count > 1)
                {
                    if (profile != nullptr)
                    {
                        profile->multithreaded = true;
                    }
                    msm_execute_chunked(out.value, points, scalars, nonzero_count, worker_count, stable_srs_bases);
                }
                else
                {
                    msm_execute_single(out.value, points, scalars, nonzero_count, stable_srs_bases);
                }
                return out;
            }
            if (nonzero_count * 4 < count * 3)
            {
                if (profile != nullptr)
                {
                    profile->zero_scalar_skipping = true;
                    profile->sparse_compaction = true;
                    profile->effective_scalar_count = nonzero_count;
                }
                thread_local std::vector<G1Point> compact_points;
                thread_local std::vector<algebra::FieldElement> compact_scalars;
                compact_points.clear();
                compact_scalars.clear();
                compact_points.reserve(nonzero_count);
                compact_scalars.reserve(nonzero_count);
                static_assert(sizeof(G1Point) == sizeof(mcl::G1), "G1Point must wrap a single mcl::G1");
                for (std::size_t i = 0; i < count; ++i)
                {
                    if (!(scalars[i] == zero_scalar))
                    {
                        compact_points.push_back(points[i]);
                        compact_scalars.push_back(scalars[i]);
                    }
                }

                if (execute_cuda_msm_when_requested(&out, compact_points.data(), compact_scalars.data(), compact_scalars.size(), false, 0))
                {
                    return out;
                }

                const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
                auto worker_count = msm_worker_count(compact_scalars.size(), cpu_count);
                if (options != nullptr && options->max_worker_count != 0)
                {
                    worker_count = std::min(worker_count, options->max_worker_count);
                }
                if (worker_count > 1)
                {
                    if (profile != nullptr)
                    {
                        profile->multithreaded = true;
                    }
                    msm_execute_chunked(out.value, compact_points.data(), compact_scalars.data(), compact_scalars.size(), worker_count, false);
                }
                else
                {
                    msm_execute_single(out.value, compact_points.data(), compact_scalars.data(), compact_scalars.size(), false);
                }
                return out;
            }
        }

        if (execute_cuda_msm_when_requested(&out, points, scalars, count, stable_srs_bases, stable_srs_capacity))
        {
            return out;
        }

        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        auto worker_count = msm_worker_count(count, cpu_count);
        if (options != nullptr && options->max_worker_count != 0)
        {
            worker_count = std::min(worker_count, options->max_worker_count);
        }
        if (worker_count > 1)
        {
            if (profile != nullptr)
            {
                profile->multithreaded = true;
            }
            msm_execute_chunked(out.value, points, scalars, count, worker_count, stable_srs_bases);
        }
        else
        {
            msm_execute_single(out.value, points, scalars, count, stable_srs_bases);
        }
        return out;
    }

    G1Point g1_multi_scalar_mul(const G1Point *points, const algebra::FieldElement *scalars, std::size_t count)
    {
        return g1_multi_scalar_mul_profiled(points, scalars, count, nullptr);
    }

    G1Point g1_multi_scalar_mul(const std::vector<G1Point> &points, const std::vector<algebra::FieldElement> &scalars)
    {
        if (points.size() != scalars.size())
        {
            throw std::runtime_error("G1 MSM size mismatch");
        }
        return g1_multi_scalar_mul(points.data(), scalars.data(), points.size());
    }

    void g1_parallel_msm_batch_workers(std::size_t worker_count, const std::function<void(std::size_t)> &worker)
    {
        if (worker_count == 0)
        {
            return;
        }

        const auto hardware_workers = std::max<std::size_t>(1, std::thread::hardware_concurrency());

        // CPU execution, single-worker execution, and nested batch schedulers
        // preserve the existing path exactly.
        if (!util::route2_options().cuda_field_hotspots || worker_count <= 1 || hardware_workers <= 1 || current_cuda_msm_batch_window() != nullptr)
        {
            msm_batch_worker_pool().parallel_for(worker_count, worker);

            return;
        }

        auto batch_window = std::make_shared<CudaMsmBatchWindow>(worker_count);

        msm_batch_worker_pool().parallel_for(worker_count,
                                             [&](std::size_t worker_index)
                                             {
                                                 ScopedCudaMsmBatchWindow batch_scope(batch_window.get());

                                                 try
                                                 {
                                                     worker(worker_index);
                                                 }
                                                 catch (...)
                                                 {
                                                     batch_window->worker_finished();

                                                     throw;
                                                 }

                                                 batch_window->worker_finished();
                                             });
    }

    // 同基点批量乘法是 KZG 承诺与 witness 构造的典型热点。（--fast-msm on）
    std::vector<G1Point> g1_mul_same_base_batch(const G1Point &point, const std::vector<algebra::FieldElement> &scalars)
    {
        algebra::ensure_mcl_field_ready();
        std::vector<G1Point> out(scalars.size());
        if (scalars.empty())
        {
            return out;
        }

        std::vector<std::size_t> miss_indices;
        std::vector<algebra::FieldElement> miss_scalars;
        miss_indices.reserve(scalars.size());
        miss_scalars.reserve(scalars.size());
        // 先把缓存命中的 `(base, scalar)` 拿出来，未命中的再交给后面的 mcl 批处理。
        for (std::size_t i = 0; i < scalars.size(); ++i)
        {
            if (!same_base_mul_cache().lookup(point, scalars[i], &out[i]))
            {
                miss_indices.push_back(i);
                miss_scalars.push_back(scalars[i]);
            }
        }
        if (miss_indices.empty())
        {
            return out;
        }

        if (!util::route2_options().fast_msm)
        {
            for (std::size_t i = 0; i < miss_scalars.size(); ++i)
            {
                auto value = g1_mul(point, miss_scalars[i]);
                out[miss_indices[i]] = value;
                same_base_mul_cache().store(point, miss_scalars[i], value);
            }
            return out;
        }

        // 快速MSM路径
        // 把未命中的标量收成一次 `mulEach` 输入，尽量让同基点批乘只过一遍曲线后端。
        std::vector<mcl::G1> points(miss_scalars.size(), point.value);
        std::vector<mcl::Fr> native_scalars;
        native_scalars.reserve(miss_scalars.size());
        for (const auto &scalar : miss_scalars)
        {
            native_scalars.push_back(scalar.native());
        }

        const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const bool run_parallel = cpu_count > 1 && miss_scalars.size() >= 256;
        if (run_parallel)
        {
            const auto task_count = std::min<std::size_t>(cpu_count, miss_scalars.size() / 128);
            if (task_count > 1)
            {
                const auto chunk_size = (miss_scalars.size() + task_count - 1) / task_count;
                msm_worker_pool().parallel_for(task_count,
                                               [&](std::size_t task)
                                               {
                                                   const auto begin = task * chunk_size;
                                                   const auto end = std::min<std::size_t>(miss_scalars.size(), begin + chunk_size);
                                                   mcl::G1::mulEach(points.data() + begin, native_scalars.data() + begin, end - begin);
                                               });
                for (std::size_t i = 0; i < points.size(); ++i)
                {
                    G1Point value;
                    value.value = points[i];
                    out[miss_indices[i]] = value;
                    same_base_mul_cache().store(point, miss_scalars[i], value);
                }
                return out;
            }
        }

        // 串行回退时也保持同一批量接口，避免逐项乘法和缓存写回交错得太碎。
        mcl::G1::mulEach(points.data(), native_scalars.data(), points.size());
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            G1Point value;
            value.value = points[i];
            out[miss_indices[i]] = value;
            same_base_mul_cache().store(point, miss_scalars[i], value);
        }
        return out;
    }

    G2Point g2_add(const G2Point &lhs, const G2Point &rhs)
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G2::add(out.value, lhs.value, rhs.value);
        return out;
    }

    G2Point g2_sub(const G2Point &lhs, const G2Point &rhs)
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G2::sub(out.value, lhs.value, rhs.value);
        return out;
    }

    G2Point g2_mul(const G2Point &point, const algebra::FieldElement &scalar)
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        mcl::G2::mul(out.value, point.value, scalar.native());
        return out;
    }

    bool operator==(const G1Point &lhs, const G1Point &rhs)
    {
        algebra::ensure_mcl_field_ready();
        return lhs.value == rhs.value;
    }

    bool operator==(const G2Point &lhs, const G2Point &rhs)
    {
        algebra::ensure_mcl_field_ready();
        return lhs.value == rhs.value;
    }

    // 序列化G1点为字节向量
    std::vector<std::uint8_t> serialize(const G1Point &point)
    {
        return serialize_impl(point.value);
    }

    std::size_t serialize(const G1Point &point, std::uint8_t *output, std::size_t capacity)
    {
        if (output == nullptr && capacity != 0)
        {
            throw std::runtime_error("G1 serialization received null output");
        }

        algebra::ensure_mcl_field_ready();

        const auto written = point.value.serialize(output, capacity);

        if (written == 0)
        {
            throw std::runtime_error("failed to serialize mcl G1 point");
        }

        return written;
    }

    std::vector<std::uint8_t> serialize(const G2Point &point)
    {
        return serialize_impl(point.value);
    }

    G1Point deserialize_g1(const std::uint8_t *bytes, std::size_t size)
    {
        if (bytes == nullptr && size != 0)
        {
            throw std::runtime_error("G1 deserialization received null input");
        }

        G1Point out;
        algebra::ensure_mcl_field_ready();

        const auto read = out.value.deserialize(bytes, size);

        if (read != size)
        {
            throw std::runtime_error("failed to deserialize G1 point");
        }

        return out;
    }

    G1Point deserialize_g1(const std::vector<std::uint8_t> &bytes)
    {
        return deserialize_g1(bytes.data(), bytes.size());
    }

    G2Point deserialize_g2(const std::vector<std::uint8_t> &bytes)
    {
        G2Point out;
        algebra::ensure_mcl_field_ready();
        const auto read = out.value.deserialize(bytes.data(), bytes.size());
        if (read != bytes.size())
        {
            throw std::runtime_error("failed to deserialize G2 point");
        }
        return out;
    }

    std::size_t serialized_size(const G1Point &point)
    {
        return serialize(point).size();
    }

    // 返回 G1 点序列化后的字节长度
    std::size_t serialized_size(const G2Point &point)
    {
        return serialize(point).size();
    }

    // 预计算G2点的配对系数，用于加速多次配对运算
    PreparedG2 prepare_g2(const G2Point &point)
    {
        algebra::ensure_mcl_field_ready();
        PreparedG2 prepared;
        mcl::precomputeG2(prepared.coeffs, point.value);
        return prepared;
    }

    bool pairing_equal(const G1Point &lhs_a, const G2Point &rhs_a, const G1Point &lhs_b, const G2Point &rhs_b)
    {
        algebra::ensure_mcl_field_ready();
        mcl::GT eval_a;
        mcl::GT eval_b;
        mcl::pairing(eval_a, lhs_a.value, rhs_a.value);
        mcl::pairing(eval_b, lhs_b.value, rhs_b.value);
        return eval_a == eval_b;
    }

    bool pairing_equal_with_prepared_rhs_a(const G1Point &lhs_a, const PreparedG2 &rhs_a_prepared, const G1Point &lhs_b, const G2Point &rhs_b)
    {
        algebra::ensure_mcl_field_ready();
        mcl::G1 neg_lhs_b;
        mcl::G1::neg(neg_lhs_b, lhs_b.value);
        mcl::Fp12 miller_product;
        mcl::precomputedMillerLoop2mixed(miller_product, neg_lhs_b, rhs_b.value, lhs_a.value, rhs_a_prepared.coeffs);
        mcl::finalExp(miller_product, miller_product);
        return miller_product.isOne();
    }

    // 判断动态点对与预计算点对的配对乘积是否为 1
    bool pairing_product_is_one_mixed_prepared(const G1Point &lhs_dynamic, const G2Point &rhs_dynamic, const G1Point &lhs_prepared,
                                               const PreparedG2 &rhs_prepared)
    {
        algebra::ensure_mcl_field_ready();
        mcl::Fp12 miller_product;
        mcl::precomputedMillerLoop2mixed(miller_product, lhs_dynamic.value, rhs_dynamic.value, lhs_prepared.value, rhs_prepared.coeffs);
        mcl::finalExp(miller_product, miller_product);
        return miller_product.isOne();
    }
}
