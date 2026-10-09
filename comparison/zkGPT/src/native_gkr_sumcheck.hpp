#pragma once

#include "polynomial.h"
#include "../gat_adapter/native_storage.hpp"

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace zkgpt::native_gkr {

inline std::uint64_t checked_domain_size(int bits, const char* name) {
    if (bits < 0) return 0;
    if (bits >= std::numeric_limits<std::uint64_t>::digits ||
        bits >= std::numeric_limits<std::size_t>::digits) {
        throw std::length_error(std::string(name) + " bit length exceeds native index width");
    }
    return std::uint64_t(1) << bits;
}

inline std::size_t checked_extent(std::uint64_t extent, const char* name) {
    if (extent > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error(std::string(name) + " extent exceeds native storage index width");
    }
    return static_cast<std::size_t>(extent);
}

class EqualityWeightProvider {
    std::vector<F> point_;
    std::vector<F> low_weights_;
    unsigned bits_ = 0;
    unsigned low_bits_ = 0;
    std::uint64_t low_mask_ = 0;
    F scale_ = F_ONE;

    void initialize_low_weights() {
        low_bits_ = std::min<unsigned>(12, bits_);
        const std::size_t count = std::size_t(1) << low_bits_;
        low_weights_.assign(count, F_ZERO);
        low_weights_[0] = scale_;
        for (unsigned bit = 0; bit < low_bits_; ++bit) {
            const std::size_t old_count = std::size_t(1) << bit;
            for (std::size_t index = 0; index < old_count; ++index) {
                const F weight = low_weights_[index];
                low_weights_[index] = weight * (F_ONE - point_[bit]);
                low_weights_[index + old_count] = weight * point_[bit];
            }
        }
        low_mask_ = count - 1;
    }

public:
    EqualityWeightProvider() = default;

    EqualityWeightProvider(const std::vector<F>& point, int bits, const F& scale = F_ONE)
        : point_(point), scale_(scale) {
        if (bits < 0 || bits >= std::numeric_limits<std::uint64_t>::digits ||
            bits >= std::numeric_limits<std::size_t>::digits) {
            throw std::length_error("native equality point bit length is invalid");
        }
        bits_ = static_cast<unsigned>(bits);
        if (point_.size() < bits_ && !scale_.isZero()) throw std::invalid_argument("native equality point is shorter than its domain");
        if (point_.size() < bits_) point_.resize(bits_, F_ZERO);
        initialize_low_weights();
    }

    std::uint64_t domain_size() const {
        return std::uint64_t(1) << bits_;
    }

    F at(std::uint64_t coordinate) const {
        if (coordinate >= domain_size()) throw std::out_of_range("native equality coordinate exceeds its domain");
        F result = low_weights_[static_cast<std::size_t>(coordinate & low_mask_)];
        for (unsigned bit = low_bits_; bit < bits_; ++bit) {
            result *= ((coordinate >> bit) & 1) ? point_[bit] : F_ONE - point_[bit];
        }
        return result;
    }
};

class CombinedEqualityWeightProvider {
    EqualityWeightProvider first_;
    EqualityWeightProvider second_;

public:
    CombinedEqualityWeightProvider(const std::vector<F>& first_point,
                                   const std::vector<F>& second_point,
                                   int bits, const F& first_scale,
                                   const F& second_scale)
        : first_(first_point, bits, first_scale),
          second_(second_point, bits, second_scale) {}

    F at(std::uint64_t coordinate) const {
        return first_.at(coordinate) + second_.at(coordinate);
    }
};

inline linear_poly fold_prefix_linear(const zkgpt::NativeVector<linear_poly>& values,
                                      std::uint64_t logical_extent,
                                      std::uint64_t pair_index,
                                      const F& previous_random) {
    if (logical_extent == 0 || pair_index >= logical_extent / 2 + logical_extent % 2) {
        throw std::out_of_range("native sumcheck prefix fold pair is outside the active extent");
    }
    const std::uint64_t even = pair_index * 2;
    const std::uint64_t odd = even + 1;
    if (even >= values.size()) throw std::out_of_range("native sumcheck prefix value is missing");
    const F zero_value = values[static_cast<std::size_t>(even)].eval(previous_random);
    const F one_value = odd < logical_extent
        ? values.at(static_cast<std::size_t>(odd)).eval(previous_random)
        : F_ZERO;
    return linear_poly(one_value - zero_value, zero_value);
}

inline quadratic_poly fold_prefix_product_round(
    const zkgpt::NativeVector<linear_poly>& mult,
    const zkgpt::NativeVector<linear_poly>& values,
    zkgpt::NativeVector<linear_poly>& folded_mult,
    zkgpt::NativeVector<linear_poly>& folded_values,
    std::uint64_t logical_extent,
    const F& previous_random) {
    if (logical_extent > mult.size() || logical_extent > values.size()) {
        throw std::out_of_range("native sumcheck active prefix exceeds its source vectors");
    }
    const std::uint64_t folded_extent = logical_extent / 2 + logical_extent % 2;
    const std::size_t storage_extent = checked_extent(std::max<std::uint64_t>(1, folded_extent), "native sumcheck fold");
    folded_mult.resize(storage_extent);
    folded_values.resize(storage_extent);
    quadratic_poly result;
    result.clear();
    for (std::uint64_t pair = 0; pair < folded_extent; ++pair) {
        folded_mult[static_cast<std::size_t>(pair)] = fold_prefix_linear(mult, logical_extent, pair, previous_random);
        folded_values[static_cast<std::size_t>(pair)] = fold_prefix_linear(values, logical_extent, pair, previous_random);
        result = result + folded_mult[static_cast<std::size_t>(pair)] * folded_values[static_cast<std::size_t>(pair)];
    }
    return result;
}

// GKR prefix values start as scalars and become affine polynomials after the
// first challenge. Keep that mode transition in one compact native arena:
// scalar entries occupy one field each, then each folded entry occupies its
// (a,b) coefficients in adjacent fields. Since the first fold writes exactly
// into the source pair and later folds compact toward lower indices, one
// ascending bounded chunk at a time is safe without a full-domain fold buffer.
class PackedAffinePrefixArena {
    enum class Mode { Empty, Scalar, Source, SourcePending, Affine };
    NativeVector<F> fields_;
    std::uint64_t logical_extent_ = 0;
    std::uint64_t source_extent_ = 0;
    std::uint64_t pending_extent_ = 0;
    Mode mode_ = Mode::Empty;
    NativeFieldView source_;

    static std::size_t required_capacity(std::uint64_t logical_extent) {
        if (logical_extent == std::numeric_limits<std::uint64_t>::max())
            throw std::length_error("packed GKR prefix extent overflows its paired capacity");
        const std::uint64_t paired = logical_extent + (logical_extent & 1U);
        const std::uint64_t required = std::max<std::uint64_t>(2U, paired);
        return checked_extent(required, "packed GKR prefix arena");
    }

public:
    void initialize_scalars(std::uint64_t logical_extent, const F& initial_value = F_ZERO, bool clear_storage = true) {
        if (!fields_.empty() || fields_.capacity() != 0U)
            throw std::logic_error("packed GKR prefix arena must be released before initialization");
        const std::size_t capacity = required_capacity(logical_extent);
        fields_.reserve(capacity);
        if (fields_.capacity() != capacity)
            throw std::logic_error("packed GKR prefix arena capacity differs from its source bound");
        fields_.resize(capacity);
        if (clear_storage) std::fill(fields_.begin(), fields_.end(), F_ZERO);
        logical_extent_ = logical_extent;
        source_extent_ = 0U;
        pending_extent_ = 0U;
        mode_ = Mode::Scalar;
        source_ = NativeFieldView();
        if (!initial_value.isZero())
            for (std::uint64_t index = 0U; index < logical_extent_; ++index)
                fields_[static_cast<std::size_t>(index)] = initial_value;
    }

    void initialize_source_view(std::uint64_t logical_extent, NativeFieldView source) {
        if (!fields_.empty() || fields_.capacity() != 0U || mode_ != Mode::Empty)
            throw std::logic_error("packed GKR source view requires an empty prefix arena");
        if (!source.valid() || source.size() != logical_extent)
            throw std::invalid_argument("packed GKR source view does not match its active prefix");
        logical_extent_ = logical_extent;
        source_extent_ = logical_extent;
        pending_extent_ = 0U;
        source_ = std::move(source);
        mode_ = Mode::Source;
    }

    std::uint64_t logical_extent() const { return logical_extent_; }
    bool affine_mode() const { return mode_ == Mode::Affine; }
    bool source_view_mode() const { return mode_ == Mode::Source || mode_ == Mode::SourcePending; }
    std::size_t capacity() const { return fields_.capacity(); }
    std::size_t capacity_bytes() const {
        if (fields_.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(F))
            throw std::overflow_error("packed GKR prefix byte count overflows");
        return fields_.capacity() * sizeof(F);
    }
    std::uint64_t physical_bytes() const { return native_storage::physical_bytes(fields_.data(), capacity_bytes()); }

    F first_slope() const {
        if (logical_extent_ == 0U || mode_ != Mode::Affine) return F_ZERO;
        return fields_[0U];
    }

    void set_scalar(std::uint64_t index, const F& value) {
        if (mode_ != Mode::Scalar || index >= logical_extent_)
            throw std::out_of_range("packed GKR scalar assignment is outside its active prefix");
        fields_[static_cast<std::size_t>(index)] = value;
    }

    void add_scalar(std::uint64_t index, const F& value) {
        if (mode_ != Mode::Scalar || index >= logical_extent_)
            throw std::out_of_range("packed GKR scalar accumulation is outside its active prefix");
        fields_[static_cast<std::size_t>(index)] += value;
    }

    F coordinate_value(std::uint64_t index, const F& point) const {
        if (index >= logical_extent_)
            throw std::out_of_range("packed GKR coordinate is outside its active prefix");
        if (mode_ == Mode::Source) return source_.at(checked_extent(index, "packed GKR source coordinate"));
        if (mode_ == Mode::SourcePending || mode_ == Mode::Empty)
            throw std::logic_error("packed GKR coordinate is unavailable before its challenge transition");
        if (mode_ == Mode::Scalar) return fields_[static_cast<std::size_t>(index)];
        const std::size_t offset = checked_extent(index * 2U, "packed GKR affine coordinate");
        const F& slope = fields_[offset];
        const F& intercept = fields_[offset + 1U];
        return slope.isZero() ? intercept : slope * point + intercept;
    }

    F first_value(const F& point) const {
        return logical_extent_ == 0U ? F_ZERO : coordinate_value(0U, point);
    }

    F first_constant() const {
        if (logical_extent_ == 0U) return F_ZERO;
        if (mode_ == Mode::Source) return source_.at(0U);
        if (mode_ == Mode::SourcePending || mode_ == Mode::Empty)
            throw std::logic_error("packed GKR terminal constant is unavailable before its challenge transition");
        return mode_ == Mode::Affine ? fields_[1U] : fields_[0U];
    }

    void evaluate_first_in_place(const F& point) {
        if (logical_extent_ == 0U) return;
        const F value = coordinate_value(0U, point);
        if (mode_ == Mode::Source) {
            initialize_scalars(logical_extent_);
            fields_[0U] = value;
        } else if (mode_ == Mode::Affine) {
            fields_[0U].clear();
            fields_[1U] = value;
        } else if (mode_ == Mode::Scalar) {
            fields_[0U] = value;
        } else throw std::logic_error("packed GKR first coordinate cannot be evaluated in its current mode");
    }

    void store_folded(std::uint64_t index, const linear_poly& value) {
        if (index >= (logical_extent_ / 2U + logical_extent_ % 2U))
            throw std::out_of_range("packed GKR folded output is outside its active prefix");
        if (mode_ == Mode::Source) return;
        if (mode_ != Mode::Scalar && mode_ != Mode::Affine)
            throw std::logic_error("packed GKR folded output is unavailable in its current mode");
        const std::size_t offset = checked_extent(index * 2U, "packed GKR folded output");
        if (offset + 1U >= fields_.size())
            throw std::out_of_range("packed GKR folded output exceeds its reserved arena");
        fields_[offset] = value.a;
        fields_[offset + 1U] = value.b;
    }

    void finish_fold(std::uint64_t folded_extent) {
        if (folded_extent != logical_extent_ / 2U + logical_extent_ % 2U)
            throw std::logic_error("packed GKR folded extent differs from the source prefix");
        if (mode_ == Mode::Source) {
            pending_extent_ = folded_extent;
            mode_ = Mode::SourcePending;
            return;
        }
        if (mode_ != Mode::Scalar && mode_ != Mode::Affine)
            throw std::logic_error("packed GKR fold cannot finish in its current mode");
        logical_extent_ = folded_extent;
        mode_ = Mode::Affine;
        const std::size_t affine_fields = checked_extent(folded_extent * 2U, "packed GKR affine live extent");
        if (affine_fields > std::numeric_limits<std::size_t>::max() / sizeof(F))
            throw std::overflow_error("packed GKR affine live byte count overflows");
        native_storage::record_live_bound(fields_.data(), affine_fields * sizeof(F), capacity_bytes());
    }

    void apply_challenge(const F& challenge) {
        if (logical_extent_ == 0U) return;
        if (mode_ == Mode::SourcePending) {
            const std::uint64_t source_extent = source_extent_;
            const std::uint64_t folded_extent = pending_extent_;
            const NativeFieldView source = source_;
            initialize_scalars(folded_extent, F_ZERO, false);
            for (std::uint64_t index = 0U; index < folded_extent; ++index) {
                const std::uint64_t even = index * 2U;
                const std::uint64_t odd = even + 1U;
                const F zero_value = source.at(checked_extent(even, "packed GKR source fold"));
                const F one_value = odd < source_extent
                    ? source.at(checked_extent(odd, "packed GKR source fold")) : F_ZERO;
                const F slope = one_value - zero_value;
                fields_[static_cast<std::size_t>(index)] = slope.isZero() ? zero_value : slope * challenge + zero_value;
            }
            source_extent_ = 0U;
            pending_extent_ = 0U;
            source_ = NativeFieldView();
            native_storage::reclaim_tail(fields_.data(), logical_extent_ * sizeof(F), capacity_bytes());
            return;
        }
        if (mode_ != Mode::Affine) throw std::logic_error("packed GKR challenge was applied more than once or before a fold");
        const std::uint64_t extent = logical_extent_;
        for (std::uint64_t index = 0U; index < extent; ++index) {
            const std::size_t offset = checked_extent(index * 2U, "packed GKR challenge compaction");
            const F& slope = fields_[offset];
            const F& intercept = fields_[offset + 1U];
            const F value = slope.isZero() ? intercept : slope * challenge + intercept;
            fields_[static_cast<std::size_t>(index)] = value;
        }
        mode_ = Mode::Scalar;
        native_storage::reclaim_tail(fields_.data(), checked_extent(extent, "packed GKR challenge live prefix") * sizeof(F), capacity_bytes());
    }

    void release() {
        NativeVector<F>().swap(fields_);
        logical_extent_ = 0U;
        source_extent_ = 0U;
        pending_extent_ = 0U;
        source_ = NativeFieldView();
        mode_ = Mode::Empty;
    }
};

class PrefixChunkBarrier {
    std::mutex mutex_;
    std::condition_variable condition_;
    const std::size_t participants_;
    std::size_t waiting_ = 0U;
    std::size_t generation_ = 0U;

public:
    explicit PrefixChunkBarrier(std::size_t participants) : participants_(participants) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const std::size_t generation = generation_;
        if (++waiting_ == participants_) {
            waiting_ = 0U;
            ++generation_;
            lock.unlock();
            condition_.notify_all();
            return;
        }
        condition_.wait(lock, [this, generation] { return generation_ != generation; });
    }
};

inline quadratic_poly fold_prefix_product_round_packed(
    PackedAffinePrefixArena& mult,
    PackedAffinePrefixArena& values,
    std::uint64_t logical_extent,
    const F& previous_random) {
    if (logical_extent != mult.logical_extent() || logical_extent != values.logical_extent())
        throw std::logic_error("packed GKR value and multiplier prefixes have different extents");
    const std::uint64_t folded_extent = logical_extent / 2U + logical_extent % 2U;
    quadratic_poly result;
    result.clear();
    if (folded_extent == 0U) return result;

    constexpr std::uint64_t kChunkPairs = 1U << 16U;
    constexpr std::size_t kMaxWorkers = 32U;
    const std::size_t scratch_size = checked_extent(std::min<std::uint64_t>(folded_extent, kChunkPairs),
                                                    "packed GKR bounded fold scratch");
    std::vector<linear_poly> folded_values(scratch_size);
    std::vector<linear_poly> folded_mult(scratch_size);
    const std::size_t workers = folded_extent < (1U << 15U)
        ? 1U
        : std::min<std::size_t>(kMaxWorkers, scratch_size);
    std::vector<quadratic_poly> partials(workers);
    auto barrier = std::make_unique<PrefixChunkBarrier>(workers);

    const auto worker = [&](std::size_t worker_id) {
        for (std::uint64_t chunk_start = 0U; chunk_start < folded_extent;) {
            const std::uint64_t chunk_count = std::min<std::uint64_t>(kChunkPairs, folded_extent - chunk_start);
            const std::uint64_t begin = chunk_start + (chunk_count * worker_id) / workers;
            const std::uint64_t end = chunk_start + (chunk_count * (worker_id + 1U)) / workers;
            quadratic_poly chunk_sum;
            chunk_sum.clear();
            for (std::uint64_t index = begin; index < end; ++index) {
                const std::uint64_t even = index * 2U;
                const std::uint64_t odd = even + 1U;
                const F value_zero = values.coordinate_value(even, previous_random);
                const F value_one = odd < logical_extent
                    ? values.coordinate_value(odd, previous_random)
                    : F_ZERO;
                const F mult_zero = mult.coordinate_value(even, previous_random);
                const F mult_one = odd < logical_extent
                    ? mult.coordinate_value(odd, previous_random)
                    : F_ZERO;
                const std::size_t local = checked_extent(index - chunk_start, "packed GKR scratch index");
                folded_values[local] = linear_poly(value_one - value_zero, value_zero);
                folded_mult[local] = linear_poly(mult_one - mult_zero, mult_zero);
                chunk_sum = chunk_sum + folded_mult[local] * folded_values[local];
            }
            partials[worker_id] = partials[worker_id] + chunk_sum;
            barrier->wait();
            for (std::uint64_t index = begin; index < end; ++index) {
                const std::size_t local = checked_extent(index - chunk_start, "packed GKR scratch copy index");
                values.store_folded(index, folded_values[local]);
                mult.store_folded(index, folded_mult[local]);
            }
            barrier->wait();
            chunk_start += chunk_count;
        }
    };

    if (workers == 1U) {
        worker(0U);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(workers - 1U);
        std::mutex start_mutex;
        std::condition_variable start_condition;
        bool start = false;
        bool cancel = false;
        try {
            for (std::size_t worker_id = 1U; worker_id < workers; ++worker_id) {
                threads.emplace_back([&, worker_id] {
                    {
                        std::unique_lock<std::mutex> lock(start_mutex);
                        start_condition.wait(lock, [&] { return start || cancel; });
                        if (cancel) return;
                    }
                    worker(worker_id);
                });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(start_mutex);
                cancel = true;
            }
            start_condition.notify_all();
            for (auto& thread : threads) if (thread.joinable()) thread.join();
            throw;
        }
        {
            std::lock_guard<std::mutex> lock(start_mutex);
            start = true;
        }
        start_condition.notify_all();
        worker(0U);
        for (auto& thread : threads) thread.join();
    }

    for (const auto& partial : partials) result = result + partial;
    values.finish_fold(folded_extent);
    mult.finish_fold(folded_extent);
    return result;
}

} // namespace zkgpt::native_gkr
