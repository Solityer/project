#pragma once

#include "circuit.h"

#include <algorithm>
#include <exception>
#include <thread>
#include <vector>

namespace zkgpt {
namespace verifier_detail {

inline F masked_eq_at(const std::vector<F> &point, u64 index) {
    if (point.size() >= 64 || (point.size() < 64 && (index >> point.size()) != 0))
        return F_ZERO;
    F value = F_ONE;
    for (std::size_t bit = 0; bit < point.size(); ++bit)
        value *= ((index >> bit) & 1) != 0 ? point[bit] : F_ONE - point[bit];
    return value;
}

class MaskedEqLookup {
public:
    explicit MaskedEqLookup(const std::vector<F> &point, bool build_table = true)
        : MaskedEqLookup(&point, build_table) {}

    explicit MaskedEqLookup(const std::vector<F> *point, bool build_table = true)
        : point_(point) {
        if (!build_table || point_ == nullptr || point_->size() < 2 ||
            point_->size() > kMaximumTableBits)
            return;
        low_bits_ = point_->size() / 2;
        const std::size_t high_bits = point_->size() - low_bits_;
        low_mask_ = (u64(1) << low_bits_) - 1;
        low_table_ = build_half(*point_, 0, low_bits_);
        high_table_ = build_half(*point_, low_bits_, high_bits);
    }

    bool uses_table() const noexcept { return !low_table_.empty(); }

    F at(u64 index) const {
        if (point_ == nullptr) return F_ZERO;
        if (!uses_table()) return masked_eq_at(*point_, index);
        if ((index >> point_->size()) != 0) return F_ZERO;
        return low_table_[static_cast<std::size_t>(index & low_mask_)] *
               high_table_[static_cast<std::size_t>(index >> low_bits_)];
    }

private:
    static constexpr std::size_t kMaximumTableBits = 36;

    static std::vector<F> build_half(const std::vector<F> &point,
                                     std::size_t offset,
                                     std::size_t bit_count) {
        std::vector<F> table(std::size_t(1) << bit_count, F_ZERO);
        table[0] = F_ONE;
        std::size_t active = 1;
        for (std::size_t bit = 0; bit < bit_count; ++bit) {
            const F one = point[offset + bit];
            const F zero = F_ONE - one;
            for (std::size_t index = 0; index < active; ++index) {
                const F prefix = table[index];
                table[index] = prefix * zero;
                table[index + active] = prefix * one;
            }
            active <<= 1;
        }
        return table;
    }

    const std::vector<F> *point_ = nullptr;
    std::size_t low_bits_ = 0;
    u64 low_mask_ = 0;
    std::vector<F> low_table_;
    std::vector<F> high_table_;
};

inline std::size_t masked_eq_table_build_cost(std::size_t point_bits) {
    if (point_bits < 2 || point_bits > 36) return 0;
    const std::size_t low_bits = point_bits / 2;
    const std::size_t high_bits = point_bits - low_bits;
    const std::size_t low_size = std::size_t(1) << low_bits;
    const std::size_t high_size = std::size_t(1) << high_bits;
    return 2 * (low_size - 1) + 2 * (high_size - 1);
}

inline std::vector<F> masked_gkr_coefficients(const layer &current,
                                              const std::vector<F> &r0,
                                              const std::vector<F> &r1,
                                              const std::vector<F> &ru,
                                              const F &alpha, const F &beta,
                                              const F &relu_rou,
                                              const std::vector<F> *rv = nullptr) {
    constexpr std::size_t kMinimumLookupGates = 4096;
    constexpr std::size_t kMinimumParallelGates = 32768;
    constexpr unsigned int kMaximumWorkers = 20;
    const std::size_t unary_count = current.uni_gates.size();
    const std::size_t binary_count = current.bin_gates.size();
    const std::size_t gate_count = unary_count + binary_count;
    if (gate_count == 0) return {F_ZERO, F_ZERO, F_ZERO, F_ZERO, F_ZERO};

    std::size_t table_build_cost = 0;
    std::size_t saved_multiplies_per_gate = 0;
    const auto consider_point = [&](const std::vector<F> *point) {
        if (point == nullptr || point->size() < 2 || point->size() > 36) return;
        table_build_cost += masked_eq_table_build_cost(point->size());
        saved_multiplies_per_gate += point->size() - 1;
    };
    consider_point(&r0);
    consider_point(&r1);
    consider_point(&ru);
    consider_point(rv);
    const bool use_tables = gate_count >= kMinimumLookupGates &&
        saved_multiplies_per_gate != 0 &&
        gate_count > table_build_cost / saved_multiplies_per_gate;

    const MaskedEqLookup r0_lookup(r0, use_tables);
    const MaskedEqLookup r1_lookup(r1, use_tables);
    const MaskedEqLookup ru_lookup(ru, use_tables);
    const MaskedEqLookup rv_lookup(rv, use_tables);
    const auto gate_weight = [&](u64 gate, u64 input, u64 right) {
        F weight = (alpha * r0_lookup.at(gate) + beta * r1_lookup.at(gate)) *
                   ru_lookup.at(input);
        if (current.zero_start_id < current.size && gate >= current.zero_start_id)
            weight *= relu_rou;
        if (rv != nullptr) weight *= rv_lookup.at(right);
        return weight;
    };
    struct Partial {
        F uni[2] = {F_ZERO, F_ZERO};
        F bin[3] = {F_ZERO, F_ZERO, F_ZERO};
    };
    const auto accumulate_range = [&](std::size_t begin, std::size_t end,
                                      Partial &partial) {
        for (std::size_t index = begin; index < end; ++index) {
            if (index < unary_count) {
                const uniGate &gate = current.uni_gates[index];
                partial.uni[gate.lu] +=
                    gate_weight(gate.g, gate.u, 0) * current.coefficient(gate);
            } else {
                const binGate &gate = current.bin_gates[index - unary_count];
                partial.bin[gate.l] +=
                    gate_weight(gate.g, gate.u, gate.v) * current.coefficient(gate);
            }
        }
    };

    const unsigned int available_workers = std::thread::hardware_concurrency();
    const unsigned int worker_count = std::min(available_workers, kMaximumWorkers);
    if (gate_count < kMinimumParallelGates || worker_count < 2) {
        Partial partial;
        accumulate_range(0, gate_count, partial);
        return {partial.bin[0], partial.bin[1], partial.bin[2],
                partial.uni[0], partial.uni[1]};
    }

    std::vector<Partial> partials(worker_count);
    std::vector<std::exception_ptr> errors(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const std::size_t base_count = gate_count / worker_count;
    const std::size_t remainder = gate_count % worker_count;
    std::size_t begin = 0;
    try {
        for (unsigned int worker = 0; worker < worker_count; ++worker) {
            const std::size_t count = base_count + (worker < remainder ? 1 : 0);
            const std::size_t end = begin + count;
            workers.emplace_back([&, worker, begin, end] {
                try {
                    accumulate_range(begin, end, partials[worker]);
                } catch (...) {
                    errors[worker] = std::current_exception();
                }
            });
            begin = end;
        }
    } catch (...) {
        for (std::thread &worker : workers)
            if (worker.joinable()) worker.join();
        throw;
    }
    for (std::thread &worker : workers) worker.join();
    for (const std::exception_ptr &error : errors)
        if (error != nullptr) std::rethrow_exception(error);

    Partial total;
    for (unsigned int worker = 0; worker < worker_count; ++worker) {
        for (std::size_t index = 0; index < 2; ++index)
            total.uni[index] += partials[worker].uni[index];
        for (std::size_t index = 0; index < 3; ++index)
            total.bin[index] += partials[worker].bin[index];
    }
    return {total.bin[0], total.bin[1], total.bin[2],
            total.uni[0], total.uni[1]};
}

} // namespace verifier_detail
} // namespace zkgpt
