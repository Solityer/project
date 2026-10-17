#include "circuit.h"
#include "utils.hpp"
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace {

template <class Gate>
std::size_t checked_gate_offset(const Gate& gate, const zkgpt::NativeVector<Gate>& gates) {
    if (gates.empty()) throw std::logic_error("native coefficient lookup has no matching gate storage");
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(gates.data());
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(&gate);
    const std::size_t bytes = gates.size() * sizeof(Gate);
    if (address < base || address - base >= bytes || (address - base) % sizeof(Gate) != 0U)
        throw std::logic_error("native coefficient lookup gate is not owned by its layer");
    return static_cast<std::size_t>((address - base) / sizeof(Gate));
}

void reserve_exact(zkgpt::NativeVector<u32>& values, std::size_t count, const char* name) {
    values.reserve(count);
    if (count != 0U && values.capacity() != count)
        throw std::length_error(std::string(name) + " capacity differs from its exact gate count");
}

}  // namespace

void layer::reserve_gate_arrays(std::size_t unary_count, std::size_t binary_count) {
    uni_gates.reserve(unary_count);
    bin_gates.reserve(binary_count);
    if ((unary_count != 0 && uni_gates.capacity() != unary_count) ||
        (binary_count != 0 && bin_gates.capacity() != binary_count))
        throw std::length_error("native layer gate reservation differs from its exact source bound");
    if (wide_coefficient_format) {
        reserve_exact(wide_uni_coefficient_ids, unary_count, "wide unary coefficient IDs");
        reserve_exact(wide_bin_coefficient_ids, binary_count, "wide binary coefficient IDs");
    }
}

void layer::reserve_coefficients(std::size_t distinct_count) {
    if (static_cast<unsigned __int128>(distinct_count) > static_cast<unsigned __int128>(std::numeric_limits<u32>::max()) + 1)
        throw std::length_error("native layer coefficient pool exceeds u32 IDs");
    const bool needs_wide_ids = distinct_count > 256U;
    if (needs_wide_ids != wide_coefficient_format) {
        if (!uni_gates.empty() || !bin_gates.empty() || !wide_uni_coefficient_ids.empty() ||
            !wide_bin_coefficient_ids.empty())
            throw std::logic_error("native coefficient ID format must be selected before gate emission");
        wide_coefficient_format = needs_wide_ids;
    }
    coefficient_pool.reserve(distinct_count);
    coefficient_lookup_.reserve(distinct_count);
    if (wide_coefficient_format) {
        reserve_exact(wide_uni_coefficient_ids, uni_gates.capacity(), "wide unary coefficient IDs");
        reserve_exact(wide_bin_coefficient_ids, bin_gates.capacity(), "wide binary coefficient IDs");
    }
}

void layer::release_native_proof_backing() noexcept {
    zkgpt::NativeVector<uniGate>().swap(uni_gates);
    zkgpt::NativeVector<binGate>().swap(bin_gates);
    zkgpt::NativeVector<ll>().swap(coefficient_pool);
    zkgpt::NativeVector<u32>().swap(wide_uni_coefficient_ids);
    zkgpt::NativeVector<u32>().swap(wide_bin_coefficient_ids);
    zkgpt::NativeVector<PackedU40>().swap(ori_id_u);
    zkgpt::NativeVector<PackedU40>().swap(ori_id_v);
    std::vector<pair<u64, u64>>().swap(uni_interval);
    std::vector<pair<u64, u64>>().swap(bin_interval);
    CoefficientLookup().swap(coefficient_lookup_);
    wide_coefficient_format = false;
}

bool layer::native_metadata_matches(const layer& other) const noexcept {
    return ty == other.ty && size == other.size && size_u[0] == other.size_u[0] && size_u[1] == other.size_u[1] &&
           size_v[0] == other.size_v[0] && size_v[1] == other.size_v[1] &&
           bit_length_u[0] == other.bit_length_u[0] && bit_length_u[1] == other.bit_length_u[1] &&
           bit_length_v[0] == other.bit_length_v[0] && bit_length_v[1] == other.bit_length_v[1] &&
           bit_length == other.bit_length && max_bl_u == other.max_bl_u && max_bl_v == other.max_bl_v &&
           need_phase2 == other.need_phase2 && zero_start_id == other.zero_start_id && fft_bit_length == other.fft_bit_length &&
           native_relation_id == other.native_relation_id && native_family_id == other.native_family_id &&
           native_rows == other.native_rows && native_inner == other.native_inner && native_cols == other.native_cols &&
           native_lhs_offset == other.native_lhs_offset && native_rhs_offset == other.native_rhs_offset &&
           native_output_offset == other.native_output_offset &&
           wide_coefficient_format == other.wide_coefficient_format;
}

void layer::restore_native_proof_backing(layer&& canonical_backing) {
    if (!native_metadata_matches(canonical_backing))
        throw std::invalid_argument("rebuilt native layer backing does not match retained logical metadata");
    uni_gates = std::move(canonical_backing.uni_gates);
    bin_gates = std::move(canonical_backing.bin_gates);
    coefficient_pool = std::move(canonical_backing.coefficient_pool);
    wide_uni_coefficient_ids = std::move(canonical_backing.wide_uni_coefficient_ids);
    wide_bin_coefficient_ids = std::move(canonical_backing.wide_bin_coefficient_ids);
    wide_coefficient_format = canonical_backing.wide_coefficient_format;
    ori_id_u = std::move(canonical_backing.ori_id_u);
    ori_id_v = std::move(canonical_backing.ori_id_v);
    uni_interval = std::move(canonical_backing.uni_interval);
    bin_interval = std::move(canonical_backing.bin_interval);
    coefficient_lookup_ = std::move(canonical_backing.coefficient_lookup_);
}

u32 layer::intern_coefficient(ll value) {
    const auto found = coefficient_lookup_.find(value);
    if (found != coefficient_lookup_.end()) {
        if (found->second >= coefficient_pool.size() || coefficient_pool[found->second] != value)
            throw std::logic_error("native layer coefficient lookup is inconsistent");
        return found->second;
    }
    if (coefficient_pool.size() > std::numeric_limits<u32>::max()) throw std::length_error("native layer coefficient pool exceeds u32 IDs");
    const u32 logical_id = static_cast<u32>(coefficient_pool.size());
    if (!wide_coefficient_format && logical_id > std::numeric_limits<u8>::max())
        throw std::length_error("native layer needs the explicitly selected wide coefficient ID format");
    const auto inserted = coefficient_lookup_.emplace(value, logical_id);
    if (!inserted.second) throw std::logic_error("native layer coefficient lookup insertion raced");
    try { coefficient_pool.push_back(value); }
    catch (...) { coefficient_lookup_.erase(inserted.first); throw; }
    return logical_id;
}

ll layer::coefficient(NativeCoefficientId id) const {
    if (id.value >= coefficient_pool.size()) throw std::out_of_range("native gate coefficient ID exceeds layer pool");
    return coefficient_pool[id.value];
}

ll layer::coefficient(const uniGate& gate) const {
    if (!wide_coefficient_format) return coefficient(gate.coefficient_id);
    const std::size_t index = checked_gate_offset(gate, uni_gates);
    if (index >= wide_uni_coefficient_ids.size()) throw std::out_of_range("wide unary coefficient ID is missing");
    if (wide_uni_coefficient_ids[index] >= coefficient_pool.size())
        throw std::out_of_range("wide unary coefficient ID exceeds layer pool");
    return coefficient_pool[wide_uni_coefficient_ids[index]];
}

ll layer::coefficient(const binGate& gate) const {
    if (!wide_coefficient_format) return coefficient(gate.coefficient_id);
    const std::size_t index = checked_gate_offset(gate, bin_gates);
    if (index >= wide_bin_coefficient_ids.size()) throw std::out_of_range("wide binary coefficient ID is missing");
    if (wide_bin_coefficient_ids[index] >= coefficient_pool.size())
        throw std::out_of_range("wide binary coefficient ID exceeds layer pool");
    return coefficient_pool[wide_bin_coefficient_ids[index]];
}

void layer::add_unary(u64 g, u64 u, u8 lu, ll value) {
    const u32 logical_id = intern_coefficient(value);
    if (!wide_coefficient_format && logical_id > std::numeric_limits<u8>::max())
        throw std::length_error("native layer needs the explicitly selected wide coefficient ID format");
    uni_gates.emplace_back(g, u, lu, NativeCoefficientId(static_cast<u8>(wide_coefficient_format ? 0U : logical_id)));
    if (wide_coefficient_format) {
        if (wide_uni_coefficient_ids.size() >= wide_uni_coefficient_ids.capacity())
            throw std::length_error("wide unary coefficient IDs exceeded the reserved gate count");
        wide_uni_coefficient_ids.push_back(logical_id);
    }
}

void layer::add_binary(u64 g, u64 u, u64 v, ll value, u8 l) {
    const u32 logical_id = intern_coefficient(value);
    if (!wide_coefficient_format && logical_id > std::numeric_limits<u8>::max())
        throw std::length_error("native layer needs the explicitly selected wide coefficient ID format");
    bin_gates.emplace_back(g, u, v, NativeCoefficientId(static_cast<u8>(wide_coefficient_format ? 0U : logical_id)), l);
    if (wide_coefficient_format) {
        if (wide_bin_coefficient_ids.size() >= wide_bin_coefficient_ids.capacity())
            throw std::length_error("wide binary coefficient IDs exceeded the reserved gate count");
        wide_bin_coefficient_ids.push_back(logical_id);
    }
}

std::uint64_t layer::coefficient_storage_bytes() const {
    const unsigned __int128 buckets = static_cast<unsigned __int128>(coefficient_lookup_.bucket_count()) * sizeof(void*);
    const unsigned __int128 nodes = static_cast<unsigned __int128>(coefficient_lookup_.size()) * (sizeof(CoefficientLookup::value_type) + 2 * sizeof(void*) + sizeof(std::size_t));
    const unsigned __int128 pool = static_cast<unsigned __int128>(coefficient_pool.capacity()) * sizeof(ll);
    const unsigned __int128 wide_ids = static_cast<unsigned __int128>(wide_uni_coefficient_ids.capacity() + wide_bin_coefficient_ids.capacity()) * sizeof(u32);
    const unsigned __int128 total = pool + buckets + nodes + wide_ids;
    if (total > std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("native layer coefficient storage estimate exceeds u64");
    return static_cast<std::uint64_t>(total);
}

std::uint64_t layer::coefficient_storage_bytes_for(std::size_t distinct_count) {
    if (static_cast<unsigned __int128>(distinct_count) > static_cast<unsigned __int128>(std::numeric_limits<u32>::max()) + 1)
        throw std::length_error("native layer coefficient pool exceeds u32 IDs");
    const unsigned __int128 count = distinct_count;
    const unsigned __int128 pool = count * sizeof(ll);
    const unsigned __int128 buckets = count * 2 * sizeof(void*);
    const unsigned __int128 nodes = count * (sizeof(CoefficientLookup::value_type) + 2 * sizeof(void*) + sizeof(std::size_t));
    const unsigned __int128 total = pool + buckets + nodes;
    if (total > std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("native layer coefficient storage estimate exceeds u64");
    return static_cast<std::uint64_t>(total);
}

void layeredCircuit::initSubset() 
{
    const std::size_t layer0_domain = circuit.empty() ? 0 : circuit.front().size;
    std::vector<std::pair<u64, u64>> support_ranges = source_support_ranges;
    if (support_ranges.empty() && layer0_domain != 0) support_ranges.emplace_back(0, static_cast<u64>(layer0_domain));
    std::vector<u64> support_prefix;
    support_prefix.reserve(support_ranges.size() + 1);
    support_prefix.push_back(0);
    u64 support_entries = 0;
    u64 previous_end = 0;
    for (const auto& range : support_ranges) {
        if (range.second == 0 || range.first < previous_end || range.first > layer0_domain ||
            range.second > static_cast<u64>(layer0_domain) - range.first ||
            range.second > std::numeric_limits<u64>::max() - support_entries)
            throw std::invalid_argument("native GKR source support range is invalid");
        previous_end = range.first + range.second;
        support_entries += range.second;
        support_prefix.push_back(support_entries);
    }
    if (support_entries > std::numeric_limits<std::size_t>::max())
        throw std::length_error("native GKR source support exceeds native size_t");
    const std::size_t bitmap_word_count = static_cast<std::size_t>(support_entries / 64 + (support_entries % 64 != 0));
    // Reuse one zeroed bitmap for all lanes. Each enumeration clears its words
    // after its gates are remapped, so only one bitmap/rank pair is live.
    zkgpt::NativeVector<std::uint64_t> source_seen(bitmap_word_count, 0);
    if (source_seen.capacity() != bitmap_word_count)
        throw std::length_error("native GKR source bitmap capacity differs from its exact support bound");
    zkgpt::NativeVector<std::uint64_t> source_rank(bitmap_word_count, 0);
    if (source_rank.capacity() != bitmap_word_count)
        throw std::length_error("native GKR source rank capacity differs from its exact support bound");
    const auto support_index_for_coordinate = [&support_ranges, &support_prefix](u64 coordinate) -> std::size_t {
        const auto after = std::upper_bound(support_ranges.begin(), support_ranges.end(), coordinate,
            [](u64 value, const auto& range) { return value < range.first; });
        if (after == support_ranges.begin()) throw std::out_of_range("native GKR input index is outside logical source support");
        const std::size_t index = static_cast<std::size_t>((after - support_ranges.begin()) - 1);
        const auto& range = support_ranges[index];
        if (coordinate < range.first || coordinate - range.first >= range.second)
            throw std::out_of_range("native GKR input index is outside logical source support");
        return static_cast<std::size_t>(support_prefix[index] + coordinate - range.first);
    };
    const auto coordinate_for_support_index = [&support_ranges, &support_prefix](u64 index) -> u64 {
        const auto after = std::upper_bound(support_prefix.begin(), support_prefix.end(), index);
        if (after == support_prefix.begin()) throw std::logic_error("native GKR source bitmap rank is invalid");
        const std::size_t range_index = static_cast<std::size_t>((after - support_prefix.begin()) - 1);
        if (range_index >= support_ranges.size() || index >= support_prefix[range_index + 1])
            throw std::logic_error("native GKR source bitmap rank is outside support ranges");
        return support_ranges[range_index].first + index - support_prefix[range_index];
    };
    for (int i = 1; i < size; ++i) 
    {
        auto &cur = circuit[i], &lst = circuit[i - 1];
        bool has_pre_layer_u = cur.ty == layerType::FFT || cur.ty == layerType::IFFT;
        bool has_pre_layer_v = false, has_v_operand = false;
        const auto checked_source_domain = [&](std::size_t source_layer, u64 coordinate) -> std::size_t {
            if (source_layer >= circuit.size()) throw std::out_of_range("native GKR source layer exceeds circuit");
            const std::size_t domain = circuit[source_layer].size;
            if (coordinate >= domain) throw std::out_of_range("native GKR input index exceeds its source-layer domain");
            return domain;
        };
        const auto make_subset_map = [&](std::size_t source_layer, const auto& visit_coordinates, auto& coordinates) {
            const std::size_t domain = circuit.at(source_layer).size;
            if (source_layer != 0 || source_seen.size() != bitmap_word_count || source_rank.size() != bitmap_word_count)
                throw std::logic_error("native GKR source-map bitmap does not match its logical source support");
            std::size_t unique_count = 0;
            visit_coordinates([&](std::size_t actual_source_layer, u64 coordinate) {
                const std::size_t actual_domain = checked_source_domain(actual_source_layer, coordinate);
                if (actual_source_layer != source_layer || actual_domain != domain)
                    throw std::logic_error("native GKR source map mixes source-layer domains");
                const std::size_t index = support_index_for_coordinate(coordinate);
                const std::uint64_t mask = std::uint64_t(1) << (index & 63);
                std::uint64_t& word = source_seen[index >> 6];
                if ((word & mask) == 0) { word |= mask; ++unique_count; }
            });
            if (unique_count == 0) return;
            std::size_t rank = 0;
            for (std::size_t word_index = 0; word_index < source_seen.size(); ++word_index) {
                source_rank[word_index] = rank;
                rank += static_cast<std::size_t>(__builtin_popcountll(source_seen[word_index]));
            }
            if (rank != unique_count) throw std::logic_error("native GKR source bitmap rank differs from its unique coordinate count");
            coordinates.reserve(unique_count);
            if (coordinates.capacity() != unique_count)
                throw std::length_error("native GKR source map capacity differs from its exact unique count");
            for (std::size_t word_index = 0; word_index < source_seen.size(); ++word_index) {
                std::uint64_t word = source_seen[word_index];
                while (word != 0) {
                    const unsigned bit = static_cast<unsigned>(__builtin_ctzll(word));
                    const u64 support_index = static_cast<u64>(word_index) * 64 + bit;
                    if (support_index >= support_entries) throw std::logic_error("native GKR source bitmap emitted an out-of-support index");
                    coordinates.push_back(coordinate_for_support_index(support_index));
                    word &= word - 1;
                }
            }
            if (coordinates.size() != unique_count)
                throw std::logic_error("native GKR source bitmap unique count changed during enumeration");
        };
        auto finish_map = [](auto& map, u64& count) {
            count = map.size();
        };
        const auto subset_index = [&](u64 coordinate) {
            const std::size_t support_index = support_index_for_coordinate(coordinate);
            const std::size_t word_index = support_index >> 6U;
            const unsigned bit_index = static_cast<unsigned>(support_index & 63U);
            if (word_index >= source_seen.size()) throw std::logic_error("native GKR input subset rank word is outside its bitmap");
            const std::uint64_t word = source_seen[word_index];
            const std::uint64_t bit = std::uint64_t(1) << bit_index;
            if ((word & bit) == 0U) throw std::logic_error("native GKR input subset map is incomplete");
            const std::uint64_t lower_mask = bit_index == 0U ? 0U : bit - 1U;
            return source_rank[word_index] + static_cast<std::uint64_t>(__builtin_popcountll(word & lower_mask));
        };
        {
            const auto source_layer_for_unary = [i](const uniGate& gate) -> std::size_t {
                return gate.lu == 0 ? 0 : static_cast<std::size_t>(i - 1);
            };
            make_subset_map(0, [&](const auto& add) {
                for (const auto& gate : cur.uni_gates) {
                    if (source_layer_for_unary(gate) == 0) add(source_layer_for_unary(gate), gate.u);
                    has_pre_layer_u |= gate.lu != 0;
                }
                for (const auto& gate : cur.bin_gates) {
                    const std::size_t source_layer = gate.getLayerIdU(i);
                    if (source_layer == 0) add(source_layer, gate.u);
                    has_v_operand = true;
                    has_pre_layer_u |= source_layer != 0;
                    has_pre_layer_v |= gate.getLayerIdV(i) != 0;
                }
            }, cur.ori_id_u);
            finish_map(cur.ori_id_u, cur.size_u[0]);
            for (auto& gate : cur.uni_gates) {
                if (!gate.lu) gate.u = subset_index(gate.u);
            }
            for (auto& gate : cur.bin_gates) {
                if (!gate.getLayerIdU(i)) gate.u = subset_index(gate.u);
            }
            std::fill(source_seen.begin(), source_seen.end(), 0U);
        }
        {
            make_subset_map(0, [&](const auto& add) {
                for (const auto& gate : cur.bin_gates) {
                    const std::size_t source_layer = gate.getLayerIdV(i);
                    if (source_layer == 0) add(source_layer, gate.v);
                }
            }, cur.ori_id_v);
            finish_map(cur.ori_id_v, cur.size_v[0]);
            for (auto& gate : cur.bin_gates) if (!gate.getLayerIdV(i)) gate.v = subset_index(gate.v);
            std::fill(source_seen.begin(), source_seen.end(), 0U);
        }
        cur.bit_length_u[0] = ceilPow2BitLength(cur.size_u[0]);
        cur.bit_length_v[0] = ceilPow2BitLength(cur.size_v[0]);

        if (has_pre_layer_u) 
            switch (cur.ty) {
                case layerType::FFT:
                    cur.size_u[1] = 1ULL << cur.fft_bit_length - 1;
                    cur.bit_length_u[1] = cur.fft_bit_length - 1;
                    break;
                case layerType::IFFT:
                    cur.size_u[1] = 1ULL << cur.fft_bit_length;
                    cur.bit_length_u[1] = cur.fft_bit_length;
                    break;
                default:
                    cur.size_u[1] = lst.size ;
                    cur.bit_length_u[1] = lst.bit_length;
                    break;
            } else {
            cur.size_u[1] = 0;
            cur.bit_length_u[1] = -1;
        }
        if (has_pre_layer_v) 
        {
            if (cur.ty == layerType::DOT_PROD) {
                cur.size_v[1] = lst.size >> cur.fft_bit_length;
                cur.bit_length_v[1] = lst.bit_length - cur.fft_bit_length;
            } else {
                cur.size_v[1] = lst.size;
                cur.bit_length_v[1] = lst.bit_length;
            }
        } 
        else 
        {
            cur.size_v[1] = 0;
            cur.bit_length_v[1] = -1;
        }
        if (cur.ty != layerType::FCONN)
            cur.need_phase2 = has_v_operand;
        cur.updateSize();
    }
}

void layeredCircuit::init(u8 q_bit_size, u8 _layer_sz) 
{
    size = _layer_sz;
    circuit.resize(size);
    source_support_ranges.clear();
}
