#include "circuit.h"
#include "utils.hpp"
#include <stdexcept>
#include <algorithm>
#include <limits>

void layer::reserve_gate_arrays(std::size_t unary_count, std::size_t binary_count) {
    uni_gates.reserve(unary_count);
    bin_gates.reserve(binary_count);
}

void layer::reserve_coefficients(std::size_t distinct_count) {
    if (static_cast<unsigned __int128>(distinct_count) > static_cast<unsigned __int128>(std::numeric_limits<u32>::max()) + 1)
        throw std::length_error("native layer coefficient pool exceeds u32 IDs");
    coefficient_pool.reserve(distinct_count);
    coefficient_lookup_.reserve(distinct_count);
}

void layer::release_native_proof_backing() noexcept {
    zkgpt::NativeVector<uniGate>().swap(uni_gates);
    zkgpt::NativeVector<binGate>().swap(bin_gates);
    zkgpt::NativeVector<ll>().swap(coefficient_pool);
    zkgpt::NativeVector<u64>().swap(ori_id_u);
    zkgpt::NativeVector<u64>().swap(ori_id_v);
    std::vector<pair<u64, u64>>().swap(uni_interval);
    std::vector<pair<u64, u64>>().swap(bin_interval);
    CoefficientLookup().swap(coefficient_lookup_);
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
           native_output_offset == other.native_output_offset;
}

void layer::restore_native_proof_backing(layer&& canonical_backing) {
    if (!native_metadata_matches(canonical_backing))
        throw std::invalid_argument("rebuilt native layer backing does not match retained logical metadata");
    uni_gates = std::move(canonical_backing.uni_gates);
    bin_gates = std::move(canonical_backing.bin_gates);
    coefficient_pool = std::move(canonical_backing.coefficient_pool);
    ori_id_u = std::move(canonical_backing.ori_id_u);
    ori_id_v = std::move(canonical_backing.ori_id_v);
    uni_interval = std::move(canonical_backing.uni_interval);
    bin_interval = std::move(canonical_backing.bin_interval);
    coefficient_lookup_ = std::move(canonical_backing.coefficient_lookup_);
}

NativeCoefficientId layer::intern_coefficient(ll value) {
    const auto found = coefficient_lookup_.find(value);
    if (found != coefficient_lookup_.end()) {
        if (found->second.value >= coefficient_pool.size() || coefficient_pool[found->second.value] != value)
            throw std::logic_error("native layer coefficient lookup is inconsistent");
        return found->second;
    }
    if (coefficient_pool.size() > std::numeric_limits<u32>::max()) throw std::length_error("native layer coefficient pool exceeds u32 IDs");
    const NativeCoefficientId id(static_cast<u32>(coefficient_pool.size()));
    const auto inserted = coefficient_lookup_.emplace(value, id);
    if (!inserted.second) throw std::logic_error("native layer coefficient lookup insertion raced");
    try { coefficient_pool.push_back(value); }
    catch (...) { coefficient_lookup_.erase(inserted.first); throw; }
    return id;
}

ll layer::coefficient(NativeCoefficientId id) const {
    if (id.value >= coefficient_pool.size()) throw std::out_of_range("native gate coefficient ID exceeds layer pool");
    return coefficient_pool[id.value];
}

ll layer::coefficient(const uniGate& gate) const { return coefficient(gate.coefficient_id); }
ll layer::coefficient(const binGate& gate) const { return coefficient(gate.coefficient_id); }

void layer::add_unary(u64 g, u64 u, u8 lu, ll value) { uni_gates.emplace_back(g, u, lu, intern_coefficient(value)); }
void layer::add_binary(u64 g, u64 u, u64 v, ll value, u8 l) { bin_gates.emplace_back(g, u, v, intern_coefficient(value), l); }

std::uint64_t layer::coefficient_storage_bytes() const {
    const unsigned __int128 buckets = static_cast<unsigned __int128>(coefficient_lookup_.bucket_count()) * sizeof(void*);
    const unsigned __int128 nodes = static_cast<unsigned __int128>(coefficient_lookup_.size()) * (sizeof(CoefficientLookup::value_type) + 2 * sizeof(void*) + sizeof(std::size_t));
    const unsigned __int128 pool = static_cast<unsigned __int128>(coefficient_pool.capacity()) * sizeof(ll);
    const unsigned __int128 total = pool + buckets + nodes;
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
    for (int i = 1; i < size; ++i) 
    {
        auto &cur = circuit[i], &lst = circuit[i - 1];
        bool has_pre_layer_u = cur.ty == layerType::FFT || cur.ty == layerType::IFFT;
        bool has_pre_layer_v = false, has_v_operand = false;
        auto validate_coordinate = [&](u64 coordinate) {
            if (coordinate >= circuit[0].size) throw std::out_of_range("native GKR input index exceeds domain");
        };
        std::size_t subset_u_reference_count = 0, subset_v_reference_count = 0;
        for (const auto& gate : cur.uni_gates) if (!gate.lu) { validate_coordinate(gate.u); ++subset_u_reference_count; }
        for (const auto& gate : cur.bin_gates) {
            if (!gate.getLayerIdU(i)) { validate_coordinate(gate.u); ++subset_u_reference_count; }
            if (!gate.getLayerIdV(i)) { validate_coordinate(gate.v); ++subset_v_reference_count; }
        }
        auto finish_map = [](auto& map, auto& original, u64& count) {
            std::sort(map.begin(), map.end());
            map.erase(std::unique(map.begin(), map.end()), map.end());
            count = map.size();
            // The sorted unique coordinate vector is itself the canonical
            // source map. Moving it avoids a second full-size map allocation
            // while preserving the source-coordinate meaning of every entry.
            original = std::move(map);
        };
        auto subset_index = [](const auto& sorted_coordinates, u64 coordinate) {
            const auto found = std::lower_bound(sorted_coordinates.begin(), sorted_coordinates.end(), coordinate);
            if (found == sorted_coordinates.end() || *found != coordinate) throw std::logic_error("native GKR input subset map is incomplete");
            return static_cast<u64>(found - sorted_coordinates.begin());
        };
        {
            zkgpt::NativeVector<u64> subset_uidx;
            subset_uidx.reserve(subset_u_reference_count);
            for (const auto& gate : cur.uni_gates) {
                if (!gate.lu) subset_uidx.push_back(gate.u);
                has_pre_layer_u |= gate.lu != 0;
            }
            for (const auto& gate : cur.bin_gates) {
                has_v_operand = true;
                if (!gate.getLayerIdU(i)) subset_uidx.push_back(gate.u);
                has_pre_layer_u |= gate.getLayerIdU(i) != 0;
                has_pre_layer_v |= gate.getLayerIdV(i) != 0;
            }
            if (subset_uidx.size() != subset_u_reference_count)
                throw std::logic_error("native GKR U subset reference count changed during construction");
            finish_map(subset_uidx, cur.ori_id_u, cur.size_u[0]);
            for (auto& gate : cur.uni_gates) if (!gate.lu) gate.u = subset_index(cur.ori_id_u, gate.u);
            for (auto& gate : cur.bin_gates) if (!gate.getLayerIdU(i)) gate.u = subset_index(cur.ori_id_u, gate.u);
        }
        {
            zkgpt::NativeVector<u64> subset_vidx;
            subset_vidx.reserve(subset_v_reference_count);
            for (const auto& gate : cur.bin_gates) if (!gate.getLayerIdV(i)) subset_vidx.push_back(gate.v);
            if (subset_vidx.size() != subset_v_reference_count)
                throw std::logic_error("native GKR V subset reference count changed during construction");
            finish_map(subset_vidx, cur.ori_id_v, cur.size_v[0]);
            for (auto& gate : cur.bin_gates) if (!gate.getLayerIdV(i)) gate.v = subset_index(cur.ori_id_v, gate.v);
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
}
