#pragma once

#include <vector>
#include <unordered_map>
#include <utility>

#include <unordered_set>
#include <iostream>
#include <string>
#include <cstdint>
#include "global_var.hpp"
#include "hyrax.hpp"
#include "../gat_adapter/native_storage.hpp"
using std::cerr;
using std::endl;
using std::vector;

struct NativeCoefficientId {
    u8 value;
    explicit NativeCoefficientId(u8 raw) : value(raw) {}
};

// The official formal GKR indices are below 2^40. Preserve u64 logical
// access while storing that authenticated range in five checked bytes.
struct PackedU40 {
    u8 bytes[5]{};
    static constexpr u64 maximum = (u64(1) << 40U) - 1U;

    PackedU40() = default;
    PackedU40(u64 value) { *this = value; }
    PackedU40& operator=(u64 value) {
        if (value > maximum) throw std::length_error("native GKR index exceeds checked 40-bit storage");
        for (unsigned i = 0U; i < 5U; ++i) bytes[i] = static_cast<u8>(value >> (8U * i));
        return *this;
    }
    operator u64() const noexcept {
        u64 value = 0;
        for (unsigned i = 0U; i < 5U; ++i) value |= static_cast<u64>(bytes[i]) << (8U * i);
        return value;
    }
};
static_assert(sizeof(PackedU40) == 5, "native GKR compact index must occupy five bytes");

struct NativeCoefficientHash {
    std::size_t operator()(ll value) const noexcept {
        const unsigned __int128 bits = static_cast<unsigned __int128>(value);
        auto mix = [](std::uint64_t word) { word += 0x9e3779b97f4a7c15ULL; word = (word ^ (word >> 30)) * 0xbf58476d1ce4e5b9ULL; word = (word ^ (word >> 27)) * 0x94d049bb133111ebULL; return word ^ (word >> 31); };
        const std::uint64_t low = static_cast<std::uint64_t>(bits), high = static_cast<std::uint64_t>(bits >> 64);
        const std::uint64_t combined = low ^ (mix(high) + 0x9e3779b97f4a7c15ULL + (low << 6) + (low >> 2));
        return static_cast<std::size_t>(mix(combined));
    }
};

#pragma pack(push, 1)
struct uniGate {
    PackedU40 g, u;
    u8 lu;
    NativeCoefficientId coefficient_id;
    uniGate(u64 _g, u64 _u, u8 _lu, NativeCoefficientId _coefficient_id) :
        g(_g), u(_u), lu(_lu), coefficient_id(_coefficient_id) {
    }
};

struct binGate 
{
    PackedU40 g, u, v;
    u8  l;
    NativeCoefficientId coefficient_id;
    binGate(u64 _g, u64 _u, u64 _v, NativeCoefficientId _coefficient_id, u8 _l):
        g(_g), u(_u), v(_v), l(_l), coefficient_id(_coefficient_id)
        {
    }
    [[nodiscard]] u8 getLayerIdU(u8 layer_id) const { return !l ? 0 : layer_id - 1; }
    [[nodiscard]] u8 getLayerIdV(u8 layer_id) const { return !(l & 1) ? 0 : layer_id - 1; }
};
#pragma pack(pop)

static_assert(sizeof(uniGate) == 12, "native unary gates must use the checked compact record");
static_assert(sizeof(binGate) == 17, "native binary gates must use the checked compact record");

enum class layerType {
    INPUT, FFT, IFFT, ADD_BIAS, RELU, Sqr, OPT_AVG_POOL, MAX_POOL, AVG_POOL, DOT_PROD, PADDING, FCONN,  LAYER_NORM_1, LAYER_NORM_2 ,LAYER_NORM_3,GELU_1,GELU_2,GELU_3,MHA_QK,SOFTMAX_1,SOFTMAX_2,SOFTMAX_3,PLACE_HOLDER
};

class layer {
public:
    layerType ty;
    std::vector<pair<u64,u64> > uni_interval,bin_interval;
	u64 size{}, size_u[2]{}, size_v[2]{};
	i8 bit_length_u[2]{}, bit_length_v[2]{}, bit_length{};
    i8 max_bl_u{}, max_bl_v{};

    bool need_phase2;

    // bit decomp related
    u64 zero_start_id;

    zkgpt::NativeVector<uniGate> uni_gates;
	zkgpt::NativeVector<binGate> bin_gates;
    zkgpt::NativeVector<ll> coefficient_pool;

    zkgpt::NativeVector<PackedU40> ori_id_u, ori_id_v;
    zkgpt::NativeVector<u32> wide_uni_coefficient_ids, wide_bin_coefficient_ids;
    bool wide_coefficient_format = false;
    i8 fft_bit_length;

    std::string native_relation_id;
    std::string native_family_id;
    u64 native_rows{}, native_inner{}, native_cols{};
    u64 native_lhs_offset{}, native_rhs_offset{}, native_output_offset{};

    // iFFT or avg pooling.
    //F scale;

	layer() 
    {
        bit_length_u[0] = bit_length_v[0] = -1;
        size_u[0] = size_v[0] = 0;
        bit_length_u[1] = bit_length_v[1] = -1;
        size_u[1] = size_v[1] = 0;
        need_phase2 = false;
        zero_start_id = 0;
        fft_bit_length = -1;
        //scale = F_ONE;
	}

	void updateSize() {
	    max_bl_u = std::max(bit_length_u[0], bit_length_u[1]);
	    max_bl_v = 0;
	    if (!need_phase2) return;

        max_bl_v = std::max(bit_length_v[0], bit_length_v[1]);
	}

    void reserve_gate_arrays(std::size_t unary_count, std::size_t binary_count);
    void reserve_coefficients(std::size_t distinct_count);
    void release_native_proof_backing() noexcept;
    bool native_metadata_matches(const layer& other) const noexcept;
    void restore_native_proof_backing(layer&& canonical_backing);
    u32 intern_coefficient(ll value);
    void add_unary(u64 g, u64 u, u8 lu, ll value);
    void add_binary(u64 g, u64 u, u64 v, ll value, u8 l);
    ll coefficient(NativeCoefficientId id) const;
    ll coefficient(const uniGate& gate) const;
    ll coefficient(const binGate& gate) const;
    std::uint64_t coefficient_storage_bytes() const;
    static std::uint64_t coefficient_storage_bytes_for(std::size_t distinct_count);

private:
    using CoefficientLookup = std::unordered_map<ll, u32, NativeCoefficientHash>;
    CoefficientLookup coefficient_lookup_;
};

class layeredCircuit {
public:
	vector<layer> circuit;
    u8 size;
	std::vector<std::pair<u64, u64>> source_support_ranges;
    void init(u8 q_bit_size, u8 _layer_sz);
	void set_source_support_ranges(std::vector<std::pair<u64, u64>> ranges) { source_support_ranges = std::move(ranges); }
	void initSubset();
};

