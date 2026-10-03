#include "../gat_adapter/native_storage.hpp"
#include "circuit.h"
#include "polynomial.h"
#include "utils.hpp"
#include <mcl/bn256.hpp>
#include <cassert>
#include <iostream>

using namespace mcl::bn;

int main() {
    initPairing(mcl::BN_SNARK1);
    const u64 high = (u64(1) << 32) + 7;
    assert(ceilPow2BitLength(0) == -1 && ceilPow2BitLength(u64(1) << 32) == 32 && ceilPow2BitLength((u64(1) << 32) + 1) == 33);
    assert(floorPow2BitLength(u64(1) << 32) == 32 && ceilPow2BitLength(~u64(0)) == 64);
    const uniGate high_gate(high, high + 1, 0, 1);
    assert(high_gate.g == high && high_gate.u == high + 1);
    layeredCircuit circuit;
    circuit.init(0, 2);
    circuit.circuit[0].size = u64(1) << 33;
    circuit.circuit[0].bit_length = 33;
    auto& output = circuit.circuit[1];
    output.ty = layerType::ADD_BIAS;
    output.size = output.zero_start_id = 3;
    output.bit_length = 2;
    output.uni_gates.emplace_back(0, high, 0, 1);
    output.uni_gates.emplace_back(1, 3, 0, 1);
    output.uni_gates.emplace_back(2, high, 0, 1);
    output.bin_gates.emplace_back(0, high, high + 1, 1, 0);
    circuit.initSubset();
    assert(output.size_u[0] == 2 && output.ori_id_u[0] == high && output.ori_id_u[1] == 3 && output.ori_id_v[0] == high + 1);
    assert(output.uni_gates[0].u == 0 && output.uni_gates[1].u == 1 && output.uni_gates[2].u == 0 && output.bin_gates[0].u == 0);
    const auto initial_mappings = zkgpt::native_storage::mappings.size();
    {
        zkgpt::NativeFieldVector fields(300000, F(0));
        assert(zkgpt::native_storage::mappings.size() == initial_mappings + 1);
        fields.front() = F(17);
        fields.back() = F(-5);
        auto copied = fields;
        auto moved = std::move(copied);
        assert(moved.front() == F(17) && moved.back() == F(-5) && fields[1000].isZero());
        moved.resize(400000, F(0));
        assert(moved.front() == F(17) && moved[299999] == F(-5) && moved.back().isZero());
        moved.clear();
        moved.shrink_to_fit();
        zkgpt::NativeVector<linear_poly> polynomials(150000);
        assert(polynomials.front().a.isZero() && polynomials.back().b.isZero());
        bool overflow_rejected = false;
        try { zkgpt::NativeScratchAllocator<F> allocator; allocator.allocate(std::numeric_limits<std::size_t>::max()); }
        catch (const std::length_error&) { overflow_rejected = true; }
        assert(overflow_rejected);
    }
    assert(zkgpt::native_storage::mappings.size() == initial_mappings);
    std::vector<F> values(5003), point(13);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = F(static_cast<int>(i % 79) - 40);
    for (std::size_t i = 0; i < point.size(); ++i) point[i] = F(static_cast<int>(i + 2));
    auto dense = values;
    dense.resize(std::size_t(1) << point.size(), F(0));
    std::size_t active = dense.size();
    for (const F& coordinate : point) { active >>= 1; for (std::size_t i = 0; i < active; ++i) dense[i] = dense[2 * i] * (F(1) - coordinate) + dense[2 * i + 1] * coordinate; }
    assert(zkgpt::native_mle_evaluate(values, point) == dense.front());
    F direct(0);
    zkgpt::native_for_each_equality_weight(point, values.size(), [&](std::size_t index, const F& weight) { F reference(1); for (std::size_t bit = 0; bit < point.size(); ++bit) reference *= ((index >> bit) & 1) ? point[bit] : F(1) - point[bit]; assert(reference == weight); direct += values[index] * weight; });
    assert(direct == dense.front());
    std::vector<F> beta_point(17, F(3));
    zkgpt::NativeFieldVector beta_table(std::size_t(1) << beta_point.size(), F(0));
    initBetaTable(beta_table, beta_point.size(), beta_point.begin(), F(1), 2);
    F beta_sum(0);
    for (const F& weight : beta_table) beta_sum += weight;
    assert(beta_sum == F(1));
    F beta_first(1);
    for (const F& coordinate : beta_point) beta_first *= F(1) - coordinate;
    assert(beta_table.front() == beta_first);
    const auto digest = zkgpt::native_field_vector_digest(values, "native-storage-test-v2");
    std::string encoded("native-storage-test-v2");
    auto u64_bytes = [&](u64 value) { for (unsigned byte = 0; byte < 8; ++byte) encoded.push_back(static_cast<char>((value >> (8 * byte)) & 255)); };
    u64_bytes(2);
    u64_bytes(values.size());
    for (const F& value : values) { const std::string field = value.getStr(16); u64_bytes(field.size()); encoded += field; }
    cybozu::Sha256 reference_hash;
    char raw[32];
    reference_hash.digest(raw, sizeof(raw), encoded.data(), encoded.size());
    const char* hex = "0123456789abcdef";
    std::string reference_digest;
    for (unsigned char byte : raw) { reference_digest += hex[byte >> 4]; reference_digest += hex[byte & 15]; }
    assert(digest == reference_digest);
    values[1] += F(1);
    assert(zkgpt::native_field_vector_digest(values, "native-storage-test-v2") != digest);
    values[1] -= F(1);
    std::swap(values[0], values[1]);
    assert(zkgpt::native_field_vector_digest(values, "native-storage-test-v2") != digest);
    values.pop_back();
    assert(zkgpt::native_field_vector_digest(values, "native-storage-test-v2") != digest);
    assert(zkgpt::native_mle_evaluate(std::vector<F>{F(9)}, {}) == F(9));
    std::cout << "NATIVE_STORAGE_CAPACITY_OK high_index=" << high << " scalar_bytes=" << sizeof(F) << " linear_poly_bytes=" << sizeof(linear_poly) << " uni_gate_bytes=" << sizeof(uniGate) << " bin_gate_bytes=" << sizeof(binGate) << " generator_bytes=" << sizeof(G1) << '\n';
}
