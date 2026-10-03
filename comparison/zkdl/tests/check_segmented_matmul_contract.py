#!/usr/bin/env python3
"""Check current production host algebra and segmented-opening wire without CUDA kernel compilation."""
from pathlib import Path
import argparse
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent

def definition(source: str, marker: str) -> str:
    start = source.index(marker)
    left = source.index('{', start)
    depth = 0
    quote = None
    escape = False
    for index in range(left, len(source)):
        character = source[index]
        if quote:
            if escape:
                escape = False
            elif character == '\\':
                escape = True
            elif character == quote:
                quote = None
            continue
        if character in ('"', "'"):
            quote = character
        elif character == '{':
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0:
                stop = index + 1
                if source[stop:stop + 1] == ';':
                    stop += 1
                return source[start:stop]
    raise ValueError(f'unterminated production definition: {marker}')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blst', action='store_true', help='use the formal vendored BLST G1 codec')
    args = parser.parse_args()
    relation = (ROOT / 'zkrelation-proof.cuh').read_text()
    proof = (ROOT / 'proof.cuh').read_text()
    matmul = (ROOT / 'zkmatmul-proof.cuh').read_text()
    # These are upstream raw storage typedefs. All field/group/codec/wire arithmetic below comes verbatim from production headers.
    prefix = '''#include <cstdint>
#include <functional>
#include <openssl/hmac.h>
#include <iostream>
#include <limits>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using uint = unsigned int;
#define FR_TENSOR_CUH
#define G1_TENSOR_CUH
constexpr uint blstrs__scalar__Scalar_LIMBS = 8;
constexpr uint blstrs__fp__Fp_LIMBS = 12;
struct blstrs__scalar__Scalar { std::uint32_t val[8]; };
struct blstrs__fp__Fp { std::uint32_t val[12]; };
struct blstrs__g1__G1Affine_jacobian { blstrs__fp__Fp x, y, z; };
using Fr_t = blstrs__scalar__Scalar;
using Fp_t = blstrs__fp__Fp;
using G1Jacobian_t = blstrs__g1__G1Affine_jacobian;
'''
    generators = (ROOT / 'g1-tensor.cuh').read_text()
    prefix += '\n' + definition(generators, 'const Fp_t G1_generator_x_mont =') + '\n' + definition(generators, 'const Fp_t G1_generator_y_mont =') + '\n'
    native_proof = proof[:proof.index('} // namespace zkdl') + len('} // namespace zkdl')] + '\n#endif\n'
    declarations = '\n'.join(definition(relation, marker) for marker in ('class SegmentedTensorProvider', 'struct ZkRelationSegmentLayout {', 'inline bool same_wire(', 'inline void append_fr_wire(', 'inline void append_digest(', 'inline bool registered_model_segment_blinders(', 'inline bool hidden_matmul_shape_valid(', 'struct HiddenMatMulOpening {', 'inline void serialize_hidden_matmul_opening(', 'inline bool deserialize_hidden_matmul_opening(', 'inline bool hidden_matmul_opening_shape_valid(', 'inline FrWire fr_from_signed_i64(', 'inline std::vector<FrWire> hidden_matmul_wire_points(', 'inline bool hidden_matmul_line_values(', 'inline void fold_values(std::vector<FrWire>& values, const FrWire& challenge) {'))
    source = prefix + native_proof + '\n#include "cpu-backend.cuh"\nnamespace zkdl {\n' + definition(matmul, 'struct MatMulShape {') + '\n}\nnamespace zkdl_gat::relation {\nusing FrWire = zkdl::FrWire;\nusing byte = zkdl::byte;\nconstexpr std::uint32_t kMaxRelationPointBits = 30U, kMaxRelationSegments = 4096U;\n' + declarations + '\n}\n'
    source += '''
using namespace zkdl_gat::relation;
static void require(bool value, const char* label) { if (!value) throw std::runtime_error(label); }
int main() {
    const auto native_weights = zkdl::cpu_ipa_mle_weights({zkdl::fr_from_u32(2), zkdl::fr_from_u32(3)});
    require(native_weights == std::vector<FrWire>({zkdl::fr_from_u32(2), zkdl::fr_neg(zkdl::fr_from_u32(3)), zkdl::fr_neg(zkdl::fr_from_u32(4)), zkdl::fr_from_u32(6)}), "native point0 is index MSB");
    const zkdl::MatMulShape shape{1, 4, 4, 4};
    std::string shape_error;
    require(hidden_matmul_shape_valid(shape, &shape_error) && !hidden_matmul_shape_valid(zkdl::MatMulShape{4,4,4,4}, &shape_error), "unimplemented cross-batch relation rejected");
    const ZkRelationSegmentLayout layout{8, 4, 2, 4};
    std::array<zkdl::byte, 32> srs{};
    srs.fill(1);
    std::vector<FrWire> shares;
    require(registered_model_segment_blinders("hidden/layer0/head0/model/seq_kernel", layout, zkdl::fr_from_u32(17), srs, shares, &shape_error), "model segment PRF");
    const auto expected_wire = [](const std::string& hex) { FrWire out{}; for (std::size_t byte = 0; byte < 32; ++byte) out[byte / 4] |= static_cast<std::uint32_t>(std::stoul(hex.substr(2 * byte, 2), nullptr, 16)) << (8 * (byte % 4)); return out; };
    require(shares == std::vector<FrWire>({expected_wire("b32044a1ae999ea8f05671336f4403a5430a18b3e7fbbbefb4b60d83b75d6911"), expected_wire("5fdfbb5e506661570e058dcc935fbaaec1cd895620dc7d4393c68fa69b498462")}) && zkdl::fr_add(shares[0], shares[1]) == zkdl::fr_from_u32(17) && shares[0] != FrWire{} && shares[1] != FrWire{}, "Python C++ registered share vector");
    std::vector<FrWire> a(16), b(16), c(16);
    for (std::size_t i = 0; i < 16; ++i) { a[i] = zkdl::fr_from_u32(i + 1); b[i] = zkdl::fr_from_u32(2 * i + 3); }
    for (std::uint32_t batch = 0; batch < shape.num; ++batch) for (std::uint32_t row = 0; row < 4; ++row) for (std::uint32_t col = 0; col < 4; ++col) for (std::uint32_t inner = 0; inner < 4; ++inner) c[16 * batch + 4 * row + col] = zkdl::fr_add(c[16 * batch + 4 * row + col], zkdl::fr_mul(a[16 * batch + 4 * row + inner], b[16 * batch + 4 * inner + col]));
    const std::vector<Fr_t> row{zkdl::fr_value(zkdl::fr_from_u32(2)), zkdl::fr_value(zkdl::fr_from_u32(3))}, column{zkdl::fr_value(zkdl::fr_from_u32(5)), zkdl::fr_value(zkdl::fr_from_u32(7))}, batch{};
    const auto rp = hidden_matmul_wire_points(row), cp = hidden_matmul_wire_points(column), bp = hidden_matmul_wire_points(batch);
    SegmentedTensorProvider ap, bprov, cprov;
    std::string error;
    require(SegmentedTensorProvider::from_values(a, 4, ap, &error) && SegmentedTensorProvider::from_values(b, 4, bprov, &error) && SegmentedTensorProvider::from_values(c, 4, cprov, &error), "production providers");
    std::vector<FrWire> aline, bline;
    require(hidden_matmul_line_values(ap, shape, row, column, batch, true, aline, &error) && hidden_matmul_line_values(bprov, shape, row, column, batch, false, bline, &error), "production contraction");
    std::vector<FrWire> output_point = bp;
    output_point.insert(output_point.end(), rp.begin(), rp.end());
    output_point.insert(output_point.end(), cp.begin(), cp.end());
    FrWire cv{};
    require(cprov.evaluate_mle(output_point, cv, &error) && cv == zkdl::cpu_ipa_inner_product(c, zkdl::cpu_ipa_mle_weights(output_point)), "segmented canonical MSB MLE");
    require(cv == zkdl::cpu_ipa_inner_product(aline, bline), "full matrix contraction relation");
    const std::vector<FrWire> inner_point{zkdl::fr_from_u32(17), zkdl::fr_from_u32(19)};
    std::vector<FrWire> afull = bp, bfull = bp;
    afull.insert(afull.end(), rp.begin(), rp.end()); afull.insert(afull.end(), inner_point.begin(), inner_point.end());
    bfull.insert(bfull.end(), inner_point.begin(), inner_point.end()); bfull.insert(bfull.end(), cp.begin(), cp.end());
    auto a_folded = aline, b_folded = bline;
    for (auto it = inner_point.rbegin(); it != inner_point.rend(); ++it) { fold_values(a_folded, *it); fold_values(b_folded, *it); }
    require(a_folded.front() == zkdl::cpu_ipa_inner_product(aline, zkdl::cpu_ipa_mle_weights(inner_point)) && b_folded.front() == zkdl::cpu_ipa_inner_product(bline, zkdl::cpu_ipa_mle_weights(inner_point)), "Sumcheck LSB folds reverse to native MSB point");
    FrWire av{}, bv{};
    require(ap.evaluate_mle(afull, av, &error) && bprov.evaluate_mle(bfull, bv, &error) && av == zkdl::cpu_ipa_inner_product(aline, zkdl::cpu_ipa_mle_weights(inner_point)) && bv == zkdl::cpu_ipa_inner_product(bline, zkdl::cpu_ipa_mle_weights(inner_point)), "canonical A/B terminal points");
    for (const std::int64_t value : {std::int64_t(4294967295), std::int64_t(4294967296), std::numeric_limits<std::int64_t>::max(), std::int64_t(-1), std::numeric_limits<std::int64_t>::min()}) { const auto wire = fr_from_signed_i64(value); const auto magnitude = value < 0 ? static_cast<std::uint64_t>(-(value + 1)) + 1 : static_cast<std::uint64_t>(value); FrWire expected{}; expected[0] = magnitude; expected[1] = magnitude >> 32; if (value < 0) expected = zkdl::fr_neg(expected); require(wire == expected, "signed integer high limb"); }
    const auto zero = zkdl::cpu_g1_to_native(zkdl::CpuG1Point::zero());
    std::array<zkdl::byte, 48> encoded{};
    G1Jacobian_t decoded{};
    require(zkdl_g1_wire::encode(zero, encoded, &error) && zkdl_g1_wire::decode(encoded, decoded, &error) && zkdl::cpu_g1_from_native(decoded).is_zero(), "canonical infinity roundtrip");
    require(zkdl::cpu_g1_sum({zkdl::CpuG1Point::zero()}, {FrWire{}}).is_zero(), "zero public segment arithmetic");
    HiddenMatMulOpening bundle;
    bundle.tau = zero;
    for (std::uint32_t index = 0; index < 2; ++index) { zkdl::HyraxOpeningProofV3 part; part.commitment_index = 1; part.evaluation_commitment_index = 9; part.generator_size = 1; part.padded_generator_size = 2; part.logical_domain = 2; part.segment_index = index; part.logical_offset = index; part.actual_length = 1; part.padded_length = 1; part.outer_point = {zkdl::fr_value(zkdl::fr_from_u32(2))}; part.xi = part.tau = part.blinding_generator = part.u_generator = part.delta = part.beta = zero; part.rounds = {zero, zero}; part.descriptor_digest[0] = part.registry_digest[0] = part.srs_digest[0] = 1; bundle.segments.push_back(part); }
    std::vector<zkdl::byte> wire;
    serialize_hidden_matmul_opening(bundle, wire);
    HiddenMatMulOpening parsed;
    std::size_t offset = 0;
    require(deserialize_hidden_matmul_opening(wire, offset, parsed) && offset == wire.size() && hidden_matmul_opening_shape_valid(parsed, 2), "native segmented opening parser roundtrip fixture");
    parsed.segments[1].logical_offset = 0;
    require(!hidden_matmul_opening_shape_valid(parsed, 2), "out-of-order segment rejection");
    wire.pop_back(); offset = 0;
    require(!deserialize_hidden_matmul_opening(wire, offset, parsed), "truncated segment rejection");
    std::cout << "segmented_matmul_contract=PASS native_host_algebra=PASS parser_fixture=PASS canonical_infinity=PASS integer_limbs=PASS model_share_vector=PASS\\n";
}
'''
    with tempfile.TemporaryDirectory(prefix='segmented_contract_', dir=ROOT / 'tests') as temp:
        cpp = Path(temp) / 'check.cpp'
        binary = Path(temp) / 'check'
        cpp.write_text(source)
        command = ['g++', '-std=c++17', '-O0', '-I', str(ROOT)]
        if args.blst:
            command += ['-DZKDL_USE_BLST_G1_CODEC']
        command += [str(cpp)]
        if args.blst:
            command += [str(ROOT / 'third_party/blst/libblst.a')]
        command += ['-lcrypto', '-o', str(binary)]
        subprocess.run(command, check=True, timeout=20)
        subprocess.run([str(binary)], check=True, timeout=10)
        print('codec=' + ('formal_vendored_blst' if args.blst else 'portable'))

if __name__ == '__main__':
    main()
