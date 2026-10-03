#!/usr/bin/env python3
"""Bounded checks of production streamed witness generators and scratch encoding."""
import os
from pathlib import Path
import subprocess
import sys
import math
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_form_resources import cartesian_model_values, registered_model_segment_blinders, R
source = (ROOT / 'gat_adapter/gat_runner.cu').read_text()
build = source[source.index('NativeRelationWitnessInput build_native_relation_witness('):]
helpers = build[build.index('    struct NativeValues {'):build.index('    const auto put = ')]
bits = build[build.index('    const auto bits_for = '):build.index('    const auto group_new = ')]
aux = build[build.index('    const auto round_aux = '):build.index('    const auto write_sorted_lookup_provider = ')]
lookup = build[build.index('    const auto write_sorted_lookup_provider = '):build.index('    const auto put_stage_base = ')]
concat_start = build.index('view(graph.nodes * head_count * width, 2U, [&source, head, head_count, width]')
concat_view = build[concat_start:build.index('));', concat_start) + 1]
cpp = r'''
#define main zkdl_runner_entry
#include "gat_adapter/gat_runner.cu"
#undef main
#include <cassert>
using namespace zkdl_gat;
int main(int argc, char **argv) {
    assert(argc == 2);
    Config config;
    config.baseline_root = std::filesystem::path(argv[1]);
    config.run_root = config.baseline_root;
    Graph graph;
    graph.nodes = 2; graph.features = 3; graph.edges = 3; graph.src = {1,0,1}; graph.dst = {0,0,1}; graph.node_edge_ptr = {0,2,3};
    RelationPlan plan;
    std::vector<NativeSortedLookupProviderInput> sorted_lookup_providers;
''' + helpers + bits + aux + lookup + r'''
    const auto check = [&](const auto &input, const std::vector<u64> &logical, const std::vector<u64> &padded, const std::vector<i64> &expected) {
        const auto storage = NativeFamilyStorage::persist_generated(config.run_root, input.size(), input.encoding, [&](u64 index) { return input[index]; });
        std::vector<NativeFrWire> values;
        std::string error;
        assert(storage->read_padded(logical, padded, 0, expected.size(), values, &error));
        for (std::size_t index = 0; index < expected.size(); ++index) assert(values.at(index) == native_wire(expected.at(index)));
    };
    const std::vector<i64> integers = {4294967295LL,4294967296LL,std::numeric_limits<i64>::max(),-1,std::numeric_limits<i64>::min()};
    for (const auto value : integers) { i64 decoded = 0; assert(stored_signed_value(native_wire(value), decoded)); assert(decoded == value); assert(native_wire(value) == stored_from_signed(value)); }
    check(native_i64_values(integers), {5}, {8}, {integers[0],integers[1],integers[2],integers[3],integers[4],0,0,0});
    Matrix matrix{2,3,{1,2,3,4,5,6}};
    check(native_matrix_values(matrix), {2,3}, {2,4}, {1,2,3,0,4,5,6,0});
    const auto broadcast = native_broadcast_node_matrix(graph, matrix);
    check(broadcast, {3,3}, {4,4}, {1,2,3,0,1,2,3,0,4,5,6,0,0,0,0,0});
    check(native_route_matrix(graph, matrix, true), {3,3}, {4,4}, {4,5,6,0,1,2,3,0,4,5,6,0,0,0,0,0});
    const std::vector<std::uint8_t> groups{1,0,1};
    check(native_prefix(broadcast, groups, 3, false), {3,3}, {4,4}, {1,2,3,0,2,4,6,0,4,5,6,0,0,0,0,0});
    check(native_prefix(broadcast, groups, 3, true), {3,3}, {4,4}, {0,0,0,0,1,2,3,0,0,0,0,0,0,0,0,0});
    const std::vector<i64> bit_source{0,3,5};
    check(bits_for(bit_source, 3, "bits"), {3,3}, {4,4}, {0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0});
    const std::vector<i64> queries{-2,1,3};
    check(lookup_bits(queries, -2, 5, 3, "lookup"), {3,3}, {4,4}, {0,0,0,0,1,1,0,0,1,0,1,0,0,0,0,0});
    SignedRoundDivWitness witness;
    witness.rows=2; witness.cols=3; witness.sigma={0,1,0,1,0,1}; witness.z={1,0,0,0,0,0}; witness.a={0,1,2,3,4,5}; witness.q={0,0,1,1,2,2}; witness.r={0,1,0,1,0,1}; witness.delta={1,0,1,0,1,0}; witness.divisor={2,2,2,2,2,2};
    std::vector<i64> round_expected;
    for (u64 row=0; row<2; ++row) for (u64 col=0; col<4; ++col) { const auto cell=row*3+col; if (col==3) round_expected.insert(round_expected.end(),8,0); else { const auto expected=std::vector<i64>{witness.sigma[cell],witness.z[cell],witness.a[cell],witness.q[cell],witness.r[cell],witness.delta[cell],witness.divisor[cell],0}; round_expected.insert(round_expected.end(),expected.begin(),expected.end()); } }
    check(round_aux(witness), {2,3,8}, {2,4,8}, round_expected);
    const Matrix &source = matrix;
    const u64 head=1, head_count=3, width=3;
    const auto concat = ''' + concat_view + r''';
    std::vector<i64> concat_expected;
    for(u64 node=0; node<2; ++node) for(u64 h=0; h<4; ++h) for(u64 col=0; col<4; ++col) concat_expected.push_back(h==head && col<3 ? matrix.at(node,col) : 0);
    check(concat, {2,3,3}, {2,4,4}, concat_expected);
    SortedLookupQueryPlan descriptor;
    descriptor.operation_id="lookup/check"; descriptor.commitment_role_map="table,query,merge"; descriptor.table_domain=4; descriptor.query_domain=4; descriptor.merge_domain=8; descriptor.segment_size=4; descriptor.table_logical_rows=3; descriptor.query_logical_rows=3; descriptor.merge_logical_rows=6; descriptor.x_bits=2; descriptor.y_bits=4; descriptor.key_max_u128="63";
    plan.sorted_lookup.queries.push_back(descriptor);
    write_sorted_lookup_provider(descriptor.operation_id, {-1,-1,1}, -1, {3,3,9}, {3,3,9});
    assert(sorted_lookup_providers.size()==1);
    std::ifstream stream(config.baseline_root / sorted_lookup_providers[0].scratch_path,std::ios::binary);
    const auto read_u64=[&]() { u64 result=0; for(unsigned byte=0;byte<8;++byte) result|=static_cast<u64>(static_cast<unsigned char>(stream.get()))<<(8*byte); return result; };
    const auto skip_string=[&]() { const auto length=read_u64(); stream.seekg(length,std::ios::cur); };
    stream.seekg(12); skip_string(); skip_string(); stream.seekg(8*7+4*2,std::ios::cur); skip_string();
    assert(read_u64()==3); stream.seekg(4*107,std::ios::cur); assert(read_u64()==3); stream.seekg(4*107,std::ios::cur); assert(read_u64()==6);
    const std::vector<u64> keys{0,0,0,16,38,38}; const std::vector<unsigned> sources{1,2,2,1,1,2}; const std::vector<unsigned> markers{1,0,0,1,1,0};
    for(std::size_t index=0;index<keys.size();++index) { std::array<unsigned char,107> record{}; stream.read(reinterpret_cast<char*>(record.data()),record.size()); assert(stream); u64 key=0; for(unsigned byte=0;byte<8;++byte) key|=static_cast<u64>(record[byte])<<(8*byte); assert(key==keys[index]); assert(record[104]==1 && record[105]==markers[index] && record[106]==sources[index]); }
    const std::vector<std::vector<u64>> model_shapes{{2,3},{3,7},{3,2,3}};
    for (std::size_t fixture=0; fixture<model_shapes.size(); ++fixture) {
        const auto logical=model_shapes[fixture];
        std::vector<u64> padded; u64 logical_count=1, padded_count=1;
        for(const auto dimension:logical) { u64 aligned=1; while(aligned<dimension) aligned<<=1; padded.push_back(aligned); logical_count*=dimension; padded_count*=aligned; }
        const auto storage=NativeFamilyStorage::persist_generated(config.run_root,logical_count,2U,[](u64 index) { return native_wire(index%5==0 ? -static_cast<i64>(index+1) : static_cast<i64>(index+1)); });
        std::vector<NativeFrWire> actual; std::string error;
        assert(storage->read_padded(logical,padded,0,padded_count,actual,&error));
        std::ifstream expected(config.run_root / ("model_padding_"+std::to_string(fixture)+".bin"),std::ios::binary);
        for(const auto& value:actual) { std::array<unsigned char,32> bytes{}; expected.read(reinterpret_cast<char*>(bytes.data()),bytes.size()); assert(expected); NativeFrWire decoded{}; for(unsigned limb=0;limb<8;++limb) for(unsigned byte=0;byte<4;++byte) decoded[limb]|=static_cast<std::uint32_t>(bytes[limb*4+byte])<<(8*byte); assert(decoded==value); }
        assert(expected.get()==std::char_traits<char>::eof());
    }
    std::cout << "COMPACT_WITNESS_OK integers padding bits round_aux route broadcast concat prefix sorted_duplicates model_cartesian_padding\n";
}
'''
with tempfile.TemporaryDirectory(prefix='.compact-check-', dir=ROOT / 'tests') as raw:
    directory = Path(raw)
    translation = directory / 'compact_check.cpp'
    binary = directory / 'compact_check'
    translation.write_text(cpp)
    subprocess.run(['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O0', '-arch=sm_121', '-Xcompiler=-ffunction-sections,-fdata-sections', '-Xlinker=--gc-sections', '-I', str(ROOT), '-x', 'cu', str(translation), '-lcrypto', '-o', str(binary)], check=True, timeout=45)
    scratch = directory / 'scratch'
    scratch.mkdir()
    os.chmod(scratch, 0o700)
    for fixture, logical in enumerate(([2, 3], [3, 7], [3, 2, 3])):
        padded = [1 << (dimension - 1).bit_length() for dimension in logical]
        values = [(-(index + 1) if index % 5 == 0 else index + 1) % R for index in range(math.prod(logical))]
        expected = cartesian_model_values(logical, padded, values)
        (scratch / f'model_padding_{fixture}.bin').write_bytes(b''.join(value.to_bytes(32, 'little') for value in expected))
    for count in (1, 2, 4):
        shares = registered_model_segment_blinders('hidden/layer0/head0/model/seq_kernel', 17, 8, 8 // count, count, '01' * 32)
        assert len(shares) == count and all(0 < share < R for share in shares) and sum(shares) % R == 17
    shares = registered_model_segment_blinders('hidden/layer0/head0/model/seq_kernel', 17, 8, 4, 2, '01' * 32)
    assert [share.to_bytes(32, 'little').hex() for share in shares] == ['b32044a1ae999ea8f05671336f4403a5430a18b3e7fbbbefb4b60d83b75d6911', '5fdfbb5e506661570e058dcc935fbaaec1cd895620dc7d4393c68fa69b498462']
    subprocess.run([str(binary), str(scratch)], check=True, timeout=5)
