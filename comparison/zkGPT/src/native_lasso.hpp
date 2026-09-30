#ifndef ZKGPT_NATIVE_LASSO_HPP
#define ZKGPT_NATIVE_LASSO_HPP

#include "hyrax_rp.hpp"
#include "hyrax.hpp"
#include "native_pedersen_relation.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct NativeLassoFamilyPublic {
    std::string family_id;
    std::string table_id;
    std::string resource_id;
    std::string registration_id;
    std::vector<F> table;
    std::vector<G1> matrix_generators;
    std::size_t segment_offset = 0;
    std::size_t logical_queries = 0;
    std::size_t padded_queries = 0;
    std::size_t padding_index = 0;
    F padding_value = F_ZERO;

    std::string statement_tables_digest;
    std::string statement_resources_digest;
};

struct NativeLassoFamilyProver : NativeLassoFamilyPublic {
    std::vector<F> matrix;
    std::vector<F> queries;
};

struct NativeLassoQueryProver {
    G1 G;
    std::vector<G1> generators;
    std::vector<F> values;
    G1 commitment;

    std::string basis_id;
    G1 blinding_generator;
    F commitment_blinding = F_ZERO;
    bool has_commitment_blinding = false;
    std::string statement_l_digest;
};

struct NativeLassoQueryPublic {
    G1 G;
    std::vector<G1> generators;
    G1 commitment;
    std::string basis_id;
    std::string statement_l_digest;
};

struct NativeLassoBundleResult {
    std::vector<std::vector<F>> family_query_points;
    std::vector<G1> family_commitments;
    std::vector<F> final_query_point;
    F final_l_evaluation = F_ZERO;

    zkgpt::native_pedersen::ScalarCommitment final_l_commitment;
    bool has_final_l_commitment = false;
};

static constexpr std::uint32_t kNativeLassoChunkVersion = 1;
static constexpr std::uint64_t kNativeLassoChunkEntries = 1ULL << 12;

struct NativeLassoChunkFamilyPublic {
    std::string family_id;
    std::string table_id;
    std::string resource_id;
    std::string registration_id;
    std::vector<F> table;
    std::size_t segment_offset = 0;
    std::size_t logical_queries = 0;
    std::size_t padded_queries = 0;
    std::size_t padding_index = 0;
    F padding_value = F_ZERO;
    std::uint64_t chunk_size = kNativeLassoChunkEntries;
    std::string statement_tables_digest;
    std::string statement_resources_digest;
};

struct NativeLassoChunkFamilyProver : NativeLassoChunkFamilyPublic {
    std::function<F(std::uint64_t)> query_value;
};

struct NativeLassoChunkFamilyWire {
    NativeLassoChunkFamilyPublic public_metadata;
    std::string row_point_digest;
    hyrax::ChunkedMleWire query;
    std::vector<hyrax::ChunkedMleWire> matrix_columns;
    std::string lookup_terminal_commitment_hex;
    std::string onehot_terminal_commitment_hex;
};

struct NativeLassoChunkBundleWire {
    std::string schema = "zkgpt-native-lasso-chunked-v1";
    std::uint32_t version = kNativeLassoChunkVersion;
    std::string tables_digest;
    std::string resources_digest;
    std::string root_digest;
    std::vector<NativeLassoChunkFamilyWire> families;
    std::string final_commitment_hex;
};

struct NativeLassoChunkBundleProverResult {
    NativeLassoChunkBundleWire wire;
    zkgpt::native_pedersen::ScalarOpening final_opening;
    F final_l_evaluation = F_ZERO;
};

bool native_lasso_prove_chunked_bundle(NativeProofSession& session, std::vector<NativeLassoChunkFamilyProver>& families,
                                       const zkgpt::native_pedersen::PedersenBasis& scalar_basis, const std::vector<F>* global_point = nullptr,
                                       NativeLassoChunkBundleProverResult* result = nullptr, std::string* reason = nullptr);

bool native_lasso_verify_chunked_bundle(const NativeProofSession& session, const std::vector<NativeLassoChunkFamilyPublic>& families,
                                        const NativeLassoChunkBundleWire& wire, const zkgpt::native_pedersen::PedersenBasis& scalar_basis,
                                        const std::vector<F>* global_point = nullptr, std::string* reason = nullptr);

struct NativeLassoTerminalOpeningLabels {
    static const char* l_qry();
    static const char* w_reserved();
    static const char* aux_reserved();
};

std::string native_lasso_table_digest(const NativeLassoFamilyPublic& family);
std::string native_lasso_tables_digest(const std::vector<NativeLassoFamilyPublic>& families);
std::string native_lasso_resources_digest(const std::vector<NativeLassoFamilyPublic>& families);
std::string native_lasso_query_digest(const G1& commitment, std::size_t size);
std::string native_lasso_generator_digest(const G1& G, const std::vector<G1>& generators);

bool native_lasso_prove_bundle(NativeProofSession& session, std::vector<NativeLassoFamilyProver>& families, NativeLassoQueryProver& query,
                               NativeLassoBundleResult* result = nullptr, std::string* reason = nullptr);

bool native_lasso_verify_bundle(const NativeProofSession& session, const std::vector<NativeLassoFamilyPublic>& families, const NativeLassoQueryPublic& query,
                                NativeLassoBundleResult* result = nullptr, std::string* reason = nullptr);

#endif
