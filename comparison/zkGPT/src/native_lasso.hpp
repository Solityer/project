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

static constexpr std::uint32_t kNativeLassoMatrixLayoutVersion = 2;

struct NativeLassoFamilyPublic {
    std::string family_id;
    std::string table_id;
    std::string resource_id;
    std::string registration_id;
    std::vector<F> table;
    struct MatrixDescriptor {
        std::uint32_t version = kNativeLassoMatrixLayoutVersion;
        std::uint64_t padded_queries = 0;
        std::uint64_t table_domain = 0;
        std::uint64_t vector_size = 0;
        std::uint64_t rows = 0;
        std::uint64_t columns = 0;
        std::uint32_t row_bits = 0;
        std::uint32_t column_bits = 0;
        std::string flatten_order = "zkgpt-lasso-q-plus-table-times-Q-lsb-first-balanced-v2";
        std::string generator_scheme = "CanonicalAbsoluteHashToCurveV1";
        std::vector<G1> column_generators;
        std::string trusted_generator_id;
    } matrix_layout;
    std::vector<G1> legacy_clear_matrix_generators;
    std::uint64_t segment_offset = 0;
    std::uint64_t logical_queries = 0;
    std::uint64_t padded_queries = 0;
    std::uint64_t padding_index = 0;
    F padding_value = F_ZERO;

    std::string statement_tables_digest;
    std::string statement_resources_digest;
};

struct NativeLassoFamilyProver : NativeLassoFamilyPublic {
    zkgpt::NativeVector<std::uint64_t> selected_table_rows;
    std::function<std::uint64_t(std::uint64_t)> selected_table_row_at;
    std::vector<F> queries;
    std::vector<F> legacy_clear_matrix;
};

using NativeLassoSparseEntryCallback = std::function<void(std::uint64_t, const F&)>;

struct NativeLassoSparseView {
    // The callbacks borrow source-family private storage; keep it alive and stable while using the view.
    std::uint64_t full_domain = 0;
    std::uint64_t nonzero_count = 0;
    std::function<void(const NativeLassoSparseEntryCallback&)> for_each_nonzero;
    std::function<F(std::uint64_t)> value_at;
    std::function<F(const std::vector<F>&)> evaluate_mle;
};

// Per-family native Lasso data bytes; family peaks include this family's ledger and exclude shared tensor, Hyrax, and transcript buffers.
struct NativeLassoCapacityProfile {
    std::uint64_t padded_queries = 0;
    std::uint64_t table_domain = 0;
    std::uint64_t full_domain = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t row_commitment_chunk_count = 0;
    std::uint64_t booleanity_rounds = 0;
    std::uint64_t onehot_rounds = 0;
    std::uint64_t lookup_rounds = 0;
    std::uint64_t hyrax_matrix_masked_openings = 4;
    std::uint64_t hyrax_l_qry_tensor_openings = 1;
    bool t2_booleanity = false;
    std::uint64_t private_witness_entries = 0;
    std::uint64_t private_witness_element_bytes = 0;
    std::uint64_t private_witness_bytes = 0;
    std::uint64_t row_blinding_bytes = 0;
    std::uint64_t row_commitment_bytes = 0;
    std::uint64_t column_generator_bytes = 0;
    std::uint64_t row_fold_weights_bytes = 0;
    std::uint64_t row_fold_values_bytes = 0;
    // One retained std::vector over table_domain (selected_table_row_mle result).
    std::uint64_t onehot_query_bytes = 0;
    // T2-only NativeFieldVector over padded_queries, live during weighted Booleanity.
    std::uint64_t t2_column_one_bytes = 0;
    // Ordinary onehot fold's simultaneously live left/current/next std::vectors.
    std::uint64_t onehot_heap_peak_bytes = 0;
    std::uint64_t masked_opening_generator_bytes = 0;
    std::uint64_t masked_opening_coefficient_bytes = 0;
    std::uint64_t masked_opening_response_bytes = 0;
    std::uint64_t row_ledger_retained_bytes = 0;
    std::uint64_t commit_peak_bytes = 0;
    std::uint64_t booleanity_peak_bytes = 0;
    std::uint64_t row_fold_peak_bytes = 0;
    std::uint64_t opening_peak_bytes = 0;
    std::uint64_t family_peak_bytes = 0;
};

NativeLassoFamilyPublic::MatrixDescriptor native_lasso_balanced_matrix_layout(std::uint64_t padded_queries, std::uint64_t table_domain);
NativeLassoSparseView native_lasso_sparse_view(const NativeLassoFamilyProver& family);
NativeLassoCapacityProfile native_lasso_capacity_profile(std::uint64_t padded_queries, std::uint64_t table_domain);

struct NativeLassoQueryProver {
    G1 G;
    std::vector<G1> generators;
    std::vector<F> values;
    hyrax::HyraxGeneratorProvider generator_provider;
    G1 commitment;

    std::string basis_id;
    G1 blinding_generator;
    F commitment_blinding = F_ZERO;
    bool has_commitment_blinding = false;
    std::string statement_l_digest;
    hyrax::HyraxTensorMleView tensor_view;
    hyrax::HyraxTensorMleCommitment tensor_commitment;
    hyrax::HyraxTensorMleProverState tensor_state;
    hyrax::HyraxScalarProvider value_provider;
};

struct NativeLassoQueryPublic {
    G1 G;
    std::vector<G1> generators;
    G1 commitment;
    std::string basis_id;
    std::string statement_l_digest;
    hyrax::HyraxGeneratorProvider generator_provider;
    hyrax::HyraxTensorMleView tensor_view;
    hyrax::HyraxTensorMleCommitment tensor_commitment;
};

using NativeLassoTensorTailProveHook = std::function<bool(NativeProofSession&, const NativeLassoQueryProver&, std::string*)>;
using NativeLassoTensorTailVerifyHook = std::function<bool(const NativeProofSession&, const NativeLassoQueryPublic&, std::size_t&, std::string*)>;

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
std::string native_lasso_tensor_query_digest(const hyrax::HyraxTensorMleView& view, const hyrax::HyraxTensorMleCommitment& commitment);
std::string native_lasso_generator_digest(const G1& G, const std::vector<G1>& generators);
std::string native_lasso_matrix_layout_digest(const NativeLassoFamilyPublic::MatrixDescriptor& layout);

bool native_lasso_prove_bundle(NativeProofSession& session, std::vector<NativeLassoFamilyProver>& families, NativeLassoQueryProver& query,
                               NativeLassoBundleResult* result = nullptr, std::string* reason = nullptr,
                               // Required by the formal typed-tensor path; runs Aux then L_qry after all M_f row chunks and before M_f β challenges.
                               const NativeLassoTensorTailProveHook& tensor_tail_stage = {});

bool native_lasso_verify_bundle(const NativeProofSession& session, const std::vector<NativeLassoFamilyPublic>& families, const NativeLassoQueryPublic& query,
                                NativeLassoBundleResult* result = nullptr, std::string* reason = nullptr,
                                // Required by the formal typed-tensor path; consumes Aux then L_qry tail scopes at the exact transcript cursor.
                                const NativeLassoTensorTailVerifyHook& tensor_tail_stage = {});

#endif
