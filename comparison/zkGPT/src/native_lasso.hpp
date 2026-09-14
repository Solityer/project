#ifndef ZKGPT_NATIVE_LASSO_HPP
#define ZKGPT_NATIVE_LASSO_HPP

#include "hyrax_rp.hpp"
#include "native_pedersen_relation.hpp"

#include <cstddef>
#include <string>
#include <vector>

// Public identity/table side of one baseline-local Lasso family.  The table,
// resource and registration identities are public authenticated inputs; the
// matrix and query vectors live only in NativeLassoFamilyProver.
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
    // Optional adapter-provided statement identities.  When present they
    // must match the native statement fields; otherwise the canonical local
    // digests below are used (for standalone native tests/legacy callers).
    std::string statement_tables_digest;
    std::string statement_resources_digest;
};

struct NativeLassoFamilyProver : NativeLassoFamilyPublic {
    // Column-major M_f: matrix[row + column * rows].  Entries must be bits;
    // every row has exactly one selected table column.
    std::vector<F> matrix;
    std::vector<F> queries;
};

struct NativeLassoQueryProver {
    G1 G;
    std::vector<G1> generators;
    std::vector<F> values;
    G1 commitment;
    // The dynamic L_qry commitment is a hiding Pedersen MLE commitment.  The
    // blinding scalar remains prover-only and is never serialized.
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
    // Authenticated scalar claim for the final L_qry opening.  The opening
    // witness stays prover-local; this commitment is safe to hand to the
    // GAT combiner together with final_query_point.
    zkgpt::native_pedersen::ScalarCommitment final_l_commitment;
    bool has_final_l_commitment = false;
};

// These labels are the only terminal-opening interface exposed by this
// stage.  W/Aux are intentionally reservations, not implemented openings.
struct NativeLassoTerminalOpeningLabels {
    static const char *l_qry();
    static const char *w_reserved();
    static const char *aux_reserved();
};

std::string native_lasso_table_digest(const NativeLassoFamilyPublic &family);
std::string native_lasso_tables_digest(
    const std::vector<NativeLassoFamilyPublic> &families);
std::string native_lasso_resources_digest(
    const std::vector<NativeLassoFamilyPublic> &families);
std::string native_lasso_query_digest(const G1 &commitment, std::size_t size);
std::string native_lasso_generator_digest(const G1 &G,
                                          const std::vector<G1> &generators);

// Produces all LC_f commitments before deriving any family query challenge,
// then emits lookup product Sumcheck, M_f Hyrax opening, query/input
// consistency opening, and one final Open_L at a fresh input point.
bool native_lasso_prove_bundle(
    NativeProofSession &session,
    std::vector<NativeLassoFamilyProver> &families,
    NativeLassoQueryProver &query,
    NativeLassoBundleResult *result = nullptr,
    std::string *reason = nullptr);

// Independent verifier.  It receives no matrix, query, or witness arrays;
// all accepted claims are reconstructed from public tables, commitments and
// the typed transcript.
bool native_lasso_verify_bundle(
    const NativeProofSession &session,
    const std::vector<NativeLassoFamilyPublic> &families,
    const NativeLassoQueryPublic &query,
    NativeLassoBundleResult *result = nullptr,
    std::string *reason = nullptr);

#endif
