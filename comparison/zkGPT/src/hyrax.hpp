#ifndef HYRAX_DEFINE
#define HYRAX_DEFINE
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>         // std::thread
#include <string>
#include <cstdint>
#include <functional>
#include <mcl/bn256.hpp>
#include "typedef.hpp"
#include "timer.hpp"
#include "hyrax_util.hpp"
#include "native_pedersen_relation.hpp"
using namespace std;
using namespace mcl::bn;
class NativeProofSession;
struct Hyrax_first_round
{
    //transmitted data
    Fr* tk;
    G1 comm_w;
};
struct Hyrax_second_round
{  
    //public data
    G1 *g, G;
    Fr* R;
    //prover only data
    Fr* LT;
    Fr eval;
};

G1 perdersen_commit(G1* g,ll* f,int n,G1* W=NULL); //support 2^80, optimized using pippenger
G1 perdersen_commit(G1* g,Fr* f,int n);
Fr lagrange(Fr *r,int l,int k);
void brute_force_compute_LR(Fr* L,Fr* R,Fr* r,int l);
Fr brute_force_compute_eval(Fr* w,Fr* r,int l);
G1 compute_Tprime(int l,Fr* L,G1* Tk) ;
G1 compute_LT(Fr*w ,Fr*L,int l,G1*g,Fr*& ret);
G1 gen_gi(G1* g,int n);
Pack bullet_reduce(G1 gamma, Fr*a,G1*g,int n,G1& G,Fr* x,Fr y,bool need_free=false);
void prove_dot_product(G1 comm_x, G1 comm_y, Fr* a, G1*g ,G1& G,Fr* x,Fr y,int n);
void prove_dot_product_with_session(G1 comm_x, G1 comm_y, Fr* a, G1*g,
                                    G1& G, Fr* x, Fr y, int n,
                                    NativeProofSession &session,
                                    const std::string &label,
                                    bool enforce_statement_basis = true);
G1* prover_commit(ll* w, G1* g, int l,int thread_n=1);
G1* prover_commit(Fr* w, G1* g, int l,int thread=1);
G1* prover_commit(int* w, G1* g, int l,int thread=1);
Fr prover_evaluate(Fr*w ,Fr*r,G1& G,G1* g, Fr*L,Fr*R,int l);  // nlogn brute force 
namespace hyrax
{
// Bounded segmented MLE opening wire.  A chunk is a real Hyrax/Pedersen
// commitment and opening; the root below only authenticates descriptor order
// and lengths and is never accepted as a replacement for the group opening.
static constexpr const char *kChunkedMleSchema =
    "zkgpt-hyrax-chunked-mle-v1";
static constexpr std::uint32_t kChunkedMleVersion = 1;
static constexpr std::uint64_t kChunkedMleEntries = 1ULL << 12;

struct ChunkedMleChunkWire {
    std::uint64_t offset = 0;
    std::uint64_t actual_length = 0;
    std::uint64_t padded_length = 0;
    std::string label;
    std::string basis_id;
    std::string commitment_hex;
    std::string evaluation_mask_commitment_hex;
    std::string masked_evaluation;
    std::string raw_evaluation_commitment_hex;
    std::string masked_evaluation_commitment_hex;
};

struct ChunkedMleWire {
    std::string schema = kChunkedMleSchema;
    std::uint32_t version = kChunkedMleVersion;
    std::string label;
    std::string basis_id;
    std::string scalar_domain;
    std::uint64_t logical_length = 0;
    std::uint64_t padded_length = 0;
    std::uint64_t chunk_size = kChunkedMleEntries;
    std::string point_digest;
    std::string root_digest;
    std::string aggregate_commitment_hex;
    std::vector<ChunkedMleChunkWire> chunks;
};

struct ChunkedMleProverResult {
    ChunkedMleWire wire;
    zkgpt::native_pedersen::ScalarOpening aggregate_opening;
};

using ChunkedMleValueProvider = std::function<Fr(std::uint64_t)>;

// Commits and opens a segmented MLE using at most chunk_size generators and
// values at a time.  The per-chunk openings are bound to an aggregate scalar
// commitment by a native Pedersen linear relation, so the canonical root is
// only an ordering digest and never the proof of an opening.
bool prove_chunked_mle_opening(
    NativeProofSession &session, const std::string &label,
    std::uint64_t logical_length, std::uint64_t padded_length,
    std::uint64_t chunk_size, const std::vector<Fr> &point,
    const ChunkedMleValueProvider &provider,
    const zkgpt::native_pedersen::PedersenBasis &scalar_basis,
    ChunkedMleProverResult &result, std::string *reason = nullptr);

// Independent verifier for the preceding wire.  It reconstructs the
// chunk-local generator bases and checks every Hyrax masked opening followed
// by the aggregate Pedersen relation; no witness/value callback is accepted.
bool verify_chunked_mle_opening(
    const NativeProofSession &session, const ChunkedMleWire &wire,
    const std::string &label, std::uint64_t logical_length,
    std::uint64_t padded_length, std::uint64_t chunk_size,
    const std::vector<Fr> &point,
    const zkgpt::native_pedersen::PedersenBasis &scalar_basis,
    std::string *reason = nullptr);

pair<double,double> open(Fr*w,Fr*r,Fr eval,G1&G,G1*g,Fr*L,Fr*R,G1*tk,int l);
pair<double,double> open_with_session(Fr*w,Fr*r,Fr eval,G1&G,G1*g,
                                      Fr*L,Fr*R,G1*tk,int l,
                                      NativeProofSession &session,
                                      const std::string &label);
// Independent verifier for the native Hyrax/Bulletproof opening transcript.
// It consumes only public generators/basis, the claimed point and the
// serialized transcript; no prover object or witness array is consulted.
bool verify_opening_with_session(
    const G1 &comm_x, const Fr &eval, const G1 &G,
    const std::vector<G1> &generators, const std::vector<Fr> &basis,
    const NativeProofSession &session, const std::string &label,
    std::string *reason = nullptr,
    bool enforce_statement_basis = true,
    bool allow_trailing_unrelated = false);

// Baseline-local zero-knowledge MLE opening.  The public vector commitment is
// C = <g,v> + H*rho, where H is derived from a domain-separated commitment
// identity and rho is retained only by the prover.  A masked evaluation claim
// is q = <basis,v> + mu together with D = J*mu.  The transcript is a
// Fiat-Shamir Schnorr proof of knowledge of (v,rho,mu); it never serializes
// the raw evaluation, folded terminal scalar, or either blinding scalar.
struct MaskedMleOpening {
    G1 evaluation_mask_commitment;
    Fr masked_evaluation;
    // Prover-local scalar bindings.  The values/blindings never cross the
    // transcript boundary; only their commitments and the joint Schnorr
    // equations are serialized by the opening implementation.
    bool has_scalar_binding = false;
    zkgpt::native_pedersen::PedersenBasis scalar_basis;
    zkgpt::native_pedersen::ScalarOpening raw_evaluation_opening;
    zkgpt::native_pedersen::ScalarOpening masked_evaluation_opening;
};

struct PedersenMleCommitment {
    G1 commitment;
    G1 blinding_generator;
    Fr blinding; // prover-only; never part of an artifact wire
    std::string basis_id;
};

G1 derive_pedersen_blinding_generator(const std::string &basis_id);
G1 derive_pedersen_evaluation_generator(const std::string &basis_id,
                                        const std::string &label);

// A transcript-global evaluation-mask generator used when several
// independently committed MLEs must expose claims under one linear masking
// relation.  The identity is domain separated from every commitment basis so
// the same J can be checked across W/Aux/L and the GKR input claim.
G1 derive_shared_pedersen_evaluation_generator(const std::string &identity);

// Computes the private scalar opening for the MLE evaluation at point.  The
// returned commitment uses the caller's native scalar Pedersen basis and is
// intended to be supplied back to open_masked_mle_* as a forced raw opening;
// no value or blinding is serialized by this helper.
bool make_mle_evaluation_opening(
    const zkgpt::native_pedersen::PedersenBasis &scalar_basis,
    const std::vector<Fr> &point, const std::vector<Fr> &values,
    zkgpt::native_pedersen::ScalarOpening &opening,
    std::string *reason = nullptr);

PedersenMleCommitment pedersen_commit_mle(
    const std::vector<G1> &generators, const std::vector<Fr> &values,
    const std::string &basis_id);

bool open_masked_mle_with_session(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const std::vector<Fr> &values, NativeProofSession &session,
    const std::string &label, MaskedMleOpening &opening,
    std::string *reason = nullptr, bool enforce_statement_basis = true,
    const std::string &mask_identity = "",
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);

// Same opening protocol, with a caller-supplied hidden evaluation claim.
// This is used when a masked Sumcheck terminal and a Pedersen MLE opening
// must bind to the same transcript value.  The raw evaluation and the
// resulting mask are never serialized; the prover only proves knowledge of
// the corresponding masked opening witness.
bool open_masked_mle_with_claim(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const std::vector<Fr> &values, const Fr &masked_evaluation,
    NativeProofSession &session, const std::string &label,
    MaskedMleOpening &opening, std::string *reason = nullptr,
    bool enforce_statement_basis = true,
    const std::string &mask_identity = "",
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);

bool verify_masked_mle_opening_with_session(
    const G1 &commitment, const G1 &evaluation_mask_commitment,
    const Fr &masked_evaluation, const std::string &basis_id,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const NativeProofSession &session, const std::string &label,
    std::string *reason = nullptr, bool enforce_statement_basis = true,
    bool allow_trailing_unrelated = false,
    const std::string &mask_identity = "",
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
    const zkgpt::native_pedersen::ScalarCommitment *expected_raw_evaluation_commitment = nullptr,
    const zkgpt::native_pedersen::ScalarCommitment *expected_masked_evaluation_commitment = nullptr);
}


#endif
