#ifndef HYRAX_DEFINE
#define HYRAX_DEFINE
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>         // std::thread
#include <string>
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
