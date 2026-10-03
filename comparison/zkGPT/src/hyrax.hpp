#include "../gat_adapter/native_storage.hpp"
#ifndef HYRAX_DEFINE
#define HYRAX_DEFINE
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread> // std::thread
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
struct NativeMaskedSumcheckResult;
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

G1 perdersen_commit(G1 *g, ll *f, int n, G1 *W = NULL); // support 2^80, optimized using pippenger
G1 perdersen_commit(G1 *g, Fr *f, int n);
Fr lagrange(Fr *r,int l,int k);
void brute_force_compute_LR(Fr* L,Fr* R,Fr* r,int l);
Fr brute_force_compute_eval(Fr* w,Fr* r,int l);
G1 compute_Tprime(int l,Fr* L,G1* Tk) ;
G1 compute_LT(Fr*w ,Fr*L,int l,G1*g,Fr*& ret);
G1 gen_gi(G1* g,int n);
Pack bullet_reduce(G1 gamma, Fr*a,G1*g,int n,G1& G,Fr* x,Fr y,bool need_free=false);
void prove_dot_product(G1 comm_x, G1 comm_y, Fr *a, G1 *g, G1 &G, Fr *x, Fr y, int n);
void prove_dot_product_with_session(G1 comm_x, G1 comm_y, Fr *a, G1 *g, G1 &G, Fr *x, Fr y, int n, NativeProofSession &session, const std::string &label,
                                    bool enforce_statement_basis = true);
G1* prover_commit(ll* w, G1* g, int l,int thread_n=1);
G1* prover_commit(Fr* w, G1* g, int l,int thread=1);
G1* prover_commit(int* w, G1* g, int l,int thread=1);
Fr prover_evaluate(Fr*w ,Fr*r,G1& G,G1* g, Fr*L,Fr*R,int l);  // nlogn brute force 
namespace hyrax
{
static constexpr const char *kChunkedMleSchema = "zkgpt-hyrax-chunked-mle-v1";
static constexpr std::uint32_t kChunkedMleVersion = 1;
static constexpr std::uint64_t kChunkedMleEntries = 1ULL << 12;
static constexpr const char *kHyraxTensorMleMethod = "hyrax-balanced-blinded-row-column-full-mle";
static constexpr std::uint32_t kHyraxTensorMleVersion = 2;
static constexpr const char *kHyraxTensorMleSchema = "zkgpt-hyrax-balanced-tensor-full-mle-v2";

struct HyraxScalarProvider
{
    std::uint64_t size = 0;
    std::function<Fr(std::uint64_t)> at;
};

enum class HyraxGeneratorProviderScheme : std::uint8_t
{
    OpaqueProvider = 0,
    CanonicalAbsoluteHashToCurveV1 = 1
};

struct HyraxGeneratorProvider
{
    std::uint64_t size = 0;
    std::string canonical_id;
    HyraxGeneratorProviderScheme scheme = HyraxGeneratorProviderScheme::OpaqueProvider;
    std::function<G1(std::uint64_t)> at;
};

struct HyraxMleBasisView
{
    std::uint64_t full_domain = 0;
    std::uint64_t logical_length = 0;
    std::uint64_t active_domain = 0;
    std::string role;
    std::string basis_id;
    std::vector<Fr> point;
    HyraxGeneratorProvider generators;
};

struct HyraxTensorMleView
{
    std::string schema = kHyraxTensorMleSchema;
    std::string method = kHyraxTensorMleMethod;
    std::uint32_t version = kHyraxTensorMleVersion;
    std::uint64_t full_domain = 0;
    std::uint64_t logical_length = 0;
    std::uint64_t active_domain = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::string role;
    std::string basis_id;
    HyraxGeneratorProvider column_generators;
};

struct HyraxTensorMleCommitment
{
    std::string schema = kHyraxTensorMleSchema;
    std::string method = kHyraxTensorMleMethod;
    std::uint32_t version = kHyraxTensorMleVersion;
    std::string role;
    std::string basis_id;
    HyraxGeneratorProviderScheme column_generator_scheme = HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1;
    std::string column_generator_id;
    std::uint64_t full_domain = 0;
    std::uint64_t logical_length = 0;
    std::uint64_t active_domain = 0;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::vector<G1> row_commitments;
    std::string root_digest;
};

struct HyraxTensorMleProverState
{
    std::vector<Fr> row_blindings;
};

struct HyraxTensorZeroInterval
{
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
};

struct ChunkedMleChunkWire
{
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

struct ChunkedMleWire
{
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

struct ChunkedMleProverResult
{
    ChunkedMleWire wire;
    zkgpt::native_pedersen::ScalarOpening aggregate_opening;
};

using ChunkedMleValueProvider = std::function<Fr(std::uint64_t)>;

bool prove_chunked_mle_opening(NativeProofSession &session, const std::string &label, std::uint64_t logical_length, std::uint64_t padded_length,
                               std::uint64_t chunk_size, const std::vector<Fr> &point, const ChunkedMleValueProvider &provider,
                               const zkgpt::native_pedersen::PedersenBasis &scalar_basis, ChunkedMleProverResult &result, std::string *reason = nullptr);

bool verify_chunked_mle_opening(const NativeProofSession &session, const ChunkedMleWire &wire, const std::string &label, std::uint64_t logical_length,
                                std::uint64_t padded_length, std::uint64_t chunk_size, const std::vector<Fr> &point,
                                const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::string *reason = nullptr);

pair<double, double> open(Fr *w, Fr *r, Fr eval, G1 &G, G1 *g, Fr *L, Fr *R, G1 *tk, int l);
pair<double, double> open_with_session(Fr *w, Fr *r, Fr eval, G1 &G, G1 *g, Fr *L, Fr *R, G1 *tk, int l, NativeProofSession &session, const std::string &label);
bool verify_opening_with_session(const G1 &comm_x, const Fr &eval, const G1 &G, const std::vector<G1> &generators, const std::vector<Fr> &basis,
                                 const NativeProofSession &session, const std::string &label, std::string *reason = nullptr,
                                 bool enforce_statement_basis = true, bool allow_trailing_unrelated = false);

struct MaskedMleOpening
{
    G1 evaluation_mask_commitment;
    Fr masked_evaluation;
    bool has_scalar_binding = false;
    zkgpt::native_pedersen::PedersenBasis scalar_basis;
    zkgpt::native_pedersen::ScalarOpening raw_evaluation_opening;
    zkgpt::native_pedersen::ScalarOpening masked_evaluation_opening;
};

struct HyraxTensorZeroIntervalCertificate
{
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::vector<Fr> beta;
    std::vector<Fr> sumcheck_challenges;
    std::vector<Fr> full_terminal_point;
};

struct PedersenMleCommitment
{
    G1 commitment;
    G1 blinding_generator;
    Fr blinding;
    std::string basis_id;
};

bool commit_hyrax_tensor_mle(const HyraxTensorMleView &view, const HyraxScalarProvider &values, HyraxTensorMleCommitment &commitment, HyraxTensorMleProverState &state, std::string *reason = nullptr);
bool open_hyrax_tensor_mle_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const std::vector<Fr> &full_point, const HyraxScalarProvider &values, NativeProofSession &session, const std::string &label, MaskedMleOpening &opening, std::string *reason = nullptr, const std::string &mask_identity = "", const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr, const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr, const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);
bool open_hyrax_tensor_mle_with_claim(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const std::vector<Fr> &full_point, const HyraxScalarProvider &values, const Fr &masked_evaluation, NativeProofSession &session, const std::string &label, MaskedMleOpening &opening, std::string *reason = nullptr, const std::string &mask_identity = "", const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr, const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr, const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);
bool verify_hyrax_tensor_mle_opening_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const std::vector<Fr> &full_point, const G1 &evaluation_mask_commitment, const Fr &masked_evaluation, const NativeProofSession &session, const std::string &label, std::string *reason = nullptr, bool allow_trailing_unrelated = false, const std::string &mask_identity = "", const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr, const zkgpt::native_pedersen::ScalarCommitment *expected_raw_evaluation_commitment = nullptr, const zkgpt::native_pedersen::ScalarCommitment *expected_masked_evaluation_commitment = nullptr);
bool prove_hyrax_tensor_zero_tail_certificate_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const HyraxScalarProvider &values, NativeProofSession &session, std::size_t &exact_position, const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::vector<Fr> &beta, std::vector<Fr> &full_terminal_point, ::NativeMaskedSumcheckResult &sumcheck_result, MaskedMleOpening &terminal_opening, std::string *reason = nullptr, std::vector<std::vector<Fr>> *unmasked_round_trace = nullptr);
bool verify_hyrax_tensor_zero_tail_certificate_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const NativeProofSession &session, std::size_t &exact_position, const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::vector<Fr> &beta, std::vector<Fr> &full_terminal_point, ::NativeMaskedSumcheckResult &sumcheck_result, std::string *reason = nullptr);
bool prove_hyrax_tensor_zero_interval_certificates_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const HyraxScalarProvider &values, const std::vector<HyraxTensorZeroInterval> &trusted_intervals, NativeProofSession &session, std::size_t &exact_position, const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::vector<HyraxTensorZeroIntervalCertificate> &certificates, std::string *reason = nullptr, std::vector<std::vector<std::vector<Fr>>> *unmasked_round_traces = nullptr);
bool verify_hyrax_tensor_zero_interval_certificates_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const std::vector<HyraxTensorZeroInterval> &trusted_intervals, const NativeProofSession &session, std::size_t &exact_position, const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::vector<HyraxTensorZeroIntervalCertificate> &certificates, std::string *reason = nullptr);

G1 derive_pedersen_blinding_generator(const std::string &basis_id);
G1 derive_pedersen_evaluation_generator(const std::string &basis_id, const std::string &label);
G1 derive_masked_response_ipa_base(const std::string &basis_id);

G1 derive_shared_pedersen_evaluation_generator(const std::string &identity);

Fr hyrax_mle_basis_at(const HyraxMleBasisView &view, std::uint64_t canonical_index);
std::string hyrax_generator_set_digest(const HyraxMleBasisView &view);

PedersenMleCommitment pedersen_commit_mle(const HyraxMleBasisView &view, const HyraxScalarProvider &values);

bool make_mle_evaluation_opening(const zkgpt::native_pedersen::PedersenBasis &scalar_basis, const std::vector<Fr> &point, const std::vector<Fr> &values,
                                 zkgpt::native_pedersen::ScalarOpening &opening, std::string *reason = nullptr);

bool make_mle_evaluation_opening(const zkgpt::native_pedersen::PedersenBasis &scalar_basis, const std::vector<Fr> &point, const zkgpt::NativeFieldVector &values,
                                 zkgpt::native_pedersen::ScalarOpening &opening, std::string *reason = nullptr);

PedersenMleCommitment pedersen_commit_mle(const std::vector<G1> &generators, const std::vector<Fr> &values, const std::string &basis_id);

PedersenMleCommitment pedersen_commit_mle(const std::vector<G1> &generators, const zkgpt::NativeFieldVector &values, const std::string &basis_id);

bool open_masked_mle_with_session(const PedersenMleCommitment &commitment, const std::vector<G1> &generators, const std::vector<Fr> &point,
                                  const std::vector<Fr> &values, NativeProofSession &session, const std::string &label, MaskedMleOpening &opening,
                                  std::string *reason = nullptr, bool enforce_statement_basis = true, const std::string &mask_identity = "",
                                  const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                                  const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr,
                                  const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);

bool open_masked_mle_with_session(const PedersenMleCommitment &commitment, const std::vector<G1> &generators, const std::vector<Fr> &point,
                                  const zkgpt::NativeFieldVector &values, NativeProofSession &session, const std::string &label, MaskedMleOpening &opening,
                                  std::string *reason = nullptr, bool enforce_statement_basis = true, const std::string &mask_identity = "",
                                  const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                                  const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr,
                                  const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);

bool open_masked_mle_with_session(const PedersenMleCommitment &commitment, const HyraxMleBasisView &view,
                                  const HyraxScalarProvider &values, NativeProofSession &session, const std::string &label,
                                  MaskedMleOpening &opening, std::string *reason = nullptr, bool enforce_statement_basis = true,
                                  const std::string &mask_identity = "",
                                  const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                                  const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr,
                                  const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);

bool open_masked_mle_with_claim(const PedersenMleCommitment &commitment, const std::vector<G1> &generators, const std::vector<Fr> &point,
                                const std::vector<Fr> &values, const Fr &masked_evaluation, NativeProofSession &session, const std::string &label,
                                MaskedMleOpening &opening, std::string *reason = nullptr, bool enforce_statement_basis = true,
                                const std::string &mask_identity = "", const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                                const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr,
                                const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);

bool open_masked_mle_with_claim(const PedersenMleCommitment &commitment, const std::vector<G1> &generators, const std::vector<Fr> &point,
                                const zkgpt::NativeFieldVector &values, const Fr &masked_evaluation, NativeProofSession &session, const std::string &label,
                                MaskedMleOpening &opening, std::string *reason = nullptr, bool enforce_statement_basis = true,
                                const std::string &mask_identity = "", const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                                const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr,
                                const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);

bool open_masked_mle_with_claim(const PedersenMleCommitment &commitment, const HyraxMleBasisView &view,
                                const HyraxScalarProvider &values, const Fr &masked_evaluation, NativeProofSession &session,
                                const std::string &label, MaskedMleOpening &opening, std::string *reason = nullptr,
                                bool enforce_statement_basis = true, const std::string &mask_identity = "",
                                const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                                const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening = nullptr,
                                const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening = nullptr);

bool verify_masked_mle_opening_with_session(const G1 &commitment, const G1 &evaluation_mask_commitment, const Fr &masked_evaluation,
                                            const std::string &basis_id, const std::vector<G1> &generators, const std::vector<Fr> &point,
                                            const NativeProofSession &session, const std::string &label, std::string *reason = nullptr,
                                            bool enforce_statement_basis = true, bool allow_trailing_unrelated = false, const std::string &mask_identity = "",
                                            const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                                            const zkgpt::native_pedersen::ScalarCommitment *expected_raw_evaluation_commitment = nullptr,
                                            const zkgpt::native_pedersen::ScalarCommitment *expected_masked_evaluation_commitment = nullptr);

bool verify_masked_mle_opening_with_session(const G1 &commitment, const G1 &evaluation_mask_commitment,
                                            const Fr &masked_evaluation, const HyraxMleBasisView &view,
                                            const NativeProofSession &session, const std::string &label,
                                            std::string *reason = nullptr, bool enforce_statement_basis = true,
                                            bool allow_trailing_unrelated = false, const std::string &mask_identity = "",
                                            const zkgpt::native_pedersen::PedersenBasis *scalar_basis = nullptr,
                                            const zkgpt::native_pedersen::ScalarCommitment *expected_raw_evaluation_commitment = nullptr,
                                            const zkgpt::native_pedersen::ScalarCommitment *expected_masked_evaluation_commitment = nullptr);
} // namespace hyrax


#endif
