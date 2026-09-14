#include "native_masked_sumcheck.hpp"

#include <mcl/bn256.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace mcl::bn;
using zkgpt::native_pedersen::LinearRelationProof;
using zkgpt::native_pedersen::MultiplicationLinks;
using zkgpt::native_pedersen::PedersenBasis;
using zkgpt::native_pedersen::Product3Links;
using zkgpt::native_pedersen::ScalarCommitment;
using zkgpt::native_pedersen::ScalarOpening;

namespace {

const char kMagic[] = "zkgpt-masked-sumcheck-v2";

void set_reason(std::string *reason, const std::string &message) {
    if (reason != nullptr) *reason = message;
}

bool prefix(const std::string &value, const std::string &wanted) {
    return value.compare(0, wanted.size(), wanted) == 0;
}

bool valid_label(const std::string &label) {
    if (label.empty() || label.size() > 512 || label.front() == '/' ||
        label.back() == '/' || label.find("//") != std::string::npos ||
        label.find('|') != std::string::npos || prefix(label, "challenge/"))
        return false;
    for (unsigned char c : label)
        if (c <= 0x20 || c == 0x7f) return false;
    return true;
}

Fr random_field() {
    Fr value;
    value.setByCSPRNG();
    return value;
}

F interpolate_at(const std::vector<F> &values, const F &point) {
    if (values.empty() || values.size() > 4)
        throw std::invalid_argument("masked Sumcheck degree must be in 1..3");
    F result = F_ZERO;
    for (std::size_t i = 0; i < values.size(); ++i) {
        F basis = F_ONE;
        const F xi = F(static_cast<int>(i));
        for (std::size_t j = 0; j < values.size(); ++j) {
            if (i == j) continue;
            const F xj = F(static_cast<int>(j));
            basis *= (point - xj) / (xi - xj);
        }
        result += values[i] * basis;
    }
    return result;
}

F mle_eval(const std::vector<F> &values, const std::vector<F> &point) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0 ||
        values.size() != (std::size_t(1) << point.size()))
        throw std::invalid_argument("masked public MLE dimensions are invalid");
    std::vector<F> current = values;
    for (const F &r : point) {
        const std::size_t half = current.size() / 2;
        std::vector<F> next(half);
        for (std::size_t i = 0; i < half; ++i)
            next[i] = current[2 * i] + r * (current[2 * i + 1] - current[2 * i]);
        current.swap(next);
    }
    return current.front();
}

void append_string(std::string &out, const std::string &value) {
    const std::uint64_t size = static_cast<std::uint64_t>(value.size());
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<char>((size >> shift) & 0xffU));
    out.append(value);
}

void append_u64(std::string &out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8)
        out.push_back(static_cast<char>((value >> shift) & 0xffU));
}

struct Reader {
    const std::string &payload;
    std::size_t offset = 0;

    bool u64(std::uint64_t &value) {
        if (payload.size() - offset < 8) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i)
            value |= static_cast<std::uint64_t>(
                         static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    }

    bool string(std::string &value, std::size_t maximum) {
        std::uint64_t size = 0;
        if (!u64(size) || size > maximum || size > payload.size() - offset)
            return false;
        value.assign(payload.data() + offset, static_cast<std::size_t>(size));
        offset += static_cast<std::size_t>(size);
        return true;
    }
    bool done() const { return offset == payload.size(); }
};

std::string point_hex(const G1 &point) { return point.serializeToHexStr(); }

bool decode_point(const std::string &encoded, G1 &point) {
    if (encoded.empty() || encoded.size() > 4096) return false;
    try {
        point.deserializeHexStr(encoded);
        return point.serializeToHexStr() == encoded && point.isValid() &&
               point.isValidOrder() && !point.isZero();
    } catch (...) {
        return false;
    }
}

void encode_points(std::string &payload, const std::vector<ScalarCommitment> &points) {
    append_u64(payload, static_cast<std::uint64_t>(points.size()));
    for (const ScalarCommitment &point : points) append_string(payload, point_hex(point.point));
}

bool decode_points(Reader &reader, const PedersenBasis &basis, std::size_t count,
                   std::vector<ScalarCommitment> &points, std::string *reason) {
    std::uint64_t encoded_count = 0;
    if (!reader.u64(encoded_count) || encoded_count != count) {
        set_reason(reason, "masked Sumcheck commitment vector length mismatch");
        return false;
    }
    points.clear();
    points.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::string encoded;
        G1 point;
        ScalarCommitment commitment;
        if (!reader.string(encoded, 4096) || !decode_point(encoded, point)) {
            set_reason(reason, "malformed masked Sumcheck commitment point");
            return false;
        }
        commitment.domain = basis.domain;
        commitment.point = point;
        if (!commitment.valid(basis, reason)) return false;
        points.push_back(commitment);
    }
    return true;
}

std::string basis_domain(const NativeProofSession &session) {
    return "zkgpt/masked-sumcheck/v2/" + session.statement_digest();
}

PedersenBasis derive_basis(const NativeProofSession &session) {
    return zkgpt::native_pedersen::derive_basis(basis_domain(session));
}

std::string initial_label(const std::string &label, const char *name) {
    return label + "/initial/" + name;
}

std::string round_label(const std::string &label, std::size_t round,
                        const char *name, std::size_t index = 0,
                        bool indexed = false) {
    std::string result = label + "/round/" + std::to_string(round) + "/" + name;
    if (indexed) result += "/" + std::to_string(index);
    return result;
}

std::string claim_label(const std::string &label, std::size_t round,
                        const char *name) {
    return label + "/round/" + std::to_string(round) + "/claim/" + name;
}

std::string terminal_label(const std::string &label, const char *name,
                           std::size_t index = 0, bool indexed = false) {
    std::string result = label + "/terminal/" + name;
    if (indexed) result += "/" + std::to_string(index);
    return result;
}

bool same_point(const ScalarCommitment &a, const ScalarCommitment &b) {
    return a.domain == b.domain && point_hex(a.point) == point_hex(b.point);
}

bool external_shape(const NativeMaskedTerminalSpec &spec, const PedersenBasis &basis,
                    std::size_t terminal_arity, bool prover, std::string *reason) {
    if (!spec.has_external_binding || !spec.initial_commitment.valid(basis, reason)) {
        set_reason(reason, "masked Sumcheck requires an authenticated external initial binding");
        return false;
    }
    if (prover && !spec.initial_opening.valid(basis, reason)) return false;
    if (prover && !same_point(spec.initial_opening.commitment, spec.initial_commitment)) {
        set_reason(reason, "masked initial opening/commitment mismatch");
        return false;
    }
    if (spec.public_initial_claim) {
        if (spec.has_masked_initial_binding) {
            set_reason(reason, "public masked Sumcheck scope cannot carry a masked initial binding");
            return false;
        }
    } else if (!spec.has_masked_initial_binding) {
        set_reason(reason, "private masked Sumcheck scope is missing its masked initial binding");
        return false;
    } else {
        if (!spec.masked_initial_commitment.valid(basis, reason)) return false;
        if (prover &&
            (!spec.masked_initial_opening.valid(basis, reason) ||
             !same_point(spec.masked_initial_opening.commitment,
                         spec.masked_initial_commitment)))
            return false;
    }
    const bool fixed = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU ||
                       spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV;
    const std::size_t fixed_arity = fixed ? 2 : 0;
    if (prover) {
        if (spec.unmasked_terminal_openings.size() != terminal_arity ||
            spec.unmasked_fixed_openings.size() != fixed_arity) {
            set_reason(reason, "masked terminal unmasked opening arity mismatch");
            return false;
        }
        if (spec.unmasked_fixed_values.size() != fixed_arity) {
            set_reason(reason, "masked fixed-side unmasked value arity mismatch");
            return false;
        }
        for (const ScalarOpening &opening : spec.unmasked_terminal_openings)
            if (!opening.valid(basis, reason)) return false;
        for (const ScalarOpening &opening : spec.unmasked_fixed_openings)
            if (!opening.valid(basis, reason)) return false;
        if ((!spec.terminal_openings.empty() &&
             spec.terminal_openings.size() != terminal_arity) ||
            (!spec.fixed_openings.empty() &&
             spec.fixed_openings.size() != fixed_arity) ||
            (!spec.terminal_commitments.empty() &&
             spec.terminal_commitments.size() != terminal_arity) ||
            (!spec.fixed_commitments.empty() &&
             spec.fixed_commitments.size() != fixed_arity)) {
            set_reason(reason, "masked terminal external opening arity mismatch");
            return false;
        }
        if (!spec.terminal_openings.empty()) {
            for (std::size_t i = 0; i < terminal_arity; ++i) {
                if (!spec.terminal_openings[i].valid(basis, reason)) return false;
                if (!spec.terminal_commitments.empty() &&
                    (!spec.terminal_commitments[i].valid(basis, reason) ||
                     !same_point(spec.terminal_openings[i].commitment,
                                 spec.terminal_commitments[i]))) return false;
            }
        }
        if (!spec.fixed_openings.empty()) {
            for (std::size_t i = 0; i < fixed_arity; ++i) {
                if (!spec.fixed_openings[i].valid(basis, reason)) return false;
                if (!spec.fixed_commitments.empty() &&
                    (!spec.fixed_commitments[i].valid(basis, reason) ||
                     !same_point(spec.fixed_openings[i].commitment,
                                 spec.fixed_commitments[i]))) return false;
            }
        }
    } else {
        // Terminal output commitments are produced inside this scope and are
        // authenticated by the unmasked relation plus the per-terminal mask
        // bindings below.  A caller may optionally provide expected output
        // commitments (for composition), but cannot be required to know them
        // before replaying the scope.  Fixed-side commitments, on the other
        // hand, are externally supplied operands and remain mandatory.
        if (!spec.terminal_commitments.empty() &&
            spec.terminal_commitments.size() != terminal_arity) {
            set_reason(reason, "masked terminal external commitment arity mismatch");
            return false;
        }
        if (spec.fixed_commitments.size() != fixed_arity) {
            set_reason(reason, "masked fixed external commitment arity mismatch");
            return false;
        }
        for (const ScalarCommitment &commitment : spec.terminal_commitments)
            if (!commitment.valid(basis, reason)) return false;
        for (const ScalarCommitment &commitment : spec.fixed_commitments)
            if (!commitment.valid(basis, reason)) return false;
    }
    return true;
}

bool terminal_shape(const NativeMaskedTerminalSpec &spec, std::size_t degree,
                    std::size_t &arity, std::string *reason) {
    if (degree < 1 || degree > 3) {
        set_reason(reason, "masked Sumcheck degree must be in 1..3");
        return false;
    }
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar: arity = 1; break;
    case NativeMaskedTerminalKind::Product2: arity = 2; break;
    case NativeMaskedTerminalKind::Product3: arity = 3; break;
    case NativeMaskedTerminalKind::Claims:
        arity = 2;
        if (spec.terminal_linear_coefficients.size() != 2) {
            set_reason(reason, "masked claims relation needs two coefficients");
            return false;
        }
        break;
    case NativeMaskedTerminalKind::GkrPredicate:
        arity = 4;
        if (spec.gkr_coefficients.size() != 5) {
            set_reason(reason, "masked GKR predicate needs five coefficients");
            return false;
        }
        break;
    case NativeMaskedTerminalKind::Product2Public:
        arity = 1;
        if (spec.public_vector.empty() ||
            (spec.public_vector.size() & (spec.public_vector.size() - 1)) != 0) {
            set_reason(reason, "masked public product relation needs a power-of-two vector");
            return false;
        }
        break;
    case NativeMaskedTerminalKind::GkrPredicateFixedU:
    case NativeMaskedTerminalKind::GkrPredicateFixedV:
        arity = 2;
        if (spec.gkr_coefficients.size() != 5 || spec.fixed_values.size() != 2) {
            set_reason(reason, "masked fixed-side GKR predicate dimensions are invalid");
            return false;
        }
        break;
    default:
        set_reason(reason, "unknown masked terminal relation");
        return false;
    }
    if ((spec.kind == NativeMaskedTerminalKind::Product2 && degree != 2) ||
        (spec.kind == NativeMaskedTerminalKind::Product3 && degree != 3) ||
        (spec.kind == NativeMaskedTerminalKind::Product2Public && degree != 2) ||
        (spec.kind == NativeMaskedTerminalKind::GkrPredicate && degree != 2) ||
        ((spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU ||
          spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV) && degree != 2)) {
        set_reason(reason, "masked terminal relation degree mismatch");
        return false;
    }
    return true;
}

std::vector<F> lagrange_coefficients(std::size_t degree, const F &point) {
    std::vector<F> result(degree + 1);
    for (std::size_t i = 0; i <= degree; ++i) {
        F value = F_ONE;
        const F xi = F(static_cast<int>(i));
        for (std::size_t j = 0; j <= degree; ++j) {
            if (i == j) continue;
            value *= (point - F(static_cast<int>(j))) /
                     (xi - F(static_cast<int>(j)));
        }
        result[i] = value;
    }
    return result;
}

// The public streaming/free wrappers are kept before the implementation
// helpers below.  Declare the transcript/terminal helpers here so that the
// wrappers and their fail-closed verifier can be compiled without relying on
// declaration order across the two implementation sections.
constexpr std::size_t kMaxRounds = 1ULL << 20;
bool equal_commitment(const ScalarCommitment &, const ScalarCommitment &);
bool decode_field_vector(const NativeProofMessage &, std::size_t,
                         std::vector<F> &, std::string *);
bool expect_message(const NativeProofSession &, std::size_t &,
                    NativeProofMessageKind, const std::string &,
                    const NativeProofMessage **, std::string *);
bool check_terminal(const NativeMaskedTerminalSpec &, const F &,
                    const std::vector<F> &, bool, std::string *);
bool sample_terminal(const NativeMaskedTerminalSpec &, const F &,
                     std::vector<F> &, std::string *);
bool unique_index(const NativeProofSession &, const std::string &, std::size_t &,
                  std::string *);
bool verify_opening_block(const NativeProofSession &, const PedersenBasis &,
                          std::size_t &, const std::string &,
                          ScalarCommitment &, std::string *);
bool verify_linear_block(const NativeProofSession &, const PedersenBasis &,
                         std::size_t &, const std::string &,
                         const std::vector<ScalarCommitment> &,
                         const std::vector<F> &, const F &, std::string *);
bool decode_challenge_at(const NativeProofSession &, std::size_t,
                         const std::string &, F &, std::string *);
bool opening_commitment_at(const NativeProofSession &, const PedersenBasis &,
                           std::size_t, const std::string &, ScalarCommitment &,
                           std::string *);
bool consume_multiplication_block(const NativeProofSession &, std::size_t &,
                                  const std::string &, std::string *);
bool consume_product3_block(const NativeProofSession &, std::size_t &,
                            const std::string &, std::string *);
std::vector<F> ones(std::size_t);

} // namespace

bool native_masked_opening_commitment_at(
    const NativeProofSession &session,
    const zkgpt::native_pedersen::PedersenBasis &basis,
    std::size_t cursor, const std::string &label,
    zkgpt::native_pedersen::ScalarCommitment &commitment,
    std::string *reason) {
    return opening_commitment_at(session, basis, cursor, label, commitment,
                                 reason);
}

namespace {

struct NativeMaskedSumcheckProverState {
    NativeProofSession *session = nullptr;
    std::string label;
    std::size_t degree = 0;
    NativeMaskedTerminalSpec terminal_spec;
    std::size_t checkpoint = 0;
    std::size_t rounds = 0;
    F masked_running = F_ZERO;
    F unmasked_running = F_ZERO;
    F mask_target = F_ZERO;
    bool has_previous_mask_eval = false;
    ScalarOpening previous_mask_eval;
    bool finished = false;
    PedersenBasis basis;
    NativeMaskedSumcheckResult result;
};

bool initialize_external(NativeMaskedSumcheckProverState &, const F *,
                         std::string *);
bool prove_round(NativeMaskedSumcheckProverState &, const std::vector<F> &, F *,
                 std::string *);
bool finish_state(NativeMaskedSumcheckProverState &, const std::vector<F> &,
                  NativeMaskedSumcheckResult *, std::string *);
bool make_terminal_opening(const PedersenBasis &, const ScalarOpening *, const F &,
                           ScalarOpening &, const std::string &, std::string *);
bool relation_opening(const ScalarOpening &, const PedersenBasis &, const F &,
                      std::string *);
bool prove_terminal_relation(NativeMaskedSumcheckProverState &,
                             const std::vector<F> &, std::vector<ScalarOpening> &,
                             std::vector<ScalarOpening> &,
                             std::vector<ScalarOpening> &,
                             std::vector<ScalarOpening> &,
                             std::vector<ScalarOpening> &, std::string *);
bool verify_initial_scope(const NativeProofSession &, const PedersenBasis &,
                          const std::string &, const F &,
                          const NativeMaskedTerminalSpec &, std::size_t &,
                          bool &, ScalarCommitment &, NativeMaskedSumcheckResult &,
                          std::string *);
bool verify_round(const NativeProofSession &, const PedersenBasis &,
                  const std::string &, std::size_t, std::size_t, const F &, bool,
                  const ScalarCommitment &, std::size_t &,
                  NativeMaskedRoundBinding &, F &, F &, ScalarCommitment &,
                  std::string *);
bool verify_terminal_relation(const NativeProofSession &, const PedersenBasis &,
                              const std::string &, const NativeMaskedTerminalSpec &,
                              const F &, const std::vector<F> &, std::size_t &,
                              NativeMaskedSumcheckResult *, std::string *);
bool verify_unmasked_terminal_relation(const NativeProofSession &, const PedersenBasis &,
                                       const std::string &, const NativeMaskedTerminalSpec &,
                                       std::size_t &, NativeMaskedSumcheckResult *,
                                       std::string *);

} // namespace

struct NativeMaskedSumcheckProver::Impl : NativeMaskedSumcheckProverState {};

NativeMaskedSumcheckProver::NativeMaskedSumcheckProver(
    NativeProofSession &session, const std::string &label, std::size_t degree,
    const F &initial_claim, const NativeMaskedTerminalSpec &terminal_spec,
    const F *unmasked_initial_claim)
    : impl_(new Impl) {
    impl_->session = &session;
    impl_->label = label;
    impl_->degree = degree;
    impl_->terminal_spec = terminal_spec;
    impl_->checkpoint = session.checkpoint();
    impl_->masked_running = initial_claim;
    impl_->unmasked_running = unmasked_initial_claim == nullptr
        ? initial_claim : *unmasked_initial_claim;
    impl_->mask_target = impl_->masked_running - impl_->unmasked_running;
    impl_->result.external_binding_required = true;
    try {
        if (!valid_label(label) || !session.valid() || degree < 1 || degree > 3)
            throw std::invalid_argument("masked streaming Sumcheck identity is invalid");
        std::size_t arity = 0;
        const bool placeholder_claims =
            terminal_spec.kind == NativeMaskedTerminalKind::Claims &&
            terminal_spec.terminal_linear_coefficients.empty();
        if ((!placeholder_claims &&
             !terminal_shape(terminal_spec, degree, arity, nullptr)) ||
            (placeholder_claims && degree < 1))
            throw std::invalid_argument("masked streaming terminal relation is invalid");
        if (terminal_spec.has_masked_initial_claim &&
            terminal_spec.masked_initial_claim != initial_claim)
            throw std::invalid_argument("masked streaming initial claim mismatch");
        if (terminal_spec.public_initial_claim) {
            if (unmasked_initial_claim != nullptr)
                throw std::invalid_argument("public masked Sumcheck scope cannot receive a private initial claim");
        } else if (unmasked_initial_claim == nullptr) {
            throw std::invalid_argument("private masked Sumcheck scope requires an unmasked initial claim");
        }
        impl_->basis = derive_basis(session);
        if (!initialize_external(*impl_, unmasked_initial_claim, nullptr))
            throw std::invalid_argument("masked streaming external binding is missing");
        impl_->result.basis = impl_->basis;
    } catch (...) {
        session.rollback(impl_->checkpoint);
        throw;
    }
}

NativeMaskedSumcheckProver::~NativeMaskedSumcheckProver() = default;

std::size_t NativeMaskedSumcheckProver::rounds() const {
    return impl_ == nullptr ? 0 : impl_->rounds;
}

const NativeMaskedSumcheckResult &NativeMaskedSumcheckProver::result() const {
    return impl_->result;
}

bool NativeMaskedSumcheckProver::append_round(const std::vector<F> &unmasked_poly,
                                              F *challenge,
                                              std::string *reason) {
    if (impl_ == nullptr || impl_->finished) {
        set_reason(reason, "masked streaming Sumcheck is already finished");
        return false;
    }
    try {
        if (!prove_round(*impl_, unmasked_poly, challenge, reason))
            throw std::runtime_error(reason != nullptr && !reason->empty()
                                         ? *reason : "masked round failed");
        return true;
    } catch (const std::exception &error) {
        impl_->session->rollback(impl_->checkpoint);
        impl_->finished = true;
        set_reason(reason, error.what());
        return false;
    }
}

bool NativeMaskedSumcheckProver::set_terminal_spec(
    const NativeMaskedTerminalSpec &terminal_spec, std::string *reason) {
    if (impl_ == nullptr || impl_->finished) {
        set_reason(reason, "masked streaming Sumcheck is already finished");
        return false;
    }
    NativeMaskedTerminalSpec effective = terminal_spec;
    // A phase-2 GKR caller may supply only the relation-specific fields; the
    // authenticated initial binding remains the one fixed at construction.
    if (!effective.has_external_binding && impl_->terminal_spec.has_external_binding) {
        effective.has_external_binding = true;
        effective.public_initial_claim = impl_->terminal_spec.public_initial_claim;
        effective.initial_opening = impl_->terminal_spec.initial_opening;
        effective.initial_commitment = impl_->terminal_spec.initial_commitment;
        effective.has_masked_initial_binding =
            impl_->terminal_spec.has_masked_initial_binding;
        effective.masked_initial_opening =
            impl_->terminal_spec.masked_initial_opening;
        effective.masked_initial_commitment =
            impl_->terminal_spec.masked_initial_commitment;
    }
    if (effective.public_initial_claim != impl_->terminal_spec.public_initial_claim) {
        set_reason(reason, "masked Sumcheck initial scope mode cannot change after rounds");
        return false;
    }
    std::size_t arity = 0;
    if (!terminal_shape(effective, impl_->degree, arity, reason)) return false;
    impl_->terminal_spec = std::move(effective);
    return true;
}

bool NativeMaskedSumcheckProver::finish(const std::vector<F> &unmasked_terminal,
                                        NativeMaskedSumcheckResult *result,
                                        std::string *reason) {
    if (impl_ == nullptr || impl_->finished) {
        set_reason(reason, "masked streaming Sumcheck is already finished");
        return false;
    }
    try {
        if (!finish_state(*impl_, unmasked_terminal, result, reason))
            throw std::runtime_error(reason != nullptr && !reason->empty()
                                         ? *reason : "masked terminal relation failed");
        return true;
    } catch (const std::exception &error) {
        impl_->session->rollback(impl_->checkpoint);
        impl_->finished = true;
        set_reason(reason, error.what());
        return false;
    }
}

bool native_masked_sumcheck_prove(
    NativeProofSession &session, const std::string &label, std::size_t degree,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const NativeMaskedTerminalSpec &terminal_spec,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    if (!terminal_spec.has_external_binding) {
        set_reason(reason, "masked Sumcheck external binding is required");
        return false;
    }
    if (round_polys.size() > kMaxRounds) {
        set_reason(reason, "masked Sumcheck round count is invalid");
        return false;
    }
    try {
        NativeMaskedSumcheckProver prover(
            session, label, degree, initial_claim, terminal_spec, nullptr);
        for (const std::vector<F> &poly : round_polys)
            if (!prover.append_round(poly, nullptr, reason)) return false;
        return prover.finish(terminal_values, result, reason);
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool native_masked_sumcheck_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t degree, std::size_t rounds, const F &initial_claim,
    const NativeMaskedTerminalSpec &terminal_spec,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    try {
        if (!valid_label(label) || rounds > kMaxRounds || !session.valid(reason) ||
            !session.replay(reason) || !terminal_spec.has_external_binding) {
            set_reason(reason, "masked Sumcheck verifier requires external binding");
            return false;
        }
        std::size_t arity = 0;
        if (!terminal_shape(terminal_spec, degree, arity, reason)) return false;
        const PedersenBasis basis = derive_basis(session);
        if (!external_shape(terminal_spec, basis, arity, false, reason)) return false;
        NativeMaskedSumcheckResult local;
        local.basis = basis;
        local.external_binding_required = true;
        std::size_t cursor = 0;
        bool has_mask = false;
        ScalarCommitment previous_mask;
        if (!verify_initial_scope(session, basis, label, initial_claim,
                                  terminal_spec, cursor, has_mask, previous_mask,
                                  local, reason)) return false;
        F running = initial_claim;
        local.challenges.reserve(rounds);
        for (std::size_t round = 0; round < rounds; ++round) {
            NativeMaskedRoundBinding binding;
            F challenge;
            F next_running;
            ScalarCommitment next_mask;
            if (!verify_round(session, basis, label, round, degree, running,
                              round != 0 || has_mask, previous_mask, cursor,
                              binding, challenge, next_running, next_mask,
                              reason)) return false;
            local.rounds.push_back(binding);
            local.challenges.push_back(challenge);
            running = next_running;
            previous_mask = next_mask;
            has_mask = true;
        }
        NativeMaskedTerminalSpec effective = terminal_spec;
        if (effective.kind == NativeMaskedTerminalKind::Product2Public)
            effective.public_evaluation = mle_eval(effective.public_vector,
                                                   local.challenges);
        if (!verify_unmasked_terminal_relation(session, basis, label, effective,
                                               cursor, &local, reason))
            return false;
        const NativeProofMessage *terminal_message = nullptr;
        if (!expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                            label + "/terminal/values", &terminal_message, reason))
            return false;
        if (!decode_field_vector(*terminal_message, arity, local.terminal_values,
                                 reason)) return false;
        if (!check_terminal(effective, running, local.terminal_values, false, reason))
            return false;
        if (!verify_terminal_relation(session, basis, label, effective, running,
                                      local.terminal_values, cursor, &local, reason))
            return false;
        local.final_claim = running;
        local.transcript_end = cursor;
        local.external_binding_verified = true;
        if (result != nullptr) *result = std::move(local);
        return true;
    } catch (const std::exception &error) {
        set_reason(reason, error.what());
        return false;
    }
}

bool native_masked_gkr_quadratic_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const std::vector<F> &gkr_coefficients,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::GkrPredicate;
    spec.gkr_coefficients = gkr_coefficients;
    set_reason(reason, "masked GKR requires an external-bound terminal spec");
    (void)session; (void)label; (void)round_polys; (void)initial_claim;
    (void)terminal_values; (void)result;
    return false;
}

bool native_masked_gkr_quadratic_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim,
    const std::vector<F> &gkr_coefficients,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::GkrPredicate;
    spec.gkr_coefficients = gkr_coefficients;
    (void)session; (void)label; (void)rounds; (void)initial_claim;
    (void)result;
    set_reason(reason, "masked GKR verifier requires external-bound terminal spec");
    return false;
}

bool native_masked_gkr_fixed_u_quadratic_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const std::vector<F> &gkr_coefficients,
    const std::vector<F> &masked_u, const std::vector<F> &unmasked_u,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    (void)session; (void)label; (void)round_polys; (void)initial_claim;
    (void)terminal_values; (void)result;
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::GkrPredicateFixedU;
    spec.gkr_coefficients = gkr_coefficients;
    spec.fixed_values = masked_u;
    spec.unmasked_fixed_values = unmasked_u;
    set_reason(reason, "masked fixed-U GKR requires an external-bound terminal spec");
    return false;
}

bool native_masked_gkr_fixed_u_quadratic_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim,
    const std::vector<F> &gkr_coefficients,
    const std::vector<F> &masked_u,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    (void)session; (void)label; (void)rounds; (void)initial_claim;
    (void)masked_u; (void)result; (void)gkr_coefficients;
    set_reason(reason, "masked fixed-U GKR verifier requires an external-bound terminal spec");
    return false;
}

bool native_masked_gkr_fixed_v_quadratic_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values,
    const std::vector<F> &gkr_coefficients,
    const std::vector<F> &masked_v, const std::vector<F> &unmasked_v,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    (void)session; (void)label; (void)round_polys; (void)initial_claim;
    (void)terminal_values; (void)result; (void)gkr_coefficients;
    (void)masked_v; (void)unmasked_v;
    set_reason(reason, "masked fixed-V GKR requires an external-bound terminal spec");
    return false;
}

bool native_masked_gkr_fixed_v_quadratic_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim,
    const std::vector<F> &gkr_coefficients,
    const std::vector<F> &masked_v,
    NativeMaskedSumcheckResult *result, std::string *reason) {
    (void)session; (void)label; (void)rounds; (void)initial_claim;
    (void)gkr_coefficients; (void)masked_v; (void)result;
    set_reason(reason, "masked fixed-V GKR verifier requires an external-bound terminal spec");
    return false;
}

bool native_masked_matmul_product_prove(
    NativeProofSession &session, const std::string &label,
    const std::vector<std::vector<F>> &round_polys, const F &initial_claim,
    const std::vector<F> &terminal_values, NativeMaskedSumcheckResult *result,
    std::string *reason) {
    (void)session; (void)label; (void)round_polys; (void)initial_claim;
    (void)terminal_values; (void)result;
    set_reason(reason, "masked MatMul requires an external-bound terminal spec");
    return false;
}

bool native_masked_matmul_product_verify(
    const NativeProofSession &session, const std::string &label,
    std::size_t rounds, const F &initial_claim, NativeMaskedSumcheckResult *result,
    std::string *reason) {
    (void)session; (void)label; (void)rounds; (void)initial_claim; (void)result;
    set_reason(reason, "masked MatMul verifier requires an external-bound terminal spec");
    return false;
}

namespace {

bool finish_state(NativeMaskedSumcheckProverState &state,
                  const std::vector<F> &unmasked_terminal,
                  NativeMaskedSumcheckResult *result, std::string *reason) {
    std::size_t arity = 0;
    if (!terminal_shape(state.terminal_spec, state.degree, arity, reason) ||
        unmasked_terminal.size() != arity)
        return false;
    if (!external_shape(state.terminal_spec, state.basis, arity, true, reason))
        return false;
    NativeMaskedTerminalSpec masked_spec = state.terminal_spec;
    if (masked_spec.kind == NativeMaskedTerminalKind::Product2Public)
        masked_spec.public_evaluation = mle_eval(masked_spec.public_vector,
                                                 state.result.challenges);
    if (!check_terminal(state.terminal_spec, state.unmasked_running,
                        unmasked_terminal, true, reason)) return false;
    if (!sample_terminal(masked_spec, state.masked_running,
                         state.result.terminal_values, reason)) return false;
    masked_spec.unmasked_fixed_values.clear();
    if (!check_terminal(masked_spec, state.masked_running,
                        state.result.terminal_values, false, reason)) return false;

    const bool fixed = masked_spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU ||
                       masked_spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV;
    state.terminal_spec.public_evaluation = masked_spec.public_evaluation;
    std::vector<ScalarOpening> terminal_openings;
    terminal_openings.reserve(arity);
    for (std::size_t i = 0; i < arity; ++i) {
        const ScalarOpening *provided = state.terminal_spec.terminal_openings.empty()
            ? nullptr : &state.terminal_spec.terminal_openings[i];
        ScalarOpening opening;
        if (!make_terminal_opening(state.basis, provided,
                                   state.result.terminal_values[i], opening,
                                   "masked terminal", reason)) return false;
        terminal_openings.push_back(opening);
    }
    std::vector<ScalarOpening> unmasked_terminal_openings =
        state.terminal_spec.unmasked_terminal_openings;
    for (std::size_t i = 0; i < arity; ++i)
        if (!relation_opening(unmasked_terminal_openings[i], state.basis,
                              unmasked_terminal[i], reason))
            return false;
    std::vector<ScalarOpening> fixed_openings;
    if (fixed) {
        fixed_openings.reserve(2);
        for (std::size_t i = 0; i < 2; ++i) {
            const ScalarOpening *provided = state.terminal_spec.fixed_openings.empty()
                ? nullptr : &state.terminal_spec.fixed_openings[i];
            ScalarOpening opening;
            if (!make_terminal_opening(state.basis, provided,
                                       state.terminal_spec.fixed_values[i], opening,
                                       "masked fixed", reason)) return false;
            fixed_openings.push_back(opening);
        }
    }
    std::vector<ScalarOpening> unmasked_fixed_openings =
        state.terminal_spec.unmasked_fixed_openings;
    for (std::size_t i = 0; i < unmasked_fixed_openings.size(); ++i)
        if (!relation_opening(unmasked_fixed_openings[i], state.basis,
                              state.terminal_spec.unmasked_fixed_values[i], reason))
            return false;
    // The terminal relation is itself an authenticated claim transition.  It
    // must be represented by a fresh actual opening before the relation proof
    // is emitted; callers use this returned commitment as the next scope's
    // external initial binding.
    state.result.unmasked_claim_opening =
        zkgpt::native_pedersen::commit_fresh(state.basis,
                                              state.unmasked_running);
    std::vector<ScalarOpening> operands;
    if (!prove_terminal_relation(state, state.result.terminal_values,
                                 terminal_openings, fixed_openings,
                                 unmasked_terminal_openings,
                                 unmasked_fixed_openings, operands, reason))
        return false;
    state.result.terminal_commitments.clear();
    state.result.fixed_commitments.clear();
    for (const ScalarOpening &opening : terminal_openings)
        state.result.terminal_commitments.push_back(opening.commitment);
    for (const ScalarOpening &opening : fixed_openings)
        state.result.fixed_commitments.push_back(opening.commitment);
    for (const ScalarOpening &opening : unmasked_terminal_openings)
        state.result.unmasked_terminal_commitments.push_back(opening.commitment);
    for (const ScalarOpening &opening : unmasked_fixed_openings)
        state.result.unmasked_fixed_commitments.push_back(opening.commitment);
    state.result.terminal_openings = terminal_openings;
    state.result.fixed_openings = fixed_openings;
    state.result.unmasked_terminal_openings = unmasked_terminal_openings;
    state.result.unmasked_fixed_openings = unmasked_fixed_openings;
    for (const ScalarOpening &opening : operands)
        state.result.operand_commitments.push_back(opening.commitment);
    if (!state.terminal_spec.terminal_commitments.empty()) {
        for (std::size_t i = 0; i < terminal_openings.size(); ++i)
            if (!equal_commitment(state.terminal_spec.terminal_commitments[i],
                                  terminal_openings[i].commitment)) {
                set_reason(reason, "masked terminal external commitment mismatch");
                return false;
            }
    }
    if (!state.terminal_spec.fixed_commitments.empty()) {
        for (std::size_t i = 0; i < fixed_openings.size(); ++i)
            if (!equal_commitment(state.terminal_spec.fixed_commitments[i],
                                  fixed_openings[i].commitment)) {
                set_reason(reason, "masked fixed external commitment mismatch");
                return false;
            }
    }
    state.result.final_claim = state.masked_running;
    state.result.basis = state.basis;
    state.result.external_binding_required = true;
    state.result.external_binding_verified = state.terminal_spec.has_external_binding;
    state.finished = true;
    if (result != nullptr) *result = state.result;
    return true;
}

bool verify_initial_scope(const NativeProofSession &session,
                          const PedersenBasis &basis, const std::string &label,
                          const F &initial_claim,
                          const NativeMaskedTerminalSpec &spec,
                          std::size_t &cursor, bool &has_mask,
                          ScalarCommitment &previous_mask,
                          NativeMaskedSumcheckResult &result,
                          std::string *reason) {
    std::size_t mode_index = 0;
    if (!unique_index(session, label + "/initial/mode", mode_index, reason)) return false;
    cursor = mode_index;
    const NativeProofMessage *mode = nullptr;
    if (!expect_message(session, cursor, NativeProofMessageKind::Data,
                        label + "/initial/mode", &mode, reason)) return false;
    const bool mode_with_mask = mode->payload == "with-mask-v2";
    if ((mode->payload != "zero-mask-v2" && mode->payload != "with-mask-v2") ||
        mode_with_mask == spec.public_initial_claim) {
        set_reason(reason, "masked Sumcheck initial mode/tag does not match scope binding");
        return false;
    }
    has_mask = mode_with_mask;
    ScalarCommitment external;
    if (!verify_opening_block(session, basis, cursor,
                              label + "/initial/opening", external, reason) ||
        !equal_commitment(external, spec.initial_commitment)) {
        set_reason(reason, "masked Sumcheck initial external commitment mismatch");
        return false;
    }
    result.initial_commitment = external;
    if (!has_mask) {
        if (!verify_linear_block(session, basis, cursor,
                                 label + "/initial/public-bind", {external},
                                 {F_ONE}, initial_claim, reason))
            return false;
    } else {
        ScalarCommitment masked, mask;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/initial/masked", masked, reason))
            return false;
        if (!equal_commitment(masked, spec.masked_initial_commitment)) {
            set_reason(reason, "masked Sumcheck initial masked commitment mismatch");
            return false;
        }
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/initial/mask", mask, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/initial/masked-bind", {masked},
                                 {F_ONE}, initial_claim, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/initial/mask-relation",
                                 {masked, external, mask},
                                 {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason))
            return false;
        previous_mask = mask;
        result.operand_commitments.push_back(masked);
        result.operand_commitments.push_back(mask);
    }
    return true;
}

bool verify_round(const NativeProofSession &session, const PedersenBasis &basis,
                  const std::string &label, std::size_t round,
                  std::size_t degree, const F &running, bool has_previous_mask,
                  const ScalarCommitment &previous_mask, std::size_t &cursor,
                  NativeMaskedRoundBinding &binding, F &challenge_out,
                  F &next_running,
                  ScalarCommitment &next_mask, std::string *reason) {
    const std::string base = label + "/round/" + std::to_string(round);
    const std::size_t width = degree + 1;
    std::vector<ScalarCommitment> p, h, q;
    p.reserve(width); h.reserve(width); q.reserve(width);
    for (std::size_t i = 0; i < width; ++i) {
        ScalarCommitment c;
        if (!verify_opening_block(session, basis, cursor,
                                  base + "/p/" + std::to_string(i), c, reason)) return false;
        p.push_back(c);
    }
    for (std::size_t i = 0; i < width; ++i) {
        ScalarCommitment c;
        if (!verify_opening_block(session, basis, cursor,
                                  base + "/h/" + std::to_string(i), c, reason)) return false;
        h.push_back(c);
    }
    for (std::size_t i = 0; i < width; ++i) {
        ScalarCommitment c;
        if (!verify_opening_block(session, basis, cursor,
                                  base + "/q/" + std::to_string(i), c, reason)) return false;
        q.push_back(c);
    }
    const NativeProofMessage *q_message = nullptr;
    if (!expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                        base + "/q-values", &q_message, reason)) return false;
    std::vector<F> q_values;
    if (!decode_field_vector(*q_message, width, q_values, reason)) return false;
    ScalarCommitment p_claim, h_claim, q_claim;
    if (!verify_opening_block(session, basis, cursor, base + "/p-claim", p_claim, reason) ||
        !verify_opening_block(session, basis, cursor, base + "/h-claim", h_claim, reason) ||
        !verify_opening_block(session, basis, cursor, base + "/q-claim", q_claim, reason))
        return false;
    for (std::size_t i = 0; i < width; ++i)
        if (!verify_linear_block(session, basis, cursor,
                                 base + "/q-public/" + std::to_string(i),
                                 {q[i]}, {F_ONE}, q_values[i], reason)) return false;
    std::vector<ScalarCommitment> q_relation = q;
    q_relation.insert(q_relation.end(), p.begin(), p.end());
    q_relation.insert(q_relation.end(), h.begin(), h.end());
    std::vector<F> q_coeff(width, F_ONE);
    q_coeff.insert(q_coeff.end(), width, -F_ONE);
    q_coeff.insert(q_coeff.end(), width, -F_ONE);
    std::vector<ScalarCommitment> p_bind = p;
    std::vector<ScalarCommitment> h_bind = h;
    std::vector<ScalarCommitment> q_bind = q;
    p_bind.push_back(p_claim);
    h_bind.push_back(h_claim);
    q_bind.push_back(q_claim);
    std::vector<F> claim_coefficients(width, F_ZERO);
    claim_coefficients[0] = F_ONE;
    claim_coefficients[1] = F_ONE;
    claim_coefficients.push_back(-F_ONE);
    if (!verify_linear_block(session, basis, cursor, base + "/q-linear",
                             q_relation, q_coeff, F_ZERO, reason) ||
        !verify_linear_block(session, basis, cursor, base + "/p-claim-bind", p_bind,
                             claim_coefficients, F_ZERO, reason) ||
        !verify_linear_block(session, basis, cursor, base + "/h-claim-bind", h_bind,
                             claim_coefficients, F_ZERO, reason) ||
        !verify_linear_block(session, basis, cursor, base + "/q-claim-bind", q_bind,
                             claim_coefficients, F_ZERO, reason) ||
        !verify_linear_block(session, basis, cursor, base + "/claim-add",
                             {q_claim, p_claim, h_claim}, {F_ONE, -F_ONE, -F_ONE},
                             F_ZERO, reason)) return false;
    if (has_previous_mask) {
        if (!verify_linear_block(session, basis, cursor, base + "/h-transition",
                                 {h_claim, previous_mask}, {F_ONE, -F_ONE},
                                 F_ZERO, reason)) return false;
    } else if (!verify_linear_block(session, basis, cursor, base + "/h-transition",
                                    {h_claim}, {F_ONE}, F_ZERO, reason)) return false;
    if (!verify_linear_block(session, basis, cursor, base + "/transition",
                             {q_claim}, {F_ONE}, running, reason)) return false;
    const NativeProofMessage *q_transcript = nullptr;
    if (!expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                        base + "/q-transcript", &q_transcript, reason)) return false;
    std::vector<F> q_again;
    if (!decode_field_vector(*q_transcript, width, q_again, reason) || q_again != q_values) {
        set_reason(reason, "masked Sumcheck q transcript mismatch");
        return false;
    }
    F challenge;
    if (!decode_challenge_at(session, cursor, label + "/r/" + std::to_string(round),
                             challenge, reason)) return false;
    ++cursor;
    challenge_out = challenge;
    next_running = interpolate_at(q_values, challenge);
    if (!verify_opening_block(session, basis, cursor, base + "/h-eval", next_mask,
                              reason)) return false;
    const std::vector<F> lambdas = lagrange_coefficients(degree, challenge);
    std::vector<ScalarCommitment> h_eval_operands = h;
    h_eval_operands.push_back(next_mask);
    std::vector<F> h_eval_coefficients = lambdas;
    h_eval_coefficients.push_back(-F_ONE);
    if (!verify_linear_block(session, basis, cursor, base + "/h-eval-bind",
                             h_eval_operands, h_eval_coefficients, F_ZERO, reason))
        return false;
    binding.p_coefficients = p;
    binding.h_coefficients = h;
    binding.q_coefficients = q;
    binding.p_claim = p_claim;
    binding.h_claim = h_claim;
    binding.q_claim = q_claim;
    binding.h_evaluation = next_mask;
    return true;
}

} // namespace

namespace {

bool verify_unmasked_terminal_relation(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, const NativeMaskedTerminalSpec &spec,
    std::size_t &cursor, NativeMaskedSumcheckResult *result,
    std::string *reason) {
    std::size_t arity = 0;
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar: arity = 1; break;
    case NativeMaskedTerminalKind::Product2: arity = 2; break;
    case NativeMaskedTerminalKind::Product3: arity = 3; break;
    case NativeMaskedTerminalKind::Claims: arity = 2; break;
    case NativeMaskedTerminalKind::GkrPredicate: arity = 4; break;
    case NativeMaskedTerminalKind::Product2Public: arity = 1; break;
    case NativeMaskedTerminalKind::GkrPredicateFixedU:
    case NativeMaskedTerminalKind::GkrPredicateFixedV: arity = 2; break;
    default:
        set_reason(reason, "unknown masked unmasked terminal relation");
        return false;
    }
    const bool fixed = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU ||
                       spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV;
    const std::size_t fixed_arity = fixed ? 2 : 0;
    ScalarCommitment claim;
    if (!verify_opening_block(session, basis, cursor,
                              label + "/terminal/unmasked/claim", claim, reason))
        return false;
    std::vector<ScalarCommitment> values, fixed_values;
    for (std::size_t i = 0; i < arity; ++i) {
        ScalarCommitment opening;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/terminal/unmasked/value/" +
                                      std::to_string(i), opening, reason)) return false;
        values.push_back(opening);
    }
    for (std::size_t i = 0; i < fixed_arity; ++i) {
        ScalarCommitment opening;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/terminal/unmasked/fixed/" +
                                      std::to_string(i), opening, reason)) return false;
        fixed_values.push_back(opening);
    }
    if (result != nullptr) {
        result->unmasked_claim_commitment = claim;
        result->unmasked_terminal_commitments = values;
        result->unmasked_fixed_commitments = fixed_values;
    }
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar:
        return verify_linear_block(session, basis, cursor,
                                   label + "/terminal/unmasked/scalar",
                                   {values[0], claim}, {F_ONE, -F_ONE},
                                   F_ZERO, reason);
    case NativeMaskedTerminalKind::Claims:
        {
            std::vector<ScalarCommitment> relation_operands = values;
            relation_operands.push_back(claim);
            std::vector<F> coefficients = spec.terminal_linear_coefficients;
            coefficients.push_back(-F_ONE);
            return verify_linear_block(session, basis, cursor,
                                       label + "/terminal/unmasked/claims",
                                       relation_operands, coefficients,
                                       spec.terminal_linear_offset, reason);
        }
    case NativeMaskedTerminalKind::Product2: {
        const MultiplicationLinks links{
            label + "/terminal/unmasked/value/0",
            label + "/terminal/unmasked/value/1",
            label + "/terminal/unmasked/claim"};
        return zkgpt::native_pedersen::verify_multiplication(
                   session, basis, label + "/terminal/unmasked/product2",
                   values[0], values[1], claim, links, reason) &&
               consume_multiplication_block(
                   session, cursor, label + "/terminal/unmasked/product2", reason);
    }
    case NativeMaskedTerminalKind::Product2Public:
        return verify_linear_block(session, basis, cursor,
                                   label + "/terminal/unmasked/public-product",
                                   {values[0], claim},
                                   {spec.public_evaluation, -F_ONE}, F_ZERO, reason);
    case NativeMaskedTerminalKind::GkrPredicate:
    case NativeMaskedTerminalKind::GkrPredicateFixedU:
    case NativeMaskedTerminalKind::GkrPredicateFixedV: {
        const bool fixed_u = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU;
        const bool fixed_v = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV;
        std::vector<ScalarCommitment> products;
        const auto value_label = [&](std::size_t i) {
            return label + "/terminal/unmasked/value/" + std::to_string(i);
        };
        const auto fixed_label = [&](std::size_t i) {
            return label + "/terminal/unmasked/fixed/" + std::to_string(i);
        };
        const std::string product_prefix = label + "/terminal/unmasked/gkr-product/";
        const auto verify_product = [&](std::size_t index,
                                        const ScalarCommitment &a,
                                        const ScalarCommitment &b,
                                        const std::string &a_label,
                                        const std::string &b_label) -> bool {
            ScalarCommitment product;
            if (!verify_opening_block(session, basis, cursor,
                                      product_prefix + std::to_string(index),
                                      product, reason)) return false;
            const MultiplicationLinks links{a_label, b_label,
                                            product_prefix + std::to_string(index)};
            if (!zkgpt::native_pedersen::verify_multiplication(
                    session, basis,
                    label + "/terminal/unmasked/gkr-mul/" + std::to_string(index),
                    a, b, product, links, reason) ||
                !consume_multiplication_block(
                    session, cursor,
                    label + "/terminal/unmasked/gkr-mul/" + std::to_string(index),
                    reason)) return false;
            products.push_back(product);
            return true;
        };
        if (fixed_u) {
            if (!verify_product(0, fixed_values[0], values[0], fixed_label(0),
                                value_label(0)) ||
                !verify_product(1, fixed_values[1], values[1], fixed_label(1),
                                value_label(1)) ||
                !verify_product(2, fixed_values[1], values[0], fixed_label(1),
                                value_label(0))) return false;
        } else if (fixed_v) {
            if (!verify_product(0, values[0], fixed_values[0], value_label(0),
                                fixed_label(0)) ||
                !verify_product(1, values[1], fixed_values[1], value_label(1),
                                fixed_label(1)) ||
                !verify_product(2, values[1], fixed_values[0], value_label(1),
                                fixed_label(0))) return false;
        } else {
            if (!verify_product(0, values[0], values[2], value_label(0),
                                value_label(2)) ||
                !verify_product(1, values[1], values[3], value_label(1),
                                value_label(3)) ||
                !verify_product(2, values[1], values[2], value_label(1),
                                value_label(2))) return false;
        }
        std::vector<ScalarCommitment> relation_operands = products;
        if (fixed_u) relation_operands.insert(relation_operands.end(), fixed_values.begin(),
                                              fixed_values.end());
        else relation_operands.insert(relation_operands.end(), values.begin(),
                                      values.begin() + 2);
        relation_operands.push_back(claim);
        std::vector<F> coefficients{spec.gkr_coefficients[0],
                                    spec.gkr_coefficients[1],
                                    spec.gkr_coefficients[2],
                                    spec.gkr_coefficients[3],
                                    spec.gkr_coefficients[4], -F_ONE};
        return verify_linear_block(session, basis, cursor,
                                   label + "/terminal/unmasked/gkr-linear",
                                   relation_operands, coefficients, F_ZERO, reason);
    }
    default:
        return false;
    }
}

bool verify_terminal_relation(const NativeProofSession &session,
                              const PedersenBasis &basis,
                              const std::string &label,
                              const NativeMaskedTerminalSpec &spec,
                              const F &running,
                              const std::vector<F> &terminal_values,
                              std::size_t &cursor,
                              NativeMaskedSumcheckResult *result,
                              std::string *reason) {
    const bool fixed = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU ||
                       spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV;
    const std::size_t fixed_arity = fixed ? 2 : 0;
    std::vector<ScalarCommitment> fixed_commitments;
    std::vector<ScalarCommitment> terminal_commitments;
    for (std::size_t i = 0; i < fixed_arity; ++i) {
        ScalarCommitment commitment;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/terminal/fixed/" + std::to_string(i),
                                  commitment, reason)) return false;
        fixed_commitments.push_back(commitment);
    }
    const std::size_t arity = terminal_values.size();
    for (std::size_t i = 0; i < arity; ++i) {
        ScalarCommitment commitment;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/terminal/value/" + std::to_string(i),
                                  commitment, reason)) return false;
        terminal_commitments.push_back(commitment);
    }
    if (!spec.terminal_commitments.empty()) {
        if (spec.terminal_commitments.size() != terminal_commitments.size()) {
            set_reason(reason, "masked terminal commitment arity mismatch");
            return false;
        }
        for (std::size_t i = 0; i < terminal_commitments.size(); ++i)
            if (!equal_commitment(spec.terminal_commitments[i], terminal_commitments[i])) {
                set_reason(reason, "masked terminal commitment binding mismatch");
                return false;
            }
    }
        if (!spec.fixed_commitments.empty()) {
            if (spec.fixed_commitments.size() != fixed_commitments.size()) {
                set_reason(reason, "masked fixed commitment arity mismatch");
            return false;
        }
        for (std::size_t i = 0; i < fixed_commitments.size(); ++i)
            if (!equal_commitment(spec.fixed_commitments[i], fixed_commitments[i])) {
                set_reason(reason, "masked fixed commitment binding mismatch");
                return false;
            }
    }
    if (result != nullptr) {
        result->terminal_commitments = terminal_commitments;
        result->fixed_commitments = fixed_commitments;
    }

    // The terminal field vector is public transcript data.  Bind every
    // decoded value to the corresponding Pedersen commitment before any
    // terminal relation is checked.  A commitment opening alone proves only
    // knowledge of some scalar; this affine hidden-linear relation is what
    // makes it the exact public scalar carried by /terminal/values.
    for (std::size_t i = 0; i < arity; ++i)
        if (!verify_linear_block(session, basis, cursor,
                                 label + "/terminal/value/public-bind/" +
                                     std::to_string(i),
                                 {terminal_commitments[i]}, {F_ONE},
                                 terminal_values[i], reason))
            return false;
    // Fixed-side terminals are public relation operands as well.  They must
    // be externally supplied commitments and bound to the verifier's fixed
    // values; a plaintext check of spec.fixed_values is insufficient.
    if (fixed_arity != 0 && spec.fixed_values.size() != fixed_arity) {
        set_reason(reason, "masked fixed terminal public-value arity mismatch");
        return false;
    }
    for (std::size_t i = 0; i < fixed_arity; ++i)
        if (!verify_linear_block(session, basis, cursor,
                                 label + "/terminal/fixed/public-bind/" +
                                     std::to_string(i),
                                 {fixed_commitments[i]}, {F_ONE},
                                 spec.fixed_values[i], reason))
            return false;

    if (result == nullptr || result->unmasked_terminal_commitments.size() != arity ||
        result->unmasked_fixed_commitments.size() != fixed_arity) {
        set_reason(reason, "masked terminal unmasked commitment binding is missing");
        return false;
    }
    for (std::size_t i = 0; i < arity; ++i) {
        ScalarCommitment mask;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/terminal/mask/" + std::to_string(i),
                                  mask, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/mask-bind/" +
                                     std::to_string(i),
                                 {terminal_commitments[i],
                                  result->unmasked_terminal_commitments[i], mask},
                                 {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason))
            return false;
    }
    for (std::size_t i = 0; i < fixed_arity; ++i) {
        ScalarCommitment mask;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/terminal/fixed-mask/" +
                                      std::to_string(i), mask, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/fixed-mask-bind/" +
                                     std::to_string(i),
                                 {fixed_commitments[i],
                                  result->unmasked_fixed_commitments[i], mask},
                                 {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason))
            return false;
    }

    const auto verify_claim_binding = [&](const ScalarCommitment &anchor) {
        ScalarCommitment mask;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/terminal/claim-mask", mask, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/claim-bind",
                                 {anchor, result->unmasked_claim_commitment, mask},
                                 {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason))
            return false;
        result->claim_commitment = anchor;
        result->claim_mask_commitment = mask;
        return true;
    };

    const auto operand = [&](const std::string &opening_label,
                             ScalarCommitment &commitment) {
        return opening_commitment_at(session, basis, 0, opening_label,
                                     commitment, reason);
    };
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar: {
        ScalarCommitment anchor;
        if (!verify_opening_block(session, basis, cursor, label + "/terminal/anchor",
                                  anchor, reason) ||
            !verify_linear_block(session, basis, cursor, label + "/terminal/scalar",
                                 terminal_commitments, {F_ONE}, running, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/anchor-bind", {anchor}, {F_ONE},
                                 running, reason) ||
            !verify_claim_binding(anchor)) return false;
        if (result != nullptr) result->operand_commitments.push_back(anchor);
        break;
    }
    case NativeMaskedTerminalKind::Claims: {
        ScalarCommitment anchor;
        if (!verify_opening_block(session, basis, cursor, label + "/terminal/anchor",
                                  anchor, reason) ||
            !verify_linear_block(session, basis, cursor, label + "/terminal/claims",
                                 terminal_commitments,
                                 spec.terminal_linear_coefficients,
                                 running - spec.terminal_linear_offset, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/anchor-bind", {anchor}, {F_ONE},
                                 running, reason) ||
            !verify_claim_binding(anchor)) return false;
        if (result != nullptr) result->operand_commitments.push_back(anchor);
        break;
    }
    case NativeMaskedTerminalKind::Product2: {
        ScalarCommitment anchor;
        if (!verify_opening_block(session, basis, cursor, label + "/terminal/anchor",
                                  anchor, reason)) return false;
        if (result != nullptr) result->operand_commitments.push_back(anchor);
        const MultiplicationLinks links{
            label + "/terminal/value/0", label + "/terminal/value/1",
            label + "/terminal/anchor"};
        if (!zkgpt::native_pedersen::verify_multiplication(
                session, basis, label + "/terminal/product2", terminal_commitments[0],
                terminal_commitments[1], anchor, links, reason) ||
            !consume_multiplication_block(session, cursor,
                                          label + "/terminal/product2", reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/anchor-bind", {anchor},
                                 {F_ONE}, running, reason) ||
            !verify_claim_binding(anchor)) return false;
        break;
    }
    case NativeMaskedTerminalKind::Product3: {
        ScalarCommitment intermediate, anchor;
        if (!verify_opening_block(session, basis, cursor,
                                  label + "/terminal/intermediate", intermediate,
                                  reason) ||
            !verify_opening_block(session, basis, cursor, label + "/terminal/anchor",
                                  anchor, reason)) return false;
        if (result != nullptr) {
            result->operand_commitments.push_back(intermediate);
            result->operand_commitments.push_back(anchor);
        }
        const Product3Links links{
            {label + "/terminal/value/0", label + "/terminal/value/1",
             label + "/terminal/intermediate"},
            {label + "/terminal/intermediate", label + "/terminal/value/2",
             label + "/terminal/anchor"}};
        if (!zkgpt::native_pedersen::verify_product3(
                session, basis, label + "/terminal/product3", terminal_commitments[0],
                terminal_commitments[1], terminal_commitments[2], intermediate, anchor,
                links, reason) ||
            !consume_product3_block(session, cursor,
                                     label + "/terminal/product3", reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/anchor-bind", {anchor},
                                 {F_ONE}, running, reason) ||
            !verify_claim_binding(anchor)) return false;
        break;
    }
    case NativeMaskedTerminalKind::Product2Public: {
        ScalarCommitment anchor;
        if (!verify_opening_block(session, basis, cursor, label + "/terminal/anchor",
                                  anchor, reason)) return false;
        if (result != nullptr) result->operand_commitments.push_back(anchor);
        if (!verify_linear_block(session, basis, cursor,
                                 label + "/terminal/public-product",
                                 {terminal_commitments[0], anchor},
                                 {spec.public_evaluation, -F_ONE}, F_ZERO, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/anchor-bind", {anchor},
                                 {F_ONE}, running, reason) ||
            !verify_claim_binding(anchor)) return false;
        break;
    }
    case NativeMaskedTerminalKind::GkrPredicate:
    case NativeMaskedTerminalKind::GkrPredicateFixedU:
    case NativeMaskedTerminalKind::GkrPredicateFixedV: {
        const bool fixed_u = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU;
        const bool fixed_v = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV;
        std::vector<ScalarCommitment> products;
        for (std::size_t i = 0; i < 3; ++i) {
            ScalarCommitment product;
            if (!verify_opening_block(session, basis, cursor,
                                      label + "/terminal/gkr-product/" +
                                          std::to_string(i), product, reason)) return false;
            products.push_back(product);
            const std::string a_label = fixed_u
                ? label + "/terminal/fixed/" + std::to_string(i == 0 ? 0 : 1)
                : (fixed_v ? label + "/terminal/value/" +
                                  std::to_string(i == 0 ? 0 : 1)
                           : label + "/terminal/value/" +
                                  std::to_string(i == 0 ? 0 : 1));
            const std::string b_label = fixed_u
                ? label + "/terminal/value/" +
                      std::to_string(i == 0 ? 0 : (i == 1 ? 1 : 0))
                : (fixed_v ? label + "/terminal/fixed/" +
                                  std::to_string(i == 0 ? 0 : (i == 1 ? 1 : 0))
                           : label + "/terminal/value/" +
                                  std::to_string(i == 0 ? 2 : (i == 1 ? 3 : 2)));
            ScalarCommitment a, b;
            if (!opening_commitment_at(session, basis, 0, a_label, a, reason) ||
                !opening_commitment_at(session, basis, 0, b_label, b, reason)) return false;
            const MultiplicationLinks links{a_label, b_label,
                                            label + "/terminal/gkr-product/" +
                                                std::to_string(i)};
            if (!zkgpt::native_pedersen::verify_multiplication(
                    session, basis, label + "/terminal/gkr-mul/" + std::to_string(i),
                    a, b, product, links, reason) ||
                !consume_multiplication_block(
                    session, cursor,
                    label + "/terminal/gkr-mul/" + std::to_string(i), reason))
                return false;
        }
        ScalarCommitment anchor;
        if (!verify_opening_block(session, basis, cursor, label + "/terminal/anchor",
                                  anchor, reason)) return false;
        if (result != nullptr) result->operand_commitments.push_back(anchor);
        std::vector<ScalarCommitment> relation_operands = products;
        if (fixed_u) relation_operands.insert(relation_operands.end(), fixed_commitments.begin(),
                                              fixed_commitments.end());
        else relation_operands.insert(relation_operands.end(), terminal_commitments.begin(),
                                      terminal_commitments.begin() + 2);
        relation_operands.push_back(anchor);
        std::vector<F> coefficients{spec.gkr_coefficients[0],
                                    spec.gkr_coefficients[1],
                                    spec.gkr_coefficients[2],
                                    spec.gkr_coefficients[3],
                                    spec.gkr_coefficients[4], -F_ONE};
        if (!verify_linear_block(session, basis, cursor,
                                 label + "/terminal/gkr-linear", relation_operands,
                                 coefficients, F_ZERO, reason) ||
            !verify_linear_block(session, basis, cursor,
                                 label + "/terminal/anchor-bind", {anchor},
                                 {F_ONE}, running, reason) ||
            !verify_claim_binding(anchor)) return false;
        break;
    }
    default:
        set_reason(reason, "unknown masked terminal relation");
        return false;
    }
    return true;
}

} // namespace

namespace {

using zkgpt::native_pedersen::PedersenBasis;
using zkgpt::native_pedersen::ScalarCommitment;
using zkgpt::native_pedersen::ScalarOpening;
using zkgpt::native_pedersen::MultiplicationLinks;
using zkgpt::native_pedersen::Product3Links;

std::string field_hex(const F &value) {
    const std::string encoded = value.getStr(16);
    if (encoded.empty() || encoded.size() > 128)
        throw std::runtime_error("masked Sumcheck field encoding is invalid");
    return encoded;
}

bool decode_field(const std::string &encoded, F &value,
                  std::string *reason = nullptr) {
    if (encoded.empty() || encoded.size() > 128) {
        set_reason(reason, "masked Sumcheck field encoding is invalid");
        return false;
    }
    try {
        value.setStr(encoded, 16);
        if (field_hex(value) != encoded) {
            set_reason(reason, "masked Sumcheck field encoding is not canonical");
            return false;
        }
    } catch (...) {
        set_reason(reason, "masked Sumcheck field decoding failed");
        return false;
    }
    return true;
}

bool decode_field_vector(const NativeProofMessage &message, std::size_t expected,
                         std::vector<F> &values, std::string *reason) {
    if (message.kind != NativeProofMessageKind::FieldVector) {
        set_reason(reason, "masked Sumcheck field vector has wrong kind");
        return false;
    }
    Reader reader{message.payload};
    std::uint64_t count = 0;
    if (!reader.u64(count) || count != expected || count > kMaxRounds) {
        set_reason(reason, "masked Sumcheck field vector arity mismatch");
        return false;
    }
    values.clear();
    values.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        std::string encoded;
        F value;
        if (!reader.string(encoded, 128) ||
            !decode_field(encoded, value, reason)) {
            set_reason(reason, "malformed masked Sumcheck field vector");
            return false;
        }
        values.push_back(value);
    }
    if (!reader.done()) {
        set_reason(reason, "masked Sumcheck field vector has trailing bytes");
        return false;
    }
    return true;
}

bool decode_field_message(const NativeProofMessage &message,
                          const std::string &label, F &value,
                          std::string *reason) {
    if (message.kind != NativeProofMessageKind::Field ||
        message.label != label) {
        set_reason(reason, "masked Sumcheck field message is invalid");
        return false;
    }
    return decode_field(message.payload, value, reason);
}

bool parse_point(const std::string &encoded, G1 &point,
                 std::string *reason = nullptr) {
    if (encoded.empty() || encoded.size() > 4096) {
        set_reason(reason, "masked Sumcheck point encoding is invalid");
        return false;
    }
    try {
        point.deserializeHexStr(encoded);
        if (point.serializeToHexStr() != encoded || !point.isValid() ||
            !point.isValidOrder()) {
            set_reason(reason, "masked Sumcheck point encoding is not canonical");
            return false;
        }
    } catch (...) {
        set_reason(reason, "masked Sumcheck point decoding failed");
        return false;
    }
    return true;
}

bool equal_commitment(const ScalarCommitment &left,
                      const ScalarCommitment &right) {
    return left.domain == right.domain &&
           left.point.serializeToHexStr() == right.point.serializeToHexStr();
}

// native_pedersen deliberately keeps its metadata parser private.  The
// masked Sumcheck verifier only needs the commitment carried by an opening;
// decode the canonical header here and then delegate all cryptographic checks
// back to native_pedersen::verify_opening.
bool opening_commitment_at(const NativeProofSession &session,
                           const PedersenBasis &basis, std::size_t index,
                           const std::string &label, ScalarCommitment &out,
                           std::string *reason) {
    // `index == 0` is the historical compositional lookup sentinel used by
    // terminal multiplication links; all cursor-based callers pass a
    // nonzero transcript position and retain strict order checking.
    if (index == 0) {
        bool found = false;
        for (std::size_t candidate = 0; candidate < session.messages().size();
             ++candidate) {
            if (session.messages()[candidate].label != label + "/meta") continue;
            if (found) {
                set_reason(reason, "masked Sumcheck opening metadata is duplicated");
                return false;
            }
            index = candidate;
            found = true;
        }
        if (!found) {
            set_reason(reason, "masked Sumcheck opening metadata is missing");
            return false;
        }
    }
    if (index >= session.messages().size()) {
        set_reason(reason, "masked Sumcheck opening metadata is missing");
        return false;
    }
    const NativeProofMessage &message = session.messages()[index];
    if (message.kind != NativeProofMessageKind::Data ||
        message.label != label + "/meta") {
        set_reason(reason, "masked Sumcheck opening metadata is out of order: expected " +
                             label + "/meta, got " + message.label);
        return false;
    }
    Reader reader{message.payload};
    std::string magic, kind, domain, g_encoded, h_encoded, c_encoded;
    G1 G, H, point;
    if (!reader.string(magic, 128) || magic != "zkgpt-native-pedersen-relation-v1" ||
        !reader.string(kind, 128) || kind != "opening" ||
        !reader.string(domain, 256) || domain != basis.domain ||
        !reader.string(g_encoded, 4096) || !reader.string(h_encoded, 4096) ||
        !reader.string(c_encoded, 4096) || !reader.done() ||
        !parse_point(g_encoded, G, reason) || !parse_point(h_encoded, H, reason) ||
        !parse_point(c_encoded, point, reason) ||
        G.serializeToHexStr() != basis.G.serializeToHexStr() ||
        H.serializeToHexStr() != basis.H.serializeToHexStr()) {
        set_reason(reason, "masked Sumcheck opening metadata is malformed");
        return false;
    }
    out.domain = basis.domain;
    out.point = point;
    return out.valid(basis, reason);
}

bool unique_index(const NativeProofSession &session, const std::string &label,
                  std::size_t &index, std::string *reason) {
    bool found = false;
    for (std::size_t i = 0; i < session.messages().size(); ++i) {
        if (session.messages()[i].label != label) continue;
        if (found) {
            set_reason(reason, "masked Sumcheck transcript contains duplicate labels");
            return false;
        }
        found = true;
        index = i;
    }
    if (!found) {
        set_reason(reason, "masked Sumcheck transcript message is missing: " + label);
        return false;
    }
    return true;
}

bool expect_message(const NativeProofSession &session, std::size_t &cursor,
                    NativeProofMessageKind kind, const std::string &label,
                    const NativeProofMessage **message, std::string *reason) {
    if (cursor >= session.messages().size() ||
        session.messages()[cursor].kind != kind ||
        session.messages()[cursor].label != label) {
        set_reason(reason, "masked Sumcheck transcript expected " + label);
        return false;
    }
    if (message != nullptr) *message = &session.messages()[cursor];
    ++cursor;
    return true;
}

bool consume_multiplication_block(const NativeProofSession &session,
                                  std::size_t &cursor, const std::string &label,
                                  std::string *reason) {
    return expect_message(session, cursor, NativeProofMessageKind::Data,
                          label + "/meta", nullptr, reason) &&
           expect_message(session, cursor, NativeProofMessageKind::Data,
                          label + "/T1", nullptr, reason) &&
           expect_message(session, cursor, NativeProofMessageKind::Data,
                          label + "/T2", nullptr, reason) &&
           expect_message(session, cursor, NativeProofMessageKind::Challenge,
                          "challenge/" + label + "/challenge", nullptr, reason) &&
           expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                          label + "/response", nullptr, reason);
}

bool consume_product3_block(const NativeProofSession &session, std::size_t &cursor,
                            const std::string &label, std::string *reason) {
    return expect_message(session, cursor, NativeProofMessageKind::Data,
                          label + "/meta", nullptr, reason) &&
           consume_multiplication_block(session, cursor, label + "/mul1", reason) &&
           consume_multiplication_block(session, cursor, label + "/mul2", reason);
}

bool decode_challenge_at(const NativeProofSession &session, std::size_t index,
                         const std::string &label, F &challenge,
                         std::string *reason) {
    if (index >= session.messages().size() ||
        session.messages()[index].kind != NativeProofMessageKind::Challenge ||
        session.messages()[index].label != "challenge/" + label ||
        !decode_field(session.messages()[index].payload, challenge, reason)) {
        set_reason(reason, "masked Sumcheck round challenge is malformed");
        return false;
    }
    return true;
}

bool verify_opening_block(const NativeProofSession &session,
                          const PedersenBasis &basis, std::size_t &cursor,
                          const std::string &label, ScalarCommitment &commitment,
                          std::string *reason) {
    const std::size_t start = cursor;
    if (!opening_commitment_at(session, basis, start, label, commitment, reason))
        return false;
    if (!expect_message(session, cursor, NativeProofMessageKind::Data,
                        label + "/meta", nullptr, reason) ||
        !expect_message(session, cursor, NativeProofMessageKind::Data,
                        label + "/T", nullptr, reason) ||
        !expect_message(session, cursor, NativeProofMessageKind::Challenge,
                        "challenge/" + label + "/challenge", nullptr, reason) ||
        !expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                        label + "/response", nullptr, reason))
        return false;
    if (!zkgpt::native_pedersen::verify_opening(session, basis, label,
                                                commitment, reason))
        return false;
    return cursor == start + 4;
}

bool verify_linear_block(const NativeProofSession &session,
                         const PedersenBasis &basis, std::size_t &cursor,
                         const std::string &label,
                         const std::vector<ScalarCommitment> &operands,
                         const std::vector<F> &coefficients,
                         const F &public_value, std::string *reason) {
    const std::size_t start = cursor;
    if (!expect_message(session, cursor, NativeProofMessageKind::Data,
                        label + "/meta", nullptr, reason) ||
        !expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                        label + "/coefficients", nullptr, reason) ||
        !expect_message(session, cursor, NativeProofMessageKind::Field,
                        label + "/public", nullptr, reason) ||
        !expect_message(session, cursor, NativeProofMessageKind::Data,
                        label + "/T", nullptr, reason) ||
        !expect_message(session, cursor, NativeProofMessageKind::Challenge,
                        "challenge/" + label + "/challenge", nullptr, reason) ||
        !expect_message(session, cursor, NativeProofMessageKind::Field,
                        label + "/response", nullptr, reason))
        return false;
    std::string relation_reason;
    if (!zkgpt::native_pedersen::verify_hidden_linear(
        session, basis, label, operands, coefficients, public_value,
        &relation_reason)) {
        set_reason(reason, "masked Sumcheck linear relation " + label + ": " +
                             relation_reason);
        return false;
    }
    return cursor == start + 6;
}

bool gkr_value(const std::vector<F> &coefficients,
               const std::vector<F> &values, std::string *reason) {
    if (coefficients.size() != 5 || values.size() != 4) {
        set_reason(reason, "masked GKR terminal dimensions are invalid");
        return false;
    }
    return true;
}

F gkr_eval(const std::vector<F> &c, const std::vector<F> &v) {
    return c[0] * v[0] * v[2] + c[1] * v[1] * v[3] +
           c[2] * v[1] * v[2] + c[3] * v[0] + c[4] * v[1];
}

F gkr_fixed_u_eval(const std::vector<F> &c, const std::vector<F> &u,
                   const std::vector<F> &v) {
    return c[0] * u[0] * v[0] + c[1] * u[1] * v[1] +
           c[2] * u[1] * v[0] + c[3] * u[0] + c[4] * u[1];
}

F gkr_fixed_v_eval(const std::vector<F> &c, const std::vector<F> &u,
                   const std::vector<F> &v) {
    return c[0] * u[0] * v[0] + c[1] * u[1] * v[1] +
           c[2] * u[1] * v[0] + c[3] * u[0] + c[4] * u[1];
}

F random_nonzero_field() {
    F result;
    do result = random_field(); while (result.isZero());
    return result;
}

bool sample_terminal(const NativeMaskedTerminalSpec &spec, const F &running,
                     std::vector<F> &terminal, std::string *reason) {
    std::size_t arity = 0;
    if (!terminal_shape(spec, 1, arity, reason)) {
        // terminal_shape enforces relation-specific degrees.  This helper is
        // called after the caller has validated the actual degree, so use the
        // lightweight arity switch below for degree-two/three relations.
        switch (spec.kind) {
        case NativeMaskedTerminalKind::Scalar: arity = 1; break;
        case NativeMaskedTerminalKind::Product2:
        case NativeMaskedTerminalKind::Product2Public:
        case NativeMaskedTerminalKind::GkrPredicateFixedU:
        case NativeMaskedTerminalKind::GkrPredicateFixedV: arity =
            spec.kind == NativeMaskedTerminalKind::Product2 ? 2 :
            (spec.kind == NativeMaskedTerminalKind::Product2Public ? 1 : 2); break;
        case NativeMaskedTerminalKind::Product3: arity = 3; break;
        case NativeMaskedTerminalKind::Claims: arity = 2; break;
        case NativeMaskedTerminalKind::GkrPredicate: arity = 4; break;
        default: break;
        }
    }
    (void)arity;
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar:
        terminal = {running};
        return true;
    case NativeMaskedTerminalKind::Claims: {
        if (spec.terminal_linear_coefficients.size() != 2) {
            set_reason(reason, "masked claims relation needs two coefficients");
            return false;
        }
        const F a = spec.terminal_linear_coefficients[0];
        const F b = spec.terminal_linear_coefficients[1];
        const F rhs = running - spec.terminal_linear_offset;
        if (!b.isZero()) {
            const F t0 = random_field();
            terminal = {t0, (rhs - a * t0) / b};
        } else if (!a.isZero()) {
            terminal = {rhs / a, random_field()};
        } else if (rhs.isZero()) {
            terminal = {random_field(), random_field()};
        } else {
            set_reason(reason, "masked claims terminal relation has no solution");
            return false;
        }
        return true;
    }
    case NativeMaskedTerminalKind::Product2: {
        const F left = random_nonzero_field();
        terminal = {left, running / left};
        return true;
    }
    case NativeMaskedTerminalKind::Product3: {
        const F left = random_nonzero_field();
        const F middle = random_nonzero_field();
        terminal = {left, middle, running / (left * middle)};
        return true;
    }
    case NativeMaskedTerminalKind::Product2Public:
        if (spec.public_evaluation.isZero()) {
            if (!running.isZero()) {
                set_reason(reason, "masked public product has a zero public factor");
                return false;
            }
            terminal = {random_field()};
        } else {
            terminal = {running / spec.public_evaluation};
        }
        return true;
    case NativeMaskedTerminalKind::GkrPredicate: {
        if (spec.gkr_coefficients.size() != 5) {
            set_reason(reason, "masked GKR predicate coefficients are invalid");
            return false;
        }
        for (unsigned attempt = 0; attempt < 128; ++attempt) {
            F u0 = random_field(), u1 = random_field();
            F v0 = random_field(), v1 = random_field();
            const F coeff = spec.gkr_coefficients[1] * u1;
            if (!coeff.isZero()) {
                const F base = spec.gkr_coefficients[0] * u0 * v0 +
                               spec.gkr_coefficients[2] * u1 * v0 +
                               spec.gkr_coefficients[3] * u0 +
                               spec.gkr_coefficients[4] * u1;
                v1 = (running - base) / coeff;
                terminal = {u0, u1, v0, v1};
                return true;
            }
            const F u0_coeff = spec.gkr_coefficients[0] * v0 +
                               spec.gkr_coefficients[3];
            if (!u0_coeff.isZero()) {
                const F base = spec.gkr_coefficients[1] * u1 * v1 +
                               spec.gkr_coefficients[2] * u1 * v0 +
                               spec.gkr_coefficients[4] * u1;
                u0 = (running - base) / u0_coeff;
                terminal = {u0, u1, v0, v1};
                return true;
            }
        }
        set_reason(reason, "masked GKR predicate terminal sampling failed");
        return false;
    }
    case NativeMaskedTerminalKind::GkrPredicateFixedU: {
        if (spec.fixed_values.size() != 2 || spec.gkr_coefficients.size() != 5) {
            set_reason(reason, "masked fixed-U terminal values are invalid");
            return false;
        }
        const F &u0 = spec.fixed_values[0], &u1 = spec.fixed_values[1];
        const F v1_coeff = spec.gkr_coefficients[1] * u1;
        if (!v1_coeff.isZero()) {
            const F v0 = random_field();
            const F base = spec.gkr_coefficients[0] * u0 * v0 +
                           spec.gkr_coefficients[2] * u1 * v0 +
                           spec.gkr_coefficients[3] * u0 +
                           spec.gkr_coefficients[4] * u1;
            terminal = {v0, (running - base) / v1_coeff};
            return true;
        }
        const F v0_coeff = spec.gkr_coefficients[0] * u0 +
                           spec.gkr_coefficients[2] * u1;
        if (!v0_coeff.isZero()) {
            const F v1 = random_field();
            const F base = spec.gkr_coefficients[1] * u1 * v1 +
                           spec.gkr_coefficients[3] * u0 +
                           spec.gkr_coefficients[4] * u1;
            terminal = {(running - base) / v0_coeff, v1};
            return true;
        }
        if ((spec.gkr_coefficients[3] * u0 +
             spec.gkr_coefficients[4] * u1 - running).isZero()) {
            terminal = {random_field(), random_field()};
            return true;
        }
        set_reason(reason, "masked fixed-U predicate terminal has no solution");
        return false;
    }
    case NativeMaskedTerminalKind::GkrPredicateFixedV: {
        if (spec.fixed_values.size() != 2 || spec.gkr_coefficients.size() != 5) {
            set_reason(reason, "masked fixed-V terminal values are invalid");
            return false;
        }
        const F &v0 = spec.fixed_values[0], &v1 = spec.fixed_values[1];
        const F u0_coeff = spec.gkr_coefficients[0] * v0 +
                           spec.gkr_coefficients[3];
        if (!u0_coeff.isZero()) {
            const F u1 = random_field();
            const F base = spec.gkr_coefficients[1] * u1 * v1 +
                           spec.gkr_coefficients[2] * u1 * v0 +
                           spec.gkr_coefficients[4] * u1;
            terminal = {(running - base) / u0_coeff, u1};
            return true;
        }
        const F u1_coeff = spec.gkr_coefficients[1] * v1 +
                           spec.gkr_coefficients[2] * v0 +
                           spec.gkr_coefficients[4];
        if (!u1_coeff.isZero()) {
            const F u0 = random_field();
            const F base = spec.gkr_coefficients[0] * u0 * v0 +
                           spec.gkr_coefficients[3] * u0;
            terminal = {u0, (running - base) / u1_coeff};
            return true;
        }
        if ((spec.gkr_coefficients[0].isZero() &&
             spec.gkr_coefficients[3].isZero() && running.isZero())) {
            terminal = {random_field(), random_field()};
            return true;
        }
        set_reason(reason, "masked fixed-V predicate terminal has no solution");
        return false;
    }
    default:
        set_reason(reason, "unknown masked terminal relation");
        return false;
    }
}

bool check_terminal(const NativeMaskedTerminalSpec &spec, const F &running,
                    const std::vector<F> &terminal, bool use_unmasked_fixed,
                    std::string *reason) {
    std::size_t arity = 0;
    // terminal_shape's degree checks are performed by the caller.  Recreate
    // only the arity here so Product2/Product3 can be checked independently.
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar: arity = 1; break;
    case NativeMaskedTerminalKind::Product2: arity = 2; break;
    case NativeMaskedTerminalKind::Product3: arity = 3; break;
    case NativeMaskedTerminalKind::Claims: arity = 2; break;
    case NativeMaskedTerminalKind::GkrPredicate: arity = 4; break;
    case NativeMaskedTerminalKind::Product2Public: arity = 1; break;
    case NativeMaskedTerminalKind::GkrPredicateFixedU:
    case NativeMaskedTerminalKind::GkrPredicateFixedV: arity = 2; break;
    default: set_reason(reason, "unknown masked terminal relation"); return false;
    }
    if (terminal.size() != arity) {
        set_reason(reason, "masked terminal relation arity mismatch");
        return false;
    }
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar:
        if (terminal[0] != running) {
            set_reason(reason, "masked scalar terminal relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::Product2:
        if (terminal[0] * terminal[1] != running) {
            set_reason(reason, "masked Product2 terminal relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::Product3:
        if (terminal[0] * terminal[1] * terminal[2] != running) {
            set_reason(reason, "masked Product3 terminal relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::Claims: {
        if (spec.terminal_linear_coefficients.size() != 2 ||
            spec.terminal_linear_coefficients[0] * terminal[0] +
                spec.terminal_linear_coefficients[1] * terminal[1] +
                spec.terminal_linear_offset != running) {
            set_reason(reason, "masked claims terminal relation failed");
            return false;
        }
        return true;
    }
    case NativeMaskedTerminalKind::Product2Public:
        if (terminal[0] * spec.public_evaluation != running) {
            set_reason(reason, "masked public Product2 terminal relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::GkrPredicate:
        if (!gkr_value(spec.gkr_coefficients, terminal, reason) ||
            gkr_eval(spec.gkr_coefficients, terminal) != running) {
            set_reason(reason, "masked GKR terminal relation failed");
            return false;
        }
        return true;
    case NativeMaskedTerminalKind::GkrPredicateFixedU: {
        const std::vector<F> &fixed = use_unmasked_fixed &&
            spec.unmasked_fixed_values.size() == 2
                ? spec.unmasked_fixed_values : spec.fixed_values;
        if (fixed.size() != 2 ||
            gkr_fixed_u_eval(spec.gkr_coefficients, fixed, terminal) != running) {
            set_reason(reason, "masked fixed-U terminal relation failed");
            return false;
        }
        return true;
    }
    case NativeMaskedTerminalKind::GkrPredicateFixedV: {
        const std::vector<F> &fixed = use_unmasked_fixed &&
            spec.unmasked_fixed_values.size() == 2
                ? spec.unmasked_fixed_values : spec.fixed_values;
        if (fixed.size() != 2 ||
            gkr_fixed_v_eval(spec.gkr_coefficients, terminal, fixed) != running) {
            set_reason(reason, "masked fixed-V terminal relation failed");
            return false;
        }
        return true;
    }
    default:
        set_reason(reason, "unknown masked terminal relation");
        return false;
    }
}

bool relation_opening(const ScalarOpening &opening, const PedersenBasis &basis,
                      const F &expected, std::string *reason) {
    if (!opening.valid(basis, reason) || opening.value != expected) {
        set_reason(reason, "masked Sumcheck opening value does not match relation");
        return false;
    }
    return true;
}

bool emit_linear(NativeProofSession &session, const PedersenBasis &basis,
                 const std::string &label, const std::vector<ScalarOpening> &openings,
                 const std::vector<F> &coefficients, const F &public_value,
                 std::string *reason) {
    LinearRelationProof ignored;
    const bool ok = zkgpt::native_pedersen::prove_hidden_linear(
        session, basis, label, openings, coefficients, public_value, &ignored,
        reason);
    if (!ok && reason != nullptr && !reason->empty()) *reason = label + ": " + *reason;
    return ok;
}

bool emit_opening(NativeProofSession &session, const PedersenBasis &basis,
                  const std::string &label, const ScalarOpening &opening,
                  std::string *reason) {
    return zkgpt::native_pedersen::prove_opening(session, basis, label, opening,
                                                  nullptr, reason);
}

std::vector<F> ones(std::size_t count) { return std::vector<F>(count, F_ONE); }

std::vector<F> signed_ones(std::size_t count, F sign) {
    return std::vector<F>(count, sign);
}

bool initialize_external(NativeMaskedSumcheckProverState &state,
                         const F *unmasked_initial_claim, std::string *reason) {
    const bool public_initial = state.terminal_spec.public_initial_claim;
    if (!state.terminal_spec.has_external_binding ||
        !state.terminal_spec.initial_commitment.valid(state.basis, reason) ||
        !state.terminal_spec.initial_opening.valid(state.basis, reason) ||
        !equal_commitment(state.terminal_spec.initial_opening.commitment,
                          state.terminal_spec.initial_commitment) ||
        (public_initial && unmasked_initial_claim != nullptr) ||
        (!public_initial && unmasked_initial_claim == nullptr)) {
        set_reason(reason, "masked Sumcheck requires a matching external initial opening");
        return false;
    }
    if (!public_initial &&
        (!state.terminal_spec.has_masked_initial_binding ||
         !state.terminal_spec.masked_initial_commitment.valid(state.basis, reason) ||
         !state.terminal_spec.masked_initial_opening.valid(state.basis, reason) ||
         !equal_commitment(state.terminal_spec.masked_initial_opening.commitment,
                           state.terminal_spec.masked_initial_commitment))) {
        set_reason(reason, "masked Sumcheck requires a matching masked initial opening");
        return false;
    }
    const F expected_external = public_initial ? state.masked_running
                                               : *unmasked_initial_claim;
    if (state.terminal_spec.initial_opening.value != expected_external) {
        set_reason(reason, "masked Sumcheck external initial value is not canonical");
        return false;
    }
    if (!public_initial &&
        state.terminal_spec.masked_initial_opening.value != state.masked_running) {
        set_reason(reason, "masked Sumcheck masked initial value is not canonical");
        return false;
    }
    state.result.initial_commitment = state.terminal_spec.initial_commitment;
    state.session->absorb(state.label + "/initial/mode",
                          public_initial ? "zero-mask-v2" : "with-mask-v2");
    if (!emit_opening(*state.session, state.basis,
                      state.label + "/initial/opening",
                      state.terminal_spec.initial_opening, reason))
        return false;

    if (public_initial) {
        // The opening proof alone proves knowledge of the opening, not that
        // it commits the public claim.  Bind the externally supplied
        // commitment to the public scalar with a native hidden-linear proof.
        return emit_linear(*state.session, state.basis,
                           state.label + "/initial/public-bind",
                           {state.terminal_spec.initial_opening}, {F_ONE},
                           state.masked_running, reason);
    }
    const ScalarOpening &masked = state.terminal_spec.masked_initial_opening;
    const ScalarOpening mask = zkgpt::native_pedersen::commit_fresh(
        state.basis, state.masked_running - *unmasked_initial_claim);
    if (!emit_opening(*state.session, state.basis,
                      state.label + "/initial/masked", masked, reason) ||
        !emit_opening(*state.session, state.basis,
                      state.label + "/initial/mask", mask, reason) ||
        !emit_linear(*state.session, state.basis,
                     state.label + "/initial/masked-bind", {masked},
                     {F_ONE}, state.masked_running, reason) ||
        !emit_linear(*state.session, state.basis,
                     state.label + "/initial/mask-relation",
                     {masked, state.terminal_spec.initial_opening, mask},
                     {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason))
        return false;
    state.previous_mask_eval = mask;
    state.has_previous_mask_eval = true;
    state.result.operand_commitments.push_back(masked.commitment);
    state.result.operand_commitments.push_back(mask.commitment);
    return true;
}

bool prove_round(NativeMaskedSumcheckProverState &state,
                 const std::vector<F> &unmasked_poly, F *challenge,
                 std::string *reason) {
    const std::size_t round = state.rounds;
    const std::string base = state.label + "/round/" + std::to_string(round);
    if (unmasked_poly.size() != state.degree + 1 || state.degree < 1 ||
        state.degree > 3 || round >= kMaxRounds) {
        set_reason(reason, "masked Sumcheck round polynomial arity is invalid");
        return false;
    }
    if (state.unmasked_running != unmasked_poly[0] + unmasked_poly[1]) {
        set_reason(reason, "unmasked Sumcheck claim transition failed");
        return false;
    }
    std::vector<F> mask(state.degree + 1);
    mask[0] = random_field();
    mask[1] = state.mask_target - mask[0];
    for (std::size_t i = 2; i < mask.size(); ++i) mask[i] = random_field();
    std::vector<F> q(unmasked_poly.size());
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = unmasked_poly[i] + mask[i];

    std::vector<ScalarOpening> p, h, q_open;
    p.reserve(q.size()); h.reserve(q.size()); q_open.reserve(q.size());
    for (const F &value : unmasked_poly) p.push_back(
        zkgpt::native_pedersen::commit_fresh(state.basis, value));
    for (const F &value : mask) h.push_back(
        zkgpt::native_pedersen::commit_fresh(state.basis, value));
    for (const F &value : q) q_open.push_back(
        zkgpt::native_pedersen::commit_fresh(state.basis, value));
    const ScalarOpening p_claim = zkgpt::native_pedersen::commit_fresh(
        state.basis, unmasked_poly[0] + unmasked_poly[1]);
    const ScalarOpening h_claim = zkgpt::native_pedersen::commit_fresh(
        state.basis, mask[0] + mask[1]);
    const ScalarOpening q_claim = zkgpt::native_pedersen::commit_fresh(
        state.basis, q[0] + q[1]);

    for (std::size_t i = 0; i < q.size(); ++i)
        if (!emit_opening(*state.session, state.basis,
                          base + "/p/" + std::to_string(i), p[i], reason)) return false;
    for (std::size_t i = 0; i < q.size(); ++i)
        if (!emit_opening(*state.session, state.basis,
                          base + "/h/" + std::to_string(i), h[i], reason)) return false;
    for (std::size_t i = 0; i < q.size(); ++i)
        if (!emit_opening(*state.session, state.basis,
                          base + "/q/" + std::to_string(i), q_open[i], reason)) return false;
    state.session->absorb_fields(base + "/q-values", q);
    if (!emit_opening(*state.session, state.basis, base + "/p-claim", p_claim, reason) ||
        !emit_opening(*state.session, state.basis, base + "/h-claim", h_claim, reason) ||
        !emit_opening(*state.session, state.basis, base + "/q-claim", q_claim, reason))
        return false;
    for (std::size_t i = 0; i < q.size(); ++i)
        if (!emit_linear(*state.session, state.basis,
                         base + "/q-public/" + std::to_string(i), {q_open[i]},
                         {F_ONE}, q[i], reason)) return false;
    std::vector<ScalarOpening> q_relation = q_open;
    q_relation.insert(q_relation.end(), p.begin(), p.end());
    q_relation.insert(q_relation.end(), h.begin(), h.end());
    std::vector<F> q_coeff(q.size(), F_ONE);
    q_coeff.insert(q_coeff.end(), q.size(), -F_ONE);
    q_coeff.insert(q_coeff.end(), q.size(), -F_ONE);
    std::vector<F> claim_coefficients(q.size(), F_ZERO);
    claim_coefficients[0] = F_ONE;
    claim_coefficients[1] = F_ONE;
    claim_coefficients.push_back(-F_ONE);
    if (!emit_linear(*state.session, state.basis, base + "/q-linear",
                     q_relation, q_coeff, F_ZERO, reason) ||
        !emit_linear(*state.session, state.basis, base + "/p-claim-bind",
                     [&]() { std::vector<ScalarOpening> v = p; v.push_back(p_claim); return v; }(),
                     claim_coefficients,
                     F_ZERO, reason) ||
        !emit_linear(*state.session, state.basis, base + "/h-claim-bind",
                     [&]() { std::vector<ScalarOpening> v = h; v.push_back(h_claim); return v; }(),
                     claim_coefficients,
                     F_ZERO, reason) ||
        !emit_linear(*state.session, state.basis, base + "/q-claim-bind",
                     [&]() { std::vector<ScalarOpening> v = q_open; v.push_back(q_claim); return v; }(),
                     claim_coefficients,
                     F_ZERO, reason) ||
        !emit_linear(*state.session, state.basis, base + "/claim-add",
                     {q_claim, p_claim, h_claim}, {F_ONE, -F_ONE, -F_ONE},
                     F_ZERO, reason)) return false;
    if (state.has_previous_mask_eval) {
        if (!emit_linear(*state.session, state.basis, base + "/h-transition",
                         {h_claim, state.previous_mask_eval}, {F_ONE, -F_ONE},
                         F_ZERO, reason)) return false;
    } else if (!emit_linear(*state.session, state.basis, base + "/h-transition",
                            {h_claim}, {F_ONE}, F_ZERO, reason)) return false;
    if (!emit_linear(*state.session, state.basis, base + "/transition",
                     {q_claim}, {F_ONE}, state.masked_running, reason)) return false;

    state.session->absorb_fields(base + "/q-transcript", q);
    const F r = state.session->challenge(state.label + "/r/" +
                                          std::to_string(round));
    const F next_unmasked = interpolate_at(unmasked_poly, r);
    const F next_masked = interpolate_at(q, r);
    const F next_mask = interpolate_at(mask, r);
    const ScalarOpening h_eval = zkgpt::native_pedersen::commit_fresh(
        state.basis, next_mask);
    if (!emit_opening(*state.session, state.basis, base + "/h-eval", h_eval,
                      reason)) return false;
    const std::vector<F> lambdas = lagrange_coefficients(state.degree, r);
    std::vector<ScalarOpening> h_eval_operands = h;
    h_eval_operands.push_back(h_eval);
    std::vector<F> h_eval_coefficients = lambdas;
    h_eval_coefficients.push_back(-F_ONE);
    if (!emit_linear(*state.session, state.basis, base + "/h-eval-bind",
                     h_eval_operands, h_eval_coefficients, F_ZERO, reason)) return false;

    NativeMaskedRoundBinding binding;
    for (const ScalarOpening &opening : p) binding.p_coefficients.push_back(opening.commitment);
    for (const ScalarOpening &opening : h) binding.h_coefficients.push_back(opening.commitment);
    for (const ScalarOpening &opening : q_open) binding.q_coefficients.push_back(opening.commitment);
    binding.p_claim = p_claim.commitment;
    binding.h_claim = h_claim.commitment;
    binding.q_claim = q_claim.commitment;
    binding.h_evaluation = h_eval.commitment;
    state.result.rounds.push_back(binding);
    state.result.operand_commitments.push_back(h_eval.commitment);
    state.result.challenges.push_back(r);
    state.unmasked_running = next_unmasked;
    state.masked_running = next_masked;
    state.mask_target = next_mask;
    state.previous_mask_eval = h_eval;
    state.has_previous_mask_eval = true;
    ++state.rounds;
    if (challenge != nullptr) *challenge = r;
    return true;
}

bool make_terminal_opening(const PedersenBasis &basis, const ScalarOpening *provided,
                          const F &value, ScalarOpening &out,
                          const std::string &what, std::string *reason) {
    if (provided != nullptr) {
        if (!relation_opening(*provided, basis, value, reason)) {
            set_reason(reason, what + " opening is not the masked terminal value");
            return false;
        }
        out = *provided;
    } else {
        out = zkgpt::native_pedersen::commit_fresh(basis, value);
    }
    return true;
}

bool prove_unmasked_terminal_relation(
    NativeMaskedSumcheckProverState &state,
    const std::vector<ScalarOpening> &terminal_openings,
    const std::vector<ScalarOpening> &fixed_openings,
    ScalarOpening &claim_opening, std::string *reason) {
    const NativeMaskedTerminalSpec &spec = state.terminal_spec;
    const std::string &label = state.label;
    claim_opening = zkgpt::native_pedersen::commit_fresh(
        state.basis, state.unmasked_running);
    if (!emit_opening(*state.session, state.basis,
                      label + "/terminal/unmasked/claim", claim_opening, reason))
        return false;
    for (std::size_t i = 0; i < terminal_openings.size(); ++i)
        if (!emit_opening(*state.session, state.basis,
                          label + "/terminal/unmasked/value/" + std::to_string(i),
                          terminal_openings[i], reason)) return false;
    for (std::size_t i = 0; i < fixed_openings.size(); ++i)
        if (!emit_opening(*state.session, state.basis,
                          label + "/terminal/unmasked/fixed/" + std::to_string(i),
                          fixed_openings[i], reason)) return false;

    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar:
        return emit_linear(*state.session, state.basis,
                           label + "/terminal/unmasked/scalar",
                           {terminal_openings[0], claim_opening},
                           {F_ONE, -F_ONE}, F_ZERO, reason);
    case NativeMaskedTerminalKind::Claims:
        {
            std::vector<ScalarOpening> relation_operands = terminal_openings;
            relation_operands.push_back(claim_opening);
            std::vector<F> coefficients = spec.terminal_linear_coefficients;
            coefficients.push_back(-F_ONE);
            return emit_linear(*state.session, state.basis,
                               label + "/terminal/unmasked/claims",
                               relation_operands, coefficients,
                               spec.terminal_linear_offset, reason);
        }
    case NativeMaskedTerminalKind::Product2: {
        const MultiplicationLinks links{
            label + "/terminal/unmasked/value/0",
            label + "/terminal/unmasked/value/1",
            label + "/terminal/unmasked/claim"};
        return zkgpt::native_pedersen::prove_multiplication(
            *state.session, state.basis,
            label + "/terminal/unmasked/product2", terminal_openings[0],
            terminal_openings[1], claim_opening, links, nullptr, reason);
    }
    case NativeMaskedTerminalKind::Product2Public:
        return emit_linear(*state.session, state.basis,
                           label + "/terminal/unmasked/public-product",
                           {terminal_openings[0], claim_opening},
                           {spec.public_evaluation, -F_ONE}, F_ZERO, reason);
    case NativeMaskedTerminalKind::GkrPredicate:
    case NativeMaskedTerminalKind::GkrPredicateFixedU:
    case NativeMaskedTerminalKind::GkrPredicateFixedV: {
        const bool fixed_u = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU;
        const bool fixed_v = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV;
        std::vector<ScalarOpening> products;
        const std::string product_prefix = label + "/terminal/unmasked/gkr-product/";
        const auto make_product = [&](std::size_t index, const ScalarOpening &a,
                                      const ScalarOpening &b,
                                      const std::string &a_label,
                                      const std::string &b_label) -> bool {
            const ScalarOpening product = zkgpt::native_pedersen::commit_fresh(
                state.basis, a.value * b.value);
            if (!emit_opening(*state.session, state.basis,
                              product_prefix + std::to_string(index), product,
                              reason)) return false;
            const MultiplicationLinks links{a_label, b_label,
                                            product_prefix + std::to_string(index)};
            if (!zkgpt::native_pedersen::prove_multiplication(
                    *state.session, state.basis,
                    label + "/terminal/unmasked/gkr-mul/" + std::to_string(index),
                    a, b, product, links, nullptr, reason)) return false;
            products.push_back(product);
            return true;
        };
        const auto value_label = [&](std::size_t i) {
            return label + "/terminal/unmasked/value/" + std::to_string(i);
        };
        const auto fixed_label = [&](std::size_t i) {
            return label + "/terminal/unmasked/fixed/" + std::to_string(i);
        };
        if (fixed_u) {
            if (!make_product(0, fixed_openings[0], terminal_openings[0],
                              fixed_label(0), value_label(0)) ||
                !make_product(1, fixed_openings[1], terminal_openings[1],
                              fixed_label(1), value_label(1)) ||
                !make_product(2, fixed_openings[1], terminal_openings[0],
                              fixed_label(1), value_label(0))) return false;
        } else if (fixed_v) {
            if (!make_product(0, terminal_openings[0], fixed_openings[0],
                              value_label(0), fixed_label(0)) ||
                !make_product(1, terminal_openings[1], fixed_openings[1],
                              value_label(1), fixed_label(1)) ||
                !make_product(2, terminal_openings[1], fixed_openings[0],
                              value_label(1), fixed_label(0))) return false;
        } else {
            if (!make_product(0, terminal_openings[0], terminal_openings[2],
                              value_label(0), value_label(2)) ||
                !make_product(1, terminal_openings[1], terminal_openings[3],
                              value_label(1), value_label(3)) ||
                !make_product(2, terminal_openings[1], terminal_openings[2],
                              value_label(1), value_label(2))) return false;
        }
        std::vector<ScalarOpening> relation_operands = products;
        if (fixed_u)
            relation_operands.insert(relation_operands.end(), fixed_openings.begin(),
                                     fixed_openings.end());
        else
            relation_operands.insert(relation_operands.end(), terminal_openings.begin(),
                                     terminal_openings.begin() + 2);
        relation_operands.push_back(claim_opening);
        std::vector<F> coefficients{spec.gkr_coefficients[0],
                                    spec.gkr_coefficients[1],
                                    spec.gkr_coefficients[2],
                                    spec.gkr_coefficients[3],
                                    spec.gkr_coefficients[4], -F_ONE};
        return emit_linear(*state.session, state.basis,
                           label + "/terminal/unmasked/gkr-linear",
                           relation_operands, coefficients, F_ZERO, reason);
    }
    default:
        set_reason(reason, "unknown masked terminal relation");
        return false;
    }
}

bool bind_masked_claim(NativeMaskedSumcheckProverState &state,
                       const ScalarOpening &claim_opening,
                       const ScalarOpening &masked_claim,
                       std::vector<ScalarOpening> &operands,
                       std::string *reason) {
    const ScalarOpening mask = zkgpt::native_pedersen::commit_fresh(
        state.basis, state.masked_running - state.unmasked_running);
    if (!emit_opening(*state.session, state.basis,
                      state.label + "/terminal/claim-mask", mask, reason) ||
        !emit_linear(*state.session, state.basis,
                     state.label + "/terminal/claim-bind",
                     {masked_claim, claim_opening, mask},
                     {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason)) return false;
    operands.push_back(masked_claim);
    state.result.claim_opening = masked_claim;
    state.result.unmasked_claim_opening = claim_opening;
    state.result.claim_mask_opening = mask;
    state.result.claim_commitment = masked_claim.commitment;
    state.result.unmasked_claim_commitment = claim_opening.commitment;
    state.result.claim_mask_commitment = mask.commitment;
    state.result.has_claim_opening = true;
    state.result.has_unmasked_claim_opening = true;
    state.result.has_claim_mask_opening = true;
    return true;
}

bool prove_terminal_relation(NativeMaskedSumcheckProverState &state,
                             const std::vector<F> &masked_terminal,
                             std::vector<ScalarOpening> &terminal_openings,
                             std::vector<ScalarOpening> &fixed_openings,
                             std::vector<ScalarOpening> &unmasked_terminal_openings,
                             std::vector<ScalarOpening> &unmasked_fixed_openings,
                             std::vector<ScalarOpening> &operands,
                             std::string *reason) {
    const NativeMaskedTerminalSpec &spec = state.terminal_spec;
    const std::string &label = state.label;
    if (!prove_unmasked_terminal_relation(state, unmasked_terminal_openings,
                                           unmasked_fixed_openings,
                                           state.result.unmasked_claim_opening,
                                           reason)) return false;
    state.session->absorb_fields(label + "/terminal/values", masked_terminal);
    for (std::size_t i = 0; i < fixed_openings.size(); ++i)
        if (!emit_opening(*state.session, state.basis,
                          label + "/terminal/fixed/" + std::to_string(i),
                          fixed_openings[i], reason)) return false;
    for (std::size_t i = 0; i < terminal_openings.size(); ++i)
        if (!emit_opening(*state.session, state.basis,
                          label + "/terminal/value/" + std::to_string(i),
                          terminal_openings[i], reason)) return false;
    for (std::size_t i = 0; i < terminal_openings.size(); ++i) {
        const ScalarOpening mask = zkgpt::native_pedersen::commit_fresh(
            state.basis, terminal_openings[i].value - unmasked_terminal_openings[i].value);
        if (!emit_opening(*state.session, state.basis,
                          label + "/terminal/mask/" + std::to_string(i), mask,
                          reason) ||
            !emit_linear(*state.session, state.basis,
                         label + "/terminal/mask-bind/" + std::to_string(i),
                         {terminal_openings[i], unmasked_terminal_openings[i], mask},
                         {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason)) return false;
    }
    for (std::size_t i = 0; i < fixed_openings.size(); ++i) {
        const ScalarOpening mask = zkgpt::native_pedersen::commit_fresh(
            state.basis, fixed_openings[i].value - unmasked_fixed_openings[i].value);
        if (!emit_opening(*state.session, state.basis,
                          label + "/terminal/fixed-mask/" + std::to_string(i),
                          mask, reason) ||
            !emit_linear(*state.session, state.basis,
                         label + "/terminal/fixed-mask-bind/" + std::to_string(i),
                         {fixed_openings[i], unmasked_fixed_openings[i], mask},
                         {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason)) return false;
    }

    const std::size_t arity = terminal_openings.size();
    switch (spec.kind) {
    case NativeMaskedTerminalKind::Scalar: {
        const ScalarOpening anchor = zkgpt::native_pedersen::commit_fresh(
            state.basis, state.masked_running);
        if (!emit_opening(*state.session, state.basis, label + "/terminal/anchor",
                          anchor, reason) ||
            !emit_linear(*state.session, state.basis, label + "/terminal/scalar",
                         terminal_openings, {F_ONE}, state.masked_running, reason) ||
            !emit_linear(*state.session, state.basis,
                         label + "/terminal/anchor-bind", {anchor}, {F_ONE},
                         state.masked_running, reason)) return false;
        return bind_masked_claim(state, state.result.unmasked_claim_opening,
                                 anchor, operands, reason);
    }
    case NativeMaskedTerminalKind::Claims: {
        const ScalarOpening anchor = zkgpt::native_pedersen::commit_fresh(
            state.basis, state.masked_running);
        if (!emit_opening(*state.session, state.basis, label + "/terminal/anchor",
                          anchor, reason) ||
            !emit_linear(*state.session, state.basis, label + "/terminal/claims",
                         terminal_openings, spec.terminal_linear_coefficients,
                         state.masked_running - spec.terminal_linear_offset, reason) ||
            !emit_linear(*state.session, state.basis,
                         label + "/terminal/anchor-bind", {anchor}, {F_ONE},
                         state.masked_running, reason)) return false;
        return bind_masked_claim(state, state.result.unmasked_claim_opening,
                                 anchor, operands, reason);
    }
    case NativeMaskedTerminalKind::Product2: {
        const ScalarOpening anchor = zkgpt::native_pedersen::commit_fresh(
            state.basis, state.masked_running);
        if (!emit_opening(*state.session, state.basis, label + "/terminal/anchor",
                          anchor, reason)) return false;
        operands.push_back(anchor);
        const MultiplicationLinks links{
            label + "/terminal/value/0", label + "/terminal/value/1",
            label + "/terminal/anchor"};
        if (!zkgpt::native_pedersen::prove_multiplication(
                *state.session, state.basis, label + "/terminal/product2",
                terminal_openings[0], terminal_openings[1], anchor, links,
                nullptr, reason)) return false;
        if (!emit_linear(*state.session, state.basis, label + "/terminal/anchor-bind",
                         {anchor}, {F_ONE}, state.masked_running, reason)) return false;
        return bind_masked_claim(state, state.result.unmasked_claim_opening,
                                 anchor, operands, reason);
    }
    case NativeMaskedTerminalKind::Product3: {
        const ScalarOpening intermediate = zkgpt::native_pedersen::commit_fresh(
            state.basis, masked_terminal[0] * masked_terminal[1]);
        const ScalarOpening anchor = zkgpt::native_pedersen::commit_fresh(
            state.basis, state.masked_running);
        if (!emit_opening(*state.session, state.basis,
                          label + "/terminal/intermediate", intermediate, reason) ||
            !emit_opening(*state.session, state.basis, label + "/terminal/anchor",
                          anchor, reason)) return false;
        operands.push_back(intermediate);
        operands.push_back(anchor);
        const Product3Links links{
            {label + "/terminal/value/0", label + "/terminal/value/1",
             label + "/terminal/intermediate"},
            {label + "/terminal/intermediate", label + "/terminal/value/2",
             label + "/terminal/anchor"}};
        if (!zkgpt::native_pedersen::prove_product3(
                *state.session, state.basis, label + "/terminal/product3",
                terminal_openings[0], terminal_openings[1], terminal_openings[2],
                intermediate, anchor, links, nullptr, reason)) return false;
        if (!emit_linear(*state.session, state.basis, label + "/terminal/anchor-bind",
                         {anchor}, {F_ONE}, state.masked_running, reason)) return false;
        return bind_masked_claim(state, state.result.unmasked_claim_opening,
                                 anchor, operands, reason);
    }
    case NativeMaskedTerminalKind::Product2Public: {
        const ScalarOpening anchor = zkgpt::native_pedersen::commit_fresh(
            state.basis, state.masked_running);
        if (!emit_opening(*state.session, state.basis, label + "/terminal/anchor",
                          anchor, reason)) return false;
        operands.push_back(anchor);
        if (!emit_linear(*state.session, state.basis, label + "/terminal/public-product",
                         {terminal_openings[0], anchor},
                         {spec.public_evaluation, -F_ONE}, F_ZERO, reason)) return false;
        if (!emit_linear(*state.session, state.basis, label + "/terminal/anchor-bind",
                         {anchor}, {F_ONE}, state.masked_running, reason)) return false;
        return bind_masked_claim(state, state.result.unmasked_claim_opening,
                                 anchor, operands, reason);
    }
    case NativeMaskedTerminalKind::GkrPredicate:
    case NativeMaskedTerminalKind::GkrPredicateFixedU:
    case NativeMaskedTerminalKind::GkrPredicateFixedV: {
        const bool fixed_u = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedU;
        const bool fixed_v = spec.kind == NativeMaskedTerminalKind::GkrPredicateFixedV;
        std::vector<ScalarOpening> left = terminal_openings;
        std::vector<ScalarOpening> fixed = fixed_openings;
        std::vector<ScalarOpening> products;
        const std::string product_prefix = label + "/terminal/gkr-product/";
        const auto make_product = [&](std::size_t index, const ScalarOpening &a,
                                      const ScalarOpening &b,
                                      const std::string &a_label,
                                      const std::string &b_label) -> bool {
            const ScalarOpening product = zkgpt::native_pedersen::commit_fresh(
                state.basis, a.value * b.value);
            if (!emit_opening(*state.session, state.basis,
                              product_prefix + std::to_string(index), product,
                              reason)) return false;
            const MultiplicationLinks links{a_label, b_label,
                                            product_prefix + std::to_string(index)};
            if (!zkgpt::native_pedersen::prove_multiplication(
                    *state.session, state.basis,
                    label + "/terminal/gkr-mul/" + std::to_string(index),
                    a, b, product, links, nullptr, reason)) return false;
            products.push_back(product);
            return true;
        };
        if (fixed_u) {
            if (!make_product(0, fixed[0], left[0], label + "/terminal/fixed/0",
                              label + "/terminal/value/0") ||
                !make_product(1, fixed[1], left[1], label + "/terminal/fixed/1",
                              label + "/terminal/value/1") ||
                !make_product(2, fixed[1], left[0], label + "/terminal/fixed/1",
                              label + "/terminal/value/0")) return false;
        } else if (fixed_v) {
            if (!make_product(0, left[0], fixed[0], label + "/terminal/value/0",
                              label + "/terminal/fixed/0") ||
                !make_product(1, left[1], fixed[1], label + "/terminal/value/1",
                              label + "/terminal/fixed/1") ||
                !make_product(2, left[1], fixed[0], label + "/terminal/value/1",
                              label + "/terminal/fixed/0")) return false;
        } else {
            if (!make_product(0, left[0], left[2], label + "/terminal/value/0",
                              label + "/terminal/value/2") ||
                !make_product(1, left[1], left[3], label + "/terminal/value/1",
                              label + "/terminal/value/3") ||
                !make_product(2, left[1], left[2], label + "/terminal/value/1",
                              label + "/terminal/value/2")) return false;
        }
        const ScalarOpening anchor = zkgpt::native_pedersen::commit_fresh(
            state.basis, state.masked_running);
        if (!emit_opening(*state.session, state.basis, label + "/terminal/anchor",
                          anchor, reason)) return false;
        operands.push_back(anchor);
        std::vector<ScalarOpening> relation_operands = products;
        if (fixed_u) relation_operands.insert(relation_operands.end(), fixed.begin(), fixed.end());
        else relation_operands.insert(relation_operands.end(), left.begin(),
                                      left.begin() + 2);
        relation_operands.push_back(anchor);
        std::vector<F> coefficients{spec.gkr_coefficients[0],
                                    spec.gkr_coefficients[1],
                                    spec.gkr_coefficients[2]};
        if (fixed_u) {
            coefficients.insert(coefficients.end(), {spec.gkr_coefficients[3],
                                                     spec.gkr_coefficients[4]});
        } else {
            coefficients.insert(coefficients.end(), {spec.gkr_coefficients[3],
                                                     spec.gkr_coefficients[4]});
        }
        coefficients.push_back(-F_ONE);
        if (!emit_linear(*state.session, state.basis, label + "/terminal/gkr-linear",
                         relation_operands, coefficients, F_ZERO, reason)) return false;
        if (!emit_linear(*state.session, state.basis, label + "/terminal/anchor-bind",
                         {anchor}, {F_ONE}, state.masked_running, reason)) return false;
        return bind_masked_claim(state, state.result.unmasked_claim_opening,
                                 anchor, operands, reason);
    }
    default:
        set_reason(reason, "unknown masked terminal relation");
        return false;
    }
}

} // namespace
