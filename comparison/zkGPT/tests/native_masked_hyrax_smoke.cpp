#include "hyrax.hpp"
#include "native_pedersen_relation.hpp"
#include "native_masked_sumcheck.hpp"
#include "proof_session.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

#define HYRAX_SMOKE_FAIL() do { \
    std::cerr << "native_masked_hyrax_smoke: early false at " << __func__ << ':' << __LINE__ << '\n'; \
    return false; \
} while (false)

std::vector<Fr> mle_basis(const std::vector<Fr>& point) {
    if (point.empty()) return std::vector<Fr>{Fr(1)};
    std::vector<Fr> basis{Fr(1) - point[0], point[0]};
    for (std::size_t bit = 1; bit < point.size(); ++bit) {
        const std::size_t old_size = std::size_t(1) << bit;
        basis.resize(old_size * 2);
        for (std::size_t index = old_size; index-- > 0;) {
            const Fr previous = basis[index];
            basis[index] = previous * (Fr(1) - point[bit]);
            basis[index + old_size] = previous * point[bit];
        }
    }
    return basis;
}

bool tensor_tail_rounds_match_dense_reference(const std::vector<Fr>& values, std::uint64_t logical_length, std::uint64_t active_domain, const std::vector<Fr>& beta, const std::vector<Fr>& challenges, const std::vector<std::vector<Fr>>& trace) {
    if (values.size() != logical_length || beta.size() != challenges.size() || trace.size() != challenges.size()) HYRAX_SMOKE_FAIL();
    std::vector<Fr> private_values(static_cast<std::size_t>(active_domain), Fr(0)), public_values(static_cast<std::size_t>(active_domain), Fr(0));
    const std::vector<Fr> beta_weights = mle_basis(beta);
    for (std::uint64_t index = 0; index < logical_length; ++index) private_values[static_cast<std::size_t>(index)] = values[static_cast<std::size_t>(index)];
    for (std::uint64_t index = logical_length; index < active_domain; ++index) public_values[static_cast<std::size_t>(index)] = beta_weights[static_cast<std::size_t>(index)];
    for (std::size_t round = 0; round < challenges.size(); ++round) {
        const std::uint64_t block_size = std::uint64_t(1) << round;
        const std::size_t remaining = static_cast<std::size_t>(active_domain / block_size);
        const std::vector<Fr> low_point(challenges.begin(), challenges.begin() + static_cast<std::ptrdiff_t>(round));
        const std::vector<Fr> low_weights = mle_basis(low_point);
        std::vector<Fr> folded_private(remaining, Fr(0)), folded_public(remaining, Fr(0));
        for (std::size_t suffix = 0; suffix < remaining; ++suffix) {
            for (std::uint64_t low = 0; low < block_size; ++low) {
                const std::size_t index = suffix * static_cast<std::size_t>(block_size) + static_cast<std::size_t>(low);
                folded_private[suffix] += low_weights[static_cast<std::size_t>(low)] * private_values[index];
                folded_public[suffix] += low_weights[static_cast<std::size_t>(low)] * public_values[index];
            }
        }
        std::vector<Fr> expected(3, Fr(0));
        for (std::size_t pair = 0; pair < remaining; pair += 2) {
            const Fr &v0 = folded_private[pair], &v1 = folded_private[pair + 1];
            const Fr &a0 = folded_public[pair], &a1 = folded_public[pair + 1];
            expected[0] += v0 * a0;
            expected[1] += v1 * a1;
            expected[2] += (Fr(2) * v1 - v0) * (Fr(2) * a1 - a0);
        }
        if (trace[round] != expected) HYRAX_SMOKE_FAIL();
    }
    return true;
}

bool tensor_zero_interval_rounds_match_dense_reference(const std::vector<Fr> &values, std::uint64_t logical_length, std::uint64_t active_domain, const hyrax::HyraxTensorZeroInterval &interval, const std::vector<Fr> &beta, const std::vector<Fr> &challenges, const std::vector<std::vector<Fr>> &trace) {
    if (values.size() != logical_length || interval.begin >= interval.end || interval.end > logical_length || beta.size() != challenges.size() || trace.size() != challenges.size()) HYRAX_SMOKE_FAIL();
    std::vector<Fr> private_values(static_cast<std::size_t>(active_domain), Fr(0)), public_values(static_cast<std::size_t>(active_domain), Fr(0));
    const std::vector<Fr> beta_weights = mle_basis(beta);
    for (std::uint64_t index = 0; index < logical_length; ++index) private_values[static_cast<std::size_t>(index)] = values[static_cast<std::size_t>(index)];
    for (std::uint64_t index = interval.begin; index < interval.end; ++index) public_values[static_cast<std::size_t>(index)] = beta_weights[static_cast<std::size_t>(index)];
    for (std::size_t round = 0; round < challenges.size(); ++round) {
        const std::uint64_t block_size = std::uint64_t(1) << round;
        const std::size_t remaining = static_cast<std::size_t>(active_domain / block_size);
        const std::vector<Fr> low_point(challenges.begin(), challenges.begin() + static_cast<std::ptrdiff_t>(round));
        const std::vector<Fr> low_weights = mle_basis(low_point);
        std::vector<Fr> folded_private(remaining, Fr(0)), folded_public(remaining, Fr(0));
        for (std::size_t suffix = 0; suffix < remaining; ++suffix) for (std::uint64_t low = 0; low < block_size; ++low) {
            const std::size_t index = suffix * static_cast<std::size_t>(block_size) + static_cast<std::size_t>(low);
            folded_private[suffix] += low_weights[static_cast<std::size_t>(low)] * private_values[index];
            folded_public[suffix] += low_weights[static_cast<std::size_t>(low)] * public_values[index];
        }
        std::vector<Fr> expected(3, Fr(0));
        for (std::size_t pair = 0; pair < remaining; pair += 2) {
            const Fr &v0 = folded_private[pair], &v1 = folded_private[pair + 1];
            const Fr &a0 = folded_public[pair], &a1 = folded_public[pair + 1];
            expected[0] += v0 * a0;
            expected[1] += v1 * a1;
            expected[2] += (Fr(2) * v1 - v0) * (Fr(2) * a1 - a0);
        }
        if (trace[round] != expected) HYRAX_SMOKE_FAIL();
    }
    return true;
}

NativeProofStatement make_statement(const std::string& basis_digest, std::uint64_t dimension) {
    const std::string zeros = "0000000000000000000000000000000000000000000000000000000000000011";
    return NativeProofStatement{zeros, "mcl-bn254-fr-v1", zeros, zeros, zeros, zeros, zeros, zeros, basis_digest, dimension, zeros};
}

bool verify_provider(const NativeProofSession& session, const hyrax::PedersenMleCommitment& commitment, const hyrax::MaskedMleOpening& claim, const hyrax::HyraxMleBasisView& view, const zkgpt::native_pedersen::PedersenBasis& scalar_basis, std::string& reason) {
    return hyrax::verify_masked_mle_opening_with_session(commitment.commitment, claim.evaluation_mask_commitment, claim.masked_evaluation, view, session, "masked/opening", &reason, true, false, "", &scalar_basis, &claim.raw_evaluation_opening.commitment, &claim.masked_evaluation_opening.commitment);
}

bool reject_evaluation_transcript_tamper(const NativeProofSession& session) {
    std::vector<NativeProofMessage> messages = session.messages();
    bool changed = false;
    for (NativeProofMessage& message : messages) {
        if (message.label != "masked/opening/response-ipa/evaluation") continue;
        const std::size_t separator = message.payload.find('|');
        if (separator == std::string::npos) HYRAX_SMOKE_FAIL();
        Fr evaluation;
        try { evaluation.setStr(message.payload.substr(separator + 1), 16); } catch (...) { HYRAX_SMOKE_FAIL(); }
        message.payload = message.payload.substr(0, separator + 1) + (evaluation + Fr(1)).getStr(16);
        changed = true;
        break;
    }
    NativeProofSession tampered;
    std::string reason;
    return changed && !NativeProofSession::from_messages(session.domain(), session.statement(), messages, tampered, &reason);
}

bool reject_tensor_tail_raw_claim_tamper(const NativeProofSession &session, const hyrax::HyraxTensorMleView &view,
                                         const hyrax::HyraxTensorMleCommitment &commitment, std::size_t start,
                                         const zkgpt::native_pedersen::PedersenBasis &scalar_basis) {
    std::vector<NativeProofMessage> messages = session.messages();
    bool changed = false;
    for (NativeProofMessage &message : messages) {
        if (message.label != "tensor/aux/tail/opening/column/claim") continue;
        std::size_t separators[4], found = 0;
        for (std::size_t index = 0; index < message.payload.size() && found < 4; ++index)
            if (message.payload[index] == '|') separators[found++] = index;
        if (found != 4) HYRAX_SMOKE_FAIL();
        const auto replacement = zkgpt::native_pedersen::commit_fresh(scalar_basis, Fr(987654321));
        message.payload.replace(separators[2] + 1, separators[3] - separators[2] - 1, replacement.commitment.point.serializeToHexStr());
        changed = true;
        break;
    }
    if (!changed) HYRAX_SMOKE_FAIL();
    NativeProofSession tampered;
    std::string reason;
    if (!NativeProofSession::from_messages(session.domain(), session.statement(), messages, tampered, &reason)) return true;
    std::size_t position = start;
    std::vector<Fr> beta, point;
    NativeMaskedSumcheckResult sumcheck;
    return !hyrax::verify_hyrax_tensor_zero_tail_certificate_with_session(view, commitment, tampered, position, scalar_basis, beta, point, sumcheck, &reason);
}

bool check_response_ipa_base_binding(const NativeProofSession& session, const G1& H, const G1& Q) {
    Fr outer_challenge, randomizer;
    bool found_outer = false, found_randomizer = false;
    for (const NativeProofMessage& message : session.messages()) {
        if (message.kind != NativeProofMessageKind::Challenge) continue;
        if (message.label == "challenge/masked/opening/c") { outer_challenge.setStr(message.payload, 16); found_outer = true; }
        if (message.label == "challenge/masked/opening/response-ipa/evaluation-base") { randomizer.setStr(message.payload, 16); found_randomizer = true; }
    }
    if (!found_outer || !found_randomizer || outer_challenge.isZero() || randomizer.isZero()) HYRAX_SMOKE_FAIL();
    Fr inverse;
    Fr::inv(inverse, randomizer);
    const Fr delta(7), evaluation(23);
    const G1 D = Q * Fr(17), U = Q * randomizer, original = D + U * evaluation;
    const G1 malformed_commitment = D + Q * (outer_challenge * delta);
    const Fr compensated_evaluation = evaluation - outer_challenge * delta * inverse;
    const G1 commitment_compensation = malformed_commitment + U * compensated_evaluation;
    const G1 rho_compensation = D - H * delta + U * (evaluation + delta * inverse);
    return commitment_compensation == original && rho_compensation != original;
}

bool run_provider_case(std::uint64_t full_domain, std::uint64_t logical_length) {
    const std::string basis_id = "00000000000000000000000000000000000000000000000000000000000000ab";
    const std::string generator_label = "masked-hyrax-smoke/G";
    const std::string derivation_id = "hashAndMapToG1(canonical_basis_label+/coordinate/+canonical_absolute_index_decimal);absolute-index-u64;v1";
    const std::size_t point_size = static_cast<std::size_t>(std::log2(full_domain));
    std::uint64_t active_domain = 1;
    while (active_domain < logical_length) active_domain <<= 1;
    std::vector<Fr> point(point_size);
    for (std::size_t bit = 0; bit < point_size; ++bit) point[bit] = Fr(static_cast<int>(bit + 2));
    std::vector<Fr> values(static_cast<std::size_t>(full_domain), Fr(0));
    for (std::uint64_t index = 0; index < logical_length; ++index) values[static_cast<std::size_t>(index)] = Fr(static_cast<int>(5 + 3 * index));
    std::vector<G1> generators(static_cast<std::size_t>(full_domain));
    for (std::uint64_t index = 0; index < full_domain; ++index) hashAndMapToG1(generators[static_cast<std::size_t>(index)], generator_label + "/coordinate/" + std::to_string(index));
    hyrax::HyraxMleBasisView view;
    view.full_domain = full_domain;
    view.logical_length = logical_length;
    view.active_domain = active_domain;
    view.role = "smoke-public-zero-extension";
    view.basis_id = basis_id;
    view.point = point;
    view.generators.size = full_domain;
    view.generators.scheme = hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1;
    view.generators.at = [&generators](std::uint64_t index) { return generators.at(static_cast<std::size_t>(index)); };
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    view.generators.canonical_id = NativeProofStatement::basis_descriptor_digest_for(H, generator_label, basis_id, derivation_id, full_domain);
    const NativeProofStatement statement = make_statement(view.generators.canonical_id, full_domain);
    hyrax::HyraxScalarProvider provider{full_domain, [&values](std::uint64_t index) { return values.at(static_cast<std::size_t>(index)); }};
    const hyrax::PedersenMleCommitment commitment = hyrax::pedersen_commit_mle(view, provider);
    G1 expected_commitment;
    expected_commitment.clear();
    for (std::uint64_t index = 0; index < active_domain; ++index) expected_commitment += generators[static_cast<std::size_t>(index)] * values[static_cast<std::size_t>(index)];
    expected_commitment += H * commitment.blinding;
    if (expected_commitment != commitment.commitment) {
        std::cerr << "provider case commitment mismatch full=" << full_domain << " logical=" << logical_length << '\n';
        HYRAX_SMOKE_FAIL();
    }
    std::vector<Fr> coefficients = mle_basis(point);
    Fr raw_evaluation(0);
    for (std::uint64_t index = 0; index < active_domain; ++index) raw_evaluation += coefficients[static_cast<std::size_t>(index)] * values[static_cast<std::size_t>(index)];
    const zkgpt::native_pedersen::PedersenBasis scalar_basis = zkgpt::native_pedersen::derive_basis("masked-hyrax-smoke/scalar/" + std::to_string(full_domain));
    const zkgpt::native_pedersen::ScalarOpening raw_opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, raw_evaluation);
    const zkgpt::native_pedersen::ScalarOpening masked_opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, raw_evaluation + Fr(static_cast<int>(19 + full_domain)));
    NativeProofSession first("masked-hyrax-smoke-v2", statement);
    hyrax::MaskedMleOpening claim;
    std::string reason;
    if (!hyrax::open_masked_mle_with_claim(commitment, view, provider, masked_opening.value, first, "masked/opening", claim, &reason, true, "", &scalar_basis, &masked_opening, &raw_opening)) {
        std::cerr << "provider case opening failed full=" << full_domain << " logical=" << logical_length << ": " << reason << '\n';
        HYRAX_SMOKE_FAIL();
    }
    if (!verify_provider(first, commitment, claim, view, scalar_basis, reason)) {
        std::cerr << "provider case verification failed full=" << full_domain << " logical=" << logical_length << ": " << reason << '\n';
        HYRAX_SMOKE_FAIL();
    }
    if (claim.masked_evaluation == raw_evaluation || claim.raw_evaluation_opening.value != raw_evaluation || claim.masked_evaluation_opening.value != claim.masked_evaluation) {
        std::cerr << "provider case scalar claim mismatch full=" << full_domain << " logical=" << logical_length << '\n';
        HYRAX_SMOKE_FAIL();
    }
    std::size_t basis_headers = 0, response_vectors = 0, ipa_folds = 0;
    for (const NativeProofMessage& message : first.messages()) {
        if (message.label == "masked/opening/basis") {
            ++basis_headers;
            if (message.payload.size() > 1024) {
                std::cerr << "provider case oversized basis header bytes=" << message.payload.size() << '\n';
                HYRAX_SMOKE_FAIL();
            }
        }
        if (message.label == "masked/opening/response") {
            ++response_vectors;
            if (message.kind != NativeProofMessageKind::FieldVector || message.payload.size() > 512) {
                std::cerr << "provider case invalid compact response kind=" << static_cast<int>(message.kind)
                          << " bytes=" << message.payload.size() << '\n';
                HYRAX_SMOKE_FAIL();
            }
        }
        if (message.label.find("masked/opening/response-ipa/fold/") == 0) ++ipa_folds;
    }
    const bool tamper_rejected = reject_evaluation_transcript_tamper(first);
    if (basis_headers != 1 || response_vectors != 1 || ipa_folds != static_cast<std::size_t>(std::log2(active_domain)) || !tamper_rejected) {
        std::cerr << "provider case compact transcript mismatch full=" << full_domain << " logical=" << logical_length
                  << " basis=" << basis_headers << " responses=" << response_vectors << " folds=" << ipa_folds
                  << " expected_folds=" << static_cast<std::size_t>(std::log2(active_domain)) << " tamper_rejected=" << tamper_rejected << '\n';
        HYRAX_SMOKE_FAIL();
    }
    hyrax::MaskedMleOpening bad_claim = claim;
    bad_claim.masked_evaluation += Fr(1);
    if (verify_provider(first, commitment, bad_claim, view, scalar_basis, reason)) {
        std::cerr << "provider case accepted bad claim full=" << full_domain << " logical=" << logical_length << '\n';
        HYRAX_SMOKE_FAIL();
    }
    if (!point.empty()) {
        hyrax::HyraxMleBasisView bad_point = view;
        bad_point.point[0] += Fr(1);
        if (verify_provider(first, commitment, claim, bad_point, scalar_basis, reason)) {
            std::cerr << "provider case accepted bad point full=" << full_domain << " logical=" << logical_length << '\n';
            HYRAX_SMOKE_FAIL();
        }
    }
    const G1 Q = hyrax::derive_masked_response_ipa_base(basis_id);
    if (verify_provider(first, hyrax::PedersenMleCommitment{commitment.commitment + Q * Fr(7), H, commitment.blinding, basis_id}, claim, view, scalar_basis, reason) || !check_response_ipa_base_binding(first, H, Q)) {
        std::cerr << "provider case commitment/base binding regression failed full=" << full_domain << " logical=" << logical_length << '\n';
        HYRAX_SMOKE_FAIL();
    }
    if (logical_length < active_domain) {
        std::vector<Fr> bad_values = values;
        bad_values[static_cast<std::size_t>(logical_length)] = Fr(1);
        hyrax::HyraxScalarProvider bad_provider{full_domain, [&bad_values](std::uint64_t index) { return bad_values.at(static_cast<std::size_t>(index)); }};
        try { (void)hyrax::pedersen_commit_mle(view, bad_provider); std::cerr << "provider case accepted nonzero tail full=" << full_domain << " logical=" << logical_length << '\n'; HYRAX_SMOKE_FAIL(); } catch (const std::invalid_argument&) {}
    }
    NativeProofSession second("masked-hyrax-smoke-v2", statement);
    hyrax::MaskedMleOpening second_claim;
    if (!hyrax::open_masked_mle_with_claim(commitment, view, provider, masked_opening.value, second, "masked/opening", second_claim, &reason, true, "", &scalar_basis, &masked_opening, &raw_opening) || !verify_provider(second, commitment, second_claim, view, scalar_basis, reason) || first.digest() == second.digest()) {
        std::cerr << "provider case randomized rerun failed full=" << full_domain << " logical=" << logical_length << ": " << reason << '\n';
        HYRAX_SMOKE_FAIL();
    }
    return true;
}

bool run_legacy_wrapper_case(std::uint64_t size) {
    const std::string basis_id = "00000000000000000000000000000000000000000000000000000000000000bb";
    std::vector<Fr> point(static_cast<std::size_t>(std::log2(size)));
    for (std::size_t bit = 0; bit < point.size(); ++bit) point[bit] = Fr(static_cast<int>(bit + 2));
    std::vector<Fr> values(static_cast<std::size_t>(size));
    for (std::uint64_t index = 0; index < size; ++index) values[static_cast<std::size_t>(index)] = Fr(static_cast<int>(5 + 3 * index));
    std::vector<G1> generators(static_cast<std::size_t>(size));
    for (std::size_t index = 0; index < generators.size(); ++index) hashAndMapToG1(generators[index], "masked-hyrax-smoke/legacy/" + std::to_string(index));
    const hyrax::PedersenMleCommitment commitment = hyrax::pedersen_commit_mle(generators, values, basis_id);
    const std::vector<Fr> basis = mle_basis(point);
    const NativeProofStatement statement = make_statement(NativeProofStatement::basis_digest_for(commitment.blinding_generator, generators, basis, generators.size()), generators.size());
    NativeProofSession session("masked-hyrax-smoke-legacy-v1", statement);
    hyrax::MaskedMleOpening claim;
    std::string reason;
    if (size == 1) {
        const auto scalar_basis = zkgpt::native_pedersen::derive_basis("masked-hyrax-smoke/legacy-scalar");
        zkgpt::native_pedersen::ScalarOpening scalar_opening;
        if (!hyrax::make_mle_evaluation_opening(scalar_basis, point, values, scalar_opening, &reason) || scalar_opening.value != values[0]) HYRAX_SMOKE_FAIL();
    }
    return hyrax::open_masked_mle_with_session(commitment, generators, point, values, session, "masked/legacy", claim, &reason) && hyrax::verify_masked_mle_opening_with_session(commitment.commitment, claim.evaluation_mask_commitment, claim.masked_evaluation, basis_id, generators, point, session, "masked/legacy", &reason);
}

bool run_tensor_case() {
    const std::uint64_t full_domain = 16, logical_length = 5;
    const std::string basis_id = "00000000000000000000000000000000000000000000000000000000000000cd";
    const std::string generator_label = "masked-hyrax-smoke/tensor-column";
    const std::string derivation_id = "hashAndMapToG1(canonical_basis_label+/coordinate/+canonical_absolute_index_decimal);absolute-index-u64;v1";
    std::vector<Fr> values{Fr(7), Fr(11), Fr(13), Fr(17), Fr(19)};
    std::vector<Fr> point{Fr(2), Fr(3), Fr(5), Fr(7)};
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    std::vector<G1> column_generators(2);
    for (std::uint64_t index = 0; index < column_generators.size(); ++index) hashAndMapToG1(column_generators[static_cast<std::size_t>(index)], generator_label + "/coordinate/" + std::to_string(index));
    hyrax::HyraxTensorMleView view;
    view.full_domain = full_domain;
    view.logical_length = logical_length;
    view.active_domain = 8;
    view.rows = 4;
    view.columns = 2;
    view.role = "canonical/Aux";
    view.basis_id = basis_id;
    view.column_generators.size = column_generators.size();
    view.column_generators.canonical_id = NativeProofStatement::basis_descriptor_digest_for(H, generator_label, basis_id, derivation_id, full_domain);
    view.column_generators.scheme = hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1;
    view.column_generators.at = [&column_generators](std::uint64_t index) { return column_generators.at(static_cast<std::size_t>(index)); };
    hyrax::HyraxScalarProvider provider{logical_length, [&values](std::uint64_t index) { return values.at(static_cast<std::size_t>(index)); }};
    hyrax::HyraxTensorMleCommitment commitment;
    hyrax::HyraxTensorMleProverState state;
    std::string reason;
    if (!hyrax::commit_hyrax_tensor_mle(view, provider, commitment, state, &reason)) HYRAX_SMOKE_FAIL();
    std::vector<Fr> dense_values(static_cast<std::size_t>(full_domain), Fr(0));
    for (std::uint64_t index = 0; index < logical_length; ++index) dense_values[static_cast<std::size_t>(index)] = values[static_cast<std::size_t>(index)];
    const std::vector<Fr> dense_weights = mle_basis(point);
    Fr dense_evaluation(0);
    for (std::uint64_t index = 0; index < full_domain; ++index) dense_evaluation += dense_weights[static_cast<std::size_t>(index)] * dense_values[static_cast<std::size_t>(index)];
    NativeProofStatement statement = make_statement(view.column_generators.canonical_id, full_domain);
    statement.aux_digest = commitment.root_digest;
    const zkgpt::native_pedersen::PedersenBasis scalar_basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + statement.root());
    const auto raw_opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, dense_evaluation);
    const auto masked_opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, dense_evaluation + Fr(41));
    NativeProofSession session("masked-hyrax-smoke-tensor-v2", statement);
    hyrax::MaskedMleOpening claim;
    if (!hyrax::open_hyrax_tensor_mle_with_claim(view, commitment, state, point, provider, masked_opening.value, session, "tensor/opening", claim, &reason, "", &scalar_basis, &masked_opening, &raw_opening) ||
        claim.raw_evaluation_opening.value != dense_evaluation || claim.masked_evaluation_opening.value != masked_opening.value ||
        !hyrax::verify_hyrax_tensor_mle_opening_with_session(view, commitment, point, claim.evaluation_mask_commitment, claim.masked_evaluation, session, "tensor/opening", &reason, false, "", &scalar_basis, &claim.raw_evaluation_opening.commitment, &claim.masked_evaluation_opening.commitment)) HYRAX_SMOKE_FAIL();
    const auto *public_positions = session.message_positions("tensor/opening/tensor/public");
    const auto *column_positions = session.message_positions("tensor/opening/column/basis");
    if (public_positions == nullptr || public_positions->size() != 1 || column_positions == nullptr || column_positions->size() != 1 || public_positions->front() + 1 != column_positions->front()) HYRAX_SMOKE_FAIL();
    hyrax::HyraxTensorMleCommitment bad_commitment = commitment;
    bad_commitment.row_commitments[1] += H;
    if (hyrax::verify_hyrax_tensor_mle_opening_with_session(view, bad_commitment, point, claim.evaluation_mask_commitment, claim.masked_evaluation, session, "tensor/opening", &reason, false, "", &scalar_basis, &claim.raw_evaluation_opening.commitment, &claim.masked_evaluation_opening.commitment)) HYRAX_SMOKE_FAIL();
    std::vector<Fr> bad_point = point;
    bad_point[3] += Fr(1);
    if (hyrax::verify_hyrax_tensor_mle_opening_with_session(view, commitment, bad_point, claim.evaluation_mask_commitment, claim.masked_evaluation, session, "tensor/opening", &reason, false, "", &scalar_basis, &claim.raw_evaluation_opening.commitment, &claim.masked_evaluation_opening.commitment)) HYRAX_SMOKE_FAIL();
    std::size_t prover_position = session.checkpoint();
    const std::size_t tail_start = prover_position;
    std::vector<Fr> beta, tail_point;
    NativeMaskedSumcheckResult tail_sumcheck;
    hyrax::MaskedMleOpening tail_opening;
    std::vector<std::vector<Fr>> tail_round_trace;
    if (!hyrax::prove_hyrax_tensor_zero_tail_certificate_with_session(view, commitment, state, provider, session, prover_position, scalar_basis, beta, tail_point, tail_sumcheck, tail_opening, &reason, &tail_round_trace) || prover_position != session.checkpoint() || beta.size() != 3 || tail_point.size() != 4 || tail_sumcheck.challenges.size() != 3 || tail_opening.raw_evaluation_opening.value != tail_sumcheck.unmasked_terminal_openings[0].value || !tensor_tail_rounds_match_dense_reference(values, logical_length, 8, beta, tail_sumcheck.challenges, tail_round_trace)) HYRAX_SMOKE_FAIL();
    std::vector<Fr> dense_private_weights = mle_basis(tail_sumcheck.challenges);
    Fr dense_private_evaluation(0), dense_public_evaluation(0);
    for (std::uint64_t index = 0; index < 8; ++index) {
        if (index < logical_length) dense_private_evaluation += values[static_cast<std::size_t>(index)] * dense_private_weights[static_cast<std::size_t>(index)];
        if (index >= logical_length) dense_public_evaluation += dense_private_weights[static_cast<std::size_t>(index)] *
            [&]() { Fr weight(1); for (std::size_t bit = 0; bit < beta.size(); ++bit) weight *= ((index >> bit) & 1U) ? beta[bit] : Fr(1) - beta[bit]; return weight; }();
    }
    if (tail_sumcheck.unmasked_terminal_openings[0].value != dense_private_evaluation ||
        tail_sumcheck.unmasked_claim_opening.value != dense_private_evaluation * dense_public_evaluation) HYRAX_SMOKE_FAIL();
    const auto *tail_positions = session.message_positions("tensor/aux/tail/public");
    if (tail_positions == nullptr || tail_positions->size() != 1) HYRAX_SMOKE_FAIL();
    std::size_t verifier_position = tail_positions->front();
    std::vector<Fr> verified_beta, verified_point;
    NativeMaskedSumcheckResult verified_sumcheck;
    if (!hyrax::verify_hyrax_tensor_zero_tail_certificate_with_session(view, commitment, session, verifier_position, scalar_basis, verified_beta,
                                                                       verified_point, verified_sumcheck, &reason) ||
        verifier_position != prover_position || verified_beta != beta || verified_point != tail_point) HYRAX_SMOKE_FAIL();
    hyrax::HyraxTensorMleCommitment bad_tail_commitment = commitment;
    bad_tail_commitment.root_digest[0] = bad_tail_commitment.root_digest[0] == '0' ? '1' : '0';
    verifier_position = tail_positions->front();
    if (hyrax::verify_hyrax_tensor_zero_tail_certificate_with_session(view, bad_tail_commitment, session, verifier_position, scalar_basis, verified_beta,
                                                                       verified_point, verified_sumcheck, &reason) ||
        !reject_tensor_tail_raw_claim_tamper(session, view, commitment, tail_start, scalar_basis)) HYRAX_SMOKE_FAIL();
    return true;
}

bool run_tensor_tail_certificate_case(std::uint64_t full_domain, std::uint64_t logical_length, const std::string &role) {
    const std::string basis_id = "00000000000000000000000000000000000000000000000000000000000000de";
    const std::string generator_label = "masked-hyrax-smoke/tail-column";
    const std::string derivation_id = "hashAndMapToG1(canonical_basis_label+/coordinate/+canonical_absolute_index_decimal);absolute-index-u64;v1";
    std::uint64_t active_domain = 1;
    while (active_domain < logical_length) active_domain <<= 1;
    std::size_t full_bits = 0, active_bits = 0;
    for (std::uint64_t size = full_domain; size > 1; size >>= 1) ++full_bits;
    for (std::uint64_t size = active_domain; size > 1; size >>= 1) ++active_bits;
    const std::uint64_t columns = std::uint64_t(1) << (active_bits / 2), rows = active_domain / columns;
    std::vector<Fr> values(static_cast<std::size_t>(logical_length));
    for (std::uint64_t index = 0; index < logical_length; ++index) values[static_cast<std::size_t>(index)] = Fr(static_cast<int>(index + 3));
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    std::vector<G1> column_generators(static_cast<std::size_t>(columns));
    for (std::uint64_t index = 0; index < columns; ++index) hashAndMapToG1(column_generators[static_cast<std::size_t>(index)], generator_label + "/coordinate/" + std::to_string(index));
    hyrax::HyraxTensorMleView view;
    view.full_domain = full_domain;
    view.logical_length = logical_length;
    view.active_domain = active_domain;
    view.rows = rows;
    view.columns = columns;
    view.role = role;
    view.basis_id = basis_id;
    view.column_generators.size = columns;
    view.column_generators.canonical_id = NativeProofStatement::basis_descriptor_digest_for(H, generator_label, basis_id, derivation_id, full_domain);
    view.column_generators.scheme = hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1;
    view.column_generators.at = [&column_generators](std::uint64_t index) { return column_generators.at(static_cast<std::size_t>(index)); };
    const hyrax::HyraxScalarProvider provider{logical_length, [&values](std::uint64_t index) { return values.at(static_cast<std::size_t>(index)); }};
    hyrax::HyraxTensorMleCommitment commitment;
    hyrax::HyraxTensorMleProverState state;
    std::string reason;
    if (!hyrax::commit_hyrax_tensor_mle(view, provider, commitment, state, &reason)) HYRAX_SMOKE_FAIL();
    NativeProofStatement statement = make_statement(view.column_generators.canonical_id, full_domain);
    if (role == "canonical/Aux") statement.aux_digest = commitment.root_digest;
    else if (role == "canonical/L_qry") statement.l_digest = commitment.root_digest;
    else HYRAX_SMOKE_FAIL();
    const zkgpt::native_pedersen::PedersenBasis scalar_basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + statement.root());
    NativeProofSession session("masked-hyrax-smoke-tail-v1", statement);
    std::size_t prover_position = session.checkpoint();
    std::vector<Fr> beta, terminal_point;
    NativeMaskedSumcheckResult sumcheck;
    hyrax::MaskedMleOpening terminal_opening;
    std::vector<std::vector<Fr>> round_trace;
    if (!hyrax::prove_hyrax_tensor_zero_tail_certificate_with_session(view, commitment, state, provider, session, prover_position, scalar_basis, beta, terminal_point, sumcheck, terminal_opening, &reason, &round_trace)) {
        std::cerr << "tail proof rejected full=" << full_domain << " logical=" << logical_length << " active=" << active_domain << ": " << reason << '\n';
        HYRAX_SMOKE_FAIL();
    }
    if (logical_length == active_domain) {
        if (prover_position != session.checkpoint() || !session.messages().empty() || !beta.empty() || !terminal_point.empty() ||
            !sumcheck.challenges.empty() || !sumcheck.unmasked_terminal_openings.empty() || !round_trace.empty()) {
            std::cerr << "empty tail prover emitted work: checkpoint=" << session.checkpoint() << " position=" << prover_position
                      << " beta=" << beta.size() << " point=" << terminal_point.size() << " challenges=" << sumcheck.challenges.size()
                      << " terminal_openings=" << sumcheck.unmasked_terminal_openings.size() << " trace=" << round_trace.size() << '\n';
            HYRAX_SMOKE_FAIL();
        }
        std::size_t verifier_position = 0;
        std::vector<Fr> verified_beta, verified_point;
        NativeMaskedSumcheckResult verified_sumcheck;
        if (!hyrax::verify_hyrax_tensor_zero_tail_certificate_with_session(view, commitment, session, verifier_position, scalar_basis,
                                                                           verified_beta, verified_point, verified_sumcheck, &reason)) {
            std::cerr << "empty tail verifier rejected vacuous proof: " << reason << '\n';
            HYRAX_SMOKE_FAIL();
        }
        if (verifier_position != 0 || !verified_beta.empty() || !verified_point.empty() || !verified_sumcheck.challenges.empty() ||
            !verified_sumcheck.unmasked_terminal_openings.empty() || !session.messages().empty()) {
            std::cerr << "empty tail verifier consumed or returned work: position=" << verifier_position << " beta=" << verified_beta.size()
                      << " point=" << verified_point.size() << " challenges=" << verified_sumcheck.challenges.size()
                      << " terminal_openings=" << verified_sumcheck.unmasked_terminal_openings.size() << '\n';
            HYRAX_SMOKE_FAIL();
        }
        return true;
    }
    if (prover_position != session.checkpoint()) {
        std::cerr << "tail prover position mismatch position=" << prover_position << " checkpoint=" << session.checkpoint() << '\n';
        HYRAX_SMOKE_FAIL();
    }
    if (beta.size() != active_bits || terminal_point.size() != full_bits || sumcheck.challenges.size() != active_bits || sumcheck.unmasked_terminal_openings.size() != 1) {
        std::cerr << "tail dimensions mismatch active_bits=" << active_bits << " beta=" << beta.size() << " point=" << terminal_point.size()
                  << " full_bits=" << full_bits << " challenges=" << sumcheck.challenges.size()
                  << " terminal_openings=" << sumcheck.unmasked_terminal_openings.size() << '\n';
        HYRAX_SMOKE_FAIL();
    }
    if (terminal_opening.raw_evaluation_opening.value != sumcheck.unmasked_terminal_openings[0].value) {
        std::cerr << "tail terminal opening mismatch raw=" << terminal_opening.raw_evaluation_opening.value.getStr(16)
                  << " sumcheck=" << sumcheck.unmasked_terminal_openings[0].value.getStr(16) << '\n';
        HYRAX_SMOKE_FAIL();
    }
    if (!tensor_tail_rounds_match_dense_reference(values, logical_length, active_domain, beta, sumcheck.challenges, round_trace)) {
        std::cerr << "tail dense round trace mismatch rounds=" << round_trace.size() << " logical=" << logical_length << " active=" << active_domain << '\n';
        HYRAX_SMOKE_FAIL();
    }
    const std::vector<Fr> point_weights = mle_basis(sumcheck.challenges);
    Fr private_eval(0), public_eval(0);
    for (std::uint64_t index = 0; index < active_domain; ++index) {
        if (index < logical_length) private_eval += values[static_cast<std::size_t>(index)] * point_weights[static_cast<std::size_t>(index)];
        if (index >= logical_length) {
            Fr beta_weight(1);
            for (std::size_t bit = 0; bit < beta.size(); ++bit) beta_weight *= ((index >> bit) & 1U) ? beta[bit] : Fr(1) - beta[bit];
            public_eval += point_weights[static_cast<std::size_t>(index)] * beta_weight;
        }
    }
    if (private_eval != sumcheck.unmasked_terminal_openings[0].value || sumcheck.unmasked_claim_opening.value != private_eval * public_eval) HYRAX_SMOKE_FAIL();
    std::size_t verifier_position = 0;
    std::vector<Fr> verified_beta, verified_point;
    NativeMaskedSumcheckResult verified_sumcheck;
    if (!hyrax::verify_hyrax_tensor_zero_tail_certificate_with_session(view, commitment, session, verifier_position, scalar_basis, verified_beta,
                                                                       verified_point, verified_sumcheck, &reason) ||
        verifier_position != prover_position || verified_beta != beta || verified_point != terminal_point) HYRAX_SMOKE_FAIL();
    return true;
}

bool run_tensor_zero_interval_case(std::uint64_t full_domain, std::uint64_t logical_length, const std::vector<hyrax::HyraxTensorZeroInterval> &intervals, bool nonzero_interval_witness = false) {
    const std::string basis_id = "00000000000000000000000000000000000000000000000000000000000000de";
    const std::string generator_label = "masked-hyrax-smoke/zero-interval-column";
    const std::string derivation_id = "hashAndMapToG1(canonical_basis_label+/coordinate/+canonical_absolute_index_decimal);absolute-index-u64;v1";
    std::uint64_t active_domain = 1;
    while (active_domain < logical_length) active_domain <<= 1;
    std::size_t full_bits = 0, active_bits = 0;
    for (std::uint64_t size = full_domain; size > 1; size >>= 1) ++full_bits;
    for (std::uint64_t size = active_domain; size > 1; size >>= 1) ++active_bits;
    const std::uint64_t columns = std::uint64_t(1) << (active_bits / 2), rows = active_domain / columns;
    std::vector<Fr> values(static_cast<std::size_t>(logical_length));
    for (std::uint64_t index = 0; index < logical_length; ++index) values[static_cast<std::size_t>(index)] = Fr(static_cast<int>(index + 7));
    if (!nonzero_interval_witness) for (const auto &interval : intervals) for (std::uint64_t index = interval.begin; index < interval.end && index < logical_length; ++index) values[static_cast<std::size_t>(index)] = Fr(0);
    const G1 H = hyrax::derive_pedersen_blinding_generator(basis_id);
    std::vector<G1> column_generators(static_cast<std::size_t>(columns));
    for (std::uint64_t index = 0; index < columns; ++index) hashAndMapToG1(column_generators[static_cast<std::size_t>(index)], generator_label + "/coordinate/" + std::to_string(index));
    hyrax::HyraxTensorMleView view;
    view.full_domain = full_domain;
    view.logical_length = logical_length;
    view.active_domain = active_domain;
    view.rows = rows;
    view.columns = columns;
    view.role = "canonical/L_qry";
    view.basis_id = basis_id;
    view.column_generators.size = columns;
    view.column_generators.canonical_id = NativeProofStatement::basis_descriptor_digest_for(H, generator_label, basis_id, derivation_id, full_domain);
    view.column_generators.scheme = hyrax::HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1;
    view.column_generators.at = [&column_generators](std::uint64_t index) { return column_generators.at(static_cast<std::size_t>(index)); };
    const hyrax::HyraxScalarProvider provider{logical_length, [&values](std::uint64_t index) { return values.at(static_cast<std::size_t>(index)); }};
    hyrax::HyraxTensorMleCommitment commitment;
    hyrax::HyraxTensorMleProverState state;
    std::string reason;
    if (!hyrax::commit_hyrax_tensor_mle(view, provider, commitment, state, &reason)) HYRAX_SMOKE_FAIL();
    NativeProofStatement statement = make_statement(view.column_generators.canonical_id, full_domain);
    statement.l_digest = commitment.root_digest;
    const zkgpt::native_pedersen::PedersenBasis scalar_basis = zkgpt::native_pedersen::derive_basis("zkgpt/masked-sumcheck/v2/" + statement.root());
    NativeProofSession session("masked-hyrax-smoke-zero-interval-v1", statement);
    std::size_t prover_position = session.checkpoint();
    std::vector<hyrax::HyraxTensorZeroIntervalCertificate> certificates;
    std::vector<std::vector<std::vector<Fr>>> traces;
    const bool proved = hyrax::prove_hyrax_tensor_zero_interval_certificates_with_session(view, commitment, state, provider, intervals, session, prover_position, scalar_basis, certificates, &reason, &traces);
    if (nonzero_interval_witness) {
        if (!proved) return true;
        if (prover_position != session.checkpoint()) HYRAX_SMOKE_FAIL();
        const auto *public_positions = session.message_positions("tensor/L_qry/zero-gap/public");
        if (public_positions == nullptr || public_positions->size() != 1) HYRAX_SMOKE_FAIL();
        std::size_t verifier_position = public_positions->front();
        std::vector<hyrax::HyraxTensorZeroIntervalCertificate> rejected_certificates;
        return !hyrax::verify_hyrax_tensor_zero_interval_certificates_with_session(view, commitment, intervals, session, verifier_position, scalar_basis, rejected_certificates, &reason);
    }
    if (!proved || prover_position != session.checkpoint() || certificates.size() != intervals.size() || traces.size() != intervals.size()) HYRAX_SMOKE_FAIL();
    for (std::size_t item = 0; item < intervals.size(); ++item) {
        const auto &certificate = certificates[item];
        if (certificate.begin != intervals[item].begin || certificate.end != intervals[item].end || certificate.beta.size() != active_bits || certificate.sumcheck_challenges.size() != active_bits || certificate.full_terminal_point.size() != full_bits || !tensor_zero_interval_rounds_match_dense_reference(values, logical_length, active_domain, intervals[item], certificate.beta, certificate.sumcheck_challenges, traces[item])) HYRAX_SMOKE_FAIL();
        for (std::size_t bit = 0; bit < active_bits; ++bit) if (certificate.full_terminal_point[bit] != certificate.sumcheck_challenges[bit]) HYRAX_SMOKE_FAIL();
        for (std::size_t bit = active_bits; bit < full_bits; ++bit) if (certificate.full_terminal_point[bit] != Fr(0)) HYRAX_SMOKE_FAIL();
    }
    const auto *public_positions = session.message_positions("tensor/L_qry/zero-gap/public");
    if (public_positions == nullptr || public_positions->size() != 1) HYRAX_SMOKE_FAIL();
    std::size_t verifier_position = public_positions->front();
    std::vector<hyrax::HyraxTensorZeroIntervalCertificate> verified_certificates;
    if (!hyrax::verify_hyrax_tensor_zero_interval_certificates_with_session(view, commitment, intervals, session, verifier_position, scalar_basis, verified_certificates, &reason) || verifier_position != prover_position || verified_certificates.size() != certificates.size()) HYRAX_SMOKE_FAIL();
    for (std::size_t item = 0; item < certificates.size(); ++item) if (verified_certificates[item].beta != certificates[item].beta || verified_certificates[item].full_terminal_point != certificates[item].full_terminal_point || verified_certificates[item].sumcheck_challenges != certificates[item].sumcheck_challenges) HYRAX_SMOKE_FAIL();
    if (!intervals.empty()) {
        std::vector<hyrax::HyraxTensorZeroInterval> wrong_intervals = intervals;
        --wrong_intervals[0].end;
        verifier_position = public_positions->front();
        if (hyrax::verify_hyrax_tensor_zero_interval_certificates_with_session(view, commitment, wrong_intervals, session, verifier_position, scalar_basis, verified_certificates, &reason)) HYRAX_SMOKE_FAIL();
    }
    std::vector<NativeProofMessage> tampered_messages = session.messages();
    for (NativeProofMessage &message : tampered_messages) if (message.label == "tensor/L_qry/zero-gap/public") { message.payload += "|tamper"; break; }
    NativeProofSession tampered_session;
    if (NativeProofSession::from_messages(session.domain(), session.statement(), tampered_messages, tampered_session, &reason)) {
        verifier_position = public_positions->front();
        if (hyrax::verify_hyrax_tensor_zero_interval_certificates_with_session(view, commitment, intervals, tampered_session, verifier_position, scalar_basis, verified_certificates, &reason)) HYRAX_SMOKE_FAIL();
    }
    if (!intervals.empty()) {
        tampered_messages = session.messages();
        bool changed_interval_proof = false;
        for (NativeProofMessage &message : tampered_messages) if (message.label == "tensor/L_qry/zero-gap/0/opening/column/claim") { message.payload += "|tamper"; changed_interval_proof = true; break; }
        NativeProofSession tampered_interval_session;
        if (!changed_interval_proof) HYRAX_SMOKE_FAIL();
        if (NativeProofSession::from_messages(session.domain(), session.statement(), tampered_messages, tampered_interval_session, &reason)) {
            verifier_position = public_positions->front();
            if (hyrax::verify_hyrax_tensor_zero_interval_certificates_with_session(view, commitment, intervals, tampered_interval_session, verifier_position, scalar_basis, verified_certificates, &reason)) HYRAX_SMOKE_FAIL();
        }
    }
    return true;
}

}

int main() {
    initPairing(mcl::BN254);
    const auto check_case = [](const char *name, const auto &run) {
        try {
            if (!run()) {
                std::cerr << "native masked Hyrax compact opening check failed at " << name << '\n';
                HYRAX_SMOKE_FAIL();
            }
        } catch (const std::exception &exception) {
            std::cerr << "native masked Hyrax exception at " << name << ": " << exception.what() << '\n';
            HYRAX_SMOKE_FAIL();
        } catch (...) {
            std::cerr << "native masked Hyrax unknown exception at " << name << '\n';
            HYRAX_SMOKE_FAIL();
        }
        return true;
    };
    if (!check_case("provider(1,1)", [] { return run_provider_case(1, 1); })) return 1;
    if (!check_case("provider(8,1)", [] { return run_provider_case(8, 1); })) return 1;
    if (!check_case("provider(8,3)", [] { return run_provider_case(8, 3); })) return 1;
    if (!check_case("provider(16,5)", [] { return run_provider_case(16, 5); })) return 1;
    if (!check_case("provider(32,9)", [] { return run_provider_case(32, 9); })) return 1;
    if (!check_case("provider(64,64)", [] { return run_provider_case(64, 64); })) return 1;
    if (!check_case("legacy(1)", [] { return run_legacy_wrapper_case(1); })) return 1;
    if (!check_case("legacy(4)", [] { return run_legacy_wrapper_case(4); })) return 1;
    if (!check_case("tensor", [] { return run_tensor_case(); })) return 1;
    if (!check_case("tail(16,6)", [] { return run_tensor_tail_certificate_case(16, 6, "canonical/L_qry"); })) return 1;
    if (!check_case("tail(32,9)", [] { return run_tensor_tail_certificate_case(32, 9, "canonical/L_qry"); })) return 1;
    if (!check_case("tail(2^37,5)", [] { return run_tensor_tail_certificate_case(std::uint64_t(1) << 37, 5, "canonical/Aux"); })) return 1;
    if (!check_case("tail(8,8)", [] { return run_tensor_tail_certificate_case(8, 8, "canonical/L_qry"); })) return 1;
    if (!check_case("tail(1,1)", [] { return run_tensor_tail_certificate_case(1, 1, "canonical/Aux"); })) return 1;
    if (!check_case("zero-interval(8,8)", [] { return run_tensor_zero_interval_case(8, 8, {{2, 6}}); })) return 1;
    if (!check_case("zero-interval(16,16)", [] { return run_tensor_zero_interval_case(16, 16, {{0, 4}, {6, 10}, {12, 16}}); })) return 1;
    if (!check_case("zero-interval(2^37,5)", [] { return run_tensor_zero_interval_case(std::uint64_t(1) << 37, 5, {{1, 3}, {4, 5}}); })) return 1;
    if (!check_case("zero-interval(empty)", [] { return run_tensor_zero_interval_case(8, 8, {}); })) return 1;
    if (!check_case("zero-interval(1,1)", [] { return run_tensor_zero_interval_case(1, 1, {{0, 1}}); })) return 1;
    if (!check_case("zero-interval(nonzero witness)", [] { return run_tensor_zero_interval_case(8, 8, {{2, 6}}, true); })) return 1;
    std::cout << "native_masked_hyrax_smoke: PASS\n";
    return 0;
}
