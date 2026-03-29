#include "gatzk/crypto/kzg.hpp"

#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::crypto {
namespace {

using algebra::FieldElement;

std::vector<FieldElement> folding_powers(const FieldElement& challenge, std::size_t count) {
    std::vector<FieldElement> out(count, FieldElement::one());
    for (std::size_t i = 1; i < count; ++i) {
        out[i] = out[i - 1] * challenge;
    }
    return out;
}

FieldElement vanishing_eval(const std::vector<FieldElement>& points, const FieldElement& x) {
    FieldElement out = FieldElement::one();
    for (const auto& point : points) {
        out *= (x - point);
    }
    return out;
}

FieldElement safe_inverse(const FieldElement& value, const std::string& label) {
    if (value == FieldElement::zero()) {
        throw std::runtime_error("cannot invert zero in " + label);
    }
    return value.inv();
}

G1Point scaled_generator(const G1Point& generator, const FieldElement& scalar) {
    return g1_mul(generator, scalar);
}

std::vector<FieldElement> interpolation_basis_at(
    const std::vector<FieldElement>& points,
    const FieldElement& x) {
    std::vector<FieldElement> out(points.size(), FieldElement::zero());
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (points[i] == x) {
            out[i] = FieldElement::one();
            return out;
        }
    }

    for (std::size_t i = 0; i < points.size(); ++i) {
        FieldElement basis = FieldElement::one();
        for (std::size_t j = 0; j < points.size(); ++j) {
            if (i == j) {
                continue;
            }
            basis *= (x - points[j]) / (points[i] - points[j]);
        }
        out[i] = basis;
    }
    return out;
}

FieldElement interpolate_with_basis(
    const std::vector<FieldElement>& basis_at_x,
    const std::vector<FieldElement>& values) {
    FieldElement out = FieldElement::zero();
    for (std::size_t i = 0; i < basis_at_x.size(); ++i) {
        out += basis_at_x[i] * values[i];
    }
    return out;
}

struct BatchOpeningPrecompute {
    FieldElement vanishing_at_tau = FieldElement::one();
    std::vector<FieldElement> interpolation_basis_at_tau;
    std::vector<FieldElement> folding_powers;
};

struct DomainCommitWeights {
    std::optional<std::size_t> direct_index;
    std::vector<mcl::Fr> native_weights;
};

BatchOpeningPrecompute prepare_batch_opening(
    const std::vector<FieldElement>& points,
    const FieldElement& folding_challenge,
    const KZGKeyPair& key,
    std::size_t commitment_count) {
    // Batch opening semantics stay unchanged: we still fold the same
    // commitments over the same point set. This helper only memoizes the
    // point-set-dependent pieces shared by all folded terms.
    BatchOpeningPrecompute out;
    out.vanishing_at_tau = vanishing_eval(points, key.tau);
    out.interpolation_basis_at_tau = interpolation_basis_at(points, key.tau);
    out.folding_powers = folding_powers(folding_challenge, commitment_count);
    return out;
}

FieldElement fold_commitment_scalar(
    const std::vector<Commitment>& commitments,
    const std::vector<std::vector<FieldElement>>& claimed_values,
    const BatchOpeningPrecompute& precompute) {
    FieldElement folded = FieldElement::zero();
    for (std::size_t i = 0; i < commitments.size(); ++i) {
        const auto interpolation_at_tau = interpolate_with_basis(precompute.interpolation_basis_at_tau, claimed_values[i]);
        folded += precompute.folding_powers[i] * (commitments[i].tau_evaluation - interpolation_at_tau);
    }
    return folded;
}

struct ExternalFoldPrecompute {
    FieldElement accumulator_eval = FieldElement::one();
    std::vector<FieldElement> denominator_weights;
    std::vector<FieldElement> folding_powers;
};

ExternalFoldPrecompute prepare_external_fold(
    const std::vector<FieldElement>& points,
    const FieldElement& folding_challenge,
    const KZGKeyPair& key) {
    // The external fold witness uses the same accumulator polynomial as before.
    // We cache the vanishing value, denominator weights and rho powers because
    // they are reused term-by-term within one witness construction.
    ExternalFoldPrecompute out;
    out.accumulator_eval = vanishing_eval(points, key.tau);
    out.denominator_weights.reserve(points.size());
    for (const auto& point : points) {
        out.denominator_weights.push_back(
            out.accumulator_eval * safe_inverse(key.tau - point, "external fold denominator"));
    }
    out.folding_powers = folding_powers(folding_challenge, points.size());
    return out;
}

G1Point fold_commitments(
    const std::vector<Commitment>& commitments,
    const std::vector<std::vector<FieldElement>>& claimed_values,
    const BatchOpeningPrecompute& precompute,
    const KZGKeyPair& key) {
    return g1_mul(key.g1_generator, fold_commitment_scalar(commitments, claimed_values, precompute));
}

FieldElement fold_external_scalar(
    const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
    const ExternalFoldPrecompute& precompute) {
    FieldElement folded = FieldElement::zero();
    for (std::size_t i = 0; i < commitments_and_values.size(); ++i) {
        const auto& [commitment, value] = commitments_and_values[i];
        folded += precompute.folding_powers[i]
            * (commitment.tau_evaluation - value)
            * precompute.denominator_weights[i];
    }
    return folded;
}

G1Point fold_external_commitments(
    const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
    const ExternalFoldPrecompute& precompute,
    const KZGKeyPair& key,
    FieldElement* accumulator_eval_out) {
    if (accumulator_eval_out != nullptr) {
        *accumulator_eval_out = precompute.accumulator_eval;
    }
    return g1_mul(key.g1_generator, fold_external_scalar(commitments_and_values, precompute));
}

}  // namespace

std::size_t serialized_size(const Commitment& commitment) {
    return serialized_size(commitment.point);
}

KZGKeyPair KZG::setup(std::uint64_t seed) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::uint64_t, KZGKeyPair> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(seed); it != cache.end()) {
            return it->second;
        }
    }

    const auto tau = FieldElement(seed + 17);

    G1Point g1_generator = g1_zero();
    G2Point g2_generator = g2_zero();

    mcl::bn::hashAndMapToG1(g1_generator.value, "gatzk:kzg:g1");
    mcl::bn::hashAndMapToG2(g2_generator.value, "gatzk:kzg:g2");

    const KZGKeyPair key{
        .tau = tau,
        .g1_generator = g1_generator,
        .g2_one = g2_generator,
        .g2_tau = g2_mul(g2_generator, tau),
        .g2_one_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_generator)),
        .g2_tau_prepared = std::make_shared<PreparedG2>(prepare_g2(g2_mul(g2_generator, tau))),
    };
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.emplace(seed, key);
    }
    return key;
}

Commitment KZG::commit(const std::string& name, const algebra::Polynomial& polynomial, const KZGKeyPair& key) {
    const auto tau_evaluation = polynomial.evaluate(key.tau);
    return commit_tau_evaluation(name, tau_evaluation, key);
}

Commitment KZG::commit_tau_evaluation(
    const std::string& name,
    const algebra::FieldElement& tau_evaluation,
    const KZGKeyPair& key) {
    return Commitment{
        .name = name,
        .point = scaled_generator(key.g1_generator, tau_evaluation),
        .tau_evaluation = tau_evaluation,
    };
}

std::vector<Commitment> KZG::commit_tau_evaluation_batch(
    const std::vector<std::pair<std::string, algebra::FieldElement>>& named_tau_evaluations,
    const KZGKeyPair& key) {
    std::vector<Commitment> out;
    out.reserve(named_tau_evaluations.size());
    if (named_tau_evaluations.empty()) {
        return out;
    }

    std::vector<FieldElement> tau_evaluations;
    tau_evaluations.reserve(named_tau_evaluations.size());
    for (const auto& [name, tau_evaluation] : named_tau_evaluations) {
        (void)name;
        tau_evaluations.push_back(tau_evaluation);
    }
    const auto points = g1_mul_same_base_batch(key.g1_generator, tau_evaluations);
    for (std::size_t i = 0; i < named_tau_evaluations.size(); ++i) {
        out.push_back(Commitment{
            .name = named_tau_evaluations[i].first,
            .point = points[i],
            .tau_evaluation = named_tau_evaluations[i].second,
        });
    }
    return out;
}

std::vector<Commitment> KZG::commit_batch(
    const std::vector<std::pair<std::string, const algebra::Polynomial*>>& named_polynomials,
    const KZGKeyPair& key) {
    std::vector<Commitment> out;
    out.reserve(named_polynomials.size());
    if (named_polynomials.empty()) {
        return out;
    }

    std::vector<FieldElement> tau_evaluations(named_polynomials.size(), FieldElement::zero());
    std::unordered_map<std::string, DomainCommitWeights> domain_weight_cache;
    for (const auto& [name, polynomial] : named_polynomials) {
        (void)name;
        if (polynomial->basis == algebra::PolynomialBasis::Coefficient) {
            continue;
        }
        if (polynomial->domain == nullptr) {
            throw std::runtime_error("evaluation polynomial is missing its domain");
        }

        const auto cache_key = polynomial->domain->name + ":" + std::to_string(polynomial->domain->size) + ":"
            + key.tau.to_string();
        if (domain_weight_cache.contains(cache_key)) {
            continue;
        }

        DomainCommitWeights entry;
        for (std::size_t i = 0; i < polynomial->domain->size; ++i) {
            if (polynomial->domain->points[i] == key.tau) {
                entry.direct_index = i;
                break;
            }
        }
        if (!entry.direct_index.has_value()) {
            if (util::route2_options().fft_backend_upgrade) {
                // The KZG commitment is still C = [p(tau)]_1. This route only
                // swaps the low-level evaluation backend used to obtain p(tau)
                // for evaluation-basis polynomials, and it reuses the same
                // (domain, tau) barycentric weights across the whole batch.
                entry.native_weights = polynomial->domain->barycentric_weights_native(key.tau);
            } else {
                entry.native_weights.resize(polynomial->domain->size);
                const auto scale = (polynomial->domain->zero_polynomial_eval(key.tau) * polynomial->domain->inv_size).native();
                for (std::size_t i = 0; i < polynomial->domain->size; ++i) {
                    mcl::Fr denominator;
                    mcl::Fr::sub(denominator, key.tau.native(), polynomial->domain->points[i].native());
                    mcl::Fr inverse;
                    mcl::Fr::inv(inverse, denominator);
                    mcl::Fr::mul(entry.native_weights[i], scale, polynomial->domain->points[i].native());
                    mcl::Fr::mul(entry.native_weights[i], entry.native_weights[i], inverse);
                }
            }
        }
        domain_weight_cache.emplace(cache_key, std::move(entry));
    }

    auto evaluate_with_cache = [&](const algebra::Polynomial& polynomial) {
        if (polynomial.basis == algebra::PolynomialBasis::Coefficient) {
            return polynomial.evaluate(key.tau);
        }
        const auto cache_key = polynomial.domain->name + ":" + std::to_string(polynomial.domain->size) + ":"
            + key.tau.to_string();
        const auto& cached = domain_weight_cache.at(cache_key);
        if (cached.direct_index.has_value()) {
            return polynomial.data.at(*cached.direct_index);
        }
        return algebra::dot_product_native_weights(polynomial.data, cached.native_weights);
    };

    const bool run_parallel = named_polynomials.size() >= 2 && std::thread::hardware_concurrency() > 1;
    if (run_parallel) {
        std::vector<std::future<FieldElement>> futures;
        futures.reserve(named_polynomials.size());
        for (const auto& [name, polynomial] : named_polynomials) {
            (void)name;
            futures.push_back(std::async(
                std::launch::async,
                [&evaluate_with_cache, polynomial]() {
                    return evaluate_with_cache(*polynomial);
                }));
        }
        for (std::size_t i = 0; i < futures.size(); ++i) {
            tau_evaluations[i] = futures[i].get();
        }
    } else {
        for (std::size_t i = 0; i < named_polynomials.size(); ++i) {
            tau_evaluations[i] = evaluate_with_cache(*named_polynomials[i].second);
        }
    }

    const auto points = g1_mul_same_base_batch(key.g1_generator, tau_evaluations);
    for (std::size_t i = 0; i < named_polynomials.size(); ++i) {
        out.push_back(Commitment{
            .name = named_polynomials[i].first,
            .point = points[i],
            .tau_evaluation = tau_evaluations[i],
        });
    }
    return out;
}

G1Point KZG::open_batch(
    const std::vector<Commitment>& commitments,
    const std::vector<FieldElement>& points,
    const std::vector<std::vector<FieldElement>>& claimed_values,
    const FieldElement& folding_challenge,
    const KZGKeyPair& key) {
    if (commitments.size() != claimed_values.size()) {
        throw std::runtime_error("batch opening mismatch between commitments and values");
    }
    const auto precompute = prepare_batch_opening(points, folding_challenge, key, commitments.size());
    return g1_mul(
        fold_commitments(commitments, claimed_values, precompute, key),
        safe_inverse(precompute.vanishing_at_tau, "batch opening"));
}

bool KZG::verify_batch(
    const std::vector<Commitment>& commitments,
    const std::vector<FieldElement>& points,
    const std::vector<std::vector<FieldElement>>& claimed_values,
    const FieldElement& folding_challenge,
    const G1Point& witness,
    const KZGKeyPair& key) {
    if (commitments.size() != claimed_values.size()) {
        throw std::runtime_error("batch verification mismatch between commitments and values");
    }

    const auto precompute = prepare_batch_opening(points, folding_challenge, key, commitments.size());
    const auto folded_commitment = g1_mul(
        key.g1_generator,
        fold_commitment_scalar(commitments, claimed_values, precompute));

    if (util::route2_options().fast_verify_pairing && key.g2_one_prepared != nullptr) {
        // This does not change the KZG verification equation. We only evaluate
        // the same product of pairings through mcl's mixed precomputed Miller
        // loop so the static G2 generator side is reused across verifications.
        const auto dynamic_rhs = g2_mul(key.g2_one, precompute.vanishing_at_tau);
        const auto neg_folded_commitment = g1_sub(g1_zero(), folded_commitment);
        return pairing_product_is_one_mixed_prepared(
            witness,
            dynamic_rhs,
            neg_folded_commitment,
            *key.g2_one_prepared);
    }

    return pairing_equal(
        folded_commitment,
        key.g2_one,
        witness,
        g2_mul(key.g2_one, precompute.vanishing_at_tau));
}

G1Point KZG::open_external_fold(
    const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
    const std::vector<FieldElement>& points,
    const FieldElement& folding_challenge,
    const KZGKeyPair& key) {
    FieldElement accumulator_eval = FieldElement::zero();
    const auto precompute = prepare_external_fold(points, folding_challenge, key);
    const auto folded_numerator = fold_external_commitments(
        commitments_and_values,
        precompute,
        key,
        &accumulator_eval);
    return g1_mul(folded_numerator, safe_inverse(accumulator_eval, "external fold accumulator"));
}

bool KZG::verify_external_fold(
    const std::vector<std::pair<Commitment, FieldElement>>& commitments_and_values,
    const std::vector<FieldElement>& points,
    const FieldElement& folding_challenge,
    const G1Point& witness,
    const KZGKeyPair& key) {
    if (commitments_and_values.size() != points.size()) {
        throw std::runtime_error("external verification mismatch between commitments and points");
    }
    const auto precompute = prepare_external_fold(points, folding_challenge, key);
    const auto folded_commitment = g1_mul(
        key.g1_generator,
        fold_external_scalar(commitments_and_values, precompute));

    if (util::route2_options().fast_verify_pairing && key.g2_one_prepared != nullptr) {
        const auto dynamic_rhs = g2_mul(key.g2_one, precompute.accumulator_eval);
        const auto neg_folded_commitment = g1_sub(g1_zero(), folded_commitment);
        return pairing_product_is_one_mixed_prepared(
            witness,
            dynamic_rhs,
            neg_folded_commitment,
            *key.g2_one_prepared);
    }

    return pairing_equal(
        folded_commitment,
        key.g2_one,
        witness,
        g2_mul(key.g2_one, precompute.accumulator_eval));
}

}  // namespace gatzk::crypto
