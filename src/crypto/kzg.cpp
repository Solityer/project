#include "gatzk/crypto/kzg.hpp"

#include <chrono>
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
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

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

std::vector<FieldElement> evaluate_polynomials_with_shared_domain_weights(
    const std::vector<const algebra::Polynomial*>& polynomials,
    const DomainCommitWeights& weights) {
    std::vector<FieldElement> out(polynomials.size(), FieldElement::zero());
    if (polynomials.empty()) {
        return out;
    }
    if (weights.direct_index.has_value()) {
        for (std::size_t i = 0; i < polynomials.size(); ++i) {
            out[i] = polynomials[i]->data.at(*weights.direct_index);
        }
        return out;
    }

    const auto domain_size = weights.native_weights.size();
    std::vector<mcl::Fr> native_out(polynomials.size());
    for (auto& value : native_out) {
        value.clear();
    }

    const auto cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const bool run_parallel = util::route2_options().parallel_fft && domain_size >= 1024 && polynomials.size() >= 4 && cpu_count > 1;
    if (run_parallel) {
        const auto task_count = std::min<std::size_t>(cpu_count, domain_size / 512);
        if (task_count > 1) {
            std::vector<std::future<std::vector<mcl::Fr>>> futures;
            futures.reserve(task_count);
            const auto chunk_size = (domain_size + task_count - 1) / task_count;
            for (std::size_t task = 0; task < task_count; ++task) {
                const auto begin = task * chunk_size;
                const auto end = std::min(domain_size, begin + chunk_size);
                if (begin >= end) {
                    break;
                }
                futures.push_back(std::async(
                    std::launch::async,
                    [begin, end, &weights, &polynomials]() {
                        std::vector<mcl::Fr> partial(polynomials.size());
                        for (auto& value : partial) {
                            value.clear();
                        }
                        for (std::size_t index = begin; index < end; ++index) {
                            const auto& weight = weights.native_weights[index];
                            for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index) {
                                mcl::Fr term;
                                mcl::Fr::mul(term, polynomials[poly_index]->data[index].native(), weight);
                                mcl::Fr::add(partial[poly_index], partial[poly_index], term);
                            }
                        }
                        return partial;
                    }));
            }
            for (auto& future : futures) {
                const auto partial = future.get();
                for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index) {
                    mcl::Fr::add(native_out[poly_index], native_out[poly_index], partial[poly_index]);
                }
            }
            for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index) {
                out[poly_index] = FieldElement::from_native(native_out[poly_index]);
            }
            return out;
        }
    }

    for (std::size_t index = 0; index < domain_size; ++index) {
        const auto& weight = weights.native_weights[index];
        for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index) {
            mcl::Fr term;
            mcl::Fr::mul(term, polynomials[poly_index]->data[index].native(), weight);
            mcl::Fr::add(native_out[poly_index], native_out[poly_index], term);
        }
    }
    for (std::size_t poly_index = 0; poly_index < polynomials.size(); ++poly_index) {
        out[poly_index] = FieldElement::from_native(native_out[poly_index]);
    }
    return out;
}

std::shared_ptr<const DomainCommitWeights> load_or_build_domain_commit_weights(
    const algebra::Polynomial& polynomial,
    const KZGKeyPair& key) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::shared_ptr<DomainCommitWeights>> cache;

    const auto cache_key = polynomial.domain->name + ":" + std::to_string(polynomial.domain->size) + ":"
        + key.tau.to_string();
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(cache_key); it != cache.end()) {
            return it->second;
        }
    }

    auto entry = std::make_shared<DomainCommitWeights>();
    for (std::size_t i = 0; i < polynomial.domain->size; ++i) {
        if (polynomial.domain->points[i] == key.tau) {
            entry->direct_index = i;
            break;
        }
    }
    if (!entry->direct_index.has_value()) {
        if (util::route2_options().fft_backend_upgrade) {
            entry->native_weights = polynomial.domain->barycentric_weights_native(key.tau);
        } else {
            entry->native_weights.resize(polynomial.domain->size);
            const auto scale = (polynomial.domain->zero_polynomial_eval(key.tau) * polynomial.domain->inv_size).native();
            for (std::size_t i = 0; i < polynomial.domain->size; ++i) {
                mcl::Fr denominator;
                mcl::Fr::sub(denominator, key.tau.native(), polynomial.domain->points[i].native());
                mcl::Fr inverse;
                mcl::Fr::inv(inverse, denominator);
                mcl::Fr::mul(entry->native_weights[i], scale, polynomial.domain->points[i].native());
                mcl::Fr::mul(entry->native_weights[i], entry->native_weights[i], inverse);
            }
        }
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto [it, _] = cache.emplace(cache_key, entry);
    return it->second;
}

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
    const KZGKeyPair& key,
    CommitBatchProfile* profile) {
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
    const auto msm_start = Clock::now();
    const auto points = g1_mul_same_base_batch(key.g1_generator, tau_evaluations);
    if (profile != nullptr) {
        profile->msm_ms += elapsed_ms(msm_start, Clock::now());
    }
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
    const KZGKeyPair& key,
    CommitBatchProfile* profile) {
    std::vector<Commitment> out;
    out.reserve(named_polynomials.size());
    if (named_polynomials.empty()) {
        return out;
    }

    std::vector<FieldElement> tau_evaluations(named_polynomials.size(), FieldElement::zero());
    const auto tau_key = key.tau.to_string();
    std::unordered_map<std::string, std::shared_ptr<const DomainCommitWeights>> domain_weight_cache;
    struct DomainBatchGroup {
        std::vector<std::size_t> indices;
        std::vector<const algebra::Polynomial*> polynomials;
        std::shared_ptr<const DomainCommitWeights> weights;
    };
    std::unordered_map<std::string, DomainBatchGroup> domain_groups;
    for (std::size_t i = 0; i < named_polynomials.size(); ++i) {
        const auto* polynomial = named_polynomials[i].second;
        if (polynomial->basis == algebra::PolynomialBasis::Coefficient) {
            continue;
        }
        if (polynomial->domain == nullptr) {
            throw std::runtime_error("evaluation polynomial is missing its domain");
        }
        const auto cache_key =
            polynomial->domain->name + ":" + std::to_string(polynomial->domain->size) + ":" + tau_key;
        if (!domain_weight_cache.contains(cache_key)) {
            // The commitment is still C = [p(tau)]_1. We only reuse the exact same
            // (domain, tau) weights across all commitment batches in this process.
            domain_weight_cache.emplace(cache_key, load_or_build_domain_commit_weights(*polynomial, key));
        }
        auto& group = domain_groups[cache_key];
        group.indices.push_back(i);
        group.polynomials.push_back(polynomial);
        group.weights = domain_weight_cache.at(cache_key);
    }

    const auto tau_eval_start = Clock::now();
    for (std::size_t i = 0; i < named_polynomials.size(); ++i) {
        const auto* polynomial = named_polynomials[i].second;
        if (polynomial->basis == algebra::PolynomialBasis::Coefficient) {
            tau_evaluations[i] = polynomial->evaluate(key.tau);
        }
    }
    for (const auto& [cache_key, group] : domain_groups) {
        (void)cache_key;
        const auto values = evaluate_polynomials_with_shared_domain_weights(group.polynomials, *group.weights);
        for (std::size_t i = 0; i < group.indices.size(); ++i) {
            tau_evaluations[group.indices[i]] = values[i];
        }
    }
    if (profile != nullptr) {
        profile->tau_eval_ms += elapsed_ms(tau_eval_start, Clock::now());
    }

    const auto msm_start = Clock::now();
    const auto points = g1_mul_same_base_batch(key.g1_generator, tau_evaluations);
    if (profile != nullptr) {
        profile->msm_ms += elapsed_ms(msm_start, Clock::now());
    }
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
    const KZGKeyPair& key,
    BatchOpeningProfile* profile) {
    if (commitments.size() != claimed_values.size()) {
        throw std::runtime_error("batch opening mismatch between commitments and values");
    }
    const auto precompute_start = Clock::now();
    const auto precompute = prepare_batch_opening(points, folding_challenge, key, commitments.size());
    if (profile != nullptr) {
        profile->precompute_ms += elapsed_ms(precompute_start, Clock::now());
    }
    const auto fold_start = Clock::now();
    const auto folded_commitment = fold_commitments(commitments, claimed_values, precompute, key);
    if (profile != nullptr) {
        profile->fold_commitment_ms += elapsed_ms(fold_start, Clock::now());
    }
    const auto finalize_start = Clock::now();
    const auto witness = g1_mul(
        folded_commitment,
        safe_inverse(precompute.vanishing_at_tau, "batch opening"));
    if (profile != nullptr) {
        profile->finalize_ms += elapsed_ms(finalize_start, Clock::now());
    }
    return witness;
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
