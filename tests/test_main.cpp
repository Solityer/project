#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <openssl/sha.h>

#include "gatzk/algebra/domain_evaluation.hpp"
#include "gatzk/algebra/eval_backend.hpp"
#include "gatzk/algebra/field_io.hpp"
#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/crypto/kzg.hpp"
#include "gatzk/data/loader.hpp"
#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/lookup.hpp"
#include "gatzk/protocol/opening_registry.hpp"
#include "gatzk/protocol/prover.hpp"
#include "gatzk/protocol/proof_codec.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"
#include "gatzk/protocol/trace.hpp"
#include "gatzk/protocol/verifier.hpp"
#include "gatzk/util/config.hpp"

namespace
{

    using gatzk::algebra::FieldElement;

    std::filesystem::path repo_root()
    {
        auto cwd = std::filesystem::current_path();
        if (std::filesystem::exists(cwd / "configs" / "cora_secure.cfg"))
        {
            return cwd;
        }
        if (std::filesystem::exists(cwd.parent_path() / "configs" / "cora_secure.cfg"))
        {
            return cwd.parent_path();
        }
        return cwd;
    }

    std::filesystem::path test_srs_path()
    {
        if (const auto *override_path = std::getenv("GATZK_TEST_SRS_PATH");
            override_path != nullptr && *override_path != '\0')
        {
            return override_path;
        }
        return repo_root() / "artifacts/srs/unit_security_srs_v1";
    }

    std::string test_srs_id()
    {
        if (const auto *override_id = std::getenv("GATZK_TEST_SRS_ID");
            override_id != nullptr && *override_id != '\0')
        {
            return override_id;
        }
        return "unit_security_srs_v1";
    }

    void require(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    FieldElement choose_non_domain_point(
        const gatzk::algebra::RootOfUnityDomain &domain);

    void test_rotation_shift_subgroup_rejection_equivalence()
    {
        const auto domain =
            gatzk::algebra::RootOfUnityDomain::create(
                "rotation_shift_regression", 64);
        const FieldElement from(7);
        auto point = from;
        for (std::size_t shift = 0; shift < domain->size; ++shift)
        {
            require(
                domain->rotation_shift(from, point) == std::optional<std::size_t>(shift),
                "rotation subgroup membership check changed an in-domain shift");
            point *= domain->omega;
        }
        const auto outside = choose_non_domain_point(*domain);
        require(
            !domain->rotation_shift(FieldElement::one(), outside).has_value(),
            "rotation subgroup membership check accepted an out-of-domain point");
        require(
            domain->rotation_shift(FieldElement::zero(), FieldElement::zero()) == std::optional<std::size_t>(0) && !domain->rotation_shift(
                                                                                                                              FieldElement::zero(), FieldElement::one())
                                                                                                                        .has_value(),
            "rotation subgroup membership check changed zero-point handling");
    }

    void test_spilled_multi_point_evaluation_equivalence()
    {
        const auto domain =
            gatzk::algebra::RootOfUnityDomain::create(
                "spilled_multi_point_regression", 128);
        std::vector<FieldElement> values(domain->size, FieldElement::zero());
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            values[index] = FieldElement(
                static_cast<std::uint64_t>(index * index + 7 * index + 13));
        }
        auto polynomial = gatzk::algebra::Polynomial::from_evaluations(
            "P_spilled_multi_point", values, domain);
        polynomial.set_zero_domain_blinder(
            {FieldElement(17), FieldElement(19), FieldElement(23)});

        const auto spill_path =
            std::filesystem::path(GATZK_TEST_BINARY_DIR) / "spilled_multi_point_regression.bin";
        struct SpillCleanup
        {
            std::filesystem::path path;
            ~SpillCleanup()
            {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        } cleanup{spill_path};
        {
            std::ofstream output(
                spill_path, std::ios::binary | std::ios::trunc);
            require(
                static_cast<bool>(output),
                "failed to create multi-point spill regression fixture");
            std::array<std::uint8_t, 32> bytes{};
            for (const auto &value : values)
            {
                value.write_little_endian(bytes.data(), bytes.size());
                output.write(
                    reinterpret_cast<const char *>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
            }
            require(
                static_cast<bool>(output),
                "failed to write multi-point spill regression fixture");
        }

        const gatzk::protocol::TraceArtifacts::SpilledEvaluationPolynomial spilled{
            .path = spill_path.string(),
            .domain_name = domain->name,
            .size = values.size(),
            .zero_domain_blinder = polynomial.zero_domain_blinder,
        };
        const auto point = choose_non_domain_point(*domain);
        const std::vector<FieldElement> points{
            point,
            point * domain->omega,
        };
        gatzk::algebra::DomainEvaluationWeightCache weight_cache;
        std::vector<const gatzk::algebra::DomainEvaluationWeights *> weights;
        weights.reserve(points.size());
        for (const auto &current_point : points)
        {
            weights.push_back(&weight_cache.get(domain, current_point));
        }

        gatzk::protocol::SpilledEvaluationProfile grouped_profile;
        const auto grouped =
            gatzk::protocol::evaluate_spilled_evaluation_polynomial_points(
                spilled, domain, points, weights, &grouped_profile);
        gatzk::protocol::SpilledEvaluationProfile single_profile;
        for (std::size_t point_index = 0;
             point_index < points.size();
             ++point_index)
        {
            const auto single =
                gatzk::protocol::evaluate_spilled_evaluation_polynomial(
                    spilled,
                    domain,
                    points[point_index],
                    *weights[point_index],
                    &single_profile);
            require(
                grouped[point_index] == single && grouped[point_index] == polynomial.evaluate(points[point_index]),
                "grouped spilled evaluation changed a rotated-point value or zero-domain blinder");
        }
        require(
            grouped_profile.file_open_count == 1 && grouped_profile.sequential_pass_count == 1 && grouped_profile.field_elements_streamed == values.size(),
            "grouped spilled evaluation must use exactly one sequential file pass");
        require(
            single_profile.file_open_count == points.size() && single_profile.sequential_pass_count == points.size() && single_profile.field_elements_streamed == points.size() * values.size(),
            "single-point spill evaluator instrumentation changed unexpectedly");

        const std::vector<FieldElement> direct_points{
            domain->point_at(3),
            domain->point_at(97),
        };
        std::vector<const gatzk::algebra::DomainEvaluationWeights *> direct_weights;
        direct_weights.reserve(direct_points.size());
        for (const auto &direct_point : direct_points)
        {
            direct_weights.push_back(&weight_cache.get(domain, direct_point));
        }
        gatzk::protocol::SpilledEvaluationProfile direct_profile;
        const auto direct_values =
            gatzk::protocol::evaluate_spilled_evaluation_polynomial_points(
                spilled,
                domain,
                direct_points,
                direct_weights,
                &direct_profile);
        require(
            direct_values[0] == values[3] && direct_values[1] == values[97] && direct_profile.file_open_count == 1 && direct_profile.sequential_pass_count == 0 && direct_profile.direct_values_resolved == 2,
            "grouped spilled evaluation changed direct-index semantics");

        const std::vector<gatzk::protocol::SpilledEvaluationRequest> requests{
            {
                .spilled = &spilled,
                .domain = domain,
                .points = points,
                .weights = weights,
            },
            {
                .spilled = &spilled,
                .domain = domain,
                .points = {points[1], points[0]},
                .weights = {weights[1], weights[0]},
            },
        };
        gatzk::protocol::SpilledEvaluationBatchProfile batch_profile;
        const auto batch_values =
            gatzk::protocol::evaluate_spilled_evaluation_requests(
                requests, 4, &batch_profile);
        require(
            batch_profile.worker_count == 1 && batch_profile.jobs.size() == requests.size() && batch_profile.jobs[0].sequential_pass_count == 1 && batch_profile.jobs[1].sequential_pass_count == 1 && batch_values[0] == grouped && batch_values[1][0] == grouped[1] && batch_values[1][1] == grouped[0],
            "small spill batches must remain serial and preserve request order");
    }

    void test_fh_precomputed_boundary_equivalence()
    {
        gatzk::protocol::ProtocolContext context;
        context.domains.fh =
            gatzk::algebra::RootOfUnityDomain::create("FH", 16);
        const auto z = FieldElement(37);
        const auto one = FieldElement::one();
        const auto last_point =
            context.domains.fh->point_at(
                context.domains.fh->size - 1);
        const auto zero =
            context.domains.fh->zero_polynomial_eval(z);
        const auto first_denominator = z - one;
        const auto last_denominator = z - last_point;
        const auto inverse_product =
            (first_denominator * last_denominator).inv();
        const auto zero_scaled =
            zero * context.domains.fh->inv_size;
        const gatzk::protocol::QuotientBoundaryValues boundary{
            .first = zero_scaled * last_denominator * inverse_product,
            .last = zero_scaled * last_point * first_denominator * inverse_product,
            .zero = zero,
            .zero_inverse = std::nullopt,
        };
        require(
            boundary.first == context.domains.fh->lagrange_basis_eval(
                                  0, z) &&
                boundary.last == context.domains.fh->lagrange_basis_eval(
                                     context.domains.fh->size - 1, z),
            "precomputed FH boundary differs from canonical "
            "Lagrange values");

        const std::map<std::string, FieldElement> challenges{
            {"alpha_quot", FieldElement(3)},
            {"beta_feat", FieldElement(5)},
            {"eta_feat", FieldElement(7)},
        };
        const gatzk::protocol::EvalFn eval =
            [](const std::string &label,
               const FieldElement &point)
        {
            std::uint64_t seed = 17;
            for (const auto ch : label)
            {
                seed = seed * 131 + static_cast<unsigned char>(ch);
            }
            return FieldElement(1 + seed % 65521) + point * FieldElement(3 + seed % 257);
        };
        const auto direct = gatzk::protocol::evaluate_t_fh(
            context, challenges, eval, z);
        const auto precomputed =
            gatzk::protocol::evaluate_t_fh(
                context, challenges, eval, z, nullptr, &boundary);
        require(
            direct == precomputed,
            "precomputed FH boundary changed the quotient equation");
        auto inverse_cached_boundary = boundary;
        inverse_cached_boundary.zero_inverse = zero.inv();
        const auto inverse_cached =
            gatzk::protocol::evaluate_t_fh(
                context,
                challenges,
                eval,
                z,
                nullptr,
                &inverse_cached_boundary);
        require(
            direct == inverse_cached,
            "cached FH zero-polynomial inverse changed the quotient equation");

        const gatzk::protocol::FHQuotientParameters fh_parameters{
            .alpha_quot = challenges.at("alpha_quot"),
            .beta_feat = challenges.at("beta_feat"),
            .eta_feat = challenges.at("eta_feat"),
        };
        const gatzk::protocol::FHQuotientValues fh_values{
            .q_table = eval("P_Q_tbl_feat", z),
            .q_query = eval("P_Q_qry_feat", z),
            .table = eval("P_Table_feat", z),
            .query = eval("P_Query_feat", z),
            .multiplicity = eval("P_m_feat", z),
            .j_table = eval("P_J_tbl_feat", z),
            .j_query = eval("P_J_qry_feat", z),
            .accumulator = eval("P_R_feat", z),
            .accumulator_next = eval(
                "P_R_feat", z * context.domains.fh->omega),
        };
        const gatzk::protocol::FHFeatureEncodingValues
            fh_feature_encoding_values{
                .table_row = eval("P_Row_feat_tbl", z),
                .table_column = eval("P_Col_feat_tbl", z),
                .table_feature = eval("P_T_H", z),
                .query_row = eval("P_I_feat_qry", z),
                .query_column = eval("P_Col_feat_qry", z),
                .query_feature = eval("P_H", z),
            };
        const auto fixed_slot_generic =
            gatzk::protocol::evaluate_t_fh_from_values(
                fh_parameters,
                fh_values,
                boundary,
                &fh_feature_encoding_values);
        const auto fixed_slot_generic_power_reference =
            gatzk::protocol::
                evaluate_t_fh_from_values_power_accumulator_for_stage_benchmark(
                    fh_parameters,
                    fh_values,
                    boundary,
                    &fh_feature_encoding_values);
        require(
            direct == fixed_slot_generic && fixed_slot_generic == fixed_slot_generic_power_reference,
            "fixed-slot FH core differs from generic named evaluation");

        const gatzk::protocol::EvalFn identity_eval =
            [&](const std::string &label,
                const FieldElement &point)
        {
            const auto row = eval("P_Row_feat_tbl", point);
            const auto col = eval("P_Col_feat_tbl", point);
            const auto feature = eval("P_T_H", point);
            if (label == "P_Table_feat" || label == "P_Query_feat")
            {
                return row + challenges.at("eta_feat") * col + challenges.at("eta_feat").pow(2) * feature;
            }
            if (label == "P_H")
            {
                return feature;
            }
            if (label == "P_Row_feat_qry" || label == "P_I_feat_qry")
            {
                return row;
            }
            if (label == "P_Col_feat_qry")
            {
                return col;
            }
            return eval(label, point);
        };
        const auto generic_identity =
            gatzk::protocol::evaluate_t_fh(
                context, challenges, identity_eval, z, nullptr, &boundary);
        const auto derived_identity =
            gatzk::protocol::evaluate_t_fh(
                context,
                challenges,
                identity_eval,
                z,
                nullptr,
                &boundary,
                true);
        require(
            generic_identity == derived_identity,
            "derived full-identity FH encoding terms changed the quotient equation");
        const gatzk::protocol::FHQuotientValues identity_values{
            .q_table = identity_eval("P_Q_tbl_feat", z),
            .q_query = identity_eval("P_Q_qry_feat", z),
            .table = identity_eval("P_Table_feat", z),
            .query = identity_eval("P_Query_feat", z),
            .multiplicity = identity_eval("P_m_feat", z),
            .j_table = identity_eval("P_J_tbl_feat", z),
            .j_query = identity_eval("P_J_qry_feat", z),
            .accumulator = identity_eval("P_R_feat", z),
            .accumulator_next = identity_eval(
                "P_R_feat", z * context.domains.fh->omega),
        };
        const auto fixed_slot_identity =
            gatzk::protocol::evaluate_t_fh_from_values(
                fh_parameters, identity_values, boundary);
        const auto fixed_slot_identity_power_reference =
            gatzk::protocol::
                evaluate_t_fh_from_values_power_accumulator_for_stage_benchmark(
                    fh_parameters, identity_values, boundary);
        require(
            derived_identity == fixed_slot_identity && fixed_slot_identity == fixed_slot_identity_power_reference,
            "fixed-slot FH identity core differs from generic source of truth");
    }

    void test_logup_inverse_accumulator_reuse_equivalence()
    {
        constexpr std::size_t domain_size = 16;
        constexpr std::size_t active_count = 9;
        std::vector<FieldElement> table(domain_size, FieldElement::zero());
        std::vector<FieldElement> query(domain_size, FieldElement::zero());
        std::vector<FieldElement> multiplicity(domain_size, FieldElement::zero());
        std::vector<FieldElement> q_table(domain_size, FieldElement::zero());
        std::vector<FieldElement> q_query(domain_size, FieldElement::zero());
        for (std::size_t i = 0; i < 5; ++i)
        {
            table[i] = FieldElement(i + 2);
            multiplicity[i] = FieldElement((i % 3) + 1);
            q_table[i] = FieldElement::one();
        }
        for (std::size_t i = 0; i < active_count; ++i)
        {
            query[i] = FieldElement((i * 3) % 5 + 2);
            q_query[i] = FieldElement::one();
        }
        const FieldElement beta(29);
        const auto inverses = gatzk::protocol::build_logup_inverse_columns(
            table, query, q_table, q_query, beta);
        const auto reference =
            gatzk::protocol::build_logup_accumulator_with_active_count(
                table,
                query,
                multiplicity,
                q_table,
                q_query,
                beta,
                active_count);
        const auto reused =
            gatzk::protocol::
                build_logup_accumulator_from_inverse_columns_with_active_count(
                    inverses.table,
                    inverses.query,
                    multiplicity,
                    q_table,
                    q_query,
                    active_count);
        require(
            reused == reference,
            "LogUp accumulator reuse must exactly match denominator recomputation");

        const auto inactive = gatzk::protocol::
            build_logup_accumulator_from_inverse_columns_with_active_count(
                inverses.table,
                inverses.query,
                multiplicity,
                q_table,
                q_query,
                0);
        require(
            std::all_of(
                inactive.begin(), inactive.end(),
                [](const FieldElement &value)
                { return value.is_zero(); }),
            "zero-active LogUp accumulator must remain zero through padding");

        constexpr std::size_t parallel_active_count = std::size_t{1} << 18;
        constexpr std::size_t parallel_domain_size = parallel_active_count + 1;
        std::vector<FieldElement> parallel_table_inverse(
            parallel_domain_size, FieldElement::one());
        std::vector<FieldElement> parallel_query_inverse(
            parallel_domain_size, FieldElement::zero());
        std::vector<FieldElement> parallel_multiplicity(
            parallel_domain_size, FieldElement(3));
        std::vector<FieldElement> parallel_q_table(
            parallel_domain_size, FieldElement::zero());
        std::vector<FieldElement> parallel_q_query(
            parallel_domain_size, FieldElement::zero());
        std::fill_n(
            parallel_q_table.begin(), parallel_active_count,
            FieldElement::one());
        const auto parallel_reference = gatzk::protocol::
            build_logup_accumulator_from_inverse_columns_with_active_count_serial_for_stage_benchmark(
                parallel_table_inverse,
                parallel_query_inverse,
                parallel_multiplicity,
                parallel_q_table,
                parallel_q_query,
                parallel_active_count);
        const auto parallel_reused = gatzk::protocol::
            build_logup_accumulator_from_inverse_columns_with_active_count(
                parallel_table_inverse,
                parallel_query_inverse,
                parallel_multiplicity,
                parallel_q_table,
                parallel_q_query,
                parallel_active_count);
        require(
            parallel_reused == parallel_reference && parallel_reused.back() == parallel_reused[parallel_active_count],
            "threshold-parallel LogUp accumulator must preserve terminal hold-last");
        parallel_q_table[parallel_active_count / 2] = FieldElement(2);
        bool rejected_non_boolean_selector = false;
        try
        {
            (void)gatzk::protocol::
                build_logup_accumulator_from_inverse_columns_with_active_count(
                    parallel_table_inverse,
                    parallel_query_inverse,
                    parallel_multiplicity,
                    parallel_q_table,
                    parallel_q_query,
                    parallel_active_count);
        }
        catch (const std::exception &)
        {
            rejected_non_boolean_selector = true;
        }
        require(
            rejected_non_boolean_selector,
            "parallel LogUp accumulator must reject a non-boolean selector");
    }

    struct TestPointGroupProof
    {
        std::vector<FieldElement> points;
        std::vector<gatzk::crypto::G1Point> witnesses;
    };

    TestPointGroupProof open_matrix_point_groups(
        const std::vector<gatzk::crypto::Commitment> &commitments,
        const std::vector<FieldElement> &points,
        const std::vector<std::vector<FieldElement>> &values,
        const FieldElement &first_challenge,
        const gatzk::crypto::KZGKeyPair &keys)
    {
        if (commitments.size() != values.size())
        {
            throw std::runtime_error("test point-group matrix shape mismatch");
        }
        TestPointGroupProof proof;
        proof.points = points;
        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
        {
            std::vector<FieldElement> point_values;
            point_values.reserve(values.size());
            for (const auto &column : values)
            {
                if (column.size() != points.size())
                {
                    throw std::runtime_error("test point-group value shape mismatch");
                }
                point_values.push_back(column[point_index]);
            }
            proof.witnesses.push_back(gatzk::crypto::KZG::open_point_group(
                commitments,
                points[point_index],
                point_values,
                first_challenge + FieldElement(point_index),
                keys));
        }
        return proof;
    }

    bool verify_matrix_point_groups(
        const std::vector<gatzk::crypto::Commitment> &commitments,
        const std::vector<FieldElement> &points,
        const std::vector<std::vector<FieldElement>> &values,
        const FieldElement &first_challenge,
        const TestPointGroupProof &proof,
        const gatzk::crypto::KZGVerifierKey &vk)
    {
        if (proof.points != points || proof.witnesses.size() != points.size() || commitments.size() != values.size())
        {
            throw std::runtime_error("test point-group proof shape mismatch");
        }
        for (std::size_t point_index = 0; point_index < points.size(); ++point_index)
        {
            std::vector<FieldElement> point_values;
            for (const auto &column : values)
            {
                if (column.size() != points.size())
                {
                    throw std::runtime_error("test point-group verify shape mismatch");
                }
                point_values.push_back(column[point_index]);
            }
            if (!gatzk::crypto::KZG::verify_point_group(
                    commitments,
                    points[point_index],
                    point_values,
                    first_challenge + FieldElement(point_index),
                    proof.witnesses[point_index],
                    vk))
            {
                return false;
            }
        }
        return true;
    }

    TestPointGroupProof open_claim_point_groups(
        const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> &claims,
        const std::vector<FieldElement> &points,
        const FieldElement &first_challenge,
        const gatzk::crypto::KZGKeyPair &keys)
    {
        if (claims.size() != points.size())
        {
            throw std::runtime_error("test external point-group shape mismatch");
        }
        TestPointGroupProof proof;
        for (const auto &point : points)
        {
            if (std::find(proof.points.begin(), proof.points.end(), point) == proof.points.end())
            {
                proof.points.push_back(point);
            }
        }
        for (std::size_t group_index = 0; group_index < proof.points.size(); ++group_index)
        {
            std::vector<gatzk::crypto::Commitment> commitments;
            std::vector<FieldElement> values;
            for (std::size_t i = 0; i < claims.size(); ++i)
            {
                if (points[i] == proof.points[group_index])
                {
                    commitments.push_back(claims[i].first);
                    values.push_back(claims[i].second);
                }
            }
            proof.witnesses.push_back(gatzk::crypto::KZG::open_point_group(
                commitments,
                proof.points[group_index],
                values,
                first_challenge + FieldElement(group_index),
                keys));
        }
        return proof;
    }

    bool verify_claim_point_groups(
        const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> &claims,
        const std::vector<FieldElement> &points,
        const FieldElement &first_challenge,
        const TestPointGroupProof &proof,
        const gatzk::crypto::KZGVerifierKey &vk)
    {
        if (claims.size() != points.size() || proof.points.size() != proof.witnesses.size())
        {
            throw std::runtime_error("test external point-group proof shape mismatch");
        }
        for (std::size_t group_index = 0; group_index < proof.points.size(); ++group_index)
        {
            std::vector<gatzk::crypto::Commitment> commitments;
            std::vector<FieldElement> values;
            for (std::size_t i = 0; i < claims.size(); ++i)
            {
                if (points[i] == proof.points[group_index])
                {
                    commitments.push_back(claims[i].first);
                    values.push_back(claims[i].second);
                }
            }
            if (!gatzk::crypto::KZG::verify_point_group(
                    commitments,
                    proof.points[group_index],
                    values,
                    first_challenge + FieldElement(group_index),
                    proof.witnesses[group_index],
                    vk))
            {
                return false;
            }
        }
        return true;
    }

    void test_parallel_split_intt_equivalence()
    {
        constexpr std::size_t size = 1ULL << 15;
        std::vector<FieldElement> serial(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            serial[i] = FieldElement(
                31 + ((i * 2654435761ULL) % 104729ULL));
        }
        auto split = serial;
        gatzk::algebra::ntt_plan(size)->intt(
            FieldElement::mutable_native_array_data(serial.data()));
        gatzk::algebra::intt_in_place(
            FieldElement::mutable_native_array_data(split.data()), split.size());
        require(serial == split,
                "parallel split INTT must equal the canonical full INTT");

        auto first_serial = split;
        auto second_serial = split;
        second_serial.front() += FieldElement(19);
        auto first_parallel = first_serial;
        auto second_parallel = second_serial;
        gatzk::algebra::ntt_plan(size)->ntt(
            FieldElement::mutable_native_array_data(first_serial.data()));
        gatzk::algebra::ntt_plan(size)->ntt(
            FieldElement::mutable_native_array_data(second_serial.data()));
        gatzk::algebra::ntt_pair_in_place(
            FieldElement::mutable_native_array_data(first_parallel.data()),
            FieldElement::mutable_native_array_data(second_parallel.data()),
            size);
        require(first_serial == first_parallel && second_serial == second_parallel,
                "parallel paired NTT must equal two canonical forward NTTs");
    }

    std::string sha256_hex(std::string_view data)
    {
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        SHA256(
            reinterpret_cast<const unsigned char *>(data.data()),
            data.size(),
            digest.data());
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto byte : digest)
            out << std::setw(2) << static_cast<unsigned>(byte);
        return out.str();
    }

    std::string decode_canonical_hex(std::string_view value)
    {
        require(!value.empty() && (value.size() & 1U) == 0U,
                "test hex field must have an even nonzero length");
        auto nibble = [](char c) -> unsigned
        {
            if (c >= '0' && c <= '9')
                return static_cast<unsigned>(c - '0');
            if (c >= 'a' && c <= 'f')
                return 10U + static_cast<unsigned>(c - 'a');
            throw std::runtime_error("test hex field is not canonical lowercase hex");
        };
        std::string out(value.size() / 2, '\0');
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] = static_cast<char>((nibble(value[2 * i]) << 4U) | nibble(value[2 * i + 1]));
        return out;
    }

    std::string canonical_hex(std::string_view value)
    {
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (const auto byte : value)
            out << std::setw(2)
                << static_cast<unsigned>(static_cast<unsigned char>(byte));
        return out.str();
    }

    void replace_artifact_scalar(
        std::string &artifact,
        const std::string &field_name,
        const std::string &value)
    {
        const auto prefix = field_name + "=";
        const auto begin = artifact.find(prefix);
        require(begin != std::string::npos && (begin == 0 || artifact[begin - 1] == '\n'),
                "test artifact field is missing: " + field_name);
        const auto value_begin = begin + prefix.size();
        const auto value_end = artifact.find('\n', value_begin);
        require(value_end != std::string::npos,
                "test artifact field is unterminated: " + field_name);
        artifact.replace(value_begin, value_end - value_begin, value);
    }

    std::string read_file(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input)
            throw std::runtime_error("failed to read test file: " + path.string());
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    void write_file(const std::filesystem::path &path, std::string_view bytes)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output)
            throw std::runtime_error("failed to write test file: " + path.string());
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    void rewrite_manifest_hash(std::string &manifest)
    {
        const auto field = manifest.rfind("  \"manifest_hash\"");
        require(field != std::string::npos, "test manifest must contain manifest_hash");
        const auto value_begin = manifest.find('"', manifest.find(':', field) + 1);
        const auto value_end = manifest.find('"', value_begin + 1);
        require(value_begin != std::string::npos && value_end != std::string::npos,
                "test manifest hash field must be well formed");
        manifest.replace(value_begin + 1, value_end - value_begin - 1,
                         sha256_hex(std::string_view(manifest).substr(0, field)));
    }

    void replace_json_string_field(
        std::string &json,
        const std::string &field_name,
        const std::string &value)
    {
        const auto field = json.find("\"" + field_name + "\"");
        require(field != std::string::npos, "test JSON field is missing: " + field_name);
        const auto value_begin = json.find('"', json.find(':', field) + 1);
        const auto value_end = json.find('"', value_begin + 1);
        require(value_begin != std::string::npos && value_end != std::string::npos,
                "test JSON string field is malformed: " + field_name);
        json.replace(value_begin + 1, value_end - value_begin - 1, value);
    }

    std::filesystem::path copy_test_srs(const std::string &suffix)
    {
        const auto root =
            std::filesystem::path(GATZK_TEST_BINARY_DIR) / ("srs_tamper_" + suffix);
        std::filesystem::remove_all(root);
        std::filesystem::copy(
            test_srs_path(), root,
            std::filesystem::copy_options::recursive);
        return root;
    }

    template <typename Fn>
    void require_throws(Fn &&fn, const std::string &message)
    {
        bool thrown = false;
        try
        {
            fn();
        }
        catch (const std::exception &)
        {
            thrown = true;
        }
        require(thrown, message);
    }

    struct SecureFixture
    {
        gatzk::protocol::ProtocolContext context;
        gatzk::protocol::TraceArtifacts trace;
        gatzk::protocol::Proof proof;
    };

    void strip_opening_material(gatzk::crypto::Commitment &commitment)
    {
        commitment.opening_polynomial.reset();
    }

    void strip_context_to_verifier_only(gatzk::protocol::ProtocolContext &context)
    {
        context.kzg.pk = {};
        for (auto &[name, commitment] : context.public_commitments)
        {
            (void)name;
            strip_opening_material(commitment);
        }
        for (auto &[name, commitment] : context.static_commitments)
        {
            (void)name;
            strip_opening_material(commitment);
        }
        auto clear_head = [](gatzk::model::AttentionHeadParameters &head)
        {
            head.seq_kernel_fp.clear();
            head.attn_dst_kernel_fp.clear();
            head.attn_dst_bias_fp = 0.0;
            head.attn_src_kernel_fp.clear();
            head.attn_src_bias_fp = 0.0;
            head.output_bias_fp.clear();
        };
        for (auto &head : context.model.hidden_heads)
            clear_head(head);
        for (auto &layer : context.model.hidden_layers)
        {
            for (auto &head : layer.heads)
                clear_head(head);
        }
        for (auto &head : context.model.output_layer.heads)
            clear_head(head);
        clear_head(context.model.output_head);
        context.model.W.clear();
        context.model.a_src.clear();
        context.model.a_dst.clear();
        context.model.W_out.clear();
        context.model.b.clear();
    }

    void strip_proof_to_verifier_only(gatzk::protocol::Proof &proof)
    {
        for (auto &[name, commitment] : proof.dynamic_commitments)
        {
            (void)name;
            strip_opening_material(commitment);
        }
        for (auto &[name, commitment] : proof.quotient_commitments)
        {
            (void)name;
            strip_opening_material(commitment);
        }
    }

    FieldElement choose_non_domain_point(const gatzk::algebra::RootOfUnityDomain &domain)
    {
        for (std::uint64_t raw = 17; raw < 10000; ++raw)
        {
            const FieldElement candidate(raw);
            bool overlap = false;
            for (std::size_t i = 0; i < domain.size; ++i)
            {
                if (domain.point_at(i) == candidate)
                {
                    overlap = true;
                    break;
                }
            }
            if (!overlap)
            {
                return candidate;
            }
        }
        throw std::runtime_error("failed to find a non-domain opening point");
    }

    const SecureFixture &cora_secure_fixture()
    {
        static const SecureFixture fixture = []
        {
            auto config = gatzk::util::load_config((repo_root() / "configs/cora_secure.cfg").string());
            auto context = gatzk::protocol::build_context(config, nullptr);
            auto trace = gatzk::protocol::build_trace(context, nullptr);
            auto proof = gatzk::protocol::prove(context, trace, nullptr);
            return SecureFixture{std::move(context), std::move(trace), std::move(proof)};
        }();
        return fixture;
    }

    const SecureFixture &unit_security_fixture()
    {
        static const SecureFixture fixture = []
        {
            auto config = gatzk::util::load_config(
                (repo_root() / "configs/unit_security_regression.cfg").string());
            auto context = gatzk::protocol::build_context(config, nullptr);
            auto trace = gatzk::protocol::build_trace(context, nullptr);
            auto proof = gatzk::protocol::prove(context, trace, nullptr);
            return SecureFixture{
                std::move(context), std::move(trace), std::move(proof)};
        }();
        return fixture;
    }

    void test_kzg_opening_selftest()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(),
            "trusted_srs",
            test_srs_id());
        require(
            keys.manifest.protocol_schema_id == sha256_hex(gatzk::protocol::protocol_template_canonical_bytes()),
            "SRS manifest must authenticate the exact formal protocol template bytes");
        auto poly = gatzk::algebra::Polynomial::from_coefficients(
            "P_test",
            {FieldElement(3), FieldElement(5), FieldElement(7)});
        const auto commitment = gatzk::crypto::KZG::commit("P_test", poly, keys);
        const std::vector<FieldElement> points{FieldElement(9)};
        const std::vector<std::vector<FieldElement>> values{{poly.evaluate(points.front())}};
        const auto proof = open_matrix_point_groups(
            {commitment}, points, values, FieldElement(13), keys);
        require(
            verify_matrix_point_groups(
                {commitment}, points, values, FieldElement(13), proof, keys.vk),
            "KZG single-point opening must verify");
        auto bad_values = values;
        bad_values[0][0] += FieldElement::one();
        require(
            !verify_matrix_point_groups(
                {commitment}, points, bad_values, FieldElement(13), proof, keys.vk),
            "KZG claimed-value tamper must reject");
        auto compromised_vk = keys.vk;
        compromised_vk.g2_tau = gatzk::crypto::g2_mul(
            compromised_vk.g2_one,
            points.front());
        require(
            !verify_matrix_point_groups(
                {commitment}, points, values, FieldElement(13), proof, compromised_vk),
            "KZG must reject a verifier key with [tau-z]_2 equal to the identity");

        auto verifier_commitment = commitment;
        strip_opening_material(verifier_commitment);
        require(
            verify_matrix_point_groups(
                {verifier_commitment}, points, values,
                FieldElement(13), proof, keys.vk),
            "KZG verifier-key-only opening must verify without prover polynomial material");

        const auto streamed_domain =
            gatzk::algebra::RootOfUnityDomain::create("N", 8);
        auto streamed_polynomial =
            gatzk::algebra::Polynomial::from_evaluations(
                "P_streamed_opening",
                {FieldElement(2), FieldElement(3), FieldElement(5),
                 FieldElement(7), FieldElement(11), FieldElement(13),
                 FieldElement(17), FieldElement(19)},
                streamed_domain);
        streamed_polynomial.set_zero_domain_blinder(
            {FieldElement(23), FieldElement(29)});
        auto streamed_commitment = gatzk::crypto::KZG::commit(
            "P_streamed_opening", streamed_polynomial, keys);
        const auto streamed_material =
            streamed_commitment.opening_polynomial;
        const FieldElement streamed_point(31);
        const auto streamed_value =
            streamed_polynomial.evaluate(streamed_point);
        streamed_commitment.opening_polynomial.reset();
        const gatzk::crypto::OpeningPolynomialResolver resolver =
            [&](const gatzk::crypto::Commitment &requested)
        {
            require(requested.name == "P_streamed_opening",
                    "streamed opening resolver received the wrong object");
            return streamed_material;
        };
        const auto streamed_witness =
            gatzk::crypto::KZG::open_point_group(
                {streamed_commitment},
                streamed_point,
                {streamed_value},
                FieldElement(37),
                keys,
                nullptr,
                gatzk::crypto::BatchOpeningClaimValidation::
                    AlreadyComputedFromOpeningPolynomials,
                nullptr,
                &resolver);
        require(
            gatzk::crypto::KZG::verify_point_group(
                {streamed_commitment},
                streamed_point,
                {streamed_value},
                FieldElement(37),
                streamed_witness,
                keys.vk),
            "file-backed streamed evaluation opening must verify");

        // The production file-backed resolver depends on the streamed fold
        // releasing each decoded source before it asks for the next one.  Keep
        // this as an explicit resource-lifetime contract: an opening group may
        // contain hundreds of 2^23 evaluation columns on the large datasets.
        std::vector<std::vector<FieldElement>> bounded_values{
            {FieldElement(2), FieldElement(3), FieldElement(5), FieldElement(7),
             FieldElement(11), FieldElement(13), FieldElement(17), FieldElement(19)},
            {FieldElement(23), FieldElement(29), FieldElement(31), FieldElement(37),
             FieldElement(41), FieldElement(43), FieldElement(47), FieldElement(53)},
            {FieldElement(59), FieldElement(61), FieldElement(67), FieldElement(71),
             FieldElement(73), FieldElement(79), FieldElement(83), FieldElement(89)},
        };
        std::vector<gatzk::crypto::Commitment> bounded_commitments;
        std::vector<FieldElement> bounded_claims;
        const FieldElement bounded_point(97);
        for (std::size_t i = 0; i < bounded_values.size(); ++i)
        {
            const auto name = "P_streamed_lifetime_" + std::to_string(i);
            const auto polynomial =
                gatzk::algebra::Polynomial::from_evaluations(
                    name, bounded_values[i], streamed_domain);
            bounded_claims.push_back(polynomial.evaluate(bounded_point));
            auto commitment = gatzk::crypto::KZG::commit(
                name, polynomial, keys);
            commitment.opening_polynomial.reset();
            bounded_commitments.push_back(std::move(commitment));
        }
        std::size_t live_resolved_materials = 0;
        std::size_t peak_resolved_materials = 0;
        std::size_t resolution_count = 0;
        const gatzk::crypto::OpeningPolynomialResolver bounded_resolver =
            [&](const gatzk::crypto::Commitment &requested)
            -> std::shared_ptr<const gatzk::algebra::Polynomial>
        {
            const auto prefix = std::string("P_streamed_lifetime_");
            require(requested.name.rfind(prefix, 0) == 0,
                    "bounded streamed resolver received the wrong object");
            const auto index = static_cast<std::size_t>(
                std::stoull(requested.name.substr(prefix.size())));
            require(index < bounded_values.size(),
                    "bounded streamed resolver index is out of range");
            auto *material = new gatzk::algebra::Polynomial(
                gatzk::algebra::Polynomial::from_evaluations(
                    requested.name,
                    bounded_values[index],
                    streamed_domain));
            ++live_resolved_materials;
            ++resolution_count;
            peak_resolved_materials = std::max(
                peak_resolved_materials, live_resolved_materials);
            return std::shared_ptr<const gatzk::algebra::Polynomial>(
                material,
                [&](const gatzk::algebra::Polynomial *released)
                {
                    delete released;
                    --live_resolved_materials;
                });
        };
        const auto bounded_witness = gatzk::crypto::KZG::open_point_group(
            bounded_commitments,
            bounded_point,
            bounded_claims,
            FieldElement(101),
            keys,
            nullptr,
            gatzk::crypto::BatchOpeningClaimValidation::
                AlreadyComputedFromOpeningPolynomials,
            nullptr,
            &bounded_resolver);
        require(resolution_count == bounded_commitments.size() && peak_resolved_materials == 1 && live_resolved_materials == 0,
                "streamed opening must release each resolved source after folding");
        require(
            gatzk::crypto::KZG::verify_point_group(
                bounded_commitments,
                bounded_point,
                bounded_claims,
                FieldElement(101),
                bounded_witness,
                keys.vk),
            "bounded-lifetime streamed evaluation opening must verify");

        require(
            keys.vk.domain_zero_g2.size() == keys.manifest.domain_sizes.size(),
            "verifier key must carry the complete zero-polynomial G2 catalogue");
        const auto model_anchor = gatzk::crypto::KZG::commit(
            "V_model_test",
            gatzk::algebra::Polynomial::from_coefficients(
                "V_model_test",
                {FieldElement(4), FieldElement(6), FieldElement(8)}),
            keys);
        const auto [proof_local, blinder] =
            gatzk::crypto::KZG::reblind_domain_polynomial(
                model_anchor,
                "V_model_test_proof",
                "S_model_test",
                "C",
                keys.manifest.domain_sizes.at("C"),
                4,
                keys);
        require(
            gatzk::crypto::KZG::verify_domain_divisibility(
                proof_local, model_anchor, blinder, "C", keys.vk),
            "proof-local model reblinding divisibility must verify");
        auto tampered_proof_local = proof_local;
        tampered_proof_local.point = gatzk::crypto::g1_add(
            tampered_proof_local.point,
            keys.g1_generator);
        require(
            !gatzk::crypto::KZG::verify_domain_divisibility(
                tampered_proof_local, model_anchor, blinder, "C", keys.vk),
            "tampered proof-local model commitment must reject");
        auto tampered_blinder = blinder;
        tampered_blinder.point = gatzk::crypto::g1_add(
            tampered_blinder.point,
            keys.g1_generator);
        require(
            !gatzk::crypto::KZG::verify_domain_divisibility(
                proof_local, model_anchor, tampered_blinder, "C", keys.vk),
            "tampered model reblinder commitment must reject");

        const auto private_domain = gatzk::algebra::RootOfUnityDomain::create(
            "N", keys.manifest.domain_sizes.at("N"));
        std::vector<FieldElement> private_values(private_domain->size, FieldElement::zero());
        for (std::size_t i = 0; i < private_values.size(); ++i)
        {
            private_values[i] = FieldElement(31 + i * 7);
        }
        auto unblinded_private = gatzk::algebra::Polynomial::from_evaluations(
            "P_private_unblinded", private_values, private_domain);
        auto blinded_private = gatzk::algebra::Polynomial::from_evaluations(
            "P_private_blinded", std::move(private_values), private_domain);
        blinded_private.set_zero_domain_blinder(
            {FieldElement(5), FieldElement(9), FieldElement(17), FieldElement(33)});
        for (std::size_t i = 0; i < private_domain->size; ++i)
        {
            require(
                blinded_private.evaluate(private_domain->point_at(i)) == unblinded_private.evaluate(private_domain->point_at(i)),
                "zero-domain blinding must preserve every canonical trace value");
        }
        const auto private_point = choose_non_domain_point(*private_domain);
        require(
            !(blinded_private.evaluate(private_point) == unblinded_private.evaluate(private_point)),
            "zero-domain blinding must change a private off-domain observation");
        const auto unblinded_commitment = gatzk::crypto::KZG::commit(
            "P_private_unblinded", unblinded_private, keys);
        const auto blinded_commitment = gatzk::crypto::KZG::commit(
            "P_private_blinded", blinded_private, keys);
        require(
            !(unblinded_commitment.point == blinded_commitment.point),
            "zero-domain blinding must change the private commitment");
        const auto private_evaluation_component =
            gatzk::crypto::KZG::evaluation_commitment_component(
                unblinded_commitment, keys);
        const auto derived_blinded_commitment =
            gatzk::crypto::KZG::commit_evaluation_from_component(
                "P_private_blinded",
                blinded_private,
                private_evaluation_component,
                keys);
        require(
            derived_blinded_commitment.point == blinded_commitment.point,
            "evaluation-component reblinding must equal the direct KZG commitment");
        const auto recovered_private_component =
            gatzk::crypto::KZG::evaluation_commitment_component(
                blinded_commitment, keys);
        require(
            recovered_private_component == private_evaluation_component,
            "evaluation-component extraction must remove the independent blinder");
        const std::vector<FieldElement> private_points{
            private_point, private_point * private_domain->omega};
        const std::vector<std::vector<FieldElement>> private_claims{{
            blinded_private.evaluate(private_points[0]),
            blinded_private.evaluate(private_points[1]),
        }};
        const auto private_opening = open_matrix_point_groups(
            {blinded_commitment}, private_points, private_claims, FieldElement(41), keys);
        require(
            verify_matrix_point_groups(
                {blinded_commitment}, private_points, private_claims,
                FieldElement(41), private_opening, keys.vk),
            "zero-domain-blinded point-group opening must verify");
        auto invalid_private_claim = private_claims;
        invalid_private_claim[0][0] += FieldElement::one();
        require_throws(
            [&]
            {
                (void)open_matrix_point_groups(
                    {blinded_commitment}, private_points, invalid_private_claim,
                    FieldElement(41), keys);
            },
            "KZG prover must reject an incorrect individual evaluation claim");
        auto private_claim_tamper = private_claims;
        private_claim_tamper[0][1] += FieldElement::one();
        require(
            !verify_matrix_point_groups(
                {blinded_commitment}, private_points, private_claim_tamper,
                FieldElement(41), private_opening, keys.vk),
            "zero-domain-blinded opening must reject a claimed-value tamper");

        const auto fold_domain = gatzk::algebra::RootOfUnityDomain::create(
            "C", keys.manifest.domain_sizes.at("C"));
        std::vector<FieldElement> fold_values_0(fold_domain->size);
        std::vector<FieldElement> fold_values_1(fold_domain->size);
        for (std::size_t i = 0; i < fold_domain->size; ++i)
        {
            fold_values_0[i] = FieldElement(17 + 3 * i);
            fold_values_1[i] = FieldElement(29 + 5 * i);
        }
        auto fold_polynomial_0 = gatzk::algebra::Polynomial::from_evaluations(
            "P_fold_0", fold_values_0, fold_domain);
        auto fold_polynomial_1 = gatzk::algebra::Polynomial::from_evaluations(
            "P_fold_1", fold_values_1, fold_domain);
        fold_polynomial_0.set_zero_domain_blinder(
            {FieldElement(7), FieldElement(11)});
        fold_polynomial_1.set_zero_domain_blinder(
            {FieldElement(13), FieldElement(19)});
        const auto fold_commitment_0 = gatzk::crypto::KZG::commit(
            "P_fold_0", fold_polynomial_0, keys);
        const auto fold_commitment_1 = gatzk::crypto::KZG::commit(
            "P_fold_1", fold_polynomial_1, keys);
        const FieldElement fold_challenge(43);
        const auto fold_point = choose_non_domain_point(*fold_domain);
        const std::vector<FieldElement> fold_points{
            fold_point, fold_point * fold_domain->omega};
        const std::vector<std::vector<FieldElement>> fold_claims{
            {fold_polynomial_0.evaluate(fold_points[0]),
             fold_polynomial_0.evaluate(fold_points[1])},
            {fold_polynomial_1.evaluate(fold_points[0]),
             fold_polynomial_1.evaluate(fold_points[1])},
        };
        const auto folded_batch_opening = open_matrix_point_groups(
            {fold_commitment_0, fold_commitment_1},
            fold_points,
            fold_claims,
            fold_challenge,
            keys);

        std::vector<FieldElement> explicit_fold_values(fold_domain->size);
        for (std::size_t i = 0; i < fold_domain->size; ++i)
        {
            explicit_fold_values[i] =
                fold_values_0[i] + fold_values_1[i] * fold_challenge;
        }
        auto explicit_fold_polynomial =
            gatzk::algebra::Polynomial::from_evaluations(
                "P_fold_explicit", std::move(explicit_fold_values), fold_domain);
        explicit_fold_polynomial.set_zero_domain_blinder({
            FieldElement(7) + FieldElement(13) * fold_challenge,
            FieldElement(11) + FieldElement(19) * fold_challenge,
        });
        const auto explicit_fold_commitment = gatzk::crypto::KZG::commit(
            "P_fold_explicit", explicit_fold_polynomial, keys);
        require(
            explicit_fold_commitment.point == gatzk::crypto::g1_add(
                                                  fold_commitment_0.point,
                                                  gatzk::crypto::g1_mul(
                                                      fold_commitment_1.point, fold_challenge)),
            "folded evaluation commitment must equal the explicit polynomial commitment");
        const std::vector<std::vector<FieldElement>> explicit_fold_claims{{
            fold_claims[0][0] + fold_claims[1][0] * fold_challenge,
            fold_claims[0][1] + fold_claims[1][1] * fold_challenge,
        }};
        const auto explicit_fold_opening = open_matrix_point_groups(
            {explicit_fold_commitment},
            {fold_points.front()},
            {{explicit_fold_claims.front().front()}},
            FieldElement(47),
            keys);
        require(
            folded_batch_opening.witnesses.front() == explicit_fold_opening.witnesses.front(),
            "single point-group folded witness must equal its explicit polynomial witness");
        require(
            verify_matrix_point_groups(
                {fold_commitment_0, fold_commitment_1},
                fold_points,
                fold_claims,
                fold_challenge,
                folded_batch_opening,
                keys.vk),
            "independently challenged point groups must verify");
    }

    void test_paired_streamed_point_group_equivalence()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(),
            "trusted_srs",
            test_srs_id());

        const auto domain =
            gatzk::algebra::RootOfUnityDomain::create(
                "N",
                keys.manifest.domain_sizes.at("N"));

        auto shared_values =
            std::make_shared<const std::vector<FieldElement>>(
                [&]()
                {
                    std::vector<FieldElement> values(domain->size);
                    for (std::size_t i = 0; i < values.size(); ++i)
                    {
                        values[i] = FieldElement(
                            static_cast<std::uint64_t>(
                                17 + 13 * i + i * i));
                    }
                    return values;
                }());

        auto polynomial_0 =
            gatzk::algebra::Polynomial::from_shared_evaluations(
                "P_paired_stream_0",
                shared_values,
                domain);

        auto polynomial_1 =
            gatzk::algebra::Polynomial::from_shared_evaluations(
                "P_paired_stream_1",
                shared_values,
                domain);

        polynomial_0.set_zero_domain_blinder(
            {FieldElement(23), FieldElement(29)});

        polynomial_1.set_zero_domain_blinder(
            {FieldElement(31), FieldElement(37)});

        auto commitment_0 =
            gatzk::crypto::KZG::commit(
                polynomial_0.name,
                polynomial_0,
                keys);

        auto commitment_1 =
            gatzk::crypto::KZG::commit(
                polynomial_1.name,
                polynomial_1,
                keys);

        const auto material_0 =
            commitment_0.opening_polynomial;

        const auto material_1 =
            commitment_1.opening_polynomial;

        commitment_0.opening_polynomial.reset();
        commitment_1.opening_polynomial.reset();

        const gatzk::crypto::OpeningPolynomialResolver
            opening_resolver =
                [&](const gatzk::crypto::Commitment &requested)
            -> std::shared_ptr<
                const gatzk::algebra::Polynomial>
        {
            if (requested.name == commitment_0.name)
            {
                return material_0;
            }
            if (requested.name == commitment_1.name)
            {
                return material_1;
            }
            throw std::runtime_error(
                "paired streamed regression received unknown commitment");
        };

        const gatzk::crypto::OpeningSemanticSourceResolver
            semantic_source_resolver =
                [&](const gatzk::crypto::Commitment &requested)
            -> std::optional<
                gatzk::crypto::OpeningSemanticSourceInfo>
        {
            const auto material =
                opening_resolver(requested);

            return gatzk::crypto::OpeningSemanticSourceInfo{
                .key =
                    "paired-streamed-unit-shared-source",
                .domain = domain,
                .value_count = domain->size,
                .zero_domain_blinder =
                    material->zero_domain_blinder,
            };
        };

        const auto first_point =
            choose_non_domain_point(*domain);

        const std::array<FieldElement, 2> points{
            first_point,
            first_point * domain->omega,
        };

        const std::array<
            std::vector<gatzk::crypto::Commitment>,
            2>
            commitments{
                std::vector<gatzk::crypto::Commitment>{
                    commitment_0,
                    commitment_1,
                },
                std::vector<gatzk::crypto::Commitment>{
                    commitment_0,
                    commitment_1,
                },
            };

        const std::array<
            std::vector<FieldElement>,
            2>
            claimed_values{
                std::vector<FieldElement>{
                    polynomial_0.evaluate(points[0]),
                    polynomial_1.evaluate(points[0]),
                },
                std::vector<FieldElement>{
                    polynomial_0.evaluate(points[1]),
                    polynomial_1.evaluate(points[1]),
                },
            };

        const auto run_case =
            [&](const std::array<FieldElement, 2> &challenges,
                const std::string &label)
        {
            const auto paired =
                gatzk::crypto::KZG::
                    open_point_group_pair_shared_stream(
                        commitments,
                        points,
                        claimed_values,
                        challenges,
                        keys,
                        nullptr,
                        gatzk::crypto::
                            BatchOpeningClaimValidation::
                                AlreadyComputedFromOpeningPolynomials,
                        nullptr,
                        &opening_resolver,
                        &semantic_source_resolver);

            for (std::size_t group = 0;
                 group < 2;
                 ++group)
            {
                const auto reference =
                    gatzk::crypto::KZG::open_point_group(
                        commitments[group],
                        points[group],
                        claimed_values[group],
                        challenges[group],
                        keys,
                        nullptr,
                        gatzk::crypto::
                            BatchOpeningClaimValidation::
                                AlreadyComputedFromOpeningPolynomials,
                        nullptr,
                        &opening_resolver,
                        &semantic_source_resolver);

                require(
                    paired[group] == reference,
                    "paired streamed opening changed the "
                    "canonical witness: " +
                        label);

                require(
                    gatzk::crypto::KZG::verify_point_group(
                        commitments[group],
                        points[group],
                        claimed_values[group],
                        challenges[group],
                        paired[group],
                        keys.vk),
                    "paired streamed opening failed verification: " + label);
            }
        };

        run_case(
            {
                FieldElement(41),
                FieldElement(43),
            },
            "shared-source");

        const auto minus_one =
            FieldElement::zero() - FieldElement::one();

        run_case(
            {
                minus_one,
                minus_one,
            },
            "zero-source-scale");
    }

    void test_coefficient_taylor_shift_fixed_shape()
    {
        const std::size_t configured_degree = 7;
        std::vector<FieldElement> coefficients(configured_degree + 1, FieldElement::zero());
        coefficients[0] = FieldElement(3);
        coefficients[1] = FieldElement(5);
        coefficients[3] = FieldElement(11);
        coefficients[7] = FieldElement(17);
        const FieldElement delta(19);
        const auto shifted = gatzk::algebra::coefficient_taylor_shift(
            coefficients, configured_degree, delta);
        require(
            shifted.size() == configured_degree + 1,
            "Taylor shift must preserve the compiler-fixed coefficient length");
        for (const auto point : {FieldElement(2), FieldElement(23), FieldElement(41)})
        {
            require(
                gatzk::algebra::horner(shifted, point) == gatzk::algebra::horner(coefficients, point + delta),
                "coefficient Taylor shift must satisfy F_shift(X)=F(X+delta)");
        }
        auto wrong_length = coefficients;
        wrong_length.pop_back();
        require_throws(
            [&]
            {
                (void)gatzk::algebra::coefficient_taylor_shift(
                    wrong_length, configured_degree, delta);
            },
            "Taylor shift must reject witness-dependent coefficient trimming");

        const auto domain = gatzk::algebra::RootOfUnityDomain::create("N", 8);
        std::vector<FieldElement> values(8, FieldElement::zero());
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] = FieldElement(7 + i * 3);
        auto evaluation = gatzk::algebra::Polynomial::from_evaluations(
            "P_shift_material", values, domain);
        evaluation.set_zero_domain_blinder({FieldElement(13), FieldElement(29)});
        const auto materialized = gatzk::algebra::materialize_blinded_coefficients(
            evaluation, 9);
        require(materialized.size() == 10,
                "blinded coefficient materialization must preserve configured length");
        for (const auto point : {FieldElement(3), FieldElement(31)})
        {
            require(
                gatzk::algebra::horner(materialized, point) == evaluation.evaluate(point),
                "materialized coefficients must include the exact domain blinder");
        }

        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(), "trusted_srs", test_srs_id());
        const auto source = gatzk::crypto::KZG::commit(
            "P_shift_source", evaluation, keys);
        const auto shifted_commitment = gatzk::crypto::KZG::commit_taylor_shifted(
            source, "P_shifted", 9, delta, keys);
        const FieldElement relation_point(43);
        const auto relation_value = shifted_commitment.opening_polynomial->evaluate(relation_point);
        require(
            relation_value == source.opening_polynomial->evaluate(relation_point + delta),
            "Taylor-shift commitment material must share the registered relation value");
        const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> claims{
            {shifted_commitment, relation_value},
            {source, relation_value},
        };
        const std::vector<FieldElement> points{relation_point, relation_point + delta};
        const auto opening = open_claim_point_groups(
            claims, points, FieldElement(47), keys);
        require(
            verify_claim_point_groups(
                claims, points, FieldElement(47), opening, keys.vk),
            "Taylor-shift relation claims must verify through standard point groups");
        auto tampered_claims = claims;
        tampered_claims[0].first.point = gatzk::crypto::g1_add(
            tampered_claims[0].first.point, keys.g1_generator);
        require(
            !verify_claim_point_groups(
                tampered_claims, points, FieldElement(47), opening, keys.vk),
            "tampered Taylor-shift commitment must be rejected");
    }

    void test_cross_domain_indexed_view_shift_relation()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(), "trusted_srs", test_srs_id());
        const auto canonical_domain = gatzk::algebra::RootOfUnityDomain::create("C", 4);
        const auto view_domain = gatzk::algebra::RootOfUnityDomain::create("N", 8);
        constexpr std::size_t class_count = 2;
        constexpr std::size_t row_count = 3;
        std::vector<FieldElement> canonical_indices(4, FieldElement::zero());
        std::vector<FieldElement> canonical_values(4, FieldElement::zero());
        std::vector<FieldElement> canonical_selector(4, FieldElement::zero());
        std::vector<FieldElement> canonical_multiplicity(4, FieldElement::zero());
        const std::array<FieldElement, class_count> bias{
            FieldElement(5), FieldElement(11)};
        for (std::size_t column = 0; column < class_count; ++column)
        {
            canonical_indices[column] = FieldElement(column);
            canonical_values[column] = bias[column];
            canonical_selector[column] = FieldElement::one();
            canonical_multiplicity[column] = FieldElement(row_count);
        }
        std::vector<FieldElement> view_indices(8, FieldElement::zero());
        std::vector<FieldElement> view_values(8, FieldElement::zero());
        std::vector<FieldElement> view_selector(8, FieldElement::zero());
        std::vector<FieldElement> view_multiplicity(8, FieldElement::zero());
        for (std::size_t row = 0; row < row_count; ++row)
        {
            for (std::size_t column = 0; column < class_count; ++column)
            {
                const auto index = row * class_count + column;
                view_indices[index] = FieldElement(column);
                view_values[index] = bias[column];
                view_selector[index] = FieldElement::one();
                view_multiplicity[index] = FieldElement::one();
            }
        }
        const FieldElement eta(31);
        const FieldElement beta(37);
        const auto canonical_side = gatzk::protocol::build_indexed_view_side(
            canonical_indices, canonical_values, canonical_selector,
            canonical_multiplicity, eta, beta);
        const auto view_side = gatzk::protocol::build_indexed_view_side(
            view_indices, view_values, view_selector,
            view_multiplicity, eta, beta);
        const auto canonical_prefix =
            gatzk::protocol::build_indexed_view_active_prefix_witness(
                canonical_side.tuples,
                class_count,
                FieldElement(row_count),
                beta);
        const auto view_prefix =
            gatzk::protocol::build_indexed_view_active_prefix_witness(
                view_side.tuples,
                row_count * class_count,
                FieldElement::one(),
                beta);
        require(
            canonical_prefix.inverses == canonical_side.inverses && canonical_prefix.accumulator == canonical_side.accumulator && view_prefix.inverses == view_side.inverses && view_prefix.accumulator == view_side.accumulator,
            "active-prefix indexed-view witness must match selector route");
        require(
            canonical_side.accumulator[class_count] == view_side.accumulator[row_count * class_count],
            "indexed-view LogUp terminals must match across native domains");

        auto canonical_accumulator = gatzk::algebra::Polynomial::from_evaluations(
            "P_bias_R_can", canonical_side.accumulator, canonical_domain);
        canonical_accumulator.set_zero_domain_blinder({FieldElement(41), FieldElement(43)});
        std::vector<FieldElement> negative_view_values(view_side.accumulator.size());
        for (std::size_t i = 0; i < negative_view_values.size(); ++i)
            negative_view_values[i] = -view_side.accumulator[i];
        auto negative_view_accumulator = gatzk::algebra::Polynomial::from_evaluations(
            "P_bias_R_qry_neg", std::move(negative_view_values), view_domain);
        negative_view_accumulator.set_zero_domain_blinder({FieldElement(47), FieldElement(53)});
        const auto canonical_commitment = gatzk::crypto::KZG::commit(
            "P_bias_R_can", canonical_accumulator, keys);
        const auto negative_view_commitment = gatzk::crypto::KZG::commit(
            "P_bias_R_qry_neg", negative_view_accumulator, keys);
        auto canonical_prefix_accumulator =
            gatzk::algebra::Polynomial::from_evaluations(
                "P_bias_R_can", canonical_prefix.accumulator, canonical_domain);
        canonical_prefix_accumulator.set_zero_domain_blinder(
            {FieldElement(41), FieldElement(43)});
        std::vector<FieldElement> negative_prefix_view_values(
            view_prefix.accumulator.size());
        for (std::size_t i = 0; i < negative_prefix_view_values.size(); ++i)
            negative_prefix_view_values[i] = -view_prefix.accumulator[i];
        auto negative_prefix_view_accumulator =
            gatzk::algebra::Polynomial::from_evaluations(
                "P_bias_R_qry_neg",
                std::move(negative_prefix_view_values),
                view_domain);
        negative_prefix_view_accumulator.set_zero_domain_blinder(
            {FieldElement(47), FieldElement(53)});
        require(
            gatzk::crypto::KZG::commit(
                "P_bias_R_can", canonical_prefix_accumulator, keys)
                        .point == canonical_commitment.point &&
                gatzk::crypto::KZG::commit(
                    "P_bias_R_qry_neg",
                    negative_prefix_view_accumulator,
                    keys)
                        .point == negative_view_commitment.point,
            "active-prefix indexed-view accumulator commitments must match selector route");
        const auto z_star = canonical_domain->point_at(class_count);
        const auto z_query = view_domain->point_at(row_count * class_count);
        const auto delta = z_query - z_star;
        constexpr std::size_t shift_degree = 15;
        const auto shifted_query = gatzk::crypto::KZG::commit_taylor_shifted(
            negative_view_commitment, "P_bias_R_qry_shifted",
            shift_degree, delta, keys);

        auto relation_coefficients = gatzk::algebra::materialize_blinded_coefficients(
            canonical_accumulator, shift_degree);
        const auto &shifted_coefficients = shifted_query.opening_polynomial->data;
        for (std::size_t i = 0; i < relation_coefficients.size(); ++i)
            relation_coefficients[i] += shifted_coefficients[i];
        const auto relation_commitment = gatzk::crypto::KZG::commit(
            "P_bias_terminal_relation",
            gatzk::algebra::Polynomial::from_coefficients(
                "P_bias_terminal_relation", std::move(relation_coefficients)),
            keys);
        require(
            relation_commitment.point == gatzk::crypto::g1_add(
                                             canonical_commitment.point, shifted_query.point),
            "indexed-view relation commitment must be verifier-reconstructible");
        require(
            relation_commitment.opening_polynomial->evaluate(z_star).is_zero(),
            "cross-domain indexed-view terminal relation must be zero at z_star");

        gatzk::crypto::Transcript relation_transcript("zkgat:kzg-shift-relation:v1");
        relation_transcript.absorb_text("relation_id", "bias_indexed_view_terminal");
        relation_transcript.absorb_commitment(
            "source", negative_view_commitment.point);
        relation_transcript.absorb_commitment("shifted", shifted_query.point);
        relation_transcript.absorb_scalar("z_star", z_star);
        relation_transcript.absorb_scalar("delta", delta);
        const auto relation_check_point = relation_transcript.challenge_excluding(
            "y_rel",
            [&](const FieldElement &candidate)
            {
                return candidate.is_zero() || view_domain->zero_polynomial_eval(candidate + delta).is_zero();
            });
        gatzk::crypto::Transcript replay_transcript("zkgat:kzg-shift-relation:v1");
        replay_transcript.absorb_text("relation_id", "bias_indexed_view_terminal");
        replay_transcript.absorb_commitment("source", negative_view_commitment.point);
        replay_transcript.absorb_commitment("shifted", shifted_query.point);
        replay_transcript.absorb_scalar("z_star", z_star);
        replay_transcript.absorb_scalar("delta", delta);
        require(
            replay_transcript.challenge_excluding(
                "y_rel",
                [&](const FieldElement &candidate)
                {
                    return candidate.is_zero() || view_domain->zero_polynomial_eval(candidate + delta).is_zero();
                }) == relation_check_point,
            "KZG shift challenge exclusion counter must replay deterministically");
        const auto shift_value = shifted_query.opening_polynomial->evaluate(
            relation_check_point);
        require(
            shift_value == negative_view_commitment.opening_polynomial->evaluate(
                               relation_check_point + delta),
            "indexed-view terminal shift must reuse one masked relation value");
        const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> claims{
            {shifted_query, shift_value},
            {negative_view_commitment, shift_value},
            {relation_commitment, FieldElement::zero()},
        };
        const std::vector<FieldElement> points{
            relation_check_point, relation_check_point + delta, z_star};
        const auto opening = open_claim_point_groups(
            claims, points, FieldElement(61), keys);
        require(
            verify_claim_point_groups(
                claims, points, FieldElement(61), opening, keys.vk),
            "cross-domain indexed-view relation must verify via standard point groups");
        auto tampered_claims = claims;
        tampered_claims[1].first.point = gatzk::crypto::g1_add(
            tampered_claims[1].first.point, keys.g1_generator);
        require(
            !verify_claim_point_groups(
                tampered_claims, points, FieldElement(61), opening, keys.vk),
            "cross-domain indexed-view source commitment tamper must reject");

        auto invalid_multiplicity = canonical_multiplicity;
        invalid_multiplicity.back() = FieldElement::one();
        require_throws(
            [&]
            {
                (void)gatzk::protocol::build_indexed_view_side(
                    canonical_indices, canonical_values, canonical_selector,
                    invalid_multiplicity, eta, beta);
            },
            "indexed-view multiplicity outside the canonical selector must reject");
    }

    void test_same_point_indexed_view_relation()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(), "trusted_srs", test_srs_id());
        const auto canonical_domain =
            gatzk::algebra::RootOfUnityDomain::create("C", 4);
        const auto view_domain =
            gatzk::algebra::RootOfUnityDomain::create("N", 8);
        constexpr std::size_t class_count = 2;
        constexpr std::size_t row_count = 2;
        const auto z_c = canonical_domain->point_at(class_count);
        const auto z_edge = view_domain->point_at(row_count * class_count);
        require(z_c == z_edge,
                "m=1 indexed-view fixture must collide after canonical point merging");

        std::vector<FieldElement> canonical_indices(4, FieldElement::zero());
        std::vector<FieldElement> canonical_values(4, FieldElement::zero());
        std::vector<FieldElement> canonical_selector(4, FieldElement::zero());
        std::vector<FieldElement> canonical_multiplicity(4, FieldElement::zero());
        for (std::size_t column = 0; column < class_count; ++column)
        {
            canonical_indices[column] = FieldElement(column);
            canonical_values[column] = FieldElement(7 + 6 * column);
            canonical_selector[column] = FieldElement::one();
            canonical_multiplicity[column] = FieldElement(row_count);
        }
        std::vector<FieldElement> view_indices(8, FieldElement::zero());
        std::vector<FieldElement> view_values(8, FieldElement::zero());
        std::vector<FieldElement> view_selector(8, FieldElement::zero());
        std::vector<FieldElement> view_multiplicity(8, FieldElement::zero());
        for (std::size_t row = 0; row < row_count; ++row)
        {
            for (std::size_t column = 0; column < class_count; ++column)
            {
                const auto index = row * class_count + column;
                view_indices[index] = FieldElement(column);
                view_values[index] = canonical_values[column];
                view_selector[index] = FieldElement::one();
                view_multiplicity[index] = FieldElement::one();
            }
        }
        const auto canonical_side = gatzk::protocol::build_indexed_view_side(
            canonical_indices, canonical_values, canonical_selector,
            canonical_multiplicity, FieldElement(31), FieldElement(37));
        const auto view_side = gatzk::protocol::build_indexed_view_side(
            view_indices, view_values, view_selector,
            view_multiplicity, FieldElement(31), FieldElement(37));
        require(canonical_side.accumulator[class_count] == view_side.accumulator[row_count * class_count],
                "m=1 indexed-view terminals must agree before blinding");

        auto r_can = gatzk::algebra::Polynomial::from_evaluations(
            "P_same_R_can", canonical_side.accumulator, canonical_domain);
        r_can.set_zero_domain_blinder({FieldElement(41), FieldElement(43)});
        auto r_qry = gatzk::algebra::Polynomial::from_evaluations(
            "P_same_R_qry", view_side.accumulator, view_domain);
        r_qry.set_zero_domain_blinder({FieldElement(47), FieldElement(53)});
        const auto c_can = gatzk::crypto::KZG::commit("P_same_R_can", r_can, keys);
        const auto c_qry = gatzk::crypto::KZG::commit("P_same_R_qry", r_qry, keys);

        constexpr std::size_t relation_degree = 15;
        auto relation_coefficients =
            gatzk::algebra::materialize_blinded_coefficients(r_can, relation_degree);
        const auto query_coefficients =
            gatzk::algebra::materialize_blinded_coefficients(r_qry, relation_degree);
        for (std::size_t i = 0; i < relation_coefficients.size(); ++i)
            relation_coefficients[i] -= query_coefficients[i];
        const auto relation_polynomial =
            gatzk::algebra::Polynomial::from_coefficients(
                "P_same_G_rel", std::move(relation_coefficients));
        const auto relation_commitment = gatzk::crypto::KZG::commit(
            "P_same_G_rel", relation_polynomial, keys);
        require(relation_commitment.point == gatzk::crypto::g1_add(
                                                 c_can.point, gatzk::crypto::g1_neg(c_qry.point)),
                "m=1 G_rel commitment must be verifier-reconstructible");
        require(relation_polynomial.evaluate(z_c).is_zero(),
                "m=1 G_rel must have the ordinary zero claim at the merged point");

        const auto multiplicity_polynomial =
            gatzk::algebra::Polynomial::from_evaluations(
                "P_same_m_can", canonical_multiplicity, canonical_domain);
        const auto multiplicity_commitment = gatzk::crypto::KZG::commit(
            "P_same_m_can", multiplicity_polynomial, keys);
        const FieldElement y_public(59);
        const auto public_value = multiplicity_polynomial.evaluate(y_public);
        const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> claims{
            {relation_commitment, FieldElement::zero()},
            {multiplicity_commitment, public_value},
        };
        const std::vector<FieldElement> points{z_c, y_public};
        const auto opening = open_claim_point_groups(
            claims, points, FieldElement(61), keys);
        require(verify_claim_point_groups(
                    claims, points, FieldElement(61), opening, keys.vk),
                "m=1 relation must verify only through ordinary point-group claims");
        auto tampered_commitment = claims;
        tampered_commitment[0].first.point = gatzk::crypto::g1_add(
            tampered_commitment[0].first.point, keys.g1_generator);
        require(!verify_claim_point_groups(
                    tampered_commitment, points, FieldElement(61), opening, keys.vk),
                "m=1 reconstructed G_rel commitment tamper must reject");
        auto tampered_zero = claims;
        tampered_zero[0].second = FieldElement::one();
        require(!verify_claim_point_groups(
                    tampered_zero, points, FieldElement(61), opening, keys.vk),
                "m=1 relation zero-claim tamper must reject");
    }

    void test_compact_eval_kzg_regression()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(),
            "trusted_srs",
            test_srs_id());

        const auto exact_n_it = keys.manifest.domain_sizes.find("N_4096");
        if (keys.manifest.domain_sizes.at("N") > 4096)
        {
            require(exact_n_it != keys.manifest.domain_sizes.end(), "expanded SRS must include exact N_4096 material");
        }
        const auto n_size = exact_n_it != keys.manifest.domain_sizes.end()
                                ? exact_n_it->second
                                : keys.manifest.domain_sizes.at("N");
        auto domain = gatzk::algebra::RootOfUnityDomain::create("N", n_size);
        auto polynomial = gatzk::algebra::Polynomial::from_evaluations(
            "P_compact_eval",
            {FieldElement(3), FieldElement(5), FieldElement(7), FieldElement(11), FieldElement(13)},
            domain);
        polynomial.set_zero_domain_blinder(
            {FieldElement(17), FieldElement(19)});
        std::vector<FieldElement> padded_values(
            domain->size, FieldElement::zero());
        std::copy(
            polynomial.values().begin(),
            polynomial.values().end(),
            padded_values.begin());
        auto padded_polynomial =
            gatzk::algebra::Polynomial::from_evaluations(
                "P_compact_eval_padded",
                std::move(padded_values),
                domain);
        padded_polynomial.set_zero_domain_blinder(
            polynomial.zero_domain_blinder);
        const auto point = choose_non_domain_point(*domain);
        const auto value = polynomial.evaluate(point);
        const auto commitment = gatzk::crypto::KZG::commit("P_compact_eval", polynomial, keys);
        const auto padded_commitment = gatzk::crypto::KZG::commit(
            "P_compact_eval_padded", padded_polynomial, keys);
        require(
            commitment.point == padded_commitment.point && value == padded_polynomial.evaluate(point),
            "compact evaluation storage must equal its explicit zero-padded polynomial");
        const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> openings{
            {commitment, value},
        };
        const std::vector<FieldElement> points{point};
        const auto proof = open_claim_point_groups(
            openings, points, FieldElement(29), keys);

        require(
            verify_claim_point_groups(
                openings, points, FieldElement(29), proof, keys.vk),
            "compact evaluation external opening must verify");

        auto tampered = openings;
        tampered[0].second += FieldElement::one();
        require(
            !verify_claim_point_groups(
                tampered, points, FieldElement(29), proof, keys.vk),
            "compact evaluation external opening must reject a tampered value");
    }

    void test_evaluation_commitment_cache_lifetime()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(),
            "trusted_srs",
            test_srs_id());
        const auto domain_size = keys.manifest.domain_sizes.at("N");
        auto domain =
            gatzk::algebra::RootOfUnityDomain::create(
                "N", domain_size);
        gatzk::crypto::EvaluationCommitmentCache cache;

        struct AggregateOwner
        {
            std::vector<FieldElement> values;
            std::vector<FieldElement> unrelated_large_member;
        };
        std::weak_ptr<AggregateOwner> aggregate_lifetime;
        {
            auto owner = std::make_shared<AggregateOwner>();
            owner->values = {
                FieldElement(2),
                FieldElement(3),
                FieldElement(5),
                FieldElement(7),
            };
            owner->unrelated_large_member.assign(
                4096, FieldElement(11));
            aggregate_lifetime = owner;
            auto aliased_values =
                std::shared_ptr<const std::vector<FieldElement>>(
                    owner, &owner->values);
            auto polynomial =
                gatzk::algebra::Polynomial::from_shared_evaluations(
                    "P_cache_lifetime",
                    aliased_values,
                    domain);
            const std::vector<std::pair<
                std::string,
                const gatzk::algebra::Polynomial *>>
                batch{
                    {"P_cache_lifetime", &polynomial},
                };
            auto commitments = gatzk::crypto::KZG::commit_batch(
                batch, keys, nullptr, &cache);
            commitments.front().opening_polynomial.reset();
        }
        require(
            aggregate_lifetime.expired(),
            "evaluation commitment cache must not retain an aggregate column owner");

        auto live_values =
            std::make_shared<const std::vector<FieldElement>>(
                std::initializer_list<FieldElement>{
                    FieldElement(13),
                    FieldElement(17),
                    FieldElement(19),
                    FieldElement(23),
                });
        auto first =
            gatzk::algebra::Polynomial::from_shared_evaluations(
                "P_cache_live_first", live_values, domain);
        const std::vector<std::pair<
            std::string,
            const gatzk::algebra::Polynomial *>>
            first_batch{
                {"P_cache_live_first", &first},
            };
        (void)gatzk::crypto::KZG::commit_batch(
            first_batch, keys, nullptr, &cache);

        auto second =
            gatzk::algebra::Polynomial::from_shared_evaluations(
                "P_cache_live_second", live_values, domain);
        const std::vector<std::pair<
            std::string,
            const gatzk::algebra::Polynomial *>>
            second_batch{
                {"P_cache_live_second", &second},
            };
        gatzk::crypto::CommitBatchProfile profile;
        const auto reused = gatzk::crypto::KZG::commit_batch(
            second_batch, keys, &profile, &cache);
        require(
            profile.objects.size() == 1 && profile.objects.front().semantic_commitment_reuse,
            "live shared evaluation values must retain semantic commitment reuse");
        require(
            reused.front().point == gatzk::crypto::KZG::commit(
                                        "P_cache_live_reference",
                                        second,
                                        keys)
                                        .point,
            "semantic commitment reuse must preserve the exact commitment point");
    }

    void test_file_backed_taylor_shift_source_resolution()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(),
            "trusted_srs",
            test_srs_id());
        const auto domain_size = keys.manifest.domain_sizes.at("N");
        auto domain = gatzk::algebra::RootOfUnityDomain::create(
            "N", domain_size);
        std::vector<FieldElement> values{
            FieldElement(2),
            FieldElement(3),
            FieldElement(5),
            FieldElement(7),
            FieldElement(11),
        };
        auto polynomial = gatzk::algebra::Polynomial::from_evaluations(
            "P_spilled_shift_source", values, domain);
        polynomial.set_zero_domain_blinder(
            {FieldElement(13), FieldElement(17)});
        auto commitment = gatzk::crypto::KZG::commit(
            polynomial.name, polynomial, keys);

        gatzk::protocol::TraceArtifacts trace;
        trace.commitments.emplace(polynomial.name, commitment);
        trace.polynomial_domains[polynomial.name] = domain->name;
        require(
            gatzk::protocol::resolve_trace_opening_polynomial(
                trace, polynomial.name, domain) == commitment.opening_polynomial,
            "resident trace opening resolution must preserve its registered material");

        const auto spill_path =
            std::filesystem::path(GATZK_TEST_BINARY_DIR) / "trace_spilled_taylor_shift_source.bin";
        struct SpillFixtureCleanup
        {
            std::filesystem::path path;
            ~SpillFixtureCleanup()
            {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        } cleanup{spill_path};
        {
            std::ofstream output(
                spill_path, std::ios::binary | std::ios::trunc);
            require(
                static_cast<bool>(output),
                "failed to create file-backed Taylor-shift fixture");
            std::array<std::uint8_t, 32> bytes{};
            for (const auto &value : values)
            {
                value.write_little_endian(bytes.data(), bytes.size());
                output.write(
                    reinterpret_cast<const char *>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
            }
            require(
                static_cast<bool>(output),
                "failed to write file-backed Taylor-shift fixture");
        }

        trace.commitments.at(polynomial.name).opening_polynomial.reset();
        trace.spilled_polynomials.emplace(
            polynomial.name,
            gatzk::protocol::TraceArtifacts::SpilledEvaluationPolynomial{
                .path = spill_path.string(),
                .domain_name = domain->name,
                .size = values.size(),
                .zero_domain_blinder = polynomial.zero_domain_blinder,
            });
        const auto resolved =
            gatzk::protocol::resolve_trace_opening_polynomial(
                trace, polynomial.name, domain);
        require(
            resolved->basis == gatzk::algebra::PolynomialBasis::Evaluation && resolved->domain != nullptr && resolved->domain->name == domain->name && resolved->values() == values && resolved->zero_domain_blinder == polynomial.zero_domain_blinder,
            "file-backed Taylor-shift source must restore exact values, domain, and blinder");

        auto restored_commitment = trace.commitments.at(polynomial.name);
        restored_commitment.opening_polynomial = resolved;
        const auto configured_degree =
            domain->size + polynomial.zero_domain_blinder.size() - 1;
        const FieldElement delta(19);
        const auto restored_shift =
            gatzk::crypto::KZG::commit_taylor_shifted(
                restored_commitment,
                "P_spilled_shift_restored",
                configured_degree,
                delta,
                keys);
        const auto direct_shift =
            gatzk::crypto::KZG::commit_taylor_shifted(
                commitment,
                "P_spilled_shift_direct",
                configured_degree,
                delta,
                keys);
        require(
            restored_shift.point == direct_shift.point && restored_shift.opening_polynomial->data == direct_shift.opening_polynomial->data,
            "file-backed Taylor shift must equal the resident source path");

        const auto wrong_domain =
            gatzk::algebra::RootOfUnityDomain::create(
                "edge", domain->size);
        require_throws(
            [&]
            {
                (void)gatzk::protocol::resolve_trace_opening_polynomial(
                    trace, polynomial.name, wrong_domain);
            },
            "file-backed Taylor-shift source must reject a registered domain mismatch");
    }

    void test_external_point_grouping_regression()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(),
            "trusted_srs",
            test_srs_id());
        const auto p0 = gatzk::algebra::Polynomial::from_coefficients(
            "P_group_0", {FieldElement(2), FieldElement(3), FieldElement(5)});
        const auto p1 = gatzk::algebra::Polynomial::from_coefficients(
            "P_group_1", {FieldElement(7), FieldElement(11)});
        const auto p2 = gatzk::algebra::Polynomial::from_coefficients(
            "P_group_2", {FieldElement(13), FieldElement(17), FieldElement(19)});
        const auto c0 = gatzk::crypto::KZG::commit("P_group_0", p0, keys);
        const auto c1 = gatzk::crypto::KZG::commit("P_group_1", p1, keys);
        const auto c2 = gatzk::crypto::KZG::commit("P_group_2", p2, keys);
        const FieldElement z0(23);
        const FieldElement z1(29);
        const std::vector<std::pair<gatzk::crypto::Commitment, FieldElement>> claims = {
            {c0, p0.evaluate(z0)},
            {c1, p1.evaluate(z1)},
            {c2, p2.evaluate(z0)},
        };
        const std::vector<FieldElement> points = {z0, z1, z0};
        gatzk::protocol::PublicMetadata metadata;
        metadata.protocol_name = "zkgat-test";
        const std::vector<gatzk::protocol::PayloadValueKind> kinds(
            claims.size(),
            gatzk::protocol::PayloadValueKind::MaskedPrivateValue);
        gatzk::protocol::StandardPointGroupRegistry registry;
        gatzk::protocol::add_external_claims(registry, claims, points, kinds);
        registry.finalize();
        require(registry.groups().size() == 2,
                "external opening must encode one witness per distinct point");
        const auto rho0 = gatzk::protocol::derive_point_group_challenge(
            metadata, registry.group_at(z0));
        const auto rho1 = gatzk::protocol::derive_point_group_challenge(
            metadata, registry.group_at(z1));
        require(!rho0.is_zero() && !rho1.is_zero() && rho0 != rho1,
                "distinct points must derive independent non-zero challenges");

        std::vector<gatzk::protocol::PointGroupOpening> openings;
        for (const auto &group : registry.groups())
        {
            std::vector<gatzk::crypto::Commitment> group_commitments;
            std::vector<FieldElement> group_values;
            for (const auto &claim : group.claims)
            {
                group_commitments.push_back(claim.commitment);
                group_values.push_back(claim.value);
            }
            const auto rho = gatzk::protocol::derive_point_group_challenge(
                metadata, group);
            const auto witness = gatzk::crypto::KZG::open_point_group(
                group_commitments, group.point, group_values, rho, keys);
            require(gatzk::crypto::KZG::verify_point_group(
                        group_commitments, group.point, group_values, rho,
                        witness, keys.vk),
                    "point-grouped external opening must verify");
            openings.push_back({
                .group_id = group.group_id,
                .owner_block = group.owner_block,
                .point = group.point,
                .witness = witness,
            });
        }
        auto duplicate_witness = openings;
        duplicate_witness.push_back(openings.front());
        require_throws(
            [&]
            {
                gatzk::protocol::require_opening_layout(
                    duplicate_witness, registry.groups());
            },
            "duplicate point-group witness must reject");

        auto tampered_claims = claims;
        tampered_claims[0].second += FieldElement::one();
        gatzk::protocol::StandardPointGroupRegistry tampered_registry;
        gatzk::protocol::add_external_claims(
            tampered_registry, tampered_claims, points, kinds);
        tampered_registry.finalize();
        require(rho0 != gatzk::protocol::derive_point_group_challenge(
                            metadata, tampered_registry.group_at(z0)),
                "claimed values must be absorbed before the point-group folding challenge");

        auto duplicate_claims = claims;
        duplicate_claims.push_back(claims.front());
        auto duplicate_points = points;
        duplicate_points.push_back(z0);
        auto duplicate_kinds = kinds;
        duplicate_kinds.push_back(kinds.front());
        gatzk::protocol::StandardPointGroupRegistry deduplicated_registry;
        gatzk::protocol::add_external_claims(
            deduplicated_registry,
            duplicate_claims,
            duplicate_points,
            duplicate_kinds);
        deduplicated_registry.finalize();
        require(deduplicated_registry.group_at(z0).claims.size() == 2,
                "identical commitment-point claims must deduplicate");

        auto conflicting_claims = duplicate_claims;
        conflicting_claims.back().second += FieldElement::one();
        gatzk::protocol::StandardPointGroupRegistry conflict_registry;
        require_throws(
            [&]
            {
                gatzk::protocol::add_external_claims(
                    conflict_registry,
                    conflicting_claims,
                    duplicate_points,
                    duplicate_kinds);
            },
            "conflicting duplicate claims must reject before challenge derivation");

        auto reverse_claims = claims;
        auto reverse_points = points;
        auto reverse_kinds = kinds;
        std::reverse(reverse_claims.begin(), reverse_claims.end());
        std::reverse(reverse_points.begin(), reverse_points.end());
        std::reverse(reverse_kinds.begin(), reverse_kinds.end());
        gatzk::protocol::StandardPointGroupRegistry reverse_registry;
        gatzk::protocol::add_external_claims(
            reverse_registry, reverse_claims, reverse_points, reverse_kinds);
        reverse_registry.finalize();
        require(
            gatzk::protocol::derive_point_group_challenge(
                metadata, reverse_registry.group_at(z0)) == rho0 &&
                gatzk::protocol::derive_point_group_challenge(
                    metadata, reverse_registry.group_at(z1)) == rho1,
            "canonical groups and challenges must not depend on insertion order");

        gatzk::protocol::StandardPointGroupRegistry domain_merge_registry;
        domain_merge_registry.add_domain_claim(
            "unit",
            z0,
            gatzk::protocol::make_standard_opening_claim(
                "claim:domain:unit:P_group_0:0",
                c0.name,
                "domain_evaluation",
                "domain_polynomial",
                gatzk::protocol::PayloadValueKind::MaskedPrivateValue,
                "Eval_dom:unit:P_group_0:0",
                c0,
                p0.evaluate(z0)));
        gatzk::protocol::add_external_claims(
            domain_merge_registry,
            {{c2, p2.evaluate(z0)}},
            {z0},
            {gatzk::protocol::PayloadValueKind::MaskedPrivateValue});
        domain_merge_registry.finalize();
        require(domain_merge_registry.groups().size() == 1 && domain_merge_registry.groups().front().owner_block == "Open_dom:unit",
                "external claim at a work-domain point must merge into Open_dom");
    }

    void test_packed_backend_padded_tail_regression()
    {
        auto domain = gatzk::algebra::RootOfUnityDomain::create("packed_pad_regression", 4096);

        {
            std::vector<FieldElement> full_values(domain->size, FieldElement(1234567));
            auto poison = gatzk::algebra::Polynomial::from_evaluations("P_poison", std::move(full_values), domain);
            gatzk::algebra::PackedEvaluationBackend poison_backend(
                domain,
                {{"P_poison", &poison}});
            (void)poison_backend;
        }

        auto polynomial = gatzk::algebra::Polynomial::from_evaluations(
            "P_pad",
            {FieldElement(3), FieldElement(5), FieldElement(7), FieldElement(11), FieldElement(13)},
            domain);
        gatzk::algebra::PackedEvaluationBackend backend(
            domain,
            {{"P_pad", &polynomial}});

        const auto direct_padding = backend.values_at_direct_index({"P_pad"}, 128);
        require(direct_padding.size() == 1 && direct_padding[0] == FieldElement::zero(), "packed backend direct padding must be zero");

        const auto point = choose_non_domain_point(*domain);
        const auto weights = gatzk::algebra::build_domain_evaluation_weights(domain, point);
        require(!weights.direct_index.has_value(), "packed backend regression point must not be in domain");
        const auto packed_values = backend.evaluate_with_native_weights({"P_pad"}, weights.native_weights);
        require(packed_values.size() == 1, "packed backend must return one value");
        require(packed_values[0] == polynomial.evaluate(point), "packed backend weighted eval must ignore padded tail");
    }

    void require_msm_matches_single_thread(
        const std::string &label,
        const std::vector<gatzk::crypto::G1Point> &points,
        std::size_t count,
        bool sparse)
    {
        require(points.size() >= count, label + " SRS input is too small");
        std::vector<FieldElement> scalars(count, FieldElement::zero());
        for (std::size_t i = 0; i < count; ++i)
        {
            if (sparse && i % 5 != 0)
            {
                continue;
            }
            scalars[i] = FieldElement(17 + (i * 131) % 65521);
        }

        const auto actual = gatzk::crypto::g1_multi_scalar_mul(points.data(), scalars.data(), count);

        std::vector<mcl::Fr> native_scalars;
        native_scalars.reserve(scalars.size());
        for (const auto &scalar : scalars)
        {
            native_scalars.push_back(scalar.native());
        }
        auto *native_points = const_cast<mcl::G1 *>(
            reinterpret_cast<const mcl::G1 *>(points.data()));
        gatzk::crypto::G1Point expected;
        expected.value.clear();
        mcl::G1::mulVec(expected.value, native_points, native_scalars.data(), native_scalars.size());
        require(actual == expected, label + " MSM must match single-thread mulVec");
    }

    void test_msm_chunked_equivalence_regression()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(),
            "trusted_srs",
            test_srs_id());
        require(keys.pk.powers_g1 != nullptr, "MSM regression requires coefficient SRS");
        require_msm_matches_single_thread("coef_21664_dense", *keys.pk.powers_g1, 21664, false);

        const auto edge_it = keys.pk.lagrange_srs_by_domain.find("edge");
        require(edge_it != keys.pk.lagrange_srs_by_domain.end() && edge_it->second != nullptr, "MSM regression requires edge SRS");
        require_msm_matches_single_thread("edge_13264_sparse", *edge_it->second, 13264, true);
        require_msm_matches_single_thread("edge_131072_dense", *edge_it->second, 131072, false);
    }

    void test_msm_non_divisible_chunk_regression()
    {
        const auto keys = gatzk::crypto::KZG::load_keys(
            test_srs_path().string(),
            "trusted_srs",
            test_srs_id());
        require(keys.pk.powers_g1 != nullptr, "MSM chunk regression requires coefficient SRS");
        require_msm_matches_single_thread("coef_257_dense", *keys.pk.powers_g1, 257, false);
        require_msm_matches_single_thread("coef_1009_dense", *keys.pk.powers_g1, 1009, false);
    }

    void test_srs_mismatch_gate()
    {
        const auto srs_path = test_srs_path().string();
        const auto expected_srs_id = test_srs_id();
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(srs_path, "mpc_srs", expected_srs_id); },
            "SRS type mismatch must hard fail");
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(srs_path, "trusted_srs", "wrong_srs_id"); },
            "SRS id mismatch must hard fail");
        const auto manifest_keys = gatzk::crypto::KZG::load_keys(
            srs_path, "trusted_srs", expected_srs_id);
        const auto canonical_d_size = manifest_keys.manifest.domain_sizes.at("d");
        (void)gatzk::crypto::KZG::load_keys_for_domains(
            srs_path,
            "trusted_srs",
            expected_srs_id,
            {{"d", canonical_d_size}});
        const auto sibling_size = manifest_keys.manifest.domain_sizes.at("N");
        require(
            sibling_size != manifest_keys.manifest.domain_sizes.at("C"),
            "unit SRS must expose a same-size sibling mismatch regression");
        require_throws(
            [&]
            {
                (void)gatzk::crypto::KZG::load_keys_for_domains(
                    srs_path,
                    "trusted_srs",
                    expected_srs_id,
                    {{"C", sibling_size}});
            },
            "SRS loader must not borrow a same-size sibling domain");

        const auto manifest_tamper = copy_test_srs("manifest");
        auto manifest = read_file(manifest_tamper / "srs_manifest.json");
        const auto format = manifest.find("zkgat_srs_manifest_v8");
        require(format != std::string::npos, "unit SRS must use the audited v8 manifest");
        manifest[format] = 'x';
        write_file(manifest_tamper / "srs_manifest.json", manifest);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  manifest_tamper.string(), "trusted_srs", expected_srs_id); },
            "manifest byte tamper must hard fail");

        const auto audit_tamper = copy_test_srs("audit_report");
        auto audit_report = read_file(
            audit_tamper / "srs_audit_report.txt");
        const auto audit_status = audit_report.find("status=passed");
        require(
            audit_status != std::string::npos,
            "unit SRS must carry a passed canonical audit report");
        audit_report[audit_status] = 'x';
        write_file(
            audit_tamper / "srs_audit_report.txt", audit_report);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  audit_tamper.string(), "trusted_srs", expected_srs_id); },
            "SRS consistency audit report tamper must hard fail");

        const auto domain_tamper = copy_test_srs("domain");
        manifest = read_file(domain_tamper / "srs_manifest.json");
        const auto generator_field = manifest.find("\"domain_generators\"");
        const auto generator_key = manifest.find('"', manifest.find(':', generator_field) + 1);
        const auto generator_key_end = manifest.find('"', generator_key + 1);
        const auto generator_value = manifest.find('"', generator_key_end + 1);
        const auto generator_hex = generator_value + 1;
        require(generator_field != std::string::npos && generator_hex < manifest.size(),
                "unit SRS must carry the domain generator catalogue");
        manifest[generator_hex] = manifest[generator_hex] == '0' ? '1' : '0';
        rewrite_manifest_hash(manifest);
        write_file(domain_tamper / "srs_manifest.json", manifest);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  domain_tamper.string(), "trusted_srs", expected_srs_id); },
            "domain generator tamper with a valid outer manifest hash must hard fail");

        const auto compiler_tamper = copy_test_srs("compiler_artifact");
        auto compiler_artifact = read_file(
            compiler_tamper / "protocol_compiler_artifact.txt");
        const auto template_field = compiler_artifact.find("protocol_template_hex=");
        require(template_field != std::string::npos,
                "unit SRS must carry the full protocol compiler artifact");
        const auto template_byte = template_field + std::string("protocol_template_hex=").size();
        compiler_artifact[template_byte] = compiler_artifact[template_byte] == '0' ? '1' : '0';
        write_file(
            compiler_tamper / "protocol_compiler_artifact.txt", compiler_artifact);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  compiler_tamper.string(), "trusted_srs", expected_srs_id); },
            "protocol compiler artifact tamper must hard fail");

        const auto compiler_rehashed_tamper = copy_test_srs("compiler_rehashed");
        compiler_artifact = read_file(
            compiler_rehashed_tamper / "protocol_compiler_artifact.txt");
        const auto rehashed_template_field = compiler_artifact.find(
            "protocol_template_hex=");
        require(rehashed_template_field != std::string::npos,
                "unit compiler artifact must carry protocol template bytes");
        const auto rehashed_template_byte = rehashed_template_field + std::string("protocol_template_hex=").size();
        compiler_artifact[rehashed_template_byte] =
            compiler_artifact[rehashed_template_byte] == '0' ? '1' : '0';
        const auto artifact_hash_field = compiler_artifact.rfind("artifact_hash=");
        require(artifact_hash_field != std::string::npos,
                "unit compiler artifact must carry its canonical hash");
        const auto artifact_hash_begin = artifact_hash_field + std::string("artifact_hash=").size();
        const auto artifact_hash_end = compiler_artifact.find('\n', artifact_hash_begin);
        compiler_artifact.replace(
            artifact_hash_begin, artifact_hash_end - artifact_hash_begin,
            sha256_hex(std::string_view(compiler_artifact).substr(0, artifact_hash_field)));
        write_file(
            compiler_rehashed_tamper / "protocol_compiler_artifact.txt",
            compiler_artifact);
        manifest = read_file(compiler_rehashed_tamper / "srs_manifest.json");
        replace_json_string_field(
            manifest,
            "protocol_compiler_artifact_hash",
            sha256_hex(compiler_artifact));
        rewrite_manifest_hash(manifest);
        write_file(compiler_rehashed_tamper / "srs_manifest.json", manifest);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  compiler_rehashed_tamper.string(), "trusted_srs", expected_srs_id); },
            "rehashed compiler template tamper must fail protocol_schema_id validation");

        const auto compiler_registry_rehashed_tamper =
            copy_test_srs("compiler_registry_rehashed");
        compiler_artifact = read_file(
            compiler_registry_rehashed_tamper / "protocol_compiler_artifact.txt");
        const auto registry_prefix = std::string("observation_ledger_template_hex=");
        const auto registry_field = compiler_artifact.find(registry_prefix);
        require(registry_field != std::string::npos,
                "unit compiler artifact must embed ObservationLedger template bytes");
        const auto registry_begin = registry_field + registry_prefix.size();
        const auto registry_end = compiler_artifact.find('\n', registry_begin);
        require(registry_end != std::string::npos && registry_begin < registry_end,
                "unit compiler artifact must embed ObservationLedger template bytes");
        compiler_artifact[registry_begin] =
            compiler_artifact[registry_begin] == '0' ? '1' : '0';
        const auto tampered_registry_hex = compiler_artifact.substr(
            registry_begin, registry_end - registry_begin);
        const auto tampered_registry_hash = sha256_hex(
            decode_canonical_hex(tampered_registry_hex));
        replace_artifact_scalar(
            compiler_artifact, "observation_ledger_hash", tampered_registry_hash);
        const auto registry_artifact_hash_field = compiler_artifact.rfind("artifact_hash=");
        require(registry_artifact_hash_field != std::string::npos,
                "unit compiler artifact must carry its canonical hash");
        replace_artifact_scalar(
            compiler_artifact,
            "artifact_hash",
            sha256_hex(std::string_view(compiler_artifact).substr(0, registry_artifact_hash_field)));
        write_file(
            compiler_registry_rehashed_tamper / "protocol_compiler_artifact.txt",
            compiler_artifact);
        manifest = read_file(
            compiler_registry_rehashed_tamper / "srs_manifest.json");
        replace_json_string_field(
            manifest, "observation_ledger_hash", tampered_registry_hash);
        replace_json_string_field(
            manifest,
            "protocol_compiler_artifact_hash",
            sha256_hex(compiler_artifact));
        rewrite_manifest_hash(manifest);
        write_file(
            compiler_registry_rehashed_tamper / "srs_manifest.json", manifest);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  compiler_registry_rehashed_tamper.string(),
                  "trusted_srs", expected_srs_id); },
            "rehashed embedded ObservationLedger template tamper must fail local registry validation");

        const auto envelope_rehashed_tamper =
            copy_test_srs("compiler_envelope_rehashed");
        compiler_artifact = read_file(
            envelope_rehashed_tamper / "protocol_compiler_artifact.txt");
        const auto envelope_prefix = std::string("max_instance_envelope_hex=");
        const auto envelope_field = compiler_artifact.find(envelope_prefix);
        require(envelope_field != std::string::npos,
                "unit compiler artifact must embed max instance envelope bytes");
        const auto envelope_begin = envelope_field + envelope_prefix.size();
        const auto envelope_end = compiler_artifact.find('\n', envelope_begin);
        auto envelope = decode_canonical_hex(compiler_artifact.substr(
            envelope_begin, envelope_end - envelope_begin));
        const auto class_bound = envelope.find("max_classes=2\n");
        require(class_bound != std::string::npos,
                "unit max envelope must carry its class bound");
        envelope[class_bound + std::string("max_classes=").size()] = '3';
        const auto envelope_hash = sha256_hex(envelope);
        replace_artifact_scalar(
            compiler_artifact, "max_instance_envelope_hex", canonical_hex(envelope));
        replace_artifact_scalar(
            compiler_artifact, "max_instance_envelope_hash", envelope_hash);
        replace_artifact_scalar(
            compiler_artifact, "max_instance_envelope_id", envelope_hash);
        const auto envelope_artifact_hash_field =
            compiler_artifact.rfind("artifact_hash=");
        require(envelope_artifact_hash_field != std::string::npos,
                "unit compiler artifact must carry its canonical hash");
        replace_artifact_scalar(
            compiler_artifact, "artifact_hash",
            sha256_hex(std::string_view(compiler_artifact).substr(0, envelope_artifact_hash_field)));
        write_file(
            envelope_rehashed_tamper / "protocol_compiler_artifact.txt",
            compiler_artifact);
        manifest = read_file(envelope_rehashed_tamper / "srs_manifest.json");
        replace_json_string_field(
            manifest, "max_instance_envelope_id", envelope_hash);
        replace_json_string_field(
            manifest, "max_instance_envelope_hash", envelope_hash);
        replace_json_string_field(
            manifest, "protocol_compiler_artifact_hash",
            sha256_hex(compiler_artifact));
        rewrite_manifest_hash(manifest);
        write_file(envelope_rehashed_tamper / "srs_manifest.json", manifest);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  envelope_rehashed_tamper.string(),
                  "trusted_srs", expected_srs_id); },
            "rehashed max-envelope substitution must fail ceremony transcript binding");

        const auto vk_tamper = copy_test_srs("vk");
        auto vk = read_file(vk_tamper / "vk.bin");
        require(vk.size() > 32, "unit verifier key must contain material");
        vk[vk.size() / 2] ^= 1;
        write_file(vk_tamper / "vk.bin", vk);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  vk_tamper.string(), "trusted_srs", expected_srs_id); },
            "verifier-key material tamper must hard fail");

        const auto pk_tamper = copy_test_srs("pk");
        auto pk = read_file(pk_tamper / "pk_coef.bin");
        require(pk.size() > 32, "unit coefficient SRS must contain material");
        pk[pk.size() / 2] ^= 1;
        write_file(pk_tamper / "pk_coef.bin", pk);
        require_throws(
            [&]
            { (void)gatzk::crypto::KZG::load_keys(
                  pk_tamper.string(), "trusted_srs", expected_srs_id); },
            "coefficient SRS material tamper must hard fail");
        for (const auto &path : {
                 manifest_tamper, domain_tamper, compiler_tamper,
                 audit_tamper,
                 compiler_rehashed_tamper, compiler_registry_rehashed_tamper,
                 envelope_rehashed_tamper,
                 vk_tamper, pk_tamper})
        {
            std::filesystem::remove_all(path);
        }
    }

    void test_metadata_and_schema_gates()
    {
        const auto config = gatzk::util::load_config((repo_root() / "configs/cora_secure.cfg").string());
        require(config.proof_task == "node_classification_full_gat", "config proof_task mismatch");
        require(config.privacy_policy == "model_and_trace_hidden", "config privacy policy mismatch");
        require(config.input_privacy == "public_graph_public_features", "config input privacy mismatch");
        require(config.output_mode == "public_logits", "config output mode mismatch");
        require(config.opening_mode == "point_grouped_batch_opening", "config opening mode mismatch");
        require(config.binding_mode == "zk_subproof_for_crpc", "config binding mode mismatch");
        require(config.srs_type == "trusted_srs", "config SRS type mismatch");
        require(config.srs_id == "cora_formal_srs", "config SRS id mismatch");
        require(
            gatzk::protocol::schema_accepts_evaluation(
                gatzk::protocol::PolynomialVisibility::PrivateWitness,
                gatzk::protocol::EvaluationKind::MaskedPrivateValue),
            "private witness must accept masked values");
        require(
            !gatzk::protocol::schema_accepts_evaluation(
                gatzk::protocol::PolynomialVisibility::PrivateWitness,
                gatzk::protocol::EvaluationKind::PublicValue),
            "private witness must reject public values");

        gatzk::protocol::ProtocolContext empty_context;
        require(
            gatzk::protocol::polynomial_visibility(empty_context, "P_Y") ==
                gatzk::protocol::PolynomialVisibility::PublicOutput,
            "only the final logits polynomial is public output");
        require(
            gatzk::protocol::polynomial_visibility(empty_context, "P_Y_lin") ==
                gatzk::protocol::PolynomialVisibility::PrivateWitness,
            "pre-bias output must remain private");
        require(
            gatzk::protocol::polynomial_visibility(empty_context, "P_Y_o0") ==
                gatzk::protocol::PolynomialVisibility::PrivateWitness,
            "per-head output must remain private");
    }

    void test_removed_compatibility_protocol_gate()
    {
        gatzk::protocol::ProtocolContext legacy_context;
        legacy_context.model.has_real_multihead = false;
        gatzk::protocol::TraceArtifacts trace;
        gatzk::protocol::Proof proof;
        require_throws(
            [&]
            { (void)gatzk::protocol::build_trace(legacy_context, nullptr); },
            "formal trace builder must reject the removed single-output protocol");
        require_throws(
            [&]
            { (void)gatzk::protocol::prove(legacy_context, trace, nullptr); },
            "formal prover must reject the removed single-output protocol");
        require(
            !gatzk::protocol::verify(legacy_context, proof, nullptr),
            "formal verifier must reject the removed single-output protocol");
    }

    void test_canonical_output_digest_contract()
    {
        const std::vector<FieldElement> logits = {
            FieldElement::from_signed(-1),
            FieldElement(2),
            FieldElement(3),
            FieldElement::from_signed(-4),
        };
        const auto digest = gatzk::protocol::canonical_output_digest(logits, 2, 2);
        require(digest.size() == 64, "output digest must be a lowercase SHA-256 hex value");
        auto tampered = logits;
        tampered[3] += FieldElement::one();
        require(!(digest == gatzk::protocol::canonical_output_digest(tampered, 2, 2)),
                "output digest must bind every canonical field element");
        require(!(digest == gatzk::protocol::canonical_output_digest(logits, 1, 4)),
                "output digest must bind the declared N/C shape");
        require_throws(
            [&]
            {
                (void)gatzk::protocol::canonical_output_digest(
                    {}, std::numeric_limits<std::size_t>::max(), 2);
            },
            "output digest must reject overflowing NC before encoding");
    }

    gatzk::data::LocalGraph valid_two_graph_fixture()
    {
        gatzk::data::LocalGraph local;
        local.num_nodes = 4;
        local.num_features = 1;
        local.num_classes = 2;
        local.graph_count = 2;
        local.absolute_ids = {10, 11, 12, 13};
        local.node_ptr = {0, 2, 4};
        local.edge_ptr = {0, 3, 6};
        local.edges = {
            {.src = 0, .dst = 0, .graph_id = 0, .stable_index = 0},
            {.src = 1, .dst = 0, .graph_id = 0, .stable_index = 1},
            {.src = 1, .dst = 1, .graph_id = 0, .stable_index = 2},
            {.src = 2, .dst = 2, .graph_id = 1, .stable_index = 0},
            {.src = 3, .dst = 2, .graph_id = 1, .stable_index = 1},
            {.src = 3, .dst = 3, .graph_id = 1, .stable_index = 2},
        };
        local.public_input.N_total = 20;
        local.public_input.N = 4;
        local.public_input.E = local.edges.size();
        local.public_input.G_batch = 2;
        local.public_input.I = local.absolute_ids;
        local.public_input.node_ptr = local.node_ptr;
        local.public_input.edge_ptr = local.edge_ptr;
        for (const auto &edge : local.edges)
        {
            local.public_input.src.push_back(edge.src);
            local.public_input.dst.push_back(edge.dst);
        }
        return local;
    }

    void test_multi_graph_public_boundary_validation()
    {
        const auto valid = valid_two_graph_fixture();
        gatzk::data::validate_normalized_graph_input(valid);

        auto cross_graph = valid;
        cross_graph.edges[3].src = 0;
        cross_graph.public_input.src[3] = 0;
        require_throws(
            [&]
            { gatzk::data::validate_normalized_graph_input(cross_graph); },
            "cross-graph edge must reject");

        auto bad_pointer = valid;
        bad_pointer.edge_ptr = {0, 4, 6};
        bad_pointer.public_input.edge_ptr = bad_pointer.edge_ptr;
        require_throws(
            [&]
            { gatzk::data::validate_normalized_graph_input(bad_pointer); },
            "edge_ptr/graph_id mismatch must reject");

        auto unsorted = valid;
        std::swap(unsorted.edges[1], unsorted.edges[2]);
        std::swap(unsorted.public_input.src[1], unsorted.public_input.src[2]);
        std::swap(unsorted.public_input.dst[1], unsorted.public_input.dst[2]);
        require_throws(
            [&]
            { gatzk::data::validate_normalized_graph_input(unsorted); },
            "non-monotone destination order must reject");

        auto duplicate_node = valid;
        duplicate_node.absolute_ids[3] = duplicate_node.absolute_ids[2];
        duplicate_node.public_input.I = duplicate_node.absolute_ids;
        require_throws(
            [&]
            { gatzk::data::validate_normalized_graph_input(duplicate_node); },
            "duplicate absolute node id must reject");
    }

    void test_compiler_observation_certificate_gate()
    {
        const auto exact_chunks = gatzk::protocol::coefficient_chunk_layout(31, 16);
        require(exact_chunks == std::vector<gatzk::protocol::CoefficientChunkLayoutEntry>{
                                    {.chunk_index = 0, .offset = 0, .length = 15},
                                    {.chunk_index = 1, .offset = 15, .length = 15},
                                    {.chunk_index = 2, .offset = 30, .length = 1}},
                "coefficient chunks must use the unique n_edge-1 layout");
        require_throws(
            []
            { (void)gatzk::protocol::coefficient_chunk_layout(0, 16); },
            "empty coefficient object must reject");
        require_throws(
            []
            { (void)gatzk::protocol::coefficient_chunk_layout(1, 1); },
            "coefficient chunk domain without a terminal row must reject");
        const auto region = gatzk::protocol::coefficient_region_layout(
            5, {0, 2, 4}, 3, 6);
        require(region.total_length == 8 && region.tail_length == 3 && region.q_sem == std::vector<std::uint8_t>({1, 0, 1, 0, 1, 0, 0, 0}) && region.q_gap == std::vector<std::uint8_t>({0, 1, 0, 1, 0, 0, 0, 0}) && region.q_tail == std::vector<std::uint8_t>({0, 0, 0, 0, 0, 1, 1, 1}) && region.chunks.size() == 2 && region.chunks[0].offset == 0 && region.chunks[0].length == 5 && region.chunks[1].offset == 5 && region.chunks[1].length == 3,
                "coefficient region must uniquely partition semantic/gap/tail positions and chunks");
        require_throws(
            []
            { (void)gatzk::protocol::coefficient_region_layout(5, {0, 0}, 1, 8); },
            "duplicate coefficient semantic positions must reject");
        require_throws(
            []
            { (void)gatzk::protocol::coefficient_region_layout(5, {0, 5}, 1, 8); },
            "out-of-range coefficient semantic position must reject");
        require_throws(
            []
            { (void)gatzk::protocol::coefficient_region_layout(5, {0}, 0, 8); },
            "zero coefficient tail must reject");
        const auto materialized_tail =
            gatzk::protocol::materialize_coefficient_tail(
                region,
                {FieldElement(2), FieldElement::zero(), FieldElement(3),
                 FieldElement::zero(), FieldElement(5)},
                {FieldElement(7), FieldElement(11), FieldElement(13)});
        require(materialized_tail == std::vector<FieldElement>({FieldElement(2), FieldElement::zero(), FieldElement(3),
                                                                FieldElement::zero(), FieldElement(5), FieldElement(7),
                                                                FieldElement(11), FieldElement(13)}),
                "coefficient-tail materializer must preserve semantic coefficients and append the exact tail");
        const auto gamma = FieldElement(17);
        auto reconstructed = FieldElement::zero();
        for (const auto &chunk : region.chunks)
        {
            auto local_terminal = FieldElement::zero();
            auto local_power = FieldElement::one();
            for (std::size_t row = 0; row < chunk.length; ++row)
            {
                local_terminal += materialized_tail[chunk.offset + row] * local_power;
                local_power *= gamma;
            }
            reconstructed += gamma.pow(chunk.offset) * local_terminal;
        }
        require(reconstructed == gatzk::algebra::horner(materialized_tail, gamma),
                "multi-chunk local accumulators must reconstruct the complete coefficient-tail evaluation");
        require_throws(
            [&]
            {
                (void)gatzk::protocol::materialize_coefficient_tail(
                    region,
                    {FieldElement(2), FieldElement(9), FieldElement(3),
                     FieldElement::zero(), FieldElement(5)},
                    {FieldElement(7), FieldElement(11), FieldElement(13)});
            },
            "nonzero low-degree gap must reject before commitment");
        require_throws(
            [&]
            {
                (void)gatzk::protocol::materialize_coefficient_tail(
                    region,
                    {FieldElement(2), FieldElement::zero(), FieldElement(3),
                     FieldElement::zero(), FieldElement(5)},
                    {FieldElement::zero(), FieldElement::zero(), FieldElement::zero()});
            },
            "all-zero coefficient tail must reject before commitment");
        gatzk::protocol::ProtocolContext context;
        context.config.opening_mode = "point_grouped_batch_opening";
        context.config.binding_mode = "zk_subproof_for_crpc";
        context.local.num_nodes = 3;
        context.local.num_features = 2;
        context.local.num_classes = 2;
        context.local.edges = {
            {.src = 0, .dst = 0, .graph_id = 0, .stable_index = 0},
            {.src = 1, .dst = 1, .graph_id = 0, .stable_index = 1},
            {.src = 2, .dst = 2, .graph_id = 0, .stable_index = 2},
        };
        context.model.L = 2;
        context.model.K_out = 1;
        context.model.C = 2;
        context.model.has_real_multihead = true;
        context.model.d_in_profile = {2};
        context.model.hidden_layers = {{
            .layer_index = 0,
            .input_dim = 2,
            .shape = {.head_count = 1, .head_dim = 2},
            .heads = {gatzk::model::AttentionHeadParameters{}},
        }};
        context.model.hidden_heads.resize(1);
        context.model.output_layer.input_dim = 2;
        context.model.output_layer.output_dim = 2;
        context.model.output_layer.head_count = 1;
        context.model.output_layer.heads.resize(1);
        context.kzg.manifest.degree_bound = 255;
        context.domains.fh = gatzk::algebra::RootOfUnityDomain::create("FH", 16);
        context.domains.edge = gatzk::algebra::RootOfUnityDomain::create("edge", 16);
        context.domains.in = gatzk::algebra::RootOfUnityDomain::create("in", 16);
        context.domains.d = gatzk::algebra::RootOfUnityDomain::create("d", 16);
        context.domains.cat = gatzk::algebra::RootOfUnityDomain::create("cat", 16);
        context.domains.c = gatzk::algebra::RootOfUnityDomain::create("C", 8);
        context.domains.n = gatzk::algebra::RootOfUnityDomain::create("N", 16);
        context.local.graph_count = 1;
        context.kzg.manifest.max_instance_envelope = {
            {"max_layers", 2},
            {"max_hidden_heads_per_layer", 1},
            {"max_hidden_head_dim", 2},
            {"max_input_dim", 2},
            {"max_output_heads", 1},
            {"max_classes", 2},
            {"max_graphs", 1},
            {"max_nodes", 3},
            {"max_edges", 3},
            {"max_total_nodes", 3},
            {"max_concat_dim", 2},
            {"max_range_bits", 10},
            {"max_coefficient_object_length", 6},
            {"max_object_family_count", 2048},
            {"max_coefficient_chunks", 2},
            {"security_target_bits", 120},
        };
        context.local.public_input.N_total = 3;
        std::string envelope_reason;
        require(gatzk::protocol::instance_within_authenticated_max_envelope(
                    context, &envelope_reason),
                "unit instance must fit its authenticated envelope: " + envelope_reason);
        auto excessive_layers = context;
        excessive_layers.model.L = 3;
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    excessive_layers, &envelope_reason),
                "max_layers must constrain the total GAT layer count");
        auto excessive_output_input = context;
        excessive_output_input.model.output_layer.input_dim = 3;
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    excessive_output_input, &envelope_reason),
                "max_input_dim must constrain the output-layer input dimension");
        auto excessive_total_nodes = context;
        excessive_total_nodes.local.public_input.N_total = 4;
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    excessive_total_nodes, &envelope_reason),
                "max_total_nodes must constrain the public dataset universe");
        auto excessive_concat = context;
        excessive_concat.kzg.manifest.max_instance_envelope["max_concat_dim"] = 1;
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    excessive_concat, &envelope_reason),
                "max_concat_dim must constrain the hidden concatenation width");
        auto excessive_range_bits = context;
        excessive_range_bits.config.range_bits = 11;
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    excessive_range_bits, &envelope_reason),
                "max_range_bits must constrain the registered range semantics");
        auto excessive_coefficient_object = context;
        excessive_coefficient_object.kzg.manifest.max_instance_envelope["max_coefficient_object_length"] = 5;
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    excessive_coefficient_object, &envelope_reason),
                "maximum coefficient object length must be enforced");
        auto excessive_object_family = context;
        excessive_object_family.kzg.manifest.max_instance_envelope["max_object_family_count"] = 1;
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    excessive_object_family, &envelope_reason),
                "maximum object family count must be enforced");
        auto missing_envelope_bound = context;
        missing_envelope_bound.kzg.manifest.max_instance_envelope.erase(
            "max_total_nodes");
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    missing_envelope_bound, &envelope_reason),
                "missing maximum-envelope field must reject before proving");
        auto inconsistent_total_nodes = context;
        inconsistent_total_nodes.kzg.manifest.max_instance_envelope["max_total_nodes"] = 2;
        require(!gatzk::protocol::instance_within_authenticated_max_envelope(
                    inconsistent_total_nodes, &envelope_reason),
                "an instance larger than the authenticated N_total bound must reject");

        auto artifact = gatzk::protocol::build_protocol_compiler_artifact(context);
        context.compiler_artifact =
            std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>(artifact);
        require(!artifact.observation_ledger.empty(),
                "compiler unit context must produce ObservationLedger entries");
        require(artifact.binding_relation_registry.size() == gatzk::protocol::build_binding_relation_metadata(context).size() && std::none_of(
                                                                                                                                     artifact.binding_relation_registry.begin(),
                                                                                                                                     artifact.binding_relation_registry.end(),
                                                                                                                                     [](const auto &record)
                                                                                                                                     { return record.empty(); }),
                "compiler artifact must authenticate every instance RelationMeta record");
        const auto derived_model_binding =
            gatzk::protocol::output_model_binding_blinder_label(0, "proj");
        const auto require_schema_basis = [&](const std::string &tag,
                                              const std::string &expected)
        {
            const auto it = std::find_if(
                artifact.proof_schema.entries.begin(),
                artifact.proof_schema.entries.end(),
                [&](const auto &entry)
                { return entry.tag == tag; });
            require(it != artifact.proof_schema.entries.end() && it->blinding_basis == expected,
                    "ProofSchema blinding basis mismatch for " + tag);
        };
        const auto hidden_star =
            gatzk::protocol::hidden_head_prefix(0) + "_H_star";
        const auto hidden_star_mask =
            gatzk::protocol::external_mask_label(hidden_star);
        require_schema_basis(hidden_star_mask, "DerivedExpression");
        require(std::none_of(
                    artifact.observation_ledger.begin(),
                    artifact.observation_ledger.end(),
                    [&](const auto &entry)
                    {
                        return entry.object_or_challenge == hidden_star_mask;
                    }),
                "derived external mask must not own independent randomness");
        const auto hidden_star_dependency = std::find_if(
            artifact.observation_dependency_dag.begin(),
            artifact.observation_dependency_dag.end(),
            [&](const auto &edge)
            {
                return edge.derived_object == hidden_star_mask;
            });
        require(hidden_star_dependency != artifact.observation_dependency_dag.end() && hidden_star_dependency->source_objects == std::vector<std::string>{hidden_star} && hidden_star_dependency->propagation_rule.find("P_committed=P_sem+Z_N*Mask") != std::string::npos,
                "external semantic opening mask must be registered as a source-derived DAG object");
        require_schema_basis("P_H", "None");
        require_schema_basis("t_FH", "DerivedExpression");
        require_schema_basis("R_t_FH", "CoefficientMask");
        require_schema_basis(derived_model_binding, "DerivedExpression");
        require_schema_basis(
            gatzk::protocol::model_reblind_witness_label(
                gatzk::protocol::output_weight_column_label(0, 0)),
            "DerivedExpression");
        require_schema_basis(
            gatzk::protocol::model_proof_local_label(
                gatzk::protocol::output_weight_column_label(0, 0)),
            "DomainVanishing");
        bool saw_formal_rank_certificate = false;
        for (const auto &entry : artifact.observation_ledger)
        {
            if (entry.expression_or_certificate.find("formal-full-row-rank-v2") != std::string::npos)
            {
                saw_formal_rank_certificate = true;
                require(entry.query_bound > 0,
                        "formal observation certificate must fix q_O_max");
            }
        }
        require(saw_formal_rank_certificate,
                "domain-blinded objects must carry a nonzero formal minor certificate");
        const auto quotient_semantic_obs = std::find_if(
            artifact.observation_ledger.begin(), artifact.observation_ledger.end(),
            [](const auto &entry)
            { return entry.object_or_challenge == "t_FH"; });
        require(quotient_semantic_obs == artifact.observation_ledger.end(),
                "derived masked quotient must not invent an independent blinder owner");
        const auto quotient_mask_obs = std::find_if(
            artifact.observation_ledger.begin(), artifact.observation_ledger.end(),
            [](const auto &entry)
            { return entry.object_or_challenge == "R_t_FH"; });
        require(quotient_mask_obs != artifact.observation_ledger.end() && quotient_mask_obs->query_bound == 8 && quotient_mask_obs->expression_or_certificate.find("identity=t_zk=t+Z_D*R_t") != std::string::npos && quotient_mask_obs->expression_or_certificate.find("source-objects=") != std::string::npos,
                "quotient mask must own the compiler resource and bind the derived quotient DAG");
        require(gatzk::protocol::registered_blinder_coefficient_count(
                    context, "R_t_FH") == quotient_mask_obs->query_bound,
                "quotient-mask sampler resource must come from ObservationLedger");
        require(std::none_of(
                    artifact.observation_ledger.begin(),
                    artifact.observation_ledger.end(),
                    [&](const auto &entry)
                    {
                        return entry.object_or_challenge == derived_model_binding;
                    }),
                "derived ModelBind_R must not invent an independent blinder owner");
        const auto output_projection_source =
            gatzk::protocol::output_head_prefix(0) + "_b_proj";
        const auto output_projection_observation = std::find_if(
            artifact.observation_ledger.begin(),
            artifact.observation_ledger.end(),
            [&](const auto &entry)
            {
                return entry.object_or_challenge == output_projection_source;
            });
        require(output_projection_observation != artifact.observation_ledger.end() && output_projection_observation->expression_or_certificate.find(
                                                                                          "derived-model-binding-observations(") != std::string::npos,
                "ModelBind_R serialized observations must propagate to their source certificate");
        const auto model_binding_dependency = std::find_if(
            artifact.observation_dependency_dag.begin(),
            artifact.observation_dependency_dag.end(),
            [&](const auto &edge)
            {
                return edge.derived_object == derived_model_binding;
            });
        require(model_binding_dependency != artifact.observation_dependency_dag.end() && model_binding_dependency->source_objects.size() == 3 && !model_binding_dependency->introduces_independent_randomness && std::find(model_binding_dependency->serialized_observations.begin(), model_binding_dependency->serialized_observations.end(), "OpeningWitness@canonical-point-group") != model_binding_dependency->serialized_observations.end(),
                "derived ModelBind_R must have a canonical cross-source observation DAG edge");
        const auto output_projection_reblind =
            gatzk::protocol::model_reblind_witness_label(
                gatzk::protocol::output_weight_column_label(0, 0));
        require(std::any_of(
                    artifact.observation_dependency_dag.begin(),
                    artifact.observation_dependency_dag.end(),
                    [&](const auto &edge)
                    {
                        return edge.derived_object == output_projection_reblind && edge.source_objects == std::vector<std::string>{
                                                                                                              gatzk::protocol::model_proof_local_label(
                                                                                                                  gatzk::protocol::output_weight_column_label(0, 0))};
                    }),
                "proof-local model blinder commitment must trace to its sampled owner");
        require(std::any_of(
                    artifact.observation_dependency_dag.begin(),
                    artifact.observation_dependency_dag.end(),
                    [](const auto &edge)
                    {
                        return edge.derived_object == "t_FH" && std::find(
                                                                    edge.source_objects.begin(), edge.source_objects.end(),
                                                                    "R_t_FH") != edge.source_objects.end();
                    }),
                "derived masked quotient must trace to its independent mask owner");
        require(std::none_of(
                    artifact.random_identity_ledger.begin(),
                    artifact.random_identity_ledger.end(),
                    [](const auto &entry)
                    {
                        return entry.object_or_challenge.find("families") != std::string::npos || entry.object_or_challenge.find(',') != std::string::npos;
                    }),
                "RandomIdentityLedger must not collapse concrete identities into family placeholders");
        const std::vector<std::string> required_formal_challenges = {
            "eta_feat",
            "beta_feat",
            "y_proj_h0",
            "xi_h0",
            "y_src_h0",
            "y_dst_h0",
            "y_star_h0",
            "eta_src_h0",
            "beta_src_h0",
            "eta_L_h0",
            "beta_L_h0",
            "beta_R_h0",
            "eta_exp_h0",
            "beta_exp_h0",
            "lambda_psq_h0",
            "eta_t_h0",
            "beta_t_h0",
            "y_agg_pre_h0",
            "eta_ELU_h0",
            "beta_ELU_h0",
            "y_agg_h0",
            "eta_dst_h0",
            "beta_dst_h0",
            "xi_out",
            "y_out",
            gatzk::protocol::output_challenge_name("y_proj_out", 0),
            gatzk::protocol::output_challenge_name("y_src_out", 0),
            gatzk::protocol::output_challenge_name("y_dst_out", 0),
            gatzk::protocol::output_challenge_name("eta_src_out", 0),
            gatzk::protocol::output_challenge_name("beta_src_out", 0),
            gatzk::protocol::output_challenge_name("eta_L_out", 0),
            gatzk::protocol::output_challenge_name("beta_L_out", 0),
            gatzk::protocol::output_challenge_name("beta_R_out", 0),
            gatzk::protocol::output_challenge_name("eta_exp_out", 0),
            gatzk::protocol::output_challenge_name("beta_exp_out", 0),
            gatzk::protocol::output_challenge_name("lambda_out", 0),
            gatzk::protocol::output_challenge_name("eta_t_out", 0),
            gatzk::protocol::output_challenge_name("beta_t_out", 0),
            gatzk::protocol::output_challenge_name("y_out_star", 0),
            gatzk::protocol::output_challenge_name("eta_dst_out", 0),
            gatzk::protocol::output_challenge_name("beta_dst_out", 0),
        };
        for (const auto &challenge : required_formal_challenges)
        {
            require(std::any_of(
                        artifact.random_identity_ledger.begin(),
                        artifact.random_identity_ledger.end(),
                        [&](const auto &entry)
                        {
                            return entry.object_or_challenge == challenge;
                        }),
                    "RandomIdentityLedger is missing formal challenge: " + challenge);
        }
        const auto signed_division_beta = gatzk::protocol::signed_division_beta_name(
            gatzk::protocol::hidden_head_prefix(0) + "_Div_", "A", 0);
        require(std::any_of(
                    artifact.random_identity_ledger.begin(),
                    artifact.random_identity_ledger.end(),
                    [&](const auto &entry)
                    {
                        return entry.object_or_challenge == signed_division_beta;
                    }),
                "RandomIdentityLedger must enumerate SignedRoundDiv limb LogUp challenges");
        require(artifact.security_budget_ledger.size() == artifact.random_identity_ledger.size() + 1,
                "each random identity must have one exact budget term plus the summary");
        std::vector<std::string> point_group_foldings;
        for (const auto *domain : {"FH", "edge", "in", "d_h", "cat", "C", "N"})
        {
            point_group_foldings.push_back(
                "point_group.rho@Open_dom:" + std::string(domain) + ":0");
            point_group_foldings.push_back(
                "point_group.rho@Open_dom:" + std::string(domain) + ":1");
        }
        point_group_foldings.push_back(
            "point_group.rho@W_ext:each-distinct-point");
        for (const auto &folding : point_group_foldings)
        {
            require(std::any_of(
                        artifact.random_identity_ledger.begin(),
                        artifact.random_identity_ledger.end(),
                        [&](const auto &entry)
                        {
                            return entry.object_or_challenge == folding;
                        }),
                    "RandomIdentityLedger is missing point-group folding: " + folding);
        }
        const auto z_fh_identity = std::find_if(
            artifact.random_identity_ledger.begin(),
            artifact.random_identity_ledger.end(),
            [](const auto &entry)
            { return entry.id == "random:quotient:t_FH"; });
        require(z_fh_identity != artifact.random_identity_ledger.end() && z_fh_identity->object_or_challenge == "z_FH" && z_fh_identity->exclusion_degree == context.domains.fh->size + 1 && z_fh_identity->expression_or_certificate.find("sub(mul(Z_D,t_D_zk),constraint_accumulator_D)") != std::string::npos,
                "FH quotient identity must bind its actual verifier DAG and opening exclusions");
        const auto shift_identity = std::find_if(
            artifact.random_identity_ledger.begin(),
            artifact.random_identity_ledger.end(),
            [](const auto &entry)
            {
                return entry.id.find("random:shift:output-bias-terminal") == 0;
            });
        require(shift_identity != artifact.random_identity_ledger.end() && shift_identity->degree_bound == context.kzg.manifest.degree_bound && shift_identity->exclusion_degree == context.domains.edge->size,
                "output-bias Taylor shift must have a concrete RandomIdentityLedger entry");
        const auto security_summary = std::find_if(
            artifact.security_budget_ledger.begin(),
            artifact.security_budget_ledger.end(),
            [](const auto &entry)
            { return entry.id == "budget:summary"; });
        require(security_summary != artifact.security_budget_ledger.end() && security_summary->expression_or_certificate.find("lambda_target=120") != std::string::npos && security_summary->expression_or_certificate.find("epsilon_total=") != std::string::npos && security_summary->expression_or_certificate.find("gate=epsilon_total<=2^-120") != std::string::npos,
                "SecurityBudgetLedger must carry an exact rational total and target gate");
        require(security_summary->expression_or_certificate.find(
                    "epsilon_crs=0/1") == std::string::npos,
                "SecurityBudgetLedger must include nonzero BadCRS_obs terms");
        require(security_summary->expression_or_certificate.find(
                    "lambda_target=128") == std::string::npos,
                "security metadata must not claim the impossible 2^-128 sum of three 128-bit base advantages");
        const auto output_bias_proof = gatzk::protocol::model_proof_local_label(
            gatzk::protocol::output_bias_label(0));
        const auto output_bias_obs = std::find_if(
            artifact.observation_ledger.begin(), artifact.observation_ledger.end(),
            [&](const auto &entry)
            {
                return entry.object_or_challenge == output_bias_proof;
            });
        require(output_bias_obs != artifact.observation_ledger.end() && output_bias_obs->query_bound == 2 && output_bias_obs->expression_or_certificate.find("S_model-row-eliminated-by-shifted-divisibility-DAG") != std::string::npos,
                "proof-local output bias must use the canonical two-coefficient shared blinder budget");
        require(!gatzk::protocol::observation_image_is_covered(
                    {{FieldElement::one()}, {FieldElement::one()}},
                    {{FieldElement::one()}, {FieldElement(2)}}),
                "o1=s+r,o2=2s+r must fail full image inclusion despite nonzero blinder rank");
        require(gatzk::protocol::observation_image_is_covered(
                    {{FieldElement::one()}, {FieldElement::one()}},
                    {{FieldElement::one()}, {FieldElement::one()}}),
                "matching secret and blinder images must pass the exact inclusion gate");
        std::string reason;
        require(gatzk::protocol::validate_protocol_compiler_artifact(
                    context, *context.compiler_artifact, &reason),
                "untampered compiler artifact must validate: " + reason);
        auto tampered = artifact;
        ++tampered.observation_ledger.front().query_bound;
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, tampered, &reason),
                "re-encoding gate must reject ledger content with stale canonical bytes");
        auto tampered_relation_registry = artifact;
        tampered_relation_registry.binding_relation_registry.front().push_back('\0');
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, tampered_relation_registry, &reason),
                "re-encoding gate must reject RelationMeta registry tamper");
        auto missing_relation_registry = artifact;
        missing_relation_registry.binding_relation_registry.pop_back();
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, missing_relation_registry, &reason),
                "missing authenticated RelationMeta record must reject");
        auto missing_dependency = artifact;
        missing_dependency.observation_dependency_dag.erase(
            missing_dependency.observation_dependency_dag.begin());
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, missing_dependency, &reason),
                "missing ObservationLedger dependency edge must reject");
        auto dangling_dependency = artifact;
        dangling_dependency.observation_dependency_dag.front().source_objects.front() += "_unknown";
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, dangling_dependency, &reason),
                "dangling ObservationLedger dependency source must reject");
        auto duplicate_randomness_owner = artifact;
        const auto derived_schema_it = std::find_if(
            duplicate_randomness_owner.proof_schema.entries.begin(),
            duplicate_randomness_owner.proof_schema.entries.end(),
            [&](const auto &entry)
            {
                return entry.tag == derived_model_binding;
            });
        require(derived_schema_it != duplicate_randomness_owner.proof_schema.entries.end(),
                "derived ModelBind_R must be present in ProofSchema");
        duplicate_randomness_owner.observation_ledger.push_back({
            .id = "obs:illegal-duplicate-owner",
            .object_or_challenge = derived_model_binding,
            .domain = derived_schema_it->domain,
            .transcript_stage = derived_schema_it->owner_block,
            .expression_or_certificate = "illegal-independent-owner-for-derived-object",
            .degree_bound = context.kzg.manifest.degree_bound,
            .query_bound = 1,
            .exclusion_degree = 1,
            .multiplicity = 1,
        });
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, duplicate_randomness_owner, &reason),
                "derived object with an invented independent blinder owner must reject");
        auto missing_qfs = artifact;
        const auto summary_index = static_cast<std::size_t>(std::distance(
            missing_qfs.security_budget_ledger.begin(),
            std::find_if(
                missing_qfs.security_budget_ledger.begin(),
                missing_qfs.security_budget_ledger.end(),
                [](const auto &entry)
                { return entry.id == "budget:summary"; })));
        require(summary_index < missing_qfs.security_budget_ledger.size(),
                "compiler artifact must contain the security summary entry");
        missing_qfs.security_budget_ledger[summary_index].query_bound = 0;
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, missing_qfs, &reason),
                "missing Q_FS_max must fail the security budget gate");
        auto floating_budget = artifact;
        floating_budget.security_budget_ledger[summary_index]
            .expression_or_certificate =
            "exact-rational:lambda_target=120|epsilon_total=1e-40";
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, floating_budget, &reason),
                "floating-point SecurityBudgetLedger encoding must reject");
        auto target_mismatch = artifact;
        auto &target_expression = target_mismatch.security_budget_ledger[summary_index]
                                      .expression_or_certificate;
        const auto target_position = target_expression.find("lambda_target=120");
        require(target_position != std::string::npos,
                "security summary must expose lambda_target");
        target_expression.replace(
            target_position, std::string("lambda_target=120").size(),
            "lambda_target=128");
        require(!gatzk::protocol::validate_protocol_compiler_artifact(
                    context, target_mismatch, &reason),
                "manifest/instance target-security mismatch must reject");

        const auto binding_meta = gatzk::protocol::build_binding_relation_metadata(context);
        const auto output_meta = std::find_if(
            binding_meta.begin(), binding_meta.end(), [](const auto &entry)
            { return entry.object_kind == "output" && !entry.shift_relations.empty(); });
        require(output_meta != binding_meta.end(),
                "compiler unit context must expand output-bias relation metadata");
        const auto *shift_ptr = gatzk::protocol::find_unique_shift_relation_by_kind(
            *output_meta, "output-bias-terminal");
        require(shift_ptr != nullptr,
                "output bias terminal relation must be uniquely addressable by kind");
        const auto &shift = *shift_ptr;
        require(shift.max_difference_degree == context.kzg.manifest.degree_bound,
                "unit shift soundness metadata must use global D_max");
        require(shift.configured_degrees.size() == 1,
                "unit output-bias relation must fix one Taylor-shift degree");

        auto colliding = context;
        colliding.local.num_nodes = 2;
        colliding.local.edges.resize(2);
        const auto colliding_meta =
            gatzk::protocol::build_binding_relation_metadata(colliding);
        const auto colliding_output = std::find_if(
            colliding_meta.begin(), colliding_meta.end(), [](const auto &entry)
            { return entry.object_kind == "output" && !entry.shift_relations.empty(); });
        require(colliding_output != colliding_meta.end(),
                "colliding terminals must retain a same-point RelationMeta");
        const auto *same_point_ptr =
            gatzk::protocol::find_unique_shift_relation_by_kind(
                *colliding_output, "output-bias-terminal");
        require(same_point_ptr != nullptr,
                "same-point output bias relation must retain its unique kind");
        const auto &same_point = *same_point_ptr;
        require(same_point.canonical_points.size() == 1 && same_point.configured_degrees.empty() && same_point.shift_domain_sizes.empty(),
                "colliding terminals must canonicalize to m=1 without shift fields");
        auto duplicate_relation_kind = *output_meta;
        duplicate_relation_kind.shift_relations.push_back(shift);
        require(gatzk::protocol::find_unique_shift_relation_by_kind(
                    duplicate_relation_kind, "output-bias-terminal") == nullptr,
                "duplicate relation kind must not resolve by vector position");

        const auto relation_challenge = [](const auto &meta)
        {
            gatzk::crypto::Transcript transcript("relation-meta-coverage-test");
            gatzk::protocol::absorb_binding_relation_metadata(transcript, meta);
            return transcript.challenge("after_relation_meta");
        };
        const auto base_relation_challenge = relation_challenge(*output_meta);
        auto require_relation_field_bound = [&](auto mutation,
                                                const std::string &field)
        {
            auto changed = *output_meta;
            mutation(changed);
            require(relation_challenge(changed) != base_relation_challenge,
                    "RelationMeta transcript omitted field: " + field);
        };
        require_relation_field_bound(
            [](auto &meta)
            { meta.source_objects.front() += "_tampered"; },
            "source_objects");
        require_relation_field_bound(
            [](auto &meta)
            { meta.target_objects.front() += "_tampered"; },
            "target_objects");
        require_relation_field_bound(
            [](auto &meta)
            { meta.work_domains.front() += "_tampered"; },
            "work_domains");
        require_relation_field_bound(
            [](auto &meta)
            { ++meta.real_lengths.front(); }, "real_lengths");
        require_relation_field_bound(
            [](auto &meta)
            { meta.quotient_relations.front() += "_tampered"; },
            "quotient_relations");
        require_relation_field_bound(
            [](auto &meta)
            { ++*meta.layer_id; }, "layer_id");
        require_relation_field_bound(
            [](auto &meta)
            { ++*meta.head_id; }, "head_id");
    }

    void test_domain_opening_challenge_exclusion_registry()
    {
        gatzk::protocol::ProtocolContext context;
        context.domains.fh = gatzk::algebra::RootOfUnityDomain::create("FH", 16);
        context.domains.edge = gatzk::algebra::RootOfUnityDomain::create("edge", 16);
        context.domains.in = gatzk::algebra::RootOfUnityDomain::create("in", 16);
        context.domains.d = gatzk::algebra::RootOfUnityDomain::create("d", 16);
        context.domains.cat = gatzk::algebra::RootOfUnityDomain::create("cat", 16);
        context.domains.c = gatzk::algebra::RootOfUnityDomain::create("C", 8);
        context.domains.n = gatzk::algebra::RootOfUnityDomain::create("N", 16);
        gatzk::crypto::Transcript transcript("domain-opening-registry-test");
        transcript.absorb_text("instance", "canonical-test-instance");
        const auto challenges =
            gatzk::protocol::derive_domain_opening_challenges(context, transcript);
        const std::vector<std::tuple<
            std::string, std::shared_ptr<gatzk::algebra::RootOfUnityDomain>>>
            specs = {
                {"z_FH", context.domains.fh},
                {"z_edge", context.domains.edge},
                {"z_in", context.domains.in},
                {"z_d_h", context.domains.d},
                {"z_cat", context.domains.cat},
                {"z_C", context.domains.c},
                {"z_N", context.domains.n},
            };
        std::vector<FieldElement> all_points;
        for (const auto &[label, domain] : specs)
        {
            const auto point = challenges.at(label);
            require(!domain->zero_polynomial_eval(point).is_zero(),
                    label + " must be outside its registered work domain");
            all_points.push_back(point);
            all_points.push_back(point * domain->omega);
        }
        std::vector<std::string> encoded;
        for (const auto &point : all_points)
            encoded.push_back(point.to_string());
        std::sort(encoded.begin(), encoded.end());
        require(std::adjacent_find(encoded.begin(), encoded.end()) == encoded.end(),
                "all fourteen work-domain opening points must be mutually distinct");
    }

    void test_real_ppi_representative_batch_boundary()
    {
        auto config = gatzk::util::load_config(
            (repo_root() / "configs/ppi_secure.cfg").string());
        config.batch_graphs = 2;
        // Development validation uses two complete source graphs. No node or edge
        // inside either graph is cropped, and this is not a formal full-PPI run.
        const auto dataset = gatzk::data::load_dataset(config);
        require(dataset.name == "ppi", "PPI boundary test must load the real PPI cache");
        require(dataset.graph_count >= config.batch_graphs, "PPI cache must contain the representative graphs");

        const auto local = gatzk::data::normalize_graph_input(dataset, config);
        gatzk::data::validate_normalized_graph_input(local);
        require(local.graph_count == config.batch_graphs, "PPI representative batch graph count mismatch");
        require(local.node_ptr.size() == config.batch_graphs + 1, "PPI node_ptr shape mismatch");
        require(local.edge_ptr.size() == config.batch_graphs + 1, "PPI edge_ptr shape mismatch");

        for (std::size_t graph = 0; graph < config.batch_graphs; ++graph)
        {
            const auto source_node_begin = dataset.node_ptr.at(graph);
            const auto source_node_end = dataset.node_ptr.at(graph + 1);
            require(
                local.node_ptr[graph + 1] - local.node_ptr[graph] == source_node_end - source_node_begin,
                "PPI normalization must retain every node of each selected source graph");
            for (std::size_t node = local.node_ptr[graph]; node < local.node_ptr[graph + 1]; ++node)
            {
                require(
                    local.absolute_ids[node] == source_node_begin + node - local.node_ptr[graph],
                    "PPI normalization must preserve the complete source-graph node range");
            }
            for (std::size_t edge = local.edge_ptr[graph]; edge < local.edge_ptr[graph + 1]; ++edge)
            {
                const auto &item = local.edges[edge];
                require(item.graph_id == graph, "PPI edge block graph id mismatch");
                require(
                    item.src >= local.node_ptr[graph] && item.src < local.node_ptr[graph + 1] &&
                        item.dst >= local.node_ptr[graph] && item.dst < local.node_ptr[graph + 1],
                    "PPI representative batch contains a cross-graph edge");
            }
        }
    }

    void test_formal_shape_and_activation_boundaries()
    {
        const auto ppi_rows = gatzk::protocol::minimum_edge_trace_length(
            56944, 875660, 8, 121);
        require(
            ppi_rows == 6890224,
            "PPI edge trace lower bound must include the complete N*C output view");
        require(
            gatzk::algebra::next_power_of_two(ppi_rows + 2) == 8388608,
            "PPI output view requires the 2^23 edge work domain");
        require_throws(
            []
            {
                (void)gatzk::protocol::minimum_edge_trace_length(
                    std::numeric_limits<std::size_t>::max(), 1, 2, 1);
            },
            "flattened edge trace length overflow must reject");

        const auto ogbn = gatzk::util::load_config(
            (repo_root() / "configs/ogbn_arxiv_secure.cfg").string());
        constexpr std::size_t observed_ogbn_hidden_projection_bound = 1069433;
        require(
            observed_ogbn_hidden_projection_bound > (1ULL << 20),
            "ogbn-arxiv regression must cross the former heuristic ELU bound");
        require(
            observed_ogbn_hidden_projection_bound <= (1ULL << ogbn.range_bits),
            "ogbn-arxiv registered ELU range must cover its signed hidden projection");
        require(
            ogbn.range_bits == 21,
            "ogbn-arxiv must use the minimum power-of-two range covering the formal input/checkpoint boundary");
        require(
            2 * (1ULL << ogbn.range_bits) + 3 <= (1ULL << 23),
            "ogbn-arxiv ELU table and terminal padding must fit its registered edge domain");
        constexpr std::size_t ogbn_nodes = 169343;
        constexpr std::size_t ogbn_edges = 1335586;
        constexpr std::size_t ogbn_hidden_dim = 8;
        constexpr std::size_t ogbn_classes = 40;
        constexpr std::size_t ogbn_edge_domain = 1ULL << 23;
        require(
            ogbn_edges < ogbn_edge_domain && ogbn_nodes * ogbn_hidden_dim < ogbn_edge_domain && ogbn_nodes * ogbn_classes < ogbn_edge_domain,
            "ogbn-arxiv compact active prefixes must fit the registered edge domain");
        require(
            ogbn_nodes * ogbn_classes == 6773720,
            "ogbn-arxiv output active-prefix boundary changed unexpectedly");
    }

    void test_ppi_deferred_commitment_chunk_dependency_order()
    {
        constexpr std::size_t ppi_edge_domain = 1ULL << 23;
        constexpr std::size_t commitment_payload_limit =
            96ULL * 1024ULL * 1024ULL;
        require(
            ppi_edge_domain * sizeof(FieldElement) > commitment_payload_limit,
            "PPI regression must force each edge polynomial into its own commitment chunk");
        const std::vector<std::vector<std::string>> produced = {
            {"P_out0_s_max"},
            {"P_out0_C_max"},
            {"P_out0_widehat_y_star"},
            {"P_out0_w"},
            {"P_out0_T"},
            {"P_out0_T_edge"},
            {"P_out0_PSQ"},
            {"P_out0_Y_star"},
            {"P_out0_Y_star_edge"},
        };
        const std::vector<std::vector<std::string>> required = {
            {},
            {},
            {},
            {"P_out0_U", "P_out0_widehat_y_star"},
            {"P_out0_Sum", "P_out0_Y_star"},
            {"P_out0_Sum_edge", "P_out0_Y_star_edge"},
            {},
            {},
            {},
        };
        const auto order =
            gatzk::protocol::stable_dynamic_commitment_chunk_order(
                produced, required);
        require(order.size() == produced.size(),
                "PPI dynamic commitment scheduler must retain every chunk");
        std::vector<std::size_t> position(order.size(), 0);
        for (std::size_t index = 0; index < order.size(); ++index)
        {
            position.at(order[index]) = index;
        }
        require(position[7] < position[4],
                "P_out0_Y_star must execute before dependent P_out0_T");
        require(position[8] < position[5],
                "P_out0_Y_star_edge must execute before dependent P_out0_T_edge");
        require(position[2] < position[3],
                "already ordered deferred dependencies must remain ordered");
    }

    void test_weighted_edge_aggregate_raw_by_destination()
    {
        const std::vector<std::vector<std::int64_t>> source_rows{
            {2, -3, 5},
            {7, 11, -13},
            {-17, 19, 23},
            {29, -31, 37},
        };
        const std::vector<std::size_t> edge_src{0, 1, 2, 1, 3, 0};
        const std::vector<std::size_t> edge_dst{0, 0, 1, 1, 1, 3};
        const std::vector<std::int64_t> weights{3, -2, 5, 7, -11, 13};
        const std::vector<std::pair<std::size_t, std::size_t>> groups{
            {0, 2}, {2, 5}, {5, 6}};
        const auto actual =
            gatzk::protocol::weighted_edge_aggregate_raw_by_destination(
                weights,
                source_rows,
                edge_src,
                edge_dst,
                groups,
                4,
                "unit grouped attention");
        std::vector<std::int64_t> expected(4 * 3, 0);
        for (std::size_t destination = 0; destination < 4; ++destination)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                __int128 sum = 0;
                for (std::size_t edge = 0; edge < edge_src.size(); ++edge)
                {
                    if (edge_dst[edge] == destination)
                    {
                        sum += static_cast<__int128>(weights[edge]) * source_rows[edge_src[edge]][column];
                    }
                }
                expected[destination * 3 + column] =
                    static_cast<std::int64_t>(sum);
            }
        }
        require(actual == expected,
                "grouped weighted edge aggregation must equal canonical edge-order sums");
        require_throws(
            [&]
            {
                (void)gatzk::protocol::weighted_edge_aggregate_raw_by_destination(
                    weights,
                    source_rows,
                    edge_src,
                    edge_dst,
                    {{0, 2}, {3, 6}},
                    4,
                    "malformed grouped attention");
            },
            "grouped weighted edge aggregation must reject a coverage gap");

        if (std::getenv("GATZK_RUN_TRACE_AGGREGATION_STAGE_TESTS") == nullptr)
        {
            return;
        }
        constexpr std::size_t node_count = 2048;
        constexpr std::size_t edges_per_node = 8;
        constexpr std::size_t width = 8;
        std::vector<std::vector<std::int64_t>> stage_rows(
            node_count,
            std::vector<std::int64_t>(width, 0));
        for (std::size_t row = 0; row < node_count; ++row)
        {
            for (std::size_t column = 0; column < width; ++column)
            {
                stage_rows[row][column] =
                    static_cast<std::int64_t>(
                        (row * 17 + column * 31) % 101) -
                    50;
            }
        }
        std::vector<std::size_t> stage_src;
        std::vector<std::size_t> stage_dst;
        std::vector<std::int64_t> stage_weights;
        std::vector<std::pair<std::size_t, std::size_t>> stage_groups;
        stage_src.reserve(node_count * edges_per_node);
        stage_dst.reserve(node_count * edges_per_node);
        stage_weights.reserve(node_count * edges_per_node);
        stage_groups.reserve(node_count);
        for (std::size_t destination = 0;
             destination < node_count; ++destination)
        {
            const auto begin = stage_src.size();
            for (std::size_t offset = 0; offset < edges_per_node; ++offset)
            {
                stage_src.push_back(
                    (destination * 13 + offset * 97) % node_count);
                stage_dst.push_back(destination);
                stage_weights.push_back(static_cast<std::int64_t>(offset + 1));
            }
            stage_groups.emplace_back(begin, stage_src.size());
        }
        const auto stage_start = std::chrono::steady_clock::now();
        const auto stage_result =
            gatzk::protocol::weighted_edge_aggregate_raw_by_destination(
                stage_weights,
                stage_rows,
                stage_src,
                stage_dst,
                stage_groups,
                node_count,
                "stage grouped attention");
        const auto stage_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - stage_start)
                                  .count();
        require(stage_result.size() == node_count * width,
                "trace aggregation stage result shape changed");
        std::uint64_t checksum = 0;
        for (const auto value : stage_result)
        {
            checksum = checksum * 1315423911ULL + static_cast<std::uint64_t>(value);
        }
        std::cout << "trace_aggregation_stage_ms=" << std::fixed
                  << std::setprecision(3) << stage_ms
                  << " checksum=" << checksum << '\n';
    }

    void test_field_element_spill_io()
    {
        const std::vector<FieldElement> values{
            FieldElement::zero(),
            FieldElement::one(),
            FieldElement(17),
            -FieldElement(29),
        };
        std::stringstream stream(
            std::ios::in | std::ios::out | std::ios::binary);
        gatzk::algebra::write_field_elements_binary(stream, values);
        stream.seekg(0);
        require(
            gatzk::algebra::read_field_elements_binary(stream, values.size()) == values,
            "field spill I/O must preserve canonical field values");

        if (std::getenv("GATZK_RUN_FIELD_IO_STAGE_TESTS") == nullptr)
        {
            return;
        }
        constexpr std::size_t value_count = 1ULL << 20;
        std::vector<FieldElement> stage_values(value_count, FieldElement::zero());
        for (std::size_t i = 0; i < stage_values.size(); ++i)
        {
            stage_values[i] = FieldElement(
                static_cast<std::uint64_t>(i * 1315423911ULL + 17));
        }
        const auto stage_path =
            std::filesystem::path(GATZK_TEST_BINARY_DIR) / "field_io_stage.bin";
        struct StageCleanup
        {
            std::filesystem::path path;
            ~StageCleanup()
            {
                std::error_code ignored;
                std::filesystem::remove(path, ignored);
            }
        } cleanup{stage_path};

        const auto write_start = std::chrono::steady_clock::now();
        {
            std::ofstream output(
                stage_path, std::ios::binary | std::ios::trunc);
            require(static_cast<bool>(output),
                    "failed to create field I/O stage file");
            gatzk::algebra::write_field_elements_binary(
                output, stage_values);
        }
        const auto write_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - write_start)
                                  .count();
        const auto read_start = std::chrono::steady_clock::now();
        std::vector<FieldElement> restored;
        {
            std::ifstream input(stage_path, std::ios::binary);
            require(static_cast<bool>(input),
                    "failed to open field I/O stage file");
            restored = gatzk::algebra::read_field_elements_binary(
                input, stage_values.size());
        }
        const auto read_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - read_start)
                                 .count();
        require(restored == stage_values,
                "field I/O stage roundtrip changed a canonical value");
        std::cout << "field_io_stage_write_ms=" << std::fixed
                  << std::setprecision(3) << write_ms
                  << " field_io_stage_read_ms=" << read_ms
                  << " bytes=" << value_count * 32 << '\n';
    }

    void test_large_degree_pi_bind_codec_boundary()
    {
        const auto &context = unit_security_fixture().context;
        auto proof = unit_security_fixture().proof;
        constexpr std::size_t formal_large_degree = 67108863;
        bool changed = false;
        for (auto &subproof : proof.pi_bind.subproofs)
        {
            for (auto &relation : subproof.relation_meta.shift_relations)
            {
                if (relation.max_difference_degree != 0)
                {
                    relation.max_difference_degree = formal_large_degree;
                    changed = true;
                }
            }
        }
        require(changed, "unit fixture must contain a nontrivial Pi_bind shift relation");
        const auto wire = gatzk::protocol::serialize_proof(proof);
        try
        {
            (void)gatzk::protocol::parse_proof(context, wire);
            throw std::runtime_error(
                "mismatched formal degree must not pass the instance-schema check");
        }
        catch (const std::runtime_error &error)
        {
            const std::string message = error.what();
            require(
                message.find("Pi_bind RelationMeta does not match instance schema") != std::string::npos,
                "large formal degree must decode as numeric metadata before strict schema comparison: " + message);
        }
    }

    void test_full_metadata_and_schema_gates()
    {
        const auto &fixture = cora_secure_fixture();
        const auto metadata = gatzk::protocol::canonical_public_metadata(fixture.context);
        require(metadata.protocol_name == "zkgat", "metadata protocol_name mismatch");
        require(metadata.proof_task == "node_classification_full_gat", "metadata proof_task mismatch");
        require(metadata.privacy_policy == "model_and_trace_hidden", "metadata privacy policy mismatch");
        require(metadata.input_privacy == "public_graph_public_features", "metadata input privacy mismatch");
        require(metadata.output_mode == "public_logits", "metadata output mode mismatch");
        require(metadata.opening_mode == "point_grouped_batch_opening", "metadata opening mode mismatch");
        require(metadata.binding_mode == "zk_subproof_for_crpc", "metadata binding mode mismatch");

        const auto schema = gatzk::protocol::build_proof_schema(fixture.context);
        require(!schema.entries.empty(), "ProofSchema must register committed objects");
        require(schema.private_plain_value_count == 0, "ProofSchema private plain value count must be zero");

        const auto binding_meta = gatzk::protocol::build_binding_relation_metadata(
            fixture.context);
        const auto output_meta = std::find_if(
            binding_meta.begin(), binding_meta.end(), [](const auto &entry)
            { return entry.object_kind == "output" && !entry.shift_relations.empty(); });
        require(output_meta != binding_meta.end(),
                "compiler metadata must register output-bias shift relation");
        const auto &shift = output_meta->shift_relations.front();
        require(shift.max_difference_degree == fixture.context.kzg.manifest.degree_bound,
                "shift soundness must use verifier-enforced global D_max");
        require(shift.configured_degrees.size() == 1,
                "output-bias relation must have one compiler-fixed shifted source");
        const auto shifted_source = shift.source_linear_maps[1].substr(3);
        const auto q_max = gatzk::protocol::registered_blinder_coefficient_count(
            fixture.context, shifted_source);
        const auto shifted_domain_size = shift.source_domains[1] == fixture.context.domains.edge->name
                                             ? fixture.context.domains.edge->size
                                             : fixture.context.domains.c->size;
        require(shift.configured_degrees.front() == shifted_domain_size + q_max - 1,
                "Taylor shift degree must come from the registered source blinder budget");
    }

    void test_reference_consistency_and_verify()
    {
        const auto &fixture = cora_secure_fixture();
        require(gatzk::protocol::verify(fixture.context, fixture.proof), "secure Cora proof must verify");
        require(fixture.proof.public_metadata.output_mode == "public_logits", "secure proof must expose public logits");
        require(fixture.proof.public_metadata.srs_type == "trusted_srs", "secure proof must bind trusted SRS type");
    }

    std::size_t find_wire_token(
        const std::vector<std::uint8_t> &bytes,
        const std::string &token)
    {
        const auto it = std::search(bytes.begin(), bytes.end(), token.begin(), token.end());
        require(it != bytes.end(), "wire token not found: " + token);
        return static_cast<std::size_t>(std::distance(bytes.begin(), it));
    }

    void test_strict_proof_codec()
    {
        const auto &fixture = cora_secure_fixture();
        const auto bytes = gatzk::protocol::serialize_proof(fixture.proof);
        require(bytes.size() == gatzk::protocol::proof_size_bytes(fixture.proof),
                "proof_size_bytes must equal canonical serialized bytes");
        const auto parsed = gatzk::protocol::parse_proof(fixture.context, bytes);
        require(gatzk::protocol::verify(fixture.context, parsed),
                "canonical proof round-trip must verify");
        require(gatzk::protocol::serialize_proof(parsed) == bytes,
                "canonical proof round-trip must be byte stable");

        auto truncated = bytes;
        truncated.pop_back();
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(fixture.context, truncated); },
            "truncated proof must be rejected by parser");

        auto appended = bytes;
        appended.push_back(0);
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(fixture.context, appended); },
            "trailing proof bytes must be rejected by parser");

        auto reordered = bytes;
        const auto m_pub = find_wire_token(reordered, "M_pub");
        const auto w_ext = find_wire_token(reordered, "W_ext");
        std::swap_ranges(reordered.begin() + static_cast<std::ptrdiff_t>(m_pub),
                         reordered.begin() + static_cast<std::ptrdiff_t>(m_pub + 5),
                         reordered.begin() + static_cast<std::ptrdiff_t>(w_ext));
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(fixture.context, reordered); },
            "reordered proof blocks must be rejected by parser");

        auto duplicated = bytes;
        std::copy_n(duplicated.begin() + static_cast<std::ptrdiff_t>(w_ext), 5,
                    duplicated.begin() + static_cast<std::ptrdiff_t>(m_pub));
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(fixture.context, duplicated); },
            "duplicate/missing proof block tags must be rejected by parser");

        auto invalid_length = bytes;
        require(invalid_length.size() > 24, "proof wire header is unexpectedly short");
        std::fill_n(invalid_length.begin() + 16, 8, 0xff);
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(fixture.context, invalid_length); },
            "invalid proof length must be rejected by parser");

        auto invalid_kind = fixture.proof;
        require(!invalid_kind.external_evaluation_kinds.empty(),
                "fixture must contain a typed external evaluation");
        invalid_kind.external_evaluation_kinds[0] =
            static_cast<gatzk::protocol::PayloadValueKind>(99);
        const auto invalid_kind_bytes = gatzk::protocol::serialize_proof(invalid_kind);
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(fixture.context, invalid_kind_bytes); },
            "invalid proof payload type must be rejected by parser");

        const auto output_binding = std::find_if(
            fixture.proof.pi_bind.subproofs.begin(),
            fixture.proof.pi_bind.subproofs.end(),
            [](const auto &subproof)
            {
                return subproof.relation_meta.object_kind == "output";
            });
        require(output_binding != fixture.proof.pi_bind.subproofs.end(),
                "fixture must contain output Pi_bind for strict parser tests");
        const auto output_index = static_cast<std::size_t>(std::distance(
            fixture.proof.pi_bind.subproofs.begin(), output_binding));

        auto missing_open_reference = fixture.proof;
        require(!missing_open_reference.pi_bind.subproofs[output_index]
                     .opening_references.empty(),
                "output Pi_bind must carry opening references");
        missing_open_reference.pi_bind.subproofs[output_index]
            .opening_references.pop_back();
        const auto missing_open_reference_bytes =
            gatzk::protocol::serialize_proof(missing_open_reference);
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(
                  fixture.context, missing_open_reference_bytes); },
            "missing Pi_bind opening reference must be rejected by parser");

        auto reordered_view_commitment = fixture.proof;
        auto &view_commitments = reordered_view_commitment.pi_bind
                                     .subproofs[output_index]
                                     .view_logup_commitments;
        require(view_commitments.size() >= 2,
                "output Pi_bind must carry ordered view commitments");
        std::swap(view_commitments[0], view_commitments[1]);
        const auto reordered_view_commitment_bytes =
            gatzk::protocol::serialize_proof(reordered_view_commitment);
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(
                  fixture.context, reordered_view_commitment_bytes); },
            "reordered Pi_bind commitment group must be rejected by parser");

        auto renamed_model_reblind = fixture.proof;
        require(!renamed_model_reblind.pi_bind.subproofs[output_index]
                     .model_reblinds.empty(),
                "output Pi_bind must carry model reblinds");
        renamed_model_reblind.pi_bind.subproofs[output_index]
            .model_reblinds.front()
            .proof_local_commitment.name += ":tampered";
        const auto renamed_model_reblind_bytes =
            gatzk::protocol::serialize_proof(renamed_model_reblind);
        require_throws(
            [&]
            { (void)gatzk::protocol::parse_proof(
                  fixture.context, renamed_model_reblind_bytes); },
            "renamed Pi_bind model commitment must be rejected by parser");
    }

    void require_verifier_only_boundary(const SecureFixture &fixture)
    {
        auto context = fixture.context;
        auto proof = fixture.proof;
        strip_context_to_verifier_only(context);
        strip_proof_to_verifier_only(proof);
        require(context.kzg.pk.powers_g1 == nullptr, "verifier-only context must not retain coefficient proving SRS");
        require(context.kzg.pk.lagrange_srs_by_domain.empty(), "verifier-only context must not retain Lagrange proving SRS");
        require(context.model.hidden_heads.front().seq_kernel_fp.empty(),
                "verifier-only context must not retain private model parameters");
        require(gatzk::protocol::verify(context, proof), "verifier-only proof path must verify without prover material");
        const auto verifier_wire = gatzk::protocol::serialize_proof(proof);
        const auto parsed = gatzk::protocol::parse_proof(context, verifier_wire);
        require(gatzk::protocol::verify(context, parsed),
                "verifier-only parser path must verify without prover material");
        require(
            gatzk::protocol::private_plain_value_count(context, proof) == 0,
            "verifier payload audit must observe zero private plain values");
    }

    void test_verifier_only_boundary()
    {
        require_verifier_only_boundary(cora_secure_fixture());
    }

    void require_rejects(
        const SecureFixture &fixture,
        gatzk::protocol::Proof proof,
        const std::string &label)
    {
        require(!gatzk::protocol::verify(fixture.context, proof), label + " must reject");
    }

    void require_point_group_proof_tampers_reject(
        const SecureFixture &fixture)
    {
        auto witness_tamper = fixture.proof;
        require(!witness_tamper.opening_groups.empty(),
                "fixture must carry standard point-group openings");
        witness_tamper.opening_groups.front().witness = gatzk::crypto::g1_add(
            witness_tamper.opening_groups.front().witness,
            fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, std::move(witness_tamper), "point-group witness tamper");

        auto group_id_tamper = fixture.proof;
        group_id_tamper.opening_groups.front().group_id += ":tampered";
        require_rejects(fixture, group_id_tamper, "point-group id tamper");
        require_throws(
            [&]
            { (void)gatzk::protocol::serialize_proof(group_id_tamper); },
            "serializer must reject a non-canonical point-group id");

        auto owner_tamper = fixture.proof;
        auto &owner_group = owner_tamper.opening_groups.front();
        owner_group.owner_block = "Open_dom:unregistered";
        owner_group.group_id = gatzk::protocol::point_group_id(
            owner_group.owner_block, owner_group.point);
        require_rejects(fixture, owner_tamper, "point-group owner tamper");
        require_throws(
            [&]
            { (void)gatzk::protocol::serialize_proof(owner_tamper); },
            "serializer must reject an unregistered point-group owner");

        auto point_tamper = fixture.proof;
        point_tamper.opening_groups.front().point += FieldElement::one();
        require_rejects(
            fixture, std::move(point_tamper),
            "point-group evaluation point tamper");

        auto duplicate_tamper = fixture.proof;
        duplicate_tamper.opening_groups.push_back(
            duplicate_tamper.opening_groups.front());
        require_rejects(
            fixture, duplicate_tamper,
            "duplicate point-group owner/witness tamper");
        require_throws(
            [&]
            { (void)gatzk::protocol::serialize_proof(duplicate_tamper); },
            "serializer must reject a duplicate point-group owner/witness");

        auto order_tamper = fixture.proof;
        require(order_tamper.opening_groups.size() >= 2,
                "fixture must carry multiple point groups");
        std::swap(order_tamper.opening_groups[0], order_tamper.opening_groups[1]);
        require_rejects(
            fixture, order_tamper, "point-group canonical order tamper");
        require_throws(
            [&]
            { (void)gatzk::protocol::serialize_proof(order_tamper); },
            "serializer must reject non-canonical point-group order");

        auto malformed_wire = gatzk::protocol::serialize_proof(fixture.proof);
        const auto &encoded_group_id = fixture.proof.opening_groups.front().group_id;
        const auto encoded_group_id_offset = find_wire_token(
            malformed_wire, encoded_group_id);
        malformed_wire[encoded_group_id_offset + encoded_group_id.size() - 1] ^= 1;
        require_throws(
            [&]
            {
                (void)gatzk::protocol::parse_proof(
                    fixture.context, malformed_wire);
            },
            "strict parser must reject a tampered point-group reference");
    }

    void test_unit_coefficient_basis_closure()
    {
        const auto &fixture = unit_security_fixture();
        require(gatzk::protocol::verify(fixture.context, fixture.proof),
                "unit coefficient-basis proof must verify");

        auto swapped_artifact_context = fixture.context;
        swapped_artifact_context.compiler_artifact =
            std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>(
                *fixture.context.compiler_artifact);
        require(!gatzk::protocol::verify(
                    swapped_artifact_context, fixture.proof),
                "an unvalidated compiler-artifact object swap must reject");

        auto mutated_artifact_context = fixture.context;
        auto mutated_artifact =
            std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>(
                *fixture.context.compiler_artifact);
        mutated_artifact->instance_canonical_bytes.push_back('\0');
        mutated_artifact_context.compiler_artifact = std::move(mutated_artifact);
        require(!gatzk::protocol::verify(
                    mutated_artifact_context, fixture.proof),
                "a mutated compiler-artifact replacement must reject");

        const auto bytes = gatzk::protocol::serialize_proof(fixture.proof);
        require_throws(
            [&]
            {
                (void)gatzk::protocol::parse_proof(
                    swapped_artifact_context, bytes);
            },
            "strict parser must reject an unvalidated compiler-artifact object swap");
        require_throws(
            [&]
            {
                (void)gatzk::protocol::parse_proof(
                    mutated_artifact_context, bytes);
            },
            "strict parser must reject a mutated compiler-artifact replacement");
        const auto parsed = gatzk::protocol::parse_proof(fixture.context, bytes);
        require(gatzk::protocol::verify(fixture.context, parsed),
                "unit coefficient-basis proof must survive strict parser round-trip");
        require(gatzk::protocol::serialize_proof(parsed) == bytes,
                "unit coefficient-basis proof wire encoding must be canonical");
        require_point_group_proof_tampers_reject(fixture);
        require_verifier_only_boundary(fixture);

        require(!fixture.proof.dynamic_commitments.empty(),
                "unit proof must contain a batched dynamic commitment group");
        const auto encoded_commitment = gatzk::crypto::serialize(
            fixture.proof.dynamic_commitments.front().second.point);
        auto invalid_subgroup_encoding = bytes;
        const auto encoded_position = std::search(
            invalid_subgroup_encoding.begin(), invalid_subgroup_encoding.end(),
            encoded_commitment.begin(), encoded_commitment.end());
        require(encoded_position != invalid_subgroup_encoding.end(),
                "unit proof wire must contain the selected commitment encoding");
        std::fill(
            encoded_position,
            encoded_position + static_cast<std::ptrdiff_t>(encoded_commitment.size()),
            0xff);
        require_throws(
            [&]
            {
                (void)gatzk::protocol::parse_proof(
                    fixture.context, invalid_subgroup_encoding);
            },
            "parallel strict parser must reject an invalid G1/subgroup encoding");

        const auto output_subproof = std::find_if(
            fixture.proof.pi_bind.subproofs.begin(),
            fixture.proof.pi_bind.subproofs.end(),
            [](const auto &subproof)
            {
                return subproof.relation_meta.object_kind == "output" && subproof.relation_meta.coefficient_basis_relations.size() == 1;
            });
        require(output_subproof != fixture.proof.pi_bind.subproofs.end(),
                "unit proof must contain the formal output coefficient-basis relation");
        const auto subproof_index = static_cast<std::size_t>(std::distance(
            fixture.proof.pi_bind.subproofs.begin(), output_subproof));
        const auto &basis =
            output_subproof->relation_meta.coefficient_basis_relations.front();
        require(output_subproof->relation_meta.canonical_matrices.size() == 3,
                "unit output relation must register Y-prime, Y-pre, and head-output canonical matrices");
        std::size_t expected_canonical_commitments = 0;
        std::set<std::string> canonical_challenge_labels;
        for (const auto &matrix : output_subproof->relation_meta.canonical_matrices)
        {
            require(matrix.column_objects.size() == matrix.column_count && matrix.row_count == fixture.context.local.num_nodes && matrix.canonical_domain == fixture.context.domains.n->name,
                    "unit canonical matrix RelationMeta shape mismatch");
            require(matrix.compressed_object == matrix.matrix_object + "_CanFold" && matrix.compression_challenge_label == "xi_can:" + matrix.relation_id && canonical_challenge_labels.insert(matrix.compression_challenge_label).second && fixture.proof.challenges.contains(matrix.compression_challenge_label),
                    "canonical matrix must own a unique post-commit compression challenge");
            expected_canonical_commitments += matrix.column_count + 1;
        }
        require(output_subproof->view_commitments.size() == expected_canonical_commitments,
                "unit Com_view must encode every canonical matrix column and fold");
        require(output_subproof->relation_meta.aggregation_relations.size() == 1,
                "unit output relation must own one independent aggregation binding");
        const auto &aggregation =
            output_subproof->relation_meta.aggregation_relations.front();
        require(aggregation.source_matrix_object == output_subproof->relation_meta.canonical_matrices[0]
                                                        .matrix_object &&
                    aggregation.target_matrix_object == output_subproof->relation_meta.canonical_matrices[1]
                                                            .matrix_object &&
                    aggregation.alpha_object == "P_out0_alpha" && aggregation.compression_challenge_label == "xi_agg_bind:" + aggregation.relation_id && fixture.proof.challenges.contains(aggregation.compression_challenge_label) && aggregation.compression_challenge_label != output_subproof->relation_meta.canonical_matrices[0].compression_challenge_label,
                "aggregation binding must use the registered independent post-commit challenge");
        require(output_subproof->aggregation_commitments.size() == aggregation.commitment_order.size() && std::equal(
                                                                                                              output_subproof->aggregation_commitments.begin(),
                                                                                                              output_subproof->aggregation_commitments.end(),
                                                                                                              aggregation.commitment_order.begin(),
                                                                                                              [](const auto &actual, const std::string &expected)
                                                                                                              {
                                                                                                                  return actual.first == expected;
                                                                                                              }),
                "unit Com_agg must follow its exact RelationMeta commitment order");
        for (const auto *route : {
                 &aggregation.source_route, &aggregation.target_route})
        {
            require(route->configured_shift_degree > 0 && route->shift_domain_size >= 2 * route->configured_shift_degree + 1 && fixture.proof.challenges.contains(route->eta_challenge_label) && fixture.proof.challenges.contains(route->beta_challenge_label) && fixture.proof.challenges.contains(route->terminal_challenge_label),
                    "aggregation broadcast must register LogUp and KZG terminal challenges");
        }
        require(
            fixture.proof.route_references == gatzk::protocol::route_opening_references(
                                                  fixture.context, fixture.proof.challenges),
            "S_route must contain only the canonical point-group references");
        require(!output_subproof->relation_meta.route_terminal_relations.empty(),
                "output relation must register native route terminal relations");
        const auto &native_route =
            output_subproof->relation_meta.route_terminal_relations.front();
        const auto native_shift_commitment = std::find_if(
            output_subproof->relation_commitments.begin(),
            output_subproof->relation_commitments.end(),
            [&](const auto &entry)
            {
                return entry.first == native_route.shifted_accumulator_object;
            });
        require(native_shift_commitment != output_subproof->relation_commitments.end(),
                "native route shifted accumulator must be serialized in Com_rel");

        auto native_shift_commitment_tamper = fixture.proof;
        auto &native_relation_commitments = native_shift_commitment_tamper.pi_bind
                                                .subproofs[subproof_index]
                                                .relation_commitments;
        const auto tampered_native_shift = std::find_if(
            native_relation_commitments.begin(),
            native_relation_commitments.end(),
            [&](const auto &entry)
            {
                return entry.first == native_route.shifted_accumulator_object;
            });
        require(tampered_native_shift != native_relation_commitments.end(),
                "native route shift commitment must be present for tamper test");
        tampered_native_shift->second.point = gatzk::crypto::g1_add(
            tampered_native_shift->second.point,
            fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, std::move(native_shift_commitment_tamper),
            "native route shifted-accumulator commitment tamper");

        auto native_shift_value_tamper = fixture.proof;
        auto &native_relation_evaluations = native_shift_value_tamper.pi_bind
                                                .subproofs[subproof_index]
                                                .evaluations;
        const auto native_shift_value = std::find_if(
            native_relation_evaluations.begin(),
            native_relation_evaluations.end(),
            [&](const auto &evaluation)
            {
                return evaluation.object_id == native_route.shifted_accumulator_object + ":u_rel";
            });
        require(native_shift_value != native_relation_evaluations.end() && native_shift_value->domain == "relation" && native_shift_value->values.size() == 1,
                "native route must encode one masked Taylor-shift value");
        native_shift_value->values.front() += FieldElement::one();
        require_rejects(
            fixture, std::move(native_shift_value_tamper),
            "native route Taylor-shift value tamper");

        auto route_reference_tamper = fixture.proof;
        require(route_reference_tamper.route_references.size() >= 2,
                "S_route must contain multiple canonical references");
        std::swap(route_reference_tamper.route_references[0],
                  route_reference_tamper.route_references[1]);
        require_rejects(
            fixture, route_reference_tamper,
            "S_route reference order tamper");
        require_throws(
            [&]
            {
                const auto tampered_bytes =
                    gatzk::protocol::serialize_proof(route_reference_tamper);
                (void)gatzk::protocol::parse_proof(
                    fixture.context, tampered_bytes);
            },
            "strict parser must reject reordered S_route references");
        const auto average_subproof = std::find_if(
            fixture.proof.pi_bind.subproofs.begin(),
            fixture.proof.pi_bind.subproofs.end(),
            [](const auto &subproof)
            {
                return subproof.relation_meta.object_kind == "average";
            });
        require(average_subproof != fixture.proof.pi_bind.subproofs.end() && average_subproof->relation_meta.canonical_matrices.size() == 2 && average_subproof->relation_meta.canonical_matrices[0].matrix_object == "P_Y_lin" && average_subproof->relation_meta.canonical_matrices[1].matrix_object == "P_Y",
                "unit average relation must register pre-average and final-output canonical matrices");
        require(!basis.chunks.empty() && output_subproof->basis_commitments.size() == basis.commitment_order.size(),
                "unit coefficient-basis commitment group must match RelationMeta");
        require(std::none_of(
                    output_subproof->basis_commitments.begin(),
                    output_subproof->basis_commitments.end(),
                    [&](const auto &item)
                    { return item.first == basis.power_object; }),
                "Gamma powers must remain a verifier-local uncommitted column");
        require(std::none_of(
                    output_subproof->evaluations.begin(),
                    output_subproof->evaluations.end(),
                    [&](const auto &item)
                    { return item.object_id == basis.power_object; }),
                "Gamma powers must not be serialized as a Pi_bind evaluation");
        require(std::none_of(
                    fixture.proof.dynamic_commitments.begin(),
                    fixture.proof.dynamic_commitments.end(),
                    [&](const auto &item)
                    { return item.first == basis.power_object; }),
                "Gamma powers must not appear in Com_dyn");
        require(std::none_of(
                    fixture.proof.domain_openings.begin(),
                    fixture.proof.domain_openings.end(),
                    [&](const auto &bundle)
                    {
                        return std::any_of(
                            bundle.second.values.begin(), bundle.second.values.end(),
                            [&](const auto &item)
                            {
                                return item.first == basis.power_object;
                            });
                    }),
                "Gamma powers must not appear in Open_dom");

        auto canonical_commitment_tamper = fixture.proof;
        auto &canonical_commitment = canonical_commitment_tamper.pi_bind
                                         .subproofs[subproof_index]
                                         .view_commitments.front()
                                         .second;
        canonical_commitment.point = gatzk::crypto::g1_add(
            canonical_commitment.point, fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, std::move(canonical_commitment_tamper),
            "canonical matrix column commitment tamper");

        auto canonical_value_tamper = fixture.proof;
        const auto canonical_label = output_subproof->view_commitments.front().first;
        auto &canonical_evaluations = canonical_value_tamper.pi_bind
                                          .subproofs[subproof_index]
                                          .evaluations;
        const auto canonical_evaluation = std::find_if(
            canonical_evaluations.begin(), canonical_evaluations.end(),
            [&](const auto &evaluation)
            {
                return evaluation.object_id == canonical_label;
            });
        require(canonical_evaluation != canonical_evaluations.end() && canonical_evaluation->domain == "N" && canonical_evaluation->values.size() == 2,
                "canonical matrix column must own two masked N-domain values");
        canonical_evaluation->values.front() += FieldElement::one();
        require_rejects(
            fixture, std::move(canonical_value_tamper),
            "canonical matrix masked-value tamper");

        const auto &first_canonical =
            output_subproof->relation_meta.canonical_matrices.front();
        auto canonical_fold_commitment_tamper = fixture.proof;
        auto &fold_commitments = canonical_fold_commitment_tamper.pi_bind
                                     .subproofs[subproof_index]
                                     .view_commitments;
        const auto fold_commitment = std::find_if(
            fold_commitments.begin(), fold_commitments.end(),
            [&](const auto &entry)
            {
                return entry.first == first_canonical.compressed_object;
            });
        require(fold_commitment != fold_commitments.end(),
                "canonical matrix fold commitment must be serialized in Com_view");
        fold_commitment->second.point = gatzk::crypto::g1_add(
            fold_commitment->second.point, fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, std::move(canonical_fold_commitment_tamper),
            "canonical matrix post-challenge fold commitment tamper");

        auto canonical_fold_value_tamper = fixture.proof;
        auto &fold_evaluations = canonical_fold_value_tamper.pi_bind
                                     .subproofs[subproof_index]
                                     .evaluations;
        const auto fold_evaluation = std::find_if(
            fold_evaluations.begin(), fold_evaluations.end(),
            [&](const auto &evaluation)
            {
                return evaluation.object_id == first_canonical.compressed_object;
            });
        require(fold_evaluation != fold_evaluations.end() && fold_evaluation->domain == "N" && fold_evaluation->values.size() == 2,
                "canonical matrix fold must own two masked N-domain values");
        fold_evaluation->values.front() += FieldElement::one();
        require_rejects(
            fixture, std::move(canonical_fold_value_tamper),
            "canonical matrix post-challenge fold value tamper");

        auto canonical_challenge_tamper = fixture.proof;
        canonical_challenge_tamper.challenges.at(
            first_canonical.compression_challenge_label) += FieldElement::one();
        require_rejects(
            fixture, std::move(canonical_challenge_tamper),
            "canonical matrix compression challenge tamper");

        auto aggregation_product_commitment_tamper = fixture.proof;
        auto &aggregation_commitments = aggregation_product_commitment_tamper
                                            .pi_bind.subproofs[subproof_index]
                                            .aggregation_commitments;
        const auto product_commitment = std::find_if(
            aggregation_commitments.begin(), aggregation_commitments.end(),
            [&](const auto &entry)
            {
                return entry.first == aggregation.product_object;
            });
        require(product_commitment != aggregation_commitments.end(),
                "aggregation product commitment must be serialized in Com_agg");
        product_commitment->second.point = gatzk::crypto::g1_add(
            product_commitment->second.point, fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, std::move(aggregation_product_commitment_tamper),
            "aggregation product commitment tamper");

        auto aggregation_product_value_tamper = fixture.proof;
        auto &aggregation_evaluations = aggregation_product_value_tamper.pi_bind
                                            .subproofs[subproof_index]
                                            .evaluations;
        const auto product_evaluation = std::find_if(
            aggregation_evaluations.begin(), aggregation_evaluations.end(),
            [&](const auto &evaluation)
            {
                return evaluation.object_id == aggregation.product_object;
            });
        require(product_evaluation != aggregation_evaluations.end() && product_evaluation->domain == "edge" && product_evaluation->values.size() == 2,
                "aggregation product must own two masked edge-domain values");
        product_evaluation->values.front() += FieldElement::one();
        require_rejects(
            fixture, std::move(aggregation_product_value_tamper),
            "aggregation product masked-value tamper");

        auto aggregation_shift_commitment_tamper = fixture.proof;
        auto &shift_commitments = aggregation_shift_commitment_tamper.pi_bind
                                      .subproofs[subproof_index]
                                      .aggregation_commitments;
        const auto shift_commitment = std::find_if(
            shift_commitments.begin(), shift_commitments.end(),
            [&](const auto &entry)
            {
                return entry.first == aggregation.source_route.shifted_accumulator_object;
            });
        require(shift_commitment != shift_commitments.end(),
                "aggregation route shifted accumulator must be serialized");
        shift_commitment->second.point = gatzk::crypto::g1_add(
            shift_commitment->second.point, fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, std::move(aggregation_shift_commitment_tamper),
            "aggregation route shifted-accumulator commitment tamper");

        auto aggregation_shift_value_tamper = fixture.proof;
        auto &shift_evaluations = aggregation_shift_value_tamper.pi_bind
                                      .subproofs[subproof_index]
                                      .evaluations;
        const auto shift_evaluation = std::find_if(
            shift_evaluations.begin(), shift_evaluations.end(),
            [&](const auto &evaluation)
            {
                return evaluation.object_id == aggregation.source_route.shifted_accumulator_object + ":u_rel";
            });
        require(shift_evaluation != shift_evaluations.end() && shift_evaluation->domain == "relation" && shift_evaluation->values.size() == 1,
                "aggregation route must encode one masked Taylor-shift value");
        shift_evaluation->values.front() += FieldElement::one();
        require_rejects(
            fixture, std::move(aggregation_shift_value_tamper),
            "aggregation route Taylor-shift value tamper");

        auto aggregation_challenge_tamper = fixture.proof;
        aggregation_challenge_tamper.challenges.at(
            aggregation.compression_challenge_label) += FieldElement::one();
        require_rejects(
            fixture, std::move(aggregation_challenge_tamper),
            "aggregation compression challenge tamper");

        auto chunk_value_tamper = fixture.proof;
        auto &chunk_evaluations =
            chunk_value_tamper.pi_bind.subproofs[subproof_index].evaluations;
        const auto chunk_evaluation = std::find_if(
            chunk_evaluations.begin(), chunk_evaluations.end(),
            [&](const auto &evaluation)
            {
                return evaluation.object_id == basis.chunks.front().object_id;
            });
        require(chunk_evaluation != chunk_evaluations.end() && chunk_evaluation->values.size() == 2,
                "unit coefficient chunk must own its two masked edge evaluations");
        chunk_evaluation->values.front() += FieldElement::one();
        require_rejects(
            fixture, std::move(chunk_value_tamper),
            "coefficient chunk masked-value tamper");

        auto basis_commitment_tamper = fixture.proof;
        auto &basis_commitment = basis_commitment_tamper.pi_bind
                                     .subproofs[subproof_index]
                                     .basis_commitments.front()
                                     .second;
        basis_commitment.point = gatzk::crypto::g1_add(
            basis_commitment.point, fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, std::move(basis_commitment_tamper),
            "Com_basis coefficient commitment tamper");

        auto basis_shift_value_tamper = fixture.proof;
        auto &basis_evaluations = basis_shift_value_tamper.pi_bind
                                      .subproofs[subproof_index]
                                      .evaluations;
        const auto shifted_value = std::find_if(
            basis_evaluations.begin(), basis_evaluations.end(),
            [&](const auto &evaluation)
            {
                return evaluation.object_id == basis.chunks.front().shifted_accumulator_object + ":u_rel";
            });
        require(shifted_value != basis_evaluations.end() && shifted_value->values.size() == 1,
                "unit basis link must encode one shared shifted value");
        shifted_value->values.front() += FieldElement::one();
        require_rejects(
            fixture, std::move(basis_shift_value_tamper),
            "coefficient basis-link shifted-value tamper");

        auto basis_metadata_tamper = fixture.proof;
        ++basis_metadata_tamper.pi_bind.subproofs[subproof_index]
              .relation_meta.coefficient_basis_relations.front()
              .chunks.front()
              .offset;
        require_rejects(
            fixture, basis_metadata_tamper,
            "coefficient chunk RelationMeta offset tamper");
        require_throws(
            [&]
            {
                const auto tampered_bytes =
                    gatzk::protocol::serialize_proof(basis_metadata_tamper);
                (void)gatzk::protocol::parse_proof(
                    fixture.context, tampered_bytes);
            },
            "strict parser must reject tampered coefficient chunk metadata");
    }

    void test_tamper_rejection_suite()
    {
        const auto &fixture = cora_secure_fixture();

        auto public_input_context_tamper = fixture.context;
        require(public_input_context_tamper.local.absolute_ids.size() >= 2,
                "fixture must expose at least two public node ids");
        std::swap(public_input_context_tamper.local.absolute_ids[0],
                  public_input_context_tamper.local.absolute_ids[1]);
        public_input_context_tamper.local.public_input.I =
            public_input_context_tamper.local.absolute_ids;
        require(!gatzk::protocol::verify(public_input_context_tamper, fixture.proof),
                "public input replacement must reject against registered public columns");

        auto model_commitment_context_tamper = fixture.context;
        const auto model_label = gatzk::protocol::hidden_weight_column_label(0, 0);
        require(model_commitment_context_tamper.static_commitments.contains(model_label),
                "fixture must contain the first registered hidden-head commitment");
        model_commitment_context_tamper.static_commitments.at(model_label).point =
            gatzk::crypto::g1_zero();
        require(!gatzk::protocol::verify(model_commitment_context_tamper, fixture.proof),
                "registered model/static commitment replacement must reject");

        auto missing_compiler_artifact = fixture.context;
        missing_compiler_artifact.compiler_artifact.reset();
        require(!gatzk::protocol::verify(missing_compiler_artifact, fixture.proof),
                "missing ProtocolCompilerArtifact must reject");

        auto tampered_compiler_artifact = fixture.context;
        auto mutable_artifact = std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>(
            *tampered_compiler_artifact.compiler_artifact);
        mutable_artifact->instance_canonical_bytes.push_back('\0');
        tampered_compiler_artifact.compiler_artifact = std::move(mutable_artifact);
        require(!gatzk::protocol::verify(tampered_compiler_artifact, fixture.proof),
                "tampered ProtocolCompilerArtifact must reject");

        auto observation_budget_tamper = fixture.context;
        auto budget_artifact = std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>(
            *observation_budget_tamper.compiler_artifact);
        require(!budget_artifact->observation_ledger.empty(),
                "fixture compiler artifact must contain ObservationLedger entries");
        ++budget_artifact->observation_ledger.front().query_bound;
        observation_budget_tamper.compiler_artifact = std::move(budget_artifact);
        require(!gatzk::protocol::verify(observation_budget_tamper, fixture.proof),
                "ObservationLedger q_O_max tamper must reject");

        auto observation_dependency_tamper = fixture.context;
        auto dependency_artifact = std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>(
            *observation_dependency_tamper.compiler_artifact);
        dependency_artifact->observation_ledger.front().expression_or_certificate +=
            ":missing-shift-dependency";
        observation_dependency_tamper.compiler_artifact = std::move(dependency_artifact);
        require(!gatzk::protocol::verify(observation_dependency_tamper, fixture.proof),
                "ObservationLedger dependency certificate tamper must reject");

        auto pi_bind_meta_tamper = fixture.proof;
        require(!pi_bind_meta_tamper.pi_bind.subproofs.empty(),
                "fixture must contain Pi_bind subproofs");
        pi_bind_meta_tamper.pi_bind.subproofs.front().relation_meta.relation_id += "_tampered";
        require_rejects(fixture, pi_bind_meta_tamper, "Pi_bind RelationMeta tamper");

        auto pi_bind_missing_tamper = fixture.proof;
        pi_bind_missing_tamper.pi_bind.subproofs.pop_back();
        require_rejects(fixture, pi_bind_missing_tamper, "Pi_bind subproof missing");

        auto pi_bind_duplicate_tamper = fixture.proof;
        pi_bind_duplicate_tamper.pi_bind.subproofs.push_back(
            pi_bind_duplicate_tamper.pi_bind.subproofs.back());
        require_rejects(fixture, pi_bind_duplicate_tamper, "Pi_bind subproof duplicate");

        const auto output_binding = std::find_if(
            fixture.proof.pi_bind.subproofs.begin(),
            fixture.proof.pi_bind.subproofs.end(),
            [](const auto &subproof)
            {
                return !subproof.relation_meta.indexed_views.empty() && !subproof.relation_meta.shift_relations.empty();
            });
        require(output_binding != fixture.proof.pi_bind.subproofs.end(),
                "fixture must contain the formal output-bias Pi_bind relation");
        const auto output_binding_index = static_cast<std::size_t>(
            std::distance(fixture.proof.pi_bind.subproofs.begin(), output_binding));

        auto bias_index_map_tamper = fixture.proof;
        bias_index_map_tamper.pi_bind.subproofs[output_binding_index]
            .relation_meta.indexed_views.front()
            .index_map_id += ":tampered";
        require_rejects(fixture, bias_index_map_tamper,
                        "output-bias indexed-view map tamper");

        auto bias_shift_degree_tamper = fixture.proof;
        ++bias_shift_degree_tamper.pi_bind.subproofs[output_binding_index]
              .relation_meta.shift_relations.front()
              .max_difference_degree;
        require_rejects(fixture, bias_shift_degree_tamper,
                        "output-bias global shift degree tamper");

        auto bias_shift_commitment_tamper = fixture.proof;
        auto &shifted_commitments = bias_shift_commitment_tamper.pi_bind
                                        .subproofs[output_binding_index]
                                        .relation_commitments;
        require(!shifted_commitments.empty(),
                "output-bias Pi_bind must contain the shifted commitment");
        shifted_commitments.front().second.point = gatzk::crypto::g1_add(
            shifted_commitments.front().second.point,
            fixture.context.kzg.g1_generator);
        require_rejects(fixture, bias_shift_commitment_tamper,
                        "output-bias shifted commitment tamper");

        auto bias_shared_value_tamper = fixture.proof;
        auto &binding_evaluations = bias_shared_value_tamper.pi_bind
                                        .subproofs[output_binding_index]
                                        .evaluations;
        require(!binding_evaluations.empty() && !binding_evaluations.front().values.empty(),
                "output-bias Pi_bind must contain u_rel");
        binding_evaluations.front().values.front() += FieldElement::one();
        require_rejects(fixture, bias_shared_value_tamper,
                        "output-bias shared u_rel tamper");

        auto bias_opening_reference_tamper = fixture.proof;
        auto &binding_opening_refs = bias_opening_reference_tamper.pi_bind
                                         .subproofs[output_binding_index]
                                         .opening_references;
        require(!binding_opening_refs.empty(),
                "output-bias Pi_bind must reference standard point groups");
        binding_opening_refs.pop_back();
        require_rejects(fixture, bias_opening_reference_tamper,
                        "output-bias opening reference deletion");

        auto model_reblind_tamper = fixture.proof;
        auto model_subproof = std::find_if(
            model_reblind_tamper.pi_bind.subproofs.begin(),
            model_reblind_tamper.pi_bind.subproofs.end(),
            [](const auto &subproof)
            { return !subproof.model_reblinds.empty(); });
        require(model_subproof != model_reblind_tamper.pi_bind.subproofs.end(),
                "fixture must contain a proof-local model reblinding");
        model_subproof->model_reblinds.front().proof_local_commitment.point =
            gatzk::crypto::g1_add(
                model_subproof->model_reblinds.front().proof_local_commitment.point,
                fixture.context.kzg.g1_generator);
        require_rejects(fixture, model_reblind_tamper, "proof-local model commitment tamper");

        auto model_reblind_witness_tamper = fixture.proof;
        auto witness_subproof = std::find_if(
            model_reblind_witness_tamper.pi_bind.subproofs.begin(),
            model_reblind_witness_tamper.pi_bind.subproofs.end(),
            [](const auto &subproof)
            { return !subproof.model_reblinds.empty(); });
        require(witness_subproof != model_reblind_witness_tamper.pi_bind.subproofs.end(),
                "fixture must contain a proof-local model reblinding witness");
        witness_subproof->model_reblinds.front().blinder_commitment.point =
            gatzk::crypto::g1_add(
                witness_subproof->model_reblinds.front().blinder_commitment.point,
                fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, model_reblind_witness_tamper, "proof-local model S_model tamper");

        auto metadata_tamper = fixture.proof;
        metadata_tamper.public_metadata.privacy_policy = "wrong_policy";
        require_rejects(fixture, metadata_tamper, "privacy_policy tamper");

        auto srs_id_tamper = fixture.proof;
        srs_id_tamper.public_metadata.srs_id = "wrong_srs";
        require_rejects(fixture, srs_id_tamper, "srs_id tamper");

        auto srs_type_tamper = fixture.proof;
        srs_type_tamper.public_metadata.srs_type = "wrong_type";
        require_rejects(fixture, srs_type_tamper, "srs_type tamper");

        auto input_privacy_tamper = fixture.proof;
        input_privacy_tamper.public_metadata.input_privacy = "hidden_graph";
        require_rejects(fixture, input_privacy_tamper, "input_privacy tamper");

        auto output_mode_tamper = fixture.proof;
        output_mode_tamper.public_metadata.output_mode = "label_only";
        require_rejects(fixture, output_mode_tamper, "output_mode tamper");

        auto opening_mode_tamper = fixture.proof;
        opening_mode_tamper.public_metadata.opening_mode = "single_opening";
        require_rejects(fixture, opening_mode_tamper, "opening_mode tamper");

        auto binding_mode_tamper = fixture.proof;
        binding_mode_tamper.public_metadata.binding_mode = "wrong_binding";
        require_rejects(fixture, binding_mode_tamper, "binding_mode tamper");

        auto proof_task_tamper = fixture.proof;
        proof_task_tamper.public_metadata.proof_task = "wrong_task";
        require_rejects(fixture, proof_task_tamper, "proof_task tamper");

        auto block_order_tamper = fixture.proof;
        block_order_tamper.block_order.push_back("unknown_field");
        require_rejects(fixture, block_order_tamper, "added unknown field tamper");

        auto commitment_order_tamper = fixture.proof;
        if (commitment_order_tamper.dynamic_commitments.size() >= 2)
        {
            std::swap(commitment_order_tamper.dynamic_commitments[0], commitment_order_tamper.dynamic_commitments[1]);
            require_rejects(fixture, commitment_order_tamper, "commitment order tamper");
        }

        auto dynamic_point_tamper = fixture.proof;
        dynamic_point_tamper.dynamic_commitments.front().second.point = gatzk::crypto::g1_zero();
        require_rejects(fixture, dynamic_point_tamper, "dynamic commitment point tamper");

        auto quotient_point_tamper = fixture.proof;
        quotient_point_tamper.quotient_commitments.front().second.point = gatzk::crypto::g1_zero();
        require_rejects(fixture, quotient_point_tamper, "quotient commitment tamper");

        auto domain_value_tamper = fixture.proof;
        domain_value_tamper.domain_openings.front().second.values.front().second.front() += FieldElement::one();
        require_rejects(fixture, domain_value_tamper, "domain opening value tamper");

        auto private_payload_kind_tamper = fixture.proof;
        bool changed_private_kind = false;
        for (auto &[bundle_name, bundle] : private_payload_kind_tamper.domain_openings)
        {
            (void)bundle_name;
            for (auto &kind : bundle.value_kinds)
            {
                if (kind == gatzk::protocol::PayloadValueKind::MaskedPrivateValue)
                {
                    kind = gatzk::protocol::PayloadValueKind::PublicValue;
                    changed_private_kind = true;
                    break;
                }
            }
            if (changed_private_kind)
            {
                break;
            }
        }
        require(changed_private_kind, "fixture must contain a private typed payload value");
        require_rejects(fixture, private_payload_kind_tamper, "private payload kind tamper");

        auto external_value_tamper = fixture.proof;
        external_value_tamper.external_evaluations.front().second += FieldElement::one();
        require_rejects(fixture, external_value_tamper, "external masked value tamper");

        auto external_mask_value_tamper = fixture.proof;
        const auto external_mask_value = std::find_if(
            external_mask_value_tamper.external_evaluations.begin(),
            external_mask_value_tamper.external_evaluations.end(),
            [](const auto &evaluation)
            {
                return evaluation.first.ends_with("_mask");
            });
        require(external_mask_value != external_mask_value_tamper.external_evaluations.end(),
                "fixture must contain an external semantic-mask evaluation");
        external_mask_value->second += FieldElement::one();
        require_rejects(
            fixture, external_mask_value_tamper,
            "external semantic-mask evaluation tamper");

        auto external_mask_commitment_tamper = fixture.proof;
        const auto external_mask_commitment = std::find_if(
            external_mask_commitment_tamper.dynamic_commitments.begin(),
            external_mask_commitment_tamper.dynamic_commitments.end(),
            [](const auto &commitment)
            {
                return commitment.first.ends_with("_ExternalMask");
            });
        require(external_mask_commitment != external_mask_commitment_tamper.dynamic_commitments.end(),
                "fixture must contain an external semantic-mask commitment");
        external_mask_commitment->second.point = gatzk::crypto::g1_add(
            external_mask_commitment->second.point,
            fixture.context.kzg.g1_generator);
        require_rejects(
            fixture, external_mask_commitment_tamper,
            "external semantic-mask commitment tamper");

        auto public_logits_tamper = fixture.proof;
        public_logits_tamper.public_logits.front() += FieldElement::one();
        require_rejects(fixture, public_logits_tamper, "public logits tamper");

        auto output_digest_tamper = fixture.proof;
        require(!output_digest_tamper.public_metadata.public_output.empty(),
                "fixture must carry canonical public-output metadata");
        output_digest_tamper.public_metadata.public_output.back() =
            output_digest_tamper.public_metadata.public_output.back() == '0' ? '1' : '0';
        require_rejects(fixture, output_digest_tamper, "public output digest tamper");

        auto proof_nonce_tamper = fixture.proof;
        require(proof_nonce_tamper.public_metadata.proof_nonce.size() == 64,
                "fixture must carry a 256-bit proof nonce");
        proof_nonce_tamper.public_metadata.proof_nonce[0] =
            proof_nonce_tamper.public_metadata.proof_nonce[0] == '0' ? '1' : '0';
        require_rejects(fixture, proof_nonce_tamper, "proof nonce tamper");

        require_point_group_proof_tampers_reject(fixture);

        auto challenge_tamper = fixture.proof;
        challenge_tamper.challenges.begin()->second += FieldElement::one();
        require_rejects(fixture, challenge_tamper, "challenge tamper");
    }

} // namespace

int main()
{
    try
    {
        std::cerr
            << "running test_rotation_shift_subgroup_rejection_equivalence\n";
        test_rotation_shift_subgroup_rejection_equivalence();
        std::cerr
            << "running test_spilled_multi_point_evaluation_equivalence\n";
        test_spilled_multi_point_evaluation_equivalence();
        std::cerr
            << "running test_fh_precomputed_boundary_equivalence\n";
        test_fh_precomputed_boundary_equivalence();
        std::cerr << "running test_logup_inverse_accumulator_reuse_equivalence\n";
        test_logup_inverse_accumulator_reuse_equivalence();
        std::cerr << "running test_parallel_split_intt_equivalence\n";
        test_parallel_split_intt_equivalence();
        std::cerr
            << "running test_quotient_split_lane_materialization_equivalence\n";
        test_quotient_split_lane_materialization_equivalence();
        void test_quotient_split_lane_materialization_equivalence()
        {
            const auto profile =
                gatzk::protocol::
                    benchmark_quotient_split_lane_materialization_stage(
                        64,
                        256,
                        1);

            require(
                profile.exact,
                "split-lane quotient materialization changed logical coset order");
        }
        const bool skip_srs_tests = std::getenv("GATZK_SKIP_SRS_TESTS") != nullptr;
        if (!skip_srs_tests)
        {
            std::cerr << "running test_kzg_opening_selftest\n";
            test_kzg_opening_selftest();
            std::cerr
                << "running test_paired_streamed_point_group_equivalence\n";
            test_paired_streamed_point_group_equivalence();
            std::cerr << "running test_coefficient_taylor_shift_fixed_shape\n";
            test_coefficient_taylor_shift_fixed_shape();
            std::cerr << "running test_cross_domain_indexed_view_shift_relation\n";
            test_cross_domain_indexed_view_shift_relation();
            std::cerr << "running test_same_point_indexed_view_relation\n";
            test_same_point_indexed_view_relation();
            std::cerr << "running test_compact_eval_kzg_regression\n";
            test_compact_eval_kzg_regression();
            std::cerr << "running test_evaluation_commitment_cache_lifetime\n";
            test_evaluation_commitment_cache_lifetime();
            std::cerr << "running test_file_backed_taylor_shift_source_resolution\n";
            test_file_backed_taylor_shift_source_resolution();
            std::cerr << "running test_external_point_grouping_regression\n";
            test_external_point_grouping_regression();
        }
        else
        {
            std::cerr << "skipping SRS-backed KZG regressions; GATZK_SKIP_SRS_TESTS is set\n";
        }
        std::cerr << "running test_packed_backend_padded_tail_regression\n";
        test_packed_backend_padded_tail_regression();
        if (!skip_srs_tests)
        {
            std::cerr << "running test_msm_non_divisible_chunk_regression\n";
            test_msm_non_divisible_chunk_regression();
            const bool run_large_msm_tests =
                std::getenv("GATZK_RUN_LARGE_MSM_TESTS") != nullptr && std::getenv("GATZK_SKIP_LARGE_MSM_TESTS") == nullptr;
            if (run_large_msm_tests)
            {
                std::cerr << "running test_msm_chunked_equivalence_regression\n";
                test_msm_chunked_equivalence_regression();
            }
            else
            {
                std::cerr << "skipping opt-in large-SRS MSM regression; set GATZK_RUN_LARGE_MSM_TESTS=1 with a sufficiently large test SRS to run it\n";
            }
            std::cerr << "running test_srs_mismatch_gate\n";
            test_srs_mismatch_gate();
        }
        std::cerr << "running test_metadata_and_schema_gates\n";
        test_metadata_and_schema_gates();
        std::cerr << "running test_removed_compatibility_protocol_gate\n";
        test_removed_compatibility_protocol_gate();
        std::cerr << "running test_canonical_output_digest_contract\n";
        test_canonical_output_digest_contract();
        std::cerr << "running test_multi_graph_public_boundary_validation\n";
        test_multi_graph_public_boundary_validation();
        std::cerr << "running test_formal_shape_and_activation_boundaries\n";
        test_formal_shape_and_activation_boundaries();
        std::cerr << "running test_weighted_edge_aggregate_raw_by_destination\n";
        test_weighted_edge_aggregate_raw_by_destination();
        std::cerr << "running test_field_element_spill_io\n";
        test_field_element_spill_io();
        std::cerr << "running test_ppi_deferred_commitment_chunk_dependency_order\n";
        test_ppi_deferred_commitment_chunk_dependency_order();
        std::cerr << "running test_compiler_observation_certificate_gate\n";
        test_compiler_observation_certificate_gate();
        std::cerr << "running test_domain_opening_challenge_exclusion_registry\n";
        test_domain_opening_challenge_exclusion_registry();
        if (std::getenv("GATZK_RUN_UNIT_SECURITY_TESTS") != nullptr)
        {
            std::cerr << "running test_unit_coefficient_basis_closure\n";
            test_unit_coefficient_basis_closure();
            std::cerr << "running test_large_degree_pi_bind_codec_boundary\n";
            test_large_degree_pi_bind_codec_boundary();
        }
        else
        {
            std::cerr << "skipping unit coefficient-basis closure regression; set GATZK_RUN_UNIT_SECURITY_TESTS=1 to run it\n";
        }
        if (std::getenv("GATZK_RUN_PPI_BOUNDARY_TESTS") != nullptr)
        {
            std::cerr << "running test_real_ppi_representative_batch_boundary\n";
            test_real_ppi_representative_batch_boundary();
        }
        else
        {
            std::cerr << "skipping real PPI representative-batch boundary regression; set GATZK_RUN_PPI_BOUNDARY_TESTS=1 to run it\n";
        }
        if (std::getenv("GATZK_RUN_FULL_SECURITY_TESTS") != nullptr)
        {
            std::cerr << "running test_full_metadata_and_schema_gates\n";
            test_full_metadata_and_schema_gates();
            std::cerr << "running test_reference_consistency_and_verify\n";
            test_reference_consistency_and_verify();
            std::cerr << "running test_strict_proof_codec\n";
            test_strict_proof_codec();
            std::cerr << "running test_verifier_only_boundary\n";
            test_verifier_only_boundary();
            std::cerr << "running test_tamper_rejection_suite\n";
            test_tamper_rejection_suite();
        }
        else
        {
            std::cerr << "skipping full Cora security regression; set GATZK_RUN_FULL_SECURITY_TESTS=1 to run it\n";
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
    std::cout << "secure regression tests passed\n";
    return 0;
}
