#include "gatzk/mechanism/internal.hpp"

#include "codec.hpp"
#include "real_input.hpp"

#include <openssl/rand.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <set>
#include <tuple>
#include <optional>

#if GATZK_ENABLE_CUDA_BACKEND
#include <cuda_runtime_api.h>
#endif

#include "gatzk/algebra/vector_ops.hpp"
#include "gatzk/crypto/curve.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::mechanism
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        double elapsed_ms(Clock::time_point start, Clock::time_point stop)
        {
            return std::chrono::duration<double, std::milli>(stop - start).count();
        }

        void require(bool condition, const std::string &message)
        {
            if (!condition)
                throw std::runtime_error(message);
        }

        constexpr std::size_t kMechanismMaskRank = 8;
        constexpr unsigned kFieldBoundBits = 254;
        constexpr unsigned kComputationalAssumptionBits = 128;
        constexpr unsigned kTargetSecurityBits = 120;
        constexpr std::size_t kQuotientMaskRank = 8;
        constexpr std::uint64_t kFiatShamirQueryBudgetMax = 1U << 20U;
        constexpr unsigned kFiatShamirSecurityBits = 128;
        constexpr std::string_view kFiatShamirAssumptionProfileId = "transcript_sha256_reject_zero_v3";
        constexpr std::string_view kScalarModulusDecimal = "52435875175126190479447740508185965837690552500527637822603658699938581184513";
        constexpr std::string_view kConservativeFieldDenominator =
            "28948022309329048855892746252171976963317496166410141009864396001978282409984"; // 2^254

        struct ObservationRows
        {
            std::uint64_t field_rows = 0;
            std::uint64_t independent_group_transforms = 0;
            std::uint64_t group_rows() const { return field_rows + 1 + independent_group_transforms; }
        };

        bool starts_with(const std::string &value, std::string_view prefix)
        {
            return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
        }

        std::string json_escape_string(const std::string &value)
        {
            static constexpr char hex[] = "0123456789abcdef";
            std::string escaped;
            escaped.reserve(value.size());
            for (const unsigned char byte : value)
            {
                switch (byte)
                {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\b': escaped += "\\b"; break;
                case '\f': escaped += "\\f"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default:
                    if (byte < 0x20)
                    {
                        escaped += "\\u00";
                        escaped.push_back(hex[byte >> 4]);
                        escaped.push_back(hex[byte & 0x0f]);
                    }
                    else
                        escaped.push_back(static_cast<char>(byte));
                }
            }
            return escaped;
        }

        bool is_sha256_hex(const std::string &value)
        {
            return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c)
            {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            });
        }

        std::string uint192_decimal(std::array<std::uint64_t, 3> limbs)
        {
            if (limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0)
                return "0";
            std::string digits;
            while (limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0)
            {
                unsigned __int128 remainder = 0;
                for (std::size_t i = limbs.size(); i-- > 0;)
                {
                    const auto current = (remainder << 64U) | limbs[i];
                    limbs[i] = static_cast<std::uint64_t>(current / 10U);
                    remainder = current % 10U;
                }
                digits.push_back(static_cast<char>('0' + remainder));
            }
            std::reverse(digits.begin(), digits.end());
            return digits;
        }

        std::string subtract_small_decimal(std::string value, std::uint64_t amount)
        {
            for (std::size_t i = value.size(); i-- > 0 && amount != 0;)
            {
                const auto sub = static_cast<unsigned>(amount % 10U);
                amount /= 10U;
                auto digit = static_cast<unsigned>(value[i] - '0');
                if (digit < sub)
                {
                    digit += 10U;
                    ++amount;
                }
                value[i] = static_cast<char>('0' + (digit - sub));
            }
            if (amount != 0)
                throw std::runtime_error("decimal subtraction underflow in mechanism security ledger");
            const auto first = value.find_first_not_of('0');
            return first == std::string::npos ? "0" : value.substr(first);
        }

        bool decimal_greater(std::string_view lhs, std::string_view rhs)
        {
            return lhs.size() != rhs.size() ? lhs.size() > rhs.size() : lhs > rhs;
        }

        std::string checked_scalar_modulus_decimal()
        {
            algebra::ensure_mcl_field_ready();
            const auto modulus = mcl::Fr::getOp().mp.getStr(10);
            if (modulus != kScalarModulusDecimal)
                throw std::runtime_error("mechanism security ledger BLS12-381 scalar modulus differs from its reviewed constant");
            return modulus;
        }

        std::optional<ObservationRows> expected_observation_rows(const PublicStatement &s, const std::string &label)
        {
            if (starts_with(label, "sn/route/") || starts_with(label, "ff/route/"))
            {
                const auto slash = label.rfind('/');
                if (slash == std::string::npos)
                    return std::nullopt;
                const auto member = label.substr(slash + 1);
                if (member == "Table" || member == "Query" || member == "J_node" || member == "J_edge")
                    return ObservationRows{1, 0};
                if (member == "R_node")
                    return ObservationRows{2, 1};
                if (member == "R_edge")
                    return ObservationRows{3, 2};
                // Q_zk and R_Q are disclosed together in one point-group
                // opening. Their pair certificate accounts for two claims and
                // one aggregate opening witness at this point.
                if (member == "R_Q" || member.rfind("Q_", 0) == 0)
                    return ObservationRows{2, 0};
                // These are coefficient-basis identities/commitment transforms, not
                // separately randomized evaluation polynomials.
                if (member == "R_edge_shifted" || member == "terminal_virtual")
                    return std::nullopt;
                return std::nullopt;
            }

            if (s.mechanism == Mechanism::SharedNeighborhood)
            {
                if (label == "sn/U_edge")
                    return ObservationRows{2, s.variant == Variant::Original ? 1U : 0U};
                if (starts_with(label, "sn/V_edge/"))
                    return ObservationRows{2, s.variant == Variant::Original ? 2U : 1U};
                if (label == "sn/Sum_node")
                    return ObservationRows{1, 1};
                if (starts_with(label, "sn/Hagg_node/"))
                    return ObservationRows{1, s.variant == Variant::Original ? 2U : 2U};
                if (label == "sn/V_star_edge")
                    return ObservationRows{2, s.variant == Variant::Original ? 1U : 0U};
                if (label == "sn/Hagg_star_node")
                    return ObservationRows{1, 1};
                if (label == "sn/w_edge")
                    return ObservationRows{2, 0};
                if (label == "sn/T_node")
                    return ObservationRows{1, 1};
                if (starts_with(label, "sn/target_edge/"))
                    return ObservationRows{2, 1};
                if (starts_with(label, "sn/psq/") && label.size() >= 6 && label.compare(label.size() - 6, 6, "/state") == 0)
                    return ObservationRows{3, 0};
                if (starts_with(label, "sn/psq/") &&
                    (label.ends_with("/Q_step") || label.ends_with("/Q_group_terminal")))
                    return ObservationRows{2, 0};
                if (starts_with(label, "sn/psq/") && label.ends_with("/R_Q"))
                    return ObservationRows{2, 0};
            }
            else if (s.mechanism == Mechanism::FeatureFolding && starts_with(label, "ff/"))
            {
                // FF inventory labels are mechanism-owned.  Its masked source
                // columns are committed once and feed at most the folded point
                // plus one route tuple; the route-specific rows above enumerate
                // their direct evaluations and Taylor transforms.
                if (label.find("/Q_") != std::string::npos || label.ends_with("/R_Q") || label.ends_with("/terminal_virtual") ||
                    label.ends_with("/R_edge_shifted"))
                    return std::nullopt;
                if (starts_with(label, "ff/H_node/"))
                    return ObservationRows{1, 1};
                if (starts_with(label, "ff/H_edge/"))
                    return ObservationRows{1, 1};
                if (label == "ff/H_star_node" || label == "ff/H_star_edge")
                    return ObservationRows{1, 1};
                if (label.find("/source/") != std::string::npos || label.find("/feature/") != std::string::npos ||
                    label.find("/folded/") != std::string::npos)
                    return ObservationRows{0, 2};
            }
            return std::nullopt;
        }

        struct SecurityBudget
        {
            std::size_t route_rounds = 0;
            std::size_t psq_channels = 0;
            std::size_t route_identity_count = 0;
            std::size_t psq_identity_count = 0;
            std::size_t identity_count = 0;
            std::uint64_t route_inverse_node_degree = 0;
            std::uint64_t route_inverse_edge_degree = 0;
            std::uint64_t route_accumulator_node_degree = 0;
            std::uint64_t route_accumulator_edge_degree = 0;
            std::uint64_t psq_step_degree = 0;
            std::uint64_t psq_terminal_degree = 0;
            std::uint64_t max_quotient_degree = 0;
            std::uint64_t tuple_compression_numerator = 0;
            std::uint64_t logup_denominator_numerator = 0;
            std::uint64_t quotient_identity_numerator = 0;
            std::uint64_t terminal_crpc_numerator = 0;
            std::uint64_t opening_fold_numerator = 0;
            std::uint64_t width_fold_numerator = 0;
            std::uint64_t shared_lambda_numerator = 0;
            std::uint64_t quotient_crs_numerator = 0;
            std::uint64_t fiat_shamir_challenge_count = 0;
            std::uint64_t fiat_shamir_query_budget_max = kFiatShamirQueryBudgetMax;
            std::uint64_t kzg_point_group_games = 0;
            std::uint64_t registry_assumption_terms = 1;
            std::uint64_t total_128bit_assumption_terms = 0;
            std::uint64_t max_quotient_exclusion_degree = 0;
            std::uint64_t max_qzk_degree = 0;
            std::uint64_t global_srs_degree_bound = 0;
            std::uint64_t route_inverse_node_identity_degree = 0;
            std::uint64_t route_inverse_edge_identity_degree = 0;
            std::uint64_t route_accumulator_node_identity_degree = 0;
            std::uint64_t route_accumulator_edge_identity_degree = 0;
            std::uint64_t psq_step_identity_degree = 0;
            std::uint64_t psq_terminal_identity_degree = 0;
            std::uint64_t tuple_compression_degree_bound = 0;
            std::uint64_t max_field_identity_degree = 0;
            std::size_t required_srs_degree = 0;
            std::uint64_t max_field_excluded_roots = 0;
            std::size_t max_field_observation_rows = 0;
            std::size_t max_group_observation_rows = 0;
            std::size_t max_quotient_mask_observations = 0;
            std::vector<std::pair<std::string, ObservationRows>> observation_registry;
            std::vector<std::string> identity_inventory;
            std::vector<std::string> challenge_inventory;
            std::vector<std::string> quotient_mask_inventory;
            std::vector<std::string> observation_lineage_inventory;
            std::vector<std::string> kzg_game_inventory;
            std::string field_error_numerator;
            std::string field_unamplified_numerator;
            std::string crs_error_numerator;
            std::string scalar_modulus_decimal;
            std::string crs_actual_denominator_decimal;
            std::string field_actual_denominator_decimal;
            std::string total_error_numerator;
            std::string target_numerator_threshold;
            bool meets_target = false;
            std::string canonical;
        };

        SecurityBudget build_security_budget(const PublicStatement &s)
        {
            SecurityBudget b;
            std::vector<std::string> route_stems;
            std::vector<std::string> psq_channels;
            if (s.mechanism == Mechanism::SharedNeighborhood)
            {
                const auto channels = s.variant == Variant::Original ? std::vector<std::string>{"shared"}
                                                                      : std::vector<std::string>{"den", "agg"};
                for (const auto &channel : channels)
                {
                    route_stems.push_back("sn/route/" + channel);
                    psq_channels.push_back(channel);
                }
            }
            else if (s.variant == Variant::Original)
                route_stems.push_back("ff/route/folded");
            else
                for (std::size_t lane = 0; lane < s.width; ++lane)
                    route_stems.push_back("ff/route/coord/" + std::to_string(lane));
            b.route_rounds = route_stems.size();
            b.psq_channels = psq_channels.size();
            if (s.mechanism == Mechanism::FeatureFolding)
            {
                for (std::size_t lane = 0; lane < s.width; ++lane)
                {
                    const auto node = "ff/H_node/" + std::to_string(lane);
                    const auto edge = "ff/H_edge/" + std::to_string(lane);
                    if (s.variant == Variant::Original)
                    {
                        b.observation_lineage_inventory.push_back(node + "->ff/H_star_node:postcommit-xi-linear-commitment");
                        b.observation_lineage_inventory.push_back(edge + "->ff/H_star_edge:postcommit-xi-linear-commitment");
                    }
                    else
                    {
                        const auto stem = "ff/route/coord/" + std::to_string(lane);
                        b.observation_lineage_inventory.push_back(node + "->" + stem + "/Table:eta-tuple-identity-at-zeta");
                        b.observation_lineage_inventory.push_back(edge + "->" + stem + "/Query:eta-tuple-identity-at-zeta");
                    }
                }
                if (s.variant == Variant::Original)
                {
                    b.observation_lineage_inventory.push_back("ff/H_star_node->ff/route/folded/Table:public-node-id-plus-eta-feature;zeta-tuple-identity");
                    b.observation_lineage_inventory.push_back("ff/H_star_edge->ff/route/folded/Query:public-edge-src-plus-eta-feature;zeta-tuple-identity");
                }
            }
            else if (s.variant == Variant::Original)
            {
                b.observation_lineage_inventory.push_back("sn/U_edge->sn/w_edge:explicit-linear-commitment;PSQ-step-and-step-next");
                b.observation_lineage_inventory.push_back("sn/Sum_node->sn/T_node:explicit-linear-commitment;route-zeta");
                for (std::size_t lane = 0; lane < s.width; ++lane)
                {
                    const auto suffix = std::to_string(lane);
                    b.observation_lineage_inventory.push_back("sn/V_edge/" + suffix + "->sn/V_star_edge->sn/w_edge:postcommit-xi-and-explicit-linear-commitments;PSQ-step-and-step-next");
                    b.observation_lineage_inventory.push_back("sn/Hagg_node/" + suffix + "->sn/Hagg_star_node->sn/T_node:postcommit-xi-and-explicit-linear-commitments;route-zeta");
                }
            }
            else
            {
                b.observation_lineage_inventory.push_back("sn/U_edge->sn/psq/den:direct-weight;PSQ-step-and-step-next");
                b.observation_lineage_inventory.push_back("sn/Sum_node->sn/route/den/Table:direct-target;route-zeta-tuple-identity");
                for (std::size_t lane = 0; lane < s.width; ++lane)
                {
                    const auto suffix = std::to_string(lane);
                    b.observation_lineage_inventory.push_back("sn/V_edge/" + suffix + "->sn/V_star_edge->sn/psq/agg:postcommit-xi-and-explicit-linear-commitment;PSQ-step-and-step-next");
                    b.observation_lineage_inventory.push_back("sn/Hagg_node/" + suffix + "->sn/Hagg_star_node->sn/route/agg/Table:postcommit-xi-and-explicit-linear-commitment;route-zeta-tuple-identity");
                }
            }
            for (const auto &channel : psq_channels)
            {
                const auto route = "sn/route/" + channel;
                const auto psq = "sn/psq/" + channel;
                b.observation_lineage_inventory.push_back("sn/target_edge/" + channel + "->" + route + "/Query:eta-tuple-mask;route-zeta");
                b.observation_lineage_inventory.push_back("sn/target_edge/" + channel + "->" + psq + "/open/group_terminal:terminal-target-opening");
                b.observation_lineage_inventory.push_back(psq + "/state->" + psq + "/open/step,/open/step_next,/open/group_terminal:three-distinct-opening-points");
            }
            if (s.mechanism == Mechanism::SharedNeighborhood || s.variant == Variant::Original)
                b.challenge_inventory.push_back(s.mechanism == Mechanism::SharedNeighborhood ? "sn/width_fold_xi" : "ff/feature_fold_xi");
            if (s.mechanism == Mechanism::SharedNeighborhood && s.variant == Variant::Original)
                b.challenge_inventory.push_back("sn/shared_lambda");
            for (const auto &stem : route_stems)
            {
                b.identity_inventory.push_back(stem + "/identity/inverse_node");
                b.identity_inventory.push_back(stem + "/identity/inverse_edge");
                b.identity_inventory.push_back(stem + "/identity/accumulator_node");
                b.identity_inventory.push_back(stem + "/identity/accumulator_edge");
            }
            for (std::size_t route_index = 0; route_index < route_stems.size(); ++route_index)
            {
                const auto &stem = route_stems[route_index];
                const bool has_psq = route_index < psq_channels.size();
                const auto channel = has_psq ? psq_channels[route_index] : std::string{};
                if (has_psq)
                    b.challenge_inventory.push_back("sn/route/" + channel + "/eta");
                else
                    b.challenge_inventory.push_back(stem + "/eta");
                b.challenge_inventory.push_back(stem + "/beta");
                b.challenge_inventory.push_back(stem + "/zeta");
                b.challenge_inventory.push_back("mechanism_open_fold:v3:" + stem + "/zeta");
                b.challenge_inventory.push_back("mechanism_open_fold:v3:" + stem + "/zeta_next_node");
                b.challenge_inventory.push_back("mechanism_open_fold:v3:" + stem + "/zeta_next_edge");
                b.challenge_inventory.push_back(stem + "/terminal_y");
                b.challenge_inventory.push_back("mechanism_open_fold:v3:" + stem + "/terminal_shifted");
                b.challenge_inventory.push_back("mechanism_open_fold:v3:" + stem + "/terminal_edge");
                b.challenge_inventory.push_back("mechanism_open_fold:v3:" + stem + "/terminal_fixed");
                if (has_psq)
                {
                    const auto prefix = "sn/psq/" + channel;
                    b.identity_inventory.push_back(prefix + "/identity/step_recurrence");
                    b.identity_inventory.push_back(prefix + "/identity/group_terminal");
                    b.challenge_inventory.push_back(prefix + "/zeta");
                    b.challenge_inventory.push_back("mechanism_open_fold:v3:" + prefix + "/open/step");
                    b.challenge_inventory.push_back("mechanism_open_fold:v3:" + prefix + "/open/step_next");
                    b.challenge_inventory.push_back(prefix + "/terminal_point");
                    b.challenge_inventory.push_back("mechanism_open_fold:v3:" + prefix + "/open/group_terminal");
                }
            }
            b.route_identity_count = route_stems.size() * 4;
            b.psq_identity_count = psq_channels.size() * 2;
            b.identity_count = b.identity_inventory.size();

            const auto n = static_cast<std::uint64_t>(s.node_domain_size);
            const auto e = static_cast<std::uint64_t>(s.edge_domain_size);
            b.global_srs_degree_bound = static_cast<std::uint64_t>(s.parent_srs_degree_bound);
            if (b.global_srs_degree_bound == 0)
                throw std::runtime_error("mechanism ledger is missing the authenticated parent SRS degree bound");
            const auto checked_degree_sum = [](std::uint64_t a, std::uint64_t b)
            {
                if (b > std::numeric_limits<std::uint64_t>::max() - a)
                    throw std::runtime_error("mechanism malicious-polynomial degree bound overflows");
                return a + b;
            };
            // Bounds include rank-8 source masks, selectors, rotations, and the
            // additional Z_D*R_Q quotient masks committed in the mechanism proof.
            b.route_inverse_node_degree = 2 * n + 13;
            b.route_inverse_edge_degree = 2 * e + 13;
            b.route_accumulator_node_degree = 2 * n + 7;
            b.route_accumulator_edge_degree = 2 * e + 7;
            b.psq_step_degree = 2 * e + 7;
            b.psq_terminal_degree = 2 * e + 7;
            b.max_quotient_degree = std::max({b.route_inverse_node_degree, b.route_inverse_edge_degree,
                                               b.route_accumulator_node_degree, b.route_accumulator_edge_degree,
                                               b.psq_step_degree, b.psq_terminal_degree});

            // Soundness must cover a malicious prover's highest-degree committed
            // polynomials, not only the lower-degree polynomials produced by this
            // honest prover. The statement binds the authenticated global SRS
            // degree; selector polynomials contribute their full domain degree.
            const auto twice_global_degree = checked_degree_sum(b.global_srs_degree_bound, b.global_srs_degree_bound);
            b.route_inverse_node_identity_degree = checked_degree_sum(twice_global_degree, n);
            b.route_inverse_edge_identity_degree = checked_degree_sum(twice_global_degree, e);
            b.route_accumulator_node_identity_degree = checked_degree_sum(b.global_srs_degree_bound, 2 * n);
            b.route_accumulator_edge_identity_degree = checked_degree_sum(b.global_srs_degree_bound, e);
            b.psq_step_identity_degree = checked_degree_sum(b.global_srs_degree_bound, 2 * e);
            b.psq_terminal_identity_degree = checked_degree_sum(b.global_srs_degree_bound, e);
            b.max_field_identity_degree = std::max({b.route_inverse_node_identity_degree,
                                                     b.route_inverse_edge_identity_degree,
                                                     b.route_accumulator_node_identity_degree,
                                                     b.route_accumulator_edge_identity_degree,
                                                     b.psq_step_identity_degree,
                                                     b.psq_terminal_identity_degree,
                                                     b.global_srs_degree_bound});

            const auto add_quotient_mask = [&](const std::string &quotient, const std::string &domain,
                                               std::uint64_t domain_rows, std::uint64_t quotient_degree,
                                               const std::string &opening_group,
                                               const std::vector<std::string> &private_sources,
                                               const std::vector<std::string> &source_opening_groups)
            {
                if (domain_rows > std::numeric_limits<std::uint64_t>::max() - kQuotientMaskRank)
                    throw std::runtime_error("mechanism quotient-mask exclusion degree overflows its ledger");
                const auto exclusion_degree = domain_rows + kQuotientMaskRank;
                const auto q_mask_degree = domain_rows + kQuotientMaskRank - 1;
                const auto qzk_degree = std::max(quotient_degree, q_mask_degree);
                b.quotient_crs_numerator += exclusion_degree;
                b.max_quotient_exclusion_degree = std::max(b.max_quotient_exclusion_degree, exclusion_degree);
                b.max_qzk_degree = std::max(b.max_qzk_degree, qzk_degree);
                b.required_srs_degree = std::max(b.required_srs_degree, static_cast<std::size_t>(qzk_degree));
                if (private_sources.empty())
                    throw std::runtime_error("mechanism quotient mask has no private source dependency: " + quotient);
                std::ostringstream entry;
                entry << "quotient=" << quotient << "|mask=" << internal::quotient_mask_label(quotient) << "|domain=" << domain << ':'
                      << domain_rows << "|raw_quotient_degree_bound=" << quotient_degree << "|Q_zk_degree_bound=" << qzk_degree
                      << "|required_coefficient_count=" << (qzk_degree + 1) << "|mask_rank=" << kQuotientMaskRank
                      << "|exclusion_degree=" << exclusion_degree << "|field_observation_rows=2|group_observation_rows=3|total_rank_rows=5"
                      << "|rank_certificate=independent-csprng-coefficient-mask;observations=Q_zk-value,R_Q-value,Q_zk-commitment,R_Q-commitment,aggregate-point-group-witness;rank<8"
                      << "|opening_group=" << opening_group << "|dependency=Q_zk=Q_raw+(X^D-1)*R_Q;Q_raw=identity_residual/Z_D"
                      << "|evaluation_dependency=Q_raw(z)=Q_zk(z)-Z_D(z)*R_Q(z);identity_residual(z) reuses source values from the listed pre-existing opening groups;no additional independent field point";
                for (const auto &source : private_sources)
                    entry << "|private_source=" << source;
                for (const auto &source_group : source_opening_groups)
                    entry << "|source_opening_group=" << source_group;
                b.quotient_mask_inventory.push_back(entry.str());
                std::ostringstream lineage;
                lineage << quotient << '+' << internal::quotient_mask_label(quotient)
                        << "->aggregate_witness(" << opening_group << "):Q_raw=Q_zk-Z_D*R_Q;two-claim-values-at-opening-point;no-new-field-point";
                for (const auto &source : private_sources)
                    lineage << "|source=" << source;
                for (const auto &source_group : source_opening_groups)
                    lineage << "|source_opening_group=" << source_group;
                b.observation_lineage_inventory.push_back(lineage.str());
                b.max_quotient_mask_observations = std::max<std::size_t>(b.max_quotient_mask_observations, 5);
            };

            for (std::size_t i = 0; i < route_stems.size(); ++i)
            {
                const auto &stem = route_stems[i];
                add_quotient_mask(stem + "/Q_inv_node", "N", n, b.route_inverse_node_degree, stem + "/zeta",
                                  {stem + "/Table", stem + "/J_node"}, {stem + "/zeta"});
                add_quotient_mask(stem + "/Q_inv_edge", "edge", e, b.route_inverse_edge_degree, stem + "/zeta",
                                  {stem + "/Query", stem + "/J_edge"}, {stem + "/zeta"});
                add_quotient_mask(stem + "/Q_acc_node", "N", n, b.route_accumulator_node_degree, stem + "/zeta",
                                  {stem + "/J_node", stem + "/R_node"}, {stem + "/zeta", stem + "/zeta_next_node"});
                add_quotient_mask(stem + "/Q_acc_edge", "edge", e, b.route_accumulator_edge_degree, stem + "/zeta",
                                  {stem + "/J_edge", stem + "/R_edge"}, {stem + "/zeta", stem + "/zeta_next_edge"});
            }
            for (std::size_t i = 0; i < psq_channels.size(); ++i)
            {
                const auto &channel = psq_channels[i];
                const auto prefix = "sn/psq/" + channel;
                const auto state = prefix + "/state";
                const auto weight = channel == "shared" ? std::string("sn/w_edge")
                                    : channel == "den" ? std::string("sn/U_edge")
                                                        : std::string("sn/V_star_edge");
                const auto target = "sn/target_edge/" + channel;
                add_quotient_mask(prefix + "/Q_step", "edge", e, b.psq_step_degree,
                                  prefix + "/open/step", {state, weight}, {prefix + "/open/step", prefix + "/open/step_next"});
                add_quotient_mask(prefix + "/Q_group_terminal", "edge", e, b.psq_terminal_degree,
                                  prefix + "/open/group_terminal", {state, target}, {prefix + "/open/group_terminal"});
            }

            const auto active_tuple_count = static_cast<std::uint64_t>(s.nodes) + static_cast<std::uint64_t>(s.edges);
            b.tuple_compression_degree_bound = checked_degree_sum(checked_degree_sum(twice_global_degree, n), e);
            b.max_field_identity_degree = std::max(b.max_field_identity_degree, b.tuple_compression_degree_bound);
            b.tuple_compression_numerator = static_cast<std::uint64_t>(b.route_rounds) * b.tuple_compression_degree_bound;
            b.logup_denominator_numerator = static_cast<std::uint64_t>(b.route_rounds) * active_tuple_count;
            const auto route_degree_sum = b.route_inverse_node_identity_degree + b.route_inverse_edge_identity_degree +
                                          b.route_accumulator_node_identity_degree + b.route_accumulator_edge_identity_degree;
            const auto psq_degree_sum = b.psq_step_identity_degree + b.psq_terminal_identity_degree;
            b.quotient_identity_numerator = static_cast<std::uint64_t>(b.route_rounds) * route_degree_sum +
                                            static_cast<std::uint64_t>(b.psq_channels) * psq_degree_sum;
            b.terminal_crpc_numerator = static_cast<std::uint64_t>(b.route_rounds) * b.global_srs_degree_bound;

            // Route zeta groups contain ten base commitments, two tuple targets,
            // and four quotient-mask commitments (15-fold degree). PSQ groups
            // contain 4, 2, and 4 members (7 total fold degree per channel).
            b.opening_fold_numerator = static_cast<std::uint64_t>(b.route_rounds) * 15 +
                                       static_cast<std::uint64_t>(b.psq_channels) * 7;
            b.fiat_shamir_challenge_count = b.challenge_inventory.size();
            b.width_fold_numerator = s.mechanism == Mechanism::SharedNeighborhood || s.variant == Variant::Original
                                         ? static_cast<std::uint64_t>(s.width - 1)
                                         : 0;
            b.shared_lambda_numerator = s.mechanism == Mechanism::SharedNeighborhood && s.variant == Variant::Original ? 1 : 0;
            for (const auto &stem : route_stems)
            {
                const auto groups = internal::route_opening_group_labels(stem);
                b.kzg_game_inventory.insert(b.kzg_game_inventory.end(), groups.begin(), groups.end());
            }
            for (const auto &channel : psq_channels)
            {
                const auto prefix = "sn/psq/" + channel;
                b.kzg_game_inventory.push_back(prefix + "/open/step");
                b.kzg_game_inventory.push_back(prefix + "/open/step_next");
                b.kzg_game_inventory.push_back(prefix + "/open/group_terminal");
            }
            b.kzg_point_group_games = b.kzg_game_inventory.size();
            // One 128-bit random-oracle assumption covers Fiat-Shamir; the native
            // Q_FS bound separately amplifies algebraic challenge-query events.
            b.total_128bit_assumption_terms = b.kzg_point_group_games + 1 + b.registry_assumption_terms;

            const auto labels = s.mechanism == Mechanism::FeatureFolding ? internal::feature_folding_commitment_labels(s)
                                                                          : internal::shared_neighborhood_commitment_labels(s);
            for (const auto &label : labels)
            {
                if (const auto rows = expected_observation_rows(s, label))
                {
                    b.observation_registry.emplace_back(label, *rows);
                    b.max_field_observation_rows = std::max(b.max_field_observation_rows, static_cast<std::size_t>(rows->field_rows));
                    b.max_group_observation_rows = std::max(b.max_group_observation_rows, static_cast<std::size_t>(rows->group_rows()));
                }
            }

            const std::array<std::uint64_t, 7> field_components{b.tuple_compression_numerator, b.logup_denominator_numerator,
                                                                 b.quotient_identity_numerator, b.terminal_crpc_numerator,
                                                                 b.opening_fold_numerator, b.width_fold_numerator, b.shared_lambda_numerator};
            std::uint64_t field_unamplified_numerator = 0;
            for (const auto component : field_components)
            {
                if (component > std::numeric_limits<std::uint64_t>::max() - field_unamplified_numerator)
                    throw std::runtime_error("mechanism exact field numerator overflows its integer ledger");
                field_unamplified_numerator += component;
            }
            if (field_unamplified_numerator > std::numeric_limits<std::uint64_t>::max() / kFiatShamirQueryBudgetMax)
                throw std::runtime_error("mechanism Fiat-Shamir query-amplified numerator overflows its integer ledger");
            const auto field_numerator = field_unamplified_numerator * kFiatShamirQueryBudgetMax;
            if (b.quotient_crs_numerator > std::numeric_limits<std::uint64_t>::max() / kFiatShamirQueryBudgetMax)
                throw std::runtime_error("mechanism Fiat-Shamir query-amplified CRS numerator overflows its integer ledger");
            const auto crs_error_numerator = b.quotient_crs_numerator * kFiatShamirQueryBudgetMax;
            if (field_numerator > std::numeric_limits<std::uint64_t>::max() - crs_error_numerator)
                throw std::runtime_error("mechanism exact security numerator overflows its integer ledger");
            const auto total_field_numerator = field_numerator + crs_error_numerator;
            const auto assumption_terms = b.total_128bit_assumption_terms;
            std::array<std::uint64_t, 3> total_limbs{
                total_field_numerator,
                (assumption_terms & 3U) << 62U,
                assumption_terms >> 2U,
            };
            const std::array<std::uint64_t, 3> threshold_limbs{0, 0, 64}; // 2^(254-120) = 2^134
            const bool under_threshold = total_limbs[2] < threshold_limbs[2] ||
                                         (total_limbs[2] == threshold_limbs[2] && total_limbs[1] < threshold_limbs[1]) ||
                                         (total_limbs[2] == threshold_limbs[2] && total_limbs[1] == threshold_limbs[1] &&
                                          total_limbs[0] <= threshold_limbs[0]);
            b.field_error_numerator = std::to_string(field_numerator);
            b.field_unamplified_numerator = std::to_string(field_unamplified_numerator);
            b.crs_error_numerator = std::to_string(crs_error_numerator);
            b.scalar_modulus_decimal = checked_scalar_modulus_decimal();
            const auto max_domain = static_cast<std::uint64_t>(std::max(s.node_domain_size, s.edge_domain_size));
            b.crs_actual_denominator_decimal = subtract_small_decimal(b.scalar_modulus_decimal, max_domain + 1);
            const auto max_excluded = std::max<std::uint64_t>(b.max_quotient_degree + 1, n + e + 1);
            b.max_field_excluded_roots = max_excluded;
            b.field_actual_denominator_decimal = subtract_small_decimal(b.scalar_modulus_decimal, b.max_field_identity_degree);
            const bool conservative_denominator_valid = decimal_greater(b.crs_actual_denominator_decimal, kConservativeFieldDenominator) &&
                                                        decimal_greater(b.field_actual_denominator_decimal, kConservativeFieldDenominator);
            b.total_error_numerator = uint192_decimal(total_limbs);
            b.target_numerator_threshold = uint192_decimal(threshold_limbs);
            b.meets_target = under_threshold && conservative_denominator_valid &&
                             kMechanismMaskRank >= b.max_field_observation_rows &&
                             kMechanismMaskRank >= b.max_group_observation_rows &&
                             kQuotientMaskRank > b.max_quotient_mask_observations;

            std::ostringstream canonical;
            canonical << "mechanism-security-ledger-v7|bound_kind=conservative_exact|target_bits=" << kTargetSecurityBits << "|field_bound_bits=" << kFieldBoundBits
                      << "|fiat_shamir_assumption_profile_id=" << kFiatShamirAssumptionProfileId << "|fiat_shamir_security_bits=" << kFiatShamirSecurityBits
                      << "|fiat_shamir_query_budget_max=" << kFiatShamirQueryBudgetMax
                      << "|mechanism=" << to_string(s.mechanism) << "|variant=" << to_string(s.variant) << "|nodes=" << s.nodes
                      << "|edges=" << s.edges << "|width=" << s.width << "|graphs=" << s.graphs
                      << "|srs_id=" << s.srs_id << "|parent_manifest=" << s.srs_manifest_sha256
                      << "|parent_schema=" << s.parent_protocol_schema_id
                      << "|computational_assumption_bits=" << kComputationalAssumptionBits << "|global_srs_degree_bound=" << b.global_srs_degree_bound
                      << "|kzg_point_group_games=" << b.kzg_point_group_games
                      << "|registry_assumption_terms=" << b.registry_assumption_terms << "|ro_assumption_terms=1"
                      << "|fiat_shamir_challenge_count=" << b.fiat_shamir_challenge_count
                      << "|field_modulus=" << b.scalar_modulus_decimal << "|field_denominator_lower_bound=2^" << kFieldBoundBits
                      << "|crs_actual_denominator=" << b.crs_actual_denominator_decimal
                      << "|field_actual_denominator=" << b.field_actual_denominator_decimal
                      << "|max_field_excluded_roots=" << b.max_field_excluded_roots << "|max_field_identity_degree=" << b.max_field_identity_degree
                      << "|tuple_compression_degree_bound=" << b.tuple_compression_degree_bound
                      << "|route_inverse_node_identity_degree=" << b.route_inverse_node_identity_degree
                      << "|route_inverse_edge_identity_degree=" << b.route_inverse_edge_identity_degree
                      << "|route_accumulator_node_identity_degree=" << b.route_accumulator_node_identity_degree
                      << "|route_accumulator_edge_identity_degree=" << b.route_accumulator_edge_identity_degree
                      << "|psq_step_identity_degree=" << b.psq_step_identity_degree
                      << "|psq_terminal_identity_degree=" << b.psq_terminal_identity_degree
                      << "|route_rounds=" << b.route_rounds << "|psq_channels=" << b.psq_channels
                      << "|route_identity_count=" << b.route_identity_count << "|psq_identity_count=" << b.psq_identity_count
                      << "|identity_count=" << b.identity_count << "|route_inverse_node_degree=" << b.route_inverse_node_degree
                      << "|route_inverse_edge_degree=" << b.route_inverse_edge_degree << "|route_accumulator_node_degree="
                      << b.route_accumulator_node_degree << "|route_accumulator_edge_degree=" << b.route_accumulator_edge_degree
                      << "|psq_step_degree=" << b.psq_step_degree << "|psq_terminal_degree=" << b.psq_terminal_degree
                      << "|max_quotient_degree=" << b.max_quotient_degree << "|tuple_compression_numerator=" << b.tuple_compression_numerator
                      << "|logup_denominator_numerator=" << b.logup_denominator_numerator
                      << "|quotient_identity_numerator=" << b.quotient_identity_numerator
                      << "|terminal_crpc_numerator=" << b.terminal_crpc_numerator << "|opening_fold_numerator=" << b.opening_fold_numerator
                      << "|width_fold_numerator=" << b.width_fold_numerator << "|shared_lambda_numerator=" << b.shared_lambda_numerator
                      << "|quotient_crs_numerator=" << b.quotient_crs_numerator << "|crs_error_numerator=" << b.crs_error_numerator
                      << "|field_error_numerator=" << b.field_error_numerator
                      << "|field_unamplified_numerator=" << b.field_unamplified_numerator
                      << "|total_error_numerator=" << b.total_error_numerator << "|target_numerator_threshold=" << b.target_numerator_threshold
                      << "|observation_model=field-distinct-opening-points;group-rows=field-rows+source-commitment+independent-derived-commitment-transforms"
                      << "|max_field_observation_rows=" << b.max_field_observation_rows << "|max_group_observation_rows="
                      << b.max_group_observation_rows << "|max_quotient_mask_observations=" << b.max_quotient_mask_observations
                      << "|chosen_zero_domain_mask_rank=" << kMechanismMaskRank << "|quotient_mask_rank=" << kQuotientMaskRank
                      << "|required_srs_degree=" << b.required_srs_degree << "|max_qzk_degree=" << b.max_qzk_degree;
            for (const auto &identity : b.identity_inventory)
                canonical << "|identity=" << identity;
            for (const auto &challenge : b.challenge_inventory)
                canonical << "|challenge=" << challenge;
            for (const auto &[label, rows] : b.observation_registry)
                canonical << "|observation=" << label << ',' << rows.field_rows << ',' << rows.group_rows();
            for (const auto &certificate : b.quotient_mask_inventory)
                canonical << "|quotient_mask_certificate=" << certificate;
            for (const auto &lineage : b.observation_lineage_inventory)
                canonical << "|observation_lineage=" << lineage;
            for (const auto &group : b.kzg_game_inventory)
                canonical << "|kzg_point_group_game=" << group;
            canonical << "|node_domain=" << s.node_domain << ':' << s.node_domain_size << "|edge_domain=" << s.edge_domain << ':' << s.edge_domain_size;
            for (const auto &edge : s.topology)
                canonical << "|edge=" << edge.graph << ',' << edge.src << ',' << edge.dst;
            for (const auto value : s.node_ptr)
                canonical << "|node_ptr=" << value;
            for (const auto value : s.edge_ptr)
                canonical << "|edge_ptr=" << value;
            b.canonical = canonical.str();
            return b;
        }

        std::vector<std::uint8_t> read_witness_seed(const std::filesystem::path &path)
        {
            const auto text = internal::read_text_file(path, 256);
            std::string hex;
            for (const auto ch : text)
                if (ch != '\n' && ch != '\r' && ch != ' ' && ch != '\t')
                    hex.push_back(ch);
            if (hex.size() != 64)
                throw std::runtime_error("witness seed file must contain exactly 64 hexadecimal characters");
            auto nibble = [](char c) -> std::uint8_t
            {
                if (c >= '0' && c <= '9')
                    return static_cast<std::uint8_t>(c - '0');
                if (c >= 'a' && c <= 'f')
                    return static_cast<std::uint8_t>(c - 'a' + 10);
                if (c >= 'A' && c <= 'F')
                    return static_cast<std::uint8_t>(c - 'A' + 10);
                throw std::runtime_error("witness seed file contains non-hexadecimal data");
            };
            std::vector<std::uint8_t> result(32);
            for (std::size_t i = 0; i < result.size(); ++i)
                result[i] = static_cast<std::uint8_t>((nibble(hex[2 * i]) << 4U) | nibble(hex[2 * i + 1]));
            return result;
        }

        std::uint64_t splitmix64(std::uint64_t &state)
        {
            std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
            value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
            return value ^ (value >> 31U);
        }

        void append_pair_field(std::string &material, const algebra::FieldElement &value)
        {
            std::array<std::uint8_t, 32> bytes{};
            value.write_little_endian(bytes.data(), bytes.size());
            material.append(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        }

        void append_pair_column(std::string &material, const std::string &label,
                                const std::vector<algebra::FieldElement> &values)
        {
            material += std::to_string(label.size()) + ":" + label + ":" + std::to_string(values.size()) + ":";
            for (const auto &value : values)
                append_pair_field(material, value);
        }

        PublicStatement generate_statement(const RunOptions &options, const crypto::KZGKeyPair &key, const std::vector<std::uint8_t> &secret)
        {
            if (options.graphs == 0 || options.graphs > options.nodes || options.nodes < options.graphs || options.edges < options.nodes ||
                options.width == 0 || options.width > 32 || options.nodes >= key.manifest.domain_sizes.at("N") ||
                options.edges >= key.manifest.domain_sizes.at("edge"))
                throw std::runtime_error("mechanism workload is outside the authenticated node/edge domains or legal graph shape");
            if (options.nodes > std::numeric_limits<std::uint32_t>::max() || options.edges > std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("mechanism workload exceeds the supported topology index width");

            PublicStatement statement;
            statement.dataset_id = "synthetic";
            statement.mechanism = options.mechanism;
            statement.variant = options.variant;
            statement.nodes = options.nodes;
            statement.edges = options.edges;
            statement.width = options.width;
            statement.graphs = options.graphs;
            statement.seed = options.seed;
            statement.srs_id = key.manifest.srs_id;
            statement.srs_manifest_sha256 = key.manifest.manifest_hash;
            statement.parent_protocol_schema_id = key.manifest.protocol_schema_id;
            statement.parent_srs_degree_bound = static_cast<std::size_t>(key.manifest.kzg_assumption_degree_bound);
            statement.node_domain = "N";
            statement.node_domain_size = key.manifest.domain_sizes.at("N");
            statement.edge_domain = "edge";
            statement.edge_domain_size = key.manifest.domain_sizes.at("edge");

            const auto node_base = options.nodes / options.graphs;
            const auto node_rem = options.nodes % options.graphs;
            statement.node_ptr.push_back(0);
            for (std::size_t graph = 0; graph < options.graphs; ++graph)
                statement.node_ptr.push_back(statement.node_ptr.back() + node_base + (graph < node_rem ? 1 : 0));

            const auto extras = options.edges - options.nodes;
            const auto extra_base = extras / options.graphs;
            const auto extra_rem = extras % options.graphs;
            statement.edge_ptr.push_back(0);
            for (std::size_t graph = 0; graph < options.graphs; ++graph)
            {
                const auto node_count = statement.node_ptr[graph + 1] - statement.node_ptr[graph];
                statement.edge_ptr.push_back(statement.edge_ptr.back() + node_count + extra_base + (graph < extra_rem ? 1 : 0));
            }

            std::uint64_t rng_state = options.seed;
            statement.topology.reserve(options.edges);
            for (std::size_t graph = 0; graph < options.graphs; ++graph)
            {
                const auto node_begin = statement.node_ptr[graph];
                const auto node_end = statement.node_ptr[graph + 1];
                for (auto node = node_begin; node < node_end; ++node)
                    statement.topology.push_back(Edge{graph, node, node});
                const auto graph_extra = extra_base + (graph < extra_rem ? 1 : 0);
                const auto graph_nodes = node_end - node_begin;
                for (std::size_t i = 0; i < graph_extra; ++i)
                {
                    const auto src = node_begin + splitmix64(rng_state) % graph_nodes;
                    const auto dst = node_begin + splitmix64(rng_state) % graph_nodes;
                    statement.topology.push_back(Edge{graph, src, dst});
                }
            }
            std::sort(statement.topology.begin(), statement.topology.end(), [](const Edge &lhs, const Edge &rhs)
                      { return std::tie(lhs.graph, lhs.dst, lhs.src) < std::tie(rhs.graph, rhs.dst, rhs.src); });

            std::string pair_material = "zkgat-mechanism-pair-input-v2|" + to_string(options.mechanism) + "|" + std::to_string(options.nodes) + "|" +
                                        std::to_string(options.edges) + "|" + std::to_string(options.width) + "|" + std::to_string(options.graphs) + "|" +
                                        std::to_string(options.seed) + "|";
            for (const auto value : statement.node_ptr)
                pair_material += "np=" + std::to_string(value) + ";";
            for (const auto value : statement.edge_ptr)
                pair_material += "ep=" + std::to_string(value) + ";";
            for (const auto &edge : statement.topology)
                pair_material += std::to_string(edge.graph) + "," + std::to_string(edge.src) + "," + std::to_string(edge.dst) + ";";
            if (options.mechanism == Mechanism::SharedNeighborhood)
            {
                append_pair_column(pair_material, "sn/U_edge", internal::derive_witness_fields(secret, "sn/U_edge", options.edges));
                for (std::size_t lane = 0; lane < options.width; ++lane)
                {
                    const auto label = "sn/V_edge/" + std::to_string(lane);
                    append_pair_column(pair_material, label, internal::derive_witness_fields(secret, label, options.edges));
                }
            }
            else
            {
                for (std::size_t lane = 0; lane < options.width; ++lane)
                {
                    const auto node_label = "ff/H_node/" + std::to_string(lane);
                    const auto node_values = internal::derive_witness_fields(secret, node_label, options.nodes);
                    std::vector<algebra::FieldElement> edge_values;
                    edge_values.reserve(options.edges);
                    for (const auto &edge : statement.topology)
                        edge_values.push_back(node_values.at(static_cast<std::size_t>(edge.src)));
                    append_pair_column(pair_material, node_label, node_values);
                    append_pair_column(pair_material, "ff/H_edge/" + std::to_string(lane), edge_values);
                }
            }
            statement.pair_input_commitment_sha256 = internal::sha256_hex(pair_material);
            statement.security_budget_ledger_sha256 = internal::compute_security_budget_ledger_sha256(statement);
            statement.mechanism_schema_id = internal::compute_mechanism_schema_id(statement);
            return statement;
        }

        PublicStatement generate_real_statement(const RunOptions &options, const crypto::KZGKeyPair &key,
                                                const internal::RealMechanismInput &input)
        {
            const auto &w = input.witness;
            require(input.dataset == "cora" || input.dataset == "citeseer" || input.dataset == "ppi",
                    "real mechanism input dataset is not registered for formal experiments");
            const auto expected_n = input.dataset == "ppi" ? 65536U : 4096U;
            const auto expected_e = input.dataset == "ppi" ? 8388608U : 32768U;
            require(key.manifest.domain_sizes.at("N") == expected_n && key.manifest.domain_sizes.at("edge") == expected_e,
                    "authenticated SRS domains do not match the real dataset catalog");
            require((options.mechanism == Mechanism::FeatureFolding &&
                     (options.width == 1 || options.width == 2 || options.width == 4 || options.width == 8)) ||
                        (options.mechanism == Mechanism::SharedNeighborhood && options.width == 8),
                    "real mechanism width must match a registered feature-folding prefix or eight-lane shared neighborhood");
            require(w.width == 8 && w.node_count == options.nodes && w.edge_count == options.edges &&
                        w.node_ptr.size() == options.graphs + 1,
                    "real mechanism workload flags disagree with the prepared first-head witness");
            require(input.input_preprocess_id == "row_normalize" && input.model_arch_id.size() != 0 && input.model_param_id.size() != 0 &&
                        input.quant_cfg_id.size() != 0 && input.static_table_id.size() != 0,
                    "real mechanism input has incomplete formal model identities");
            require(is_sha256_hex(input.config_sha256) && is_sha256_hex(input.checkpoint_bundle_sha256) &&
                        is_sha256_hex(input.source_manifest_sha256) && is_sha256_hex(input.bundle_sha256) &&
                        is_sha256_hex(input.metadata_sha256) && is_sha256_hex(input.witness_sha256),
                    "real mechanism input provenance digest is missing or malformed");
            require(w.node_ptr.size() == w.edge_ptr.size() && w.node_ptr.front() == 0 && w.node_ptr.back() == w.node_count &&
                        w.edge_ptr.front() == 0 && w.edge_ptr.back() == w.edge_count,
                    "real mechanism input graph pointer endpoints do not match its workload");

            PublicStatement statement;
            statement.dataset_id = input.dataset;
            statement.input_preprocess_id = input.input_preprocess_id;
            statement.real_input_bundle_sha256 = input.bundle_sha256;
            statement.real_input_witness_sha256 = input.witness_sha256;
            statement.real_input_metadata_sha256 = input.metadata_sha256;
            statement.checkpoint_bundle_sha256 = input.checkpoint_bundle_sha256;
            statement.input_source_manifest_sha256 = input.source_manifest_sha256;
            statement.source_layer = 0;
            statement.source_head = 0;
            statement.formal_head_width = 8;
            statement.mechanism = options.mechanism;
            statement.variant = options.variant;
            statement.nodes = w.node_count;
            statement.edges = w.edge_count;
            statement.width = options.width;
            statement.graphs = w.node_ptr.size() - 1;
            statement.seed = options.seed;
            statement.node_ptr.assign(w.node_ptr.begin(), w.node_ptr.end());
            statement.edge_ptr.assign(w.edge_ptr.begin(), w.edge_ptr.end());
            statement.topology.reserve(w.edge_count);
            for (std::size_t graph = 0; graph < statement.graphs; ++graph)
                for (std::size_t edge = statement.edge_ptr[graph]; edge < statement.edge_ptr[graph + 1]; ++edge)
                    statement.topology.push_back(Edge{graph, w.edge_src_indices[edge], w.edge_dst_indices[edge]});
            statement.srs_id = key.manifest.srs_id;
            statement.srs_manifest_sha256 = key.manifest.manifest_hash;
            statement.parent_protocol_schema_id = key.manifest.protocol_schema_id;
            statement.parent_srs_degree_bound = static_cast<std::size_t>(key.manifest.kzg_assumption_degree_bound);
            statement.node_domain = "N";
            statement.node_domain_size = key.manifest.domain_sizes.at("N");
            statement.edge_domain = "edge";
            statement.edge_domain_size = key.manifest.domain_sizes.at("edge");
            statement.pair_input_commitment_sha256 = internal::real_mechanism_relation_digest(input, options.mechanism, options.width);
            statement.security_budget_ledger_sha256 = internal::compute_security_budget_ledger_sha256(statement);
            statement.mechanism_schema_id = internal::compute_mechanism_schema_id(statement);
            return statement;
        }

        void validate_statement(const PublicStatement &s, const crypto::KZGKeyPair &key)
        {
            require(s.nodes > 0 && s.edges >= s.nodes && s.graphs > 0 && s.graphs <= s.nodes && s.width > 0 && s.width <= 32,
                    "mechanism statement has an illegal workload shape");
            require(s.srs_id == key.manifest.srs_id && s.srs_manifest_sha256 == key.manifest.manifest_hash &&
                        s.parent_protocol_schema_id == key.manifest.protocol_schema_id &&
                        s.parent_srs_degree_bound == key.manifest.kzg_assumption_degree_bound,
                    "mechanism statement is not bound to the loaded authenticated SRS manifest");
            const auto budget = build_security_budget(s);
            require(budget.meets_target, "mechanism security ledger does not meet its exact 120-bit bound");
            require(key.manifest.kzg_assumption_profile_id == "shifted_kzg_divisibility" &&
                        key.manifest.kzg_assumption_security_bits >= kComputationalAssumptionBits &&
                        key.manifest.kzg_assumption_degree_bound >= budget.required_srs_degree &&
                        key.pk.degree_bound >= budget.required_srs_degree && key.vk.degree_bound >= budget.required_srs_degree,
                    "authenticated SRS KZG profile, security level, or degree does not cover this mechanism schema");
            require(s.node_domain == "N" && s.edge_domain == "edge" && s.node_domain_size == key.manifest.domain_sizes.at("N") &&
                        s.edge_domain_size == key.manifest.domain_sizes.at("edge"),
                    "mechanism statement domain catalog differs from authenticated SRS domains");
            if (s.dataset_id == "synthetic")
            {
                require(s.input_preprocess_id.empty() && s.real_input_bundle_sha256.empty() && s.real_input_metadata_sha256.empty() &&
                            s.real_input_witness_sha256.empty() && s.checkpoint_bundle_sha256.empty() && s.input_source_manifest_sha256.empty() && s.formal_head_width == 0,
                        "synthetic mechanism statement unexpectedly carries real-input provenance");
                require(s.node_domain_size == 4096 && s.edge_domain_size == 32768,
                        "synthetic mechanism statement is restricted to its legacy native test domain");
            }
            else
            {
                const auto expected_n = s.dataset_id == "ppi" ? 65536U : 4096U;
                const auto expected_e = s.dataset_id == "ppi" ? 8388608U : 32768U;
                require((s.dataset_id == "cora" || s.dataset_id == "citeseer" || s.dataset_id == "ppi") &&
                            s.node_domain_size == expected_n && s.edge_domain_size == expected_e && s.input_preprocess_id == "row_normalize" &&
                            s.source_layer == 0 && s.source_head == 0 && s.formal_head_width == 8 && s.width <= 8 &&
                            (s.mechanism != Mechanism::SharedNeighborhood || s.width == 8) &&
                            (s.mechanism != Mechanism::FeatureFolding || s.width == 1 || s.width == 2 || s.width == 4 || s.width == 8) &&
                            is_sha256_hex(s.real_input_bundle_sha256) && is_sha256_hex(s.real_input_witness_sha256) &&
                            is_sha256_hex(s.real_input_metadata_sha256) && is_sha256_hex(s.checkpoint_bundle_sha256) &&
                            is_sha256_hex(s.input_source_manifest_sha256),
                        "real mechanism statement has invalid dataset, source-head, width, or provenance identities");
            }
            require(s.nodes < s.node_domain_size && s.edges < s.edge_domain_size, "mechanism workload leaves no route terminal row");
            require(s.node_ptr.size() == s.graphs + 1 && s.edge_ptr.size() == s.graphs + 1 && s.topology.size() == s.edges,
                    "mechanism statement pointer/topology shape mismatch");
            require(s.node_ptr.front() == 0 && s.edge_ptr.front() == 0 && s.node_ptr.back() == s.nodes && s.edge_ptr.back() == s.edges,
                    "mechanism statement pointer endpoints are invalid");
            for (std::size_t graph = 0; graph < s.graphs; ++graph)
            {
                require(s.node_ptr[graph] < s.node_ptr[graph + 1] && s.edge_ptr[graph] < s.edge_ptr[graph + 1],
                        "each mechanism graph must have at least one node and edge");
                require(s.edge_ptr[graph + 1] - s.edge_ptr[graph] >= s.node_ptr[graph + 1] - s.node_ptr[graph],
                        "each mechanism graph must include at least one incoming edge per node");
                for (std::size_t edge_index = s.edge_ptr[graph]; edge_index < s.edge_ptr[graph + 1]; ++edge_index)
                    require(s.topology[edge_index].graph == graph,
                            "mechanism edge_ptr slice does not match the graph identity of its topology rows");
            }
            std::vector<bool> self_edge(s.nodes, false);
            std::uint64_t prior_graph = 0, prior_dst = 0;
            std::set<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>> unique_edges;
            for (std::size_t i = 0; i < s.topology.size(); ++i)
            {
                const auto &e = s.topology[i];
                require(e.graph < s.graphs && e.src >= s.node_ptr[e.graph] && e.src < s.node_ptr[e.graph + 1] && e.dst >= s.node_ptr[e.graph] &&
                            e.dst < s.node_ptr[e.graph + 1],
                        "mechanism topology contains a cross-graph or out-of-range edge");
                if (i != 0)
                    require(std::tie(prior_graph, prior_dst) <= std::tie(e.graph, e.dst),
                            "mechanism topology is not in canonical graph/destination group order");
                require(unique_edges.emplace(e.graph, e.src, e.dst).second, "mechanism topology contains a duplicate edge");
                prior_graph = e.graph;
                prior_dst = e.dst;
                if (e.src == e.dst)
                    self_edge[e.src] = true;
            }
            require(std::all_of(self_edge.begin(), self_edge.end(), [](bool value) { return value; }),
                    "mechanism topology must provide a self-edge for every node");
            require(s.security_budget_ledger_sha256 == internal::compute_security_budget_ledger_sha256(s) &&
                        s.mechanism_schema_id == internal::compute_mechanism_schema_id(s),
                    "mechanism statement schema/security ledger digest mismatch");
        }

        crypto::KZGKeyPair load_mechanism_key(const std::filesystem::path &srs_path, std::size_t node_domain_size = 4096,
                                              std::size_t edge_domain_size = 32768, const std::string &expected_srs_id = {})
        {
            require(edge_domain_size <= (std::numeric_limits<std::size_t>::max() - 14) / 2,
                    "mechanism registered edge domain overflows the trusted coefficient prefix");
            const auto required_coefficients = 2 * edge_domain_size + 14;
            return crypto::KZG::load_keys_for_domains(srs_path.string(), "trusted_srs", expected_srs_id,
                                                      {{"N", node_domain_size}, {"edge", edge_domain_size}},
                                                      required_coefficients);
        }

        std::string registered_srs_id(const std::string &dataset)
        {
            if (dataset == "cora")
                return "cora_formal_srs";
            if (dataset == "citeseer")
                return "citeseer_formal_srs";
            if (dataset == "ppi")
                return "ppi_formal_srs";
            throw std::runtime_error("real mechanism input does not have a registered trusted SRS identity: " + dataset);
        }

        std::pair<std::size_t, std::size_t> statement_domain_catalog_prefix(const std::string &bytes)
        {
            constexpr std::string_view marker = ",\"domains\":{\"node\":\"N\",\"node_size\":";
            const auto start = bytes.find(marker);
            if (start == std::string::npos)
                throw std::runtime_error("mechanism statement has no canonical registered-domain prefix");
            auto parse_number = [&](std::size_t &offset, char terminator)
            {
                const auto end = bytes.find(terminator, offset);
                if (end == std::string::npos || end == offset)
                    throw std::runtime_error("mechanism statement registered-domain prefix is malformed");
                std::uint64_t value = 0;
                const auto parsed = std::from_chars(bytes.data() + offset, bytes.data() + end, value, 10);
                if (parsed.ec != std::errc{} || parsed.ptr != bytes.data() + end || value > std::numeric_limits<std::size_t>::max())
                    throw std::runtime_error("mechanism statement registered-domain size is invalid");
                offset = end + 1;
                return static_cast<std::size_t>(value);
            };
            auto offset = start + marker.size();
            const auto nodes = parse_number(offset, ',');
            constexpr std::string_view edge_marker = "\"edge\":\"edge\",\"edge_size\":";
            if (bytes.compare(offset, edge_marker.size(), edge_marker) != 0)
                throw std::runtime_error("mechanism statement registered edge-domain prefix is malformed");
            offset += edge_marker.size();
            const auto edges = parse_number(offset, '}');
            const bool cora = nodes == 4096 && edges == 32768;
            const bool ppi = nodes == 65536 && edges == 8388608;
            if (!cora && !ppi)
                throw std::runtime_error("mechanism statement requests a domain catalog outside the trusted registry");
            return {nodes, edges};
        }

        crypto::Transcript mechanism_transcript(const PublicStatement &statement, const MechanismProof &proof)
        {
            crypto::Transcript transcript("zkgat-independent-mechanism-proof-v3");
            transcript.absorb_text("statement_sha256", proof.statement_sha256);
            transcript.absorb_text("mechanism", to_string(statement.mechanism));
            transcript.absorb_text("variant", to_string(statement.variant));
            transcript.absorb_text("parent_srs_id", statement.srs_id);
            transcript.absorb_text("parent_manifest_sha256", statement.srs_manifest_sha256);
            transcript.absorb_text("parent_protocol_schema_id", statement.parent_protocol_schema_id);
            transcript.absorb_text("mechanism_schema_id", statement.mechanism_schema_id);
            transcript.absorb_text("security_budget_ledger_sha256", statement.security_budget_ledger_sha256);
            transcript.absorb_scalar("nodes", algebra::FieldElement(statement.nodes));
            transcript.absorb_scalar("edges", algebra::FieldElement(statement.edges));
            transcript.absorb_scalar("width", algebra::FieldElement(statement.width));
            transcript.absorb_scalar("graphs", algebra::FieldElement(statement.graphs));
            for (std::size_t i = 0; i < statement.node_ptr.size(); ++i)
                transcript.absorb_scalar("node_ptr:" + std::to_string(i), algebra::FieldElement(statement.node_ptr[i]));
            for (std::size_t i = 0; i < statement.edge_ptr.size(); ++i)
                transcript.absorb_scalar("edge_ptr:" + std::to_string(i), algebra::FieldElement(statement.edge_ptr[i]));
            for (std::size_t i = 0; i < statement.topology.size(); ++i)
            {
                transcript.absorb_scalar("edge_graph:" + std::to_string(i), algebra::FieldElement(statement.topology[i].graph));
                transcript.absorb_scalar("edge_src:" + std::to_string(i), algebra::FieldElement(statement.topology[i].src));
                transcript.absorb_scalar("edge_dst:" + std::to_string(i), algebra::FieldElement(statement.topology[i].dst));
            }
            return transcript;
        }

        std::vector<std::string> expected_commitments(const PublicStatement &statement)
        {
            if (statement.mechanism == Mechanism::FeatureFolding)
                return internal::feature_folding_commitment_labels(statement);
            return internal::shared_neighborhood_commitment_labels(statement);
        }
        std::vector<std::string> expected_groups(const PublicStatement &statement)
        {
            if (statement.mechanism == Mechanism::FeatureFolding)
                return internal::feature_folding_opening_group_labels(statement);
            return internal::shared_neighborhood_opening_group_labels(statement);
        }

        void validate_proof_inventory(const PublicStatement &statement, const MechanismProof &proof)
        {
            const auto expected_c = expected_commitments(statement);
            const auto expected_g = expected_groups(statement);
            require(proof.commitments.size() == expected_c.size(), "mechanism proof omitted or added commitment objects");
            require(proof.opening_groups.size() == expected_g.size(), "mechanism proof omitted or added point-opening groups");
            for (std::size_t i = 0; i < expected_c.size(); ++i)
                require(proof.commitments[i].label == expected_c[i], "mechanism proof commitment schema/order mismatch");
            for (std::size_t i = 0; i < expected_g.size(); ++i)
                require(proof.opening_groups[i].label == expected_g[i], "mechanism proof opening-group schema/order mismatch");
        }

        void configure_backend(Backend backend, int device, bool verifier)
        {
            auto options = util::route2_options();
            options.cuda_field_hotspots = backend == Backend::Cuda && !verifier;
            options.cuda_trace_hotspots = false;
            options.cuda_force_ntt_for_mechanism = backend == Backend::Cuda && !verifier;
            util::set_route2_options(options);
            if (backend == Backend::Cpu || verifier)
                return;
#if GATZK_ENABLE_CUDA_BACKEND
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count || cudaSetDevice(device) != cudaSuccess)
                throw std::runtime_error("requested CUDA device is unavailable");
#else
            (void)device;
            throw std::runtime_error("CUDA mechanism proving requested from a CPU-only build");
#endif
        }

        std::string cuda_device_name(int device)
        {
#if GATZK_ENABLE_CUDA_BACKEND
            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
                return "unknown";
            return properties.name;
#else
            (void)device;
            return "unavailable";
#endif
        }

        void initialize_proof_metadata(const PublicStatement &s, MechanismProof &p)
        {
            p.mechanism = s.mechanism;
            p.variant = s.variant;
            p.statement_sha256 = internal::sha256_hex(s.canonical_json());
            p.srs_id = s.srs_id;
            p.srs_manifest_sha256 = s.srs_manifest_sha256;
            p.parent_protocol_schema_id = s.parent_protocol_schema_id;
            p.mechanism_schema_id = s.mechanism_schema_id;
            p.security_budget_ledger_sha256 = s.security_budget_ledger_sha256;
        }
    }

    namespace internal
    {
        std::vector<FieldElement> sample_zero_domain_blinder(const std::vector<std::uint8_t> &, const std::string &label, std::size_t count)
        {
            if (count == 0 || count > 64)
                throw std::runtime_error("mechanism zero-domain mask rank is outside the reviewed bound: " + label);
            std::vector<FieldElement> output(count, FieldElement::zero());
            std::array<std::uint8_t, 64> random_bytes{};
            bool any_nonzero = false;
            for (auto &value : output)
            {
                if (RAND_bytes(random_bytes.data(), static_cast<int>(random_bytes.size())) != 1)
                    throw std::runtime_error("CSPRNG failed while sampling mechanism zero-domain masks");
                value = FieldElement::from_little_endian_mod(random_bytes.data(), random_bytes.size());
                any_nonzero = any_nonzero || !value.is_zero();
            }
            if (!any_nonzero)
                throw std::runtime_error("CSPRNG produced an all-zero mechanism mask: " + label);
            return output;
        }

        std::vector<FieldElement> derive_witness_fields(const std::vector<std::uint8_t> &seed, const std::string &label, std::size_t count)
        {
            std::vector<FieldElement> output;
            output.reserve(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                std::string material = "zkgat-mechanism-witness-v1|" + label + "|" + std::to_string(i) + "|";
                material.append(reinterpret_cast<const char *>(seed.data()), seed.size());
                std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
                SHA256(reinterpret_cast<const unsigned char *>(material.data()), material.size(), digest.data());
                output.push_back(FieldElement::from_little_endian_mod(digest.data(), digest.size()));
            }
            return output;
        }

        Polynomial make_masked_evaluation_polynomial(const std::string &label, const std::vector<FieldElement> &values,
                                                     const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                                                     const std::vector<FieldElement> &blinder)
        {
            if (domain == nullptr || values.size() != domain->size || blinder.size() != kMechanismMaskRank)
                throw std::runtime_error("mechanism masked polynomial must cover its authenticated physical domain");
            auto polynomial = Polynomial::from_evaluations(label, values, domain);
            if (!blinder.empty())
                polynomial.set_zero_domain_blinder(blinder);
            return polynomial;
        }

        void record_commitment(ProofBuildContext &context, const std::string &label, Commitment commitment, bool absorb)
        {
            if (label.empty() || context.objects.contains(label))
                throw std::runtime_error("mechanism commitment label is empty or duplicated: " + label);
            if (commitment.opening_polynomial != nullptr &&
                commitment.opening_polynomial->basis == algebra::PolynomialBasis::Evaluation)
            {
                const auto &polynomial = *commitment.opening_polynomial;
                const auto observations = expected_observation_rows(context.statement, label);
                if (!observations || polynomial.zero_domain_blinder.size() != kMechanismMaskRank ||
                    polynomial.zero_domain_blinder.size() < observations->field_rows ||
                    polynomial.zero_domain_blinder.size() < observations->group_rows())
                    throw std::runtime_error("mechanism masked commitment does not meet its label-bound observation ledger: " + label);
            }
            commitment.name = label;
            context.proof.commitments.push_back(CommitmentRecord{label, commitment.point});
            context.objects.emplace(label, std::move(commitment));
            if (absorb)
                context.transcript.absorb_commitment(label, context.objects.at(label).point);
        }

        Commitment commit_polynomial(ProofBuildContext &context, const std::string &label, Polynomial polynomial, bool absorb)
        {
            const auto start = Clock::now();
            auto commitment = crypto::KZG::commit(label, polynomial, context.key);
            context.commitment_time_ms += elapsed_ms(start, Clock::now());
            commitment.opening_polynomial = std::make_shared<Polynomial>(std::move(polynomial));
            record_commitment(context, label, commitment, absorb);
            return commitment;
        }

        Commitment commit_masked_evaluations(ProofBuildContext &context, const std::string &label,
                                             const std::vector<FieldElement> &values,
                                             const std::shared_ptr<algebra::RootOfUnityDomain> &domain,
                                             std::size_t observation_rank, bool absorb)
        {
            const auto observations = expected_observation_rows(context.statement, label);
            if (observation_rank != kMechanismMaskRank || !observations ||
                observation_rank < observations->field_rows || observation_rank < observations->group_rows())
                throw std::runtime_error("mechanism mask rank/label does not match the reviewed observation registry: " + label);
            auto mask = sample_zero_domain_blinder(context.witness_seed, label, observation_rank);
            return commit_polynomial(context, label, make_masked_evaluation_polynomial(label, values, domain, mask), absorb);
        }

        std::string quotient_mask_label(const std::string &quotient_label)
        {
            if (quotient_label.empty())
                throw std::runtime_error("mechanism quotient label is empty");
            return quotient_label + "/R_Q";
        }

        void commit_quotient_with_mask(ProofBuildContext &context, const std::string &quotient_label,
                                       Polynomial quotient, std::size_t domain_size)
        {
            if (quotient.basis != algebra::PolynomialBasis::Coefficient || domain_size == 0 ||
                domain_size > std::numeric_limits<std::size_t>::max() - kMechanismMaskRank)
                throw std::runtime_error("mechanism masked quotient has an invalid coefficient basis or domain: " + quotient_label);
            const auto mask_label = quotient_mask_label(quotient_label);
            auto mask = sample_zero_domain_blinder({}, mask_label, kMechanismMaskRank);
            auto masked_coefficients = quotient.values();
            const auto minimum_size = domain_size + kMechanismMaskRank;
            if (masked_coefficients.size() < minimum_size)
                masked_coefficients.resize(minimum_size, FieldElement::zero());
            for (std::size_t i = 0; i < mask.size(); ++i)
            {
                masked_coefficients[i] -= mask[i];
                masked_coefficients[domain_size + i] += mask[i];
            }
            if (masked_coefficients.size() > context.key.pk.degree_bound + 1)
                throw std::runtime_error("mechanism quotient mask exceeds the authenticated SRS degree bound: " + quotient_label);

            auto masked_quotient = Polynomial::from_coefficients(quotient_label, std::move(masked_coefficients));
            auto mask_polynomial = Polynomial::from_coefficients(mask_label, std::move(mask));
            commit_polynomial(context, quotient_label, std::move(masked_quotient));
            commit_polynomial(context, mask_label, std::move(mask_polynomial));
        }

        Commitment derive_linear_commitment(ProofBuildContext &context, const std::string &label,
                                            const std::vector<std::pair<std::string, FieldElement>> &terms, bool absorb)
        {
            if (terms.empty())
                throw std::runtime_error("empty mechanism linear commitment derivation");
            std::vector<crypto::G1Point> points;
            std::vector<FieldElement> scalars;
            std::optional<algebra::PolynomialBasis> result_basis;
            std::shared_ptr<algebra::RootOfUnityDomain> result_domain;
            std::vector<FieldElement> result_values;
            std::vector<FieldElement> combined_mask;
            for (const auto &[source_label, scalar] : terms)
            {
                const auto source_it = context.objects.find(source_label);
                if (source_it == context.objects.end() || source_it->second.opening_polynomial == nullptr)
                    throw std::runtime_error("mechanism linear commitment source is missing opening material: " + source_label);
                const auto &source = *source_it->second.opening_polynomial;
                if (!result_basis.has_value())
                {
                    result_basis = source.basis;
                    result_domain = source.domain;
                    result_values.assign(source.size(), FieldElement::zero());
                }
                else
                {
                    if (*result_basis != source.basis || ((result_domain == nullptr) != (source.domain == nullptr)) ||
                        (result_domain != nullptr && (result_domain->name != source.domain->name || result_domain->size != source.domain->size)) ||
                        result_values.size() != source.size())
                        throw std::runtime_error("mechanism linear commitment mixes incompatible polynomial domains");
                }
                for (std::size_t i = 0; i < source.size(); ++i)
                    result_values[i] += scalar * source.value_at(i);
                if (!source.zero_domain_blinder.empty())
                {
                    if (combined_mask.size() < source.zero_domain_blinder.size())
                        combined_mask.resize(source.zero_domain_blinder.size(), FieldElement::zero());
                    for (std::size_t i = 0; i < source.zero_domain_blinder.size(); ++i)
                        combined_mask[i] += scalar * source.zero_domain_blinder[i];
                }
                points.push_back(source_it->second.point);
                scalars.push_back(scalar);
            }
            std::shared_ptr<Polynomial> result;
            if (*result_basis == algebra::PolynomialBasis::Evaluation)
            {
                result = std::make_shared<Polynomial>(Polynomial::from_evaluations(label, std::move(result_values), result_domain));
                if (!combined_mask.empty())
                    result->set_zero_domain_blinder(std::move(combined_mask));
            }
            else
                result = std::make_shared<Polynomial>(Polynomial::from_coefficients(label, std::move(result_values)));
            const auto commitment_start = Clock::now();
            auto point = crypto::g1_multi_scalar_mul(points, scalars);
            context.commitment_time_ms += elapsed_ms(commitment_start, Clock::now());
            Commitment out{.name = label,
                           .point = std::move(point),
                           .opening_polynomial = std::move(result)};
            record_commitment(context, label, out, absorb);
            return out;
        }

        void add_opening_group(ProofBuildContext &context, const std::string &label,
                               const std::vector<std::string> &labels, const std::vector<FieldElement> &values,
                               const FieldElement &point)
        {
            if (labels.empty() || labels.size() != values.size())
                throw std::runtime_error("mechanism KZG opening group shape mismatch: " + label);
            std::vector<Commitment> commitments;
            commitments.reserve(labels.size());
            for (const auto &item : labels)
            {
                const auto it = context.objects.find(item);
                if (it == context.objects.end() || it->second.opening_polynomial == nullptr)
                    throw std::runtime_error("mechanism opening group references missing polynomial material: " + item);
                commitments.push_back(it->second);
            }
            context.transcript.absorb_text("mechanism_open_group:label", label);
            context.transcript.absorb_scalar("mechanism_open_group:point", point);
            context.transcript.absorb_scalar("mechanism_open_group:member_count", FieldElement(labels.size()));
            for (std::size_t i = 0; i < labels.size(); ++i)
            {
                context.transcript.absorb_text("mechanism_open_group:member_label:" + std::to_string(i), labels[i]);
                context.transcript.absorb_commitment("mechanism_open_group:member_point:" + std::to_string(i), commitments[i].point);
                context.transcript.absorb_scalar("mechanism_open_group:member_value:" + std::to_string(i), values[i]);
            }
            const auto fold = context.transcript.challenge_excluding("mechanism_open_fold:v3:" + label,
                                                                       [](const FieldElement &value) { return value.is_zero(); });
            const auto witness = crypto::KZG::open_point_group(commitments, point, values, fold, context.key);
            context.proof.opening_groups.push_back(OpeningGroup{label, labels, values, witness});
        }

        const CommitmentRecord &required_commitment(const MechanismProof &proof, const std::string &label)
        {
            const auto it = std::find_if(proof.commitments.begin(), proof.commitments.end(), [&](const auto &item) { return item.label == label; });
            if (it == proof.commitments.end())
                throw std::runtime_error("mechanism proof is missing commitment: " + label);
            return *it;
        }

        const OpeningGroup &required_opening_group(const MechanismProof &proof, const std::string &label)
        {
            const auto it = std::find_if(proof.opening_groups.begin(), proof.opening_groups.end(), [&](const auto &item) { return item.label == label; });
            if (it == proof.opening_groups.end())
                throw std::runtime_error("mechanism proof is missing opening group: " + label);
            return *it;
        }

        FieldElement required_opening_value(const OpeningGroup &group, const std::string &commitment_label)
        {
            for (std::size_t i = 0; i < group.commitment_labels.size(); ++i)
                if (group.commitment_labels[i] == commitment_label)
                    return group.values.at(i);
            throw std::runtime_error("mechanism opening group is missing claimed value: " + commitment_label);
        }

        bool verify_opening_group(ProofVerifyContext &context, const std::string &label, const std::vector<std::string> &expected_labels,
                                  const FieldElement &point)
        {
            const auto &group = required_opening_group(context.proof, label);
            if (group.commitment_labels != expected_labels || group.values.size() != expected_labels.size())
                return false;
            std::vector<Commitment> commitments;
            commitments.reserve(expected_labels.size());
            for (const auto &item : expected_labels)
            {
                const auto &record = required_commitment(context.proof, item);
                commitments.push_back(Commitment{.name = item, .point = record.point, .opening_polynomial = nullptr});
            }
            context.transcript.absorb_text("mechanism_open_group:label", label);
            context.transcript.absorb_scalar("mechanism_open_group:point", point);
            context.transcript.absorb_scalar("mechanism_open_group:member_count", FieldElement(expected_labels.size()));
            for (std::size_t i = 0; i < expected_labels.size(); ++i)
            {
                context.transcript.absorb_text("mechanism_open_group:member_label:" + std::to_string(i), expected_labels[i]);
                context.transcript.absorb_commitment("mechanism_open_group:member_point:" + std::to_string(i), commitments[i].point);
                context.transcript.absorb_scalar("mechanism_open_group:member_value:" + std::to_string(i), group.values[i]);
            }
            const auto fold = context.transcript.challenge_excluding("mechanism_open_fold:v3:" + label,
                                                                       [](const FieldElement &value) { return value.is_zero(); });
            return crypto::KZG::verify_point_group(commitments, point, group.values, fold, group.witness, context.key.vk);
        }
    }

    RunReceipt run(const RunOptions &options)
    {
        const auto secret = read_witness_seed(options.witness_seed_file);
        configure_backend(options.backend, options.device, false);
        std::optional<internal::RealMechanismInput> real_input;
        if (options.real_input_path.empty() != options.real_input_metadata_path.empty())
            throw std::runtime_error("real mechanism runs require both --real-input and --real-input-metadata");
        if (!options.real_input_path.empty())
        {
            if (options.backend != Backend::Cuda)
                throw std::runtime_error("real-graph mechanism proving is CUDA-only; CPU proving remains available for synthetic mechanism tests");
            const auto bundle_link = std::filesystem::symlink_status(options.real_input_path);
            const auto metadata_link = std::filesystem::symlink_status(options.real_input_metadata_path);
            if (std::filesystem::is_symlink(bundle_link) || !std::filesystem::is_regular_file(bundle_link) ||
                std::filesystem::is_symlink(metadata_link) || !std::filesystem::is_regular_file(metadata_link))
                throw std::runtime_error("real mechanism bundle and metadata must be regular non-symlink files");
            real_input = internal::read_real_mechanism_input(options.real_input_path);
            const auto metadata_bytes = internal::read_text_file(options.real_input_metadata_path, 64U << 20);
            real_input->metadata_sha256 = internal::sha256_hex(metadata_bytes);
            const auto dataset = real_input->dataset;
            const auto expected_srs_id = registered_srs_id(dataset);
            const auto registered_n_domain = dataset == "ppi" ? 65536U : 4096U;
            const auto registered_e_domain = dataset == "ppi" ? 8388608U : 32768U;
            const auto required_coefficients = 2U * registered_e_domain + 14U;
            const auto bundle_path = json_escape_string(std::filesystem::absolute(options.real_input_path).string());
            const auto g = real_input->witness.node_ptr.size() - 1;
            const bool has_component_zero = metadata_bytes.find("{\"canonical_id\":0,") != std::string::npos;
            const bool has_component_nine = metadata_bytes.find("{\"canonical_id\":9,") != std::string::npos;
            const auto checkpoint_marker = metadata_bytes.find("\"checkpoint_bundle\":{\"path\":");
            const auto checkpoint_hash_marker = checkpoint_marker == std::string::npos ? std::string::npos :
                metadata_bytes.find("\"sha256\":\"" + real_input->checkpoint_bundle_sha256 + "\",\"files\":", checkpoint_marker);
            if (dataset != "cora" && dataset != "citeseer" && dataset != "ppi")
                throw std::runtime_error("real mechanism bundle dataset is outside the registered formal catalog");
            if (metadata_bytes.find("\"schema\":\"zkgat_real_mechanism_input_v1\"") == std::string::npos ||
                metadata_bytes.find("\"dataset\":\"" + dataset + "\"") == std::string::npos ||
                metadata_bytes.find("\"input_preprocess_id\":\"" + real_input->input_preprocess_id + "\"") == std::string::npos ||
                real_input->input_preprocess_id != "row_normalize" ||
                metadata_bytes.find("\"model_arch_id\":\"" + real_input->model_arch_id + "\"") == std::string::npos ||
                metadata_bytes.find("\"model_param_id\":\"" + real_input->model_param_id + "\"") == std::string::npos ||
                metadata_bytes.find("\"quant_cfg_id\":\"" + real_input->quant_cfg_id + "\"") == std::string::npos ||
                metadata_bytes.find("\"static_table_id\":\"" + real_input->static_table_id + "\"") == std::string::npos ||
                metadata_bytes.find("\"bundle\":{\"path\":\"" + bundle_path + "\",\"sha256\":\"" + real_input->bundle_sha256 + "\"}") == std::string::npos ||
                checkpoint_hash_marker == std::string::npos ||
                metadata_bytes.find("\"witness_sha256\":\"" + real_input->witness_sha256 + "\"") == std::string::npos ||
                metadata_bytes.find("\"source_manifest_sha256\":\"" + real_input->source_manifest_sha256 + "\"") == std::string::npos ||
                metadata_bytes.find("\"config_sha256\":\"" + real_input->config_sha256 + "\"") == std::string::npos ||
                metadata_bytes.find("\"layer\":0,\"head\":0,\"formal_head_width\":8") == std::string::npos ||
                metadata_bytes.find("\"actual\":{\"N\":" + std::to_string(real_input->witness.node_count) + ",\"E\":" +
                                   std::to_string(real_input->witness.edge_count) + ",\"G\":" + std::to_string(g) + "}") == std::string::npos ||
                metadata_bytes.find("\"srs_expectation\":{\"dataset_id\":\"" + dataset + "\",\"srs_id\":\"" +
                                   expected_srs_id + "\",\"n_domain\":" + std::to_string(registered_n_domain) +
                                   ",\"e_domain\":" + std::to_string(registered_e_domain) +
                                   ",\"required_coefficient_count\":" + std::to_string(required_coefficients) + "}") == std::string::npos ||
                !has_component_zero || (dataset == "ppi" && g == 2 && !has_component_nine) ||
                ((dataset != "ppi" || g == 1) && has_component_nine))
                throw std::runtime_error("real mechanism metadata does not match the canonical witness bundle and data identity");
        }
        const auto node_domain_size = real_input.has_value() ? (real_input->dataset == "ppi" ? 65536U : 4096U) : 4096U;
        const auto edge_domain_size = real_input.has_value() ? (real_input->dataset == "ppi" ? 8388608U : 32768U) : 32768U;
        const auto expected_srs_id = real_input.has_value() ? registered_srs_id(real_input->dataset) : std::string{};
        const auto key = load_mechanism_key(options.srs_path, node_domain_size, edge_domain_size, expected_srs_id);
        const auto statement = real_input.has_value() ? generate_real_statement(options, key, *real_input) : generate_statement(options, key, secret);
        validate_statement(statement, key);
        const auto statement_bytes = statement.canonical_json();
        const auto statement_hash = internal::sha256_hex(statement_bytes);

        double cuda_warmup_time_ms = 0.0;
        std::uint64_t cuda_warmup_kernel_invocations = 0;
        if (options.backend == Backend::Cuda)
        {
            const auto warmup_start = Clock::now();
            const auto warmup_before = algebra::cuda_execution_stats_snapshot();
            const auto warmup_kernel_before = algebra::cuda_kernel_invocation_count();
            constexpr std::size_t kWarmupSize = 32768;
            constexpr std::size_t kWarmupMsmSize = 1U << 16U;
            std::vector<algebra::FieldElement> coefficients(kWarmupSize, algebra::FieldElement::zero());
            if (real_input.has_value())
            {
                const auto &warm_source = real_input->witness.h_prime;
                for (std::size_t i = 0; i < coefficients.size(); ++i)
                    coefficients[i] = warm_source[i % warm_source.size()][i % 8] + algebra::FieldElement::one();
            }
            else
            {
                for (std::size_t i = 0; i < coefficients.size(); ++i)
                    coefficients[i] = algebra::FieldElement(i + 1);
            }
            auto transform = coefficients;
            algebra::ntt_in_place(algebra::FieldElement::mutable_native_array_data(transform.data()), transform.size());
            algebra::intt_in_place(algebra::FieldElement::mutable_native_array_data(transform.data()), transform.size());
            if (transform != coefficients)
                throw std::runtime_error("mechanism CUDA warmup NTT/INTT roundtrip changed its field vector");
            crypto::G1Point ephemeral_base = crypto::g1_zero();
            mcl::bn::hashAndMapToG1(ephemeral_base.value, "zkgat:mechanism:untimed-msm-warmup:v1");
            std::vector<crypto::G1Point> ephemeral_points;
            std::vector<algebra::FieldElement> ephemeral_scalars;
            ephemeral_points.reserve(kWarmupMsmSize);
            ephemeral_scalars.reserve(kWarmupMsmSize);
            auto point = ephemeral_base;
            for (std::size_t i = 0; i < kWarmupMsmSize; ++i)
            {
                ephemeral_points.push_back(point);
                ephemeral_scalars.emplace_back(static_cast<std::uint64_t>(i + 1));
                point = crypto::g1_add(point, ephemeral_base);
            }
            const crypto::G1MSMExecutionOptions warmup_msm_options{
                .stable_srs_bases = false,
                .stable_srs_capacity = 0,
            };
            crypto::G1MSMProfile warmup_msm_profile;
            (void)crypto::g1_multi_scalar_mul_profiled(ephemeral_points.data(), ephemeral_scalars.data(), kWarmupMsmSize,
                                                       &warmup_msm_profile, &warmup_msm_options);
            const auto warmup_after = algebra::cuda_execution_stats_snapshot();
            const auto warmup_kernel_after = algebra::cuda_kernel_invocation_count();
            if (warmup_kernel_after < warmup_kernel_before)
                throw std::runtime_error("mechanism CUDA warmup kernel counter moved backwards");
            cuda_warmup_kernel_invocations = warmup_kernel_after - warmup_kernel_before;
            if (cuda_warmup_kernel_invocations == 0 || warmup_after.ntt_calls == warmup_before.ntt_calls ||
                warmup_after.intt_calls == warmup_before.intt_calls ||
                (warmup_after.msm_single_calls == warmup_before.msm_single_calls && warmup_after.msm_batch_calls == warmup_before.msm_batch_calls) ||
                warmup_msm_profile.input_count != kWarmupMsmSize || warmup_msm_profile.nonzero_scalar_count != kWarmupMsmSize ||
                warmup_msm_profile.effective_scalar_count != kWarmupMsmSize)
                throw std::runtime_error("CUDA warmup did not exercise NTT, inverse NTT, and a non-SRS ephemeral CUDA MSM before the timed proof");
            cuda_warmup_time_ms = elapsed_ms(warmup_start, Clock::now());
        }

        MechanismProof proof;
        initialize_proof_metadata(statement, proof);
        auto transcript = mechanism_transcript(statement, proof);
        const auto cuda_before = algebra::cuda_execution_stats_snapshot();
        const auto cuda_kernels_before = algebra::cuda_kernel_invocation_count();
        double commitment_time_ms = 0.0;
        const auto proof_start = Clock::now();
        internal::ProofBuildContext context{statement, key, transcript, secret, proof, commitment_time_ms, {},
                                             real_input.has_value() ? &*real_input : nullptr};
        if (options.mechanism == Mechanism::FeatureFolding)
            internal::prove_feature_folding(context);
        else
            internal::prove_shared_neighborhood(context);
        validate_proof_inventory(statement, proof);
        const auto proof_bytes = internal::encode_proof(proof);
        const auto crypto_prove_ms = elapsed_ms(proof_start, Clock::now());

        const auto cuda_after = algebra::cuda_execution_stats_snapshot();
        const auto cuda_kernels_after = algebra::cuda_kernel_invocation_count();
        const auto counter_delta = [](std::uint64_t after, std::uint64_t before, const char *counter)
        {
            if (after < before)
                throw std::runtime_error(std::string("mechanism native counter moved backwards: ") + counter);
            return after - before;
        };
        const auto cuda_kernel_delta = counter_delta(static_cast<std::uint64_t>(cuda_kernels_after),
                                                     static_cast<std::uint64_t>(cuda_kernels_before), "cuda_kernel_invocations");
        const auto actual_cuda = cuda_kernel_delta > 0;
        if (options.backend == Backend::Cuda && !actual_cuda)
            throw std::runtime_error("CUDA proving was requested but no CUDA cryptographic kernel executed");
        if (options.backend == Backend::Cpu && actual_cuda)
            throw std::runtime_error("CPU proving unexpectedly executed a CUDA kernel");

        internal::write_text_file(options.statement_out, statement_bytes);
        internal::write_binary_file(options.proof_out, proof_bytes);
        RunReceipt receipt;
        receipt.commitment_time_ms = commitment_time_ms;
        receipt.prove_time_ms = std::max(0.0, crypto_prove_ms - commitment_time_ms);
        receipt.proof_size_bytes = proof_bytes.size();
        receipt.proof_sha256 = internal::sha256_hex(std::string_view(reinterpret_cast<const char *>(proof_bytes.data()), proof_bytes.size()));
        receipt.statement_sha256 = statement_hash;
        receipt.pair_input_commitment_sha256 = statement.pair_input_commitment_sha256;
        receipt.srs_id = key.manifest.srs_id;
        receipt.srs_manifest_sha256 = key.manifest.manifest_hash;
        receipt.parent_protocol_schema_id = key.manifest.protocol_schema_id;
        receipt.actual_cuda = actual_cuda;
        receipt.cuda_ntt_forced = options.backend == Backend::Cuda;
        receipt.device = options.backend == Backend::Cuda ? options.device : -1;
        receipt.device_name = options.backend == Backend::Cuda ? cuda_device_name(options.device) : "cpu";
        receipt.cuda_kernel_invocations_delta = cuda_kernel_delta;
        receipt.cuda_h2d_bytes_delta = counter_delta(cuda_after.h2d_bytes, cuda_before.h2d_bytes, "h2d_bytes");
        receipt.cuda_d2h_bytes_delta = counter_delta(cuda_after.d2h_bytes, cuda_before.d2h_bytes, "d2h_bytes");
        receipt.cuda_h2d_operations_delta = counter_delta(cuda_after.h2d_operations, cuda_before.h2d_operations, "h2d_operations");
        receipt.cuda_d2h_operations_delta = counter_delta(cuda_after.d2h_operations, cuda_before.d2h_operations, "d2h_operations");
        receipt.cuda_stream_synchronizations_delta = counter_delta(cuda_after.stream_synchronizations, cuda_before.stream_synchronizations, "stream_synchronizations");
        receipt.cuda_ntt_calls_delta = counter_delta(cuda_after.ntt_calls, cuda_before.ntt_calls, "ntt_calls");
        receipt.cuda_intt_calls_delta = counter_delta(cuda_after.intt_calls, cuda_before.intt_calls, "intt_calls");
        receipt.cuda_quotient_calls_delta = counter_delta(cuda_after.quotient_calls, cuda_before.quotient_calls, "quotient_calls");
        receipt.cuda_t_edge_calls_delta = counter_delta(cuda_after.t_edge_calls, cuda_before.t_edge_calls, "t_edge_calls");
        receipt.cuda_t_fh_calls_delta = counter_delta(cuda_after.t_fh_calls, cuda_before.t_fh_calls, "t_fh_calls");
        receipt.cuda_opening_fold_calls_delta = counter_delta(cuda_after.opening_fold_calls, cuda_before.opening_fold_calls, "opening_fold_calls");
        receipt.cuda_msm_single_calls_delta = counter_delta(cuda_after.msm_single_calls, cuda_before.msm_single_calls, "msm_single_calls");
        receipt.cuda_msm_batch_calls_delta = counter_delta(cuda_after.msm_batch_calls, cuda_before.msm_batch_calls, "msm_batch_calls");
        receipt.cuda_resident_hits_delta = counter_delta(cuda_after.resident_hits, cuda_before.resident_hits, "resident_hits");
        receipt.cuda_resident_misses_delta = counter_delta(cuda_after.resident_misses, cuda_before.resident_misses, "resident_misses");
        if (options.backend == Backend::Cpu &&
            (receipt.cuda_h2d_bytes_delta != 0 || receipt.cuda_d2h_bytes_delta != 0 || receipt.cuda_h2d_operations_delta != 0 ||
             receipt.cuda_d2h_operations_delta != 0 || receipt.cuda_stream_synchronizations_delta != 0 || receipt.cuda_ntt_calls_delta != 0 ||
             receipt.cuda_intt_calls_delta != 0 || receipt.cuda_quotient_calls_delta != 0 || receipt.cuda_t_edge_calls_delta != 0 ||
             receipt.cuda_t_fh_calls_delta != 0 || receipt.cuda_opening_fold_calls_delta != 0 || receipt.cuda_msm_single_calls_delta != 0 ||
             receipt.cuda_msm_batch_calls_delta != 0 || receipt.cuda_resident_hits_delta != 0 || receipt.cuda_resident_misses_delta != 0))
            throw std::runtime_error("CPU proving unexpectedly changed native CUDA execution counters");
        if (options.backend == Backend::Cuda &&
            (receipt.cuda_ntt_calls_delta == 0 || receipt.cuda_intt_calls_delta == 0 ||
             (receipt.cuda_msm_single_calls_delta == 0 && receipt.cuda_msm_batch_calls_delta == 0)))
            throw std::runtime_error("CUDA mechanism proof did not execute the required NTT, inverse NTT, and MSM primitives");
        receipt.security_budget_ledger_sha256 = statement.security_budget_ledger_sha256;
        receipt.security_budget_ledger_json = internal::security_budget_ledger_json(statement);
        receipt.real_input = real_input.has_value();
        if (real_input.has_value())
        {
            receipt.real_input_dataset = real_input->dataset;
            receipt.real_input_preprocess_id = real_input->input_preprocess_id;
            receipt.real_input_bundle_sha256 = real_input->bundle_sha256;
            receipt.real_input_witness_sha256 = real_input->witness_sha256;
            receipt.real_input_metadata_sha256 = real_input->metadata_sha256;
            receipt.real_input_checkpoint_bundle_sha256 = real_input->checkpoint_bundle_sha256;
            receipt.real_input_source_manifest_sha256 = real_input->source_manifest_sha256;
            receipt.real_input_nodes = real_input->witness.node_count;
            receipt.real_input_edges = real_input->witness.edge_count;
            receipt.real_input_graphs = real_input->witness.node_ptr.size() - 1;
            receipt.real_input_selected_width = options.width;
            receipt.real_input_layer = 0;
            receipt.real_input_head = 0;
            receipt.real_input_formal_head_width = 8;
        }
        receipt.cuda_warmup_time_ms = cuda_warmup_time_ms;
        receipt.cuda_warmup_kernel_invocations = cuda_warmup_kernel_invocations;
        internal::write_text_file(options.result_json, internal::run_receipt_json(options, receipt));
        return receipt;
    }

    VerifyReceipt verify(const VerifyOptions &options)
    {
        configure_backend(Backend::Cpu, 0, true);
        const auto statement_bytes = internal::read_text_file(options.statement_path, 1U << 30);
        const auto proof_bytes = internal::read_binary_file(options.proof_path, 1U << 30);
        const auto [node_domain_size, edge_domain_size] = statement_domain_catalog_prefix(statement_bytes);
        const auto key = load_mechanism_key(options.srs_path, node_domain_size, edge_domain_size);
        const auto verifier_cuda_kernels_before = algebra::cuda_kernel_invocation_count();
        const auto verify_start = Clock::now();
        const auto statement = PublicStatement::parse_canonical_json(statement_bytes);
        const auto proof_hash = internal::sha256_hex(std::string_view(reinterpret_cast<const char *>(proof_bytes.data()), proof_bytes.size()));
        const auto statement_hash = internal::sha256_hex(statement_bytes);
        auto proof = internal::decode_proof(proof_bytes);
        validate_statement(statement, key);
        if (proof.mechanism != statement.mechanism || proof.variant != statement.variant || proof.statement_sha256 != statement_hash ||
            proof.srs_id != key.manifest.srs_id || proof.srs_manifest_sha256 != key.manifest.manifest_hash ||
            proof.parent_protocol_schema_id != key.manifest.protocol_schema_id || proof.mechanism_schema_id != statement.mechanism_schema_id ||
            proof.security_budget_ledger_sha256 != statement.security_budget_ledger_sha256)
            throw std::runtime_error("mechanism proof metadata does not match statement or authenticated SRS");
        validate_proof_inventory(statement, proof);
        auto transcript = mechanism_transcript(statement, proof);
        internal::ProofVerifyContext context{statement, key, transcript, proof, {}};
        bool accepted = false;
        if (statement.mechanism == Mechanism::FeatureFolding)
            accepted = internal::verify_feature_folding(context);
        else
            accepted = internal::verify_shared_neighborhood(context);
        if (!accepted)
            throw std::runtime_error("mechanism proof rejected by independent verifier");
        const auto verify_elapsed_ms = elapsed_ms(verify_start, Clock::now());
        const auto verifier_cuda_kernels_after = algebra::cuda_kernel_invocation_count();
        if (verifier_cuda_kernels_after < verifier_cuda_kernels_before)
            throw std::runtime_error("mechanism verifier CUDA kernel counter moved backwards");
        const auto verifier_cuda_delta = verifier_cuda_kernels_after - verifier_cuda_kernels_before;
        if (verifier_cuda_delta != 0)
            throw std::runtime_error("CPU mechanism verifier unexpectedly executed a CUDA kernel");
        VerifyReceipt receipt;
        receipt.verify_time_ms = verify_elapsed_ms;
        receipt.proof_sha256 = proof_hash;
        receipt.statement_sha256 = statement_hash;
        receipt.srs_id = key.manifest.srs_id;
        receipt.srs_manifest_sha256 = key.manifest.manifest_hash;
        receipt.parent_protocol_schema_id = key.manifest.protocol_schema_id;
        receipt.security_budget_ledger_sha256 = statement.security_budget_ledger_sha256;
        receipt.security_budget_ledger_json = internal::security_budget_ledger_json(statement);
        receipt.verifier_cuda_kernel_invocations_delta = static_cast<std::uint64_t>(verifier_cuda_delta);
        internal::write_text_file(options.result_json, internal::verify_receipt_json(receipt));
        return receipt;
    }

    namespace internal
    {
        std::string compute_mechanism_schema_id(const PublicStatement &s)
        {
            std::ostringstream canonical;
            canonical << "zkgat-independent-mechanism-schema-v8|dataset=" << s.dataset_id << "|input_preprocess_id=" << s.input_preprocess_id
                      << "|real_input_bundle_sha256=" << s.real_input_bundle_sha256 << "|real_input_metadata_sha256=" << s.real_input_metadata_sha256
                      << "|checkpoint_bundle_sha256=" << s.checkpoint_bundle_sha256 << "|source_manifest_sha256=" << s.input_source_manifest_sha256
                      << "|source_layer=" << s.source_layer << "|source_head=" << s.source_head << "|formal_head_width=" << s.formal_head_width
                      << "|mechanism=" << to_string(s.mechanism) << "|variant=" << to_string(s.variant)
                      << "|N=" << s.node_domain << ':' << s.node_domain_size << "|E=" << s.edge_domain << ':' << s.edge_domain_size << "|nodes=" << s.nodes
                      << "|edges=" << s.edges << "|width=" << s.width << "|graphs=" << s.graphs << "|parent_manifest=" << s.srs_manifest_sha256
                      << "|parent_schema=" << s.parent_protocol_schema_id << "|parent_srs_degree_bound=" << s.parent_srs_degree_bound
                      << "|statement_codec=zkgat_mechanism_statement_v8|proof_codec=GZKMECH2-v2|fiat_shamir=Transcript-SHA256-reject-zero-v3"
                      << "|open_fold=ordered-label-point-commitments-claims-before-challenge-v3"
                      << "|route=LogUp-inverse+accumulator+masked-quotient+Taylor-terminal-CRPC|security_ledger=numerical-union-bound-v5-global-SRS-degree-QFS"
                      << "|quotient_mask=Qzk=Q+(X^D-1)RQ;commit-Qzk-and-RQ;verify-residual=Z*(Qzk-Z*RQ);independent-csprng-rank8"
                      << "|quotient_mask_privacy=rank8-source-dependency-and-aggregate-witness-certificate-v3-source-opening-groups;crs-bound=D+8-over-p-minus-one-minus-max-domain"
                      << "|observation_lineage=field-points-plus-source-commitment-plus-verified-derived-point-transform-v3;quotient-pair-field2-group3-source-opening-groups"
                      << "|zero_domain_mask_rank=8|target_bits=120";
            for (const auto value : s.node_ptr)
                canonical << "|node_ptr=" << value;
            for (const auto value : s.edge_ptr)
                canonical << "|edge_ptr=" << value;
            for (const auto &edge : s.topology)
                canonical << "|edge=" << edge.graph << ',' << edge.src << ',' << edge.dst;
            if (s.mechanism == Mechanism::FeatureFolding)
                canonical << "|relation=(id,feature)|fold=postcommit-xi|control=per-coordinate-route";
            else
                canonical << "|relation=(graph,dst,weighted-sum)|fold=postcommit-xi|original=shared-lambda|control=dual-PSQ";
            return sha256_hex(canonical.str());
        }

        std::string compute_security_budget_ledger_sha256(const PublicStatement &s)
        {
            const auto budget = build_security_budget(s);
            require(budget.meets_target, "mechanism numerical security ledger does not meet the 120-bit target");
            return sha256_hex(budget.canonical);
        }

        std::string security_budget_ledger_json(const PublicStatement &s)
        {
            const auto b = build_security_budget(s);
            require(b.meets_target, "mechanism numerical security ledger does not meet the 120-bit target");
            std::ostringstream out;
            out << "{\"target_bits\":" << kTargetSecurityBits << ",\"budget_exact\":true,\"bound_kind\":\"conservative_exact\",\"field_bound_bits\":" << kFieldBoundBits
                << ",\"fiat_shamir_query_budget_max\":" << kFiatShamirQueryBudgetMax
                << ",\"parent_srs_degree_bound\":" << b.global_srs_degree_bound
                << ",\"field_modulus_decimal\":\"" << b.scalar_modulus_decimal << "\",\"field_denominator_lower_bound\":\"2^" << kFieldBoundBits
                << "\",\"field_actual_denominator_decimal\":\"" << b.field_actual_denominator_decimal
                << "\",\"max_field_excluded_roots\":" << b.max_field_excluded_roots
                << ",\"max_field_identity_degree\":" << b.max_field_identity_degree
                << ",\"crs_actual_denominator_decimal\":\"" << b.crs_actual_denominator_decimal
                << "\",\"computational_assumption_bits\":" << kComputationalAssumptionBits << ",\"mechanism\":\"" << to_string(s.mechanism)
                << "\",\"variant\":\"" << to_string(s.variant) << "\",\"route_rounds\":" << b.route_rounds
                << ",\"psq_channels\":" << b.psq_channels << ",\"identity_count\":" << b.identity_count
                << ",\"identity_counts\":{\"route\":" << b.route_identity_count << ",\"psq\":" << b.psq_identity_count << "}"
                << ",\"degree_bounds\":{\"route_inverse_node\":" << b.route_inverse_node_degree
                << ",\"route_inverse_edge\":" << b.route_inverse_edge_degree
                << ",\"route_accumulator_node\":" << b.route_accumulator_node_degree
                << ",\"route_accumulator_edge\":" << b.route_accumulator_edge_degree << ",\"psq_step\":" << b.psq_step_degree
                << ",\"psq_terminal\":" << b.psq_terminal_degree << ",\"max_quotient\":" << b.max_quotient_degree
                << ",\"global_srs_malicious_identity_bound\":" << b.global_srs_degree_bound
                << ",\"route_inverse_node_identity\":" << b.route_inverse_node_identity_degree
                << ",\"route_inverse_edge_identity\":" << b.route_inverse_edge_identity_degree
                << ",\"route_accumulator_node_identity\":" << b.route_accumulator_node_identity_degree
                << ",\"route_accumulator_edge_identity\":" << b.route_accumulator_edge_identity_degree
                << ",\"psq_step_identity\":" << b.psq_step_identity_degree
                << ",\"psq_terminal_identity\":" << b.psq_terminal_identity_degree
                << ",\"tuple_compression\":" << b.tuple_compression_degree_bound
                << ",\"max_field_identity\":" << b.max_field_identity_degree
                << ",\"max_masked_quotient\":" << b.max_qzk_degree << ",\"required_srs_degree\":" << b.required_srs_degree << "}"
                << ",\"field_error_bound\":{\"denominator_lower_bound\":\"2^" << kFieldBoundBits << "\""
                << ",\"fiat_shamir_query_budget_max\":" << kFiatShamirQueryBudgetMax
                << ",\"tuple_compression_numerator\":" << b.tuple_compression_numerator
                << ",\"logup_denominator_numerator\":" << b.logup_denominator_numerator
                << ",\"quotient_identity_numerator\":" << b.quotient_identity_numerator
                << ",\"terminal_crpc_numerator\":" << b.terminal_crpc_numerator
                << ",\"opening_fold_numerator\":" << b.opening_fold_numerator << ",\"width_fold_numerator\":" << b.width_fold_numerator
                << ",\"shared_lambda_numerator\":" << b.shared_lambda_numerator
                << ",\"unamplified_numerator\":\"" << b.field_unamplified_numerator
                << "\",\"field_numerator\":\"" << b.field_error_numerator << "\"}"
                << ",\"crs_error_bound\":{\"numerator\":\"" << b.quotient_crs_numerator << "\",\"actual_denominator\":\""
                << b.crs_actual_denominator_decimal << "\",\"denominator_lower_bound\":\"2^" << kFieldBoundBits << "\",\"terms\":"
                << b.quotient_mask_inventory.size() << ",\"fiat_shamir_query_budget_max\":" << kFiatShamirQueryBudgetMax
                << ",\"query_amplified_numerator\":\"" << b.crs_error_numerator << "\"}"
                << ",\"assumptions\":{\"kzg_assumption_profile_id\":\"shifted_kzg_divisibility\",\"kzg_assumption_bits\":128"
                << ",\"kzg_point_group_games\":" << b.kzg_point_group_games
                << ",\"fiat_shamir_assumption_profile_id\":\"" << kFiatShamirAssumptionProfileId << "\",\"fiat_shamir_security_bits\":"
                << kFiatShamirSecurityBits << ",\"fiat_shamir_query_budget_max\":" << kFiatShamirQueryBudgetMax
                << ",\"fiat_shamir_random_oracle_terms_128bit\":1,\"fiat_shamir_challenge_count\":" << b.fiat_shamir_challenge_count
                << ",\"registry_assumption_id\":\"mechanism_schema_transcript_binding_v1\",\"registry_terms_128bit\":"
                << b.registry_assumption_terms << ",\"total_128bit_terms\":" << b.total_128bit_assumption_terms << "}"
                << ",\"identity_inventory\":[";
            for (std::size_t i = 0; i < b.identity_inventory.size(); ++i)
            {
                if (i != 0)
                    out << ',';
                out << "\"" << b.identity_inventory[i] << "\"";
            }
            out << "],\"fiat_shamir_challenge_inventory\":[";
            for (std::size_t i = 0; i < b.challenge_inventory.size(); ++i)
            {
                if (i != 0)
                    out << ',';
                out << "\"" << b.challenge_inventory[i] << "\"";
            }
            out << ']'
                << ",\"kzg_point_group_game_inventory\":[";
            for (std::size_t i = 0; i < b.kzg_game_inventory.size(); ++i)
            {
                if (i != 0)
                    out << ',';
                out << "\"" << b.kzg_game_inventory[i] << "\"";
            }
            out << ']'
                << ",\"quotient_mask_registry\":[";
            for (std::size_t i = 0; i < b.quotient_mask_inventory.size(); ++i)
            {
                if (i != 0)
                    out << ',';
                out << "\"" << b.quotient_mask_inventory[i] << "\"";
            }
            out << ']'
                << ",\"total_error_bound\":{\"denominator_lower_bound\":\"2^" << kFieldBoundBits << "\",\"numerator\":\"" << b.total_error_numerator
                << "\",\"target_numerator_threshold\":\"" << b.target_numerator_threshold << "\",\"meets_target\":true}"
                << ",\"mask_observation_ledger\":{\"chosen_rank\":" << kMechanismMaskRank << ",\"max_field_rows\":"
                << b.max_field_observation_rows << ",\"max_group_rows\":" << b.max_group_observation_rows
                << ",\"quotient_mask_rank\":" << kQuotientMaskRank << ",\"max_quotient_observations\":"
                << b.max_quotient_mask_observations << ",\"profiles\":[";
            for (std::size_t i = 0; i < b.observation_registry.size(); ++i)
            {
                if (i != 0)
                    out << ',';
                const auto &[label, rows] = b.observation_registry[i];
                require(label.find_first_of("\\\"") == std::string::npos, "mask observation label is not canonical JSON text");
                out << "{\"label\":\"" << label << "\",\"field_rows\":" << rows.field_rows
                    << ",\"group_rows\":" << rows.group_rows() << '}';
            }
            out << "],\"lineage_inventory\":[";
            for (std::size_t i = 0; i < b.observation_lineage_inventory.size(); ++i)
            {
                if (i != 0)
                    out << ',';
                const auto &lineage = b.observation_lineage_inventory[i];
                require(lineage.find_first_of("\\\"") == std::string::npos, "mask observation lineage is not canonical JSON text");
                out << "\"" << lineage << "\"";
            }
            out << "]},\"budget_ledger_sha256\":\"" << sha256_hex(b.canonical) << "\"}";
            return out.str();
        }
    }
}
