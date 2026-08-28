#include "gatzk/protocol/schema.hpp"

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/proof.hpp"
#include "gatzk/protocol/quotients.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <memory>
#include <openssl/bn.h>

// 实现正式 proof schema 的命名与顺序规则。任何标签拼接方式变化都会影响 trace、proof、artifact 与 verifier 的一致性。

namespace gatzk::protocol
{

    namespace
    {
        struct BnDeleter
        {
            void operator()(BIGNUM *value) const { BN_free(value); }
        };

        class ExactInteger
        {
        public:
            ExactInteger(std::uint64_t value = 0) : value_(BN_new())
            {
                if (value_ == nullptr || BN_set_word(value_.get(), value) != 1)
                    throw std::runtime_error("exact integer allocation failed");
            }

            explicit ExactInteger(const std::string &decimal) : value_(nullptr)
            {
                BIGNUM *raw = nullptr;
                if (BN_dec2bn(&raw, decimal.c_str()) == 0 || raw == nullptr)
                    throw std::runtime_error("exact integer decimal parse failed");
                value_.reset(raw);
            }

            ExactInteger(const ExactInteger &other)
                : value_(BN_dup(other.value_.get()))
            {
                if (value_ == nullptr)
                    throw std::runtime_error("exact integer copy failed");
            }

            ExactInteger(ExactInteger &&) noexcept = default;
            ExactInteger &operator=(ExactInteger other) noexcept
            {
                value_.swap(other.value_);
                return *this;
            }

            bool is_zero() const { return BN_is_zero(value_.get()); }
            bool is_positive() const
            {
                return !BN_is_negative(value_.get()) && !is_zero();
            }
            int compare(const ExactInteger &other) const
            {
                return BN_cmp(value_.get(), other.value_.get());
            }
            std::string decimal() const
            {
                char *raw = BN_bn2dec(value_.get());
                if (raw == nullptr)
                    throw std::runtime_error("exact integer formatting failed");
                std::string out(raw);
                OPENSSL_free(raw);
                return out;
            }

            friend ExactInteger operator+(
                const ExactInteger &lhs, const ExactInteger &rhs)
            {
                ExactInteger out;
                if (BN_add(out.value_.get(), lhs.value_.get(), rhs.value_.get()) != 1)
                    throw std::runtime_error("exact integer addition failed");
                return out;
            }
            friend ExactInteger operator-(
                const ExactInteger &lhs, const ExactInteger &rhs)
            {
                ExactInteger out;
                if (BN_sub(out.value_.get(), lhs.value_.get(), rhs.value_.get()) != 1)
                    throw std::runtime_error("exact integer subtraction failed");
                return out;
            }
            friend ExactInteger operator*(
                const ExactInteger &lhs, const ExactInteger &rhs)
            {
                ExactInteger out;
                std::unique_ptr<BN_CTX, decltype(&BN_CTX_free)> context(
                    BN_CTX_new(), BN_CTX_free);
                if (context == nullptr || BN_mul(out.value_.get(), lhs.value_.get(), rhs.value_.get(),
                                                 context.get()) != 1)
                    throw std::runtime_error("exact integer multiplication failed");
                return out;
            }
            friend ExactInteger operator/(
                const ExactInteger &lhs, const ExactInteger &rhs)
            {
                ExactInteger out;
                std::unique_ptr<BN_CTX, decltype(&BN_CTX_free)> context(
                    BN_CTX_new(), BN_CTX_free);
                if (context == nullptr || rhs.is_zero() || BN_div(out.value_.get(), nullptr, lhs.value_.get(), rhs.value_.get(), context.get()) != 1)
                    throw std::runtime_error("exact integer division failed");
                return out;
            }
            friend ExactInteger operator%(
                const ExactInteger &lhs, const ExactInteger &rhs)
            {
                ExactInteger out;
                std::unique_ptr<BN_CTX, decltype(&BN_CTX_free)> context(
                    BN_CTX_new(), BN_CTX_free);
                if (context == nullptr || rhs.is_zero() || BN_mod(out.value_.get(), lhs.value_.get(), rhs.value_.get(), context.get()) != 1)
                    throw std::runtime_error("exact integer remainder failed");
                return out;
            }
            friend ExactInteger operator<<(
                const ExactInteger &value, std::size_t bits)
            {
                if (bits > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    throw std::runtime_error("exact integer shift is too large");
                ExactInteger out;
                if (BN_lshift(out.value_.get(), value.value_.get(),
                              static_cast<int>(bits)) != 1)
                    throw std::runtime_error("exact integer shift failed");
                return out;
            }
            ExactInteger &operator/=(const ExactInteger &rhs)
            {
                *this = *this / rhs;
                return *this;
            }

        private:
            std::unique_ptr<BIGNUM, BnDeleter> value_;
        };

        ExactInteger exact_integer(std::size_t value)
        {
            if (value > std::numeric_limits<std::uint64_t>::max())
                throw std::runtime_error("exact integer input exceeds u64");
            return ExactInteger(static_cast<std::uint64_t>(value));
        }

        ExactInteger exact_gcd(ExactInteger lhs, ExactInteger rhs)
        {
            while (!rhs.is_zero())
            {
                const auto remainder = lhs % rhs;
                lhs = rhs;
                rhs = remainder;
            }
            return lhs;
        }

        struct ExactRational
        {
            ExactInteger numerator = ExactInteger(0);
            ExactInteger denominator = ExactInteger(1);

            void normalize()
            {
                if (!denominator.is_positive())
                    throw std::runtime_error(
                        "SecurityBudgetLedger rational denominator is not positive");
                const auto divisor = exact_gcd(numerator, denominator);
                if (!divisor.is_zero())
                {
                    numerator /= divisor;
                    denominator /= divisor;
                }
            }

            std::string canonical() const
            {
                return numerator.decimal() + "/" + denominator.decimal();
            }
        };

        ExactRational exact_add(ExactRational lhs, const ExactRational &rhs)
        {
            const auto common = exact_gcd(lhs.denominator, rhs.denominator);
            const auto lhs_scale = rhs.denominator / common;
            const auto rhs_scale = lhs.denominator / common;
            lhs.numerator = lhs.numerator * lhs_scale + rhs.numerator * rhs_scale;
            lhs.denominator = lhs.denominator * lhs_scale;
            lhs.normalize();
            return lhs;
        }

        bool exact_at_most_power_of_two(
            const ExactRational &value,
            std::size_t negative_exponent)
        {
            return (value.numerator << negative_exponent).compare(value.denominator) <= 0;
        }

        const ExactInteger &bls12_381_scalar_modulus()
        {
            static const ExactInteger modulus(
                "52435875175126190479447740508185965837690552500527637822603658699938581184513");
            return modulus;
        }

        bool canonical_field_less(
            const algebra::FieldElement &lhs,
            const algebra::FieldElement &rhs)
        {
            std::array<std::uint8_t, 32> lhs_bytes{};
            std::array<std::uint8_t, 32> rhs_bytes{};
            lhs.write_little_endian(lhs_bytes.data(), lhs_bytes.size());
            rhs.write_little_endian(rhs_bytes.data(), rhs_bytes.size());
            return std::lexicographical_compare(
                lhs_bytes.begin(), lhs_bytes.end(), rhs_bytes.begin(), rhs_bytes.end());
        }

        std::size_t next_power_of_two_checked(std::size_t value)
        {
            if (value == 0)
                return 1;
            std::size_t out = 1;
            while (out < value)
            {
                if (out > std::numeric_limits<std::size_t>::max() / 2)
                    throw std::runtime_error("binding shift domain size overflow");
                out *= 2;
            }
            return out;
        }

        std::optional<std::size_t> registered_domain_size(
            const ProtocolContext &context,
            const std::string &domain_name)
        {
            for (const auto &domain : {
                     context.domains.fh, context.domains.edge, context.domains.in,
                     context.domains.d, context.domains.cat, context.domains.c,
                     context.domains.n})
            {
                if (domain != nullptr && domain->name == domain_name)
                    return domain->size;
            }
            return std::nullopt;
        }

        algebra::FieldElement matrix_determinant(
            std::vector<std::vector<algebra::FieldElement>> matrix)
        {
            if (matrix.empty())
                return algebra::FieldElement::one();
            const auto size = matrix.size();
            for (const auto &row : matrix)
                if (row.size() != size)
                    throw std::runtime_error("rank certificate matrix is not square");
            auto determinant = algebra::FieldElement::one();
            for (std::size_t column = 0; column < size; ++column)
            {
                std::size_t pivot = column;
                while (pivot < size && matrix[pivot][column].is_zero())
                    ++pivot;
                if (pivot == size)
                    return algebra::FieldElement::zero();
                if (pivot != column)
                {
                    std::swap(matrix[pivot], matrix[column]);
                    determinant = -determinant;
                }
                const auto pivot_value = matrix[column][column];
                determinant *= pivot_value;
                const auto inverse = pivot_value.inv();
                for (std::size_t row = column + 1; row < size; ++row)
                {
                    if (matrix[row][column].is_zero())
                        continue;
                    const auto scale = matrix[row][column] * inverse;
                    for (std::size_t item = column; item < size; ++item)
                        matrix[row][item] -= scale * matrix[column][item];
                }
            }
            return determinant;
        }

        algebra::FieldElement domain_blinder_basis(
            std::size_t domain_size,
            std::size_t coefficient,
            const algebra::FieldElement &point)
        {
            return (point.pow(domain_size) - algebra::FieldElement::one()) * point.pow(coefficient);
        }

        std::string formal_observation_rank_certificate(
            const std::string &object_id,
            std::size_t domain_size,
            bool includes_domain_openings,
            bool includes_model_blinder_commitment,
            bool includes_first_relation_witness,
            bool includes_shift_observations,
            const algebra::FieldElement &relation_pivot,
            const algebra::FieldElement &shift_delta,
            std::size_t expected_field_rows,
            std::size_t expected_group_rows)
        {
            std::string first_minor_pair;
            for (std::uint64_t attempt = 0; attempt < 256; ++attempt)
            {
                const algebra::FieldElement trapdoor(1009 + 16 * attempt);
                const algebra::FieldElement z1(1013 + 16 * attempt);
                const algebra::FieldElement z2(1019 + 16 * attempt);
                const algebra::FieldElement y(1021 + 16 * attempt);
                if (trapdoor == z1 || trapdoor == z2 || trapdoor == y || trapdoor == y + shift_delta || ((includes_first_relation_witness || includes_shift_observations) && trapdoor == relation_pivot))
                    continue;

                using Row = std::function<algebra::FieldElement(std::size_t)>;
                const auto basis = [&](std::size_t a, const algebra::FieldElement &x)
                { return domain_blinder_basis(domain_size, a, x); };
                const auto secret_basis = [](
                                              std::size_t coefficient, const algebra::FieldElement &x)
                { return x.pow(coefficient); };
                std::vector<Row> field_rows;
                std::vector<Row> field_secret_rows;
                std::vector<Row> group_rows;
                std::vector<Row> group_secret_rows;
                if (!includes_model_blinder_commitment)
                {
                    group_rows.push_back(
                        [&](std::size_t a)
                        { return basis(a, trapdoor); });
                    group_secret_rows.push_back(
                        [&](std::size_t i)
                        { return secret_basis(i, trapdoor); });
                }
                if (includes_domain_openings)
                {
                    field_rows.push_back(
                        [&](std::size_t a)
                        { return basis(a, z1); });
                    field_rows.push_back(
                        [&](std::size_t a)
                        { return basis(a, z2); });
                    field_secret_rows.push_back(
                        [&](std::size_t i)
                        { return secret_basis(i, z1); });
                    field_secret_rows.push_back(
                        [&](std::size_t i)
                        { return secret_basis(i, z2); });
                    group_rows.push_back([&](std::size_t a)
                                         { return (basis(a, trapdoor) - basis(a, z1)) / (trapdoor - z1); });
                    group_rows.push_back([&](std::size_t a)
                                         { return (basis(a, trapdoor) - basis(a, z2)) / (trapdoor - z2); });
                    group_secret_rows.push_back([&](std::size_t i)
                                                { return (secret_basis(i, trapdoor) - secret_basis(i, z1)) / (trapdoor - z1); });
                    group_secret_rows.push_back([&](std::size_t i)
                                                { return (secret_basis(i, trapdoor) - secret_basis(i, z2)) / (trapdoor - z2); });
                }
                if (includes_model_blinder_commitment && !includes_domain_openings)
                {
                    group_rows.push_back([&](std::size_t a)
                                         { return trapdoor.pow(a); });
                    // [S_model] is an independent-randomness observation.  It
                    // carries no unblinded model coefficient direction after
                    // the public VK anchor is eliminated from the DAG.
                    group_secret_rows.push_back([](std::size_t)
                                                { return algebra::FieldElement::zero(); });
                }
                if (includes_first_relation_witness)
                {
                    // z_last is a native domain point, hence b_a(z_last)=0;
                    // this row is a public scalar multiple of the commitment
                    // row and is retained in the DAG but not the pivot minor.
                }
                if (includes_shift_observations)
                {
                    field_rows.push_back([&](std::size_t a)
                                         { return basis(a, y + shift_delta); });
                    field_secret_rows.push_back([&](std::size_t i)
                                                { return secret_basis(i, y + shift_delta); });
                    group_rows.push_back([&](std::size_t a)
                                         { return basis(a, trapdoor + shift_delta); });
                    group_secret_rows.push_back([&](std::size_t i)
                                                { return secret_basis(i, trapdoor + shift_delta); });
                    // Both shift-check witness rows remain in the expression
                    // DAG.  After the common u_rel reference and the exact
                    // Taylor-shift source map are substituted, neither adds a
                    // pivot beyond the source observations and shifted
                    // commitment at this transcript stage.
                    // The G_rel pivot row has source point
                    // relation_pivot+shift_delta=z_last in the second domain,
                    // so it is dependent on the shifted commitment row.
                }
                if (field_rows.size() != expected_field_rows || group_rows.size() != expected_group_rows || field_secret_rows.size() != field_rows.size() || group_secret_rows.size() != group_rows.size())
                    throw std::runtime_error(
                        "ObservationLedger dependency row accounting mismatch");

                auto minor = [](const std::vector<Row> &rows)
                {
                    std::vector<std::vector<algebra::FieldElement>> matrix(
                        rows.size(),
                        std::vector<algebra::FieldElement>(
                            rows.size(), algebra::FieldElement::zero()));
                    for (std::size_t row = 0; row < rows.size(); ++row)
                        for (std::size_t column = 0; column < rows.size(); ++column)
                            matrix[row][column] = rows[row](column);
                    return matrix_determinant(std::move(matrix));
                };
                auto rank = [](std::vector<std::vector<algebra::FieldElement>> matrix)
                {
                    if (matrix.empty())
                        return std::size_t{0};
                    const auto column_count = matrix.front().size();
                    for (const auto &row : matrix)
                        if (row.size() != column_count)
                            throw std::runtime_error(
                                "rank certificate matrix is ragged");
                    std::size_t pivot_row = 0;
                    for (std::size_t column = 0;
                         column < column_count && pivot_row < matrix.size();
                         ++column)
                    {
                        std::size_t pivot = pivot_row;
                        while (pivot < matrix.size() && matrix[pivot][column].is_zero())
                            ++pivot;
                        if (pivot == matrix.size())
                            continue;
                        std::swap(matrix[pivot_row], matrix[pivot]);
                        const auto inverse = matrix[pivot_row][column].inv();
                        for (auto &value : matrix[pivot_row])
                            value *= inverse;
                        for (std::size_t row = 0; row < matrix.size(); ++row)
                        {
                            if (row == pivot_row || matrix[row][column].is_zero())
                                continue;
                            const auto scale = matrix[row][column];
                            for (std::size_t item = column;
                                 item < column_count; ++item)
                                matrix[row][item] -=
                                    scale * matrix[pivot_row][item];
                        }
                        ++pivot_row;
                    }
                    return pivot_row;
                };
                auto minimal_prefix_certificate = [&](const std::vector<Row> &blind,
                                                      const std::vector<Row> &secret)
                {
                    if (blind.empty())
                        return false;
                    const auto short_prefix = blind.size() - 1;
                    std::vector<std::vector<algebra::FieldElement>> blind_matrix(
                        blind.size(),
                        std::vector<algebra::FieldElement>(
                            short_prefix, algebra::FieldElement::zero()));
                    std::vector<std::vector<algebra::FieldElement>> combined_matrix(
                        blind.size(),
                        std::vector<algebra::FieldElement>(
                            short_prefix + secret.size(),
                            algebra::FieldElement::zero()));
                    for (std::size_t row = 0; row < blind.size(); ++row)
                    {
                        for (std::size_t column = 0;
                             column < short_prefix; ++column)
                        {
                            const auto value = blind[row](column);
                            blind_matrix[row][column] = value;
                            combined_matrix[row][column] = value;
                        }
                        for (std::size_t column = 0;
                             column < secret.size(); ++column)
                            combined_matrix[row][short_prefix + column] =
                                secret[row](column);
                    }
                    return rank(std::move(combined_matrix)) > rank(std::move(blind_matrix));
                };
                const auto field_minor = minor(field_rows);
                const auto group_minor = minor(group_rows);
                const auto field_secret_minor = minor(field_secret_rows);
                const auto group_secret_minor = minor(group_secret_rows);
                const auto field_prefix_minimal = minimal_prefix_certificate(
                    field_rows, field_secret_rows);
                const auto group_prefix_minimal = minimal_prefix_certificate(
                    group_rows, group_secret_rows);
                if (attempt == 0)
                    first_minor_pair = field_minor.to_string() + "," + group_minor.to_string();
                if (field_minor.is_zero() || group_minor.is_zero())
                    continue;
                const auto max_rows = std::max(field_rows.size(), group_rows.size());
                const bool max_prefix_is_minimal =
                    (field_rows.size() == max_rows && field_prefix_minimal) || (group_rows.size() == max_rows && group_prefix_minimal);
                const bool random_only_minimum =
                    includes_model_blinder_commitment && !includes_domain_openings && max_rows == 1;
                if (max_rows != 0 && !max_prefix_is_minimal && !random_only_minimum)
                    continue;
                return "formal-full-row-rank-v2|specialization=" + trapdoor.to_string() + "," + z1.to_string() + "," + z2.to_string() + "," + y.to_string() + "|A_blind_field_minor=" + field_minor.to_string() + "|A_blind_group_minor=" + group_minor.to_string() + "|A_sec_field_minor=" + field_secret_minor.to_string() + "|A_sec_group_minor=" + group_secret_minor.to_string() + "|coverage=full-row-A_blind-implies-Im(A_sec)-subset" + "|field_qminus1_fails=" + std::to_string(field_prefix_minimal) + "|group_qminus1_fails=" + std::to_string(group_prefix_minimal) + "|minimality=" + (random_only_minimum ? std::string("nonzero-S_model-randomness") : std::string("rank([A_blind_prefix|A_sec])"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ">rank(A_blind_prefix)")) +
                       "|certificate=nonzero-specialization-implies-formal-nonidentity";
            }
            throw std::runtime_error(
                "failed to construct nonzero formal ObservationLedger minor: " + object_id + " first_minors=" + first_minor_pair);
        }
    }

    // block 顺序是 proof serializer / parser 与 verifier 共享的固定契约。
    const std::vector<std::string> &proof_block_order()
    {
        static const std::vector<std::string> order = {
            "M_pub",
            "Com_dyn",
            "S_route",
            "Eval_ext",
            "Eval_dom",
            "Com_quot",
            "Open_dom",
            "W_ext",
            "Pi_bind",
        };
        return order;
    }

    std::vector<BindingRelationMeta> build_binding_relation_metadata(
        const ProtocolContext &context)
    {
        std::vector<BindingRelationMeta> out;
        const auto canonical_matrix = [&](
                                          std::string relation_id,
                                          std::string matrix_object,
                                          std::size_t rows,
                                          std::size_t columns)
        {
            CanonicalMatrixRelationMeta meta{
                .relation_id = relation_id,
                .matrix_object = matrix_object,
                .canonical_domain = context.domains.n->name,
                .row_count = rows,
                .column_count = columns,
                .compressed_object = matrix_object + "_CanFold",
                .compression_challenge_label =
                    "xi_can:" + relation_id,
            };
            meta.column_objects.reserve(meta.column_count);
            for (std::size_t column = 0; column < meta.column_count; ++column)
                meta.column_objects.push_back(canonical_matrix_column_label(
                    meta.matrix_object, column));
            return meta;
        };
        const auto aggregation_route = [&](
                                           const std::string &relation_id,
                                           const std::string &prefix,
                                           std::string_view role,
                                           const std::string &index_object,
                                           const std::string &node_value_object,
                                           const std::string &edge_value_object)
        {
            const auto stem = prefix + std::string(role) + "_";
            AggregationRouteRelationMeta route{
                .relation_id = relation_id + ":" + std::string(role),
                .index_object = index_object,
                .node_value_object = node_value_object,
                .edge_value_object = edge_value_object,
                .table_object = stem + "Table",
                .query_object = stem + "Query",
                .multiplicity_object = std::string(role) == "src"
                                           ? "P_AggBind_m_src"
                                           : "P_AggBind_m_dst",
                .node_inverse_object = stem + "J_node",
                .edge_inverse_object = stem + "J_edge",
                .node_accumulator_object = stem + "R_node",
                .edge_accumulator_object = stem + "R_edge",
                .shifted_accumulator_object = stem + "R_edge_shifted",
                .relation_virtual_object = stem + "terminal_relation_virtual",
                .eta_challenge_label = "eta:" + relation_id + ":" + std::string(role),
                .beta_challenge_label = "beta:" + relation_id + ":" + std::string(role),
                .terminal_challenge_label = "y_rel:" + relation_id + ":" + std::string(role) + ":terminal",
                .node_terminal_point = context.domains.n->point_at(
                    context.local.num_nodes),
                .edge_terminal_point = context.domains.edge->point_at(
                    context.local.edges.size()),
            };
            route.terminal_delta =
                route.edge_terminal_point - route.node_terminal_point;
            route.configured_shift_degree = context.domains.edge->size + registered_blinder_coefficient_count(context, route.edge_accumulator_object) - 1;
            route.shift_domain_size = next_power_of_two_checked(
                2 * route.configured_shift_degree + 1);
            return route;
        };
        const auto aggregation_relation = [&](
                                              std::string relation_id,
                                              std::string prefix,
                                              std::string source_matrix,
                                              std::string target_matrix,
                                              std::string alpha_object,
                                              std::size_t columns)
        {
            AggregationRelationMeta relation{
                .relation_id = relation_id,
                .source_matrix_object = std::move(source_matrix),
                .target_matrix_object = std::move(target_matrix),
                .alpha_object = std::move(alpha_object),
                .node_domain = context.domains.n->name,
                .edge_domain = context.domains.edge->name,
                .row_count = context.local.num_nodes,
                .column_count = columns,
                .compression_challenge_label = "xi_agg_bind:" + relation_id,
                .source_node_object = prefix + "Src_node",
                .target_node_object = prefix + "Dst_node",
                .source_edge_object = prefix + "Src_edge",
                .target_edge_object = prefix + "Dst_edge",
                .product_object = prefix + "V",
                .psq_object = prefix + "PSQ",
            };
            relation.source_route = aggregation_route(
                relation_id, prefix, "src", "P_src",
                relation.source_node_object, relation.source_edge_object);
            relation.target_route = aggregation_route(
                relation_id, prefix, "dst", "P_dst",
                relation.target_node_object, relation.target_edge_object);
            relation.commitment_order = {
                relation.source_node_object,
                relation.target_node_object,
                relation.source_edge_object,
                relation.target_edge_object,
                relation.product_object,
                relation.psq_object,
            };
            const auto append_route = [&](const AggregationRouteRelationMeta &route)
            {
                for (const auto *object : {
                         &route.table_object, &route.query_object,
                         &route.node_inverse_object,
                         &route.edge_inverse_object,
                         &route.node_accumulator_object,
                         &route.edge_accumulator_object})
                    relation.commitment_order.push_back(*object);
                relation.commitment_order.push_back(
                    route.shifted_accumulator_object);
            };
            append_route(relation.source_route);
            append_route(relation.target_route);
            return relation;
        };
        const auto route_terminal_relation = [&](
                                                 std::string relation_id,
                                                 std::string node_accumulator,
                                                 std::string edge_accumulator)
        {
            RouteTerminalRelationMeta relation{
                .relation_id = std::move(relation_id),
                .node_accumulator_object = std::move(node_accumulator),
                .edge_accumulator_object = std::move(edge_accumulator),
                .node_domain = context.domains.n->name,
                .edge_domain = context.domains.edge->name,
                .node_terminal_point = context.domains.n->point_at(
                    context.domains.n->size - 1),
                .edge_terminal_point = context.domains.edge->point_at(
                    context.domains.edge->size - 1),
            };
            relation.shifted_accumulator_object =
                relation.edge_accumulator_object + "_RouteShifted";
            relation.relation_virtual_object =
                relation.node_accumulator_object + "_RouteTerminalVirtual";
            relation.challenge_label =
                "y_route_terminal:" + relation.relation_id;
            relation.terminal_delta = relation.edge_terminal_point - relation.node_terminal_point;
            relation.configured_shift_degree = context.domains.edge->size + registered_blinder_coefficient_count(context, relation.edge_accumulator_object) - 1;
            relation.shift_domain_size = next_power_of_two_checked(
                2 * relation.configured_shift_degree + 1);
            return relation;
        };
        out.push_back(BindingRelationMeta{
            .relation_id = "bind:feat",
            .object_kind = "feature",
            .source_objects = {"P_T_H", "I"},
            .target_objects = {"P_H"},
            .work_domains = {context.domains.fh->name},
            .real_lengths = {context.local.num_nodes * context.local.num_features},
            .quotient_relations = {"t_FH:feature_lookup"},
        });
        for (std::size_t layer = 0; layer < context.model.hidden_layers.size(); ++layer)
        {
            for (std::size_t head = 0;
                 head < context.model.hidden_layers[layer].heads.size();
                 ++head)
            {
                BindingRelationMeta meta{
                    .relation_id = "bind:hidden:" + std::to_string(layer + 1) + ":" + std::to_string(head),
                    .object_kind = "hidden",
                    .layer_id = layer + 1,
                    .head_id = head,
                    .source_objects = {hidden_head_prefix(layer, head)},
                    .target_objects = {hidden_head_prefix(layer, head) + "_H_agg"},
                    .work_domains = {
                        context.domains.in->name, context.domains.d->name,
                        context.domains.edge->name, context.domains.n->name},
                    .real_lengths = {context.model.d_in_profile.at(layer), context.model.hidden_layers[layer].shape.head_dim, context.local.edges.size(), context.local.num_nodes},
                    .quotient_relations = {"t_in:hidden_binding", "t_d_h:hidden_binding", "t_edge:hidden_binding", "t_N:hidden_binding"},
                };
                const auto prefix = hidden_head_prefix(layer, head);
                const auto width =
                    context.model.hidden_layers[layer].shape.head_dim;
                meta.canonical_matrices.push_back(canonical_matrix(
                    "canonical:hidden:H_prime:l" + std::to_string(layer)
                        + ":h" + std::to_string(head),
                    prefix + "_H_prime", context.local.num_nodes, width));
                meta.canonical_matrices.push_back(canonical_matrix(
                    "canonical:hidden:H_agg_pre:l" + std::to_string(layer)
                        + ":h" + std::to_string(head),
                    prefix + "_H_agg_pre", context.local.num_nodes, width));
                meta.canonical_matrices.push_back(canonical_matrix(
                    "canonical:hidden:H_agg:l" + std::to_string(layer)
                        + ":h" + std::to_string(head),
                    prefix + "_H_agg", context.local.num_nodes, width));
                meta.aggregation_relations.push_back(aggregation_relation(
                    "aggregation:hidden:l" + std::to_string(layer)
                        + ":h" + std::to_string(head),
                    hidden_aggregation_binding_label(layer, head, ""),
                    prefix + "_H_prime", prefix + "_H_agg_pre",
                    prefix + "_alpha", width));
                for (const auto *role : {"src", "dst", "t"})
                    meta.route_terminal_relations.push_back(
                        route_terminal_relation(
                            "route-terminal:hidden:l" + std::to_string(layer)
                                + ":h" + std::to_string(head) + ":" + role,
                            prefix + "_R_" + role + "_node",
                            prefix + "_R_" + role));
                out.push_back(std::move(meta));
            }
            BindingRelationMeta concat_meta{
                .relation_id = "bind:concat:" + std::to_string(layer + 1),
                .object_kind = "concat",
                .layer_id = layer + 1,
                .source_objects = {hidden_layer_concat_label(layer)},
                .target_objects = {hidden_layer_concat_star_label(layer)},
                .work_domains = {context.domains.cat->name, context.domains.n->name},
                .real_lengths = {
                    context.model.hidden_layers[layer].shape.head_count * context.model.hidden_layers[layer].shape.head_dim,
                    context.local.num_nodes},
                .quotient_relations = {"t_cat:concat_binding", "t_N:concat_binding"},
            };
            const auto concat_width =
                context.model.hidden_layers[layer].shape.head_count * context.model.hidden_layers[layer].shape.head_dim;
            concat_meta.canonical_matrices.push_back(canonical_matrix(
                "canonical:concat:" + std::to_string(layer + 1),
                hidden_layer_concat_label(layer),
                context.local.num_nodes, concat_width));
            out.push_back(std::move(concat_meta));
        }
        for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
        {
            BindingRelationMeta meta{
                .relation_id = "bind:output:" + std::to_string(head),
                .object_kind = "output",
                .layer_id = context.model.L,
                .head_id = head,
                .source_objects = {output_head_prefix(head)},
                .target_objects = {output_y_label(head)},
                .work_domains = {
                    context.domains.cat->name, context.domains.c->name,
                    context.domains.edge->name, context.domains.n->name},
                .real_lengths = {context.model.output_layer.input_dim, context.local.num_classes, context.local.edges.size(), context.local.num_nodes},
                .quotient_relations = {"t_cat:output_binding", "t_C:output_binding", "t_edge:output_binding", "t_N:output_binding"},
            };
            const auto matrix_prefix = output_head_prefix(head);
            meta.canonical_matrices.push_back(canonical_matrix(
                "canonical:output:Y_prime:" + std::to_string(head),
                matrix_prefix + "_Y_prime", context.local.num_nodes,
                context.local.num_classes));
            meta.canonical_matrices.push_back(canonical_matrix(
                "canonical:output:Y_pre:" + std::to_string(head),
                output_y_lin_label(head), context.local.num_nodes,
                context.local.num_classes));
            meta.canonical_matrices.push_back(canonical_matrix(
                "canonical:output:Y_head:" + std::to_string(head),
                output_y_label(head), context.local.num_nodes,
                context.local.num_classes));
            meta.aggregation_relations.push_back(aggregation_relation(
                "aggregation:output:" + std::to_string(head),
                output_aggregation_binding_label(head, ""),
                matrix_prefix + "_Y_prime",
                output_y_lin_label(head), matrix_prefix + "_alpha",
                context.local.num_classes));
            for (const auto *role : {"src", "dst", "t"})
                meta.route_terminal_relations.push_back(
                    route_terminal_relation(
                        "route-terminal:output:" + std::to_string(head) + ":" + role,
                        matrix_prefix + "_R_" + role + "_node",
                        matrix_prefix + "_R_" + role));
            const auto prefix = output_bias_view_prefix(head);
            const std::vector<std::string> logup_order = {
                output_bias_view_label(head, "Table_can"),
                output_bias_view_label(head, "Query"),
                output_bias_view_label(head, "m_can"),
                output_bias_view_label(head, "J_can"),
                output_bias_view_label(head, "J_qry"),
                output_bias_view_label(head, "R_can"),
                output_bias_view_label(head, "R_qry"),
            };
            meta.indexed_views.push_back(IndexedViewRelationMeta{
                .relation_id = "indexed-view:output-bias:" + std::to_string(head),
                .canonical_object = model_proof_local_label(output_bias_label(head)),
                .view_object = output_head_prefix(head) + "_Bias_flat",
                .canonical_domain = context.domains.c->name,
                .view_domain = context.domains.edge->name,
                .canonical_length = context.local.num_classes,
                .view_length = context.local.num_nodes * context.local.num_classes,
                .index_map_id = "row-major-q=i*C+c-to-c-v1",
                .canonical_selector_id = "Q_C_valid",
                .query_selector_id = "Q_NC_valid",
                .multiplicity_id = "m_bias_can[c]=N",
                .eta_challenge_label = output_bias_view_eta_label(head),
                .beta_challenge_label = output_bias_view_beta_label(head),
                .commitment_order = logup_order,
            });

            if (context.compiler_artifact == nullptr)
                throw std::runtime_error(
                    "coefficient basis RelationMeta requires ProtocolCompilerArtifact");
            const auto coefficient_object = output_bias_basis_label(head, "Coeff");
            const auto tail_length = registered_blinder_coefficient_count(
                context, coefficient_object);
            std::vector<std::size_t> semantic_positions(
                context.local.num_classes, 0);
            std::iota(
                semantic_positions.begin(), semantic_positions.end(),
                std::size_t{0});
            const auto region = coefficient_region_layout(
                context.local.num_classes, semantic_positions, tail_length,
                context.domains.edge->size);
            if (region.semantic_positions.back() >= region.chunks.front().length)
                throw std::runtime_error(
                    "output bias semantic coefficient view crosses the first chunk");
            std::vector<CoefficientBasisChunkMeta> basis_chunks;
            std::vector<std::string> basis_commitment_order = {
                coefficient_object};
            basis_chunks.reserve(region.chunks.size());
            for (const auto &chunk : region.chunks)
            {
                const auto object = output_bias_basis_chunk_label(
                    head, chunk.chunk_index);
                basis_chunks.push_back(CoefficientBasisChunkMeta{
                    .object_id = object,
                    .accumulator_object =
                        output_bias_basis_chunk_accumulator_label(
                            head, chunk.chunk_index),
                    .shifted_accumulator_object =
                        output_bias_basis_chunk_shifted_accumulator_label(
                            head, chunk.chunk_index),
                    .chunk_index = chunk.chunk_index,
                    .offset = chunk.offset,
                    .length = chunk.length,
                    .terminal_point = context.domains.edge->point_at(
                        chunk.length),
                    .configured_shift_degree = context.domains.edge->size + registered_blinder_coefficient_count(context, output_bias_basis_chunk_accumulator_label(head, chunk.chunk_index)) - 1,
                });
                basis_chunks.back().shift_domain_size =
                    next_power_of_two_checked(
                        2 * basis_chunks.back().configured_shift_degree + 1);
                basis_commitment_order.push_back(object);
            }
            const auto power_object = output_bias_basis_label(head, "Powers");
            for (const auto &chunk : basis_chunks)
                basis_commitment_order.push_back(chunk.accumulator_object);
            std::vector<std::string> opening_claim_order;
            for (const auto &chunk : basis_chunks)
            {
                opening_claim_order.push_back(
                    chunk.shifted_accumulator_object + "@relation-check");
                opening_claim_order.push_back(
                    chunk.accumulator_object + "@relation-check-plus-delta");
            }
            opening_claim_order.push_back(
                output_bias_basis_prefix(head) + "G_rel@gamma");
            meta.coefficient_basis_relations.push_back(
                CoefficientBasisRelationMeta{
                    .relation_id = "coefficient-basis:output-bias:" + std::to_string(head),
                    .canonical_object = output_head_prefix(head) + "_Bias_flat",
                    .coefficient_object = coefficient_object,
                    .carrier_domain = context.domains.edge->name,
                    .low_degree_length = region.low_degree_length,
                    .tail_length = region.tail_length,
                    .total_length = region.total_length,
                    .semantic_positions = region.semantic_positions,
                    .chunks = std::move(basis_chunks),
                    .power_object = power_object,
                    .gamma_challenge_label = output_bias_basis_gamma_label(head),
                    .relation_challenge_label =
                        output_bias_basis_relation_challenge_label(head),
                    .commitment_order = std::move(basis_commitment_order),
                    .opening_claim_order = std::move(opening_claim_order),
                });

            const auto z_c = context.domains.c->point_at(context.local.num_classes);
            const auto z_edge = context.domains.edge->point_at(
                context.local.num_nodes * context.local.num_classes);
            if (z_c == z_edge)
            {
                meta.shift_relations.push_back(KZGShiftRelationMeta{
                    .relation_id = "same-point:output-bias-terminal:" + std::to_string(head),
                    .source_linear_maps = {
                        "+1*" + output_bias_view_label(head, "R_can") + "-1*" + output_bias_view_label(head, "R_qry")},
                    .source_domains = {context.domains.c->name + "+" + context.domains.edge->name},
                    .canonical_points = {z_c},
                    .lambda0 = algebra::FieldElement::zero(),
                    .deltas = {algebra::FieldElement::zero()},
                    .source_value_kinds = {"MaskedPrivateValue"},
                    .challenge_label = output_bias_view_relation_challenge_label(head),
                    .exclusion_degree = 0,
                    .max_difference_degree = 0,
                    .configured_degrees = {},
                    .shift_domain_sizes = {},
                    .opening_claim_order = {prefix + "G_rel@same-point-pivot"},
                });
                out.push_back(std::move(meta));
                continue;
            }
            const bool c_first = canonical_field_less(z_c, z_edge);
            const auto first_point = c_first ? z_c : z_edge;
            const auto second_point = c_first ? z_edge : z_c;
            const auto second_domain = c_first ? context.domains.edge : context.domains.c;
            const auto shifted_source_label = c_first
                                                  ? output_bias_view_label(head, "R_qry")
                                                  : output_bias_view_label(head, "R_can");
            const auto kRegisteredBlinderCoefficients =
                registered_blinder_coefficient_count(context, shifted_source_label);
            const auto shift_degree = second_domain->size + kRegisteredBlinderCoefficients - 1;
            meta.shift_relations.push_back(KZGShiftRelationMeta{
                .relation_id = "shift:output-bias-terminal:" + std::to_string(head),
                .source_linear_maps = c_first
                                          ? std::vector<std::string>{
                                                "+1*" + output_bias_view_label(head, "R_can"),
                                                "-1*" + output_bias_view_label(head, "R_qry")}
                                          : std::vector<std::string>{"-1*" + output_bias_view_label(head, "R_qry"), "+1*" + output_bias_view_label(head, "R_can")},
                .source_domains = c_first ? std::vector<std::string>{context.domains.c->name, context.domains.edge->name} : std::vector<std::string>{context.domains.edge->name, context.domains.c->name},
                .canonical_points = {first_point, second_point},
                .lambda0 = algebra::FieldElement::zero(),
                .deltas = {algebra::FieldElement::zero(), second_point - first_point},
                .source_value_kinds = {"MaskedPrivateValue", "MaskedPrivateValue"},
                .challenge_label = output_bias_view_relation_challenge_label(head),
                .exclusion_degree = second_domain->size,
                // Soundness is compiled against the verifier-enforced global
                // KZG bound, never the honest prover's smaller shift resource.
                .max_difference_degree = context.kzg.manifest.degree_bound,
                .configured_degrees = {shift_degree},
                .shift_domain_sizes = {next_power_of_two_checked(2 * shift_degree + 1)},
                .opening_claim_order = {prefix + "R_shifted@relation-check", "F2@relation-check-plus-delta", "G_rel@pivot"},
            });
            out.push_back(std::move(meta));
        }
        BindingRelationMeta average_meta{
            .relation_id = "bind:output:avg",
            .object_kind = "average",
            .layer_id = context.model.L,
            .source_objects = {"output_head_family"},
            .target_objects = {"P_Y"},
            .work_domains = {
                context.domains.edge->name, context.domains.c->name,
                context.domains.n->name},
            .real_lengths = {context.local.num_nodes * context.local.num_classes},
            .quotient_relations = {"t_edge:output_average", "t_C:output_average", "t_N:public_output"},
        };
        average_meta.canonical_matrices.push_back(canonical_matrix(
            "canonical:output:Y_average_pre", "P_Y_lin",
            context.local.num_nodes, context.local.num_classes));
        average_meta.canonical_matrices.push_back(canonical_matrix(
            "canonical:output:Y_average", "P_Y",
            context.local.num_nodes, context.local.num_classes));
        out.push_back(std::move(average_meta));
        return out;
    }

    std::string opening_group_reference(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges,
        const algebra::FieldElement &point)
    {
        struct DomainPointSpec
        {
            const char *domain;
            const char *challenge;
            std::shared_ptr<algebra::RootOfUnityDomain> root_domain;
        };
        const std::vector<DomainPointSpec> domains = {
            {"FH", "z_FH", context.domains.fh},
            {"edge", "z_edge", context.domains.edge},
            {"in", "z_in", context.domains.in},
            {"d_h", "z_d_h", context.domains.d},
            {"cat", "z_cat", context.domains.cat},
            {"C", "z_C", context.domains.c},
            {"N", "z_N", context.domains.n},
        };
        for (const auto &spec : domains)
        {
            const auto challenge = challenges.find(spec.challenge);
            if (challenge == challenges.end() || spec.root_domain == nullptr)
            {
                throw std::runtime_error(
                    "point-group owner lookup is missing domain challenge");
            }
            if (point == challenge->second || point == challenge->second * spec.root_domain->omega)
            {
                return "Open_dom:" + std::string(spec.domain) + ":" + point.to_string();
            }
        }
        return "W_ext:" + point.to_string();
    }

    std::vector<MetadataReference> route_opening_references(
        const ProtocolContext &context,
        const std::map<std::string, algebra::FieldElement> &challenges)
    {
        std::vector<MetadataReference> references;
        const auto append_route = [&](const auto &route,
                                      const std::string &challenge_label)
        {
            const auto challenge = challenges.find(challenge_label);
            if (challenge == challenges.end())
                throw std::runtime_error(
                    "S_route is missing terminal challenge: " + challenge_label);
            references.push_back({opening_group_reference(
                context, challenges, challenge->second)});
            references.push_back({opening_group_reference(
                context,
                challenges,
                challenge->second + route.terminal_delta)});
            references.push_back({opening_group_reference(
                context, challenges, route.node_terminal_point)});
        };

        for (const auto &binding : build_binding_relation_metadata(context))
        {
            for (const auto &route : binding.route_terminal_relations)
                append_route(route, route.challenge_label);
            for (const auto &aggregation : binding.aggregation_relations)
                for (const auto *route : {
                         &aggregation.source_route,
                         &aggregation.target_route})
                    append_route(*route, route->terminal_challenge_label);
        }
        return references;
    }

    const KZGShiftRelationMeta *find_unique_shift_relation_by_kind(
        const BindingRelationMeta &metadata,
        std::string_view relation_kind)
    {
        const KZGShiftRelationMeta *match = nullptr;
        for (const auto &relation : metadata.shift_relations)
        {
            if (relation.relation_id.find(relation_kind) == std::string::npos)
                continue;
            if (match != nullptr)
                return nullptr;
            match = &relation;
        }
        return match;
    }

    ProofSchema build_proof_schema(const ProtocolContext &context)
    {
        ProofSchema schema;
        auto padding_mode = [](const std::string &tag)
        {
            if (tag.find("_Acc") != std::string::npos || tag.find("_PSQ") != std::string::npos || tag.find("_R_") != std::string::npos || tag.ends_with("_R") || tag.ends_with("C_max"))
                return std::string("HoldLastState");
            return std::string("ZeroPad");
        };
        auto add = [&](std::string tag, std::string domain, std::string owner_block,
                       std::string field_type, std::string padding,
                       PolynomialVisibility visibility, bool commitment, bool opening, bool quotient)
        {
            std::string blinding_basis = "None";
            if (visibility == PolynomialVisibility::PrivateWitness)
            {
                const bool derived = tag.ends_with("_S_model") || tag.ends_with("_BiasView_R_shifted") || tag.ends_with("_Acc_shifted") || tag.find("_ScoreMask_") != std::string::npos || tag.ends_with("_ExternalMask") || tag.find("_ModelBind_") != std::string::npos || (owner_block == "Com_quot" && tag.rfind("t_", 0) == 0);
                if (derived)
                    blinding_basis = "DerivedExpression";
                else if (owner_block == "Com_quot" && tag.rfind("R_t_", 0) == 0)
                    blinding_basis = "CoefficientMask";
                else
                    blinding_basis = "DomainVanishing";
            }
            schema.entries.push_back(ProofSchemaEntry{
                .tag = std::move(tag),
                .domain = std::move(domain),
                .owner_block = std::move(owner_block),
                .field_type = std::move(field_type),
                .padding_mode = std::move(padding),
                .owner_policy = opening ? "global_standard_point_group" : "none",
                .visibility = visibility,
                .opening_policy = context.config.opening_mode,
                .binding_mode = context.config.binding_mode,
                .blinding_basis = std::move(blinding_basis),
                .enters_commitment = commitment,
                .enters_opening = opening,
                .enters_quotient = quotient,
            });
        };
        for (const auto &label : dynamic_commitment_labels(context))
        {
            const auto visibility = polynomial_visibility(context, label);
            const auto domain = dynamic_commitment_domain_name(context, label);
            if (!domain.has_value())
            {
                throw std::runtime_error("ProofSchema has no domain for dynamic object: " + label);
            }
            const bool is_binding_reference =
                label.find("_ModelBind_") != std::string::npos || label.find("_ScoreMask_") != std::string::npos || label.ends_with("_ExternalMask");
            add(label, *domain, "Com_dyn", "Commitment", padding_mode(label),
                visibility, true, true, !is_binding_reference);
        }
        for (const auto &label : quotient_commitment_labels(context))
        {
            const auto base = label.rfind("R_t_", 0) == 0
                                  ? label.substr(2)
                                  : label;
            std::string domain;
            if (base == "t_FH")
                domain = context.domains.fh->name;
            else if (base == "t_edge")
                domain = context.domains.edge->name;
            else if (base == "t_in")
                domain = context.domains.in->name;
            else if (base == "t_d_h" || base == "t_d")
                domain = context.domains.d->name;
            else if (base == "t_cat")
                domain = context.domains.cat->name;
            else if (base == "t_C")
                domain = context.domains.c->name;
            else if (base == "t_N")
                domain = context.domains.n->name;
            else
                throw std::runtime_error(
                    "ProofSchema has no work domain for quotient object: " + label);
            add(label, std::move(domain), "Com_quot", "Commitment", "ZeroPad",
                PolynomialVisibility::PrivateWitness, true, true, true);
        }
        auto add_model = [&](const std::string &anchor, const std::string &domain,
                             bool opens_in_binding = false)
        {
            add(model_proof_local_label(anchor), domain, "Pi_bind.Com_modelReblind",
                "Commitment", "ZeroPad", PolynomialVisibility::PrivateWitness,
                true, opens_in_binding, opens_in_binding);
            add(model_reblind_witness_label(anchor), domain, "Pi_bind.Com_modelReblind",
                "Commitment", "NotApplicable", PolynomialVisibility::PrivateWitness,
                true, false, false);
        };
        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size();
             ++layer)
        {
            const auto &hidden_layer = context.model.hidden_layers[layer];
            for (std::size_t head = 0;
                 head < hidden_layer.heads.size();
                 ++head)
            {
                for (std::size_t column = 0;; ++column)
                {
                    const auto anchor = hidden_weight_column_label(
                        layer, head, column);
                    if (!context.static_commitments.contains(anchor))
                        break;
                    add_model(anchor, context.domains.in->name);
                }
                add_model(hidden_src_label(layer, head), context.domains.d->name);
                add_model(hidden_dst_label(layer, head), context.domains.d->name);
            }
        }
        for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
        {
            for (std::size_t column = 0; column < context.local.num_classes; ++column)
                add_model(output_weight_column_label(head, column), context.domains.cat->name);
            add_model(output_src_label(head), context.domains.c->name);
            add_model(output_dst_label(head), context.domains.c->name);
            add_model(output_bias_label(head), context.domains.c->name, true);
            for (const auto &[suffix, domain, visibility] : std::vector<std::tuple<
                     std::string, std::string, PolynomialVisibility>>{
                     {"Table_can", context.domains.c->name, PolynomialVisibility::PrivateWitness},
                     {"Query", context.domains.edge->name, PolynomialVisibility::PrivateWitness},
                     {"m_can", context.domains.c->name, PolynomialVisibility::Public},
                     {"J_can", context.domains.c->name, PolynomialVisibility::PrivateWitness},
                     {"J_qry", context.domains.edge->name, PolynomialVisibility::PrivateWitness},
                     {"R_can", context.domains.c->name, PolynomialVisibility::PrivateWitness},
                     {"R_qry", context.domains.edge->name, PolynomialVisibility::PrivateWitness}})
            {
                const auto label = output_bias_view_label(head, suffix);
                add(label, domain, "Pi_bind.Com_viewLogUp", "Commitment",
                    padding_mode(label), visibility, true, true, true);
            }
            const auto z_c = context.domains.c->point_at(context.local.num_classes);
            const auto z_edge = context.domains.edge->point_at(
                context.local.num_nodes * context.local.num_classes);
            if (!(z_c == z_edge))
                add(output_bias_view_label(head, "R_shifted"), "coef",
                    "Pi_bind.Com_rel", "Commitment", "FixedDegreeZeroPad",
                    PolynomialVisibility::PrivateWitness, true, true, false);

            // Coefficient-tail owners use a coefficient-basis mask, while the
            // H_edge chunks and fold accumulator use ordinary domain-vanishing
            // masks.  Three is the compiler's canonical base observation
            // prefix for an independently committed/opened private object;
            // build_binding_relation_metadata verifies the completed ledger
            // value before fixing the final region layout.
            constexpr std::size_t kCoefficientTailBaseObservationCount = 3;
            const auto coefficient_label = output_bias_basis_label(head, "Coeff");
            add(coefficient_label, "coef", "Pi_bind.Com_basis", "Commitment",
                "CoefficientTailSuffix", PolynomialVisibility::PrivateWitness,
                true, true, false);
            schema.entries.back().blinding_basis = "CoefficientTail";
            const auto chunks = coefficient_chunk_layout(
                context.local.num_classes + kCoefficientTailBaseObservationCount,
                context.domains.edge->size);
            for (const auto &chunk : chunks)
            {
                add(output_bias_basis_chunk_label(head, chunk.chunk_index),
                    context.domains.edge->name, "Pi_bind.Com_basis", "Commitment",
                    "CoefficientChunkZeroPad", PolynomialVisibility::PrivateWitness,
                    true, true, true);
                add(output_bias_basis_chunk_accumulator_label(
                        head, chunk.chunk_index),
                    context.domains.edge->name, "Pi_bind.Com_basis", "Commitment",
                    "HoldLastState", PolynomialVisibility::PrivateWitness,
                    true, true, true);
                add(output_bias_basis_chunk_shifted_accumulator_label(
                        head, chunk.chunk_index),
                    "coef", "Pi_bind.Com_rel", "Commitment",
                    "FixedDegreeZeroPad", PolynomialVisibility::PrivateWitness,
                    true, true, false);
            }
        }
        auto add_canonical_columns = [&](const std::string &matrix,
                                         std::size_t columns)
        {
            for (std::size_t column = 0; column < columns; ++column)
                add(canonical_matrix_column_label(matrix, column),
                    context.domains.n->name, "Pi_bind.Com_view",
                    "Commitment", "ZeroPad",
                    PolynomialVisibility::PrivateWitness,
                    true, true, true);
            add(matrix + "_CanFold", context.domains.n->name,
                "Pi_bind.Com_view", "Commitment", "ZeroPad",
                PolynomialVisibility::PrivateWitness,
                true, true, true);
        };
        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size(); ++layer)
        {
            const auto width =
                context.model.hidden_layers[layer].shape.head_dim;
            for (std::size_t head = 0;
                 head < context.model.hidden_layers[layer].heads.size();
                 ++head)
            {
                const auto prefix = hidden_head_prefix(layer, head);
                add_canonical_columns(prefix + "_H_prime", width);
                add_canonical_columns(prefix + "_H_agg_pre", width);
                add_canonical_columns(prefix + "_H_agg", width);
            }
            add_canonical_columns(
                hidden_layer_concat_label(layer),
                context.model.hidden_layers[layer].shape.head_count * width);
        }
        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size(); ++head)
        {
            add_canonical_columns(
                output_head_prefix(head) + "_Y_prime",
                context.local.num_classes);
            add_canonical_columns(
                output_y_lin_label(head),
                context.local.num_classes);
            add_canonical_columns(
                output_y_label(head),
                context.local.num_classes);
        }
        add_canonical_columns("P_Y_lin", context.local.num_classes);
        add_canonical_columns("P_Y", context.local.num_classes);
        const auto add_aggregation_binding = [&](const std::string &prefix)
        {
            const auto add_private = [&](
                                         const std::string &label,
                                         const std::string &domain,
                                         const std::string &padding,
                                         bool enters_quotient = true)
            {
                add(label, domain, "Pi_bind.Com_agg", "Commitment", padding,
                    PolynomialVisibility::PrivateWitness,
                    true, true, enters_quotient);
            };
            for (const auto *suffix : {"Src_node", "Dst_node"})
                add_private(prefix + suffix, context.domains.n->name, "ZeroPad");
            for (const auto *suffix : {"Src_edge", "Dst_edge", "V"})
                add_private(prefix + suffix, context.domains.edge->name, "ZeroPad");
            add_private(prefix + "PSQ", context.domains.edge->name, "HoldLastState");
            for (const auto *role : {"src", "dst"})
            {
                const auto stem = prefix + role + "_";
                add_private(stem + "Table", context.domains.n->name, "ZeroPad");
                add_private(stem + "Query", context.domains.edge->name, "ZeroPad");
                add_private(stem + "J_node", context.domains.n->name, "ZeroPad");
                add_private(stem + "J_edge", context.domains.edge->name, "ZeroPad");
                add_private(stem + "R_node", context.domains.n->name, "HoldLastState");
                add_private(stem + "R_edge", context.domains.edge->name, "HoldLastState");
                add_private(
                    stem + "R_edge_shifted", "coef", "FixedDegreeZeroPad",
                    false);
            }
        };
        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size();
             ++layer)
            for (std::size_t head = 0;
                 head < context.model.hidden_layers[layer].heads.size();
                 ++head)
                add_aggregation_binding(
                    hidden_aggregation_binding_label(layer, head, ""));
        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size(); ++head)
            add_aggregation_binding(
                output_aggregation_binding_label(head, ""));
        const auto add_route_terminal_shift = [&](const std::string &prefix)
        {
            for (const auto *role : {"src", "dst", "t"})
                add(
                    prefix + "_R_" + role + "_RouteShifted",
                    "coef", "Pi_bind.Com_rel", "Commitment",
                    "FixedDegreeZeroPad",
                    PolynomialVisibility::PrivateWitness,
                    true, true, false);
        };
        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size();
             ++layer)
            for (std::size_t head = 0;
                 head < context.model.hidden_layers[layer].heads.size();
                 ++head)
                add_route_terminal_shift(hidden_head_prefix(layer, head));
        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size(); ++head)
            add_route_terminal_shift(output_head_prefix(head));
        schema.private_plain_value_count = 0;
        return schema;
    }

    namespace
    {
        void append_canonical_string(std::string &out, std::string_view value)
        {
            const auto size = static_cast<std::uint64_t>(value.size());
            for (std::size_t i = 0; i < 8; ++i)
            {
                out.push_back(static_cast<char>((size >> (8U * i)) & 0xffU));
            }
            out.append(value);
        }

        void append_canonical_vector(
            std::string &out,
            std::string_view tag,
            const std::vector<std::string> &values)
        {
            append_canonical_string(out, tag);
            append_canonical_string(out, std::to_string(values.size()));
            for (const auto &value : values)
                append_canonical_string(out, value);
        }

        std::string ledger_string(const CompilerLedgerEntry &entry)
        {
            return entry.id + "|object=" + entry.object_or_challenge + "|domain=" + entry.domain + "|stage=" + entry.transcript_stage + "|expr=" + entry.expression_or_certificate + "|degree=" + std::to_string(entry.degree_bound) + "|queries=" + std::to_string(entry.query_bound) + "|B=" + std::to_string(entry.exclusion_degree) + "|multiplicity=" + std::to_string(entry.multiplicity);
        }

        std::string observation_dependency_string(
            const ObservationDependencyEdge &edge)
        {
            std::string out = edge.id + "|derived=" + edge.derived_object + "|stage=" + edge.transcript_stage + "|rule=" + edge.propagation_rule + "|fresh_randomness=" + std::to_string(edge.introduces_independent_randomness) + "|sources=";
            for (std::size_t index = 0; index < edge.source_objects.size(); ++index)
            {
                if (index != 0)
                    out += ',';
                out += edge.source_objects[index];
            }
            out += "|observations=";
            for (std::size_t index = 0;
                 index < edge.serialized_observations.size(); ++index)
            {
                if (index != 0)
                    out += ',';
                out += edge.serialized_observations[index];
            }
            return out;
        }

        std::string binding_relation_string(const BindingRelationMeta &meta)
        {
            std::string out;
            append_canonical_string(out, meta.relation_id);
            append_canonical_string(out, meta.object_kind);
            append_canonical_string(
                out, meta.layer_id.has_value() ? "layer:present" : "layer:none");
            if (meta.layer_id.has_value())
                append_canonical_string(out, std::to_string(*meta.layer_id));
            append_canonical_string(
                out, meta.head_id.has_value() ? "head:present" : "head:none");
            if (meta.head_id.has_value())
                append_canonical_string(out, std::to_string(*meta.head_id));
            append_canonical_vector(out, "source_objects", meta.source_objects);
            append_canonical_vector(out, "target_objects", meta.target_objects);
            append_canonical_vector(out, "work_domains", meta.work_domains);
            std::vector<std::string> real_lengths;
            real_lengths.reserve(meta.real_lengths.size());
            for (const auto value : meta.real_lengths)
                real_lengths.push_back(std::to_string(value));
            append_canonical_vector(out, "real_lengths", real_lengths);
            append_canonical_vector(
                out, "quotient_relations", meta.quotient_relations);

            append_canonical_string(out, "canonical_matrices");
            append_canonical_string(
                out, std::to_string(meta.canonical_matrices.size()));
            for (const auto &matrix : meta.canonical_matrices)
            {
                append_canonical_string(out, matrix.relation_id);
                append_canonical_string(out, matrix.matrix_object);
                append_canonical_string(out, matrix.canonical_domain);
                append_canonical_string(out, std::to_string(matrix.row_count));
                append_canonical_string(out, std::to_string(matrix.column_count));
                append_canonical_vector(
                    out, "column_objects", matrix.column_objects);
                append_canonical_string(out, matrix.compressed_object);
                append_canonical_string(
                    out, matrix.compression_challenge_label);
            }

            append_canonical_string(out, "aggregation_relations");
            append_canonical_string(
                out, std::to_string(meta.aggregation_relations.size()));
            for (const auto &relation : meta.aggregation_relations)
            {
                for (const auto *value : {
                         &relation.relation_id,
                         &relation.source_matrix_object,
                         &relation.target_matrix_object,
                         &relation.alpha_object,
                         &relation.node_domain,
                         &relation.edge_domain,
                         &relation.compression_challenge_label,
                         &relation.source_node_object,
                         &relation.target_node_object,
                         &relation.source_edge_object,
                         &relation.target_edge_object,
                         &relation.product_object,
                         &relation.psq_object})
                    append_canonical_string(out, *value);
                append_canonical_string(
                    out, std::to_string(relation.row_count));
                append_canonical_string(
                    out, std::to_string(relation.column_count));
                const auto append_route = [&](const auto &route)
                {
                    for (const auto *value : {
                             &route.relation_id,
                             &route.index_object,
                             &route.node_value_object,
                             &route.edge_value_object,
                             &route.table_object,
                             &route.query_object,
                             &route.multiplicity_object,
                             &route.node_inverse_object,
                             &route.edge_inverse_object,
                             &route.node_accumulator_object,
                             &route.edge_accumulator_object,
                             &route.shifted_accumulator_object,
                             &route.relation_virtual_object,
                             &route.eta_challenge_label,
                             &route.beta_challenge_label,
                             &route.terminal_challenge_label})
                        append_canonical_string(out, *value);
                    append_canonical_string(
                        out, route.node_terminal_point.to_string());
                    append_canonical_string(
                        out, route.edge_terminal_point.to_string());
                    append_canonical_string(
                        out, route.terminal_delta.to_string());
                    append_canonical_string(
                        out, std::to_string(route.configured_shift_degree));
                    append_canonical_string(
                        out, std::to_string(route.shift_domain_size));
                };
                append_route(relation.source_route);
                append_route(relation.target_route);
                append_canonical_vector(
                    out, "commitment_order", relation.commitment_order);
            }

            append_canonical_string(out, "route_terminal_relations");
            append_canonical_string(
                out, std::to_string(meta.route_terminal_relations.size()));
            for (const auto &relation : meta.route_terminal_relations)
            {
                for (const auto *value : {
                         &relation.relation_id,
                         &relation.node_accumulator_object,
                         &relation.edge_accumulator_object,
                         &relation.shifted_accumulator_object,
                         &relation.relation_virtual_object,
                         &relation.node_domain,
                         &relation.edge_domain,
                         &relation.challenge_label})
                    append_canonical_string(out, *value);
                append_canonical_string(
                    out, relation.node_terminal_point.to_string());
                append_canonical_string(
                    out, relation.edge_terminal_point.to_string());
                append_canonical_string(
                    out, relation.terminal_delta.to_string());
                append_canonical_string(
                    out, std::to_string(relation.configured_shift_degree));
                append_canonical_string(
                    out, std::to_string(relation.shift_domain_size));
            }

            append_canonical_string(out, "indexed_views");
            append_canonical_string(out, std::to_string(meta.indexed_views.size()));
            for (const auto &view : meta.indexed_views)
            {
                for (const auto *value : {
                         &view.relation_id, &view.canonical_object,
                         &view.view_object, &view.canonical_domain,
                         &view.view_domain, &view.index_map_id,
                         &view.canonical_selector_id, &view.query_selector_id,
                         &view.multiplicity_id, &view.eta_challenge_label,
                         &view.beta_challenge_label})
                    append_canonical_string(out, *value);
                append_canonical_string(
                    out, std::to_string(view.canonical_length));
                append_canonical_string(out, std::to_string(view.view_length));
                append_canonical_vector(
                    out, "commitment_order", view.commitment_order);
            }

            append_canonical_string(out, "coefficient_basis_relations");
            append_canonical_string(
                out, std::to_string(meta.coefficient_basis_relations.size()));
            for (const auto &relation : meta.coefficient_basis_relations)
            {
                for (const auto *value : {
                         &relation.relation_id, &relation.canonical_object,
                         &relation.coefficient_object, &relation.carrier_domain,
                         &relation.power_object,
                         &relation.gamma_challenge_label,
                         &relation.relation_challenge_label})
                    append_canonical_string(out, *value);
                append_canonical_string(
                    out, std::to_string(relation.low_degree_length));
                append_canonical_string(
                    out, std::to_string(relation.tail_length));
                append_canonical_string(
                    out, std::to_string(relation.total_length));
                std::vector<std::string> semantic_positions;
                for (const auto position : relation.semantic_positions)
                    semantic_positions.push_back(std::to_string(position));
                append_canonical_vector(
                    out, "semantic_positions", semantic_positions);
                append_canonical_string(out, "chunks");
                append_canonical_string(
                    out, std::to_string(relation.chunks.size()));
                for (const auto &chunk : relation.chunks)
                {
                    append_canonical_string(out, chunk.object_id);
                    append_canonical_string(out, chunk.accumulator_object);
                    append_canonical_string(
                        out, chunk.shifted_accumulator_object);
                    append_canonical_string(
                        out, std::to_string(chunk.chunk_index));
                    append_canonical_string(out, std::to_string(chunk.offset));
                    append_canonical_string(out, std::to_string(chunk.length));
                    append_canonical_string(
                        out, chunk.terminal_point.to_string());
                    append_canonical_string(
                        out, std::to_string(chunk.configured_shift_degree));
                    append_canonical_string(
                        out, std::to_string(chunk.shift_domain_size));
                }
                append_canonical_vector(
                    out, "commitment_order", relation.commitment_order);
                append_canonical_vector(
                    out, "opening_claim_order", relation.opening_claim_order);
            }

            append_canonical_string(out, "shift_relations");
            append_canonical_string(out, std::to_string(meta.shift_relations.size()));
            for (const auto &relation : meta.shift_relations)
            {
                append_canonical_string(out, relation.relation_id);
                append_canonical_vector(
                    out, "source_linear_maps", relation.source_linear_maps);
                append_canonical_vector(
                    out, "source_domains", relation.source_domains);
                std::vector<std::string> points;
                points.reserve(relation.canonical_points.size());
                for (const auto &point : relation.canonical_points)
                    points.push_back(point.to_string());
                append_canonical_vector(out, "canonical_points", points);
                append_canonical_string(out, relation.lambda0.to_string());
                std::vector<std::string> deltas;
                deltas.reserve(relation.deltas.size());
                for (const auto &delta : relation.deltas)
                    deltas.push_back(delta.to_string());
                append_canonical_vector(out, "deltas", deltas);
                append_canonical_vector(
                    out, "source_value_kinds", relation.source_value_kinds);
                append_canonical_string(out, relation.challenge_label);
                append_canonical_string(
                    out, std::to_string(relation.exclusion_degree));
                append_canonical_string(
                    out, std::to_string(relation.max_difference_degree));
                std::vector<std::string> degrees;
                for (const auto value : relation.configured_degrees)
                    degrees.push_back(std::to_string(value));
                append_canonical_vector(out, "configured_degrees", degrees);
                std::vector<std::string> shift_sizes;
                for (const auto value : relation.shift_domain_sizes)
                    shift_sizes.push_back(std::to_string(value));
                append_canonical_vector(out, "shift_domain_sizes", shift_sizes);
                append_canonical_vector(
                    out, "opening_claim_order", relation.opening_claim_order);
            }
            return out;
        }

        void populate_protocol_template(ProtocolCompilerArtifact &artifact)
        {
            artifact.proof_blocks = proof_block_order();
            artifact.serializer_tag_table = {
                "wire:v3:type-tagged-length-prefixed:no-trailing-bytes"
                "+M_pub:PublicMetadata[25]+Y_public",
                "Com_dyn:Commitment[]",
                "S_route:MetadataReference[]",
                "Eval_ext:(PublicValue|MaskedPrivateValue)[]",
                "Eval_dom:(PublicValue|MaskedPrivateValue)[]",
                "Com_quot:Commitment[]",
                "Open_dom:PointGroupOwner(group_id+point+witness)[]",
                "W_ext:PointGroupOwner(group_id+point+witness)[]",
                "Pi_bind:RelationMeta(indexed_view+KZGShiftRelationMeta)"
                "+Com_modelReblind+Com_view(ref|commitment)+Com_viewLogUp"
                "+Com_agg+Com_basis+Com_CRPC+Com_rel+Eval_bind+OpenRef_bind",
            };
            artifact.object_canonical_order = {
                "feature",
                "hidden:layer_then_head",
                "concat:layer",
                "output:head",
                "output:average",
            };
            artifact.transcript_stages = {
                "M_pub",
                "public_shape_and_registered_columns",
                "feature_lookup",
                "hidden_layers_in_layer_head_order",
                "hidden_concat_per_layer",
                "output_Y_prime_ordered_list",
                "output_heads_in_order",
                "output_bias_view:canonical+view_then_eta_then_TableQuerym"
                "_then_beta_then_JR_then_shift_commitment_then_y_rel",
                "signed_division_limb_lookups",
                "public_output_binding",
                "model_and_relation_binding_registry",
                "alpha_quot",
                "seven_quotient_commitments",
                "fourteen_domain_points",
                "standard_point_group_rho_per_distinct_point",
            };
            artifact.point_group_registry_template = {
                "global:one_group_owner_witness_per_canonical_field_point",
                "domain_points:FH,edge,in,d_h,cat,C,N:z,z_omega:all_distinct",
                "external_route_binding_at_domain_point:merge_into_Open_dom",
                "non_domain_point:owner=W_ext",
                "same_commitment_point_equal_value:deduplicate",
                "same_commitment_point_conflicting_value:reject_before_rho",
                "rho:independent_nonzero_per_distinct_point",
                "rho_tuple:claim_id+object_id+commitment_reference"
                "+commitment_point+evaluation_kind+claim_role+visibility_tag"
                "+value_type+value_reference_id+resolved_canonical_value",
                "canonical_order:field_point_then_full_claim_tuple",
                "route_and_Pi_bind:MetadataReference_only",
                "public_output:single_group_owner",
            };
            artifact.verifier_equation_registry = {
                "metadata_and_public_statement_exact_match",
                "global_standard_point_group_openings_one_equation_per_point",
                "seven_masked_quotient_identities",
                "shifted_kzg_model_divisibility",
                "indexed_view_logup",
                "coefficient_region_and_basis_link",
                "crpc_same_point_identity",
                "kzg_shift_relation",
                "aggregation_and_bias_global_quotient",
                "public_logits_commitment_binding",
                "three_ledger_canonical_gate",
                "private_payload_leakage_gate",
            };
            artifact.ledger_templates = {
                "ObservationLedger:v2:formal_T_and_FS_coordinates"
                ":field_group_direct_sum:shared_blinder_dependency_DAG"
                ":canonical_RREF:Im(A_sec)_subset_Im(A_blind)"
                ":lexicographic_qmax:BadCRS_obs_certificates",
                "RandomIdentityLedger:v2:verifier_expression_DAG"
                ":challenge_stage:canonical_exclusion_polynomial"
                ":global_Dmax_degree:multiplicity",
                "SecurityBudgetLedger:v2:canonical_exact_rational"
                ":Q_FS_max:KZG+register+RO+algebraic+abort+CRS",
            };
            artifact.kzg_assumption_profile = "shifted_kzg_divisibility_v1";
            artifact.shift_algorithm_id = "coefficient_taylor_shift";
            artifact.shift_algorithm_spec =
                "coefficient_taylor_shift:v1|fixed_d_shift_cfg_plus_one"
                "|factorial_inverse_factorial_0_through_Dmax"
                "|reverse_convolution|n_shift=next_pow2(2*d_shift+1)"
                "|canonical_root_derivation|PK_coef_fixed_length_MSM"
                "|cache_key=proof_nonce+object_id|no_cross_proof_private_cache";

            std::string protocol_bytes;
            append_canonical_vector(protocol_bytes, "proof_blocks", artifact.proof_blocks);
            append_canonical_vector(protocol_bytes, "serializer_tags", artifact.serializer_tag_table);
            append_canonical_vector(protocol_bytes, "object_order", artifact.object_canonical_order);
            append_canonical_vector(protocol_bytes, "transcript_stages", artifact.transcript_stages);
            append_canonical_vector(protocol_bytes, "point_groups", artifact.point_group_registry_template);
            append_canonical_vector(protocol_bytes, "verifier_equations", artifact.verifier_equation_registry);
            append_canonical_vector(protocol_bytes, "ledger_templates", artifact.ledger_templates);
            append_canonical_string(protocol_bytes, artifact.kzg_assumption_profile);
            append_canonical_string(protocol_bytes, artifact.shift_algorithm_id);
            append_canonical_string(protocol_bytes, artifact.shift_algorithm_spec);
            artifact.protocol_canonical_bytes = std::move(protocol_bytes);
        }
    }

    std::string protocol_template_canonical_bytes()
    {
        ProtocolCompilerArtifact artifact;
        populate_protocol_template(artifact);
        return artifact.protocol_canonical_bytes;
    }

    std::string protocol_proof_schema_template_canonical_bytes()
    {
        ProtocolCompilerArtifact artifact;
        populate_protocol_template(artifact);
        std::string out;
        append_canonical_vector(out, "proof_blocks", artifact.proof_blocks);
        append_canonical_vector(out, "serializer_tags", artifact.serializer_tag_table);
        append_canonical_vector(out, "object_order", artifact.object_canonical_order);
        append_canonical_vector(
            out, "point_group_registry", artifact.point_group_registry_template);
        append_canonical_vector(
            out, "verifier_equations", artifact.verifier_equation_registry);
        return out;
    }

    std::string protocol_observation_ledger_template_canonical_bytes()
    {
        ProtocolCompilerArtifact artifact;
        populate_protocol_template(artifact);
        std::string out;
        append_canonical_string(out, artifact.ledger_templates.at(0));
        append_canonical_vector(
            out, "transcript_stages", artifact.transcript_stages);
        append_canonical_vector(
            out, "verifier_equations", artifact.verifier_equation_registry);
        append_canonical_string(out, artifact.shift_algorithm_spec);
        return out;
    }

    std::string protocol_random_identity_ledger_template_canonical_bytes()
    {
        ProtocolCompilerArtifact artifact;
        populate_protocol_template(artifact);
        std::string out;
        append_canonical_string(out, artifact.ledger_templates.at(1));
        append_canonical_vector(
            out, "transcript_stages", artifact.transcript_stages);
        append_canonical_vector(
            out, "verifier_equations", artifact.verifier_equation_registry);
        append_canonical_string(out, artifact.shift_algorithm_spec);
        return out;
    }

    std::string protocol_security_budget_ledger_template_canonical_bytes()
    {
        ProtocolCompilerArtifact artifact;
        populate_protocol_template(artifact);
        std::string out;
        append_canonical_string(out, artifact.ledger_templates.at(2));
        append_canonical_string(out, artifact.kzg_assumption_profile);
        append_canonical_vector(
            out, "verifier_equations", artifact.verifier_equation_registry);
        return out;
    }

    ProtocolCompilerArtifact build_protocol_compiler_artifact(const ProtocolContext &context)
    {
        ProtocolCompilerArtifact artifact;
        artifact.proof_schema = build_proof_schema(context);
        populate_protocol_template(artifact);

        std::set<std::string> model_reblind_witnesses;
        for (const auto &entry : artifact.proof_schema.entries)
        {
            if (entry.tag.ends_with("_S_model"))
                model_reblind_witnesses.insert(entry.tag);
        }
        std::vector<std::pair<std::string, std::vector<std::string>>>
            derived_model_binding_dependencies;
        std::vector<std::pair<std::string, std::vector<std::string>>>
            derived_score_mask_dependencies;
        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size();
             ++layer)
        {
            const auto &hidden_layer = context.model.hidden_layers[layer];
            for (std::size_t head = 0;
                 head < hidden_layer.heads.size();
                 ++head)
            {
                const auto prefix = hidden_head_prefix(layer, head);
                std::vector<std::string> projection_dependencies = {
                    prefix + "_b_proj"};
                for (std::size_t column = 0;; ++column)
                {
                    const auto anchor = hidden_weight_column_label(
                        layer, head, column);
                    if (!context.static_commitments.contains(anchor))
                        break;
                    projection_dependencies.push_back(model_proof_local_label(anchor));
                }
                derived_model_binding_dependencies.push_back({
                    hidden_model_binding_blinder_label(layer, head, "proj"),
                    std::move(projection_dependencies)});
                derived_model_binding_dependencies.push_back({
                    hidden_model_binding_blinder_label(layer, head, "src"),
                    {prefix + "_b_src",
                     model_proof_local_label(hidden_src_label(layer, head))}});
                derived_model_binding_dependencies.push_back({
                    hidden_model_binding_blinder_label(layer, head, "dst"),
                    {prefix + "_b_dst",
                     model_proof_local_label(hidden_dst_label(layer, head))}});
                derived_score_mask_dependencies.push_back({
                    hidden_score_mask_label(layer, head, "src"),
                    {prefix + "_E_src"}});
                derived_score_mask_dependencies.push_back({
                    hidden_score_mask_label(layer, head, "dst"),
                    {prefix + "_E_dst"}});
                for (const auto *suffix : {
                         "_H_star", "_H_agg_pre_star", "_H_agg_star"})
                {
                    const auto source = prefix + suffix;
                    derived_score_mask_dependencies.push_back({external_mask_label(source), {source}});
                }
            }
        }
        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size(); ++layer)
        {
            const auto source = hidden_layer_concat_star_label(layer);
            derived_score_mask_dependencies.push_back({external_mask_label(source), {source}});
        }
        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size(); ++head)
        {
            std::vector<std::string> projection_dependencies = {
                output_head_prefix(head) + "_b_proj"};
            for (std::size_t column = 0;
                 column < context.local.num_classes; ++column)
                projection_dependencies.push_back(model_proof_local_label(
                    output_weight_column_label(head, column)));
            derived_model_binding_dependencies.push_back({output_model_binding_blinder_label(head, "proj"),
                                                          std::move(projection_dependencies)});
            derived_model_binding_dependencies.push_back({output_model_binding_blinder_label(head, "src"),
                                                          {output_head_prefix(head) + "_b_src",
                                                           model_proof_local_label(output_src_label(head))}});
            derived_model_binding_dependencies.push_back({output_model_binding_blinder_label(head, "dst"),
                                                          {output_head_prefix(head) + "_b_dst",
                                                           model_proof_local_label(output_dst_label(head))}});
            derived_score_mask_dependencies.push_back({output_score_mask_label(head, "src"),
                                                       {output_head_prefix(head) + "_E_src"}});
            derived_score_mask_dependencies.push_back({output_score_mask_label(head, "dst"),
                                                       {output_head_prefix(head) + "_E_dst"}});
            const auto output_star =
                output_head_prefix(head) + "_Y_star";
            derived_score_mask_dependencies.push_back({external_mask_label(output_star), {output_star}});
        }
        auto append_dependency = [&](ObservationDependencyEdge edge)
        {
            std::sort(edge.source_objects.begin(), edge.source_objects.end());
            edge.source_objects.erase(
                std::unique(edge.source_objects.begin(), edge.source_objects.end()),
                edge.source_objects.end());
            artifact.observation_dependency_dag.push_back(std::move(edge));
        };
        for (const auto &[derived, sources] : derived_model_binding_dependencies)
        {
            append_dependency(ObservationDependencyEdge{
                .id = "obs-dep:" + derived,
                .derived_object = derived,
                .source_objects = sources,
                .serialized_observations = {
                    "Commitment", "MaskedPrivateValue@z",
                    "MaskedPrivateValue@z_omega",
                    "OpeningWitness@canonical-point-group"},
                .transcript_stage = "Com_dyn->Eval_dom->Open_dom",
                .propagation_rule = "linear-blinder-difference:source-view-minus-proof-local-model-terms",
                .introduces_independent_randomness = false,
            });
        }
        for (const auto &[derived, sources] : derived_score_mask_dependencies)
        {
            append_dependency(ObservationDependencyEdge{
                .id = "obs-dep:" + derived,
                .derived_object = derived,
                .source_objects = sources,
                .serialized_observations = {
                    "Commitment", "MaskedPrivateValue@external-point",
                    "OpeningWitness@external-point-group"},
                .transcript_stage = "Com_dyn->Eval_ext->Open_ext",
                .propagation_rule = "zero-domain-external-mask:P_committed=P_sem+Z_N*Mask",
                .introduces_independent_randomness = false,
            });
        }
        for (const auto &entry : artifact.proof_schema.entries)
        {
            if (!entry.tag.ends_with("_S_model"))
                continue;
            const auto proof_label = entry.tag.substr(
                                         0, entry.tag.size() - std::string("_S_model").size()) +
                                     "_proof";
            append_dependency(ObservationDependencyEdge{
                .id = "obs-dep:" + entry.tag,
                .derived_object = entry.tag,
                .source_objects = {proof_label},
                .serialized_observations = {"Commitment"},
                .transcript_stage = "Pi_bind.Com_modelReblind",
                .propagation_rule =
                    "shifted-kzg-divisibility:P_proof-P_vk=Z_D*S_model;same-random-coordinates",
                .introduces_independent_randomness = false,
            });
        }
        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size(); ++head)
        {
            const auto chunks = coefficient_chunk_layout(
                context.local.num_classes + 3,
                context.domains.edge->size);
            for (const auto &chunk : chunks)
            {
                const auto basis_shifted =
                    output_bias_basis_chunk_shifted_accumulator_label(
                        head, chunk.chunk_index);
                append_dependency(ObservationDependencyEdge{
                    .id = "obs-dep:" + basis_shifted,
                    .derived_object = basis_shifted,
                    .source_objects = {
                        output_bias_basis_chunk_accumulator_label(
                            head, chunk.chunk_index)},
                    .serialized_observations = {"Commitment", "MaskedPrivateValue@relation-check", "OpeningWitness@relation-check", "OpeningWitness@relation-check-plus-delta"},
                    .transcript_stage = "Pi_bind.Com_rel->Eval_bind->W_ext",
                    .propagation_rule = "coefficient-taylor-shift:transcript-gamma-to-chunk-terminal"
                                        ":shared-source-random-coordinates",
                    .introduces_independent_randomness = false,
                });
            }
            const auto shifted = output_bias_view_label(head, "R_shifted");
            if (std::none_of(
                    artifact.proof_schema.entries.begin(),
                    artifact.proof_schema.entries.end(),
                    [&](const ProofSchemaEntry &entry)
                    { return entry.tag == shifted; }))
                continue;
            const auto z_c = context.domains.c->point_at(context.local.num_classes);
            const auto z_edge = context.domains.edge->point_at(
                context.local.num_nodes * context.local.num_classes);
            const auto source = canonical_field_less(z_c, z_edge)
                                    ? output_bias_view_label(head, "R_qry")
                                    : output_bias_view_label(head, "R_can");
            append_dependency(ObservationDependencyEdge{
                .id = "obs-dep:" + shifted,
                .derived_object = shifted,
                .source_objects = {source},
                .serialized_observations = {
                    "Commitment", "MaskedPrivateValue@relation-check",
                    "OpeningWitness@relation-check",
                    "OpeningWitness@relation-check-plus-delta"},
                .transcript_stage = "Pi_bind.Com_rel->Eval_bind->W_ext",
                .propagation_rule = "coefficient-taylor-shift:shared-source-random-coordinates",
                .introduces_independent_randomness = false,
            });
        }
        for (const auto &entry : artifact.proof_schema.entries)
        {
            if (entry.enters_quotient)
            {
                artifact.constraint_registry.push_back(
                    "constraint:" + entry.domain + ":" + entry.tag);
            }
            artifact.degree_registry.push_back(
                entry.tag + "|domain=" + entry.domain + "|D_max=" + std::to_string(context.kzg.manifest.degree_bound));
            artifact.padding_registry.push_back(
                entry.tag + "|domain=" + entry.domain + "|rule=selector-zero-or-hold-last-as-registered");
            if (entry.visibility == PolynomialVisibility::PrivateWitness)
            {
                const bool is_derived_masked_quotient =
                    entry.owner_block == "Com_quot" && entry.tag.rfind("t_", 0) == 0;
                const bool is_independent_quotient_mask =
                    entry.owner_block == "Com_quot" && entry.tag.rfind("R_t_", 0) == 0;
                // t_D^zk is not a private base object with a fresh blinder.  It
                // is uniquely derived as t_D + Z_D R_t,D, so giving it a
                // separate ObservationLedger row would invent randomness that
                // the prover never samples.  The independent owner is R_t,D;
                // the source-object dependencies remain attached to every
                // private constraint input in the same domain below.
                if (is_derived_masked_quotient)
                    continue;
                if (is_independent_quotient_mask)
                {
                    std::vector<std::string> sources;
                    for (const auto &source : artifact.proof_schema.entries)
                    {
                        if (source.owner_block != "Com_quot" && source.visibility == PolynomialVisibility::PrivateWitness && source.enters_quotient && source.domain == entry.domain)
                        {
                            sources.push_back(source.tag);
                        }
                    }
                    std::sort(sources.begin(), sources.end());
                    std::string certificate =
                        "independent-quotient-mask-v1|derived-object=" + entry.tag.substr(2) + "|identity=t_zk=t+Z_D*R_t" + "|t=constraint_accumulator_D/Z_D" + "|field-evaluations-eliminated-by-verifier-quotient-DAG" + "|source-objects=";
                    if (sources.empty())
                    {
                        certificate += "none(public-only-or-zero-constraint-accumulator)";
                    }
                    for (std::size_t i = 0; i < sources.size(); ++i)
                    {
                        if (i != 0)
                            certificate += ',';
                        certificate += sources[i];
                    }
                    auto quotient_sources = sources;
                    quotient_sources.push_back(entry.tag);
                    append_dependency(ObservationDependencyEdge{
                        .id = "obs-dep:" + entry.tag.substr(2),
                        .derived_object = entry.tag.substr(2),
                        .source_objects = std::move(quotient_sources),
                        .serialized_observations = {
                            "Commitment", "MaskedPrivateValue@z_D",
                            "OpeningWitness@domain-point-group"},
                        .transcript_stage = "Com_quot->Eval_dom->Open_dom",
                        .propagation_rule = "global-quotient:t_D_zk=constraint_accumulator_D/Z_D+Z_D*R_t_D",
                        .introduces_independent_randomness = false,
                    });
                    // This is the currently registered independent mask
                    // resource.  The prover consumes this value through the
                    // same lookup used for all other blinder owners; it is no
                    // longer a hidden implementation constant.
                    constexpr std::size_t kQuotientMaskCoefficientCount = 8;
                    const auto quotient_domain_size =
                        registered_domain_size(context, entry.domain).value_or(0);
                    if (quotient_domain_size > std::numeric_limits<std::size_t>::max() - kQuotientMaskCoefficientCount)
                        throw std::runtime_error(
                            "quotient mask BadCRS degree overflow: " + entry.tag);
                    artifact.observation_ledger.push_back({
                        .id = "obs:" + entry.tag,
                        .object_or_challenge = entry.tag,
                        .domain = entry.domain,
                        .transcript_stage = entry.owner_block,
                        .expression_or_certificate = std::move(certificate),
                        .degree_bound = context.kzg.manifest.degree_bound,
                        .query_bound = kQuotientMaskCoefficientCount,
                        .exclusion_degree = quotient_domain_size + kQuotientMaskCoefficientCount,
                        .multiplicity = 1,
                    });
                    continue;
                }
                const bool is_derived_model_binding = std::any_of(
                    derived_model_binding_dependencies.begin(),
                    derived_model_binding_dependencies.end(),
                    [&](const auto &dependency)
                    { return dependency.first == entry.tag; });
                const bool is_derived_score_mask = std::any_of(
                    derived_score_mask_dependencies.begin(),
                    derived_score_mask_dependencies.end(),
                    [&](const auto &dependency)
                    { return dependency.first == entry.tag; });
                // _S_model, R_shifted and ModelBind_R use randomness from
                // their registered source objects.  They are observations in
                // those base-object certificates, never independent owners.
                if (entry.tag.ends_with("_S_model") || entry.tag.ends_with("_BiasView_R_shifted") || entry.tag.ends_with("_Acc_shifted") || is_derived_model_binding || is_derived_score_mask)
                {
                    continue;
                }
                std::size_t field_rows = entry.enters_opening ? 2 : 0;
                std::size_t group_rows = 1 + (entry.enters_opening ? 2 : 0);
                bool includes_model_blinder_commitment = false;
                bool includes_first_relation_witness = false;
                bool includes_shift_observations = false;
                auto relation_pivot = algebra::FieldElement::zero();
                auto shift_delta = algebra::FieldElement::zero();
                std::string certificate =
                    "canonical-rref-prefix-v1:commitment" + std::string(entry.enters_opening
                                                                            ? "+domain-eval(z,z_omega)+point-group-witness(z,z_omega)"
                                                                            : "");
                for (const auto &[derived, dependencies] :
                     derived_model_binding_dependencies)
                {
                    if (std::find(
                            dependencies.begin(), dependencies.end(), entry.tag) == dependencies.end())
                        continue;
                    certificate +=
                        "+derived-model-binding-observations(" + derived + ":commitment+domain-eval(z,z_omega)"
                                                                           "+point-group-witness(z,z_omega)"
                                                                           ":global-cross-source-DAG)";
                }
                for (const auto &[derived, dependencies] :
                     derived_score_mask_dependencies)
                {
                    if (std::find(
                            dependencies.begin(), dependencies.end(), entry.tag) == dependencies.end())
                        continue;
                    certificate +=
                        "+derived-external-mask-observations(" + derived + ":commitment+external-eval(same-point)"
                                                                           "+external-point-group-witness"
                                                                           ":global-cross-source-DAG)";
                }
                if (entry.enters_quotient)
                {
                    certificate +=
                        "+global-quotient-DAG-source(domain=" + entry.domain + ":constraint-accumulator->t_zk)";
                }
                if (entry.tag.ends_with("_proof") && model_reblind_witnesses.contains(
                                                         entry.tag.substr(0, entry.tag.size() - 6) + "_S_model"))
                {
                    includes_model_blinder_commitment = true;
                    // The proof-local commitment is anchor + Z_D*S_model.
                    // Its anchor is already a verifier-known group element, so
                    // it contributes no new A_sec direction and is eliminated
                    // through the public expression DAG before RREF.
                    if (entry.enters_opening)
                        --group_rows;
                    certificate +=
                        "+shared-source-S_model-commitment"
                        "+derived-proof-commitment-eliminated-against-public-anchor"
                        "+S_model-row-eliminated-by-shifted-divisibility-DAG";
                }
                for (std::size_t head = 0;
                     head < context.model.output_layer.heads.size(); ++head)
                {
                    const auto z_c = context.domains.c->point_at(context.local.num_classes);
                    const auto z_edge = context.domains.edge->point_at(
                        context.local.num_nodes * context.local.num_classes);
                    const bool c_first = canonical_field_less(z_c, z_edge);
                    const auto first_source = c_first
                                                  ? output_bias_view_label(head, "R_can")
                                                  : output_bias_view_label(head, "R_qry");
                    const auto shifted_source = c_first
                                                    ? output_bias_view_label(head, "R_qry")
                                                    : output_bias_view_label(head, "R_can");
                    if (!(z_c == z_edge) && entry.tag == first_source)
                    {
                        includes_first_relation_witness = true;
                        relation_pivot = c_first ? z_c : z_edge;
                        certificate +=
                            "+G_rel-pivot-point-group-witness"
                            "(dependent:Z_D(z_last)=0)";
                    }
                    if (!(z_c == z_edge) && entry.tag == shifted_source)
                    {
                        ++field_rows;
                        group_rows += 1;
                        includes_shift_observations = true;
                        relation_pivot = c_first ? z_c : z_edge;
                        shift_delta = (c_first ? z_edge : z_c) - relation_pivot;
                        certificate +=
                            "+Taylor-shift-commitment+shared-u_rel"
                            "+shifted/source-point-group-witness"
                            "(dependent:shared-u_rel+exact-shift-map)"
                            "+G_rel-pivot-witness(dependent:Z_D(z_last)=0)";
                    }
                }
                // Field observations and formal G1-exponent observations live
                // in separate direct-sum components.  Each component is
                // covered independently; using their sum would violate the
                // compiler's lexicographically minimal prefix rule.
                const auto q_max = std::max(field_rows, group_rows);
                std::size_t bad_crs_degree = 0;
                certificate += "|field_rows=" + std::to_string(field_rows) + "|group_rows=" + std::to_string(group_rows) + "|qmax=max(field_rows,group_rows)=" + std::to_string(q_max);
                if (const auto domain_size = registered_domain_size(
                        context, entry.domain);
                    domain_size.has_value())
                {
                    certificate += "|" + formal_observation_rank_certificate(
                                             entry.tag,
                                             *domain_size,
                                             entry.enters_opening,
                                             includes_model_blinder_commitment,
                                             includes_first_relation_witness,
                                             includes_shift_observations,
                                             relation_pivot,
                                             shift_delta,
                                             field_rows,
                                             group_rows);
                    const auto certificate_factor_degree = *domain_size + q_max + (includes_shift_observations ? *domain_size : 0) + 2;
                    const auto certificate_factor_count =
                        field_rows + group_rows + 1;
                    if (certificate_factor_count != 0 && certificate_factor_degree > std::numeric_limits<std::size_t>::max() / certificate_factor_count / certificate_factor_count)
                        throw std::runtime_error(
                            "ObservationLedger BadCRS degree overflow: " + entry.tag);
                    bad_crs_degree = certificate_factor_degree * certificate_factor_count * certificate_factor_count;
                    certificate += "|BadCRS_obs_degree_bound=" + std::to_string(bad_crs_degree);
                }
                else
                {
                    certificate += "|basis-certificate=separate-schema-basis-required";
                }
                artifact.observation_ledger.push_back({
                    .id = "obs:" + entry.tag,
                    .object_or_challenge = entry.tag,
                    .domain = entry.domain,
                    .transcript_stage = entry.owner_block,
                    .expression_or_certificate = std::move(certificate),
                    .degree_bound = context.kzg.manifest.degree_bound,
                    .query_bound = q_max,
                    .exclusion_degree = bad_crs_degree,
                    .multiplicity = 1,
                });
            }
        }
        for (std::size_t layer = 0;
             layer < context.model.hidden_layers.size(); ++layer)
            for (std::size_t head = 0;
                 head < context.model.hidden_layers[layer].heads.size(); ++head)
                for (const auto *role : {"proj", "src", "dst"})
                {
                    artifact.constraint_registry.push_back(
                        "shifted-divisibility:hidden:l"
                        + std::to_string(layer) + ":h"
                        + std::to_string(head) + ":" + role);
                }
        for (std::size_t head = 0; head < context.model.output_layer.heads.size(); ++head)
        {
            for (const auto *role : {"proj", "src", "dst"})
            {
                artifact.constraint_registry.push_back(
                    "shifted-divisibility:output:" + std::to_string(head) + ":" + role);
            }
        }
        constexpr std::size_t kQFsMax = 1048576;
        constexpr std::size_t kLambdaTarget = 120;
        constexpr std::size_t kLambdaKzg = 128;
        constexpr std::size_t kLambdaRegister = 128;
        constexpr std::size_t kLambdaRandomOracle = 128;
        std::map<std::size_t, ExactInteger>
            epsilon_alg_numerators_by_exclusion;
        const auto checked_mul = [](std::size_t lhs, std::size_t rhs,
                                    const std::string &label)
        {
            if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
                throw std::runtime_error(
                    "RandomIdentityLedger degree overflow: " + label);
            return lhs * rhs;
        };
        const auto checked_add = [](std::size_t lhs, std::size_t rhs,
                                    const std::string &label)
        {
            if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
                throw std::runtime_error(
                    "RandomIdentityLedger degree overflow: " + label);
            return lhs + rhs;
        };
        auto add_random_identity = [&](std::string id,
                                       std::string challenge,
                                       std::string stage,
                                       std::string expression,
                                       std::size_t degree,
                                       std::size_t exclusion_degree,
                                       std::size_t multiplicity)
        {
            if (degree == 0 || multiplicity == 0)
                throw std::runtime_error(
                    "RandomIdentityLedger entry has an empty bound: " + id);
            artifact.random_identity_ledger.push_back({
                .id = "random:" + id,
                .object_or_challenge = challenge,
                .domain = "Fr",
                .transcript_stage = std::move(stage),
                .expression_or_certificate = std::move(expression),
                .degree_bound = degree,
                .query_bound = 1,
                .exclusion_degree = exclusion_degree,
                .multiplicity = multiplicity,
            });
            artifact.security_budget_ledger.push_back({
                .id = "budget:" + id,
                .object_or_challenge = std::move(challenge),
                .domain = "Fr",
                .transcript_stage = artifact.random_identity_ledger.back()
                                        .transcript_stage,
                .expression_or_certificate = "",
                .degree_bound = degree,
                .query_bound = kQFsMax,
                .exclusion_degree = exclusion_degree,
                .multiplicity = multiplicity,
            });
            const ExactInteger numerator = exact_integer(kQFsMax) * exact_integer(multiplicity) * exact_integer(degree);
            const ExactInteger denominator = bls12_381_scalar_modulus() - exact_integer(exclusion_degree);
            if (!denominator.is_positive())
                throw std::runtime_error(
                    "SecurityBudgetLedger algebraic denominator is not positive: " + id);
            ExactRational term{numerator, denominator};
            term.normalize();
            artifact.security_budget_ledger.back().expression_or_certificate =
                "exact-rational:Q_FS_max*multiplicity*d_max/(p-B)"
                "|value=" +
                term.canonical() + "|Q_FS_max=" + std::to_string(kQFsMax) + "|d_max=" + std::to_string(degree) + "|B=" + std::to_string(exclusion_degree) + "|multiplicity=" + std::to_string(multiplicity);
            const auto [numerator_it, inserted] =
                epsilon_alg_numerators_by_exclusion.emplace(
                    exclusion_degree, numerator);
            if (!inserted)
            {
                numerator_it->second = numerator_it->second + numerator;
            }
        };

        const auto d_max = context.kzg.manifest.degree_bound;
        struct QuotientIdentitySpec
        {
            std::string quotient;
            std::string challenge;
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
            std::size_t committed_product_arity;
        };
        const std::vector<QuotientIdentitySpec> quotient_identities = {
            {"t_FH", "z_FH", context.domains.fh, 3},
            {"t_edge", "z_edge", context.domains.edge, 4},
            {"t_in", "z_in", context.domains.in, 3},
            {"t_d_h", "z_d_h", context.domains.d, 3},
            {"t_cat", "z_cat", context.domains.cat, 3},
            {"t_C", "z_C", context.domains.c, 3},
            {"t_N", "z_N", context.domains.n, 3},
        };
        std::size_t prior_evaluation_challenge_count = 0;
        for (const auto &identity : quotient_identities)
        {
            if (identity.domain == nullptr)
                throw std::runtime_error(
                    "RandomIdentityLedger quotient domain is missing: " + identity.quotient);
            const auto constraint_degree = checked_mul(
                identity.committed_product_arity, d_max, identity.quotient);
            const auto masked_quotient_degree = checked_add(
                d_max, identity.domain->size, identity.quotient);
            const auto identity_degree = std::max(
                constraint_degree, masked_quotient_degree);
            const auto collision_degree = checked_mul(
                std::size_t{2}, prior_evaluation_challenge_count,
                identity.challenge + ":collision");
            const auto exclusion_degree = checked_add(
                checked_add(
                    identity.domain->size, collision_degree,
                    identity.challenge + ":domain+collision"),
                std::size_t{1}, identity.challenge + ":zero-pair");
            add_random_identity(
                "quotient:" + identity.quotient,
                identity.challenge,
                "seven_quotient_commitments",
                "sub(mul(Z_D,t_D_zk),constraint_accumulator_D)"
                "|substitute(t_D_zk=t_D+Z_D*R_t_D)"
                "|malicious-input-degree=global-D_max"
                "|max-committed-product-arity=" +
                    std::to_string(identity.committed_product_arity) + "|opening-point-exclusion=Z_D(Y)*prior-point-collision-factors*Y",
                identity_degree,
                exclusion_degree,
                1);
            prior_evaluation_challenge_count += 2;
        }

        const auto alpha_degree = std::max<std::size_t>(
            1, artifact.constraint_registry.size() - 1);
        add_random_identity(
            "constraint-fold:alpha_quot", "alpha_quot",
            "alpha_quot",
            "sum_i(alpha_quot^i*constraint_identity_i)"
            "|canonical-constraint-order|degree-in-alpha=m-1",
            alpha_degree, 0, 1);

        auto add_tuple_compression = [&](const std::string &id,
                                         const std::string &challenge,
                                         const std::string &stage,
                                         std::size_t tuple_length,
                                         std::size_t comparison_count)
        {
            if (tuple_length < 2 || comparison_count == 0)
                throw std::runtime_error(
                    "invalid tuple-compression registry shape: " + id);
            add_random_identity(
                id + ":tuple-compression", challenge, stage,
                "sub(sum_j(lhs_j*Y^j),sum_j(rhs_j*Y^j))"
                "|tuple-length=" +
                    std::to_string(tuple_length) + "|collision-degree=tuple-length-1" + "|comparison-count=" + std::to_string(comparison_count),
                tuple_length - 1, 0, comparison_count);
        };
        auto add_logup_endpoint = [&](const std::string &id,
                                      const std::string &challenge,
                                      const std::string &stage,
                                      std::size_t active_denominator_count)
        {
            if (active_denominator_count == 0)
                throw std::runtime_error(
                    "invalid LogUp denominator registry shape: " + id);
            add_random_identity(
                id + ":logup-cleared-endpoint", challenge, stage,
                "clear_denominators(sum_i(m_i/(Table_i+Y))"
                "-sum_j(1/(Query_j+Y)))"
                "|active-denominator-count=" +
                    std::to_string(active_denominator_count) + "|degree-from-rational-expression-DAG=q_den-1",
                std::max<std::size_t>(1, active_denominator_count - 1),
                0, 1);
        };
        auto add_random_point_identity = [&](const std::string &id,
                                             const std::string &challenge,
                                             const std::string &stage,
                                             const std::string &expression,
                                             std::size_t multiplicity = 1)
        {
            add_random_identity(
                id, challenge, stage,
                expression + "|malicious-committed-input-degree=global-D_max",
                d_max, 0, multiplicity);
        };
        const auto route_denominators = checked_add(
            context.local.num_nodes, context.local.edges.size(),
            "route-active-denominators");
        const auto feature_rows = checked_mul(
            context.local.num_nodes, context.local.num_features,
            "feature-query-rows");
        const auto feature_table_rows = checked_mul(
            context.dataset.num_nodes, context.local.num_features,
            "feature-table-rows");
        const auto feature_denominators = checked_add(
            feature_table_rows, feature_rows, "feature-active-denominators");
        add_tuple_compression(
            "feature-indexed-view", "eta_feat", "feature_lookup",
            3, std::max<std::size_t>(1, feature_denominators));
        add_logup_endpoint(
            "feature-indexed-view", "beta_feat", "feature_lookup",
            feature_denominators);

        auto add_pair_lookup = [&](const std::string &id,
                                   const std::string &eta,
                                   const std::string &beta,
                                   std::size_t table_rows,
                                   std::size_t query_rows,
                                   const std::string &stage)
        {
            const auto active = checked_add(
                table_rows, query_rows, id + ":active-denominators");
            add_tuple_compression(id, eta, stage, 2, active);
            add_logup_endpoint(id, beta, stage, active);
        };
        auto add_route = [&](const std::string &id,
                             const std::string &eta,
                             const std::string &beta,
                             std::size_t tuple_length,
                             const std::string &stage)
        {
            add_tuple_compression(
                id, eta, stage, tuple_length,
                std::max<std::size_t>(1, route_denominators));
            add_logup_endpoint(id, beta, stage, route_denominators);
        };
        auto add_signed_division_ranges = [&](const std::string &prefix,
                                              std::size_t active_rows,
                                              const std::string &stage)
        {
            const auto denominator_count = checked_add(
                context.tables.range.size(), active_rows,
                prefix + ":range-active-denominators");
            const auto limb_count = signed_division_limb_count(
                context.config.range_bits);
            for (const auto *value_name : {
                     "A", "DMinus1", "Q", "R", "Delta"})
            {
                for (std::size_t limb = 0; limb < limb_count; ++limb)
                {
                    const auto beta = signed_division_beta_name(
                        prefix, value_name, limb);
                    add_logup_endpoint(
                        beta, beta, stage, denominator_count);
                }
            }
        };

        for (const auto &layer : context.model.hidden_layers)
        {
            for (std::size_t local_head = 0;
                 local_head < layer.shape.head_count; ++local_head)
            {
                const auto layer_index = layer.layer_index;
                const auto suffix = "l" + std::to_string(layer_index)
                    + ":h" + std::to_string(local_head);
                const auto prefix = hidden_head_prefix(layer_index, local_head);
                const auto stage = "hidden-head:" + suffix;
                const auto head_dim = layer.shape.head_dim;
                const auto point_multiplicity = std::max<std::size_t>(1, head_dim);
                add_random_point_identity(
                    "hidden:" + suffix + ":projection",
                    hidden_challenge_name("y_proj_h", layer_index, local_head),
                    stage,
                    "row-major-projection-fold(Y)-committed-H_prime(Y)");
                add_tuple_compression(
                    "hidden:" + suffix + ":column-fold",
                    hidden_challenge_name("xi_h", layer_index, local_head),
                    stage, std::max<std::size_t>(2, head_dim),
                    checked_mul(std::size_t{3}, context.local.num_nodes,
                                "hidden-column-fold-comparisons"));
                for (const auto *role : {"src", "dst", "star", "agg_pre", "agg"})
                {
                    const auto challenge = hidden_challenge_name(
                        std::string(role) == "agg_pre"
                            ? "y_agg_pre_h"
                            : std::string(role) == "agg"
                                  ? "y_agg_h"
                                  : "y_" + std::string(role) + "_h",
                        layer_index, local_head);
                    add_random_point_identity(
                        "hidden:" + suffix + ":" + role,
                        challenge, stage,
                        "folded-column-inner-product-or-compressed-view-identity(Y)",
                        point_multiplicity);
                }
                add_route(
                    "hidden:" + suffix + ":src-route",
                    hidden_challenge_name("eta_src_h", layer_index, local_head),
                    hidden_challenge_name("beta_src_h", layer_index, local_head),
                    3, stage);
                add_pair_lookup(
                    "hidden:" + suffix + ":lrelu",
                    hidden_challenge_name("eta_L_h", layer_index, local_head),
                    hidden_challenge_name("beta_L_h", layer_index, local_head),
                    context.tables.lrelu.size(),
                    context.local.edges.size(), stage);
                add_logup_endpoint(
                    "hidden:" + suffix + ":range",
                    hidden_challenge_name("beta_R_h", layer_index, local_head),
                    stage, checked_add(context.tables.range.size(), context.local.edges.size(), "hidden-range-denominators"));
                add_pair_lookup(
                    "hidden:" + suffix + ":exp",
                    hidden_challenge_name("eta_exp_h", layer_index, local_head),
                    hidden_challenge_name("beta_exp_h", layer_index, local_head),
                    context.tables.exp.size(),
                    context.local.edges.size(), stage);
                add_tuple_compression(
                    "hidden:" + suffix + ":psq-fold",
                    hidden_challenge_name("lambda_psq_h", layer_index, local_head),
                    stage, 2, std::max<std::size_t>(1, route_denominators));
                add_route(
                    "hidden:" + suffix + ":target-route",
                    hidden_challenge_name("eta_t_h", layer_index, local_head),
                    hidden_challenge_name("beta_t_h", layer_index, local_head),
                    2, stage);
                const auto hidden_value_rows = checked_mul(
                    context.local.num_nodes, head_dim,
                    "hidden-value-rows");
                add_pair_lookup(
                    "hidden:" + suffix + ":elu",
                    hidden_challenge_name("eta_ELU_h", layer_index, local_head),
                    hidden_challenge_name("beta_ELU_h", layer_index, local_head),
                    context.tables.elu.size(),
                    hidden_value_rows, stage);
                add_route(
                    "hidden:" + suffix + ":dst-route",
                    hidden_challenge_name("eta_dst_h", layer_index, local_head),
                    hidden_challenge_name("beta_dst_h", layer_index, local_head),
                    6, stage);
                add_signed_division_ranges(
                    prefix + "_Div_", hidden_value_rows, stage);
            }
            const auto layer_suffix = std::to_string(layer.layer_index);
            const auto concat_width = checked_mul(
                layer.shape.head_count, layer.shape.head_dim,
                "hidden-concat-width");
            add_tuple_compression(
                "hidden-concat:" + layer_suffix,
                hidden_concat_xi_name(layer.layer_index),
                "hidden-concat:" + layer_suffix,
                std::max<std::size_t>(2, concat_width),
                std::max<std::size_t>(1, context.local.num_nodes));
            add_random_point_identity(
                "hidden-concat:" + layer_suffix + ":binding",
                hidden_concat_y_name(layer.layer_index),
                "hidden-concat:" + layer_suffix,
                "concatenated-head-layout-polynomial-identity(Y)");
        }

        const auto class_count = context.local.num_classes;
        add_tuple_compression(
            "output:shared-column-fold", "xi_out", "output-heads",
            std::max<std::size_t>(2, class_count),
            checked_mul(
                std::max<std::size_t>(1, context.model.output_layer.heads.size()),
                std::max<std::size_t>(1, context.local.num_nodes),
                "output-column-fold-comparisons"));
        for (std::size_t head = 0;
             head < context.model.output_layer.heads.size(); ++head)
        {
            const auto suffix = std::to_string(head);
            const auto stage = "output-head:" + suffix;
            auto out_challenge = [&](std::string_view base)
            { return output_challenge_name(base, head); };
            for (const auto *role : {"proj", "src", "dst", "out_star"})
            {
                const auto challenge = std::string(role) == "out_star"
                                           ? out_challenge("y_out_star")
                                           : out_challenge("y_" + std::string(role) + "_out");
                add_random_point_identity(
                    "output:" + suffix + ":" + role, challenge, stage,
                    "output-folded-column-or-model-inner-product-identity(Y)",
                    std::max<std::size_t>(1, class_count));
            }
            add_route(
                "output:" + suffix + ":src-route", out_challenge("eta_src_out"),
                out_challenge("beta_src_out"), 2, stage);
            add_pair_lookup(
                "output:" + suffix + ":lrelu", out_challenge("eta_L_out"),
                out_challenge("beta_L_out"), context.tables.lrelu.size(),
                context.local.edges.size(), stage);
            add_logup_endpoint(
                "output:" + suffix + ":range", out_challenge("beta_R_out"),
                stage, checked_add(context.tables.range.size(), context.local.edges.size(), "output-range-denominators"));
            add_pair_lookup(
                "output:" + suffix + ":exp", out_challenge("eta_exp_out"),
                out_challenge("beta_exp_out"), context.tables.exp.size(),
                context.local.edges.size(), stage);
            add_tuple_compression(
                "output:" + suffix + ":psq-fold", out_challenge("lambda_out"),
                stage, 2, std::max<std::size_t>(1, route_denominators));
            add_route(
                "output:" + suffix + ":target-route", out_challenge("eta_t_out"),
                out_challenge("beta_t_out"), 2, stage);
            add_route(
                "output:" + suffix + ":dst-route", out_challenge("eta_dst_out"),
                out_challenge("beta_dst_out"), 6, stage);
            add_signed_division_ranges(
                output_head_prefix(head) + "_Div_",
                checked_mul(context.local.num_nodes, class_count,
                            "output-value-rows"),
                stage);
        }
        add_random_point_identity(
            "output:average-and-public-binding", "y_out", "output-average",
            "average-head-logits-and-public-output-compression-identity(Y)",
            std::max<std::size_t>(1, class_count));
        add_signed_division_ranges(
            "P_avg_Div_",
            checked_mul(context.local.num_nodes, class_count,
                        "average-value-rows"),
            "output-average");

        auto relation_context = context;
        relation_context.compiler_artifact =
            std::make_shared<ProtocolCompilerArtifact>(artifact);
        for (const auto &meta : build_binding_relation_metadata(relation_context))
        {
            for (const auto &matrix : meta.canonical_matrices)
                add_tuple_compression(
                    matrix.relation_id + ":column-compression",
                    matrix.compression_challenge_label,
                    "canonical_matrix:compression",
                    std::max<std::size_t>(2, matrix.column_count),
                    std::max<std::size_t>(1, matrix.row_count));
            for (const auto &aggregation : meta.aggregation_relations)
            {
                add_tuple_compression(
                    aggregation.relation_id + ":coordinate-compression",
                    aggregation.compression_challenge_label,
                    "aggregation_binding:compression",
                    std::max<std::size_t>(2, aggregation.column_count),
                    std::max<std::size_t>(1, aggregation.row_count));
                for (const auto *route : {
                         &aggregation.source_route,
                         &aggregation.target_route})
                {
                    add_tuple_compression(
                        route->relation_id,
                        route->eta_challenge_label,
                        "aggregation_binding:eta", 2,
                        std::max<std::size_t>(1, route_denominators));
                    add_logup_endpoint(
                        route->relation_id,
                        route->beta_challenge_label,
                        "aggregation_binding:beta", route_denominators);
                    add_random_identity(
                        route->relation_id + ":terminal-shift",
                        route->terminal_challenge_label,
                        "aggregation_binding:y_rel",
                        "TaylorShift(R_edge,Delta)(Y)-R_edge(Y+Delta)"
                        "|malicious-degree=global-D_max",
                        d_max, context.domains.edge->size, 1);
                }
            }
            for (const auto &relation : meta.route_terminal_relations)
                add_random_identity(
                    relation.relation_id + ":terminal-shift",
                    relation.challenge_label,
                    "route_terminal:y_rel",
                    "TaylorShift(R_edge,Delta)(Y)-R_edge(Y+Delta)"
                    "|malicious-degree=global-D_max",
                    d_max, context.domains.edge->size, 1);
            if (meta.object_kind != "output")
                continue;
            const auto &view = meta.indexed_views.at(0);
            const auto &relation = meta.shift_relations.at(0);
            const auto tuple_uses = checked_add(
                view.canonical_length, view.view_length,
                view.relation_id + ":tuple-uses");
            add_random_identity(
                view.relation_id + ":tuple-compression",
                view.eta_challenge_label,
                "output_bias_view:eta",
                "tuple(idx,value)=idx+eta_view*value"
                "|tuple-length=2|collision-degree=1",
                1, 0, tuple_uses);
            add_logup_endpoint(
                view.relation_id,
                view.beta_challenge_label,
                "output_bias_view:beta",
                tuple_uses);
            if (!relation.configured_degrees.empty())
            {
                add_random_identity(
                    relation.relation_id,
                    relation.challenge_label,
                    "output_bias_view:y_rel",
                    "TaylorShift(F_j,Delta_j)(Y)-F_j(Y+Delta_j)"
                    "|malicious-degree=global-D_max",
                    relation.max_difference_degree,
                    relation.exclusion_degree,
                    relation.configured_degrees.size());
            }
        }
        add_random_identity(
            "public-output-polynomial-binding",
            "public_output.z",
            "public_output_binding",
            "P_Y_committed(Y)-P_Y_public(Y)"
            "|malicious-degree=global-D_max",
            d_max, 0, 1);

        const auto opening_schema_claim_count =
            static_cast<std::size_t>(std::count_if(
                artifact.proof_schema.entries.begin(),
                artifact.proof_schema.entries.end(),
                [](const ProofSchemaEntry &entry)
                { return entry.enters_opening; }));
        // The compiler registers a conservative instance-bound envelope for
        // external/route/binding claims.  Runtime grouping can only merge or
        // deduplicate this set, never exceed it.
        // The formal envelope is counted from the nested layer/head registry;
        // the checkpoint/parity flattening view is not part of the schema.
        std::size_t formal_hidden_head_count = 0;
        for (const auto &layer : context.model.hidden_layers)
            formal_hidden_head_count += layer.heads.size();
        const auto relation_family_envelope = checked_add(
            artifact.proof_schema.entries.size(),
            checked_add(
                formal_hidden_head_count,
                checked_add(
                    context.model.hidden_layers.size(),
                    context.model.output_layer.heads.size(),
                    "point-group-relation-family"),
                "point-group-relation-family"),
            "point-group-relation-family");
        const auto relation_envelope = checked_add(
            std::size_t{1},
            checked_mul(
                std::size_t{16}, relation_family_envelope,
                "point-group-relation-envelope"),
            "point-group-relation-envelope");
        const auto external_claim_count = checked_add(
            checked_mul(
                std::size_t{4}, opening_schema_claim_count,
                "external-point-group-schema-envelope"),
            relation_envelope,
            "external-point-group-claim-envelope");
        const auto max_group_claim_count = checked_add(
            opening_schema_claim_count,
            external_claim_count,
            "global-point-group-claim-envelope");
        if (opening_schema_claim_count < 2 || external_claim_count < 2)
            throw std::runtime_error(
                "standard point-group claim envelope is too small");

        for (const auto &[group_id, domain_name] :
             std::vector<std::pair<std::string, std::string>>{
                 {"FH", context.domains.fh->name},
                 {"edge", context.domains.edge->name},
                 {"in", context.domains.in->name},
                 {"d_h", context.domains.d->name},
                 {"cat", context.domains.cat->name},
                 {"C", context.domains.c->name},
                 {"N", context.domains.n->name}})
        {
            const auto domain_claim_count =
                static_cast<std::size_t>(std::count_if(
                    artifact.proof_schema.entries.begin(),
                    artifact.proof_schema.entries.end(),
                    [&](const ProofSchemaEntry &entry)
                    {
                        return entry.enters_opening && entry.domain == domain_name;
                    }));
            if (domain_claim_count < 2)
                throw std::runtime_error(
                    "point-group registry has fewer than two domain claims: " + group_id);
            for (std::size_t point_index = 0; point_index < 2; ++point_index)
            {
                add_random_identity(
                    "point-group-fold:" + group_id + ":" + std::to_string(point_index),
                    "point_group.rho@Open_dom:" + group_id + ":" + std::to_string(point_index),
                    "standard_point_group_folding",
                    "sum_i(rho_z^i*(P_i(z)-v_i))"
                    "|one-independent-nonzero-rho-per-point"
                    "|full-canonical-claim-tuple"
                    "|external-collision-merges-into-domain-owner"
                    "|domain=" +
                        group_id + "|point-index=" + std::to_string(point_index) + "|honest-domain-claim-count=" + std::to_string(domain_claim_count) + "|max-group-claim-count=" + std::to_string(max_group_claim_count),
                    max_group_claim_count - 1, 0, 1);
            }
        }
        add_random_identity(
            "point-group-fold:W_ext-family",
            "point_group.rho@W_ext:each-distinct-point",
            "standard_point_group_folding",
            "sum_i(rho_z^i*(P_i(z)-v_i))"
            "|one-independent-nonzero-rho-per-point"
            "|full-canonical-claim-tuple"
            "|same-commitment-point-dedup-conflict-reject"
            "|max-group-claim-count=" +
                std::to_string(max_group_claim_count) + "|max-distinct-external-points=" + std::to_string(external_claim_count),
            max_group_claim_count - 1,
            0,
            external_claim_count);

        ExactInteger epsilon_abort_numerator(0);
        for (const auto &entry : artifact.random_identity_ledger)
        {
            if (entry.id.find(":logup-cleared-endpoint") == std::string::npos)
                continue;
            constexpr std::string_view marker =
                "|active-denominator-count=";
            const auto marker_position =
                entry.expression_or_certificate.find(marker);
            if (marker_position == std::string::npos)
                throw std::runtime_error(
                    "LogUp RandomIdentityLedger entry lacks q_den: " + entry.id);
            const auto number_begin = marker_position + marker.size();
            const auto number_end = entry.expression_or_certificate.find(
                '|', number_begin);
            const auto encoded_count = entry.expression_or_certificate.substr(
                number_begin, number_end - number_begin);
            if (encoded_count.empty() || (encoded_count.size() > 1 && encoded_count.front() == '0') || !std::all_of(encoded_count.begin(), encoded_count.end(), [](unsigned char value)
                                                                                                                    { return value >= '0' && value <= '9'; }))
                throw std::runtime_error(
                    "LogUp RandomIdentityLedger q_den is not canonical: " + entry.id);
            const auto denominator_count = static_cast<std::size_t>(
                std::stoull(encoded_count));
            if (denominator_count == 0)
                throw std::runtime_error(
                    "LogUp RandomIdentityLedger q_den is zero: " + entry.id);
            epsilon_abort_numerator = epsilon_abort_numerator + exact_integer(denominator_count);
        }
        ExactRational epsilon_abort{
            epsilon_abort_numerator,
            bls12_381_scalar_modulus()};
        epsilon_abort.normalize();
        const auto max_domain_size = std::max({context.domains.fh->size, context.domains.edge->size,
                                               context.domains.in->size, context.domains.d->size,
                                               context.domains.cat->size, context.domains.c->size,
                                               context.domains.n->size});
        const ExactInteger n_tau = bls12_381_scalar_modulus() - 1 - exact_integer(max_domain_size);
        ExactInteger epsilon_crs_numerator(0);
        for (const auto &entry : artifact.observation_ledger)
        {
            if (entry.exclusion_degree == 0)
                continue;
            epsilon_crs_numerator = epsilon_crs_numerator + exact_integer(entry.exclusion_degree);
        }
        ExactRational epsilon_crs{epsilon_crs_numerator, n_tau};
        epsilon_crs.normalize();
        ExactRational epsilon_alg;
        for (const auto &[exclusion_degree, numerator] :
             epsilon_alg_numerators_by_exclusion)
        {
            ExactRational term{
                numerator,
                bls12_381_scalar_modulus() - exact_integer(exclusion_degree)};
            term.normalize();
            epsilon_alg = exact_add(std::move(epsilon_alg), term);
        }
        ExactRational epsilon_total = epsilon_alg;
        epsilon_total = exact_add(epsilon_total, epsilon_abort);
        epsilon_total = exact_add(epsilon_total, epsilon_crs);
        for (const auto lambda : {
                 kLambdaKzg, kLambdaRegister, kLambdaRandomOracle})
        {
            ExactRational direct{ExactInteger(1), ExactInteger(1) << lambda};
            epsilon_total = exact_add(std::move(epsilon_total), direct);
        }
        if (!exact_at_most_power_of_two(epsilon_total, kLambdaTarget))
            throw std::runtime_error(
                "SecurityBudgetLedger total advantage exceeds target");
        artifact.security_budget_ledger.push_back({
            .id = "budget:summary",
            .object_or_challenge = "epsilon_total",
            .domain = "exact-rational",
            .transcript_stage = "compiler-security-gate",
            .expression_or_certificate =
                "exact-rational:lambda_target=" + std::to_string(kLambdaTarget) + "|Q_FS_max=" + std::to_string(kQFsMax) + "|lambda_KZG=" + std::to_string(kLambdaKzg) + "|lambda_register=" + std::to_string(kLambdaRegister) + "|lambda_RO=" + std::to_string(kLambdaRandomOracle) + "|epsilon_alg=" + epsilon_alg.canonical() + "|epsilon_abort=" + epsilon_abort.canonical() + "|epsilon_crs=" + epsilon_crs.canonical() + "|epsilon_total=" + epsilon_total.canonical() + "|gate=epsilon_total<=2^-120",
            .degree_bound = 1,
            .query_bound = kQFsMax,
            .exclusion_degree = 0,
            .multiplicity = 1,
        });

        // Shift-relation degrees consume the completed ObservationLedger, so
        // compile RelationMeta only after all independent blinder resources
        // have been fixed.  The temporary context is read-only and prevents a
        // prover-selected resource from entering the registry.
        auto binding_context = context;
        binding_context.compiler_artifact =
            std::make_shared<ProtocolCompilerArtifact>(artifact);
        for (const auto &meta : build_binding_relation_metadata(binding_context))
            artifact.binding_relation_registry.push_back(
                binding_relation_string(meta));

        std::string instance_bytes = artifact.protocol_canonical_bytes;
        append_canonical_vector(
            instance_bytes, "binding_relation_registry",
            artifact.binding_relation_registry);
        append_canonical_string(instance_bytes, "proof_schema");
        for (const auto &entry : artifact.proof_schema.entries)
        {
            append_canonical_string(
                instance_bytes,
                entry.tag + "|" + entry.domain + "|visibility=" + std::to_string(static_cast<unsigned>(entry.visibility)) + "|owner=" + entry.owner_block + "|type=" + entry.field_type + "|padding=" + entry.padding_mode + "|owner_policy=" + entry.owner_policy + "|blinding_basis=" + entry.blinding_basis + "|C=" + std::to_string(entry.enters_commitment) + "|O=" + std::to_string(entry.enters_opening) + "|Q=" + std::to_string(entry.enters_quotient));
        }
        append_canonical_vector(instance_bytes, "constraints", artifact.constraint_registry);
        append_canonical_vector(instance_bytes, "degrees", artifact.degree_registry);
        append_canonical_vector(instance_bytes, "padding", artifact.padding_registry);
        append_canonical_string(instance_bytes, "observation_dependency_dag");
        for (const auto &edge : artifact.observation_dependency_dag)
            append_canonical_string(instance_bytes, observation_dependency_string(edge));
        for (const auto &entry : artifact.observation_ledger)
            append_canonical_string(instance_bytes, ledger_string(entry));
        for (const auto &entry : artifact.random_identity_ledger)
            append_canonical_string(instance_bytes, ledger_string(entry));
        for (const auto &entry : artifact.security_budget_ledger)
            append_canonical_string(instance_bytes, ledger_string(entry));
        artifact.instance_canonical_bytes = std::move(instance_bytes);
        return artifact;
    }

    bool validate_protocol_compiler_artifact(
        const ProtocolContext &context,
        const ProtocolCompilerArtifact &artifact,
        std::string *reason)
    {
        auto fail = [&](std::string value)
        {
            if (reason != nullptr)
                *reason = std::move(value);
            return false;
        };
        if (artifact.proof_blocks != proof_block_order() || artifact.proof_blocks.size() != 9)
            return fail("proof block registry mismatch");
        if (artifact.proof_schema.entries.empty() || artifact.binding_relation_registry.empty() || artifact.constraint_registry.empty() || artifact.point_group_registry_template.empty() || artifact.degree_registry.empty() || artifact.padding_registry.empty() || artifact.observation_ledger.empty() || artifact.random_identity_ledger.empty() || artifact.security_budget_ledger.empty() || artifact.serializer_tag_table.size() != artifact.proof_blocks.size() || artifact.object_canonical_order.empty() || artifact.verifier_equation_registry.empty() || artifact.ledger_templates.size() != 3 || artifact.shift_algorithm_id != "coefficient_taylor_shift" || artifact.shift_algorithm_spec.empty())
            return fail("compiler artifact registry is incomplete");
        std::set<std::string> tags;
        for (const auto &entry : artifact.proof_schema.entries)
        {
            if (!tags.insert(entry.tag).second)
                return fail("duplicate proof schema tag: " + entry.tag);
            if (entry.domain.empty())
                return fail("proof schema domain is empty: " + entry.tag);
            if (entry.owner_block.empty() || entry.field_type.empty() || entry.padding_mode.empty() || entry.owner_policy.empty() || entry.blinding_basis.empty())
                return fail("proof schema typed ownership is incomplete: " + entry.tag);
            if (entry.field_type != "Commitment" && entry.field_type != "MetadataReference" && entry.field_type != "PublicValue" && entry.field_type != "MaskedPrivateValue" && entry.field_type != "OpeningWitness")
                return fail("proof schema field type is invalid: " + entry.tag);
            if (entry.owner_block.starts_with("Pi_bind") && entry.field_type == "OpeningWitness")
                return fail("Pi_bind cannot own an opening witness: " + entry.tag);
            if (entry.owner_block.starts_with("Pi_bind") && entry.enters_opening && entry.owner_policy != "global_standard_point_group")
                return fail("Pi_bind opening must reference global point-group owner: " + entry.tag);
            if (entry.blinding_basis != "None" && entry.blinding_basis != "DomainVanishing" && entry.blinding_basis != "CoefficientMask" && entry.blinding_basis != "DerivedExpression" && entry.blinding_basis != "CoefficientTail")
                return fail("proof schema blinding basis is invalid: " + entry.tag);
            if (entry.visibility == PolynomialVisibility::PrivateWitness
                    ? entry.blinding_basis == "None"
                    : entry.blinding_basis != "None")
                return fail("proof schema visibility/blinding basis mismatch: " + entry.tag);
        }
        std::set<std::string> observed_objects;
        for (const auto &entry : artifact.observation_ledger)
        {
            if (entry.id.empty() || entry.object_or_challenge.empty() || entry.transcript_stage.empty() || entry.expression_or_certificate.empty() || entry.query_bound == 0 || entry.multiplicity == 0)
                return fail("ObservationLedger entry is incomplete: " + entry.id);
            if (!observed_objects.insert(entry.object_or_challenge).second)
                return fail("duplicate ObservationLedger base object: " + entry.object_or_challenge);
            if (const auto domain_size = registered_domain_size(context, entry.domain);
                domain_size.has_value() && (*domain_size > context.kzg.manifest.degree_bound || entry.query_bound - 1 > context.kzg.manifest.degree_bound - *domain_size))
            {
                return fail("blinding degree resource exceeds D_max: " + entry.object_or_challenge);
            }
        }
        if (artifact.observation_dependency_dag.empty())
            return fail("ObservationLedger dependency DAG is empty");
        std::set<std::string> dependency_ids;
        std::set<std::string> dependency_targets;
        std::set<std::string> schema_tags;
        for (const auto &entry : artifact.proof_schema.entries)
            schema_tags.insert(entry.tag);
        for (const auto &edge : artifact.observation_dependency_dag)
        {
            if (edge.id.empty() || edge.derived_object.empty() || edge.source_objects.empty() || edge.serialized_observations.empty() || edge.transcript_stage.empty() || edge.propagation_rule.empty() || edge.introduces_independent_randomness || !dependency_ids.insert(edge.id).second || !dependency_targets.insert(edge.derived_object).second)
                return fail("ObservationLedger dependency edge is incomplete or duplicate: " + edge.id);
            if (!std::is_sorted(edge.source_objects.begin(), edge.source_objects.end()) || std::adjacent_find(
                                                                                               edge.source_objects.begin(), edge.source_objects.end()) != edge.source_objects.end())
                return fail("ObservationLedger dependency sources are not canonical: " + edge.id);
            if (!schema_tags.contains(edge.derived_object))
                return fail("ObservationLedger dependency target is not in ProofSchema: " + edge.derived_object);
            for (const auto &source : edge.source_objects)
            {
                if (!schema_tags.contains(source) || !observed_objects.contains(source))
                    return fail("ObservationLedger dependency source is unresolved: " + edge.id + ":" + source);
            }
        }
        for (const auto &entry : artifact.proof_schema.entries)
        {
            if (entry.visibility != PolynomialVisibility::PrivateWitness)
                continue;
            const bool owns_independent_randomness =
                observed_objects.contains(entry.tag);
            const bool is_derived = dependency_targets.contains(entry.tag);
            if (owns_independent_randomness == is_derived)
            {
                return fail(
                    "private ProofSchema object must be exactly one of independent owner or derived target: " + entry.tag);
            }
            if (is_derived != (entry.blinding_basis == "DerivedExpression"))
                return fail("ObservationLedger derived target/blinding basis mismatch: " + entry.tag);
            if (owns_independent_randomness && entry.blinding_basis != "DomainVanishing" && entry.blinding_basis != "CoefficientMask" && entry.blinding_basis != "CoefficientTail")
                return fail("ObservationLedger owner has no independent blinding basis: " + entry.tag);
        }
        std::set<std::string> random_identity_ids;
        for (const auto &entry : artifact.random_identity_ledger)
        {
            if (entry.id.empty() || entry.object_or_challenge.empty() || entry.transcript_stage.empty() || entry.expression_or_certificate.empty() || entry.degree_bound == 0 || entry.query_bound == 0 || entry.multiplicity == 0 || !random_identity_ids.insert(entry.id).second)
                return fail("RandomIdentityLedger entry is incomplete: " + entry.id);
        }
        for (const auto &entry : artifact.security_budget_ledger)
        {
            if (entry.query_bound == 0 || entry.multiplicity == 0 || !entry.expression_or_certificate.starts_with("exact-rational:"))
                return fail("SecurityBudgetLedger entry is incomplete: " + entry.id);
        }
        const auto rebuilt = build_protocol_compiler_artifact(context);
        if (!(artifact.proof_schema == rebuilt.proof_schema) || artifact.binding_relation_registry != rebuilt.binding_relation_registry || artifact.proof_blocks != rebuilt.proof_blocks || artifact.serializer_tag_table != rebuilt.serializer_tag_table || artifact.object_canonical_order != rebuilt.object_canonical_order || artifact.transcript_stages != rebuilt.transcript_stages || artifact.verifier_equation_registry != rebuilt.verifier_equation_registry || artifact.ledger_templates != rebuilt.ledger_templates || artifact.constraint_registry != rebuilt.constraint_registry || artifact.point_group_registry_template != rebuilt.point_group_registry_template || artifact.degree_registry != rebuilt.degree_registry || artifact.padding_registry != rebuilt.padding_registry || artifact.observation_dependency_dag != rebuilt.observation_dependency_dag || artifact.observation_ledger != rebuilt.observation_ledger || artifact.random_identity_ledger != rebuilt.random_identity_ledger || artifact.security_budget_ledger != rebuilt.security_budget_ledger || artifact.kzg_assumption_profile != rebuilt.kzg_assumption_profile || artifact.shift_algorithm_id != rebuilt.shift_algorithm_id || artifact.shift_algorithm_spec != rebuilt.shift_algorithm_spec || artifact.protocol_canonical_bytes != rebuilt.protocol_canonical_bytes || artifact.instance_canonical_bytes != rebuilt.instance_canonical_bytes)
            return fail("compiler artifact canonical re-encoding mismatch");
        if (reason != nullptr)
            reason->clear();
        return true;
    }

    bool instance_within_authenticated_max_envelope(
        const ProtocolContext &context,
        std::string *reason)
    {
        auto fail = [&](std::string message)
        {
            if (reason != nullptr)
                *reason = std::move(message);
            return false;
        };
        const auto &envelope = context.kzg.manifest.max_instance_envelope;
        auto bound = [&](const std::string &key) -> std::optional<std::size_t>
        {
            const auto it = envelope.find(key);
            if (it == envelope.end() || it->second == 0)
                return std::nullopt;
            return it->second;
        };
        const std::vector<std::string> required = {
            "max_layers", "max_hidden_heads_per_layer", "max_hidden_head_dim",
            "max_input_dim", "max_output_heads", "max_classes", "max_graphs",
            "max_nodes", "max_edges", "max_total_nodes", "max_concat_dim",
            "max_range_bits", "max_coefficient_object_length",
            "max_object_family_count", "max_coefficient_chunks",
            "security_target_bits"};
        for (const auto &key : required)
            if (!bound(key).has_value())
                return fail("SRS max instance envelope is missing bound: " + key);

        std::size_t max_layer_heads = 0;
        std::size_t max_head_dim = 0;
        std::size_t max_input_dim = context.local.num_features;
        for (const auto &layer : context.model.hidden_layers)
        {
            max_layer_heads = std::max(max_layer_heads, layer.shape.head_count);
            max_head_dim = std::max(max_head_dim, layer.shape.head_dim);
            max_input_dim = std::max(max_input_dim, layer.input_dim);
        }
        max_input_dim = std::max(
            max_input_dim, context.model.output_layer.input_dim);
        if (context.model.L > *bound("max_layers"))
            return fail("instance total layer count exceeds SRS max envelope");
        if (max_layer_heads > *bound("max_hidden_heads_per_layer"))
            return fail("instance hidden head count exceeds SRS max envelope");
        if (max_head_dim > *bound("max_hidden_head_dim"))
            return fail("instance hidden head dimension exceeds SRS max envelope");
        if (max_input_dim > *bound("max_input_dim"))
            return fail("instance layer input dimension exceeds SRS max envelope");
        if (context.model.output_layer.heads.size() > *bound("max_output_heads"))
            return fail("instance output head count exceeds SRS max envelope");
        if (context.local.num_classes > *bound("max_classes"))
            return fail("instance class count exceeds SRS max envelope");
        if (context.local.graph_count > *bound("max_graphs"))
            return fail("instance graph count exceeds SRS max envelope");
        if (context.local.num_nodes > *bound("max_nodes"))
            return fail("instance node count exceeds SRS max envelope");
        if (context.local.edges.size() > *bound("max_edges"))
            return fail("instance edge count exceeds SRS max envelope");
        if (context.local.public_input.N_total > *bound("max_total_nodes"))
            return fail("instance total dataset node count exceeds SRS max envelope");
        const auto concat_dim = model::max_hidden_concat_width(context.model);
        if (concat_dim > *bound("max_concat_dim"))
            return fail("instance hidden concat dimension exceeds SRS max envelope");
        if (context.config.range_bits > *bound("max_range_bits"))
            return fail("instance range bit width exceeds SRS max envelope");
        auto checked_product = [&](std::size_t left, std::size_t right)
            -> std::optional<std::size_t>
        {
            if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
                return std::nullopt;
            return left * right;
        };
        std::size_t max_coefficient_length = 1;
        const auto include_product = [&](std::size_t left, std::size_t right)
        {
            const auto product = checked_product(left, right);
            if (!product.has_value())
                return false;
            max_coefficient_length = std::max(max_coefficient_length, *product);
            return true;
        };
        if (!include_product(context.local.num_nodes, context.local.num_features) || !include_product(context.local.num_nodes, concat_dim) || !include_product(context.local.num_nodes, context.local.num_classes))
            return fail("instance coefficient object length overflows size_t");
        for (const auto &layer : context.model.hidden_layers)
        {
            if (!include_product(layer.input_dim, layer.shape.head_dim) || !include_product(context.local.num_nodes, layer.shape.head_dim))
                return fail("instance hidden coefficient object length overflows size_t");
        }
        if (!include_product(
                context.model.output_layer.input_dim,
                context.model.output_layer.output_dim))
            return fail("instance output coefficient object length overflows size_t");
        if (max_coefficient_length > *bound("max_coefficient_object_length"))
            return fail("instance coefficient object length exceeds SRS max envelope");
        const auto required_chunks = coefficient_chunk_layout(
            max_coefficient_length, context.domains.edge->size);
        if (required_chunks.size() > *bound("max_coefficient_chunks"))
            return fail("instance coefficient object exceeds SRS chunk-count envelope");
        const auto instance_schema = build_proof_schema(context);
        if (instance_schema.entries.size() > *bound("max_object_family_count"))
            return fail("instance object family count exceeds SRS max envelope");
        if (*bound("security_target_bits") != 120)
            return fail("SRS max envelope security target mismatch");
        if (reason != nullptr)
            reason->clear();
        return true;
    }

    std::vector<CoefficientChunkLayoutEntry> coefficient_chunk_layout(
        std::size_t total_coefficient_length,
        std::size_t edge_domain_size)
    {
        if (total_coefficient_length == 0)
            throw std::runtime_error(
                "coefficient object length must be positive");
        if (edge_domain_size < 2)
            throw std::runtime_error(
                "coefficient chunks require an edge domain with a terminal row");
        const auto capacity = edge_domain_size - 1;
        const auto quotient = total_coefficient_length / capacity;
        const auto remainder = total_coefficient_length % capacity;
        if (quotient == std::numeric_limits<std::size_t>::max() && remainder != 0)
            throw std::runtime_error("coefficient chunk count overflows size_t");
        const auto count = quotient + (remainder == 0 ? 0 : 1);
        std::vector<CoefficientChunkLayoutEntry> out;
        out.reserve(count);
        for (std::size_t chunk = 0; chunk < count; ++chunk)
        {
            if (chunk > std::numeric_limits<std::size_t>::max() / capacity)
                throw std::runtime_error("coefficient chunk offset overflows size_t");
            const auto offset = chunk * capacity;
            const auto remaining = total_coefficient_length - offset;
            out.push_back(CoefficientChunkLayoutEntry{
                .chunk_index = chunk,
                .offset = offset,
                .length = std::min(capacity, remaining),
            });
        }
        return out;
    }

    std::size_t minimum_edge_trace_length(
        std::size_t node_count,
        std::size_t edge_count,
        std::size_t hidden_width,
        std::size_t class_count)
    {
        const auto checked_product = [node_count](
                                         std::size_t width, std::string_view label)
        {
            if (width != 0 && node_count > std::numeric_limits<std::size_t>::max() / width)
            {
                throw std::runtime_error(
                    "edge-domain flattened length overflows size_t: " + std::string(label));
            }
            return node_count * width;
        };
        return std::max({
            node_count,
            edge_count,
            checked_product(hidden_width, "N*d_h"),
            checked_product(class_count, "N*C"),
        });
    }

    CoefficientRegionLayout coefficient_region_layout(
        std::size_t low_degree_length,
        const std::vector<std::size_t> &semantic_positions,
        std::size_t tail_length,
        std::size_t edge_domain_size)
    {
        if (low_degree_length == 0)
            throw std::runtime_error(
                "coefficient region low-degree length must be nonzero");
        if (tail_length == 0)
            throw std::runtime_error(
                "coefficient region tail length must be nonzero");
        if (semantic_positions.empty())
            throw std::runtime_error(
                "coefficient region must contain a semantic position");
        if (!std::is_sorted(
                semantic_positions.begin(), semantic_positions.end()) ||
            std::adjacent_find(
                semantic_positions.begin(), semantic_positions.end()) != semantic_positions.end() ||
            semantic_positions.back() >= low_degree_length)
        {
            throw std::runtime_error(
                "coefficient semantic positions are not canonical");
        }
        if (tail_length > std::numeric_limits<std::size_t>::max() - low_degree_length)
            throw std::runtime_error("coefficient region length overflows size_t");
        const auto total_length = low_degree_length + tail_length;
        CoefficientRegionLayout out{
            .low_degree_length = low_degree_length,
            .tail_length = tail_length,
            .total_length = total_length,
            .semantic_positions = semantic_positions,
            .q_sem = std::vector<std::uint8_t>(total_length, 0),
            .q_gap = std::vector<std::uint8_t>(total_length, 0),
            .q_tail = std::vector<std::uint8_t>(total_length, 0),
            .chunks = coefficient_chunk_layout(total_length, edge_domain_size),
        };
        for (const auto position : semantic_positions)
            out.q_sem[position] = 1;
        for (std::size_t position = 0; position < low_degree_length; ++position)
            if (out.q_sem[position] == 0)
                out.q_gap[position] = 1;
        for (std::size_t position = low_degree_length;
             position < total_length; ++position)
            out.q_tail[position] = 1;
        for (std::size_t position = 0; position < total_length; ++position)
        {
            const auto sum = static_cast<unsigned>(out.q_sem[position]) + static_cast<unsigned>(out.q_gap[position]) + static_cast<unsigned>(out.q_tail[position]);
            if (sum != 1)
                throw std::runtime_error(
                    "coefficient region selectors do not form a partition");
        }
        return out;
    }

    std::vector<algebra::FieldElement> materialize_coefficient_tail(
        const CoefficientRegionLayout &layout,
        const std::vector<algebra::FieldElement> &low_degree_coefficients,
        const std::vector<algebra::FieldElement> &tail_randomness)
    {
        if (layout.low_degree_length == 0 || layout.tail_length == 0 || layout.total_length != layout.low_degree_length + layout.tail_length || low_degree_coefficients.size() != layout.low_degree_length || tail_randomness.size() != layout.tail_length || layout.q_sem.size() != layout.total_length || layout.q_gap.size() != layout.total_length || layout.q_tail.size() != layout.total_length)
            throw std::runtime_error(
                "coefficient-tail materialization shape mismatch");
        for (std::size_t position = 0;
             position < layout.low_degree_length; ++position)
        {
            const auto selector_sum =
                static_cast<unsigned>(layout.q_sem[position]) + static_cast<unsigned>(layout.q_gap[position]) + static_cast<unsigned>(layout.q_tail[position]);
            if (selector_sum != 1 || layout.q_tail[position] != 0 || (layout.q_gap[position] != 0 && !low_degree_coefficients[position].is_zero()))
                throw std::runtime_error(
                    "coefficient-tail low-degree selector/gap violation");
        }
        for (std::size_t position = layout.low_degree_length;
             position < layout.total_length; ++position)
        {
            const auto selector_sum =
                static_cast<unsigned>(layout.q_sem[position]) + static_cast<unsigned>(layout.q_gap[position]) + static_cast<unsigned>(layout.q_tail[position]);
            if (selector_sum != 1 || layout.q_tail[position] != 1)
                throw std::runtime_error(
                    "coefficient-tail suffix selector violation");
        }
        if (std::all_of(
                tail_randomness.begin(), tail_randomness.end(),
                [](const algebra::FieldElement &value)
                { return value.is_zero(); }))
            throw std::runtime_error(
                "coefficient-tail randomness cannot be the all-zero vector");
        std::vector<algebra::FieldElement> out;
        out.reserve(layout.total_length);
        out.insert(
            out.end(), low_degree_coefficients.begin(),
            low_degree_coefficients.end());
        out.insert(out.end(), tail_randomness.begin(), tail_randomness.end());
        return out;
    }

    std::size_t registered_blinder_coefficient_count(
        const ProtocolContext &context,
        const std::string &object_id)
    {
        if (context.compiler_artifact == nullptr)
            throw std::runtime_error(
                "blinder resource lookup requires ProtocolCompilerArtifact: " + object_id);
        const auto &ledger = context.compiler_artifact->observation_ledger;
        const auto it = std::find_if(
            ledger.begin(), ledger.end(), [&](const CompilerLedgerEntry &entry)
            { return entry.object_or_challenge == object_id; });
        if (it == ledger.end() || it->query_bound == 0)
            throw std::runtime_error(
                "missing ObservationLedger blinder resource: " + object_id);
        return it->query_bound;
    }

    PolynomialVisibility polynomial_visibility(
        const ProtocolContext &context,
        const std::string &label)
    {
        if (context.public_commitments.contains(label) || label == "P_H" || label.ends_with("_BiasView_m_can"))
        {
            return PolynomialVisibility::Public;
        }
        // These are the only final-output polynomial identities. Prefix matches
        // are forbidden: `P_Y_lin`, `P_Y_o0`, and head-local compressed outputs
        // contain private intermediate state.
        if (label == "P_Y" || label == "P_Y_star" || label == "P_Y_star_edge")
        {
            return PolynomialVisibility::PublicOutput;
        }
        return PolynomialVisibility::PrivateWitness;
    }

    bool schema_accepts_evaluation(PolynomialVisibility visibility, EvaluationKind kind)
    {
        if (visibility == PolynomialVisibility::PrivateWitness)
        {
            return kind == EvaluationKind::MaskedPrivateValue;
        }
        return kind == EvaluationKind::PublicValue;
    }

} // namespace gatzk::protocol
