#pragma once

#include <cstddef>
#include <vector>

#include "gatzk/algebra/field.hpp"
#include "gatzk/data/dataset.hpp"

namespace gatzk::model {

using Matrix = std::vector<std::vector<algebra::FieldElement>>;

struct ModelParameters {
    Matrix W;
    std::vector<algebra::FieldElement> a_src;
    std::vector<algebra::FieldElement> a_dst;
    Matrix W_out;
    std::vector<algebra::FieldElement> b;
};

ModelParameters build_model_parameters(
    std::size_t input_dim,
    std::size_t hidden_dim,
    std::size_t num_classes,
    std::uint64_t seed);

Matrix project_features(const Matrix& left, const Matrix& right);
std::vector<algebra::FieldElement> matvec_projection(const Matrix& matrix, const std::vector<algebra::FieldElement>& vector);
std::vector<algebra::FieldElement> compress_rows(const Matrix& matrix, const algebra::FieldElement& challenge);
Matrix aggregate_by_edges(
    const Matrix& h_prime,
    const std::vector<algebra::FieldElement>& alpha,
    const std::vector<data::Edge>& edges,
    std::size_t num_nodes);
Matrix output_projection(
    const Matrix& h_agg,
    const Matrix& w_out,
    const std::vector<algebra::FieldElement>& bias,
    Matrix* linear_part);

}  // namespace gatzk::model
