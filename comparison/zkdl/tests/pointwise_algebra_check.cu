#include "../zkrelation-proof.cuh"
#include <iostream>
#include <stdexcept>
using namespace zkdl_gat::relation;
static void require(bool condition, const char* label) { if (!condition) throw std::runtime_error(label); }
static std::vector<SegmentedTensorProvider> providers(const std::vector<std::vector<FrWire>>& values, std::size_t basis) {
    std::vector<SegmentedTensorProvider> result;
    std::string error;
    require(make_segmented_tensor_providers(values, basis, result, &error), "providers");
    return result;
}
static FrWire endpoint(const ConstraintDescriptor& descriptor, const std::vector<SegmentedTensorProvider>& input, const std::vector<FrWire>& rho) {
    std::vector<SegmentedTensorProvider> weighted;
    std::string error;
    require(pointwise_providers(descriptor, input, rho, weighted, &error), "weighted providers");
    const auto expanded = pointwise_sumcheck_descriptor(descriptor, input.size());
    std::vector<FrWire> values;
    require(zksc_segmented_round_evaluations(expanded, weighted, expanded.degree, 0U, {}, values, &error), "round algebra");
    return zkdl::fr_add(values[0], values[1]);
}
int main() {
    try {
        ConstraintDescriptor linear;
        linear.id = "check/cancel";
        linear.degree = 1U;
        linear.terms = {Monomial{zkdl::fr_from_u32(1U), {0U}}};
        require(endpoint(linear, providers({{zkdl::fr_from_u32(1U), zkdl::fr_neg(zkdl::fr_from_u32(1U))}}, 2U), {zkdl::fr_from_u32(2U)}) != FrWire{}, "canceling slots must fail weighted zero claim");
        ConstraintDescriptor inverse;
        inverse.id = "check/inverse";
        inverse.degree = 2U;
        inverse.constant = zkdl::fr_neg(zkdl::fr_from_u32(1U));
        inverse.terms = {Monomial{zkdl::fr_from_u32(1U), {0U, 1U}}};
        inverse.logical_shape = {3U, 3U};
        inverse.padded_shape = {4U, 4U};
        std::vector<FrWire> tensor(16U);
        for (std::size_t slot = 0U; slot < tensor.size(); ++slot) if (pointwise_active_slot(inverse, slot)) tensor[slot] = zkdl::fr_from_u32(1U);
        const std::vector<FrWire> rho = {zkdl::fr_from_u32(2U), zkdl::fr_from_u32(3U), zkdl::fr_from_u32(5U), zkdl::fr_from_u32(7U)};
        auto input = providers({tensor, tensor}, 4U);
        require(endpoint(inverse, input, rho) == FrWire{}, "Cartesian padded inverse constants must be masked");
        std::vector<SegmentedTensorProvider> weighted;
        std::string error;
        require(pointwise_providers(inverse, input, rho, weighted, &error), "inverse providers");
        const std::vector<FrWire> lsb_point = {zkdl::fr_from_u32(11U), zkdl::fr_from_u32(13U), zkdl::fr_from_u32(17U), zkdl::fr_from_u32(19U)};
        std::vector<FrWire> all_terminal, base_terminal;
        require(zksc_segmented_terminal_values(weighted, lsb_point, all_terminal, &error) && zksc_segmented_terminal_values(input, lsb_point, base_terminal, &error), "segmented public folds");
        const std::vector<FrWire> msb_point(lsb_point.rbegin(), lsb_point.rend());
        require(eval_constraint(pointwise_sumcheck_descriptor(inverse, input.size()), all_terminal) == eval_constraint(pointwise_terminal_descriptor(inverse, rho, msb_point), base_terminal), "EqMask terminal ordering/Cartesian MLE mismatch");
        auto corrupted = tensor;
        corrupted[0] = FrWire{};
        require(endpoint(inverse, providers({corrupted, tensor}, 4U), rho) != FrWire{}, "invalid active inverse slot must fail");
        std::vector<FrWire> padded_error(16U), padding_selector(16U);
        padded_error[3U] = zkdl::fr_from_u32(1U);
        padded_error[7U] = zkdl::fr_neg(zkdl::fr_from_u32(1U));
        for (std::size_t slot = 0U; slot < padding_selector.size(); ++slot) padding_selector[slot] = zkdl::fr_from_u32(pointwise_active_slot(inverse, slot) ? 0U : 1U);
        require(endpoint(qpad_constraint_descriptor("check/padding"), providers({padded_error, padding_selector}, 4U), rho) != FrWire{}, "canceling padding slots must fail");
        const auto consistency = field_integer_consistency_constraints("check/consistency", 0U, 1U, 2U, 3U);
        const std::vector<FrWire> malicious = {zkdl::fr_from_u32(2U), zkdl::fr_from_u32(3U), zkdl::fr_from_u32(5U), zkdl::fr_from_u32(1U)};
        require(eval_constraint(consistency[0], malicious) != FrWire{} && eval_constraint(consistency[1], malicious) != FrWire{}, "free consistency residual must not repair relation");
        const std::vector<FrWire> honest = {zkdl::fr_from_u32(2U), zkdl::fr_from_u32(3U), zkdl::fr_from_u32(6U), FrWire{}};
        require(eval_constraint(consistency[0], honest) == FrWire{} && eval_constraint(consistency[1], honest) == FrWire{}, "honest consistency relation");
        std::vector<zkdl::byte> encoded;
        append_constraint(encoded, inverse);
        ConstraintDescriptor decoded;
        std::size_t offset = 0U;
        require(read_constraint(encoded, offset, decoded) && offset == encoded.size() && decoded.logical_shape == inverse.logical_shape && decoded.padded_shape == inverse.padded_shape, "active shape descriptor wire");
        require(Proof::kVersion == 7U && pointwise_sumcheck_descriptor(inverse, 2U).degree == 4U, "new protocol/degree");
        std::cout << "pointwise_algebra_check=PASS canceling_slots=REJECT padded_constants=MASKED padding_cancel=REJECT free_residual=REJECT terminal_mle=MATCH protocol=GATR7\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
