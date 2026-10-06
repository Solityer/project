#include "zkrelation-proof.cuh"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace zkdl_gat::relation;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

OperationDescriptor canonical_sorted_lookup() {
    OperationDescriptor operation;
    operation.id = "hidden/layer0/head0/leaky_relu";
    operation.kind = RelationKind::TableLookup;
    operation.degree = 2U;
    operation.mapping = "operation_view_v1->canonical_mle:" + operation.id;
    operation.padding = "fixed-zero-padding;authenticated-sorted-lookup-v1";
    operation.terminal_claim = "terminal_claim_v1|lookup";
    operation.terminal_point = "FS:operation:" + operation.id;
    operation.requires_reindex = true;
    operation.batch_family = "authenticated_sorted_lookup";
    operation.public_table_ref = "static_lrelu";
    operation.source_refs = {
        "hidden/layer0/head0/trace/score",
        "static_lrelu",
        "hidden/layer0/head0/trace/lrelu_idx",
        "hidden/layer0/head0/trace/lrelu_bits",
        "hidden/layer0/head0/trace/leaky_relu",
        operation.id + "/view/query",
        operation.id + "/view/index",
        operation.id + "/view/output0",
    };
    operation.tensor_refs = {
        operation.id + "/view/query",
        operation.id + "/view/index",
        operation.id + "/view/output0",
    };

    const auto add_identity_binding = [&](const std::string& view, const std::string& canonical) {
        FacBindingDescriptor binding;
        binding.view_ref = view;
        binding.canonical_ref = canonical;
        binding.view_outer_domain = 2U;
        binding.canonical_outer_domain = 2U;
        binding.view_feature_domain = 2U;
        binding.canonical_feature_domain = 2U;
        binding.source_outer_index = {0U, 1U};
        binding.source_feature_index = {0U, 1U};
        binding.selector_active = {1U, 1U};
        operation.fac_bindings.push_back(std::move(binding));
    };
    add_identity_binding(operation.id + "/view/query", operation.source_refs[0U]);
    add_identity_binding(operation.id + "/view/index", operation.source_refs[2U]);
    add_identity_binding(operation.id + "/view/output0", operation.source_refs[4U]);
    std::sort(operation.fac_bindings.begin(), operation.fac_bindings.end(),
              [](const FacBindingDescriptor& lhs, const FacBindingDescriptor& rhs) {
                  return std::pair<std::string, std::string>(lhs.view_ref, lhs.canonical_ref) <
                         std::pair<std::string, std::string>(rhs.view_ref, rhs.canonical_ref);
              });
    return operation;
}

void require_rejected(const OperationDescriptor& operation, const std::string& label) {
    require(!operation.authenticated_sorted_lookup_body_valid(), label + " passed specialized body validation");
    std::string error;
    require(!operation.valid(&error), label + " passed complete descriptor validation");
}

}  // namespace

int main() {
    try {
        const OperationDescriptor canonical = canonical_sorted_lookup();
        require(canonical.authenticated_sorted_lookup_body_valid(), "canonical specialized body rejected");
        std::string error;
        require(canonical.valid(&error), "canonical specialized descriptor rejected: " + error);

        auto wrong_family = canonical;
        wrong_family.batch_family = "unknown_lookup_family";
        require_rejected(wrong_family, "wrong family");

        auto wrong_kind = canonical;
        wrong_kind.kind = RelationKind::Projection;
        require_rejected(wrong_kind, "wrong relation kind");

        auto missing_table_source = canonical;
        missing_table_source.source_refs.erase(missing_table_source.source_refs.begin() + 1);
        require_rejected(missing_table_source, "missing public table source");

        auto table_as_tensor = canonical;
        table_as_tensor.tensor_refs[0U] = table_as_tensor.public_table_ref;
        require_rejected(table_as_tensor, "public table used as witness tensor");

        auto missing_fac = canonical;
        missing_fac.fac_bindings.pop_back();
        require_rejected(missing_fac, "missing identity FAC");

        auto wrong_fac_map = canonical;
        wrong_fac_map.fac_bindings[0U].source_outer_index = {1U, 0U};
        require_rejected(wrong_fac_map, "non-identity FAC map");

        auto wrong_fac_target = canonical;
        const auto query_binding = std::find_if(
            wrong_fac_target.fac_bindings.begin(), wrong_fac_target.fac_bindings.end(),
            [&](const FacBindingDescriptor& binding) {
                return binding.view_ref == wrong_fac_target.id + "/view/query";
            });
        require(query_binding != wrong_fac_target.fac_bindings.end(), "query FAC binding absent in test fixture");
        query_binding->canonical_ref = wrong_fac_target.source_refs[2U];
        require_rejected(wrong_fac_target, "wrong FAC canonical target");

        auto wrong_view = canonical;
        wrong_view.fac_bindings[0U].view_ref = "foreign/view";
        require_rejected(wrong_view, "wrong FAC view");

        std::cout << "authenticated_sorted_lookup_descriptor=PASS positive=1 negative=8\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "authenticated_sorted_lookup_descriptor=FAIL error=" << exception.what() << "\n";
        return 1;
    }
}
