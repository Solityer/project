#include "gatzk/protocol/verifier.hpp"

#include <future>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "gatzk/protocol/challenges.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/util/route2.hpp"

namespace gatzk::protocol {
namespace {

using algebra::FieldElement;
using crypto::Commitment;

struct BundleVerificationInput {
    const DomainOpeningBundle* bundle = nullptr;
    FieldElement folding_challenge = FieldElement::zero();
    std::vector<Commitment> commitments;
    std::vector<std::vector<FieldElement>> values;
};

const DomainOpeningBundle* bundle_by_name(const Proof& proof, const std::string& name) {
    for (const auto& [bundle_name, bundle] : proof.domain_openings) {
        if (bundle_name == name) {
            return &bundle;
        }
    }
    throw std::runtime_error("missing domain bundle: " + name);
}

FieldElement bundle_value(const DomainOpeningBundle& bundle, const std::string& label, const FieldElement& point) {
    for (const auto& [name, values] : bundle.values) {
        if (name != label) {
            continue;
        }
        for (std::size_t i = 0; i < bundle.points.size(); ++i) {
            if (bundle.points[i] == point) {
                return values[i];
            }
        }
    }
    throw std::runtime_error("missing opened value for " + label);
}

FieldElement external_value(const Proof& proof, const std::string& label) {
    for (const auto& [name, value] : proof.external_evaluations) {
        if (name == label) {
            return value;
        }
    }
    throw std::runtime_error("missing external evaluation: " + label);
}

FieldElement bias_fold(const ProtocolContext& context, const FieldElement& y_out) {
    FieldElement out = FieldElement::zero();
    for (std::size_t i = 0; i < context.local.num_nodes; ++i) {
        for (std::size_t j = 0; j < context.model.b.size(); ++j) {
            out += context.model.b[j] * y_out.pow(static_cast<std::uint64_t>(i * context.model.b.size() + j));
        }
    }
    return out;
}

std::unordered_map<std::string, Commitment> to_map(const std::vector<std::pair<std::string, Commitment>>& entries) {
    std::unordered_map<std::string, Commitment> out;
    for (const auto& [name, commitment] : entries) {
        out[name] = commitment;
    }
    return out;
}

std::map<std::string, FieldElement> to_field_map(const std::vector<std::pair<std::string, FieldElement>>& entries) {
    std::map<std::string, FieldElement> out;
    for (const auto& [name, value] : entries) {
        out[name] = value;
    }
    return out;
}

std::size_t hidden_head_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead && !parameters.hidden_heads.empty()) {
        return parameters.hidden_heads.front().output_bias_fp.size();
    }
    return config.hidden_dim;
}

std::size_t concat_width(const model::ModelParameters& parameters, const util::AppConfig& config) {
    if (parameters.has_real_multihead && !parameters.hidden_heads.empty()) {
        return hidden_head_width(parameters, config) * parameters.hidden_heads.size();
    }
    return config.hidden_dim;
}

PublicMetadata build_public_metadata(const ProtocolContext& context) {
    PublicMetadata metadata;
    metadata.protocol_id = "gatzkml";
    metadata.model_arch_id = context.model.has_real_multihead
        ? "single_layer_gat_hidden8_output1"
        : "legacy_single_head_debug";
    metadata.model_param_id = context.model.has_real_multihead
        ? ("checkpoint_bundle:" + context.config.checkpoint_bundle)
        : ("synthetic_seed:" + std::to_string(context.config.seed));
    metadata.static_table_id = "tables:lrelu+exp+range";
    metadata.quant_cfg_id = "range_bits=" + std::to_string(context.config.range_bits);
    metadata.domain_cfg =
        "FH=" + context.domains.fh->name + ":" + std::to_string(context.domains.fh->size)
        + ",edge=" + context.domains.edge->name + ":" + std::to_string(context.domains.edge->size)
        + ",in=" + context.domains.in->name + ":" + std::to_string(context.domains.in->size)
        + ",d_h=" + context.domains.d->name + ":" + std::to_string(context.domains.d->size)
        + ",cat=" + context.domains.cat->name + ":" + std::to_string(context.domains.cat->size)
        + ",C=" + context.domains.c->name + ":" + std::to_string(context.domains.c->size)
        + ",N=" + context.domains.n->name + ":" + std::to_string(context.domains.n->size);
    metadata.dim_cfg =
        "N=" + std::to_string(context.local.num_nodes)
        + ",E=" + std::to_string(context.local.edges.size())
        + ",d_in=" + std::to_string(context.local.num_features)
        + ",d_h=" + std::to_string(hidden_head_width(context.model, context.config))
        + ",d_cat=" + std::to_string(concat_width(context.model, context.config))
        + ",C=" + std::to_string(context.local.num_classes);
    metadata.encoding_id = "project-fixed-order-v1";
    metadata.padding_rule_id = "zero-pad+selector-mask";
    metadata.degree_bound_id = context.model.has_real_multihead
        ? "note-target:FH,edge,in,d_h,cat,C,N"
        : "legacy:FH,edge,in,d,N";
    return metadata;
}

bool metadata_matches(const PublicMetadata& lhs, const PublicMetadata& rhs) {
    return lhs.protocol_id == rhs.protocol_id
        && lhs.model_arch_id == rhs.model_arch_id
        && lhs.model_param_id == rhs.model_param_id
        && lhs.static_table_id == rhs.static_table_id
        && lhs.quant_cfg_id == rhs.quant_cfg_id
        && lhs.domain_cfg == rhs.domain_cfg
        && lhs.dim_cfg == rhs.dim_cfg
        && lhs.encoding_id == rhs.encoding_id
        && lhs.padding_rule_id == rhs.padding_rule_id
        && lhs.degree_bound_id == rhs.degree_bound_id;
}

std::vector<std::string> fixed_proof_block_order() {
    return {"M_pub", "Com_dyn", "S_route", "Eval_ext", "Eval_dom", "Com_quot", "Open_dom", "W_ext", "Pi_bind"};
}

struct ExternalEvalSpec {
    std::string proof_name;
    std::string label;
    std::string challenge_name;
};

std::vector<ExternalEvalSpec> multihead_external_specs(const ProtocolContext& context) {
    std::vector<ExternalEvalSpec> specs;
    for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
        const auto suffix = "h" + std::to_string(head_index);
        const auto prefix = "P_h" + std::to_string(head_index) + "_";
        specs.push_back({"mu_" + suffix + "_proj", prefix + "H_prime", "y_proj_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_src", prefix + "E_src", "y_src_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_dst", prefix + "E_dst", "y_dst_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_star", prefix + "H_star", "y_star_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_agg_pre", prefix + "H_agg_pre_star", "y_agg_pre_h" + std::to_string(head_index)});
        specs.push_back({"mu_" + suffix + "_agg", prefix + "H_agg_star", "y_agg_h" + std::to_string(head_index)});
    }
    specs.push_back({"mu_cat", "P_H_cat_star", "y_cat"});
    specs.push_back({"mu_out_proj", "P_out_Y_prime", "y_proj_out"});
    specs.push_back({"mu_out_src", "P_out_E_src", "y_src_out"});
    specs.push_back({"mu_out_dst", "P_out_E_dst", "y_dst_out"});
    specs.push_back({"mu_out_star", "P_out_Y_star", "y_out_star"});
    specs.push_back({"mu_out", "P_Y", "y_out"});
    return specs;
}

}  // namespace

bool verify(const ProtocolContext& context, const Proof& proof) {
    if (!metadata_matches(proof.public_metadata, build_public_metadata(context))) {
        return false;
    }
    if (proof.block_order != fixed_proof_block_order()) {
        return false;
    }
    if (context.model.has_real_multihead) {
        if (proof.dynamic_commitments.size() != dynamic_commitment_labels(context).size()) {
            return false;
        }
        for (std::size_t i = 0; i < proof.dynamic_commitments.size(); ++i) {
            if (proof.dynamic_commitments[i].first != dynamic_commitment_labels(context)[i]) {
                return false;
            }
        }
        if (proof.quotient_commitments.size() != quotient_commitment_labels(context).size()) {
            return false;
        }
        for (std::size_t i = 0; i < proof.quotient_commitments.size(); ++i) {
            if (proof.quotient_commitments[i].first != quotient_commitment_labels(context)[i]) {
                return false;
            }
        }
        const auto dynamic_commitments = to_map(proof.dynamic_commitments);
        const auto quotient_commitments = to_map(proof.quotient_commitments);
        const auto witness_scalars = to_field_map(proof.witness_scalars);
        const auto external_evaluations = to_field_map(proof.external_evaluations);
        const auto challenges = replay_challenges(context, dynamic_commitments, quotient_commitments);
        if (proof.challenges != challenges) {
            return false;
        }
        for (std::size_t head_index = 0; head_index < context.model.hidden_heads.size(); ++head_index) {
            if (!witness_scalars.contains("S_src_h" + std::to_string(head_index))
                || !witness_scalars.contains("S_dst_h" + std::to_string(head_index))) {
                return false;
            }
        }
        if (!witness_scalars.contains("S_src_out") || !witness_scalars.contains("S_dst_out")) {
            return false;
        }
        for (const auto& spec : multihead_external_specs(context)) {
            if (!external_evaluations.contains(spec.proof_name)) {
                return false;
            }
        }
        struct BundleSpec {
            std::string bundle_name;
            std::string trace_domain_name;
            std::shared_ptr<algebra::RootOfUnityDomain> domain;
            std::string z_name;
            std::string v_name;
            std::string quotient_name;
        };
        const std::vector<BundleSpec> bundle_specs = {
            {"FH", "FH", context.domains.fh, "z_FH", "v_FH", "t_FH"},
            {"edge", "edge", context.domains.edge, "z_edge", "v_edge", "t_edge"},
            {"in", "in", context.domains.in, "z_in", "v_in", "t_in"},
            {"d_h", "d", context.domains.d, "z_d_h", "v_d_h", "t_d_h"},
            {"cat", "cat", context.domains.cat, "z_cat", "v_cat", "t_cat"},
            {"C", "C", context.domains.c, "z_C", "v_C", "t_C"},
            {"N", "N", context.domains.n, "z_N", "v_N", "t_N"},
        };
        const auto eval = [&](const std::string& name, const FieldElement& point) -> FieldElement {
            if (const auto it = context.public_polynomials.find(name); it != context.public_polynomials.end()) {
                return it->second.evaluate(point);
            }
            for (const auto& spec : bundle_specs) {
                const auto* bundle = bundle_by_name(proof, spec.bundle_name);
                for (const auto& [label, _] : bundle->values) {
                    if (label == name) {
                        return bundle_value(*bundle, name, point);
                    }
                }
            }
            throw std::runtime_error("missing opened value for " + name);
        };
        auto verify_bundle_spec = [&](const BundleSpec& spec) {
            const auto* bundle = bundle_by_name(proof, spec.bundle_name);
            auto labels = domain_opening_labels(context, spec.trace_domain_name);
            labels.push_back(spec.quotient_name);
            if (bundle->values.size() != labels.size()) {
                return false;
            }
            if (bundle->points.size() != 2
                || bundle->points[0] != challenges.at(spec.z_name)
                || bundle->points[1] != challenges.at(spec.z_name) * spec.domain->omega) {
                return false;
            }
            std::vector<Commitment> commitments;
            std::vector<std::vector<FieldElement>> values;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (bundle->values[i].first != labels[i]) {
                    return false;
                }
                if (dynamic_commitments.contains(labels[i])) {
                    commitments.push_back(dynamic_commitments.at(labels[i]));
                } else if (quotient_commitments.contains(labels[i])) {
                    commitments.push_back(quotient_commitments.at(labels[i]));
                } else {
                    return false;
                }
                values.push_back(bundle->values[i].second);
            }
            if (!crypto::KZG::verify_batch(
                    commitments,
                    bundle->points,
                    values,
                    challenges.at(spec.v_name),
                    bundle->witness,
                    context.kzg)) {
                return false;
            }
            FieldElement expected_t = FieldElement::zero();
            if (spec.quotient_name == "t_FH") {
                expected_t = evaluate_t_fh(context, challenges, eval, challenges.at(spec.z_name));
            } else if (spec.quotient_name == "t_edge") {
                expected_t = evaluate_t_edge(context, challenges, witness_scalars, eval, challenges.at(spec.z_name));
            } else if (spec.quotient_name == "t_in") {
                expected_t = evaluate_t_in(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
            } else if (spec.quotient_name == "t_d_h") {
                expected_t = evaluate_t_d(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
            } else if (spec.quotient_name == "t_cat") {
                expected_t = evaluate_t_cat(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
            } else if (spec.quotient_name == "t_C") {
                expected_t = evaluate_t_c(context, challenges, external_evaluations, eval, challenges.at(spec.z_name));
            } else if (spec.quotient_name == "t_N") {
                expected_t = evaluate_t_n(context, challenges, witness_scalars, eval, challenges.at(spec.z_name));
            } else {
                return false;
            }
            if (bundle_value(*bundle, spec.quotient_name, challenges.at(spec.z_name)) != expected_t) {
                return false;
            }
            return true;
        };
        std::vector<std::pair<Commitment, FieldElement>> external_commitments;
        std::vector<FieldElement> external_points;
        for (const auto& spec : multihead_external_specs(context)) {
            if (!dynamic_commitments.contains(spec.label)) {
                return false;
            }
            external_commitments.emplace_back(dynamic_commitments.at(spec.label), external_value(proof, spec.proof_name));
            external_points.push_back(challenges.at(spec.challenge_name));
        }
        if (!crypto::KZG::verify_external_fold(
                external_commitments,
                external_points,
                challenges.at("rho_ext"),
                proof.external_witness,
                context.kzg)) {
            return false;
        }
        for (const auto& spec : bundle_specs) {
            if (!verify_bundle_spec(spec)) {
                return false;
            }
        }
        return true;
    }
    const auto dynamic_commitments = to_map(proof.dynamic_commitments);
    const auto quotient_commitments = to_map(proof.quotient_commitments);
    const auto witness_scalars = to_field_map(proof.witness_scalars);
    if (!witness_scalars.contains("S_src") || !witness_scalars.contains("S_dst")) {
        return false;
    }
    const auto challenges = replay_challenges(context, dynamic_commitments, quotient_commitments);
    if (proof.challenges != challenges) {
        return false;
    }

    const auto mu_bias_out = bias_fold(context, challenges.at("y_out"));
    if (external_value(proof, "mu_out") != external_value(proof, "mu_Y_lin") + mu_bias_out) {
        return false;
    }

    const std::vector<std::pair<Commitment, FieldElement>> external_commitments = {
        {dynamic_commitments.at("P_H_prime"), external_value(proof, "mu_proj")},
        {dynamic_commitments.at("P_E_src"), external_value(proof, "mu_src")},
        {dynamic_commitments.at("P_E_dst"), external_value(proof, "mu_dst")},
        {dynamic_commitments.at("P_H_star"), external_value(proof, "mu_star")},
        {dynamic_commitments.at("P_H_agg_star"), external_value(proof, "mu_agg")},
        {dynamic_commitments.at("P_Y_lin"), external_value(proof, "mu_Y_lin")},
        {dynamic_commitments.at("P_Y"), external_value(proof, "mu_out")},
    };
    const std::vector<FieldElement> external_points = {
        challenges.at("y_proj"),
        challenges.at("y_src"),
        challenges.at("y_dst"),
        challenges.at("y_star"),
        challenges.at("y_agg"),
        challenges.at("y_out"),
        challenges.at("y_out"),
    };

    const auto* bundle_fh = bundle_by_name(proof, "FH");
    const auto* bundle_edge = bundle_by_name(proof, "edge");
    const auto* bundle_in = bundle_by_name(proof, "in");
    const auto* bundle_d = bundle_by_name(proof, "d");
    const auto* bundle_n = bundle_by_name(proof, "N");

    const auto eval = [&](const std::string& name, const FieldElement& point) -> FieldElement {
        if (context.public_polynomials.contains(name)) {
            return context.public_polynomials.at(name).evaluate(point);
        }
        if (name == "t_FH" || name == "P_Table_feat" || name == "P_Query_feat" || name == "P_m_feat" || name == "P_R_feat") {
            return bundle_value(*bundle_fh, name, point);
        }
        if (name == "t_edge"
            || name == "P_E_src_edge"
            || name == "P_H_src_star_edge"
            || name == "P_Query_src"
            || name == "P_R_src"
            || name == "P_E_dst_edge"
            || name == "P_M_edge"
            || name == "P_Sum_edge"
            || name == "P_inv_edge"
            || name == "P_H_agg_star_edge"
            || name == "P_Query_dst"
            || name == "P_R_dst"
            || name == "P_Table_L"
            || name == "P_Query_L"
            || name == "P_m_L"
            || name == "P_R_L"
            || name == "P_Table_R"
            || name == "P_Query_R"
            || name == "P_m_R"
            || name == "P_R_R"
            || name == "P_Table_exp"
            || name == "P_Query_exp"
            || name == "P_m_exp"
            || name == "P_R_exp"
            || name == "P_Delta"
            || name == "P_S"
            || name == "P_Z"
            || name == "P_s_max"
            || name == "P_C_max"
            || name == "P_alpha"
            || name == "P_U"
            || name == "P_v_hat"
            || name == "P_PSQ"
            || name == "P_w_psq"
            || name == "P_T_psq_edge") {
            return bundle_value(*bundle_edge, name, point);
        }
        if (name == "t_in" || name == "P_a_proj" || name == "P_b_proj" || name == "P_Acc_proj") {
            return bundle_value(*bundle_in, name, point);
        }
        if (name == "t_d"
            || name == "P_a_src"
            || name == "P_b_src"
            || name == "P_Acc_src"
            || name == "P_a_dst"
            || name == "P_b_dst"
            || name == "P_Acc_dst"
            || name == "P_a_star"
            || name == "P_b_star"
            || name == "P_Acc_star"
            || name == "P_a_agg"
            || name == "P_b_agg"
            || name == "P_Acc_agg"
            || name == "P_a_out"
            || name == "P_b_out"
            || name == "P_Acc_out") {
            return bundle_value(*bundle_d, name, point);
        }
        if (name == "t_N"
            || name == "P_E_src"
            || name == "P_H_star"
            || name == "P_Table_src"
            || name == "P_m_src"
            || name == "P_R_src_node"
            || name == "P_E_dst"
            || name == "P_M"
            || name == "P_Sum"
            || name == "P_inv"
            || name == "P_H_agg_star"
            || name == "P_Table_dst"
            || name == "P_m_dst"
            || name == "P_R_dst_node") {
            return bundle_value(*bundle_n, name, point);
        }
        throw std::runtime_error("missing evaluation in verifier: " + name);
    };

    const std::map<std::string, FieldElement> external_evals = {
        {"mu_proj", external_value(proof, "mu_proj")},
        {"mu_src", external_value(proof, "mu_src")},
        {"mu_dst", external_value(proof, "mu_dst")},
        {"mu_star", external_value(proof, "mu_star")},
        {"mu_agg", external_value(proof, "mu_agg")},
        {"mu_Y_lin", external_value(proof, "mu_Y_lin")},
        {"mu_out", external_value(proof, "mu_out")},
    };

    if (evaluate_t_fh(context, challenges, eval, challenges.at("z_FH")) != eval("t_FH", challenges.at("z_FH"))) {
        return false;
    }
    if (evaluate_t_edge(context, challenges, witness_scalars, eval, challenges.at("z_edge")) != eval("t_edge", challenges.at("z_edge"))) {
        return false;
    }
    if (evaluate_t_n(context, challenges, witness_scalars, eval, challenges.at("z_N")) != eval("t_N", challenges.at("z_N"))) {
        return false;
    }
    if (evaluate_t_in(context, challenges, external_evals, eval, challenges.at("z_in")) != eval("t_in", challenges.at("z_in"))) {
        return false;
    }
    if (evaluate_t_d(context, challenges, external_evals, eval, challenges.at("z_d")) != eval("t_d", challenges.at("z_d"))) {
        return false;
    }

    auto prepare_bundle = [&](const DomainOpeningBundle& bundle, const std::string& challenge_name) {
        BundleVerificationInput input;
        input.bundle = &bundle;
        input.folding_challenge = challenges.at(challenge_name);
        input.commitments.reserve(bundle.values.size());
        input.values.reserve(bundle.values.size());
        for (const auto& [label, opened_values] : bundle.values) {
            if (dynamic_commitments.contains(label)) {
                input.commitments.push_back(dynamic_commitments.at(label));
            } else {
                input.commitments.push_back(quotient_commitments.at(label));
            }
            input.values.push_back(opened_values);
        }
        return input;
    };

    auto verify_bundle = [&](const BundleVerificationInput& input) {
        return crypto::KZG::verify_batch(
            input.commitments,
            input.bundle->points,
            input.values,
            input.folding_challenge,
            input.bundle->witness,
            context.kzg);
    };

    const auto prepared_fh = prepare_bundle(*bundle_fh, "v_FH");
    const auto prepared_edge = prepare_bundle(*bundle_edge, "v_edge");
    const auto prepared_in = prepare_bundle(*bundle_in, "v_in");
    const auto prepared_d = prepare_bundle(*bundle_d, "v_d");
    const auto prepared_n = prepare_bundle(*bundle_n, "v_N");

    const bool run_parallel_verification = std::thread::hardware_concurrency() > 1;
    if (run_parallel_verification) {
        auto external_future = std::async(
            std::launch::async,
            [&]() {
                return crypto::KZG::verify_external_fold(
                    external_commitments,
                    external_points,
                    challenges.at("rho_ext"),
                    proof.external_witness,
                    context.kzg);
            });
        auto fh_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_fh); });
        auto edge_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_edge); });
        auto in_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_in); });
        auto d_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_d); });
        auto n_future = std::async(std::launch::async, [&]() { return verify_bundle(prepared_n); });
        if (!external_future.get() || !fh_future.get() || !edge_future.get() || !in_future.get() || !d_future.get() || !n_future.get()) {
            return false;
        }
    } else {
        if (!crypto::KZG::verify_external_fold(
                external_commitments,
                external_points,
                challenges.at("rho_ext"),
                proof.external_witness,
                context.kzg)) {
            return false;
        }
        if (!verify_bundle(prepared_fh)
            || !verify_bundle(prepared_edge)
            || !verify_bundle(prepared_in)
            || !verify_bundle(prepared_d)
            || !verify_bundle(prepared_n)) {
            return false;
        }
    }

    return true;
}

}  // namespace gatzk::protocol
