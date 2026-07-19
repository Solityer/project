#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "gatzk/algebra/polynomial.hpp"
#include "gatzk/protocol/proof.hpp"
#include "gatzk/protocol/quotients.hpp"
#include "gatzk/protocol/schema.hpp"

namespace {

using gatzk::algebra::FieldElement;
using gatzk::algebra::RootOfUnityDomain;
using gatzk::protocol::ProtocolContext;

struct CaseSpec {
    std::string dataset;
    std::size_t edge_domain_size = 0;
    std::size_t hidden_head_count = 0;
    std::size_t output_head_count = 1;
    std::size_t class_count = 0;
    std::size_t range_bits = 0;
};

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void set_field(std::map<std::string, FieldElement>& values, const std::string& name, std::uint64_t seed) {
    values.emplace(name, FieldElement(17 + ((seed * 1315423911ULL) % 104729ULL)));
}

void add_hidden_challenges(
    std::map<std::string, FieldElement>& challenges,
    std::size_t head_index,
    std::uint64_t& seed) {
    const auto suffix = "_h" + std::to_string(head_index);
    for (std::string_view base : {
             "beta_src",
             "beta_dst",
             "eta_src",
             "eta_dst",
             "beta_L",
             "eta_L",
             "beta_R",
             "beta_exp",
             "eta_exp",
             "beta_ELU",
             "eta_ELU",
             "beta_t",
             "eta_t",
         }) {
        set_field(challenges, std::string(base) + suffix, ++seed);
    }
}

ProtocolContext make_context(const CaseSpec& spec) {
    if (spec.edge_domain_size == 0 || (spec.edge_domain_size & (spec.edge_domain_size - 1U)) != 0U) {
        throw std::runtime_error("edge domain size must be a non-zero power of two");
    }
    ProtocolContext context;
    context.config.range_bits = spec.range_bits;
    context.model.has_real_multihead = true;
    context.model.L = 2;
    context.model.hidden_heads.resize(spec.hidden_head_count);
    context.model.hidden_layers = {{
        .layer_index = 0,
        .input_dim = 1,
        .shape = {.head_count = spec.hidden_head_count, .head_dim = 8},
        .heads = std::vector<gatzk::model::AttentionHeadParameters>(
            spec.hidden_head_count),
    }};
    context.model.hidden_profile = {{
        .head_count = spec.hidden_head_count,
        .head_dim = 8,
    }};
    context.model.d_in_profile = {1};
    context.model.K_out = spec.output_head_count;
    context.model.C = spec.class_count;
    context.model.output_layer.input_dim = spec.hidden_head_count * 8;
    context.model.output_layer.output_dim = spec.class_count;
    context.model.output_layer.head_count = spec.output_head_count;
    context.model.output_layer.heads.resize(spec.output_head_count);
    context.local.num_classes = spec.class_count;
    context.domains.edge = RootOfUnityDomain::create("edge", spec.edge_domain_size);
    context.domains.fh = RootOfUnityDomain::create("FH", 2);
    context.domains.n = RootOfUnityDomain::create("N", 2);
    context.domains.in = RootOfUnityDomain::create("in", 2);
    context.domains.d = RootOfUnityDomain::create("d", 2);
    context.domains.cat = RootOfUnityDomain::create("cat", 2);
    std::size_t class_domain_size = 2;
    while (class_domain_size <= spec.class_count) {
        class_domain_size *= 2;
    }
    context.domains.c = RootOfUnityDomain::create("C", class_domain_size);
    auto compiler_artifact =
        std::make_shared<gatzk::protocol::ProtocolCompilerArtifact>();
    auto register_blinder = [&](const std::string& object_id) {
        compiler_artifact->observation_ledger.push_back({
            .id = "benchmark:" + object_id,
            .object_or_challenge = object_id,
            .query_bound = 2,
        });
    };
    auto register_head_relations = [&](const std::string& prefix) {
        for (const auto* role : {"src", "dst"}) {
            register_blinder(
                prefix + "_AggBind_" + role + "_R_edge");
        }
        for (const auto* role : {"src", "dst", "t"}) {
            register_blinder(prefix + "_R_" + role);
        }
    };
    for (std::size_t head = 0; head < spec.hidden_head_count; ++head) {
        register_head_relations(
            gatzk::protocol::hidden_head_prefix(head));
    }
    for (std::size_t head = 0; head < spec.output_head_count; ++head) {
        const auto prefix = gatzk::protocol::output_head_prefix(head);
        register_head_relations(prefix);
        register_blinder(
            gatzk::protocol::output_bias_basis_label(head, "Coeff"));
        register_blinder(
            gatzk::protocol::output_bias_basis_chunk_accumulator_label(
                head, 0));
        register_blinder(
            gatzk::protocol::output_bias_view_label(head, "R_can"));
        register_blinder(
            gatzk::protocol::output_bias_view_label(head, "R_qry"));
    }
    context.compiler_artifact = std::move(compiler_artifact);
    return context;
}

std::map<std::string, FieldElement> make_challenges(const CaseSpec& spec) {
    std::map<std::string, FieldElement> challenges;
    std::uint64_t seed = 1;
    set_field(challenges, "alpha_quot", ++seed);
    set_field(challenges, "y_out", ++seed);
    for (std::size_t head = 0; head < spec.hidden_head_count; ++head) {
        add_hidden_challenges(challenges, head, seed);
    }
    for (std::size_t head = 0; head < spec.output_head_count; ++head) {
        for (std::string_view base : {
                 "beta_src_out",
                 "beta_dst_out",
                 "eta_src_out",
                 "eta_dst_out",
                 "beta_L_out",
                 "eta_L_out",
                 "beta_R_out",
                 "beta_exp_out",
                 "eta_exp_out",
                 "beta_t_out",
                 "eta_t_out",
             }) {
            set_field(
                challenges,
                gatzk::protocol::output_challenge_name(base, head),
                ++seed);
        }
    }
    return challenges;
}

void add_signed_division_challenges(
    std::map<std::string, FieldElement>& challenges,
    const std::string& prefix,
    std::size_t range_bits,
    std::uint64_t& seed) {
    for (const auto* value_name : {"A", "DMinus1", "Q", "R", "Delta"}) {
        for (std::size_t limb = 0;
             limb < gatzk::protocol::signed_division_limb_count(range_bits);
             ++limb) {
            set_field(
                challenges,
                gatzk::protocol::signed_division_beta_name(prefix, value_name, limb),
                ++seed);
        }
    }
}

FieldElement synthetic_eval(const std::string& label, const FieldElement&) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto ch : label) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 1099511628211ULL;
    }
    return FieldElement(23 + (hash % 1000003ULL));
}

void run_case(
    const CaseSpec& spec,
    std::size_t repeats,
    std::size_t requested_sample_points,
    std::size_t requested_workers,
    bool use_precomputed_powers) {
    const auto context = make_context(spec);
    auto challenges = make_challenges(spec);
    std::uint64_t challenge_seed = 100000;
    for (std::size_t head = 0; head < spec.hidden_head_count; ++head) {
        add_signed_division_challenges(
            challenges,
            "P_h" + std::to_string(head) + "_Div_",
            context.config.range_bits,
            challenge_seed);
    }
    for (std::size_t head = 0; head < spec.output_head_count; ++head) {
        add_signed_division_challenges(
            challenges,
            gatzk::protocol::output_head_prefix(head) + "_Div_",
            context.config.range_bits,
            challenge_seed);
        set_field(
            challenges,
            gatzk::protocol::output_bias_view_beta_label(head),
            ++challenge_seed);
        set_field(
            challenges,
            gatzk::protocol::output_bias_view_eta_label(head),
            ++challenge_seed);
        set_field(
            challenges,
            gatzk::protocol::output_bias_basis_gamma_label(head),
            ++challenge_seed);
    }
    std::map<std::string, FieldElement> external_evaluations;
    const auto evaluation_plan =
        gatzk::protocol::build_t_edge_evaluation_plan(context, challenges);
    add_signed_division_challenges(
        challenges,
        "P_avg_Div_",
        context.config.range_bits,
        challenge_seed);
    set_field(external_evaluations, "mu_out", ++challenge_seed);
    for (std::size_t head = 0; head < spec.output_head_count; ++head) {
        set_field(
            external_evaluations,
            gatzk::protocol::output_external_eval_name("y", head),
            ++challenge_seed);
    }
    for (const auto& binding : gatzk::protocol::build_binding_relation_metadata(context)) {
        for (const auto& relation : binding.aggregation_relations) {
            for (const auto* route : {&relation.source_route, &relation.target_route}) {
                set_field(challenges, route->beta_challenge_label, ++challenge_seed);
                set_field(challenges, route->eta_challenge_label, ++challenge_seed);
            }
        }
    }
    constexpr std::size_t kFormalQuotientCosetMultiplier = 4;
    const auto quotient_coset_size =
        spec.edge_domain_size * kFormalQuotientCosetMultiplier;
    const auto sample_points = std::min(
        requested_sample_points == 0 ? quotient_coset_size : requested_sample_points,
        quotient_coset_size);
    const auto sample_stride = std::max<std::size_t>(
        1, quotient_coset_size / sample_points);
    const auto step_root = FieldElement::root_of_unity(
        quotient_coset_size).pow(sample_stride);
    const auto shift = FieldElement(7);
    std::vector<FieldElement> quotient_alpha_powers;
    const auto edge_head_count = context.model.hidden_heads.size()
        + context.model.output_layer.heads.size();
    if (use_precomputed_powers) {
        const auto power_count = std::size_t{2048}
            + std::size_t{1024} * edge_head_count;
        quotient_alpha_powers.reserve(power_count);
        auto alpha_power = FieldElement::one();
        const auto alpha = challenges.at("alpha_quot");
        for (std::size_t i = 0; i < power_count; ++i) {
            quotient_alpha_powers.push_back(alpha_power);
            alpha_power *= alpha;
        }
    }

    std::size_t audited_eval_calls = 0;
    std::unordered_set<std::string> audited_eval_keys;
    std::unordered_set<std::string> audited_labels;
    const auto audit_omega = shift * context.domains.edge->omega;
    const gatzk::protocol::EvalFn audit_eval = [&](const std::string& label, const FieldElement& point) {
        ++audited_eval_calls;
        audited_labels.insert(label);
        audited_eval_keys.insert(label + (point == shift ? "@z" : point == audit_omega ? "@omega" : "@other"));
        return synthetic_eval(label, point);
    };
    const auto audit_direct = gatzk::protocol::evaluate_t_edge(
        context, challenges, external_evaluations, audit_eval, shift);
    const gatzk::protocol::EvalFn plan_audit_eval = [&](
        const std::string& label, const FieldElement& point) {
        return synthetic_eval(label, point);
    };
    const auto audit_planned = gatzk::protocol::evaluate_t_edge(
        context, challenges, external_evaluations, plan_audit_eval, shift,
        false, nullptr, nullptr, nullptr, evaluation_plan.get());
    const gatzk::protocol::QuotientBoundaryValues audit_boundary{
        .first = context.domains.edge->lagrange_basis_eval(0, shift),
        .last = context.domains.edge->lagrange_basis_eval(
            context.domains.edge->size - 1, shift),
        .zero = context.domains.edge->zero_polynomial_eval(shift),
    };
    const auto audit_precomputed_boundary =
        gatzk::protocol::evaluate_t_edge(
            context, challenges, external_evaluations, plan_audit_eval, shift,
            false, nullptr, nullptr, nullptr, evaluation_plan.get(),
            &audit_boundary);
    if (audit_direct != audit_planned
        || audit_direct != audit_precomputed_boundary) {
        throw std::runtime_error(
            "t_edge evaluation plan/boundary changed the quotient value");
    }
    std::unordered_map<std::string, FieldElement> eval_values;
    eval_values.reserve(audited_labels.size() * 2);
    for (const auto& label : audited_labels) {
        eval_values.emplace(label, synthetic_eval(label, shift));
    }

    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto worker_count = std::max<std::size_t>(
            1, std::min(requested_workers, sample_points));
        auto run_range = [&](std::size_t begin, std::size_t end) {
            std::size_t local_eval_calls = 0;
            const gatzk::protocol::EvalFn eval = [&](
                const std::string& label, const FieldElement&) {
                ++local_eval_calls;
                const auto it = eval_values.find(label);
                if (it == eval_values.end()) {
                    throw std::runtime_error(
                        "t_edge benchmark lookup is missing label: " + label);
                }
                return it->second;
            };
            auto local_z = shift * step_root.pow(begin);
            auto local_sink = FieldElement::zero();
            for (std::size_t i = begin; i < end; ++i) {
                local_sink += gatzk::protocol::evaluate_t_edge(
                    context,
                    challenges,
                    external_evaluations,
                    eval,
                    local_z,
                    false,
                    nullptr,
                    nullptr,
                    quotient_alpha_powers.empty()
                        ? nullptr
                        : &quotient_alpha_powers,
                    evaluation_plan.get());
                local_z *= step_root;
            }
            return std::pair{local_sink, local_eval_calls};
        };

        auto sink = FieldElement::zero();
        std::size_t eval_calls = 0;
        const auto start = std::chrono::steady_clock::now();
        if (worker_count == 1) {
            auto result = run_range(0, sample_points);
            sink = std::move(result.first);
            eval_calls = result.second;
        } else {
            const auto chunk_size = (sample_points + worker_count - 1) / worker_count;
            std::vector<std::future<std::pair<FieldElement, std::size_t>>> futures;
            futures.reserve(worker_count);
            for (std::size_t worker = 0; worker < worker_count; ++worker) {
                const auto begin = worker * chunk_size;
                const auto end = std::min(sample_points, begin + chunk_size);
                if (begin >= end) {
                    break;
                }
                futures.push_back(std::async(std::launch::async, run_range, begin, end));
            }
            for (auto& future : futures) {
                auto result = future.get();
                sink += result.first;
                eval_calls += result.second;
            }
        }
        const auto total_ms = elapsed_ms(start);
        const auto projected_full_domain_ms =
            total_ms * (static_cast<double>(quotient_coset_size)
                / static_cast<double>(sample_points));
        std::cout
            << spec.dataset << ','
            << spec.edge_domain_size << ','
            << quotient_coset_size << ','
            << sample_points << ','
            << sample_stride << ','
            << spec.hidden_head_count << ','
            << spec.output_head_count << ','
            << spec.range_bits << ','
            << repeat << ','
            << worker_count << ','
            << (use_precomputed_powers ? "precomputed" : "incremental") << ','
            << eval_calls << ','
            << audited_eval_calls << ','
            << audited_eval_keys.size() << ','
            << std::fixed << std::setprecision(3)
            << total_ms << ','
            << projected_full_domain_ms << ','
            << ((total_ms * 1000000.0) / static_cast<double>(sample_points)) << ','
            << sink.to_string() << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::size_t repeats = 1;
        if (argc > 1) {
            repeats = static_cast<std::size_t>(std::stoull(argv[1]));
        }
        std::size_t sample_points = 8192;
        if (argc > 2) {
            sample_points = static_cast<std::size_t>(std::stoull(argv[2]));
        }
        const std::string selected_case = argc > 3 ? argv[3] : "all";
        const auto workers = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4]))
            : 1;
        const std::string power_mode = argc > 5 ? argv[5] : "auto";
        if (workers == 0 || argc > 6
            || (power_mode != "auto" && power_mode != "on" && power_mode != "off")) {
            throw std::runtime_error(
                "usage: gatzk_t_edge_eval_stage_bench [repeats] [sample_points] [case|all] [workers] [auto|on|off]");
        }
        const std::vector<CaseSpec> cases = {
            {"cora_formal_shape", 32768, 8, 1, 7, 12},
            {"citeseer_formal_shape", 32768, 8, 1, 6, 12},
            {"pubmed_formal_shape", 262144, 8, 1, 3, 12},
            {"ppi_formal_shape", 1048576, 1, 1, 121, 14},
            {"ogbn-arxiv_formal_shape", 1048576, 1, 1, 40, 12},
        };
        std::cout
            << "dataset,edge_domain_size,quotient_coset_size,sample_points,sample_stride,hidden_head_count,"
            << "output_head_count,range_bits,repeat,workers,alpha_power_mode,eval_calls,"
            << "audited_eval_calls_per_point,"
            << "audited_unique_eval_keys_per_point,total_ms,projected_full_domain_ms,"
            << "ns_per_sample_eval,accumulator\n";
        bool ran_case = false;
        for (const auto& spec : cases) {
            if (selected_case == "all" || selected_case == spec.dataset) {
                const auto head_count = spec.hidden_head_count + spec.output_head_count;
                const bool use_precomputed_powers = power_mode == "on"
                    || (power_mode == "auto" && head_count >= 2);
                run_case(
                    spec, repeats, sample_points, workers,
                    use_precomputed_powers);
                ran_case = true;
            }
        }
        if (!ran_case) {
            throw std::runtime_error("unknown t_edge benchmark case: " + selected_case);
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
