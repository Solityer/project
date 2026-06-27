#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
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

void add_output_challenges(
    std::map<std::string, FieldElement>& challenges,
    std::size_t head_index,
    bool legacy_single_output,
    std::uint64_t& seed) {
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
            gatzk::protocol::output_challenge_name(base, head_index, legacy_single_output),
            ++seed);
    }
}

ProtocolContext make_context(const CaseSpec& spec) {
    if (spec.edge_domain_size == 0 || (spec.edge_domain_size & (spec.edge_domain_size - 1U)) != 0U) {
        throw std::runtime_error("edge domain size must be a non-zero power of two");
    }
    ProtocolContext context;
    context.model.has_real_multihead = true;
    context.model.hidden_heads.resize(spec.hidden_head_count);
    context.model.output_layer.head_count = spec.output_head_count;
    context.model.output_layer.heads.resize(spec.output_head_count);
    context.model.K_out = spec.output_head_count;
    context.domains.edge = RootOfUnityDomain::create("edge", spec.edge_domain_size);
    return context;
}

std::map<std::string, FieldElement> make_challenges(const CaseSpec& spec) {
    std::map<std::string, FieldElement> challenges;
    std::uint64_t seed = 1;
    set_field(challenges, "alpha_quot", ++seed);
    for (std::size_t head = 0; head < spec.hidden_head_count; ++head) {
        add_hidden_challenges(challenges, head, seed);
    }
    const bool legacy_single_output = spec.output_head_count == 1;
    for (std::size_t head = 0; head < spec.output_head_count; ++head) {
        add_output_challenges(challenges, head, legacy_single_output, seed);
    }
    return challenges;
}

std::map<std::string, FieldElement> make_witness_scalars(const CaseSpec& spec) {
    std::map<std::string, FieldElement> witness;
    std::uint64_t seed = 2000;
    for (std::size_t head = 0; head < spec.hidden_head_count; ++head) {
        const auto suffix = "_h" + std::to_string(head);
        set_field(witness, "S_src" + suffix, ++seed);
        set_field(witness, "S_t" + suffix, ++seed);
        set_field(witness, "S_dst" + suffix, ++seed);
    }
    const bool legacy_single_output = spec.output_head_count == 1;
    for (std::size_t head = 0; head < spec.output_head_count; ++head) {
        set_field(
            witness,
            gatzk::protocol::output_witness_scalar_name("src", head, legacy_single_output),
            ++seed);
        set_field(
            witness,
            gatzk::protocol::output_witness_scalar_name("t", head, legacy_single_output),
            ++seed);
        set_field(
            witness,
            gatzk::protocol::output_witness_scalar_name("dst", head, legacy_single_output),
            ++seed);
    }
    return witness;
}

FieldElement synthetic_eval(const std::string& label, const FieldElement&) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto ch : label) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 1099511628211ULL;
    }
    return FieldElement(23 + (hash % 1000003ULL));
}

void run_case(const CaseSpec& spec, std::size_t repeats, std::size_t requested_sample_points) {
    const auto context = make_context(spec);
    const auto challenges = make_challenges(spec);
    const auto witness = make_witness_scalars(spec);
    const auto sample_points = std::min(
        requested_sample_points == 0 ? spec.edge_domain_size : requested_sample_points,
        spec.edge_domain_size);
    const auto sample_stride = std::max<std::size_t>(1, spec.edge_domain_size / sample_points);
    const auto step_root = context.domains.edge->omega.pow(sample_stride);
    const auto shift = FieldElement(7);

    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
        std::size_t eval_calls = 0;
        const gatzk::protocol::EvalFn eval = [&](const std::string& label, const FieldElement& point) {
            ++eval_calls;
            return synthetic_eval(label, point);
        };

        auto z = shift;
        auto sink = FieldElement::zero();
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < sample_points; ++i) {
            sink += gatzk::protocol::evaluate_t_edge(context, challenges, witness, eval, z);
            z *= step_root;
        }
        const auto total_ms = elapsed_ms(start);
        const auto projected_full_domain_ms =
            total_ms * (static_cast<double>(spec.edge_domain_size) / static_cast<double>(sample_points));
        std::cout
            << spec.dataset << ','
            << spec.edge_domain_size << ','
            << sample_points << ','
            << sample_stride << ','
            << spec.hidden_head_count << ','
            << spec.output_head_count << ','
            << repeat << ','
            << eval_calls << ','
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
        const std::vector<CaseSpec> cases = {
            {"cora", 131072, 8, 1},
            {"citeseer", 131072, 8, 1},
            {"pubmed", 1048576, 8, 1},
            {"ppi", 1048576, 1, 1},
            {"ogbn-arxiv", 1048576, 1, 1},
        };
        std::cout
            << "dataset,edge_domain_size,sample_points,sample_stride,hidden_head_count,"
            << "output_head_count,repeat,eval_calls,total_ms,projected_full_domain_ms,"
            << "ns_per_sample_eval,accumulator\n";
        for (const auto& spec : cases) {
            run_case(spec, repeats, sample_points);
        }
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
