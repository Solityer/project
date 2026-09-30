#include "gat_sparse_gkr.hpp"

#include <algorithm>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace zkgpt {
namespace gat {
namespace {

void append_u64(std::string& out, std::uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) out.push_back(static_cast<char>((value >> (byte * 8)) & 0xffU));
}

void append_string(std::string& out, const std::string& value) {
    append_u64(out, static_cast<std::uint64_t>(value.size()));
    out.append(value);
}

void set_reason(std::string* reason, const std::string& value) {
    if (reason != nullptr) *reason = value;
}

bool relation_is_edge_domain(SparseGkrRelationKind kind) {
    return kind == SparseGkrRelationKind::AttentionScore || kind == SparseGkrRelationKind::MaskedSoftmax || kind == SparseGkrRelationKind::Aggregation;
}

bool relation_is_node_domain(SparseGkrRelationKind kind) {
    return kind == SparseGkrRelationKind::SignedRoundDiv || kind == SparseGkrRelationKind::Residual || kind == SparseGkrRelationKind::OutputBinding;
}

bool relation_domain(const SparseGkrRelationSpec& relation, std::uint64_t* entries) {
    const std::uint64_t width = std::max<std::uint64_t>(1, relation.value_width);
    return sparse_gkr_checked_mul(relation.item_count, width, entries) && *entries != 0;
}

std::size_t domain_rounds(const SparseGkrRelationSpec& relation) {
    std::uint64_t entries = 0;
    if (!relation_domain(relation, &entries)) throw std::invalid_argument("sparse GKR relation domain is invalid");
    std::uint64_t domain = 0;
    if (!sparse_gkr_next_power_of_two(entries, &domain)) throw std::overflow_error("sparse GKR relation domain overflows u64");
    std::size_t rounds = 0;
    while (domain > 1) {
        domain >>= 1;
        ++rounds;
    }
    return rounds;
}

bool same_message(const NativeProofMessage& left, const NativeProofMessage& right) {
    return left.kind == right.kind && left.label == right.label && left.payload == right.payload && left.sequence == right.sequence;
}

bool schedule_prefix_matches(const NativeProofSession& session, const SparseGkrPlan& plan, std::string* reason) {
    NativeProofSession expected(session.domain(), session.statement());
    if (!absorb_sparse_gkr_schedule(expected, plan, reason)) return false;
    const auto& actual_messages = session.messages();
    const auto& expected_messages = expected.messages();
    if (actual_messages.size() < expected_messages.size()) {
        set_reason(reason, "sparse GKR transcript is missing public schedule binding");
        return false;
    }
    for (std::size_t index = 0; index < expected_messages.size(); ++index) {
        if (!same_message(actual_messages[index], expected_messages[index])) {
            set_reason(reason, "sparse GKR transcript schedule prefix mismatch");
            return false;
        }
    }
    return true;
}

bool append_relation_encoding(std::string& out, const SparseGkrRelationSpec& relation) {
    append_string(out, relation.relation_id);
    append_string(out, relation.family_id);
    out.push_back(static_cast<char>(relation.kind));
    append_u64(out, relation.item_count);
    append_u64(out, relation.value_width);
    append_u64(out, relation.input_offset);
    append_u64(out, relation.source_offset);
    append_u64(out, relation.destination_offset);
    append_u64(out, relation.output_offset);
    append_u64(out, relation.chunk_size);
    append_string(out, relation.coordinate_rule);
    append_string(out, relation.group_rule);
    append_string(out, relation.arithmetic_rule);
    append_string(out, relation.overflow_rule);
    append_string(out, relation.terminal_rule);
    append_u64(out, relation.equations.size());
    for (const std::string& equation : relation.equations) append_string(out, equation);
    return true;
}

bool validate_plan(const Graph& graph, const SparseGkrPlan& plan, std::string* reason) {
    const auto fail = [reason](const std::string& message) {
        set_reason(reason, message);
        return false;
    };
    if (plan.schema != kSparseGkrRelationSchema || plan.version != kSparseGkrRelationVersion)
        return fail("sparse GKR relation schema/version is not supported");
    if (plan.chunk_size == 0 || (plan.chunk_size & (plan.chunk_size - 1)) != 0) return fail("sparse GKR chunk size is not a power of two");
    if (plan.graph_nodes != graph.num_nodes || plan.graph_edges != graph.edges() || plan.graph_count != graph.graph_count)
        return fail("sparse GKR graph identity does not match the public graph");
    if (plan.relations.empty()) return fail("sparse GKR plan contains no relations");
    std::uint64_t previous_domain = 0;
    std::set<std::string> relation_ids;
    for (const SparseGkrRelationSpec& relation : plan.relations) {
        std::uint64_t domain = 0;
        if (relation.relation_id.empty() || !relation_ids.insert(relation.relation_id).second || relation.family_id.empty() ||
            !sparse_gkr_relation_kind_valid(static_cast<std::uint8_t>(relation.kind)) || relation.chunk_size != plan.chunk_size ||
            relation.arithmetic_rule != kSparseGkrArithmeticRule || relation.overflow_rule != kSparseGkrOverflowRule || !relation_domain(relation, &domain))
            return fail("sparse GKR relation shape or arithmetic binding is invalid");
        if (relation_is_edge_domain(relation.kind) && relation.item_count != graph.edges())
            return fail("edge-domain sparse relation does not cover graph edges");
        if (relation_is_node_domain(relation.kind) && relation.item_count != graph.num_nodes)
            return fail("node-domain sparse relation does not cover graph nodes");
        if (relation_is_edge_domain(relation.kind) &&
            (relation.coordinate_rule.find("dst") == std::string::npos ||
             (relation.kind != SparseGkrRelationKind::MaskedSoftmax && relation.coordinate_rule.find("src") == std::string::npos)))
            return fail("edge-domain relation lacks public src/dst schedule");
        if (relation.equations.empty()) return fail("sparse GKR relation has no arithmetic equations");
        if (previous_domain > domain && relation.kind == SparseGkrRelationKind::InputBinding) return fail("sparse GKR relation order is not canonical");
        previous_domain = domain;
    }
    if (plan.claim_links.size() != plan.relations.size() - 1) return fail("sparse GKR terminal chain does not cover every adjacent relation");
    std::set<std::string> claim_link_ids;
    for (std::size_t index = 0; index < plan.claim_links.size(); ++index) {
        const SparseGkrClaimLink& link = plan.claim_links[index];
        if (link.from_relation != plan.relations[index].relation_id || link.to_relation != plan.relations[index + 1].relation_id || link.claim_name.empty() ||
            !claim_link_ids.insert(link.from_relation + "->" + link.to_relation + "/" + link.claim_name).second)
            return fail("sparse GKR terminal chain link is missing or duplicated");
    }
    std::set<std::string> binding_link_ids;
    for (const SparseGkrClaimLink& link : plan.binding_links) {
        if (relation_ids.find(link.from_relation) == relation_ids.end() || relation_ids.find(link.to_relation) == relation_ids.end() ||
            link.claim_name.empty() || !binding_link_ids.insert(link.from_relation + "->" + link.to_relation + "/" + link.claim_name).second)
            return fail("sparse GKR binding link is missing or duplicated");
    }
    return true;
}

bool relation_terminal_and_initial(const SparseGkrRelationSpec& relation, const SparseGkrProverCallbacks& callbacks, F& initial, NativeMaskedTerminalSpec& spec,
                                   std::string* reason) {
    if (!callbacks.initial_claim || !callbacks.terminal_spec || !callbacks.terminal_values || !callbacks.round_polynomial) return false;
    if (!callbacks.initial_claim(relation, initial, reason) || !callbacks.terminal_spec(relation, spec, reason)) return false;
    return true;
}

bool relation_terminal_and_initial(const SparseGkrRelationSpec& relation, const SparseGkrVerifierCallbacks& callbacks, F& initial,
                                   NativeMaskedTerminalSpec& spec, std::string* reason) {
    if (!callbacks.initial_claim || !callbacks.terminal_spec) return false;
    return callbacks.initial_claim(relation, initial, reason) && callbacks.terminal_spec(relation, spec, reason);
}

}

const char* sparse_gkr_relation_kind_name(SparseGkrRelationKind kind) {
    switch (kind) {
    case SparseGkrRelationKind::AttentionScore:
        return "attention_score";
    case SparseGkrRelationKind::MaskedSoftmax:
        return "masked_softmax";
    case SparseGkrRelationKind::Aggregation:
        return "aggregation";
    case SparseGkrRelationKind::SignedRoundDiv:
        return "signed_round_div";
    case SparseGkrRelationKind::Residual:
        return "residual";
    case SparseGkrRelationKind::Fconn:
        return "fconn";
    case SparseGkrRelationKind::InputBinding:
        return "input_binding";
    case SparseGkrRelationKind::OutputBinding:
        return "output_binding";
    }
    return "invalid";
}

bool sparse_gkr_relation_kind_valid(std::uint8_t kind) {
    return kind >= static_cast<std::uint8_t>(SparseGkrRelationKind::AttentionScore) && kind <= static_cast<std::uint8_t>(SparseGkrRelationKind::OutputBinding);
}

bool sparse_gkr_checked_add(std::uint64_t left, std::uint64_t right, std::uint64_t* out) {
    if (out == nullptr || right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    *out = left + right;
    return true;
}

bool sparse_gkr_checked_mul(std::uint64_t left, std::uint64_t right, std::uint64_t* out) {
    if (out == nullptr || (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)) return false;
    *out = left * right;
    return true;
}

bool sparse_gkr_next_power_of_two(std::uint64_t value, std::uint64_t* out) {
    if (out == nullptr || value == 0) return false;
    std::uint64_t power = 1;
    while (power < value) {
        if (power > (std::numeric_limits<std::uint64_t>::max() >> 1)) return false;
        power <<= 1;
    }
    *out = power;
    return true;
}

std::string SparseGkrPlan::encoding() const {
    std::string out;
    append_string(out, schema);
    append_u64(out, version);
    append_u64(out, graph_nodes);
    append_u64(out, graph_edges);
    append_u64(out, graph_count);
    append_u64(out, chunk_size);
    append_string(out, graph_topology_digest);
    append_string(out, static_tables_id);
    append_u64(out, relations.size());
    for (const SparseGkrRelationSpec& relation : relations) append_relation_encoding(out, relation);
    append_u64(out, claim_links.size());
    for (const SparseGkrClaimLink& link : claim_links) {
        append_string(out, link.from_relation);
        append_string(out, link.to_relation);
        append_string(out, link.claim_name);
        append_u64(out, link.from_terminal);
        append_u64(out, link.to_initial);
    }
    append_u64(out, binding_links.size());
    for (const SparseGkrClaimLink& link : binding_links) {
        append_string(out, link.from_relation);
        append_string(out, link.to_relation);
        append_string(out, link.claim_name);
        append_u64(out, link.from_terminal);
        append_u64(out, link.to_initial);
    }
    return out;
}

SparseGkrPlan build_sparse_gkr_plan(const Graph& graph, const std::string& graph_topology_digest, const std::string& static_tables_id,
                                    const std::vector<SparseGkrRelationSpec>& relations) {
    graph.validate();
    SparseGkrPlan plan;
    plan.graph_nodes = graph.num_nodes;
    plan.graph_edges = graph.edges();
    plan.graph_count = graph.graph_count;
    plan.graph_topology_digest = graph_topology_digest;
    plan.static_tables_id = static_tables_id;
    plan.relations = relations;
    for (std::size_t index = 1; index < plan.relations.size(); ++index) {
        SparseGkrClaimLink link;
        link.from_relation = plan.relations[index - 1].relation_id;
        link.to_relation = plan.relations[index].relation_id;
        link.claim_name = "terminal->initial/" + link.from_relation;
        plan.claim_links.push_back(std::move(link));
    }
    std::string reason;
    if (!validate_plan(graph, plan, &reason)) throw std::invalid_argument("invalid sparse GKR plan: " + reason);
    return plan;
}

bool for_each_sparse_gkr_chunk(const Graph& graph, const SparseGkrRelationSpec& relation, const SparseGkrChunkCallback& callback, std::string* reason) {
    if (!callback) {
        set_reason(reason, "sparse GKR chunk callback is empty");
        return false;
    }
    graph.validate();
    std::uint64_t entries = 0;
    if (!relation_domain(relation, &entries) || relation.chunk_size == 0) {
        set_reason(reason, "sparse GKR chunk domain is invalid");
        return false;
    }
    const std::uint64_t width = std::max<std::uint64_t>(1, relation.value_width);
    for (std::uint64_t offset = 0; offset < entries;) {
        const std::uint64_t length = std::min<std::uint64_t>(relation.chunk_size, entries - offset);
        SparseGkrChunk chunk;
        chunk.offset = offset;
        chunk.length = length;
        const std::uint64_t tuple = offset / width;
        if (relation_is_edge_domain(relation.kind)) chunk.first_edge = tuple;
        if (relation_is_node_domain(relation.kind)) chunk.first_node = tuple;
        if (!callback(relation, chunk, reason)) return false;
        std::uint64_t next = 0;
        if (!sparse_gkr_checked_add(offset, length, &next) || next <= offset) {
            set_reason(reason, "sparse GKR chunk offset overflow");
            return false;
        }
        offset = next;
    }
    return true;
}

bool absorb_sparse_gkr_schedule(NativeProofSession& session, const SparseGkrPlan& plan, std::string* reason) {
    session.absorb("sparse/schema", plan.schema + "/" + std::to_string(plan.version));
    session.absorb("sparse/graph", std::to_string(plan.graph_nodes) + ":" + std::to_string(plan.graph_edges) + ":" + std::to_string(plan.graph_count) + ":" +
                                       plan.graph_topology_digest);
    session.absorb("sparse/static_tables", plan.static_tables_id);
    session.absorb("sparse/chunk", std::to_string(plan.chunk_size));
    session.absorb("sparse/arithmetic", kSparseGkrArithmeticRule);
    session.absorb("sparse/overflow", kSparseGkrOverflowRule);
    for (std::size_t index = 0; index < plan.relations.size(); ++index) {
        const SparseGkrRelationSpec& relation = plan.relations[index];
        std::ostringstream label;
        label << "sparse/relation/" << index << "/" << relation.relation_id;
        std::ostringstream payload;
        payload << sparse_gkr_relation_kind_name(relation.kind) << '|' << relation.family_id << '|' << relation.item_count << '|' << relation.value_width << '|'
                << relation.input_offset << '|' << relation.source_offset << '|' << relation.destination_offset << '|' << relation.output_offset << '|'
                << relation.chunk_size << '|' << relation.coordinate_rule << '|' << relation.group_rule << '|' << relation.arithmetic_rule << '|'
                << relation.overflow_rule << '|' << relation.terminal_rule;
        for (const std::string& equation : relation.equations) payload << '|' << equation;
        session.absorb(label.str(), payload.str());
    }
    for (std::size_t index = 0; index < plan.claim_links.size(); ++index) {
        const SparseGkrClaimLink& link = plan.claim_links[index];
        session.absorb("sparse/link/" + std::to_string(index), link.from_relation + "->" + link.to_relation + "|" + link.claim_name + "|" +
                                                                   std::to_string(link.from_terminal) + "|" + std::to_string(link.to_initial));
    }
    for (std::size_t index = 0; index < plan.binding_links.size(); ++index) {
        const SparseGkrClaimLink& link = plan.binding_links[index];
        session.absorb("sparse/binding/" + std::to_string(index), link.from_relation + "->" + link.to_relation + "|" + link.claim_name + "|" +
                                                                      std::to_string(link.from_terminal) + "|" + std::to_string(link.to_initial));
    }
    (void)reason;
    return true;
}

bool prove_sparse_gkr_plan(NativeProofSession& session, const Graph& graph, const SparseGkrPlan& plan, const SparseGkrProverCallbacks& callbacks,
                           std::vector<SparseGkrRelationResult>* results, std::string* reason) {
    if (!validate_plan(graph, plan, reason)) return false;
    if (!absorb_sparse_gkr_schedule(session, plan, reason)) return false;
    if (results != nullptr) results->clear();
    std::vector<F> previous_terminal;
    std::string previous_relation;
    for (const SparseGkrRelationSpec& relation : plan.relations) {
        F initial = F_ZERO;
        NativeMaskedTerminalSpec terminal_spec;
        if (!relation_terminal_and_initial(relation, callbacks, initial, terminal_spec, reason)) {
            if (reason != nullptr && reason->empty()) *reason = "sparse GKR prover callback did not provide relation inputs";
            return false;
        }
        if (!previous_relation.empty()) {
            for (const SparseGkrClaimLink& link : plan.claim_links) {
                if (link.from_relation == previous_relation && link.to_relation == relation.relation_id && !previous_terminal.empty() &&
                    link.to_initial < previous_terminal.size() && initial != previous_terminal[link.to_initial]) {
                    set_reason(reason, "sparse GKR terminal claim link is inconsistent");
                    return false;
                }
            }
        }
        NativeMaskedSumcheckProver prover(session, "sparse/gkr/" + relation.relation_id, 2, initial, terminal_spec);
        std::vector<F> prefix;
        const std::size_t rounds = domain_rounds(relation);
        for (std::size_t round = 0; round < rounds; ++round) {
            std::vector<F> polynomial(3, F_ZERO);
            std::string callback_reason;
            const bool chunks_ok = for_each_sparse_gkr_chunk(
                graph, relation,
                [&](const SparseGkrRelationSpec& chunk_relation, const SparseGkrChunk& chunk, std::string* chunk_reason) {
                    std::vector<F> local;
                    if (!callbacks.round_polynomial(chunk_relation, chunk, round, prefix, local, &callback_reason)) {
                        if (chunk_reason != nullptr) *chunk_reason = callback_reason;
                        return false;
                    }
                    if (local.size() != polynomial.size()) {
                        if (chunk_reason != nullptr) *chunk_reason = "sparse GKR round polynomial degree mismatch";
                        return false;
                    }
                    for (std::size_t index = 0; index < polynomial.size(); ++index) polynomial[index] += local[index];
                    return true;
                },
                reason);
            if (!chunks_ok) return false;
            F challenge = F_ZERO;
            if (!prover.append_round(polynomial, &challenge, reason)) return false;
            prefix.push_back(challenge);
        }
        std::vector<F> terminal_values;
        if (!callbacks.terminal_values(relation, terminal_values, reason)) return false;
        NativeMaskedSumcheckResult result;
        if (!prover.finish(terminal_values, &result, reason)) return false;
        previous_terminal = result.terminal_values;
        previous_relation = relation.relation_id;
        SparseGkrRelationResult finished{relation.relation_id, std::move(result)};
        if (callbacks.relation_finished) callbacks.relation_finished(finished);
        if (results != nullptr) results->push_back(finished);
    }
    return true;
}

bool verify_sparse_gkr_plan(const NativeProofSession& session, const Graph& graph, const SparseGkrPlan& plan, const SparseGkrVerifierCallbacks& callbacks,
                            std::vector<SparseGkrRelationResult>* results, std::string* reason) {
    if (!validate_plan(graph, plan, reason) || !schedule_prefix_matches(session, plan, reason)) return false;
    if (results != nullptr) results->clear();
    std::vector<F> previous_terminal;
    std::string previous_relation;
    for (const SparseGkrRelationSpec& relation : plan.relations) {
        F initial = F_ZERO;
        NativeMaskedTerminalSpec terminal_spec;
        if (!relation_terminal_and_initial(relation, callbacks, initial, terminal_spec, reason)) {
            if (reason != nullptr && reason->empty()) *reason = "sparse GKR verifier callback did not provide public relation inputs";
            return false;
        }
        if (!previous_relation.empty()) {
            for (const SparseGkrClaimLink& link : plan.claim_links) {
                if (link.from_relation == previous_relation && link.to_relation == relation.relation_id && !previous_terminal.empty() &&
                    link.to_initial < previous_terminal.size() && initial != previous_terminal[link.to_initial]) {
                    set_reason(reason, "sparse GKR verifier rejected terminal claim link");
                    return false;
                }
            }
        }
        NativeMaskedSumcheckResult result;
        if (!native_masked_sumcheck_verify(session, "sparse/gkr/" + relation.relation_id, 2, domain_rounds(relation), initial, terminal_spec, &result, reason))
            return false;
        previous_terminal = result.terminal_values;
        previous_relation = relation.relation_id;
        SparseGkrRelationResult finished{relation.relation_id, std::move(result)};
        if (callbacks.relation_finished) callbacks.relation_finished(finished);
        if (results != nullptr) results->push_back(finished);
    }
    return true;
}

}
}
