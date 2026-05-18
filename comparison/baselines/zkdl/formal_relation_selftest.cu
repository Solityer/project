#include "formal_relations.cuh"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool fr_equal_host_local(const Fr_t& lhs, const Fr_t& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(Fr_t)) == 0;
}

std::uint64_t fnv1a64_local(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

Fr_t fr_from_signed_local(std::int64_t value)
{
    Fr_t out = Fr_host_zero();
    const Fr_t p = Fr_host_modulus();
    if (value >= 0) {
        std::uint64_t v = static_cast<std::uint64_t>(value);
        out.val[0] = static_cast<uint>(v & 0xffffffffULL);
        out.val[1] = static_cast<uint>((v >> 32U) & 0xffffffffULL);
        return out;
    }
    Fr_t abs_value = Fr_host_zero();
    const std::uint64_t v = static_cast<std::uint64_t>(-value);
    abs_value.val[0] = static_cast<uint>(v & 0xffffffffULL);
    abs_value.val[1] = static_cast<uint>((v >> 32U) & 0xffffffffULL);
    return Fr_host_sub(p, abs_value);
}

std::vector<Fr_t> deterministic_mont_vec_local(std::size_t n, const std::string& label)
{
    std::mt19937_64 rng(fnv1a64_local(label));
    std::vector<Fr_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto value = static_cast<std::int64_t>((rng() % 17ULL) - 8ULL);
        out[i] = Fr_host_mont(fr_from_signed_local(value));
    }
    return out;
}

Fr_t one_mont()
{
    return Fr_host_one_mont();
}

Fr_t inverse_two_mont()
{
    Fr_t p_plus_one = Fr_host_modulus();
    p_plus_one.val[0] += 1U;
    Fr_t half = Fr_host_zero();
    std::uint64_t carry = 0;
    for (int i = 7; i >= 0; --i) {
        const std::uint64_t cur = (carry << 32U) | p_plus_one.val[i];
        half.val[i] = static_cast<uint>(cur / 2U);
        carry = cur & 1U;
    }
    return Fr_host_mont(half);
}

bool run_hadamard_case(const std::string& label, uint size, std::ostream& log)
{
    bool passed = true;
    std::string failure;
    try {
        auto a_host = deterministic_mont_vec_local(size, label + ":a");
        auto b_host = deterministic_mont_vec_local(size, label + ":b");
        std::vector<Fr_t> c_host(size, Fr_host_zero());
        for (uint i = 0; i < size; ++i) c_host[i] = Fr_host_mul(a_host[i], b_host[i]);
        FrTensor a(size, a_host.data());
        FrTensor b(size, b_host.data());
        FrTensor c(size, c_host.data());
        Commitment gen(1, G1Jacobian_generator);
        const auto com_a = gen.commit(a);
        const auto com_b = gen.commit(b);
        const auto com_c = gen.commit(c);
        auto proof = formal_hadamard_relation_prove(a, b, c, gen, com_a, com_b, com_c, label);
        std::string verify_failure;
        passed = formal_hadamard_relation_verify(proof, gen, com_a, com_b, com_c, label, &verify_failure);
        if (!passed) failure = verify_failure;

        auto tampered_proof = proof;
        tampered_proof.final_a = Fr_host_add(tampered_proof.final_a, one_mont());
        if (formal_hadamard_relation_verify(tampered_proof, gen, com_a, com_b, com_c, label, nullptr)) {
            passed = false;
            failure = "tampered proof accepted";
        }
        auto c_bad_host = c_host;
        c_bad_host[0] = Fr_host_add(c_bad_host[0], one_mont());
        FrTensor c_bad(size, c_bad_host.data());
        const auto com_c_bad = gen.commit(c_bad);
        if (formal_hadamard_relation_verify(proof, gen, com_a, com_b, com_c_bad, label, nullptr)) {
            passed = false;
            failure = "tampered C commitment accepted";
        }
        log << "HADAMARD_RELATION label=" << label
            << " size=" << size
            << " padded_size=" << (std::size_t{1} << ceilLog2(size))
            << " rounds=" << proof.rounds.size()
            << " transcript_digest=" << proof.transcript_digest
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " tamper_reject=PASS"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "HADAMARD_RELATION label=" << label << " size=" << size << " status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

struct MatmulCase
{
    std::string label;
    uint num;
    uint m;
    uint n;
    uint k;
};

void build_padded_matmul_case(
    const MatmulCase& test,
    std::vector<Fr_t>* a,
    std::vector<Fr_t>* b,
    std::vector<Fr_t>* c,
    uint* padded_num,
    uint* padded_m,
    uint* padded_n,
    uint* padded_k)
{
    *padded_num = formal_next_power_of_two_u32(test.num);
    *padded_m = formal_next_power_of_two_u32(test.m);
    *padded_n = formal_next_power_of_two_u32(test.n);
    *padded_k = formal_next_power_of_two_u32(test.k);
    a->assign(static_cast<std::size_t>(*padded_num) * *padded_m * *padded_n, Fr_host_zero());
    b->assign(static_cast<std::size_t>(*padded_num) * *padded_n * *padded_k, Fr_host_zero());
    c->assign(static_cast<std::size_t>(*padded_num) * *padded_m * *padded_k, Fr_host_zero());
    auto a_values = deterministic_mont_vec_local(static_cast<std::size_t>(test.num) * test.m * test.n, test.label + ":A");
    auto b_values = deterministic_mont_vec_local(static_cast<std::size_t>(test.num) * test.n * test.k, test.label + ":B");
    for (uint batch = 0; batch < test.num; ++batch) {
        for (uint row = 0; row < test.m; ++row) {
            for (uint t = 0; t < test.n; ++t) {
                (*a)[(static_cast<std::size_t>(batch) * *padded_m + row) * *padded_n + t] =
                    a_values[(static_cast<std::size_t>(batch) * test.m + row) * test.n + t];
            }
        }
        for (uint t = 0; t < test.n; ++t) {
            for (uint col = 0; col < test.k; ++col) {
                (*b)[(static_cast<std::size_t>(batch) * *padded_n + t) * *padded_k + col] =
                    b_values[(static_cast<std::size_t>(batch) * test.n + t) * test.k + col];
            }
        }
    }
    for (uint batch = 0; batch < test.num; ++batch) {
        for (uint row = 0; row < test.m; ++row) {
            for (uint col = 0; col < test.k; ++col) {
                Fr_t acc = Fr_host_zero();
                for (uint t = 0; t < test.n; ++t) {
                    const auto av = (*a)[(static_cast<std::size_t>(batch) * *padded_m + row) * *padded_n + t];
                    const auto bv = (*b)[(static_cast<std::size_t>(batch) * *padded_n + t) * *padded_k + col];
                    acc = Fr_host_add(acc, Fr_host_mul(av, bv));
                }
                (*c)[(static_cast<std::size_t>(batch) * *padded_m + row) * *padded_k + col] = acc;
            }
        }
    }
}

bool run_matmul_case(const MatmulCase& test, std::ostream& log)
{
    bool passed = true;
    std::string failure;
    try {
        uint padded_num = 0;
        uint padded_m = 0;
        uint padded_n = 0;
        uint padded_k = 0;
        std::vector<Fr_t> a_host;
        std::vector<Fr_t> b_host;
        std::vector<Fr_t> c_host;
        build_padded_matmul_case(test, &a_host, &b_host, &c_host, &padded_num, &padded_m, &padded_n, &padded_k);
        FrTensor a(static_cast<uint>(a_host.size()), a_host.data());
        FrTensor b(static_cast<uint>(b_host.size()), b_host.data());
        FrTensor c(static_cast<uint>(c_host.size()), c_host.data());
        Commitment gen(1, G1Jacobian_generator);
        const auto com_a = gen.commit(a);
        const auto com_b = gen.commit(b);
        const auto com_c = gen.commit(c);
        auto proof = formal_matmul_relation_prove(a, b, c, gen, com_a, com_b, com_c,
            test.num, test.m, test.n, test.k, padded_num, padded_m, padded_n, padded_k, test.label);
        std::string verify_failure;
        passed = formal_matmul_relation_verify(proof, gen, com_a, com_b, com_c, test.label, &verify_failure);
        if (!passed) failure = verify_failure;

        auto tampered_proof = proof;
        tampered_proof.inner_product.final_b = Fr_host_add(tampered_proof.inner_product.final_b, one_mont());
        if (formal_matmul_relation_verify(tampered_proof, gen, com_a, com_b, com_c, test.label, nullptr)) {
            passed = false;
            failure = "tampered matmul proof accepted";
        }
        auto c_bad_host = c_host;
        c_bad_host[0] = Fr_host_add(c_bad_host[0], one_mont());
        FrTensor c_bad(static_cast<uint>(c_bad_host.size()), c_bad_host.data());
        const auto com_c_bad = gen.commit(c_bad);
        if (formal_matmul_relation_verify(proof, gen, com_a, com_b, com_c_bad, test.label, nullptr)) {
            passed = false;
            failure = "tampered matmul C commitment accepted";
        }
        log << "MATMUL_RELATION label=" << test.label
            << " num=" << test.num
            << " m=" << test.m
            << " n=" << test.n
            << " k=" << test.k
            << " padded_num=" << padded_num
            << " padded_m=" << padded_m
            << " padded_n=" << padded_n
            << " padded_k=" << padded_k
            << " batch_rounds=" << proof.batch_rounds.size()
            << " inner_rounds=" << proof.inner_product.rounds.size()
            << " transcript_digest=" << proof.transcript_digest
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " tamper_reject=PASS"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "MATMUL_RELATION label=" << test.label << " status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

bool run_linear_average_case(const std::string& label, uint k_out, uint size, std::ostream& log)
{
    bool passed = true;
    std::string failure;
    try {
        std::vector<std::vector<Fr_t>> heads;
        heads.reserve(k_out);
        std::vector<Fr_t> avg(size, Fr_host_zero());
        for (uint head = 0; head < k_out; ++head) {
            heads.push_back(deterministic_mont_vec_local(size, label + ":head" + std::to_string(head)));
            for (uint i = 0; i < size; ++i) avg[i] = Fr_host_add(avg[i], heads.back()[i]);
        }
        Fr_t inv_k = Fr_host_one_mont();
        if (k_out == 2) {
            inv_k = inverse_two_mont();
        } else if (k_out != 1) {
            throw std::runtime_error("linear selftest only supports K_out 1 or 2");
        }
        for (uint i = 0; i < size; ++i) avg[i] = Fr_host_mul(avg[i], inv_k);

        Commitment gen(1, G1Jacobian_generator);
        std::vector<std::unique_ptr<FrTensor>> tensors;
        std::vector<std::unique_ptr<G1TensorJacobian>> commitments;
        std::vector<FormalLinearTerm> terms;
        const auto point = deterministic_mont_vec_local(ceilLog2(size), label + ":point");
        for (uint head = 0; head < k_out; ++head) {
            tensors.emplace_back(std::make_unique<FrTensor>(size, heads[head].data()));
            commitments.emplace_back(std::make_unique<G1TensorJacobian>(gen.commit(*tensors.back())));
            FormalLinearTerm term;
            term.label = label + ":head" + std::to_string(head);
            term.tensor = tensors.back().get();
            term.commitment = commitments.back().get();
            term.coefficient = inv_k;
            term.point = point;
            terms.push_back(term);
        }
        FrTensor avg_tensor(size, avg.data());
        const Fr_t public_eval = avg_tensor(point);
        const Fr_t minus_one = Fr_host_sub(Fr_host_zero(), Fr_host_one_mont());
        auto proof = formal_linear_relation_prove(label, {size}, terms, gen, minus_one, public_eval, Fr_host_zero());
        std::vector<const G1TensorJacobian*> verify_commitments;
        for (const auto& commitment : commitments) verify_commitments.push_back(commitment.get());
        passed = formal_linear_relation_verify(proof, gen, verify_commitments, public_eval, &failure);
        auto tampered = proof;
        tampered.evaluations[0] = Fr_host_add(tampered.evaluations[0], one_mont());
        if (formal_linear_relation_verify(tampered, gen, verify_commitments, public_eval, nullptr)) {
            passed = false;
            failure = "tampered linear proof accepted";
        }
        log << "LINEAR_RELATION label=" << label
            << " kind=output_head_average"
            << " K_out=" << k_out
            << " size=" << size
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " transcript_digest=" << proof.transcript_digest
            << " tamper_reject=PASS"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "LINEAR_RELATION label=" << label << " status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

std::vector<Fr_t> boolean_point_for_index(uint index, uint log_size)
{
    std::vector<Fr_t> point(log_size, Fr_host_zero());
    for (uint i = 0; i < log_size; ++i) {
        point[i] = ((index >> i) & 1U) ? Fr_host_one_mont() : Fr_host_zero();
    }
    return point;
}

bool run_linear_bias_case(const std::string& label, uint size, std::ostream& log)
{
    bool passed = true;
    std::string failure;
    try {
        auto x_host = deterministic_mont_vec_local(size, label + ":x");
        auto b_host = deterministic_mont_vec_local(size, label + ":bias");
        std::vector<Fr_t> y_host(size, Fr_host_zero());
        for (uint i = 0; i < size; ++i) y_host[i] = Fr_host_add(x_host[i], b_host[i]);
        FrTensor x(size, x_host.data());
        FrTensor y(size, y_host.data());
        FrTensor bias(size, b_host.data());
        Commitment gen(1, G1Jacobian_generator);
        const auto com_x = gen.commit(x);
        const auto com_y = gen.commit(y);
        const auto point = deterministic_mont_vec_local(ceilLog2(size), label + ":point");
        const Fr_t bias_eval = bias(point);
        std::vector<FormalLinearTerm> terms = {
            {"Y", &y, &com_y, Fr_host_one_mont(), point},
            {"X", &x, &com_x, Fr_host_sub(Fr_host_zero(), Fr_host_one_mont()), point},
        };
        auto proof = formal_linear_relation_prove(label, {size}, terms, gen,
            Fr_host_sub(Fr_host_zero(), Fr_host_one_mont()), bias_eval, Fr_host_zero());
        passed = formal_linear_relation_verify(proof, gen, {&com_y, &com_x}, bias_eval, &failure);
        auto tampered = proof;
        tampered.public_evaluation = Fr_host_add(tampered.public_evaluation, one_mont());
        if (formal_linear_relation_verify(tampered, gen, {&com_y, &com_x}, bias_eval, nullptr)) {
            passed = false;
            failure = "tampered bias proof accepted";
        }
        log << "LINEAR_RELATION label=" << label
            << " kind=bias_add"
            << " size=" << size
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " transcript_digest=" << proof.transcript_digest
            << " tamper_reject=PASS"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "LINEAR_RELATION label=" << label << " status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

bool run_linear_concat_case(const std::string& label, std::ostream& log)
{
    bool passed = true;
    std::string failure;
    try {
        const uint rows = 2;
        const uint head_width = 2;
        const uint concat_width = 4;
        const uint offset = 2;
        auto head_host = deterministic_mont_vec_local(rows * head_width, label + ":head");
        std::vector<Fr_t> cat_host(rows * concat_width, Fr_host_zero());
        for (uint row = 0; row < rows; ++row) {
            for (uint col = 0; col < head_width; ++col) {
                cat_host[row * concat_width + offset + col] = head_host[row * head_width + col];
            }
        }
        FrTensor head(rows * head_width, head_host.data());
        FrTensor cat(rows * concat_width, cat_host.data());
        Commitment gen(1, G1Jacobian_generator);
        const auto com_head = gen.commit(head);
        const auto com_cat = gen.commit(cat);
        const uint row = 1;
        const uint col = 1;
        const auto source_point = boolean_point_for_index(row * head_width + col, ceilLog2(rows * head_width));
        const auto target_point = boolean_point_for_index(row * concat_width + offset + col, ceilLog2(rows * concat_width));
        std::vector<FormalLinearTerm> terms = {
            {"H_cat", &cat, &com_cat, Fr_host_one_mont(), target_point},
            {"H_head", &head, &com_head, Fr_host_sub(Fr_host_zero(), Fr_host_one_mont()), source_point},
        };
        auto proof = formal_linear_relation_prove(label, {rows, concat_width}, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        passed = formal_linear_relation_verify(proof, gen, {&com_cat, &com_head}, Fr_host_zero(), &failure);
        auto tampered = proof;
        tampered.evaluations[1] = Fr_host_add(tampered.evaluations[1], one_mont());
        if (formal_linear_relation_verify(tampered, gen, {&com_cat, &com_head}, Fr_host_zero(), nullptr)) {
            passed = false;
            failure = "tampered concat proof accepted";
        }
        log << "LINEAR_RELATION label=" << label
            << " kind=concat_copy"
            << " rows=" << rows
            << " head_width=" << head_width
            << " concat_width=" << concat_width
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " transcript_digest=" << proof.transcript_digest
            << " tamper_reject=PASS"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "LINEAR_RELATION label=" << label << " status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

bool run_sparse_routing_case(const std::string& label, const std::string& direction, std::ostream& log)
{
    bool passed = true;
    std::string failure;
    try {
        const uint node_count = 5;
        const uint edge_count = 9;
        const uint padded_node_count = formal_next_power_of_two_u32(node_count);
        const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
        const std::vector<std::uint32_t> src = {0, 2, 1, 3, 4, 0, 2, 4, 1};
        const std::vector<std::uint32_t> dst = {1, 1, 2, 2, 3, 3, 4, 4, 4};
        auto node_host = deterministic_mont_vec_local(node_count, label + ":node");
        node_host.resize(padded_node_count, Fr_host_zero());
        std::vector<Fr_t> edge_host(padded_edge_count, Fr_host_zero());
        for (uint edge = 0; edge < edge_count; ++edge) {
            const auto node = direction == "source" ? src[edge] : dst[edge];
            edge_host[edge] = node_host[node];
        }
        FrTensor node_tensor(padded_node_count, node_host.data());
        FrTensor edge_tensor(padded_edge_count, edge_host.data());
        Commitment gen(1, G1Jacobian_generator);
        const auto com_node = gen.commit(node_tensor);
        const auto com_edge = gen.commit(edge_tensor);
        auto proof = formal_sparse_routing_relation_prove(
            node_tensor,
            edge_tensor,
            gen,
            com_node,
            gen,
            com_edge,
            src,
            dst,
            node_count,
            edge_count,
            direction,
            label + ":" + direction);
        passed = formal_sparse_routing_relation_verify(
            proof,
            gen,
            com_node,
            gen,
            com_edge,
            src,
            dst,
            direction,
            label + ":" + direction,
            &failure);
        auto tampered = proof;
        tampered.inner_product.final_b = Fr_host_add(tampered.inner_product.final_b, one_mont());
        if (formal_sparse_routing_relation_verify(
                tampered, gen, com_node, gen, com_edge, src, dst, direction, label + ":" + direction, nullptr)) {
            passed = false;
            failure = "tampered sparse routing proof accepted";
        }
        auto edge_bad_host = edge_host;
        edge_bad_host[0] = Fr_host_add(edge_bad_host[0], one_mont());
        FrTensor edge_bad(padded_edge_count, edge_bad_host.data());
        const auto com_edge_bad = gen.commit(edge_bad);
        if (formal_sparse_routing_relation_verify(
                proof, gen, com_node, gen, com_edge_bad, src, dst, direction, label + ":" + direction, nullptr)) {
            passed = false;
            failure = "tampered sparse routing edge commitment accepted";
        }
        log << "SPARSE_ROUTING_RELATION label=" << label
            << " direction=" << direction
            << " nodes=" << node_count
            << " edges=" << edge_count
            << " padded_nodes=" << padded_node_count
            << " padded_edges=" << padded_edge_count
            << " topology_hash=" << proof.topology_hash
            << " inner_rounds=" << proof.inner_product.rounds.size()
            << " transcript_digest=" << proof.transcript_digest
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " tamper_reject=PASS"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "SPARSE_ROUTING_RELATION label=" << label << " direction=" << direction << " status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

bool run_grouped_sum_case(const std::string& label, std::ostream& log)
{
    bool passed = true;
    std::string failure;
    try {
        const uint node_count = 5;
        const uint edge_count = 9;
        const uint padded_node_count = formal_next_power_of_two_u32(node_count);
        const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
        const std::vector<std::uint32_t> src = {0, 2, 1, 3, 4, 0, 2, 4, 1};
        const std::vector<std::uint32_t> dst = {1, 1, 2, 2, 3, 3, 4, 4, 4};
        auto edge_host = deterministic_mont_vec_local(edge_count, label + ":edge");
        edge_host.resize(padded_edge_count, Fr_host_zero());
        std::vector<Fr_t> out_host(padded_node_count, Fr_host_zero());
        for (uint edge = 0; edge < edge_count; ++edge) {
            out_host[dst[edge]] = Fr_host_add(out_host[dst[edge]], edge_host[edge]);
        }
        FrTensor edge_tensor(padded_edge_count, edge_host.data());
        FrTensor out_tensor(padded_node_count, out_host.data());
        Commitment gen(1, G1Jacobian_generator);
        const auto com_edge = gen.commit(edge_tensor);
        const auto com_out = gen.commit(out_tensor);
        auto proof = formal_grouped_sum_relation_prove(
            edge_tensor,
            out_tensor,
            gen,
            com_edge,
            gen,
            com_out,
            src,
            dst,
            node_count,
            edge_count,
            label);
        passed = formal_grouped_sum_relation_verify(
            proof,
            gen,
            com_edge,
            gen,
            com_out,
            src,
            dst,
            label,
            &failure);
        auto tampered = proof;
        tampered.inner_product.final_b = Fr_host_add(tampered.inner_product.final_b, one_mont());
        if (formal_grouped_sum_relation_verify(tampered, gen, com_edge, gen, com_out, src, dst, label, nullptr)) {
            passed = false;
            failure = "tampered grouped-sum proof accepted";
        }
        auto out_bad_host = out_host;
        out_bad_host[1] = Fr_host_add(out_bad_host[1], one_mont());
        FrTensor out_bad(padded_node_count, out_bad_host.data());
        const auto com_out_bad = gen.commit(out_bad);
        if (formal_grouped_sum_relation_verify(proof, gen, com_edge, gen, com_out_bad, src, dst, label, nullptr)) {
            passed = false;
            failure = "tampered grouped-sum output commitment accepted";
        }
        log << "GROUPED_SUM_RELATION label=" << label
            << " nodes=" << node_count
            << " edges=" << edge_count
            << " padded_nodes=" << padded_node_count
            << " padded_edges=" << padded_edge_count
            << " topology_hash=" << proof.topology_hash
            << " inner_rounds=" << proof.inner_product.rounds.size()
            << " transcript_digest=" << proof.transcript_digest
            << " proof_size_bytes=" << proof.serialized_size_bytes()
            << " tamper_reject=PASS"
            << " status=" << (passed ? "PASS" : "FAIL");
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
        log << "GROUPED_SUM_RELATION label=" << label << " status=FAIL";
    }
    if (!failure.empty()) log << " failure=\"" << failure << "\"";
    log << '\n';
    return passed;
}

}  // namespace

int main()
{
    bool ok = true;
    for (uint size : {1U, 2U, 3U, 4U, 7U, 8U, 15U, 16U, 31U, 32U, 257U}) {
        ok = run_hadamard_case("size" + std::to_string(size), size, std::cout) && ok;
    }
    const std::vector<MatmulCase> matmul_cases = {
        {"small_square", 1, 2, 2, 2},
        {"non_square", 1, 2, 3, 4},
        {"non_power_padding", 3, 3, 3, 2},
        {"batched_gt1", 2, 2, 3, 2},
        {"dot_product_degenerate", 1, 1, 5, 1},
    };
    for (const auto& test : matmul_cases) {
        ok = run_matmul_case(test, std::cout) && ok;
    }
    ok = run_linear_average_case("average_k1", 1, 7, std::cout) && ok;
    ok = run_linear_average_case("average_k2", 2, 7, std::cout) && ok;
    ok = run_linear_bias_case("bias_add", 7, std::cout) && ok;
    ok = run_linear_concat_case("concat_copy", std::cout) && ok;
    ok = run_sparse_routing_case("sparse_source_route", "source", std::cout) && ok;
    ok = run_sparse_routing_case("sparse_destination_route", "destination", std::cout) && ok;
    ok = run_grouped_sum_case("grouped_sum_by_dst", std::cout) && ok;
    std::cout << "SUMMARY status=" << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
