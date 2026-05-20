#include "commitment.cuh"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool fr_equal_host(const Fr_t& lhs, const Fr_t& rhs)
{
    return std::memcmp(&lhs, &rhs, sizeof(Fr_t)) == 0;
}

Fr_t fr_from_signed_host(std::int64_t value)
{
    Fr_t out = {{0, 0, 0, 0, 0, 0, 0, 0}};
    const std::uint32_t p[8] = {1U, 4294967295U, 4294859774U, 1404937218U, 161601541U, 859428872U, 698187080U, 1944954707U};
    if (value >= 0) {
        std::uint64_t v = static_cast<std::uint64_t>(value);
        out.val[0] = static_cast<std::uint32_t>(v & 0xffffffffULL);
        out.val[1] = static_cast<std::uint32_t>((v >> 32U) & 0xffffffffULL);
        return out;
    }
    std::uint64_t abs_value = static_cast<std::uint64_t>(-value);
    std::uint32_t sub[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    sub[0] = static_cast<std::uint32_t>(abs_value & 0xffffffffULL);
    sub[1] = static_cast<std::uint32_t>((abs_value >> 32U) & 0xffffffffULL);
    std::uint64_t borrow = 0;
    for (int i = 0; i < 8; ++i) {
        const std::uint64_t lhs = p[i];
        const std::uint64_t rhs = static_cast<std::uint64_t>(sub[i]) + borrow;
        if (lhs >= rhs) {
            out.val[i] = static_cast<std::uint32_t>(lhs - rhs);
            borrow = 0;
        } else {
            out.val[i] = static_cast<std::uint32_t>((1ULL << 32U) + lhs - rhs);
            borrow = 1;
        }
    }
    return out;
}

std::uint64_t fnv1a64(const std::string& value)
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::vector<Fr_t> deterministic_raw_fr_vec(std::size_t n, const std::string& label)
{
    std::mt19937_64 rng(fnv1a64(label));
    std::vector<Fr_t> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto value = static_cast<std::int64_t>((rng() % 97ULL) - 48ULL);
        out[i] = fr_from_signed_host(value);
    }
    return out;
}

std::vector<Fr_t> deterministic_mont_fr_vec(std::size_t n, const std::string& label)
{
    auto raw = deterministic_raw_fr_vec(n, label);
    if (raw.empty()) {
        return raw;
    }
    FrTensor tensor(static_cast<uint>(raw.size()), raw.data());
    tensor.mont();
    std::vector<Fr_t> out(raw.size());
    for (uint i = 0; i < tensor.size; ++i) {
        out[i] = tensor(i);
    }
    return out;
}

Commitment deterministic_generators(uint size, const std::string& label)
{
    auto scalars = deterministic_raw_fr_vec(size, "formal-opening-selftest:gens:" + label);
    FrTensor scalar_tensor(size, scalars.data());
    Commitment out(size, G1Jacobian_generator);
    out *= scalar_tensor;
    return out;
}

struct OpeningCase {
    std::string label;
    uint rows;
    uint inner_size;
};

struct PartialCase {
    std::string label;
    uint size;
    uint rounds;
    uint window_size;
};

bool run_opening_case(const OpeningCase& test_case, std::ostream& log)
{
    const uint outer_log = ceilLog2(test_case.rows);
    const uint inner_log = ceilLog2(test_case.inner_size);
    const auto total_size = static_cast<std::size_t>(test_case.rows) * test_case.inner_size;

    auto raw_scalars = deterministic_raw_fr_vec(total_size, test_case.label + ":scalars");
    FrTensor tensor(static_cast<uint>(raw_scalars.size()), raw_scalars.data());
    tensor.mont();
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        log << "OPENING label=" << test_case.label
            << " rows=" << test_case.rows
            << " inner_size=" << test_case.inner_size
            << " outer_log=" << outer_log
            << " inner_log=" << inner_log
            << " total_size=" << total_size
            << " status=FAIL failure=\"tensor.mont failed: " << cudaGetErrorString(status) << "\"\n";
        return false;
    }
    auto generators = deterministic_generators(test_case.inner_size, test_case.label);
    auto commitment = generators.commit(tensor);
    auto point = deterministic_mont_fr_vec(outer_log + inner_log, test_case.label + ":point");

    CommitmentOpeningProof proof;
    bool opened = false;
    bool verified = false;
    bool value_match = false;
    std::string failure;
    try {
        generators.open_with_proof(tensor, commitment, point, proof);
        opened = true;
        value_match = fr_equal_host(proof.value, tensor(point));
        verified = Commitment::verify_opening(generators, commitment, proof);
        if (!verified) {
            failure = "verify_opening returned false";
        }
        if (!value_match) {
            if (!failure.empty()) {
                failure += "; ";
            }
            failure += "opened value mismatched direct MLE";
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }

    log << "OPENING label=" << test_case.label
        << " rows=" << test_case.rows
        << " inner_size=" << test_case.inner_size
        << " tensor_size=" << total_size
        << " padded_size=" << ((std::size_t{1} << outer_log) * (std::size_t{1} << inner_log))
        << " com.size=" << commitment.size
        << " outer_log=" << outer_log
        << " inner_log=" << inner_log
        << " window_size=" << test_case.inner_size
        << " round=all"
        << " backend=hybrid_cpu_field_gpu_g1"
        << " status=" << (opened && verified && value_match ? "PASS" : "FAIL");
    if (!failure.empty()) {
        log << " failure=\"" << failure << "\"";
    }
    log << '\n';
    return opened && verified && value_match;
}

bool run_partial_case(const PartialCase& test_case, std::ostream& log)
{
    auto raw_scalars = deterministic_raw_fr_vec(test_case.size, test_case.label + ":partial-scalars");
    FrTensor tensor(test_case.size, raw_scalars.data());
    tensor.mont();
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        log << "PARTIAL label=" << test_case.label
            << " size=" << test_case.size
            << " rounds=" << test_case.rounds
            << " window_size=" << test_case.window_size
            << " status=FAIL failure=\"tensor.mont failed: " << cudaGetErrorString(status) << "\"\n";
        return false;
    }
    auto point = deterministic_mont_fr_vec(test_case.rounds, test_case.label + ":partial-point");

    bool passed = true;
    std::string failure;
    try {
        const auto checked = Commitment::partial_me_checked(tensor, point.begin(), point.end(), test_case.window_size, 0);
        const auto direct = Fr_partial_me(tensor, point.begin(), point.end(), test_case.window_size);
        if (checked.size != direct.size) {
            passed = false;
            failure = "size mismatch";
        } else {
            for (uint i = 0; i < checked.size; ++i) {
                if (!fr_equal_host(checked(i), direct(i))) {
                    passed = false;
                    std::ostringstream out;
                    out << "value mismatch at index=" << i;
                    failure = out.str();
                    break;
                }
            }
        }
    } catch (const std::exception& error) {
        passed = false;
        failure = error.what();
    }

    log << "PARTIAL label=" << test_case.label
        << " size=" << test_case.size
        << " padded_size=" << (static_cast<std::size_t>(test_case.window_size) << test_case.rounds)
        << " com.size=NA"
        << " inner_log=NA"
        << " outer_log=" << test_case.rounds
        << " rounds=" << test_case.rounds
        << " window_size=" << test_case.window_size
        << " round=all"
        << " backend=cpu_field"
        << " status=" << (passed ? "PASS" : "FAIL");
    if (!failure.empty()) {
        log << " failure=\"" << failure << "\"";
    }
    log << '\n';
    return passed;
}

}  // namespace

int main(int argc, char** argv)
{
    std::vector<OpeningCase> opening_cases = {
        {"size_2_row_vector", 2, 1},
        {"size_3_row_vector", 3, 1},
        {"size_4_row_vector", 4, 1},
        {"size_7_row_vector", 7, 1},
        {"size_8_row_vector", 8, 1},
        {"size_15_row_vector", 15, 1},
        {"size_16_row_vector", 16, 1},
        {"size_31_row_vector", 31, 1},
        {"size_32_row_vector", 32, 1},
        {"com1_inner2", 1, 2},
        {"com1_inner4", 1, 4},
        {"com1_inner8", 1, 8},
        {"com1_inner16", 1, 16},
        {"rows3_inner2", 3, 2},
        {"rows3_inner4", 3, 4},
        {"rows3_inner8", 3, 8},
        {"rows3_inner16", 3, 16},
        {"rows5_inner4", 5, 4},
        {"dataset_cora_logits_size", 18956, 1},
        {"dataset_citeseer_logits_size", 19962, 1},
        {"dataset_pubmed_logits_size", 59151, 1},
    };

    std::vector<PartialCase> partial_cases = {
        {"window1_size3", 3, 2, 1},
        {"window1_size7", 7, 3, 1},
        {"window2_size6", 6, 2, 2},
        {"window2_size10", 10, 3, 2},
        {"window4_size12", 12, 2, 4},
        {"window4_size20", 20, 3, 4},
        {"window8_size24", 24, 2, 8},
        {"window16_size48", 48, 2, 16},
        {"dataset_ppi_logits_size_partial", 6890224, 23, 1},
        {"dataset_ogbn_arxiv_logits_size_partial", 6773720, 23, 1},
    };

    bool ok = true;
    for (const auto& test_case : opening_cases) {
        if (argc > 1 && test_case.label != argv[1]) {
            continue;
        }
        ok = run_opening_case(test_case, std::cout) && ok;
    }
    for (const auto& test_case : partial_cases) {
        if (argc > 1 && test_case.label != argv[1]) {
            continue;
        }
        ok = run_partial_case(test_case, std::cout) && ok;
    }

    std::cout << "SUMMARY status=" << (ok ? "PASS" : "FAIL")
        << " opening_cases=" << opening_cases.size()
        << " partial_cases=" << partial_cases.size() << '\n';
    return ok ? 0 : 1;
}
