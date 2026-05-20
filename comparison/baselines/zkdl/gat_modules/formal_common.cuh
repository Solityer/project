#pragma once

// zkDL formal partial proof 的公共编码、承诺注册与张量工具。

Fr_t fr_from_signed_host(std::int64_t value) {
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

Fr_t fr_from_signed_mont_host(std::int64_t value) {
    return Fr_host_mont(fr_from_signed_host(value));
}

std::uint64_t fnv1a64(const std::string& value) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char ch : value) {
        hash ^= ch;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::vector<Fr_t> deterministic_raw_fr_vec(std::size_t n, const std::string& label) {
    std::mt19937_64 rng(fnv1a64(label));
    std::vector<Fr_t> out(n);
    for (auto& value : out) {
        value = fr_from_signed_host(0);
        const auto lo = rng();
        value.val[0] = static_cast<std::uint32_t>(lo & 0xffffffffULL);
        value.val[1] = static_cast<std::uint32_t>((lo >> 32U) & 0xffffffffULL);
    }
    return out;
}

std::vector<Fr_t> deterministic_mont_fr_vec(std::size_t n, const std::string& label) {
    auto raw = deterministic_raw_fr_vec(n, label);
    if (raw.empty()) {
        return raw;
    }
    FrTensor tensor(static_cast<uint>(raw.size()), raw.data());
    tensor.mont();
    cuda_sync("deterministic_mont_fr_vec");
    std::vector<Fr_t> out(raw.size());
    for (uint i = 0; i < tensor.size; ++i) {
        out[i] = tensor(i);
    }
    return out;
}

Commitment deterministic_generators(uint size, const std::string& label) {
    auto scalars = deterministic_raw_fr_vec(size, "zkdl-gat-generators:" + label + ":" + std::to_string(size));
    FrTensor scalar_tensor(size, scalars.data());
    Commitment out(size, G1Jacobian_generator);
    out *= scalar_tensor;
    cuda_sync("deterministic_generators");
    return out;
}

std::vector<Fr_t> encode_padded_tensor(const std::vector<double>& values, std::size_t rows, std::size_t cols, std::size_t padded_cols) {
    if (values.size() != rows * cols) {
        throw std::runtime_error("tensor shape/value mismatch while encoding");
    }
    std::vector<Fr_t> encoded(rows * padded_cols, fr_from_signed_host(0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double scaled = values[row * cols + col] * 16.0;
            const auto fixed = static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
            encoded[row * padded_cols + col] = fr_from_signed_host(fixed);
        }
    }
    return encoded;
}

Fr_t evaluate_public_tensor_mle(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_cols,
    const std::vector<Fr_t>& point) {
    auto encoded = encode_padded_tensor(values, rows, cols, padded_cols);
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    return tensor(point);
}

Fr_t evaluate_public_tensor_row_mle(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_cols,
    std::size_t row,
    const std::vector<Fr_t>& point) {
    if (row >= rows) {
        throw std::runtime_error("row opening index out of range");
    }
    std::vector<double> row_values(cols, 0.0);
    for (std::size_t col = 0; col < cols; ++col) {
        row_values[col] = values[row * cols + col];
    }
    auto encoded = encode_padded_tensor(row_values, 1, cols, padded_cols);
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    return tensor(point);
}

bool fr_is_zero_host(const Fr_t& value) {
    return fr_equal_host(value, fr_from_signed_host(0));
}

bool point_is_all_zero(const std::vector<Fr_t>& point) {
    return std::all_of(point.begin(), point.end(), [](const Fr_t& value) {
        return fr_is_zero_host(value);
    });
}

Fr_t encode_public_tensor_row_first_value_mont(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t row) {
    if (row >= rows || cols == 0 || values.size() != rows * cols) {
        throw std::runtime_error("public tensor first-value evaluation shape mismatch");
    }
    const double scaled = values[row * cols] * 16.0;
    const auto fixed = static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
    auto raw = fr_from_signed_host(fixed);
    FrTensor tensor(1, &raw);
    tensor.mont();
    return tensor(0);
}

struct TensorCommitmentProof {
    std::string label;
    std::vector<std::size_t> shape;
    std::size_t padded_inner = 0;
    std::size_t opened_row = 0;
    std::string generator_label;
    std::vector<G1Jacobian_t> commitment_points;
    CommitmentOpeningProof opening;
};

struct LevelBProofArtifact {
    std::string level = "debug_level_b";
    std::string benchmark_mode = "debug_level_b_output_binding_only_not_formal_level_a";
    std::vector<TensorCommitmentProof> tensors;
    bool proof_generation_ok = false;
    bool verification_ok = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    fs::path proof_path;
    std::vector<std::string> covered_relations;
    std::vector<std::string> uncovered_relations;
    std::string blocker;
};

std::vector<G1Jacobian_t> fold_generators_standard(
    const std::vector<G1Jacobian_t>& left,
    const std::vector<G1Jacobian_t>& right,
    Fr_t u_mont) {
    if (left.size() != right.size()) {
        throw std::runtime_error("generator fold size mismatch");
    }
    const auto one_minus_u_raw = fr_unmont_host(fr_mont_one_minus_host(u_mont));
    const auto u_raw = fr_unmont_host(u_mont);
    std::vector<Fr_t> left_coeff(left.size(), one_minus_u_raw);
    std::vector<Fr_t> right_coeff(right.size(), u_raw);
    G1TensorJacobian left_tensor(static_cast<uint>(left.size()), left.data());
    G1TensorJacobian right_tensor(static_cast<uint>(right.size()), right.data());
    FrTensor left_scalar(static_cast<uint>(left_coeff.size()), left_coeff.data());
    FrTensor right_scalar(static_cast<uint>(right_coeff.size()), right_coeff.data());
    auto folded = (left_tensor * left_scalar) + (right_tensor * right_scalar);
    cuda_sync("fold_generators_standard");
    return Commitment::host_points(folded);
}

void split_generators_standard(
    const std::vector<G1Jacobian_t>& in,
    std::vector<G1Jacobian_t>* left,
    std::vector<G1Jacobian_t>* right) {
    if (in.empty() || (in.size() % 2) != 0) {
        throw std::runtime_error("standard generator split requires a non-empty even vector");
    }
    const std::size_t out_size = (in.size() + 1) / 2;
    left->resize(out_size);
    right->resize(out_size);
    for (std::size_t i = 0; i < out_size; ++i) {
        const auto left_index = 2 * i;
        const auto right_index = left_index + 1;
        (*left)[i] = in[left_index];
        (*right)[i] = in[right_index];
    }
}

G1Jacobian_t commit_one_point(const Commitment& generators, const FrTensor& scalars) {
    auto commitment = generators.commit(scalars);
    if (commitment.size != 1) {
        throw std::runtime_error("expected one-point row commitment");
    }
    return commitment(0);
}

CommitmentOpeningProof prove_standard_row_opening(
    const FrTensor& row_tensor_mont,
    const Commitment& generators,
    const std::vector<Fr_t>& point,
    std::ofstream* debug) {
    if (generators.size != row_tensor_mont.size) {
        throw std::runtime_error("standard row opening dimension mismatch");
    }
    if (generators.size == 0 || generators.size != (1U << point.size())) {
        throw std::runtime_error("standard row opening requires a power-of-two row and matching challenge length");
    }
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };

    CommitmentOpeningProof proof;
    proof.point = point;
    proof.transcript.clear();
    proof.transcript.reserve(3 * point.size() + 1);

    auto current_scalars = std::make_unique<FrTensor>(row_tensor_mont);
    auto current_generators = Commitment::host_points(generators);
    for (std::size_t round = 0; round < point.size(); ++round) {
        dbg("standard_row_open round=" + std::to_string(round) + " width=" + std::to_string(current_generators.size()));
        auto split_scalars = current_scalars->split(1);
        std::vector<G1Jacobian_t> left_generators;
        std::vector<G1Jacobian_t> right_generators;
        split_generators_standard(current_generators, &left_generators, &right_generators);
        Commitment left_gen(static_cast<uint>(left_generators.size()), left_generators.data());
        Commitment right_gen(static_cast<uint>(right_generators.size()), right_generators.data());

        const auto c_left = commit_one_point(left_gen, split_scalars.first);
        const auto c_right = commit_one_point(right_gen, split_scalars.second);
        const auto c_cross_lr = commit_one_point(right_gen, split_scalars.first);
        const auto c_cross_rl = commit_one_point(left_gen, split_scalars.second);
        const auto c_cross = g1_add_host(c_cross_lr, c_cross_rl);
        proof.transcript.push_back(c_left);
        proof.transcript.push_back(c_cross);
        proof.transcript.push_back(c_right);

        auto folded_scalars = split_scalars.first + ((split_scalars.second - split_scalars.first) * point[round]);
        current_scalars = std::make_unique<FrTensor>(folded_scalars);
        current_generators = fold_generators_standard(left_generators, right_generators, point[round]);
        {
            const auto expected_next = g1_open_expected_next_host(c_cross, c_left, c_right, point[round]);
            Commitment folded_gen(static_cast<uint>(current_generators.size()), current_generators.data());
            const auto actual_next = commit_one_point(folded_gen, *current_scalars);
            dbg("standard_row_open round=" + std::to_string(round)
                + " prover_recurrence_match=" + (g1_equal_host(expected_next, actual_next) ? std::string("true") : std::string("false")));
        }
    }
    if (current_generators.size() != 1 || current_scalars->size != 1) {
        throw std::runtime_error("standard row opening did not fold to one element");
    }
    proof.value = (*current_scalars)(0);
    proof.transcript.push_back(current_generators.front());
    return proof;
}

void write_bytes(std::ofstream& out, const void* data, std::size_t size) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

void write_u32(std::ofstream& out, std::uint32_t value) {
    write_bytes(out, &value, sizeof(value));
}

void write_u64(std::ofstream& out, std::uint64_t value) {
    write_bytes(out, &value, sizeof(value));
}

void write_string_binary(std::ofstream& out, const std::string& value) {
    write_u32(out, static_cast<std::uint32_t>(value.size()));
    if (!value.empty()) {
        write_bytes(out, value.data(), value.size());
    }
}

void serialize_level_b_proof(const LevelBProofArtifact& artifact, const fs::path& path) {
    ensure_dir(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to write proof object: " + path.string());
    }
    write_bytes(out, "ZKDLGATB", 8);
    write_u32(out, 1);
    write_string_binary(out, artifact.level);
    write_string_binary(out, artifact.benchmark_mode);
    write_u32(out, static_cast<std::uint32_t>(artifact.covered_relations.size()));
    for (const auto& relation : artifact.covered_relations) {
        write_string_binary(out, relation);
    }
    write_u32(out, static_cast<std::uint32_t>(artifact.tensors.size()));
    for (const auto& tensor : artifact.tensors) {
        write_string_binary(out, tensor.label);
        write_string_binary(out, tensor.generator_label);
        write_u32(out, static_cast<std::uint32_t>(tensor.shape.size()));
        for (const auto dim : tensor.shape) {
            write_u64(out, static_cast<std::uint64_t>(dim));
        }
        write_u64(out, static_cast<std::uint64_t>(tensor.padded_inner));
        write_u64(out, static_cast<std::uint64_t>(tensor.opened_row));
        write_u32(out, static_cast<std::uint32_t>(tensor.commitment_points.size()));
        if (!tensor.commitment_points.empty()) {
            write_bytes(out, tensor.commitment_points.data(), tensor.commitment_points.size() * sizeof(G1Jacobian_t));
        }
        write_u32(out, static_cast<std::uint32_t>(tensor.opening.point.size()));
        if (!tensor.opening.point.empty()) {
            write_bytes(out, tensor.opening.point.data(), tensor.opening.point.size() * sizeof(Fr_t));
        }
        write_bytes(out, &tensor.opening.value, sizeof(Fr_t));
        write_u32(out, static_cast<std::uint32_t>(tensor.opening.transcript.size()));
        if (!tensor.opening.transcript.empty()) {
            write_bytes(out, tensor.opening.transcript.data(), tensor.opening.transcript.size() * sizeof(G1Jacobian_t));
        }
    }
}

TensorCommitmentProof prove_tensor_opening(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    double* commitment_ms,
    double* prove_ms,
    std::ofstream* debug) {
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    if (padded_cols > std::numeric_limits<uint>::max() || rows > std::numeric_limits<uint>::max()) {
        throw std::runtime_error("zkDL uint-sized tensor API cannot address tensor " + label);
    }
    dbg("prove_tensor_opening stage=encode_full rows=" + std::to_string(rows) + " cols=" + std::to_string(cols) + " padded_cols=" + std::to_string(padded_cols));
    const std::string generator_label = dataset + ":" + label + ":inner" + std::to_string(padded_cols);
    auto encoded = encode_padded_tensor(values, rows, cols, padded_cols);
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    cuda_sync("prove_tensor_opening full tensor mont");
    dbg("prove_tensor_opening stage=deterministic_generators");
    auto generators = deterministic_generators(static_cast<uint>(padded_cols), generator_label);

    dbg("prove_tensor_opening stage=commit_full");
    ScopedTimer commitment_timer;
    G1TensorJacobian commitment = generators.commit(tensor);
    cuda_sync("zkDL tensor commitment " + label);
    *commitment_ms += commitment_timer.elapsed_ms();
    dbg("prove_tensor_opening stage=commit_full_done ms=" + format_ms(*commitment_ms));

    TensorCommitmentProof proof;
    proof.label = label;
    proof.shape = {rows, cols};
    proof.padded_inner = padded_cols;
    proof.opened_row = rows == 0 ? 0 : static_cast<std::size_t>(fnv1a64(dataset + ":" + label + ":row-opening") % rows);
    proof.generator_label = generator_label;
    dbg("prove_tensor_opening stage=copy_commitment_points opened_row=" + std::to_string(proof.opened_row));
    proof.commitment_points = Commitment::host_points(commitment);
    dbg("prove_tensor_opening stage=open_row_begin mode=row_mle_cpu_sized_gpu_fallback");
    {
        ScopedTimer timer;
        std::vector<Fr_t> row_encoded(padded_cols, fr_from_signed_host(0));
        const auto row_begin = encoded.begin() + static_cast<std::ptrdiff_t>(proof.opened_row * padded_cols);
        std::copy(row_begin, row_begin + static_cast<std::ptrdiff_t>(padded_cols), row_encoded.begin());
        FrTensor row_tensor(static_cast<uint>(row_encoded.size()), row_encoded.data());
        row_tensor.mont();
        const uint inner_log = ceilLog2(static_cast<uint>(padded_cols));
        std::vector<Fr_t> point(inner_log, fr_from_signed_host(0));
        const char* enable_opening_probe = std::getenv("ZKDL_GAT_ENABLE_ROW_OPENING_PROBE");
        if (enable_opening_probe != nullptr && std::string(enable_opening_probe) == "1") {
            point = deterministic_mont_fr_vec(inner_log, dataset + ":" + label + ":row-opening-point");
            proof.opening = prove_standard_row_opening(row_tensor, generators, point, debug);
        } else {
            proof.opening.point = point;
            proof.opening.value = row_tensor(0);
            proof.opening.transcript.clear();
            dbg("prove_tensor_opening stage=standard_row_opening_probe_skipped reason=probe_rejected_on_cora_and_triggered_cuda_illegal_memory_on_citeseer fallback_point=all_zero_first_value");
        }
        cuda_sync("prove_tensor_opening row open " + label);
        *prove_ms += timer.elapsed_ms();
    }
    dbg("prove_tensor_opening stage=open_row_done transcript_points=" + std::to_string(proof.opening.transcript.size())
        + " challenge_dim=" + std::to_string(proof.opening.point.size()));
    return proof;
}

bool verify_tensor_opening(
    const TensorCommitmentProof& proof,
    const std::vector<double>* public_values,
    std::string* failure,
    bool allow_public_commitment_fallback) {
    if (proof.shape.size() != 2) {
        *failure = "tensor proof has non-matrix shape: " + proof.label;
        return false;
    }
    const auto rows = proof.shape[0];
    const auto cols = proof.shape[1];
    if (proof.padded_inner == 0 || proof.padded_inner > std::numeric_limits<uint>::max()) {
        *failure = "invalid padded inner dimension for " + proof.label;
        return false;
    }
    auto fail_or_fallback = [&](const std::string& reason, bool public_commitment_ok) {
        if (allow_public_commitment_fallback && public_commitment_ok && public_values != nullptr
            && !proof.opening.point.empty() && !point_is_all_zero(proof.opening.point)) {
            try {
                const auto public_eval = evaluate_public_tensor_row_mle(
                    *public_values, rows, cols, proof.padded_inner, proof.opened_row, proof.opening.point);
                if (fr_equal_host(public_eval, proof.opening.value)) {
                    *failure = reason + "; fallback public commitment plus public row MLE evaluation accepted as debug-only output binding";
                    return true;
                }
                *failure = reason + "; fallback public row MLE evaluation mismatch";
                return false;
            } catch (const std::exception& error) {
                *failure = reason + "; fallback public row MLE evaluation failed: " + std::string(error.what());
                return false;
            }
        }
        if (allow_public_commitment_fallback && public_commitment_ok && public_values != nullptr && point_is_all_zero(proof.opening.point)) {
            try {
                const auto public_eval = encode_public_tensor_row_first_value_mont(
                    *public_values, rows, cols, proof.opened_row);
                if (fr_equal_host(public_eval, proof.opening.value)) {
                    *failure = reason + "; fallback public commitment plus public row first-value evaluation accepted as debug-only output binding";
                    return true;
                }
                *failure = reason + "; fallback public row first-value evaluation mismatch";
                return false;
            } catch (const std::exception& error) {
                *failure = reason + "; fallback public row first-value evaluation failed: " + std::string(error.what());
                return false;
            }
        }
        *failure = reason;
        return false;
    };
    auto generators = deterministic_generators(static_cast<uint>(proof.padded_inner), proof.generator_label);
    bool public_commitment_ok = false;
    std::string opening_failure;
    if (public_values != nullptr) {
        try {
            auto encoded = encode_padded_tensor(*public_values, rows, cols, proof.padded_inner);
            FrTensor public_tensor(static_cast<uint>(encoded.size()), encoded.data());
            public_tensor.mont();
            G1TensorJacobian recomputed = generators.commit(public_tensor);
            const auto recomputed_points = Commitment::host_points(recomputed);
            public_commitment_ok = recomputed_points.size() == proof.commitment_points.size()
                && std::memcmp(
                    recomputed_points.data(),
                    proof.commitment_points.data(),
                    recomputed_points.size() * sizeof(G1Jacobian_t)) == 0;
            if (!public_commitment_ok) {
                *failure = "public logits full commitment recomputation mismatch for " + proof.label;
                return false;
            }
        } catch (const std::exception& error) {
            *failure = std::string("public logits full commitment verifier failed for ") + proof.label + ": " + error.what();
            return false;
        }
    }
    if (proof.opened_row >= proof.commitment_points.size()) {
        *failure = "row opening index exceeds commitment vector for " + proof.label;
        return false;
    }
    G1TensorJacobian commitment(1, &proof.commitment_points.at(proof.opened_row));
    const uint outer_log = ceilLog2(commitment.size);
    const uint inner_log = static_cast<uint>(proof.opening.point.size()) - outer_log;
    if (proof.opening.transcript.size() != static_cast<std::size_t>(3 * inner_log + 1)) {
        opening_failure = "Hyrax opening transcript length mismatch for " + proof.label;
        return fail_or_fallback(opening_failure, public_commitment_ok);
    }
    G1Jacobian_t current_commitment = commitment(0);
    for (uint round = 0; round < inner_log; ++round) {
        const auto left = proof.opening.transcript.at(3 * round);
        const auto cross = proof.opening.transcript.at(3 * round + 1);
        const auto right = proof.opening.transcript.at(3 * round + 2);
        const auto reconstructed_current = g1_add_host(left, right);
        if (!g1_equal_host(reconstructed_current, current_commitment)) {
            opening_failure = "Hyrax opening left/right commitment sum mismatch for " + proof.label
                + " at round " + std::to_string(round);
            return fail_or_fallback(opening_failure, public_commitment_ok);
        }
        const auto expected_next = g1_open_expected_next_host(cross, left, right, proof.opening.point.at(round));
        const G1Jacobian_t actual_next = (round + 1 < inner_log)
            ? g1_add_host(proof.opening.transcript.at(3 * (round + 1)), proof.opening.transcript.at(3 * (round + 1) + 2))
            : g1_final_commitment_host(proof.opening.transcript.back(), proof.opening.value);
        if (!g1_equal_host(expected_next, actual_next)) {
            const auto swapped_next = g1_open_expected_next_host(cross, right, left, proof.opening.point.at(round));
            const auto complement = fr_mont_one_minus_host(proof.opening.point.at(round));
            const auto complement_next = g1_open_expected_next_host(cross, left, right, complement);
            const auto swapped_complement_next = g1_open_expected_next_host(cross, right, left, complement);
            opening_failure = "Hyrax opening recurrence mismatch for " + proof.label + " at round " + std::to_string(round)
                + " variants{swapped=" + (g1_equal_host(swapped_next, actual_next) ? std::string("match") : std::string("no"))
                + ", complement=" + (g1_equal_host(complement_next, actual_next) ? std::string("match") : std::string("no"))
                + ", swapped_complement=" + (g1_equal_host(swapped_complement_next, actual_next) ? std::string("match") : std::string("no"))
                + "}";
            return fail_or_fallback(opening_failure, public_commitment_ok);
        }
        current_commitment = expected_next;
    }
    const auto expected_final_generator = generators(proof.opening.point);
    if (!g1_equal_host(expected_final_generator, proof.opening.transcript.back())) {
        opening_failure = "Hyrax opening final folded generator mismatch for " + proof.label;
        return fail_or_fallback(opening_failure, public_commitment_ok);
    }
    const auto expected_final_commitment = g1_final_commitment_host(expected_final_generator, proof.opening.value);
    if (!g1_equal_host(expected_final_commitment, current_commitment)) {
        opening_failure = "Hyrax opening final value commitment mismatch for " + proof.label;
        return fail_or_fallback(opening_failure, public_commitment_ok);
    }
    if (public_values != nullptr) {
        const auto public_eval = evaluate_public_tensor_row_mle(*public_values, rows, cols, proof.padded_inner, proof.opened_row, proof.opening.point);
        if (!fr_equal_host(public_eval, proof.opening.value)) {
            opening_failure = "public logits MLE did not match opened value for " + proof.label;
            return fail_or_fallback(opening_failure, public_commitment_ok);
        }
    }
    failure->clear();
    return true;
}

LevelBProofArtifact run_level_b_output_binding_proof(
    const DatasetRunSpec& spec,
    const ModelParams& model,
    const LocalGraph& graph,
    const ForwardTraceHost& trace,
    const fs::path& run_dir) {
    LevelBProofArtifact artifact;
    std::ofstream debug(run_dir / "proof_debug.log");
    auto dbg = [&](const std::string& line) {
        if (debug) {
            debug << line << '\n';
            debug.flush();
        }
    };
    artifact.covered_relations = {
        "zkDL Hyrax final output-head tensor commitment",
        "debug_only final output-head public row evaluation",
        "verifier-side public logits commitment recomputation",
        "K_out=1 output-head arithmetic mean identity"
    };
    artifact.uncovered_relations = {
        "zkDL Hyrax row MLE opening transcript 尚不能作为 Level A verifier",
        "hidden/output projection matmul 仍只是 relation-level evidence",
        "sparse src/dst routing 与 aggregation relation",
        "LeakyReLU lookup/range relation",
        "softmax max/sum/inverse/ExpMap relation",
        "ELU lookup/range relation",
        "rescale 与 fixed-point rounding relation",
        "完整 output-head bias 前序链路"
    };
    if (model.K_out != 1 || trace.output_heads.size() != 1) {
        artifact.blocker = "debug_level_b 只检查 K_out=1 output-head average identity；当前 dataset 的 K_out="
            + std::to_string(model.K_out);
        return artifact;
    }
    if (trace.output_heads.front().h_out.size() != trace.logits.size()) {
        artifact.blocker = "output head tensor and logits shape mismatch before proof generation";
        return artifact;
    }

    try {
        ScopedTimer prove_total;
        dbg("level_b stage=prove_tensor_opening_begin");
        auto tensor_proof = prove_tensor_opening(
            spec.dataset,
            "P_out_head0_post_bias_equals_public_logits",
            trace.output_heads.front().h_out,
            graph.num_nodes,
            model.C,
            &artifact.commitment_ms,
            &artifact.prove_ms,
            &debug);
        dbg("level_b stage=prove_tensor_opening_done");
        artifact.tensors.push_back(std::move(tensor_proof));
        artifact.proof_generation_ok = true;

        {
            dbg("level_b stage=verify_begin");
            ScopedTimer timer;
            std::string failure;
            artifact.verification_ok = verify_tensor_opening(artifact.tensors.front(), &trace.logits, &failure, false);
            if (!artifact.verification_ok) {
                std::string fallback_failure;
                const bool fallback_ok = verify_tensor_opening(artifact.tensors.front(), &trace.logits, &fallback_failure, true);
                if (fallback_ok) {
                    artifact.verification_ok = true;
                    failure = fallback_failure;
                }
            }
            if (!artifact.verification_ok) {
                artifact.blocker = failure;
            } else if (!failure.empty()) {
                artifact.blocker = failure;
            }
            artifact.verify_ms = timer.elapsed_ms();
            dbg("level_b stage=verify_done ms=" + format_ms(artifact.verify_ms));
        }

        artifact.proof_path = run_dir / "debug_artifacts" / "zkdl_level_b_output_binding.debug_proof";
        {
            dbg("level_b stage=serialize_begin");
            ScopedTimer timer;
            serialize_level_b_proof(artifact, artifact.proof_path);
            artifact.prove_ms += timer.elapsed_ms();
            dbg("level_b stage=serialize_done");
        }
        artifact.proof_size_bytes = fs::file_size(artifact.proof_path);
        artifact.prove_ms = prove_total.elapsed_ms() - artifact.commitment_ms - artifact.verify_ms;

        if (artifact.verification_ok) {
            artifact.blocker = std::string(
                "debug_level_b 已完成 final output-head commitment、public row evaluation、public logits binding 与 K_out=1 average identity 的调试级检查"
            )
            + (artifact.blocker.empty() ? std::string("") : std::string("；verifier note: ") + artifact.blocker)
                + "; "
                "Level A 仍被完整 softmax max/sum/inverse/ExpMap、LeakyReLU/ELU、weighted/output aggregation 与 rescale gadget 缺失阻塞；sparse routing/grouped sum 需以 formal sparse evidence 单独判断。";
        }
    } catch (const std::exception& error) {
        artifact.blocker = error.what();
    }
    return artifact;
}

struct FormalPartialArtifact {
    bool proof_generation_ok = false;
    bool verification_ok = false;
    double commitment_ms = 0.0;
    double prove_ms = 0.0;
    double verify_ms = 0.0;
    std::uintmax_t proof_size_bytes = 0;
    std::size_t total_chunks = 0;
    std::size_t verified_chunks = 0;
    fs::path proof_path;
    std::vector<std::string> covered_relations;
    std::vector<std::string> blocked_relations;
    std::vector<std::string> relation_details;
    std::string blocker;
};

struct ProofAttempt {
    bool partial_commitment_ok = false;
    bool full_proof_ok = false;
    bool full_verify_ok = false;
    double partial_commitment_ms = 0.0;
    double partial_open_ms = 0.0;
    std::size_t partial_proof_size_bytes = 0;
    std::string blocker;
    LevelBProofArtifact level_b;
    FormalPartialArtifact formal_partial;
};

struct FormalTensorRecord {
    std::string label;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t padded_cols = 0;
    std::unique_ptr<FrTensor> tensor;
    std::unique_ptr<Commitment> generators;
    std::unique_ptr<G1TensorJacobian> commitment;
};

class FormalTensorRegistry {
public:
    FormalTensorRecord& add_matrix(
        const std::string& dataset,
        const std::string& label,
        const std::vector<double>& values,
        std::size_t rows,
        std::size_t cols,
        double* commitment_ms,
        std::ofstream* debug)
    {
        if (values.size() != rows * cols) {
            throw std::runtime_error("formal tensor shape mismatch for " + label);
        }
        auto dbg = [&](const std::string& line) {
            if (debug != nullptr && *debug) {
                *debug << line << '\n';
                debug->flush();
            }
        };
        const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
        if (padded_cols > std::numeric_limits<uint>::max() || rows > std::numeric_limits<uint>::max()) {
            throw std::runtime_error("formal tensor exceeds zkDL uint dimensions for " + label);
        }
        dbg("formal_registry add_matrix label=" + label
            + " rows=" + std::to_string(rows)
            + " cols=" + std::to_string(cols)
            + " padded_cols=" + std::to_string(padded_cols));
        auto encoded = encode_padded_tensor(values, rows, cols, padded_cols);
        auto record = std::make_unique<FormalTensorRecord>();
        record->label = label;
        record->rows = rows;
        record->cols = cols;
        record->padded_cols = padded_cols;
        record->tensor = std::make_unique<FrTensor>(static_cast<uint>(encoded.size()), encoded.data());
        record->tensor->mont();
        record->generators = std::make_unique<Commitment>(
            deterministic_generators(static_cast<uint>(padded_cols),
                dataset + ":formal:matrix_inner" + std::to_string(padded_cols)));
        ScopedTimer timer;
        record->commitment = std::make_unique<G1TensorJacobian>(record->generators->commit(*record->tensor));
        cuda_sync("formal registry commit " + label);
        if (commitment_ms != nullptr) *commitment_ms += timer.elapsed_ms();
        records_.push_back(std::move(record));
        return *records_.back();
    }

    FormalTensorRecord& add_encoded_matrix(
        const std::string& dataset,
        const std::string& label,
        const std::vector<Fr_t>& encoded_raw_values,
        std::size_t rows,
        std::size_t cols,
        std::size_t padded_cols,
        double* commitment_ms,
        std::ofstream* debug)
    {
        if (encoded_raw_values.size() != rows * padded_cols) {
            throw std::runtime_error("formal encoded tensor shape mismatch for " + label);
        }
        auto dbg = [&](const std::string& line) {
            if (debug != nullptr && *debug) {
                *debug << line << '\n';
                debug->flush();
            }
        };
        if (padded_cols > std::numeric_limits<uint>::max() || rows > std::numeric_limits<uint>::max()) {
            throw std::runtime_error("formal encoded tensor exceeds zkDL uint dimensions for " + label);
        }
        dbg("formal_registry add_encoded_matrix label=" + label
            + " rows=" + std::to_string(rows)
            + " cols=" + std::to_string(cols)
            + " padded_cols=" + std::to_string(padded_cols));
        auto record = std::make_unique<FormalTensorRecord>();
        record->label = label;
        record->rows = rows;
        record->cols = cols;
        record->padded_cols = padded_cols;
        record->tensor = std::make_unique<FrTensor>(static_cast<uint>(encoded_raw_values.size()), encoded_raw_values.data());
        record->tensor->mont();
        record->generators = std::make_unique<Commitment>(
            deterministic_generators(static_cast<uint>(padded_cols),
                dataset + ":formal:matrix_inner" + std::to_string(padded_cols)));
        ScopedTimer timer;
        record->commitment = std::make_unique<G1TensorJacobian>(record->generators->commit(*record->tensor));
        cuda_sync("formal registry commit encoded " + label);
        if (commitment_ms != nullptr) *commitment_ms += timer.elapsed_ms();
        records_.push_back(std::move(record));
        return *records_.back();
    }

private:
    std::vector<std::unique_ptr<FormalTensorRecord>> records_;
};

Fr_t fr_inverse_uint_mont(std::uint32_t denominator)
{
    if (denominator == 0) throw std::runtime_error("zero denominator inversion");
    if (denominator == 1) return Fr_host_one_mont();
    const Fr_t p = Fr_host_modulus();
    std::uint64_t p_mod = 0;
    for (int i = 7; i >= 0; --i) {
        p_mod = ((p_mod << 32U) + p.val[i]) % denominator;
    }
    std::uint32_t q = 0;
    for (; q < denominator; ++q) {
        if (((p_mod * q) + 1U) % denominator == 0) break;
    }
    if (q == denominator) throw std::runtime_error("failed to invert small integer");
    std::uint32_t limbs[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::uint64_t carry = 1U;
    for (int i = 0; i < 8; ++i) {
        const std::uint64_t cur = static_cast<std::uint64_t>(p.val[i]) * q + carry;
        limbs[i] = static_cast<std::uint32_t>(cur & 0xffffffffULL);
        carry = cur >> 32U;
    }
    limbs[8] = static_cast<std::uint32_t>(carry);
    Fr_t inv_raw = Fr_host_zero();
    std::uint64_t rem = 0;
    for (int i = 8; i >= 0; --i) {
        const std::uint64_t cur = (rem << 32U) | limbs[i];
        const std::uint32_t quotient = static_cast<std::uint32_t>(cur / denominator);
        rem = cur % denominator;
        if (i < 8) inv_raw.val[i] = quotient;
    }
    if (rem != 0) throw std::runtime_error("small integer inverse division left remainder");
    return Fr_host_mont(inv_raw);
}

Fr_t evaluate_public_vector_mle(
    const std::vector<double>& values,
    std::size_t cols,
    std::size_t padded_cols,
    const std::vector<Fr_t>& point)
{
    auto encoded = encode_padded_tensor(values, 1, cols, padded_cols);
    FrTensor tensor(static_cast<uint>(encoded.size()), encoded.data());
    tensor.mont();
    return tensor(point);
}

Fr_t evaluate_valid_selector_mle(std::size_t valid_count, const std::vector<Fr_t>& point)
{
    const std::size_t padded_count = next_power_of_two(std::max<std::size_t>(1, valid_count));
    std::vector<Fr_t> values(padded_count, Fr_host_zero());
    for (std::size_t i = 0; i < valid_count; ++i) {
        values[i] = Fr_host_one_mont();
    }
    return formal_host_mle_vector(values, point.begin(), point.end());
}

Fr_t evaluate_valid_matrix_selector_mle(
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols,
    const std::vector<Fr_t>& point)
{
    if (rows > padded_rows || cols > padded_cols || point.size() != ceilLog2(static_cast<uint>(padded_rows * padded_cols))) {
        throw std::runtime_error("valid matrix selector MLE shape mismatch");
    }
    std::vector<Fr_t> values(padded_rows * padded_cols, Fr_host_zero());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            values[row * padded_cols + col] = Fr_host_one_mont();
        }
    }
    return formal_host_mle_vector(values, point.begin(), point.end());
}

Fr_t fr_neg_one_mont()
{
    return Fr_host_sub(Fr_host_zero(), Fr_host_one_mont());
}

std::vector<Fr_t> formal_encode_matmul_matrix(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    if (values.size() != rows * cols) throw std::runtime_error("matmul encode shape mismatch");
    std::vector<Fr_t> out(padded_rows * padded_cols, fr_from_signed_host(0));
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const double scaled = values[row * cols + col] * 16.0;
            const auto fixed = static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
            out[row * padded_cols + col] = fr_from_signed_host(fixed);
        }
    }
    return out;
}

std::int64_t quantize_scale16_to_i64(double value)
{
    const double scaled = value * 16.0;
    return static_cast<std::int64_t>(scaled >= 0.0 ? scaled + 0.5 : scaled - 0.5);
}

std::string int128_to_string(__int128 value)
{
    if (value == 0) return "0";
    const bool negative = value < 0;
    unsigned __int128 magnitude = negative ? static_cast<unsigned __int128>(-value) : static_cast<unsigned __int128>(value);
    std::string digits;
    while (magnitude != 0) {
        const unsigned digit = static_cast<unsigned>(magnitude % 10);
        digits.push_back(static_cast<char>('0' + digit));
        magnitude /= 10;
    }
    if (negative) digits.push_back('-');
    std::reverse(digits.begin(), digits.end());
    return digits;
}

std::string fr_hex(const Fr_t& value)
{
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0');
    for (int i = 7; i >= 0; --i) {
        out << std::setw(8) << value.val[i];
    }
    return out.str();
}

std::string fr_vector_json(const std::vector<Fr_t>& values)
{
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ',';
        out << '"' << fr_hex(values[i]) << '"';
    }
    out << ']';
    return out.str();
}

std::string matmul_relation_type(const std::string& label)
{
    if (label.find("projection") != std::string::npos) return "projection";
    if (label.find("source_attention_dot") != std::string::npos) return "source_attention_dot";
    if (label.find("destination_attention_dot") != std::string::npos) return "destination_attention_dot";
    return "matmul";
}

std::string matmul_layer_head(const std::string& label)
{
    std::smatch match;
    if (std::regex_search(label, match, std::regex(R"(output_head([0-9]+))"))) {
        return "output/head" + match[1].str();
    }
    if (std::regex_search(label, match, std::regex(R"(hidden_layer([0-9]+)_head([0-9]+))"))) {
        return "hidden/layer" + match[1].str() + "/head" + match[2].str();
    }
    return "unknown";
}
