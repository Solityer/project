#pragma once

// projection、concat、bias、average、rescale 相关 formal trace 尝试。

std::vector<Fr_t> formal_matmul_product_mont(
    const std::vector<Fr_t>& a_mont,
    std::size_t rows,
    std::size_t shared,
    std::size_t padded_rows,
    std::size_t padded_shared,
    const std::vector<Fr_t>& b_mont,
    std::size_t cols,
    std::size_t padded_cols)
{
    if (a_mont.size() != padded_rows * padded_shared || b_mont.size() != padded_shared * padded_cols) {
        throw std::runtime_error("fixed-point matmul product shape mismatch");
    }
    std::vector<Fr_t> out(padded_rows * padded_cols, Fr_host_zero());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            Fr_t sum = Fr_host_zero();
            for (std::size_t inner = 0; inner < shared; ++inner) {
                const Fr_t lhs = a_mont[row * padded_shared + inner];
                const Fr_t rhs = b_mont[inner * padded_cols + col];
                sum = Fr_host_add(sum, Fr_host_mul(lhs, rhs));
            }
            out[row * padded_cols + col] = sum;
        }
    }
    return out;
}

std::size_t matmul_chunk_rows(std::size_t shared, std::size_t cols)
{
    constexpr std::size_t kMaxFlatCommitmentPoints = 50000;
    const std::size_t padded_shared = next_power_of_two(std::max<std::size_t>(1, shared));
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    const std::size_t max_row_factor = std::max<std::size_t>(padded_shared, padded_cols);
    std::size_t max_padded_rows = 1;
    while ((max_padded_rows << 1U) * max_row_factor <= kMaxFlatCommitmentPoints && max_padded_rows < 4096U) {
        max_padded_rows <<= 1U;
    }
    return std::max<std::size_t>(1, max_padded_rows);
}

std::string maybe_write_matmul_first_failure_evidence(
    const std::string& dataset,
    const std::string& label,
    std::size_t row_begin,
    std::size_t row_count,
    std::size_t total_chunks,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    std::size_t padded_rows,
    std::size_t padded_shared,
    std::size_t padded_cols,
    const FrTensor& a,
    const FrTensor& b,
    const FrTensor& c_trace,
    const G1TensorJacobian& com_a,
    const G1TensorJacobian& com_b,
    const G1TensorJacobian& com_c_trace,
    const std::vector<Fr_t>& fixed_c_mont)
{
    static std::unordered_set<std::string> written_datasets;

    FormalTranscript transcript;
    const std::string relation_label = dataset + ":" + label;
    formal_absorb_matmul_metadata(transcript, relation_label,
        1U,
        static_cast<uint>(rows),
        static_cast<uint>(shared),
        static_cast<uint>(cols),
        1U,
        static_cast<uint>(padded_rows),
        static_cast<uint>(padded_shared),
        static_cast<uint>(padded_cols),
        com_a,
        com_b,
        com_c_trace);
    const auto point_num = formal_make_challenge_point(transcript, 0, "matmul_num_point");
    const auto point_m = formal_make_challenge_point(transcript, ceilLog2(static_cast<uint>(padded_rows)), "matmul_m_point");
    const auto point_k = formal_make_challenge_point(transcript, ceilLog2(static_cast<uint>(padded_cols)), "matmul_k_point");
    const auto point_c = formal_concat_points({point_k, point_m, point_num});
    const Fr_t claimed_c_eval = c_trace(point_c);
    transcript.absorb_fr(claimed_c_eval);

    auto reduced_a = formal_matmul_reduce_a_over_m(a.to_host(), 1U, static_cast<uint>(padded_rows), static_cast<uint>(padded_shared), point_m);
    auto reduced_b = formal_matmul_reduce_b_over_k(b.to_host(), 1U, static_cast<uint>(padded_shared), static_cast<uint>(padded_cols), point_k);
    const Fr_t actual_inner_claim = formal_inner_product_claim_host(reduced_a, reduced_b);
    const Fr_t mismatch_delta = Fr_host_sub(actual_inner_claim, claimed_c_eval);
    if (Fr_host_eq(mismatch_delta, Fr_host_zero())) {
        return "";
    }
    if (!written_datasets.insert(dataset).second) {
        return "";
    }

    FrTensor fixed_c(static_cast<uint>(fixed_c_mont.size()), const_cast<Fr_t*>(fixed_c_mont.data()));
    const Fr_t fixed_c_eval = fixed_c(point_c);
    const bool fixed_eval_matches = Fr_host_eq(fixed_c_eval, actual_inner_claim);

    std::size_t sample_row = 0;
    std::size_t sample_col = 0;
    std::int64_t sample_trace_c = c_values.empty() ? 0 : quantize_scale16_to_i64(c_values.front());
    __int128 sample_product = 0;
    bool sample_matches = true;
    bool sample_found = false;
    if (!a_values.empty() && !b_values.empty() && !c_values.empty() && cols != 0) {
        for (std::size_t row = 0; row < rows && !sample_found; ++row) {
            for (std::size_t col = 0; col < cols; ++col) {
                __int128 product = 0;
                for (std::size_t inner = 0; inner < shared; ++inner) {
                    const std::int64_t lhs = quantize_scale16_to_i64(a_values[row * shared + inner]);
                    const std::int64_t rhs = quantize_scale16_to_i64(b_values[inner * cols + col]);
                    product += static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
                }
                const std::int64_t trace_c = quantize_scale16_to_i64(c_values[row * cols + col]);
                if (static_cast<__int128>(trace_c) != product || (row + 1 == rows && col + 1 == cols)) {
                    sample_row = row;
                    sample_col = col;
                    sample_trace_c = trace_c;
                    sample_product = product;
                    sample_matches = (static_cast<__int128>(trace_c) == product);
                    sample_found = true;
                    break;
                }
            }
        }
    }

    ensure_dir(fs::path("/home/pzh/project/comparison/baselines/zkdl/debug"));
    const fs::path json_path = fs::path("/home/pzh/project/comparison/baselines/zkdl/debug")
        / ("matmul_first_failure_" + dataset + ".json");
    const fs::path md_path = fs::path("/home/pzh/project/comparison/baselines/zkdl/debug")
        / ("matmul_first_failure_" + dataset + ".md");
    if (fs::exists(json_path) || fs::exists(md_path)) {
        return "";
    }
    const std::string conclusion = fixed_eval_matches && !sample_matches
        ? "域编码与量化路径不一致：float forward 中的 C 是 scale-16 rounded output，而 MatmulRelationProof 需要 scale-256 pre-rescale product witness"
        : "fixed-point host check 后仍无法唯一定位 matmul mismatch，需要继续检查 flatten、order 与 point metadata";

    {
        std::ofstream out(json_path);
        out << "{\n";
        out << "  \"dataset\": \"" << json_escape(dataset) << "\",\n";
        out << "  \"relation_label\": \"" << json_escape(label) << "\",\n";
        out << "  \"layer_head\": \"" << json_escape(matmul_layer_head(label)) << "\",\n";
        out << "  \"relation_type\": \"" << json_escape(matmul_relation_type(label)) << "\",\n";
        out << "  \"chunk_index\": 0,\n";
        out << "  \"row_range\": [\"" << row_begin << "\", \"" << (row_begin + row_count) << "\"],\n";
        out << "  \"logical_shape\": {\"rows\":\"" << rows << "\",\"shared\":\"" << shared << "\",\"cols\":\"" << cols << "\"},\n";
        out << "  \"padded_shape\": {\"num\":\"1\",\"rows\":\"" << padded_rows << "\",\"shared\":\"" << padded_shared << "\",\"cols\":\"" << padded_cols << "\"},\n";
        out << "  \"total_chunks\": \"" << total_chunks << "\",\n";
        out << "  \"quant_cfg_id\": \"adapter_fixed_scale_16_current_path\",\n";
        out << "  \"current_scale\": \"16\",\n";
        out << "  \"pre_rescale_product_scale\": \"256\",\n";
        out << "  \"c_from_float_forward\": true,\n";
        out << "  \"proof_c_source_after_fix\": \"fixed_point_pre_rescale_product_witness\",\n";
        out << "  \"claimed_c_evaluation\": \"" << fr_hex(claimed_c_eval) << "\",\n";
        out << "  \"reduced_inner_product_claim\": \"" << fr_hex(claimed_c_eval) << "\",\n";
        out << "  \"actual_inner_product_proof_claim\": \"" << fr_hex(actual_inner_claim) << "\",\n";
        out << "  \"fixed_point_c_evaluation\": \"" << fr_hex(fixed_c_eval) << "\",\n";
        out << "  \"mismatch_delta_actual_minus_claimed\": \"" << fr_hex(mismatch_delta) << "\",\n";
        out << "  \"opening_point\": {\"point_k\":" << fr_vector_json(point_k)
            << ",\"point_m\":" << fr_vector_json(point_m)
            << ",\"point_num\":" << fr_vector_json(point_num)
            << ",\"point_c\":" << fr_vector_json(point_c) << "},\n";
        out << "  \"transcript_digest_or_challenge_id\": \"" << transcript.digest_hex() << "\",\n";
        out << "  \"source_file\": \"gat_zkdl.cu\",\n";
        out << "  \"source_function\": \"run_formal_matmul_trace_attempt / maybe_write_matmul_first_failure_evidence\",\n";
        out << "  \"evidence_file_path\": \"" << json_escape(json_path.string()) << "\",\n";
        out << "  \"minimal_host_reference_check\": {\n";
        out << "    \"sample_row_in_chunk\": \"" << sample_row << "\",\n";
        out << "    \"sample_col\": \"" << sample_col << "\",\n";
        out << "    \"trace_c_fixed_scale16\": \"" << sample_trace_c << "\",\n";
        out << "    \"pre_rescale_product_of_fixed_inputs_scale256\": \"" << int128_to_string(sample_product) << "\",\n";
        out << "    \"sample_matches_without_rescale\": " << (sample_matches ? "true" : "false") << ",\n";
        out << "    \"fixed_point_mle_matches_inner_claim\": " << (fixed_eval_matches ? "true" : "false") << ",\n";
        out << "    \"conclusion\": \"" << json_escape(conclusion) << "\"\n";
        out << "  }\n";
        out << "}\n";
    }
    {
        std::ofstream out(md_path);
        out << "# Matmul 首个失败证据：" << dataset << "\n\n";
        out << "- relation: `" << label << "`\n";
        out << "- layer/head: `" << matmul_layer_head(label) << "`\n";
        out << "- chunk: `0`，rows `" << row_begin << ".." << (row_begin + row_count) << "`，total chunks `" << total_chunks << "`\n";
        out << "- logical shape: rows `" << rows << "`，shared `" << shared << "`，cols `" << cols << "`\n";
        out << "- padded shape: num `1`，rows `" << padded_rows << "`，shared `" << padded_shared << "`，cols `" << padded_cols << "`\n";
        out << "- 当前量化路径：scale `16`；pre-rescale product scale 为 `256`\n";
        out << "- C 来自 float forward：`true`\n";
        out << "- claimed C evaluation: `" << fr_hex(claimed_c_eval) << "`\n";
        out << "- actual inner-product proof claim: `" << fr_hex(actual_inner_claim) << "`\n";
        out << "- delta actual-claimed: `" << fr_hex(mismatch_delta) << "`\n";
        out << "- transcript digest/challenge id: `" << transcript.digest_hex() << "`\n";
        out << "- opening point C: `" << fr_vector_json(point_c) << "`\n";
        out << "- evidence path: `" << json_path.string() << "`\n\n";
        out << "最小 host reference check：trace `C[" << sample_row << "," << sample_col << "]` 在 scale 16 下的量化值是 `" << sample_trace_c
            << "`，fixed input 在 rescale 前的乘积是 `" << int128_to_string(sample_product)
            << "`，其 scale 为 256。结论：" << conclusion << "。\n\n";
        out << "本证据不包含完整 witness tensor、完整 intermediate tensor、完整 output-head tensor 或完整 logits tensor。\n";
    }
    return json_path.string();
}

struct ScaleDecompositionStats {
    std::size_t checked_values = 0;
    std::size_t outside_nearest_rounding_bound = 0;
    std::string max_abs_residual = "0";
    std::string sample_row = "0";
    std::string sample_col = "0";
    std::string sample_product_scale256 = "0";
    std::string sample_trace_scale16 = "0";
    std::string sample_residual = "0";
};

ScaleDecompositionStats compute_scale_decomposition_stats(
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values)
{
    ScaleDecompositionStats stats;
    if (a_values.size() != rows * shared || b_values.size() != shared * cols || c_values.size() != rows * cols) {
        return stats;
    }
    bool sampled = false;
    __int128 max_abs_residual = 0;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            __int128 product = 0;
            for (std::size_t inner = 0; inner < shared; ++inner) {
                const std::int64_t lhs = quantize_scale16_to_i64(a_values[row * shared + inner]);
                const std::int64_t rhs = quantize_scale16_to_i64(b_values[inner * cols + col]);
                product += static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
            }
            const std::int64_t trace = quantize_scale16_to_i64(c_values[row * cols + col]);
            const __int128 residual = product - static_cast<__int128>(16) * static_cast<__int128>(trace);
            const __int128 abs_residual = residual < 0 ? -residual : residual;
            if (abs_residual > max_abs_residual) {
                max_abs_residual = abs_residual;
            }
            ++stats.checked_values;
            if (abs_residual > 8) {
                ++stats.outside_nearest_rounding_bound;
                if (!sampled) {
                    stats.sample_row = std::to_string(row);
                    stats.sample_col = std::to_string(col);
                    stats.sample_product_scale256 = int128_to_string(product);
                    stats.sample_trace_scale16 = std::to_string(trace);
                    stats.sample_residual = int128_to_string(residual);
                    sampled = true;
                }
            }
            if (!sampled && row + 1 == rows && col + 1 == cols) {
                stats.sample_row = std::to_string(row);
                stats.sample_col = std::to_string(col);
                stats.sample_product_scale256 = int128_to_string(product);
                stats.sample_trace_scale16 = std::to_string(trace);
                stats.sample_residual = int128_to_string(residual);
            }
        }
    }
    stats.max_abs_residual = int128_to_string(max_abs_residual);
    return stats;
}

std::uint64_t checked_int128_to_u64(__int128 value, const std::string& label)
{
    if (value < 0 || value > static_cast<__int128>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::runtime_error(label + " is outside uint64 range");
    }
    return static_cast<std::uint64_t>(value);
}

Fr_t fr_from_u64_mont_host(std::uint64_t value)
{
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("field conversion value exceeds signed int64 helper range");
    }
    return fr_from_signed_mont_host(static_cast<std::int64_t>(value));
}

struct ResidualRangeWitness {
    std::vector<std::vector<Fr_t>> bit_raw_values;
    std::size_t checked_values = 0;
    std::size_t bit_count = 0;
    std::uint64_t offset = 0;
    __int128 max_abs_residual = 0;
    std::string sample_row = "0";
    std::string sample_col = "0";
    std::string sample_residual = "0";
    std::string sample_shifted = "0";
};

ResidualRangeWitness build_residual_range_witness(
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    if (a_values.size() != rows * shared || b_values.size() != shared * cols || c_values.size() != rows * cols) {
        throw std::runtime_error("residual range witness shape mismatch");
    }
    std::vector<__int128> residuals(rows * cols, 0);
    ResidualRangeWitness witness;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            __int128 product = 0;
            for (std::size_t inner = 0; inner < shared; ++inner) {
                const std::int64_t lhs = quantize_scale16_to_i64(a_values[row * shared + inner]);
                const std::int64_t rhs = quantize_scale16_to_i64(b_values[inner * cols + col]);
                product += static_cast<__int128>(lhs) * static_cast<__int128>(rhs);
            }
            const std::int64_t trace = quantize_scale16_to_i64(c_values[row * cols + col]);
            const __int128 residual = product - static_cast<__int128>(16) * static_cast<__int128>(trace);
            const __int128 abs_residual = residual < 0 ? -residual : residual;
            residuals[row * cols + col] = residual;
            if (abs_residual > witness.max_abs_residual) {
                witness.max_abs_residual = abs_residual;
                witness.sample_row = std::to_string(row);
                witness.sample_col = std::to_string(col);
                witness.sample_residual = int128_to_string(residual);
            }
            ++witness.checked_values;
        }
    }

    std::uint64_t offset = 1;
    const auto max_abs = checked_int128_to_u64(witness.max_abs_residual, "max residual magnitude");
    while (offset <= max_abs) {
        if (offset > (std::uint64_t{1} << 62U)) {
            throw std::runtime_error("residual range proof exceeds supported 63-bit shifted range");
        }
        offset <<= 1U;
    }
    witness.offset = offset;
    const std::uint64_t shifted_max = offset + max_abs;
    std::size_t bit_count = 1;
    while (bit_count < 63 && (std::uint64_t{1} << bit_count) <= shifted_max) {
        ++bit_count;
    }
    if (bit_count > 63 || (bit_count == 63 && shifted_max >= (std::uint64_t{1} << 63U))) {
        throw std::runtime_error("residual range proof bit width exceeds supported field helper range");
    }
    witness.bit_count = bit_count;
    witness.bit_raw_values.assign(bit_count, std::vector<Fr_t>(padded_rows * padded_cols, fr_from_signed_host(0)));

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            const __int128 shifted_signed = residuals[row * cols + col] + static_cast<__int128>(offset);
            const std::uint64_t shifted = checked_int128_to_u64(shifted_signed, "shifted residual");
            if (shifted > shifted_max) {
                throw std::runtime_error("shifted residual exceeds declared range");
            }
            if (row == std::stoull(witness.sample_row) && col == std::stoull(witness.sample_col)) {
                witness.sample_shifted = std::to_string(shifted);
            }
            const std::size_t flat = row * padded_cols + col;
            for (std::size_t bit = 0; bit < bit_count; ++bit) {
                if (((shifted >> bit) & 1ULL) != 0ULL) {
                    witness.bit_raw_values[bit][flat] = fr_from_signed_host(1);
                }
            }
        }
    }
    return witness;
}

bool run_residual_range_bound_relation(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    std::size_t padded_rows,
    std::size_t padded_cols,
    FrTensor& residual,
    const G1TensorJacobian& com_residual,
    const Commitment& gen,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (artifact == nullptr) return false;
    try {
        auto witness = build_residual_range_witness(
            a_values, rows, shared, b_values, cols, c_values, padded_rows, padded_cols);
        std::vector<std::unique_ptr<FrTensor>> bit_tensors;
        std::vector<std::unique_ptr<G1TensorJacobian>> bit_commitments;
        bit_tensors.reserve(witness.bit_count);
        bit_commitments.reserve(witness.bit_count);

        for (std::size_t bit = 0; bit < witness.bit_count; ++bit) {
            auto bit_tensor = std::make_unique<FrTensor>(
                static_cast<uint>(witness.bit_raw_values[bit].size()),
                witness.bit_raw_values[bit].data());
            bit_tensor->mont();
            ScopedTimer commit_timer;
            auto bit_commitment = std::make_unique<G1TensorJacobian>(gen.commit(*bit_tensor));
            cuda_sync("formal residual range bit commit " + label + " bit " + std::to_string(bit));
            artifact->commitment_ms += commit_timer.elapsed_ms();

            ++artifact->total_chunks;
            ScopedTimer prove_timer;
            auto bit_proof = formal_hadamard_relation_prove(
                *bit_tensor,
                *bit_tensor,
                *bit_tensor,
                gen,
                *bit_commitment,
                *bit_commitment,
                *bit_commitment,
                dataset + ":" + label + ":residual_range_bit_" + std::to_string(bit));
            artifact->prove_ms += prove_timer.elapsed_ms();
            ScopedTimer verify_timer;
            std::string failure;
            const bool ok = formal_hadamard_relation_verify(
                bit_proof,
                gen,
                *bit_commitment,
                *bit_commitment,
                *bit_commitment,
                dataset + ":" + label + ":residual_range_bit_" + std::to_string(bit),
                &failure);
            artifact->verify_ms += verify_timer.elapsed_ms();
            if (!ok) {
                artifact->blocked_relations.push_back(
                    label + ": residual range bitness verifier rejected bit "
                    + std::to_string(bit) + ": " + failure);
                dbg("formal_residual_range label=" + label + " bit=" + std::to_string(bit)
                    + " status=REJECT failure=\"" + failure + "\"");
                return false;
            }
            ++artifact->verified_chunks;
            artifact->proof_size_bytes += bit_proof.serialized_size_bytes();
            bit_tensors.push_back(std::move(bit_tensor));
            bit_commitments.push_back(std::move(bit_commitment));
        }

        const auto point = deterministic_mont_fr_vec(
            ceilLog2(static_cast<uint>(padded_rows * padded_cols)),
            dataset + ":" + label + ":residual_range_decomposition_point");
        const Fr_t selector_eval = evaluate_valid_matrix_selector_mle(rows, cols, padded_rows, padded_cols, point);
        std::vector<FormalLinearTerm> terms;
        std::vector<const G1TensorJacobian*> commitments;
        terms.push_back({"residual_signed", &residual, &com_residual, Fr_host_one_mont(), point});
        commitments.push_back(&com_residual);
        for (std::size_t bit = 0; bit < witness.bit_count; ++bit) {
            const std::uint64_t coeff = std::uint64_t{1} << bit;
            const Fr_t neg_coeff = Fr_host_sub(Fr_host_zero(), fr_from_u64_mont_host(coeff));
            terms.push_back({
                "shifted_residual_bit_" + std::to_string(bit),
                bit_tensors[bit].get(),
                bit_commitments[bit].get(),
                neg_coeff,
                point});
            commitments.push_back(bit_commitments[bit].get());
        }

        ++artifact->total_chunks;
        ScopedTimer prove_timer;
        auto decomposition_proof = formal_linear_relation_prove(
            dataset + ":" + label + ":residual_range_shifted_decomposition",
            {static_cast<uint>(padded_rows), static_cast<uint>(padded_cols)},
            terms,
            gen,
            fr_from_u64_mont_host(witness.offset),
            selector_eval,
            Fr_host_zero());
        artifact->prove_ms += prove_timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        const bool ok = formal_linear_relation_verify(
            decomposition_proof,
            gen,
            commitments,
            selector_eval,
            &failure);
        artifact->verify_ms += verify_timer.elapsed_ms();
        if (!ok) {
            artifact->blocked_relations.push_back(
                label + ": residual shifted range decomposition verifier rejected: " + failure);
            dbg("formal_residual_range label=" + label + " decomposition status=REJECT failure=\"" + failure + "\"");
            return false;
        }
        ++artifact->verified_chunks;
        artifact->proof_size_bytes += decomposition_proof.serialized_size_bytes();
        artifact->covered_relations.push_back(
            label + ": partial residual bounded range proof verified by bitness plus residual+offset decomposition");
        artifact->relation_details.push_back(
            label
            + " residual_range_bound status=PASS bit_count=" + std::to_string(witness.bit_count)
            + " offset=" + std::to_string(witness.offset)
            + " max_abs_residual=" + int128_to_string(witness.max_abs_residual)
            + " checked_values=" + std::to_string(witness.checked_values)
            + " sample[row=" + witness.sample_row
            + ",col=" + witness.sample_col
            + ",residual=" + witness.sample_residual
            + ",shifted=" + witness.sample_shifted
            + "] proof_size_bytes="
            + std::to_string(decomposition_proof.serialized_size_bytes())
            + " transcript_digest=" + decomposition_proof.transcript_digest);
        artifact->blocked_relations.push_back(
            label + ": residual bounded range is verifier-bound, but sign-specific rounding mode and tie-breaking are still not proven");
        dbg("formal_residual_range label=" + label
            + " status=PASS bit_count=" + std::to_string(witness.bit_count)
            + " offset=" + std::to_string(witness.offset)
            + " max_abs_residual=" + int128_to_string(witness.max_abs_residual));
        return true;
    } catch (const std::exception& error) {
        artifact->blocked_relations.push_back(label + ": residual range proof failed: " + std::string(error.what()));
        dbg("formal_residual_range label=" + label + " status=FAILED reason=\"" + std::string(error.what()) + "\"");
        return false;
    }
}

bool run_fixed_to_trace_scale_decomposition_relation(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    std::size_t padded_rows,
    std::size_t padded_cols,
    const std::vector<Fr_t>& c_fixed_mont,
    const FrTensor& c_trace,
    const G1TensorJacobian& com_c_fixed,
    const G1TensorJacobian& com_c_trace,
    const Commitment& gen,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (artifact == nullptr) {
        return false;
    }
    const auto c_trace_mont = c_trace.to_host();
    if (c_fixed_mont.size() != c_trace_mont.size() || c_fixed_mont.size() != padded_rows * padded_cols) {
        artifact->blocked_relations.push_back(label + ": fixed-to-trace scale decomposition shape mismatch");
        return false;
    }
    const Fr_t sixteen = fr_from_signed_mont_host(16);
    std::vector<Fr_t> residual_mont(c_fixed_mont.size(), Fr_host_zero());
    for (std::size_t i = 0; i < c_fixed_mont.size(); ++i) {
        residual_mont[i] = Fr_host_sub(c_fixed_mont[i], Fr_host_mul(sixteen, c_trace_mont[i]));
    }

    FrTensor residual(static_cast<uint>(residual_mont.size()), residual_mont.data());
    ScopedTimer commit_timer;
    const auto com_residual = gen.commit(residual);
    cuda_sync("formal fixed-to-trace residual commit " + label);
    artifact->commitment_ms += commit_timer.elapsed_ms();

    ++artifact->total_chunks;
    const auto point = deterministic_mont_fr_vec(
        ceilLog2(static_cast<uint>(c_fixed_mont.size())),
        dataset + ":" + label + ":fixed_to_trace_scale_decomposition_point");
    const Fr_t neg_sixteen = Fr_host_sub(Fr_host_zero(), sixteen);
    std::vector<FormalLinearTerm> terms;
    terms.push_back({"C_fixed_pre_rescale", nullptr, &com_c_fixed, Fr_host_one_mont(), point});
    terms.push_back({"C_trace_rounded_scale16", &c_trace, &com_c_trace, neg_sixteen, point});
    terms.push_back({"residual_unranged", &residual, &com_residual, fr_neg_one_mont(), point});

    // `formal_linear_relation_prove` 需要所有项都有 tensor 指针；这里用已是 Montgomery 形式的值构造轻量包装。
    FrTensor c_fixed_tensor(static_cast<uint>(c_fixed_mont.size()), const_cast<Fr_t*>(c_fixed_mont.data()));
    terms[0].tensor = &c_fixed_tensor;

    try {
        ScopedTimer prove_timer;
        auto proof = formal_linear_relation_prove(
            dataset + ":" + label + ":fixed_to_trace_scale_decomposition",
            {static_cast<uint>(padded_rows), static_cast<uint>(padded_cols)},
            terms,
            gen,
            Fr_host_zero(),
            Fr_host_zero(),
            Fr_host_zero());
        artifact->prove_ms += prove_timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        const bool ok = formal_linear_relation_verify(
            proof,
            gen,
            {&com_c_fixed, &com_c_trace, &com_residual},
            Fr_host_zero(),
            &failure);
        artifact->verify_ms += verify_timer.elapsed_ms();
        if (!ok) {
            artifact->blocked_relations.push_back(label + ": fixed-to-trace scale decomposition verifier rejected: " + failure);
            dbg("formal_rescale_decomposition label=" + label + " status=REJECT failure=\"" + failure + "\"");
            return false;
        }
        const auto stats = compute_scale_decomposition_stats(a_values, rows, shared, b_values, cols, c_values);
        ++artifact->verified_chunks;
        artifact->proof_size_bytes += proof.serialized_size_bytes();
        artifact->covered_relations.push_back(
            label + ": partial fixed-to-trace scale decomposition LinearRelationProof verified C_fixed - 16*C_trace - residual = 0");
        artifact->relation_details.push_back(
            label
            + " fixed_to_trace_scale_decomposition status=PASS proof_size_bytes="
            + std::to_string(proof.serialized_size_bytes())
            + " transcript_digest=" + proof.transcript_digest
            + " residual_outside_nearest_rounding_bound="
            + std::to_string(stats.outside_nearest_rounding_bound)
            + "/" + std::to_string(stats.checked_values)
            + " max_abs_residual=" + stats.max_abs_residual
            + " sample[row=" + stats.sample_row
            + ",col=" + stats.sample_col
            + ",product_scale256=" + stats.sample_product_scale256
            + ",trace_scale16=" + stats.sample_trace_scale16
            + ",residual=" + stats.sample_residual + "]");
        artifact->blocked_relations.push_back(
            label
            + ": fixed-to-trace scale decomposition is verifier-bound, but full rounding remains blocked because residual range/sign constraints are not implemented as a zkDL verifier gadget; residual_outside_nearest_rounding_bound="
            + std::to_string(stats.outside_nearest_rounding_bound)
            + "/" + std::to_string(stats.checked_values));
        dbg("formal_rescale_decomposition label=" + label + " status=PASS proof_size_bytes="
            + std::to_string(proof.serialized_size_bytes())
            + " residual_outside_nearest_rounding_bound="
            + std::to_string(stats.outside_nearest_rounding_bound)
            + "/" + std::to_string(stats.checked_values));
        run_residual_range_bound_relation(
            dataset,
            label,
            a_values,
            rows,
            shared,
            b_values,
            cols,
            c_values,
            padded_rows,
            padded_cols,
            residual,
            com_residual,
            gen,
            artifact,
            debug);
        return true;
    } catch (const std::exception& error) {
        artifact->blocked_relations.push_back(label + ": fixed-to-trace scale decomposition failed: " + std::string(error.what()));
        dbg("formal_rescale_decomposition label=" + label + " status=FAILED reason=\"" + std::string(error.what()) + "\"");
        return false;
    }
}

std::vector<double> matrix_row_slice(
    const std::vector<double>& values,
    std::size_t total_rows,
    std::size_t cols,
    std::size_t row_begin,
    std::size_t row_count)
{
    if (row_begin > total_rows || row_begin + row_count > total_rows || values.size() != total_rows * cols) {
        throw std::runtime_error("matrix row slice shape mismatch");
    }
    std::vector<double> out(row_count * cols, 0.0);
    for (std::size_t row = 0; row < row_count; ++row) {
        const auto src = values.begin() + static_cast<std::ptrdiff_t>((row_begin + row) * cols);
        std::copy(src, src + static_cast<std::ptrdiff_t>(cols), out.begin() + static_cast<std::ptrdiff_t>(row * cols));
    }
    return out;
}

std::vector<Fr_t> encode_pre_rescaled_from_head_and_bias(
    const std::vector<double>& head_values,
    const std::vector<double>& bias,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_cols)
{
    if (bias.size() != cols) {
        throw std::runtime_error("bias vector shape mismatch for encoded pre-rescaled tensor");
    }
    auto head_encoded = encode_padded_tensor(head_values, rows, cols, padded_cols);
    auto bias_encoded = encode_padded_tensor(bias, 1, cols, padded_cols);
    std::vector<Fr_t> out(rows * padded_cols, Fr_host_zero());
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < padded_cols; ++col) {
            out[row * padded_cols + col] = Fr_host_sub(head_encoded[row * padded_cols + col], bias_encoded[col]);
        }
    }
    return out;
}

std::vector<Fr_t> concat_column_point_for_head(
    std::size_t head_index,
    std::size_t head_width,
    std::size_t concat_width,
    const std::vector<Fr_t>& head_col_point)
{
    const std::size_t padded_head_width = next_power_of_two(std::max<std::size_t>(1, head_width));
    const std::size_t padded_concat_width = next_power_of_two(std::max<std::size_t>(1, concat_width));
    const uint head_log = ceilLog2(static_cast<uint>(padded_head_width));
    const uint concat_log = ceilLog2(static_cast<uint>(padded_concat_width));
    if (head_col_point.size() != head_log || concat_log < head_log) {
        throw std::runtime_error("concat column point shape mismatch");
    }
    if (padded_head_width != head_width || padded_concat_width != concat_width) {
        throw std::runtime_error("concat proof currently requires power-of-two head and concat widths");
    }
    std::vector<Fr_t> out(concat_log, Fr_host_zero());
    for (uint bit = 0; bit < head_log; ++bit) {
        out[bit] = head_col_point[bit];
    }
    for (uint bit = head_log; bit < concat_log; ++bit) {
        const std::size_t selector_bit = bit - head_log;
        out[bit] = ((head_index >> selector_bit) & 1U) ? Fr_host_one_mont() : Fr_host_zero();
    }
    return out;
}

std::string chunk_label(const std::string& base, std::size_t chunk_idx)
{
    return base + "_chunk" + std::to_string(chunk_idx);
}

std::string chunk_relation_label(
    const std::string& base,
    std::size_t chunk_idx,
    std::size_t row_begin,
    std::size_t row_count,
    std::size_t total_chunks)
{
    return base
        + ":chunk" + std::to_string(chunk_idx)
        + ":offset" + std::to_string(row_begin)
        + ":rows" + std::to_string(row_count)
        + ":total_chunks" + std::to_string(total_chunks);
}

void run_output_linear_chunked_relations(
    const DatasetRunSpec& spec,
    const ModelParams& model,
    const ForwardTraceHost& trace,
    std::size_t rows,
    std::size_t cols,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (trace.output_heads.size() != model.K_out) {
        throw std::runtime_error("output head trace count mismatch in chunked linear relations");
    }
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    const std::size_t max_chunk_values = 262144;
    const std::size_t max_chunk_rows = 4096;
    const std::size_t chunk_rows = std::max<std::size_t>(1, std::min(max_chunk_rows, std::max<std::size_t>(1, max_chunk_values / padded_cols)));
    const std::size_t total_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    std::size_t average_verified = 0;
    std::size_t average_attempted = 0;
    std::size_t bias_verified = 0;
    std::size_t bias_attempted = 0;
    std::size_t bias_rejected = 0;
    std::size_t average_rejected = 0;
    dbg("formal_chunked_linear rows=" + std::to_string(rows)
        + " cols=" + std::to_string(cols)
        + " padded_cols=" + std::to_string(padded_cols)
        + " chunk_rows=" + std::to_string(chunk_rows)
        + " total_chunks=" + std::to_string(total_chunks));

    for (std::size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const std::size_t row_begin = chunk_idx * chunk_rows;
        const std::size_t row_count = std::min(chunk_rows, rows - row_begin);
        FormalTensorRegistry chunk_registry;
        const uint point_dim = ceilLog2(static_cast<uint>(padded_cols)) + ceilLog2(static_cast<uint>(row_count));
        const auto point = deterministic_mont_fr_vec(
            point_dim,
            spec.dataset + ":formal:output-linear-chunk:" + std::to_string(chunk_idx));
        const uint col_log = ceilLog2(static_cast<uint>(padded_cols));
        const std::vector<Fr_t> col_point(point.begin(), point.begin() + col_log);
        const std::vector<Fr_t> row_point(point.begin() + col_log, point.end());
        const Fr_t row_valid_eval = evaluate_valid_selector_mle(row_count, row_point);
        std::vector<FormalTensorRecord*> post_records;
        post_records.reserve(trace.output_heads.size());
        for (std::size_t head = 0; head < trace.output_heads.size(); ++head) {
            const auto& head_trace = trace.output_heads[head];
            const auto post_values = matrix_row_slice(head_trace.h_out, rows, cols, row_begin, row_count);
            const auto pre_rescaled_encoded = encode_pre_rescaled_from_head_and_bias(
                post_values,
                model.output_heads[head].output_bias,
                row_count,
                cols,
                padded_cols);
            auto& pre_rescaled = chunk_registry.add_encoded_matrix(
                spec.dataset,
                chunk_label("output_head" + std::to_string(head) + "_Y_pre_rescaled", chunk_idx),
                pre_rescaled_encoded,
                row_count,
                cols,
                padded_cols,
                &artifact->commitment_ms,
                debug);
            auto& post = chunk_registry.add_matrix(
                spec.dataset,
                chunk_label("output_head" + std::to_string(head) + "_Y_head", chunk_idx),
                post_values,
                row_count,
                cols,
                &artifact->commitment_ms,
                debug);
            post_records.push_back(&post);
            const Fr_t bias_eval = Fr_host_mul(
                evaluate_public_vector_mle(model.output_heads[head].output_bias, cols, padded_cols, col_point),
                row_valid_eval);
            std::vector<FormalLinearTerm> terms;
            terms.push_back({"Y_head", post.tensor.get(), post.commitment.get(), Fr_host_one_mont(), point});
            terms.push_back({"Y_pre_rescaled", pre_rescaled.tensor.get(), pre_rescaled.commitment.get(), fr_neg_one_mont(), point});
            ++artifact->total_chunks;
            ++bias_attempted;
            try {
                ScopedTimer prove_timer;
                auto proof = formal_linear_relation_prove(
                    chunk_relation_label(
                        spec.dataset + ":output_head" + std::to_string(head) + ":bias_add",
                        chunk_idx,
                        row_begin,
                        row_count,
                        total_chunks),
                    {static_cast<uint>(row_count), static_cast<uint>(cols)},
                    terms,
                    *post.generators,
                    fr_neg_one_mont(),
                    bias_eval,
                    Fr_host_zero());
                artifact->prove_ms += prove_timer.elapsed_ms();
                ScopedTimer verify_timer;
                std::string failure;
                const bool ok = formal_linear_relation_verify(
                    proof,
                    *post.generators,
                    {post.commitment.get(), pre_rescaled.commitment.get()},
                    bias_eval,
                    &failure);
                artifact->verify_ms += verify_timer.elapsed_ms();
                if (ok) {
                    ++artifact->verified_chunks;
                    ++bias_verified;
                    artifact->proof_size_bytes += proof.serialized_size_bytes();
                    artifact->relation_details.push_back(
                        "bias_add chunk=" + std::to_string(chunk_idx)
                        + " head=" + std::to_string(head)
                        + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                        + " status=PASS proof_size_bytes=" + std::to_string(proof.serialized_size_bytes())
                        + " transcript_digest=" + proof.transcript_digest);
                } else {
                    ++bias_rejected;
                    artifact->relation_details.push_back(
                        "bias_add chunk=" + std::to_string(chunk_idx)
                        + " head=" + std::to_string(head)
                        + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                        + " status=REJECT failure=\"" + failure + "\"");
                }
            } catch (const std::exception& error) {
                ++bias_rejected;
                artifact->relation_details.push_back(
                    "bias_add chunk=" + std::to_string(chunk_idx)
                    + " head=" + std::to_string(head)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=FAILED failure=\"" + std::string(error.what()) + "\"");
            }
        }

        const auto logits_chunk = matrix_row_slice(trace.logits, rows, cols, row_begin, row_count);
        const Fr_t public_logits_eval = evaluate_public_tensor_mle(logits_chunk, row_count, cols, padded_cols, point);
        const Fr_t inv_k = fr_inverse_uint_mont(static_cast<std::uint32_t>(model.K_out));
        std::vector<FormalLinearTerm> terms;
        std::vector<const G1TensorJacobian*> commitments;
        for (std::size_t head = 0; head < post_records.size(); ++head) {
            auto* record = post_records[head];
            terms.push_back({"Y_head_" + std::to_string(head), record->tensor.get(), record->commitment.get(), inv_k, point});
            commitments.push_back(record->commitment.get());
        }
        ++artifact->total_chunks;
        ++average_attempted;
        try {
            ScopedTimer prove_timer;
            auto proof = formal_linear_relation_prove(
                chunk_relation_label(
                    spec.dataset + ":output_head_average_public_logits",
                    chunk_idx,
                    row_begin,
                    row_count,
                    total_chunks),
                {static_cast<uint>(row_count), static_cast<uint>(cols)},
                terms,
                *post_records.front()->generators,
                fr_neg_one_mont(),
                public_logits_eval,
                Fr_host_zero());
            artifact->prove_ms += prove_timer.elapsed_ms();
            ScopedTimer verify_timer;
            std::string failure;
            const bool ok = formal_linear_relation_verify(proof, *post_records.front()->generators, commitments, public_logits_eval, &failure);
            artifact->verify_ms += verify_timer.elapsed_ms();
            if (ok) {
                ++artifact->verified_chunks;
                ++average_verified;
                artifact->proof_size_bytes += proof.serialized_size_bytes();
                artifact->relation_details.push_back(
                    "output_head_average_public_logits chunk=" + std::to_string(chunk_idx)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=PASS proof_size_bytes=" + std::to_string(proof.serialized_size_bytes())
                    + " transcript_digest=" + proof.transcript_digest);
            } else {
                ++average_rejected;
                artifact->relation_details.push_back(
                    "output_head_average_public_logits chunk=" + std::to_string(chunk_idx)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=REJECT failure=\"" + failure + "\"");
            }
        } catch (const std::exception& error) {
            ++average_rejected;
            artifact->relation_details.push_back(
                "output_head_average_public_logits chunk=" + std::to_string(chunk_idx)
                + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                + " status=FAILED failure=\"" + std::string(error.what()) + "\"");
        }
    }

    if (average_verified == average_attempted && average_attempted != 0) {
        artifact->covered_relations.push_back(
            "output-head average and public logits binding: chunked LinearRelationProof verifier passed for "
            + std::to_string(average_verified) + "/" + std::to_string(average_attempted) + " full-dataset chunks");
    } else {
        artifact->blocked_relations.push_back(
            "output-head average/public logits binding chunked verifier incomplete: verified "
            + std::to_string(average_verified) + "/" + std::to_string(average_attempted)
            + ", rejected=" + std::to_string(average_rejected));
    }
    if (bias_verified == bias_attempted && bias_attempted != 0) {
        artifact->covered_relations.push_back(
            "output bias add: chunked LinearRelationProof verifier passed for "
            + std::to_string(bias_verified) + "/" + std::to_string(bias_attempted)
            + " chunks over the protocol tensor Y_pre_rescaled");
    } else {
        artifact->blocked_relations.push_back(
            "output bias add chunked verifier incomplete: verified "
            + std::to_string(bias_verified) + "/" + std::to_string(bias_attempted)
            + ", rejected=" + std::to_string(bias_rejected)
            + "; fixed-point rescale/rounding witness is required before marking the real GAT bias relation supported");
    }
}

void run_hidden_concat_chunked_relations(
    const DatasetRunSpec& spec,
    const ForwardTraceHost& trace,
    std::size_t rows,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (trace.hidden_heads.empty() || trace.hidden_heads.front().empty() || trace.output_layer_input.empty()) {
        artifact->blocked_relations.push_back("hidden concat blocked: hidden head trace or output-layer input trace is unavailable");
        return;
    }
    const auto& layer_heads = trace.hidden_heads.front();
    const std::size_t concat_width = trace.output_layer_input_dim;
    std::size_t expected_width = 0;
    for (const auto& head_trace : layer_heads) expected_width += head_trace.width;
    if (expected_width != concat_width || trace.output_layer_input.size() != rows * concat_width) {
        artifact->blocked_relations.push_back("hidden concat blocked: concat trace width does not match hidden head widths");
        return;
    }
    const std::size_t padded_concat_width = next_power_of_two(std::max<std::size_t>(1, concat_width));
    const std::size_t max_chunk_values = 262144;
    const std::size_t max_chunk_rows = 4096;
    const std::size_t chunk_rows = std::max<std::size_t>(1, std::min(max_chunk_rows, std::max<std::size_t>(1, max_chunk_values / padded_concat_width)));
    const std::size_t total_row_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    const std::size_t total_relation_chunks = total_row_chunks * layer_heads.size();
    artifact->total_chunks += total_relation_chunks;
    std::size_t verified = 0;
    std::size_t rejected = 0;
    dbg("formal_hidden_concat rows=" + std::to_string(rows)
        + " concat_width=" + std::to_string(concat_width)
        + " heads=" + std::to_string(layer_heads.size())
        + " chunk_rows=" + std::to_string(chunk_rows)
        + " total_relation_chunks=" + std::to_string(total_relation_chunks));

    for (std::size_t chunk_idx = 0; chunk_idx < total_row_chunks; ++chunk_idx) {
        const std::size_t row_begin = chunk_idx * chunk_rows;
        const std::size_t row_count = std::min(chunk_rows, rows - row_begin);
        FormalTensorRegistry chunk_registry;
        const auto concat_values = matrix_row_slice(trace.output_layer_input, rows, concat_width, row_begin, row_count);
        auto& concat_record = chunk_registry.add_matrix(
            spec.dataset,
            chunk_label("hidden_layer0_H_cat", chunk_idx),
            concat_values,
            row_count,
            concat_width,
            &artifact->commitment_ms,
            debug);
        for (std::size_t head = 0; head < layer_heads.size(); ++head) {
            const auto& head_trace = layer_heads[head];
            const auto head_values = matrix_row_slice(head_trace.h_out, rows, head_trace.width, row_begin, row_count);
            std::vector<double> embedded_head(row_count * padded_concat_width, 0.0);
            const std::size_t head_offset = 0;
            for (std::size_t prior = 0; prior < head; ++prior) {
                // 与列点计算分开，方便未来非均匀 head width 失败时定位清楚。
                if (layer_heads[prior].width != head_trace.width) {
                    throw std::runtime_error("hidden concat proof currently requires uniform hidden head widths");
                }
            }
            const std::size_t column_offset = head * head_trace.width + head_offset;
            for (std::size_t row = 0; row < row_count; ++row) {
                for (std::size_t col = 0; col < head_trace.width; ++col) {
                    embedded_head[row * padded_concat_width + column_offset + col] =
                        head_values[row * head_trace.width + col];
                }
            }
            auto& head_record = chunk_registry.add_encoded_matrix(
                spec.dataset,
                chunk_label("hidden_layer0_head" + std::to_string(head) + "_H_agg_embedded", chunk_idx),
                encode_padded_tensor(embedded_head, row_count, padded_concat_width, padded_concat_width),
                row_count,
                concat_width,
                padded_concat_width,
                &artifact->commitment_ms,
                debug);
            const uint head_col_log = ceilLog2(static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, head_trace.width))));
            const uint row_log = ceilLog2(static_cast<uint>(row_count));
            const auto head_point = deterministic_mont_fr_vec(
                head_col_log + row_log,
                spec.dataset + ":formal:hidden-concat:chunk:" + std::to_string(chunk_idx) + ":head:" + std::to_string(head));
            const std::vector<Fr_t> head_col_point(head_point.begin(), head_point.begin() + head_col_log);
            const std::vector<Fr_t> row_point(head_point.begin() + head_col_log, head_point.end());
            const auto concat_col_point = concat_column_point_for_head(head, head_trace.width, concat_width, head_col_point);
            const auto concat_point = formal_concat_points({concat_col_point, row_point});
            std::vector<FormalLinearTerm> terms;
            terms.push_back({"H_cat_slice", concat_record.tensor.get(), concat_record.commitment.get(), Fr_host_one_mont(), concat_point});
            terms.push_back({"H_agg_head_embedded", head_record.tensor.get(), head_record.commitment.get(), fr_neg_one_mont(), concat_point});
            try {
                ScopedTimer prove_timer;
                auto proof = formal_linear_relation_prove(
                    chunk_relation_label(
                        spec.dataset + ":hidden_layer0_head" + std::to_string(head) + ":concat_copy",
                        chunk_idx,
                        row_begin,
                        row_count,
                        total_row_chunks),
                    {static_cast<uint>(row_count), static_cast<uint>(concat_width)},
                    terms,
                    *concat_record.generators,
                    Fr_host_zero(),
                    Fr_host_zero(),
                    Fr_host_zero());
                artifact->prove_ms += prove_timer.elapsed_ms();
                ScopedTimer verify_timer;
                std::string failure;
                const bool ok = formal_linear_relation_verify(
                    proof,
                    *concat_record.generators,
                    {concat_record.commitment.get(), head_record.commitment.get()},
                    Fr_host_zero(),
                    &failure);
                artifact->verify_ms += verify_timer.elapsed_ms();
                if (!ok) {
                    ++rejected;
                    artifact->relation_details.push_back(
                        "hidden_concat chunk=" + std::to_string(chunk_idx)
                        + " head=" + std::to_string(head)
                        + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                        + " status=REJECT failure=\"" + failure + "\"");
                    artifact->blocked_relations.push_back(
                        "hidden concat chunked verifier rejected chunk " + std::to_string(chunk_idx)
                        + " head " + std::to_string(head) + ": " + failure);
                    return;
                }
                ++verified;
                ++artifact->verified_chunks;
                artifact->proof_size_bytes += proof.serialized_size_bytes();
                artifact->relation_details.push_back(
                    "hidden_concat chunk=" + std::to_string(chunk_idx)
                    + " head=" + std::to_string(head)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=PASS proof_size_bytes=" + std::to_string(proof.serialized_size_bytes())
                    + " transcript_digest=" + proof.transcript_digest);
            } catch (const std::exception& error) {
                ++rejected;
                artifact->relation_details.push_back(
                    "hidden_concat chunk=" + std::to_string(chunk_idx)
                    + " head=" + std::to_string(head)
                    + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                    + " status=FAILED failure=\"" + std::string(error.what()) + "\"");
                artifact->blocked_relations.push_back(
                    "hidden concat chunked prover/verifier failed at chunk " + std::to_string(chunk_idx)
                    + " head " + std::to_string(head) + ": " + std::string(error.what()));
                return;
            }
        }
    }
    if (verified == total_relation_chunks && rejected == 0) {
        artifact->covered_relations.push_back(
            "hidden multi-head concat: chunked LinearRelationProof verifier passed for "
            + std::to_string(verified) + "/" + std::to_string(total_relation_chunks)
            + " head/chunk relations");
    } else {
        artifact->blocked_relations.push_back(
            "hidden concat chunked verifier incomplete: verified "
            + std::to_string(verified) + "/" + std::to_string(total_relation_chunks));
    }
}

bool run_formal_matmul_trace_attempt(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    FormalPartialArtifact* artifact,
    std::ofstream* debug,
    std::size_t row_begin_for_evidence = 0,
    std::size_t total_chunks_for_evidence = 1)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    const std::size_t padded_rows = next_power_of_two(std::max<std::size_t>(1, rows));
    const std::size_t padded_shared = next_power_of_two(std::max<std::size_t>(1, shared));
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    const std::size_t a_size = padded_rows * padded_shared;
    const std::size_t b_size = padded_shared * padded_cols;
    const std::size_t c_size = padded_rows * padded_cols;
    constexpr std::size_t kMaxFlatCommitmentPoints = 50000;
    if (std::max({a_size, b_size, c_size}) > kMaxFlatCommitmentPoints) {
        artifact->blocked_relations.push_back(label + ": blocked because current formal matmul uses flat gen=1 commitments and padded tensor size "
            + std::to_string(std::max({a_size, b_size, c_size})) + " exceeds cap "
            + std::to_string(kMaxFlatCommitmentPoints) + "; row-generator matmul openings are not implemented yet");
        dbg("formal_matmul label=" + label + " status=BLOCKED reason=flat_commitment_cap");
        return false;
    }
    try {
        dbg("formal_matmul label=" + label + " stage=encode");
        auto a_host = formal_encode_matmul_matrix(a_values, rows, shared, padded_rows, padded_shared);
        auto b_host = formal_encode_matmul_matrix(b_values, shared, cols, padded_shared, padded_cols);
        auto c_trace_host = formal_encode_matmul_matrix(c_values, rows, cols, padded_rows, padded_cols);
        FrTensor a(static_cast<uint>(a_host.size()), a_host.data());
        FrTensor b(static_cast<uint>(b_host.size()), b_host.data());
        FrTensor c_trace(static_cast<uint>(c_trace_host.size()), c_trace_host.data());
        a.mont();
        b.mont();
        c_trace.mont();
        auto c_fixed_host = formal_matmul_product_mont(
            a.to_host(),
            rows,
            shared,
            padded_rows,
            padded_shared,
            b.to_host(),
            cols,
            padded_cols);
        FrTensor c(static_cast<uint>(c_fixed_host.size()), c_fixed_host.data());
        Commitment gen(1, G1Jacobian_generator);
        ScopedTimer commit_timer;
        const auto com_a = gen.commit(a);
        const auto com_b = gen.commit(b);
        const auto com_c = gen.commit(c);
        const auto com_c_trace = gen.commit(c_trace);
        cuda_sync("formal matmul commit " + label);
        artifact->commitment_ms += commit_timer.elapsed_ms();
        if (debug != nullptr) {
            const std::string evidence_path = maybe_write_matmul_first_failure_evidence(
                dataset,
                label,
                row_begin_for_evidence,
                rows,
                total_chunks_for_evidence,
                a_values,
                rows,
                shared,
                b_values,
                cols,
                c_values,
                padded_rows,
                padded_shared,
                padded_cols,
                a,
                b,
                c_trace,
                com_a,
                com_b,
                com_c_trace,
                c_fixed_host);
            if (!evidence_path.empty()) {
                artifact->blocked_relations.push_back(label + ": float forward C does not equal the fixed-point pre-rescale matmul witness; rescale/rounding/equality remains blocked; evidence=" + evidence_path);
                dbg("formal_matmul label=" + label + " evidence=\"" + evidence_path + "\" classification=\"float_C_vs_fixed_pre_rescale_mismatch\"");
            }
        }
        ScopedTimer prove_timer;
        auto proof = formal_matmul_relation_prove(a, b, c, gen, com_a, com_b, com_c,
            1U,
            static_cast<uint>(rows),
            static_cast<uint>(shared),
            static_cast<uint>(cols),
            1U,
            static_cast<uint>(padded_rows),
            static_cast<uint>(padded_shared),
            static_cast<uint>(padded_cols),
            dataset + ":" + label);
        artifact->prove_ms += prove_timer.elapsed_ms();
        ScopedTimer verify_timer;
        std::string failure;
        const bool ok = formal_matmul_relation_verify(proof, gen, com_a, com_b, com_c, dataset + ":" + label, &failure);
        artifact->verify_ms += verify_timer.elapsed_ms();
        if (!ok) {
            artifact->blocked_relations.push_back(label + ": verifier rejected formal matmul: " + failure);
            dbg("formal_matmul label=" + label + " status=REJECT failure=\"" + failure + "\"");
            return false;
        }
        artifact->proof_size_bytes += proof.serialized_size_bytes();
        artifact->covered_relations.push_back(label + ": formal fixed-point pre-rescale MatmulRelationProof verifier passed");
        dbg("formal_matmul label=" + label + " status=PASS proof_size_bytes=" + std::to_string(proof.serialized_size_bytes()));
        run_fixed_to_trace_scale_decomposition_relation(
            dataset,
            label,
            a_values,
            rows,
            shared,
            b_values,
            cols,
            c_values,
            padded_rows,
            padded_cols,
            c_fixed_host,
            c_trace,
            com_c,
            com_c_trace,
            gen,
            artifact,
            debug);
        return true;
    } catch (const std::exception& error) {
        artifact->blocked_relations.push_back(label + ": formal matmul attempt failed: " + std::string(error.what()));
        dbg("formal_matmul label=" + label + " status=FAILED reason=\"" + std::string(error.what()) + "\"");
        return false;
    }
}

bool run_formal_matmul_trace_attempt_row_chunked(
    const std::string& dataset,
    const std::string& label,
    const std::vector<double>& a_values,
    std::size_t rows,
    std::size_t shared,
    const std::vector<double>& b_values,
    std::size_t cols,
    const std::vector<double>& c_values,
    FormalPartialArtifact* artifact,
    std::ofstream* debug)
{
    auto dbg = [&](const std::string& line) {
        if (debug != nullptr && *debug) {
            *debug << line << '\n';
            debug->flush();
        }
    };
    if (a_values.size() != rows * shared || b_values.size() != shared * cols || c_values.size() != rows * cols) {
        throw std::runtime_error("chunked matmul trace shape mismatch for " + label);
    }
    constexpr std::size_t kMaxFlatCommitmentPoints = 50000;
    const std::size_t padded_shared = next_power_of_two(std::max<std::size_t>(1, shared));
    const std::size_t padded_cols = next_power_of_two(std::max<std::size_t>(1, cols));
    const std::size_t b_size = padded_shared * padded_cols;
    if (b_size > kMaxFlatCommitmentPoints) {
        artifact->blocked_relations.push_back(label + ": chunked matmul blocked because padded parameter tensor size "
            + std::to_string(b_size) + " exceeds flat commitment cap " + std::to_string(kMaxFlatCommitmentPoints));
        dbg("formal_matmul_chunked label=" + label + " status=BLOCKED reason=parameter_flat_cap");
        return false;
    }
    const std::size_t max_row_factor = std::max<std::size_t>(padded_shared, padded_cols);
    std::size_t max_padded_rows = 1;
    while ((max_padded_rows << 1U) * max_row_factor <= kMaxFlatCommitmentPoints && max_padded_rows < 4096U) {
        max_padded_rows <<= 1U;
    }
    const std::size_t chunk_rows = std::max<std::size_t>(1, max_padded_rows);
    const std::size_t total_chunks = (rows + chunk_rows - 1U) / chunk_rows;
    artifact->total_chunks += total_chunks;
    dbg("formal_matmul_chunked label=" + label
        + " rows=" + std::to_string(rows)
        + " shared=" + std::to_string(shared)
        + " cols=" + std::to_string(cols)
        + " chunk_rows=" + std::to_string(chunk_rows)
        + " total_chunks=" + std::to_string(total_chunks));

    std::size_t verified = 0;
    for (std::size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        const std::size_t row_begin = chunk_idx * chunk_rows;
        const std::size_t row_count = std::min(chunk_rows, rows - row_begin);
        const auto a_chunk = matrix_row_slice(a_values, rows, shared, row_begin, row_count);
        const auto c_chunk = matrix_row_slice(c_values, rows, cols, row_begin, row_count);
        const std::string chunked_label = chunk_relation_label(label, chunk_idx, row_begin, row_count, total_chunks);
        const bool ok = run_formal_matmul_trace_attempt(
            dataset,
            chunked_label,
            a_chunk,
            row_count,
            shared,
            b_values,
            cols,
            c_chunk,
            artifact,
            debug,
            row_begin,
            total_chunks);
        if (!ok) {
            artifact->relation_details.push_back(label
                + " chunk=" + std::to_string(chunk_idx)
                + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
                + " status=REJECT_OR_BLOCKED verified_chunks=" + std::to_string(verified)
                + "/" + std::to_string(total_chunks));
            artifact->blocked_relations.push_back(label + ": chunked MatmulRelationProof incomplete; verified "
                + std::to_string(verified) + "/" + std::to_string(total_chunks)
                + " chunks before first rejected or blocked chunk " + std::to_string(chunk_idx));
            return false;
        }
        ++verified;
        ++artifact->verified_chunks;
        artifact->relation_details.push_back(label
            + " chunk=" + std::to_string(chunk_idx)
            + " rows=" + std::to_string(row_begin) + ".." + std::to_string(row_begin + row_count)
            + " status=PASS");
    }
    artifact->covered_relations.push_back(label + ": chunked MatmulRelationProof verifier passed for "
        + std::to_string(verified) + "/" + std::to_string(total_chunks) + " full-dataset row chunks");
    return true;
}

void serialize_formal_partial_artifact(const FormalPartialArtifact& artifact, const fs::path& path)
{
    ensure_dir(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("failed to write formal partial proof artifact: " + path.string());
    out << "ZKDLGATF\n";
    out << "version=1\n";
    out << "proof_generation_ok=" << (artifact.proof_generation_ok ? "true" : "false") << "\n";
    out << "verification_ok=" << (artifact.verification_ok ? "true" : "false") << "\n";
    out << "commitment_time_ms=" << format_ms(artifact.commitment_ms) << "\n";
    out << "prove_time_ms=" << format_ms(artifact.prove_ms) << "\n";
    out << "verify_time_ms=" << format_ms(artifact.verify_ms) << "\n";
    out << "proof_size_bytes=" << artifact.proof_size_bytes << "\n";
    out << "total_chunks=" << artifact.total_chunks << "\n";
    out << "verified_chunks=" << artifact.verified_chunks << "\n";
    out << "covered_relations=" << artifact.covered_relations.size() << "\n";
    for (const auto& relation : artifact.covered_relations) out << relation << "\n";
    out << "blocked_relations=" << artifact.blocked_relations.size() << "\n";
    for (const auto& relation : artifact.blocked_relations) out << relation << "\n";
    out << "relation_details=" << artifact.relation_details.size() << "\n";
    for (const auto& detail : artifact.relation_details) out << detail << "\n";
    out << "blocker=" << artifact.blocker << "\n";
}
