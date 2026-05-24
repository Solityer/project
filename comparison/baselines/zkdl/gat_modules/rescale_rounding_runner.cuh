#pragma once

// projection 后的 fixed-point rescale / rounding 子证明。
// 该模块只把承诺、opening 与 sumcheck 摘要写入 full proof object；完整 tensor 只在 prover 内存中使用。

struct RescaleRoundingWitness {
    std::vector<Fr_t> pre_product;
    std::vector<Fr_t> rounded_trace;
    std::vector<Fr_t> residual;
    std::vector<Fr_t> shifted_residual;
    std::vector<std::vector<Fr_t>> shifted_bits;
    uint rows = 0;
    uint cols = 0;
    uint padded_rows = 0;
    uint padded_cols = 0;
    uint bit_width = 0;
    std::int64_t residual_offset = 0;
    std::int64_t max_abs_residual = 0;
    std::size_t canonical_rounding_mismatch_count = 0;
};

std::vector<std::int64_t> matrix_scale16_i64(
    const std::vector<double>& values,
    std::size_t rows,
    std::size_t cols,
    std::size_t padded_rows,
    std::size_t padded_cols)
{
    if (values.size() != rows * cols) throw std::runtime_error("scale16 i64 matrix shape mismatch");
    std::vector<std::int64_t> out(padded_rows * padded_cols, 0);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            out[row * padded_cols + col] = quantize_scale16_to_i64(values[row * cols + col]);
        }
    }
    return out;
}

std::int64_t checked_i128_to_i64(__int128 value, const std::string& label)
{
    const __int128 lo = static_cast<__int128>(std::numeric_limits<std::int64_t>::min());
    const __int128 hi = static_cast<__int128>(std::numeric_limits<std::int64_t>::max());
    if (value < lo || value > hi) {
        throw std::runtime_error(label + " 超出 int64 fixed-point 范围");
    }
    return static_cast<std::int64_t>(value);
}

std::int64_t round_half_away_div_i64(std::int64_t numerator, std::int64_t denominator)
{
    if (denominator <= 0) throw std::runtime_error("round denominator 非法");
    if (numerator >= 0) {
        return (numerator + denominator / 2) / denominator;
    }
    const auto magnitude = static_cast<std::uint64_t>(-(numerator + 1)) + 1U;
    return -static_cast<std::int64_t>((magnitude + static_cast<std::uint64_t>(denominator / 2)) / static_cast<std::uint64_t>(denominator));
}

std::vector<std::int64_t> projection_product_i64(
    const std::vector<std::int64_t>& a,
    const std::vector<std::int64_t>& b,
    std::size_t padded_m,
    std::size_t padded_n,
    std::size_t padded_k)
{
    if (a.size() != padded_m * padded_n || b.size() != padded_n * padded_k) {
        throw std::runtime_error("projection product i64 输入形状不一致");
    }
    std::vector<std::int64_t> out(padded_m * padded_k, 0);
    const std::size_t task_count = padded_m * padded_k;
    const std::size_t cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min<std::size_t>(cpu_count, task_count == 0 ? 1 : task_count);
    auto worker = [&](std::size_t begin, std::size_t end) {
        for (std::size_t idx = begin; idx < end; ++idx) {
            const std::size_t row = idx / padded_k;
            const std::size_t col = idx % padded_k;
            __int128 acc = 0;
            for (std::size_t t = 0; t < padded_n; ++t) {
                acc += static_cast<__int128>(a[row * padded_n + t]) * static_cast<__int128>(b[t * padded_k + col]);
            }
            out[idx] = checked_i128_to_i64(acc, "projection pre-rescale product");
        }
    };
    if (worker_count <= 1 || task_count < 4096) {
        worker(0, task_count);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        const std::size_t chunk = (task_count + worker_count - 1) / worker_count;
        for (std::size_t i = 0; i < worker_count; ++i) {
            const std::size_t begin = i * chunk;
            const std::size_t end = std::min(task_count, begin + chunk);
            if (begin >= end) break;
            workers.emplace_back(worker, begin, end);
        }
        for (auto& thread : workers) thread.join();
    }
    return out;
}

std::vector<Fr_t> signed_i64_to_mont_vector(const std::vector<std::int64_t>& values)
{
    std::vector<Fr_t> out(values.size(), Fr_host_zero());
    for (std::size_t i = 0; i < values.size(); ++i) out[i] = fr_from_signed_mont_host(values[i]);
    return out;
}

uint rescale_residual_bit_width(std::int64_t max_abs)
{
    const auto abs_value = static_cast<std::uint64_t>(max_abs < 0 ? 0 : max_abs);
    const auto need = abs_value > (std::numeric_limits<std::uint64_t>::max() - 1ULL) / 2ULL
        ? std::numeric_limits<std::uint64_t>::max()
        : 2ULL * abs_value + 1ULL;
    uint bits = 0;
    std::uint64_t limit = 1;
    while (limit < need && bits < 61U) {
        limit <<= 1U;
        ++bits;
    }
    return std::max<uint>(2U, bits);
}

std::size_t rescale_aligned_extent(std::size_t logical, std::size_t alignment)
{
    if (logical == 0) return alignment;
    const std::size_t align = std::max<std::size_t>(1, alignment);
    const std::size_t rem = logical % align;
    return rem == 0 ? logical : logical + (align - rem);
}

RescaleRoundingWitness build_rescale_rounding_witness(
    const std::vector<double>& input_matrix,
    std::size_t input_dim,
    const HeadParams& head,
    const HeadTraceHost& head_trace,
    const ZkdlGatQuantContract& quant,
    const LocalGraph& graph)
{
    if (input_matrix.size() != graph.num_nodes * input_dim || head.W.size() != input_dim * head.output_dim) {
        throw std::runtime_error("rescale/rounding 输入矩阵或权重形状不一致");
    }
    if (head_trace.h_prime.size() != graph.num_nodes * head.output_dim) {
        throw std::runtime_error("rescale/rounding rounded trace shape mismatch");
    }
    if (quant.scale == 0 || quant.scale > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("rescale/rounding scale 非法");
    }
    RescaleRoundingWitness witness;
    witness.rows = static_cast<uint>(graph.num_nodes);
    witness.cols = static_cast<uint>(head.output_dim);
    witness.padded_rows = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.num_nodes)));
    witness.padded_cols = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, head.output_dim)));

    const auto padded_input_dim = next_power_of_two(std::max<std::size_t>(1, input_dim));
    const auto a = matrix_scale16_i64(input_matrix, graph.num_nodes, input_dim, witness.padded_rows, padded_input_dim);
    const auto b = matrix_scale16_i64(head.W, input_dim, head.output_dim, padded_input_dim, witness.padded_cols);
    const auto pre_i64 = projection_product_i64(a, b, witness.padded_rows, padded_input_dim, witness.padded_cols);
    auto rounded_i64 = matrix_scale16_i64(head_trace.h_prime, graph.num_nodes, head.output_dim, witness.padded_rows, witness.padded_cols);

    std::vector<std::int64_t> residual_i64(pre_i64.size(), 0);
    witness.max_abs_residual = 0;
    const auto scale = static_cast<std::int64_t>(quant.scale);
    for (std::size_t idx = 0; idx < pre_i64.size(); ++idx) {
        const std::size_t row = idx / witness.padded_cols;
        const std::size_t col = idx % witness.padded_cols;
        if (row >= graph.num_nodes || col >= head.output_dim) {
            rounded_i64[idx] = 0;
            continue;
        }
        const auto canonical = round_half_away_div_i64(pre_i64[idx], scale);
        if (canonical != rounded_i64[idx]) ++witness.canonical_rounding_mismatch_count;
        const __int128 residual = static_cast<__int128>(pre_i64[idx]) - static_cast<__int128>(scale) * static_cast<__int128>(rounded_i64[idx]);
        residual_i64[idx] = checked_i128_to_i64(residual, "rescale residual");
        const std::int64_t abs_residual = residual_i64[idx] == std::numeric_limits<std::int64_t>::min()
            ? std::numeric_limits<std::int64_t>::max()
            : static_cast<std::int64_t>(std::llabs(residual_i64[idx]));
        witness.max_abs_residual = std::max(witness.max_abs_residual, abs_residual);
    }

    witness.bit_width = rescale_residual_bit_width(witness.max_abs_residual);
    witness.residual_offset = static_cast<std::int64_t>(1ULL << (witness.bit_width - 1U));
    std::vector<std::int64_t> shifted_i64(pre_i64.size(), 0);
    witness.shifted_bits.assign(witness.bit_width, std::vector<Fr_t>(pre_i64.size(), Fr_host_zero()));
    for (std::size_t idx = 0; idx < pre_i64.size(); ++idx) {
        const std::size_t row = idx / witness.padded_cols;
        const std::size_t col = idx % witness.padded_cols;
        if (row >= graph.num_nodes || col >= head.output_dim) continue;
        const __int128 shifted = static_cast<__int128>(residual_i64[idx]) + static_cast<__int128>(witness.residual_offset);
        if (shifted < 0 || shifted >= (static_cast<__int128>(1) << witness.bit_width)) {
            throw std::runtime_error("rescale residual shifted range 非法");
        }
        shifted_i64[idx] = static_cast<std::int64_t>(shifted);
        const auto shifted_u64 = static_cast<std::uint64_t>(shifted_i64[idx]);
        for (uint bit = 0; bit < witness.bit_width; ++bit) {
            witness.shifted_bits[bit][idx] = ((shifted_u64 >> bit) & 1ULL) != 0ULL
                ? Fr_host_one_mont()
                : Fr_host_zero();
        }
    }
    witness.pre_product = signed_i64_to_mont_vector(pre_i64);
    witness.rounded_trace = signed_i64_to_mont_vector(rounded_i64);
    witness.residual = signed_i64_to_mont_vector(residual_i64);
    witness.shifted_residual = signed_i64_to_mont_vector(shifted_i64);
    return witness;
}

std::size_t rescale_common_commitment_block_size(std::size_t requested, std::size_t size)
{
    if (size == 0) throw std::runtime_error("rescale commitment size 输入为空");
    if (requested == 0) requested = 64;
    const std::size_t capped = std::min(requested, size);
    return std::max<std::size_t>(1, floor_power_of_two_size(capped));
}

ZkdlFullRelationRecord prove_gpu_rescale_rounding_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const std::vector<double>& input_matrix,
    std::size_t input_dim,
    const HeadParams& head,
    const HeadTraceHost& head_trace,
    const ZkdlGatQuantContract& quant,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_rescale_rounding_witness(input_matrix, input_dim, head, head_trace, quant, graph);
    const std::size_t size = witness.pre_product.size();
    const std::size_t block_size = rescale_common_commitment_block_size(requested_commitment_block_size, size);

    ZkdlFullRelationRecord record;
    record.stage = "rescale_rounding";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":projection_rescale_rounding";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    Commitment gen = make_formal_commitment_generators(static_cast<uint>(block_size));
    FrTensor pre_tensor(static_cast<uint>(size), witness.pre_product.data());
    FrTensor rounded_tensor(static_cast<uint>(size), witness.rounded_trace.data());
    FrTensor residual_tensor(static_cast<uint>(size), witness.residual.data());
    FrTensor shifted_tensor(static_cast<uint>(size), witness.shifted_residual.data());
    std::vector<std::unique_ptr<FrTensor>> bit_tensors;
    bit_tensors.reserve(witness.bit_width);
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        bit_tensors.emplace_back(std::make_unique<FrTensor>(static_cast<uint>(size), witness.shifted_bits[bit].data()));
    }

    ScopedTimer commit_timer;
    const auto com_pre = gen.commit(pre_tensor);
    const auto com_rounded = gen.commit(rounded_tensor);
    const auto com_residual = gen.commit(residual_tensor);
    const auto com_shifted = gen.commit(shifted_tensor);
    std::vector<std::unique_ptr<G1TensorJacobian>> bit_commitments;
    bit_commitments.reserve(witness.bit_width);
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        bit_commitments.emplace_back(std::make_unique<G1TensorJacobian>(gen.commit(*bit_tensors[bit])));
    }
    cuda_sync("rescale/rounding commit " + record.relation_scope);
    record.commitment_ms += commit_timer.elapsed_ms();

    const auto point = deterministic_mont_fr_vec(ceilLog2(static_cast<uint>(size)), record.relation_scope + ":point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    const auto neg_scale = fr_from_signed_mont_host(-static_cast<std::int64_t>(quant.scale));
    const std::vector<uint> shape = {witness.padded_rows, witness.padded_cols};

    LinearRelationProof identity;
    {
        std::vector<FormalLinearTerm> terms = {
            {"pre_product", &pre_tensor, &com_pre, one, point},
            {"rounded_trace", &rounded_tensor, &com_rounded, neg_scale, point},
            {"residual", &residual_tensor, &com_residual, neg_one, point},
        };
        ScopedTimer timer;
        identity = formal_linear_relation_prove(
            record.relation_scope + ":identity", shape, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        std::vector<const G1TensorJacobian*> commitments = {&com_pre, &com_rounded, &com_residual};
        ScopedTimer timer;
        std::string failure;
        if (!formal_linear_relation_verify(identity, gen, commitments, Fr_host_zero(), &failure)) {
            throw std::runtime_error("GPU rescale identity verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }

    LinearRelationProof shifted;
    const auto valid_eval = evaluate_valid_matrix_selector_mle(
        witness.rows, witness.cols, witness.padded_rows, witness.padded_cols, point);
    {
        std::vector<FormalLinearTerm> terms = {
            {"residual", &residual_tensor, &com_residual, one, point},
            {"shifted_residual", &shifted_tensor, &com_shifted, neg_one, point},
        };
        ScopedTimer timer;
        shifted = formal_linear_relation_prove(
            record.relation_scope + ":shifted_residual", shape, terms, gen,
            fr_from_signed_mont_host(witness.residual_offset), valid_eval, Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        std::vector<const G1TensorJacobian*> commitments = {&com_residual, &com_shifted};
        ScopedTimer timer;
        std::string failure;
        if (!formal_linear_relation_verify(shifted, gen, commitments, valid_eval, &failure)) {
            throw std::runtime_error("GPU rescale shifted verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }

    LinearRelationProof decomposition;
    std::vector<const G1TensorJacobian*> decomposition_commitments;
    {
        std::vector<FormalLinearTerm> terms;
        terms.push_back({"shifted_residual", &shifted_tensor, &com_shifted, one, point});
        decomposition_commitments.push_back(&com_shifted);
        for (uint bit = 0; bit < witness.bit_width; ++bit) {
            terms.push_back({
                "shifted_bit" + std::to_string(bit),
                bit_tensors[bit].get(),
                bit_commitments[bit].get(),
                fr_from_signed_mont_host(-static_cast<std::int64_t>(1ULL << bit)),
                point,
            });
            decomposition_commitments.push_back(bit_commitments[bit].get());
        }
        ScopedTimer timer;
        decomposition = formal_linear_relation_prove(
            record.relation_scope + ":shifted_bit_decomposition", shape, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer verify_timer;
        std::string failure;
        if (!formal_linear_relation_verify(decomposition, gen, decomposition_commitments, Fr_host_zero(), &failure)) {
            throw std::runtime_error("GPU rescale bit decomposition verifier rejected: " + failure);
        }
        record.verify_ms += verify_timer.elapsed_ms();
    }

    std::uintmax_t bitness_size = 0;
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        const auto label = record.relation_scope + ":shifted_bit" + std::to_string(bit) + ":boolean";
        HadamardSumcheckProof bitness;
        {
            ScopedTimer timer;
            bitness = formal_hadamard_relation_prove(
                *bit_tensors[bit], *bit_tensors[bit], *bit_tensors[bit],
                gen, *bit_commitments[bit], *bit_commitments[bit], *bit_commitments[bit], label);
            record.prove_ms += timer.elapsed_ms();
        }
        {
            ScopedTimer timer;
            std::string failure;
            if (!formal_hadamard_relation_verify(
                    bitness, gen, *bit_commitments[bit], *bit_commitments[bit], *bit_commitments[bit], label, &failure)) {
                throw std::runtime_error("GPU rescale bitness verifier rejected: " + failure);
            }
            record.verify_ms += timer.elapsed_ms();
        }
        bitness_size += bitness.serialized_size_bytes();
        record.proof_digests.push_back("bitness" + std::to_string(bit) + "=" + bitness.transcript_digest);
    }

    record.proof_size_bytes = identity.serialized_size_bytes()
        + shifted.serialized_size_bytes()
        + decomposition.serialized_size_bytes()
        + bitness_size;
    record.proof_digests.push_back("identity=" + identity.transcript_digest);
    record.proof_digests.push_back("shifted=" + shifted.transcript_digest);
    record.proof_digests.push_back("decomposition=" + decomposition.transcript_digest);
    record.proof_digests.push_back("quant_contract_id=" + quant.contract_id);
    record.proof_digests.push_back("scale=" + std::to_string(quant.scale));
    record.proof_digests.push_back("rem_bits=" + std::to_string(witness.bit_width));
    record.proof_digests.push_back("tie_policy=" + quant.rounding_policy);
    record.proof_digests.push_back("max_abs_residual=" + std::to_string(witness.max_abs_residual));
    record.proof_digests.push_back("canonical_rounding_mismatch_count=" + std::to_string(witness.canonical_rounding_mismatch_count));
    record.proof_generated = true;
    record.verified = true;
    return record;
}

ZkdlFullRelationRecord prove_cpu_rescale_rounding_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const std::vector<double>& input_matrix,
    std::size_t input_dim,
    const HeadParams& head,
    const HeadTraceHost& head_trace,
    const ZkdlGatQuantContract& quant,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    auto witness = build_rescale_rounding_witness(input_matrix, input_dim, head, head_trace, quant, graph);
    const std::size_t size = witness.pre_product.size();
    const std::size_t block_size = rescale_common_commitment_block_size(requested_commitment_block_size, size);

    ZkdlFullRelationRecord record;
    record.stage = "cpu_rescale_rounding";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":projection_rescale_rounding";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    CpuCommitment gen = CpuCommitment::deterministic(
        static_cast<uint>(block_size),
        spec.dataset + ":cpu_rescale_block" + std::to_string(block_size));
    CpuFrTensor pre_tensor(witness.pre_product);
    CpuFrTensor rounded_tensor(witness.rounded_trace);
    CpuFrTensor residual_tensor(witness.residual);
    CpuFrTensor shifted_tensor(witness.shifted_residual);
    std::vector<std::unique_ptr<CpuFrTensor>> bit_tensors;
    bit_tensors.reserve(witness.bit_width);
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        bit_tensors.emplace_back(std::make_unique<CpuFrTensor>(witness.shifted_bits[bit]));
    }

    ScopedTimer commit_timer;
    const auto com_pre = gen.commit(pre_tensor);
    const auto com_rounded = gen.commit(rounded_tensor);
    const auto com_residual = gen.commit(residual_tensor);
    const auto com_shifted = gen.commit(shifted_tensor);
    std::vector<std::unique_ptr<CpuG1Tensor>> bit_commitments;
    bit_commitments.reserve(witness.bit_width);
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        bit_commitments.emplace_back(std::make_unique<CpuG1Tensor>(gen.commit(*bit_tensors[bit])));
    }
    record.commitment_ms += commit_timer.elapsed_ms();

    const auto point = cpu_deterministic_mont_fr_vec(ceilLog2(static_cast<uint>(size)), record.relation_scope + ":point");
    const auto one = Fr_host_one_mont();
    const auto neg_one = fr_neg_one_mont();
    const auto neg_scale = fr_from_signed_mont_host(-static_cast<std::int64_t>(quant.scale));
    const std::vector<uint> shape = {witness.padded_rows, witness.padded_cols};

    LinearRelationProof identity;
    {
        std::vector<CpuLinearTerm> terms = {
            {"pre_product", &pre_tensor, &com_pre, one, point},
            {"rounded_trace", &rounded_tensor, &com_rounded, neg_scale, point},
            {"residual", &residual_tensor, &com_residual, neg_one, point},
        };
        ScopedTimer timer;
        identity = cpu_linear_relation_prove(
            record.relation_scope + ":identity", shape, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        std::vector<const CpuG1Tensor*> commitments = {&com_pre, &com_rounded, &com_residual};
        ScopedTimer timer;
        std::string failure;
        if (!cpu_linear_relation_verify(identity, gen, commitments, Fr_host_zero(), &failure)) {
            throw std::runtime_error("CPU rescale identity verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }

    LinearRelationProof shifted;
    const auto valid_eval = evaluate_valid_matrix_selector_mle(
        witness.rows, witness.cols, witness.padded_rows, witness.padded_cols, point);
    {
        std::vector<CpuLinearTerm> terms = {
            {"residual", &residual_tensor, &com_residual, one, point},
            {"shifted_residual", &shifted_tensor, &com_shifted, neg_one, point},
        };
        ScopedTimer timer;
        shifted = cpu_linear_relation_prove(
            record.relation_scope + ":shifted_residual", shape, terms, gen,
            fr_from_signed_mont_host(witness.residual_offset), valid_eval, Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        std::vector<const CpuG1Tensor*> commitments = {&com_residual, &com_shifted};
        ScopedTimer timer;
        std::string failure;
        if (!cpu_linear_relation_verify(shifted, gen, commitments, valid_eval, &failure)) {
            throw std::runtime_error("CPU rescale shifted verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }

    LinearRelationProof decomposition;
    std::vector<const CpuG1Tensor*> decomposition_commitments;
    {
        std::vector<CpuLinearTerm> terms;
        terms.push_back({"shifted_residual", &shifted_tensor, &com_shifted, one, point});
        decomposition_commitments.push_back(&com_shifted);
        for (uint bit = 0; bit < witness.bit_width; ++bit) {
            terms.push_back({
                "shifted_bit" + std::to_string(bit),
                bit_tensors[bit].get(),
                bit_commitments[bit].get(),
                fr_from_signed_mont_host(-static_cast<std::int64_t>(1ULL << bit)),
                point,
            });
            decomposition_commitments.push_back(bit_commitments[bit].get());
        }
        ScopedTimer timer;
        decomposition = cpu_linear_relation_prove(
            record.relation_scope + ":shifted_bit_decomposition", shape, terms, gen,
            Fr_host_zero(), Fr_host_zero(), Fr_host_zero());
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer verify_timer;
        std::string failure;
        if (!cpu_linear_relation_verify(decomposition, gen, decomposition_commitments, Fr_host_zero(), &failure)) {
            throw std::runtime_error("CPU rescale bit decomposition verifier rejected: " + failure);
        }
        record.verify_ms += verify_timer.elapsed_ms();
    }

    std::uintmax_t bitness_size = 0;
    for (uint bit = 0; bit < witness.bit_width; ++bit) {
        const auto label = record.relation_scope + ":shifted_bit" + std::to_string(bit) + ":boolean";
        HadamardSumcheckProof bitness;
        {
            ScopedTimer timer;
            bitness = cpu_hadamard_relation_prove(
                *bit_tensors[bit], *bit_tensors[bit], *bit_tensors[bit],
                gen, *bit_commitments[bit], *bit_commitments[bit], *bit_commitments[bit], label);
            record.prove_ms += timer.elapsed_ms();
        }
        {
            ScopedTimer timer;
            std::string failure;
            if (!cpu_hadamard_relation_verify(
                    bitness, gen, *bit_commitments[bit], *bit_commitments[bit], *bit_commitments[bit], label, &failure)) {
                throw std::runtime_error("CPU rescale bitness verifier rejected: " + failure);
            }
            record.verify_ms += timer.elapsed_ms();
        }
        bitness_size += bitness.serialized_size_bytes();
        record.proof_digests.push_back("bitness" + std::to_string(bit) + "=" + bitness.transcript_digest);
    }

    record.proof_size_bytes = identity.serialized_size_bytes()
        + shifted.serialized_size_bytes()
        + decomposition.serialized_size_bytes()
        + bitness_size;
    record.proof_digests.push_back("identity=" + identity.transcript_digest);
    record.proof_digests.push_back("shifted=" + shifted.transcript_digest);
    record.proof_digests.push_back("decomposition=" + decomposition.transcript_digest);
    record.proof_digests.push_back("quant_contract_id=" + quant.contract_id);
    record.proof_digests.push_back("scale=" + std::to_string(quant.scale));
    record.proof_digests.push_back("rem_bits=" + std::to_string(witness.bit_width));
    record.proof_digests.push_back("tie_policy=" + quant.rounding_policy);
    record.proof_digests.push_back("max_abs_residual=" + std::to_string(witness.max_abs_residual));
    record.proof_digests.push_back("canonical_rounding_mismatch_count=" + std::to_string(witness.canonical_rounding_mismatch_count));
    record.proof_generated = true;
    record.verified = true;
    return record;
}
