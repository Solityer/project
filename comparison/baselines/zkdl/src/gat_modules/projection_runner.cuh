#pragma once

// 选择投影矩阵乘法 proof 的 commitment block；影响 opening proof_size_bytes。
std::size_t projection_common_commitment_block_size(
    std::size_t requested,
    std::size_t a_size,
    std::size_t b_size,
    std::size_t c_size)
{
    if (a_size == 0 || b_size == 0 || c_size == 0) {
        throw std::runtime_error("projection commitment block size 输入为空");
    }
    if (requested == 0) requested = 64;
    const std::size_t divisor = std::gcd(std::gcd(a_size, b_size), c_size);
    const std::size_t capped = std::min(requested, divisor);
    return std::max<std::size_t>(1, floor_power_of_two_size(capped));
}

// 计算投影矩阵乘法的 scale16 Fr witness；只在 prover 侧构造，不写入 proof object。
// 计算 H*W 的 Montgomery witness；输出写入 projection matmul relation proof。
std::vector<Fr_t> projection_product_mont(
    const std::vector<Fr_t>& a_mont,
    const std::vector<Fr_t>& b_mont,
    uint padded_num,
    uint padded_m,
    uint padded_n,
    uint padded_k)
{
    if (a_mont.size() != static_cast<std::size_t>(padded_num) * padded_m * padded_n
        || b_mont.size() != static_cast<std::size_t>(padded_num) * padded_n * padded_k) {
        throw std::runtime_error("projection matmul 输入形状不一致");
    }
    std::vector<Fr_t> c_mont(static_cast<std::size_t>(padded_num) * padded_m * padded_k, Fr_host_zero());
    const std::size_t task_count = static_cast<std::size_t>(padded_num) * padded_m * padded_k;
    const std::size_t cpu_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min<std::size_t>(cpu_count, task_count == 0 ? 1 : task_count);
    auto worker = [&](std::size_t begin, std::size_t end) {
        for (std::size_t idx = begin; idx < end; ++idx) {
            const uint col = static_cast<uint>(idx % padded_k);
            const uint row = static_cast<uint>((idx / padded_k) % padded_m);
            const uint batch = static_cast<uint>(idx / (static_cast<std::size_t>(padded_m) * padded_k));
            Fr_t acc = Fr_host_zero();
            for (uint t = 0; t < padded_n; ++t) {
                const auto av = a_mont[(static_cast<std::size_t>(batch) * padded_m + row) * padded_n + t];
                const auto bv = b_mont[(static_cast<std::size_t>(batch) * padded_n + t) * padded_k + col];
                acc = Fr_host_add(acc, Fr_host_mul(av, bv));
            }
            c_mont[idx] = acc;
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
    return c_mont;
}

// CPU matmul relation transcript 元数据；绑定形状和 A/B/C commitment。
void cpu_absorb_matmul_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    uint num,
    uint m,
    uint n,
    uint k,
    uint padded_num,
    uint padded_m,
    uint padded_n,
    uint padded_k,
    const CpuG1Tensor& com_a,
    const CpuG1Tensor& com_b,
    const CpuG1Tensor& com_c)
{
    transcript.absorb_label("matmul_relation_cpu");
    transcript.absorb_label(label);
    transcript.absorb_u64(num);
    transcript.absorb_u64(m);
    transcript.absorb_u64(n);
    transcript.absorb_u64(k);
    transcript.absorb_u64(padded_num);
    transcript.absorb_u64(padded_m);
    transcript.absorb_u64(padded_n);
    transcript.absorb_u64(padded_k);
    cpu_absorb_g1_tensor(transcript, com_a);
    cpu_absorb_g1_tensor(transcript, com_b);
    cpu_absorb_g1_tensor(transcript, com_c);
}

// CPU matmul prover 使用与 GPU full-chain 相同的 MLE 关系，避免 CPU 结果借用 GPU artifact。
MatmulRelationProof cpu_matmul_relation_prove(
    const CpuFrTensor& a,
    const CpuFrTensor& b,
    const CpuFrTensor& c,
    const CpuCommitment& generators,
    const CpuG1Tensor& com_a,
    const CpuG1Tensor& com_b,
    const CpuG1Tensor& com_c,
    uint num,
    uint m,
    uint n,
    uint k,
    uint padded_num,
    uint padded_m,
    uint padded_n,
    uint padded_k,
    const std::string& relation_label)
{
    if (a.values.size() != static_cast<std::size_t>(padded_num) * padded_m * padded_n
        || b.values.size() != static_cast<std::size_t>(padded_num) * padded_n * padded_k
        || c.values.size() != static_cast<std::size_t>(padded_num) * padded_m * padded_k) {
        throw std::runtime_error("CPU projection matmul padded tensor size mismatch");
    }
    MatmulRelationProof proof;
    proof.label = relation_label;
    proof.num = num;
    proof.m = m;
    proof.n = n;
    proof.k = k;
    proof.padded_num = padded_num;
    proof.padded_m = padded_m;
    proof.padded_n = padded_n;
    proof.padded_k = padded_k;
    proof.commitment_a_label = relation_label + ":A";
    proof.commitment_b_label = relation_label + ":B";
    proof.commitment_c_label = relation_label + ":C";

    FormalTranscript transcript;
    cpu_absorb_matmul_metadata(transcript, relation_label, num, m, n, k,
        padded_num, padded_m, padded_n, padded_k, com_a, com_b, com_c);
    proof.point_num = formal_make_challenge_point(transcript, ceilLog2(padded_num), "matmul_num_point");
    proof.point_m = formal_make_challenge_point(transcript, ceilLog2(padded_m), "matmul_m_point");
    proof.point_k = formal_make_challenge_point(transcript, ceilLog2(padded_k), "matmul_k_point");
    proof.point_c = formal_concat_points({proof.point_k, proof.point_m, proof.point_num});
    proof.eval_c = generators.open_with_proof(c, com_c, proof.point_c, proof.opening_c);
    transcript.absorb_fr(proof.eval_c);

    std::vector<Fr_t> reduced_a = formal_matmul_reduce_a_over_m(a.values, padded_num, padded_m, padded_n, proof.point_m);
    std::vector<Fr_t> reduced_b = formal_matmul_reduce_b_over_k(b.values, padded_num, padded_n, padded_k, proof.point_k);
    uint cur_num = padded_num;
    Fr_t running_claim = proof.eval_c;
    for (uint round_idx = 0; round_idx < proof.point_num.size(); ++round_idx) {
        const uint out_num = (cur_num + 1U) / 2U;
        std::vector<Fr_t> coeff0(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        std::vector<Fr_t> coeff1(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        std::vector<Fr_t> coeff2(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        for (uint batch = 0; batch < out_num; ++batch) {
            for (uint t = 0; t < padded_n; ++t) {
                const uint low_batch = 2U * batch;
                const uint high_batch = low_batch + 1U;
                const Fr_t a0 = low_batch < cur_num ? reduced_a[static_cast<std::size_t>(low_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t b0 = low_batch < cur_num ? reduced_b[static_cast<std::size_t>(low_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t a1 = high_batch < cur_num ? reduced_a[static_cast<std::size_t>(high_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t b1 = high_batch < cur_num ? reduced_b[static_cast<std::size_t>(high_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t da = Fr_host_sub(a1, a0);
                const Fr_t db = Fr_host_sub(b1, b0);
                const std::size_t idx = static_cast<std::size_t>(batch) * padded_n + t;
                coeff0[idx] = Fr_host_mul(a0, b0);
                coeff1[idx] = Fr_host_add(Fr_host_mul(a0, db), Fr_host_mul(b0, da));
                coeff2[idx] = Fr_host_mul(da, db);
            }
        }
        SumcheckQuadraticRound round;
        round.c0 = formal_host_sum(formal_partial_mle_by_window_host(
            coeff0, proof.point_num.begin() + round_idx + 1, proof.point_num.end(), padded_n));
        round.c1 = formal_host_sum(formal_partial_mle_by_window_host(
            coeff1, proof.point_num.begin() + round_idx + 1, proof.point_num.end(), padded_n));
        round.c2 = formal_host_sum(formal_partial_mle_by_window_host(
            coeff2, proof.point_num.begin() + round_idx + 1, proof.point_num.end(), padded_n));
        if (!Fr_host_eq(running_claim, Fr_host_weighted_quadratic_endpoints(round, proof.point_num[round_idx]))) {
            throw std::runtime_error("CPU projection matmul prover batch round 不一致");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("matmul_batch_fold");

        std::vector<Fr_t> next_a(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        std::vector<Fr_t> next_b(static_cast<std::size_t>(out_num) * padded_n, Fr_host_zero());
        for (uint batch = 0; batch < out_num; ++batch) {
            for (uint t = 0; t < padded_n; ++t) {
                const uint low_batch = 2U * batch;
                const uint high_batch = low_batch + 1U;
                const Fr_t a0 = low_batch < cur_num ? reduced_a[static_cast<std::size_t>(low_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t b0 = low_batch < cur_num ? reduced_b[static_cast<std::size_t>(low_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t a1 = high_batch < cur_num ? reduced_a[static_cast<std::size_t>(high_batch) * padded_n + t] : Fr_host_zero();
                const Fr_t b1 = high_batch < cur_num ? reduced_b[static_cast<std::size_t>(high_batch) * padded_n + t] : Fr_host_zero();
                const std::size_t idx = static_cast<std::size_t>(batch) * padded_n + t;
                next_a[idx] = Fr_host_mle_step(a0, a1, challenge);
                next_b[idx] = Fr_host_mle_step(b0, b1, challenge);
            }
        }
        running_claim = Fr_host_quadratic_eval(round, challenge);
        proof.batch_rounds.push_back(round);
        proof.batch_challenges.push_back(challenge);
        reduced_a.swap(next_a);
        reduced_b.swap(next_b);
        cur_num = out_num;
    }
    if (reduced_a.size() != padded_n || reduced_b.size() != padded_n) {
        throw std::runtime_error("CPU projection matmul batch reduction 未落到 inner dimension");
    }

    proof.inner_product = cpu_inner_product_sumcheck_prove(reduced_a, reduced_b, relation_label + ":inner");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, running_claim)) {
        throw std::runtime_error("CPU projection matmul inner-product claim mismatch");
    }
    proof.reduced_inner_product_claim = running_claim;
    const auto point_a = formal_concat_points({proof.inner_product.challenges, proof.point_m, proof.batch_challenges});
    const auto point_b = formal_concat_points({proof.point_k, proof.inner_product.challenges, proof.batch_challenges});
    proof.inner_product.final_a = generators.open_with_proof(a, com_a, point_a, proof.opening_a);
    proof.inner_product.final_b = generators.open_with_proof(b, com_b, point_b, proof.opening_b);
    proof.inner_product.opening_a = proof.opening_a;
    proof.inner_product.opening_b = proof.opening_b;
    proof.openings = {proof.opening_a, proof.opening_b, proof.opening_c};
    transcript.absorb_label(proof.inner_product.transcript_digest);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

// CPU matmul verifier 重放 sumcheck/opening；失败会阻止对应 relation record 写入 verified=true。
// CPU 验证矩阵乘法 relation proof；只使用承诺 opening，不读取 witness。
bool cpu_matmul_relation_verify(
    const MatmulRelationProof& proof,
    const CpuCommitment& generators,
    const CpuG1Tensor& com_a,
    const CpuG1Tensor& com_b,
    const CpuG1Tensor& com_c,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("CPU projection matmul label mismatch");
    if (proof.padded_num == 0 || proof.padded_m == 0 || proof.padded_n == 0 || proof.padded_k == 0) {
        return fail("CPU projection matmul padded shape 非法");
    }
    if (proof.point_num.size() != ceilLog2(proof.padded_num)
        || proof.point_m.size() != ceilLog2(proof.padded_m)
        || proof.point_k.size() != ceilLog2(proof.padded_k)
        || proof.batch_rounds.size() != proof.point_num.size()
        || proof.batch_challenges.size() != proof.point_num.size()) {
        return fail("CPU projection matmul point/round 数量不一致");
    }

    FormalTranscript transcript;
    cpu_absorb_matmul_metadata(transcript, relation_label, proof.num, proof.m, proof.n, proof.k,
        proof.padded_num, proof.padded_m, proof.padded_n, proof.padded_k, com_a, com_b, com_c);
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.padded_num), "matmul_num_point"), proof.point_num)) {
        return fail("CPU projection matmul num point replay mismatch");
    }
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.padded_m), "matmul_m_point"), proof.point_m)) {
        return fail("CPU projection matmul m point replay mismatch");
    }
    if (!formal_points_equal(formal_make_challenge_point(transcript, ceilLog2(proof.padded_k), "matmul_k_point"), proof.point_k)) {
        return fail("CPU projection matmul k point replay mismatch");
    }
    const auto expected_c_point = formal_concat_points({proof.point_k, proof.point_m, proof.point_num});
    if (!formal_points_equal(expected_c_point, proof.point_c)) return fail("CPU projection matmul C point mismatch");
    if (!formal_points_equal(proof.opening_c.point, proof.point_c)) return fail("CPU projection matmul C opening point mismatch");
    if (!Fr_host_eq(proof.opening_c.value, proof.eval_c)) return fail("CPU projection matmul C opening value mismatch");
    if (!CpuCommitment::verify_opening(generators, com_c, proof.opening_c)) return fail("CPU projection matmul C opening rejected");
    transcript.absorb_fr(proof.eval_c);

    Fr_t running_claim = proof.eval_c;
    for (uint round_idx = 0; round_idx < proof.batch_rounds.size(); ++round_idx) {
        const auto& round = proof.batch_rounds[round_idx];
        if (!Fr_host_eq(running_claim, Fr_host_weighted_quadratic_endpoints(round, proof.point_num[round_idx]))) {
            return fail("CPU projection matmul batch weighted endpoint mismatch");
        }
        transcript.absorb_u64(round_idx);
        transcript.absorb_fr(round.c0);
        transcript.absorb_fr(round.c1);
        transcript.absorb_fr(round.c2);
        const Fr_t challenge = transcript.challenge("matmul_batch_fold");
        if (!Fr_host_eq(challenge, proof.batch_challenges[round_idx])) return fail("CPU projection matmul batch challenge mismatch");
        running_claim = Fr_host_quadratic_eval(round, challenge);
    }
    if (!Fr_host_eq(proof.reduced_inner_product_claim, running_claim)) return fail("CPU projection matmul reduced claim mismatch");
    if (!Fr_host_eq(proof.inner_product.claimed_sum, running_claim)) return fail("CPU projection matmul inner claimed sum mismatch");
    std::string inner_failure;
    if (!formal_inner_product_sumcheck_verify(proof.inner_product, relation_label + ":inner", &inner_failure)) {
        return fail("CPU projection matmul inner sumcheck failed: " + inner_failure);
    }
    const auto expected_a_point = formal_concat_points({proof.inner_product.challenges, proof.point_m, proof.batch_challenges});
    const auto expected_b_point = formal_concat_points({proof.point_k, proof.inner_product.challenges, proof.batch_challenges});
    if (!formal_points_equal(proof.opening_a.point, expected_a_point)) return fail("CPU projection matmul A opening point mismatch");
    if (!formal_points_equal(proof.opening_b.point, expected_b_point)) return fail("CPU projection matmul B opening point mismatch");
    if (!Fr_host_eq(proof.opening_a.value, proof.inner_product.final_a)
        || !Fr_host_eq(proof.opening_b.value, proof.inner_product.final_b)) {
        return fail("CPU projection matmul terminal opening value mismatch");
    }
    if (!CpuCommitment::verify_opening(generators, com_a, proof.opening_a)) return fail("CPU projection matmul A opening rejected");
    if (!CpuCommitment::verify_opening(generators, com_b, proof.opening_b)) return fail("CPU projection matmul B opening rejected");
    transcript.absorb_label(proof.inner_product.transcript_digest);
    if (proof.transcript_digest != transcript.digest_hex()) return fail("CPU projection matmul transcript digest mismatch");
    return true;
}

// GPU full-chain projection relation：承诺输入、权重和投影输出，并记录 matmul proof 摘要。
// GPU 路径生成 hidden/output 投影 matmul record；写入 full proof manifest 的 relation。
ZkdlFullRelationRecord prove_gpu_projection_matmul_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const std::vector<double>& input_matrix,
    std::size_t input_dim,
    const HeadParams& head,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    if (input_matrix.size() != graph.num_nodes * input_dim || head.W.size() != input_dim * head.output_dim) {
        throw std::runtime_error("GPU projection matmul 输入/权重形状不一致");
    }
    const uint num = 1;
    const uint m = static_cast<uint>(graph.num_nodes);
    const uint n = static_cast<uint>(input_dim);
    const uint k = static_cast<uint>(head.output_dim);
    const uint padded_num = 1;
    const uint padded_m = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.num_nodes)));
    const uint padded_n = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, input_dim)));
    const uint padded_k = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, head.output_dim)));

    auto a_mont = mont_vector_from_raw(matrix_scale16_raw(input_matrix, graph.num_nodes, input_dim, padded_m, padded_n));
    auto b_mont = mont_vector_from_raw(matrix_scale16_raw(head.W, input_dim, head.output_dim, padded_n, padded_k));
    auto c_mont = projection_product_mont(a_mont, b_mont, padded_num, padded_m, padded_n, padded_k);
    const std::size_t block_size = projection_common_commitment_block_size(
        requested_commitment_block_size, a_mont.size(), b_mont.size(), c_mont.size());

    ZkdlFullRelationRecord record;
    record.stage = "projection_matmul";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":pre_rescale_product";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    Commitment gen = make_formal_commitment_generators(block_size);
    FrTensor a_tensor(static_cast<uint>(a_mont.size()), a_mont.data());
    FrTensor b_tensor(static_cast<uint>(b_mont.size()), b_mont.data());
    FrTensor c_tensor(static_cast<uint>(c_mont.size()), c_mont.data());
    ScopedTimer commit_timer;
    const auto com_a = gen.commit(a_tensor);
    const auto com_b = gen.commit(b_tensor);
    const auto com_c = gen.commit(c_tensor);
    cuda_sync("projection matmul commit " + record.relation_scope);
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string label = record.relation_scope + ":matmul";
    MatmulRelationProof proof;
    {
        ScopedTimer timer;
        proof = formal_matmul_relation_prove(
            a_tensor, b_tensor, c_tensor, gen, com_a, com_b, com_c,
            num, m, n, k, padded_num, padded_m, padded_n, padded_k, label);
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!formal_matmul_relation_verify(proof, gen, com_a, com_b, com_c, label, &failure)) {
            throw std::runtime_error("GPU projection matmul verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = proof.serialized_size_bytes();
    record.proof_digests.push_back("matmul=" + proof.transcript_digest);
    record.proof_generated = true;
    record.verified = true;
    return record;
}

// CPU full-chain projection relation：语义与 GPU 相同，后端换成 CPU commitment/opening。
// CPU 路径生成 projection matmul record；CPU/GPU artifact 目录隔离。
ZkdlFullRelationRecord prove_cpu_projection_matmul_full_record(
    const DatasetRunSpec& spec,
    const LocalGraph& graph,
    const std::vector<double>& input_matrix,
    std::size_t input_dim,
    const HeadParams& head,
    const std::string& layer_kind,
    std::size_t layer_index,
    std::size_t head_index,
    std::size_t requested_commitment_block_size)
{
    if (input_matrix.size() != graph.num_nodes * input_dim || head.W.size() != input_dim * head.output_dim) {
        throw std::runtime_error("CPU projection matmul 输入/权重形状不一致");
    }
    const uint num = 1;
    const uint m = static_cast<uint>(graph.num_nodes);
    const uint n = static_cast<uint>(input_dim);
    const uint k = static_cast<uint>(head.output_dim);
    const uint padded_num = 1;
    const uint padded_m = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, graph.num_nodes)));
    const uint padded_n = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, input_dim)));
    const uint padded_k = static_cast<uint>(next_power_of_two(std::max<std::size_t>(1, head.output_dim)));

    auto a_mont = mont_vector_from_raw(matrix_scale16_raw(input_matrix, graph.num_nodes, input_dim, padded_m, padded_n));
    auto b_mont = mont_vector_from_raw(matrix_scale16_raw(head.W, input_dim, head.output_dim, padded_n, padded_k));
    auto c_mont = projection_product_mont(a_mont, b_mont, padded_num, padded_m, padded_n, padded_k);
    const std::size_t block_size = projection_common_commitment_block_size(
        requested_commitment_block_size, a_mont.size(), b_mont.size(), c_mont.size());

    ZkdlFullRelationRecord record;
    record.stage = "cpu_projection_matmul";
    record.relation_scope = spec.dataset + ":" + layer_kind + "_layer" + std::to_string(layer_index)
        + ":head" + std::to_string(head_index) + ":pre_rescale_product";
    record.layer_kind = layer_kind;
    record.layer_index = layer_index;
    record.head_index = head_index;

    CpuCommitment gen = CpuCommitment::deterministic(
        static_cast<uint>(block_size),
        spec.dataset + ":cpu_projection_block" + std::to_string(block_size));
    CpuFrTensor a_tensor(std::move(a_mont));
    CpuFrTensor b_tensor(std::move(b_mont));
    CpuFrTensor c_tensor(std::move(c_mont));
    ScopedTimer commit_timer;
    const auto com_a = gen.commit(a_tensor);
    const auto com_b = gen.commit(b_tensor);
    const auto com_c = gen.commit(c_tensor);
    record.commitment_ms += commit_timer.elapsed_ms();

    const std::string label = record.relation_scope + ":matmul";
    MatmulRelationProof proof;
    {
        ScopedTimer timer;
        proof = cpu_matmul_relation_prove(
            a_tensor, b_tensor, c_tensor, gen, com_a, com_b, com_c,
            num, m, n, k, padded_num, padded_m, padded_n, padded_k, label);
        record.prove_ms += timer.elapsed_ms();
    }
    {
        ScopedTimer timer;
        std::string failure;
        if (!cpu_matmul_relation_verify(proof, gen, com_a, com_b, com_c, label, &failure)) {
            throw std::runtime_error("CPU projection matmul verifier rejected: " + failure);
        }
        record.verify_ms += timer.elapsed_ms();
    }
    record.proof_size_bytes = proof.serialized_size_bytes();
    record.proof_digests.push_back("matmul=" + proof.transcript_digest);
    record.proof_generated = true;
    record.verified = true;
    return record;
}

// 从模型参数中按层/头取 hidden head；缺失时直接失败，避免 proof scope 静默缩小。
const HeadParams& require_hidden_head_params(
    const ModelParams& model,
    std::size_t layer_index,
    std::size_t local_head_index)
{
    std::size_t seen = 0;
    for (const auto& head : model.hidden_heads) {
        if (head.layer_index != layer_index) continue;
        if (seen == local_head_index) return head;
        ++seen;
    }
    throw std::runtime_error("hidden projection head 参数不存在");
}
