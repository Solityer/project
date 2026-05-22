#pragma once

// CPU 多列 relation：把矩阵列维先用 transcript 挑战压缩，再复用 CPU-only
// sparse/grouped/hadamard relation。该模块不依赖 CUDA storage，也不引入主仓库证明方案。

std::vector<Fr_t> cpu_matrix_column_compression_host(
    const CpuFrTensor& matrix,
    uint logical_rows,
    uint padded_rows,
    uint logical_columns,
    uint padded_columns,
    const std::vector<Fr_t>& column_point)
{
    if (logical_rows > padded_rows || logical_columns > padded_columns) {
        throw std::runtime_error("CPU multi-column compression logical shape exceeds padded shape");
    }
    if (matrix.size() != padded_rows * padded_columns) {
        throw std::runtime_error("CPU multi-column compression matrix size mismatch");
    }
    if (column_point.size() != ceilLog2(padded_columns)) {
        throw std::runtime_error("CPU multi-column compression column point dimension mismatch");
    }
    std::vector<Fr_t> out(padded_rows, Fr_host_zero());
    std::vector<Fr_t> row_values(padded_columns, Fr_host_zero());
    for (uint row = 0; row < padded_rows; ++row) {
        for (uint col = 0; col < padded_columns; ++col) {
            row_values[col] = matrix.values[static_cast<std::size_t>(row) * padded_columns + col];
        }
        out[row] = formal_host_mle_vector(row_values, column_point.begin(), column_point.end());
    }
    return out;
}

void cpu_absorb_multi_column_sparse_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    const std::string& direction,
    std::uint64_t topology_hash,
    uint node_count,
    uint edge_count,
    uint column_count,
    uint padded_node_count,
    uint padded_edge_count,
    uint padded_column_count,
    const CpuG1Tensor& com_node_matrix,
    const CpuG1Tensor& com_edge_matrix)
{
    transcript.absorb_label("zkdl_gat_multi_column_sparse_routing_cpu");
    transcript.absorb_label(label);
    transcript.absorb_label(direction);
    transcript.absorb_u64(topology_hash);
    transcript.absorb_u64(node_count);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(column_count);
    transcript.absorb_u64(padded_node_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(padded_column_count);
    cpu_absorb_g1_tensor(transcript, com_node_matrix);
    cpu_absorb_g1_tensor(transcript, com_edge_matrix);
}

void cpu_absorb_multi_column_hadamard_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    uint edge_count,
    uint column_count,
    uint padded_edge_count,
    uint padded_column_count,
    const CpuG1Tensor& com_alpha,
    const CpuG1Tensor& com_message_matrix,
    const CpuG1Tensor& com_weighted_matrix)
{
    transcript.absorb_label("zkdl_gat_multi_column_hadamard_cpu");
    transcript.absorb_label(label);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(column_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(padded_column_count);
    cpu_absorb_g1_tensor(transcript, com_alpha);
    cpu_absorb_g1_tensor(transcript, com_message_matrix);
    cpu_absorb_g1_tensor(transcript, com_weighted_matrix);
}

void cpu_absorb_multi_column_grouped_sum_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    std::uint64_t topology_hash,
    uint node_count,
    uint edge_count,
    uint column_count,
    uint padded_node_count,
    uint padded_edge_count,
    uint padded_column_count,
    const CpuG1Tensor& com_edge_matrix,
    const CpuG1Tensor& com_node_matrix)
{
    transcript.absorb_label("zkdl_gat_multi_column_grouped_sum_cpu");
    transcript.absorb_label(label);
    transcript.absorb_u64(topology_hash);
    transcript.absorb_u64(node_count);
    transcript.absorb_u64(edge_count);
    transcript.absorb_u64(column_count);
    transcript.absorb_u64(padded_node_count);
    transcript.absorb_u64(padded_edge_count);
    transcript.absorb_u64(padded_column_count);
    cpu_absorb_g1_tensor(transcript, com_edge_matrix);
    cpu_absorb_g1_tensor(transcript, com_node_matrix);
}

MultiColumnSparseRoutingGadgetProof cpu_prove_multi_column_sparse_routing(
    const CpuFrTensor& node_matrix,
    const CpuFrTensor& edge_matrix,
    const CpuCommitment& matrix_generators,
    const CpuG1Tensor& com_node_matrix,
    const CpuG1Tensor& com_edge_matrix,
    const CpuCommitment& vector_generators,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    uint column_count,
    uint padded_column_count,
    const std::string& direction,
    const std::string& relation_label)
{
    if (direction != "source" && direction != "destination") {
        throw std::runtime_error("CPU multi-column sparse routing direction 必须是 source 或 destination");
    }
    const uint padded_node_count = formal_next_power_of_two_u32(node_count);
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (padded_column_count != formal_next_power_of_two_u32(column_count)) {
        throw std::runtime_error("CPU multi-column sparse routing padded column mismatch");
    }
    if (node_matrix.size() != padded_node_count * padded_column_count
        || edge_matrix.size() != padded_edge_count * padded_column_count) {
        throw std::runtime_error("CPU multi-column sparse routing matrix size mismatch");
    }

    MultiColumnSparseRoutingGadgetProof proof;
    proof.label = relation_label;
    proof.direction = direction;
    proof.topology_hash = formal_sparse_topology_hash(src, dst, node_count, edge_count);
    proof.node_count = node_count;
    proof.edge_count = edge_count;
    proof.column_count = column_count;
    proof.padded_node_count = padded_node_count;
    proof.padded_edge_count = padded_edge_count;
    proof.padded_column_count = padded_column_count;

    FormalTranscript transcript;
    cpu_absorb_multi_column_sparse_metadata(
        transcript, relation_label, direction, proof.topology_hash,
        node_count, edge_count, column_count, padded_node_count, padded_edge_count,
        padded_column_count, com_node_matrix, com_edge_matrix);
    proof.column_point = formal_make_challenge_point(
        transcript, ceilLog2(padded_column_count), "multi_column_sparse_column_point");

    const auto compressed_node_host = cpu_matrix_column_compression_host(
        node_matrix, node_count, padded_node_count, column_count, padded_column_count, proof.column_point);
    const auto compressed_edge_host = cpu_matrix_column_compression_host(
        edge_matrix, edge_count, padded_edge_count, column_count, padded_column_count, proof.column_point);
    CpuFrTensor compressed_node(compressed_node_host);
    CpuFrTensor compressed_edge(compressed_edge_host);
    const auto com_compressed_node = vector_generators.commit(compressed_node);
    const auto com_compressed_edge = vector_generators.commit(compressed_edge);
    proof.compressed_node_commitment_points = com_compressed_node.points;
    proof.compressed_edge_commitment_points = com_compressed_edge.points;
    proof.compressed_routing = cpu_sparse_routing_relation_prove(
        compressed_node, compressed_edge,
        vector_generators, com_compressed_node,
        vector_generators, com_compressed_edge,
        src, dst, node_count, edge_count, direction,
        relation_label + ":compressed_sparse_routing");

    const auto node_matrix_point = formal_concat_points({proof.column_point, proof.compressed_routing.opening_node.point});
    const auto edge_matrix_point = formal_concat_points({proof.column_point, proof.compressed_routing.opening_edge.point});
    proof.node_matrix_eval = matrix_generators.open_with_proof(
        node_matrix, com_node_matrix, node_matrix_point, proof.opening_node_matrix);
    proof.edge_matrix_eval = matrix_generators.open_with_proof(
        edge_matrix, com_edge_matrix, edge_matrix_point, proof.opening_edge_matrix);
    if (!Fr_host_eq(proof.node_matrix_eval, proof.compressed_routing.opening_node.value)
        || !Fr_host_eq(proof.edge_matrix_eval, proof.compressed_routing.opening_edge.value)) {
        throw std::runtime_error("CPU multi-column sparse routing compression binding mismatch");
    }
    transcript.absorb_label(proof.compressed_routing.transcript_digest);
    transcript.absorb_fr(proof.node_matrix_eval);
    transcript.absorb_fr(proof.edge_matrix_eval);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool cpu_verify_multi_column_sparse_routing(
    const MultiColumnSparseRoutingGadgetProof& proof,
    const CpuCommitment& matrix_generators,
    const CpuG1Tensor& com_node_matrix,
    const CpuG1Tensor& com_edge_matrix,
    const CpuCommitment& vector_generators,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    const std::string& direction,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label || proof.direction != direction) {
        return fail("CPU multi-column sparse routing label/direction mismatch");
    }
    if (proof.padded_node_count != formal_next_power_of_two_u32(proof.node_count)
        || proof.padded_edge_count != formal_next_power_of_two_u32(proof.edge_count)
        || proof.padded_column_count != formal_next_power_of_two_u32(proof.column_count)) {
        return fail("CPU multi-column sparse routing padded shape mismatch");
    }
    const auto expected_hash = formal_sparse_topology_hash(src, dst, proof.node_count, proof.edge_count);
    if (proof.topology_hash != expected_hash) return fail("CPU multi-column sparse routing topology hash mismatch");

    FormalTranscript transcript;
    cpu_absorb_multi_column_sparse_metadata(
        transcript, relation_label, direction, proof.topology_hash,
        proof.node_count, proof.edge_count, proof.column_count, proof.padded_node_count,
        proof.padded_edge_count, proof.padded_column_count, com_node_matrix, com_edge_matrix);
    const auto expected_column_point = formal_make_challenge_point(
        transcript, ceilLog2(proof.padded_column_count), "multi_column_sparse_column_point");
    if (!formal_points_equal(expected_column_point, proof.column_point)) {
        return fail("CPU multi-column sparse routing column point mismatch");
    }
    CpuG1Tensor com_compressed_node(proof.compressed_node_commitment_points);
    CpuG1Tensor com_compressed_edge(proof.compressed_edge_commitment_points);
    std::string inner_failure;
    if (!cpu_sparse_routing_relation_verify(
            proof.compressed_routing,
            vector_generators, com_compressed_node,
            vector_generators, com_compressed_edge,
            src, dst, direction,
            relation_label + ":compressed_sparse_routing",
            &inner_failure)) {
        return fail("CPU multi-column sparse routing compressed verifier failed: " + inner_failure);
    }
    const auto expected_node_matrix_point = formal_concat_points({proof.column_point, proof.compressed_routing.opening_node.point});
    const auto expected_edge_matrix_point = formal_concat_points({proof.column_point, proof.compressed_routing.opening_edge.point});
    if (!formal_points_equal(proof.opening_node_matrix.point, expected_node_matrix_point)
        || !formal_points_equal(proof.opening_edge_matrix.point, expected_edge_matrix_point)) {
        return fail("CPU multi-column sparse routing matrix opening point mismatch");
    }
    if (!Fr_host_eq(proof.opening_node_matrix.value, proof.node_matrix_eval)
        || !Fr_host_eq(proof.opening_edge_matrix.value, proof.edge_matrix_eval)) {
        return fail("CPU multi-column sparse routing matrix opening value mismatch");
    }
    if (!Fr_host_eq(proof.node_matrix_eval, proof.compressed_routing.opening_node.value)
        || !Fr_host_eq(proof.edge_matrix_eval, proof.compressed_routing.opening_edge.value)) {
        return fail("CPU multi-column sparse routing compression binding mismatch");
    }
    if (!CpuCommitment::verify_opening(matrix_generators, com_node_matrix, proof.opening_node_matrix)
        || !CpuCommitment::verify_opening(matrix_generators, com_edge_matrix, proof.opening_edge_matrix)) {
        return fail("CPU multi-column sparse routing matrix opening rejected");
    }
    transcript.absorb_label(proof.compressed_routing.transcript_digest);
    transcript.absorb_fr(proof.node_matrix_eval);
    transcript.absorb_fr(proof.edge_matrix_eval);
    if (proof.transcript_digest != transcript.digest_hex()) {
        return fail("CPU multi-column sparse routing transcript digest mismatch");
    }
    return true;
}

MultiColumnHadamardGadgetProof cpu_prove_multi_column_alpha_hadamard(
    const CpuFrTensor& alpha,
    const CpuFrTensor& message_matrix,
    const CpuFrTensor& weighted_matrix,
    const CpuCommitment& vector_generators,
    const CpuG1Tensor& com_alpha,
    const CpuCommitment& matrix_generators,
    const CpuG1Tensor& com_message_matrix,
    const CpuG1Tensor& com_weighted_matrix,
    uint edge_count,
    uint column_count,
    uint padded_column_count,
    const std::string& relation_label)
{
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (padded_column_count != formal_next_power_of_two_u32(column_count)) {
        throw std::runtime_error("CPU multi-column Hadamard padded column mismatch");
    }
    if (alpha.size() != padded_edge_count
        || message_matrix.size() != padded_edge_count * padded_column_count
        || weighted_matrix.size() != padded_edge_count * padded_column_count) {
        throw std::runtime_error("CPU multi-column Hadamard tensor size mismatch");
    }

    MultiColumnHadamardGadgetProof proof;
    proof.label = relation_label;
    proof.edge_count = edge_count;
    proof.column_count = column_count;
    proof.padded_edge_count = padded_edge_count;
    proof.padded_column_count = padded_column_count;

    FormalTranscript transcript;
    cpu_absorb_multi_column_hadamard_metadata(
        transcript, relation_label, edge_count, column_count, padded_edge_count,
        padded_column_count, com_alpha, com_message_matrix, com_weighted_matrix);
    proof.column_point = formal_make_challenge_point(
        transcript, ceilLog2(padded_column_count), "multi_column_hadamard_column_point");
    const auto compressed_message_host = cpu_matrix_column_compression_host(
        message_matrix, edge_count, padded_edge_count, column_count, padded_column_count, proof.column_point);
    const auto compressed_weighted_host = cpu_matrix_column_compression_host(
        weighted_matrix, edge_count, padded_edge_count, column_count, padded_column_count, proof.column_point);
    CpuFrTensor compressed_message(compressed_message_host);
    CpuFrTensor compressed_weighted(compressed_weighted_host);
    const auto com_compressed_message = vector_generators.commit(compressed_message);
    const auto com_compressed_weighted = vector_generators.commit(compressed_weighted);
    proof.compressed_message_commitment_points = com_compressed_message.points;
    proof.compressed_weighted_commitment_points = com_compressed_weighted.points;
    proof.compressed_hadamard = cpu_hadamard_relation_prove(
        alpha, compressed_message, compressed_weighted,
        vector_generators, com_alpha, com_compressed_message, com_compressed_weighted,
        relation_label + ":compressed_alpha_times_message");
    const auto message_matrix_point = formal_concat_points({proof.column_point, proof.compressed_hadamard.opening_b.point});
    const auto weighted_matrix_point = formal_concat_points({proof.column_point, proof.compressed_hadamard.opening_c.point});
    proof.message_matrix_eval = matrix_generators.open_with_proof(
        message_matrix, com_message_matrix, message_matrix_point, proof.opening_message_matrix);
    proof.weighted_matrix_eval = matrix_generators.open_with_proof(
        weighted_matrix, com_weighted_matrix, weighted_matrix_point, proof.opening_weighted_matrix);
    if (!Fr_host_eq(proof.message_matrix_eval, proof.compressed_hadamard.opening_b.value)
        || !Fr_host_eq(proof.weighted_matrix_eval, proof.compressed_hadamard.opening_c.value)) {
        throw std::runtime_error("CPU multi-column Hadamard compression binding mismatch");
    }
    transcript.absorb_label(proof.compressed_hadamard.transcript_digest);
    transcript.absorb_fr(proof.message_matrix_eval);
    transcript.absorb_fr(proof.weighted_matrix_eval);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool cpu_verify_multi_column_alpha_hadamard(
    const MultiColumnHadamardGadgetProof& proof,
    const CpuCommitment& vector_generators,
    const CpuG1Tensor& com_alpha,
    const CpuCommitment& matrix_generators,
    const CpuG1Tensor& com_message_matrix,
    const CpuG1Tensor& com_weighted_matrix,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("CPU multi-column Hadamard label mismatch");
    if (proof.padded_edge_count != formal_next_power_of_two_u32(proof.edge_count)
        || proof.padded_column_count != formal_next_power_of_two_u32(proof.column_count)) {
        return fail("CPU multi-column Hadamard padded shape mismatch");
    }
    FormalTranscript transcript;
    cpu_absorb_multi_column_hadamard_metadata(
        transcript, relation_label, proof.edge_count, proof.column_count,
        proof.padded_edge_count, proof.padded_column_count,
        com_alpha, com_message_matrix, com_weighted_matrix);
    const auto expected_column_point = formal_make_challenge_point(
        transcript, ceilLog2(proof.padded_column_count), "multi_column_hadamard_column_point");
    if (!formal_points_equal(expected_column_point, proof.column_point)) {
        return fail("CPU multi-column Hadamard column point mismatch");
    }
    CpuG1Tensor com_compressed_message(proof.compressed_message_commitment_points);
    CpuG1Tensor com_compressed_weighted(proof.compressed_weighted_commitment_points);
    std::string inner_failure;
    if (!cpu_hadamard_relation_verify(
            proof.compressed_hadamard,
            vector_generators, com_alpha, com_compressed_message, com_compressed_weighted,
            relation_label + ":compressed_alpha_times_message",
            &inner_failure)) {
        return fail("CPU multi-column Hadamard compressed verifier failed: " + inner_failure);
    }
    const auto expected_message_matrix_point = formal_concat_points({proof.column_point, proof.compressed_hadamard.opening_b.point});
    const auto expected_weighted_matrix_point = formal_concat_points({proof.column_point, proof.compressed_hadamard.opening_c.point});
    if (!formal_points_equal(proof.opening_message_matrix.point, expected_message_matrix_point)
        || !formal_points_equal(proof.opening_weighted_matrix.point, expected_weighted_matrix_point)) {
        return fail("CPU multi-column Hadamard matrix opening point mismatch");
    }
    if (!Fr_host_eq(proof.opening_message_matrix.value, proof.message_matrix_eval)
        || !Fr_host_eq(proof.opening_weighted_matrix.value, proof.weighted_matrix_eval)) {
        return fail("CPU multi-column Hadamard matrix opening value mismatch");
    }
    if (!Fr_host_eq(proof.message_matrix_eval, proof.compressed_hadamard.opening_b.value)
        || !Fr_host_eq(proof.weighted_matrix_eval, proof.compressed_hadamard.opening_c.value)) {
        return fail("CPU multi-column Hadamard compression binding mismatch");
    }
    if (!CpuCommitment::verify_opening(matrix_generators, com_message_matrix, proof.opening_message_matrix)
        || !CpuCommitment::verify_opening(matrix_generators, com_weighted_matrix, proof.opening_weighted_matrix)) {
        return fail("CPU multi-column Hadamard matrix opening rejected");
    }
    transcript.absorb_label(proof.compressed_hadamard.transcript_digest);
    transcript.absorb_fr(proof.message_matrix_eval);
    transcript.absorb_fr(proof.weighted_matrix_eval);
    if (proof.transcript_digest != transcript.digest_hex()) {
        return fail("CPU multi-column Hadamard transcript digest mismatch");
    }
    return true;
}

MultiColumnGroupedSumRelationProof cpu_multi_column_grouped_sum_relation_prove(
    const CpuFrTensor& edge_matrix,
    const CpuFrTensor& node_matrix,
    const CpuCommitment& edge_matrix_generators,
    const CpuG1Tensor& com_edge_matrix,
    const CpuCommitment& node_matrix_generators,
    const CpuG1Tensor& com_node_matrix,
    const CpuCommitment& vector_generators,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    uint node_count,
    uint edge_count,
    uint column_count,
    uint padded_column_count,
    const std::string& relation_label)
{
    (void)src;
    const uint padded_node_count = formal_next_power_of_two_u32(node_count);
    const uint padded_edge_count = formal_next_power_of_two_u32(edge_count);
    if (padded_column_count != formal_next_power_of_two_u32(column_count)) {
        throw std::runtime_error("CPU multi-column grouped-sum padded column mismatch");
    }
    if (edge_matrix.size() != padded_edge_count * padded_column_count
        || node_matrix.size() != padded_node_count * padded_column_count) {
        throw std::runtime_error("CPU multi-column grouped-sum matrix size mismatch");
    }
    MultiColumnGroupedSumRelationProof proof;
    proof.label = relation_label;
    proof.topology_hash = formal_sparse_topology_hash(src, dst, node_count, edge_count);
    proof.node_count = node_count;
    proof.edge_count = edge_count;
    proof.column_count = column_count;
    proof.padded_node_count = padded_node_count;
    proof.padded_edge_count = padded_edge_count;
    proof.padded_column_count = padded_column_count;
    proof.commitment_edge_matrix_label = relation_label + ":edge_matrix";
    proof.commitment_node_matrix_label = relation_label + ":node_matrix";
    proof.commitment_compressed_edge_label = relation_label + ":compressed_edge";
    proof.commitment_compressed_node_label = relation_label + ":compressed_node";

    FormalTranscript transcript;
    cpu_absorb_multi_column_grouped_sum_metadata(
        transcript, relation_label, proof.topology_hash,
        node_count, edge_count, column_count, padded_node_count, padded_edge_count,
        padded_column_count, com_edge_matrix, com_node_matrix);
    proof.column_point = formal_make_challenge_point(
        transcript, ceilLog2(padded_column_count), "multi_column_grouped_sum_column_point");

    const auto compressed_edge_host = cpu_matrix_column_compression_host(
        edge_matrix, edge_count, padded_edge_count, column_count, padded_column_count, proof.column_point);
    const auto compressed_node_host = cpu_matrix_column_compression_host(
        node_matrix, node_count, padded_node_count, column_count, padded_column_count, proof.column_point);
    CpuFrTensor compressed_edge(compressed_edge_host);
    CpuFrTensor compressed_node(compressed_node_host);
    const auto com_compressed_edge = vector_generators.commit(compressed_edge);
    const auto com_compressed_node = vector_generators.commit(compressed_node);
    proof.compressed_edge_commitment_points = com_compressed_edge.points;
    proof.compressed_node_commitment_points = com_compressed_node.points;

    proof.compressed_grouped_sum = cpu_grouped_sum_relation_prove(
        compressed_edge,
        compressed_node,
        vector_generators,
        com_compressed_edge,
        vector_generators,
        com_compressed_node,
        src,
        dst,
        node_count,
        edge_count,
        relation_label + ":compressed_grouped_sum");

    const auto edge_matrix_point = formal_concat_points({proof.column_point, proof.compressed_grouped_sum.opening_edge_input.point});
    const auto node_matrix_point = formal_concat_points({proof.column_point, proof.compressed_grouped_sum.opening_node_output.point});
    proof.edge_matrix_eval = edge_matrix_generators.open_with_proof(
        edge_matrix, com_edge_matrix, edge_matrix_point, proof.opening_edge_matrix);
    proof.node_matrix_eval = node_matrix_generators.open_with_proof(
        node_matrix, com_node_matrix, node_matrix_point, proof.opening_node_matrix);
    if (!Fr_host_eq(proof.edge_matrix_eval, proof.compressed_grouped_sum.opening_edge_input.value)
        || !Fr_host_eq(proof.node_matrix_eval, proof.compressed_grouped_sum.opening_node_output.value)) {
        throw std::runtime_error("CPU multi-column grouped-sum compression binding mismatch");
    }
    transcript.absorb_label(proof.compressed_grouped_sum.transcript_digest);
    transcript.absorb_fr(proof.edge_matrix_eval);
    transcript.absorb_fr(proof.node_matrix_eval);
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool cpu_multi_column_grouped_sum_relation_verify(
    const MultiColumnGroupedSumRelationProof& proof,
    const CpuCommitment& edge_matrix_generators,
    const CpuG1Tensor& com_edge_matrix,
    const CpuCommitment& node_matrix_generators,
    const CpuG1Tensor& com_node_matrix,
    const CpuCommitment& vector_generators,
    const std::vector<std::uint32_t>& src,
    const std::vector<std::uint32_t>& dst,
    const std::string& relation_label,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (proof.label != relation_label) return fail("CPU multi-column grouped-sum label mismatch");
    if (proof.node_count == 0 || proof.edge_count == 0 || proof.column_count == 0) {
        return fail("CPU empty multi-column grouped-sum relation");
    }
    if (proof.padded_node_count != formal_next_power_of_two_u32(proof.node_count)
        || proof.padded_edge_count != formal_next_power_of_two_u32(proof.edge_count)
        || proof.padded_column_count != formal_next_power_of_two_u32(proof.column_count)) {
        return fail("CPU multi-column grouped-sum padded shape mismatch");
    }
    const auto expected_hash = formal_sparse_topology_hash(src, dst, proof.node_count, proof.edge_count);
    if (proof.topology_hash != expected_hash) return fail("CPU multi-column grouped-sum topology hash mismatch");

    FormalTranscript transcript;
    cpu_absorb_multi_column_grouped_sum_metadata(
        transcript, relation_label, proof.topology_hash,
        proof.node_count, proof.edge_count, proof.column_count, proof.padded_node_count,
        proof.padded_edge_count, proof.padded_column_count, com_edge_matrix, com_node_matrix);
    const auto expected_column_point = formal_make_challenge_point(
        transcript, ceilLog2(proof.padded_column_count), "multi_column_grouped_sum_column_point");
    if (!formal_points_equal(expected_column_point, proof.column_point)) {
        return fail("CPU multi-column grouped-sum column point replay mismatch");
    }

    CpuG1Tensor com_compressed_edge(proof.compressed_edge_commitment_points);
    CpuG1Tensor com_compressed_node(proof.compressed_node_commitment_points);
    std::string grouped_failure;
    if (!cpu_grouped_sum_relation_verify(
            proof.compressed_grouped_sum,
            vector_generators,
            com_compressed_edge,
            vector_generators,
            com_compressed_node,
            src,
            dst,
            relation_label + ":compressed_grouped_sum",
            &grouped_failure)) {
        return fail("CPU multi-column grouped-sum compressed verifier failed: " + grouped_failure);
    }

    const auto expected_edge_matrix_point = formal_concat_points({proof.column_point, proof.compressed_grouped_sum.opening_edge_input.point});
    const auto expected_node_matrix_point = formal_concat_points({proof.column_point, proof.compressed_grouped_sum.opening_node_output.point});
    if (!formal_points_equal(proof.opening_edge_matrix.point, expected_edge_matrix_point)
        || !formal_points_equal(proof.opening_node_matrix.point, expected_node_matrix_point)) {
        return fail("CPU multi-column grouped-sum matrix opening point mismatch");
    }
    if (!Fr_host_eq(proof.opening_edge_matrix.value, proof.edge_matrix_eval)
        || !Fr_host_eq(proof.opening_node_matrix.value, proof.node_matrix_eval)) {
        return fail("CPU multi-column grouped-sum matrix opening value mismatch");
    }
    if (!Fr_host_eq(proof.edge_matrix_eval, proof.compressed_grouped_sum.opening_edge_input.value)
        || !Fr_host_eq(proof.node_matrix_eval, proof.compressed_grouped_sum.opening_node_output.value)) {
        return fail("CPU multi-column grouped-sum compression binding mismatch");
    }
    if (!CpuCommitment::verify_opening(edge_matrix_generators, com_edge_matrix, proof.opening_edge_matrix)
        || !CpuCommitment::verify_opening(node_matrix_generators, com_node_matrix, proof.opening_node_matrix)) {
        return fail("CPU multi-column grouped-sum matrix opening rejected");
    }

    transcript.absorb_label(proof.compressed_grouped_sum.transcript_digest);
    transcript.absorb_fr(proof.edge_matrix_eval);
    transcript.absorb_fr(proof.node_matrix_eval);
    if (proof.transcript_digest != transcript.digest_hex()) {
        return fail("CPU multi-column grouped-sum transcript digest mismatch");
    }
    return true;
}
