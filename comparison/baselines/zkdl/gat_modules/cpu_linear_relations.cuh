#pragma once

// CPU-only linear opening relation。该模块只复验承诺开值与线性等式，
// 不读取 witness，也不依赖 CUDA storage。

struct CpuLinearTerm
{
    std::string label;
    const CpuFrTensor* tensor = nullptr;
    const CpuG1Tensor* commitment = nullptr;
    Fr_t coefficient = Fr_host_zero();
    std::vector<Fr_t> point;
};

std::vector<Fr_t> cpu_deterministic_mont_fr_vec(std::size_t n, const std::string& label)
{
    auto raw = deterministic_raw_fr_vec(n, label);
    for (auto& value : raw) {
        value = Fr_host_mont(value);
    }
    return raw;
}

void cpu_absorb_linear_metadata(
    FormalTranscript& transcript,
    const std::string& label,
    const std::vector<uint>& shape,
    const std::vector<std::string>& tensor_labels,
    const std::vector<Fr_t>& coefficients,
    const std::vector<std::vector<Fr_t>>& points,
    const Fr_t& public_coefficient,
    const Fr_t& public_evaluation,
    const Fr_t& constant,
    const std::vector<const CpuG1Tensor*>& commitments)
{
    transcript.absorb_label("linear_relation_cpu");
    transcript.absorb_label(label);
    transcript.absorb_u64(shape.size());
    for (const auto dim : shape) transcript.absorb_u64(dim);
    formal_absorb_string_vector(transcript, tensor_labels);
    transcript.absorb_u64(coefficients.size());
    for (const auto& coeff : coefficients) transcript.absorb_fr(coeff);
    transcript.absorb_u64(points.size());
    for (const auto& point : points) {
        transcript.absorb_u64(point.size());
        for (const auto& coord : point) transcript.absorb_fr(coord);
    }
    transcript.absorb_fr(public_coefficient);
    transcript.absorb_fr(public_evaluation);
    transcript.absorb_fr(constant);
    transcript.absorb_u64(commitments.size());
    for (const auto* commitment : commitments) {
        if (commitment == nullptr) throw std::runtime_error("CPU linear commitment 为空");
        cpu_absorb_g1_tensor(transcript, *commitment);
    }
}

LinearRelationProof cpu_linear_relation_prove(
    const std::string& relation_label,
    const std::vector<uint>& shape,
    const std::vector<CpuLinearTerm>& terms,
    const CpuCommitment& generators,
    const Fr_t& public_coefficient,
    const Fr_t& public_evaluation,
    const Fr_t& constant)
{
    if (terms.empty()) throw std::runtime_error("CPU linear relation 至少需要一个 committed term");
    LinearRelationProof proof;
    proof.label = relation_label;
    proof.shape = shape;
    proof.public_coefficient = public_coefficient;
    proof.public_evaluation = public_evaluation;
    proof.constant = constant;

    std::vector<const CpuG1Tensor*> commitments;
    for (const auto& term : terms) {
        if (term.tensor == nullptr || term.commitment == nullptr) {
            throw std::runtime_error("CPU linear relation term 为空");
        }
        proof.tensor_labels.push_back(term.label);
        proof.coefficients.push_back(term.coefficient);
        proof.points.push_back(term.point);
        commitments.push_back(term.commitment);
    }

    FormalTranscript transcript;
    cpu_absorb_linear_metadata(
        transcript, relation_label, shape, proof.tensor_labels, proof.coefficients,
        proof.points, public_coefficient, public_evaluation, constant, commitments);
    proof.openings.resize(terms.size());
    proof.evaluations.resize(terms.size(), Fr_host_zero());
    for (std::size_t i = 0; i < terms.size(); ++i) {
        proof.evaluations[i] = generators.open_with_proof(
            *terms[i].tensor, *terms[i].commitment, terms[i].point, proof.openings[i]);
        transcript.absorb_fr(proof.evaluations[i]);
    }
    proof.transcript_digest = transcript.digest_hex();
    return proof;
}

bool cpu_linear_relation_verify(
    const LinearRelationProof& proof,
    const CpuCommitment& generators,
    const std::vector<const CpuG1Tensor*>& commitments,
    const Fr_t& verifier_public_evaluation,
    std::string* failure)
{
    const auto fail = [&](const std::string& message) {
        if (failure != nullptr) *failure = message;
        return false;
    };
    if (commitments.size() != proof.coefficients.size()
        || proof.openings.size() != proof.coefficients.size()
        || proof.evaluations.size() != proof.coefficients.size()
        || proof.points.size() != proof.coefficients.size()
        || proof.tensor_labels.size() != proof.coefficients.size()) {
        return fail("CPU linear relation vector size mismatch");
    }
    if (!Fr_host_eq(proof.public_evaluation, verifier_public_evaluation)) {
        return fail("CPU linear public evaluation mismatch");
    }
    FormalTranscript transcript;
    cpu_absorb_linear_metadata(
        transcript, proof.label, proof.shape, proof.tensor_labels, proof.coefficients,
        proof.points, proof.public_coefficient, proof.public_evaluation, proof.constant, commitments);

    Fr_t acc = proof.constant;
    acc = Fr_host_add(acc, Fr_host_mul(proof.public_coefficient, verifier_public_evaluation));
    for (std::size_t i = 0; i < proof.coefficients.size(); ++i) {
        if (commitments[i] == nullptr) return fail("CPU linear verifier commitment 为空");
        if (!formal_points_equal(proof.openings[i].point, proof.points[i])) {
            return fail("CPU linear opening point mismatch");
        }
        if (!Fr_host_eq(proof.openings[i].value, proof.evaluations[i])) {
            return fail("CPU linear opening value mismatch");
        }
        if (!CpuCommitment::verify_opening(generators, *commitments[i], proof.openings[i])) {
            return fail("CPU linear opening rejected");
        }
        acc = Fr_host_add(acc, Fr_host_mul(proof.coefficients[i], proof.evaluations[i]));
        transcript.absorb_fr(proof.evaluations[i]);
    }
    if (!Fr_host_eq(acc, Fr_host_zero())) return fail("CPU linear equation mismatch");
    if (proof.transcript_digest != transcript.digest_hex()) return fail("CPU linear transcript digest mismatch");
    return true;
}
