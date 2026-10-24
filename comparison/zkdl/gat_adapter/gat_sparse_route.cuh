#ifndef ZKDL_GAT_SPARSE_ROUTE_CUH
#define ZKDL_GAT_SPARSE_ROUTE_CUH

inline std::vector<FrWire> sparse_route_point(const std::vector<Fr_t> &point) {
    std::vector<FrWire> result;
    result.reserve(point.size());
    for (const auto &coordinate : point) result.push_back(zkdl::fr_wire(coordinate));
    return result;
}

inline std::vector<FrWire> sparse_route_hyrax_point(const std::vector<FrWire> &fold_order_point) {
    return std::vector<FrWire>(fold_order_point.rbegin(), fold_order_point.rend());
}

inline bool sparse_route_public_inputs_valid(const OperationDescriptor &descriptor, const std::vector<CommitmentWire> &commitments,
                                             const std::vector<zkdl::CpuG1Point> &generators, const RelationStatement &statement, std::uint32_t &source_index,
                                             std::uint32_t &output_index, std::string *error) {
    if (!descriptor.valid(error) || (descriptor.kind != RelationKind::SparseRoute && descriptor.kind != RelationKind::Alpha) ||
        !zktr_all_statement_digests_valid(statement, error) || !route_commitment_index(commitments, descriptor.selector_source_ref, source_index, error) ||
        !route_commitment_index(commitments, descriptor.selector_output_ref, output_index, error))
        return false;
    const std::uint64_t source_domain = std::uint64_t(descriptor.selector_node_domain) * descriptor.selector_source_feature_domain;
    const std::uint64_t output_domain = std::uint64_t(descriptor.selector_edge_domain) * descriptor.selector_feature_domain;
    if (source_domain == 0U || output_domain == 0U || source_domain > (std::uint64_t(1U) << kMaxRelationPointBits) ||
        output_domain > (std::uint64_t(1U) << kMaxRelationPointBits) || commitments[source_index].domain_size != source_domain ||
        commitments[output_index].domain_size != output_domain || source_index >= kZkTerminalEvaluationIndexBase ||
        output_index >= kZkTerminalEvaluationIndexBase) {
        if (error) *error = "sparse route canonical commitment/domain mismatch";
        return false;
    }
    ZkRelationSegmentLayout source_layout, output_layout;
    if (!zktr_segment_layout(source_domain, generators.size(), source_layout, error) ||
        !zktr_segment_layout(output_domain, generators.size(), output_layout, error))
        return false;
    return true;
}

inline std::array<byte, 32U> sparse_route_absorb_header(const OperationDescriptor &descriptor, const RelationStatement &statement,
                                                        const std::vector<CommitmentWire> &commitments, std::uint32_t source_index, std::uint32_t output_index,
                                                        const zkdl::CpuG1Point &h, const zkdl::CpuG1Point &u, zkdl::Transcript &transcript,
                                                        const std::vector<byte>* route_context = nullptr) {
    std::vector<byte> encoded;
    zkdl::append_u32(encoded, SparseRouteProof::kVersion);
    append_statement(encoded, statement);
    append_operation(encoded, descriptor);
    zkdl::append_u32(encoded, source_index);
    zkdl::append_u32(encoded, output_index);
    append_commitment(encoded, commitments[source_index]);
    append_commitment(encoded, commitments[output_index]);
    if (route_context != nullptr) {
        zkdl::append_u64(encoded, static_cast<std::uint64_t>(route_context->size()));
        encoded.insert(encoded.end(), route_context->begin(), route_context->end());
    }
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(h));
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(u));
    transcript.absorb("gat.route.hidden.header", encoded);
    return zkdl::sha256(encoded);
}

inline ConstraintDescriptor sparse_route_sumcheck_descriptor(const OperationDescriptor &operation, const std::array<byte, 32U> &binding_digest) {
    ConstraintDescriptor descriptor;
    descriptor.id = operation.id + "/hidden-route/";
    static const char digits[] = "0123456789abcdef";
    for (const auto value : binding_digest) {
        descriptor.id.push_back(digits[value >> 4U]);
        descriptor.id.push_back(digits[value & 15U]);
    }
    descriptor.degree = 2U;
    descriptor.terms = {Monomial{zkdl::fr_from_u32(1U), {0U, 1U}}, Monomial{zkdl::fr_neg(zkdl::fr_from_u32(1U)), {2U, 3U}}};
    return descriptor;
}

inline std::vector<FrWire> sparse_route_public_selector(const OperationDescriptor &descriptor, const std::vector<FrWire> &edge_point,
                                                        const std::vector<FrWire> &feature_point) {
    const auto outer = public_route_selector_mle(descriptor, edge_point, descriptor.route_by_destination);
    const auto feature = public_route_feature_selector_mle(descriptor, feature_point);
    const std::size_t source_domain = std::size_t(descriptor.selector_node_domain) * descriptor.selector_source_feature_domain;
    std::vector<FrWire> selector(std::max<std::size_t>(2U, source_domain));
    for (std::size_t node = 0U; node < outer.size(); ++node)
        for (std::size_t column = 0U; column < feature.size(); ++column) selector[node * feature.size() + column] = zkdl::fr_mul(outer[node], feature[column]);
    return selector;
}

inline bool sparse_route_sumcheck_providers(
    const std::vector<FrWire> &source_values,
    const std::vector<FrWire> &selector_outer,
    const std::vector<FrWire> &selector_feature,
    const FrWire &output_value, std::uint32_t domain_size,
    std::size_t basis_size,
    std::vector<SegmentedTensorProvider> &providers,
    std::string *error) {
    std::uint32_t segment_size = 0U, segment_count = 0U;
    const std::uint64_t selector_size = static_cast<std::uint64_t>(selector_outer.size()) * selector_feature.size();
    if (source_values.size() != domain_size ||
        std::max<std::uint64_t>(2U, selector_size) != domain_size ||
        selector_outer.empty() || selector_feature.empty() ||
        !SegmentedTensorProvider::shape(domain_size, basis_size, segment_size,
                                        segment_count, error)) {
        if (error && error->empty()) *error = "sparse route virtual Sumcheck provider shape mismatch";
        return false;
    }
    providers.clear();
    providers.reserve(4U);
    const auto* outer_values = &selector_outer;
    const auto* feature_values = &selector_feature;
    providers.emplace_back(
        domain_size, segment_size,
        [outer_values, feature_values, domain_size](
            std::uint64_t offset, std::uint32_t count,
            std::vector<FrWire> &output, std::string *read_error) {
            if (offset > domain_size || count > domain_size - offset) {
                if (read_error) *read_error = "sparse route selector provider range is out of bounds";
                return false;
            }
            output.assign(count, FrWire{});
            const std::uint64_t selector_size =
                static_cast<std::uint64_t>(outer_values->size()) * feature_values->size();
            for (std::uint32_t index = 0U; index < count; ++index) {
                const std::uint64_t global = offset + index;
                if (global < selector_size) {
                    const std::size_t node = static_cast<std::size_t>(global / feature_values->size());
                    const std::size_t feature = static_cast<std::size_t>(global % feature_values->size());
                    output[index] = zkdl::fr_mul((*outer_values)[node], (*feature_values)[feature]);
                }
            }
            return true;
        });
    SegmentedTensorProvider source_provider;
    if (!SegmentedTensorProvider::from_values(source_values, basis_size,
                                               source_provider, error)) return false;
    providers.push_back(std::move(source_provider));
    providers.emplace_back(
        domain_size, segment_size,
        [domain_size](std::uint64_t offset, std::uint32_t count,
                      std::vector<FrWire> &output, std::string *read_error) {
            if (offset > domain_size || count > domain_size - offset) {
                if (read_error) *read_error = "sparse route zero provider range is out of bounds";
                return false;
            }
            output.assign(count, FrWire{});
            if (offset == 0U && count != 0U) output[0] = zkdl::fr_from_u32(1U);
            return true;
        });
    providers.emplace_back(
        domain_size, segment_size,
        [domain_size, output_value](std::uint64_t offset, std::uint32_t count,
                                    std::vector<FrWire> &output,
                                    std::string *read_error) {
            if (offset > domain_size || count > domain_size - offset) {
                if (read_error) *read_error = "sparse route output provider range is out of bounds";
                return false;
            }
            output.assign(count, output_value);
            return true;
        });
    (void)segment_count;
    return true;
}

inline std::array<byte, 32U> sparse_route_terminal_digest(const std::array<byte, 32U> &header_digest, const SparseRouteProof &route) {
    std::vector<byte> encoded(header_digest.begin(), header_digest.end());
    append_native_fr_vector(encoded, route.edge_point);
    append_native_fr_vector(encoded, route.feature_point);
    append_native_fr_vector(encoded, route.source_feature_point);
    append_native_fr_vector(encoded, route.node_point);
    zkdl::append_g1(encoded, route.sumcheck.final_claim_commitment);
    return zkdl::sha256(encoded);
}

inline zkdl::PedersenProofContext sparse_route_terminal_context(const std::array<byte, 32U> &digest, const RelationStatement &statement,
                                                                const zkdl::CpuG1Point &h, const zkdl::CpuG1Point &u) {
    return zksc_make_context(zkdl::PedersenProofContext::kOperationScalarEquality, 1U, 0U, kZkTerminalResidualIndex, kZkTerminalExternalIndex,
                             zkdl::cpu_ipa_basis_digest({}, h, u, 0U), digest, statement);
}

inline void sparse_route_absorb_terminal(zkdl::Transcript &transcript, const SparseRouteProof &route) {
    std::vector<byte> encoded;
    zkdl::serialize_hyrax_opening_v3(route.source_opening, encoded);
    zkdl::serialize_hyrax_opening_v3(route.output_opening, encoded);
    zkdl::serialize_pedersen_scalar_equality(route.terminal_binding, encoded);
    transcript.absorb("gat.route.hidden.terminal", encoded);
}

inline bool sparse_route_aggregate_opening_inputs(const TensorWitness &tensor, const std::vector<FrWire> &point,
                                                  const std::vector<zkdl::CpuG1Point> &generators, const zkdl::CpuG1Point &ipa_blinding_generator,
                                                  const zkdl::CpuG1Point &ipa_u_generator,
                                                  const std::vector<CommitmentWire> &commitments, std::uint32_t commitment_index,
                                                  const std::map<std::string, FrWire> &blinders,
                                                  const std::map<std::string, std::vector<FrWire>> *segment_blinding_scalars,
                                                  std::vector<FrWire> &aggregate_values, FrWire &aggregate_blinder, FrWire &aggregate_value,
                                                  zkdl::CpuG1Point &aggregate_xi, std::string *error,
                                                  const std::vector<FrWire> *validated_values = nullptr,
                                                  const RouteWorkProgressCallback* progress = nullptr,
                                                  const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
                                                  zkdl::CpuG1MsmSession* msm_session = nullptr,
                                                  const SegmentCommitmentValidationReceipt* validation_receipt = nullptr) {
    if (commitment_index >= commitments.size() || commitments[commitment_index].tensor_id != tensor.id) {
        if (error) *error = "sparse route aggregate commitment index is not canonical";
        return false;
    }
    SegmentedTensorProvider provider;
    if (validated_values != nullptr) {
        if (validated_values->size() != tensor.domain_size ||
            !SegmentedTensorProvider::from_values(*validated_values, generators.size(), provider, error)) {
            if (error && error->empty()) *error = "sparse route validated tensor shape mismatch";
            return false;
        }
    } else if (!tensor.segmented(generators.size(), provider, error)) return false;
    ZkRelationSegmentLayout layout;
    if (!zktr_segment_layout(provider.logical_domain(), generators.size(), layout, error) || point.size() != log2_exact(layout.logical_domain)) return false;
    SegmentCommitmentMaterial material;
    if (segment_blinding_scalars != nullptr) {
        if (!zktr_material_from_wire(commitments[commitment_index], *segment_blinding_scalars, material, error)) return false;
    } else if (layout.segment_count == 1U) {
        material.blinders = {relation_blinding_scalar(tensor, blinders)};
        material.commitments = {zkdl::cpu_g1_from_native(commitments[commitment_index].commitment)};
    } else {
        if (error) *error = "sparse route lacks authenticated multi-segment material";
        return false;
    }
    if (material.blinders.size() != layout.segment_count || material.commitments.size() != layout.segment_count) {
        if (error) *error = "sparse route segment material count mismatch";
        return false;
    }
    if (validation_receipt != nullptr &&
        !validation_receipt->matches_context(
            tensor.id, tensor.visibility, provider, generators,
            ipa_blinding_generator, ipa_u_generator, authenticated_basis,
            commitments[commitment_index], material, error))
        return false;
    const std::vector<FrWire> low_point = zktr_segment_point(point, layout.segment_size);
    const std::vector<FrWire> high_point = zktr_segment_prefix_point(point, layout.segment_size);
    const std::vector<FrWire> high_weights = zkdl::cpu_ipa_mle_weights(high_point);
    if (high_weights.size() != layout.segment_count) {
        if (error) *error = "sparse route high-point segment weight shape mismatch";
        return false;
    }
    for (const auto &weight : high_weights) {
        if (zkdl::fr_compare(weight, FrWire{}) == 0) {
            if (error) *error = "sparse route high-point segment weight is zero";
            return false;
        }
    }
    const std::vector<zkdl::CpuG1Point> segment_generators(generators.begin(), generators.begin() + layout.segment_size);
    aggregate_values.assign(layout.segment_size, FrWire{});
    aggregate_blinder = FrWire{};
    aggregate_xi = zkdl::CpuG1Point::zero();
    for (std::uint32_t segment = 0U; segment < layout.segment_count; ++segment) {
        std::vector<FrWire> values;
        if (validated_values != nullptr) {
            const std::size_t offset = static_cast<std::size_t>(segment) * layout.segment_size;
            if (offset > validated_values->size() ||
                layout.segment_size > validated_values->size() - offset) {
                if (error) *error = "sparse route validated tensor range is out of bounds";
                return false;
            }
            values.assign(validated_values->begin() + static_cast<std::ptrdiff_t>(offset),
                          validated_values->begin() + static_cast<std::ptrdiff_t>(offset + layout.segment_size));
        } else if (!provider.read_segment(segment, values, error) || values.size() != layout.segment_size) return false;
        if (validation_receipt != nullptr &&
            !validation_receipt->matches_segment(segment, values, material, error))
            return false;
        for (std::size_t index = 0U; index < values.size(); ++index)
            aggregate_values[index] = zkdl::fr_add(aggregate_values[index], zkdl::fr_mul(high_weights[segment], values[index]));
        aggregate_blinder = zkdl::fr_add(aggregate_blinder, zkdl::fr_mul(high_weights[segment], material.blinders[segment]));
        aggregate_xi = zkdl::cpu_g1_add(aggregate_xi, zkdl::cpu_g1_mul(material.commitments[segment], high_weights[segment]));
        if (progress != nullptr)
            (*progress)("opening_aggregate_segments", segment + 1U,
                        layout.segment_count);
    }
    if (validation_receipt == nullptr) {
        const zkdl::CpuG1Point aggregate_generator_sum =
            authenticated_basis != nullptr && msm_session != nullptr
                ? msm_session->sum(*authenticated_basis, generators,
                                   ipa_blinding_generator,
                                   ipa_u_generator, aggregate_values.data(),
                                   layout.segment_size)
                : zkdl::cpu_g1_sum(segment_generators, aggregate_values);
        const zkdl::CpuG1Point recomputed_xi = zkdl::cpu_g1_add(
            aggregate_generator_sum,
            zkdl::cpu_g1_mul(ipa_blinding_generator, aggregate_blinder));
        if (!zkdl::cpu_g1_equal(recomputed_xi, aggregate_xi)) {
            if (error) *error = "sparse route aggregate material does not match witness values";
            return false;
        }
    }
    aggregate_value = zkdl::cpu_ipa_inner_product(aggregate_values, zkdl::cpu_ipa_mle_weights(low_point));
    return true;
}

inline bool prove_sparse_route(const OperationDescriptor &descriptor, const std::map<std::string, const TensorWitness *> &tensor_map,
                               const std::map<std::string, FrWire> &blinders, const std::vector<zkdl::CpuG1Point> &generators,
                               const zkdl::CpuG1Point &ipa_blinding_generator, const zkdl::CpuG1Point &ipa_u_generator,
                               const std::vector<CommitmentWire> &commitments, const RelationStatement &statement, zkdl::Transcript &transcript,
                               SparseRouteProof &route, std::string *error,
                               const std::map<std::string, std::vector<FrWire>> *segment_blinding_scalars,
                               const zkdl::CpuIpaBasisContext* authenticated_basis,
                               zkdl::CpuG1MsmSession* msm_session,
                               const std::map<std::string, SegmentCommitmentValidationReceipt>* validation_receipts) {
    try {
        RouteSumcheckProgressReporter route_work_reporter{descriptor.id, 0U};
        const RouteWorkProgressCallback route_work_progress =
            [&route_work_reporter](const char* phase, std::uint64_t completed,
                                  std::uint64_t total) {
                route_work_reporter.update_route_work(phase, completed, total);
            };
        const TensorWitness *source = nullptr;
        const TensorWitness *output = nullptr;
        std::vector<FrWire> source_values;
        route = SparseRouteProof{};
        if (!route_tensor_shapes_valid(descriptor, tensor_map, source, output, error) || !source->metadata_valid(error) || !output->metadata_valid(error) ||
            !sparse_route_public_inputs_valid(descriptor, commitments, generators, statement, route.source_commitment_index, route.output_commitment_index,
                                              error) ||
            !relation_segment_commitments_valid(statement, commitments, error) ||
            !source->materialize_validated(source_values, error,
                                           &route_work_progress))
            return false;
        route.edge_domain = descriptor.selector_edge_domain;
        route.node_domain = descriptor.selector_node_domain;
        route.feature_domain = descriptor.selector_feature_domain;
        route.source_feature_domain = descriptor.selector_source_feature_domain;
        const auto header_digest = sparse_route_absorb_header(descriptor, statement, commitments, route.source_commitment_index, route.output_commitment_index,
                                                              ipa_blinding_generator, ipa_u_generator, transcript);
        for (std::uint32_t bit = 0U; bit < log2_exact(route.edge_domain); ++bit)
            route.edge_point.push_back(transcript.challenge("gat.route." + descriptor.id + ".edge." + std::to_string(bit)));
        for (std::uint32_t bit = 0U; bit < log2_exact(route.feature_domain); ++bit)
            route.feature_point.push_back(transcript.challenge("gat.route." + descriptor.id + ".feature." + std::to_string(bit)));
        const auto edge_point = sparse_route_point(route.edge_point);
        const auto feature_point = sparse_route_point(route.feature_point);
        std::vector<FrWire> output_fold_point = feature_point;
        output_fold_point.insert(output_fold_point.end(), edge_point.begin(), edge_point.end());
        const std::vector<FrWire> output_point = sparse_route_hyrax_point(output_fold_point);
        std::vector<FrWire> output_values;
        FrWire output_aggregate_blinder{}, output_value{};
        zkdl::CpuG1Point output_xi;
        const SegmentCommitmentValidationReceipt* output_receipt = nullptr;
        if (validation_receipts != nullptr) {
            const auto found = validation_receipts->find(output->id);
            if (found != validation_receipts->end()) output_receipt = &found->second;
        }
        if (!sparse_route_aggregate_opening_inputs(*output, output_point, generators, ipa_blinding_generator, ipa_u_generator, commitments, route.output_commitment_index,
                                                   blinders, segment_blinding_scalars, output_values, output_aggregate_blinder, output_value, output_xi, error,
                                                   nullptr, &route_work_progress,
                                                   authenticated_basis, msm_session,
                                                   output_receipt))
            return false;
        const std::uint32_t domain = std::max(2U, source->domain_size);
        const bool duplicate_singleton = source_values.size() == 1U;
        if (duplicate_singleton) source_values.push_back(source_values.front());
        const auto constraint = sparse_route_sumcheck_descriptor(descriptor, header_digest);
        ZkSumcheckProverResult sumcheck_result;
        const auto selector_outer = public_route_selector_mle(
            descriptor, edge_point, descriptor.route_by_destination,
            &route_work_progress);
        const auto selector_feature = public_route_feature_selector_mle(
            descriptor, feature_point, &route_work_progress);
        std::vector<SegmentedTensorProvider> sumcheck_providers;
        if (!sparse_route_sumcheck_providers(
                source_values, selector_outer, selector_feature, output_value,
                domain, generators.size(), sumcheck_providers, error) ||
            !prove_zk_sumcheck_segmented(constraint, sumcheck_providers,
                               2U, domain, generators, ipa_blinding_generator, ipa_u_generator, FrWire{}, statement, constraint.id, transcript, sumcheck_result,
                               error, 0U, true))
            return false;
        sumcheck_providers.clear();
        if (duplicate_singleton) source_values.pop_back();
        route.sumcheck = std::move(sumcheck_result.proof);
        const std::uint32_t source_feature_bits = log2_exact(route.source_feature_domain);
        const std::uint32_t source_bits = log2_exact(source->domain_size);
        route.source_feature_point.assign(sumcheck_result.challenges.begin(), sumcheck_result.challenges.begin() + source_feature_bits);
        route.node_point.assign(sumcheck_result.challenges.begin() + source_feature_bits, sumcheck_result.challenges.begin() + source_bits);
        std::vector<FrWire> source_fold_point = sparse_route_point(route.source_feature_point);
        const auto node_point = sparse_route_point(route.node_point);
        source_fold_point.insert(source_fold_point.end(), node_point.begin(), node_point.end());
        const std::vector<FrWire> source_point = sparse_route_hyrax_point(source_fold_point);
        std::vector<FrWire> source_values_aggregate;
        FrWire source_aggregate_blinder{}, source_value_aggregate{};
        zkdl::CpuG1Point source_xi;
        const SegmentCommitmentValidationReceipt* source_receipt = nullptr;
        if (validation_receipts != nullptr) {
            const auto found = validation_receipts->find(source->id);
            if (found != validation_receipts->end()) source_receipt = &found->second;
        }
        if (!sparse_route_aggregate_opening_inputs(*source, source_point, generators, ipa_blinding_generator, ipa_u_generator, commitments, route.source_commitment_index,
                                                   blinders, segment_blinding_scalars, source_values_aggregate, source_aggregate_blinder,
                                                   source_value_aggregate, source_xi, error, &source_values,
                                                   &route_work_progress,
                                                   authenticated_basis, msm_session,
                                                   source_receipt))
            return false;
        const auto terminal_digest = sparse_route_terminal_digest(header_digest, route);
        const auto source_metadata = zktr_make_opening_metadata(terminal_digest, statement, source_point, route.source_commitment_index,
                                                                zktr_terminal_evaluation_index(0U));
        const auto output_metadata = zktr_make_opening_metadata(terminal_digest, statement, output_point, route.output_commitment_index,
                                                                zktr_terminal_evaluation_index(1U));
        ZkRelationSegmentLayout source_layout, output_layout;
        if (!zktr_segment_layout(source->domain_size, generators.size(), source_layout, error) ||
            !zktr_segment_layout(output->domain_size, generators.size(), output_layout, error))
            return false;
        const std::vector<zkdl::CpuG1Point> source_generators(generators.begin(), generators.begin() + source_layout.segment_size);
        const std::vector<zkdl::CpuG1Point> output_generators(generators.begin(), generators.begin() + output_layout.segment_size);
        auto source_segment_metadata = source_metadata;
        source_segment_metadata.logical_domain = source_layout.logical_domain;
        source_segment_metadata.segment_index = 0U;
        source_segment_metadata.logical_offset = 0U;
        source_segment_metadata.actual_length = source_layout.segment_size;
        source_segment_metadata.padded_length = source_layout.segment_size;
        auto output_segment_metadata = output_metadata;
        output_segment_metadata.logical_domain = output_layout.logical_domain;
        output_segment_metadata.segment_index = 0U;
        output_segment_metadata.logical_offset = 0U;
        output_segment_metadata.actual_length = output_layout.segment_size;
        output_segment_metadata.padded_length = output_layout.segment_size;
        const FrWire source_blinder = zkdl::cpu_fresh_csprng_scalar();
        const FrWire output_blinder = zkdl::cpu_fresh_csprng_scalar();
        const FrWire source_value = sumcheck_result.folded_terminal_values[1U];
        if (source_value != source_value_aggregate) {
            if (error) *error = "sparse route source terminal value disagrees with segment aggregate";
            return false;
        }
        const auto source_tau = zkdl::cpu_pedersen_scalar_commit(source_value_aggregate, source_blinder, ipa_blinding_generator, ipa_u_generator);
        const auto output_tau = zkdl::cpu_pedersen_scalar_commit(output_value, output_blinder, ipa_blinding_generator, ipa_u_generator);
        route.source_opening = authenticated_basis != nullptr
            ? zkdl::cpu_open_hyrax_v3_authenticated(
                  *authenticated_basis, source_values_aggregate,
                  zktr_segment_point(source_point, source_layout.segment_size),
                  source_aggregate_blinder, source_blinder, source_tau,
                  source_segment_metadata, source_xi, msm_session).proof
            : zkdl::cpu_open_hyrax_v3(
                  source_generators, source_values_aggregate,
                  zktr_segment_point(source_point, source_layout.segment_size),
                  ipa_blinding_generator, ipa_u_generator, source_aggregate_blinder,
                  source_blinder, source_tau, source_segment_metadata).proof;
        route.output_opening = authenticated_basis != nullptr
            ? zkdl::cpu_open_hyrax_v3_authenticated(
                  *authenticated_basis, output_values,
                  zktr_segment_point(output_point, output_layout.segment_size),
                  output_aggregate_blinder, output_blinder, output_tau,
                  output_segment_metadata, output_xi, msm_session).proof
            : zkdl::cpu_open_hyrax_v3(
                  output_generators, output_values,
                  zktr_segment_point(output_point, output_layout.segment_size),
                  ipa_blinding_generator, ipa_u_generator, output_aggregate_blinder,
                  output_blinder, output_tau, output_segment_metadata).proof;
        if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(route.source_opening.xi), source_xi) ||
            !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(route.output_opening.xi), output_xi)) {
            if (error) *error = "sparse route witness is not the weighted canonical segment commitment";
            return false;
        }
        const FrWire p = sumcheck_result.folded_terminal_values[0U];
        const FrWire q0 = sumcheck_result.folded_terminal_values[2U];
        const FrWire residual = zkdl::fr_sub(zkdl::fr_mul(p, source_value), zkdl::fr_mul(q0, output_value));
        const FrWire residual_blinder = zkdl::fr_sub(zkdl::fr_mul(p, source_blinder), zkdl::fr_mul(q0, output_blinder));
        const auto residual_commitment = zkdl::cpu_g1_add(zkdl::cpu_g1_mul(source_tau, p), zkdl::cpu_g1_mul(output_tau, zkdl::fr_neg(q0)));
        route.terminal_binding = zkdl::cpu_open_pedersen_scalar_equality(residual, residual_blinder, sumcheck_result.final_q_blinder, residual_commitment,
                                                                         zkdl::cpu_g1_from_native(route.sumcheck.final_claim_commitment),
                                                                         ipa_blinding_generator, ipa_u_generator,
                                                                         sparse_route_terminal_context(terminal_digest, statement, ipa_blinding_generator,
                                                                                                       ipa_u_generator))
                                     .proof;
        sparse_route_absorb_terminal(transcript, route);
        return true;
    } catch (const std::exception &exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool verify_sparse_route(const OperationDescriptor &descriptor, const std::vector<CommitmentWire> &commitments,
                                const std::vector<zkdl::CpuG1Point> &generators, const zkdl::CpuG1Point &ipa_blinding_generator,
                                const zkdl::CpuG1Point &ipa_u_generator, const RelationStatement &statement, zkdl::Transcript &transcript,
                                const SparseRouteProof &route, std::string *error,
                                const zkdl::CpuIpaBasisContext* authenticated_basis,
                                zkdl::CpuG1MsmSession* msm_session) {
    try {
        std::uint32_t source_index = 0U, output_index = 0U;
        if (!sparse_route_public_inputs_valid(descriptor, commitments, generators, statement, source_index, output_index, error) ||
            !relation_segment_commitments_valid(statement, commitments, error))
            return false;
        const std::uint32_t source_domain = commitments[source_index].domain_size;
        const std::uint32_t output_domain = commitments[output_index].domain_size;
        const std::uint32_t source_feature_bits = log2_exact(descriptor.selector_source_feature_domain);
        if (route.version != SparseRouteProof::kVersion || route.source_commitment_index != source_index || route.output_commitment_index != output_index ||
            route.edge_domain != descriptor.selector_edge_domain || route.node_domain != descriptor.selector_node_domain ||
            route.feature_domain != descriptor.selector_feature_domain || route.source_feature_domain != descriptor.selector_source_feature_domain ||
            route.edge_point.size() != log2_exact(route.edge_domain) || route.feature_point.size() != log2_exact(route.feature_domain) ||
            route.source_feature_point.size() != source_feature_bits || route.node_point.size() != log2_exact(route.node_domain)) {
            if (error) *error = "sparse route wire does not match the public descriptor";
            return false;
        }
        const auto header_digest = sparse_route_absorb_header(descriptor, statement, commitments, source_index, output_index, ipa_blinding_generator,
                                                              ipa_u_generator, transcript);
        for (std::size_t bit = 0U; bit < route.edge_point.size(); ++bit)
            if (fr_compare_mixed(route.edge_point[bit], transcript.challenge("gat.route." + descriptor.id + ".edge." + std::to_string(bit))) != 0) {
                if (error) *error = "sparse route edge challenge mismatch";
                return false;
            }
        for (std::size_t bit = 0U; bit < route.feature_point.size(); ++bit)
            if (fr_compare_mixed(route.feature_point[bit], transcript.challenge("gat.route." + descriptor.id + ".feature." + std::to_string(bit))) != 0) {
                if (error) *error = "sparse route feature challenge mismatch";
                return false;
            }
        const auto constraint = sparse_route_sumcheck_descriptor(descriptor, header_digest);
        ZkSumcheckVerifierResult sumcheck_result;
        if (!verify_zk_sumcheck(route.sumcheck, constraint, 2U, std::max(2U, source_domain), generators, ipa_blinding_generator, ipa_u_generator, FrWire{},
                                statement, constraint.id, transcript, sumcheck_result, error))
            return false;
        std::vector<FrWire> source_fold_point;
        for (std::size_t bit = 0U; bit < log2_exact(source_domain); ++bit) {
            const auto &coordinate = bit < source_feature_bits ? route.source_feature_point[bit] : route.node_point[bit - source_feature_bits];
            if (fr_compare_mixed(coordinate, sumcheck_result.challenges[bit]) != 0) {
                if (error) *error = "sparse route source challenge mismatch";
                return false;
            }
            source_fold_point.push_back(zkdl::fr_wire(coordinate));
        }
        const std::vector<FrWire> source_point = sparse_route_hyrax_point(source_fold_point);
        const auto edge_point = sparse_route_point(route.edge_point);
        const auto feature_point = sparse_route_point(route.feature_point);
        std::vector<FrWire> output_fold_point = feature_point;
        output_fold_point.insert(output_fold_point.end(), edge_point.begin(), edge_point.end());
        const std::vector<FrWire> output_point = sparse_route_hyrax_point(output_fold_point);
        ZkRelationSegmentLayout source_layout, output_layout;
        if (!zktr_segment_layout(source_domain, generators.size(), source_layout, error) ||
            !zktr_segment_layout(output_domain, generators.size(), output_layout, error))
            return false;
        const auto weighted_commitment = [&](std::uint32_t commitment_index, const std::vector<FrWire> &point, const ZkRelationSegmentLayout &layout,
                                             zkdl::CpuG1Point &weighted) {
            const auto &wire = commitments[commitment_index];
            if (wire.segment_count != layout.segment_count || wire.segment_size != layout.segment_size ||
                wire.segment_commitments.size() != layout.segment_count) {
                if (error) *error = "sparse route statement segment layout mismatch";
                return false;
            }
            const std::vector<FrWire> high_point = zktr_segment_prefix_point(point, layout.segment_size);
            const std::vector<FrWire> weights = zkdl::cpu_ipa_mle_weights(high_point);
            if (weights.size() != layout.segment_count) {
                if (error) *error = "sparse route statement segment weight shape mismatch";
                return false;
            }
            weighted = zkdl::CpuG1Point::zero();
            for (std::size_t index = 0U; index < weights.size(); ++index) {
                if (zkdl::fr_compare(weights[index], FrWire{}) == 0) {
                    if (error) *error = "sparse route statement segment weight is zero";
                    return false;
                }
                weighted = zkdl::cpu_g1_add(weighted, zkdl::cpu_g1_mul(zkdl::cpu_g1_from_native(wire.segment_commitments[index].commitment), weights[index]));
            }
            return true;
        };
        zkdl::CpuG1Point source_xi, output_xi;
        if (!weighted_commitment(source_index, source_point, source_layout, source_xi) ||
            !weighted_commitment(output_index, output_point, output_layout, output_xi))
            return false;
        const auto terminal_digest = sparse_route_terminal_digest(header_digest, route);
        const auto source_tau = zkdl::cpu_g1_from_native(route.source_opening.tau);
        const auto output_tau = zkdl::cpu_g1_from_native(route.output_opening.tau);
        auto source_metadata = zktr_make_opening_metadata(terminal_digest, statement, source_point, source_index, zktr_terminal_evaluation_index(0U));
        source_metadata.logical_domain = source_layout.logical_domain;
        source_metadata.segment_index = 0U;
        source_metadata.logical_offset = 0U;
        source_metadata.actual_length = source_layout.segment_size;
        source_metadata.padded_length = source_layout.segment_size;
        auto output_metadata = zktr_make_opening_metadata(terminal_digest, statement, output_point, output_index, zktr_terminal_evaluation_index(1U));
        output_metadata.logical_domain = output_layout.logical_domain;
        output_metadata.segment_index = 0U;
        output_metadata.logical_offset = 0U;
        output_metadata.actual_length = output_layout.segment_size;
        output_metadata.padded_length = output_layout.segment_size;
        const auto verify_route_opening = [&](const zkdl::HyraxOpeningProofV3& opening,
                                              const zkdl::CpuG1Point& xi,
                                              const zkdl::CpuG1Point& tau,
                                              const std::vector<FrWire>& point,
                                              const zkdl::CpuIpaOpeningMetadata& metadata,
                                              std::uint32_t generator_count) {
            if (authenticated_basis != nullptr)
                return zkdl::cpu_verify_hyrax_v3_authenticated(
                    opening, xi, tau, *authenticated_basis, point, metadata,
                    error, msm_session);
            const std::vector<zkdl::CpuG1Point> prefix(
                generators.begin(), generators.begin() + generator_count);
            return zkdl::cpu_verify_hyrax_v3(
                opening, xi, tau, prefix, point, ipa_blinding_generator,
                ipa_u_generator, metadata, error);
        };
        if (!verify_route_opening(
                route.source_opening, source_xi, source_tau,
                zktr_segment_point(source_point, source_layout.segment_size),
                source_metadata, source_layout.segment_size) ||
            !verify_route_opening(
                route.output_opening, output_xi, output_tau,
                zktr_segment_point(output_point, output_layout.segment_size),
                output_metadata, output_layout.segment_size))
            return false;
        auto selector = sparse_route_public_selector(descriptor, edge_point, feature_point);
        FrWire q0 = zkdl::fr_from_u32(1U);
        for (const auto &challenge : sumcheck_result.challenges) {
            const auto coordinate = zkdl::fr_wire(challenge);
            fold_values(selector, coordinate);
            q0 = zkdl::fr_mul(q0, zkdl::fr_sub(zkdl::fr_from_u32(1U), coordinate));
        }
        const auto residual_commitment = zkdl::cpu_g1_add(zkdl::cpu_g1_mul(source_tau, selector.front()), zkdl::cpu_g1_mul(output_tau, zkdl::fr_neg(q0)));
        if (!zkdl::cpu_verify_pedersen_scalar_equality(route.terminal_binding, residual_commitment,
                                                       zkdl::cpu_g1_from_native(sumcheck_result.final_claim_commitment), ipa_blinding_generator,
                                                       ipa_u_generator,
                                                       sparse_route_terminal_context(terminal_digest, statement, ipa_blinding_generator, ipa_u_generator),
                                                       error))
            return false;
        sparse_route_absorb_terminal(transcript, route);
        return true;
    } catch (const std::exception &exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool sparse_route_feature_rlc_feature(
    const std::vector<OperationDescriptor>& members,
    const std::vector<FrWire>& coefficients,
    const std::vector<FrWire>& feature_point,
    std::vector<FrWire>& feature_star, std::string* error) {
    if (members.size() < 2U || members.size() != coefficients.size()) {
        if (error) *error = "sparse route feature-RLC member shape mismatch";
        return false;
    }
    const auto& first = members.front();
    feature_star.assign(first.selector_source_feature_domain, FrWire{});
    for (std::size_t member = 0U; member < members.size(); ++member) {
        const auto& descriptor = members[member];
        if (descriptor.selector_feature_domain != first.selector_feature_domain ||
            descriptor.selector_source_feature_domain !=
                first.selector_source_feature_domain ||
            descriptor.selector_node_domain != first.selector_node_domain ||
            descriptor.selector_edge_domain != first.selector_edge_domain) {
            if (error) *error = "sparse route feature-RLC member layout mismatch";
            return false;
        }
        std::vector<FrWire> feature;
        try {
            feature = public_route_feature_selector_mle(
                descriptor, feature_point);
        } catch (const std::exception& exception) {
            if (error) *error = exception.what();
            return false;
        }
        if (feature.size() != feature_star.size()) {
            if (error) *error = "sparse route feature-RLC feature vector mismatch";
            return false;
        }
        for (std::size_t index = 0U; index < feature.size(); ++index)
            feature_star[index] = zkdl::fr_add(
                feature_star[index],
                zkdl::fr_mul(coefficients[member], feature[index]));
    }
    return true;
}

inline std::vector<FrWire> sparse_route_selector_from_factors(
    const OperationDescriptor& descriptor,
    const std::vector<FrWire>& selector_outer,
    const std::vector<FrWire>& selector_feature) {
    const std::size_t source_domain =
        std::size_t(descriptor.selector_node_domain) *
        descriptor.selector_source_feature_domain;
    std::vector<FrWire> selector(std::max<std::size_t>(2U, source_domain));
    for (std::size_t node = 0U; node < selector_outer.size(); ++node)
        for (std::size_t column = 0U; column < selector_feature.size(); ++column)
            selector[node * selector_feature.size() + column] =
                zkdl::fr_mul(selector_outer[node], selector_feature[column]);
    return selector;
}

inline bool prove_sparse_route_feature_rlc(
    const OperationDescriptor& descriptor,
    const std::vector<OperationDescriptor>& member_descriptors,
    const std::vector<FrWire>& coefficients,
    const std::vector<byte>& group_context,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, SparseRouteProof& route,
    std::string* error,
    const std::map<std::string, std::vector<FrWire>>* segment_blinding_scalars,
    const zkdl::CpuIpaBasisContext* authenticated_basis,
    zkdl::CpuG1MsmSession* msm_session,
    const std::map<std::string, SegmentCommitmentValidationReceipt>*
        validation_receipts) {
    try {
        RouteSumcheckProgressReporter route_work_reporter{descriptor.id, 0U};
        const RouteWorkProgressCallback route_work_progress =
            [&route_work_reporter](const char* phase, std::uint64_t completed,
                                  std::uint64_t total) {
                route_work_reporter.update_route_work(phase, completed, total);
            };
        const TensorWitness* source = nullptr;
        const TensorWitness* anchor_output = nullptr;
        std::vector<FrWire> source_values;
        route = SparseRouteProof{};
        if (member_descriptors.size() < 2U ||
            member_descriptors.size() != coefficients.size() ||
            !route_tensor_shapes_valid(descriptor, tensor_map, source,
                                       anchor_output, error) ||
            !source->metadata_valid(error) || !anchor_output->metadata_valid(error) ||
            !sparse_route_public_inputs_valid(
                descriptor, commitments, generators, statement,
                route.source_commitment_index, route.output_commitment_index,
                error) ||
            !relation_segment_commitments_valid(statement, commitments, error) ||
            !source->materialize_validated(source_values, error,
                                           &route_work_progress))
            return false;
        for (std::size_t member = 0U; member < member_descriptors.size(); ++member) {
            const auto& member_descriptor = member_descriptors[member];
            if (member_descriptor.selector_source_ref != descriptor.selector_source_ref ||
                member_descriptor.selector_node_domain != descriptor.selector_node_domain ||
                member_descriptor.selector_source_feature_domain !=
                    descriptor.selector_source_feature_domain ||
                member_descriptor.selector_edge_domain != descriptor.selector_edge_domain ||
                member_descriptor.selector_feature_domain != descriptor.selector_feature_domain) {
                if (error) *error = "FAC feature-RLC route member descriptor mismatch";
                return false;
            }
            const auto output = tensor_map.find(member_descriptor.selector_output_ref);
            if (output == tensor_map.end() || output->second == nullptr ||
                !output->second->metadata_valid(error) ||
                output->second->domain_size != anchor_output->domain_size) {
                if (error && error->empty())
                    *error = "FAC feature-RLC target tensor is missing or layout-incompatible";
                return false;
            }
        }
        route.edge_domain = descriptor.selector_edge_domain;
        route.node_domain = descriptor.selector_node_domain;
        route.feature_domain = descriptor.selector_feature_domain;
        route.source_feature_domain = descriptor.selector_source_feature_domain;
        const auto header_digest = sparse_route_absorb_header(
            descriptor, statement, commitments, route.source_commitment_index,
            route.output_commitment_index, ipa_blinding_generator,
            ipa_u_generator, transcript, &group_context);
        for (std::uint32_t bit = 0U; bit < log2_exact(route.edge_domain); ++bit)
            route.edge_point.push_back(transcript.challenge(
                "gat.route." + descriptor.id + ".edge." + std::to_string(bit)));
        for (std::uint32_t bit = 0U; bit < log2_exact(route.feature_domain); ++bit)
            route.feature_point.push_back(transcript.challenge(
                "gat.route." + descriptor.id + ".feature." + std::to_string(bit)));
        const auto edge_point = sparse_route_point(route.edge_point);
        const auto feature_point = sparse_route_point(route.feature_point);
        std::vector<FrWire> output_fold_point = feature_point;
        output_fold_point.insert(output_fold_point.end(), edge_point.begin(),
                                 edge_point.end());
        const std::vector<FrWire> output_point =
            sparse_route_hyrax_point(output_fold_point);
        const auto output_layout = [&]() {
            ZkRelationSegmentLayout layout;
            std::string layout_error;
            if (!zktr_segment_layout(anchor_output->domain_size,
                                     generators.size(), layout, &layout_error))
                throw std::invalid_argument(layout_error);
            return layout;
        }();
        std::vector<FrWire> output_values;
        FrWire output_aggregate_blinder{}, output_value{};
        zkdl::CpuG1Point output_xi = zkdl::CpuG1Point::zero();
        for (std::size_t member = 0U; member < member_descriptors.size(); ++member) {
            const auto& member_descriptor = member_descriptors[member];
            const auto output = tensor_map.find(member_descriptor.selector_output_ref);
            std::uint32_t output_index = 0U;
            if (output == tensor_map.end() || output->second == nullptr ||
                !route_commitment_index(commitments,
                                        member_descriptor.selector_output_ref,
                                        output_index, error)) return false;
            const SegmentCommitmentValidationReceipt* output_receipt = nullptr;
            if (validation_receipts != nullptr) {
                const auto found = validation_receipts->find(
                    member_descriptor.selector_output_ref);
                if (found != validation_receipts->end())
                    output_receipt = &found->second;
            }
            std::vector<FrWire> member_values;
            FrWire member_blinder{}, member_value{};
            zkdl::CpuG1Point member_xi;
            if (!sparse_route_aggregate_opening_inputs(
                    *output->second, output_point, generators,
                    ipa_blinding_generator, ipa_u_generator, commitments,
                    output_index, blinders, segment_blinding_scalars,
                    member_values, member_blinder, member_value, member_xi,
                    error, nullptr, &route_work_progress, authenticated_basis,
                    msm_session, output_receipt)) return false;
            if (member_values.size() != output_layout.segment_size) {
                if (error) *error = "FAC feature-RLC target segment shape mismatch";
                return false;
            }
            if (output_values.empty())
                output_values.assign(member_values.size(), FrWire{});
            for (std::size_t index = 0U; index < member_values.size(); ++index)
                output_values[index] = zkdl::fr_add(
                    output_values[index],
                    zkdl::fr_mul(coefficients[member], member_values[index]));
            output_aggregate_blinder = zkdl::fr_add(
                output_aggregate_blinder,
                zkdl::fr_mul(coefficients[member], member_blinder));
            output_value = zkdl::fr_add(
                output_value,
                zkdl::fr_mul(coefficients[member], member_value));
            output_xi = zkdl::cpu_g1_add(
                output_xi,
                zkdl::cpu_g1_mul(member_xi, coefficients[member]));
        }
        const std::uint32_t domain = std::max(2U, source->domain_size);
        const bool duplicate_singleton = source_values.size() == 1U;
        if (duplicate_singleton) source_values.push_back(source_values.front());
        const auto constraint = sparse_route_sumcheck_descriptor(
            descriptor, header_digest);
        ZkSumcheckProverResult sumcheck_result;
        const auto selector_outer = public_route_selector_mle(
            descriptor, edge_point, descriptor.route_by_destination,
            &route_work_progress);
        std::vector<FrWire> selector_feature;
        if (!sparse_route_feature_rlc_feature(
                member_descriptors, coefficients, feature_point,
                selector_feature, error)) return false;
        std::vector<SegmentedTensorProvider> sumcheck_providers;
        if (!sparse_route_sumcheck_providers(
                source_values, selector_outer, selector_feature, output_value,
                domain, generators.size(), sumcheck_providers, error) ||
            !prove_zk_sumcheck_segmented(
                constraint, sumcheck_providers, 2U, domain, generators,
                ipa_blinding_generator, ipa_u_generator, FrWire{}, statement,
                constraint.id, transcript, sumcheck_result, error, 0U, true))
            return false;
        sumcheck_providers.clear();
        if (duplicate_singleton) source_values.pop_back();
        route.sumcheck = std::move(sumcheck_result.proof);
        const std::uint32_t source_feature_bits =
            log2_exact(route.source_feature_domain);
        const std::uint32_t source_bits = log2_exact(source->domain_size);
        route.source_feature_point.assign(
            sumcheck_result.challenges.begin(),
            sumcheck_result.challenges.begin() + source_feature_bits);
        route.node_point.assign(
            sumcheck_result.challenges.begin() + source_feature_bits,
            sumcheck_result.challenges.begin() + source_bits);
        std::vector<FrWire> source_fold_point =
            sparse_route_point(route.source_feature_point);
        const auto node_point = sparse_route_point(route.node_point);
        source_fold_point.insert(source_fold_point.end(), node_point.begin(),
                                 node_point.end());
        const std::vector<FrWire> source_point =
            sparse_route_hyrax_point(source_fold_point);
        std::vector<FrWire> source_values_aggregate;
        FrWire source_aggregate_blinder{}, source_value_aggregate{};
        zkdl::CpuG1Point source_xi;
        const SegmentCommitmentValidationReceipt* source_receipt = nullptr;
        if (validation_receipts != nullptr) {
            const auto found = validation_receipts->find(source->id);
            if (found != validation_receipts->end())
                source_receipt = &found->second;
        }
        if (!sparse_route_aggregate_opening_inputs(
                *source, source_point, generators, ipa_blinding_generator,
                ipa_u_generator, commitments, route.source_commitment_index,
                blinders, segment_blinding_scalars, source_values_aggregate,
                source_aggregate_blinder, source_value_aggregate, source_xi,
                error, &source_values, &route_work_progress, authenticated_basis,
                msm_session, source_receipt)) return false;
        const auto terminal_digest = sparse_route_terminal_digest(
            header_digest, route);
        const auto source_metadata = zktr_make_opening_metadata(
            terminal_digest, statement, source_point,
            route.source_commitment_index, zktr_terminal_evaluation_index(0U));
        const auto output_metadata = zktr_make_opening_metadata(
            terminal_digest, statement, output_point,
            route.output_commitment_index, zktr_terminal_evaluation_index(1U));
        ZkRelationSegmentLayout source_layout;
        if (!zktr_segment_layout(source->domain_size, generators.size(),
                                 source_layout, error)) return false;
        const std::vector<zkdl::CpuG1Point> source_generators(
            generators.begin(), generators.begin() + source_layout.segment_size);
        const std::vector<zkdl::CpuG1Point> output_generators(
            generators.begin(), generators.begin() + output_layout.segment_size);
        auto source_segment_metadata = source_metadata;
        source_segment_metadata.logical_domain = source_layout.logical_domain;
        source_segment_metadata.segment_index = 0U;
        source_segment_metadata.logical_offset = 0U;
        source_segment_metadata.actual_length = source_layout.segment_size;
        source_segment_metadata.padded_length = source_layout.segment_size;
        auto output_segment_metadata = output_metadata;
        output_segment_metadata.logical_domain = output_layout.logical_domain;
        output_segment_metadata.segment_index = 0U;
        output_segment_metadata.logical_offset = 0U;
        output_segment_metadata.actual_length = output_layout.segment_size;
        output_segment_metadata.padded_length = output_layout.segment_size;
        const FrWire source_blinder = zkdl::cpu_fresh_csprng_scalar();
        const FrWire output_blinder = zkdl::cpu_fresh_csprng_scalar();
        const FrWire source_value = sumcheck_result.folded_terminal_values[1U];
        if (source_value != source_value_aggregate) {
            if (error) *error = "sparse route source terminal value disagrees with segment aggregate";
            return false;
        }
        const auto source_tau = zkdl::cpu_pedersen_scalar_commit(
            source_value_aggregate, source_blinder,
            ipa_blinding_generator, ipa_u_generator);
        const auto output_tau = zkdl::cpu_pedersen_scalar_commit(
            output_value, output_blinder,
            ipa_blinding_generator, ipa_u_generator);
        route.source_opening = authenticated_basis != nullptr
            ? zkdl::cpu_open_hyrax_v3_authenticated(
                  *authenticated_basis, source_values_aggregate,
                  zktr_segment_point(source_point, source_layout.segment_size),
                  source_aggregate_blinder, source_blinder, source_tau,
                  source_segment_metadata, source_xi, msm_session).proof
            : zkdl::cpu_open_hyrax_v3(
                  source_generators, source_values_aggregate,
                  zktr_segment_point(source_point, source_layout.segment_size),
                  ipa_blinding_generator, ipa_u_generator,
                  source_aggregate_blinder, source_blinder, source_tau,
                  source_segment_metadata).proof;
        route.output_opening = authenticated_basis != nullptr
            ? zkdl::cpu_open_hyrax_v3_authenticated(
                  *authenticated_basis, output_values,
                  zktr_segment_point(output_point, output_layout.segment_size),
                  output_aggregate_blinder, output_blinder, output_tau,
                  output_segment_metadata, output_xi, msm_session).proof
            : zkdl::cpu_open_hyrax_v3(
                  output_generators, output_values,
                  zktr_segment_point(output_point, output_layout.segment_size),
                  ipa_blinding_generator, ipa_u_generator,
                  output_aggregate_blinder, output_blinder, output_tau,
                  output_segment_metadata).proof;
        if (!zkdl::cpu_g1_equal(
                zkdl::cpu_g1_from_native(route.source_opening.xi), source_xi) ||
            !zkdl::cpu_g1_equal(
                zkdl::cpu_g1_from_native(route.output_opening.xi), output_xi)) {
            if (error) *error = "FAC feature-RLC opening does not match derived commitments";
            return false;
        }
        const FrWire p = sumcheck_result.folded_terminal_values[0U];
        const FrWire q0 = sumcheck_result.folded_terminal_values[2U];
        const FrWire residual = zkdl::fr_sub(
            zkdl::fr_mul(p, source_value), zkdl::fr_mul(q0, output_value));
        const FrWire residual_blinder = zkdl::fr_sub(
            zkdl::fr_mul(p, source_blinder),
            zkdl::fr_mul(q0, output_blinder));
        const auto residual_commitment = zkdl::cpu_g1_add(
            zkdl::cpu_g1_mul(source_tau, p),
            zkdl::cpu_g1_mul(output_tau, zkdl::fr_neg(q0)));
        route.terminal_binding = zkdl::cpu_open_pedersen_scalar_equality(
            residual, residual_blinder, sumcheck_result.final_q_blinder,
            residual_commitment,
            zkdl::cpu_g1_from_native(route.sumcheck.final_claim_commitment),
            ipa_blinding_generator, ipa_u_generator,
            sparse_route_terminal_context(
                terminal_digest, statement, ipa_blinding_generator,
                ipa_u_generator)).proof;
        sparse_route_absorb_terminal(transcript, route);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool verify_sparse_route_feature_rlc(
    const OperationDescriptor& descriptor,
    const std::vector<OperationDescriptor>& member_descriptors,
    const std::vector<FrWire>& coefficients,
    const std::vector<byte>& group_context,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, const SparseRouteProof& route,
    std::string* error,
    const zkdl::CpuIpaBasisContext* authenticated_basis,
    zkdl::CpuG1MsmSession* msm_session) {
    try {
        if (member_descriptors.size() < 2U ||
            member_descriptors.size() != coefficients.size()) {
            if (error) *error = "sparse route feature-RLC verifier shape mismatch";
            return false;
        }
        std::uint32_t source_index = 0U, output_index = 0U;
        if (!sparse_route_public_inputs_valid(
                descriptor, commitments, generators, statement, source_index,
                output_index, error) ||
            !relation_segment_commitments_valid(statement, commitments, error))
            return false;
        const std::uint32_t source_domain = commitments[source_index].domain_size;
        const std::uint32_t output_domain = commitments[output_index].domain_size;
        const std::uint32_t source_feature_bits =
            log2_exact(descriptor.selector_source_feature_domain);
        if (route.version != SparseRouteProof::kVersion ||
            route.source_commitment_index != source_index ||
            route.output_commitment_index != output_index ||
            route.edge_domain != descriptor.selector_edge_domain ||
            route.node_domain != descriptor.selector_node_domain ||
            route.feature_domain != descriptor.selector_feature_domain ||
            route.source_feature_domain !=
                descriptor.selector_source_feature_domain ||
            route.edge_point.size() != log2_exact(route.edge_domain) ||
            route.feature_point.size() != log2_exact(route.feature_domain) ||
            route.source_feature_point.size() != source_feature_bits ||
            route.node_point.size() != log2_exact(route.node_domain)) {
            if (error) *error = "FAC feature-RLC route wire does not match its anchor descriptor";
            return false;
        }
        std::vector<std::uint32_t> member_output_indices;
        member_output_indices.reserve(member_descriptors.size());
        for (const auto& member : member_descriptors) {
            if (member.selector_source_ref != descriptor.selector_source_ref ||
                member.selector_node_domain != descriptor.selector_node_domain ||
                member.selector_source_feature_domain !=
                    descriptor.selector_source_feature_domain ||
                member.selector_edge_domain != descriptor.selector_edge_domain ||
                member.selector_feature_domain != descriptor.selector_feature_domain) {
                if (error) *error = "FAC feature-RLC member descriptor layout mismatch";
                return false;
            }
            std::uint32_t member_index = 0U;
            if (!route_commitment_index(commitments, member.selector_output_ref,
                                        member_index, error) ||
                commitments[member_index].domain_size != output_domain) {
                if (error && error->empty())
                    *error = "FAC feature-RLC target commitment layout mismatch";
                return false;
            }
            member_output_indices.push_back(member_index);
        }
        const auto header_digest = sparse_route_absorb_header(
            descriptor, statement, commitments, source_index, output_index,
            ipa_blinding_generator, ipa_u_generator, transcript,
            &group_context);
        for (std::size_t bit = 0U; bit < route.edge_point.size(); ++bit)
            if (fr_compare_mixed(
                    route.edge_point[bit], transcript.challenge(
                        "gat.route." + descriptor.id + ".edge." +
                        std::to_string(bit))) != 0) {
                if (error) *error = "FAC feature-RLC edge challenge mismatch";
                return false;
            }
        for (std::size_t bit = 0U; bit < route.feature_point.size(); ++bit)
            if (fr_compare_mixed(
                    route.feature_point[bit], transcript.challenge(
                        "gat.route." + descriptor.id + ".feature." +
                        std::to_string(bit))) != 0) {
                if (error) *error = "FAC feature-RLC feature challenge mismatch";
                return false;
            }
        const auto constraint = sparse_route_sumcheck_descriptor(
            descriptor, header_digest);
        ZkSumcheckVerifierResult sumcheck_result;
        if (!verify_zk_sumcheck(
                route.sumcheck, constraint, 2U,
                std::max(2U, source_domain), generators,
                ipa_blinding_generator, ipa_u_generator, FrWire{}, statement,
                constraint.id, transcript, sumcheck_result, error)) return false;
        std::vector<FrWire> source_fold_point;
        for (std::size_t bit = 0U; bit < log2_exact(source_domain); ++bit) {
            const auto& coordinate = bit < source_feature_bits
                ? route.source_feature_point[bit]
                : route.node_point[bit - source_feature_bits];
            if (fr_compare_mixed(coordinate,
                                 sumcheck_result.challenges[bit]) != 0) {
                if (error) *error = "FAC feature-RLC source challenge mismatch";
                return false;
            }
            source_fold_point.push_back(zkdl::fr_wire(coordinate));
        }
        const std::vector<FrWire> source_point =
            sparse_route_hyrax_point(source_fold_point);
        const auto edge_point = sparse_route_point(route.edge_point);
        const auto feature_point = sparse_route_point(route.feature_point);
        std::vector<FrWire> output_fold_point = feature_point;
        output_fold_point.insert(output_fold_point.end(), edge_point.begin(),
                                 edge_point.end());
        const std::vector<FrWire> output_point =
            sparse_route_hyrax_point(output_fold_point);
        ZkRelationSegmentLayout source_layout, output_layout;
        if (!zktr_segment_layout(source_domain, generators.size(),
                                 source_layout, error) ||
            !zktr_segment_layout(output_domain, generators.size(),
                                 output_layout, error)) return false;
        const auto weighted_commitment = [&](
                std::uint32_t commitment_index,
                const std::vector<FrWire>& point,
                const ZkRelationSegmentLayout& layout,
                zkdl::CpuG1Point& weighted) {
            const auto& wire = commitments[commitment_index];
            if (wire.segment_count != layout.segment_count ||
                wire.segment_size != layout.segment_size ||
                wire.segment_commitments.size() != layout.segment_count) {
                if (error) *error = "FAC feature-RLC segment layout mismatch";
                return false;
            }
            const auto high_point =
                zktr_segment_prefix_point(point, layout.segment_size);
            const auto weights = zkdl::cpu_ipa_mle_weights(high_point);
            if (weights.size() != layout.segment_count) {
                if (error) *error = "FAC feature-RLC segment weight shape mismatch";
                return false;
            }
            weighted = zkdl::CpuG1Point::zero();
            for (std::size_t segment = 0U; segment < weights.size(); ++segment) {
                if (zkdl::fr_compare(weights[segment], FrWire{}) == 0) {
                    if (error) *error = "FAC feature-RLC segment weight is zero";
                    return false;
                }
                weighted = zkdl::cpu_g1_add(
                    weighted,
                    zkdl::cpu_g1_mul(
                        zkdl::cpu_g1_from_native(
                            wire.segment_commitments[segment].commitment),
                        weights[segment]));
            }
            return true;
        };
        zkdl::CpuG1Point source_xi, output_xi = zkdl::CpuG1Point::zero();
        if (!weighted_commitment(source_index, source_point, source_layout,
                                 source_xi)) return false;
        for (std::size_t member = 0U;
             member < member_output_indices.size(); ++member) {
            zkdl::CpuG1Point member_xi;
            if (!weighted_commitment(member_output_indices[member], output_point,
                                     output_layout, member_xi)) return false;
            output_xi = zkdl::cpu_g1_add(
                output_xi,
                zkdl::cpu_g1_mul(member_xi, coefficients[member]));
        }
        const auto terminal_digest = sparse_route_terminal_digest(
            header_digest, route);
        auto source_metadata = zktr_make_opening_metadata(
            terminal_digest, statement, source_point, source_index,
            zktr_terminal_evaluation_index(0U));
        source_metadata.logical_domain = source_layout.logical_domain;
        source_metadata.segment_index = 0U;
        source_metadata.logical_offset = 0U;
        source_metadata.actual_length = source_layout.segment_size;
        source_metadata.padded_length = source_layout.segment_size;
        auto output_metadata = zktr_make_opening_metadata(
            terminal_digest, statement, output_point, output_index,
            zktr_terminal_evaluation_index(1U));
        output_metadata.logical_domain = output_layout.logical_domain;
        output_metadata.segment_index = 0U;
        output_metadata.logical_offset = 0U;
        output_metadata.actual_length = output_layout.segment_size;
        output_metadata.padded_length = output_layout.segment_size;
        const auto verify_route_opening = [&](
                const zkdl::HyraxOpeningProofV3& opening,
                const zkdl::CpuG1Point& xi, const zkdl::CpuG1Point& tau,
                const std::vector<FrWire>& point,
                const zkdl::CpuIpaOpeningMetadata& metadata,
                std::uint32_t generator_count) {
            if (authenticated_basis != nullptr)
                return zkdl::cpu_verify_hyrax_v3_authenticated(
                    opening, xi, tau, *authenticated_basis, point, metadata,
                    error, msm_session);
            const std::vector<zkdl::CpuG1Point> prefix(
                generators.begin(), generators.begin() + generator_count);
            return zkdl::cpu_verify_hyrax_v3(
                opening, xi, tau, prefix, point, ipa_blinding_generator,
                ipa_u_generator, metadata, error);
        };
        const auto source_tau =
            zkdl::cpu_g1_from_native(route.source_opening.tau);
        const auto output_tau =
            zkdl::cpu_g1_from_native(route.output_opening.tau);
        if (!verify_route_opening(
                route.source_opening, source_xi, source_tau,
                zktr_segment_point(source_point, source_layout.segment_size),
                source_metadata, source_layout.segment_size) ||
            !verify_route_opening(
                route.output_opening, output_xi, output_tau,
                zktr_segment_point(output_point, output_layout.segment_size),
                output_metadata, output_layout.segment_size)) return false;
        std::vector<FrWire> selector_feature;
        if (!sparse_route_feature_rlc_feature(
                member_descriptors, coefficients, feature_point,
                selector_feature, error)) return false;
        const auto selector_outer = public_route_selector_mle(
            descriptor, edge_point, descriptor.route_by_destination);
        auto selector = sparse_route_selector_from_factors(
            descriptor, selector_outer, selector_feature);
        FrWire q0 = zkdl::fr_from_u32(1U);
        for (const auto& challenge : sumcheck_result.challenges) {
            const auto coordinate = zkdl::fr_wire(challenge);
            fold_values(selector, coordinate);
            q0 = zkdl::fr_mul(
                q0, zkdl::fr_sub(zkdl::fr_from_u32(1U), coordinate));
        }
        const auto residual_commitment = zkdl::cpu_g1_add(
            zkdl::cpu_g1_mul(source_tau, selector.front()),
            zkdl::cpu_g1_mul(output_tau, zkdl::fr_neg(q0)));
        if (!zkdl::cpu_verify_pedersen_scalar_equality(
                route.terminal_binding, residual_commitment,
                zkdl::cpu_g1_from_native(sumcheck_result.final_claim_commitment),
                ipa_blinding_generator, ipa_u_generator,
                sparse_route_terminal_context(
                    terminal_digest, statement, ipa_blinding_generator,
                    ipa_u_generator), error)) return false;
        sparse_route_absorb_terminal(transcript, route);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

#endif
