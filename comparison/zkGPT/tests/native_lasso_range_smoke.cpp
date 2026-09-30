#include "native_lasso.hpp"
#include "native_range.hpp"
#include "hyrax.hpp"

#include <mcl/bn256.hpp>

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl::bn;

namespace {

std::vector<G1> generators(const std::string& prefix, std::size_t count) {
    std::vector<G1> result(count);
    for (std::size_t i = 0; i < count; ++i) hashAndMapToG1(result[i], prefix + "/" + std::to_string(i));
    return result;
}

NativeProofStatement statement_for(const G1& G, const std::vector<G1>& l_generators, const std::vector<NativeLassoFamilyPublic>& families, const G1& com_l,
                                   const G1& query_blinding_generator) {
    const std::string context_root = "00000000000000000000000000000000000000000000000000000000000000aa";
    std::vector<Fr> setup_basis(l_generators.size(), Fr(0));
    NativeProofStatement statement{context_root,
                                   "mcl-bn254-fr-v1",
                                   "00000000000000000000000000000000000000000000000000000000000000ab",
                                   native_lasso_resources_digest(families),
                                   native_lasso_tables_digest(families),
                                   "00000000000000000000000000000000000000000000000000000000000000ac",
                                   "00000000000000000000000000000000000000000000000000000000000000ad",
                                   native_lasso_query_digest(com_l, l_generators.size()),
                                   NativeProofStatement::basis_digest_for(query_blinding_generator, l_generators, setup_basis, l_generators.size()),
                                   l_generators.size(),
                                   context_root};
    return statement;
}

bool decode_field(const std::string& encoded, F& value) {
    try {
        value.setStr(encoded, 16);
    } catch (...) { return false; }
    return value.getStr(16) == encoded;
}

bool decode_vector(const std::string& payload, std::vector<F>& values) {
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t& value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(static_cast<unsigned char>(payload[offset++])) << (8 * i);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count)) return false;
    values.clear();
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size == 0 || size > payload.size() - offset) return false;
        F value;
        if (!decode_field(payload.substr(offset, static_cast<std::size_t>(size)), value)) return false;
        offset += static_cast<std::size_t>(size);
        values.push_back(value);
    }
    return offset == payload.size();
}

NativeProofSession rebuild(const NativeProofSession& source, const std::vector<std::size_t>& order) {
    NativeProofSession rebuilt(source.domain(), source.statement());
    for (std::size_t index : order) {
        const NativeProofMessage& message = source.messages().at(index);
        switch (message.kind) {
        case NativeProofMessageKind::Data:
            rebuilt.absorb(message.label, message.payload);
            break;
        case NativeProofMessageKind::Field: {
            F value;
            if (!decode_field(message.payload, value)) throw std::runtime_error("field");
            rebuilt.absorb_field(message.label, value);
            break;
        }
        case NativeProofMessageKind::FieldVector: {
            std::vector<F> values;
            if (!decode_vector(message.payload, values)) throw std::runtime_error("vector");
            rebuilt.absorb_fields(message.label, values);
            break;
        }
        case NativeProofMessageKind::Challenge:
            rebuilt.challenge(message.label.substr(sizeof("challenge/") - 1));
            break;
        }
    }
    return rebuilt;
}

std::vector<F> basis_for(const std::vector<F>& point) {
    std::vector<F> result(1ULL << point.size());
    result[0] = F(1) - point[0];
    result[1] = point[0];
    for (std::size_t bit = 1; bit < point.size(); ++bit) {
        const std::size_t old_size = 1ULL << bit;
        for (std::size_t i = old_size; i-- > 0;) {
            const F previous = result[i];
            result[i] = previous * (F(1) - point[bit]);
            result[i + old_size] = previous * point[bit];
        }
    }
    return result;
}

}

int main() {
    initPairing(mcl::BN254);
    G1 G;
    hashAndMapToG1(G, "native-lasso/G");

    const std::vector<G1> l_generators = generators("native-lasso/L", 2);

    std::vector<NativeLassoFamilyProver> families(1);
    const std::vector<std::vector<F>> tables = {{Fr(5), Fr(7)}};
    for (std::size_t family_index = 0; family_index < families.size(); ++family_index) {
        NativeLassoFamilyProver& family = families[family_index];
        family.family_id = "a";
        family.table_id = "t_a";
        family.resource_id = "resource-a-v1";
        family.registration_id = "registration-a-v1";
        family.table = tables[family_index];
        family.matrix_generators = generators("native-lasso/M/" + family.family_id, 4);
        family.segment_offset = 0;
        family.logical_queries = 2;
        family.padded_queries = 2;
        family.padding_index = 0;
        family.padding_value = family.table[0];
        family.queries = family.table;
        family.matrix.assign(4, Fr(0));
        for (std::size_t row = 0; row < 2; ++row) family.matrix[row + row * 2] = Fr(1);
    }
    std::vector<NativeLassoFamilyPublic> public_families;
    for (const NativeLassoFamilyProver& family : families) public_families.push_back(family);

    NativeLassoQueryProver query;
    query.G = G;
    query.generators = l_generators;
    query.values = {Fr(5), Fr(7)};
    query.basis_id = "0000000000000000000000000000000000000000000000000000000000000001";
    const hyrax::PedersenMleCommitment query_commitment = hyrax::pedersen_commit_mle(query.generators, query.values, query.basis_id);
    query.commitment = query_commitment.commitment;
    query.blinding_generator = query_commitment.blinding_generator;
    query.commitment_blinding = query_commitment.blinding;
    query.has_commitment_blinding = true;
    query.statement_l_digest = native_lasso_query_digest(query.commitment, query.values.size());
    const std::string tables_digest = native_lasso_tables_digest(public_families);
    const std::string resources_digest = native_lasso_resources_digest(public_families);
    for (NativeLassoFamilyProver& family : families) {
        family.statement_tables_digest = tables_digest;
        family.statement_resources_digest = resources_digest;
    }
    public_families.clear();
    for (const NativeLassoFamilyProver& family : families) public_families.push_back(family);
    const NativeProofStatement statement = statement_for(G, l_generators, public_families, query.commitment, query.blinding_generator);
    NativeProofSession produced("native-lasso-range-v1", statement);
    NativeLassoBundleResult proved;
    std::string reason;
    if (!native_lasso_prove_bundle(produced, families, query, &proved, &reason)) {
        std::cerr << "native Lasso prove failed: " << reason << '\n';
        return 1;
    }
    NativeLassoQueryPublic public_query{G, l_generators, query.commitment, query.basis_id, query.statement_l_digest};
    if (!native_lasso_verify_bundle(produced, public_families, public_query, nullptr, &reason)) {
        std::cerr << "native Lasso independent verify failed: " << reason << '\n';
        return 2;
    }

    bool saw_masked_terminal = false;
    for (const NativeProofMessage& message : produced.messages()) {
        if (message.label.find("/terminal/values") != std::string::npos) {
            if (message.kind != NativeProofMessageKind::FieldVector) {
                std::cerr << "masked terminal value message kind is invalid\n";
                return 3;
            }
            saw_masked_terminal = true;
        }
        if (message.label.find("/d-onehot/sum/terminal/values") != std::string::npos) {
            std::vector<F> values;
            if (!decode_vector(message.payload, values) || values.size() != 1 || values[0] == Fr(1)) {
                std::cerr << "raw one-hot terminal leaked\n";
                return 4;
            }
        }
        if (message.label.compare(0, sizeof("challenge/") - 1, "challenge/") != 0 && message.label.find("/terminal/unmasked/value/") != std::string::npos) {
            const std::size_t slash = message.label.rfind('/');
            const std::string suffix = slash == std::string::npos ? std::string() : message.label.substr(slash + 1);
            const bool opening_wire = suffix == "meta" || suffix == "T" || suffix == "response";
            const bool expected_kind =
                (suffix == "meta" || suffix == "T") ? message.kind == NativeProofMessageKind::Data : message.kind == NativeProofMessageKind::FieldVector;
            if (!opening_wire || !expected_kind) {
                std::cerr << "raw terminal opening was serialized as a scalar\n";
                return 4;
            }
        }
    }
    if (!saw_masked_terminal) {
        std::cerr << "masked terminal surface is missing\n";
        return 5;
    }
    if (!proved.has_final_l_commitment || proved.final_l_commitment.point.isZero()) {
        std::cerr << "authenticated final L claim is missing\n";
        return 5;
    }

    std::size_t first_challenge = produced.messages().size();
    for (std::size_t i = 0; i < produced.messages().size(); ++i) {
        if (produced.messages()[i].kind == NativeProofMessageKind::Challenge) {
            first_challenge = i;
            break;
        }
    }
    for (std::size_t i = 0; i < families.size(); ++i) {
        const std::string label = "lasso/" + families[i].family_id + "/commitment";
        bool found = false;
        for (std::size_t j = 0; j < first_challenge; ++j) found |= produced.messages()[j].label == label;
        if (!found) {
            std::cerr << "LC_f was not fixed before the first challenge\n";
            return 6;
        }
    }

    const std::string path = "/tmp/zkgpt-native-lasso-range.bin";
    if (!produced.write(path, &reason)) {
        std::cerr << "native Lasso wire write failed: " << reason << '\n';
        return 7;
    }
    NativeProofSession restored;
    if (!NativeProofSession::read(path, restored, &reason) || !native_lasso_verify_bundle(restored, public_families, public_query, nullptr, &reason)) {
        std::cerr << "native Lasso serialized verify failed: " << reason << '\n';
        std::remove(path.c_str());
        return 8;
    }

    std::vector<std::size_t> reordered(restored.messages().size());
    for (std::size_t i = 0; i < reordered.size(); ++i) reordered[i] = i;
    std::size_t commitment_a = restored.messages().size();
    for (std::size_t i = 0; i < reordered.size(); ++i) {
        if (restored.messages()[i].label == "lasso/a/commitment") commitment_a = i;
    }
    if (commitment_a == restored.messages().size()) {
        std::cerr << "LC_f commitment messages are missing\n";
        std::remove(path.c_str());
        return 9;
    }

    std::swap(reordered[commitment_a], reordered[first_challenge]);
    NativeProofSession reordered_session = rebuild(restored, reordered);
    if (native_lasso_verify_bundle(reordered_session, public_families, public_query, nullptr, &reason)) {
        std::cerr << "LC_f post-challenge reorder was accepted\n";
        std::remove(path.c_str());
        return 10;
    }

    NativeLassoQueryPublic wrong_query = public_query;
    hashAndMapToG1(wrong_query.commitment, "native-lasso/wrong-com-L");
    if (native_lasso_verify_bundle(restored, public_families, wrong_query, nullptr, &reason)) {
        std::cerr << "Open_L commitment tamper was accepted\n";
        std::remove(path.c_str());
        return 11;
    }
    std::vector<NativeLassoFamilyPublic> wrong_tables = public_families;
    wrong_tables[0].table[0] += Fr(1);
    if (native_lasso_verify_bundle(restored, wrong_tables, public_query, nullptr, &reason)) {
        std::cerr << "table identity tamper was accepted\n";
        std::remove(path.c_str());
        return 12;
    }
    std::vector<NativeLassoFamilyPublic> wrong_matrix = public_families;
    hashAndMapToG1(wrong_matrix[0].matrix_generators[0], "native-lasso/wrong-M");
    if (native_lasso_verify_bundle(restored, wrong_matrix, public_query, nullptr, &reason)) {
        std::cerr << "M_f basis identity tamper was accepted\n";
        std::remove(path.c_str());
        return 13;
    }
    NativeProofSession post_terminal = restored;
    post_terminal.absorb("lasso/post-terminal", "forbidden");
    if (native_lasso_verify_bundle(post_terminal, public_families, public_query, nullptr, &reason)) {
        std::cerr << "post-terminal insertion was accepted\n";
        std::remove(path.c_str());
        return 14;
    }

    std::vector<NativeLassoFamilyPublic> wrong_resources = public_families;
    wrong_resources[0].resource_id = "resource-a-v2";
    if (native_lasso_verify_bundle(restored, wrong_resources, public_query, nullptr, &reason)) {
        std::cerr << "resource identity tamper was accepted\n";
        std::remove(path.c_str());
        return 15;
    }
    std::vector<NativeLassoFamilyPublic> wrong_bound_identity = public_families;
    wrong_bound_identity[0].statement_tables_digest = std::string(64, 'b');
    if (native_lasso_verify_bundle(restored, wrong_bound_identity, public_query, nullptr, &reason)) {
        std::cerr << "adapter table digest tamper was accepted\n";
        std::remove(path.c_str());
        return 16;
    }
    NativeLassoQueryPublic wrong_query_identity = public_query;
    wrong_query_identity.statement_l_digest = std::string(64, 'c');
    if (native_lasso_verify_bundle(restored, public_families, wrong_query_identity, nullptr, &reason)) {
        std::cerr << "adapter L_qry digest tamper was accepted\n";
        std::remove(path.c_str());
        return 17;
    }

    std::vector<NativeProofMessage> tampered_messages = restored.messages();
    bool changed_message = false;
    for (NativeProofMessage& message : tampered_messages) {
        if (message.kind == NativeProofMessageKind::Field && message.label == "lasso/a/b-boolean/claim") {
            message.payload = Fr(1).getStr(16);
            changed_message = true;
            break;
        }
    }
    NativeProofSession tampered_session;
    const bool tampered_wire_ok =
        changed_message && NativeProofSession::from_messages(restored.domain(), restored.statement(), tampered_messages, tampered_session, &reason);
    if (tampered_wire_ok && native_lasso_verify_bundle(tampered_session, public_families, public_query, nullptr, &reason)) {
        std::cerr << "transcript tamper was accepted\n";
        std::remove(path.c_str());
        return 18;
    }

    NativeLassoQueryProver fresh_query = query;
    const hyrax::PedersenMleCommitment fresh_commitment = hyrax::pedersen_commit_mle(fresh_query.generators, fresh_query.values, fresh_query.basis_id);
    fresh_query.commitment = fresh_commitment.commitment;
    fresh_query.blinding_generator = fresh_commitment.blinding_generator;
    fresh_query.commitment_blinding = fresh_commitment.blinding;
    fresh_query.has_commitment_blinding = true;
    fresh_query.statement_l_digest = native_lasso_query_digest(fresh_query.commitment, fresh_query.values.size());
    const NativeProofStatement fresh_statement = statement_for(G, l_generators, public_families, fresh_query.commitment, fresh_query.blinding_generator);
    NativeProofSession fresh_session("native-lasso-range-v1", fresh_statement);
    std::vector<NativeLassoFamilyProver> fresh_families = families;
    if (!native_lasso_prove_bundle(fresh_session, fresh_families, fresh_query, nullptr, &reason)) {
        std::cerr << "fresh masked Lasso proof failed: " << reason << '\n';
        std::remove(path.c_str());
        return 19;
    }
    NativeLassoQueryPublic fresh_public_query{G, l_generators, fresh_query.commitment, fresh_query.basis_id, fresh_query.statement_l_digest};
    if (!native_lasso_verify_bundle(fresh_session, public_families, fresh_public_query, nullptr, &reason) || fresh_query.commitment == query.commitment ||
        fresh_session.digest() == produced.digest()) {
        std::cerr << "fresh masked Lasso commitment/transcript was reused\n";
        std::remove(path.c_str());
        return 20;
    }

    NativeProofSession range_session("native-range-v1", statement);
    std::vector<F> range_values{Fr(1), Fr(2)};
    F range_claim = 0;
    for (const F& value : range_values) range_claim += value;
    if (!native_range_prove_deg1(range_session, "range/a", range_values, range_claim, nullptr, &reason) ||
        !native_range_verify_deg1(range_session, "range/a", range_claim, nullptr, &reason) ||
        native_range_verify_deg1(range_session, "range/a", range_claim + Fr(1), nullptr, &reason)) {
        std::cerr << "degree-one range wire/tamper check failed: " << reason << '\n';
        std::remove(path.c_str());
        return 12;
    }
    std::vector<F> eq{Fr(1), Fr(2)};
    std::vector<F> f{Fr(2), Fr(3)};
    std::vector<F> g{Fr(11), Fr(13)};
    F degree3_claim = 0;
    for (std::size_t i = 0; i < eq.size(); ++i) degree3_claim += eq[i] * f[i] * g[i];
    if (!native_range_prove_deg3(range_session, "range/b", eq, f, g, degree3_claim, nullptr, &reason) ||
        !native_range_verify_deg3(range_session, "range/b", degree3_claim, nullptr, &reason)) {
        std::cerr << "degree-three range wire check failed: " << reason << '\n';
        std::remove(path.c_str());
        return 13;
    }

    NativeProofStatement rp_statement = statement_for(G, l_generators, public_families, query.commitment, G);
    NativeProofSession rp_session("native-rp-v1", rp_statement);
    std::vector<F> rp_point{Fr(2)};
    G1 rp_commitment;
    Fr rp_eval;
    if (!range_proof_open_with_session(query.values, rp_point, G, l_generators, rp_commitment, rp_eval, rp_session, "rp/open", &reason, true)) {
        std::cerr << "Hyrax-RP session prove failed: " << reason << '\n';
        std::remove(path.c_str());
        return 14;
    }
    const std::vector<F> rp_basis = basis_for(rp_point);
    if (!range_proof_verify_opening_with_session(rp_commitment, rp_eval, G, l_generators, rp_basis, rp_session, "rp/open", &reason, true)) {
        std::cerr << "Hyrax-RP independent verify failed: " << reason << '\n';
        std::remove(path.c_str());
        return 15;
    }

    std::remove(path.c_str());
    std::remove("/tmp/zkgpt-native-lasso-range.bin");
    std::cout << "native_lasso_range_smoke: PASS\n";
    return 0;
}
