#include "hyrax.hpp"
#include "proof_session.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mcl::bn;

static NativeProofSession rebuild_session(
    const NativeProofSession &source, const std::vector<std::size_t> &order) {
    NativeProofSession rebuilt(source.domain(), source.statement());
    for (std::size_t index : order) {
        const NativeProofMessage &message = source.messages().at(index);
        if (message.kind == NativeProofMessageKind::Data)
            rebuilt.absorb(message.label, message.payload);
        else if (message.kind == NativeProofMessageKind::Challenge)
            rebuilt.challenge(message.label.substr(sizeof("challenge/") - 1));
        else
            throw std::runtime_error("unexpected Hyrax test message kind");
    }
    return rebuilt;
}

int main() {
    initPairing(mcl::BN254);
    const std::size_t n = 4;
    std::vector<Fr> basis{Fr(1), Fr(2), Fr(3), Fr(4)};
    std::vector<Fr> witness{Fr(5), Fr(7), Fr(11), Fr(13)};
    Fr eval = 0;
    for (std::size_t i = 0; i < n; ++i) eval += basis[i] * witness[i];

    G1 G;
    hashAndMapToG1(G, "native-hyrax-session/G");
    std::vector<G1> generators(n);
    for (std::size_t i = 0; i < n; ++i)
        hashAndMapToG1(generators[i], "native-hyrax-session/g/" + std::to_string(i));

    NativeProofStatement statement{
        "0000000000000000000000000000000000000000000000000000000000000011",
        "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000012",
        "0000000000000000000000000000000000000000000000000000000000000013",
        "0000000000000000000000000000000000000000000000000000000000000014",
        "0000000000000000000000000000000000000000000000000000000000000015",
        "0000000000000000000000000000000000000000000000000000000000000016",
        "0000000000000000000000000000000000000000000000000000000000000017",
        NativeProofStatement::basis_digest_for(G, generators, basis, n),
        n,
        "0000000000000000000000000000000000000000000000000000000000000011"};

    const G1 comm_x = perdersen_commit(generators.data(), witness.data(), static_cast<int>(n));
    const G1 comm_y = G * eval;
    NativeProofSession produced("native-hyrax-session-v1", statement);
    prove_dot_product_with_session(comm_x, comm_y, basis.data(), generators.data(),
                                   G, witness.data(), eval, static_cast<int>(n),
                                   produced, "hyrax/opening");

    std::string reason;
    if (!hyrax::verify_opening_with_session(comm_x, eval, G, generators, basis,
                                            produced, "hyrax/opening", &reason)) {
        std::cerr << "independent Hyrax verify failed: " << reason << '\n';
        return 1;
    }

    const std::string path = "/tmp/zkgpt-native-hyrax-session-smoke.bin";
    if (!produced.write(path, &reason)) {
        std::cerr << "session write failed: " << reason << '\n';
        return 2;
    }
    NativeProofSession restored;
    if (!NativeProofSession::read(path, restored, &reason) ||
        !hyrax::verify_opening_with_session(comm_x, eval, G, generators, basis,
                                            restored, "hyrax/opening", &reason)) {
        std::cerr << "serialized independent Hyrax verify failed: " << reason << '\n';
        std::remove(path.c_str());
        return 3;
    }
    std::vector<std::size_t> reordered_order(restored.messages().size());
    for (std::size_t i = 0; i < reordered_order.size(); ++i) reordered_order[i] = i;
    std::swap(reordered_order[0], reordered_order[1]);
    NativeProofSession reordered = rebuild_session(restored, reordered_order);
    if (hyrax::verify_opening_with_session(comm_x, eval, G, generators, basis,
                                           reordered, "hyrax/opening", &reason)) {
        std::cerr << "Hyrax non-adjacent reorder was accepted\n";
        std::remove(path.c_str());
        return 4;
    }
    std::vector<std::size_t> exact_order(restored.messages().size());
    for (std::size_t i = 0; i < exact_order.size(); ++i) exact_order[i] = i;
    NativeProofSession post_terminal = rebuild_session(restored, exact_order);
    post_terminal.absorb("hyrax/opening/post-terminal", "forbidden");
    if (hyrax::verify_opening_with_session(comm_x, eval, G, generators, basis,
                                           post_terminal, "hyrax/opening", &reason)) {
        std::cerr << "Hyrax post-terminal insertion was accepted\n";
        std::remove(path.c_str());
        return 5;
    }
    if (hyrax::verify_opening_with_session(comm_x, eval + Fr(1), G, generators,
                                           basis, restored, "hyrax/opening",
                                           &reason)) {
        std::cerr << "public evaluation tamper was accepted\n";
        std::remove(path.c_str());
        return 6;
    }
    std::vector<Fr> wrong_basis = basis;
    wrong_basis[0] += Fr(1);
    if (hyrax::verify_opening_with_session(comm_x, eval, G, generators,
                                           wrong_basis, restored, "hyrax/opening",
                                           &reason)) {
        std::cerr << "basis tamper was accepted\n";
        std::remove(path.c_str());
        return 7;
    }
    {
        std::fstream wire(path, std::ios::in | std::ios::out | std::ios::binary);
        wire.seekg(0, std::ios::end);
        const std::streamoff size = wire.tellg();
        if (size <= 0) {
            std::cerr << "serialized session is empty\n";
            std::remove(path.c_str());
            return 8;
        }
        wire.seekg(size - 1);
        char byte = 0;
        wire.read(&byte, 1);
        byte ^= static_cast<char>(1);
        wire.seekp(size - 1);
        wire.write(&byte, 1);
        wire.flush();
    }
    NativeProofSession tampered;
    if (NativeProofSession::read(path, tampered, &reason)) {
        std::cerr << "proof wire tamper was accepted\n";
        std::remove(path.c_str());
        return 9;
    }
    std::remove(path.c_str());
    std::cout << "native_hyrax_session_smoke: PASS\n";
    return 0;
}
