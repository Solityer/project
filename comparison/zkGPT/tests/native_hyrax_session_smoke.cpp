#include "hyrax.hpp"
#include "proof_session.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace mcl::bn;

int main() {
    initPairing(mcl::BN254);
    NativeProofStatement statement{
        "0000000000000000000000000000000000000000000000000000000000000011",
        "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000012",
        "0000000000000000000000000000000000000000000000000000000000000013",
        "0000000000000000000000000000000000000000000000000000000000000014",
        "0000000000000000000000000000000000000000000000000000000000000015",
        "0000000000000000000000000000000000000000000000000000000000000016",
        "0000000000000000000000000000000000000000000000000000000000000017",
        "0000000000000000000000000000000000000000000000000000000000000018",
        4};
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
    if (hyrax::verify_opening_with_session(comm_x, eval + Fr(1), G, generators,
                                           basis, restored, "hyrax/opening",
                                           &reason)) {
        std::cerr << "public evaluation tamper was accepted\n";
        std::remove(path.c_str());
        return 4;
    }
    {
        std::fstream wire(path, std::ios::in | std::ios::out | std::ios::binary);
        wire.seekg(0, std::ios::end);
        const std::streamoff size = wire.tellg();
        if (size <= 0) {
            std::cerr << "serialized session is empty\n";
            std::remove(path.c_str());
            return 5;
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
        return 6;
    }
    std::remove(path.c_str());
    std::cout << "native_hyrax_session_smoke: PASS\n";
    return 0;
}
