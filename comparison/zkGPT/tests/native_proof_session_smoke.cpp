#include "proof_session.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using namespace mcl::bn;

int main() {
    initPairing(mcl::BN254);
    const std::string path = "/tmp/zkgpt-native-proof-session-smoke.bin";
    NativeProofStatement statement{
        "0000000000000000000000000000000000000000000000000000000000000001",
        "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000002",
        "0000000000000000000000000000000000000000000000000000000000000003",
        "0000000000000000000000000000000000000000000000000000000000000004",
        "0000000000000000000000000000000000000000000000000000000000000005",
        "0000000000000000000000000000000000000000000000000000000000000006",
        "0000000000000000000000000000000000000000000000000000000000000007",
        "0000000000000000000000000000000000000000000000000000000000000008",
        4};
    NativeProofSession produced("native-session-smoke-v1", statement);
    produced.absorb("commitment", "public-commitment");
    const Fr first = produced.challenge("gkr/r/0");
    produced.absorb_field("poly/0", Fr(7));
    const Fr second = produced.challenge("gkr/r/1");
    produced.absorb_fields("claims", std::vector<F>{first, second});
    std::string reason;
    if (!produced.write(path, &reason)) {
        std::cerr << "write failed: " << reason << '\n';
        return 1;
    }
    NativeProofSession restored;
    if (!NativeProofSession::read(path, restored, &reason) ||
        restored.digest() != produced.digest() ||
        restored.messages().size() != produced.messages().size() ||
        !restored.replay(&reason)) {
        std::cerr << "round-trip failed: " << reason << '\n';
        std::remove(path.c_str());
        return 2;
    }

    std::ifstream in(path.c_str(), std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    if (bytes.size() < 16) return 3;
    bytes[bytes.size() - 1] ^= 1;
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    NativeProofSession tampered;
    if (NativeProofSession::read(path, tampered, &reason)) {
        std::remove(path.c_str());
        return 4;
    }
    std::remove(path.c_str());
    std::cout << "native_proof_session_smoke: PASS first=" << first.getStr(16)
              << " second=" << second.getStr(16) << '\n';
    return 0;
}
