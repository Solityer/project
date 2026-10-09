#include "proof_session.hpp"

#include <cstdio>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using namespace mcl::bn;

int main() {
    initPairing(mcl::BN254);
    const std::string path = "/tmp/zkgpt-native-proof-session-smoke.bin";
    NativeProofStatement statement{
        "0000000000000000000000000000000000000000000000000000000000000001", "mcl-bn254-fr-v1",
        "0000000000000000000000000000000000000000000000000000000000000002", "0000000000000000000000000000000000000000000000000000000000000003",
        "0000000000000000000000000000000000000000000000000000000000000004", "0000000000000000000000000000000000000000000000000000000000000005",
        "0000000000000000000000000000000000000000000000000000000000000006", "0000000000000000000000000000000000000000000000000000000000000007",
        "0000000000000000000000000000000000000000000000000000000000000008", 4,
        "0000000000000000000000000000000000000000000000000000000000000001"};
    NativeProofStatement invalid_statement = statement;
    invalid_statement.plan_digest = "arbitrary-nonempty-label";
    bool invalid_rejected = false;
    try {
        NativeProofSession invalid("native-session-smoke-v1", invalid_statement);
    } catch (...) { invalid_rejected = true; }
    if (!invalid_rejected) {
        std::cerr << "unstructured statement identity was accepted\n";
        return 1;
    }
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
    if (!NativeProofSession::read(path, restored, &reason) || restored.digest() != produced.digest() ||
        restored.messages().size() != produced.messages().size() || !restored.replay(&reason)) {
        std::cerr << "round-trip failed: " << reason << '\n';
        std::remove(path.c_str());
        return 2;
    }

    for (unsigned round = 0; round < 24; ++round) {
        restored.absorb_field("incremental/poly/" + std::to_string(round), Fr(round + 11));
        restored.challenge("incremental/r/" + std::to_string(round));
        if (!restored.valid(&reason) || !restored.replay(&reason)) return 5;
        NativeProofSession full_replay;
        if (!NativeProofSession::from_messages(restored.domain(), restored.statement(), restored.messages(), full_replay, &reason) || full_replay.digest() != restored.digest()) return 6;
    }
    restored.absorb("consecutive/boundary", "committed");
    for (unsigned round = 0; round < 24; ++round) {
        restored.challenge("consecutive/r/" + std::to_string(round));
        if (!restored.valid(&reason) || !restored.replay(&reason)) return 17;
    }
    NativeProofSession lasso_schedule("native-session-smoke-v1", statement);
    for (unsigned round = 0; round < 12; ++round) lasso_schedule.challenge("lasso/function/lrelu/a-query/r/" + std::to_string(round));
    lasso_schedule.challenge("lasso/function/lrelu/c-boolean-open/c/0");
    lasso_schedule.challenge("lasso/function/lrelu/c-boolean-complement-open/c/0");
    lasso_schedule.absorb("lasso/function/lrelu/d-onehot/claim", "committed");
    lasso_schedule.challenge("lasso/function/lrelu/d-onehot/r/0");
    if (!lasso_schedule.valid(&reason) || !lasso_schedule.replay(&reason)) return 22;
    NativeProofSession skipped_round("native-session-smoke-v1", statement);
    skipped_round.challenge("lasso/function/lrelu/a-query/r/0");
    bool skipped_round_rejected = false;
    try { skipped_round.challenge("lasso/function/lrelu/a-query/r/2"); } catch (const std::invalid_argument&) { skipped_round_rejected = true; }
    if (!skipped_round_rejected) return 23;
    NativeProofSession skipped_weight_round("native-session-smoke-v1", statement);
    skipped_weight_round.challenge("tensor/aux/tail/weight/0");
    bool skipped_weight_rejected = false;
    try { skipped_weight_round.challenge("tensor/aux/tail/weight/2"); } catch (const std::invalid_argument&) { skipped_weight_rejected = true; }
    if (!skipped_weight_rejected) return 25;
    NativeProofSession reordered_weight_rounds("native-session-smoke-v1", statement);
    reordered_weight_rounds.challenge("tensor/L_qry/tail/weight/0");
    reordered_weight_rounds.challenge("tensor/L_qry/tail/weight/1");
    auto reordered_weight_messages = reordered_weight_rounds.messages();
    std::swap(reordered_weight_messages[0], reordered_weight_messages[1]);
    reordered_weight_messages[0].sequence = 0;
    reordered_weight_messages[1].sequence = 1;
    NativeProofSession reordered_weight_replay;
    if (NativeProofSession::from_messages(reordered_weight_rounds.domain(), reordered_weight_rounds.statement(), reordered_weight_messages,
                                          reordered_weight_replay, &reason) ||
        reason.find("round schedule") == std::string::npos)
        return 26;
    NativeProofSession reordered_rounds("native-session-smoke-v1", statement);
    for (unsigned round = 0; round < 3; ++round) reordered_rounds.challenge("lasso/function/lrelu/a-query/r/" + std::to_string(round));
    auto reordered_messages = reordered_rounds.messages();
    std::swap(reordered_messages[0], reordered_messages[1]);
    reordered_messages[0].sequence = 0;
    reordered_messages[1].sequence = 1;
    NativeProofSession reordered_replay;
    if (NativeProofSession::from_messages(reordered_rounds.domain(), reordered_rounds.statement(), reordered_messages, reordered_replay, &reason) ||
        reason.find("round schedule") == std::string::npos)
        return 24;
    bool wrong_order = false;
    try { restored.challenge("consecutive/r/1"); } catch (const std::invalid_argument&) { wrong_order = true; }
    if (!wrong_order) return 18;
    for (const std::string& invalid : {"consecutive/r/010", "consecutive/r/18446744073709551616", "consecutive/r/not-a-number", "new-phase/r/1"}) {
        bool rejected_label = false;
        try { restored.challenge(invalid); } catch (const std::invalid_argument&) { rejected_label = true; }
        if (!rejected_label) return 20;
    }
    auto wrong_phase = restored.messages();
    wrong_phase.back().label = "challenge/changed-phase/r/23";
    NativeProofSession phase_rejected;
    if (NativeProofSession::from_messages(restored.domain(), restored.statement(), wrong_phase, phase_rejected, &reason)) return 21;
    NativeProofSession consecutive_replay;
    if (!NativeProofSession::from_messages(restored.domain(), restored.statement(), restored.messages(), consecutive_replay, &reason) || consecutive_replay.digest() != restored.digest()) return 19;
    const std::size_t prefix = restored.checkpoint();
    restored.absorb("rollback/temporary", "discarded");
    restored.challenge("rollback/temporary/r");
    if (!restored.replay(&reason) || !restored.rollback(prefix, &reason)) return 7;
    restored.absorb("rollback/replacement", "fresh");
    restored.challenge("rollback/replacement/r");
    if (!restored.valid(&reason) || !restored.replay(&reason)) return 8;
    auto early_tamper = restored.messages();
    early_tamper.front().payload = "different-early-commitment";
    NativeProofSession rejected;
    if (NativeProofSession::from_messages(restored.domain(), restored.statement(), early_tamper, rejected, &reason)) return 9;
    auto challenge_tamper = restored.messages();
    challenge_tamper[1].payload = "0";
    if (NativeProofSession::from_messages(restored.domain(), restored.statement(), challenge_tamper, rejected, &reason)) return 10;
    auto order_tamper = restored.messages();
    order_tamper.back().sequence = 0;
    if (NativeProofSession::from_messages(restored.domain(), restored.statement(), order_tamper, rejected, &reason)) return 11;
    auto duplicate = restored;
    const std::size_t duplicate_checkpoint = duplicate.checkpoint();
    duplicate.absorb("duplicate/separator", "message");
    bool duplicate_rejected = false;
    try { duplicate.challenge("gkr/r/0"); } catch (const std::invalid_argument&) { duplicate_rejected = true; }
    if (!duplicate_rejected || !duplicate.rollback(duplicate_checkpoint, &reason)) return 12;
    duplicate.absorb("duplicate/repaired", "message");
    duplicate.challenge("duplicate/fresh/r");
    if (!duplicate.valid(&reason) || !duplicate.replay(&reason)) return 13;
    auto truncation = restored.messages();
    truncation.resize(3);
    NativeProofSession truncated;
    if (!NativeProofSession::from_messages(restored.domain(), restored.statement(), truncation, truncated, &reason)) return 15;
    truncated.absorb_field("truncation/poly", Fr(19));
    truncated.challenge("truncation/r");
    if (!truncated.valid(&reason) || !truncated.replay(&reason)) return 16;

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

    Fr::setHashFunc(mcl::fp::sha512);
    NativeProofSession custom_hash_fallback("native-session-custom-hash-smoke-v1", statement);
    custom_hash_fallback.absorb("custom-hash/commitment", "legacy-hash-binding");
    custom_hash_fallback.challenge("custom-hash/r/0");
    if (!custom_hash_fallback.valid(&reason) || !custom_hash_fallback.replay(&reason)) {
        Fr::setHashFunc(mcl::fp::sha256);
        std::cerr << "native proof session did not preserve a configured non-default hash: " << reason << '\n';
        return 27;
    }
    Fr::setHashFunc(mcl::fp::sha256);
    std::cout << "native_proof_session_smoke: PASS first=" << first.getStr(16) << " second=" << second.getStr(16) << '\n';
    return 0;
}
