#ifndef ZKGPT_NATIVE_PROOF_SESSION_HPP
#define ZKGPT_NATIVE_PROOF_SESSION_HPP

#include "global_var.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

struct NativeProofStatement {
    // Every field below is an externally authenticated digest or immutable
    // field identifier.  The root is the explicit, authenticated
    // VerifierContext digest; all structured fields remain serialized and
    // validated so callers cannot replace the statement with an arbitrary
    // label.
    std::string verifier_context_digest;
    std::string field_id;
    std::string plan_digest;
    std::string resources_digest;
    std::string tables_digest;
    std::string com_w_digest;
    std::string aux_digest;
    std::string l_digest;
    std::string basis_digest;
    std::uint64_t l_dimension = 0;
    // The formal VerifierContext digest is the statement root.  Keeping the
    // equality explicit prevents a caller from silently substituting a
    // transcript-local label for the authenticated context identity.
    std::string statement_root_digest;

    std::string root() const;
    std::string structured_root() const;
    static std::string basis_digest_for(const G1 &G,
                                        const std::vector<G1> &generators,
                                        const std::vector<F> &basis,
                                        std::uint64_t l_dimension);
    bool valid(std::string *reason = nullptr) const;
};

enum class NativeProofMessageKind : std::uint8_t {
    Data = 1,
    Field = 2,
    FieldVector = 3,
    Challenge = 4,
};

// A typed, ordered transcript boundary used by the native zkGPT
// prover/verifier extensions.  Witness arrays and opening randomness never
// cross this boundary.  The native GKR/Lasso code remains the source of all
// arithmetic checks, while this class supplies deterministic Fiat-Shamir and
// an authenticated round-trip wire.
struct NativeProofMessage {
    NativeProofMessageKind kind = NativeProofMessageKind::Data;
    std::string label;
    std::string payload;
};

class NativeProofSession {
public:
    static const std::uint32_t kVersion = 2;

    NativeProofSession() = default;
    NativeProofSession(const std::string &domain,
                       const NativeProofStatement &statement);

    void absorb(const std::string &label, const std::string &payload);
    void absorb_field(const std::string &label, const F &value);
    void absorb_fields(const std::string &label, const std::vector<F> &values);

    // Derive and append a challenge.  Challenges are domain separated and
    // include the statement and every preceding message.
    F challenge(const std::string &label);

    // Roll back all messages emitted after a saved point.  Prover helpers use
    // this transaction boundary so failed proof attempts cannot leave a
    // partially valid transcript behind.
    std::size_t checkpoint() const { return messages_.size(); }
    bool rollback(std::size_t checkpoint, std::string *reason = nullptr);

    // Recompute every recorded challenge message without trusting the wire's
    // challenge payloads.  This is used by an independent verifier before it
    // evaluates any protocol relation.
    bool replay(std::string *reason = nullptr) const;

    std::string digest() const;
    bool valid(std::string *reason = nullptr) const;

    bool write(const std::string &path, std::string *reason = nullptr) const;
    static bool read(const std::string &path, NativeProofSession &out,
                     std::string *reason = nullptr);

    const std::string &domain() const { return domain_; }
    const NativeProofStatement &statement() const { return statement_; }
    std::string statement_digest() const { return statement_.root(); }
    const std::vector<NativeProofMessage> &messages() const { return messages_; }

private:
    std::string domain_;
    NativeProofStatement statement_;
    std::vector<NativeProofMessage> messages_;

    F derive(const std::string &label) const;
    static std::string field_encoding(const F &value);
    static bool decode_field(const std::string &payload, F &value);
    static bool decode_fields(const std::string &payload, std::vector<F> &values);
};

#endif
