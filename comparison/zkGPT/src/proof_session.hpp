#ifndef ZKGPT_NATIVE_PROOF_SESSION_HPP
#define ZKGPT_NATIVE_PROOF_SESSION_HPP

#include "global_var.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class NativeProofSession;

namespace zkgpt {
namespace native_pedersen {
struct PedersenBasis;
struct ScalarCommitment;
bool verify_opening(const NativeProofSession& session, const PedersenBasis& basis, const std::string& label, const ScalarCommitment& commitment,
                    std::string* reason);
}
}

struct NativeProofStatement {
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

    std::string statement_root_digest;

    std::string relation_schema = "zkgpt-gat-relation-v1";
    std::string relation_digest = std::string(64, '0');
    std::vector<std::string> relation_order;

    std::string root() const;
    std::string structured_root() const;
    static std::string basis_digest_for(const G1& G, const std::vector<G1>& generators, const std::vector<F>& basis, std::uint64_t l_dimension);
    bool valid(std::string* reason = nullptr) const;
};

enum class NativeProofMessageKind : std::uint8_t {
    Data = 1,
    Field = 2,
    FieldVector = 3,
    Challenge = 4,
};

struct NativeProofMessage {
    NativeProofMessageKind kind = NativeProofMessageKind::Data;
    std::string label;
    std::string payload;

    std::uint64_t sequence = 0;
};

class NativeProofSession {
  public:
    static const std::uint32_t kVersion = 4;

    NativeProofSession() = default;
    NativeProofSession(const std::string& domain, const NativeProofStatement& statement);

    void absorb(const std::string& label, const std::string& payload);
    void absorb_field(const std::string& label, const F& value);
    void absorb_fields(const std::string& label, const std::vector<F>& values);

    F challenge(const std::string& label);

    std::size_t checkpoint() const { return messages_.size(); }
    bool rollback(std::size_t checkpoint, std::string* reason = nullptr);

    bool replay(std::string* reason = nullptr) const;

    std::string digest() const;
    bool valid(std::string* reason = nullptr) const;

    bool write(const std::string& path, std::string* reason = nullptr) const;
    static bool read(const std::string& path, NativeProofSession& out, std::string* reason = nullptr);

    static bool from_messages(const std::string& domain, const NativeProofStatement& statement, const std::vector<NativeProofMessage>& messages,
                              NativeProofSession& out, std::string* reason = nullptr);

    const std::string& domain() const { return domain_; }
    const NativeProofStatement& statement() const { return statement_; }
    std::string statement_digest() const { return statement_.root(); }
    const std::vector<NativeProofMessage>& messages() const { return messages_; }
    std::uint64_t content_version() const { return content_version_; }
    const std::vector<std::size_t>* message_positions(const std::string& label) const {
        const auto found = label_positions_.find(label);
        return found == label_positions_.end() ? nullptr : &found->second;
    }

    bool has_label(const std::string& label) const { return label_positions_.find(label) != label_positions_.end(); }

  private:
    friend bool zkgpt::native_pedersen::verify_opening(const NativeProofSession&, const zkgpt::native_pedersen::PedersenBasis&, const std::string&,
                                                       const zkgpt::native_pedersen::ScalarCommitment&, std::string*);

    bool has_cached_verified_opening(const std::string& key) const { return verified_opening_cache_.find(key) != verified_opening_cache_.end(); }
    void cache_verified_opening(const std::string& key) const { verified_opening_cache_.insert(key); }
    std::string domain_;
    NativeProofStatement statement_;
    std::vector<NativeProofMessage> messages_;

    std::string message_body_cache_;
    std::unordered_map<std::string, std::vector<std::size_t>> label_positions_;
    mutable std::unordered_set<std::string> verified_opening_cache_;

    std::uint64_t content_version_ = 0;
    mutable bool valid_cache_ = false;
    mutable bool replay_cache_ = false;
    mutable std::uint64_t valid_cache_version_ = 0;
    mutable std::uint64_t replay_cache_version_ = 0;
    mutable bool valid_cache_result_ = false;
    mutable bool replay_cache_result_ = false;
    mutable std::string valid_cache_reason_;
    mutable std::string replay_cache_reason_;

    F derive(const std::string& label) const;
    void invalidate_validation_cache();
    void rebuild_message_body_cache();
    bool valid_uncached(std::string* reason) const;
    bool replay_uncached(std::string* reason) const;
    static std::string field_encoding(const F& value);
    static bool decode_field(const std::string& payload, F& value);
    static bool decode_fields(const std::string& payload, std::vector<F>& values);
};

#endif
