#ifndef ZKGPT_NATIVE_RANGE_HPP
#define ZKGPT_NATIVE_RANGE_HPP

#include "proof_session.hpp"

#include <string>
#include <vector>

struct NativeRangeResult {
    F terminal = F_ZERO;
    F terminal_left = F_ZERO;
    F terminal_right = F_ZERO;
    std::vector<F> challenges;
};

bool native_range_prove_deg1(NativeProofSession& session, const std::string& label, const std::vector<F>& values, const F& claim,
                             NativeRangeResult* result = nullptr, std::string* reason = nullptr);

bool native_range_verify_deg1(const NativeProofSession& session, const std::string& label, const F& claim, F* terminal = nullptr,
                              std::string* reason = nullptr);

bool native_range_prove_deg3(NativeProofSession& session, const std::string& label, const std::vector<F>& eq, const std::vector<F>& f, const std::vector<F>& g,
                             const F& claim, NativeRangeResult* result = nullptr, std::string* reason = nullptr);

bool native_range_verify_deg3(const NativeProofSession& session, const std::string& label, const F& claim, F* terminal = nullptr,
                              std::string* reason = nullptr);

bool native_range_prove_product(NativeProofSession& session, const std::string& label, const std::vector<F>& left, const std::vector<F>& right, const F& claim,
                                NativeRangeResult* result = nullptr, std::string* reason = nullptr);

bool native_range_verify_product(const NativeProofSession& session, const std::string& label, const F& claim, F* terminal_left = nullptr,
                                 F* terminal_right = nullptr, std::string* reason = nullptr);

#endif
