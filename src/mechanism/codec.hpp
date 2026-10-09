#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "gatzk/mechanism/runtime.hpp"

namespace gatzk::mechanism::internal
{
    std::string sha256_hex(std::string_view bytes);
    std::vector<std::uint8_t> encode_proof(const MechanismProof &proof);
    MechanismProof decode_proof(const std::vector<std::uint8_t> &bytes);
    void write_binary_file(const std::filesystem::path &path, const std::vector<std::uint8_t> &bytes);
    std::vector<std::uint8_t> read_binary_file(const std::filesystem::path &path, std::size_t max_bytes);
    std::string read_text_file(const std::filesystem::path &path, std::size_t max_bytes);
    void write_text_file(const std::filesystem::path &path, const std::string &bytes);
    std::string run_receipt_json(const RunOptions &options, const RunReceipt &receipt);
    std::string verify_receipt_json(const VerifyReceipt &receipt);
}
